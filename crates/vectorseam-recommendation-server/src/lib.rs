//! Small HTTP server for effective VectorSeam recommendations.
//!
//! The server reads `calibrations/<cohort>/latest.json` from an object store
//! and exposes only `effective.recommended_ef`. Successful lookups, including
//! the absence of an effective recommendation, are cached by cohort. Storage
//! and malformed-artifact failures are not cached.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use clap::Args;
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use serde::Deserialize;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use vectorseam_core::cohort::CohortName;

const LATEST_JSON_MAX_BYTES: u64 = 256 * 1024;

const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:7738";
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 100;
const DEFAULT_CACHE_TTL_SECONDS: u64 = 60;
const DEFAULT_MAX_CACHED_COHORTS: u64 = 10_000;

/// CLI and environment options for hosting the recommendation server.
#[derive(Args, Clone, Debug, PartialEq, Eq)]
pub struct ServerOptions {
    /// TCP address for the effective-recommendation HTTP server.
    #[arg(
        long = "recommendation-listen",
        env = "VECTORSEAM_RECOMMENDATION_LISTEN",
        default_value = DEFAULT_LISTEN_ADDR,
        value_name = "ADDR"
    )]
    pub listen_addr: SocketAddr,
    /// Maximum concurrently handled recommendation HTTP requests.
    #[arg(
        long = "recommendation-max-concurrent-requests",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CONCURRENT_REQUESTS",
        default_value_t = DEFAULT_MAX_CONCURRENT_REQUESTS
    )]
    pub max_concurrent_requests: usize,
    /// Successful recommendation lookup cache TTL in seconds.
    #[arg(
        long = "recommendation-cache-ttl-seconds",
        env = "VECTORSEAM_RECOMMENDATION_CACHE_TTL_SECONDS",
        default_value_t = DEFAULT_CACHE_TTL_SECONDS
    )]
    pub cache_ttl_seconds: u64,
    /// Maximum cohort entries retained by the recommendation cache.
    #[arg(
        long = "recommendation-max-cached-cohorts",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CACHED_COHORTS",
        default_value_t = DEFAULT_MAX_CACHED_COHORTS
    )]
    pub max_cached_cohorts: u64,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            listen_addr: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default recommendation listen address must be valid"),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            cache_ttl_seconds: DEFAULT_CACHE_TTL_SECONDS,
            max_cached_cohorts: DEFAULT_MAX_CACHED_COHORTS,
        }
    }
}

impl ServerOptions {
    /// Converts host-facing options into recommendation server configuration.
    pub fn server_config(&self) -> Config {
        Config {
            listen_addr: self.listen_addr,
            max_concurrent_requests: self.max_concurrent_requests,
            cache_ttl: Duration::from_secs(self.cache_ttl_seconds),
            max_cached_cohorts: self.max_cached_cohorts,
        }
    }
}

/// Recommendation HTTP server configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// TCP address on which the HTTP server listens.
    pub listen_addr: SocketAddr,
    /// Maximum number of request handlers allowed to run concurrently.
    pub max_concurrent_requests: usize,
    /// Time-to-live for successful per-cohort cache entries.
    pub cache_ttl: Duration,
    /// Maximum number of per-cohort entries retained in memory.
    pub max_cached_cohorts: u64,
}

impl Default for Config {
    fn default() -> Self {
        ServerOptions::default().server_config()
    }
}

/// Startup or runtime failure of the recommendation server.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Server configuration is invalid.
    #[error("invalid recommendation server configuration: {0}")]
    InvalidConfig(&'static str),
    /// The HTTP listener could not bind.
    #[error("binding recommendation HTTP listener {address} failed: {source}")]
    Bind {
        /// Requested listener address.
        address: SocketAddr,
        /// Operating-system bind error.
        #[source]
        source: std::io::Error,
    },
    /// The HTTP server stopped with an I/O error.
    #[error("recommendation HTTP server failed: {0}")]
    Serve(#[source] std::io::Error),
}

/// A bound recommendation server that can be hosted by any Tokio process.
pub struct Server {
    listener: TcpListener,
    router: Router,
}

impl Server {
    /// Binds a recommendation server backed by `store`.
    pub async fn bind(config: Config, store: Arc<dyn ObjectStore>) -> Result<Self, ServerError> {
        validate_config(&config)?;
        let listener = TcpListener::bind(config.listen_addr)
            .await
            .map_err(|source| ServerError::Bind {
                address: config.listen_addr,
                source,
            })?;
        let cache = Cache::builder()
            .max_capacity(config.max_cached_cohorts)
            .time_to_live(config.cache_ttl)
            .build();
        let state = AppState {
            store,
            cache,
            request_permits: Arc::new(Semaphore::new(config.max_concurrent_requests)),
        };
        let router = Router::new()
            .route("/v1/ef-search/{*cohort}", get(get_ef_search))
            .with_state(state);
        Ok(Self { listener, router })
    }

    /// Returns the bound listener address, including an assigned ephemeral port.
    pub fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        self.listener.local_addr()
    }

    /// Serves requests until `shutdown` completes, then drains active requests.
    pub async fn serve<S>(self, shutdown: S) -> Result<(), ServerError>
    where
        S: Future<Output = ()> + Send + 'static,
    {
        let address = self.listener.local_addr().ok();
        info!(?address, "recommendation HTTP server started");
        axum::serve(self.listener, self.router)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(ServerError::Serve)
    }

    /// Spawns the bound server with cancellation-driven graceful shutdown.
    pub fn spawn(self, shutdown: CancellationToken) -> JoinHandle<Result<(), ServerError>> {
        tokio::spawn(self.serve(shutdown.cancelled_owned()))
    }
}

#[derive(Clone)]
struct AppState {
    store: Arc<dyn ObjectStore>,
    cache: Cache<CohortName, Option<i32>>,
    request_permits: Arc<Semaphore>,
}

#[derive(Debug, Deserialize)]
struct LatestRound {
    format_version: u32,
    cohort: String,
    effective: Option<EffectiveRecommendation>,
}

#[derive(Debug, Deserialize)]
struct EffectiveRecommendation {
    recommended_ef: i32,
}

#[derive(Debug, Error)]
enum LookupError {
    #[error("GET {path} failed: {source}")]
    Get {
        path: String,
        #[source]
        source: object_store::Error,
    },
    #[error("GET body {path} failed: {source}")]
    Body {
        path: String,
        #[source]
        source: object_store::Error,
    },
    #[error("{path} is {actual_bytes} bytes; maximum is {LATEST_JSON_MAX_BYTES}")]
    TooLarge { path: String, actual_bytes: u64 },
    #[error("{path} is malformed: {source}")]
    Malformed {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path} has unsupported format_version {format_version}")]
    UnsupportedFormat { path: String, format_version: u32 },
    #[error("{path} contains cohort {actual:?}, expected {expected:?}")]
    CohortMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("{path} contains out-of-range effective recommended_ef {ef}")]
    InvalidEf { path: String, ef: i32 },
}

async fn get_ef_search(
    State(state): State<AppState>,
    AxumPath(raw_cohort): AxumPath<String>,
) -> Response {
    let Ok(_permit) = state.request_permits.clone().try_acquire_owned() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let cohort = match CohortName::try_from(raw_cohort) {
        Ok(cohort) => cohort,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let loader_store = state.store.clone();
    let loader_cohort = cohort.clone();
    match state
        .cache
        .try_get_with(cohort.clone(), async move {
            load_effective_ef(&loader_store, &loader_cohort).await
        })
        .await
    {
        Ok(Some(ef_search)) => ef_search.to_string().into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(error) => {
            warn!(cohort = %cohort, error = %error, "recommendation lookup failed");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    }
}

async fn load_effective_ef(
    store: &Arc<dyn ObjectStore>,
    cohort: &CohortName,
) -> Result<Option<i32>, LookupError> {
    let path = Path::from(format!("calibrations/{cohort}/latest.json"));
    let result = match store.get(&path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(source) => {
            return Err(LookupError::Get {
                path: path.to_string(),
                source,
            });
        }
    };
    if result.meta.size > LATEST_JSON_MAX_BYTES {
        return Err(LookupError::TooLarge {
            path: path.to_string(),
            actual_bytes: result.meta.size,
        });
    }
    let bytes = result.bytes().await.map_err(|source| LookupError::Body {
        path: path.to_string(),
        source,
    })?;
    let latest: LatestRound =
        serde_json::from_slice(&bytes).map_err(|source| LookupError::Malformed {
            path: path.to_string(),
            source,
        })?;
    if latest.format_version != 1 {
        return Err(LookupError::UnsupportedFormat {
            path: path.to_string(),
            format_version: latest.format_version,
        });
    }
    if latest.cohort != cohort.as_str() {
        return Err(LookupError::CohortMismatch {
            path: path.to_string(),
            expected: cohort.to_string(),
            actual: latest.cohort,
        });
    }
    let Some(effective) = latest.effective else {
        return Ok(None);
    };
    if !(1..=1_000).contains(&effective.recommended_ef) {
        return Err(LookupError::InvalidEf {
            path: path.to_string(),
            ef: effective.recommended_ef,
        });
    }
    Ok(Some(effective.recommended_ef))
}

fn validate_config(config: &Config) -> Result<(), ServerError> {
    if config.max_concurrent_requests == 0 {
        return Err(ServerError::InvalidConfig(
            "max_concurrent_requests must be greater than zero",
        ));
    }
    if config.cache_ttl.is_zero() {
        return Err(ServerError::InvalidConfig(
            "cache_ttl must be greater than zero",
        ));
    }
    if config.max_cached_cohorts == 0 {
        return Err(ServerError::InvalidConfig(
            "max_cached_cohorts must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use bytes::Bytes;
    use clap::Parser;
    use object_store::PutPayload;
    use object_store::memory::InMemory;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        recommendation: ServerOptions,
    }

    #[test]
    fn host_options_supply_defaults_and_parse_overrides() {
        let defaults = TestCli::try_parse_from(["test"]).unwrap().recommendation;
        assert_eq!(
            defaults.listen_addr,
            DEFAULT_LISTEN_ADDR.parse::<SocketAddr>().unwrap()
        );
        assert_eq!(
            defaults.max_concurrent_requests,
            DEFAULT_MAX_CONCURRENT_REQUESTS
        );
        assert_eq!(defaults.cache_ttl_seconds, DEFAULT_CACHE_TTL_SECONDS);
        assert_eq!(defaults.max_cached_cohorts, DEFAULT_MAX_CACHED_COHORTS);

        let overridden = TestCli::try_parse_from([
            "test",
            "--recommendation-listen",
            "127.0.0.1:9000",
            "--recommendation-max-concurrent-requests",
            "7",
            "--recommendation-cache-ttl-seconds",
            "8",
            "--recommendation-max-cached-cohorts",
            "9",
        ])
        .unwrap()
        .recommendation;
        assert_eq!(overridden.listen_addr.port(), 9000);
        assert_eq!(overridden.max_concurrent_requests, 7);
        assert_eq!(overridden.cache_ttl_seconds, 8);
        assert_eq!(overridden.max_cached_cohorts, 9);
    }

    #[test]
    fn rejects_zero_resource_limits() {
        let config = Config {
            max_concurrent_requests: 0,
            ..Config::default()
        };
        assert!(
            validate_config(&config)
                .unwrap_err()
                .to_string()
                .contains("max_concurrent_requests")
        );

        let config = Config {
            cache_ttl: Duration::ZERO,
            ..Config::default()
        };
        assert!(
            validate_config(&config)
                .unwrap_err()
                .to_string()
                .contains("cache_ttl")
        );

        let config = Config {
            max_cached_cohorts: 0,
            ..Config::default()
        };
        assert!(
            validate_config(&config)
                .unwrap_err()
                .to_string()
                .contains("max_cached_cohorts")
        );
    }

    #[tokio::test]
    async fn serves_effective_ef_as_plain_text_for_hierarchical_cohort() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_latest(&store, "prod/tenant-a/products", Some(60)).await;
        let harness = Harness::start(store, Config::default().cache_ttl).await;

        let response = get(harness.address, "/v1/ef-search/prod/tenant-a/products").await;

        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
        assert!(response.ends_with("\r\n\r\n60"), "{response}");
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn returns_not_found_without_an_effective_recommendation() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_latest(&store, "prod", None).await;
        let harness = Harness::start(store, Config::default().cache_ttl).await;

        let response = get(harness.address, "/v1/ef-search/prod").await;

        assert!(
            response.starts_with("HTTP/1.1 404 Not Found\r\n"),
            "{response}"
        );
        assert!(response.ends_with("\r\n\r\n"), "{response}");
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn caches_successful_lookup_until_ttl_expires() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_latest(&store, "prod", Some(20)).await;
        let harness = Harness::start(store.clone(), Duration::from_millis(30)).await;

        let first = get(harness.address, "/v1/ef-search/prod").await;
        put_latest(&store, "prod", Some(40)).await;
        let cached = get(harness.address, "/v1/ef-search/prod").await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let refreshed = get(harness.address, "/v1/ef-search/prod").await;

        assert!(first.ends_with("\r\n\r\n20"), "{first}");
        assert!(cached.ends_with("\r\n\r\n20"), "{cached}");
        assert!(refreshed.ends_with("\r\n\r\n40"), "{refreshed}");
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn rejects_invalid_cohort_without_reading_storage() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let harness = Harness::start(store, Config::default().cache_ttl).await;

        let response = get(harness.address, "/v1/ef-search/prod%20tenant").await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request\r\n"),
            "{response}"
        );
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn rejects_request_when_concurrency_limit_is_in_use() {
        let state = AppState {
            store: Arc::new(InMemory::new()),
            cache: Cache::builder()
                .max_capacity(1)
                .time_to_live(Config::default().cache_ttl)
                .build(),
            request_permits: Arc::new(Semaphore::new(1)),
        };
        let permit = state.request_permits.clone().acquire_owned().await.unwrap();

        let response = get_ef_search(State(state), AxumPath("prod".to_owned())).await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        drop(permit);
    }

    async fn put_latest(store: &Arc<dyn ObjectStore>, cohort: &str, ef: Option<i32>) {
        let effective = ef.map(|recommended_ef| {
            serde_json::json!({
                "recommended_ef": recommended_ef,
                "confidence": 0.95,
                "source_round": "2026-08-05T12:00:00Z",
                "carried": false
            })
        });
        let body = serde_json::json!({
            "format_version": 1,
            "cohort": cohort,
            "recommended_ef": 999,
            "effective": effective
        });
        let path = Path::from(format!("calibrations/{cohort}/latest.json"));
        store
            .put(
                &path,
                PutPayload::from(Bytes::from(serde_json::to_vec(&body).unwrap())),
            )
            .await
            .unwrap();
    }

    struct Harness {
        address: SocketAddr,
        shutdown: CancellationToken,
        task: tokio::task::JoinHandle<Result<(), ServerError>>,
    }

    impl Harness {
        async fn start(store: Arc<dyn ObjectStore>, cache_ttl: Duration) -> Self {
            let config = Config {
                listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
                cache_ttl,
                ..Config::default()
            };
            let server = Server::bind(config, store).await.unwrap();
            let address = server.local_addr().unwrap();
            let shutdown = CancellationToken::new();
            let task = server.spawn(shutdown.clone());
            Self {
                address,
                shutdown,
                task,
            }
        }

        async fn shutdown(self) {
            self.shutdown.cancel();
            self.task.await.unwrap().unwrap();
        }
    }

    async fn get(address: SocketAddr, path: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }
}
