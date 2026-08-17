//! Small HTTP server for effective VectorSeam recommendations.
//!
//! The server reads `calibrations/<cohort>/latest.json` from an object store
//! and exposes only `effective.recommended_ef`. Positive recommendations and
//! artifact defects are cached separately so arbitrary missing cohorts cannot
//! evict valid recommendations. Transient storage failures are not cached.

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
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use serde::Deserialize;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::{JoinError, JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use vectorseam_core::cohort::{CohortName, MAX_COHORT_NAME_BYTES};
use vectorseam_core::recommendation::{MAX_EF_SEARCH, MIN_EF_SEARCH};

const LATEST_JSON_MAX_BYTES: u64 = 256 * 1024;
const HTTP1_MAX_BUFFER_BYTES: usize = 16 * 1024;

const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:7738";
const DEFAULT_MAX_CONNECTIONS: usize = 100;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 100;
const DEFAULT_LOOKUP_TIMEOUT_SECONDS: u64 = 3;
const DEFAULT_REQUEST_HEAD_TIMEOUT_SECONDS: u64 = 4;
const DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS: u64 = 5;
const DEFAULT_CACHE_TTL_SECONDS: u64 = 60;
const DEFAULT_MAX_CACHED_COHORTS: u64 = 10_000;
const DEFAULT_MAX_CACHED_NEGATIVE_COHORTS: u64 = 256;

const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(100);
const LOAD_COALESCING_TTL: Duration = Duration::from_millis(10);

/// CLI and environment options for a host running the recommendation server.
#[derive(Args, Clone, Debug, PartialEq, Eq)]
pub struct RecommendationServerOptions {
    /// TCP address for the effective-recommendation HTTP server.
    #[arg(
        long = "recommendation-listen",
        env = "VECTORSEAM_RECOMMENDATION_LISTEN",
        default_value = DEFAULT_LISTEN_ADDR,
        value_name = "ADDR"
    )]
    pub listen_addr: SocketAddr,
    /// Maximum accepted recommendation HTTP connections.
    #[arg(
        id = "recommendation_max_connections",
        long = "recommendation-max-connections",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CONNECTIONS",
        default_value_t = DEFAULT_MAX_CONNECTIONS
    )]
    pub max_connections: usize,
    /// Maximum recommendation cache misses handled concurrently.
    #[arg(
        long = "recommendation-max-concurrent-requests",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CONCURRENT_REQUESTS",
        default_value_t = DEFAULT_MAX_CONCURRENT_REQUESTS
    )]
    pub max_concurrent_requests: usize,
    /// Object-store recommendation lookup deadline in seconds.
    #[arg(
        long = "recommendation-lookup-timeout-seconds",
        env = "VECTORSEAM_RECOMMENDATION_LOOKUP_TIMEOUT_SECONDS",
        default_value_t = DEFAULT_LOOKUP_TIMEOUT_SECONDS
    )]
    pub lookup_timeout_seconds: u64,
    /// HTTP/1 request-head deadline in seconds.
    #[arg(
        long = "recommendation-request-head-timeout-seconds",
        env = "VECTORSEAM_RECOMMENDATION_REQUEST_HEAD_TIMEOUT_SECONDS",
        default_value_t = DEFAULT_REQUEST_HEAD_TIMEOUT_SECONDS
    )]
    pub request_head_timeout_seconds: u64,
    /// Graceful connection-drain deadline in seconds.
    #[arg(
        long = "recommendation-shutdown-drain-timeout-seconds",
        env = "VECTORSEAM_RECOMMENDATION_SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
        default_value_t = DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS
    )]
    pub shutdown_drain_timeout_seconds: u64,
    /// Recommendation cache TTL in seconds.
    #[arg(
        long = "recommendation-cache-ttl-seconds",
        env = "VECTORSEAM_RECOMMENDATION_CACHE_TTL_SECONDS",
        default_value_t = DEFAULT_CACHE_TTL_SECONDS
    )]
    pub cache_ttl_seconds: u64,
    /// Maximum positive cohort entries retained by the cache.
    #[arg(
        long = "recommendation-max-cached-cohorts",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CACHED_COHORTS",
        default_value_t = DEFAULT_MAX_CACHED_COHORTS
    )]
    pub max_cached_cohorts: u64,
    /// Maximum missing or defective cohort entries retained by the cache.
    #[arg(
        long = "recommendation-max-cached-negative-cohorts",
        env = "VECTORSEAM_RECOMMENDATION_MAX_CACHED_NEGATIVE_COHORTS",
        default_value_t = DEFAULT_MAX_CACHED_NEGATIVE_COHORTS
    )]
    pub max_cached_negative_cohorts: u64,
}

impl Default for RecommendationServerOptions {
    fn default() -> Self {
        Self {
            listen_addr: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default recommendation listen address must be valid"),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            lookup_timeout_seconds: DEFAULT_LOOKUP_TIMEOUT_SECONDS,
            request_head_timeout_seconds: DEFAULT_REQUEST_HEAD_TIMEOUT_SECONDS,
            shutdown_drain_timeout_seconds: DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
            cache_ttl_seconds: DEFAULT_CACHE_TTL_SECONDS,
            max_cached_cohorts: DEFAULT_MAX_CACHED_COHORTS,
            max_cached_negative_cohorts: DEFAULT_MAX_CACHED_NEGATIVE_COHORTS,
        }
    }
}

impl RecommendationServerOptions {
    /// Converts host-facing options into server configuration.
    pub fn server_config(&self) -> Config {
        Config {
            listen_addr: self.listen_addr,
            max_connections: self.max_connections,
            max_concurrent_requests: self.max_concurrent_requests,
            lookup_timeout: Duration::from_secs(self.lookup_timeout_seconds),
            request_head_timeout: Duration::from_secs(self.request_head_timeout_seconds),
            shutdown_drain_timeout: Duration::from_secs(self.shutdown_drain_timeout_seconds),
            cache_ttl: Duration::from_secs(self.cache_ttl_seconds),
            max_cached_cohorts: self.max_cached_cohorts,
            max_cached_negative_cohorts: self.max_cached_negative_cohorts,
        }
    }
}

/// Recommendation HTTP server configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// TCP address on which the HTTP server listens.
    pub listen_addr: SocketAddr,
    /// Maximum number of accepted HTTP connections and connection tasks.
    pub max_connections: usize,
    /// Maximum number of cache-miss request handlers allowed to run concurrently.
    pub max_concurrent_requests: usize,
    /// Total deadline for one object-store recommendation lookup.
    pub lookup_timeout: Duration,
    /// Maximum time allowed to receive each HTTP/1 request head.
    pub request_head_timeout: Duration,
    /// Maximum time allowed for active HTTP connections to drain on shutdown.
    pub shutdown_drain_timeout: Duration,
    /// Time-to-live for successful per-cohort cache entries.
    pub cache_ttl: Duration,
    /// Maximum number of positive per-cohort entries retained in memory.
    pub max_cached_cohorts: u64,
    /// Maximum number of missing or defective cohort entries retained in memory.
    pub max_cached_negative_cohorts: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default recommendation listen address must be valid"),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            lookup_timeout: Duration::from_secs(DEFAULT_LOOKUP_TIMEOUT_SECONDS),
            request_head_timeout: Duration::from_secs(DEFAULT_REQUEST_HEAD_TIMEOUT_SECONDS),
            shutdown_drain_timeout: Duration::from_secs(DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS),
            cache_ttl: Duration::from_secs(DEFAULT_CACHE_TTL_SECONDS),
            max_cached_cohorts: DEFAULT_MAX_CACHED_COHORTS,
            max_cached_negative_cohorts: DEFAULT_MAX_CACHED_NEGATIVE_COHORTS,
        }
    }
}

/// Startup failure of the recommendation server.
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
}

/// A bound recommendation server that can be hosted by any Tokio process.
pub struct Server {
    listener: TcpListener,
    router: Router,
    max_connections: usize,
    request_head_timeout: Duration,
    shutdown_drain_timeout: Duration,
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
        let recommendations = Cache::builder()
            .max_capacity(config.max_cached_cohorts)
            .time_to_live(config.cache_ttl)
            .build();
        let negative_recommendations = Cache::builder()
            .max_capacity(config.max_cached_negative_cohorts)
            .time_to_live(config.cache_ttl)
            .build();
        let coalesced_loads = Cache::builder()
            .max_capacity(config.max_concurrent_requests as u64)
            .time_to_live(config.cache_ttl.min(LOAD_COALESCING_TTL))
            .build();
        let state = AppState {
            store,
            recommendations,
            negative_recommendations,
            coalesced_loads,
            load_permits: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            lookup_timeout: config.lookup_timeout,
        };
        let router = Router::new()
            .route("/v1/ef-search/{*cohort}", get(get_ef_search))
            .with_state(state);
        Ok(Self {
            listener,
            router,
            max_connections: config.max_connections,
            request_head_timeout: config.request_head_timeout,
            shutdown_drain_timeout: config.shutdown_drain_timeout,
        })
    }

    /// Returns the bound listener address, including an assigned ephemeral port.
    pub fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        self.listener.local_addr()
    }

    /// Serves requests until `shutdown` completes, then drains active requests.
    ///
    /// Accepted connections and their tasks are capped by `Config::max_connections`.
    /// Remaining connection tasks are aborted and joined after
    /// `Config::shutdown_drain_timeout`.
    pub async fn serve<S>(self, shutdown: S)
    where
        S: Future<Output = ()> + Send,
    {
        let Self {
            listener,
            router,
            max_connections,
            request_head_timeout,
            shutdown_drain_timeout,
        } = self;
        let address = listener.local_addr().ok();
        info!(?address, "recommendation HTTP server started");
        let connection_shutdown = CancellationToken::new();
        let mut connections = JoinSet::new();
        tokio::pin!(shutdown);

        loop {
            while let Some(joined) = connections.try_join_next() {
                handle_connection_join(joined);
            }
            if connections.len() >= max_connections {
                tokio::select! {
                    _ = &mut shutdown => break,
                    joined = connections.join_next() => {
                        if let Some(joined) = joined {
                            handle_connection_join(joined);
                        }
                    }
                }
                continue;
            }

            tokio::select! {
                _ = &mut shutdown => break,
                joined = connections.join_next(), if !connections.is_empty() => {
                    if let Some(joined) = joined {
                        handle_connection_join(joined);
                    }
                }
                accepted = listener.accept() => {
                    match accepted {
                        Ok((stream, peer)) => {
                            let connection_router = router.clone();
                            let task_shutdown = connection_shutdown.clone();
                            connections.spawn(async move {
                                serve_connection(
                                    stream,
                                    connection_router,
                                    task_shutdown,
                                    request_head_timeout,
                                )
                                .await
                                .map_err(|error| (peer, error))
                            });
                        }
                        Err(error) => {
                            warn!(%error, "recommendation HTTP accept failed; retrying");
                            tokio::select! {
                                _ = &mut shutdown => break,
                                _ = tokio::time::sleep(ACCEPT_ERROR_BACKOFF) => {}
                            }
                        }
                    }
                }
            }
        }

        connection_shutdown.cancel();
        drain_connections(&mut connections, shutdown_drain_timeout).await;
        info!("recommendation HTTP server stopped");
    }

    /// Spawns the bound server with cancellation-driven graceful shutdown.
    pub fn spawn(self, shutdown: CancellationToken) -> JoinHandle<()> {
        tokio::spawn(self.serve(shutdown.cancelled_owned()))
    }
}

async fn serve_connection(
    stream: tokio::net::TcpStream,
    router: Router,
    shutdown: CancellationToken,
    request_head_timeout: Duration,
) -> Result<(), hyper::Error> {
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(request_head_timeout)
        .max_buf_size(HTTP1_MAX_BUFFER_BYTES);
    let connection =
        builder.serve_connection(TokioIo::new(stream), TowerToHyperService::new(router));
    tokio::pin!(connection);
    tokio::select! {
        result = &mut connection => result,
        _ = shutdown.cancelled() => {
            connection.as_mut().graceful_shutdown();
            connection.await
        }
    }
}

fn handle_connection_join(joined: Result<Result<(), (SocketAddr, hyper::Error)>, JoinError>) {
    match joined {
        Ok(Ok(())) => {}
        Ok(Err((peer, error))) => {
            tracing::debug!(%peer, %error, "recommendation HTTP connection closed with error");
        }
        Err(error) if error.is_cancelled() => {
            tracing::debug!(%error, "recommendation HTTP connection task cancelled");
        }
        Err(error) => warn!(%error, "recommendation HTTP connection task failed"),
    }
}

async fn drain_connections(
    connections: &mut JoinSet<Result<(), (SocketAddr, hyper::Error)>>,
    timeout: Duration,
) {
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    while !connections.is_empty() {
        tokio::select! {
            joined = connections.join_next() => {
                if let Some(joined) = joined {
                    handle_connection_join(joined);
                }
            }
            _ = &mut deadline => {
                warn!(
                    remaining_connections = connections.len(),
                    timeout_seconds = timeout.as_secs_f64(),
                    "recommendation HTTP connection drain timed out; aborting connections"
                );
                connections.abort_all();
                break;
            }
        }
    }
    while let Some(joined) = connections.join_next().await {
        handle_connection_join(joined);
    }
}

#[derive(Clone)]
struct AppState {
    store: Arc<dyn ObjectStore>,
    recommendations: Cache<CohortName, i32>,
    negative_recommendations: Cache<CohortName, NegativeLookup>,
    coalesced_loads: Cache<CohortName, CachedLookup>,
    load_permits: Arc<Semaphore>,
    lookup_timeout: Duration,
}

#[derive(Clone, Debug)]
enum CachedLookup {
    Found(i32),
    Negative(NegativeLookup),
}

#[derive(Clone, Debug)]
enum NegativeLookup {
    Missing,
    Defect(Arc<PermanentDefect>),
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
enum TransientLookupError {
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
    #[error("recommendation lookup {path} exceeded {timeout_seconds} seconds")]
    Timeout { path: String, timeout_seconds: f64 },
}

#[derive(Clone, Debug, Error)]
enum PermanentDefect {
    #[error("{path} is {actual_bytes} bytes; maximum is {LATEST_JSON_MAX_BYTES}")]
    TooLarge { path: String, actual_bytes: u64 },
    #[error("{path} is malformed: {reason}")]
    Malformed { path: String, reason: String },
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
    let cohort = match CohortName::try_from(raw_cohort) {
        Ok(cohort) => cohort,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    if let Some(ef_search) = state.recommendations.get(&cohort).await {
        return ef_search.to_string().into_response();
    }
    if let Some(negative) = state.negative_recommendations.get(&cohort).await {
        return negative_response(&negative);
    }
    let Ok(_permit) = state.load_permits.clone().try_acquire_owned() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };

    let loader_store = state.store.clone();
    let loader_cohort = cohort.clone();
    let lookup_timeout = state.lookup_timeout;
    let loaded = state
        .coalesced_loads
        .try_get_with(cohort.clone(), async move {
            let result = load_with_timeout(&loader_store, &loader_cohort, lookup_timeout).await;
            match &result {
                Ok(CachedLookup::Negative(NegativeLookup::Defect(error))) => {
                    warn!(cohort = %loader_cohort, error = %error, "recommendation artifact rejected");
                }
                Err(error) => {
                    warn!(cohort = %loader_cohort, error = %error, "recommendation lookup failed");
                }
                _ => {}
            }
            result
        })
        .await;
    match loaded {
        Ok(CachedLookup::Found(ef_search)) => {
            state.recommendations.insert(cohort, ef_search).await;
            ef_search.to_string().into_response()
        }
        Ok(CachedLookup::Negative(negative)) => {
            state
                .negative_recommendations
                .insert(cohort, negative.clone())
                .await;
            negative_response(&negative)
        }
        Err(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

fn negative_response(negative: &NegativeLookup) -> Response {
    match negative {
        NegativeLookup::Missing => StatusCode::NOT_FOUND.into_response(),
        NegativeLookup::Defect(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn load_with_timeout(
    store: &Arc<dyn ObjectStore>,
    cohort: &CohortName,
    timeout: Duration,
) -> Result<CachedLookup, TransientLookupError> {
    match tokio::time::timeout(timeout, load_effective_ef(store, cohort)).await {
        Ok(result) => result,
        Err(_) => Err(TransientLookupError::Timeout {
            path: latest_path(cohort).to_string(),
            timeout_seconds: timeout.as_secs_f64(),
        }),
    }
}

async fn load_effective_ef(
    store: &Arc<dyn ObjectStore>,
    cohort: &CohortName,
) -> Result<CachedLookup, TransientLookupError> {
    let path = latest_path(cohort);
    let result = match store.get(&path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => {
            return Ok(CachedLookup::Negative(NegativeLookup::Missing));
        }
        Err(source) => {
            return Err(TransientLookupError::Get {
                path: path.to_string(),
                source,
            });
        }
    };
    if result.meta.size > LATEST_JSON_MAX_BYTES {
        return Ok(CachedLookup::Negative(NegativeLookup::Defect(Arc::new(
            PermanentDefect::TooLarge {
                path: path.to_string(),
                actual_bytes: result.meta.size,
            },
        ))));
    }
    let bytes = result
        .bytes()
        .await
        .map_err(|source| TransientLookupError::Body {
            path: path.to_string(),
            source,
        })?;
    let latest: LatestRound = match serde_json::from_slice(&bytes) {
        Ok(latest) => latest,
        Err(error) => {
            return Ok(CachedLookup::Negative(NegativeLookup::Defect(Arc::new(
                PermanentDefect::Malformed {
                    path: path.to_string(),
                    reason: error.to_string(),
                },
            ))));
        }
    };
    if latest.format_version != 1 {
        return Ok(CachedLookup::Negative(NegativeLookup::Defect(Arc::new(
            PermanentDefect::UnsupportedFormat {
                path: path.to_string(),
                format_version: latest.format_version,
            },
        ))));
    }
    if latest.cohort != cohort.as_str() {
        return Ok(CachedLookup::Negative(NegativeLookup::Defect(Arc::new(
            PermanentDefect::CohortMismatch {
                path: path.to_string(),
                expected: cohort.to_string(),
                actual: truncate_cohort_diagnostic(latest.cohort),
            },
        ))));
    }
    let Some(effective) = latest.effective else {
        return Ok(CachedLookup::Negative(NegativeLookup::Missing));
    };
    if !(MIN_EF_SEARCH..=MAX_EF_SEARCH).contains(&effective.recommended_ef) {
        return Ok(CachedLookup::Negative(NegativeLookup::Defect(Arc::new(
            PermanentDefect::InvalidEf {
                path: path.to_string(),
                ef: effective.recommended_ef,
            },
        ))));
    }
    Ok(CachedLookup::Found(effective.recommended_ef))
}

fn latest_path(cohort: &CohortName) -> Path {
    Path::from(format!("calibrations/{cohort}/latest.json"))
}

fn truncate_cohort_diagnostic(mut cohort: String) -> String {
    if cohort.len() <= MAX_COHORT_NAME_BYTES {
        return cohort;
    }
    let mut end = MAX_COHORT_NAME_BYTES;
    while !cohort.is_char_boundary(end) {
        end -= 1;
    }
    cohort.truncate(end);
    cohort
}

fn validate_config(config: &Config) -> Result<(), ServerError> {
    if config.max_connections == 0 {
        return Err(ServerError::InvalidConfig(
            "max_connections must be greater than zero",
        ));
    }
    if config.max_concurrent_requests == 0 {
        return Err(ServerError::InvalidConfig(
            "max_concurrent_requests must be greater than zero",
        ));
    }
    if config.lookup_timeout.is_zero() {
        return Err(ServerError::InvalidConfig(
            "lookup_timeout must be greater than zero",
        ));
    }
    if config.request_head_timeout.is_zero() {
        return Err(ServerError::InvalidConfig(
            "request_head_timeout must be greater than zero",
        ));
    }
    if config.shutdown_drain_timeout.is_zero() {
        return Err(ServerError::InvalidConfig(
            "shutdown_drain_timeout must be greater than zero",
        ));
    }
    if config.lookup_timeout >= config.shutdown_drain_timeout {
        return Err(ServerError::InvalidConfig(
            "lookup_timeout must be less than shutdown_drain_timeout",
        ));
    }
    if config.request_head_timeout >= config.shutdown_drain_timeout {
        return Err(ServerError::InvalidConfig(
            "request_head_timeout must be less than shutdown_drain_timeout",
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
    if config.max_cached_negative_cohorts == 0 {
        return Err(ServerError::InvalidConfig(
            "max_cached_negative_cohorts must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use async_trait::async_trait;
    use bytes::Bytes;
    use clap::Parser;
    use futures_util::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as StoreResult,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::oneshot;

    #[derive(Debug, Parser)]
    struct OptionsHarness {
        #[command(flatten)]
        recommendation_server: RecommendationServerOptions,
    }

    #[test]
    fn recommendation_server_options_supply_defaults_and_parse_overrides() {
        assert_eq!(
            OptionsHarness::try_parse_from(["test-host"])
                .unwrap()
                .recommendation_server,
            RecommendationServerOptions::default()
        );

        let options = OptionsHarness::try_parse_from([
            "test-host",
            "--recommendation-listen",
            "127.0.0.1:9000",
            "--recommendation-max-connections",
            "6",
            "--recommendation-max-concurrent-requests",
            "7",
            "--recommendation-lookup-timeout-seconds",
            "8",
            "--recommendation-request-head-timeout-seconds",
            "9",
            "--recommendation-shutdown-drain-timeout-seconds",
            "10",
            "--recommendation-cache-ttl-seconds",
            "11",
            "--recommendation-max-cached-cohorts",
            "12",
            "--recommendation-max-cached-negative-cohorts",
            "13",
        ])
        .unwrap()
        .recommendation_server;

        assert_eq!(options.listen_addr.port(), 9000);
        assert_eq!(options.max_connections, 6);
        assert_eq!(options.max_concurrent_requests, 7);
        assert_eq!(options.lookup_timeout_seconds, 8);
        assert_eq!(options.request_head_timeout_seconds, 9);
        assert_eq!(options.shutdown_drain_timeout_seconds, 10);
        assert_eq!(options.cache_ttl_seconds, 11);
        assert_eq!(options.max_cached_cohorts, 12);
        assert_eq!(options.max_cached_negative_cohorts, 13);
    }

    #[test]
    fn rejects_zero_resource_limits() {
        let config = Config {
            max_connections: 0,
            ..Config::default()
        };
        assert_invalid(config, "max_connections");

        let config = Config {
            max_concurrent_requests: 0,
            ..Config::default()
        };
        assert_invalid(config, "max_concurrent_requests");

        let config = Config {
            lookup_timeout: Duration::ZERO,
            ..Config::default()
        };
        assert_invalid(config, "lookup_timeout");

        let config = Config {
            lookup_timeout: Duration::from_secs(5),
            shutdown_drain_timeout: Duration::from_secs(5),
            ..Config::default()
        };
        assert_invalid(config, "lookup_timeout");

        let config = Config {
            request_head_timeout: Duration::ZERO,
            ..Config::default()
        };
        assert_invalid(config, "request_head_timeout");

        let config = Config {
            request_head_timeout: Duration::from_secs(5),
            shutdown_drain_timeout: Duration::from_secs(5),
            ..Config::default()
        };
        assert_invalid(config, "request_head_timeout");

        let config = Config {
            shutdown_drain_timeout: Duration::ZERO,
            ..Config::default()
        };
        assert_invalid(config, "shutdown_drain_timeout");

        let config = Config {
            cache_ttl: Duration::ZERO,
            ..Config::default()
        };
        assert_invalid(config, "cache_ttl");

        let config = Config {
            max_cached_cohorts: 0,
            ..Config::default()
        };
        assert_invalid(config, "max_cached_cohorts");

        let config = Config {
            max_cached_negative_cohorts: 0,
            ..Config::default()
        };
        assert_invalid(config, "max_cached_negative_cohorts");
    }

    #[tokio::test]
    async fn serves_effective_ef_as_plain_text_for_hierarchical_cohort() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_latest(&store, "prod/tenant-a/products", Some(60)).await;
        let harness = Harness::start(store, Config::default()).await;

        let response = get(harness.address, "/v1/ef-search/prod/tenant-a/products").await;

        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
        assert!(response.ends_with("\r\n\r\n60"), "{response}");
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn returns_not_found_without_an_effective_recommendation() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_latest(&store, "prod", None).await;
        let harness = Harness::start(store, Config::default()).await;

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
        let harness = Harness::start(
            store.clone(),
            Config {
                cache_ttl: Duration::from_millis(30),
                ..Config::default()
            },
        )
        .await;

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
        let harness = Harness::start(store, Config::default()).await;

        let response = get(harness.address, "/v1/ef-search/prod%20tenant").await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request\r\n"),
            "{response}"
        );
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn lookup_timeout_releases_request_capacity_and_recovers() {
        let store = Arc::new(TestStore::with_delay(Duration::from_millis(200)));
        let trait_store: Arc<dyn ObjectStore> = store.clone();
        put_latest(&trait_store, "prod", Some(20)).await;
        let harness = Harness::start(
            trait_store,
            Config {
                max_concurrent_requests: 1,
                lookup_timeout: Duration::from_millis(50),
                request_head_timeout: Duration::from_millis(100),
                shutdown_drain_timeout: Duration::from_millis(200),
                ..Config::default()
            },
        )
        .await;

        let first_address = harness.address;
        let first = tokio::spawn(async move { get(first_address, "/v1/ef-search/prod").await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let overloaded = get(harness.address, "/v1/ef-search/prod").await;
        let timed_out = first.await.unwrap();
        store.set_delay(Duration::ZERO);
        let recovered = get(harness.address, "/v1/ef-search/prod").await;

        assert!(overloaded.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(timed_out.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(recovered.ends_with("\r\n\r\n20"), "{recovered}");
        assert_eq!(store.get_count(), 2);
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn cache_hits_and_invalid_cohorts_bypass_busy_load_capacity() {
        let store = Arc::new(TestStore::default());
        let trait_store: Arc<dyn ObjectStore> = store.clone();
        put_latest(&trait_store, "hot", Some(20)).await;
        put_latest(&trait_store, "cold", Some(40)).await;
        let harness = Harness::start(
            trait_store,
            Config {
                max_concurrent_requests: 1,
                ..Config::default()
            },
        )
        .await;

        let primed = get(harness.address, "/v1/ef-search/hot").await;
        assert!(primed.ends_with("\r\n\r\n20"), "{primed}");
        store.set_delay(Duration::from_millis(200));
        let cold_address = harness.address;
        let cold = tokio::spawn(async move { get(cold_address, "/v1/ef-search/cold").await });
        tokio::time::timeout(Duration::from_millis(100), async {
            while store.get_count() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let cached = get(harness.address, "/v1/ef-search/hot").await;
        let invalid = get(harness.address, "/v1/ef-search/not%20valid").await;

        assert!(cached.ends_with("\r\n\r\n20"), "{cached}");
        assert!(invalid.starts_with("HTTP/1.1 400 Bad Request"), "{invalid}");
        assert!(cold.await.unwrap().ends_with("\r\n\r\n40"));
        harness.shutdown().await;
    }

    #[test]
    fn truncates_artifact_cohort_diagnostics_on_a_utf8_boundary() {
        let truncated = truncate_cohort_diagnostic("é".repeat(MAX_COHORT_NAME_BYTES));

        assert!(truncated.len() <= MAX_COHORT_NAME_BYTES);
        assert_eq!(truncated.chars().count(), MAX_COHORT_NAME_BYTES / 2);
    }

    #[tokio::test]
    async fn caches_permanent_artifact_defects_as_internal_errors() {
        let store = Arc::new(TestStore::default());
        let trait_store: Arc<dyn ObjectStore> = store.clone();
        put_raw(&trait_store, "prod", b"not json").await;
        let harness = Harness::start(trait_store, Config::default()).await;

        let first = get(harness.address, "/v1/ef-search/prod").await;
        let cached = get(harness.address, "/v1/ef-search/prod").await;

        assert!(first.starts_with("HTTP/1.1 500 Internal Server Error"));
        assert!(cached.starts_with("HTTP/1.1 500 Internal Server Error"));
        assert_eq!(store.get_count(), 1);
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn negative_cache_churn_does_not_evict_positive_recommendations() {
        let store = Arc::new(TestStore::default());
        let trait_store: Arc<dyn ObjectStore> = store.clone();
        put_latest(&trait_store, "prod", Some(20)).await;
        let harness = Harness::start(
            trait_store,
            Config {
                max_cached_cohorts: 1,
                max_cached_negative_cohorts: 1,
                cache_ttl: Duration::from_secs(1),
                ..Config::default()
            },
        )
        .await;

        let initial = get(harness.address, "/v1/ef-search/prod").await;
        for index in 0..8 {
            let response = get(harness.address, &format!("/v1/ef-search/junk-{index}")).await;
            assert!(response.starts_with("HTTP/1.1 404 Not Found"));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        let cached = get(harness.address, "/v1/ef-search/prod").await;

        assert!(initial.ends_with("\r\n\r\n20"));
        assert!(cached.ends_with("\r\n\r\n20"));
        assert_eq!(store.get_count(), 9);
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn concurrent_misses_share_one_object_store_lookup() {
        let store = Arc::new(TestStore::with_delay(Duration::from_millis(50)));
        let trait_store: Arc<dyn ObjectStore> = store.clone();
        put_latest(&trait_store, "prod", Some(20)).await;
        let harness = Harness::start(
            trait_store,
            Config {
                max_connections: 10,
                max_concurrent_requests: 10,
                lookup_timeout: Duration::from_millis(200),
                request_head_timeout: Duration::from_millis(250),
                shutdown_drain_timeout: Duration::from_millis(300),
                ..Config::default()
            },
        )
        .await;

        let mut requests = JoinSet::new();
        for _ in 0..10 {
            let address = harness.address;
            requests.spawn(async move { get(address, "/v1/ef-search/prod").await });
        }
        while let Some(joined) = requests.join_next().await {
            assert!(joined.unwrap().ends_with("\r\n\r\n20"));
        }

        assert_eq!(store.get_count(), 1);
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn connection_cap_blocks_accepting_more_connection_tasks() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let harness = Harness::start(
            store,
            Config {
                max_connections: 1,
                request_head_timeout: Duration::from_secs(1),
                ..Config::default()
            },
        )
        .await;
        let mut parked = tokio::net::TcpStream::connect(harness.address)
            .await
            .unwrap();
        parked
            .write_all(b"GET /v1/ef-search/prod HTTP/1.1\r\nHost: localhost\r\n")
            .await
            .unwrap();
        let mut queued = tokio::net::TcpStream::connect(harness.address)
            .await
            .unwrap();
        queued
            .write_all(
                b"GET /v1/ef-search/prod HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        let mut response = Vec::new();

        assert!(
            tokio::time::timeout(Duration::from_millis(30), queued.read_to_end(&mut response))
                .await
                .is_err()
        );
        drop(parked);
        tokio::time::timeout(
            Duration::from_millis(500),
            queued.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(
            String::from_utf8(response)
                .unwrap()
                .starts_with("HTTP/1.1 404 Not Found")
        );
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn partial_request_head_cannot_block_server_shutdown() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let harness = Harness::start(
            store,
            Config {
                request_head_timeout: Duration::from_millis(50),
                shutdown_drain_timeout: Duration::from_millis(200),
                lookup_timeout: Duration::from_millis(10),
                ..Config::default()
            },
        )
        .await;
        let mut parked = tokio::net::TcpStream::connect(harness.address)
            .await
            .unwrap();
        parked
            .write_all(b"GET /v1/ef-search/prod HTTP/1.1\r\nHost: localhost\r\n")
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_millis(300), harness.shutdown())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn connection_drain_aborts_and_joins_tasks_after_deadline() {
        let dropped = Arc::new(AtomicBool::new(false));
        let task_dropped = dropped.clone();
        let (started_tx, started_rx) = oneshot::channel();
        let mut connections = JoinSet::new();
        connections.spawn(async move {
            let _drop_flag = DropFlag(task_dropped);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
            Ok::<(), (SocketAddr, hyper::Error)>(())
        });
        started_rx.await.unwrap();

        drain_connections(&mut connections, Duration::from_millis(10)).await;

        assert!(connections.is_empty());
        assert!(dropped.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn request_head_timeout_closes_partial_requests() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let harness = Harness::start(
            store,
            Config {
                request_head_timeout: Duration::from_millis(30),
                ..Config::default()
            },
        )
        .await;
        let mut stream = tokio::net::TcpStream::connect(harness.address)
            .await
            .unwrap();
        stream
            .write_all(b"GET /v1/ef-search/prod HTTP/1.1\r\nHost: localhost\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();

        tokio::time::timeout(
            Duration::from_millis(300),
            stream.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        harness.shutdown().await;
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

    async fn put_raw(store: &Arc<dyn ObjectStore>, cohort: &str, body: &[u8]) {
        store
            .put(
                &latest_path(&CohortName::try_from(cohort).unwrap()),
                PutPayload::from(Bytes::copy_from_slice(body)),
            )
            .await
            .unwrap();
    }

    fn assert_invalid(config: Config, field: &str) {
        assert!(
            validate_config(&config)
                .unwrap_err()
                .to_string()
                .contains(field)
        );
    }

    struct Harness {
        address: SocketAddr,
        shutdown: CancellationToken,
        task: tokio::task::JoinHandle<()>,
    }

    impl Harness {
        async fn start(store: Arc<dyn ObjectStore>, config: Config) -> Self {
            let config = Config {
                listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
                ..config
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
            self.task.await.unwrap();
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

    #[derive(Debug, Default)]
    struct TestStore {
        inner: InMemory,
        delay_ms: AtomicU64,
        gets: AtomicUsize,
    }

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    impl TestStore {
        fn with_delay(delay: Duration) -> Self {
            Self {
                delay_ms: AtomicU64::new(delay.as_millis().try_into().unwrap()),
                ..Self::default()
            }
        }

        fn set_delay(&self, delay: Duration) {
            self.delay_ms
                .store(delay.as_millis().try_into().unwrap(), Ordering::Relaxed);
        }

        fn get_count(&self) -> usize {
            self.gets.load(Ordering::Relaxed)
        }
    }

    impl fmt::Display for TestStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("TestStore")
        }
    }

    #[async_trait]
    impl ObjectStore for TestStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            options: PutOptions,
        ) -> StoreResult<PutResult> {
            self.inner.put_opts(location, payload, options).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            options: PutMultipartOptions,
        ) -> StoreResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, options).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
            self.gets.fetch_add(1, Ordering::Relaxed);
            let delay = Duration::from_millis(self.delay_ms.load(Ordering::Relaxed));
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, StoreResult<Path>>,
        ) -> BoxStream<'static, StoreResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }
}
