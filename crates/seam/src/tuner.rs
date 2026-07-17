//! Stage 3 orchestration over data sources and configured cohorts.

use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

use crate::config::Config;
use crate::database::DatabaseConnection;
use crate::measure::{SampleMeasureError, SampleMeasurement, SampleMeasurer};
use crate::model::{AggregationConfig, RoundOutput};
use crate::pipeline::{CohortRoundOutcome, run_cohort_round};

const DATABASE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

/// Failure to prepare the local object store used by the tuner.
#[derive(Debug, Error)]
pub enum TunerStartError {
    /// The storage-root directory could not be created.
    #[error("could not create storage root {path}: {source}")]
    CreateStorageRoot {
        /// Configured local storage root.
        path: String,
        /// Filesystem error.
        #[source]
        source: io::Error,
    },
    /// The local object-store adapter rejected the configured root.
    #[error("could not open local object store at {path}: {source}")]
    OpenLocalStore {
        /// Configured local storage root.
        path: String,
        /// Object-store construction error.
        #[source]
        source: object_store::Error,
    },
    /// The bounded startup filesystem task panicked or was cancelled.
    #[error("local object-store startup task failed: {0}")]
    StartupTask(#[source] tokio::task::JoinError),
}

/// Results from one serialized pass over all configured cohorts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundRunReport {
    /// Fully published cohort rounds, keyed by exact cohort name.
    pub published: BTreeMap<String, RoundOutput>,
    /// Cohorts aborted before publication, with retryable diagnostic strings.
    pub failed_cohorts: BTreeMap<String, String>,
    /// Whether cancellation stopped the pass before all cohorts ran.
    pub cancelled: bool,
}

/// Long-lived tuner resources: one runtime per data source and one object store.
pub struct Tuner {
    config: Config,
    store: Arc<dyn ObjectStore>,
    data_sources: BTreeMap<String, DataSourceRuntime>,
}

impl Tuner {
    /// Creates the local storage adapter and connects each configured data source.
    ///
    /// Database connection failures are deliberately degraded into failed-sample
    /// measurers so cached Phase B work can still publish.
    pub async fn start(config: Config) -> Result<Self, TunerStartError> {
        let root = config.storage.root.clone();
        let store = tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&root).map_err(|source| {
                TunerStartError::CreateStorageRoot {
                    path: root.display().to_string(),
                    source,
                }
            })?;
            LocalFileSystem::new_with_prefix(&root).map_err(|source| {
                TunerStartError::OpenLocalStore {
                    path: root.display().to_string(),
                    source,
                }
            })
        })
        .await
        .map_err(TunerStartError::StartupTask)??;
        let store: Arc<dyn ObjectStore> = Arc::new(store);
        Ok(Self::start_with_store(config, store).await)
    }

    /// Connects data sources while using a caller-provided object store.
    pub(crate) async fn start_with_store(config: Config, store: Arc<dyn ObjectStore>) -> Self {
        let mut data_sources = BTreeMap::new();
        for (name, data_source) in &config.data_sources {
            let runtime = match DatabaseConnection::connect(data_source, &config.budget).await {
                Ok(connection) => DataSourceRuntime::Connected(Box::new(connection)),
                Err(error) => {
                    warn!(
                        data_source = %name,
                        server = %data_source.server,
                        database = %data_source.database,
                        %error,
                        "database unavailable; uncached samples will be recorded as failed"
                    );
                    DataSourceRuntime::Unavailable(error.to_string())
                }
            };
            data_sources.insert(name.clone(), runtime);
        }
        Self {
            config,
            store,
            data_sources,
        }
    }

    /// Runs every configured cohort sequentially for one already-aligned round.
    pub async fn run_round(
        &mut self,
        round_end: u64,
        computed_at: String,
        computed_at_us: u64,
        cancellation: &CancellationToken,
    ) -> RoundRunReport {
        let mut report = RoundRunReport::default();
        let cohorts = self.config.cohorts.keys().cloned().collect::<Vec<_>>();
        for cohort in cohorts {
            if cancellation.is_cancelled() {
                report.cancelled = true;
                break;
            }
            let Some(config) = AggregationConfig::for_cohort(&self.config, &cohort) else {
                let message = "validated cohort projection unexpectedly failed".to_owned();
                error!(cohort = %cohort, error = %message);
                report.failed_cohorts.insert(cohort.to_string(), message);
                continue;
            };
            let Some(cohort_config) = self.config.cohorts.get(&cohort) else {
                let message = "validated cohort configuration disappeared".to_owned();
                error!(cohort = %cohort, error = %message);
                report.failed_cohorts.insert(cohort.to_string(), message);
                continue;
            };
            let Some(index) = self.config.indexes.get(&cohort_config.index) else {
                let message = "validated index configuration disappeared".to_owned();
                error!(cohort = %cohort, error = %message);
                report.failed_cohorts.insert(cohort.to_string(), message);
                continue;
            };
            let Some(data_source) = self.data_sources.get_mut(&index.data_source) else {
                let message = "validated data source runtime disappeared".to_owned();
                error!(cohort = %cohort, error = %message);
                report.failed_cohorts.insert(cohort.to_string(), message);
                continue;
            };
            match run_cohort_round(
                Arc::clone(&self.store),
                config,
                round_end,
                computed_at.clone(),
                computed_at_us,
                data_source,
                cancellation,
            )
            .await
            {
                Ok(CohortRoundOutcome::Published(output)) => {
                    report.published.insert(cohort.to_string(), *output);
                }
                Ok(CohortRoundOutcome::Cancelled) => {
                    report.cancelled = true;
                    break;
                }
                Err(error) => {
                    error!(cohort = %cohort, %error, "cohort round aborted before publication");
                    report
                        .failed_cohorts
                        .insert(cohort.to_string(), error.to_string());
                }
            }
        }
        report
    }

    /// Drops every client and observes every owned connection-driver task.
    pub async fn shutdown(self) {
        let close_all = async move {
            for (name, runtime) in self.data_sources {
                let DataSourceRuntime::Connected(connection) = runtime else {
                    continue;
                };
                if let Err(error) = (*connection).close().await {
                    warn!(data_source = %name, %error, "database connection shutdown failed");
                }
            }
        };
        if tokio::time::timeout(DATABASE_SHUTDOWN_TIMEOUT, close_all)
            .await
            .is_err()
        {
            warn!("database shutdown deadline elapsed; remaining drivers were aborted");
        }
    }
}

enum DataSourceRuntime {
    Connected(Box<DatabaseConnection>),
    Unavailable(String),
}

#[async_trait]
impl SampleMeasurer for DataSourceRuntime {
    async fn measure_sample(
        &mut self,
        vector: &[f32],
        config: &AggregationConfig,
    ) -> Result<SampleMeasurement, SampleMeasureError> {
        match self {
            Self::Connected(connection) => connection.measure_sample(vector, config).await,
            Self::Unavailable(error) => Err(SampleMeasureError::Database(error.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use object_store::ObjectStoreExt;
    use object_store::PutPayload;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use tokio_postgres::NoTls;
    use vectorseam_core::cohort::CohortName;
    use vectorseam_core::frame::{FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION};
    use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, write_segment};
    use vectorseam_core::window::format_window_timestamp;

    use crate::config::{
        BudgetConfig, CalibrationConfig, CohortConfig, DataSourceConfig, IndexConfig,
        StorageConfig, TargetConfig,
    };
    use crate::model::RoundStatus;

    const WINDOW_START: u64 = 1_784_116_800;
    const WINDOW_SECONDS: u32 = 600;
    const PART_ULID: &str = "01K0A000000000000000000000";

    #[tokio::test]
    async fn c6_f_pg_table_smaller_stops_after_one_exact_and_other_cohort_continues() {
        if !f_pg_required() {
            return;
        }
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let query = axis_vector(1.0);
        seed_segment(
            &store,
            "acceptance/c6-a-small",
            PART_ULID,
            std::slice::from_ref(&query),
        )
        .await;
        seed_segment(
            &store,
            "acceptance/c6-b-other",
            PART_ULID,
            std::slice::from_ref(&query),
        )
        .await;
        let mut tuner = Tuner::start_with_store(c6_config(), Arc::clone(&store)).await;

        let report = tuner
            .run_round(
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-15T12:10:00Z".to_owned(),
                1_784_117_400_000_000,
                &CancellationToken::new(),
            )
            .await;

        let affected = &report.published["acceptance/c6-a-small"];
        assert_eq!(affected.status, RoundStatus::InsufficientSamples);
        assert!(affected.error.is_some());
        assert_eq!(affected.samples.measured, 0);
        let other = &report.published["acceptance/c6-b-other"];
        assert_eq!(other.samples.measured, 1);
        assert_eq!(other.samples.failed, 0);
        assert!(report.failed_cohorts.is_empty());
        // Five statements for the aborted transaction and eight for the
        // successful one-ef transaction. Any affected-cohort sweep would make
        // this larger.
        assert_eq!(statement_count(&tuner), 13);
        tuner.shutdown().await;
    }

    #[tokio::test]
    async fn d3_f_pg_one_millisecond_timeout_fails_without_retries_or_leaks() {
        if !f_pg_required() {
            return;
        }
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let vectors = (0..20)
            .map(|index| axis_vector(0.10 + index as f32 * 0.01))
            .collect::<Vec<_>>();
        seed_segment(&store, "acceptance/d3-timeout", PART_ULID, &vectors).await;
        let mut tuner = Tuner::start_with_store(d3_config(), Arc::clone(&store)).await;
        // Keep the exact scan blocked so the frozen 1 ms timeout is
        // deterministic even on a warm, unusually fast local PostgreSQL.
        let (lock_client, lock_driver) = lock_fixture_table().await;

        let first = tuner
            .run_round(
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-15T12:10:00Z".to_owned(),
                1_784_117_400_000_000,
                &CancellationToken::new(),
            )
            .await;
        let output = &first.published["acceptance/d3-timeout"];
        assert_eq!(output.samples.available, 20);
        assert_eq!(output.samples.measured, 0);
        assert_eq!(output.samples.failed, 20);
        assert_eq!(sample_transaction_count(&tuner), 20);

        let transactions_after_first = sample_transaction_count(&tuner);
        let second = tuner
            .run_round(
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-15T12:10:01Z".to_owned(),
                1_784_117_401_000_000,
                &CancellationToken::new(),
            )
            .await;
        assert_eq!(second.published["acceptance/d3-timeout"].samples.failed, 20);
        assert_eq!(sample_transaction_count(&tuner), transactions_after_first);
        lock_client.batch_execute("ROLLBACK").await.unwrap();
        drop(lock_client);
        lock_driver.await.unwrap().unwrap();
        assert_eq!(active_tuner_statements().await, 0);
        tuner.shutdown().await;
    }

    fn c6_config() -> Config {
        let mut config = base_config(Duration::from_secs(5));
        config.indexes = BTreeMap::from([
            (
                "small".to_owned(),
                IndexConfig {
                    data_source: "primary".to_owned(),
                    table: "docs_seam_tie_fixture".to_owned(),
                    key: "doc_id".to_owned(),
                    column: "embedding".to_owned(),
                },
            ),
            (
                "other".to_owned(),
                IndexConfig {
                    data_source: "primary".to_owned(),
                    table: "docs_seam_fixture".to_owned(),
                    key: "doc_id".to_owned(),
                    column: "embedding".to_owned(),
                },
            ),
        ]);
        config.targets = BTreeMap::from([
            ("small".to_owned(), target(20)),
            ("other".to_owned(), target(10)),
        ]);
        config.cohorts = BTreeMap::from([
            (
                CohortName::try_from("acceptance/c6-a-small").unwrap(),
                CohortConfig {
                    index: "small".to_owned(),
                    target: "small".to_owned(),
                },
            ),
            (
                CohortName::try_from("acceptance/c6-b-other").unwrap(),
                CohortConfig {
                    index: "other".to_owned(),
                    target: "other".to_owned(),
                },
            ),
        ]);
        config
    }

    fn d3_config() -> Config {
        let mut config = base_config(Duration::from_millis(1));
        config.indexes = BTreeMap::from([(
            "fixture".to_owned(),
            IndexConfig {
                data_source: "primary".to_owned(),
                table: "docs_seam_timeout_fixture".to_owned(),
                key: "doc_id".to_owned(),
                column: "embedding".to_owned(),
            },
        )]);
        config.targets = BTreeMap::from([("timeout".to_owned(), target(10))]);
        config.cohorts = BTreeMap::from([(
            CohortName::try_from("acceptance/d3-timeout").unwrap(),
            CohortConfig {
                index: "fixture".to_owned(),
                target: "timeout".to_owned(),
            },
        )]);
        config
    }

    fn base_config(statement_timeout: Duration) -> Config {
        let port = std::env::var("SEAM_PG_PORT").unwrap_or_else(|_| "55432".to_owned());
        Config {
            calibration: CalibrationConfig {
                interval: Duration::from_secs(600),
                ef_search: vec![20],
                train_fraction: 0.7,
                split_seed: 7,
                min_samples: 100,
            },
            storage: StorageConfig {
                root: "/unused-in-memory".into(),
                window_seconds: WINDOW_SECONDS,
            },
            budget: BudgetConfig {
                db_share: 1.0,
                statement_timeout,
            },
            data_sources: BTreeMap::from([(
                "primary".to_owned(),
                DataSourceConfig {
                    server: format!("127.0.0.1:{port}"),
                    database: "postgres".to_owned(),
                    user: "postgres".to_owned(),
                    password_env: Some("SEAM_TEST_PG_PASSWORD".to_owned()),
                },
            )]),
            indexes: BTreeMap::new(),
            targets: BTreeMap::new(),
            cohorts: BTreeMap::new(),
        }
    }

    fn target(k: u32) -> TargetConfig {
        TargetConfig {
            k,
            value: 0.9,
            percentile: 0.95,
            window: Duration::from_secs(u64::from(WINDOW_SECONDS)),
        }
    }

    fn statement_count(tuner: &Tuner) -> u64 {
        match &tuner.data_sources["primary"] {
            DataSourceRuntime::Connected(connection) => connection.statement_count(),
            DataSourceRuntime::Unavailable(error) => {
                panic!("F-pg data source unexpectedly unavailable: {error}")
            }
        }
    }

    fn sample_transaction_count(tuner: &Tuner) -> u64 {
        match &tuner.data_sources["primary"] {
            DataSourceRuntime::Connected(connection) => connection.sample_transaction_count(),
            DataSourceRuntime::Unavailable(error) => {
                panic!("F-pg data source unexpectedly unavailable: {error}")
            }
        }
    }

    async fn active_tuner_statements() -> i64 {
        let (client, driver) = tokio_postgres::connect(&database_url(), NoTls)
            .await
            .expect("D3 activity-inspection connection must open");
        let driver = tokio::spawn(driver);
        let count = client
            .query_one(
                "SELECT count(*)::bigint
                 FROM pg_stat_activity
                 WHERE application_name = 'vectorseam-seam'
                   AND state = 'active'",
                &[],
            )
            .await
            .unwrap()
            .get(0);
        drop(client);
        driver.await.unwrap().unwrap();
        count
    }

    async fn lock_fixture_table() -> (
        tokio_postgres::Client,
        tokio::task::JoinHandle<Result<(), tokio_postgres::Error>>,
    ) {
        let (client, connection) = tokio_postgres::connect(&database_url(), NoTls)
            .await
            .expect("D3 fixture-lock connection must open");
        let driver = tokio::spawn(connection);
        client
            .batch_execute(
                "BEGIN;
                 LOCK TABLE docs_seam_timeout_fixture IN ACCESS EXCLUSIVE MODE;",
            )
            .await
            .expect("D3 fixture table lock must be acquired");
        (client, driver)
    }

    fn database_url() -> String {
        std::env::var("SEAM_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:password@localhost:55432/postgres".to_owned()
        })
    }

    async fn seed_segment(
        store: &Arc<dyn ObjectStore>,
        cohort: &str,
        part_ulid: &str,
        vectors: &[Vec<f32>],
    ) {
        let frames = vectors
            .iter()
            .map(|vector| frame(cohort, vector))
            .collect::<Vec<_>>();
        let records = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| SegmentRecordRef {
                receive_time: WINDOW_START * 1_000_000 + index as u64,
                frame,
            })
            .collect::<Vec<_>>();
        let bytes = write_segment(
            &SegmentHeader {
                window_start: WINDOW_START,
                window_seconds: WINDOW_SECONDS,
                first_receive: WINDOW_START * 1_000_000,
                last_receive: WINDOW_START * 1_000_000 + vectors.len() as u64 - 1,
                received_frame_count: vectors.len() as u64,
                record_count: vectors.len() as u64,
                cohort: CohortName::try_from(cohort).unwrap(),
            },
            &records,
        )
        .unwrap();
        let timestamp = format_window_timestamp(WINDOW_START).unwrap();
        let path = Path::from(format!(
            "cohorts/{cohort}/window={timestamp}/part-{part_ulid}.vseam"
        ));
        store.put(&path, PutPayload::from(bytes)).await.unwrap();
    }

    fn frame(cohort: &str, vector: &[f32]) -> Vec<u8> {
        let name = cohort.as_bytes();
        let vector_bytes = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let total_len = FIXED_FRAME_HEADER_LEN + name.len() + vector_bytes.len();
        let mut bytes = Vec::with_capacity(total_len);
        bytes.extend_from_slice(&u32::try_from(total_len - 4).unwrap().to_le_bytes());
        bytes.extend_from_slice(&FRAME_MAGIC);
        bytes.extend_from_slice(&FRAME_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(name.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector_bytes.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&vector_bytes);
        bytes
    }

    fn axis_vector(first: f32) -> Vec<f32> {
        let mut vector = vec![0.0; 64];
        vector[0] = first;
        vector[1] = (1.0 - first * first).sqrt();
        vector
    }

    fn f_pg_required() -> bool {
        std::env::var_os("SEAM_REQUIRE_F_PG").is_some()
    }
}
