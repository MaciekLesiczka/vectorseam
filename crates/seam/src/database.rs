//! One-connection PostgreSQL measurement and transaction construction.

use std::env;
use std::time::Duration;

use async_trait::async_trait;
use pgvector::Vector;
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_postgres::types::Type;
use tokio_postgres::{Client, Config as PostgresConfig, IsolationLevel, NoTls, Row, Transaction};
use tracing::warn;

use crate::config::{BudgetConfig, DataSourceConfig};
use crate::math::recall_at_k;
use crate::measure::{SampleMeasureError, SampleMeasurement, SampleMeasurer, SampleSweepResult};
use crate::model::AggregationConfig;
use crate::pacer::{DutyCyclePacer, PacerError};

const APPLICATION_NAME: &str = "vectorseam-seam";

/// Creating, driving, or closing one PostgreSQL connection failed.
#[derive(Debug, Error)]
pub(crate) enum DatabaseConnectionError {
    #[error("invalid data-source server {server:?}; expected host:port")]
    InvalidServer { server: String },
    #[error("password environment variable {name:?} disappeared after startup validation")]
    MissingPasswordEnvironment { name: String },
    #[error("password environment variable {name:?} is not valid Unicode")]
    NonUnicodePasswordEnvironment { name: String },
    #[error("invalid database duty-cycle configuration: {0}")]
    Pacer(#[from] PacerError),
    #[error("could not connect to PostgreSQL: {0}")]
    Connect(#[source] tokio_postgres::Error),
    #[error("PostgreSQL connection driver failed: {0}")]
    Driver(#[source] tokio_postgres::Error),
    #[error("PostgreSQL connection driver task failed: {0}")]
    DriverTask(#[source] tokio::task::JoinError),
}

/// The frozen MVP's single serialized connection and pacer for one data source.
///
/// The pacer's unit of work is one whole sample transaction: statements
/// inside a transaction run back-to-back, and the duty-cycle cooldown is
/// taken between transactions from the full transaction wall time.
pub(crate) struct DatabaseConnection {
    client: Option<Client>,
    driver: Option<JoinHandle<Result<(), tokio_postgres::Error>>>,
    pacer: DutyCyclePacer,
    statement_timeout: Duration,
    /// Client-side count of issued statements; read by acceptance tests.
    statement_count: u64,
    #[cfg(test)]
    sample_transaction_count: u64,
}

impl DatabaseConnection {
    /// Connects once and starts the owned Tokio task that drives the socket.
    pub(crate) async fn connect(
        data_source: &DataSourceConfig,
        budget: &BudgetConfig,
    ) -> Result<Self, DatabaseConnectionError> {
        let (host, port) = parse_server(&data_source.server)?;
        let mut config = PostgresConfig::new();
        config
            .host(&host)
            .port(port)
            .dbname(&data_source.database)
            .user(&data_source.user)
            .application_name(APPLICATION_NAME);
        if let Some(name) = data_source.password_env.as_deref() {
            let password = env::var(name).map_err(|error| match error {
                env::VarError::NotPresent => DatabaseConnectionError::MissingPasswordEnvironment {
                    name: name.to_owned(),
                },
                env::VarError::NotUnicode(_) => {
                    DatabaseConnectionError::NonUnicodePasswordEnvironment {
                        name: name.to_owned(),
                    }
                }
            })?;
            config.password(password);
        }

        let (client, connection) = config
            .connect(NoTls)
            .await
            .map_err(DatabaseConnectionError::Connect)?;
        let driver = tokio::spawn(connection);
        Ok(Self {
            client: Some(client),
            driver: Some(driver),
            pacer: DutyCyclePacer::new(budget.db_share)?,
            statement_timeout: budget.statement_timeout,
            statement_count: 0,
            #[cfg(test)]
            sample_transaction_count: 0,
        })
    }

    /// Closes the client and observes both layers of the driver task's result.
    pub(crate) async fn close(mut self) -> Result<(), DatabaseConnectionError> {
        drop(self.client.take());
        let Some(driver) = self.driver.take() else {
            return Ok(());
        };
        driver
            .await
            .map_err(DatabaseConnectionError::DriverTask)?
            .map_err(DatabaseConnectionError::Driver)
    }

    #[cfg(test)]
    pub(crate) fn statement_count(&self) -> u64 {
        self.statement_count
    }

    #[cfg(test)]
    pub(crate) fn sample_transaction_count(&self) -> u64 {
        self.sample_transaction_count
    }

    #[cfg(test)]
    async fn ground_truth_only(
        &mut self,
        vector: &[f32],
        config: &AggregationConfig,
    ) -> Result<(Vec<i64>, Vec<f64>), SampleMeasureError> {
        let statement_timeout = self.statement_timeout;
        let statements = &mut self.statement_count;
        let client = self.client.as_mut().ok_or_else(closed_sample_error)?;
        self.pacer
            .run(async {
                let transaction = begin_sample_transaction(statements, client).await?;
                let result = async {
                    execute_counted(
                        statements,
                        &transaction,
                        &statement_timeout_sql(statement_timeout),
                    )
                    .await?;
                    execute_counted(statements, &transaction, "SET LOCAL enable_indexscan = off")
                        .await?;
                    let vector = Vector::from(vector.to_vec());
                    *statements = statements.saturating_add(1);
                    let rows = transaction
                        .query(&ground_truth_sql(config), &[&vector])
                        .await
                        .map_err(database_sample_error)?;
                    decode_ground_truth(&rows)
                }
                .await;
                finish_transaction(statements, transaction, result).await
            })
            .await
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        if let Some(driver) = self.driver.take() {
            driver.abort();
        }
    }
}

#[async_trait]
impl SampleMeasurer for DatabaseConnection {
    async fn measure_sample(
        &mut self,
        vector: &[f32],
        config: &AggregationConfig,
    ) -> Result<SampleMeasurement, SampleMeasureError> {
        #[cfg(test)]
        {
            self.sample_transaction_count += 1;
        }
        let statement_timeout = self.statement_timeout;
        let statements = &mut self.statement_count;
        let client = self.client.as_mut().ok_or_else(closed_sample_error)?;
        self.pacer
            .run(run_sample_transaction(
                client,
                statements,
                statement_timeout,
                vector,
                config,
            ))
            .await
    }
}

/// One whole paced unit: `BEGIN` through `COMMIT`/`ROLLBACK`, back-to-back.
async fn run_sample_transaction(
    client: &mut Client,
    statements: &mut u64,
    statement_timeout: Duration,
    vector: &[f32],
    config: &AggregationConfig,
) -> Result<SampleMeasurement, SampleMeasureError> {
    let transaction = begin_sample_transaction(statements, client).await?;
    let result =
        measure_in_transaction(statements, statement_timeout, &transaction, vector, config).await;
    finish_transaction(statements, transaction, result).await
}

async fn measure_in_transaction(
    statements: &mut u64,
    statement_timeout: Duration,
    transaction: &Transaction<'_>,
    vector: &[f32],
    config: &AggregationConfig,
) -> Result<SampleMeasurement, SampleMeasureError> {
    let timeout_sql = statement_timeout_sql(statement_timeout);
    execute_counted(statements, transaction, &timeout_sql).await?;
    execute_counted(statements, transaction, "SET LOCAL enable_indexscan = off").await?;

    let query_vector = Vector::from(vector.to_vec());
    *statements = statements.saturating_add(1);
    let truth_rows = transaction
        .query(&ground_truth_sql(config), &[&query_vector])
        .await
        .map_err(database_sample_error)?;
    if truth_rows.len() < config.k as usize {
        return Err(SampleMeasureError::TableSmallerThanK {
            returned: truth_rows.len(),
            k: config.k,
        });
    }
    let (gt_keys, gt_distances) = decode_ground_truth(&truth_rows)?;

    execute_counted(statements, transaction, "SET LOCAL enable_indexscan = on").await?;
    let ann_sql = ann_sql(config);
    let mut sweeps = Vec::with_capacity(config.ef_grid.len());
    for &ef in &config.ef_grid {
        execute_counted(
            statements,
            transaction,
            &format!("SET LOCAL hnsw.ef_search = {ef}"),
        )
        .await?;
        *statements = statements.saturating_add(1);
        let started = Instant::now();
        let rows = transaction
            .query(&ann_sql, &[&query_vector])
            .await
            .map_err(database_sample_error)?;
        let elapsed = started.elapsed();
        let returned_keys = decode_keys(&rows)?;
        let recall = recall_at_k(&gt_keys, &returned_keys, config.k)
            .map_err(|error| SampleMeasureError::Database(error.to_string()))?;
        sweeps.push(SampleSweepResult {
            ef,
            returned_keys,
            recall,
            latency_ms: elapsed.as_secs_f64() * 1_000.0,
        });
    }
    Ok(SampleMeasurement {
        gt_keys,
        gt_distances,
        sweeps,
    })
}

async fn begin_sample_transaction<'client>(
    statements: &mut u64,
    client: &'client mut Client,
) -> Result<Transaction<'client>, SampleMeasureError> {
    *statements = statements.saturating_add(1);
    client
        .build_transaction()
        .isolation_level(IsolationLevel::RepeatableRead)
        .start()
        .await
        .map_err(database_sample_error)
}

fn closed_sample_error() -> SampleMeasureError {
    SampleMeasureError::Database("PostgreSQL connection is closed".to_owned())
}

async fn execute_counted(
    statements: &mut u64,
    transaction: &Transaction<'_>,
    sql: &str,
) -> Result<(), SampleMeasureError> {
    *statements = statements.saturating_add(1);
    transaction
        .batch_execute(sql)
        .await
        .map_err(database_sample_error)
}

async fn finish_transaction<T>(
    statements: &mut u64,
    transaction: Transaction<'_>,
    result: Result<T, SampleMeasureError>,
) -> Result<T, SampleMeasureError> {
    *statements = statements.saturating_add(1);
    match result {
        Ok(value) => {
            transaction.commit().await.map_err(database_sample_error)?;
            Ok(value)
        }
        Err(primary) => match transaction.rollback().await {
            Ok(()) => Err(primary),
            Err(rollback) if matches!(&primary, SampleMeasureError::TableSmallerThanK { .. }) => {
                warn!(
                    %rollback,
                    "rollback failed after table-smaller-than-k detection; preserving cohort abort"
                );
                Err(primary)
            }
            Err(rollback) => Err(SampleMeasureError::Database(format!(
                "{primary}; transaction rollback also failed: {rollback}"
            ))),
        },
    }
}

fn database_sample_error(error: tokio_postgres::Error) -> SampleMeasureError {
    SampleMeasureError::Database(error.to_string())
}

fn decode_ground_truth(rows: &[Row]) -> Result<(Vec<i64>, Vec<f64>), SampleMeasureError> {
    let keys = decode_keys(rows)?;
    let distances = rows
        .iter()
        .map(|row| row.try_get::<_, f64>(1).map_err(database_sample_error))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((keys, distances))
}

fn decode_keys(rows: &[Row]) -> Result<Vec<i64>, SampleMeasureError> {
    rows.iter().map(decode_key).collect()
}

fn decode_key(row: &Row) -> Result<i64, SampleMeasureError> {
    match *row.columns()[0].type_() {
        Type::INT2 => row
            .try_get::<_, i16>(0)
            .map(i64::from)
            .map_err(database_sample_error),
        Type::INT4 => row
            .try_get::<_, i32>(0)
            .map(i64::from)
            .map_err(database_sample_error),
        Type::INT8 => row.try_get::<_, i64>(0).map_err(database_sample_error),
        ref observed => Err(SampleMeasureError::Database(format!(
            "key column must be int2, int4, or int8; PostgreSQL returned {observed}"
        ))),
    }
}

fn parse_server(server: &str) -> Result<(String, u16), DatabaseConnectionError> {
    let (host, port) =
        server
            .rsplit_once(':')
            .ok_or_else(|| DatabaseConnectionError::InvalidServer {
                server: server.to_owned(),
            })?;
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    let port = port
        .parse::<u16>()
        .map_err(|_error| DatabaseConnectionError::InvalidServer {
            server: server.to_owned(),
        })?;
    if host.is_empty() {
        return Err(DatabaseConnectionError::InvalidServer {
            server: server.to_owned(),
        });
    }
    Ok((host.to_owned(), port))
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn ground_truth_sql(config: &AggregationConfig) -> String {
    let key = quote_identifier(&config.key);
    let column = quote_identifier(&config.column);
    let table = quote_identifier(&config.table);
    format!(
        "SELECT {key}, {column} <=> $1::vector AS distance \
         FROM {table} \
         ORDER BY {column} <=> $1::vector ASC, {key} ASC \
         LIMIT {}",
        config.k
    )
}

fn ann_sql(config: &AggregationConfig) -> String {
    let key = quote_identifier(&config.key);
    let column = quote_identifier(&config.column);
    let table = quote_identifier(&config.table);
    format!(
        "SELECT {key} FROM {table} \
         ORDER BY {column} <=> $1::vector ASC \
         LIMIT {}",
        config.k
    )
}

fn statement_timeout_sql(timeout: Duration) -> String {
    let timeout_ms = timeout.as_nanos().div_ceil(1_000_000);
    format!("SET LOCAL statement_timeout = '{timeout_ms}ms'")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{BudgetConfig, DataSourceConfig};

    fn aggregation_config() -> AggregationConfig {
        AggregationConfig {
            cohort: "acceptance/b2".to_owned(),
            target_name: "recall".to_owned(),
            index: "docs".to_owned(),
            table: "doc\"s".to_owned(),
            column: "embedding".to_owned(),
            key: "doc_id".to_owned(),
            k: 10,
            value: 0.95,
            percentile: 0.95,
            window_duration_seconds: 3_600,
            storage_window_seconds: 600,
            ef_grid: vec![20, 40],
            train_fraction: 0.8,
            split_seed: 42,
            min_samples: 100,
        }
    }

    #[test]
    fn b2_ground_truth_sql_quotes_identifiers_and_tie_breaks_by_key() {
        assert_eq!(
            ground_truth_sql(&aggregation_config()),
            "SELECT \"doc_id\", \"embedding\" <=> $1::vector AS distance \
             FROM \"doc\"\"s\" \
             ORDER BY \"embedding\" <=> $1::vector ASC, \"doc_id\" ASC \
             LIMIT 10"
        );
    }

    #[test]
    fn ann_query_has_no_key_tie_break() {
        let sql = ann_sql(&aggregation_config());
        assert_eq!(
            sql,
            "SELECT \"doc_id\" FROM \"doc\"\"s\" \
             ORDER BY \"embedding\" <=> $1::vector ASC \
             LIMIT 10"
        );
        assert!(!sql.contains("\"doc_id\" ASC"));
    }

    #[test]
    fn statement_timeout_rounds_up_to_nonzero_milliseconds() {
        assert_eq!(
            statement_timeout_sql(Duration::from_micros(1)),
            "SET LOCAL statement_timeout = '1ms'"
        );
        assert_eq!(
            statement_timeout_sql(Duration::from_secs(5)),
            "SET LOCAL statement_timeout = '5000ms'"
        );
    }

    #[test]
    fn server_parser_accepts_dns_and_bracketed_ipv6() {
        assert_eq!(
            parse_server("localhost:5432").unwrap(),
            ("localhost".to_owned(), 5432)
        );
        assert_eq!(
            parse_server("[::1]:5433").unwrap(),
            ("::1".to_owned(), 5433)
        );
    }

    #[tokio::test]
    async fn b2_f_pg_ground_truth_tie_break_prefers_key_7_over_9() {
        if std::env::var_os("SEAM_REQUIRE_F_PG").is_none() {
            return;
        }
        let port = std::env::var("SEAM_PG_PORT").unwrap_or_else(|_| "55432".to_owned());
        let data_source = DataSourceConfig {
            server: format!("127.0.0.1:{port}"),
            database: "postgres".to_owned(),
            user: "postgres".to_owned(),
            password_env: Some("SEAM_TEST_PG_PASSWORD".to_owned()),
        };
        let budget = BudgetConfig {
            db_share: 1.0,
            statement_timeout: Duration::from_secs(5),
        };
        let mut connection = DatabaseConnection::connect(&data_source, &budget)
            .await
            .expect("B2 F-pg data source must connect");
        let mut config = aggregation_config();
        config.table = "docs_seam_tie_fixture".to_owned();
        config.ef_grid = vec![20];
        let mut vector = vec![0.0_f32; 64];
        vector[0] = 1.0;

        for _ in 0..3 {
            let (keys, _distances) = connection
                .ground_truth_only(&vector, &config)
                .await
                .expect("B2 ground-truth transaction must succeed");
            assert!(keys.contains(&7));
            assert!(!keys.contains(&9));
        }
        connection.close().await.unwrap();
    }
}
