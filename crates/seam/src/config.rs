//! Tuner configuration parsing and startup validation.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;
use vectorseam_core::cohort::CohortName;

const POSTGRES_IDENTIFIER_MAX_BYTES: usize = 63;

/// A fully parsed and startup-validated tuner configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Calibration and estimator settings.
    pub calibration: CalibrationConfig,
    /// Shared object-store layout settings.
    pub storage: StorageConfig,
    /// Database traffic controls.
    pub budget: BudgetConfig,
    /// Named PostgreSQL connection data sources.
    pub data_sources: BTreeMap<String, DataSourceConfig>,
    /// Named PostgreSQL indexes.
    pub indexes: BTreeMap<String, IndexConfig>,
    /// Named recall targets.
    pub targets: BTreeMap<String, TargetConfig>,
    /// Exact configured cohorts.
    pub cohorts: BTreeMap<CohortName, CohortConfig>,
}

/// Calibration and estimator settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationConfig {
    /// Time between round ticks.
    pub interval: Duration,
    /// Ascending `hnsw.ef_search` sweep grid.
    pub ef_search: Vec<i32>,
    /// Deterministic train fraction.
    pub train_fraction: f64,
    /// Deterministic split seed.
    pub split_seed: u64,
    /// Minimum deduplicated population required for selection.
    pub min_samples: usize,
}

/// Object-store layout settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageConfig {
    /// Object-store root shared with the collector.
    pub root: PathBuf,
    /// Collector tumbling-window duration.
    pub window_seconds: u32,
}

/// Database traffic controls.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetConfig {
    /// Maximum database-transaction wall-time share per data source.
    pub db_share: f64,
    /// Per-statement PostgreSQL timeout.
    pub statement_timeout: Duration,
    /// Client deadline for connecting and for each PostgreSQL protocol operation.
    pub client_timeout: Duration,
}

/// One uniquely addressed PostgreSQL connection target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataSourceConfig {
    /// Server in `host:port` form.
    pub server: String,
    /// Database name.
    pub database: String,
    /// Database user.
    pub user: String,
    /// Optional environment variable containing the password.
    pub password_env: Option<String>,
}

/// One named PostgreSQL vector index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexConfig {
    /// Named data-source reference.
    pub data_source: String,
    /// Quoted PostgreSQL table identifier.
    pub table: String,
    /// Quoted PostgreSQL key-column identifier.
    pub key: String,
    /// Quoted PostgreSQL vector-column identifier.
    pub column: String,
}

/// One named recall target.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetConfig {
    /// Recall denominator and result count.
    pub k: u32,
    /// Required recall value.
    pub value: f64,
    /// Required compliant population fraction.
    pub percentile: f64,
    /// Rolling calibration-window duration.
    pub window: Duration,
}

/// References connecting an exact cohort to an index and target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CohortConfig {
    /// Named index reference.
    pub index: String,
    /// Named target reference.
    pub target: String,
}

/// Configuration parsing or validation failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    /// YAML could not be deserialized into the configuration schema.
    #[error("invalid tuner YAML: {0}")]
    Yaml(String),
    /// A startup validation rule was violated.
    #[error("invalid tuner configuration: {0}")]
    Validation(String),
}

impl Config {
    /// Parses YAML and applies every frozen-spec startup validation rule.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, ConfigError> {
        parse_with_password_env(yaml, |name| std::env::var_os(name).is_some())
    }
}

fn parse_with_password_env(
    yaml: &str,
    password_env_exists: impl Fn(&str) -> bool,
) -> Result<Config, ConfigError> {
    let raw: RawConfig =
        serde_saphyr::from_str(yaml).map_err(|error| ConfigError::Yaml(error.to_string()))?;
    Config::from_raw(raw, password_env_exists)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    calibration: RawCalibrationConfig,
    storage: RawStorageConfig,
    #[serde(default)]
    budget: RawBudgetConfig,
    data_sources: BTreeMap<String, RawDataSourceConfig>,
    indexes: BTreeMap<String, RawIndexConfig>,
    targets: BTreeMap<String, RawTargetConfig>,
    cohorts: BTreeMap<String, RawCohortConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCalibrationConfig {
    #[serde(deserialize_with = "deserialize_duration")]
    interval: Duration,
    ef_search: Vec<i32>,
    #[serde(default = "default_train_fraction")]
    train_fraction: f64,
    #[serde(default = "default_split_seed")]
    split_seed: u64,
    #[serde(default = "default_min_samples")]
    min_samples: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStorageConfig {
    root: PathBuf,
    #[serde(default = "default_window_seconds")]
    window_seconds: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBudgetConfig {
    #[serde(default = "default_db_share")]
    db_share: f64,
    #[serde(
        default = "default_statement_timeout",
        deserialize_with = "deserialize_duration"
    )]
    statement_timeout: Duration,
    #[serde(
        default = "default_client_timeout",
        deserialize_with = "deserialize_duration"
    )]
    client_timeout: Duration,
}

impl Default for RawBudgetConfig {
    fn default() -> Self {
        Self {
            db_share: default_db_share(),
            statement_timeout: default_statement_timeout(),
            client_timeout: default_client_timeout(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDataSourceConfig {
    server: String,
    database: String,
    user: String,
    password_env: Option<String>,
    #[serde(default)]
    password: InlinePassword,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndexConfig {
    data_source: String,
    table: String,
    key: String,
    column: String,
}

#[derive(Debug, Default)]
struct InlinePassword(bool);

impl<'de> Deserialize<'de> for InlinePassword {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let _discarded = serde::de::IgnoredAny::deserialize(deserializer)?;
        Ok(Self(true))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTargetConfig {
    k: u32,
    value: f64,
    percentile: f64,
    #[serde(deserialize_with = "deserialize_duration")]
    window: Duration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCohortConfig {
    index: String,
    target: String,
}

impl Config {
    fn from_raw(
        raw: RawConfig,
        password_env_exists: impl Fn(&str) -> bool,
    ) -> Result<Self, ConfigError> {
        validate_calibration(&raw.calibration)?;
        validate_storage(&raw.storage)?;
        validate_budget(&raw.budget)?;
        validate_data_sources(&raw.data_sources, password_env_exists)?;
        validate_indexes(&raw.indexes, &raw.data_sources)?;
        validate_targets(&raw.targets, raw.storage.window_seconds)?;
        validate_grid_against_targets(&raw.calibration.ef_search, &raw.targets)?;

        let cohorts = raw
            .cohorts
            .into_iter()
            .map(|(name, cohort)| {
                let validated_name = CohortName::try_from(name.as_str())
                    .map_err(|error| invalid(format!("cohort {name:?} is invalid: {error}")))?;
                if !raw.indexes.contains_key(&cohort.index) {
                    return Err(invalid(format!(
                        "cohort {name:?} references unknown index {:?}",
                        cohort.index
                    )));
                }
                if !raw.targets.contains_key(&cohort.target) {
                    return Err(invalid(format!(
                        "cohort {name:?} references unknown target {:?}",
                        cohort.target
                    )));
                }
                Ok((
                    validated_name,
                    CohortConfig {
                        index: cohort.index,
                        target: cohort.target,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, ConfigError>>()?;

        let data_sources = raw
            .data_sources
            .into_iter()
            .map(|(name, data_source)| {
                (
                    name,
                    DataSourceConfig {
                        server: data_source.server,
                        database: data_source.database,
                        user: data_source.user,
                        password_env: data_source.password_env,
                    },
                )
            })
            .collect();
        let indexes = raw
            .indexes
            .into_iter()
            .map(|(name, index)| {
                (
                    name,
                    IndexConfig {
                        data_source: index.data_source,
                        table: index.table,
                        key: index.key,
                        column: index.column,
                    },
                )
            })
            .collect();
        let targets = raw
            .targets
            .into_iter()
            .map(|(name, target)| {
                (
                    name,
                    TargetConfig {
                        k: target.k,
                        value: target.value,
                        percentile: target.percentile,
                        window: target.window,
                    },
                )
            })
            .collect();

        Ok(Self {
            calibration: CalibrationConfig {
                interval: raw.calibration.interval,
                ef_search: raw.calibration.ef_search,
                train_fraction: raw.calibration.train_fraction,
                split_seed: raw.calibration.split_seed,
                min_samples: raw.calibration.min_samples,
            },
            storage: StorageConfig {
                root: raw.storage.root,
                window_seconds: raw.storage.window_seconds,
            },
            budget: BudgetConfig {
                db_share: raw.budget.db_share,
                statement_timeout: raw.budget.statement_timeout,
                client_timeout: raw.budget.client_timeout,
            },
            data_sources,
            indexes,
            targets,
            cohorts,
        })
    }
}

fn validate_calibration(config: &RawCalibrationConfig) -> Result<(), ConfigError> {
    if config.interval.is_zero() {
        return Err(invalid("calibration.interval must be greater than zero"));
    }
    if config.ef_search.is_empty() {
        return Err(invalid("calibration.ef_search is required and non-empty"));
    }
    if config.ef_search.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid("calibration.ef_search must be strictly increasing"));
    }
    if config.ef_search.iter().any(|ef| *ef > 1000) {
        return Err(invalid("calibration.ef_search values must be <= 1000"));
    }
    if !(config.train_fraction > 0.0 && config.train_fraction < 1.0) {
        return Err(invalid("calibration.train_fraction must be in (0, 1)"));
    }
    let split_threshold = (config.train_fraction * 10_000.0).round();
    if !(1.0..=9_999.0).contains(&split_threshold) {
        return Err(invalid(
            "calibration.train_fraction rounds to an empty train/holdout split",
        ));
    }
    if config.min_samples < 100 {
        return Err(invalid("calibration.min_samples must be >= 100"));
    }
    Ok(())
}

fn validate_storage(config: &RawStorageConfig) -> Result<(), ConfigError> {
    if config.window_seconds == 0 {
        return Err(invalid("storage.window_seconds must be greater than zero"));
    }
    if config.window_seconds % 60 != 0 {
        return Err(invalid("storage.window_seconds must be a multiple of 60"));
    }
    Ok(())
}

fn validate_budget(config: &RawBudgetConfig) -> Result<(), ConfigError> {
    if !(config.db_share > 0.0 && config.db_share <= 1.0) {
        return Err(invalid("budget.db_share must be in (0, 1]"));
    }
    if config.statement_timeout.is_zero() {
        return Err(invalid(
            "budget.statement_timeout must be greater than zero",
        ));
    }
    if config.client_timeout.is_zero() {
        return Err(invalid("budget.client_timeout must be greater than zero"));
    }
    Ok(())
}

fn validate_data_sources(
    data_sources: &BTreeMap<String, RawDataSourceConfig>,
    password_env_exists: impl Fn(&str) -> bool,
) -> Result<(), ConfigError> {
    let mut pairs = BTreeMap::<(&str, &str), &str>::new();
    for (name, data_source) in data_sources {
        if data_source.password.0 {
            return Err(invalid(format!(
                "data source {name:?} contains inline password; use password_env"
            )));
        }
        for (field, value) in [
            ("server", data_source.server.as_str()),
            ("database", data_source.database.as_str()),
            ("user", data_source.user.as_str()),
        ] {
            if contains_inline_userinfo(value) {
                return Err(invalid(format!(
                    "data source {name:?} {field} contains inline userinfo; use password_env"
                )));
            }
        }
        match data_source.password_env.as_deref() {
            Some(password_env) if !password_env_exists(password_env) => {
                return Err(invalid(format!(
                    "data source {name:?} password_env {password_env:?} is not present"
                )));
            }
            _ => {}
        }
        let pair = (data_source.server.as_str(), data_source.database.as_str());
        if let Some(existing) = pairs.insert(pair, name.as_str()) {
            return Err(invalid(format!(
                "data sources {existing:?} and {name:?} duplicate (server, database) pair ({:?}, {:?})",
                data_source.server, data_source.database
            )));
        }
    }
    Ok(())
}

fn validate_indexes(
    indexes: &BTreeMap<String, RawIndexConfig>,
    data_sources: &BTreeMap<String, RawDataSourceConfig>,
) -> Result<(), ConfigError> {
    for (name, index) in indexes {
        if !data_sources.contains_key(&index.data_source) {
            return Err(invalid(format!(
                "index {name:?} references unknown data source {:?}",
                index.data_source
            )));
        }
        for (field, identifier) in [
            ("table", index.table.as_str()),
            ("column", index.column.as_str()),
            ("key", index.key.as_str()),
        ] {
            validate_postgres_identifier(name, field, identifier)?;
        }
    }
    Ok(())
}

fn contains_inline_userinfo(value: &str) -> bool {
    value
        .split_once('@')
        .is_some_and(|(userinfo, _rest)| userinfo.contains(':'))
}

fn validate_postgres_identifier(
    index_name: &str,
    field: &str,
    identifier: &str,
) -> Result<(), ConfigError> {
    if identifier.is_empty() {
        return Err(invalid(format!(
            "index {index_name:?} {field} PostgreSQL identifier must not be empty"
        )));
    }
    if identifier.len() > POSTGRES_IDENTIFIER_MAX_BYTES {
        return Err(invalid(format!(
            "index {index_name:?} {field} PostgreSQL identifier exceeds {POSTGRES_IDENTIFIER_MAX_BYTES} bytes"
        )));
    }
    if identifier.contains('\0') {
        return Err(invalid(format!(
            "index {index_name:?} {field} PostgreSQL identifier contains NUL"
        )));
    }
    Ok(())
}

fn validate_targets(
    targets: &BTreeMap<String, RawTargetConfig>,
    storage_window_seconds: u32,
) -> Result<(), ConfigError> {
    for (name, target) in targets {
        if target.k == 0 {
            return Err(invalid(format!("target {name:?} k must be >= 1")));
        }
        if !(target.value > 0.0 && target.value <= 1.0) {
            return Err(invalid(format!("target {name:?} value must be in (0, 1]")));
        }
        if !(target.percentile > 0.0 && target.percentile < 1.0) {
            return Err(invalid(format!(
                "target {name:?} percentile must be in (0, 1)"
            )));
        }
        if target.window.as_secs() < u64::from(storage_window_seconds) {
            return Err(invalid(format!(
                "target {name:?} window must be at least storage.window_seconds"
            )));
        }
        if target.window.subsec_nanos() != 0
            || target.window.as_secs() % u64::from(storage_window_seconds) != 0
        {
            return Err(invalid(format!(
                "target {name:?} window must be a multiple of storage.window_seconds"
            )));
        }
    }
    Ok(())
}

fn validate_grid_against_targets(
    ef_search: &[i32],
    targets: &BTreeMap<String, RawTargetConfig>,
) -> Result<(), ConfigError> {
    let minimum = ef_search
        .first()
        .copied()
        .ok_or_else(|| invalid("calibration.ef_search is required and non-empty"))?;
    for (name, target) in targets {
        if i64::from(minimum) < i64::from(target.k) {
            return Err(invalid(format!(
                "calibration.ef_search minimum {minimum} is below target {name:?} k {}",
                target.k
            )));
        }
    }
    Ok(())
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let text = String::deserialize(deserializer)?;
    humantime::parse_duration(&text).map_err(serde::de::Error::custom)
}

fn default_train_fraction() -> f64 {
    0.7
}

fn default_split_seed() -> u64 {
    7
}

fn default_min_samples() -> usize {
    1000
}

fn default_window_seconds() -> u32 {
    600
}

fn default_db_share() -> f64 {
    0.1
}

fn default_statement_timeout() -> Duration {
    Duration::from_secs(5)
}

fn default_client_timeout() -> Duration {
    Duration::from_secs(10)
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Validation(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_CONFIG: &str = r#"
calibration:
  interval: 10min
  ef_search: [20, 40, 80, 160]
storage:
  root: /tmp/vectorseam
  window_seconds: 600
data_sources:
  primary:
    server: localhost:5432
    database: postgres
    user: postgres
indexes:
  fixture:
    data_source: primary
    table: docs_fixture
    key: doc_id
    column: embedding
targets:
  recall:
    k: 20
    value: 0.9
    percentile: 0.95
    window: 1h
cohorts:
  acceptance/f-agg:
    index: fixture
    target: recall
"#;

    #[test]
    fn parses_defaults_and_quoted_identifiers() {
        let config = Config::from_yaml_str(VALID_CONFIG).unwrap();
        assert_eq!(config.calibration.train_fraction, 0.7);
        assert_eq!(config.calibration.split_seed, 7);
        assert_eq!(config.calibration.min_samples, 1000);
        assert_eq!(config.budget.statement_timeout, Duration::from_secs(5));
        assert_eq!(config.budget.client_timeout, Duration::from_secs(10));
        assert_eq!(config.indexes["fixture"].data_source, "primary");
        assert_eq!(config.data_sources["primary"].database, "postgres");

        let quoted = VALID_CONFIG.replace("docs_fixture", "odd.\"table");
        assert!(Config::from_yaml_str(&quoted).is_ok());
    }

    #[test]
    fn rejects_guaranteed_empty_split_threshold() {
        let yaml = VALID_CONFIG.replace(
            "  ef_search: [20, 40, 80, 160]",
            "  ef_search: [20, 40, 80, 160]\n  train_fraction: 0.00001",
        );
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();
        assert!(error.contains("empty train/holdout split"));
    }

    #[test]
    fn rejects_inline_password_key_even_when_null() {
        let yaml = VALID_CONFIG.replace("    server:", "    password: null\n    server:");
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();
        assert!(error.contains("password_env"));
    }

    #[test]
    fn c5_missing_password_env_is_rejected_only_when_configured() {
        assert!(parse_with_password_env(VALID_CONFIG, |_name| false).is_ok());

        let yaml = VALID_CONFIG.replace(
            "    user: postgres",
            "    user: postgres\n    password_env: SEAM_PG_MISSING",
        );
        let error = parse_with_password_env(&yaml, |_name| false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("SEAM_PG_MISSING"));
        assert!(error.contains("not present"));
        assert!(parse_with_password_env(&yaml, |name| name == "SEAM_PG_MISSING").is_ok());
    }

    #[test]
    fn c5_duplicate_data_source_pair_is_rejected() {
        let yaml = VALID_CONFIG.replace(
            "indexes:",
            "  secondary:\n    server: localhost:5432\n    database: postgres\n    user: maciek\nindexes:",
        );
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();

        assert!(error.contains("duplicate (server, database) pair"));
        assert!(error.contains("primary"));
        assert!(error.contains("secondary"));
    }

    #[test]
    fn distinct_data_source_pairs_are_allowed() {
        let yaml = VALID_CONFIG.replace(
            "indexes:",
            "  secondary:\n    server: localhost:5432\n    database: maciek\n    user: maciek\nindexes:",
        );

        assert!(Config::from_yaml_str(&yaml).is_ok());
    }

    #[test]
    fn removed_max_concurrent_queries_is_rejected() {
        let yaml =
            VALID_CONFIG.replace("storage:", "budget:\n  max_concurrent_queries: 1\nstorage:");
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();

        assert!(error.contains("max_concurrent_queries"));
    }

    #[test]
    fn rejects_zero_client_timeout() {
        let yaml = VALID_CONFIG.replace("storage:", "budget:\n  client_timeout: 0s\nstorage:");
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();

        assert!(error.contains("client_timeout"));
        assert!(error.contains("greater than zero"));
    }

    #[test]
    fn rejects_non_minute_storage_window() {
        let yaml = VALID_CONFIG.replace("window_seconds: 600", "window_seconds: 610");
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();
        assert!(error.contains("multiple of 60"));
    }

    #[test]
    fn rejects_target_window_not_divisible_by_storage_window() {
        let yaml = VALID_CONFIG.replace("window: 1h", "window: 15min");
        let error = Config::from_yaml_str(&yaml).unwrap_err().to_string();
        assert!(error.contains("multiple of storage.window_seconds"));
    }
}
