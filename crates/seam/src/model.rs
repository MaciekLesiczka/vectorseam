//! Pure Phase B input, intermediate, and round-output data contracts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use vectorseam_core::cohort::CohortName;

use crate::config::Config;

/// Configuration values needed by one cohort's pure aggregation.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregationConfig {
    /// Exact cohort name.
    pub cohort: String,
    /// Named target.
    pub target_name: String,
    /// Named index.
    pub index: String,
    /// Configured table identifier.
    pub table: String,
    /// Configured vector-column identifier.
    pub column: String,
    /// Configured key-column identifier.
    pub key: String,
    /// Recall denominator.
    pub k: u32,
    /// Recall threshold.
    pub value: f64,
    /// Required compliant population fraction.
    pub percentile: f64,
    /// Rolling target duration in seconds.
    pub window_duration_seconds: u64,
    /// Collector storage-window duration in seconds.
    pub storage_window_seconds: u32,
    /// Ascending ef grid.
    pub ef_grid: Vec<i32>,
    /// Train fraction used by the deterministic split.
    pub train_fraction: f64,
    /// Deterministic split seed.
    pub split_seed: u64,
    /// Minimum deduplicated sample count.
    pub min_samples: usize,
}

impl AggregationConfig {
    /// Projects one validated top-level config into pure cohort aggregation.
    ///
    /// Returns `None` only when the exact cohort is not configured. Startup
    /// validation guarantees that configured index and target references exist.
    pub fn for_cohort(config: &Config, cohort: &CohortName) -> Option<Self> {
        let cohort_config = config.cohorts.get(cohort)?;
        let index = config.indexes.get(&cohort_config.index)?;
        let target = config.targets.get(&cohort_config.target)?;
        Some(Self {
            cohort: cohort.to_string(),
            target_name: cohort_config.target.clone(),
            index: cohort_config.index.clone(),
            table: index.table.clone(),
            column: index.column.clone(),
            key: index.key.clone(),
            k: target.k,
            value: target.value,
            percentile: target.percentile,
            window_duration_seconds: target.window.as_secs(),
            storage_window_seconds: config.storage.window_seconds,
            ef_grid: config.calibration.ef_search.clone(),
            train_fraction: config.calibration.train_fraction,
            split_seed: config.calibration.split_seed,
            min_samples: config.calibration.min_samples,
        })
    }
}

/// Header values from one listed `.vseam` part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedPart {
    /// Stable part ULID string.
    pub part_ulid: String,
    /// Storage-window start as Unix seconds.
    pub window_start: u64,
    /// Header window duration.
    pub window_seconds: u32,
    /// Frames accepted before collector-side overflow drops.
    pub received_frame_count: u64,
    /// Frames durably written into the segment.
    pub record_count: u64,
}

/// Parquet metadata needed for compatibility and accounting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntermediateMetadata {
    /// Storage format version.
    pub format_version: u32,
    /// Exact cohort name.
    pub cohort: String,
    /// Stable part ULID string.
    pub part_ulid: String,
    /// Storage-window start as Unix seconds.
    pub window_start: u64,
    /// Header window duration.
    pub window_seconds: u32,
    /// Segment received-frame count copied at measure time.
    pub received_frame_count: u64,
    /// Segment record count copied at measure time.
    pub record_count: u64,
    /// Configured index name used during measurement.
    pub index: String,
    /// Configured table identifier used during measurement.
    pub table: String,
    /// Configured vector-column identifier used during measurement.
    pub column: String,
    /// Configured key-column identifier used during measurement.
    pub key: String,
    /// Recall denominator used during measurement.
    pub k: u32,
    /// ef grid used during measurement.
    pub ef_grid: Vec<i32>,
    /// Failed samples in this part.
    pub failed_count: u64,
    /// Successfully measured distinct samples in this part.
    pub measured_count: u64,
    /// Measurement completion timestamp in Unix microseconds.
    pub computed_at_us: u64,
}

/// One authoritative stored sweep observation.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepMeasurement {
    /// Stored recall, authoritative for Phase B.
    pub recall: f64,
    /// Client-observed statement latency.
    pub latency_ms: f64,
}

/// One successfully measured distinct vector within a part.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredSample {
    /// First-occurrence record ordinal in the part.
    pub record_index: i32,
    /// FNV-1a hash of the exact raw f32 bytes.
    pub vector_hash: u64,
    /// Count of equal vectors within this part.
    pub dup_count: i32,
    /// One observation for every configured ef value.
    pub sweeps: BTreeMap<i32, SweepMeasurement>,
}

/// One durable truth/sweep intermediate pair.
#[derive(Clone, Debug, PartialEq)]
pub struct IntermediatePart {
    /// Pair metadata.
    pub metadata: IntermediateMetadata,
    /// Successfully measured distinct samples.
    pub samples: Vec<MeasuredSample>,
}

/// Pure aggregation inputs for one cohort and round end.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregationInput {
    /// Cohort-specific aggregation configuration.
    pub config: AggregationConfig,
    /// Aligned, closed round end as Unix seconds.
    pub round_end: u64,
    /// Caller-supplied presentation timestamp; Phase B never reads a clock.
    pub computed_at: String,
    /// Optional cohort-level Phase A abort that forces an insufficient round.
    pub phase_a_abort: Option<PhaseAAbort>,
    /// Incompatible pairs Phase A observed and replaced before aggregation.
    pub phase_a_incompatible_parts: u64,
    /// All listed part headers that may overlap the round.
    pub listed_parts: Vec<ListedPart>,
    /// All durable intermediate pairs that may overlap the round.
    pub intermediates: Vec<IntermediatePart>,
}

/// A cohort-level Phase A abort with frozen publication semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseAAbort {
    /// Ground truth returned fewer than the target's `k` visible rows.
    TableSmallerThanK {
        /// Human-readable error published in the round record.
        error: String,
    },
}

impl PhaseAAbort {
    pub(crate) fn error(&self) -> &str {
        match self {
            Self::TableSmallerThanK { error } => error,
        }
    }
}

/// Published round status.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundStatus {
    /// The smallest clearing ef was selected.
    Ok,
    /// No ef cleared, so the maximum grid value was selected.
    TargetUnmet,
    /// Selection was refused due to population/split size or a Phase A abort.
    InsufficientSamples,
}

/// Published target description.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoundTarget {
    /// Target config name.
    pub name: String,
    /// Recall denominator.
    pub k: u32,
    /// Required recall value.
    pub value: f64,
    /// Required compliant fraction.
    pub percentile: f64,
}

/// Published rolling-window description.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RoundWindow {
    /// Inclusive ISO-8601 start.
    pub start: String,
    /// Exclusive ISO-8601 end.
    pub end: String,
    /// Rolling duration in seconds.
    pub duration_seconds: u64,
}

/// Published sample counters.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct SampleCounts {
    /// Sum of distinct in-scope listed-part record counts.
    pub available: u64,
    /// Sum of compatible intermediate measured counts.
    pub measured: u64,
    /// Sum of compatible intermediate failed counts.
    pub failed: u64,
    /// Population after window-wide vector-hash deduplication.
    pub unique: u64,
    /// Deduplicated train count.
    pub train: u64,
    /// Deduplicated holdout count.
    pub test: u64,
}

/// Published storage-window coverage counters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Coverage {
    /// Fraction of expected in-scope windows without any listed part.
    pub empty_window_fraction: f64,
    /// Number of fully in-scope storage windows.
    pub windows_in_scope: u64,
    /// Number of those windows with at least one listed part.
    pub windows_with_parts: u64,
}

/// Informational full-population summary for one ef value.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PerEfSummary {
    /// ef value.
    pub ef: i32,
    /// Full-population compliance quantile.
    pub quantile_recall: f64,
    /// Full-population arithmetic mean recall.
    pub mean_recall: f64,
    /// Full-population type-7 median client latency.
    pub latency_p50_ms: f64,
}

/// Complete deterministic round record.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoundOutput {
    /// Round format version.
    pub format_version: u32,
    /// Exact cohort name.
    pub cohort: String,
    /// Caller-supplied computation timestamp.
    pub computed_at: String,
    /// Rolling window.
    pub window: RoundWindow,
    /// Target contract.
    pub target: RoundTarget,
    /// Configured index name.
    pub index: String,
    /// Ascending ef grid.
    pub ef_grid: Vec<i32>,
    /// Serialized as `"ok"`, `"target_unmet"`, or `"insufficient_samples"`.
    pub status: RoundStatus,
    /// Optional Phase A error.
    pub error: Option<String>,
    /// Selected ef, absent when insufficient.
    pub recommended_ef: Option<i32>,
    /// Holdout posterior confidence, absent when insufficient.
    pub confidence: Option<f64>,
    /// Whether holdout quantile transferred, absent when insufficient.
    pub transferred: Option<bool>,
    /// Train compliance quantile at the selected ef.
    pub train_quantile_recall: Option<f64>,
    /// Holdout compliance quantile at the selected ef.
    pub test_quantile_recall: Option<f64>,
    /// Sample counters.
    pub samples: SampleCounts,
    /// Collector-side drop fraction.
    pub dropped_frame_fraction: f64,
    /// Storage-window coverage.
    pub coverage: Coverage,
    /// Compatible in-scope intermediate pairs used.
    pub parts_used: u64,
    /// In-scope intermediate pairs skipped for config mismatch.
    pub incompatible_parts: u64,
    /// Informational summaries over the full deduplicated population.
    pub per_ef: Vec<PerEfSummary>,
}
