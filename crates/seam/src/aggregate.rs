//! Pure, deterministic Phase B aggregation.

use std::collections::BTreeMap;

use thiserror::Error;
use vectorseam_core::window::{WindowError, format_window_timestamp};

use crate::accounting::{
    coverage, dropped_frame_fraction, in_scope_window_starts, unique_in_scope_parts,
};
use crate::math::{MathError, is_train_member, split_threshold};
use crate::model::{
    AggregationConfig, AggregationInput, EffectiveRecommendation, IntermediatePart, RoundOutput,
    RoundStatus, RoundTarget, RoundWindow, SampleCounts,
};
use crate::population::{
    deduplicate_samples, per_ef_summaries, select_and_validate, validate_population,
};

/// Invalid or internally inconsistent Phase B input.
#[derive(Debug, Error)]
pub enum AggregateError {
    /// A frozen numeric primitive rejected its input.
    #[error(transparent)]
    Math(#[from] MathError),
    /// Window arithmetic or timestamp formatting failed.
    #[error(transparent)]
    Window(#[from] WindowError),
    /// The round end precedes the rolling-window duration.
    #[error("round_end precedes the configured rolling-window duration")]
    WindowUnderflow,
    /// The caller supplied a round end not aligned to the storage window.
    #[error("round_end must align to storage.window_seconds")]
    UnalignedRoundEnd,
    /// A directly constructed aggregation config violated startup invariants.
    #[error("invalid aggregation config: {0}")]
    InvalidConfig(String),
    /// A listed part ULID appeared with different header values.
    #[error("listed part {0:?} appears with conflicting headers")]
    ConflictingListedPart(String),
    /// An intermediate part ULID appeared with different pair contents.
    #[error("intermediate part {0:?} appears with conflicting contents")]
    ConflictingIntermediate(String),
    /// A part header contained impossible accounting values.
    #[error("part {0:?} has record_count greater than received_frame_count")]
    InvalidPartCounts(String),
    /// A published counter exceeded `u64`.
    #[error("round counter overflow while accumulating {0}")]
    CounterOverflow(&'static str),
    /// Part window end overflowed Unix-second arithmetic.
    #[error("part {0:?} window end overflows Unix seconds")]
    PartWindowOverflow(String),
    /// A compatible sample did not contain exactly the configured ef grid.
    #[error("part {part_ulid:?} record {record_index} is missing ef {ef}")]
    MissingSweep {
        /// Part identity.
        part_ulid: String,
        /// Record ordinal.
        record_index: i32,
        /// Missing ef value.
        ef: i32,
    },
    /// Stored recall or latency was non-finite or outside its valid domain.
    #[error("part {part_ulid:?} record {record_index} ef {ef} has invalid stored observation")]
    InvalidObservation {
        /// Part identity.
        part_ulid: String,
        /// Record ordinal.
        record_index: i32,
        /// ef value.
        ef: i32,
    },
    /// Stored ground-truth latency was non-finite or negative.
    #[error("part {part_ulid:?} record {record_index} has invalid stored ground-truth latency")]
    InvalidGroundTruthLatency {
        /// Part identity.
        part_ulid: String,
        /// Record ordinal.
        record_index: i32,
    },
    /// Round JSON serialization failed.
    #[error("serialize deterministic round JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Aggregates one cohort round as a pure function of inputs.
pub fn aggregate(input: &AggregationInput) -> Result<RoundOutput, AggregateError> {
    validate_aggregation_config(&input.config)?;
    let expected_windows = in_scope_window_starts(
        input.round_end,
        input.config.window_duration_seconds,
        input.config.storage_window_seconds,
    )?;
    let lower = input
        .round_end
        .checked_sub(input.config.window_duration_seconds)
        .ok_or(AggregateError::WindowUnderflow)?;
    let listed = unique_in_scope_parts(&input.listed_parts, lower, input.round_end)?;
    let coverage = coverage(&expected_windows, &listed);

    let available = listed.values().try_fold(0_u64, |total, part| {
        if part.record_count > part.received_frame_count {
            return Err(AggregateError::InvalidPartCounts(part.part_ulid.clone()));
        }
        total
            .checked_add(part.record_count)
            .ok_or(AggregateError::CounterOverflow("samples.available"))
    })?;
    let drop_fraction = dropped_frame_fraction(listed.values().map(|part| {
        (
            part.part_ulid.as_str(),
            part.received_frame_count,
            part.record_count,
        )
    }))?;

    let intermediates = unique_intermediates(&input.intermediates)?;
    let mut used = Vec::new();
    let mut incompatible_parts = input.phase_a_incompatible_parts;
    let mut measured = 0_u64;
    let mut failed = 0_u64;
    for (part_ulid, intermediate) in intermediates {
        if !listed.contains_key(part_ulid) {
            continue;
        }
        if !is_compatible(&input.config, intermediate) {
            incompatible_parts = incompatible_parts
                .checked_add(1)
                .ok_or(AggregateError::CounterOverflow("incompatible_parts"))?;
            continue;
        }
        measured = measured
            .checked_add(intermediate.metadata.measured_count)
            .ok_or(AggregateError::CounterOverflow("samples.measured"))?;
        failed = failed
            .checked_add(intermediate.metadata.failed_count)
            .ok_or(AggregateError::CounterOverflow("samples.failed"))?;
        used.push(intermediate);
    }

    let population = deduplicate_samples(used.iter().flat_map(|part| {
        part.samples
            .iter()
            .map(move |sample| (part.metadata.part_ulid.as_str(), sample))
    }));
    validate_population(&population, &input.config.ef_grid)?;
    let ground_truth_latency_mean_ms = (!population.is_empty()).then(|| {
        population
            .iter()
            .map(|sample| sample.ground_truth_latency_ms)
            .sum::<f64>()
            / population.len() as f64
    });
    let per_ef = per_ef_summaries(&population, &input.config)?;

    let mut train = Vec::new();
    let mut test = Vec::new();
    for sample in &population {
        if is_train_member(
            sample.vector_hash,
            input.config.split_seed,
            input.config.train_fraction,
        )? {
            train.push(sample);
        } else {
            test.push(sample);
        }
    }

    let samples = SampleCounts {
        available,
        measured,
        failed,
        unique: population.len() as u64,
        train: train.len() as u64,
        test: test.len() as u64,
    };
    let insufficient = input.phase_a_abort.is_some()
        || population.len() < input.config.min_samples
        || train.is_empty()
        || test.is_empty();

    let selection = if insufficient {
        None
    } else {
        Some(select_and_validate(&input.config, &train, &test)?)
    };
    let (status, recommended_ef, confidence, transferred, train_quantile, test_quantile) =
        match selection {
            None => (
                RoundStatus::InsufficientSamples,
                None,
                None,
                None,
                None,
                None,
            ),
            Some(selection) => (
                selection.status,
                Some(selection.recommended_ef),
                Some(selection.confidence),
                Some(selection.transferred),
                Some(selection.train_quantile),
                Some(selection.test_quantile),
            ),
        };
    let window_end = iso8601_seconds(input.round_end)?;
    let effective = match (recommended_ef, confidence) {
        (Some(recommended_ef), Some(confidence)) => Some(EffectiveRecommendation {
            recommended_ef,
            confidence,
            source_round: window_end.clone(),
            carried: false,
        }),
        (None, None) => input
            .previous_round
            .as_ref()
            .filter(|previous| carry_fingerprint_matches(&input.config, previous))
            .and_then(|previous| previous.effective.clone())
            .map(|mut effective| {
                effective.carried = true;
                effective
            }),
        _ => unreachable!("selection always emits recommendation and confidence together"),
    };
    Ok(RoundOutput {
        format_version: 1,
        cohort: input.config.cohort.clone(),
        computed_at: input.computed_at.clone(),
        window: RoundWindow {
            start: iso8601_seconds(lower)?,
            end: window_end,
            duration_seconds: input.config.window_duration_seconds,
        },
        target: RoundTarget {
            name: input.config.target_name.clone(),
            k: input.config.k,
            value: input.config.value,
            percentile: input.config.percentile,
        },
        index: input.config.index.clone(),
        ef_grid: input.config.ef_grid.clone(),
        status,
        error: input
            .phase_a_abort
            .as_ref()
            .map(|abort| abort.error().to_owned()),
        recommended_ef,
        confidence,
        transferred,
        train_quantile_recall: train_quantile,
        test_quantile_recall: test_quantile,
        effective,
        samples,
        dropped_frame_fraction: drop_fraction,
        coverage,
        parts_used: used.len() as u64,
        incompatible_parts,
        ground_truth_latency_mean_ms,
        per_ef,
    })
}

fn carry_fingerprint_matches(config: &AggregationConfig, previous: &RoundOutput) -> bool {
    previous.cohort == config.cohort
        && previous.index == config.index
        && previous.ef_grid == config.ef_grid
        && previous.target.k == config.k
        && previous.target.value == config.value
        && previous.target.percentile == config.percentile
}

/// Serializes a round record with deterministic struct-field ordering.
pub fn round_json_bytes(output: &RoundOutput) -> Result<Vec<u8>, AggregateError> {
    Ok(serde_json::to_vec(output)?)
}

fn unique_intermediates(
    parts: &[IntermediatePart],
) -> Result<BTreeMap<&str, &IntermediatePart>, AggregateError> {
    let mut unique = BTreeMap::new();
    for part in parts {
        if let Some(existing) = unique.insert(part.metadata.part_ulid.as_str(), part) {
            if existing != part {
                return Err(AggregateError::ConflictingIntermediate(
                    part.metadata.part_ulid.clone(),
                ));
            }
        }
    }
    Ok(unique)
}

pub(crate) fn is_compatible(config: &AggregationConfig, part: &IntermediatePart) -> bool {
    let metadata = &part.metadata;
    metadata.format_version == 1
        && metadata.k == config.k
        && metadata.index == config.index
        && metadata.table == config.table
        && metadata.column == config.column
        && metadata.key == config.key
        && metadata.ef_grid == config.ef_grid
}

fn validate_aggregation_config(config: &AggregationConfig) -> Result<(), AggregateError> {
    if config.storage_window_seconds == 0 {
        return Err(AggregateError::InvalidConfig(
            "storage_window_seconds must be greater than zero".to_owned(),
        ));
    }
    if config.storage_window_seconds % 60 != 0 {
        return Err(AggregateError::InvalidConfig(
            "storage_window_seconds must be a multiple of 60".to_owned(),
        ));
    }
    if config.window_duration_seconds < u64::from(config.storage_window_seconds) {
        return Err(AggregateError::InvalidConfig(
            "window duration must be at least one storage window".to_owned(),
        ));
    }
    if config.window_duration_seconds % u64::from(config.storage_window_seconds) != 0 {
        return Err(AggregateError::InvalidConfig(
            "window duration must be a multiple of storage_window_seconds".to_owned(),
        ));
    }
    if config.k == 0 {
        return Err(AggregateError::InvalidConfig("k must be >= 1".to_owned()));
    }
    if !(config.value > 0.0 && config.value <= 1.0) {
        return Err(AggregateError::InvalidConfig(
            "value must be in (0, 1]".to_owned(),
        ));
    }
    if !(config.percentile > 0.0 && config.percentile < 1.0) {
        return Err(AggregateError::InvalidConfig(
            "percentile must be in (0, 1)".to_owned(),
        ));
    }
    if config.ef_grid.is_empty()
        || config.ef_grid.windows(2).any(|pair| pair[0] >= pair[1])
        || config.ef_grid[0] < i32::try_from(config.k).unwrap_or(i32::MAX)
        || config.ef_grid.iter().any(|ef| *ef > 1000)
    {
        return Err(AggregateError::InvalidConfig(
            "ef_grid must be non-empty, strictly increasing, >= k, and <= 1000".to_owned(),
        ));
    }
    split_threshold(config.train_fraction)?;
    if config.min_samples < 100 {
        return Err(AggregateError::InvalidConfig(
            "min_samples must be >= 100".to_owned(),
        ));
    }
    Ok(())
}

fn iso8601_seconds(unix_seconds: u64) -> Result<String, WindowError> {
    let compact = format_window_timestamp(unix_seconds)?;
    Ok(format!(
        "{}-{}-{}T{}:{}:00Z",
        &compact[0..4],
        &compact[4..6],
        &compact[6..8],
        &compact[9..11],
        &compact[11..13]
    ))
}
