//! Pure population deduplication, summaries, selection, and transfer checks.

use std::collections::BTreeMap;

use crate::aggregate::AggregateError;
use crate::math::{quantile_type7, select_ef, transfer_confidence};
use crate::model::{
    AggregationConfig, MeasuredSample, PerEfSummary, RoundStatus, SweepMeasurement,
};

/// A window-wide deduplication survivor, exposed for acceptance assertions.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PopulationSample {
    /// Survivor part ULID.
    pub part_ulid: String,
    /// Survivor first-occurrence record ordinal.
    pub record_index: i32,
    /// Exact-vector hash and deduplication key.
    pub vector_hash: u64,
    /// Client-observed ground-truth statement duration.
    pub ground_truth_latency_ms: f64,
    /// Authoritative sweep measurements.
    pub sweeps: BTreeMap<i32, SweepMeasurement>,
}

/// Deduplicates compatible samples by vector hash and survivor ordering.
pub(crate) fn deduplicate_samples<'a>(
    samples: impl IntoIterator<Item = (&'a str, &'a MeasuredSample)>,
) -> Vec<PopulationSample> {
    let mut survivors = BTreeMap::<u64, PopulationSample>::new();
    for (part_ulid, sample) in samples {
        let candidate = PopulationSample {
            part_ulid: part_ulid.to_owned(),
            record_index: sample.record_index,
            vector_hash: sample.vector_hash,
            ground_truth_latency_ms: sample.ground_truth_latency_ms,
            sweeps: sample.sweeps.clone(),
        };
        match survivors.get_mut(&sample.vector_hash) {
            Some(current)
                if (candidate.part_ulid.as_str(), candidate.record_index)
                    < (current.part_ulid.as_str(), current.record_index) =>
            {
                *current = candidate;
            }
            None => {
                survivors.insert(sample.vector_hash, candidate);
            }
            Some(_) => {}
        }
    }
    survivors.into_values().collect()
}

pub(crate) fn validate_population(
    population: &[PopulationSample],
    ef_grid: &[i32],
) -> Result<(), AggregateError> {
    for sample in population {
        if !sample.ground_truth_latency_ms.is_finite() || sample.ground_truth_latency_ms < 0.0 {
            return Err(AggregateError::InvalidGroundTruthLatency {
                part_ulid: sample.part_ulid.clone(),
                record_index: sample.record_index,
            });
        }
        for ef in ef_grid {
            let observation =
                sample
                    .sweeps
                    .get(ef)
                    .ok_or_else(|| AggregateError::MissingSweep {
                        part_ulid: sample.part_ulid.clone(),
                        record_index: sample.record_index,
                        ef: *ef,
                    })?;
            if !observation.recall.is_finite()
                || !(0.0..=1.0).contains(&observation.recall)
                || !observation.latency_ms.is_finite()
                || observation.latency_ms < 0.0
            {
                return Err(AggregateError::InvalidObservation {
                    part_ulid: sample.part_ulid.clone(),
                    record_index: sample.record_index,
                    ef: *ef,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn per_ef_summaries(
    population: &[PopulationSample],
    config: &AggregationConfig,
) -> Result<Vec<PerEfSummary>, AggregateError> {
    if population.is_empty() {
        return Ok(Vec::new());
    }
    let q = 1.0 - config.percentile;
    config
        .ef_grid
        .iter()
        .map(|ef| {
            let observations = population
                .iter()
                .map(|sample| &sample.sweeps[ef])
                .collect::<Vec<_>>();
            let recalls = observations
                .iter()
                .map(|observation| observation.recall)
                .collect::<Vec<_>>();
            let latencies = observations
                .iter()
                .map(|observation| observation.latency_ms)
                .collect::<Vec<_>>();
            Ok(PerEfSummary {
                ef: *ef,
                quantile_recall: quantile_type7(&recalls, q)?,
                mean_recall: recalls.iter().sum::<f64>() / recalls.len() as f64,
                latency_p50_ms: quantile_type7(&latencies, 0.5)?,
            })
        })
        .collect()
}

pub(crate) struct CompletedSelection {
    pub(crate) recommended_ef: i32,
    pub(crate) status: RoundStatus,
    pub(crate) confidence: f64,
    pub(crate) transferred: bool,
    pub(crate) train_quantile: f64,
    pub(crate) test_quantile: f64,
}

pub(crate) fn select_and_validate(
    config: &AggregationConfig,
    train: &[&PopulationSample],
    test: &[&PopulationSample],
) -> Result<CompletedSelection, AggregateError> {
    let q = 1.0 - config.percentile;
    let train_quantiles = config
        .ef_grid
        .iter()
        .map(|ef| {
            let recalls = train
                .iter()
                .map(|sample| sample.sweeps[ef].recall)
                .collect::<Vec<_>>();
            Ok((*ef, quantile_type7(&recalls, q)?))
        })
        .collect::<Result<BTreeMap<_, _>, AggregateError>>()?;
    let selected = select_ef(&train_quantiles, config.value)?;
    let test_recalls = test
        .iter()
        .map(|sample| sample.sweeps[&selected.recommended_ef].recall)
        .collect::<Vec<_>>();
    let test_quantile = quantile_type7(&test_recalls, q)?;
    let successes = test_recalls
        .iter()
        .filter(|recall| **recall >= config.value)
        .count();
    Ok(CompletedSelection {
        recommended_ef: selected.recommended_ef,
        status: selected.status,
        confidence: transfer_confidence(test.len(), successes, config.percentile)?,
        transferred: test_quantile >= config.value,
        train_quantile: train_quantiles[&selected.recommended_ef],
        test_quantile,
    })
}
