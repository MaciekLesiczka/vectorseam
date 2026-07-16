//! Small, pure numeric primitives fixed by the tuner specification.

use std::collections::{BTreeMap, HashSet};

use statrs::function::beta::beta_reg;
use thiserror::Error;

use crate::model::RoundStatus;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const SPLIT_BUCKETS: u64 = 10_000;

/// Invalid input to a frozen estimator primitive.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MathError {
    /// Recall denominator was zero.
    #[error("recall k must be at least 1")]
    ZeroK,
    /// Quantile input was empty.
    #[error("type-7 quantile requires at least one value")]
    EmptyQuantile,
    /// Quantile probability was outside `[0, 1]` or non-finite.
    #[error("quantile probability must be finite and in [0, 1]")]
    InvalidQuantileProbability,
    /// Quantile input contained a non-finite value.
    #[error("quantile values must all be finite")]
    NonFiniteQuantileValue,
    /// Selection received no ef values.
    #[error("ef selection requires a non-empty grid")]
    EmptyEfGrid,
    /// Selection inputs were non-finite or outside recall bounds.
    #[error("ef selection requires finite quantiles in [0, 1] and target value in (0, 1]")]
    InvalidSelectionInput,
    /// Confidence inputs violated `m <= n` or percentile bounds.
    #[error("confidence requires m <= n and finite percentile in (0, 1)")]
    InvalidConfidenceInput,
    /// Split fraction did not map to an interior bucket threshold.
    #[error("train fraction must round to a split threshold in 1..=9999")]
    InvalidTrainFraction,
}

/// A non-insufficient ef selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EfSelection {
    /// Selected ef value.
    pub recommended_ef: i32,
    /// `ok` or `target_unmet`.
    pub status: RoundStatus,
}

/// Computes recall@k with distinct-key set semantics.
pub fn recall_at_k(gt_keys: &[i64], returned_keys: &[i64], k: u32) -> Result<f64, MathError> {
    if k == 0 {
        return Err(MathError::ZeroK);
    }
    let ground_truth = gt_keys.iter().copied().collect::<HashSet<_>>();
    let returned = returned_keys.iter().copied().collect::<HashSet<_>>();
    let hits = ground_truth.intersection(&returned).count();
    Ok(hits as f64 / f64::from(k))
}

/// Computes FNV-1a 64 exactly, with wrapping multiplication.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

/// Returns the frozen 10,000-bucket split threshold.
pub fn split_threshold(train_fraction: f64) -> Result<u64, MathError> {
    if !train_fraction.is_finite() {
        return Err(MathError::InvalidTrainFraction);
    }
    let rounded = (train_fraction * SPLIT_BUCKETS as f64).round();
    if !(1.0..=9_999.0).contains(&rounded) {
        return Err(MathError::InvalidTrainFraction);
    }
    Ok(rounded as u64)
}

/// Returns whether a vector hash belongs to the deterministic train split.
pub fn is_train_member(
    vector_hash: u64,
    split_seed: u64,
    train_fraction: f64,
) -> Result<bool, MathError> {
    let threshold = split_threshold(train_fraction)?;
    let split_key = format!("s:{split_seed}:{vector_hash}");
    Ok(fnv1a64(split_key.as_bytes()) % SPLIT_BUCKETS < threshold)
}

/// Computes Hyndman–Fan type-7 linear quantile.
pub fn quantile_type7(values: &[f64], q: f64) -> Result<f64, MathError> {
    if values.is_empty() {
        return Err(MathError::EmptyQuantile);
    }
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(MathError::InvalidQuantileProbability);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MathError::NonFiniteQuantileValue);
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    let h = (sorted.len() - 1) as f64 * q;
    let lower = h.floor() as usize;
    let fraction = h - lower as f64;
    if fraction == 0.0 {
        return Ok(sorted[lower]);
    }
    Ok(sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]))
}

/// Selects the smallest clearing ef, or the maximum ef if none clears.
pub fn select_ef(
    train_quantiles: &BTreeMap<i32, f64>,
    target_value: f64,
) -> Result<EfSelection, MathError> {
    if !target_value.is_finite()
        || !(0.0..=1.0).contains(&target_value)
        || target_value == 0.0
        || train_quantiles
            .values()
            .any(|quantile| !quantile.is_finite() || !(0.0..=1.0).contains(quantile))
    {
        return Err(MathError::InvalidSelectionInput);
    }
    let (&maximum, _) = train_quantiles
        .last_key_value()
        .ok_or(MathError::EmptyEfGrid)?;
    if let Some((&ef, _)) = train_quantiles
        .iter()
        .find(|(_ef, quantile)| **quantile >= target_value)
    {
        return Ok(EfSelection {
            recommended_ef: ef,
            status: RoundStatus::Ok,
        });
    }
    Ok(EfSelection {
        recommended_ef: maximum,
        status: RoundStatus::TargetUnmet,
    })
}

/// Computes the Beta-posterior survival probability from the frozen formula.
pub fn transfer_confidence(n: usize, m: usize, percentile: f64) -> Result<f64, MathError> {
    if m > n || !percentile.is_finite() || !(0.0..1.0).contains(&percentile) {
        return Err(MathError::InvalidConfidenceInput);
    }
    let successes = m as f64 + 1.0;
    let failures = (n - m) as f64 + 1.0;
    // I_(1-p)(b, a) is mathematically 1 - I_p(a, b), but avoids destructive
    // cancellation in the high-confidence tail and matches a Beta SF.
    Ok(beta_reg(failures, successes, 1.0 - percentile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_requires_non_empty_grid() {
        assert_eq!(
            select_ef(&BTreeMap::new(), 0.9),
            Err(MathError::EmptyEfGrid)
        );
    }
}
