//! Pure rolling-window membership and collector accounting.

use std::collections::{BTreeMap, BTreeSet};

use vectorseam_core::window::{WindowError, aligned_window_start};

use crate::aggregate::AggregateError;
use crate::model::{Coverage, ListedPart};

/// Aligns a tick time down to the current storage-window start.
pub fn aligned_round_end(
    now_unix_seconds: u64,
    storage_window_seconds: u32,
) -> Result<u64, AggregateError> {
    Ok(aligned_window_start(
        now_unix_seconds,
        storage_window_seconds,
    )?)
}

/// Enumerates fully-contained storage-window starts in ascending order.
pub fn in_scope_window_starts(
    round_end: u64,
    window_duration_seconds: u64,
    storage_window_seconds: u32,
) -> Result<Vec<u64>, AggregateError> {
    let storage_width = u64::from(storage_window_seconds);
    if storage_width == 0 {
        return Err(WindowError::ZeroDuration.into());
    }
    if round_end % storage_width != 0 {
        return Err(AggregateError::UnalignedRoundEnd);
    }
    let lower = round_end
        .checked_sub(window_duration_seconds)
        .ok_or(AggregateError::WindowUnderflow)?;
    let mut starts = Vec::new();
    let mut candidate_end = round_end;
    while let Some(start) = candidate_end.checked_sub(storage_width) {
        if start < lower {
            break;
        }
        starts.push(start);
        candidate_end = start;
    }
    starts.reverse();
    Ok(starts)
}

/// Computes collector-side drop fraction from distinct part header counts.
pub fn dropped_frame_fraction<'a>(
    headers: impl IntoIterator<Item = (&'a str, u64, u64)>,
) -> Result<f64, AggregateError> {
    let mut received = 0_u64;
    let mut records = 0_u64;
    for (part_ulid, part_received, part_records) in headers {
        if part_records > part_received {
            return Err(AggregateError::InvalidPartCounts(part_ulid.to_owned()));
        }
        received = received
            .checked_add(part_received)
            .ok_or(AggregateError::CounterOverflow("received_frame_count"))?;
        records = records
            .checked_add(part_records)
            .ok_or(AggregateError::CounterOverflow("record_count"))?;
    }
    if received == 0 {
        return Ok(0.0);
    }
    Ok(1.0 - records as f64 / received as f64)
}

pub(crate) fn unique_in_scope_parts(
    parts: &[ListedPart],
    lower: u64,
    round_end: u64,
) -> Result<BTreeMap<&str, &ListedPart>, AggregateError> {
    let mut unique = BTreeMap::new();
    for part in parts {
        let part_end = part
            .window_start
            .checked_add(u64::from(part.window_seconds))
            .ok_or_else(|| AggregateError::PartWindowOverflow(part.part_ulid.clone()))?;
        if part.window_start < lower || part_end > round_end {
            continue;
        }
        if let Some(existing) = unique.insert(part.part_ulid.as_str(), part) {
            if existing != part {
                return Err(AggregateError::ConflictingListedPart(
                    part.part_ulid.clone(),
                ));
            }
        }
    }
    Ok(unique)
}

pub(crate) fn coverage(expected_windows: &[u64], listed: &BTreeMap<&str, &ListedPart>) -> Coverage {
    let expected = expected_windows.iter().copied().collect::<BTreeSet<_>>();
    let windows_with_parts = listed
        .values()
        .filter_map(|part| {
            expected
                .contains(&part.window_start)
                .then_some(part.window_start)
        })
        .collect::<BTreeSet<_>>()
        .len();
    let windows_in_scope = expected_windows.len();
    let empty_window_fraction = if windows_in_scope == 0 {
        1.0
    } else {
        (windows_in_scope - windows_with_parts) as f64 / windows_in_scope as f64
    };
    Coverage {
        empty_window_fraction,
        windows_in_scope: windows_in_scope as u64,
        windows_with_parts: windows_with_parts as u64,
    }
}
