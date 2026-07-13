//! Window arithmetic and object key formatting.

use thiserror::Error;
use ulid::Ulid;

use crate::cohort::CohortName;

const SECONDS_PER_DAY: u64 = 86_400;
const SECONDS_PER_HOUR: u64 = 3_600;
const SECONDS_PER_MINUTE: u64 = 60;

/// Errors returned by window arithmetic and key formatting.
#[derive(Debug, Error)]
pub enum WindowError {
    /// Window duration must be greater than zero.
    #[error("window duration must be greater than zero")]
    ZeroDuration,
    /// The timestamp cannot be formatted as UTC.
    #[error("timestamp is out of supported range")]
    TimestampOutOfRange,
}

/// Returns the aligned tumbling-window start for `unix_seconds`.
///
/// For a 600-second window, for example, timestamps from `12:10:00` through
/// `12:19:59` align to `12:10:00`.
pub fn aligned_window_start(unix_seconds: u64, window_seconds: u32) -> Result<u64, WindowError> {
    if window_seconds == 0 {
        return Err(WindowError::ZeroDuration);
    }
    let width = u64::from(window_seconds);
    Ok(unix_seconds - unix_seconds % width)
}

/// Formats the immutable object key for a flushed segment part.
///
/// The key layout is
/// `cohorts/<cohort>/window=<YYYYMMDD>T<HHMM>Z/part-<ulid>.vseam`.
pub fn object_key(
    cohort: &CohortName,
    window_start: u64,
    part: Ulid,
) -> Result<String, WindowError> {
    let timestamp = format_window_timestamp(window_start)?;
    Ok(format!(
        "cohorts/{cohort}/window={timestamp}/part-{part}.vseam"
    ))
}

/// Formats an aligned UTC window timestamp as `YYYYMMDDTHHMMZ`.
pub fn format_window_timestamp(window_start: u64) -> Result<String, WindowError> {
    let days = window_start / SECONDS_PER_DAY;
    let seconds_of_day = window_start % SECONDS_PER_DAY;
    let days_i64 = i64::try_from(days).map_err(|_| WindowError::TimestampOutOfRange)?;
    let (year, month, day) = civil_from_days(days_i64)?;
    let hour = seconds_of_day / SECONDS_PER_HOUR;
    let minute = (seconds_of_day % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    Ok(format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}Z"))
}

fn civil_from_days(days_since_unix_epoch: i64) -> Result<(i64, i64, i64), WindowError> {
    // Rust's standard library has timestamps and durations, but no UTC
    // calendar-date formatter. This is Howard Hinnant's common Gregorian
    // "civil_from_days" algorithm for converting days since 1970-01-01 into
    // year/month/day without adding a date-time dependency.
    let z = days_since_unix_epoch
        .checked_add(719_468)
        .ok_or(WindowError::TimestampOutOfRange)?;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let adjusted_year = year + if month <= 2 { 1 } else { 0 };
    Ok((adjusted_year, month, day))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_window_start() {
        assert_eq!(aligned_window_start(0, 600).unwrap(), 0);
        assert_eq!(aligned_window_start(599, 600).unwrap(), 0);
        assert_eq!(aligned_window_start(600, 600).unwrap(), 600);
        assert_eq!(aligned_window_start(601, 600).unwrap(), 600);
        assert_eq!(
            aligned_window_start(1_783_513_999, 600).unwrap(),
            1_783_513_800
        );
    }

    #[test]
    fn rejects_zero_window_duration() {
        assert!(matches!(
            aligned_window_start(10, 0),
            Err(WindowError::ZeroDuration)
        ));
    }

    #[test]
    fn formats_object_key() {
        let ulid = "01J00000000000000000000000".parse::<Ulid>().unwrap();
        let cohort = CohortName::try_from("env=Prod/tenant=te.nant").unwrap();

        let key = object_key(&cohort, 1_783_513_800, ulid).unwrap();

        assert_eq!(
            key,
            "cohorts/env=Prod/tenant=te.nant/window=20260708T1230Z/part-01J00000000000000000000000.vseam"
        );
    }
}
