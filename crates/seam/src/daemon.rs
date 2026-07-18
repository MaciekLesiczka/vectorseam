//! Long-running single-flight round scheduling and graceful shutdown.

use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use thiserror::Error;
use tokio::task::JoinError;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use vectorseam_core::window::{WindowError, aligned_window_start, format_window_timestamp};

use crate::config::Config;
use crate::tuner::{Tuner, TunerStartError};

/// Failure to start or supervise the long-running tuner.
#[derive(Debug, Error)]
pub enum DaemonError {
    /// Long-lived tuner resources could not be initialized.
    #[error(transparent)]
    Start(#[from] TunerStartError),
    /// The system clock was earlier than the Unix epoch.
    #[error("system time is before the Unix epoch")]
    ClockBeforeUnixEpoch,
    /// The current timestamp could not be represented in microseconds.
    #[error("current Unix timestamp overflowed u64 microseconds")]
    TimestampOverflow,
    /// A directly constructed config bypassed startup interval validation.
    #[error("calibration.interval must be greater than zero")]
    ZeroInterval,
    /// A UTC timestamp could not be formatted or aligned.
    #[error(transparent)]
    Window(#[from] WindowError),
    /// The owned signal task panicked.
    #[error("shutdown signal task failed: {0}")]
    SignalTask(#[source] JoinError),
    /// The operating-system signal handler failed.
    #[error("shutdown signal handler failed: {0}")]
    Signal(#[source] io::Error),
}

/// Runs immediate and periodic rounds until SIGINT or SIGTERM.
pub async fn run(config: Config) -> Result<(), DaemonError> {
    let interval = config.calibration.interval;
    if interval.is_zero() {
        return Err(DaemonError::ZeroInterval);
    }
    let mut tuner = Tuner::start(config).await?;
    let cancellation = CancellationToken::new();
    let signal_cancellation = cancellation.clone();
    let mut signal_task = tokio::spawn(async move {
        let result = shutdown_signal().await;
        signal_cancellation.cancel();
        result
    });

    let run_result = run_round_loop(&mut tuner, interval, &cancellation).await;
    cancellation.cancel();
    tuner.shutdown().await;

    let signal_result = if run_result.is_err() && !signal_task.is_finished() {
        signal_task.abort();
        match signal_task.await {
            Err(error) if error.is_cancelled() => Ok(()),
            Err(error) => Err(DaemonError::SignalTask(error)),
            Ok(result) => result.map_err(DaemonError::Signal),
        }
    } else {
        (&mut signal_task)
            .await
            .map_err(DaemonError::SignalTask)?
            .map_err(DaemonError::Signal)
    };

    run_result?;
    signal_result
}

async fn run_round_loop(
    tuner: &mut Tuner,
    interval: Duration,
    cancellation: &CancellationToken,
) -> Result<(), DaemonError> {
    let mut scheduled_tick = Instant::now();
    loop {
        tokio::select! {
            () = cancellation.cancelled() => break,
            () = tokio::time::sleep_until(scheduled_tick) => {}
        }
        if cancellation.is_cancelled() {
            break;
        }

        let timestamps = current_round_timestamps(tuner.storage_window_seconds())?;
        info!(round_end = timestamps.round_end, "starting tuner round");
        let report = tuner
            .run_round(
                timestamps.round_end,
                timestamps.computed_at,
                timestamps.computed_at_us,
                cancellation,
            )
            .await;
        info!(
            round_end = timestamps.round_end,
            published_cohorts = report.published.len(),
            failed_cohorts = report.failed_cohorts.len(),
            cancelled = report.cancelled,
            "tuner round finished"
        );
        if cancellation.is_cancelled() || report.cancelled {
            break;
        }

        let (next_tick, skipped_ticks) =
            next_tick_after_round(scheduled_tick, Instant::now(), interval);
        scheduled_tick = next_tick;
        if skipped_ticks > 0 {
            warn!(
                skipped_ticks,
                "round exceeded one or more tick boundaries; skipped queued work"
            );
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct RoundTimestamps {
    round_end: u64,
    computed_at: String,
    computed_at_us: u64,
}

fn current_round_timestamps(storage_window_seconds: u32) -> Result<RoundTimestamps, DaemonError> {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_error| DaemonError::ClockBeforeUnixEpoch)?;
    round_timestamps(since_epoch, storage_window_seconds)
}

fn round_timestamps(
    since_epoch: Duration,
    storage_window_seconds: u32,
) -> Result<RoundTimestamps, DaemonError> {
    let unix_seconds = since_epoch.as_secs();
    let compact = format_window_timestamp(unix_seconds)?;
    let computed_at = format!(
        "{}-{}-{}T{}:{}:{:02}Z",
        &compact[0..4],
        &compact[4..6],
        &compact[6..8],
        &compact[9..11],
        &compact[11..13],
        unix_seconds % 60
    );
    let computed_at_us = unix_seconds
        .checked_mul(1_000_000)
        .and_then(|micros| micros.checked_add(u64::from(since_epoch.subsec_micros())))
        .ok_or(DaemonError::TimestampOverflow)?;
    Ok(RoundTimestamps {
        round_end: aligned_window_start(unix_seconds, storage_window_seconds)?,
        computed_at,
        computed_at_us,
    })
}

fn next_tick_after_round(
    previous_tick: Instant,
    now: Instant,
    interval: Duration,
) -> (Instant, u128) {
    let candidate = previous_tick + interval;
    if candidate > now {
        return (candidate, 0);
    }

    let overdue = now.duration_since(candidate).as_nanos();
    let interval_nanos = interval.as_nanos();
    let skipped_ticks = overdue / interval_nanos + 1;
    let elapsed_in_interval = overdue % interval_nanos;
    let until_next = interval_nanos - elapsed_in_interval;
    (now + duration_from_nanos(until_next), skipped_ticks)
}

fn duration_from_nanos(nanos: u128) -> Duration {
    let seconds = nanos / 1_000_000_000;
    let subsecond_nanos = nanos % 1_000_000_000;
    Duration::new(
        u64::try_from(seconds).expect("nanoseconds came from a std::time::Duration"),
        u32::try_from(subsecond_nanos).expect("subsecond nanoseconds are less than one billion"),
    )
}

#[cfg(unix)]
async fn shutdown_signal() -> io::Result<()> {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result,
        _ = sigterm.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> io::Result<()> {
    tokio::signal::ctrl_c().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage4_round_timestamps_align_and_preserve_seconds_and_micros() {
        let timestamps = round_timestamps(Duration::new(1_784_117_243, 456_789_000), 600).unwrap();
        assert_eq!(
            timestamps,
            RoundTimestamps {
                round_end: 1_784_116_800,
                computed_at: "2026-07-15T12:07:23Z".to_owned(),
                computed_at_us: 1_784_117_243_456_789,
            }
        );
    }

    #[test]
    fn stage4_single_flight_scheduler_skips_ticks_crossed_by_a_round() {
        let first_tick = Instant::now();
        let (next, skipped) = next_tick_after_round(
            first_tick,
            first_tick + Duration::from_secs(25),
            Duration::from_secs(10),
        );
        assert_eq!(next, first_tick + Duration::from_secs(30));
        assert_eq!(skipped, 2);
    }

    #[test]
    fn stage4_single_flight_scheduler_retains_next_uncrossed_tick() {
        let first_tick = Instant::now();
        let (next, skipped) = next_tick_after_round(
            first_tick,
            first_tick + Duration::from_secs(5),
            Duration::from_secs(10),
        );
        assert_eq!(next, first_tick + Duration::from_secs(10));
        assert_eq!(skipped, 0);
    }
}
