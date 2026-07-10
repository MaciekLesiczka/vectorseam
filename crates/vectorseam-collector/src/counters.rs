use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tracing::{error, info};
use vectorseam_core::cohort::CohortName;

const SUMMARY_INTERVAL_SECONDS: u64 = 60;

#[derive(Default)]
pub(crate) struct CollectorCounters {
    pub(crate) connections_accepted: AtomicU64,
    pub(crate) frames_read: AtomicU64,
    pub(crate) valid_frames_received: AtomicU64,
    pub(crate) invalid_frames: AtomicU64,
    pub(crate) invalid_names: AtomicU64,
    pub(crate) oversized_frames: AtomicU64,
    pub(crate) channel_dropped_frames: AtomicU64,
    pub(crate) writer_dropped_frames: AtomicU64,
    pub(crate) kept_frames: AtomicU64,
    pub(crate) flushed_parts: AtomicU64,
    pub(crate) flush_failures: AtomicU64,
    pub(crate) connection_errors: AtomicU64,
    pending_cohort_drops: Mutex<HashMap<CohortName, u64>>,
}

impl CollectorCounters {
    pub(crate) fn record_channel_drop(&self, cohort: &CohortName) {
        self.channel_dropped_frames.fetch_add(1, Ordering::Relaxed);
        match self.pending_cohort_drops.lock() {
            Ok(mut drops) => {
                let entry = drops.entry(cohort.clone()).or_default();
                *entry = entry.saturating_add(1);
            }
            Err(error) => {
                error!(%error, "cohort drop ledger is poisoned");
            }
        }
    }

    pub(crate) fn take_pending_cohort_drops(&self, cohort: &CohortName) -> u64 {
        match self.pending_cohort_drops.lock() {
            Ok(mut drops) => drops.remove(cohort).unwrap_or(0),
            Err(error) => {
                error!(%error, "cohort drop ledger is poisoned");
                0
            }
        }
    }

    pub(crate) fn take_all_pending_cohort_drops(&self) -> HashMap<CohortName, u64> {
        match self.pending_cohort_drops.lock() {
            Ok(mut drops) => std::mem::take(&mut *drops),
            Err(error) => {
                error!(%error, "cohort drop ledger is poisoned");
                HashMap::new()
            }
        }
    }

    fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            frames_read: self.frames_read.load(Ordering::Relaxed),
            valid_frames_received: self.valid_frames_received.load(Ordering::Relaxed),
            invalid_frames: self.invalid_frames.load(Ordering::Relaxed),
            invalid_names: self.invalid_names.load(Ordering::Relaxed),
            oversized_frames: self.oversized_frames.load(Ordering::Relaxed),
            channel_dropped_frames: self.channel_dropped_frames.load(Ordering::Relaxed),
            writer_dropped_frames: self.writer_dropped_frames.load(Ordering::Relaxed),
            kept_frames: self.kept_frames.load(Ordering::Relaxed),
            flushed_parts: self.flushed_parts.load(Ordering::Relaxed),
            flush_failures: self.flush_failures.load(Ordering::Relaxed),
            connection_errors: self.connection_errors.load(Ordering::Relaxed),
        }
    }
}

struct CounterSnapshot {
    connections_accepted: u64,
    frames_read: u64,
    valid_frames_received: u64,
    invalid_frames: u64,
    invalid_names: u64,
    oversized_frames: u64,
    channel_dropped_frames: u64,
    writer_dropped_frames: u64,
    kept_frames: u64,
    flushed_parts: u64,
    flush_failures: u64,
    connection_errors: u64,
}

pub(crate) async fn summary_loop(
    counters: Arc<CollectorCounters>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(SUMMARY_INTERVAL_SECONDS)) => {
                let snapshot = counters.snapshot();
                info!(
                    connections_accepted = snapshot.connections_accepted,
                    frames_read = snapshot.frames_read,
                    valid_frames_received = snapshot.valid_frames_received,
                    invalid_frames = snapshot.invalid_frames,
                    invalid_names = snapshot.invalid_names,
                    oversized_frames = snapshot.oversized_frames,
                    channel_dropped_frames = snapshot.channel_dropped_frames,
                    writer_dropped_frames = snapshot.writer_dropped_frames,
                    kept_frames = snapshot.kept_frames,
                    flushed_parts = snapshot.flushed_parts,
                    flush_failures = snapshot.flush_failures,
                    connection_errors = snapshot.connection_errors,
                    "collector summary"
                );
            }
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}
