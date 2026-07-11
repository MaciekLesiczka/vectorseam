use std::io;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, warn};
use vectorseam_core::cohort::CohortName;
use vectorseam_core::frame::parse_frame_header;

use crate::config::ReaderConfig;
use crate::counters::CollectorCounters;
use crate::memory::{MemoryGuard, MemoryTracker};
use crate::time::unix_micros_now;

#[derive(Debug)]
pub(crate) struct FrameEvent {
    pub(crate) cohort: CohortName,
    pub(crate) receive_time: u64,
    pub(crate) frame: Bytes,
    pub(crate) memory: MemoryGuard,
}

pub(crate) async fn handle_connection<S>(
    mut stream: S,
    tx: mpsc::Sender<FrameEvent>,
    counters: Arc<CollectorCounters>,
    memory: Arc<MemoryTracker>,
    config: ReaderConfig,
    mut shutdown: watch::Receiver<bool>,
) where
    S: AsyncRead + Unpin,
{
    loop {
        let mut len_buf = [0_u8; 4];
        match read_exact_or_shutdown(&mut stream, &mut len_buf, &mut shutdown).await {
            ReadOutcome::Read => {}
            ReadOutcome::Closed | ReadOutcome::Shutdown => return,
            ReadOutcome::Error(error) => {
                counters.connection_errors.fetch_add(1, Ordering::Relaxed);
                debug!(%error, "connection read failed");
                return;
            }
        }

        counters.frames_read.fetch_add(1, Ordering::Relaxed);
        let frame_len = u32::from_le_bytes(len_buf);
        let frame_total_len = match usize::try_from(frame_len)
            .ok()
            .and_then(|value| value.checked_add(4))
        {
            Some(value) => value,
            None => {
                counters.invalid_frames.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if frame_total_len > config.max_frame_size {
            counters.oversized_frames.fetch_add(1, Ordering::Relaxed);
            warn!(
                frame_bytes = frame_total_len,
                "closing connection after oversized frame"
            );
            return;
        }

        let mut frame = vec![0_u8; frame_total_len];
        frame[0..4].copy_from_slice(&len_buf);
        match read_exact_or_shutdown(&mut stream, &mut frame[4..], &mut shutdown).await {
            ReadOutcome::Read => {}
            ReadOutcome::Closed | ReadOutcome::Shutdown => return,
            ReadOutcome::Error(error) => {
                counters.connection_errors.fetch_add(1, Ordering::Relaxed);
                debug!(%error, "connection frame read failed");
                return;
            }
        }

        let parsed = match parse_frame_header(&frame) {
            Ok(parsed) => parsed,
            Err(error) => {
                counters.invalid_frames.fetch_add(1, Ordering::Relaxed);
                debug!(%error, "dropping invalid frame");
                continue;
            }
        };
        let cohort = match CohortName::try_from(parsed.name) {
            Ok(cohort) => cohort,
            Err(error) => {
                counters.invalid_names.fetch_add(1, Ordering::Relaxed);
                debug!(cohort = %parsed.name, %error, "dropping invalid cohort name");
                continue;
            }
        };

        let receive_time = match unix_micros_now() {
            Ok(value) => value,
            Err(error) => {
                error!(%error, "failed to stamp receive time");
                continue;
            }
        };
        counters
            .valid_frames_received
            .fetch_add(1, Ordering::Relaxed);
        let memory_bytes = match frame.len().checked_add(8) {
            Some(value) => value,
            None => {
                counters.record_memory_drop(&cohort);
                continue;
            }
        };
        let Some(memory_guard) = memory.try_reserve(memory_bytes, config.global_memory_bytes)
        else {
            counters.record_memory_drop(&cohort);
            continue;
        };
        let event = FrameEvent {
            cohort,
            receive_time,
            frame: Bytes::from(frame),
            memory: memory_guard,
        };
        match tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                counters.record_channel_drop(&event.cohort);
            }
            Err(mpsc::error::TrySendError::Closed(_event)) => return,
        }
    }
}

enum ReadOutcome {
    Read,
    Closed,
    Shutdown,
    Error(io::Error),
}

async fn read_exact_or_shutdown<S>(
    stream: &mut S,
    buffer: &mut [u8],
    shutdown: &mut watch::Receiver<bool>,
) -> ReadOutcome
where
    S: AsyncRead + Unpin,
{
    // Dropping read_exact can abandon a partially read frame. That is
    // intentional only for shutdown: collection is best effort, and the
    // simpler cancellation path is preferable to finishing every in-progress
    // connection while the daemon is stopping.
    tokio::select! {
        result = stream.read_exact(buffer) => {
            match result {
                Ok(_bytes) => ReadOutcome::Read,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => ReadOutcome::Closed,
                Err(error) => ReadOutcome::Error(error),
            }
        }
        _ = shutdown.changed() => ReadOutcome::Shutdown,
    }
}
