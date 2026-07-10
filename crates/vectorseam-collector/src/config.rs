use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Parser;
use vectorseam_core::frame::FIXED_FRAME_HEADER_LEN;

const DEFAULT_WINDOW_SECONDS: u32 = 600;
const DEFAULT_PER_COHORT_MEMORY_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_GLOBAL_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_FRAME_SIZE_BYTES: usize = 32 * 1024;
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;
const DEFAULT_MAX_CONNECTIONS: usize = 1024;

/// Command-line and environment configuration for the collector.
#[derive(Debug, Clone, Parser)]
#[command(name = "vectorseam-collector")]
pub struct Config {
    /// Unix socket path to listen on.
    #[arg(long = "socket", env = "VECTORSEAM_SOCKET", value_name = "PATH")]
    pub socket: PathBuf,
    /// Local object-store root directory.
    #[arg(
        long = "storage-root",
        env = "VECTORSEAM_STORAGE_ROOT",
        value_name = "DIR"
    )]
    pub storage_root: PathBuf,
    /// Tumbling window duration in seconds.
    #[arg(
        long = "window-seconds",
        env = "VECTORSEAM_WINDOW_SECONDS",
        default_value_t = DEFAULT_WINDOW_SECONDS
    )]
    pub window_seconds: u32,
    /// Maximum writer-buffered record bytes for one cohort.
    ///
    /// Records count as frame bytes plus the 8-byte segment receive timestamp.
    /// A cohort that would exceed this cap is flushed early as a spill part.
    #[arg(
        long = "per-cohort-memory-bytes",
        env = "VECTORSEAM_PER_COHORT_MEMORY_BYTES",
        default_value_t = DEFAULT_PER_COHORT_MEMORY_BYTES
    )]
    pub per_cohort_memory_bytes: usize,
    /// Maximum writer memory for buffered records plus serialization reserve.
    ///
    /// The writer accounts for all cohort buffers and enough reserve to build
    /// one contiguous `.vseam` segment before sending it to the object store.
    #[arg(
        long = "global-memory-bytes",
        env = "VECTORSEAM_GLOBAL_MEMORY_BYTES",
        default_value_t = DEFAULT_GLOBAL_MEMORY_BYTES
    )]
    pub global_memory_bytes: usize,
    /// Maximum accepted protocol frame size, including the length field.
    ///
    /// A frame larger than this is treated as hostile or corrupt input; the
    /// collector counts it and closes that connection.
    #[arg(
        long = "max-frame-size",
        env = "VECTORSEAM_MAX_FRAME_SIZE",
        default_value_t = DEFAULT_MAX_FRAME_SIZE_BYTES
    )]
    pub max_frame_size: usize,
    /// Bounded reader-to-writer channel capacity, in frames.
    ///
    /// Readers use non-blocking sends and drop frames when the channel is full.
    #[arg(
        long = "channel-capacity",
        env = "VECTORSEAM_CHANNEL_CAPACITY",
        default_value_t = DEFAULT_CHANNEL_CAPACITY
    )]
    pub channel_capacity: usize,
    /// Maximum concurrently handled client connections.
    #[arg(
        long = "max-connections",
        env = "VECTORSEAM_MAX_CONNECTIONS",
        default_value_t = DEFAULT_MAX_CONNECTIONS
    )]
    pub max_connections: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct ReaderConfig {
    pub(crate) max_frame_size: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct WriterConfig {
    pub(crate) window_seconds: u32,
    pub(crate) per_cohort_memory_bytes: usize,
    pub(crate) global_memory_bytes: usize,
}

pub(crate) fn validate_config(config: &Config) -> Result<()> {
    if config.window_seconds == 0 {
        return Err(anyhow!("window seconds must be greater than zero"));
    }
    if config.per_cohort_memory_bytes == 0 {
        return Err(anyhow!("per-cohort memory cap must be greater than zero"));
    }
    if config.global_memory_bytes == 0 {
        return Err(anyhow!("global memory cap must be greater than zero"));
    }
    if config.max_frame_size < FIXED_FRAME_HEADER_LEN {
        return Err(anyhow!(
            "max frame size must be at least {FIXED_FRAME_HEADER_LEN} bytes"
        ));
    }
    if config.channel_capacity == 0 {
        return Err(anyhow!("channel capacity must be greater than zero"));
    }
    if config.max_connections == 0 {
        return Err(anyhow!("max connections must be greater than zero"));
    }
    Ok(())
}
