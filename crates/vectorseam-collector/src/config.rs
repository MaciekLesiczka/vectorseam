use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::{ArgAction, Parser};
use vectorseam_core::frame::FIXED_FRAME_HEADER_LEN;
use vectorseam_core::segment::MAX_SEGMENT_OVERHEAD_BYTES;
use vectorseam_recommendation_server::RecommendationServerOptions;

const DEFAULT_WINDOW_SECONDS: u32 = 600;
const DEFAULT_PER_COHORT_MEMORY_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_GLOBAL_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_FRAME_SIZE_BYTES: usize = 32 * 1024;
const DEFAULT_CHANNEL_CAPACITY: usize = 2048;
const DEFAULT_MAX_CONNECTIONS: usize = 2048;
const DEFAULT_IDLE_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_PUT_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:7737";

/// Command-line and environment configuration for the collector.
#[derive(Debug, Clone, Parser)]
#[command(name = "vectorseam-collector")]
pub struct Config {
    /// TCP address to listen on when no Unix socket is configured.
    #[arg(
        long = "listen",
        env = "VECTORSEAM_LISTEN",
        default_value = DEFAULT_LISTEN_ADDR,
        value_name = "ADDR"
    )]
    pub listen: SocketAddr,
    /// Whether the collector should host the effective-recommendation API.
    #[arg(
        long = "recommendation-enabled",
        env = "VECTORSEAM_RECOMMENDATION_ENABLED",
        default_value_t = true,
        action = ArgAction::Set
    )]
    pub recommendation_enabled: bool,
    /// Effective-recommendation HTTP server options.
    #[command(flatten)]
    pub recommendation_server: RecommendationServerOptions,
    /// Optional Unix socket path for same-host demos and tests.
    #[arg(
        long = "unix-socket",
        env = "VECTORSEAM_UNIX_SOCKET",
        value_name = "PATH"
    )]
    pub unix_socket: Option<PathBuf>,
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
    /// Maximum live frame memory plus fixed serialization reserve.
    ///
    /// This includes frames in the reader-to-writer channel, writer cohort
    /// buffers, and fixed reserve for one max-size cohort segment.
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
    /// Maximum idle time while waiting for frame bytes on one connection.
    #[arg(
        long = "idle-timeout-seconds",
        env = "VECTORSEAM_IDLE_TIMEOUT_SECONDS",
        default_value_t = DEFAULT_IDLE_TIMEOUT_SECONDS
    )]
    pub idle_timeout_seconds: u64,
    /// Object-store PUT timeout in seconds.
    #[arg(
        long = "put-timeout-seconds",
        env = "VECTORSEAM_PUT_TIMEOUT_SECONDS",
        default_value_t = DEFAULT_PUT_TIMEOUT_SECONDS
    )]
    pub put_timeout_seconds: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct ReaderConfig {
    pub(crate) max_frame_size: usize,
    pub(crate) live_memory_bytes: usize,
    pub(crate) idle_timeout: Duration,
}

#[derive(Clone, Copy)]
pub(crate) struct WriterConfig {
    pub(crate) window_seconds: u32,
    pub(crate) per_cohort_memory_bytes: usize,
    pub(crate) live_memory_bytes: usize,
    pub(crate) put_timeout: Duration,
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
    let max_record_bytes = config
        .max_frame_size
        .checked_add(8)
        .ok_or_else(|| anyhow!("max frame size is too large"))?;
    if max_record_bytes > config.per_cohort_memory_bytes {
        return Err(anyhow!(
            "per-cohort memory cap must fit one max-size record"
        ));
    }
    let live_memory_bytes = live_memory_bytes(config)?;
    if max_record_bytes > live_memory_bytes {
        return Err(anyhow!(
            "global memory cap must fit one max-size record after serialization reserve"
        ));
    }
    if config.channel_capacity == 0 {
        return Err(anyhow!("channel capacity must be greater than zero"));
    }
    if config.max_connections == 0 {
        return Err(anyhow!("max connections must be greater than zero"));
    }
    if config.idle_timeout_seconds == 0 {
        return Err(anyhow!("idle timeout seconds must be greater than zero"));
    }
    if config.put_timeout_seconds == 0 {
        return Err(anyhow!("put timeout seconds must be greater than zero"));
    }
    Ok(())
}

pub(crate) fn live_memory_bytes(config: &Config) -> Result<usize> {
    let flush_reserve = config
        .per_cohort_memory_bytes
        .checked_add(MAX_SEGMENT_OVERHEAD_BYTES)
        .ok_or_else(|| anyhow!("per-cohort memory cap is too large"))?;
    config
        .global_memory_bytes
        .checked_sub(flush_reserve)
        .ok_or_else(|| {
            anyhow!("global memory cap must exceed per-cohort cap plus serialization reserve")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> Config {
        Config {
            listen: "127.0.0.1:7737".parse().unwrap(),
            recommendation_enabled: true,
            recommendation_server: RecommendationServerOptions::default(),
            unix_socket: None,
            storage_root: PathBuf::from("/tmp/vseam"),
            window_seconds: DEFAULT_WINDOW_SECONDS,
            per_cohort_memory_bytes: DEFAULT_PER_COHORT_MEMORY_BYTES,
            global_memory_bytes: DEFAULT_GLOBAL_MEMORY_BYTES,
            max_frame_size: DEFAULT_MAX_FRAME_SIZE_BYTES,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            idle_timeout_seconds: DEFAULT_IDLE_TIMEOUT_SECONDS,
            put_timeout_seconds: DEFAULT_PUT_TIMEOUT_SECONDS,
        }
    }

    #[test]
    fn parses_recommendation_server_options() {
        let config = Config::try_parse_from([
            "vectorseam-collector",
            "--storage-root",
            "/tmp/vseam",
            "--recommendation-enabled=false",
            "--recommendation-listen",
            "127.0.0.1:9000",
            "--recommendation-max-connections",
            "6",
            "--recommendation-max-concurrent-requests",
            "7",
            "--recommendation-lookup-timeout-seconds",
            "8",
            "--recommendation-request-head-timeout-seconds",
            "9",
            "--recommendation-shutdown-drain-timeout-seconds",
            "10",
            "--recommendation-cache-ttl-seconds",
            "11",
            "--recommendation-max-cached-cohorts",
            "12",
            "--recommendation-max-cached-negative-cohorts",
            "13",
        ])
        .unwrap();

        assert!(!config.recommendation_enabled);
        assert_eq!(config.recommendation_server.listen_addr.port(), 9000);
        assert_eq!(config.recommendation_server.max_connections, 6);
        assert_eq!(config.recommendation_server.max_concurrent_requests, 7);
        assert_eq!(config.recommendation_server.lookup_timeout_seconds, 8);
        assert_eq!(config.recommendation_server.request_head_timeout_seconds, 9);
        assert_eq!(
            config.recommendation_server.shutdown_drain_timeout_seconds,
            10
        );
        assert_eq!(config.recommendation_server.cache_ttl_seconds, 11);
        assert_eq!(config.recommendation_server.max_cached_cohorts, 12);
        assert_eq!(config.recommendation_server.max_cached_negative_cohorts, 13);
    }

    #[test]
    fn rejects_memory_caps_that_cannot_fit_one_max_size_record() {
        let mut config = valid_config();
        config.max_frame_size = 1024;
        config.per_cohort_memory_bytes = 1024 + 7;

        let error = validate_config(&config).unwrap_err();

        assert!(error.to_string().contains("per-cohort memory cap"));
    }

    #[test]
    fn rejects_global_cap_that_cannot_fit_serialization_reserve() {
        let mut config = valid_config();
        config.max_frame_size = 1024;
        config.per_cohort_memory_bytes = 1024 + 8;
        config.global_memory_bytes = config.per_cohort_memory_bytes + MAX_SEGMENT_OVERHEAD_BYTES;

        let error = validate_config(&config).unwrap_err();

        assert!(error.to_string().contains("global memory cap"));
    }

    #[test]
    fn rejects_zero_idle_timeout() {
        let mut config = valid_config();
        config.idle_timeout_seconds = 0;

        let error = validate_config(&config).unwrap_err();

        assert!(error.to_string().contains("idle timeout"));
    }
}
