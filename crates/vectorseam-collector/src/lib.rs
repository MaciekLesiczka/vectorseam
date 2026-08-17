mod config;
mod counters;
mod listener;
mod memory;
mod reader;
mod time;
mod writer;

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinError, JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use vectorseam_recommendation_server::Server as RecommendationServer;
use vectorseam_runtime::{
    await_task_shutdown, await_unit_task_shutdown, unexpected_task_result,
    unexpected_unit_task_result,
};

pub use crate::config::Config;
use crate::config::{ReaderConfig, WriterConfig, live_memory_bytes, validate_config};
use crate::counters::{CollectorCounters, summary_loop};
use crate::listener::{AcceptedConnection, BoundListener};
use crate::memory::MemoryTracker;
use crate::reader::handle_connection;
use crate::writer::Writer;

const CONNECTION_SHUTDOWN_DRAIN_MS: u64 = 250;
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(100);
const WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);
const SUMMARY_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

/// Runs the collector using a local filesystem object store.
pub async fn run(config: Config) -> Result<()> {
    std::fs::create_dir_all(&config.storage_root)
        .with_context(|| format!("creating storage root {}", config.storage_root.display()))?;
    let store = LocalFileSystem::new_with_prefix(&config.storage_root)
        .with_context(|| format!("opening storage root {}", config.storage_root.display()))?;
    let store: Arc<dyn ObjectStore> = Arc::new(store);

    run_with_store(config, store, async {
        if let Err(error) = shutdown_signal().await {
            error!(%error, "shutdown signal handler failed");
        }
    })
    .await
}

/// Runs the collector with a caller-provided object store and shutdown future.
///
/// Tests use this entry point to start the daemon in-process. Production code
/// can use it to construct a remote `ObjectStore` without changing collector
/// behavior.
pub async fn run_with_store<S>(
    config: Config,
    store: Arc<dyn ObjectStore>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send,
{
    validate_config(&config)?;
    let recommendation_server = if config.recommendation_enabled {
        Some(
            RecommendationServer::bind(config.recommendation_server.server_config(), store.clone())
                .await
                .context("starting recommendation HTTP server")?,
        )
    } else {
        info!("recommendation HTTP server disabled");
        None
    };
    let listener = BoundListener::bind(&config).await?;
    run_with_listener(config, store, shutdown, listener, recommendation_server).await
}

trait ConnectionListener {
    fn accept(&self) -> impl Future<Output = io::Result<AcceptedConnection>> + Send;
    fn cleanup(&self);
}

impl ConnectionListener for BoundListener {
    fn accept(&self) -> impl Future<Output = io::Result<AcceptedConnection>> + Send {
        BoundListener::accept(self)
    }

    fn cleanup(&self) {
        BoundListener::cleanup(self);
    }
}

async fn run_with_listener<S, L>(
    config: Config,
    store: Arc<dyn ObjectStore>,
    shutdown: S,
    listener: L,
    recommendation_server: Option<RecommendationServer>,
) -> Result<()>
where
    S: Future<Output = ()> + Send,
    L: ConnectionListener + Sync,
{
    let live_memory_bytes = live_memory_bytes(&config)?;

    let counters = Arc::new(CollectorCounters::default());
    let memory = Arc::new(MemoryTracker::default());
    let (writer_tx, writer_rx) = mpsc::channel(config.channel_capacity);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let writer_config = WriterConfig {
        window_seconds: config.window_seconds,
        per_cohort_memory_bytes: config.per_cohort_memory_bytes,
        live_memory_bytes,
        put_timeout: Duration::from_secs(config.put_timeout_seconds),
    };
    let writer = Writer::new(writer_config, store, counters.clone(), memory.clone())?;
    let mut writer_handle = tokio::spawn(writer.run(writer_rx));
    let mut writer_result = None;

    let recommendation_shutdown_timeout = Duration::from_secs(
        config
            .recommendation_server
            .shutdown_drain_timeout_seconds
            .saturating_add(1),
    );
    let (recommendation_shutdown, mut recommendation_handle) = match recommendation_server {
        Some(server) => {
            let shutdown = CancellationToken::new();
            let handle = server.spawn(shutdown.clone());
            (Some(shutdown), Some(handle))
        }
        None => (None, None),
    };
    let mut recommendation_result = None;

    let summary_counters = counters.clone();
    let summary_shutdown = shutdown_rx.clone();
    let summary_handle = tokio::spawn(async move {
        summary_loop(summary_counters, summary_shutdown).await;
        Ok::<(), anyhow::Error>(())
    });
    let reader_config = ReaderConfig {
        max_frame_size: config.max_frame_size,
        live_memory_bytes,
        idle_timeout: Duration::from_secs(config.idle_timeout_seconds),
    };
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);

    loop {
        reap_finished_connections(&mut connections);
        if connections.len() >= config.max_connections {
            tokio::select! {
                _ = &mut shutdown => {
                    info!("shutdown requested");
                    break;
                }
                joined = connections.join_next() => {
                    handle_connection_join(joined);
                }
                joined = &mut writer_handle => {
                    let result = unexpected_task_result(joined, "writer");
                    if let Err(error) = &result {
                        error!(%error, "writer task stopped; shutting down collector");
                    }
                    writer_result = Some(result);
                    break;
                }
                joined = join_optional_unit_task(&mut recommendation_handle) => {
                    let result = unexpected_unit_task_result(
                        joined.expect("disabled recommendation task cannot complete"),
                        "recommendation server",
                    );
                    recommendation_handle = None;
                    recommendation_result = Some(result);
                    break;
                }
            }
            continue;
        }

        tokio::select! {
            _ = &mut shutdown => {
                info!("shutdown requested");
                break;
            }
            joined = &mut writer_handle => {
                let result = unexpected_task_result(joined, "writer");
                if let Err(error) = &result {
                    error!(%error, "writer task stopped; shutting down collector");
                }
                writer_result = Some(result);
                break;
            }
            joined = join_optional_unit_task(&mut recommendation_handle) => {
                let result = unexpected_unit_task_result(
                    joined.expect("disabled recommendation task cannot complete"),
                    "recommendation server",
                );
                recommendation_handle = None;
                recommendation_result = Some(result);
                break;
            }
            accept_result = listener.accept() => {
                let connection = match accept_result {
                    Ok(connection) => connection,
                    Err(error) => {
                        handle_accept_error(error, &counters).await;
                        continue;
                    }
                };
                counters.connections_accepted.fetch_add(1, Ordering::Relaxed);
                let task_tx = writer_tx.clone();
                let task_counters = counters.clone();
                let task_memory = memory.clone();
                let task_config = reader_config;
                let task_shutdown = shutdown_rx.clone();
                spawn_connection(
                    &mut connections,
                    connection,
                    task_tx,
                    task_counters,
                    task_memory,
                    task_config,
                    task_shutdown,
                );
                reap_finished_connections(&mut connections);
            }
        }
    }

    if let Some(shutdown) = &recommendation_shutdown {
        shutdown.cancel();
    }
    drain_connections(&mut connections, &shutdown_tx).await;
    drop(writer_tx);

    let writer_result = match writer_result {
        Some(result) => result,
        None => await_task_shutdown(writer_handle, "writer", WRITER_SHUTDOWN_TIMEOUT).await,
    };
    let summary_result =
        await_task_shutdown(summary_handle, "summary", SUMMARY_SHUTDOWN_TIMEOUT).await;
    let recommendation_result = match (recommendation_result, recommendation_handle) {
        (Some(result), _) => result,
        (None, Some(handle)) => {
            await_unit_task_shutdown(
                handle,
                "recommendation server",
                recommendation_shutdown_timeout,
            )
            .await
        }
        (None, None) => Ok(()),
    };
    listener.cleanup();

    if let Err(error) = &summary_result {
        error!(%error, "summary task shutdown failed");
    }
    if let Err(error) = &recommendation_result {
        error!(%error, "recommendation server task failed");
    }
    writer_result?;
    summary_result?;
    recommendation_result?;
    Ok(())
}

async fn join_optional_unit_task(
    handle: &mut Option<tokio::task::JoinHandle<()>>,
) -> Option<Result<(), JoinError>> {
    match handle {
        Some(handle) => Some(handle.await),
        None => std::future::pending().await,
    }
}

async fn handle_accept_error(error: io::Error, counters: &CollectorCounters) {
    counters.accept_errors.fetch_add(1, Ordering::Relaxed);
    warn!(
        %error,
        kind = ?error.kind(),
        os_error = ?error.raw_os_error(),
        backoff_ms = ACCEPT_ERROR_BACKOFF.as_millis(),
        "accept failed"
    );
    tokio::time::sleep(ACCEPT_ERROR_BACKOFF).await;
}

fn spawn_connection(
    connections: &mut JoinSet<()>,
    connection: AcceptedConnection,
    tx: mpsc::Sender<reader::FrameEvent>,
    counters: Arc<CollectorCounters>,
    memory: Arc<MemoryTracker>,
    config: ReaderConfig,
    shutdown: watch::Receiver<bool>,
) {
    match connection {
        AcceptedConnection::Tcp(stream) => {
            connections.spawn(async move {
                handle_connection(stream, tx, counters, memory, config, shutdown).await;
            });
        }
        AcceptedConnection::Unix(stream) => {
            connections.spawn(async move {
                handle_connection(stream, tx, counters, memory, config, shutdown).await;
            });
        }
    }
}

fn reap_finished_connections(connections: &mut JoinSet<()>) {
    while let Some(joined) = connections.try_join_next() {
        handle_connection_join(Some(joined));
    }
}

async fn drain_connections(connections: &mut JoinSet<()>, shutdown_tx: &watch::Sender<bool>) {
    let connection_drain = tokio::time::sleep(Duration::from_millis(CONNECTION_SHUTDOWN_DRAIN_MS));
    tokio::pin!(connection_drain);
    while !connections.is_empty() {
        tokio::select! {
            joined = connections.join_next() => {
                handle_connection_join(joined);
            }
            _ = &mut connection_drain => {
                break;
            }
        }
    }

    let _ = shutdown_tx.send(true);
    let connection_stop = tokio::time::sleep(Duration::from_millis(CONNECTION_SHUTDOWN_DRAIN_MS));
    tokio::pin!(connection_stop);
    while !connections.is_empty() {
        tokio::select! {
            joined = connections.join_next() => {
                handle_connection_join(joined);
            }
            _ = &mut connection_stop => {
                connections.abort_all();
                break;
            }
        }
    }

    while let Some(joined) = connections.join_next().await {
        handle_connection_join(Some(joined));
    }
}

fn handle_connection_join(joined: Option<Result<(), JoinError>>) {
    match joined {
        Some(Ok(())) | None => {}
        Some(Err(error)) if error.is_cancelled() => {
            debug!(%error, "connection task cancelled during shutdown");
        }
        Some(Err(error)) => {
            warn!(%error, "connection task failed");
        }
    }
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .context("installing SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.context("installing Ctrl-C handler")?;
            }
            _ = sigterm.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .context("installing Ctrl-C handler")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use object_store::memory::InMemory;
    use vectorseam_recommendation_server::RecommendationServerOptions;

    #[tokio::test]
    async fn accept_error_handler_counts_and_returns() {
        let counters = CollectorCounters::default();

        tokio::time::timeout(
            Duration::from_secs(1),
            handle_accept_error(io::Error::from_raw_os_error(24), &counters),
        )
        .await
        .unwrap();

        assert_eq!(counters.accept_errors.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn accept_error_does_not_bypass_shutdown_or_cleanup() {
        let cleanup_called = Arc::new(AtomicBool::new(false));
        let accept_calls = Arc::new(AtomicU64::new(0));
        let listener = FailingListener {
            cleanup_called: cleanup_called.clone(),
            accept_calls: accept_calls.clone(),
        };
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            run_with_listener(
                test_config(),
                store,
                async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                },
                listener,
                Some(
                    RecommendationServer::bind(
                        test_config().recommendation_server.server_config(),
                        Arc::new(InMemory::new()),
                    )
                    .await
                    .unwrap(),
                ),
            ),
        )
        .await
        .unwrap();

        result.unwrap();
        assert!(accept_calls.load(Ordering::Relaxed) >= 1);
        assert!(cleanup_called.load(Ordering::Relaxed));
    }

    struct FailingListener {
        cleanup_called: Arc<AtomicBool>,
        accept_calls: Arc<AtomicU64>,
    }

    impl ConnectionListener for FailingListener {
        fn accept(&self) -> impl Future<Output = io::Result<AcceptedConnection>> + Send {
            let call = self.accept_calls.fetch_add(1, Ordering::Relaxed);
            async move {
                if call == 0 {
                    return Err(io::Error::from_raw_os_error(24));
                }
                std::future::pending().await
            }
        }

        fn cleanup(&self) {
            self.cleanup_called.store(true, Ordering::Relaxed);
        }
    }

    fn test_config() -> Config {
        Config {
            listen: "127.0.0.1:0".parse().unwrap(),
            recommendation_enabled: true,
            recommendation_server: RecommendationServerOptions {
                listen_addr: "127.0.0.1:0".parse().unwrap(),
                ..RecommendationServerOptions::default()
            },
            unix_socket: None,
            storage_root: PathBuf::from("/unused"),
            window_seconds: 60,
            per_cohort_memory_bytes: 8 * 1024 * 1024,
            global_memory_bytes: 64 * 1024 * 1024,
            max_frame_size: 32 * 1024,
            channel_capacity: 16,
            max_connections: 16,
            idle_timeout_seconds: 300,
            put_timeout_seconds: 60,
        }
    }
}
