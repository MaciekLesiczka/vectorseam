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

use anyhow::{Context, Result, anyhow};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinError, JoinHandle, JoinSet};
use tracing::{debug, error, info, warn};

pub use crate::config::Config;
use crate::config::{ReaderConfig, WriterConfig, live_memory_bytes, validate_config};
use crate::counters::{CollectorCounters, summary_loop};
use crate::listener::{AcceptedConnection, BoundListener};
use crate::memory::MemoryTracker;
use crate::reader::handle_connection;
use crate::writer::Writer;

const CONNECTION_SHUTDOWN_DRAIN_MS: u64 = 250;
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(100);
const WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
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
    let listener = BoundListener::bind(&config).await?;
    run_with_listener(config, store, shutdown, listener).await
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
) -> Result<()>
where
    S: Future<Output = ()> + Send,
    L: ConnectionListener + Sync,
{
    validate_config(&config)?;
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

    let summary_counters = counters.clone();
    let summary_shutdown = shutdown_rx.clone();
    let summary_handle = tokio::spawn(async move {
        summary_loop(summary_counters, summary_shutdown).await;
        Ok(())
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
                    let result = writer_runtime_result(joined);
                    if let Err(error) = &result {
                        error!(%error, "writer task stopped; shutting down collector");
                    }
                    writer_result = Some(result);
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
                let result = writer_runtime_result(joined);
                if let Err(error) = &result {
                    error!(%error, "writer task stopped; shutting down collector");
                }
                writer_result = Some(result);
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

    drain_connections(&mut connections, &shutdown_tx).await;
    drop(writer_tx);

    let writer_result = match writer_result {
        Some(result) => result,
        None => await_task_shutdown(writer_handle, "writer", WRITER_SHUTDOWN_TIMEOUT).await,
    };
    let summary_result =
        await_task_shutdown(summary_handle, "summary", SUMMARY_SHUTDOWN_TIMEOUT).await;
    listener.cleanup();

    writer_result?;
    summary_result?;
    Ok(())
}

fn writer_runtime_result(joined: Result<Result<()>, JoinError>) -> Result<()> {
    match joined {
        Ok(Ok(())) => Err(anyhow!("writer task exited unexpectedly")),
        Ok(Err(error)) => Err(error.context("writer task failed")),
        Err(error) => Err(anyhow!("writer task failed: {error}")),
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

async fn await_task_shutdown(
    mut handle: JoinHandle<Result<()>>,
    task_name: &'static str,
    timeout: Duration,
) -> Result<()> {
    match tokio::time::timeout(timeout, &mut handle).await {
        Ok(joined) => joined.map_err(|error| anyhow!("{task_name} task failed: {error}"))?,
        Err(_elapsed) => {
            error!(
                task = task_name,
                timeout_seconds = timeout.as_secs_f64(),
                "task shutdown timed out; aborting task"
            );
            handle.abort();
            match handle.await {
                Ok(result) => result,
                Err(error) if error.is_cancelled() => {
                    warn!(task = task_name, "task aborted after shutdown timeout");
                    Ok(())
                }
                Err(error) => Err(anyhow!("{task_name} task failed after abort: {error}")),
            }
        }
    }
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
