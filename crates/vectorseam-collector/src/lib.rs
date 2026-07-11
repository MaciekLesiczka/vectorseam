mod config;
mod counters;
mod listener;
mod memory;
mod reader;
mod time;
mod writer;

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinError, JoinSet};
use tracing::{debug, error, info, warn};

pub use crate::config::Config;
use crate::config::{ReaderConfig, WriterConfig, validate_config};
use crate::counters::{CollectorCounters, summary_loop};
use crate::listener::{AcceptedConnection, BoundListener};
use crate::memory::MemoryTracker;
use crate::reader::handle_connection;
//use crate::writer::Writer;

const CONNECTION_SHUTDOWN_DRAIN_MS: u64 = 250;

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

    let listener = BoundListener::bind(&config).await?;
    let counters = Arc::new(CollectorCounters::default());
    let memory = Arc::new(MemoryTracker::default());
    let (writer_tx, writer_rx) = mpsc::channel(config.channel_capacity);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let writer_config = WriterConfig {
        window_seconds: config.window_seconds,
        per_cohort_memory_bytes: config.per_cohort_memory_bytes,
        global_memory_bytes: config.global_memory_bytes,
    };
    //let writer = Writer::new(writer_config, store, counters.clone(), memory.clone())?;
    //let writer_handle = tokio::spawn(writer.run(writer_rx));

    let summary_handle = tokio::spawn(summary_loop(counters.clone(), shutdown_rx.clone()));
    let reader_config = ReaderConfig {
        max_frame_size: config.max_frame_size,
        global_memory_bytes: config.global_memory_bytes,
    };
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);

    loop {
        if connections.len() >= config.max_connections {
            tokio::select! {
                _ = &mut shutdown => {
                    info!("shutdown requested");
                    break;
                }
                joined = connections.join_next() => {
                    handle_connection_join(joined);
                }
            }
            continue;
        }

        tokio::select! {
            _ = &mut shutdown => {
                info!("shutdown requested");
                break;
            }
            accept_result = listener.accept() => {
                let connection = accept_result?;
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
            }
        }
    }

    drain_connections(&mut connections, &shutdown_tx).await;
    drop(writer_tx);

    // writer_handle
    //     .await
    //     .map_err(|error| anyhow!("writer task failed: {error}"))??;

    summary_handle
        .await
        .map_err(|error| anyhow!("summary task failed: {error}"))?;

    listener.cleanup();
    Ok(())
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
