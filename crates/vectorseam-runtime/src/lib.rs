//! Shared Tokio task-supervision helpers for VectorSeam services.

use std::time::Duration;

use anyhow::{Result, anyhow};
use tokio::task::{JoinError, JoinHandle};
use tracing::{error, warn};

/// Converts an unexpectedly completed critical task into a host error.
///
/// This observes both the Tokio `JoinError` and the task's own result. A clean
/// task return is also an error because the caller uses this helper only while
/// the host is expected to remain running.
pub fn unexpected_task_result<E>(
    joined: Result<Result<(), E>, JoinError>,
    task_name: &str,
) -> Result<()>
where
    E: Into<anyhow::Error>,
{
    match joined {
        Ok(Ok(())) => Err(anyhow!("{task_name} exited unexpectedly")),
        Ok(Err(error)) => Err(error.into().context(format!("{task_name} failed"))),
        Err(error) => Err(anyhow!("{task_name} task failed: {error}")),
    }
}

/// Converts an unexpectedly completed critical unit-output task into a host error.
pub fn unexpected_unit_task_result(joined: Result<(), JoinError>, task_name: &str) -> Result<()> {
    match joined {
        Ok(()) => Err(anyhow!("{task_name} exited unexpectedly")),
        Err(error) => Err(anyhow!("{task_name} task failed: {error}")),
    }
}

/// Awaits a task during graceful shutdown and aborts it after `timeout`.
///
/// The task must receive its normal shutdown notification before this function
/// is called. Abortion is a forced fallback; the aborted task is still joined
/// so its termination is observed.
pub async fn await_task_shutdown<E>(
    mut handle: JoinHandle<Result<(), E>>,
    task_name: &str,
    timeout: Duration,
) -> Result<()>
where
    E: Into<anyhow::Error>,
{
    match tokio::time::timeout(timeout, &mut handle).await {
        Ok(joined) => joined
            .map_err(|error| anyhow!("{task_name} task failed: {error}"))?
            .map_err(Into::into),
        Err(_elapsed) => {
            error!(
                task = task_name,
                timeout_seconds = timeout.as_secs_f64(),
                "task shutdown timed out; aborting task"
            );
            handle.abort();
            match handle.await {
                Ok(result) => result.map_err(Into::into),
                Err(error) if error.is_cancelled() => {
                    warn!(task = task_name, "task aborted after shutdown timeout");
                    Ok(())
                }
                Err(error) => Err(anyhow!("{task_name} task failed after abort: {error}")),
            }
        }
    }
}

/// Awaits a unit-output task during graceful shutdown and aborts it after `timeout`.
///
/// This is the unit-output counterpart to [`await_task_shutdown`] for services
/// whose operational connection failures are handled inside their task.
pub async fn await_unit_task_shutdown(
    mut handle: JoinHandle<()>,
    task_name: &str,
    timeout: Duration,
) -> Result<()> {
    match tokio::time::timeout(timeout, &mut handle).await {
        Ok(joined) => joined.map_err(|error| anyhow!("{task_name} task failed: {error}")),
        Err(_elapsed) => {
            error!(
                task = task_name,
                timeout_seconds = timeout.as_secs_f64(),
                "task shutdown timed out; aborting task"
            );
            handle.abort();
            match handle.await {
                Ok(()) => Ok(()),
                Err(error) if error.is_cancelled() => {
                    warn!(task = task_name, "task aborted after shutdown timeout");
                    Ok(())
                }
                Err(error) => Err(anyhow!("{task_name} task failed after abort: {error}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::future;

    use thiserror::Error;

    #[derive(Debug, Error)]
    #[error("test failure")]
    struct TestError;

    #[tokio::test]
    async fn observes_task_error() {
        let handle = tokio::spawn(async { Err::<(), _>(TestError) });

        let error = await_task_shutdown(handle, "test", Duration::from_secs(1))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "test failure");
    }

    #[tokio::test]
    async fn timeout_aborts_and_joins_task() {
        let handle = tokio::spawn(future::pending::<Result<(), TestError>>());

        let result = await_task_shutdown(handle, "test", Duration::from_millis(1)).await;

        result.unwrap();
    }

    #[tokio::test]
    async fn timeout_aborts_and_joins_unit_task() {
        let handle = tokio::spawn(future::pending::<()>());

        let result = await_unit_task_shutdown(handle, "test", Duration::from_millis(1)).await;

        result.unwrap();
    }

    #[tokio::test]
    async fn clean_critical_task_exit_is_unexpected() {
        let joined = tokio::spawn(async { Ok::<(), TestError>(()) }).await;

        let error = unexpected_task_result(joined, "test").unwrap_err();

        assert_eq!(error.to_string(), "test exited unexpectedly");
    }

    #[tokio::test]
    async fn clean_critical_unit_task_exit_is_unexpected() {
        let joined = tokio::spawn(async {}).await;

        let error = unexpected_unit_task_result(joined, "test").unwrap_err();

        assert_eq!(error.to_string(), "test exited unexpectedly");
    }
}
