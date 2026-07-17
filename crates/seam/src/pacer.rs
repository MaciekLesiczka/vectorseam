//! Serialized per-data-source duty-cycle pacing.

use std::future::Future;
use std::time::Duration;

use thiserror::Error;
use tokio::time::{Instant, sleep_until};

/// Invalid duty-cycle pacing configuration.
#[derive(Debug, Error)]
pub(crate) enum PacerError {
    #[error("db_share must be finite and in (0, 1]")]
    InvalidShare,
}

#[derive(Debug)]
pub(crate) struct StatementPacer {
    db_share: f64,
    next_allowed: Instant,
    statement_count: u64,
    total_busy: Duration,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Timed<T> {
    pub(crate) value: T,
    pub(crate) elapsed: Duration,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PacerSnapshot {
    pub(crate) statement_count: u64,
    pub(crate) total_busy: Duration,
}

impl StatementPacer {
    pub(crate) fn new(db_share: f64) -> Result<Self, PacerError> {
        if !(db_share.is_finite() && db_share > 0.0 && db_share <= 1.0) {
            return Err(PacerError::InvalidShare);
        }
        Ok(Self {
            db_share,
            next_allowed: Instant::now(),
            statement_count: 0,
            total_busy: Duration::ZERO,
        })
    }

    /// Runs one already-serialized statement after the prior cooldown.
    pub(crate) async fn run<F, T, E>(&mut self, statement: F) -> Result<Timed<T>, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        sleep_until(self.next_allowed).await;

        let started = Instant::now();
        let result = statement.await;
        let elapsed = started.elapsed();
        self.statement_count = self.statement_count.saturating_add(1);
        self.total_busy = self.total_busy.saturating_add(elapsed);

        let cooldown = elapsed.mul_f64((1.0 - self.db_share) / self.db_share);
        self.next_allowed = Instant::now() + cooldown;
        result.map(|value| Timed { value, elapsed })
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> PacerSnapshot {
        PacerSnapshot {
            statement_count: self.statement_count,
            total_busy: self.total_busy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn d1_duty_cycle_20_percent_wall_time_bound() {
        let mut pacer = StatementPacer::new(0.20).unwrap();
        let started = Instant::now();
        for _ in 0..50 {
            pacer
                .run(async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok::<_, ()>(())
                })
                .await
                .unwrap();
        }
        let total_elapsed = started.elapsed().as_secs_f64();
        let snapshot = pacer.snapshot();
        let total_busy = snapshot.total_busy.as_secs_f64();

        assert_eq!(snapshot.statement_count, 50);
        assert!(total_elapsed >= 0.95 * (total_busy / 0.20));
        assert!(total_busy / total_elapsed <= 0.21);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_statement_consumes_the_same_cooldown_budget() {
        let mut pacer = StatementPacer::new(0.20).unwrap();
        let started = Instant::now();
        let failure = pacer
            .run(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Err::<(), _>("expected failure")
            })
            .await;
        assert_eq!(failure.unwrap_err(), "expected failure");
        pacer.run(async { Ok::<_, ()>(()) }).await.unwrap();

        assert!(started.elapsed() >= Duration::from_millis(250));
        assert_eq!(pacer.snapshot().statement_count, 2);
    }
}
