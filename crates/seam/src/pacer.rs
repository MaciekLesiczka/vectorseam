//! Serialized per-data-source duty-cycle pacing.
//!
//! The paced unit is one whole piece of database work — for the tuner, one
//! complete sample transaction. Pacing sleeps only between units, never
//! inside one, so an open `REPEATABLE READ` snapshot is held for the minimum
//! possible time; the cooldown after a unit is proportional to the full wall
//! time that unit kept the database busy.

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
pub(crate) struct DutyCyclePacer {
    db_share: f64,
    next_allowed: Instant,
    work_count: u64,
    total_busy: Duration,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PacerSnapshot {
    pub(crate) work_count: u64,
    pub(crate) total_busy: Duration,
}

impl DutyCyclePacer {
    pub(crate) fn new(db_share: f64) -> Result<Self, PacerError> {
        if !(db_share.is_finite() && db_share > 0.0 && db_share <= 1.0) {
            return Err(PacerError::InvalidShare);
        }
        Ok(Self {
            db_share,
            next_allowed: Instant::now(),
            work_count: 0,
            total_busy: Duration::ZERO,
        })
    }

    /// Runs one already-serialized unit of database work — a whole sample
    /// transaction or a single statement — after the prior cooldown, then
    /// charges its full wall time to the duty cycle. Failed units consume
    /// the same budget as successful ones.
    pub(crate) async fn run<F, T, E>(&mut self, work: F) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        sleep_until(self.next_allowed).await;

        let started = Instant::now();
        let result = work.await;
        let elapsed = started.elapsed();
        self.work_count = self.work_count.saturating_add(1);
        self.total_busy = self.total_busy.saturating_add(elapsed);

        let cooldown = elapsed.mul_f64((1.0 - self.db_share) / self.db_share);
        self.next_allowed = Instant::now() + cooldown;
        result
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> PacerSnapshot {
        PacerSnapshot {
            work_count: self.work_count,
            total_busy: self.total_busy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn d1_duty_cycle_20_percent_wall_time_bound() {
        let mut pacer = DutyCyclePacer::new(0.20).unwrap();
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

        assert_eq!(snapshot.work_count, 50);
        assert!(total_elapsed >= 0.95 * (total_busy / 0.20));
        assert!(total_busy / total_elapsed <= 0.21);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_work_consumes_the_same_cooldown_budget() {
        let mut pacer = DutyCyclePacer::new(0.20).unwrap();
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
        assert_eq!(pacer.snapshot().work_count, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn no_sleep_occurs_inside_a_paced_unit() {
        let mut pacer = DutyCyclePacer::new(0.20).unwrap();
        pacer
            .run(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok::<_, ()>(())
            })
            .await
            .unwrap();

        // The next unit starts only after the cooldown, but once started it
        // runs to completion with no pacer-injected delay between its steps.
        let unit_started = Instant::now();
        pacer
            .run(async {
                let step_started = Instant::now();
                tokio::time::sleep(Duration::from_millis(10)).await;
                tokio::time::sleep(Duration::from_millis(10)).await;
                assert_eq!(step_started.elapsed(), Duration::from_millis(20));
                Ok::<_, ()>(())
            })
            .await
            .unwrap();
        // 200 ms cooldown from the first 50 ms unit, then the 20 ms unit.
        assert_eq!(unit_started.elapsed(), Duration::from_millis(220));
    }
}
