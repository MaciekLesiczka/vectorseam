use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::error;

#[derive(Debug, Default)]
pub(crate) struct MemoryTracker {
    used_bytes: AtomicUsize,
}

impl MemoryTracker {
    pub(crate) fn try_reserve(self: &Arc<Self>, bytes: usize, limit: usize) -> Option<MemoryGuard> {
        let mut current = self.used_bytes.load(Ordering::Relaxed);
        loop {
            let next = current.checked_add(bytes)?;
            if next > limit {
                return None;
            }

            match self.used_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_previous) => {
                    return Some(MemoryGuard {
                        tracker: self.clone(),
                        bytes,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn release(&self, bytes: usize) {
        let mut current = self.used_bytes.load(Ordering::Relaxed);
        let mut release_bytes = bytes;
        let mut underflow_logged = false;
        loop {
            if release_bytes > current {
                if !underflow_logged {
                    error!(
                        used_bytes = current,
                        release_bytes = bytes,
                        "memory tracker release exceeded reserved bytes"
                    );
                    underflow_logged = true;
                }
                release_bytes = current;
            }
            let next = current - release_bytes;

            match self.used_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_previous) => return,
                Err(actual) => current = actual,
            }
        }
    }

    pub(crate) fn used_bytes(&self) -> usize {
        self.used_bytes.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub(crate) struct MemoryGuard {
    tracker: Arc<MemoryTracker>,
    bytes: usize,
}

impl MemoryGuard {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for MemoryGuard {
    fn drop(&mut self) {
        self.tracker.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_until_limit_and_releases() {
        let tracker = Arc::new(MemoryTracker::default());

        let first = tracker.try_reserve(8, 10).unwrap();
        assert!(tracker.try_reserve(3, 10).is_none());
        assert_eq!(tracker.used_bytes(), 8);

        drop(first);
        assert_eq!(tracker.used_bytes(), 0);

        let second = tracker.try_reserve(10, 10).unwrap();
        assert_eq!(tracker.used_bytes(), 10);
        drop(second);
        assert_eq!(tracker.used_bytes(), 0);
    }

    #[test]
    fn release_underflow_clamps_without_store_reset() {
        let tracker = MemoryTracker::default();
        tracker.used_bytes.store(5, Ordering::Relaxed);

        tracker.release(8);

        assert_eq!(tracker.used_bytes(), 0);
    }
}
