use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};

pub(crate) fn duration_until_unix_second(target_second: u64) -> Duration {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(now) => {
            let now_seconds = now.as_secs();
            if target_second <= now_seconds {
                Duration::ZERO
            } else {
                Duration::from_secs(target_second - now_seconds)
            }
        }
        Err(_error) => Duration::ZERO,
    }
}

pub(crate) fn unix_seconds_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before unix epoch")?
        .as_secs())
}

pub(crate) fn unix_micros_now() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before unix epoch")?;
    duration
        .as_secs()
        .checked_mul(1_000_000)
        .and_then(|micros| micros.checked_add(u64::from(duration.subsec_micros())))
        .ok_or_else(|| anyhow!("unix microsecond timestamp overflowed"))
}
