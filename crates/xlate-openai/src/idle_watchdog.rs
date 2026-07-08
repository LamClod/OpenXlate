//! Stream idle timeout watchdog.
//!
//! Cancels the stream if no effective content arrives within a configurable
//! timeout window (default: 4 minutes).
//!
//! Ported from the original Go implementation's `providerStreamIdleWatchdog`.

use std::time::Duration;
use tokio::sync::watch;
use tokio::time::Instant;

const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(240); // 4 minutes
const MIN_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// A watchdog that fires if `mark_effective_content()` is not called within
/// the configured timeout window.
pub struct IdleWatchdog {
    timeout: Duration,
    last_activity_tx: watch::Sender<Instant>,
    timed_out: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl IdleWatchdog {
    /// Create a new idle watchdog. Returns the watchdog and a
    /// `tokio::sync::watch::Receiver` that the background task uses.
    pub fn new(timeout_ms: Option<u64>) -> (Self, tokio::task::JoinHandle<()>) {
        let timeout = normalize_timeout(timeout_ms);
        let now = Instant::now();
        let (tx, mut rx) = watch::channel(now);
        let timed_out = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let timed_out_clone = timed_out.clone();

        let handle = tokio::spawn(async move {
            loop {
                let deadline = *rx.borrow() + timeout;
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                        break;
                    }
                    result = rx.changed() => {
                        if result.is_err() {
                            // Sender dropped, watchdog no longer needed
                            break;
                        }
                        // Activity was recorded, loop again with new deadline
                    }
                }
            }
        });

        let watchdog = Self {
            timeout,
            last_activity_tx: tx,
            timed_out,
        };
        (watchdog, handle)
    }

    /// Signal that effective content was received, resetting the idle timer.
    pub fn mark_effective_content(&self) {
        let _ = self.last_activity_tx.send(Instant::now());
    }

    /// Check if the watchdog has fired (timed out).
    pub fn is_timed_out(&self) -> bool {
        self.timed_out.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Return an idle timeout error if the watchdog has fired.
    pub fn check(&self) -> Result<(), xlate_core::XlateError> {
        if self.is_timed_out() {
            Err(xlate_core::XlateError::IdleTimeout(self.timeout.as_millis() as u64))
        } else {
            Ok(())
        }
    }
}

fn normalize_timeout(timeout_ms: Option<u64>) -> Duration {
    match timeout_ms {
        None | Some(0) => DEFAULT_IDLE_TIMEOUT,
        Some(ms) => {
            let duration = Duration::from_millis(ms);
            if duration < MIN_IDLE_TIMEOUT {
                MIN_IDLE_TIMEOUT
            } else {
                duration
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_timeout() {
        assert_eq!(normalize_timeout(None), DEFAULT_IDLE_TIMEOUT);
        assert_eq!(normalize_timeout(Some(0)), DEFAULT_IDLE_TIMEOUT);
        assert_eq!(normalize_timeout(Some(10_000)), MIN_IDLE_TIMEOUT);
        assert_eq!(normalize_timeout(Some(300_000)), Duration::from_millis(300_000));
    }
}
