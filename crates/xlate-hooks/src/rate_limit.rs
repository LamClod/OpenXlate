use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use xlate_core::error::XlateError;
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::kernel::EventBus;
use xlate_core::message::KernelEventPayload;

#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitMode {
    Record,
    Enforce,
}

struct WindowCounter {
    timestamps: VecDeque<Instant>,
    window: Duration,
}

impl WindowCounter {
    fn new(window: Duration) -> Self {
        Self {
            timestamps: VecDeque::new(),
            window,
        }
    }

    fn record(&mut self) -> u64 {
        let now = Instant::now();
        let cutoff = now - self.window;
        while self.timestamps.front().is_some_and(|t| *t < cutoff) {
            self.timestamps.pop_front();
        }
        self.timestamps.push_back(now);
        self.timestamps.len() as u64
    }
}

struct TokenWindow {
    entries: VecDeque<(Instant, i64)>,
    window: Duration,
}

impl TokenWindow {
    fn new(window: Duration) -> Self {
        Self {
            entries: VecDeque::new(),
            window,
        }
    }

    fn record(&mut self, tokens: i64) -> i64 {
        let now = Instant::now();
        let cutoff = now - self.window;
        while self.entries.front().is_some_and(|(t, _)| *t < cutoff) {
            self.entries.pop_front();
        }
        self.entries.push_back((now, tokens));
        self.entries.iter().map(|(_, t)| *t).sum()
    }
}

pub struct RateLimitHook {
    rpm_counters: DashMap<String, Mutex<WindowCounter>>,
    tpm_counters: DashMap<String, Mutex<TokenWindow>>,
    window: Duration,
    rpm_limit: u32,
    tpm_limit: u64,
    mode: RateLimitMode,
    event_bus: Option<Arc<EventBus>>,
}

impl RateLimitHook {
    pub fn new() -> Self {
        Self::with_config(600, 1_000_000, None)
    }

    pub fn with_config(rpm: u32, tpm: u64, event_bus: Option<Arc<EventBus>>) -> Self {
        Self {
            rpm_counters: DashMap::new(),
            tpm_counters: DashMap::new(),
            window: Duration::from_secs(60),
            rpm_limit: rpm,
            tpm_limit: tpm,
            mode: RateLimitMode::Record,
            event_bus,
        }
    }

    pub fn with_mode(mut self, mode: &str) -> Self {
        self.mode = match mode {
            "enforce" => RateLimitMode::Enforce,
            _ => RateLimitMode::Record,
        };
        self
    }
}

impl Default for RateLimitHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for RateLimitHook {
    fn name(&self) -> &str {
        "rate-limit"
    }

    fn priority(&self) -> i32 {
        40
    }

    async fn pre_route(&self, ctx: &mut HookContext) -> HookVerdict {
        let key = ctx.request.model.clone();
        let entry = self
            .rpm_counters
            .entry(key.clone())
            .or_insert_with(|| Mutex::new(WindowCounter::new(self.window)));

        if let Ok(mut counter) = entry.lock() {
            let rpm = counter.record();
            tracing::debug!(model = %key, rpm, "rate-limit recorded");

            if rpm > self.rpm_limit as u64 {
                tracing::warn!(model = %key, rpm, limit = self.rpm_limit, "RPM limit exceeded");
                if let Some(ref eb) = self.event_bus {
                    eb.emit(KernelEventPayload::RateLimitExceeded {
                        metric: format!("rpm:{}", key),
                        current: rpm,
                        limit: self.rpm_limit as u64,
                    });
                }
                if self.mode == RateLimitMode::Enforce {
                    return HookVerdict::Abort(XlateError::RateLimited(format!(
                        "RPM limit exceeded: {} > {}",
                        rpm, self.rpm_limit
                    )));
                }
            }
        }

        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let total_tokens = ctx.metrics.usage.input_tokens.unwrap_or(0)
            + ctx.metrics.usage.output_tokens.unwrap_or(0);

        if total_tokens > 0 {
            let key = ctx.request.model.clone();

            let tpm = {
                let entry = self
                    .tpm_counters
                    .entry(key.clone())
                    .or_insert_with(|| Mutex::new(TokenWindow::new(self.window)));
                entry.lock().ok().map(|mut c| c.record(total_tokens))
            };

            if let Some(tpm) = tpm {
                if tpm as u64 > self.tpm_limit {
                    tracing::warn!(model = %key, tpm, limit = self.tpm_limit, "TPM limit exceeded");
                    if let Some(ref eb) = self.event_bus {
                        eb.emit(KernelEventPayload::RateLimitExceeded {
                            metric: format!("tpm:{}", key),
                            current: tpm as u64,
                            limit: self.tpm_limit,
                        });
                    }
                }
            }
        }

        HookVerdict::Continue
    }

    fn snapshot(&self) -> Option<serde_json::Value> {
        let current_rpm: u64 = self
            .rpm_counters
            .iter()
            .filter_map(|entry| {
                entry.value().lock().ok().map(|c| c.timestamps.len() as u64)
            })
            .sum();
        let current_tpm: u64 = self
            .tpm_counters
            .iter()
            .filter_map(|entry| {
                entry
                    .value()
                    .lock()
                    .ok()
                    .map(|c| c.entries.iter().map(|(_, t)| *t).sum::<i64>() as u64)
            })
            .sum();
        Some(json!({
            "current_rpm": current_rpm,
            "current_tpm": current_tpm,
            "limit_rpm": self.rpm_limit as u64,
            "limit_tpm": self.tpm_limit,
        }))
    }
}
