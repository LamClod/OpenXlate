use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xlate_core::error::XlateError;
use xlate_core::hook::{Hook, HookContext, HookVerdict, RectifyAction};
use xlate_core::kernel::EventBus;
use xlate_core::message::KernelEventPayload;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Closed,
    Open { since: Instant },
    HalfOpen,
}

struct BreakerState {
    state: State,
    failure_count: u32,
    success_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Granularity {
    PerTarget,
    PerProvider,
    PerVendor,
    All,
}

pub struct CircuitOpenCount(pub u64);

pub struct CircuitBreakerHook {
    breakers: DashMap<String, BreakerState>,
    failure_threshold: u32,
    success_threshold: u32,
    open_timeout: Duration,
    granularity: Granularity,
    event_bus: Option<Arc<EventBus>>,
}

impl CircuitBreakerHook {
    pub fn new(failure_threshold: u32, success_threshold: u32, open_timeout_ms: u64) -> Self {
        Self {
            breakers: DashMap::new(),
            failure_threshold,
            success_threshold,
            open_timeout: Duration::from_millis(open_timeout_ms),
            granularity: Granularity::All,
            event_bus: None,
        }
    }

    pub fn with_event_bus(mut self, eb: Arc<EventBus>) -> Self {
        self.event_bus = Some(eb);
        self
    }

    pub fn with_granularity(mut self, g: &str) -> Self {
        self.granularity = match g {
            "per-target" => Granularity::PerTarget,
            "per-provider" => Granularity::PerProvider,
            "per-vendor" => Granularity::PerVendor,
            _ => Granularity::All,
        };
        self
    }

    fn target_key(ctx: &HookContext) -> Option<String> {
        ctx.route
            .as_ref()
            .map(|r| format!("target:{}", r.target.id))
    }

    fn provider_key(ctx: &HookContext) -> Option<String> {
        ctx.provider_config
            .as_ref()
            .map(|c| format!("provider:{}:{}", c.plugin, c.base_url))
    }

    fn vendor_key(ctx: &HookContext) -> Option<String> {
        ctx.provider_config
            .as_ref()
            .map(|c| format!("vendor:{}", c.plugin))
    }

    fn keys_for_granularity(&self, ctx: &HookContext) -> Vec<String> {
        match self.granularity {
            Granularity::PerTarget => Self::target_key(ctx).into_iter().collect(),
            Granularity::PerProvider => Self::provider_key(ctx).into_iter().collect(),
            Granularity::PerVendor => Self::vendor_key(ctx).into_iter().collect(),
            Granularity::All => [
                Self::target_key(ctx),
                Self::provider_key(ctx),
                Self::vendor_key(ctx),
            ]
            .into_iter()
            .flatten()
            .collect(),
        }
    }
}

impl Default for CircuitBreakerHook {
    fn default() -> Self {
        Self::new(5, 2, 60_000)
    }
}

#[async_trait]
impl Hook for CircuitBreakerHook {
    fn name(&self) -> &str {
        "circuit-breaker"
    }

    fn priority(&self) -> i32 {
        300
    }

    async fn pre_send(&self, ctx: &mut HookContext) -> HookVerdict {
        let open_count = self
            .breakers
            .iter()
            .filter(|e| matches!(e.state, State::Open { .. }))
            .count();
        ctx.extensions.insert(CircuitOpenCount(open_count as u64));

        for key in self.keys_for_granularity(ctx) {
            if let Some(mut entry) = self.breakers.get_mut(&key) {
                if let State::Open { since } = entry.state {
                    if since.elapsed() >= self.open_timeout {
                        entry.state = State::HalfOpen;
                        entry.success_count = 0;
                        tracing::info!(breaker = %key, "circuit half-open");
                    } else {
                        tracing::warn!(breaker = %key, "circuit open, switching target");
                        return HookVerdict::Rectify(RectifyAction::SwitchTarget);
                    }
                }
            }
        }

        HookVerdict::Continue
    }

    async fn on_error(&self, ctx: &mut HookContext, _error: &XlateError) -> HookVerdict {
        for key in self.keys_for_granularity(ctx) {
            let mut entry =
                self.breakers
                    .entry(key.clone())
                    .or_insert_with(|| BreakerState {
                        state: State::Closed,
                        failure_count: 0,
                        success_count: 0,
                    });

            entry.failure_count += 1;
            entry.success_count = 0;

            if entry.failure_count >= self.failure_threshold
                && !matches!(entry.state, State::Open { .. })
            {
                entry.state = State::Open {
                    since: Instant::now(),
                };
                tracing::warn!(
                    breaker = %key,
                    failures = entry.failure_count,
                    "circuit opened"
                );
                if let Some(ref eb) = self.event_bus {
                    eb.emit(KernelEventPayload::CircuitOpened {
                        target_id: key.clone(),
                        failure_count: entry.failure_count,
                    });
                }
            }
        }

        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        if !ctx.metrics.success {
            return HookVerdict::Continue;
        }

        for key in self.keys_for_granularity(ctx) {
            if let Some(mut entry) = self.breakers.get_mut(&key) {
                match entry.state {
                    State::HalfOpen => {
                        entry.success_count += 1;
                        if entry.success_count >= self.success_threshold {
                            entry.state = State::Closed;
                            entry.failure_count = 0;
                            tracing::info!(breaker = %key, "circuit closed");
                        }
                    }
                    State::Closed => {
                        entry.failure_count = 0;
                    }
                    _ => {}
                }
            }
        }

        HookVerdict::Continue
    }

    fn snapshot(&self) -> Option<serde_json::Value> {
        let breakers: Vec<serde_json::Value> = self
            .breakers
            .iter()
            .map(|entry| {
                let (state_str, since_ms) = match entry.state {
                    State::Closed => ("closed", None),
                    State::Open { since } => ("open", Some(since.elapsed().as_millis() as u64)),
                    State::HalfOpen => ("half_open", None),
                };
                json!({
                    "key": entry.key().clone(),
                    "state": state_str,
                    "open_since_ms": since_ms,
                    "failure_count": entry.failure_count,
                    "success_count": entry.success_count,
                })
            })
            .collect();
        Some(json!({ "circuit_breakers": breakers }))
    }
}
