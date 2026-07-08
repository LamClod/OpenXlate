use crate::circuit_breaker::CircuitOpenCount;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use xlate_core::config::AlertRuleConfig;
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::kernel::EventBus;
use xlate_core::message::KernelEventPayload;
use xlate_core::store::{AlertEvent, Store};

struct RuleState {
    config: AlertRuleConfig,
    window: VecDeque<(Instant, f64)>,
    last_triggered: Option<Instant>,
}

impl RuleState {
    fn new(config: AlertRuleConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            last_triggered: None,
        }
    }

    fn record(&mut self, value: f64) {
        let now = Instant::now();
        let window_dur = Duration::from_secs(self.config.window_minutes as u64 * 60);
        let cutoff = now - window_dur;
        while self.window.front().is_some_and(|(t, _)| *t < cutoff) {
            self.window.pop_front();
        }
        self.window.push_back((now, value));
    }

    fn check(&mut self) -> Option<(f64, f64)> {
        if self.window.is_empty() {
            return None;
        }

        if let Some(last) = self.last_triggered {
            let cooldown = Duration::from_secs(self.config.cooldown_minutes as u64 * 60);
            if last.elapsed() < cooldown {
                return None;
            }
        }

        let current = match self.config.metric.as_str() {
            "error_rate" => {
                let errors = self.window.iter().filter(|(_, v)| *v < 0.5).count();
                errors as f64 / self.window.len() as f64
            }
            _ => {
                let sum: f64 = self.window.iter().map(|(_, v)| v).sum();
                sum / self.window.len() as f64
            }
        };

        let triggered = match self.config.operator.as_str() {
            ">" => current > self.config.threshold,
            ">=" => current >= self.config.threshold,
            "<" => current < self.config.threshold,
            "<=" => current <= self.config.threshold,
            _ => current > self.config.threshold,
        };

        if triggered {
            self.last_triggered = Some(Instant::now());
            Some((current, self.config.threshold))
        } else {
            None
        }
    }
}

pub struct AlertHook {
    rules: Mutex<Vec<RuleState>>,
    event_bus: Option<Arc<EventBus>>,
    store: Option<Arc<dyn Store>>,
}

impl AlertHook {
    pub fn new(latency_threshold_ms: f64, error_rate_threshold: f64, _window_size: usize) -> Self {
        let rules = vec![
            RuleState::new(AlertRuleConfig {
                name: "high-latency".into(),
                metric: "ttft_p99".into(),
                operator: ">".into(),
                threshold: latency_threshold_ms,
                window_minutes: 5,
                cooldown_minutes: 30,
            }),
            RuleState::new(AlertRuleConfig {
                name: "high-error-rate".into(),
                metric: "error_rate".into(),
                operator: ">".into(),
                threshold: error_rate_threshold,
                window_minutes: 5,
                cooldown_minutes: 30,
            }),
            RuleState::new(AlertRuleConfig {
                name: "circuit-open".into(),
                metric: "circuit_open_count".into(),
                operator: ">".into(),
                threshold: 0.0,
                window_minutes: 5,
                cooldown_minutes: 30,
            }),
        ];
        Self {
            rules: Mutex::new(rules),
            event_bus: None,
            store: None,
        }
    }

    pub fn from_config(rules: Vec<AlertRuleConfig>) -> Self {
        let states = rules.into_iter().map(RuleState::new).collect();
        Self {
            rules: Mutex::new(states),
            event_bus: None,
            store: None,
        }
    }

    pub fn with_event_bus(mut self, eb: Arc<EventBus>) -> Self {
        self.event_bus = Some(eb);
        self
    }

    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    fn extract_metric(ctx: &HookContext, metric: &str) -> Option<f64> {
        match metric {
            "error_rate" => Some(if ctx.metrics.success { 1.0 } else { 0.0 }),
            "ttft_p99" | "ttft" => ctx.metrics.ttft_ms,
            "duration" | "latency" => ctx.metrics.total_ms,
            "input_tokens" => ctx.metrics.usage.input_tokens.map(|t| t as f64),
            "output_tokens" => ctx.metrics.usage.output_tokens.map(|t| t as f64),
            "circuit_open_count" => ctx.extensions.get::<CircuitOpenCount>().map(|c| c.0 as f64),
            _ => ctx.metrics.total_ms,
        }
    }
}

impl Default for AlertHook {
    fn default() -> Self {
        Self::new(10_000.0, 0.5, 100)
    }
}

#[async_trait]
impl Hook for AlertHook {
    fn name(&self) -> &str {
        "alert"
    }

    fn priority(&self) -> i32 {
        50
    }

    fn snapshot(&self) -> Option<serde_json::Value> {
        let rules: Vec<serde_json::Value> = self
            .rules
            .lock()
            .map(|rules| {
                rules
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "name": r.config.name,
                            "metric": r.config.metric,
                            "operator": r.config.operator,
                            "threshold": r.config.threshold,
                            "window_size": r.window.len(),
                            "triggered": r.last_triggered.is_some(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Some(serde_json::json!({ "rules": rules }))
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let mut alerts = Vec::new();

        if let Ok(mut rules) = self.rules.lock() {
            for rule in rules.iter_mut() {
                if let Some(value) = Self::extract_metric(ctx, &rule.config.metric) {
                    rule.record(value);
                    if let Some((current, threshold)) = rule.check() {
                        alerts.push((
                            rule.config.name.clone(),
                            rule.config.metric.clone(),
                            current,
                            threshold,
                        ));
                    }
                }
            }
        }

        for (name, metric, current, threshold) in alerts {
            let message = format!(
                "{} = {:.4} exceeds threshold {:.4}",
                metric, current, threshold
            );
            tracing::warn!(
                alert = %name,
                metric = %metric,
                current = current,
                threshold = threshold,
                "alert triggered"
            );
            if let Some(ref eb) = self.event_bus {
                eb.emit(KernelEventPayload::Alert {
                    name: name.clone(),
                    message: message.clone(),
                    value: Some(current),
                    threshold: Some(threshold),
                });
            }
            if let Some(ref store) = self.store {
                let record = AlertEvent {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64,
                    alert_type: name,
                    message,
                    model: Some(ctx.request.model.clone()),
                    provider: ctx.provider_config.as_ref().map(|c| c.plugin.clone()),
                    value: Some(current),
                    threshold: Some(threshold),
                };
                let store = store.clone();
                tokio::spawn(async move {
                    if let Err(e) = store.record_alert(record).await {
                        tracing::warn!(error = %e, "failed to persist alert");
                    }
                });
            }
        }

        HookVerdict::Continue
    }
}
