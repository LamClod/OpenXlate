use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::config::RouteRuleConfig;
use crate::error::XlateError;
use crate::hook::HookContext;
use crate::pricing::PricingCatalog;
use crate::provider::*;

pub struct LatencyTracker {
    targets: DashMap<String, Mutex<VecDeque<f64>>>,
    window_size: usize,
}

impl LatencyTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            targets: DashMap::new(),
            window_size,
        }
    }

    pub fn record(&self, target_id: &str, latency_ms: f64) {
        let ws = self.window_size;
        self.targets
            .entry(target_id.to_string())
            .or_insert_with(|| Mutex::new(VecDeque::with_capacity(ws)));
        if let Some(entry) = self.targets.get(target_id) {
            if let Ok(mut buf) = entry.lock() {
                if buf.len() >= self.window_size {
                    buf.pop_front();
                }
                buf.push_back(latency_ms);
            }
        }
    }

    pub fn p50(&self, target_id: &str) -> Option<f64> {
        self.percentile(target_id, 0.50)
    }

    pub fn p95(&self, target_id: &str) -> Option<f64> {
        self.percentile(target_id, 0.95)
    }

    pub fn p99(&self, target_id: &str) -> Option<f64> {
        self.percentile(target_id, 0.99)
    }

    fn percentile(&self, target_id: &str, p: f64) -> Option<f64> {
        let entry = self.targets.get(target_id)?;
        let buf = entry.lock().ok()?;
        if buf.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = buf.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * p).ceil() as usize;
        sorted.get(idx.min(sorted.len() - 1)).copied()
    }
}

pub struct Router {
    routes: Vec<RouteRuleConfig>,
    round_robin: DashMap<String, AtomicUsize>,
    weight_counters: DashMap<String, AtomicUsize>,
    latency_tracker: Option<Arc<LatencyTracker>>,
    pricing_catalog: Option<Arc<dyn PricingCatalog>>,
}

impl Router {
    pub fn new(routes: Vec<RouteRuleConfig>) -> Self {
        Self {
            routes,
            round_robin: DashMap::new(),
            weight_counters: DashMap::new(),
            latency_tracker: None,
            pricing_catalog: None,
        }
    }

    pub fn with_latency_tracker(mut self, tracker: Arc<LatencyTracker>) -> Self {
        self.latency_tracker = Some(tracker);
        self
    }

    pub fn with_pricing_catalog(mut self, catalog: Arc<dyn PricingCatalog>) -> Self {
        self.pricing_catalog = Some(catalog);
        self
    }

    pub fn latency_tracker(&self) -> Option<&Arc<LatencyTracker>> {
        self.latency_tracker.as_ref()
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    pub fn resolve(&self, ctx: &HookContext) -> Result<RouteResult, XlateError> {
        self.resolve_with_group(ctx, &ctx.request.group)
    }

    pub fn resolve_with_group(
        &self,
        ctx: &HookContext,
        group: &str,
    ) -> Result<RouteResult, XlateError> {
        let model = &ctx.request.model;
        for rule in &self.routes {
            if rule.group != group {
                continue;
            }
            if glob_match::glob_match(&rule.model, model) {
                let enabled: Vec<_> =
                    rule.targets.iter().filter(|t| t.enabled).cloned().collect();
                if enabled.is_empty() {
                    continue;
                }
                let mut result =
                    self.select_target(&rule.strategy, &rule.model, enabled, model)?;
                result.failover = rule.failover.clone();
                return Ok(result);
            }
        }
        Err(XlateError::UnsupportedProvider(format!(
            "no route for model '{model}' in group '{group}'"
        )))
    }

    pub fn record_failure(&self, ctx: &HookContext, _error: &XlateError) {
        if let Some(ref tracker) = self.latency_tracker {
            if let Some(ref route) = ctx.route {
                tracker.record(&route.target.id, f64::MAX);
            }
        }
    }

    pub fn next_failover_target(&self, ctx: &HookContext) -> Option<RouteTarget> {
        let route = ctx.route.as_ref()?;
        let idx = (ctx.metrics.attempt as usize).saturating_sub(1);
        route.alternatives.get(idx).cloned()
    }

    pub fn select_target(
        &self,
        strategy: &RouteStrategy,
        route_key: &str,
        mut targets: Vec<RouteTarget>,
        requested_model: &str,
    ) -> Result<RouteResult, XlateError> {
        match strategy {
            RouteStrategy::PriorityWeighted => {
                targets.sort_by(|a, b| a.priority.cmp(&b.priority));
                let best_priority = targets[0].priority;
                let tier: Vec<_> = targets
                    .iter()
                    .filter(|t| t.priority == best_priority)
                    .cloned()
                    .collect();
                let rest: Vec<_> = targets
                    .iter()
                    .filter(|t| t.priority != best_priority)
                    .cloned()
                    .collect();

                let selected = if tier.len() == 1 {
                    tier[0].clone()
                } else {
                    let total_weight: usize =
                        tier.iter().map(|t| t.weight as usize).sum();
                    if total_weight == 0 {
                        tier[0].clone()
                    } else {
                        let counter = self
                            .weight_counters
                            .entry(route_key.to_string())
                            .or_insert_with(|| AtomicUsize::new(0));
                        let val =
                            counter.fetch_add(1, Ordering::Relaxed) % total_weight;
                        let mut cumulative = 0;
                        let mut pick = &tier[0];
                        for t in &tier {
                            cumulative += t.weight as usize;
                            if val < cumulative {
                                pick = t;
                                break;
                            }
                        }
                        pick.clone()
                    }
                };

                let mut alternatives: Vec<_> = tier
                    .into_iter()
                    .filter(|t| t.id != selected.id)
                    .collect();
                alternatives.extend(rest);

                Ok(RouteResult {
                    target: selected,
                    alternatives,
                    failover: Default::default(),
                })
            }
            RouteStrategy::RoundRobin => {
                let counter = self
                    .round_robin
                    .entry(route_key.to_string())
                    .or_insert_with(|| AtomicUsize::new(0));
                let idx =
                    counter.fetch_add(1, Ordering::Relaxed) % targets.len();
                let mut ordered = Vec::with_capacity(targets.len());
                for i in 0..targets.len() {
                    ordered.push(targets[(idx + i) % targets.len()].clone());
                }
                let target = ordered.remove(0);
                Ok(RouteResult {
                    target,
                    alternatives: ordered,
                    failover: Default::default(),
                })
            }
            RouteStrategy::LeastLatency => {
                if let Some(ref tracker) = self.latency_tracker {
                    targets.sort_by(|a, b| {
                        let la = tracker.p50(&a.id).unwrap_or(f64::MAX);
                        let lb = tracker.p50(&b.id).unwrap_or(f64::MAX);
                        la.partial_cmp(&lb)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.priority.cmp(&b.priority))
                    });
                } else {
                    targets.sort_by(|a, b| a.priority.cmp(&b.priority));
                }
                let target = targets.remove(0);
                Ok(RouteResult {
                    target,
                    alternatives: targets,
                    failover: Default::default(),
                })
            }
            RouteStrategy::CostOptimized => {
                if let Some(ref catalog) = self.pricing_catalog {
                    targets.sort_by(|a, b| {
                        let model_a = a.config.upstream_model.as_deref().unwrap_or(requested_model);
                        let model_b = b.config.upstream_model.as_deref().unwrap_or(requested_model);
                        let cost_a = catalog
                            .get_pricing_for(model_a, Some(&a.config.plugin))
                            .map(|p| p.input_per_mtok + p.output_per_mtok);
                        let cost_b = catalog
                            .get_pricing_for(model_b, Some(&b.config.plugin))
                            .map(|p| p.input_per_mtok + p.output_per_mtok);
                        match (cost_a, cost_b) {
                            (Some(ca), Some(cb)) => ca.cmp(&cb).then(a.priority.cmp(&b.priority)),
                            (Some(_), None) => std::cmp::Ordering::Less,
                            (None, Some(_)) => std::cmp::Ordering::Greater,
                            (None, None) => a.priority.cmp(&b.priority),
                        }
                    });
                } else {
                    targets.sort_by(|a, b| a.priority.cmp(&b.priority));
                }
                let target = targets.remove(0);
                Ok(RouteResult {
                    target,
                    alternatives: targets,
                    failover: Default::default(),
                })
            }
        }
    }
}
