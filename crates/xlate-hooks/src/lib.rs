mod sanitize;
mod model_map;
mod latency;
mod usage;
mod circuit_breaker;
mod media_sanitizer;
mod rate_limit;
mod affinity;
mod patch;
mod param_heal;
mod think_tag;
mod rectifier;
mod cache_stats;
pub mod cost_calc;
mod billing;
mod trace;
mod alert;

use std::sync::Arc;
use xlate_core::config::KernelConfig;
use xlate_core::hook::Hook;
use xlate_core::kernel::EventBus;
use xlate_core::pricing::PricingCatalog;
use xlate_core::registry::ModelRegistry;
use xlate_core::router::LatencyTracker;
use xlate_core::store::Store;
use xlate_core::supervisor::Supervisor;

pub use sanitize::SanitizeHook;
pub use model_map::ModelMapHook;
pub use latency::LatencyHook;
pub use usage::UsageHook;
pub use circuit_breaker::{CircuitBreakerHook, CircuitOpenCount};
pub use media_sanitizer::MediaSanitizerHook;
pub use rate_limit::RateLimitHook;
pub use affinity::AffinityHook;
pub use patch::{PatchHook, PatchRule, PatchOp};
pub use param_heal::{ParamHealHook, RemovedParams};
pub use think_tag::ThinkTagHook;
pub use rectifier::RectifierHook;
pub use cache_stats::CacheStatsHook;
pub use cost_calc::{CostCalcHook, CostResult, ModelPricing};
pub use billing::BillingHook;
pub use trace::{TraceHook, TraceEntry};
pub use alert::AlertHook;

pub struct StandardHooksResult {
    pub hooks: Vec<Arc<dyn Hook>>,
    pub latency_tracker: Arc<LatencyTracker>,
    pub param_heal: Option<Arc<ParamHealHook>>,
}

pub fn standard_hooks(
    config: &KernelConfig,
    store: Option<Arc<dyn Store>>,
    registry: Option<Arc<ModelRegistry>>,
    event_bus: Option<Arc<EventBus>>,
    pricing_catalog: Option<Arc<dyn PricingCatalog>>,
    supervisor: Option<Arc<Supervisor>>,
) -> StandardHooksResult {
    let patch_rules: Vec<PatchRule> = config
        .routes
        .iter()
        .filter(|r| !r.patches.is_empty())
        .map(|r| PatchRule {
            model: r.model.clone(),
            patches: r
                .patches
                .iter()
                .map(|p| PatchOp {
                    op: p.op.clone(),
                    path: p.path.clone(),
                    value: p.value.clone(),
                    condition: p.condition.clone(),
                })
                .collect(),
        })
        .collect();

    let latency_tracker = Arc::new(LatencyTracker::new(1000));

    let mut hooks: Vec<Arc<dyn Hook>> = vec![
        Arc::new(SanitizeHook::from_config(&config.sanitize)),
        {
            let model_map = ModelMapHook::with_rules(
                config
                    .model_map
                    .rules
                    .iter()
                    .map(|r| (r.from.clone(), r.to.clone()))
                    .collect(),
                config.model_map.chain_redirect,
            );
            let model_map = if let Some(ref r) = registry {
                model_map.with_registry(r.clone())
            } else {
                model_map
            };
            Arc::new(model_map)
        },
        Arc::new(LatencyHook::with_tracker(latency_tracker.clone())),
        Arc::new(UsageHook::new()),
        {
            let cost_calc = CostCalcHook::new()
                .with_rate_multiplier(config.billing.rate_multiplier);
            let cost_calc = if let Some(ref catalog) = pricing_catalog {
                cost_calc.with_external_pricing(catalog.clone())
            } else {
                cost_calc
            };
            Arc::new(cost_calc)
        },
        Arc::new(ThinkTagHook::new()),
        Arc::new(CacheStatsHook::new()),
        {
            let rectifier = RectifierHook::from_config(&config.rectifier);
            let rectifier = if let Some(ref r) = registry {
                rectifier.with_registry(r.clone())
            } else {
                rectifier
            };
            Arc::new(rectifier)
        },
        Arc::new(PatchHook::with_rules(patch_rules)),
        {
            let cb = CircuitBreakerHook::new(
                config.circuit_breaker.failure_threshold,
                config.circuit_breaker.success_threshold,
                config.circuit_breaker.open_timeout_ms,
            )
            .with_granularity(&config.circuit_breaker.granularity);
            let cb = if let Some(ref eb) = event_bus {
                cb.with_event_bus(eb.clone())
            } else {
                cb
            };
            Arc::new(cb)
        },
        {
            let trace = TraceHook::new(config.trace.ring_buffer_size)
                .with_capture_config(
                    config.trace.capture_request_body,
                    config.trace.max_body_size,
                );
            let trace = if config.trace.persist_to_store {
                if let Some(ref s) = store {
                    trace.with_store(s.clone())
                } else {
                    trace
                }
            } else {
                trace
            };
            Arc::new(trace)
        },
    ];

    if config.rate_limit.enabled {
        hooks.push(Arc::new(
            RateLimitHook::with_config(
                config.rate_limit.rpm,
                config.rate_limit.tpm,
                event_bus.clone(),
            )
            .with_mode(&config.rate_limit.mode),
        ));
    }

    if config.session_affinity.enabled {
        let affinity = AffinityHook::with_ttl(std::time::Duration::from_secs(
            config.session_affinity.ttl_s,
        ))
        .with_lazy_binding(config.session_affinity.lazy_binding);
        let affinity = if let Some(ref sup) = supervisor {
            affinity.with_supervisor(sup.clone())
        } else {
            affinity
        };
        hooks.push(Arc::new(affinity));
    }

    let mut param_heal_ref = None;
    if config.param_heal.enabled {
        let mut param_heal = ParamHealHook::new()
            .with_max_attempts(config.param_heal.max_attempts);
        if config.param_heal.persist_cache {
            if let Some(ref s) = store {
                param_heal = param_heal.with_store(s.clone());
            }
        }
        let ph = Arc::new(param_heal);
        param_heal_ref = Some(ph.clone());
        hooks.push(ph);
    }

    if config.media_sanitizer.enabled {
        let media_hook = MediaSanitizerHook::new()
            .with_strategy(&config.media_sanitizer.strategy);
        let media_hook = if let Some(ref registry) = registry {
            media_hook.with_registry(registry.clone())
        } else {
            media_hook
        };
        hooks.push(Arc::new(media_hook));
    }

    let alert = if config.alerts.rules.is_empty() {
        AlertHook::new(500.0, 0.1, 100)
    } else {
        AlertHook::from_config(config.alerts.rules.clone())
    };
    let alert = if let Some(ref eb) = event_bus {
        alert.with_event_bus(eb.clone())
    } else {
        alert
    };
    let alert = if let Some(ref s) = store {
        alert.with_store(s.clone())
    } else {
        alert
    };
    hooks.push(Arc::new(alert));

    if config.billing.enabled {
        if let Some(store) = store {
            let billing = BillingHook::with_writer(
                store,
                config.billing.rate_multiplier,
                config.store.usage_batch_size,
                std::time::Duration::from_millis(config.store.usage_flush_interval_ms),
            );
            hooks.push(Arc::new(billing));
        }
    }

    StandardHooksResult {
        hooks,
        latency_tracker,
        param_heal: param_heal_ref,
    }
}
