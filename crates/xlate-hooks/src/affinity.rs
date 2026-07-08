use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::provider::ProviderConfig;
use xlate_core::supervisor::Supervisor;

struct AffinityBinding {
    config: ProviderConfig,
    bound_at: Instant,
}

pub struct AffinityHook {
    bindings: DashMap<String, AffinityBinding>,
    ttl: Duration,
    lazy_binding: bool,
    supervisor: Option<Arc<Supervisor>>,
}

impl AffinityHook {
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(300))
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            bindings: DashMap::new(),
            ttl,
            lazy_binding: true,
            supervisor: None,
        }
    }

    pub fn with_lazy_binding(mut self, lazy: bool) -> Self {
        self.lazy_binding = lazy;
        self
    }

    pub fn with_supervisor(mut self, supervisor: Arc<Supervisor>) -> Self {
        self.supervisor = Some(supervisor);
        self
    }

    fn affinity_key(ctx: &HookContext) -> Option<String> {
        let client_id = ctx.request.custom_headers.get("x-client-id")?;
        Some(format!("{}:{}", client_id, ctx.request.model))
    }

    fn is_target_healthy(&self, plugin: &str) -> bool {
        match &self.supervisor {
            Some(sup) => sup.is_available(plugin),
            None => true,
        }
    }
}

impl Default for AffinityHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for AffinityHook {
    fn name(&self) -> &str {
        "affinity"
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn pre_route(&self, ctx: &mut HookContext) -> HookVerdict {
        let key = match Self::affinity_key(ctx) {
            Some(k) => k,
            None => return HookVerdict::Continue,
        };

        if let Some(entry) = self.bindings.get(&key) {
            if entry.bound_at.elapsed() >= self.ttl {
                drop(entry);
                self.bindings.remove(&key);
            } else if !self.is_target_healthy(&entry.config.plugin) {
                tracing::debug!(key = %key, plugin = %entry.config.plugin, "affinity target unhealthy, unbinding");
                drop(entry);
                self.bindings.remove(&key);
            } else {
                tracing::debug!(key = %key, plugin = %entry.config.plugin, "affinity hit");
                ctx.provider_config = Some(entry.config.clone());
            }
        }

        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        if self.lazy_binding && !ctx.metrics.success {
            return HookVerdict::Continue;
        }

        let key = match Self::affinity_key(ctx) {
            Some(k) => k,
            None => return HookVerdict::Continue,
        };

        if let Some(config) = &ctx.provider_config {
            self.bindings.insert(
                key,
                AffinityBinding {
                    config: config.clone(),
                    bound_at: Instant::now(),
                },
            );
        }

        HookVerdict::Continue
    }
}
