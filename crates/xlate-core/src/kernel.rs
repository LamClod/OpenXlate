use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::capability::{Capability, CapabilityRights, CapabilityType, PluginCapabilities};
use crate::config::KernelConfig;
use crate::error::XlateError;
use crate::event::{ModelEvent, Usage};
use crate::hook::*;
use crate::inbound::{self, BuiltinInboundPlugin};
use crate::message::KernelEventPayload;
use crate::plugin::{EventSink, InboundPlugin, OutboundPlugin, PluginManifest, ServicePlugin};
use crate::registry::ModelRegistry;
use crate::supervisor::Supervisor;
use crate::types::NormalizedRequest;

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

pub struct EventBus {
    events: Mutex<VecDeque<KernelEventPayload>>,
    capacity: usize,
    notify: tokio::sync::Notify,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Mutex::new(VecDeque::with_capacity(capacity.min(256))),
            capacity,
            notify: tokio::sync::Notify::new(),
        }
    }

    pub fn emit(&self, event: KernelEventPayload) {
        let mut queue = self.events.lock().unwrap_or_else(|e| e.into_inner());
        if queue.len() >= self.capacity {
            queue.pop_front();
        }
        queue.push_back(event);
        drop(queue);
        self.notify.notify_one();
    }

    pub fn poll(&self) -> Option<KernelEventPayload> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
    }

    pub async fn poll_timeout(&self, timeout_ms: i32) -> Option<KernelEventPayload> {
        if let Some(event) = self.poll() {
            return Some(event);
        }
        match timeout_ms {
            0 => None,
            ms if ms < 0 => {
                loop {
                    let notified = self.notify.notified();
                    if let Some(event) = self.poll() {
                        return Some(event);
                    }
                    notified.await;
                }
            }
            ms => {
                let deadline = std::time::Duration::from_millis(ms as u64);
                match tokio::time::timeout(deadline, async {
                    loop {
                        let notified = self.notify.notified();
                        if let Some(event) = self.poll() {
                            return event;
                        }
                        notified.await;
                    }
                })
                .await
                {
                    Ok(event) => Some(event),
                    Err(_) => None,
                }
            }
        }
    }

    pub fn poll_batch(&self, max: usize) -> Vec<KernelEventPayload> {
        let mut queue = self.events.lock().unwrap_or_else(|e| e.into_inner());
        let n = max.min(queue.len());
        queue.drain(..n).collect()
    }
}

pub struct Kernel {
    config: KernelConfig,
    outbound: Vec<Arc<dyn OutboundPlugin>>,
    inbound: Arc<dyn InboundPlugin>,
    hooks: Vec<Arc<dyn Hook>>,
    services: Vec<Arc<dyn ServicePlugin>>,
    router: crate::router::Router,
    semaphore: Arc<Semaphore>,
    supervisor: Arc<Supervisor>,
    registry: Arc<ModelRegistry>,
    store: Option<Arc<dyn crate::store::Store>>,
    pricing_catalog: Option<Arc<dyn crate::pricing::PricingCatalog>>,
    event_bus: Arc<EventBus>,
    shutdown: AtomicBool,
    active_streams: AtomicU64,
    created_at: Instant,
    total_streams: AtomicU64,
    total_errors: AtomicU64,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
    total_duration_ms: AtomicU64,
    total_cache_read_tokens: AtomicU64,
    total_cache_write_tokens: AtomicU64,
    total_reasoning_tokens: AtomicU64,
    total_cost_microcents: AtomicU64,
    total_adjusted_cost_microcents: AtomicU64,
    total_ttft_ms: AtomicU64,
    ttft_count: AtomicU64,
    model_request_counts: dashmap::DashMap<String, AtomicU64>,
}

impl Kernel {
    pub fn new(
        config: KernelConfig,
        outbound: Vec<Arc<dyn OutboundPlugin>>,
        hooks: Vec<Arc<dyn Hook>>,
    ) -> Self {
        let max = config.kernel.max_concurrent_streams;
        let mut sorted_hooks = hooks;
        sorted_hooks.sort_by_key(|h| h.priority());
        Self {
            router: crate::router::Router::new(config.routes.clone()),
            semaphore: Arc::new(Semaphore::new(max)),
            config,
            outbound,
            inbound: Arc::new(BuiltinInboundPlugin::new()),
            hooks: sorted_hooks,
            services: Vec::new(),
            supervisor: Arc::new(Supervisor::new()),
            registry: Arc::new(ModelRegistry::new()),
            store: None,
            pricing_catalog: None,
            event_bus: Arc::new(EventBus::new(1024)),
            shutdown: AtomicBool::new(false),
            active_streams: AtomicU64::new(0),
            created_at: Instant::now(),
            total_streams: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
            total_cache_read_tokens: AtomicU64::new(0),
            total_cache_write_tokens: AtomicU64::new(0),
            total_reasoning_tokens: AtomicU64::new(0),
            total_duration_ms: AtomicU64::new(0),
            total_cost_microcents: AtomicU64::new(0),
            total_adjusted_cost_microcents: AtomicU64::new(0),
            total_ttft_ms: AtomicU64::new(0),
            ttft_count: AtomicU64::new(0),
            model_request_counts: dashmap::DashMap::new(),
        }
    }

    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    pub fn store(&self) -> Option<&dyn crate::store::Store> {
        self.store.as_deref()
    }

    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    pub fn create_capabilities(&self, manifest: &PluginManifest) -> PluginCapabilities {
        let mut caps = PluginCapabilities::empty();
        for cap_type in &manifest.required_capabilities {
            match cap_type {
                CapabilityType::Router => {
                    caps.router = Some(Capability::new(
                        Arc::new(crate::router::Router::new(self.config.routes.clone())),
                        CapabilityRights::read_only(),
                    ));
                }
                CapabilityType::ModelRegistry => {
                    caps.model_registry = Some(Capability::new(
                        self.registry.clone(),
                        CapabilityRights::read_only(),
                    ));
                }
                CapabilityType::EventBus => {
                    caps.event_bus = Some(Capability::new(
                        self.event_bus.clone(),
                        CapabilityRights::read_write(),
                    ));
                }
                CapabilityType::Store => {
                    if let Some(ref store) = self.store {
                        caps.store = Some(Capability::new(
                            store.clone(),
                            CapabilityRights::read_write(),
                        ));
                    }
                }
                CapabilityType::PricingCatalog => {
                    if let Some(ref catalog) = self.pricing_catalog {
                        caps.pricing = Some(Capability::new(
                            catalog.clone(),
                            CapabilityRights::read_only(),
                        ));
                    }
                }
                CapabilityType::Config => {
                    let config_arc: Arc<dyn std::any::Any + Send + Sync> =
                        Arc::new(self.config.clone());
                    caps.config = Some(Capability::new(
                        config_arc,
                        CapabilityRights::read_only(),
                    ));
                }
            }
        }
        caps
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.semaphore.close();

        if !self.services.is_empty() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let services = self.services.clone();
                handle.spawn(async move {
                    for svc in &services {
                        if let Err(e) = svc.stop().await {
                            tracing::warn!(service = svc.name(), error = %e, "service stop failed during shutdown");
                        }
                    }
                });
            }
        }
    }

    pub async fn shutdown_graceful(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.semaphore.close();

        let timeout = std::time::Duration::from_millis(
            self.config.kernel.shutdown_timeout_ms,
        );
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            if self.active_streams.load(Ordering::Relaxed) == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                tracing::warn!(
                    active = self.active_streams.load(Ordering::Relaxed),
                    "shutdown timeout reached, proceeding with active streams"
                );
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        for hook in &self.hooks {
            hook.shutdown().await;
        }

        for svc in &self.services {
            if let Err(e) = svc.stop().await {
                tracing::warn!(
                    service = svc.name(),
                    error = %e,
                    "service stop failed during graceful shutdown"
                );
            }
        }
    }

    pub async fn start_services(&self) {
        for svc in &self.services {
            let svc = svc.clone();
            let name = svc.name().to_string();
            let caps = self.create_capabilities(&svc.manifest());
            let supervisor = self.supervisor.clone();
            let event_bus = self.event_bus.clone();
            tokio::spawn(async move {
                match svc.start(caps).await {
                    Ok(()) => {
                        supervisor.mark_running(&name);
                        event_bus.emit(KernelEventPayload::PluginRecovered {
                            plugin: name,
                        });
                    }
                    Err(e) => {
                        let crash_count =
                            supervisor.report_crash(&name, &e.to_string());
                        event_bus.emit(KernelEventPayload::PluginCrashed {
                            plugin: name,
                            error: e.to_string(),
                            restart_count: crash_count,
                        });
                    }
                }
            });
        }

        if !self.services.is_empty() {
            let supervisor = self.supervisor.clone();
            let semaphore = self.semaphore.clone();
            let services = self.services.clone();
            let event_bus = self.event_bus.clone();
            let store = self.store.clone();
            let registry = self.registry.clone();
            let routes = self.config.routes.clone();
            let pricing_catalog = self.pricing_catalog.clone();
            let config_snapshot: Arc<dyn std::any::Any + Send + Sync> =
                Arc::new(self.config.clone());
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    if semaphore.is_closed() {
                        break;
                    }
                    supervisor.check_heartbeats();

                    let restart_names = supervisor.plugins_needing_restart();
                    for name in restart_names {
                        if let Some(svc) = services.iter().find(|s| s.name() == name) {
                            tracing::info!(plugin = %name, "auto-restarting crashed service");
                            supervisor.mark_restarting(&name);
                            let svc = svc.clone();
                            let sup = supervisor.clone();
                            let eb = event_bus.clone();
                            let caps = {
                                let manifest = svc.manifest();
                                let mut c = PluginCapabilities::empty();
                                for cap_type in &manifest.required_capabilities {
                                    match cap_type {
                                        CapabilityType::EventBus => {
                                            c.event_bus = Some(Capability::new(
                                                eb.clone(),
                                                CapabilityRights::read_write(),
                                            ));
                                        }
                                        CapabilityType::Store => {
                                            if let Some(ref s) = store {
                                                c.store = Some(Capability::new(
                                                    s.clone(),
                                                    CapabilityRights::read_write(),
                                                ));
                                            }
                                        }
                                        CapabilityType::ModelRegistry => {
                                            c.model_registry = Some(Capability::new(
                                                registry.clone(),
                                                CapabilityRights::read_only(),
                                            ));
                                        }
                                        CapabilityType::Router => {
                                            c.router = Some(Capability::new(
                                                Arc::new(crate::router::Router::new(routes.clone())),
                                                CapabilityRights::read_only(),
                                            ));
                                        }
                                        CapabilityType::PricingCatalog => {
                                            if let Some(ref catalog) = pricing_catalog {
                                                c.pricing = Some(Capability::new(
                                                    catalog.clone(),
                                                    CapabilityRights::read_only(),
                                                ));
                                            }
                                        }
                                        CapabilityType::Config => {
                                            c.config = Some(Capability::new(
                                                config_snapshot.clone(),
                                                CapabilityRights::read_only(),
                                            ));
                                        }
                                    }
                                }
                                c
                            };
                            tokio::spawn(async move {
                                match svc.start(caps).await {
                                    Ok(()) => {
                                        sup.mark_running(&name);
                                        eb.emit(KernelEventPayload::PluginRecovered {
                                            plugin: name,
                                        });
                                    }
                                    Err(e) => {
                                        let crash_count =
                                            sup.report_crash(&name, &e.to_string());
                                        eb.emit(KernelEventPayload::PluginCrashed {
                                            plugin: name,
                                            error: e.to_string(),
                                            restart_count: crash_count,
                                        });
                                    }
                                }
                            });
                        }
                    }
                }
            });
        }
    }

    pub async fn stop_services(&self) {
        for svc in &self.services {
            if let Err(e) = svc.stop().await {
                tracing::warn!(
                    service = svc.name(),
                    error = %e,
                    "service plugin stop failed"
                );
            }
        }
    }

    pub fn stats(&self) -> KernelStats {
        let max = self.config.kernel.max_concurrent_streams;
        let available = self.semaphore.available_permits();
        let total = self.total_streams.load(Ordering::Relaxed);
        let errors = self.total_errors.load(Ordering::Relaxed);
        let error_rate = if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        };
        let ttft_c = self.ttft_count.load(Ordering::Relaxed);
        let avg_ttft = if ttft_c > 0 {
            self.total_ttft_ms.load(Ordering::Relaxed) as f64 / ttft_c as f64
        } else {
            0.0
        };
        let avg_duration = if total > 0 {
            self.total_duration_ms.load(Ordering::Relaxed) as f64 / total as f64
        } else {
            0.0
        };
        let input_tok = self.total_input_tokens.load(Ordering::Relaxed);
        let cache_read_tok = self.total_cache_read_tokens.load(Ordering::Relaxed);
        let cache_hit_rate = if input_tok + cache_read_tok > 0 {
            cache_read_tok as f64 / (input_tok + cache_read_tok) as f64
        } else {
            0.0
        };

        let cost_mc = self.total_cost_microcents.load(Ordering::Relaxed);
        let adj_mc = self.total_adjusted_cost_microcents.load(Ordering::Relaxed);
        let cost_dec = rust_decimal::Decimal::from(cost_mc) / rust_decimal::Decimal::from(1_000_000);
        let adj_dec = rust_decimal::Decimal::from(adj_mc) / rust_decimal::Decimal::from(1_000_000);

        let mut top_models: Vec<_> = self
            .model_request_counts
            .iter()
            .map(|e| ModelUsageSnapshot {
                model: e.key().clone(),
                request_count: e.value().load(Ordering::Relaxed),
            })
            .collect();
        top_models.sort_by(|a, b| b.request_count.cmp(&a.request_count));
        top_models.truncate(10);

        let mut result = KernelStats {
            version: "0.1.0".into(),
            uptime_s: self.created_at.elapsed().as_secs_f64(),
            max_concurrent_streams: max,
            active_streams: self.active_streams.load(Ordering::Relaxed) as u32,
            available_permits: available,
            total_streams: total,
            total_tokens: TotalTokens {
                input: input_tok as i64,
                output: self.total_output_tokens.load(Ordering::Relaxed) as i64,
                cache_read: cache_read_tok as i64,
                cache_write: self.total_cache_write_tokens.load(Ordering::Relaxed) as i64,
                reasoning: self.total_reasoning_tokens.load(Ordering::Relaxed) as i64,
            },
            total_cost: TotalCost {
                total_usd: cost_dec.to_string(),
                adjusted_usd: adj_dec.to_string(),
            },
            total_errors: errors,
            error_rate,
            cache_hit_rate,
            avg_ttft_ms: avg_ttft,
            avg_duration_ms: avg_duration,
            hook_count: self.hooks.len(),
            hooks: self.hooks.iter().map(|h| h.name().to_string()).collect(),
            outbound_plugins: self.outbound.iter().map(|p| p.name().to_string()).collect(),
            route_count: self.router.route_count(),
            model_count: self.registry.list_models().len(),
            circuit_breakers: Vec::new(),
            rate_limit: None,
            plugins: self
                .supervisor
                .stats()
                .iter()
                .map(|(name, status, crashes, kind)| PluginSnapshot {
                    name: name.clone(),
                    kind: serde_json::to_value(kind)
                        .ok()
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default(),
                    state: status.clone(),
                    crash_count: *crashes,
                })
                .collect(),
            top_models,
            supervisor_stats: self.supervisor.stats(),
            hook_snapshots: serde_json::Map::new(),
            shutdown: self.is_shutdown(),
        };

        for hook in &self.hooks {
            if let Some(snap) = hook.snapshot() {
                match hook.name() {
                    "circuit-breaker" => {
                        if let Some(arr) = snap.get("circuit_breakers").and_then(|v| v.as_array()) {
                            for item in arr {
                                if let Ok(cb) =
                                    serde_json::from_value::<CircuitBreakerSnapshot>(item.clone())
                                {
                                    result.circuit_breakers.push(cb);
                                }
                            }
                        }
                    }
                    "rate-limit" => {
                        if let Ok(rl) = serde_json::from_value::<RateLimitSnapshot>(snap.clone()) {
                            result.rate_limit = Some(rl);
                        }
                    }
                    _ => {}
                }
                result
                    .hook_snapshots
                    .insert(hook.name().to_string(), snap);
            }
        }

        result
    }

    pub async fn stream(
        &self,
        body: &str,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let body: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| XlateError::InvalidRequest(e.to_string()))?;
        self.stream_raw(&body, sink).await
    }

    pub async fn stream_normalized(
        &self,
        request: NormalizedRequest,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        if self.is_shutdown() {
            return Err(XlateError::Internal("kernel is shutting down".into()));
        }

        let _permit = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.kernel.backpressure_timeout_ms),
            self.semaphore.acquire(),
        )
        .await
        .map_err(|_| XlateError::Overloaded)?
        .map_err(|_| {
            if self.is_shutdown() {
                XlateError::Internal("kernel is shutting down".into())
            } else {
                XlateError::Internal("semaphore closed".into())
            }
        })?;

        self.total_streams.fetch_add(1, Ordering::Relaxed);
        self.active_streams.fetch_add(1, Ordering::Relaxed);

        let mut request = request;
        if request.stream_idle_timeout_ms.is_none() {
            request.stream_idle_timeout_ms = Some(self.config.kernel.default_idle_timeout_ms);
        }

        let mut ctx = HookContext::new(request);
        ctx.metrics.max_attempts = self.config.kernel.max_failover_attempts;

        let result = self.stream_inner(&mut ctx, sink).await;

        ctx.metrics.success = result.is_ok();
        let total_ms = ctx.created_at.elapsed().as_secs_f64() * 1000.0;
        ctx.metrics.total_ms = Some(total_ms);
        ctx.metrics.latency = Some(LatencyMetrics {
            total_ms,
            ttft_ms: ctx.metrics.ttft_ms,
            provider_ms: ctx.metrics.provider_ms,
        });
        if let Err(ref e) = result {
            ctx.metrics.error = Some(e.clone());
        }
        let _ = self
            .run_hooks_guarded(HookPhase::PostComplete, &mut ctx)
            .await;

        if result.is_err() {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(input) = ctx.metrics.usage.input_tokens {
            self.total_input_tokens
                .fetch_add(input.max(0) as u64, Ordering::Relaxed);
        }
        if let Some(output) = ctx.metrics.usage.output_tokens {
            self.total_output_tokens
                .fetch_add(output.max(0) as u64, Ordering::Relaxed);
        }
        if let Some(cr) = ctx.metrics.usage.cache_read_tokens {
            self.total_cache_read_tokens
                .fetch_add(cr.max(0) as u64, Ordering::Relaxed);
        }
        if let Some(cw) = ctx.metrics.usage.cache_write_tokens {
            self.total_cache_write_tokens
                .fetch_add(cw.max(0) as u64, Ordering::Relaxed);
        }
        if let Some(r) = ctx.metrics.usage.reasoning_tokens {
            self.total_reasoning_tokens
                .fetch_add(r.max(0) as u64, Ordering::Relaxed);
        }
        if let Some(ms) = ctx.metrics.total_ms {
            self.total_duration_ms
                .fetch_add(ms as u64, Ordering::Relaxed);
        }
        if let Some(ttft) = ctx.metrics.ttft_ms {
            self.total_ttft_ms
                .fetch_add(ttft as u64, Ordering::Relaxed);
            self.ttft_count.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(ref cost) = ctx.metrics.cost {
            use rust_decimal::prelude::ToPrimitive;
            let mc = (cost.total_cost * rust_decimal::Decimal::from(1_000_000))
                .to_u64()
                .unwrap_or(0);
            let adj = (cost.adjusted_cost * rust_decimal::Decimal::from(1_000_000))
                .to_u64()
                .unwrap_or(0);
            self.total_cost_microcents.fetch_add(mc, Ordering::Relaxed);
            self.total_adjusted_cost_microcents
                .fetch_add(adj, Ordering::Relaxed);
        }
        self.model_request_counts
            .entry(ctx.request.model.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        self.active_streams.fetch_sub(1, Ordering::Relaxed);
        result
    }

    async fn stream_inner(
        &self,
        ctx: &mut HookContext,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        self.run_hooks_guarded(HookPhase::PreRoute, ctx).await?;

        if ctx.provider_config.is_none() {
            let route = self.router.resolve(ctx)?;
            ctx.provider_config = Some(route.target.config.clone());
            ctx.route = Some(route);
        } else if ctx.route.is_none() {
            if let Ok(route) = self.router.resolve(ctx) {
                ctx.route = Some(route);
            }
        }

        self.execute_loop(ctx, sink).await
    }

    pub async fn stream_raw(
        &self,
        body: &serde_json::Value,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let metadata = inbound::extract_metadata(body);
        let format = inbound::resolve_format(&metadata, body);

        let clean_body = if body.get("_metadata").is_some() {
            let mut stripped = body.clone();
            if let Some(obj) = stripped.as_object_mut() {
                obj.remove("_metadata");
            }
            stripped
        } else {
            body.clone()
        };

        let mut request = self.inbound.decode(format, &clean_body)?;

        if let Some(ref client_id) = metadata.client_id {
            request
                .custom_headers
                .insert("x-client-id".into(), client_id.clone());
        }
        if let Some(ref trace_id) = metadata.trace_id {
            request
                .custom_headers
                .insert("x-trace-id".into(), trace_id.clone());
        }
        if let Some(ref group) = metadata.group {
            request.group = group.clone();
        }
        for (key, value) in &metadata.extra {
            if let Some(s) = value.as_str() {
                request
                    .custom_headers
                    .insert(format!("x-metadata-{key}"), s.to_string());
            }
        }
        request.custom_headers.insert(
            "x-source-format".into(),
            serde_json::to_value(format)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| format!("{:?}", format)),
        );

        self.stream_normalized(request, sink).await
    }

    async fn execute_loop(
        &self,
        ctx: &mut HookContext,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        loop {
            self.run_hooks_guarded(HookPhase::PreSend, ctx).await?;

            if let Some(PendingRectify(action)) = ctx.extensions.remove::<PendingRectify>() {
                match action {
                    RectifyAction::SwitchTarget => {
                        ctx.metrics.attempt += 1;
                        if ctx.metrics.attempt >= ctx.metrics.max_attempts {
                            return Err(XlateError::UnsupportedProvider(
                                "all targets exhausted by pre-send hooks".into(),
                            ));
                        }
                        if let Some(next) = self.router.next_failover_target(ctx) {
                            self.apply_failover_cooldown(ctx).await;
                            ctx.provider_config = Some(next.config.clone());
                            if let Some(ref mut route) = ctx.route {
                                route.target = next;
                            }
                            continue;
                        }
                        return Err(XlateError::UnsupportedProvider(
                            "no alternative target after pre-send switch".into(),
                        ));
                    }
                    RectifyAction::RetryAfter(delay) => {
                        ctx.metrics.attempt += 1;
                        if ctx.metrics.attempt >= ctx.metrics.max_attempts {
                            return Err(XlateError::Internal("retry limit exhausted".into()));
                        }
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    RectifyAction::ModifyRequest => {}
                }
            }

            let config = ctx.provider_config.as_ref().ok_or_else(|| {
                XlateError::Internal("no provider config after routing".into())
            })?;

            let outbound = self.find_outbound(&config.plugin)?;

            let mut hooked = HookedEventSink {
                inner: sink,
                hooks: &self.hooks,
                state: std::cell::UnsafeCell::new(EventSinkState {
                    usage: Usage::default(),
                    finish_reason: None,
                    ttft_ms: None,
                    created_at: ctx.created_at,
                    output_char_count: 0,
                }),
                ctx_snapshot: &*ctx,
                supervisor: &self.supervisor,
                event_bus: &self.event_bus,
            };

            let provider_start = Instant::now();
            let result = match std::panic::AssertUnwindSafe(
                outbound.stream(&ctx.request, config, &mut hooked),
            )
            .catch_unwind()
            .await
            {
                Ok(r) => r,
                Err(panic_info) => {
                    let reason = panic_info
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic_info.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    self.supervisor
                        .report_crash(&config.plugin, reason);
                    Err(XlateError::Internal(format!(
                        "outbound plugin '{}' panicked: {reason}",
                        config.plugin
                    )))
                }
            };
            let provider_elapsed_ms = provider_start.elapsed().as_secs_f64() * 1000.0;

            let collected = hooked.take_state();
            drop(hooked);
            ctx.metrics.provider_ms = Some(provider_elapsed_ms);
            if collected.output_char_count > 0 {
                ctx.extensions
                    .insert(OutputCharCount(collected.output_char_count));
            }
            if collected.usage.input_tokens.is_some() || collected.usage.output_tokens.is_some() {
                ctx.metrics.usage = collected.usage;
            }
            if let Some(fr) = collected.finish_reason {
                ctx.metrics.finish_reason = Some(fr);
            }
            if let Some(ttft) = collected.ttft_ms {
                if ctx.metrics.ttft_ms.is_none() {
                    ctx.metrics.ttft_ms = Some(ttft);
                }
            }

            match result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    ctx.metrics.attempt += 1;
                    self.router.record_failure(ctx, &e);

                    let verdict = self.run_error_hooks_guarded(ctx, &e).await;

                    match verdict {
                        HookVerdict::Rectify(RectifyAction::ModifyRequest) => continue,
                        HookVerdict::Rectify(RectifyAction::SwitchTarget) => {
                            if ctx.metrics.attempt >= ctx.metrics.max_attempts {
                                return Err(e);
                            }
                            if !self.should_failover(ctx, &e) {
                                return Err(e);
                            }
                            if let Some(next) = self.router.next_failover_target(ctx) {
                                self.apply_failover_cooldown(ctx).await;
                                ctx.provider_config = Some(next.config.clone());
                                if let Some(ref mut route) = ctx.route {
                                    route.target = next;
                                }
                                continue;
                            }
                            return Err(e);
                        }
                        HookVerdict::Rectify(RectifyAction::RetryAfter(delay)) => {
                            if ctx.metrics.attempt >= ctx.metrics.max_attempts {
                                return Err(e);
                            }
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                        HookVerdict::Abort(abort_err) => return Err(abort_err),
                        _ => {
                            if ctx.metrics.attempt >= ctx.metrics.max_attempts {
                                return Err(e);
                            }
                            if !self.should_failover(ctx, &e) {
                                return Err(e);
                            }
                            if let Some(next) = self.router.next_failover_target(ctx) {
                                self.apply_failover_cooldown(ctx).await;
                                ctx.provider_config = Some(next.config.clone());
                                if let Some(ref mut route) = ctx.route {
                                    route.target = next;
                                }
                                continue;
                            }
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    fn should_failover(&self, ctx: &HookContext, error: &XlateError) -> bool {
        let trigger = match ctx.route {
            Some(ref r) if !r.failover.trigger_statuses.is_empty() => {
                &r.failover.trigger_statuses
            }
            _ => return true,
        };
        let status = match error {
            XlateError::Provider { status: Some(s), .. } => *s,
            XlateError::RateLimited(_) => 429,
            XlateError::Overloaded => 529,
            _ => return true,
        };
        trigger.contains(&status)
    }

    async fn apply_failover_cooldown(&self, ctx: &HookContext) {
        let cooldown = match ctx.route {
            Some(ref r) if r.failover.cooldown_ms > 0 => r.failover.cooldown_ms,
            _ => return,
        };
        tokio::time::sleep(std::time::Duration::from_millis(cooldown)).await;
    }


    fn find_outbound(&self, plugin_name: &str) -> Result<&dyn OutboundPlugin, XlateError> {
        if !self.supervisor.is_available(plugin_name) {
            return Err(XlateError::UnsupportedProvider(format!(
                "outbound plugin '{plugin_name}' is crashed/unavailable"
            )));
        }
        self.outbound
            .iter()
            .find(|p| p.name() == plugin_name)
            .map(|p| p.as_ref())
            .ok_or_else(|| {
                XlateError::UnsupportedProvider(format!(
                    "no outbound plugin '{plugin_name}'"
                ))
            })
    }

    async fn run_hooks_guarded(
        &self,
        phase: HookPhase,
        ctx: &mut HookContext,
    ) -> Result<(), XlateError> {
        let len = self.hooks.len();

        for i in 0..len {
            let idx = i;
            let hook = &self.hooks[idx];

            if !self.supervisor.is_available(hook.name()) {
                continue;
            }

            let verdict = {
                let fut = match phase {
                    HookPhase::PreRoute => hook.pre_route(ctx),
                    HookPhase::PreSend => hook.pre_send(ctx),
                    HookPhase::PostComplete => hook.post_complete(ctx),
                    _ => return Ok(()),
                };

                match std::panic::AssertUnwindSafe(fut).catch_unwind().await {
                    Ok(v) => {
                        self.supervisor.mark_running(hook.name());
                        if ctx.extensions.get::<HooksFired>().is_none() {
                            ctx.extensions.insert(HooksFired(Vec::new()));
                        }
                        ctx.extensions
                            .get_mut::<HooksFired>()
                            .unwrap()
                            .0
                            .push(hook.name().to_string());
                        v
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        let crash_count = self.supervisor.report_crash(hook.name(), reason);
                        self.event_bus.emit(KernelEventPayload::PluginCrashed {
                            plugin: hook.name().to_string(),
                            error: reason.to_string(),
                            restart_count: crash_count,
                        });
                        tracing::error!(
                            hook = hook.name(),
                            phase = ?phase,
                            reason,
                            "hook panicked, isolated by supervisor"
                        );
                        HookVerdict::Continue
                    }
                }
            };

            match verdict {
                HookVerdict::Continue => continue,
                HookVerdict::Skip if phase != HookPhase::PostComplete => break,
                HookVerdict::Abort(e) if phase != HookPhase::PostComplete => return Err(e),
                HookVerdict::Abort(e) => {
                    tracing::warn!(
                        hook = hook.name(),
                        error = %e,
                        "PostComplete hook returned Abort, ignoring to ensure all hooks run"
                    );
                }
                HookVerdict::Rectify(action)
                    if phase == HookPhase::PreSend || phase == HookPhase::PreRoute =>
                {
                    ctx.extensions.insert(PendingRectify(action));
                    break;
                }
                _ => continue,
            }
        }
        Ok(())
    }

    async fn run_error_hooks_guarded(
        &self,
        ctx: &mut HookContext,
        error: &XlateError,
    ) -> HookVerdict {
        for hook in self.hooks.iter().rev() {
            if !self.supervisor.is_available(hook.name()) {
                continue;
            }

            let verdict =
                match std::panic::AssertUnwindSafe(hook.on_error(ctx, error))
                    .catch_unwind()
                    .await
                {
                    Ok(v) => {
                        self.supervisor.mark_running(hook.name());
                        v
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        let crash_count =
                            self.supervisor.report_crash(hook.name(), reason);
                        self.event_bus.emit(KernelEventPayload::PluginCrashed {
                            plugin: hook.name().to_string(),
                            error: reason.to_string(),
                            restart_count: crash_count,
                        });
                        HookVerdict::Continue
                    }
                };

            match verdict {
                HookVerdict::Continue => continue,
                other => return other,
            }
        }
        HookVerdict::Continue
    }
}

// ---------------------------------------------------------------------------
// KernelBuilder
// ---------------------------------------------------------------------------

pub struct KernelBuilder {
    config: KernelConfig,
    outbound: Vec<Arc<dyn OutboundPlugin>>,
    inbound: Option<Arc<dyn InboundPlugin>>,
    hooks: Vec<Arc<dyn Hook>>,
    services: Vec<Arc<dyn ServicePlugin>>,
    supervisor: Option<Arc<Supervisor>>,
    registry: Option<Arc<ModelRegistry>>,
    store: Option<Arc<dyn crate::store::Store>>,
    event_bus: Option<Arc<EventBus>>,
    latency_tracker: Option<Arc<crate::router::LatencyTracker>>,
    pricing_catalog: Option<Arc<dyn crate::pricing::PricingCatalog>>,
}

impl KernelBuilder {
    pub fn new(config: KernelConfig) -> Self {
        Self {
            config,
            outbound: Vec::new(),
            inbound: None,
            hooks: Vec::new(),
            services: Vec::new(),
            supervisor: None,
            registry: None,
            store: None,
            event_bus: None,
            latency_tracker: None,
            pricing_catalog: None,
        }
    }

    pub fn outbound(mut self, plugin: Arc<dyn OutboundPlugin>) -> Self {
        self.outbound.push(plugin);
        self
    }

    pub fn outbound_vec(mut self, plugins: Vec<Arc<dyn OutboundPlugin>>) -> Self {
        self.outbound = plugins;
        self
    }

    pub fn inbound(mut self, plugin: Arc<dyn InboundPlugin>) -> Self {
        self.inbound = Some(plugin);
        self
    }

    pub fn hook(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    pub fn hooks(mut self, hooks: Vec<Arc<dyn Hook>>) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn service(mut self, svc: Arc<dyn ServicePlugin>) -> Self {
        self.services.push(svc);
        self
    }

    pub fn services(mut self, svcs: Vec<Arc<dyn ServicePlugin>>) -> Self {
        self.services = svcs;
        self
    }

    pub fn supervisor(mut self, supervisor: Arc<Supervisor>) -> Self {
        self.supervisor = Some(supervisor);
        self
    }

    pub fn registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn store(mut self, store: Arc<dyn crate::store::Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn event_bus(mut self, eb: Arc<EventBus>) -> Self {
        self.event_bus = Some(eb);
        self
    }

    pub fn latency_tracker(mut self, tracker: Arc<crate::router::LatencyTracker>) -> Self {
        self.latency_tracker = Some(tracker);
        self
    }

    pub fn pricing_catalog(mut self, catalog: Arc<dyn crate::pricing::PricingCatalog>) -> Self {
        self.pricing_catalog = Some(catalog);
        self
    }

    pub fn build(self) -> Kernel {
        let max = self.config.kernel.max_concurrent_streams;
        let mut sorted_hooks = self.hooks;
        sorted_hooks.sort_by_key(|h| h.priority());

        let config = expand_config_env_vars(self.config);
        let supervisor = self
            .supervisor
            .unwrap_or_else(|| Arc::new(Supervisor::new()));

        for svc in &self.services {
            supervisor.register_plugin(svc.name(), crate::plugin::PluginKind::Service);
        }
        for hook in &sorted_hooks {
            supervisor.register_plugin(hook.name(), crate::plugin::PluginKind::Hook);
            supervisor.mark_running(hook.name());
        }
        for outbound in &self.outbound {
            supervisor.register_plugin(outbound.name(), crate::plugin::PluginKind::Outbound);
            supervisor.mark_running(outbound.name());
        }

        let mut router = crate::router::Router::new(config.routes.clone());
        if let Some(tracker) = self.latency_tracker {
            router = router.with_latency_tracker(tracker);
        }
        let pricing_catalog = self.pricing_catalog;
        if let Some(ref catalog) = pricing_catalog {
            router = router.with_pricing_catalog(catalog.clone());
        }

        Kernel {
            router,
            semaphore: Arc::new(Semaphore::new(max)),
            config,
            outbound: self.outbound,
            inbound: self
                .inbound
                .unwrap_or_else(|| Arc::new(BuiltinInboundPlugin::new())),
            hooks: sorted_hooks,
            services: self.services,
            supervisor,
            registry: self
                .registry
                .unwrap_or_else(|| Arc::new(ModelRegistry::new())),
            store: self.store,
            pricing_catalog,
            event_bus: self.event_bus.unwrap_or_else(|| Arc::new(EventBus::new(1024))),
            shutdown: AtomicBool::new(false),
            active_streams: AtomicU64::new(0),
            created_at: Instant::now(),
            total_streams: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
            total_cache_read_tokens: AtomicU64::new(0),
            total_cache_write_tokens: AtomicU64::new(0),
            total_reasoning_tokens: AtomicU64::new(0),
            total_cost_microcents: AtomicU64::new(0),
            total_adjusted_cost_microcents: AtomicU64::new(0),
            total_duration_ms: AtomicU64::new(0),
            total_ttft_ms: AtomicU64::new(0),
            ttft_count: AtomicU64::new(0),
            model_request_counts: dashmap::DashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// KernelStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStats {
    pub version: String,
    pub uptime_s: f64,
    pub max_concurrent_streams: usize,
    pub active_streams: u32,
    pub available_permits: usize,
    pub total_streams: u64,
    pub total_tokens: TotalTokens,
    pub total_cost: TotalCost,
    pub total_errors: u64,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
    pub avg_ttft_ms: f64,
    pub avg_duration_ms: f64,
    pub hook_count: usize,
    pub hooks: Vec<String>,
    pub outbound_plugins: Vec<String>,
    pub route_count: usize,
    pub model_count: usize,
    pub circuit_breakers: Vec<CircuitBreakerSnapshot>,
    pub rate_limit: Option<RateLimitSnapshot>,
    pub plugins: Vec<PluginSnapshot>,
    pub top_models: Vec<ModelUsageSnapshot>,
    pub supervisor_stats: Vec<(String, String, u32, crate::plugin::PluginKind)>,
    pub hook_snapshots: serde_json::Map<String, serde_json::Value>,
    pub shutdown: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TotalTokens {
    pub input: i64,
    pub output: i64,
    pub cache_read: i64,
    pub cache_write: i64,
    pub reasoning: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TotalCost {
    pub total_usd: String,
    pub adjusted_usd: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsageSnapshot {
    pub model: String,
    pub request_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitBreakerSnapshot {
    pub key: String,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_since_ms: Option<u64>,
    pub failure_count: u32,
    pub success_count: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitSnapshot {
    #[serde(default)]
    pub current_rpm: u64,
    #[serde(default)]
    pub current_tpm: u64,
    #[serde(default)]
    pub limit_rpm: u64,
    #[serde(default)]
    pub limit_tpm: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSnapshot {
    pub name: String,
    #[serde(default)]
    pub kind: String,
    pub state: String,
    pub crash_count: u32,
}

// ---------------------------------------------------------------------------
// Env var expansion: ${VAR} and ${VAR:-default}
// ---------------------------------------------------------------------------

pub fn expand_env(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next();
            let mut var_expr = String::new();
            let mut depth = 1;
            for ch in chars.by_ref() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                var_expr.push(ch);
            }

            if let Some(sep) = var_expr.find(":-") {
                let var_name = &var_expr[..sep];
                let default_val = &var_expr[sep + 2..];
                result.push_str(
                    &std::env::var(var_name).unwrap_or_else(|_| default_val.to_string()),
                );
            } else {
                match std::env::var(&var_expr) {
                    Ok(val) => result.push_str(&val),
                    Err(_) => {
                        tracing::warn!(var = %var_expr, "environment variable not set, using empty string");
                    }
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn expand_config_env_vars(mut config: KernelConfig) -> KernelConfig {
    for route in &mut config.routes {
        for target in &mut route.targets {
            target.config.api_key = expand_env(&target.config.api_key);
            target.config.base_url = expand_env(&target.config.base_url);
        }
    }
    config
}

// ---------------------------------------------------------------------------
// HookedEventSink
// ---------------------------------------------------------------------------

struct EventSinkState {
    usage: Usage,
    finish_reason: Option<String>,
    ttft_ms: Option<f64>,
    created_at: Instant,
    output_char_count: usize,
}

struct HookedEventSink<'a> {
    inner: &'a mut dyn EventSink,
    hooks: &'a [Arc<dyn Hook>],
    state: std::cell::UnsafeCell<EventSinkState>,
    ctx_snapshot: &'a HookContext,
    supervisor: &'a Supervisor,
    event_bus: &'a EventBus,
}

unsafe impl Send for HookedEventSink<'_> {}

impl HookedEventSink<'_> {
    fn take_state(&self) -> EventSinkState {
        let state = unsafe { &mut *self.state.get() };
        EventSinkState {
            usage: std::mem::take(&mut state.usage),
            finish_reason: state.finish_reason.take(),
            ttft_ms: state.ttft_ms.take(),
            created_at: state.created_at,
            output_char_count: state.output_char_count,
        }
    }
}

#[async_trait::async_trait]
impl EventSink for HookedEventSink<'_> {
    async fn send(&mut self, mut event: ModelEvent) -> Result<(), XlateError> {
        {
            let state = unsafe { &mut *self.state.get() };
            if let ModelEvent::TurnFinished {
                ref usage,
                ref finish_reason,
                ..
            } = event
            {
                state.usage = usage.clone();
                state.finish_reason = Some(finish_reason.clone());
            }
            if let ModelEvent::TextDelta { ref text, .. } = event {
                if state.ttft_ms.is_none() {
                    state.ttft_ms =
                        Some(state.created_at.elapsed().as_secs_f64() * 1000.0);
                }
                state.output_char_count += text.chars().count();
            } else if matches!(event, ModelEvent::ThinkingDelta { .. }) {
                if state.ttft_ms.is_none() {
                    state.ttft_ms =
                        Some(state.created_at.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }

        for hook in self.hooks.iter() {
            if !self.supervisor.is_available(hook.name()) {
                continue;
            }
            let verdict =
                match std::panic::AssertUnwindSafe(hook.on_event(self.ctx_snapshot, &mut event))
                    .catch_unwind()
                    .await
                {
                    Ok(v) => {
                        self.supervisor.mark_running(hook.name());
                        v
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        let crash_count = self.supervisor.report_crash(hook.name(), reason);
                        self.event_bus.emit(KernelEventPayload::PluginCrashed {
                            plugin: hook.name().to_string(),
                            error: reason.to_string(),
                            restart_count: crash_count,
                        });
                        tracing::error!(
                            hook = hook.name(),
                            reason,
                            "on_event hook panicked, isolated by supervisor"
                        );
                        HookVerdict::Continue
                    }
                };
            match verdict {
                HookVerdict::Continue => {}
                HookVerdict::Skip => return Ok(()),
                HookVerdict::Abort(e) => return Err(e),
                _ => {}
            }
        }
        self.inner.send(event).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::*;

    fn make_target(id: &str, priority: u32, weight: u32) -> RouteTarget {
        RouteTarget {
            id: id.into(),
            plugin: "openai".into(),
            config: ProviderConfig {
                plugin: "openai".into(),
                base_url: format!("https://{id}"),
                api_key: "k".into(),
                upstream_model: None,
                endpoint: None,
                extra_params: None,
                extra_headers: Default::default(),
                anthropic_version: None,
                max_tokens_override: None,
            },
            priority,
            weight,
            enabled: true,
        }
    }

    #[test]
    fn env_expansion_basic() {
        std::env::set_var("XLATE_TEST_KEY", "secret123");
        assert_eq!(expand_env("${XLATE_TEST_KEY}"), "secret123");
        assert_eq!(
            expand_env("prefix_${XLATE_TEST_KEY}_suffix"),
            "prefix_secret123_suffix"
        );
        std::env::remove_var("XLATE_TEST_KEY");
    }

    #[test]
    fn env_expansion_default() {
        std::env::remove_var("XLATE_NOEXIST_KEY");
        assert_eq!(expand_env("${XLATE_NOEXIST_KEY:-fallback}"), "fallback");
        assert_eq!(expand_env("${XLATE_NOEXIST_KEY}"), "");
    }

    #[test]
    fn env_expansion_no_vars() {
        assert_eq!(expand_env("no variables here"), "no variables here");
        assert_eq!(expand_env(""), "");
    }

    #[test]
    fn kernel_builder_defaults() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();
        assert_eq!(kernel.hooks.len(), 0);
        assert_eq!(kernel.outbound.len(), 0);
        assert!(!kernel.is_shutdown());
    }

    #[test]
    fn kernel_shutdown() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();
        assert!(!kernel.is_shutdown());
        kernel.shutdown();
        assert!(kernel.is_shutdown());
    }

    #[test]
    fn kernel_stats_comprehensive() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();
        let stats = kernel.stats();
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("0.1.0"));
        assert!(json.contains("uptime_s"));
        assert!(json.contains("active_streams"));
        assert!(json.contains("total_streams"));
        assert!(json.contains("available_permits"));
        assert!(json.contains("hooks"));
    }

    #[test]
    fn round_robin_strategy() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();

        let targets = vec![make_target("a", 1, 100), make_target("b", 1, 100)];

        let r1 = kernel.router
            .select_target(&RouteStrategy::RoundRobin, "test", targets.clone(), "gpt-4o")
            .unwrap();
        let r2 = kernel.router
            .select_target(&RouteStrategy::RoundRobin, "test", targets.clone(), "gpt-4o")
            .unwrap();
        assert_ne!(r1.target.id, r2.target.id);
    }

    #[test]
    fn priority_weighted_selects_lowest_priority() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();

        let targets = vec![
            make_target("low", 1, 100),
            make_target("high", 10, 100),
        ];

        let r = kernel.router
            .select_target(&RouteStrategy::PriorityWeighted, "pw", targets, "gpt-4o")
            .unwrap();
        assert_eq!(r.target.id, "low");
        assert_eq!(r.alternatives.len(), 1);
        assert_eq!(r.alternatives[0].id, "high");
    }

    #[test]
    fn priority_weighted_distributes_by_weight() {
        let config = KernelConfig::default();
        let kernel = KernelBuilder::new(config).build();

        let targets = vec![
            make_target("heavy", 1, 90),
            make_target("light", 1, 10),
        ];

        let mut heavy_count = 0;
        for _ in 0..100 {
            let r = kernel.router
                .select_target(&RouteStrategy::PriorityWeighted, "wd", targets.clone(), "gpt-4o")
                .unwrap();
            if r.target.id == "heavy" {
                heavy_count += 1;
            }
        }
        assert!(heavy_count >= 85, "heavy should get ~90%, got {heavy_count}%");
    }
}
