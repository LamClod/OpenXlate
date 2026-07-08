use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use xlate_core::event::ModelEvent;
use xlate_core::hook::{Hook, HookContext, HookVerdict, StreamId};
use xlate_core::router::LatencyTracker;

pub struct LatencyHook {
    first_token_times: DashMap<StreamId, Instant>,
    tracker: Arc<LatencyTracker>,
}

impl LatencyHook {
    pub fn new() -> Self {
        Self {
            first_token_times: DashMap::new(),
            tracker: Arc::new(LatencyTracker::new(1000)),
        }
    }

    pub fn with_tracker(tracker: Arc<LatencyTracker>) -> Self {
        Self {
            first_token_times: DashMap::new(),
            tracker,
        }
    }

    pub fn tracker(&self) -> &Arc<LatencyTracker> {
        &self.tracker
    }

    pub fn p50(&self, target_id: &str) -> Option<f64> {
        self.tracker.p50(target_id)
    }

    pub fn p95(&self, target_id: &str) -> Option<f64> {
        self.tracker.p95(target_id)
    }

    pub fn p99(&self, target_id: &str) -> Option<f64> {
        self.tracker.p99(target_id)
    }
}

impl Default for LatencyHook {
    fn default() -> Self {
        Self::new()
    }
}

fn is_content_bearing(event: &ModelEvent) -> bool {
    matches!(
        event,
        ModelEvent::TextDelta { .. } | ModelEvent::ThinkingDelta { .. }
    )
}

#[async_trait]
impl Hook for LatencyHook {
    fn name(&self) -> &str {
        "latency"
    }

    fn priority(&self) -> i32 {
        20
    }

    async fn on_event(&self, ctx: &HookContext, event: &mut ModelEvent) -> HookVerdict {
        if is_content_bearing(event) && !self.first_token_times.contains_key(&ctx.stream_id) {
            self.first_token_times
                .insert(ctx.stream_id, Instant::now());
        }
        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let total = ctx.created_at.elapsed().as_secs_f64() * 1000.0;
        ctx.metrics.total_ms = Some(total);

        if let Some((_, first_token_at)) = self.first_token_times.remove(&ctx.stream_id) {
            let ttft = (first_token_at - ctx.created_at).as_secs_f64() * 1000.0;
            ctx.metrics.ttft_ms = Some(ttft);
        }

        let target_id = ctx
            .route
            .as_ref()
            .map(|r| r.target.id.as_str())
            .unwrap_or("unknown");
        self.tracker.record(target_id, total);

        HookVerdict::Continue
    }
}
