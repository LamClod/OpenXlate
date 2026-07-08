use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::store::{Store, RequestTrace};

use crate::param_heal::RemovedParams;

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub stream_id: u64,
    pub timestamp: i64,

    pub model: String,
    pub original_model: String,
    pub provider: String,
    pub target_id: String,

    pub message_count: usize,
    pub tool_count: usize,

    pub success: bool,
    pub duration_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub attempt_count: u32,

    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_tokens: Option<i64>,
    pub total_cost: Option<String>,

    pub error_kind: Option<String>,
    pub error_message: Option<String>,
    pub finish_reason: Option<String>,

    pub format: String,
    pub removed_params: Vec<String>,
    pub request_body: Option<String>,
}

pub struct TraceHook {
    traces: Mutex<VecDeque<TraceEntry>>,
    max_entries: usize,
    store: Option<Arc<dyn Store>>,
    capture_request_body: bool,
    max_body_size: usize,
}

impl TraceHook {
    pub fn new(max_entries: usize) -> Self {
        Self {
            traces: Mutex::new(VecDeque::with_capacity(max_entries)),
            max_entries,
            store: None,
            capture_request_body: true,
            max_body_size: 65536,
        }
    }

    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_capture_config(mut self, capture: bool, max_size: usize) -> Self {
        self.capture_request_body = capture;
        self.max_body_size = max_size;
        self
    }

    pub fn recent(&self, limit: usize) -> Vec<TraceEntry> {
        self.traces
            .lock()
            .map(|t| t.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for TraceHook {
    fn default() -> Self {
        Self::new(500)
    }
}

#[async_trait]
impl Hook for TraceHook {
    fn name(&self) -> &str {
        "trace"
    }

    fn priority(&self) -> i32 {
        40
    }

    fn snapshot(&self) -> Option<serde_json::Value> {
        let (count, recent_error_count) = self
            .traces
            .lock()
            .map(|t| {
                let errors = t.iter().filter(|e| !e.success).count();
                (t.len(), errors)
            })
            .unwrap_or_default();
        Some(serde_json::json!({
            "ring_buffer_size": self.max_entries,
            "current_count": count,
            "recent_error_count": recent_error_count,
            "capture_request_body": self.capture_request_body,
        }))
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let entry = TraceEntry {
            stream_id: ctx.stream_id,
            timestamp: xlate_core::now_ms(),
            model: ctx.request.model.clone(),
            original_model: ctx.original_model.clone(),
            provider: ctx
                .provider_config
                .as_ref()
                .map(|c| c.plugin.clone())
                .unwrap_or_default(),
            target_id: ctx
                .route
                .as_ref()
                .map(|r| r.target.id.clone())
                .unwrap_or_default(),
            message_count: ctx.request.messages.len(),
            tool_count: ctx.request.tools.len(),
            success: ctx.metrics.success,
            duration_ms: ctx.metrics.total_ms,
            ttft_ms: ctx.metrics.ttft_ms,
            attempt_count: ctx.metrics.attempt,
            input_tokens: ctx.metrics.usage.input_tokens,
            output_tokens: ctx.metrics.usage.output_tokens,
            cache_read_tokens: ctx.metrics.usage.cache_read_tokens,
            total_cost: ctx
                .metrics
                .cost
                .as_ref()
                .map(|c| c.total_cost.to_string()),
            error_kind: ctx.metrics.error.as_ref().map(|e| format!("{e:?}")),
            error_message: ctx.metrics.error.as_ref().map(|e| e.to_string()),
            finish_reason: ctx.metrics.finish_reason.clone(),
            format: ctx
                .request
                .custom_headers
                .get("x-source-format")
                .cloned()
                .unwrap_or_else(|| "normalized".into()),
            removed_params: ctx
                .extensions
                .get::<RemovedParams>()
                .map(|r| r.0.clone())
                .unwrap_or_default(),
            request_body: if self.capture_request_body {
                let body = serde_json::to_string(&ctx.request).unwrap_or_default();
                if body.len() > self.max_body_size {
                    Some(body[..self.max_body_size].to_string())
                } else {
                    Some(body)
                }
            } else {
                None
            },
        };

        if let Some(ref store) = self.store {
            let record = RequestTrace {
                stream_id: entry.stream_id,
                timestamp: entry.timestamp,
                model: entry.model.clone(),
                provider: entry.provider.clone(),
                duration_ms: entry.duration_ms,
                success: entry.success,
                hooks_fired: ctx
                    .extensions
                    .get::<xlate_core::hook::HooksFired>()
                    .map(|h| h.0.clone())
                    .unwrap_or_default(),
            };
            let store = store.clone();
            tokio::spawn(async move {
                if let Err(e) = store.record_trace(record).await {
                    tracing::warn!(error = %e, "failed to persist trace");
                }
            });
        }

        if let Ok(mut traces) = self.traces.lock() {
            if traces.len() >= self.max_entries {
                traces.pop_front();
            }
            traces.push_back(entry);
        }

        HookVerdict::Continue
    }
}
