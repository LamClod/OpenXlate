use async_trait::async_trait;
use rust_decimal::prelude::*;
use std::sync::Arc;
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::store::{Store, UsageLog, UsageWriter};

use crate::cost_calc::CostResult;

pub struct BillingHook {
    store: Arc<dyn Store>,
    writer: Option<UsageWriter>,
    rate_multiplier: f64,
}

impl BillingHook {
    pub fn new(store: Arc<dyn Store>, rate_multiplier: f64) -> Self {
        Self {
            store,
            writer: None,
            rate_multiplier,
        }
    }

    pub fn with_writer(
        store: Arc<dyn Store>,
        rate_multiplier: f64,
        batch_size: usize,
        flush_interval: std::time::Duration,
    ) -> Self {
        let writer = UsageWriter::new(store.clone(), batch_size, flush_interval);
        Self {
            store,
            writer: Some(writer),
            rate_multiplier,
        }
    }
}

#[async_trait]
impl Hook for BillingHook {
    fn name(&self) -> &str {
        "billing"
    }

    fn priority(&self) -> i32 {
        30
    }

    async fn shutdown(&self) {
        if let Some(ref writer) = self.writer {
            writer.flush().await;
        }
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let breakdown = ctx
            .extensions
            .get::<CostResult>()
            .map(|c| c.breakdown.clone())
            .unwrap_or_default();

        let multiplier =
            Decimal::from_f64(self.rate_multiplier).unwrap_or(Decimal::ONE);
        let adjusted = breakdown.total_cost * multiplier;

        let provider = ctx
            .provider_config
            .as_ref()
            .map(|c| c.plugin.clone())
            .unwrap_or_default();

        let target_id = ctx
            .route
            .as_ref()
            .map(|r| r.target.id.clone())
            .unwrap_or_default();

        let log = UsageLog {
            id: None,
            request_id: ctx.stream_id.to_string(),
            timestamp: xlate_core::now_ms(),
            model: ctx.request.model.clone(),
            requested_model: ctx.original_model.clone(),
            upstream_model: ctx
                .provider_config
                .as_ref()
                .and_then(|c| c.upstream_model.clone()),
            provider,
            target_id,
            source_format: ctx
                .request
                .custom_headers
                .get("x-source-format")
                .cloned()
                .unwrap_or_else(|| "normalized".into()),
            stream: ctx.request.stream,
            input_tokens: ctx.metrics.usage.input_tokens,
            output_tokens: ctx.metrics.usage.output_tokens,
            cache_read_tokens: ctx.metrics.usage.cache_read_tokens,
            cache_write_tokens: ctx.metrics.usage.cache_write_tokens,
            reasoning_tokens: ctx.metrics.usage.reasoning_tokens,
            total_tokens: ctx.metrics.usage.total_tokens.or_else(|| {
                let i = ctx.metrics.usage.input_tokens?;
                let o = ctx.metrics.usage.output_tokens?;
                Some(i + o)
            }),
            usage_estimated: ctx.metrics.usage.estimated,
            input_cost: breakdown.input_cost.to_string(),
            output_cost: breakdown.output_cost.to_string(),
            cache_read_cost: breakdown.cache_read_cost.to_string(),
            cache_write_cost: breakdown.cache_write_cost.to_string(),
            reasoning_cost: breakdown.reasoning_cost.to_string(),
            total_cost: breakdown.total_cost.to_string(),
            rate_multiplier: multiplier.to_string(),
            adjusted_cost: adjusted.to_string(),
            service_tier: breakdown.service_tier.clone(),
            track: breakdown.track.clone(),
            duration_ms: ctx.metrics.total_ms,
            ttft_ms: ctx.metrics.ttft_ms,
            attempt_count: ctx.metrics.attempt,
            success: ctx.metrics.success,
            finish_reason: ctx.metrics.finish_reason.clone(),
            error_kind: ctx.metrics.error.as_ref().map(|e| format!("{e:?}")),
            error_message: ctx.metrics.error.as_ref().map(|e| e.to_string()),
            http_status: ctx.metrics.error.as_ref().and_then(|e| match e {
                xlate_core::XlateError::Provider { status, .. } => *status,
                xlate_core::XlateError::RateLimited(_) => Some(429),
                xlate_core::XlateError::Overloaded => Some(529),
                _ => None,
            }),
            client_id: ctx
                .request
                .custom_headers
                .get("x-client-id")
                .cloned(),
        };

        if let Some(ref writer) = self.writer {
            if let Err(e) = writer.try_send(log) {
                tracing::warn!(error = %e, "usage writer send failed");
            }
        } else if let Err(e) = self.store.record_usage(log).await {
            tracing::warn!(error = %e, "failed to record usage");
        }

        HookVerdict::Continue
    }
}
