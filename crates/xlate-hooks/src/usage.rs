use async_trait::async_trait;
use xlate_core::event::ModelEvent;
use xlate_core::hook::{Hook, HookContext, HookVerdict, OutputCharCount};
use xlate_token::TokenEstimator;

pub struct UsageHook {
    estimator: TokenEstimator,
}

impl UsageHook {
    pub fn new() -> Self {
        Self {
            estimator: TokenEstimator::new(),
        }
    }
}

impl Default for UsageHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for UsageHook {
    fn name(&self) -> &str {
        "usage"
    }

    fn priority(&self) -> i32 {
        15
    }

    async fn on_event(&self, _ctx: &HookContext, event: &mut ModelEvent) -> HookVerdict {
        if let ModelEvent::TurnFinished {
            usage,
            finish_reason,
            ..
        } = event
        {
            tracing::debug!(
                input = ?usage.input_tokens,
                output = ?usage.output_tokens,
                cache_read = ?usage.cache_read_tokens,
                finish = %finish_reason,
                "usage extracted"
            );
        }
        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let mut estimated = false;

        if ctx.metrics.usage.input_tokens.is_none() {
            let est = self
                .estimator
                .estimate_input(&ctx.request.messages, &ctx.request.model);
            ctx.metrics.usage.input_tokens = Some(est);
            estimated = true;
            tracing::debug!(estimated_input = est, "estimated input tokens via heuristic");
        }

        if ctx.metrics.usage.output_tokens.is_none() {
            if let Some(occ) = ctx.extensions.get::<OutputCharCount>() {
                let est = self.estimator.estimate_from_char_count(occ.0);
                ctx.metrics.usage.output_tokens = Some(est);
                estimated = true;
                tracing::debug!(
                    estimated_output = est,
                    output_chars = occ.0,
                    "estimated output tokens from streamed text"
                );
            }
        }

        if estimated {
            ctx.metrics.usage.estimated = true;
        }

        HookVerdict::Continue
    }
}
