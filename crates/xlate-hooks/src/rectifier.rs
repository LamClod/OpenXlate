use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use xlate_core::config::RectifierConfig;
use xlate_core::error::XlateError;
use xlate_core::hook::{Hook, HookContext, HookVerdict, RectifyAction};
use xlate_core::registry::{ModelRegistry, ThinkingSupport};

pub struct RectifierHook {
    thinking_signature: bool,
    thinking_budget: bool,
    thinking_effort_conflict: bool,
    context_length: bool,
    retry_after: bool,
    registry: Option<Arc<ModelRegistry>>,
}

impl RectifierHook {
    pub fn new() -> Self {
        Self {
            thinking_signature: true,
            thinking_budget: true,
            thinking_effort_conflict: true,
            context_length: true,
            retry_after: true,
            registry: None,
        }
    }

    pub fn from_config(config: &RectifierConfig) -> Self {
        Self {
            thinking_signature: config.thinking_signature,
            thinking_budget: config.thinking_budget,
            thinking_effort_conflict: config.thinking_effort_conflict,
            context_length: config.context_length,
            retry_after: config.retry_after,
            registry: None,
        }
    }

    pub fn with_registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    fn lookup_budget_min(&self, model: &str) -> u32 {
        if let Some(ref registry) = self.registry {
            if let Some(meta) = registry.get(model) {
                if let ThinkingSupport::Mandatory { budget_min } = meta.thinking {
                    return budget_min;
                }
            }
        }
        1024
    }
}

impl Default for RectifierHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for RectifierHook {
    fn name(&self) -> &str {
        "rectifier"
    }

    fn priority(&self) -> i32 {
        10
    }

    async fn on_error(&self, ctx: &mut HookContext, error: &XlateError) -> HookVerdict {
        match error {
            XlateError::RateLimited(_) => {
                if self.retry_after {
                    tracing::info!("rate limited, retrying after 1s");
                    HookVerdict::Rectify(RectifyAction::RetryAfter(Duration::from_secs(1)))
                } else {
                    HookVerdict::Continue
                }
            }
            XlateError::Overloaded => {
                tracing::info!("overloaded, retrying after 2s");
                HookVerdict::Rectify(RectifyAction::RetryAfter(Duration::from_secs(2)))
            }
            XlateError::Provider { message, status } => {
                let msg_lower = message.to_lowercase();

                if self.thinking_signature
                    && (msg_lower.contains("thinking_signature")
                        || msg_lower.contains("reasoning_signature"))
                {
                    if let Some(last_asst) = ctx
                        .request
                        .messages
                        .iter_mut()
                        .rev()
                        .find(|m| m.role == "assistant")
                    {
                        if !last_asst.reasoning_signature.is_empty() {
                            last_asst.reasoning_signature.clear();
                            last_asst.reasoning_signature_source.clear();
                            tracing::info!("cleared reasoning_signature from last assistant message");
                            return HookVerdict::Rectify(RectifyAction::ModifyRequest);
                        }
                    }
                }

                if self.thinking_budget
                    && msg_lower.contains("budget_tokens")
                    && (msg_lower.contains("too low")
                        || msg_lower.contains("minimum")
                        || msg_lower.contains("at least"))
                {
                    if let Some(ref mut thinking) = ctx.request.thinking {
                        let min = self.lookup_budget_min(&ctx.request.model);
                        thinking.budget_tokens = Some(min);
                        tracing::info!(budget_min = min, "adjusted thinking budget_tokens to model minimum");
                        return HookVerdict::Rectify(RectifyAction::ModifyRequest);
                    }
                }

                if self.thinking_effort_conflict
                    && msg_lower.contains("reasoning_effort")
                    && (msg_lower.contains("conflict") || msg_lower.contains("not supported"))
                    && ctx.request.thinking.is_some()
                {
                    ctx.request.reasoning_effort = None;
                    tracing::info!("removed reasoning_effort due to conflict with thinking");
                    return HookVerdict::Rectify(RectifyAction::ModifyRequest);
                }

                if self.context_length
                    && (msg_lower.contains("context_length")
                        || msg_lower.contains("context length")
                        || msg_lower.contains("maximum context")
                        || msg_lower.contains("too many tokens"))
                    && ctx.request.messages.len() > 2
                {
                    let idx = ctx
                        .request
                        .messages
                        .iter()
                        .position(|m| m.role != "system")
                        .unwrap_or(1);
                    let removed = ctx.request.messages.remove(idx);
                    tracing::info!(
                        role = %removed.role,
                        "trimmed oldest non-system message for context length"
                    );
                    return HookVerdict::Rectify(RectifyAction::ModifyRequest);
                }

                if self.retry_after && matches!(status, Some(429)) {
                    let delay = parse_retry_after(message).unwrap_or(1);
                    tracing::info!(delay_s = delay, "HTTP 429, retrying after delay");
                    return HookVerdict::Rectify(RectifyAction::RetryAfter(
                        Duration::from_secs(delay),
                    ));
                }

                if matches!(status, Some(529)) {
                    tracing::info!("HTTP 529 overloaded, switching target");
                    return HookVerdict::Rectify(RectifyAction::SwitchTarget);
                }

                HookVerdict::Continue
            }
            _ => HookVerdict::Continue,
        }
    }
}

fn parse_retry_after(message: &str) -> Option<u64> {
    let lower = message.to_lowercase();
    if let Some(idx) = lower.find("retry-after") {
        let rest = &lower[idx + "retry-after".len()..];
        let trimmed = rest.trim_start_matches(|c: char| c == '=' || c == ':' || c.is_whitespace());
        let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(secs) = digits.parse::<u64>() {
            if secs > 0 && secs < 300 {
                return Some(secs);
            }
        }
    }
    None
}
