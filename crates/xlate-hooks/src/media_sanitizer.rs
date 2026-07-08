use std::sync::Arc;

use async_trait::async_trait;
use xlate_core::error::XlateError;
use xlate_core::hook::{Hook, HookContext, HookVerdict, RectifyAction};
use xlate_core::registry::ModelRegistry;
use xlate_core::types::{ContentPart, ContentPartType};

#[derive(Debug, Clone, PartialEq)]
pub enum SanitizerStrategy {
    Preventive,
    Reactive,
    Both,
}

impl SanitizerStrategy {
    pub fn from_str(s: &str) -> Self {
        match s {
            "preventive" => Self::Preventive,
            "reactive" => Self::Reactive,
            _ => Self::Both,
        }
    }
}

pub struct MediaSanitizerHook {
    vision_patterns: Vec<String>,
    registry: Option<Arc<ModelRegistry>>,
    strategy: SanitizerStrategy,
}

impl MediaSanitizerHook {
    pub fn new() -> Self {
        Self {
            vision_patterns: vec![
                "*vision*".into(),
                "*4o*".into(),
                "*gpt-4-turbo*".into(),
                "claude-*".into(),
                "*gemini*".into(),
            ],
            registry: None,
            strategy: SanitizerStrategy::Both,
        }
    }

    pub fn with_patterns(patterns: Vec<String>) -> Self {
        Self {
            vision_patterns: patterns,
            registry: None,
            strategy: SanitizerStrategy::Both,
        }
    }

    pub fn with_registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn with_strategy(mut self, strategy: &str) -> Self {
        self.strategy = SanitizerStrategy::from_str(strategy);
        self
    }

    fn strip_images(ctx: &mut HookContext) -> bool {
        let mut stripped = false;
        for msg in &mut ctx.request.messages {
            let mut i = 0;
            while i < msg.content_parts.len() {
                if matches!(msg.content_parts[i].kind, ContentPartType::Image) {
                    let mime = msg.content_parts[i]
                        .image
                        .as_ref()
                        .map(|img| img.mime_type.as_str())
                        .unwrap_or("unknown");
                    let placeholder = format!("[image removed: {}]", mime);
                    msg.content_parts[i] = ContentPart {
                        kind: ContentPartType::Text,
                        text: placeholder,
                        image: None,
                    };
                    stripped = true;
                }
                i += 1;
            }
        }
        stripped
    }

    fn is_image_rejection_error(msg: &str) -> bool {
        let lower = msg.to_lowercase();
        lower.contains("does not support image")
            || lower.contains("vision is not supported")
            || lower.contains("invalid content type: image")
            || lower.contains("image input is not supported")
            || lower.contains("image_not_supported")
    }

    fn supports_vision(&self, model: &str) -> bool {
        if let Some(ref reg) = self.registry {
            return reg.supports_vision(model);
        }
        self.vision_patterns
            .iter()
            .any(|p| glob_match::glob_match(p, model))
    }
}

impl Default for MediaSanitizerHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for MediaSanitizerHook {
    fn name(&self) -> &str {
        "media-sanitizer"
    }

    fn priority(&self) -> i32 {
        30
    }

    async fn pre_route(&self, ctx: &mut HookContext) -> HookVerdict {
        if self.strategy == SanitizerStrategy::Reactive {
            return HookVerdict::Continue;
        }

        if self.supports_vision(&ctx.request.model) {
            return HookVerdict::Continue;
        }

        if Self::strip_images(ctx) {
            tracing::debug!(
                model = %ctx.request.model,
                "stripped image content for text-only model"
            );
        }

        HookVerdict::Continue
    }

    async fn on_error(&self, ctx: &mut HookContext, error: &XlateError) -> HookVerdict {
        if self.strategy == SanitizerStrategy::Preventive {
            return HookVerdict::Continue;
        }

        if let XlateError::Provider { message, .. } = error {
            if Self::is_image_rejection_error(message) && Self::strip_images(ctx) {
                tracing::info!(
                    model = %ctx.request.model,
                    "reactively stripped images after provider rejection"
                );
                return HookVerdict::Rectify(RectifyAction::ModifyRequest);
            }
        }
        HookVerdict::Continue
    }
}
