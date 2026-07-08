use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Mutex;
use xlate_core::event::ModelEvent;
use xlate_core::hook::{Hook, HookContext, HookVerdict, StreamId};
use xlate_core::think_tag::{ContentPartKind, ThinkTagParser};

pub struct ThinkTagHook {
    parsers: DashMap<StreamId, Mutex<ThinkTagParser>>,
}

impl ThinkTagHook {
    pub fn new() -> Self {
        Self {
            parsers: DashMap::new(),
        }
    }
}

impl Default for ThinkTagHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for ThinkTagHook {
    fn name(&self) -> &str {
        "think-tag"
    }

    fn priority(&self) -> i32 {
        10
    }

    async fn on_event(&self, ctx: &HookContext, event: &mut ModelEvent) -> HookVerdict {
        match event {
            ModelEvent::TextDelta { text, meta } => {
                let entry = self
                    .parsers
                    .entry(ctx.stream_id)
                    .or_insert_with(|| Mutex::new(ThinkTagParser::new()));

                let parts = match entry.lock() {
                    Ok(mut parser) => parser.consume(text),
                    Err(_) => return HookVerdict::Continue,
                };

                if parts.is_empty() {
                    return HookVerdict::Skip;
                }

                let mut reasoning_text = String::new();
                let mut plain_text = String::new();
                let mut saw_completed = false;

                for part in &parts {
                    match part.kind {
                        ContentPartKind::Reasoning => reasoning_text.push_str(&part.text),
                        ContentPartKind::Text => plain_text.push_str(&part.text),
                        ContentPartKind::ThinkingCompleted => saw_completed = true,
                    }
                }

                if !reasoning_text.is_empty() {
                    *event = ModelEvent::ThinkingDelta {
                        text: reasoning_text,
                        style: Some("think-tag".into()),
                        meta: meta.clone(),
                    };
                } else if saw_completed {
                    *event = ModelEvent::ThinkingCompleted {
                        duration_ms: None,
                        signature: String::new(),
                        signature_source: String::new(),
                        meta: meta.clone(),
                    };
                } else if !plain_text.is_empty() {
                    *text = plain_text;
                } else {
                    return HookVerdict::Skip;
                }
            }
            ModelEvent::TurnFinished { .. } => {
                self.parsers.remove(&ctx.stream_id);
            }
            _ => {}
        }

        HookVerdict::Continue
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        self.parsers.remove(&ctx.stream_id);
        HookVerdict::Continue
    }
}
