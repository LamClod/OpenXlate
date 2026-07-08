use async_trait::async_trait;
use xlate_core::config::SanitizeConfig;
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::sanitize::sanitize_messages_with;

pub struct SanitizeHook {
    remove_empty_assistants: bool,
    merge_adjacent_tool_calls: bool,
    trim_dangling_tool_calls: bool,
    trim_trailing_prefill: bool,
}

impl SanitizeHook {
    pub fn new() -> Self {
        Self {
            remove_empty_assistants: true,
            merge_adjacent_tool_calls: true,
            trim_dangling_tool_calls: true,
            trim_trailing_prefill: true,
        }
    }

    pub fn from_config(config: &SanitizeConfig) -> Self {
        Self {
            remove_empty_assistants: config.remove_empty_assistants,
            merge_adjacent_tool_calls: config.merge_adjacent_tool_calls,
            trim_dangling_tool_calls: config.trim_dangling_tool_calls,
            trim_trailing_prefill: config.trim_trailing_prefill,
        }
    }
}

impl Default for SanitizeHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for SanitizeHook {
    fn name(&self) -> &str {
        "sanitize"
    }

    fn priority(&self) -> i32 {
        10
    }

    async fn pre_route(&self, ctx: &mut HookContext) -> HookVerdict {
        ctx.request.messages = sanitize_messages_with(
            std::mem::take(&mut ctx.request.messages),
            self.remove_empty_assistants,
            self.merge_adjacent_tool_calls,
            self.trim_dangling_tool_calls,
            self.trim_trailing_prefill,
        );
        HookVerdict::Continue
    }
}
