use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;
use xlate_core::error::XlateError;
use xlate_core::hook::{Hook, HookContext, HookVerdict, RectifyAction};
use xlate_core::store::{HealCacheEntry, Store};

pub struct ParamHealHook {
    bad_params: DashMap<String, HashSet<String>>,
    store: Option<Arc<dyn Store>>,
    max_attempts: u32,
}

impl ParamHealHook {
    pub fn new() -> Self {
        Self {
            bad_params: DashMap::new(),
            store: None,
            max_attempts: 3,
        }
    }

    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    pub async fn load_from_store(&self) {
        if let Some(ref store) = self.store {
            match store.load_heal_cache().await {
                Ok(entries) => {
                    for entry in entries {
                        let key = format!("{}:{}", entry.provider, entry.model);
                        self.bad_params
                            .entry(key)
                            .or_default()
                            .insert(entry.param.clone());
                    }
                    tracing::info!(
                        entries = self.bad_params.len(),
                        "loaded param heal cache from store"
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load heal cache");
                }
            }
        }
    }

    fn cache_key(ctx: &HookContext) -> String {
        let provider = ctx
            .provider_config
            .as_ref()
            .map(|c| c.plugin.as_str())
            .unwrap_or("unknown");
        format!("{}:{}", provider, ctx.request.model)
    }

    fn extract_bad_param(message: &str) -> Option<String> {
        let patterns = [
            "unsupported parameter: ",
            "unknown parameter: ",
            "Unsupported parameter: ",
            "invalid parameter: ",
            "is not supported with ",
        ];
        for pat in &patterns {
            if let Some(idx) = message.find(pat) {
                let rest = &message[idx + pat.len()..];
                let param: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                    .collect();
                if !param.is_empty() {
                    return Some(param);
                }
            }
        }

        let msg_lower = message.to_lowercase();
        let known_params = [
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "reasoning_effort",
            "logprobs",
            "seed",
        ];
        for param in &known_params {
            if msg_lower.contains(param)
                && (msg_lower.contains("not supported")
                    || msg_lower.contains("not allowed")
                    || msg_lower.contains("invalid"))
            {
                return Some(param.to_string());
            }
        }

        None
    }

}

impl Default for ParamHealHook {
    fn default() -> Self {
        Self::new()
    }
}

struct HealAttempts(u32);
pub struct RemovedParams(pub Vec<String>);

#[async_trait]
impl Hook for ParamHealHook {
    fn name(&self) -> &str {
        "param-heal"
    }

    fn priority(&self) -> i32 {
        210
    }

    async fn pre_send(&self, ctx: &mut HookContext) -> HookVerdict {
        let key = Self::cache_key(ctx);
        let mut removed = Vec::new();

        if let Some(params) = self.bad_params.get(&key) {
            if let Some(ref mut map) = ctx.request.extra_params {
                for param in params.iter() {
                    if map.remove(param).is_some() {
                        tracing::debug!(param, key = %key, "removed cached bad param");
                        removed.push(param.clone());
                    }
                }
            }

            for param in params.iter() {
                match param.as_str() {
                    "temperature" | "top_p" | "top_k" | "frequency_penalty"
                    | "presence_penalty" | "seed" | "logprobs" => {
                        if let Some(ref mut map) = ctx.request.extra_params {
                            if map.remove(param.as_str()).is_some() {
                                removed.push(param.clone());
                            }
                        }
                    }
                    "reasoning_effort" => {
                        if ctx.request.reasoning_effort.is_some() {
                            ctx.request.reasoning_effort = None;
                            removed.push(param.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        if !removed.is_empty() {
            ctx.extensions.insert(RemovedParams(removed));
        }

        HookVerdict::Continue
    }

    async fn on_error(&self, ctx: &mut HookContext, error: &XlateError) -> HookVerdict {
        let attempts = ctx
            .extensions
            .get::<HealAttempts>()
            .map(|a| a.0)
            .unwrap_or(0);
        if attempts >= self.max_attempts {
            return HookVerdict::Continue;
        }

        let message = match error {
            XlateError::Provider { message, .. } => message.as_str(),
            XlateError::InvalidRequest(msg) => msg.as_str(),
            _ => return HookVerdict::Continue,
        };

        if let Some(param) = Self::extract_bad_param(message) {
            let key = Self::cache_key(ctx);
            tracing::info!(
                param = %param,
                key = %key,
                attempt = attempts + 1,
                "caching unsupported parameter for auto-removal"
            );

            self.bad_params
                .entry(key.clone())
                .or_default()
                .insert(param.clone());

            if let Some(ref store) = self.store {
                let provider = ctx
                    .provider_config
                    .as_ref()
                    .map(|c| c.plugin.clone())
                    .unwrap_or_else(|| "unknown".into());
                let entry = HealCacheEntry {
                    model: ctx.request.model.clone(),
                    provider,
                    param: param.clone(),
                    removed_at: xlate_core::now_ms(),
                };
                let _ = store.save_heal_entry(entry).await;
            }

            ctx.extensions.insert(HealAttempts(attempts + 1));

            return HookVerdict::Rectify(RectifyAction::ModifyRequest);
        }

        HookVerdict::Continue
    }
}
