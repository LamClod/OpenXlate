use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use xlate_core::hook::{Hook, HookContext, HookVerdict};
use xlate_core::registry::ModelRegistry;

pub struct ModelMapHook {
    rules: HashMap<String, String>,
    chain_redirect: bool,
    registry: Option<Arc<ModelRegistry>>,
}

impl ModelMapHook {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            chain_redirect: false,
            registry: None,
        }
    }

    pub fn with_rules(rules: Vec<(String, String)>, chain: bool) -> Self {
        Self {
            rules: rules.into_iter().collect(),
            chain_redirect: chain,
            registry: None,
        }
    }

    pub fn with_registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    fn resolve(&self, model: &str) -> Option<String> {
        let mut current = model.to_string();
        let mut seen = std::collections::HashSet::new();
        let max_depth = if self.chain_redirect { 10 } else { 1 };

        for _ in 0..max_depth {
            if !seen.insert(current.clone()) {
                break;
            }
            if let Some(target) = self.rules.get(&current) {
                current = target.clone();
                continue;
            }
            if let Some(ref reg) = self.registry {
                if let Some(alias_target) = reg.resolve_alias(&current) {
                    current = alias_target;
                    continue;
                }
            }
            break;
        }

        if current != model {
            Some(current)
        } else {
            None
        }
    }
}

impl Default for ModelMapHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for ModelMapHook {
    fn name(&self) -> &str {
        "model-map"
    }

    fn priority(&self) -> i32 {
        20
    }

    async fn pre_route(&self, ctx: &mut HookContext) -> HookVerdict {
        if let Some(mapped) = self.resolve(&ctx.request.model) {
            tracing::debug!(
                from = %ctx.request.model,
                to = %mapped,
                "model mapped"
            );
            ctx.request.model = mapped;
        }
        HookVerdict::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mapping() {
        let hook = ModelMapHook::with_rules(
            vec![("gpt-5".into(), "gpt-4o".into())],
            false,
        );
        assert_eq!(hook.resolve("gpt-5"), Some("gpt-4o".into()));
        assert_eq!(hook.resolve("gpt-4o"), None);
    }

    #[test]
    fn chain_redirect() {
        let hook = ModelMapHook::with_rules(
            vec![
                ("a".into(), "b".into()),
                ("b".into(), "c".into()),
            ],
            true,
        );
        assert_eq!(hook.resolve("a"), Some("c".into()));
    }

    #[test]
    fn no_chain_without_flag() {
        let hook = ModelMapHook::with_rules(
            vec![
                ("a".into(), "b".into()),
                ("b".into(), "c".into()),
            ],
            false,
        );
        assert_eq!(hook.resolve("a"), Some("b".into()));
    }
}
