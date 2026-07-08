use async_trait::async_trait;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use xlate_core::hook::{CostBreakdown, Hook, HookContext, HookVerdict};
use xlate_core::pricing::{PricingCatalog, PricingInfo};

#[derive(Debug, Clone)]
pub struct ModelPricing {
    pub input_per_mtok: Decimal,
    pub output_per_mtok: Decimal,
    pub cache_read_per_mtok: Decimal,
    pub cache_write_per_mtok: Decimal,
    pub reasoning_per_mtok: Decimal,
    pub source: Option<String>,
    pub service_tier: Option<String>,
    pub track: Option<String>,
}

impl ModelPricing {
    pub fn from_f64(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self {
            input_per_mtok: Decimal::from_f64(input).unwrap_or_default(),
            output_per_mtok: Decimal::from_f64(output).unwrap_or_default(),
            cache_read_per_mtok: Decimal::from_f64(cache_read).unwrap_or_default(),
            cache_write_per_mtok: Decimal::from_f64(cache_write).unwrap_or_default(),
            reasoning_per_mtok: Decimal::from_f64(output).unwrap_or_default(),
            source: None,
            service_tier: None,
            track: None,
        }
    }
}

impl From<&PricingInfo> for ModelPricing {
    fn from(info: &PricingInfo) -> Self {
        Self {
            input_per_mtok: info.input_per_mtok,
            output_per_mtok: info.output_per_mtok,
            cache_read_per_mtok: info.cache_read_per_mtok,
            cache_write_per_mtok: info.cache_write_per_mtok,
            reasoning_per_mtok: info.reasoning_per_mtok,
            source: info.source.clone(),
            service_tier: info.service_tier.clone(),
            track: info.track.clone(),
        }
    }
}

pub struct CostResult {
    pub breakdown: CostBreakdown,
}

pub struct CostCalcHook {
    catalog: HashMap<String, ModelPricing>,
    external: Option<Arc<dyn PricingCatalog>>,
    rate_multiplier: Decimal,
}

impl CostCalcHook {
    pub fn new() -> Self {
        Self {
            catalog: default_catalog(),
            external: None,
            rate_multiplier: Decimal::ONE,
        }
    }

    pub fn with_rate_multiplier(mut self, multiplier: f64) -> Self {
        self.rate_multiplier = Decimal::from_f64(multiplier).unwrap_or(Decimal::ONE);
        self
    }

    pub fn with_catalog(catalog: HashMap<String, ModelPricing>) -> Self {
        Self {
            catalog,
            external: None,
            rate_multiplier: Decimal::ONE,
        }
    }

    pub fn with_external_pricing(mut self, pricing: Arc<dyn PricingCatalog>) -> Self {
        self.external = Some(pricing);
        self
    }

    fn lookup(&self, model: &str, provider: Option<&str>) -> Option<ModelPricing> {
        if let Some(ext) = &self.external {
            if let Some(info) = ext.get_pricing_for(model, provider) {
                return Some(ModelPricing::from(&info));
            }
        }
        if let Some(p) = self.catalog.get(model) {
            return Some(p.clone());
        }
        for (pattern, pricing) in &self.catalog {
            if glob_match::glob_match(pattern, model) {
                return Some(pricing.clone());
            }
        }
        None
    }
}

impl Default for CostCalcHook {
    fn default() -> Self {
        Self::new()
    }
}

pub fn default_catalog() -> HashMap<String, ModelPricing> {
    let mut m = HashMap::new();
    let add = |m: &mut HashMap<String, ModelPricing>,
               pat: &str,
               input: f64,
               output: f64,
               cache_r: f64,
               cache_w: f64| {
        m.insert(pat.to_string(), ModelPricing::from_f64(input, output, cache_r, cache_w));
    };
    // Anthropic
    add(&mut m, "claude-sonnet-4*", 3.0, 15.0, 0.3, 3.75);
    add(&mut m, "claude-opus-4*", 15.0, 75.0, 1.5, 18.75);
    add(&mut m, "claude-3-5-sonnet*", 3.0, 15.0, 0.3, 3.75);
    add(&mut m, "claude-3-5-haiku*", 0.8, 4.0, 0.08, 1.0);
    add(&mut m, "claude-3-opus*", 15.0, 75.0, 1.5, 18.75);
    // OpenAI
    add(&mut m, "gpt-4o", 2.5, 10.0, 1.25, 0.0);
    add(&mut m, "gpt-4o-mini", 0.15, 0.6, 0.075, 0.0);
    add(&mut m, "gpt-4.1", 2.0, 8.0, 0.5, 0.0);
    add(&mut m, "gpt-4.1-mini", 0.4, 1.6, 0.1, 0.0);
    add(&mut m, "gpt-4.1-nano", 0.1, 0.4, 0.025, 0.0);
    add(&mut m, "o3", 2.0, 8.0, 0.5, 0.0);
    add(&mut m, "o3-mini", 1.1, 4.4, 0.275, 0.0);
    add(&mut m, "o4-mini", 1.1, 4.4, 0.275, 0.0);
    // Google
    add(&mut m, "gemini-2.5-pro*", 1.25, 10.0, 0.0, 0.0);
    add(&mut m, "gemini-2.5-flash*", 0.15, 0.6, 0.0, 0.0);
    add(&mut m, "gemini-2.0-flash*", 0.1, 0.4, 0.0, 0.0);
    m
}

fn tokens_cost(tokens: Option<i64>, price_per_mtok: Decimal) -> Decimal {
    tokens
        .map(|t| Decimal::from(t) * price_per_mtok / Decimal::from(1_000_000))
        .unwrap_or_default()
}

#[async_trait]
impl Hook for CostCalcHook {
    fn name(&self) -> &str {
        "cost-calc"
    }

    fn priority(&self) -> i32 {
        20
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let provider = ctx.provider_config.as_ref().map(|c| c.plugin.as_str());
        let pricing = self.lookup(&ctx.request.model, provider).or_else(|| {
            ctx.extensions
                .get::<xlate_core::registry::ModelMeta>()
                .and_then(|m| m.pricing_slug.as_ref())
                .and_then(|slug| self.lookup(slug, provider))
        });

        let pricing = match pricing {
            Some(p) => p,
            None => {
                let breakdown = CostBreakdown::default();
                ctx.metrics.cost = Some(breakdown.clone());
                ctx.extensions.insert(CostResult { breakdown });
                return HookVerdict::Continue;
            }
        };

        let usage = &ctx.metrics.usage;

        let input_cost = tokens_cost(usage.input_tokens, pricing.input_per_mtok);
        let output_cost = tokens_cost(usage.output_tokens, pricing.output_per_mtok);
        let cache_read_cost = tokens_cost(usage.cache_read_tokens, pricing.cache_read_per_mtok);
        let cache_write_cost = tokens_cost(usage.cache_write_tokens, pricing.cache_write_per_mtok);
        let reasoning_cost = tokens_cost(usage.reasoning_tokens, pricing.reasoning_per_mtok);

        let total = input_cost + output_cost + cache_read_cost + cache_write_cost + reasoning_cost;
        let rate_multiplier = self.rate_multiplier;
        let adjusted = total * rate_multiplier;

        let breakdown = CostBreakdown {
            input_cost,
            output_cost,
            cache_read_cost,
            cache_write_cost,
            reasoning_cost,
            total_cost: total,
            rate_multiplier,
            adjusted_cost: adjusted,
            currency: "USD".into(),
            pricing_source: pricing.source,
            service_tier: pricing.service_tier,
            track: pricing.track,
        };

        tracing::debug!(
            model = %ctx.request.model,
            total_cost = %total,
            "cost calculated"
        );

        ctx.metrics.cost = Some(breakdown.clone());
        ctx.extensions.insert(CostResult { breakdown });

        HookVerdict::Continue
    }
}
