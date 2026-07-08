use async_trait::async_trait;
use dashmap::DashMap;
use rust_decimal::Decimal;
use std::sync::atomic::{AtomicI64, Ordering};
use xlate_core::pricing::{PricingCatalog, PricingInfo};
use xlate_core::registry::{ModelMeta, ModelRegistry};

pub struct RemotePricingCatalog {
    models: DashMap<String, PricingInfo>,
    provider_models: DashMap<(String, String), PricingInfo>,
    last_updated: AtomicI64,
}

impl RemotePricingCatalog {
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
            provider_models: DashMap::new(),
            last_updated: AtomicI64::new(0),
        }
    }

    pub fn load_from_json(&self, data: &serde_json::Value) -> usize {
        let mut count = 0;

        let models = match data.as_array() {
            Some(arr) => arr,
            None => match data.get("data").and_then(|d| d.as_array()) {
                Some(arr) => arr,
                None => return 0,
            },
        };

        for entry in models {
            let id = match entry.get("model").or_else(|| entry.get("id")).and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None => continue,
            };

            let input = parse_price(entry, &["input_price", "prompt_price", "input_cost_per_token"]);
            let output = parse_price(entry, &["output_price", "completion_price", "output_cost_per_token"]);
            let cache_read = parse_price(entry, &["cache_read_price", "cached_input_price"]);
            let cache_write = parse_price(entry, &["cache_write_price", "cached_output_price"]);
            let reasoning = parse_price(entry, &["reasoning_price"]).or(Some(output.unwrap_or_default()));

            let source = entry
                .get("source")
                .or_else(|| entry.get("provider"))
                .and_then(|v| v.as_str())
                .map(String::from);

            if let (Some(inp), Some(out)) = (input, output) {
                let service_tier = entry
                    .get("service_tier")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let track = entry
                    .get("track")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                let info = PricingInfo {
                    input_per_mtok: inp,
                    output_per_mtok: out,
                    cache_read_per_mtok: cache_read.unwrap_or_default(),
                    cache_write_per_mtok: cache_write.unwrap_or_default(),
                    reasoning_per_mtok: reasoning.unwrap_or(out),
                    source: source.clone(),
                    service_tier,
                    track,
                };

                if let Some(ref provider) = source {
                    self.provider_models
                        .insert((provider.clone(), id.clone()), info.clone());
                }
                self.models.insert(id, info);
                count += 1;
            }
        }

        self.last_updated.store(xlate_core::now_ms(), Ordering::Relaxed);
        count
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn merge_into_registry(&self, registry: &ModelRegistry) {
        let mut count = 0;
        for entry in self.models.iter() {
            let model_id = entry.key().clone();
            let vendor = entry.value().source.clone().unwrap_or_default();
            let meta = ModelMeta {
                id: model_id.clone(),
                display_name: model_id,
                vendor,
                ..Default::default()
            };
            registry.register_if_absent(meta);
            count += 1;
        }
        if count > 0 {
            tracing::debug!(count, "merged pricing catalog models into registry");
        }
    }
}

impl Default for RemotePricingCatalog {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_price(entry: &serde_json::Value, keys: &[&str]) -> Option<Decimal> {
    for key in keys {
        if let Some(val) = entry.get(*key) {
            if let Some(s) = val.as_str() {
                if let Ok(d) = s.parse::<Decimal>() {
                    return Some(d);
                }
            }
            if val.is_number() {
                if let Ok(d) = val.to_string().parse::<Decimal>() {
                    return Some(d);
                }
            }
        }
    }
    None
}

#[async_trait]
impl PricingCatalog for RemotePricingCatalog {
    fn get_pricing_for(&self, model: &str, provider: Option<&str>) -> Option<PricingInfo> {
        if let Some(p) = provider {
            let key = (p.to_string(), model.to_string());
            if let Some(info) = self.provider_models.get(&key) {
                return Some(info.clone());
            }
            for entry in self.provider_models.iter() {
                if entry.key().0 == p && glob_match::glob_match(&entry.key().1, model) {
                    return Some(entry.value().clone());
                }
            }
        }
        if let Some(info) = self.models.get(model) {
            return Some(info.clone());
        }
        for entry in self.models.iter() {
            if glob_match::glob_match(entry.key(), model) {
                return Some(entry.value().clone());
            }
        }
        None
    }

    async fn refresh(&self) -> Result<(), xlate_core::error::XlateError> {
        Ok(())
    }

    fn last_updated(&self) -> Option<i64> {
        let ts = self.last_updated.load(Ordering::Relaxed);
        if ts > 0 { Some(ts) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_array_format() {
        let data = serde_json::json!([
            {
                "model": "gpt-4o",
                "input_price": 2.5,
                "output_price": 10.0,
                "cache_read_price": 1.25,
                "source": "test"
            },
            {
                "model": "claude-sonnet-4*",
                "input_price": 3.0,
                "output_price": 15.0,
                "cache_read_price": 0.3,
                "cache_write_price": 3.75,
                "source": "test"
            }
        ]);

        let catalog = RemotePricingCatalog::new();
        let count = catalog.load_from_json(&data);
        assert_eq!(count, 2);

        let gpt = catalog.get_pricing("gpt-4o").unwrap();
        assert_eq!(gpt.input_per_mtok, Decimal::from_str_exact("2.5").unwrap());

        let claude = catalog.get_pricing("claude-sonnet-4-20250514").unwrap();
        assert_eq!(claude.output_per_mtok, Decimal::from_str_exact("15").unwrap());
    }

    #[test]
    fn load_data_wrapper_format() {
        let data = serde_json::json!({
            "data": [
                { "id": "test-model", "input_price": 1.0, "output_price": 2.0 }
            ]
        });

        let catalog = RemotePricingCatalog::new();
        assert_eq!(catalog.load_from_json(&data), 1);
        assert!(catalog.get_pricing("test-model").is_some());
    }
}
