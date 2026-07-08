use async_trait::async_trait;
use rust_decimal::Decimal;

#[derive(Debug, Clone)]
pub struct PricingInfo {
    pub input_per_mtok: Decimal,
    pub output_per_mtok: Decimal,
    pub cache_read_per_mtok: Decimal,
    pub cache_write_per_mtok: Decimal,
    pub reasoning_per_mtok: Decimal,
    pub source: Option<String>,
    pub service_tier: Option<String>,
    pub track: Option<String>,
}

#[async_trait]
pub trait PricingCatalog: Send + Sync {
    fn get_pricing(&self, model: &str) -> Option<PricingInfo> {
        self.get_pricing_for(model, None)
    }
    fn get_pricing_for(&self, model: &str, provider: Option<&str>) -> Option<PricingInfo>;
    async fn refresh(&self) -> Result<(), crate::error::XlateError> {
        Ok(())
    }
    fn last_updated(&self) -> Option<i64> {
        None
    }
}
