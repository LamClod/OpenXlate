use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type TargetId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub plugin: String,
    pub base_url: String,
    pub api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_override: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTarget {
    pub id: TargetId,
    pub plugin: String,
    pub config: ProviderConfig,
    #[serde(default = "default_priority")]
    pub priority: u32,
    #[serde(default = "default_weight")]
    pub weight: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_priority() -> u32 {
    1
}
fn default_weight() -> u32 {
    100
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub target: RouteTarget,
    pub alternatives: Vec<RouteTarget>,
    pub failover: crate::config::FailoverConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteStrategy {
    PriorityWeighted,
    RoundRobin,
    LeastLatency,
    CostOptimized,
}

impl Default for RouteStrategy {
    fn default() -> Self {
        Self::PriorityWeighted
    }
}
