use serde::{Deserialize, Serialize};

use crate::provider::{RouteStrategy, RouteTarget};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelConfig {
    #[serde(default)]
    pub kernel: KernelSettings,
    #[serde(default)]
    pub routes: Vec<RouteRuleConfig>,
    #[serde(default)]
    pub model_map: ModelMapConfig,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub billing: BillingConfig,
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
    #[serde(default)]
    pub session_affinity: SessionAffinityConfig,
    #[serde(default)]
    pub sanitize: SanitizeConfig,
    #[serde(default)]
    pub param_heal: ParamHealConfig,
    #[serde(default)]
    pub rectifier: RectifierConfig,
    #[serde(default)]
    pub media_sanitizer: MediaSanitizerConfig,
    #[serde(default)]
    pub trace: TraceConfig,
    #[serde(default)]
    pub alerts: AlertsConfig,
    #[serde(default)]
    pub store: StoreConfig,
    #[serde(default)]
    pub model_registry: ModelRegistryConfig,
    #[serde(default)]
    pub plugins: PluginsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSettings {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_streams: usize,
    #[serde(default = "default_max_failover")]
    pub max_failover_attempts: u32,
    #[serde(default = "default_buffer_size")]
    pub stream_buffer_size: usize,
    #[serde(default = "default_idle_timeout")]
    pub default_idle_timeout_ms: u64,
    #[serde(default = "default_backpressure_timeout")]
    pub backpressure_timeout_ms: u64,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_ms: u64,
}

fn default_max_concurrent() -> usize {
    64
}
fn default_max_failover() -> u32 {
    3
}
fn default_buffer_size() -> usize {
    64
}
fn default_idle_timeout() -> u64 {
    240_000
}
fn default_backpressure_timeout() -> u64 {
    30_000
}
fn default_shutdown_timeout() -> u64 {
    30_000
}

impl Default for KernelSettings {
    fn default() -> Self {
        Self {
            max_concurrent_streams: default_max_concurrent(),
            max_failover_attempts: default_max_failover(),
            stream_buffer_size: default_buffer_size(),
            default_idle_timeout_ms: default_idle_timeout(),
            backpressure_timeout_ms: default_backpressure_timeout(),
            shutdown_timeout_ms: default_shutdown_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRuleConfig {
    #[serde(default = "default_group")]
    pub group: String,
    pub model: String,
    #[serde(default)]
    pub strategy: RouteStrategy,
    pub targets: Vec<RouteTarget>,
    #[serde(default)]
    pub failover: FailoverConfig,
    #[serde(default)]
    pub patches: Vec<PatchOpConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchOpConfig {
    pub op: String,
    pub path: String,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub condition: Option<String>,
}

fn default_group() -> String {
    "default".into()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailoverConfig {
    #[serde(default)]
    pub trigger_statuses: Vec<u16>,
    #[serde(default = "default_cooldown")]
    pub cooldown_ms: u64,
}

fn default_cooldown() -> u64 {
    10_000
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMapConfig {
    #[serde(default)]
    pub rules: Vec<ModelMapRule>,
    #[serde(default)]
    pub chain_redirect: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMapRule {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_success_threshold")]
    pub success_threshold: u32,
    #[serde(default = "default_open_timeout")]
    pub open_timeout_ms: u64,
    #[serde(default = "default_granularity")]
    pub granularity: String,
}

fn default_granularity() -> String {
    "per-target".into()
}

fn default_failure_threshold() -> u32 {
    5
}
fn default_success_threshold() -> u32 {
    2
}
fn default_open_timeout() -> u64 {
    60_000
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: default_failure_threshold(),
            success_threshold: default_success_threshold(),
            open_timeout_ms: default_open_timeout(),
            granularity: default_granularity(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_billing_mode")]
    pub mode: String,
    #[serde(default = "default_rate_multiplier")]
    pub rate_multiplier: f64,
    #[serde(default)]
    pub pricing: PricingSourceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingSourceConfig {
    #[serde(default = "default_pricing_source")]
    pub source: String,
    #[serde(default = "default_pricing_refresh")]
    pub refresh_interval_s: u64,
    #[serde(default)]
    pub fallback_file: Option<String>,
}

fn default_pricing_source() -> String {
    "https://cch-plus.com/pricing/v1/models.json".into()
}
fn default_pricing_refresh() -> u64 {
    3600
}

impl Default for PricingSourceConfig {
    fn default() -> Self {
        Self {
            source: default_pricing_source(),
            refresh_interval_s: default_pricing_refresh(),
            fallback_file: None,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_billing_mode() -> String {
    "record".into()
}
fn default_rate_multiplier() -> f64 {
    1.0
}

impl Default for BillingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: default_billing_mode(),
            rate_multiplier: default_rate_multiplier(),
            pricing: PricingSourceConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rate limit config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_rate_limit_mode")]
    pub mode: String,
    #[serde(default = "default_rpm")]
    pub rpm: u32,
    #[serde(default = "default_tpm")]
    pub tpm: u64,
}

fn default_rate_limit_mode() -> String { "record".into() }
fn default_rpm() -> u32 { 600 }
fn default_tpm() -> u64 { 1_000_000 }

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: default_rate_limit_mode(),
            rpm: default_rpm(),
            tpm: default_tpm(),
        }
    }
}

// ---------------------------------------------------------------------------
// Session affinity config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAffinityConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_affinity_ttl")]
    pub ttl_s: u64,
    #[serde(default = "default_true")]
    pub lazy_binding: bool,
}

fn default_affinity_ttl() -> u64 { 300 }

impl Default for SessionAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl_s: default_affinity_ttl(),
            lazy_binding: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Sanitize config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizeConfig {
    #[serde(default = "default_true")]
    pub remove_empty_assistants: bool,
    #[serde(default = "default_true")]
    pub merge_adjacent_tool_calls: bool,
    #[serde(default = "default_true")]
    pub trim_dangling_tool_calls: bool,
    #[serde(default = "default_true")]
    pub trim_trailing_prefill: bool,
}

impl Default for SanitizeConfig {
    fn default() -> Self {
        Self {
            remove_empty_assistants: true,
            merge_adjacent_tool_calls: true,
            trim_dangling_tool_calls: true,
            trim_trailing_prefill: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Param heal config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamHealConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_heal_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_true")]
    pub persist_cache: bool,
}

fn default_max_heal_attempts() -> u32 { 3 }

impl Default for ParamHealConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: default_max_heal_attempts(),
            persist_cache: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Rectifier config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RectifierConfig {
    #[serde(default = "default_true")]
    pub thinking_signature: bool,
    #[serde(default = "default_true")]
    pub thinking_budget: bool,
    #[serde(default = "default_true")]
    pub thinking_effort_conflict: bool,
    #[serde(default = "default_true")]
    pub context_length: bool,
    #[serde(default = "default_true")]
    pub retry_after: bool,
}

impl Default for RectifierConfig {
    fn default() -> Self {
        Self {
            thinking_signature: true,
            thinking_budget: true,
            thinking_effort_conflict: true,
            context_length: true,
            retry_after: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Media sanitizer config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaSanitizerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_media_strategy")]
    pub strategy: String,
}

fn default_media_strategy() -> String { "both".into() }

impl Default for MediaSanitizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: default_media_strategy(),
        }
    }
}

// ---------------------------------------------------------------------------
// Trace config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    #[serde(default = "default_ring_buffer_size")]
    pub ring_buffer_size: usize,
    #[serde(default)]
    pub persist_to_store: bool,
    #[serde(default = "default_true")]
    pub capture_request_body: bool,
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
}

fn default_ring_buffer_size() -> usize { 500 }
fn default_max_body_size() -> usize { 65536 }

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            ring_buffer_size: default_ring_buffer_size(),
            persist_to_store: false,
            capture_request_body: true,
            max_body_size: default_max_body_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// Alerts config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlertsConfig {
    #[serde(default)]
    pub rules: Vec<AlertRuleConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRuleConfig {
    pub name: String,
    pub metric: String,
    #[serde(default = "default_gt")]
    pub operator: String,
    pub threshold: f64,
    #[serde(default = "default_alert_window")]
    pub window_minutes: u32,
    #[serde(default = "default_alert_cooldown")]
    pub cooldown_minutes: u32,
}

fn default_gt() -> String { ">".into() }
fn default_alert_window() -> u32 { 5 }
fn default_alert_cooldown() -> u32 { 30 }

// ---------------------------------------------------------------------------
// Store config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    #[serde(default = "default_store_backend")]
    pub backend: String,
    #[serde(default = "default_sqlite_path")]
    pub sqlite_path: String,
    #[serde(default = "default_usage_batch_size")]
    pub usage_batch_size: usize,
    #[serde(default = "default_usage_flush_interval")]
    pub usage_flush_interval_ms: u64,
    #[serde(default = "default_stats_aggregation_interval")]
    pub stats_aggregation_interval_s: u64,
}

fn default_store_backend() -> String { "sqlite".into() }
fn default_sqlite_path() -> String { "~/.openxlate/data.db".into() }
fn default_usage_batch_size() -> usize { 100 }
fn default_usage_flush_interval() -> u64 { 5000 }
fn default_stats_aggregation_interval() -> u64 { 3600 }

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            backend: default_store_backend(),
            sqlite_path: default_sqlite_path(),
            usage_batch_size: default_usage_batch_size(),
            usage_flush_interval_ms: default_usage_flush_interval(),
            stats_aggregation_interval_s: default_stats_aggregation_interval(),
        }
    }
}

// ---------------------------------------------------------------------------
// Model registry config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistryConfig {
    #[serde(default)]
    pub overrides: std::collections::HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Outbound plugin configs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginsConfig {
    #[serde(default)]
    pub outbound: OutboundPluginsConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutboundPluginsConfig {
    #[serde(default)]
    pub openai: OutboundPluginSettings,
    #[serde(default)]
    pub anthropic: AnthropicPluginSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundPluginSettings {
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    #[serde(default = "default_pool_size")]
    pub connection_pool_size: usize,
    #[serde(default = "default_connection_idle_timeout")]
    pub idle_connection_timeout_s: u64,
}

fn default_user_agent() -> String {
    "xlate-openai/0.1.0".into()
}
fn default_pool_size() -> usize {
    16
}
fn default_connection_idle_timeout() -> u64 {
    90
}

impl Default for OutboundPluginSettings {
    fn default() -> Self {
        Self {
            user_agent: default_user_agent(),
            connection_pool_size: default_pool_size(),
            idle_connection_timeout_s: 90,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicPluginSettings {
    #[serde(default = "default_anthropic_user_agent")]
    pub user_agent: String,
    #[serde(default = "default_anthropic_max_tokens")]
    pub default_max_tokens: u32,
    #[serde(default = "default_anthropic_version")]
    pub anthropic_version: String,
    #[serde(default = "default_pool_size")]
    pub connection_pool_size: usize,
    #[serde(default = "default_anthropic_idle_timeout")]
    pub idle_connection_timeout_s: u64,
    #[serde(default = "default_max_cache_breakpoints")]
    pub max_cache_breakpoints: u32,
}

fn default_anthropic_user_agent() -> String {
    "xlate-anthropic/0.1.0".into()
}
fn default_anthropic_max_tokens() -> u32 {
    65536
}
fn default_anthropic_version() -> String {
    "2023-06-01".into()
}
fn default_anthropic_idle_timeout() -> u64 {
    90
}
fn default_max_cache_breakpoints() -> u32 {
    4
}

impl Default for AnthropicPluginSettings {
    fn default() -> Self {
        Self {
            user_agent: default_anthropic_user_agent(),
            default_max_tokens: default_anthropic_max_tokens(),
            anthropic_version: default_anthropic_version(),
            connection_pool_size: default_pool_size(),
            idle_connection_timeout_s: default_anthropic_idle_timeout(),
            max_cache_breakpoints: default_max_cache_breakpoints(),
        }
    }
}
