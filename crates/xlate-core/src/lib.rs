pub mod capability;
pub mod config;
pub mod error;
pub mod event;
pub mod hook;
pub mod inbound;
pub mod kernel;
pub mod message;
pub mod plugin;
pub mod pricing;
pub mod provider;
pub mod registry;
pub mod router;
pub mod sanitize;
pub mod stats;
pub mod store;
pub mod supervisor;
pub mod think_tag;
pub mod types;

pub use capability::{Capability, CapabilityRights, CapabilityType, PluginCapabilities};
pub use config::KernelConfig;
pub use error::XlateError;
pub use event::ModelEvent;
pub use hook::{CostBreakdown, Hook, HookContext, HookPhase, HookVerdict, LatencyMetrics};
pub use inbound::{BuiltinInboundPlugin, RequestMetadata};
pub use kernel::{EventBus, Kernel, KernelBuilder, KernelStats, ModelUsageSnapshot, TotalCost, TotalTokens};
pub use message::{KernelEventPayload, KernelMessage};
pub use plugin::{ApiFormat, EventSink, InboundPlugin, OutboundPlugin, Plugin, PluginError, PluginId, PluginKind, PluginManifest, ServicePlugin};
pub use pricing::{PricingCatalog, PricingInfo};
pub use provider::ProviderConfig;
pub use registry::{ModelMeta, ModelRegistry, ModelType};
pub use router::{LatencyTracker, Router};
pub use stats::StatsAggregator;
pub use store::{Store, UsageWriter};
pub use supervisor::Supervisor;
pub use think_tag::ThinkTagParser;
pub use types::NormalizedRequest;

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
