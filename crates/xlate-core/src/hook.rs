use async_trait::async_trait;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::XlateError;
use crate::event::ModelEvent;
use crate::event::Usage;
use crate::provider::{ProviderConfig, RouteResult};
use crate::types::NormalizedRequest;

static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub type StreamId = u64;

pub fn next_stream_id() -> StreamId {
    STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookPhase {
    PreRoute,
    PreSend,
    OnEvent,
    OnError,
    PostComplete,
}

pub enum HookVerdict {
    Continue,
    Skip,
    Abort(XlateError),
    Rectify(RectifyAction),
}

pub enum RectifyAction {
    ModifyRequest,
    SwitchTarget,
    RetryAfter(Duration),
}

pub struct HookContext {
    pub stream_id: StreamId,
    pub created_at: Instant,
    pub request: NormalizedRequest,
    pub original_model: String,
    pub route: Option<RouteResult>,
    pub provider_config: Option<ProviderConfig>,
    pub metrics: StreamMetrics,
    pub extensions: Extensions,
}

impl HookContext {
    pub fn new(request: NormalizedRequest) -> Self {
        let original_model = request.model.clone();
        Self {
            stream_id: next_stream_id(),
            created_at: Instant::now(),
            original_model,
            request,
            route: None,
            provider_config: None,
            metrics: StreamMetrics::default(),
            extensions: Extensions::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct StreamMetrics {
    pub usage: Usage,
    pub cost: Option<CostBreakdown>,
    pub latency: Option<LatencyMetrics>,
    pub attempt: u32,
    pub max_attempts: u32,
    pub finish_reason: Option<String>,
    pub success: bool,
    pub ttft_ms: Option<f64>,
    pub total_ms: Option<f64>,
    pub provider_ms: Option<f64>,
    pub error: Option<crate::error::XlateError>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LatencyMetrics {
    pub total_ms: f64,
    pub ttft_ms: Option<f64>,
    pub provider_ms: Option<f64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CostBreakdown {
    pub input_cost: rust_decimal::Decimal,
    pub output_cost: rust_decimal::Decimal,
    pub cache_read_cost: rust_decimal::Decimal,
    pub cache_write_cost: rust_decimal::Decimal,
    pub reasoning_cost: rust_decimal::Decimal,
    pub total_cost: rust_decimal::Decimal,
    pub rate_multiplier: rust_decimal::Decimal,
    pub adjusted_cost: rust_decimal::Decimal,
    #[serde(default = "default_currency")]
    pub currency: String,
    #[serde(default)]
    pub pricing_source: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub track: Option<String>,
}

fn default_currency() -> String {
    "USD".into()
}

/// Type-map for hook-to-hook communication.
pub struct Extensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(val));
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref())
    }

    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut())
    }

    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast().ok())
            .map(|b| *b)
    }
}

impl Default for Extensions {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
pub trait Hook: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32;

    async fn pre_route(&self, _ctx: &mut HookContext) -> HookVerdict {
        HookVerdict::Continue
    }
    async fn pre_send(&self, _ctx: &mut HookContext) -> HookVerdict {
        HookVerdict::Continue
    }
    async fn on_event(&self, _ctx: &HookContext, _event: &mut ModelEvent) -> HookVerdict {
        HookVerdict::Continue
    }
    async fn on_error(&self, _ctx: &mut HookContext, _error: &XlateError) -> HookVerdict {
        HookVerdict::Continue
    }
    async fn post_complete(&self, _ctx: &mut HookContext) -> HookVerdict {
        HookVerdict::Continue
    }

    async fn shutdown(&self) {}

    fn snapshot(&self) -> Option<serde_json::Value> {
        None
    }
}

pub struct HooksFired(pub Vec<String>);

pub struct OutputCharCount(pub usize);

pub struct PendingRectify(pub RectifyAction);
