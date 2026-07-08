//! OpenAI-compatible outbound plugin.
//!
//! This crate uses standard, transparent request methods: no client identity
//! spoofing, no forged billing headers, no impersonation of other tools.
//!
//! ## Implemented
//!
//! - **Chat Completions** streaming (`/v1/chat/completions` and compatible
//!   endpoints including domestic relay gateways like Z.AI /v4)
//! - Endpoint URL auto-detection and version deduplication
//! - Message normalization (content_parts, tool_calls, reasoning_content)
//! - SSE stream parsing with incremental tool call accumulation
//! - `<think>...</think>` tag splitting for providers that embed reasoning in content
//! - Provider-specific thinking disable parameters (DeepSeek, Qwen, GLM, GPT-5.1+)
//! - Extra params injection and custom headers
//! - Stream idle timeout watchdog
//! - Usage extraction with `Option` semantics for cache fields
//!
//! - **Responses API** streaming (`/v1/responses`) — see `responses.rs`.

mod chat_completions;
pub mod endpoint;
mod idle_watchdog;
pub mod messages;
mod responses;
mod responses_messages;
pub mod think_tag;
pub mod thinking_disable;

use async_trait::async_trait;
use xlate_core::{EventSink, NormalizedRequest, ProviderConfig, XlateError};

use crate::endpoint::endpoint_shape;

const DEFAULT_USER_AGENT: &str = concat!("xlate-openai/", env!("CARGO_PKG_VERSION"));

pub struct OpenAiAdapter {
    client: reqwest::Client,
}

impl OpenAiAdapter {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(DEFAULT_USER_AGENT)
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn with_pool(pool_size: usize, idle_timeout_s: u64) -> Self {
        Self::with_pool_ua(pool_size, idle_timeout_s, DEFAULT_USER_AGENT)
    }

    pub fn with_pool_ua(pool_size: usize, idle_timeout_s: u64, user_agent: &str) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(user_agent)
                .pool_max_idle_per_host(pool_size)
                .pool_idle_timeout(std::time::Duration::from_secs(idle_timeout_s))
                .build()
                .unwrap_or_default(),
        }
    }
}

impl Default for OpenAiAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiAdapter {
    async fn dispatch(
        &self,
        req: &NormalizedRequest,
        config: &ProviderConfig,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let endpoint_hint = config
            .endpoint
            .as_deref()
            .or_else(|| req.extra_params.as_ref()?.get("_endpoint")?.as_str())
            .unwrap_or("");
        let resolved = endpoint::resolve_endpoint(&config.base_url, endpoint_hint);

        let shape = match &resolved {
            Some(ep) => endpoint_shape(ep),
            None => "chat/completions",
        };

        match shape {
            "responses" => {
                tracing::debug!("routing to Responses API");
                responses::stream_responses(&self.client, req, config, sink).await
            }
            _ => {
                tracing::debug!("routing to Chat Completions API");
                chat_completions::stream_chat_completions(&self.client, req, config, sink).await
            }
        }
    }
}

impl xlate_core::plugin::Plugin for OpenAiAdapter {
    fn manifest(&self) -> xlate_core::plugin::PluginManifest {
        xlate_core::plugin::PluginManifest {
            id: "openai".into(),
            name: "openai".into(),
            version: "0.1.0".into(),
            kind: xlate_core::plugin::PluginKind::Outbound,
            required_capabilities: vec![],
        }
    }
}

#[async_trait]
impl xlate_core::OutboundPlugin for OpenAiAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    async fn stream(
        &self,
        request: &NormalizedRequest,
        config: &ProviderConfig,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let mut req = request.clone();
        if let Some(model) = &config.upstream_model {
            req.model = model.clone();
        }
        for (k, v) in &config.extra_headers {
            req.custom_headers.entry(k.clone()).or_insert_with(|| v.clone());
        }
        self.dispatch(&req, config, sink).await
    }
}
