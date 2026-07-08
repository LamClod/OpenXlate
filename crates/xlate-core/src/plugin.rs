use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::capability::PluginCapabilities;
use crate::error::XlateError;
use crate::event::ModelEvent;
use crate::provider::ProviderConfig;
use crate::registry::ModelMeta;
use crate::types::NormalizedRequest;

pub type PluginId = String;

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin initialization failed: {0}")]
    Init(String),
    #[error("plugin error: {0}")]
    Other(String),
}

impl From<XlateError> for PluginError {
    fn from(e: XlateError) -> Self {
        PluginError::Other(e.to_string())
    }
}

pub trait Plugin: Send + Sync {
    fn manifest(&self) -> PluginManifest;
    fn init(&mut self, _caps: PluginCapabilities) -> Result<(), PluginError> {
        Ok(())
    }
    fn shutdown(&self) {}
}

#[async_trait]
pub trait EventSink: Send {
    async fn send(&mut self, event: ModelEvent) -> Result<(), XlateError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiFormat {
    #[serde(rename = "normalized")]
    Normalized,
    #[serde(rename = "openai-chat")]
    OpenAiChatCompletions,
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
    #[serde(rename = "gemini")]
    Gemini,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginKind {
    Inbound,
    Outbound,
    Service,
    Hook,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: PluginId,
    pub name: String,
    pub version: String,
    pub kind: PluginKind,
    #[serde(default)]
    pub required_capabilities: Vec<crate::capability::CapabilityType>,
}

#[async_trait]
pub trait OutboundPlugin: Plugin {
    fn name(&self) -> &str;

    fn supported_providers(&self) -> &[&str] {
        &[]
    }

    async fn stream(
        &self,
        request: &NormalizedRequest,
        config: &ProviderConfig,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError>;

    fn get_model_list(&self) -> Vec<ModelMeta> {
        vec![]
    }
}

#[async_trait]
pub trait InboundPlugin: Plugin {
    fn name(&self) -> &str;
    fn supported_formats(&self) -> &[ApiFormat];

    fn decode_request(
        &self,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError>;

    fn decode(
        &self,
        format: ApiFormat,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        let _ = format;
        self.decode_request(body)
    }

    fn encode_event(
        &self,
        event: &ModelEvent,
        format: ApiFormat,
    ) -> Result<serde_json::Value, XlateError>;
}

#[async_trait]
pub trait ServicePlugin: Plugin {
    fn name(&self) -> &str;

    async fn start(&self, caps: crate::capability::PluginCapabilities) -> Result<(), PluginError>;
    async fn stop(&self) -> Result<(), PluginError>;

    async fn health_check(&self) -> crate::supervisor::PluginStatus {
        crate::supervisor::PluginStatus::Running
    }
}
