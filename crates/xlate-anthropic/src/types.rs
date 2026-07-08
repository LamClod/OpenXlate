//! Internal types for the Anthropic Messages API request/response wire format.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request-side types
// ---------------------------------------------------------------------------

/// A single message in the Anthropic Messages API format.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicMessage {
    pub role: String,
    pub content: Vec<serde_json::Value>,
}

/// A tool definition in the Anthropic Messages API format.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
}

pub(crate) fn ephemeral_cache_control() -> CacheControl {
    CacheControl {
        kind: "ephemeral".to_string(),
    }
}

// ---------------------------------------------------------------------------
// SSE response-side types
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicUsage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ContentBlock {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub text: String,
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicEventPayload {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub content_block: Option<ContentBlock>,
    #[serde(default)]
    pub message: Option<AnthropicMessageMeta>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    #[serde(default)]
    pub delta: Option<AnthropicDelta>,
    #[serde(default)]
    pub error: Option<AnthropicError>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicMessageMeta {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicDelta {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub thinking: String,
    #[serde(default)]
    pub partial_json: String,
    #[serde(default)]
    pub signature: String,
    #[serde(default)]
    pub stop_reason: String,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicError {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
}

// ---------------------------------------------------------------------------
// Tool accumulator for streaming tool_use blocks
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct ToolAccumulator {
    pub call_id: String,
    pub name: String,
    pub args: String,
}

impl ToolAccumulator {
    pub fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            args: String::new(),
        }
    }
}
