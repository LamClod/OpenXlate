use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelEvent {
    TextDelta {
        text: String,
        #[serde(flatten)]
        meta: EventMeta,
    },
    ThinkingDelta {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<String>,
        #[serde(flatten)]
        meta: EventMeta,
    },
    ThinkingCompleted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<i32>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        signature: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        signature_source: String,
        #[serde(flatten)]
        meta: EventMeta,
    },
    PartialToolCall {
        tool_call_id: String,
        name: String,
        #[serde(flatten)]
        meta: EventMeta,
    },
    ToolCallDelta {
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        args_text_delta: String,
        #[serde(flatten)]
        meta: EventMeta,
    },
    ToolLikeCompleted {
        tool_call_id: String,
        name: String,
        arguments_json: String,
        #[serde(flatten)]
        meta: EventMeta,
    },
    TurnFinished {
        #[serde(default, skip_serializing_if = "String::is_empty")]
        finish_reason: String,
        usage: Usage,
        #[serde(flatten)]
        meta: EventMeta,
    },
    ProviderError {
        error: crate::error::XlateErrorPayload,
        #[serde(flatten)]
        meta: EventMeta,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMeta {
    pub occurred_at_ms: i64,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_item_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_call_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i64>,
    #[serde(default)]
    pub estimated: bool,
}
