use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedRequest {
    pub model: String,
    #[serde(default = "default_group", skip_serializing_if = "is_default_group")]
    pub group: String,

    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stable_message_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_true")]
    pub stream: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_headers: BTreeMap<String, String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_idle_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_parts: Vec<ContentPart>,

    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning_content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning_signature: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning_signature_source: String,

    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_responses_reasoning_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_responses_reasoning_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_responses_reasoning_summary: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDescriptor>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContentPartType {
    Text,
    Image,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: ContentPartType,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContent {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mime_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCallDescriptor {
    pub id: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub index: usize,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunctionShape,

    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_responses_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_responses_call_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_responses_status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunctionShape {
    pub name: String,
    pub arguments: String,
}

impl Default for NormalizedRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            group: default_group(),
            messages: Vec::new(),
            stable_message_count: 0,
            tools: Vec::new(),
            max_tokens: None,
            stream: true,
            thinking: None,
            reasoning_effort: None,
            extra_params: None,
            custom_headers: Default::default(),
            cache_key: None,
            stream_idle_timeout_ms: None,
        }
    }
}

fn default_true() -> bool {
    true
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

fn default_group() -> String {
    "default".into()
}

fn is_default_group(v: &str) -> bool {
    v == "default"
}
