use serde::{Deserialize, Serialize};

use crate::error::XlateError;
use crate::event::ModelEvent;
use crate::plugin::{ApiFormat, InboundPlugin};
use crate::types::NormalizedRequest;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestMetadata {
    #[serde(default, rename = "format")]
    pub format_hint: Option<ApiFormat>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

pub fn extract_metadata(body: &serde_json::Value) -> RequestMetadata {
    body.get("_metadata")
        .and_then(|m| serde_json::from_value::<RequestMetadata>(m.clone()).ok())
        .unwrap_or_default()
}

pub fn detect_format(body: &serde_json::Value) -> ApiFormat {
    if body.get("contents").is_some() {
        return ApiFormat::Gemini;
    }
    if body.get("input").is_some() && body.get("messages").is_none() {
        return ApiFormat::OpenAiResponses;
    }
    if body.get("system").is_some_and(|s| s.is_array() || s.is_string())
        && body.get("messages").is_some()
        && body.get("provider").is_none()
    {
        return ApiFormat::AnthropicMessages;
    }
    if body.get("messages").is_some() && body.get("provider").is_none() {
        return ApiFormat::OpenAiChatCompletions;
    }
    ApiFormat::Normalized
}

pub fn resolve_format(metadata: &RequestMetadata, body: &serde_json::Value) -> ApiFormat {
    if let Some(hint) = metadata.format_hint {
        return hint;
    }
    detect_format(body)
}

pub struct BuiltinInboundPlugin;

impl BuiltinInboundPlugin {
    pub fn new() -> Self {
        Self
    }

    pub fn decode(
        &self,
        format: ApiFormat,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        match format {
            ApiFormat::Normalized => serde_json::from_value(body.clone())
                .map_err(|e| XlateError::InvalidRequest(format!("invalid normalized request: {e}"))),
            ApiFormat::OpenAiChatCompletions => self.decode_openai_chat(body),
            ApiFormat::OpenAiResponses => self.decode_openai_responses(body),
            ApiFormat::AnthropicMessages => self.decode_anthropic(body),
            ApiFormat::Gemini => Err(XlateError::InvalidRequest(
                "gemini format not yet supported".into(),
            )),
        }
    }

    fn decode_openai_chat(
        &self,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let messages: Vec<crate::types::Message> = body
            .get("messages")
            .cloned()
            .map(|m| serde_json::from_value(m).unwrap_or_default())
            .unwrap_or_default();

        let tools: Vec<serde_json::Value> = body
            .get("tools")
            .cloned()
            .map(|t| serde_json::from_value(t).unwrap_or_default())
            .unwrap_or_default();

        let max_tokens = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let mut extra_params = serde_json::Map::new();
        let passthrough_keys = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "logprobs",
            "top_logprobs",
            "response_format",
        ];
        for key in &passthrough_keys {
            if let Some(val) = body.get(*key) {
                extra_params.insert(key.to_string(), val.clone());
            }
        }

        let reasoning_effort = body
            .get("reasoning_effort")
            .or_else(|| body.get("reasoning").and_then(|r| r.get("effort")))
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(NormalizedRequest {
            model,
            messages,
            tools,
            max_tokens,
            stream,
            reasoning_effort,
            extra_params: if extra_params.is_empty() {
                None
            } else {
                Some(extra_params)
            },
            ..Default::default()
        })
    }

    fn decode_openai_responses(
        &self,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let input = body.get("input");
        let messages = if let Some(input_val) = input {
            if let Some(s) = input_val.as_str() {
                vec![crate::types::Message {
                    role: "user".into(),
                    content: s.to_string(),
                    ..Default::default()
                }]
            } else if let Some(arr) = input_val.as_array() {
                serde_json::from_value(serde_json::Value::Array(arr.clone()))
                    .unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let tools: Vec<serde_json::Value> = body
            .get("tools")
            .cloned()
            .map(|t| serde_json::from_value(t).unwrap_or_default())
            .unwrap_or_default();

        let stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(NormalizedRequest {
            model,
            messages,
            tools,
            stream,
            extra_params: {
                let mut m = serde_json::Map::new();
                m.insert("_endpoint".into(), serde_json::Value::String("/v1/responses".into()));
                Some(m)
            },
            ..Default::default()
        })
    }

    fn decode_anthropic(
        &self,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut messages: Vec<crate::types::Message> = Vec::new();

        if let Some(system) = body.get("system") {
            let system_text = if let Some(s) = system.as_str() {
                s.to_string()
            } else if let Some(arr) = system.as_array() {
                arr.iter()
                    .filter_map(|block| {
                        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                            block.get("text").and_then(|t| t.as_str()).map(String::from)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                String::new()
            };
            if !system_text.is_empty() {
                messages.push(crate::types::Message {
                    role: "system".into(),
                    content: system_text,
                    ..Default::default()
                });
            }
        }

        let raw_messages = body.get("messages").cloned().unwrap_or_default();
        let user_messages: Vec<crate::types::Message> =
            serde_json::from_value(raw_messages).unwrap_or_default();
        messages.extend(user_messages);

        let tools: Vec<serde_json::Value> = body
            .get("tools")
            .cloned()
            .map(|t| serde_json::from_value(t).unwrap_or_default())
            .unwrap_or_default();

        let max_tokens = body
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let thinking = body.get("thinking").map(|t| {
            let effort = t.get("type").and_then(|v| v.as_str()).map(String::from);
            let budget = t
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            crate::types::ThinkingConfig {
                effort,
                budget_tokens: budget,
            }
        });

        let mut extra_params = serde_json::Map::new();
        let passthrough_keys = ["temperature", "top_p", "top_k", "stop_sequences"];
        for key in &passthrough_keys {
            if let Some(val) = body.get(*key) {
                extra_params.insert(key.to_string(), val.clone());
            }
        }

        Ok(NormalizedRequest {
            model,
            messages,
            tools,
            max_tokens,
            stream,
            thinking,
            extra_params: if extra_params.is_empty() {
                None
            } else {
                Some(extra_params)
            },
            ..Default::default()
        })
    }
}

impl Default for BuiltinInboundPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::plugin::Plugin for BuiltinInboundPlugin {
    fn manifest(&self) -> crate::plugin::PluginManifest {
        crate::plugin::PluginManifest {
            id: "builtin".into(),
            name: "builtin".into(),
            version: "0.1.0".into(),
            kind: crate::plugin::PluginKind::Inbound,
            required_capabilities: vec![],
        }
    }
}

#[async_trait::async_trait]
impl InboundPlugin for BuiltinInboundPlugin {
    fn name(&self) -> &str {
        "builtin"
    }

    fn supported_formats(&self) -> &[ApiFormat] {
        &[
            ApiFormat::Normalized,
            ApiFormat::OpenAiChatCompletions,
            ApiFormat::OpenAiResponses,
            ApiFormat::AnthropicMessages,
        ]
    }

    fn decode_request(&self, body: &serde_json::Value) -> Result<NormalizedRequest, XlateError> {
        let format = detect_format(body);
        BuiltinInboundPlugin::decode(self, format, body)
    }

    fn decode(
        &self,
        format: ApiFormat,
        body: &serde_json::Value,
    ) -> Result<NormalizedRequest, XlateError> {
        BuiltinInboundPlugin::decode(self, format, body)
    }

    fn encode_event(&self, event: &ModelEvent, _format: ApiFormat) -> Result<serde_json::Value, XlateError> {
        serde_json::to_value(event)
            .map_err(|e| XlateError::Internal(format!("encode event: {e}")))
    }
}
