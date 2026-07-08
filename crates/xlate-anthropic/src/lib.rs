//! Anthropic-compatible outbound plugin.
//!
//! This crate uses standard, transparent request methods: no client identity
//! spoofing, no forged billing headers, no impersonation of other tools.
//!
//! Implements the full Anthropic Messages streaming API:
//! - Message normalization (system extraction, tool_result batching, thinking blocks)
//! - Prompt cache breakpoint algorithm
//! - SSE stream parsing (text, thinking, tool_use, signatures)
//! - Extra params / custom headers injection
//! - Idle timeout watchdog

mod cache;
mod messages;
mod stream;
mod types;

use async_trait::async_trait;
use xlate_core::{EventSink, NormalizedRequest, XlateError};

use crate::messages::build_system_blocks;
use crate::types::AnthropicTool;

const DEFAULT_USER_AGENT: &str = concat!("xlate-anthropic/", env!("CARGO_PKG_VERSION"));

/// Default max_tokens if none provided.
const DEFAULT_MAX_TOKENS: u32 = 65536;

pub struct AnthropicAdapter {
    client: reqwest::Client,
    default_max_tokens: u32,
    max_cache_breakpoints: usize,
    anthropic_version: String,
}

impl AnthropicAdapter {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(DEFAULT_USER_AGENT)
                .build()
                .unwrap_or_default(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            max_cache_breakpoints: 4,
            anthropic_version: "2023-06-01".into(),
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
            default_max_tokens: DEFAULT_MAX_TOKENS,
            max_cache_breakpoints: 4,
            anthropic_version: "2023-06-01".into(),
        }
    }

    pub fn with_settings(mut self, max_tokens: u32, max_breakpoints: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self.max_cache_breakpoints = max_breakpoints as usize;
        self
    }

    pub fn with_anthropic_version(mut self, version: &str) -> Self {
        self.anthropic_version = version.to_string();
        self
    }
}

impl Default for AnthropicAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicAdapter {
    async fn dispatch(
        &self,
        req: &NormalizedRequest,
        config: &xlate_core::ProviderConfig,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        // ------------------------------------------------------------------
        // Validate inputs
        // ------------------------------------------------------------------
        let base_url = config.base_url.trim().trim_end_matches('/');
        if base_url.is_empty() {
            return Err(XlateError::InvalidRequest(
                "anthropic base url is empty".into(),
            ));
        }
        let api_key = config.api_key.trim();
        if api_key.is_empty() {
            return Err(XlateError::InvalidRequest(
                "anthropic api key is empty".into(),
            ));
        }
        let model_id = req.model.trim();
        if model_id.is_empty() {
            return Err(XlateError::InvalidRequest(
                "anthropic model id is empty".into(),
            ));
        }

        let request_url = endpoint_url(base_url);

        // ------------------------------------------------------------------
        // Build thinking config
        // ------------------------------------------------------------------
        let thinking_config = build_thinking_config(req);
        let thinking_enabled = thinking_config.is_some()
            && !matches!(
                thinking_config.as_ref().and_then(|v| v.get("type")).and_then(|v| v.as_str()),
                Some("disabled")
            );

        // ------------------------------------------------------------------
        // Normalize messages
        // ------------------------------------------------------------------
        let stable_msg_count = messages::stable_provider_message_count(
            &req.messages,
            req.stable_message_count,
            thinking_enabled,
        );
        let normalized = messages::normalize_messages(&req.messages, thinking_enabled)?;

        // ------------------------------------------------------------------
        // Build tools
        // ------------------------------------------------------------------
        let tools = parse_tools(&req.tools)?;

        // ------------------------------------------------------------------
        // Build request body
        // ------------------------------------------------------------------
        let max_tokens = max_anthropic_tokens(req, config, self.default_max_tokens);
        let system_blocks = build_system_blocks(&normalized.system_parts);

        // Serialize messages to JSON values.
        let messages_json: Vec<serde_json::Value> = normalized
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).map_err(|e| XlateError::Internal(format!("serialize message: {e}"))))
            .collect::<Result<Vec<_>, _>>()?;

        let mut body = serde_json::json!({
            "model": model_id,
            "messages": messages_json,
            "stream": true,
            "max_tokens": max_tokens,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| XlateError::Internal(format!("serialize tools: {e}")))?;
        }
        if !system_blocks.is_empty() {
            body["system"] = serde_json::Value::Array(system_blocks);
        }
        if let Some(ref tc) = thinking_config {
            body["thinking"] = tc.clone();
        }

        // ------------------------------------------------------------------
        // Apply cache breakpoints
        // ------------------------------------------------------------------
        cache::apply_cache_breakpoints_with_limit(&mut body, stable_msg_count, self.max_cache_breakpoints);

        // ------------------------------------------------------------------
        // Apply extra params
        // ------------------------------------------------------------------
        // Merge extra params (request-level then provider-level)
        if let Some(ref extra) = req.extra_params {
            if let Some(obj) = body.as_object_mut() {
                for (key, value) in extra {
                    let name = key.trim();
                    if !name.is_empty() && !name.starts_with('_') {
                        obj.insert(name.to_string(), value.clone());
                    }
                }
            }
        }
        if let Some(ref extra) = config.extra_params {
            if let Some(obj) = body.as_object_mut() {
                for (key, value) in extra {
                    let name = key.trim();
                    if !name.is_empty() {
                        obj.insert(name.to_string(), value.clone());
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Serialize body
        // ------------------------------------------------------------------
        let payload = serde_json::to_vec(&body).map_err(|e| {
            XlateError::Internal(format!("failed to serialize request body: {e}"))
        })?;

        tracing::debug!(
            url = %request_url,
            model = %model_id,
            message_count = messages_json.len(),
            tool_count = tools.len(),
            "sending anthropic stream request"
        );

        // ------------------------------------------------------------------
        // Build and send HTTP request
        // ------------------------------------------------------------------
        let auth_token = extract_auth_token(api_key);
        let mut http_req = self
            .client
            .post(&request_url)
            .header("x-api-key", &auth_token)
            .header("Authorization", format!("Bearer {auth_token}"))
            .header(
                "anthropic-version",
                config
                    .anthropic_version
                    .as_deref()
                    .unwrap_or(&self.anthropic_version),
            )
            .header("content-type", "application/json");

        // Apply custom headers.
        for (key, value) in &req.custom_headers {
            let name = key.trim();
            if !name.is_empty() {
                http_req = http_req.header(name, value.trim());
            }
        }

        let resp = http_req
            .body(payload)
            .send()
            .await
            .map_err(|e| XlateError::Transport(format!("anthropic request failed: {e}")))?;

        // ------------------------------------------------------------------
        // Check HTTP status
        // ------------------------------------------------------------------
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(|v| format!(" retry-after={v}"));
            let body_text = resp.text().await.unwrap_or_default();
            let body_detail = if body_text.trim().is_empty() {
                String::new()
            } else {
                format!(": {body_text}")
            };
            return Err(XlateError::Provider {
                status: Some(status_code),
                message: format!(
                    "anthropic adapter: HTTP {status_code}{}{body_detail}",
                    retry_after.as_deref().unwrap_or("")
                ),
            });
        }

        // ------------------------------------------------------------------
        // Process SSE stream
        // ------------------------------------------------------------------
        stream::process_sse_stream(resp, model_id, req.stream_idle_timeout_ms, sink).await
    }
}

impl xlate_core::plugin::Plugin for AnthropicAdapter {
    fn manifest(&self) -> xlate_core::plugin::PluginManifest {
        xlate_core::plugin::PluginManifest {
            id: "anthropic".into(),
            name: "anthropic".into(),
            version: "0.1.0".into(),
            kind: xlate_core::plugin::PluginKind::Outbound,
            required_capabilities: vec![],
        }
    }
}

#[async_trait]
impl xlate_core::OutboundPlugin for AnthropicAdapter {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn stream(
        &self,
        request: &NormalizedRequest,
        config: &xlate_core::ProviderConfig,
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the Anthropic endpoint URL, appending `/v1/messages` if needed.
fn endpoint_url(base_url: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if base.ends_with("/v1/messages") || base.ends_with("/messages") {
        return base.to_string();
    }
    format!("{base}/v1/messages")
}

/// Extract the bare API token, stripping a leading "Bearer " if present.
fn extract_auth_token(api_key: &str) -> String {
    let token = api_key.trim();
    if token.len() >= 7 && token[..7].eq_ignore_ascii_case("Bearer ") {
        token[7..].trim().to_string()
    } else {
        token.to_string()
    }
}

/// Determine max_tokens for the request.
fn max_anthropic_tokens(
    req: &NormalizedRequest,
    config: &xlate_core::ProviderConfig,
    default: u32,
) -> u32 {
    if let Some(v) = config.max_tokens_override {
        if v > 0 {
            return v;
        }
    }
    if let Some(v) = req.max_tokens {
        if v > 0 {
            return v;
        }
    }
    default
}

/// Parse OpenAI-style tool definitions into Anthropic tool format.
fn parse_tools(raw_tools: &[serde_json::Value]) -> Result<Vec<AnthropicTool>, XlateError> {
    let mut tools = Vec::with_capacity(raw_tools.len());
    for raw in raw_tools {
        let func = raw
            .get("function")
            .ok_or_else(|| {
                XlateError::InvalidRequest(
                    "tool definition missing 'function' field".into(),
                )
            })?;
        let name = func
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let description = func
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let input_schema = func
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        tools.push(AnthropicTool {
            name,
            description,
            input_schema,
            cache_control: None,
        });
    }
    Ok(tools)
}

/// Build the `thinking` config object for the request body.
fn build_thinking_config(req: &NormalizedRequest) -> Option<serde_json::Value> {
    let thinking = req.thinking.as_ref()?;
    let effort = thinking
        .effort
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();

    match effort.as_str() {
        "disabled" | "disable" | "off" | "none" | "false" | "no" | "0" => {
            Some(serde_json::json!({ "type": "disabled" }))
        }
        "" => None,
        _ => Some(serde_json::json!({
            "type": "adaptive",
            "display": "summarized",
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_url() {
        assert_eq!(
            endpoint_url("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            endpoint_url("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            endpoint_url("https://proxy.example.com/messages"),
            "https://proxy.example.com/messages"
        );
        assert_eq!(
            endpoint_url("https://api.anthropic.com/"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_extract_auth_token() {
        assert_eq!(extract_auth_token("sk-abc123"), "sk-abc123");
        assert_eq!(extract_auth_token("Bearer sk-abc123"), "sk-abc123");
        assert_eq!(extract_auth_token("bearer sk-abc123"), "sk-abc123");
    }

    #[test]
    fn test_max_anthropic_tokens() {
        let mut req = NormalizedRequest {
            model: "claude-3".into(),
            ..Default::default()
        };
        let config = xlate_core::ProviderConfig {
            plugin: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            api_key: "test".into(),
            ..default_test_config()
        };
        assert_eq!(max_anthropic_tokens(&req, &config, DEFAULT_MAX_TOKENS), DEFAULT_MAX_TOKENS);

        req.max_tokens = Some(1024);
        assert_eq!(max_anthropic_tokens(&req, &config, DEFAULT_MAX_TOKENS), 1024);

        let config_override = xlate_core::ProviderConfig {
            max_tokens_override: Some(4096),
            ..config.clone()
        };
        assert_eq!(max_anthropic_tokens(&req, &config_override, DEFAULT_MAX_TOKENS), 4096);
    }

    fn default_test_config() -> xlate_core::ProviderConfig {
        xlate_core::ProviderConfig {
            plugin: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            api_key: "test".into(),
            upstream_model: None,
            endpoint: None,
            extra_params: None,
            extra_headers: Default::default(),
            anthropic_version: None,
            max_tokens_override: None,
        }
    }

    #[test]
    fn test_build_thinking_config() {
        let mut req = NormalizedRequest {
            model: "claude-3".into(),
            ..Default::default()
        };
        assert!(build_thinking_config(&req).is_none());

        req.thinking = Some(xlate_core::types::ThinkingConfig {
            effort: Some("disabled".into()),
            budget_tokens: None,
        });
        let cfg = build_thinking_config(&req).unwrap();
        assert_eq!(cfg["type"], "disabled");

        req.thinking = Some(xlate_core::types::ThinkingConfig {
            effort: Some("high".into()),
            budget_tokens: None,
        });
        let cfg = build_thinking_config(&req).unwrap();
        assert_eq!(cfg["type"], "adaptive");
    }

    #[test]
    fn test_parse_tools() {
        let raw = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                }
            }
        })];
        let tools = parse_tools(&raw).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "Get weather info");
    }
}
