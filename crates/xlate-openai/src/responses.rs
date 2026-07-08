//! Responses API streaming implementation.
//!
//! Sends a POST to the Responses API endpoint, reads the SSE stream line by
//! line, and emits `ModelEvent`s through an `EventSink`.
//!
//! Ported from the original Go implementation's `streamResponses`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use futures_util::StreamExt;
use serde::Deserialize;

use xlate_core::event::{EventMeta, ModelEvent, Usage};
use xlate_core::types::NormalizedRequest;
use xlate_core::{EventSink, XlateError, now_ms};

use crate::endpoint;
use crate::idle_watchdog::IdleWatchdog;
use crate::responses_messages;
use crate::think_tag::{ContentPartKind, ThinkTagParser};
use crate::thinking_disable;

// ---------------------------------------------------------------------------
// SSE event types for Responses API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ResponsesStreamEvent {
    #[serde(rename = "type", default)]
    event_type: String,
    #[serde(default)]
    request_id: String,
    #[serde(default)]
    delta: String,
    #[serde(default)]
    arguments: String,
    #[serde(default)]
    output_index: usize,
    #[serde(default)]
    item_id: String,
    #[serde(default)]
    item: Option<ResponsesOutputItem>,
    #[serde(default)]
    response: Option<ResponsesResponse>,
    #[serde(default)]
    error: Option<ResponsesError>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutputItem {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    call_id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
    #[serde(default)]
    encrypted_content: String,
    #[serde(default)]
    summary: Option<serde_json::Value>,
    #[serde(default)]
    content: Vec<ResponsesOutputContent>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutputContent {
    #[serde(rename = "type", default)]
    content_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
    #[serde(default)]
    output_text: String,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    error: Option<ResponsesError>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: i64,
    #[serde(default)]
    output_tokens: i64,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesError {
    #[serde(default)]
    message: String,
    #[serde(rename = "type", default)]
    error_type: String,
    #[serde(default)]
    code: String,
}

// ---------------------------------------------------------------------------
// Tool accumulator
// ---------------------------------------------------------------------------

struct ToolAccumulator {
    call_id: String,
    name: String,
    args: String,
    provider_item_id: String,
    provider_call_id: String,
    provider_status: String,
}

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

struct StreamState {
    current_model: String,
    tools: HashMap<String, ToolAccumulator>,
    completed_tools: HashSet<String>,
    usage: Usage,
    usage_present: bool,
    cache_read_present: bool,
    finish_reason: String,
    turn_finished_pending: bool,
    emitted_tool_invocation: bool,
    emitted_text: bool,
    thinking_started: Option<Instant>,
    thinking_active: bool,
    emitted_reasoning_signature: String,
    think_parser: ThinkTagParser,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            current_model: model,
            tools: HashMap::new(),
            completed_tools: HashSet::new(),
            usage: Usage::default(),
            usage_present: false,
            cache_read_present: false,
            finish_reason: String::new(),
            turn_finished_pending: false,
            emitted_tool_invocation: false,
            emitted_text: false,
            thinking_started: None,
            thinking_active: false,
            emitted_reasoning_signature: String::new(),
            think_parser: ThinkTagParser::new(),
        }
    }

    fn meta(&self) -> EventMeta {
        EventMeta {
            occurred_at_ms: now_ms(),
            provider: "openai".into(),
            model: self.current_model.clone(),
            provider_item_id: String::new(),
            provider_status: String::new(),
            provider_summary: None,
            provider_call_id: String::new(),
        }
    }

    fn meta_with_provider_fields(
        &self,
        item_id: &str,
        status: &str,
        summary: Option<serde_json::Value>,
        call_id: &str,
    ) -> EventMeta {
        EventMeta {
            occurred_at_ms: now_ms(),
            provider: "openai".into(),
            model: self.current_model.clone(),
            provider_item_id: item_id.trim().to_string(),
            provider_status: status.trim().to_string(),
            provider_summary: summary,
            provider_call_id: call_id.trim().to_string(),
        }
    }

    fn apply_usage(&mut self, usage: &ResponsesUsage) {
        self.usage_present = true;
        let input_tokens = usage.input_tokens.max(0);
        let mut cached_tokens: i64 = 0;
        if let Some(ref details) = usage.input_tokens_details {
            self.cache_read_present = true;
            cached_tokens = details.cached_tokens.max(0);
        }
        if cached_tokens > input_tokens {
            cached_tokens = input_tokens;
        }
        self.usage.input_tokens = Some(input_tokens - cached_tokens);
        self.usage.output_tokens = Some(usage.output_tokens.max(0));
        self.usage.cache_read_tokens = Some(cached_tokens);
        self.usage.cache_write_tokens = Some(0);
    }

    fn build_turn_finished_usage(&self) -> Usage {
        if !self.usage_present {
            return Usage::default();
        }
        let mut u = self.usage.clone();
        if !self.cache_read_present {
            u.cache_read_tokens = None;
        }
        u
    }

    fn effective_finish_reason(&self) -> String {
        let reason = self.finish_reason.trim();
        if self.emitted_tool_invocation && (reason.is_empty() || reason == "completed") {
            "tool_calls".to_string()
        } else {
            reason.to_string()
        }
    }

    fn tool_key(&self, item_id: &str, output_index: usize) -> String {
        let trimmed = item_id.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            format!("output:{output_index}")
        }
    }
}

// ---------------------------------------------------------------------------
// Event emitters
// ---------------------------------------------------------------------------

async fn emit_text_delta(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    watchdog: &IdleWatchdog,
    text: &str,
) -> Result<(), XlateError> {
    if text.is_empty() {
        return Ok(());
    }
    watchdog.mark_effective_content();
    flush_thinking_completed(sink, state).await?;
    state.emitted_text = true;
    sink.send(ModelEvent::TextDelta {
        text: text.to_string(),
        meta: state.meta(),
    })
    .await
}

async fn emit_thinking_delta(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    watchdog: &IdleWatchdog,
    reasoning: &str,
) -> Result<(), XlateError> {
    if reasoning.is_empty() {
        return Ok(());
    }
    watchdog.mark_effective_content();
    if !state.thinking_active {
        state.thinking_started = Some(Instant::now());
        state.thinking_active = true;
    }
    sink.send(ModelEvent::ThinkingDelta {
        text: reasoning.to_string(),
        style: Some("default".to_string()),
        meta: state.meta(),
    })
    .await
}

async fn flush_thinking_completed(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
) -> Result<(), XlateError> {
    if !state.thinking_active {
        return Ok(());
    }
    let duration = state
        .thinking_started
        .map(|start| start.elapsed().as_millis() as i32)
        .unwrap_or(0)
        .max(0);
    state.thinking_active = false;
    state.thinking_started = None;
    sink.send(ModelEvent::ThinkingCompleted {
        duration_ms: Some(duration),
        signature: String::new(),
        signature_source: String::new(),
        meta: state.meta(),
    })
    .await
}

async fn emit_tagged_content_parts(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    watchdog: &IdleWatchdog,
    parts: Vec<crate::think_tag::ContentPart>,
) -> Result<(), XlateError> {
    for part in parts {
        match part.kind {
            ContentPartKind::Text => {
                emit_text_delta(sink, state, watchdog, &part.text).await?;
            }
            ContentPartKind::Reasoning => {
                emit_thinking_delta(sink, state, watchdog, &part.text).await?;
            }
            ContentPartKind::ThinkingCompleted => {
                flush_thinking_completed(sink, state).await?;
            }
        }
    }
    Ok(())
}

async fn flush_tagged_content_tail(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    watchdog: &IdleWatchdog,
) -> Result<(), XlateError> {
    let parts = state.think_parser.flush();
    emit_tagged_content_parts(sink, state, watchdog, parts).await
}

async fn emit_reasoning_signature(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    signature: &str,
    provider_item_id: &str,
    provider_status: &str,
    provider_summary: Option<serde_json::Value>,
) -> Result<(), XlateError> {
    let trimmed = signature.trim();
    if trimmed.is_empty() || trimmed == state.emitted_reasoning_signature {
        return Ok(());
    }

    let duration = if state.thinking_active {
        let d = state
            .thinking_started
            .map(|start| start.elapsed().as_millis() as i32)
            .unwrap_or(0)
            .max(0);
        state.thinking_active = false;
        state.thinking_started = None;
        d
    } else {
        0
    };

    state.emitted_reasoning_signature = trimmed.to_string();

    sink.send(ModelEvent::ThinkingCompleted {
        duration_ms: Some(duration),
        signature: trimmed.to_string(),
        signature_source: "openai_responses".to_string(),
        meta: state.meta_with_provider_fields(
            provider_item_id,
            provider_status,
            provider_summary,
            "",
        ),
    })
    .await
}

async fn complete_tool(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    key: &str,
    acc: &ToolAccumulator,
) -> Result<(), XlateError> {
    // Dedup by key and call_id
    let completion_key = if !key.trim().is_empty() {
        key.trim().to_string()
    } else if !acc.call_id.trim().is_empty() {
        acc.call_id.trim().to_string()
    } else {
        format!("{}:{}", acc.name, acc.args)
    };

    if state.completed_tools.contains(&completion_key) {
        return Ok(());
    }
    if !acc.call_id.trim().is_empty()
        && state
            .completed_tools
            .contains(acc.call_id.trim())
    {
        return Ok(());
    }

    state.completed_tools.insert(completion_key);
    if !acc.call_id.trim().is_empty() {
        state
            .completed_tools
            .insert(acc.call_id.trim().to_string());
    }

    state.emitted_tool_invocation = true;

    let meta = state.meta_with_provider_fields(
        &acc.provider_item_id,
        &acc.provider_status,
        None,
        &acc.provider_call_id,
    );

    sink.send(ModelEvent::ToolLikeCompleted {
        tool_call_id: acc.call_id.trim().to_string(),
        name: acc.name.trim().to_string(),
        arguments_json: acc.args.clone(),
        meta,
    })
    .await?;

    state.emitted_tool_invocation = true;

    Ok(())
}

async fn flush_turn_finished(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
) -> Result<(), XlateError> {
    if !state.turn_finished_pending {
        return Ok(());
    }
    state.turn_finished_pending = false;
    let usage = state.build_turn_finished_usage();
    let finish_reason = state.effective_finish_reason();
    sink.send(ModelEvent::TurnFinished {
        finish_reason,
        usage,
        meta: state.meta(),
    })
    .await
}

// ---------------------------------------------------------------------------
// Output item handlers
// ---------------------------------------------------------------------------

fn apply_function_call_item_fields(
    acc: &mut ToolAccumulator,
    item: &ResponsesOutputItem,
) {
    if !item.id.trim().is_empty() {
        acc.provider_item_id = item.id.trim().to_string();
    }
    if !item.status.trim().is_empty() {
        acc.provider_status = item.status.trim().to_string();
    }
    if !item.call_id.trim().is_empty() {
        acc.provider_call_id = item.call_id.trim().to_string();
        acc.call_id = item.call_id.trim().to_string();
    } else if !item.id.trim().is_empty() && acc.call_id.is_empty() {
        acc.call_id = item.id.trim().to_string();
    }
    if !item.name.trim().is_empty() {
        acc.name = item.name.trim().to_string();
    }
    // If arguments are present and accumulator is empty, seed it
    if !item.arguments.is_empty() && acc.args.is_empty() {
        acc.args.push_str(&item.arguments);
    }
}

async fn apply_output_item(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
    watchdog: &IdleWatchdog,
    item: &ResponsesOutputItem,
    output_index: usize,
    complete: bool,
) -> Result<(), XlateError> {
    match item.item_type.trim() {
        "reasoning" => {
            emit_reasoning_signature(
                sink,
                state,
                &item.encrypted_content,
                &item.id,
                &item.status,
                item.summary.clone(),
            )
            .await
        }
        "function_call" => {
            watchdog.mark_effective_content();
            let key_id = if !item.id.trim().is_empty() {
                item.id.trim().to_string()
            } else {
                item.call_id.trim().to_string()
            };
            let key = state.tool_key(&key_id, output_index);

            if !state.tools.contains_key(&key) {
                state.tools.insert(
                    key.clone(),
                    ToolAccumulator {
                        call_id: String::new(),
                        name: String::new(),
                        args: String::new(),
                        provider_item_id: String::new(),
                        provider_call_id: String::new(),
                        provider_status: String::new(),
                    },
                );
            }

            // Apply fields to accumulator
            {
                let acc = state.tools.get_mut(&key).unwrap();
                apply_function_call_item_fields(acc, item);
            }

            // Emit PartialToolCall when first seen (not complete)
            if !complete {
                let acc = state.tools.get(&key).unwrap();
                if !acc.call_id.trim().is_empty() && !acc.name.trim().is_empty() {
                    let call_id = acc.call_id.trim().to_string();
                    let name = acc.name.trim().to_string();
                    let p_item_id = acc.provider_item_id.clone();
                    let p_status = acc.provider_status.clone();
                    let p_call_id = acc.provider_call_id.clone();
                    let meta = state.meta_with_provider_fields(
                        &p_item_id,
                        &p_status,
                        None,
                        &p_call_id,
                    );
                    sink.send(ModelEvent::PartialToolCall {
                        tool_call_id: call_id,
                        name,
                        meta,
                    })
                    .await?;
                }
            }

            if complete {
                let acc = state.tools.remove(&key).unwrap();
                complete_tool(sink, state, &key, &acc).await?;
            }
            Ok(())
        }
        "image_generation_call" => {
            // Image generation: safely skip, just log
            tracing::debug!(
                item_id = %item.id.trim(),
                output_index = output_index,
                complete = complete,
                "skipping image_generation_call output item (not implemented)"
            );
            Ok(())
        }
        other => {
            tracing::debug!(item_type = %other, "ignoring unknown output item type");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Error extraction
// ---------------------------------------------------------------------------

fn error_from_event(event: &ResponsesStreamEvent) -> XlateError {
    if let Some(ref err) = event.error {
        if !err.message.trim().is_empty() {
            let details = stream_error_details(&err.error_type, &err.code, &event.request_id);
            return XlateError::Provider {
                status: None,
                message: format!(
                    "openai responses stream error {}: {}",
                    details,
                    err.message.trim()
                ),
            };
        }
    }
    if let Some(ref resp) = event.response {
        if let Some(ref err) = resp.error {
            if !err.message.trim().is_empty() {
                let details =
                    stream_error_details(&err.error_type, &err.code, &event.request_id);
                return XlateError::Provider {
                    status: None,
                    message: format!(
                        "openai responses stream error {}: {}",
                        details,
                        err.message.trim()
                    ),
                };
            }
        }
    }
    XlateError::Provider {
        status: None,
        message: "openai responses stream failed".into(),
    }
}

fn stream_error_details(error_type: &str, code: &str, request_id: &str) -> String {
    let mut parts = Vec::new();
    if !error_type.trim().is_empty() {
        parts.push(format!("type={}", error_type.trim()));
    }
    if !code.trim().is_empty() {
        parts.push(format!("code={}", code.trim()));
    }
    if !request_id.trim().is_empty() {
        parts.push(format!("request_id={}", request_id.trim()));
    }
    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Main stream function
// ---------------------------------------------------------------------------

pub(crate) async fn stream_responses(
    client: &reqwest::Client,
    req: &NormalizedRequest,
    config: &xlate_core::ProviderConfig,
    sink: &mut dyn EventSink,
) -> Result<(), XlateError> {
    let base_url = config.base_url.trim().trim_end_matches('/');
    if base_url.is_empty() {
        return Err(XlateError::InvalidRequest("openai base url is empty".into()));
    }
    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(XlateError::InvalidRequest("openai api key is empty".into()));
    }
    let model_id = req.model.trim();
    if model_id.is_empty() {
        return Err(XlateError::InvalidRequest("openai model id is empty".into()));
    }

    let endpoint_hint = config
        .endpoint
        .as_deref()
        .or_else(|| req.extra_params.as_ref()?.get("_endpoint")?.as_str())
        .unwrap_or("");
    let resolved_endpoint = endpoint::resolve_endpoint(base_url, endpoint_hint)
        .ok_or_else(|| {
            XlateError::InvalidRequest(format!(
                "openai endpoint is unsupported: {endpoint_hint}"
            ))
        })?;

    // Determine thinking state
    let thinking_effort = req
        .thinking
        .as_ref()
        .and_then(|t| t.effort.as_deref())
        .unwrap_or("");

    // Normalize messages into Responses API format
    let (instructions, input) = responses_messages::normalize_responses_input(&req.messages)?;

    // Build request body
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model_id.to_string()));

    if !instructions.is_empty() {
        body.insert(
            "instructions".into(),
            serde_json::Value::String(instructions),
        );
    }
    body.insert("input".into(), serde_json::Value::Array(input));
    body.insert("stream".into(), serde_json::Value::Bool(true));
    body.insert("store".into(), serde_json::Value::Bool(false));

    if let Some(max_tokens) = req.max_tokens {
        body.insert(
            "max_output_tokens".into(),
            serde_json::Value::Number(max_tokens.into()),
        );
    }

    // Prompt cache key
    if let Some(ref cache_key) = req.cache_key {
        if model_id.to_lowercase().contains("gpt") && !cache_key.trim().is_empty() {
            body.insert(
                "prompt_cache_key".into(),
                serde_json::Value::String(cache_key.clone()),
            );
        }
    }

    // Tools
    if !req.tools.is_empty() {
        let tools = responses_messages::normalize_responses_tools(&req.tools)?;
        body.insert("tools".into(), serde_json::Value::Array(
            tools.into_iter().collect(),
        ));
    }

    // Reasoning effort
    let normalized_effort = thinking_disable::normalize_thinking_effort(thinking_effort);
    if !thinking_effort.is_empty() && normalized_effort != "disabled" {
        body.insert(
            "reasoning".into(),
            serde_json::json!({"effort": thinking_effort}),
        );
        body.insert(
            "include".into(),
            serde_json::json!(["reasoning.encrypted_content"]),
        );
    }

    // Apply thinking disable
    thinking_disable::apply_thinking_disable(
        &mut body,
        thinking_effort,
        base_url,
        model_id,
        &resolved_endpoint,
    );

    // Merge extra params (request-level then provider-level)
    if let Some(ref extra) = req.extra_params {
        for (key, value) in extra {
            let name = key.trim();
            if !name.is_empty() && !name.starts_with('_') {
                body.insert(name.to_string(), value.clone());
            }
        }
    }
    if let Some(ref extra) = config.extra_params {
        for (key, value) in extra {
            let name = key.trim();
            if !name.is_empty() {
                body.insert(name.to_string(), value.clone());
            }
        }
    }

    let request_url = endpoint::build_endpoint_url(base_url, &resolved_endpoint);
    tracing::debug!(url = %request_url, model = %model_id, "sending openai responses request");

    let payload = serde_json::to_vec(&body).map_err(|e| {
        XlateError::Internal(format!("failed to serialize request body: {e}"))
    })?;

    // Build HTTP request
    let mut http_req = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .body(payload);

    // Apply custom headers
    for (key, value) in &req.custom_headers {
        let name = key.trim();
        if !name.is_empty() {
            http_req = http_req.header(name, value.as_str());
        }
    }

    let resp = http_req.send().await.map_err(|e| {
        XlateError::Transport(format!("openai request failed: {e}"))
    })?;

    let status = resp.status();
    if !status.is_success() {
        let status_code = status.as_u16();
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|v| format!(" retry-after={v}"));
        let body_text = resp.text().await.unwrap_or_default();
        let body_summary = if body_text.trim().is_empty() {
            String::new()
        } else {
            format!(" body={}", body_text.trim())
        };
        return Err(XlateError::Provider {
            status: Some(status_code),
            message: format!(
                "openai adapter status={status_code}{}{body_summary}",
                retry_after.as_deref().unwrap_or("")
            ),
        });
    }

    // Set up idle watchdog
    let (watchdog, watchdog_handle) = IdleWatchdog::new(req.stream_idle_timeout_ms);

    // Stream SSE from response body
    let mut state = StreamState::new(model_id.to_string());
    let mut byte_stream = resp.bytes_stream();
    let result = stream_responses_sse(
        &mut byte_stream,
        &mut state,
        sink,
        &watchdog,
    )
    .await;

    // Clean up watchdog
    watchdog_handle.abort();

    // If we got a watchdog timeout, prefer that error
    if result.is_err() {
        watchdog.check()?
    }

    result
}

// ---------------------------------------------------------------------------
// SSE stream processing
// ---------------------------------------------------------------------------

async fn stream_responses_sse(
    byte_stream: &mut (impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin),
    state: &mut StreamState,
    sink: &mut dyn EventSink,
    watchdog: &IdleWatchdog,
) -> Result<(), XlateError> {
    let mut remainder = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        watchdog.check()?;

        let chunk = chunk_result.map_err(|e| {
            XlateError::Transport(format!("openai stream read error: {e}"))
        })?;

        let text = String::from_utf8_lossy(&chunk);
        remainder.push_str(&text);

        while let Some(newline_pos) = remainder.find('\n') {
            let line = remainder[..newline_pos].trim().to_string();
            remainder = remainder[newline_pos + 1..].to_string();

            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }

            let payload_line = line.strip_prefix("data:").unwrap_or(&line).trim();

            if payload_line == "[DONE]" {
                flush_tagged_content_tail(sink, state, watchdog).await?;
                flush_thinking_completed(sink, state).await?;
                // Complete any remaining tools
                let remaining_tools: Vec<(String, ToolAccumulator)> =
                    state.tools.drain().collect();
                for (key, acc) in &remaining_tools {
                    complete_tool(sink, state, key, acc).await?;
                }
                flush_turn_finished(sink, state).await?;
                return Ok(());
            }

            let event: ResponsesStreamEvent =
                serde_json::from_str(payload_line).map_err(|e| {
                    XlateError::Internal(format!("failed to parse openai responses event: {e}"))
                })?;

            // Apply response-level fields (model, usage)
            if let Some(ref resp) = event.response {
                if !resp.model.trim().is_empty() {
                    state.current_model = resp.model.trim().to_string();
                }
                if let Some(ref usage) = resp.usage {
                    state.apply_usage(usage);
                }
            }

            // Dispatch by event type
            match event.event_type.trim() {
                "response.output_text.delta" => {
                    let parts = state.think_parser.consume(&event.delta);
                    emit_tagged_content_parts(sink, state, watchdog, parts).await?;
                }

                "response.output_item.added" => {
                    if let Some(ref item) = event.item {
                        apply_output_item(
                            sink,
                            state,
                            watchdog,
                            item,
                            event.output_index,
                            false,
                        )
                        .await?;
                    }
                }

                "response.function_call_arguments.delta" => {
                    let key = state.tool_key(&event.item_id, event.output_index);
                    if !state.tools.contains_key(&key) {
                        state.tools.insert(
                            key.clone(),
                            ToolAccumulator {
                                call_id: String::new(),
                                name: String::new(),
                                args: String::new(),
                                provider_item_id: String::new(),
                                provider_call_id: String::new(),
                                provider_status: String::new(),
                            },
                        );
                    }
                    if !event.delta.is_empty() {
                        {
                            let acc = state.tools.get_mut(&key).unwrap();
                            acc.args.push_str(&event.delta);
                        }
                        watchdog.mark_effective_content();

                        // Emit ToolCallDelta
                        let acc = state.tools.get(&key).unwrap();
                        let call_id = acc.call_id.trim().to_string();
                        let p_item_id = acc.provider_item_id.clone();
                        let p_status = acc.provider_status.clone();
                        let p_call_id = acc.provider_call_id.clone();
                        let meta = state.meta_with_provider_fields(
                            &p_item_id,
                            &p_status,
                            None,
                            &p_call_id,
                        );
                        sink.send(ModelEvent::ToolCallDelta {
                            tool_call_id: call_id,
                            args_text_delta: event.delta.clone(),
                            meta,
                        })
                        .await?;
                    }
                }

                "response.function_call_arguments.done" => {
                    let key = state.tool_key(&event.item_id, event.output_index);
                    if !state.tools.contains_key(&key) {
                        state.tools.insert(
                            key.clone(),
                            ToolAccumulator {
                                call_id: String::new(),
                                name: String::new(),
                                args: String::new(),
                                provider_item_id: String::new(),
                                provider_call_id: String::new(),
                                provider_status: String::new(),
                            },
                        );
                    }
                    let acc = state.tools.get_mut(&key).unwrap();
                    // Only apply final arguments if we haven't accumulated any yet
                    if !event.arguments.is_empty() && acc.args.is_empty() {
                        acc.args.push_str(&event.arguments);
                        watchdog.mark_effective_content();
                    }
                }

                "response.image_generation_call.partial_image" => {
                    tracing::debug!(
                        item_id = %event.item_id.trim(),
                        output_index = event.output_index,
                        "skipping partial_image event (image generation not implemented)"
                    );
                }

                "response.output_item.done" => {
                    if let Some(ref item) = event.item {
                        apply_output_item(
                            sink,
                            state,
                            watchdog,
                            item,
                            event.output_index,
                            true,
                        )
                        .await?;
                    }
                }

                "response.reasoning_summary_text.delta"
                | "response.reasoning_text.delta" => {
                    emit_thinking_delta(sink, state, watchdog, &event.delta).await?;
                }

                "response.completed" | "response.incomplete" => {
                    // If we haven't emitted any text yet, try to recover from
                    // the response object's output_text or output items
                    if let Some(ref resp) = event.response {
                        if !state.emitted_text {
                            if !resp.output_text.trim().is_empty() {
                                let parts =
                                    state.think_parser.consume(&resp.output_text);
                                emit_tagged_content_parts(sink, state, watchdog, parts)
                                    .await?;
                            } else {
                                for item in &resp.output {
                                    for content in &item.content {
                                        let ct = content.content_type.trim();
                                        if ct != "output_text" && ct != "text" {
                                            continue;
                                        }
                                        let parts =
                                            state.think_parser.consume(&content.text);
                                        emit_tagged_content_parts(
                                            sink, state, watchdog, parts,
                                        )
                                        .await?;
                                    }
                                }
                            }
                        }
                    }

                    flush_tagged_content_tail(sink, state, watchdog).await?;
                    flush_thinking_completed(sink, state).await?;

                    // Complete any output items from the response
                    if let Some(ref resp) = event.response {
                        // We need to clone the output items to avoid borrow issues
                        let output_items: Vec<(usize, ResponsesOutputItem)> = resp
                            .output
                            .iter()
                            .enumerate()
                            .map(|(i, item)| (i, clone_output_item(item)))
                            .collect();

                        for (index, item) in &output_items {
                            apply_output_item(
                                sink, state, watchdog, item, *index, true,
                            )
                            .await?;
                        }

                        state.finish_reason = resp.status.trim().to_string();
                        if let Some(ref details) = resp.incomplete_details {
                            if !details.reason.trim().is_empty() {
                                state.finish_reason =
                                    details.reason.trim().to_string();
                            }
                        }
                    }

                    state.turn_finished_pending = true;
                }

                "response.failed" | "error" => {
                    return Err(error_from_event(&event));
                }

                other => {
                    tracing::debug!(event_type = %other, "ignoring unhandled responses stream event");
                }
            }
        }
    }

    // Stream ended without [DONE] -- drain remaining state
    let remaining_tools: Vec<(String, ToolAccumulator)> = state.tools.drain().collect();
    for (key, acc) in &remaining_tools {
        complete_tool(sink, state, key, acc).await?;
    }

    watchdog.check()?;

    flush_tagged_content_tail(sink, state, watchdog).await?;
    flush_thinking_completed(sink, state).await?;
    flush_turn_finished(sink, state).await?;

    Ok(())
}

/// Clone a ResponsesOutputItem (needed because Deserialize doesn't auto-derive Clone)
fn clone_output_item(item: &ResponsesOutputItem) -> ResponsesOutputItem {
    ResponsesOutputItem {
        id: item.id.clone(),
        item_type: item.item_type.clone(),
        status: item.status.clone(),
        call_id: item.call_id.clone(),
        name: item.name.clone(),
        arguments: item.arguments.clone(),
        encrypted_content: item.encrypted_content.clone(),
        summary: item.summary.clone(),
        content: item
            .content
            .iter()
            .map(|c| ResponsesOutputContent {
                content_type: c.content_type.clone(),
                text: c.text.clone(),
            })
            .collect(),
    }
}

