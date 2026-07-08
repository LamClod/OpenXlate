//! Chat Completions streaming implementation.
//!
//! Sends a POST to the Chat Completions endpoint, reads the SSE stream line by
//! line, and emits `ModelEvent`s through an `EventSink`.
//!
//! Ported from the original Go implementation's `streamChatCompletions`.

use std::collections::HashMap;
use std::time::Instant;

use futures_util::StreamExt;
use serde::Deserialize;

use xlate_core::event::{EventMeta, ModelEvent, Usage};
use xlate_core::types::NormalizedRequest;
use xlate_core::{EventSink, XlateError, now_ms};

use crate::endpoint;
use crate::idle_watchdog::IdleWatchdog;
use crate::messages;
use crate::think_tag::{ContentPartKind, ThinkTagParser};
use crate::thinking_disable;

// ---------------------------------------------------------------------------
// SSE chunk types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAIChunk {
    #[serde(rename = "type", default)]
    chunk_type: String,
    #[serde(default)]
    request_id: String,
    #[serde(default)]
    error: Option<ChunkError>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkError {
    #[serde(default)]
    message: String,
    #[serde(rename = "type", default)]
    error_type: String,
    #[serde(default)]
    code: String,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: String,
    #[serde(default)]
    reasoning_content: String,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: ToolCallFunctionDelta,
}

#[derive(Debug, Default, Deserialize)]
struct ToolCallFunctionDelta {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: i64,
    #[serde(default)]
    completion_tokens: i64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: i64,
}

// ---------------------------------------------------------------------------
// Tool accumulator
// ---------------------------------------------------------------------------

struct ToolAccumulator {
    call_id: String,
    name: String,
    args: String,
}

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

struct StreamState {
    current_model: String,
    tools: HashMap<usize, ToolAccumulator>,
    usage: Usage,
    usage_present: bool,
    cache_read_present: bool,
    finish_reason: String,
    turn_finished_pending: bool,
    thinking_started: Option<Instant>,
    thinking_active: bool,
    think_parser: ThinkTagParser,
    first_event: bool,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            current_model: model,
            tools: HashMap::new(),
            usage: Usage::default(),
            usage_present: false,
            cache_read_present: false,
            finish_reason: String::new(),
            turn_finished_pending: false,
            thinking_started: None,
            thinking_active: false,
            think_parser: ThinkTagParser::new(),
            first_event: true,
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

    fn apply_usage(&mut self, usage: &ChunkUsage) {
        self.usage_present = true;
        let prompt_tokens = usage.prompt_tokens.max(0);
        let mut cached_tokens: i64 = 0;
        if let Some(ref details) = usage.prompt_tokens_details {
            self.cache_read_present = true;
            cached_tokens = details.cached_tokens.max(0);
        }
        if cached_tokens > prompt_tokens {
            cached_tokens = prompt_tokens;
        }
        self.usage.input_tokens = Some(prompt_tokens - cached_tokens);
        self.usage.output_tokens = Some(usage.completion_tokens.max(0));
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

async fn flush_turn_finished(
    sink: &mut dyn EventSink,
    state: &mut StreamState,
) -> Result<(), XlateError> {
    if !state.turn_finished_pending {
        return Ok(());
    }
    state.turn_finished_pending = false;
    let usage = state.build_turn_finished_usage();
    sink.send(ModelEvent::TurnFinished {
        finish_reason: state.finish_reason.clone(),
        usage,
        meta: state.meta(),
    })
    .await
}

fn error_from_chunk(chunk: &OpenAIChunk) -> XlateError {
    if let Some(ref err) = chunk.error {
        let mut parts = Vec::new();
        if !err.error_type.trim().is_empty() {
            parts.push(format!("type={}", err.error_type.trim()));
        }
        if !err.code.trim().is_empty() {
            parts.push(format!("code={}", err.code.trim()));
        }
        if !chunk.request_id.trim().is_empty() {
            parts.push(format!("request_id={}", chunk.request_id.trim()));
        }
        let message = err.message.trim();
        if !message.is_empty() {
            if parts.is_empty() {
                return XlateError::Provider {
                    status: None,
                    message: format!("openai chat stream error: {message}"),
                };
            }
            return XlateError::Provider {
                status: None,
                message: format!("openai chat stream error {}: {message}", parts.join(" ")),
            };
        }
        if !parts.is_empty() {
            return XlateError::Provider {
                status: None,
                message: format!("openai chat stream error {}", parts.join(" ")),
            };
        }
    }
    XlateError::Provider {
        status: None,
        message: "openai chat stream error".into(),
    }
}

// ---------------------------------------------------------------------------
// Main stream function
// ---------------------------------------------------------------------------

pub(crate) async fn stream_chat_completions(
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

    // Determine thinking state from request
    let thinking_effort = req
        .thinking
        .as_ref()
        .and_then(|t| t.effort.as_deref())
        .unwrap_or("");
    let thinking_enabled = !thinking_effort.is_empty()
        && thinking_disable::normalize_thinking_effort(thinking_effort) != "disabled";

    // Build messages
    let normalized_messages = messages::normalize_messages(&req.messages, thinking_enabled)?;

    // Build request body as a JSON map for easy manipulation
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model_id.to_string()));
    body.insert("messages".into(), serde_json::Value::Array(normalized_messages));
    body.insert("stream".into(), serde_json::Value::Bool(true));

    if let Some(max_tokens) = req.max_tokens {
        body.insert("max_tokens".into(), serde_json::Value::Number(max_tokens.into()));
    }

    body.insert(
        "stream_options".into(),
        serde_json::json!({"include_usage": true}),
    );

    // Prompt cache key (only for GPT models)
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
        body.insert("tools".into(), serde_json::Value::Array(req.tools.clone()));
    }

    // Reasoning effort
    if !thinking_effort.is_empty()
        && thinking_disable::normalize_thinking_effort(thinking_effort) != "disabled"
    {
        body.insert(
            "reasoning_effort".into(),
            serde_json::Value::String(thinking_effort.to_string()),
        );
    }

    // Apply thinking disable if needed
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
    tracing::debug!(url = %request_url, model = %model_id, "sending openai chat completions request");

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
    let result = stream_sse_lines(
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

async fn stream_sse_lines(
    byte_stream: &mut (impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin),
    state: &mut StreamState,
    sink: &mut dyn EventSink,
    watchdog: &IdleWatchdog,
) -> Result<(), XlateError> {
    // Read bytes and split into lines manually (like Go's bufio.Scanner)
    let mut remainder = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        // Check watchdog on each chunk
        watchdog.check()?;

        let chunk = chunk_result.map_err(|e| {
            XlateError::Transport(format!("openai stream read error: {e}"))
        })?;

        let text = String::from_utf8_lossy(&chunk);
        remainder.push_str(&text);

        // Process complete lines
        while let Some(newline_pos) = remainder.find('\n') {
            let line = remainder[..newline_pos].trim().to_string();
            remainder = remainder[newline_pos + 1..].to_string();

            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }

            if state.first_event {
                state.first_event = false;
            }

            let payload_line = line.strip_prefix("data:").unwrap_or(&line).trim();

            if payload_line == "[DONE]" {
                flush_tagged_content_tail(sink, state, watchdog).await?;
                flush_thinking_completed(sink, state).await?;
                flush_turn_finished(sink, state).await?;
                return Ok(());
            }

            let chunk: OpenAIChunk = serde_json::from_str(payload_line).map_err(|e| {
                XlateError::Internal(format!("failed to parse openai chunk: {e}"))
            })?;

            // Handle error chunks
            if chunk.chunk_type.trim() == "error" || chunk.error.is_some() {
                return Err(error_from_chunk(&chunk));
            }

            // Update model if present
            if !chunk.model.trim().is_empty() {
                state.current_model = chunk.model.trim().to_string();
            }

            // Apply usage
            if let Some(ref usage) = chunk.usage {
                state.apply_usage(usage);
            }

            if chunk.choices.is_empty() {
                // Usage-only chunk or model update
                flush_tagged_content_tail(sink, state, watchdog).await?;
                flush_thinking_completed(sink, state).await?;
                flush_turn_finished(sink, state).await?;
                continue;
            }

            let choice = &chunk.choices[0];

            // Content delta
            if !choice.delta.content.is_empty() {
                let parts = state.think_parser.consume(&choice.delta.content);
                emit_tagged_content_parts(sink, state, watchdog, parts).await?;
            }

            // Reasoning content delta
            if !choice.delta.reasoning_content.is_empty() {
                emit_thinking_delta(
                    sink,
                    state,
                    watchdog,
                    &choice.delta.reasoning_content,
                )
                .await?;
            }

            // Tool calls — flush thinking first if only tool calls in this chunk
            if !choice.delta.tool_calls.is_empty()
                && choice.delta.content.is_empty()
                && choice.delta.reasoning_content.is_empty()
            {
                flush_tagged_content_tail(sink, state, watchdog).await?;
                flush_thinking_completed(sink, state).await?;
            }

            for tc_delta in &choice.delta.tool_calls {
                watchdog.mark_effective_content();
                let accumulator =
                    state.tools.entry(tc_delta.index).or_insert_with(|| ToolAccumulator {
                        call_id: String::new(),
                        name: String::new(),
                        args: String::new(),
                    });
                if !tc_delta.id.trim().is_empty() {
                    accumulator.call_id = tc_delta.id.trim().to_string();
                }
                if !tc_delta.function.name.trim().is_empty() {
                    accumulator.name = tc_delta.function.name.trim().to_string();
                }
                if !tc_delta.function.arguments.is_empty() {
                    accumulator.args.push_str(&tc_delta.function.arguments);
                }
            }

            // Finish reason
            if let Some(ref reason) = choice.finish_reason {
                flush_tagged_content_tail(sink, state, watchdog).await?;
                flush_thinking_completed(sink, state).await?;

                // Emit all accumulated tool calls in index order
                let mut tools: Vec<(usize, ToolAccumulator)> =
                    state.tools.drain().collect();
                tools.sort_by_key(|(idx, _)| *idx);
                for (_idx, acc) in tools {
                    watchdog.mark_effective_content();
                    sink.send(ModelEvent::ToolLikeCompleted {
                        tool_call_id: acc.call_id.clone(),
                        name: acc.name.clone(),
                        arguments_json: acc.args.clone(),
                        meta: state.meta(),
                    })
                    .await?;
                }

                state.finish_reason = reason.trim().to_string();
                state.turn_finished_pending = true;
            }
        }
    }

    // Drain any remaining tools (stream ended without [DONE])
    let mut tools: Vec<(usize, ToolAccumulator)> = state.tools.drain().collect();
    tools.sort_by_key(|(idx, _)| *idx);
    for (_idx, acc) in tools {
        watchdog.mark_effective_content();
        sink.send(ModelEvent::ToolLikeCompleted {
            tool_call_id: acc.call_id.clone(),
            name: acc.name.clone(),
            arguments_json: acc.args.clone(),
            meta: state.meta(),
        })
        .await?;
    }

    // Check for idle timeout
    watchdog.check()?;

    // Final flush
    flush_tagged_content_tail(sink, state, watchdog).await?;
    flush_thinking_completed(sink, state).await?;
    flush_turn_finished(sink, state).await?;

    Ok(())
}

