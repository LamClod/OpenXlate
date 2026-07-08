//! SSE stream parser and event dispatcher for the Anthropic Messages API.
//!
//! Reads the HTTP response body line-by-line and dispatches `ModelEvent`s through
//! the `EventSink`.

use std::collections::HashMap;
use std::time::Instant;

use futures_util::StreamExt;

use xlate_core::event::{EventMeta, ModelEvent, Usage};
use xlate_core::think_tag::{ContentPart, ContentPartKind, ThinkTagParser};
use xlate_core::{EventSink, XlateError, now_ms};

use crate::types::{AnthropicEventPayload, AnthropicUsage, ToolAccumulator};

/// Process the Anthropic SSE stream from the response body.
///
/// Reads chunks incrementally via `bytes_stream()` so the caller receives
/// `ModelEvent`s as they arrive — no buffering of the full response.
pub(crate) async fn process_sse_stream(
    body: reqwest::Response,
    model_id: &str,
    idle_timeout_ms: Option<u64>,
    sink: &mut dyn EventSink,
) -> Result<(), XlateError> {
    let mut state = StreamState::new(model_id.to_string());
    let timeout = idle_timeout(idle_timeout_ms);

    const MAX_DATA_LINES: usize = 10_000;

    let mut byte_stream = body.bytes_stream();
    let mut remainder = String::new();
    let mut current_event = String::new();
    let mut data_lines: Vec<String> = Vec::new();

    loop {
        let chunk_result = match tokio::time::timeout(timeout, byte_stream.next()).await {
            Ok(Some(result)) => result,
            Ok(None) => break,
            Err(_) => return Err(XlateError::IdleTimeout(timeout.as_millis() as u64)),
        };

        let chunk = chunk_result.map_err(|e| {
            XlateError::Transport(format!("anthropic stream read error: {e}"))
        })?;

        let text = String::from_utf8_lossy(&chunk);
        remainder.push_str(&text);

        while let Some(newline_pos) = remainder.find('\n') {
            let line = remainder[..newline_pos].trim().to_string();
            remainder = remainder[newline_pos + 1..].to_string();

            if line.is_empty() {
                if !current_event.is_empty() && !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    data_lines.clear();
                    if payload.trim() != "[DONE]" {
                        state.handle_event(&current_event, &payload, sink).await?;
                    }
                } else {
                    data_lines.clear();
                }
                current_event.clear();
                continue;
            }

            if let Some(event_type) = line.strip_prefix("event:") {
                current_event = event_type.trim().to_string();
                continue;
            }

            if let Some(data) = line.strip_prefix("data:") {
                if data_lines.len() >= MAX_DATA_LINES {
                    return Err(XlateError::Internal(
                        "SSE event exceeded maximum data line count".into(),
                    ));
                }
                data_lines.push(data.trim().to_string());
            }
        }
    }

    if !current_event.is_empty() && !data_lines.is_empty() {
        let payload = data_lines.join("\n");
        if payload.trim() != "[DONE]" {
            state.handle_event(&current_event, &payload, sink).await?;
        }
    }

    state.flush_think_tag_tail(sink).await?;
    state.flush_thinking_completed(sink).await?;

    Ok(())
}

fn idle_timeout(timeout_ms: Option<u64>) -> std::time::Duration {
    const DEFAULT_MS: u64 = 240_000;
    const MIN_MS: u64 = 30_000;
    let ms = match timeout_ms {
        None | Some(0) => DEFAULT_MS,
        Some(v) => v.max(MIN_MS),
    };
    std::time::Duration::from_millis(ms)
}

// ---------------------------------------------------------------------------
// Stream state machine
// ---------------------------------------------------------------------------

struct StreamState {
    current_model: String,
    tool_blocks: HashMap<usize, ToolAccumulator>,
    thinking_started: Option<Instant>,
    current_thinking_signature: String,
    think_parser: ThinkTagParser,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cache_read_tokens: Option<i64>,
    cache_write_tokens: Option<i64>,
    finish_reason: String,
    idle_mark_count: u64,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            current_model: model,
            tool_blocks: HashMap::new(),
            thinking_started: None,
            current_thinking_signature: String::new(),
            think_parser: ThinkTagParser::new(),
            input_tokens: None,
            output_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            finish_reason: "message_stop".to_string(),
            idle_mark_count: 0,
        }
    }

    fn meta(&self) -> EventMeta {
        EventMeta {
            occurred_at_ms: now_ms(),
            provider: "anthropic".to_string(),
            model: self.current_model.clone(),
            provider_item_id: String::new(),
            provider_status: String::new(),
            provider_summary: None,
            provider_call_id: String::new(),
        }
    }

    async fn handle_event(
        &mut self,
        event_type: &str,
        payload: &str,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let event: AnthropicEventPayload =
            serde_json::from_str(payload).map_err(|e| {
                XlateError::Internal(format!(
                    "failed to parse anthropic SSE event: {e}"
                ))
            })?;

        // Check for error events.
        if event_type == "error" || event.r#type.trim() == "error" {
            return Err(self.build_error_from_event(&event));
        }

        match event_type {
            "message_start" => {
                if let Some(ref msg) = event.message {
                    let model = msg.model.trim();
                    if !model.is_empty() {
                        self.current_model = model.to_string();
                    }
                    if let Some(ref usage) = msg.usage {
                        self.apply_usage(usage);
                    }
                }
            }

            "content_block_start" => {
                if let Some(ref cb) = event.content_block {
                    if cb.r#type.trim() == "tool_use" {
                        self.flush_think_tag_tail(sink).await?;
                        self.flush_thinking_completed(sink).await?;

                        let mut acc = ToolAccumulator::new(
                            cb.id.trim().to_string(),
                            cb.name.trim().to_string(),
                        );
                        // If the content_block_start already carries input, include it.
                        if let Some(ref input) = cb.input {
                            if !is_empty_tool_input(input) {
                                if let Ok(encoded) = serde_json::to_string(input) {
                                    if encoded != "null" {
                                        acc.args = encoded;
                                    }
                                }
                            }
                        }
                        self.idle_mark_count += 1;
                        // Emit PartialToolCall event.
                        sink.send(ModelEvent::PartialToolCall {
                            tool_call_id: acc.call_id.clone(),
                            name: acc.name.clone(),
                            meta: self.meta(),
                        })
                        .await?;
                        self.tool_blocks.insert(event.index, acc);
                    }
                }
            }

            "content_block_delta" => {
                if let Some(ref delta) = event.delta {
                    // text_delta
                    if delta.r#type == "text_delta" && !delta.text.is_empty() {
                        let parts = self.think_parser.consume(&delta.text);
                        self.emit_tagged_parts(&parts, sink).await?;
                    }
                    // thinking_delta
                    if delta.r#type == "thinking_delta" && !delta.thinking.is_empty() {
                        self.emit_thinking_delta(&delta.thinking, sink).await?;
                    }
                    // signature
                    if !delta.signature.is_empty() {
                        self.current_thinking_signature =
                            delta.signature.trim().to_string();
                    }
                    // input_json_delta (tool arguments)
                    if delta.r#type == "input_json_delta" {
                        if let Some(acc) = self.tool_blocks.get_mut(&event.index) {
                            if !delta.partial_json.is_empty() {
                                acc.args.push_str(&delta.partial_json);
                                self.idle_mark_count += 1;
                            }
                        }
                        // Emit ToolCallDelta outside the mutable borrow scope.
                        if !delta.partial_json.is_empty() {
                            if let Some(acc) = self.tool_blocks.get(&event.index) {
                                let meta = self.meta();
                                let call_id = acc.call_id.clone();
                                let args_delta = delta.partial_json.clone();
                                sink.send(ModelEvent::ToolCallDelta {
                                    tool_call_id: call_id,
                                    args_text_delta: args_delta,
                                    meta,
                                })
                                .await?;
                            }
                        }
                    }
                }
            }

            "content_block_stop" => {
                if let Some(acc) = self.tool_blocks.remove(&event.index) {
                    // Validate and complete tool args.
                    let args_json = complete_tool_args_json(&acc)?;
                    self.idle_mark_count += 1;
                    sink.send(ModelEvent::ToolLikeCompleted {
                        tool_call_id: acc.call_id,
                        name: acc.name,
                        arguments_json: args_json,
                        meta: self.meta(),
                    })
                    .await?;
                } else {
                    // Not a tool block stop -- flush any pending text/thinking.
                    self.flush_think_tag_tail(sink).await?;
                    self.flush_thinking_completed(sink).await?;
                }
            }

            "message_delta" => {
                if let Some(ref usage) = event.usage {
                    self.apply_usage(usage);
                }
                if let Some(ref delta) = event.delta {
                    let stop = delta.stop_reason.trim();
                    if !stop.is_empty() {
                        self.finish_reason = stop.to_string();
                    }
                }
            }

            "message_stop" => {
                self.flush_think_tag_tail(sink).await?;
                self.flush_thinking_completed(sink).await?;
                sink.send(ModelEvent::TurnFinished {
                    finish_reason: self.finish_reason.clone(),
                    usage: Usage {
                        input_tokens: self.input_tokens,
                        output_tokens: self.output_tokens,
                        cache_read_tokens: self.cache_read_tokens,
                        cache_write_tokens: self.cache_write_tokens,
                        ..Default::default()
                    },
                    meta: self.meta(),
                })
                .await?;
            }

            other => {
                tracing::trace!(event_type = other, "ignoring unknown anthropic SSE event");
            }
        }

        Ok(())
    }

    fn apply_usage(&mut self, usage: &AnthropicUsage) {
        if let Some(v) = usage.input_tokens {
            self.input_tokens = Some(v.max(0));
        }
        if let Some(v) = usage.output_tokens {
            self.output_tokens = Some(v.max(0));
        }
        if let Some(v) = usage.cache_read_input_tokens {
            self.cache_read_tokens = Some(v.max(0));
        }
        if let Some(v) = usage.cache_creation_input_tokens {
            self.cache_write_tokens = Some(v.max(0));
        }
    }

    async fn emit_text_delta(
        &mut self,
        text: &str,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        if text.is_empty() {
            return Ok(());
        }
        self.idle_mark_count += 1;
        self.flush_thinking_completed(sink).await?;
        sink.send(ModelEvent::TextDelta {
            text: text.to_string(),
            meta: self.meta(),
        })
        .await
    }

    async fn emit_thinking_delta(
        &mut self,
        reasoning: &str,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        if reasoning.is_empty() {
            return Ok(());
        }
        self.idle_mark_count += 1;
        if self.thinking_started.is_none() {
            self.thinking_started = Some(Instant::now());
        }
        sink.send(ModelEvent::ThinkingDelta {
            text: reasoning.to_string(),
            style: None,
            meta: self.meta(),
        })
        .await
    }

    async fn flush_thinking_completed(
        &mut self,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let started = match self.thinking_started.take() {
            Some(s) => s,
            None => return Ok(()),
        };
        let duration_ms = started.elapsed().as_millis() as i32;
        let signature = std::mem::take(&mut self.current_thinking_signature);
        let signature_source = if signature.is_empty() {
            String::new()
        } else {
            "anthropic".to_string()
        };
        sink.send(ModelEvent::ThinkingCompleted {
            duration_ms: Some(duration_ms.max(0)),
            signature,
            signature_source,
            meta: self.meta(),
        })
        .await
    }

    async fn emit_tagged_parts(
        &mut self,
        parts: &[ContentPart],
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        for part in parts {
            match part.kind {
                ContentPartKind::Text => {
                    self.emit_text_delta(&part.text, sink).await?;
                }
                ContentPartKind::Reasoning => {
                    self.emit_thinking_delta(&part.text, sink).await?;
                }
                ContentPartKind::ThinkingCompleted => {
                    self.flush_thinking_completed(sink).await?;
                }
            }
        }
        Ok(())
    }

    async fn flush_think_tag_tail(
        &mut self,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let parts = self.think_parser.flush();
        // Need to handle parts without borrowing self.think_parser.
        for part in &parts {
            match part.kind {
                ContentPartKind::Text => {
                    self.emit_text_delta(&part.text, sink).await?;
                }
                ContentPartKind::Reasoning => {
                    self.emit_thinking_delta(&part.text, sink).await?;
                }
                ContentPartKind::ThinkingCompleted => {
                    self.flush_thinking_completed(sink).await?;
                }
            }
        }
        Ok(())
    }

    fn build_error_from_event(&mut self, event: &AnthropicEventPayload) -> XlateError {
        self.finish_reason = "error".to_string();
        if let Some(ref err) = event.error {
            let mut parts = Vec::new();
            let t = err.r#type.trim();
            if !t.is_empty() {
                parts.push(format!("type={t}"));
            }
            let c = err.code.trim();
            if !c.is_empty() {
                parts.push(format!("code={c}"));
            }
            let msg = err.message.trim();
            if !msg.is_empty() {
                let prefix = if parts.is_empty() {
                    String::new()
                } else {
                    format!("{} ", parts.join(" "))
                };
                return XlateError::Provider {
                    status: None,
                    message: format!("anthropic provider error {prefix}: {msg}"),
                };
            }
            if !parts.is_empty() {
                return XlateError::Provider {
                    status: None,
                    message: format!("anthropic provider error {}", parts.join(" ")),
                };
            }
        }
        XlateError::Provider {
            status: None,
            message: "anthropic provider error".to_string(),
        }
    }
}

fn is_empty_tool_input(input: &serde_json::Value) -> bool {
    match input {
        serde_json::Value::Null => true,
        serde_json::Value::Object(m) if m.is_empty() => true,
        serde_json::Value::Array(a) if a.is_empty() => true,
        _ => false,
    }
}

fn complete_tool_args_json(acc: &ToolAccumulator) -> Result<String, XlateError> {
    let trimmed = acc.args.trim();
    if trimmed.is_empty() {
        return Ok("{}".to_string());
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
        let name = if acc.name.trim().is_empty() {
            "tool"
        } else {
            acc.name.trim()
        };
        XlateError::Internal(format!(
            "anthropic returned incomplete or malformed tool input for {name}: {e}"
        ))
    })?;
    if !value.is_object() {
        let name = if acc.name.trim().is_empty() {
            "tool"
        } else {
            acc.name.trim()
        };
        return Err(XlateError::Internal(format!(
            "anthropic returned non-object tool input for {name}"
        )));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_tool_args_json() {
        let acc = ToolAccumulator::new("id".into(), "test".into());
        assert_eq!(complete_tool_args_json(&acc).unwrap(), "{}");

        let mut acc2 = ToolAccumulator::new("id".into(), "test".into());
        acc2.args = r#"{"key": "value"}"#.to_string();
        assert_eq!(
            complete_tool_args_json(&acc2).unwrap(),
            r#"{"key": "value"}"#
        );
    }
}
