//! Message and tool normalization for the OpenAI Responses API.
//!
//! Converts `xlate_core::Message` and tool descriptors into the formats
//! expected by the Responses API (`/v1/responses`), which differ from
//! Chat Completions:
//!
//! - System messages become the top-level `instructions` string.
//! - User/assistant messages become `input` items with `input_text`/`output_text`
//!   content types.
//! - Tool calls use `function_call` / `function_call_output` item types.
//! - Reasoning signatures replay as `reasoning` items with `encrypted_content`.
//! - Tool descriptors are flattened (no wrapping `{"type":"function","function":{...}}`).
//!
//! Ported from the original Go implementation's `normalizeOpenAIResponsesInput`,
//! `openAIResponsesMessageContent`, `normalizeOpenAIResponsesTools`, etc.

use std::collections::HashMap;

use xlate_core::types::{ContentPartType, Message};
use xlate_core::XlateError;

use crate::messages;

/// The signature source value for reasoning signatures originating from the
/// OpenAI Responses API encrypted reasoning content.
const REASONING_SIGNATURE_SOURCE_OPENAI_RESPONSES: &str = "openai_responses";

// ---------------------------------------------------------------------------
// Input normalization
// ---------------------------------------------------------------------------

/// Convert a slice of `Message` into the Responses API format.
///
/// Returns `(instructions, input_items)` where `instructions` is the
/// concatenation of all system messages and `input_items` is the array that
/// goes into the `input` field.
pub(crate) fn normalize_responses_input(
    messages: &[Message],
) -> Result<(String, Vec<serde_json::Value>), XlateError> {
    if messages.is_empty() {
        return Ok((String::new(), Vec::new()));
    }

    let mut instruction_parts: Vec<String> = Vec::new();
    let mut items: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    let mut responses_call_ids: HashMap<String, String> = HashMap::new();
    let mut active_assistant_reasoning_key = String::new();

    for message in messages {
        let role = message.role.trim();

        // System messages -> instructions
        if role == "system" {
            let text = responses_message_text(message);
            if !text.trim().is_empty() {
                instruction_parts.push(text.trim().to_string());
            }
            active_assistant_reasoning_key.clear();
            continue;
        }

        // Tool result messages
        if role == "tool" && !message.tool_call_id.trim().is_empty() {
            let call_id =
                responses_tool_message_call_id(message, &responses_call_ids);
            items.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": responses_message_text(message),
            }));
            active_assistant_reasoning_key.clear();
            continue;
        }

        if role != "assistant" {
            active_assistant_reasoning_key.clear();
        }

        // Reasoning item replay (only for assistant messages with OpenAI Responses signature)
        if should_include_responses_reasoning_item(message) {
            let reasoning_key = responses_reasoning_replay_key(message);
            if reasoning_key != active_assistant_reasoning_key {
                items.push(responses_reasoning_item(message));
                active_assistant_reasoning_key = reasoning_key;
            }
        }

        // Content
        if !message.content.trim().is_empty() || !message.content_parts.is_empty() {
            let content = responses_message_content(message, role == "assistant")?;
            if !content.is_empty() {
                items.push(serde_json::json!({
                    "role": responses_message_role(role),
                    "content": content,
                }));
            }
        }

        // Assistant tool calls
        if role == "assistant" && !message.tool_calls.is_empty() {
            for tc in &message.tool_calls {
                let name = tc.function.name.trim();
                if name.is_empty() {
                    continue;
                }
                let mut call_id = responses_tool_call_call_id(tc);
                if call_id.trim().is_empty() {
                    call_id = responses_provider_call_id(&tc.id);
                }

                let internal_id = tc.id.trim();
                if !internal_id.is_empty() && !call_id.trim().is_empty() {
                    responses_call_ids
                        .insert(internal_id.to_string(), call_id.trim().to_string());
                }

                let mut tool_item = serde_json::json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": tc.function.arguments,
                });

                if !tc.openai_responses_id.trim().is_empty() {
                    tool_item["id"] =
                        serde_json::Value::String(tc.openai_responses_id.trim().to_string());
                }

                let status = tc.openai_responses_status.trim();
                if !status.is_empty() {
                    tool_item["status"] = serde_json::Value::String(status.to_string());
                } else {
                    tool_item["status"] = serde_json::Value::String("completed".to_string());
                }

                items.push(tool_item);
            }
        }
    }

    let instructions = instruction_parts.join("\n\n");
    Ok((instructions, items))
}

// ---------------------------------------------------------------------------
// Tool normalization
// ---------------------------------------------------------------------------

/// Convert Chat Completions-style tool descriptors into Responses API format.
///
/// The Responses API uses a flat `{"type":"function","name":...,"parameters":...}`
/// format instead of the nested `{"type":"function","function":{...}}` shape.
pub(crate) fn normalize_responses_tools(
    items: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>, XlateError> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut tools = Vec::with_capacity(items.len());
    for item in items {
        let raw = item
            .as_object()
            .ok_or_else(|| {
                XlateError::InvalidRequest(
                    "openai responses tool descriptor must be an object".into(),
                )
            })?;

        // If there's a "function" sub-object, use that as the source
        let source = if let Some(func_val) = raw.get("function") {
            func_val.as_object().unwrap_or(raw)
        } else {
            raw
        };

        let name = source
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if name.is_empty() {
            return Err(XlateError::InvalidRequest(
                "openai responses tool descriptor name is required".into(),
            ));
        }

        let mut tool = serde_json::json!({
            "type": "function",
            "name": name,
        });

        if let Some(desc) = source.get("description").and_then(|v| v.as_str()) {
            let desc = desc.trim();
            if !desc.is_empty() {
                tool["description"] = serde_json::Value::String(desc.to_string());
            }
        }

        if let Some(params) = source.get("parameters") {
            tool["parameters"] = params.clone();
        } else {
            tool["parameters"] = serde_json::json!({"type": "object", "properties": {}});
        }

        // strict flag: check source first, then raw (in case it's on the outer object)
        if let Some(strict) = source.get("strict") {
            tool["strict"] = strict.clone();
        } else if let Some(strict) = raw.get("strict") {
            tool["strict"] = strict.clone();
        }

        tools.push(tool);
    }

    Ok(tools)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn responses_message_text(message: &Message) -> String {
    if !message.content.trim().is_empty() {
        return message.content.clone();
    }
    if !message.content_parts.is_empty() {
        return messages::collapse_text_content_parts(&message.content_parts);
    }
    String::new()
}

fn responses_message_role(role: &str) -> &'static str {
    match role.trim() {
        "assistant" => "assistant",
        _ => "user",
    }
}

fn responses_message_content(
    message: &Message,
    assistant: bool,
) -> Result<Vec<serde_json::Value>, XlateError> {
    let text_type = if assistant { "output_text" } else { "input_text" };

    if !messages::has_image_content_parts(&message.content_parts) {
        let text = responses_message_text(message);
        if text.is_empty() {
            return Ok(Vec::new());
        }
        return Ok(vec![serde_json::json!({
            "type": text_type,
            "text": text,
        })]);
    }

    let mut parts = Vec::with_capacity(message.content_parts.len() + 1);

    if message.content_parts.is_empty() && !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": text_type,
            "text": message.content,
        }));
    }

    for part in &message.content_parts {
        match part.kind {
            ContentPartType::Text => {
                if part.text.is_empty() {
                    continue;
                }
                parts.push(serde_json::json!({
                    "type": text_type,
                    "text": part.text,
                }));
            }
            ContentPartType::Image => {
                let data_url = messages::image_content_data_url(part.image.as_ref())?;
                parts.push(serde_json::json!({
                    "type": "input_image",
                    "image_url": data_url,
                }));
            }
        }
    }

    Ok(parts)
}

fn should_include_responses_reasoning_item(message: &Message) -> bool {
    if message.role.trim() != "assistant" || message.reasoning_signature.trim().is_empty() {
        return false;
    }
    message.reasoning_signature_source.trim() == REASONING_SIGNATURE_SOURCE_OPENAI_RESPONSES
}

fn responses_reasoning_replay_key(message: &Message) -> String {
    format!(
        "{}\x00{}\x00{}\x00{}",
        message.reasoning_signature.trim(),
        message.openai_responses_reasoning_id.trim(),
        message.openai_responses_reasoning_status.trim(),
        message
            .openai_responses_reasoning_summary
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_default(),
    )
}

fn responses_reasoning_item(message: &Message) -> serde_json::Value {
    let mut item = serde_json::json!({
        "type": "reasoning",
        "encrypted_content": message.reasoning_signature.trim(),
    });

    let reasoning_id = message.openai_responses_reasoning_id.trim();
    if !reasoning_id.is_empty() {
        item["id"] = serde_json::Value::String(reasoning_id.to_string());
    }

    let reasoning_status = message.openai_responses_reasoning_status.trim();
    if !reasoning_status.is_empty() {
        item["status"] = serde_json::Value::String(reasoning_status.to_string());
    }

    if let Some(ref summary) = message.openai_responses_reasoning_summary {
        item["summary"] = summary.clone();
    } else {
        item["summary"] = serde_json::json!([]);
    }

    item
}

fn responses_tool_message_call_id(
    message: &Message,
    responses_call_ids: &HashMap<String, String>,
) -> String {
    let internal_id = message.tool_call_id.trim();
    if internal_id.is_empty() {
        return String::new();
    }
    if let Some(call_id) = responses_call_ids.get(internal_id) {
        if !call_id.trim().is_empty() {
            return call_id.trim().to_string();
        }
    }
    responses_provider_call_id(internal_id)
}

fn responses_tool_call_call_id(tc: &xlate_core::types::ToolCallDescriptor) -> String {
    let call_id = tc.openai_responses_call_id.trim();
    if !call_id.is_empty() {
        return call_id.to_string();
    }
    responses_provider_call_id(&tc.id)
}

/// Derive a provider-safe call ID from an internal tool call ID.
///
/// Simplified from the Go `openAIResponsesProviderCallID`:
/// - If it contains `::`, take the part after `::`.
/// - If it starts with `tc_` and has 3 underscore-separated parts, take the last part.
/// - Otherwise use as-is.
fn responses_provider_call_id(tool_call_id: &str) -> String {
    let trimmed = tool_call_id.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Legacy format: "namespace::raw"
    if let Some((_ns, raw)) = trimmed.split_once("::") {
        let raw = raw.trim();
        if !raw.is_empty() {
            return raw.to_string();
        }
    }

    // tc_{namespace}_{raw} format
    if trimmed.starts_with("tc_") {
        let parts: Vec<&str> = trimmed.splitn(3, '_').collect();
        if parts.len() == 3 && !parts[2].trim().is_empty() {
            return parts[2].trim().to_string();
        }
    }

    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use xlate_core::types::{ToolCallDescriptor, ToolCallFunctionShape};

    #[test]
    fn normalize_basic_messages() {
        let messages = vec![
            Message {
                role: "system".into(),
                content: "You are helpful.".into(),
                ..Default::default()
            },
            Message {
                role: "user".into(),
                content: "Hello".into(),
                ..Default::default()
            },
            Message {
                role: "assistant".into(),
                content: "Hi there!".into(),
                ..Default::default()
            },
        ];
        let (instructions, items) = normalize_responses_input(&messages).unwrap();
        assert_eq!(instructions, "You are helpful.");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[0]["content"][0]["type"], "input_text");
        assert_eq!(items[0]["content"][0]["text"], "Hello");
        assert_eq!(items[1]["role"], "assistant");
        assert_eq!(items[1]["content"][0]["type"], "output_text");
        assert_eq!(items[1]["content"][0]["text"], "Hi there!");
    }

    #[test]
    fn normalize_tool_messages() {
        let messages = vec![
            Message {
                role: "assistant".into(),
                tool_calls: vec![ToolCallDescriptor {
                    id: "call_1".into(),
                    kind: "function".into(),
                    function: ToolCallFunctionShape {
                        name: "read_file".into(),
                        arguments: r#"{"path":"test.txt"}"#.into(),
                    },
                    openai_responses_call_id: "call_abc".into(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "file contents".into(),
                tool_call_id: "call_1".into(),
                ..Default::default()
            },
        ];
        let (_instructions, items) = normalize_responses_input(&messages).unwrap();
        // First item: function_call
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call_abc");
        assert_eq!(items[0]["name"], "read_file");
        // Second item: function_call_output (should map call_1 -> call_abc)
        assert_eq!(items[1]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call_abc");
    }

    #[test]
    fn normalize_reasoning_item() {
        let msg = Message {
            role: "assistant".into(),
            content: "hello".into(),
            reasoning_signature: "encrypted_data".into(),
            reasoning_signature_source: "openai_responses".into(),
            openai_responses_reasoning_id: "rs_123".into(),
            openai_responses_reasoning_status: "completed".into(),
            ..Default::default()
        };
        let (_inst, items) = normalize_responses_input(&[msg]).unwrap();
        // First item: reasoning, second: content
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["encrypted_content"], "encrypted_data");
        assert_eq!(items[0]["id"], "rs_123");
        assert_eq!(items[1]["content"][0]["type"], "output_text");
    }

    #[test]
    fn normalize_tools_flat_format() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }
            }
        })];
        let result = normalize_responses_tools(&tools).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["name"], "read_file");
        assert_eq!(result[0]["type"], "function");
        assert!(result[0].get("function").is_none());
    }

    #[test]
    fn provider_call_id_extraction() {
        assert_eq!(responses_provider_call_id("ns::raw_id"), "raw_id");
        assert_eq!(responses_provider_call_id("tc_ns_raw"), "raw");
        assert_eq!(responses_provider_call_id("simple_id"), "simple_id");
        assert_eq!(responses_provider_call_id(""), "");
    }
}
