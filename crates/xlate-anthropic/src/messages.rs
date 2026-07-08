//! Message normalization: converts `xlate_core::Message` into Anthropic API format.
//!
//! This mirrors the Go `normalizeAnthropicProviderMessages` and related helpers.

use xlate_core::types::{ContentPartType, Message};
use xlate_core::XlateError;

use crate::types::AnthropicMessage;

/// Result of normalizing input messages for the Anthropic Messages API.
pub(crate) struct NormalizedMessages {
    /// Extracted system prompt parts (joined later into the `system` field).
    pub system_parts: Vec<String>,
    /// Provider-formatted messages array.
    pub messages: Vec<AnthropicMessage>,
}

/// Normalize the input message list into Anthropic Messages API format.
///
/// - System messages are extracted into `system_parts`.
/// - Tool result messages are batched together and emitted as a single `user`
///   message with `tool_result` blocks.
/// - Assistant messages with `reasoning_content` get a leading `thinking` block
///   when `thinking_enabled` is true.
/// - Consecutive assistant messages where the second only carries tool_use blocks
///   (and shares the same thinking block) are merged.
pub(crate) fn normalize_messages(
    input: &[Message],
    thinking_enabled: bool,
) -> Result<NormalizedMessages, XlateError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();
    let mut pending_tool_results: Vec<serde_json::Value> = Vec::new();

    let flush_tool_results =
        |pending: &mut Vec<serde_json::Value>, messages: &mut Vec<AnthropicMessage>| {
            if pending.is_empty() {
                return;
            }
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: std::mem::take(pending),
            });
        };

    for message in input {
        let role = message.role.trim();
        match role {
            "system" => {
                // System messages with image content are not supported.
                if has_image_content_parts(&message.content_parts) {
                    return Err(XlateError::InvalidRequest(
                        "anthropic system message does not support image content".into(),
                    ));
                }
                let content = effective_text_content(message);
                if !content.trim().is_empty() {
                    system_parts.push(content);
                }
            }
            "tool" => {
                let tool_use_id = provider_tool_call_id(&message.tool_call_id);
                if tool_use_id.is_empty() {
                    return Err(XlateError::InvalidRequest(
                        "anthropic tool message requires tool_call_id".into(),
                    ));
                }
                pending_tool_results.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": message.content,
                }));
            }
            "user" | "assistant" => {
                flush_tool_results(&mut pending_tool_results, &mut messages);
                let content_blocks = anthropic_content_blocks(message, thinking_enabled)?;
                let mut blocks = content_blocks;

                if role == "assistant" {
                    for tc in &message.tool_calls {
                        let input_json = decode_tool_input(&tc.function.arguments)?;
                        blocks.push(serde_json::json!({
                            "type": "tool_use",
                            "id": provider_tool_call_id(&tc.id),
                            "name": tc.function.name.trim(),
                            "input": input_json,
                        }));
                    }
                }

                if blocks.is_empty() {
                    continue;
                }

                // Try to merge tool_use-only assistant messages with previous
                // assistant message that shares the same thinking block.
                if role == "assistant"
                    && merge_assistant_tool_use_with_previous(
                        &mut messages,
                        message,
                        &blocks,
                    )
                {
                    continue;
                }

                messages.push(AnthropicMessage {
                    role: role.to_string(),
                    content: blocks,
                });
            }
            _ => {
                flush_tool_results(&mut pending_tool_results, &mut messages);
                let text = message.content.trim();
                if text.is_empty() {
                    continue;
                }
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![serde_json::json!({
                        "type": "text",
                        "text": message.content,
                    })],
                });
            }
        }
    }
    flush_tool_results(&mut pending_tool_results, &mut messages);

    Ok(NormalizedMessages {
        system_parts,
        messages,
    })
}

/// Build the `system` field blocks for the request body.
pub(crate) fn build_system_blocks(system_parts: &[String]) -> Vec<serde_json::Value> {
    let mut blocks = Vec::new();
    if !system_parts.is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": system_parts.join("\n\n"),
        }));
    }
    blocks
}

/// Count how many provider-level messages correspond to the first
/// `stable_message_count` input messages (skipping system messages).
pub(crate) fn stable_provider_message_count(
    input: &[Message],
    stable_message_count: usize,
    thinking_enabled: bool,
) -> usize {
    if input.is_empty() || stable_message_count == 0 {
        return 0;
    }
    let mut stable_replay: Vec<Message> = Vec::with_capacity(stable_message_count);
    for msg in input {
        if msg.role.trim() == "system" {
            continue;
        }
        if stable_replay.len() >= stable_message_count {
            break;
        }
        stable_replay.push(msg.clone());
    }
    if stable_replay.is_empty() {
        return 0;
    }
    match normalize_messages(&stable_replay, thinking_enabled) {
        Ok(result) => result.messages.len(),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_image_content_parts(parts: &[xlate_core::types::ContentPart]) -> bool {
    parts
        .iter()
        .any(|p| matches!(p.kind, ContentPartType::Image))
}

/// Get effective text content: either `content` or collapsed text content parts.
fn effective_text_content(message: &Message) -> String {
    if !message.content.trim().is_empty() {
        return message.content.clone();
    }
    if !message.content_parts.is_empty() {
        return collapse_text_content_parts(&message.content_parts);
    }
    String::new()
}

fn collapse_text_content_parts(parts: &[xlate_core::types::ContentPart]) -> String {
    parts
        .iter()
        .filter(|p| matches!(p.kind, ContentPartType::Text))
        .filter(|p| !p.text.trim().is_empty())
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("")
}

/// Build Anthropic content blocks from a message, optionally prepending
/// a thinking block.
fn anthropic_content_blocks(
    message: &Message,
    thinking_enabled: bool,
) -> Result<Vec<serde_json::Value>, XlateError> {
    let mut blocks = raw_content_blocks(message)?;

    if should_include_thinking_block(message, thinking_enabled) {
        let mut thinking_block = serde_json::json!({
            "type": "thinking",
            "thinking": message.reasoning_content,
        });
        let sig = anthropic_thinking_signature(message);
        if !sig.is_empty() {
            thinking_block["signature"] = serde_json::Value::String(sig);
        }
        blocks.insert(0, thinking_block);
    }

    Ok(blocks)
}

/// Convert a message's content/content_parts into raw Anthropic blocks (text + image).
fn raw_content_blocks(message: &Message) -> Result<Vec<serde_json::Value>, XlateError> {
    if message.content_parts.is_empty() {
        if message.content.trim().is_empty() {
            return Ok(Vec::new());
        }
        return Ok(vec![serde_json::json!({
            "type": "text",
            "text": message.content,
        })]);
    }

    let mut blocks = Vec::with_capacity(message.content_parts.len());
    for part in &message.content_parts {
        match part.kind {
            ContentPartType::Text => {
                if part.text.is_empty() {
                    continue;
                }
                blocks.push(serde_json::json!({
                    "type": "text",
                    "text": part.text,
                }));
            }
            ContentPartType::Image => {
                let image = part.image.as_ref().ok_or_else(|| {
                    XlateError::InvalidRequest("image content part is missing image data".into())
                })?;
                let (data_b64, media_type) = resolve_image_content(image)?;
                blocks.push(serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data_b64,
                    },
                }));
            }
        }
    }

    // Fallback: if all content_parts produced nothing, use the raw content field.
    if blocks.is_empty() && !message.content.trim().is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    Ok(blocks)
}

fn resolve_image_content(
    image: &xlate_core::types::ImageContent,
) -> Result<(String, String), XlateError> {
    // If inline base64 data is present, use it directly.
    if let Some(ref data_b64) = image.data {
        if !data_b64.is_empty() {
            let media_type = normalize_image_mime_type(&image.mime_type, &image.path);
            return Ok((data_b64.clone(), media_type));
        }
    }
    // Otherwise try to read from path.
    let path = image.path.trim();
    if path.is_empty() {
        return Err(XlateError::InvalidRequest(
            "image content is missing data and path".into(),
        ));
    }
    let payload = std::fs::read(path).map_err(|e| {
        XlateError::InvalidRequest(format!("read image content failed: {e}"))
    })?;
    let media_type = normalize_image_mime_type(&image.mime_type, path);
    use base64::Engine;
    let data_b64 = base64::engine::general_purpose::STANDARD.encode(&payload);
    Ok((data_b64, media_type))
}

fn normalize_image_mime_type(mime_type: &str, path: &str) -> String {
    let trimmed = mime_type.trim().to_lowercase();
    if !trimmed.is_empty() {
        return trimmed;
    }
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg".to_string(),
        Some("png") => "image/png".to_string(),
        Some("gif") => "image/gif".to_string(),
        Some("webp") => "image/webp".to_string(),
        _ => "image/png".to_string(),
    }
}


fn should_include_thinking_block(message: &Message, thinking_enabled: bool) -> bool {
    if !thinking_enabled {
        return false;
    }
    if message.role.trim() != "assistant" {
        return false;
    }
    !message.reasoning_content.trim().is_empty()
}

fn anthropic_thinking_signature(message: &Message) -> String {
    let signature = message.reasoning_signature.trim();
    if signature.is_empty() {
        return String::new();
    }
    let source = message.reasoning_signature_source.trim();
    if source.is_empty() || source == "anthropic" {
        return signature.to_string();
    }
    String::new()
}

/// Try to merge an assistant message that only carries tool_use blocks (and shares
/// the same thinking block) with the previous assistant message.
fn merge_assistant_tool_use_with_previous(
    messages: &mut [AnthropicMessage],
    message: &Message,
    blocks: &[serde_json::Value],
) -> bool {
    if messages.is_empty() {
        return false;
    }
    if message.role.trim() != "assistant" || message.tool_calls.is_empty() {
        return false;
    }
    // Only merge if the message has no text/content_parts (just reasoning + tool_calls).
    if !message.content.trim().is_empty() || !message.content_parts.is_empty() {
        return false;
    }
    let reasoning = message.reasoning_content.trim();
    if reasoning.is_empty() {
        return false;
    }
    let signature = anthropic_thinking_signature(message);
    let last = messages.last().unwrap();
    if last.role.trim() != "assistant" {
        return false;
    }
    if !message_has_leading_thinking(last, reasoning, &signature) {
        return false;
    }
    // Collect non-thinking blocks.
    let tool_use_blocks: Vec<serde_json::Value> = blocks
        .iter()
        .filter(|b| {
            b.get("type")
                .and_then(|v| v.as_str())
                .map(|t| t.trim() != "thinking")
                .unwrap_or(true)
        })
        .cloned()
        .collect();
    if tool_use_blocks.is_empty() {
        return false;
    }
    let last_mut = messages.last_mut().unwrap();
    last_mut.content.extend(tool_use_blocks);
    true
}

fn message_has_leading_thinking(
    message: &AnthropicMessage,
    reasoning: &str,
    signature: &str,
) -> bool {
    if message.content.is_empty() {
        return false;
    }
    let first = &message.content[0];
    let block_type = first
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if block_type != "thinking" {
        return false;
    }
    let block_thinking = first
        .get("thinking")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let block_sig = first
        .get("signature")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    block_thinking == reasoning && block_sig == signature
}

fn decode_tool_input(arguments: &str) -> Result<serde_json::Value, XlateError> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(trimmed).map_err(|e| {
        XlateError::InvalidRequest(format!("decode anthropic tool input failed: {e}"))
    })
}

/// Extract the raw provider tool_call_id from the internal tool_call_id.
/// Simplified version: just returns the id trimmed and truncated to 64 chars.
fn provider_tool_call_id(tool_call_id: &str) -> String {
    let trimmed = tool_call_id.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // If it contains "::", take everything after the last "::"
    if let Some(pos) = trimmed.rfind("::") {
        let raw = trimmed[pos + 2..].trim();
        if !raw.is_empty() {
            return truncate_id(raw);
        }
    }
    truncate_id(trimmed)
}

fn truncate_id(id: &str) -> String {
    // Anthropic tool_use IDs have a max length limit.
    if id.len() <= 64 {
        id.to_string()
    } else {
        id[..64].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlate_core::types::Message;

    #[test]
    fn test_system_extraction() {
        let input = vec![
            Message {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                ..Default::default()
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                ..Default::default()
            },
        ];
        let result = normalize_messages(&input, false).unwrap();
        assert_eq!(result.system_parts, vec!["You are helpful."]);
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }

    #[test]
    fn test_tool_result_batching() {
        let input = vec![
            Message {
                role: "tool".to_string(),
                content: "result1".to_string(),
                tool_call_id: "call_1".to_string(),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: "result2".to_string(),
                tool_call_id: "call_2".to_string(),
                ..Default::default()
            },
        ];
        let result = normalize_messages(&input, false).unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content.len(), 2);
    }

    #[test]
    fn test_provider_tool_call_id() {
        assert_eq!(provider_tool_call_id("  abc  "), "abc");
        assert_eq!(provider_tool_call_id("ns::raw_id"), "raw_id");
        assert_eq!(provider_tool_call_id(""), "");
    }
}
