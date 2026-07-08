//! Message normalization for OpenAI Chat Completions API.
//!
//! Converts `xlate_core::Message` into the `[{"role":"...","content":...}]`
//! format that the OpenAI API expects, including content_parts / tool_calls /
//! reasoning_content handling.
//!
//! Ported from the original Go implementation's `normalizeOpenAIProviderMessages`,
//! `openAIContentValue`, and related helpers.

use xlate_core::types::{ContentPart, ContentPartType, ImageContent, Message};
use xlate_core::XlateError;

/// Convert a slice of internal `Message` into OpenAI Chat Completions format.
pub fn normalize_messages(
    messages: &[Message],
    thinking_enabled: bool,
) -> Result<Vec<serde_json::Value>, XlateError> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let mut items = Vec::with_capacity(messages.len());
    for message in messages {
        let content = openai_content_value(message)?;

        let mut item = serde_json::Map::new();
        item.insert("role".into(), serde_json::Value::String(message.role.trim().to_string()));
        item.insert("content".into(), content);

        // When thinking is enabled, assistant messages with tool_calls also need
        // explicit reasoning_content (even if empty).
        if should_include_reasoning_content(message, thinking_enabled) {
            item.insert(
                "reasoning_content".into(),
                serde_json::Value::String(message.reasoning_content.clone()),
            );
        }

        if !message.tool_calls.is_empty() {
            let tool_calls: Vec<serde_json::Value> = message
                .tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id.trim(),
                        "index": tc.index,
                        "type": tc.kind.trim(),
                        "function": {
                            "name": tc.function.name.trim(),
                            "arguments": tc.function.arguments,
                        }
                    })
                })
                .collect();
            item.insert("tool_calls".into(), serde_json::Value::Array(tool_calls));
        }

        if !message.tool_call_id.trim().is_empty() {
            item.insert(
                "tool_call_id".into(),
                serde_json::Value::String(message.tool_call_id.trim().to_string()),
            );
        }
        if !message.name.trim().is_empty() {
            item.insert(
                "name".into(),
                serde_json::Value::String(message.name.trim().to_string()),
            );
        }

        items.push(serde_json::Value::Object(item));
    }
    Ok(items)
}

fn should_include_reasoning_content(message: &Message, thinking_enabled: bool) -> bool {
    if !message.reasoning_content.trim().is_empty() {
        return true;
    }
    if !thinking_enabled {
        return false;
    }
    if message.role.trim() != "assistant" {
        return false;
    }
    !message.tool_calls.is_empty()
}

/// Build the `content` value for an OpenAI message.
/// If image parts are present, returns an array of content parts.
/// Otherwise returns a plain string.
fn openai_content_value(message: &Message) -> Result<serde_json::Value, XlateError> {
    if !has_image_content_parts(&message.content_parts) {
        let content = if message.content.trim().is_empty() && !message.content_parts.is_empty() {
            collapse_text_content_parts(&message.content_parts)
        } else {
            message.content.clone()
        };
        return Ok(serde_json::Value::String(content));
    }

    let mut parts = Vec::with_capacity(message.content_parts.len() + 1);

    if message.content_parts.is_empty() && !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
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
                    "type": "text",
                    "text": part.text,
                }));
            }
            ContentPartType::Image => {
                let data_url = image_content_data_url(part.image.as_ref())?;
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": data_url },
                }));
            }
        }
    }

    if parts.is_empty() {
        return Ok(serde_json::Value::String(message.content.clone()));
    }
    Ok(serde_json::Value::Array(parts))
}

pub(crate) fn has_image_content_parts(parts: &[ContentPart]) -> bool {
    parts.iter().any(|p| matches!(p.kind, ContentPartType::Image))
}

pub(crate) fn collapse_text_content_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter(|p| matches!(p.kind, ContentPartType::Text) && !p.text.trim().is_empty())
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("")
}

pub(crate) fn image_content_data_url(image: Option<&ImageContent>) -> Result<String, XlateError> {
    let image = image.ok_or_else(|| XlateError::InvalidRequest("image content is required".into()))?;

    let (data_b64, mime_type) = if let Some(ref b64) = image.data {
        let mime = normalize_image_mime_type(&image.mime_type, &image.path);
        (b64.clone(), mime)
    } else if !image.path.trim().is_empty() {
        // Read from filesystem
        let payload = std::fs::read(image.path.trim()).map_err(|e| {
            XlateError::InvalidRequest(format!("read image content failed: {e}"))
        })?;
        let mime = normalize_image_mime_type(&image.mime_type, &image.path);
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&payload);
        (b64, mime)
    } else {
        return Err(XlateError::InvalidRequest(
            "image content is missing data and path".into(),
        ));
    };

    Ok(format!("data:{mime_type};base64,{data_b64}"))
}

fn normalize_image_mime_type(mime_type: &str, path: &str) -> String {
    let trimmed = mime_type.trim().to_lowercase();
    if !trimmed.is_empty() {
        return trimmed;
    }
    // Fallback to extension detection
    let ext = path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg".to_string(),
        "png" => "image/png".to_string(),
        "gif" => "image/gif".to_string(),
        "webp" => "image/webp".to_string(),
        _ => "image/png".to_string(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use xlate_core::types::ToolCallDescriptor;
    use xlate_core::types::ToolCallFunctionShape;

    fn user_message(text: &str) -> Message {
        Message {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }
    }

    fn assistant_message(text: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn normalizes_basic_messages() {
        let messages = vec![user_message("hello"), assistant_message("world")];
        let result = normalize_messages(&messages, false).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "hello");
        assert_eq!(result[1]["role"], "assistant");
        assert_eq!(result[1]["content"], "world");
    }

    #[test]
    fn includes_tool_calls() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCallDescriptor {
                id: "call_1".into(),
                index: 0,
                kind: "function".into(),
                function: ToolCallFunctionShape {
                    name: "read_file".into(),
                    arguments: r#"{"path":"test.txt"}"#.into(),
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = normalize_messages(&[msg], false).unwrap();
        assert!(result[0].get("tool_calls").is_some());
        let tc = &result[0]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "read_file");
    }

    #[test]
    fn includes_reasoning_for_thinking_enabled_with_tool_calls() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCallDescriptor {
                id: "call_1".into(),
                index: 0,
                kind: "function".into(),
                function: ToolCallFunctionShape {
                    name: "test".into(),
                    arguments: "{}".into(),
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = normalize_messages(&[msg], true).unwrap();
        assert!(result[0].get("reasoning_content").is_some());
    }

    #[test]
    fn includes_tool_call_id() {
        let msg = Message {
            role: "tool".into(),
            content: "result".into(),
            tool_call_id: "call_1".into(),
            ..Default::default()
        };
        let result = normalize_messages(&[msg], false).unwrap();
        assert_eq!(result[0]["tool_call_id"], "call_1");
    }
}
