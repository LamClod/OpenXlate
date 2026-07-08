//! Provider-specific logic for disabling thinking/reasoning.
//!
//! Different providers use different parameter formats to disable thinking:
//! - DeepSeek/GLM/Z.AI: `{"thinking": {"type": "disabled"}}`
//! - Qwen/DashScope/Aliyun: `{"enable_thinking": false}`
//! - GPT-5.1+/GPT-6+: `{"reasoning_effort": "none"}` (chat completions)
//!
//! Ported from the original Go implementation's `applyOpenAIThinkingDisable` and
//! `openAIThinkingDisableKind`.

use crate::endpoint;

/// Normalize a runtime thinking effort string to a canonical form.
pub fn normalize_thinking_effort(raw: &str) -> &str {
    match raw.trim().to_lowercase().as_str() {
        "disabled" | "low" | "medium" | "high" | "xhigh" | "max" => {
            // Return the lowercase trimmed version — we match in the caller
            // and return a static str for the common cases.
            match raw.trim().to_lowercase().as_str() {
                "disabled" => "disabled",
                "low" => "low",
                "medium" => "medium",
                "high" => "high",
                "xhigh" => "xhigh",
                "max" => "max",
                _ => "",
            }
        }
        "disable" | "off" | "none" | "false" | "no" | "0" => "disabled",
        "very_high" | "very-high" | "veryhigh" | "x-high" | "extra_high" | "extra-high"
        | "extrahigh" => "xhigh",
        "maximum" => "max",
        _ => "",
    }
}

/// The kind of parameter to use when disabling thinking.
#[derive(Debug, PartialEq, Eq)]
enum ThinkingDisableKind {
    /// `{"thinking": {"type": "disabled"}}`
    ThinkingType,
    /// `{"enable_thinking": false}`
    EnableThinking,
    /// `{"reasoning_effort": "none"}` (chat completions) or
    /// `{"reasoning": {"effort": "none"}}` (responses)
    ReasoningNone,
    /// No known disable mechanism for this provider.
    None,
}

fn thinking_disable_kind(base_url: &str, model_id: &str) -> ThinkingDisableKind {
    let base = base_url.trim().to_lowercase();
    let model = model_id.trim().to_lowercase();

    if base.contains("dashscope")
        || base.contains("qwen")
        || base.contains("aliyun")
        || model.contains("qwen")
    {
        return ThinkingDisableKind::EnableThinking;
    }

    if base.contains("deepseek")
        || base.contains("bigmodel")
        || base.contains("z.ai")
        || base.contains("zhipu")
        || model.contains("deepseek")
        || model.contains("glm")
        || model.contains("zai")
        || model.contains("zhipu")
    {
        return ThinkingDisableKind::ThinkingType;
    }

    if model_supports_reasoning_none(&model) {
        return ThinkingDisableKind::ReasoningNone;
    }

    ThinkingDisableKind::None
}

fn model_supports_reasoning_none(model: &str) -> bool {
    let model = model.trim().to_lowercase();
    if model.starts_with("gpt-6") {
        return true;
    }
    if model.contains("gpt-5.1") {
        return true;
    }
    if !model.starts_with("gpt-5.") {
        return false;
    }
    let minor_text = &model["gpt-5.".len()..];
    let minor_end = minor_text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(minor_text.len());
    if minor_end == 0 {
        return false;
    }
    match minor_text[..minor_end].parse::<u32>() {
        Ok(minor) => minor >= 1,
        Err(_) => false,
    }
}

/// Apply thinking disable parameters to a request body map.
///
/// Only acts when the user's thinking effort is "disabled".
pub fn apply_thinking_disable(
    body: &mut serde_json::Map<String, serde_json::Value>,
    thinking_effort: &str,
    base_url: &str,
    model_id: &str,
    openai_endpoint: &str,
) {
    if body.is_empty() || normalize_thinking_effort(thinking_effort) != "disabled" {
        return;
    }

    match thinking_disable_kind(base_url, model_id) {
        ThinkingDisableKind::ThinkingType => {
            body.insert(
                "thinking".into(),
                serde_json::json!({"type": "disabled"}),
            );
            body.remove("reasoning_effort");
            tracing::debug!("thinking disabled via thinking.type=disabled");
        }
        ThinkingDisableKind::EnableThinking => {
            body.insert("enable_thinking".into(), serde_json::Value::Bool(false));
            body.remove("reasoning_effort");
            tracing::debug!("thinking disabled via enable_thinking=false");
        }
        ThinkingDisableKind::ReasoningNone => {
            if endpoint::endpoint_shape(openai_endpoint) == "responses" {
                body.insert(
                    "reasoning".into(),
                    serde_json::json!({"effort": "none"}),
                );
            } else {
                body.insert(
                    "reasoning_effort".into(),
                    serde_json::Value::String("none".into()),
                );
            }
            tracing::debug!("thinking disabled via reasoning.effort=none");
        }
        ThinkingDisableKind::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_thinking_effort() {
        assert_eq!(normalize_thinking_effort("disabled"), "disabled");
        assert_eq!(normalize_thinking_effort("off"), "disabled");
        assert_eq!(normalize_thinking_effort("none"), "disabled");
        assert_eq!(normalize_thinking_effort("false"), "disabled");
        assert_eq!(normalize_thinking_effort("low"), "low");
        assert_eq!(normalize_thinking_effort("very_high"), "xhigh");
        assert_eq!(normalize_thinking_effort("maximum"), "max");
        assert_eq!(normalize_thinking_effort("random"), "");
    }

    #[test]
    fn test_thinking_disable_kind_deepseek() {
        assert_eq!(
            thinking_disable_kind("https://api.deepseek.com/v1", "deepseek-reasoner"),
            ThinkingDisableKind::ThinkingType
        );
    }

    #[test]
    fn test_thinking_disable_kind_qwen() {
        assert_eq!(
            thinking_disable_kind("https://dashscope.aliyuncs.com", "qwen-max"),
            ThinkingDisableKind::EnableThinking
        );
    }

    #[test]
    fn test_thinking_disable_kind_gpt6() {
        assert_eq!(
            thinking_disable_kind("https://api.openai.com/v1", "gpt-6"),
            ThinkingDisableKind::ReasoningNone
        );
    }

    #[test]
    fn test_model_supports_reasoning_none() {
        assert!(model_supports_reasoning_none("gpt-6"));
        assert!(model_supports_reasoning_none("gpt-5.1"));
        assert!(model_supports_reasoning_none("gpt-5.2"));
        assert!(!model_supports_reasoning_none("gpt-5.0"));
        assert!(!model_supports_reasoning_none("gpt-4o"));
    }

    #[test]
    fn test_apply_thinking_disable_deepseek() {
        let mut body = serde_json::Map::new();
        body.insert(
            "reasoning_effort".into(),
            serde_json::Value::String("high".into()),
        );
        apply_thinking_disable(&mut body, "disabled", "https://api.deepseek.com", "deepseek-r1", "/v1/chat/completions");
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(!body.contains_key("reasoning_effort"));
    }
}
