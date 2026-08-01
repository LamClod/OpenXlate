//! Anthropic (Claude) shim — IR ↔ Anthropic Messages API
//!
//! 零拷贝：encode 直接 to_vec，decode 直接 from_slice + Cow::Borrowed
//! 支持 thinking/redacted_thinking、cache_control、tool_result 多块内容

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::{
    DecodeRequest, DecodeResponse, DecodeStream, EncodeRequest, EncodeResponse, EncodeStream,
};
use crate::codec::error::CodecError;
use crate::codec::ir::*;

// ─── Anthropic 有线格式（编码）────────────────────────────────────

#[derive(Serialize)]
struct AntRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<serde_json::Value>,
    messages: Vec<AntMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [Cow<'a, str>]>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
}

#[derive(Serialize)]
struct AntMessage {
    role: &'static str,
    content: serde_json::Value,
}

// ─── 反序列化（响应）────────────────────────────────────────────────

#[derive(Deserialize)]
struct AntResponse<'a> {
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow, default)]
    model: Cow<'a, str>,
    #[serde(borrow, default)]
    #[allow(dead_code)]
    r#type: Cow<'a, str>,
    #[serde(default)]
    content: Vec<AntContentBlock<'a>>,
    #[serde(borrow, default)]
    stop_reason: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    stop_sequence: Option<Cow<'a, str>>,
    usage: Option<AntUsage>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AntContentBlock<'a> {
    Text {
        #[serde(borrow)]
        text: Cow<'a, str>,
        #[serde(default)]
        citations: Option<Vec<AntCitation<'a>>>,
    },
    Thinking {
        #[serde(borrow)]
        thinking: Cow<'a, str>,
        #[serde(borrow, default)]
        signature: Option<Cow<'a, str>>,
    },
    RedactedThinking {
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    ToolUse {
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        name: Cow<'a, str>,
        input: serde_json::Value,
    },
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Anthropic citation 结构体
#[derive(Deserialize)]
struct AntCitation<'a> {
    #[serde(borrow, default)]
    r#type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    url: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    title: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    cited_text: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    encrypted_index: Option<Cow<'a, str>>,
}

/// SSE 事件包装
#[derive(Deserialize)]
struct AntSseEvent<'a> {
    #[serde(borrow)]
    r#type: Cow<'a, str>,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    content_block: Option<AntContentBlock<'a>>,
    #[serde(default)]
    delta: Option<AntDelta<'a>>,
    #[serde(default)]
    message: Option<AntSseMessage<'a>>,
    #[serde(default)]
    usage: Option<AntUsage>,
}

#[derive(Deserialize)]
struct AntSseMessage<'a> {
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow)]
    model: Cow<'a, str>,
    usage: Option<AntUsage>,
}

/// content_block_delta 的 delta 子类型。
///
/// untagged 按字段名区分（text/thinking/partial_json/signature 互斥），
/// 使 message_delta 事件的 delta 载荷（{"stop_reason":...}，无 type 字段）
/// 也能落入 Unknown 而不使整个事件反序列化失败。
#[derive(Deserialize)]
#[serde(untagged)]
enum AntDelta<'a> {
    TextDelta {
        #[serde(borrow)]
        text: Cow<'a, str>,
    },
    ThinkingDelta {
        #[serde(borrow)]
        thinking: Cow<'a, str>,
    },
    InputJsonDelta {
        #[serde(borrow)]
        partial_json: Cow<'a, str>,
    },
    /// thinking 块的签名，在块尾部到达
    SignatureDelta {
        #[serde(borrow)]
        signature: Cow<'a, str>,
    },
    /// message_delta 事件的 stop_reason（避免二次 JSON 解析）
    MessageDelta {
        #[serde(borrow)]
        stop_reason: Cow<'a, str>,
    },
    CitationsDelta {
        citation: serde_json::Value,
    },
    Unknown(serde::de::IgnoredAny),
}

#[derive(Deserialize, Clone)]
struct AntUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

// ─── 实现 ──────────────────────────────────────────────────────────

pub struct AnthropicShim;

impl AnthropicShim {
    fn ir_content_to_ant_blocks(
        content: &IrContent<'_>,
        cache_control: Option<&IrCacheControl<'_>>,
        preserved: &mut Vec<serde_json::Value>,
    ) -> serde_json::Value {
        let cc_val = cache_control.map(|cc| {
            let mut obj = serde_json::json!({ "type": cc.r#type });
            if let Some(ref ttl) = cc.ttl {
                obj["ttl"] = serde_json::Value::String(ttl.to_string());
            }
            obj
        });

        match content {
            IrContent::Text(s) => {
                if let Some(cc) = cc_val {
                    // 需要 array 格式才能附加 cache_control
                    serde_json::json!([{
                        "type": "text",
                        "text": s,
                        "cache_control": cc,
                    }])
                } else {
                    serde_json::json!(s)
                }
            }
            IrContent::Parts(parts) => {
                let mut blocks: Vec<serde_json::Value> = parts
                    .iter()
                    .filter_map(|p| match p {
                        IrContentPart::Text { text, .. } => {
                            Some(serde_json::json!({ "type": "text", "text": text }))
                        }
                        IrContentPart::ImageUrl { url, .. } => Some(serde_json::json!({
                            "type": "image",
                            "source": { "type": "url", "url": url }
                        })),
                        IrContentPart::ImageBase64 {
                            media_type, data, ..
                        } => Some(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        })),
                        IrContentPart::Document {
                            media_type,
                            data,
                            filename,
                        } => {
                            let mut block = serde_json::json!({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            });
                            if let Some(ref name) = filename {
                                block["title"] = serde_json::Value::String(name.to_string());
                            }
                            Some(block)
                        }
                        IrContentPart::Reasoning { text, signature } => {
                            let mut block = serde_json::json!({
                                "type": "thinking",
                                "thinking": text,
                            });
                            if let Some(sig) = signature {
                                block["signature"] = serde_json::Value::String(sig.to_string());
                            }
                            Some(block)
                        }
                        IrContentPart::RedactedReasoning { data } => Some(serde_json::json!({
                            "type": "redacted_thinking",
                            "data": data,
                        })),
                        IrContentPart::FileRef { file_id } => {
                            // Anthropic Files API 引用
                            Some(serde_json::json!({
                                "type": "document",
                                "source": { "type": "file", "file_id": file_id }
                            }))
                        }
                        IrContentPart::FunctionCall {
                            id,
                            name,
                            arguments,
                        } => {
                            let input: serde_json::Value =
                                serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
                            Some(serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }))
                        }
                        IrContentPart::FunctionResponse { id, response, .. } => {
                            Some(serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": response.as_str()
                                    .map(|s| serde_json::json!(s))
                                    .unwrap_or_else(|| serde_json::json!(response.to_string())),
                            }))
                        }
                        // Audio/Video — Anthropic 不支持，无损保留
                        IrContentPart::Audio { .. } | IrContentPart::Video { .. } => {
                            if let Ok(v) = serde_json::to_value(p) {
                                preserved.push(v);
                            }
                            None
                        }
                        // 同源 Opaque → 直接回写为原生块
                        IrContentPart::Opaque {
                            provider, payload, ..
                        } if provider == "anthropic" => Some(payload.clone()),
                        IrContentPart::Opaque { .. } => {
                            if let Ok(v) = serde_json::to_value(p) {
                                preserved.push(v);
                            }
                            None
                        }
                    })
                    .collect();
                // 最后一个块附加 cache_control
                if let Some(cc) = cc_val {
                    if let Some(last) = blocks.last_mut() {
                        last["cache_control"] = cc;
                    }
                }
                serde_json::Value::Array(blocks)
            }
        }
    }

    fn parse_stop_reason(s: &str) -> IrFinishReason {
        match s {
            "end_turn" => IrFinishReason::Stop,
            "stop_sequence" => IrFinishReason::StopSequence,
            "max_tokens" => IrFinishReason::Length,
            "tool_use" => IrFinishReason::ToolCalls,
            "pause_turn" => IrFinishReason::PauseTurn,
            "refusal" => IrFinishReason::ContentFilter,
            _ => IrFinishReason::Stop,
        }
    }

    fn convert_usage(u: &AntUsage) -> IrUsage {
        let input = u.input_tokens.unwrap_or(0);
        let output = u.output_tokens.unwrap_or(0);
        IrUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input.saturating_add(output),
            cache_read_tokens: u.cache_read_input_tokens,
            cache_creation_tokens: u.cache_creation_input_tokens,
            reasoning_tokens: None,
            audio_tokens: None,
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }

    /// 解析 IrTool 对应的 Anthropic 工具 `type` 字符串。
    ///
    /// 返回 None → 应作为 custom（函数）工具编码（`input_schema` 形式）；
    /// 返回 Some(t) → 特殊工具（computer/text_editor/bash/mcp），按 `type` + 专有字段编码。
    /// 优先取 `extra.type` 保留的精确版本串（如 `computer_20241022`），
    /// 缺失时按 `tool_type` 回退到默认版本串。此设计使 bash（无独立 IrToolType，
    /// 解码为 Function）也能凭 `extra.type` 无损往返。
    fn ant_tool_type_str(t: &IrTool<'_>) -> Option<String> {
        if let Some(serde_json::Value::Object(map)) = t.extra.as_ref() {
            if let Some(serde_json::Value::String(s)) = map.get("type") {
                return if s == "custom" { None } else { Some(s.clone()) };
            }
        }
        match t.tool_type {
            IrToolType::ComputerUse => Some("computer_20241022".to_string()),
            IrToolType::TextEditor => Some("text_editor_20241022".to_string()),
            IrToolType::WebSearch => Some("web_search_20250305".to_string()),
            IrToolType::CodeInterpreter => Some("code_execution_20250522".to_string()),
            IrToolType::Mcp => Some("mcp".to_string()),
            _ => None,
        }
    }
}

impl EncodeRequest for AnthropicShim {
    fn encode_request(&self, ir: &IrRequest<'_>) -> Result<Vec<u8>, CodecError> {
        super::validate_tool_arguments(&ir.messages)?;
        // 无损保留：收集 Anthropic 不支持的 IR 内容部件
        let mut preserved: Vec<serde_json::Value> = Vec::new();

        // 提取 system messages
        let mut system_parts: Vec<serde_json::Value> = Vec::new();
        let mut messages: Vec<AntMessage> = Vec::new();

        for msg in &ir.messages {
            if msg.role == Role::System || msg.role == Role::Developer {
                // 非文本部件无法放入 system → 保留到 preserved
                if let IrContent::Parts(ref parts) = msg.content {
                    for p in parts {
                        match p {
                            IrContentPart::Text { .. } => {}
                            _ => {
                                if let Ok(v) = serde_json::to_value(p) {
                                    preserved.push(v);
                                }
                            }
                        }
                    }
                }
                let text = msg.content.text_concat();
                let mut block = serde_json::json!({ "type": "text", "text": text });
                if let Some(ref cc) = msg.cache_control {
                    let mut cc_obj = serde_json::json!({ "type": cc.r#type });
                    if let Some(ref ttl) = cc.ttl {
                        cc_obj["ttl"] = serde_json::Value::String(ttl.to_string());
                    }
                    block["cache_control"] = cc_obj;
                }
                system_parts.push(block);
                continue;
            }

            let role_str = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "assistant",
                Role::System | Role::Developer => unreachable!(),
            };

            let content = if msg.role == Role::Tool {
                // tool result → tool_result 块，支持多块内容
                let tool_use_id = msg.tool_call_id.as_deref().unwrap_or("");
                let result_content = match &msg.content {
                    IrContent::Text(s) => serde_json::json!(s),
                    IrContent::Parts(parts) => {
                        let blocks: Vec<serde_json::Value> = parts
                            .iter()
                            .filter_map(|p| match p {
                                IrContentPart::Text { text, .. } => {
                                    Some(serde_json::json!({ "type": "text", "text": text }))
                                }
                                IrContentPart::ImageBase64 {
                                    media_type, data, ..
                                } => Some(serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                })),
                                IrContentPart::Document {
                                    media_type,
                                    data,
                                    filename,
                                } => {
                                    let mut block = serde_json::json!({
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data
                                        }
                                    });
                                    if let Some(ref name) = filename {
                                        block["title"] =
                                            serde_json::Value::String(name.to_string());
                                    }
                                    Some(block)
                                }
                                // 同源 Opaque → 直接回写为原生块
                                IrContentPart::Opaque {
                                    provider, payload, ..
                                } if provider == "anthropic" => Some(payload.clone()),
                                // tool_result 不支持的变体 — 无损保留
                                IrContentPart::ImageUrl { .. }
                                | IrContentPart::FileRef { .. }
                                | IrContentPart::Audio { .. }
                                | IrContentPart::Video { .. }
                                | IrContentPart::Opaque { .. }
                                | IrContentPart::Reasoning { .. }
                                | IrContentPart::RedactedReasoning { .. }
                                | IrContentPart::FunctionCall { .. }
                                | IrContentPart::FunctionResponse { .. } => {
                                    if let Ok(v) = serde_json::to_value(p) {
                                        preserved.push(v);
                                    }
                                    None
                                }
                            })
                            .collect();
                        serde_json::Value::Array(blocks)
                    }
                };
                let mut tool_result = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                });
                // cache_control 附加到 tool_result 块
                if let Some(ref cc) = msg.cache_control {
                    let mut cc_obj = serde_json::json!({ "type": cc.r#type });
                    if let Some(ref ttl) = cc.ttl {
                        cc_obj["ttl"] = serde_json::Value::String(ttl.to_string());
                    }
                    tool_result["cache_control"] = cc_obj;
                }
                serde_json::json!([tool_result])
            } else if let Some(ref tcs) = msg.tool_calls {
                // assistant with tool_use blocks
                let mut blocks = Vec::new();
                // 先输出 content 部件（包括 thinking 块）
                match &msg.content {
                    IrContent::Text(s) => {
                        if !s.is_empty() {
                            blocks.push(serde_json::json!({ "type": "text", "text": s }));
                        }
                    }
                    IrContent::Parts(parts) => {
                        for p in parts {
                            match p {
                                IrContentPart::Text { text, .. } => {
                                    if !text.is_empty() {
                                        blocks.push(
                                            serde_json::json!({ "type": "text", "text": text }),
                                        );
                                    }
                                }
                                IrContentPart::Reasoning { text, signature } => {
                                    let mut block = serde_json::json!({
                                        "type": "thinking",
                                        "thinking": text,
                                    });
                                    if let Some(sig) = signature {
                                        block["signature"] =
                                            serde_json::Value::String(sig.to_string());
                                    }
                                    blocks.push(block);
                                }
                                IrContentPart::RedactedReasoning { data } => {
                                    blocks.push(serde_json::json!({
                                        "type": "redacted_thinking",
                                        "data": data,
                                    }));
                                }
                                // 同源 Opaque → 直接回写为原生块
                                IrContentPart::Opaque {
                                    provider, payload, ..
                                } if provider == "anthropic" => {
                                    blocks.push(payload.clone());
                                }
                                // assistant + tool_calls 不支持的多模态变体 — 无损保留
                                IrContentPart::ImageUrl { .. }
                                | IrContentPart::ImageBase64 { .. }
                                | IrContentPart::Document { .. }
                                | IrContentPart::FileRef { .. }
                                | IrContentPart::Audio { .. }
                                | IrContentPart::Video { .. }
                                | IrContentPart::Opaque { .. }
                                | IrContentPart::FunctionCall { .. }
                                | IrContentPart::FunctionResponse { .. } => {
                                    if let Ok(v) = serde_json::to_value(p) {
                                        preserved.push(v);
                                    }
                                }
                            }
                        }
                    }
                }
                for tc in tcs {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input
                    }));
                }
                // cache_control 附加到最后一个内容块
                if let Some(ref cc) = msg.cache_control {
                    let mut cc_obj = serde_json::json!({ "type": cc.r#type });
                    if let Some(ref ttl) = cc.ttl {
                        cc_obj["ttl"] = serde_json::Value::String(ttl.to_string());
                    }
                    if let Some(last) = blocks.last_mut() {
                        last["cache_control"] = cc_obj;
                    }
                }
                serde_json::Value::Array(blocks)
            } else {
                Self::ir_content_to_ant_blocks(
                    &msg.content,
                    msg.cache_control.as_ref(),
                    &mut preserved,
                )
            };

            messages.push(AntMessage {
                role: role_str,
                content,
            });
        }

        let system = if system_parts.is_empty() {
            None
        } else if system_parts.len() == 1 && system_parts[0].get("cache_control").is_none() {
            // 单条无 cache_control 的 system 直接用字符串
            system_parts[0].get("text").cloned()
        } else {
            Some(serde_json::Value::Array(system_parts))
        };

        let tools: Option<Vec<serde_json::Value>> = ir.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| {
                    let cc_val = t.cache_control.as_ref().map(|cc| {
                        let mut obj = serde_json::json!({ "type": cc.r#type });
                        if let Some(ref ttl) = cc.ttl {
                            obj["ttl"] = serde_json::Value::String(ttl.to_string());
                        }
                        obj
                    });
                    match Self::ant_tool_type_str(t) {
                        None => {
                            // custom（函数）工具：input_schema 形式
                            let mut obj = serde_json::json!({
                                "name": t.name,
                                "input_schema": t.parameters,
                            });
                            if let Some(desc) = t.description.as_deref() {
                                obj["description"] = serde_json::Value::String(desc.to_string());
                            }
                            if let Some(serde_json::Value::Object(extra)) = t.extra.as_ref() {
                                for (k, v) in extra {
                                    if k == "name"
                                        || k == "description"
                                        || k == "input_schema"
                                        || k == "cache_control"
                                    {
                                        continue;
                                    }
                                    obj[k.as_str()] = v.clone();
                                }
                            }
                            if let Some(cc) = cc_val {
                                obj["cache_control"] = cc;
                            }
                            obj
                        }
                        Some(type_str) => {
                            // 特殊工具（computer/text_editor/bash/mcp）：
                            // type + name + 工具专有字段（不含 input_schema）
                            let mut obj = serde_json::json!({
                                "type": type_str,
                                "name": t.name,
                            });
                            if let Some(serde_json::Value::Object(map)) = t.extra.as_ref() {
                                for (k, v) in map {
                                    if k == "type" {
                                        continue;
                                    }
                                    obj[k.as_str()] = v.clone();
                                }
                            }
                            if let Some(cc) = cc_val {
                                obj["cache_control"] = cc;
                            }
                            obj
                        }
                    }
                })
                .collect()
        });

        let tool_choice = ir.tool_choice.as_ref().map(|tc| {
            let mut val = match tc {
                IrToolChoice::Auto => serde_json::json!({ "type": "auto" }),
                IrToolChoice::None => serde_json::json!({ "type": "none" }),
                IrToolChoice::Required => serde_json::json!({ "type": "any" }),
                IrToolChoice::Specific { name } => serde_json::json!({
                    "type": "tool",
                    "name": name
                }),
            };
            if ir.parallel_tool_calls == Some(false) {
                val["disable_parallel_tool_use"] = serde_json::json!(true);
            }
            val
        });
        // If no tool_choice but parallel_tool_calls == false, emit tool_choice with disable
        let tool_choice =
            if tool_choice.is_none() && ir.parallel_tool_calls == Some(false) && ir.tools.is_some()
            {
                Some(serde_json::json!({
                    "type": "auto",
                    "disable_parallel_tool_use": true,
                }))
            } else {
                tool_choice
            };

        // reasoning → thinking 配置
        let thinking = ir.reasoning.as_ref().and_then(|r| {
            if r.mode == ReasoningMode::Disabled {
                None
            } else {
                let mut obj = serde_json::json!({ "type": "enabled" });
                // Anthropic 在 thinking 启用时强制要求 budget_tokens；缺省（如从只带
                // reasoning_effort 的 OpenAI/Responses 格式转换而来）会返回 400。
                // 因此 budget_tokens 为空时按 effort 推导一个默认值。
                let effective_max = ir.max_tokens.unwrap_or(4096);
                let mut budget = match r.budget_tokens {
                    Some(b) => b,
                    None => match r.effort.as_deref() {
                        Some("low") | Some("minimal") => 2048,
                        Some("high") => 16384,
                        _ => 8192, // "medium" 或未指定
                    },
                };
                // budget_tokens 必须严格小于 max_tokens
                if effective_max > 0 && budget >= effective_max {
                    budget = effective_max - 1;
                }
                obj["budget_tokens"] = serde_json::json!(budget);
                Some(obj)
            }
        });

        let effective_max = ir.max_tokens.unwrap_or(4096);

        // metadata.user_id
        let metadata = ir.metadata.as_ref().and_then(|m| {
            m.user_id
                .as_ref()
                .map(|uid| serde_json::json!({ "user_id": uid }))
        });

        // service_tier
        let service_tier = ir.metadata.as_ref().and_then(|m| m.service_tier.as_deref());

        let req = AntRequest {
            model: &ir.model,
            max_tokens: effective_max,
            system,
            messages,
            temperature: ir.temperature,
            top_p: ir.top_p,
            top_k: ir.top_k,
            stop_sequences: ir.stop.as_deref(),
            stream: ir.stream,
            tools,
            tool_choice,
            thinking,
            metadata,
            service_tier,
        };

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        let mut body = serde_json::to_value(&req)?;
        super::attach_preserved(&mut body, preserved);
        serde_json::to_vec(&body).map_err(CodecError::from)
    }

    fn endpoint(&self, base_url: &str) -> String {
        format!("{base_url}/v1/messages")
    }

    fn headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![
            ("x-api-key", api_key.to_string()),
            ("anthropic-version", "2023-06-01".into()),
            ("content-type", "application/json".into()),
        ]
    }
}

impl DecodeResponse for AnthropicShim {
    fn decode_response<'a>(&self, body: &'a [u8]) -> Result<IrResponse<'a>, CodecError> {
        let raw_json: serde_json::Value = serde_json::from_slice(body)?;
        let preserved = super::extract_preserved(&raw_json);
        let ant: AntResponse<'a> = serde_json::from_slice(body)?;

        let mut parts: Vec<IrContentPart<'a>> = Vec::new();

        for block in ant.content {
            match block {
                AntContentBlock::Text { text, citations } => {
                    let ir_citations = citations.map(|cits| {
                        cits.into_iter()
                            .map(|c| IrCitation {
                                r#type: c.r#type.unwrap_or(Cow::Borrowed("")),
                                url: c.url,
                                title: c.title,
                                cited_text: c.cited_text,
                                encrypted_index: c.encrypted_index,
                            })
                            .collect()
                    });
                    parts.push(IrContentPart::Text {
                        text,
                        citations: ir_citations,
                    });
                }
                AntContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    parts.push(IrContentPart::Reasoning {
                        text: thinking,
                        signature,
                    });
                }
                AntContentBlock::RedactedThinking { data } => {
                    parts.push(IrContentPart::RedactedReasoning { data });
                }
                AntContentBlock::ToolUse { id, name, input } => {
                    let args = serde_json::to_string(&input).unwrap_or_default();
                    parts.push(IrContentPart::FunctionCall {
                        id,
                        name,
                        arguments: Cow::Owned(args),
                    });
                }
                AntContentBlock::Unknown(raw) => {
                    parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("anthropic"),
                        payload: raw,
                    });
                }
            }
        }

        let content = match parts.len() {
            0 => IrContent::Text(Cow::Borrowed("")),
            1 => {
                if matches!(
                    &parts[0],
                    IrContentPart::Text {
                        citations: None,
                        ..
                    }
                ) {
                    if let IrContentPart::Text { text, .. } = parts.remove(0) {
                        IrContent::Text(text)
                    } else {
                        unreachable!()
                    }
                } else {
                    IrContent::Parts(parts)
                }
            }
            _ => IrContent::Parts(parts),
        };

        let message = IrMessage {
            role: Role::Assistant,
            content,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            cache_control: None,
            refusal: None,
        };

        let finish_reason = ant.stop_reason.as_deref().map(Self::parse_stop_reason);

        let usage = ant.usage.as_ref().map(Self::convert_usage);

        // service_tier / stop_sequence → provider_metadata
        let mut pm_map = serde_json::Map::new();
        if let Some(ref st) = ant.service_tier {
            pm_map.insert("service_tier".into(), serde_json::json!(st));
        }
        if let Some(ref ss) = ant.stop_sequence {
            pm_map.insert("stop_sequence".into(), serde_json::json!(ss));
        }
        let mut provider_metadata = if pm_map.is_empty() {
            None
        } else {
            Some(Box::new(serde_json::Value::Object(pm_map)))
        };
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        Ok(IrResponse {
            id: ant.id,
            model: ant.model,
            choices: vec![IrChoice {
                index: 0,
                message,
                finish_reason,
                logprobs: None,
            }],
            usage,
            provider_metadata,
        })
    }
}

/// 流式块类型 — Anthropic 解码器用于精确区分 content_block_stop
#[derive(Clone)]
enum AntBlockKind {
    Text,
    Thinking {
        signature: Option<String>,
    },
    ToolUse {
        seq: u32,
        id: String,
        name: String,
        args: String,
    },
    Other(serde_json::Value),
}

/// Anthropic 流式解码器 — 每条 SSE 流一个实例。
/// 跟踪各 index 的块类型，将 content_block_stop 精确映射为
/// ContentDone / ReasoningDone(+signature) / ToolCallDone(+完整 arguments)。
pub struct AntStreamDecoder {
    started: bool,
    finished: bool,
    choice_finished: bool,
    /// 线上块索引 → 块类型（含累积状态）
    blocks: Vec<(u32, AntBlockKind)>,
    /// 已分配的工具序号计数（tool 事件的 index 输出全流唯一序号）
    tool_seq: u32,
    /// message_start 携带的 usage（含 cache token），供 message_delta 合并
    base_usage: Option<IrUsage>,
}

impl AntStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            finished: false,
            choice_finished: false,
            blocks: Vec::new(),
            tool_seq: 0,
            base_usage: None,
        }
    }

    fn block_mut(&mut self, index: u32) -> Option<&mut AntBlockKind> {
        self.blocks
            .iter_mut()
            .find(|(i, _)| *i == index)
            .map(|(_, k)| k)
    }

    fn set_block(&mut self, index: u32, kind: AntBlockKind) {
        if let Some(k) = self.block_mut(index) {
            *k = kind;
        } else {
            self.blocks.push((index, kind));
        }
    }

    fn take_block(&mut self, index: u32) -> Option<AntBlockKind> {
        self.blocks
            .iter()
            .position(|(i, _)| *i == index)
            .map(|pos| self.blocks.remove(pos).1)
    }

    fn finalize(&mut self) -> Vec<IrStreamEvent<'static>> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();
        for (_, block) in self.blocks.drain(..) {
            match block {
                AntBlockKind::Text => events.push(IrStreamEvent::ContentDone { index: 0 }),
                AntBlockKind::Thinking { signature } => {
                    events.push(IrStreamEvent::ReasoningDone {
                        index: 0,
                        signature: signature.map(Cow::Owned),
                    });
                }
                AntBlockKind::ToolUse {
                    seq,
                    id,
                    name,
                    args,
                } => {
                    events.push(IrStreamEvent::ToolCallDone {
                        index: seq,
                        choice_index: 0,
                        id: Cow::Owned(id),
                        name: Cow::Owned(name),
                        arguments: Cow::Owned(args),
                    });
                }
                AntBlockKind::Other(payload) if !payload.is_null() => {
                    events.push(IrStreamEvent::OpaqueBlock {
                        index: 0,
                        provider: Cow::Borrowed("anthropic"),
                        payload,
                    });
                }
                AntBlockKind::Other(_) => {}
            }
        }
        if self.started && !self.choice_finished {
            self.choice_finished = true;
            events.push(IrStreamEvent::ChoiceFinish {
                index: 0,
                finish_reason: IrFinishReason::Stop,
            });
        }
        events.push(IrStreamEvent::Done);
        events
    }
}

impl Default for AntStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeStream for AntStreamDecoder {
    fn decode_sse_data<'a>(
        &mut self,
        data: &'a [u8],
    ) -> Result<Vec<IrStreamEvent<'a>>, CodecError> {
        let evt: AntSseEvent<'a> = serde_json::from_slice(data)?;
        if self.finished {
            return if evt.r#type.as_ref() == "message_stop" {
                Ok(Vec::new())
            } else {
                Err(CodecError::InvalidState(
                    "Anthropic stream received data after completion".to_string(),
                ))
            };
        }
        let mut events = Vec::with_capacity(2);

        match evt.r#type.as_ref() {
            "message_start" => {
                self.started = true;
                if let Some(msg) = evt.message {
                    let usage = msg.usage.as_ref().map(AnthropicShim::convert_usage);
                    events.push(IrStreamEvent::Start {
                        id: msg.id,
                        model: msg.model,
                        usage,
                    });
                    if let Some(ref u) = msg.usage {
                        self.base_usage = Some(AnthropicShim::convert_usage(u));
                    }
                }
            }
            "content_block_start" => {
                self.started = true;
                let idx = evt.index.unwrap_or(0);
                if let Some(block) = evt.content_block {
                    match block {
                        AntContentBlock::ToolUse { id, name, .. } => {
                            let seq = self.tool_seq;
                            self.tool_seq += 1;
                            self.set_block(
                                idx,
                                AntBlockKind::ToolUse {
                                    seq,
                                    id: id.to_string(),
                                    name: name.to_string(),
                                    args: String::new(),
                                },
                            );
                            events.push(IrStreamEvent::ToolCallStart {
                                index: seq,
                                choice_index: 0,
                                id,
                                name,
                            });
                        }
                        AntContentBlock::Text { .. } => {
                            self.set_block(idx, AntBlockKind::Text);
                        }
                        AntContentBlock::Thinking { .. } => {
                            self.set_block(idx, AntBlockKind::Thinking { signature: None });
                        }
                        AntContentBlock::RedactedThinking { data } => {
                            self.set_block(idx, AntBlockKind::Other(serde_json::Value::Null));
                            events.push(IrStreamEvent::RedactedReasoning { index: 0, data });
                        }
                        AntContentBlock::Unknown(raw) => {
                            self.set_block(idx, AntBlockKind::Other(raw.clone()));
                        }
                    }
                }
            }
            "content_block_delta" => {
                let idx = evt.index.unwrap_or(0);
                if let Some(delta) = evt.delta {
                    match delta {
                        AntDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                events.push(IrStreamEvent::ContentDelta {
                                    index: 0,
                                    delta: text,
                                });
                            }
                        }
                        AntDelta::ThinkingDelta { thinking } => {
                            if !thinking.is_empty() {
                                events.push(IrStreamEvent::ReasoningDelta {
                                    index: 0,
                                    delta: thinking,
                                });
                            }
                        }
                        AntDelta::InputJsonDelta { partial_json } => {
                            if let Some(AntBlockKind::ToolUse { seq, args, .. }) =
                                self.block_mut(idx)
                            {
                                args.push_str(&partial_json);
                                events.push(IrStreamEvent::ToolCallDelta {
                                    index: *seq,
                                    choice_index: 0,
                                    arguments_delta: partial_json,
                                });
                            }
                        }
                        AntDelta::SignatureDelta { signature } => {
                            // 签名不产生独立事件，缓存到块状态，在 stop 时随 ReasoningDone 发出
                            if let Some(AntBlockKind::Thinking { signature: sig }) =
                                self.block_mut(idx)
                            {
                                *sig = Some(signature.to_string());
                            }
                        }
                        AntDelta::CitationsDelta { citation } => {
                            events.push(IrStreamEvent::Citation { index: 0, citation });
                        }
                        AntDelta::MessageDelta { .. } | AntDelta::Unknown(_) => {}
                    }
                }
            }
            "content_block_stop" => {
                let idx = evt.index.unwrap_or(0);
                match self.take_block(idx) {
                    Some(AntBlockKind::Thinking { signature }) => {
                        events.push(IrStreamEvent::ReasoningDone {
                            index: 0,
                            signature: signature.map(Cow::Owned),
                        });
                    }
                    Some(AntBlockKind::ToolUse {
                        seq,
                        id,
                        name,
                        args,
                    }) => {
                        events.push(IrStreamEvent::ToolCallDone {
                            index: seq,
                            choice_index: 0,
                            id: Cow::Owned(id),
                            name: Cow::Owned(name),
                            arguments: Cow::Owned(args),
                        });
                    }
                    Some(AntBlockKind::Other(payload)) => {
                        if !payload.is_null() {
                            events.push(IrStreamEvent::OpaqueBlock {
                                index: 0,
                                provider: Cow::Borrowed("anthropic"),
                                payload,
                            });
                        }
                    }
                    // Text 或未跟踪的块（容错：未收到 block_start）→ ContentDone
                    _ => {
                        events.push(IrStreamEvent::ContentDone { index: 0 });
                    }
                }
            }
            "message_delta" => {
                if let Some(AntDelta::MessageDelta { ref stop_reason }) = evt.delta {
                    if !self.choice_finished {
                        self.choice_finished = true;
                        events.push(IrStreamEvent::ChoiceFinish {
                            index: 0,
                            finish_reason: AnthropicShim::parse_stop_reason(stop_reason),
                        });
                    }
                }
                if let Some(ref u) = evt.usage {
                    let delta_usage = AnthropicShim::convert_usage(u);
                    let merged = if let Some(ref base) = self.base_usage {
                        IrUsage {
                            prompt_tokens: if delta_usage.prompt_tokens > 0 {
                                delta_usage.prompt_tokens
                            } else {
                                base.prompt_tokens
                            },
                            completion_tokens: if delta_usage.completion_tokens > 0 {
                                delta_usage.completion_tokens
                            } else {
                                base.completion_tokens
                            },
                            total_tokens: 0, // 下方重算
                            cache_read_tokens: delta_usage
                                .cache_read_tokens
                                .or(base.cache_read_tokens),
                            cache_creation_tokens: delta_usage
                                .cache_creation_tokens
                                .or(base.cache_creation_tokens),
                            reasoning_tokens: delta_usage
                                .reasoning_tokens
                                .or(base.reasoning_tokens),
                            audio_tokens: delta_usage.audio_tokens.or(base.audio_tokens),
                            accepted_prediction_tokens: delta_usage
                                .accepted_prediction_tokens
                                .or(base.accepted_prediction_tokens),
                            rejected_prediction_tokens: delta_usage
                                .rejected_prediction_tokens
                                .or(base.rejected_prediction_tokens),
                        }
                    } else {
                        delta_usage
                    };
                    let merged = IrUsage {
                        total_tokens: merged
                            .prompt_tokens
                            .saturating_add(merged.completion_tokens),
                        ..merged
                    };
                    events.push(IrStreamEvent::Usage(merged));
                }
            }
            "message_stop" => {
                events.extend(self.finalize());
            }
            "error" => {
                // Anthropic 错误事件形如 {"type":"error","error":{"type":"...","message":"..."}}，
                // AntSseEvent 无专门 error 字段，故重新解析 data 提取 error.message，避免静默丢弃。
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(data) {
                    let msg = v
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown error");
                    events.push(IrStreamEvent::Error {
                        message: Cow::Owned(msg.to_string()),
                    });
                    self.finished = true;
                }
            }
            _ => {} // ping 等忽略
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<IrStreamEvent<'static>>, CodecError> {
        Ok(self.finalize())
    }
}

// ─── DecodeRequest ──────────────────────────────────────────────────

/// Anthropic Messages API 请求 — 反序列化用
#[derive(Deserialize)]
struct AntRequestIn<'a> {
    #[serde(borrow)]
    model: Cow<'a, str>,
    max_tokens: Option<u32>,
    #[serde(default)]
    system: Option<serde_json::Value>,
    #[serde(default)]
    messages: Vec<AntMessageIn<'a>>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(borrow, default)]
    stop_sequences: Option<Vec<Cow<'a, str>>>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<AntToolIn<'a>>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    thinking: Option<serde_json::Value>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct AntMessageIn<'a> {
    #[serde(borrow)]
    role: Cow<'a, str>,
    content: serde_json::Value,
    #[serde(default)]
    cache_control: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AntToolIn<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(borrow, default)]
    description: Option<Cow<'a, str>>,
    #[serde(default)]
    input_schema: Option<serde_json::Value>,
    #[serde(default)]
    cache_control: Option<serde_json::Value>,
    /// 特殊工具类型标识（computer_20241022 / text_editor_20241022 /
    /// bash_20241022 / mcp / custom），缺省即 custom 函数工具
    #[serde(borrow, default)]
    r#type: Option<Cow<'a, str>>,
}

impl AnthropicShim {
    fn parse_ant_tool_choice(v: &serde_json::Value) -> IrToolChoice<'static> {
        let t = v.get("type").and_then(|t| t.as_str()).unwrap_or("auto");
        match t {
            "auto" => IrToolChoice::Auto,
            "none" => IrToolChoice::None,
            "any" => IrToolChoice::Required,
            "tool" => {
                let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("");
                IrToolChoice::Specific {
                    name: Cow::Owned(name.to_string()),
                }
            }
            _ => IrToolChoice::Auto,
        }
    }

    fn parse_ant_content(
        content: &serde_json::Value,
    ) -> (IrContent<'static>, Option<Vec<IrToolCall<'static>>>) {
        // content 可以是字符串或 content block 数组
        if let Some(s) = content.as_str() {
            return (IrContent::Text(Cow::Owned(s.to_string())), None);
        }

        let arr = match content.as_array() {
            Some(a) => a,
            None => return (IrContent::Text(Cow::Owned(String::new())), None),
        };

        let mut parts: Vec<IrContentPart<'static>> = Vec::new();

        for block in arr {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match block_type {
                "text" => {
                    let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    parts.push(IrContentPart::Text {
                        text: Cow::Owned(text.to_string()),
                        citations: None,
                    });
                }
                "image" => {
                    if let Some(source) = block.get("source") {
                        let source_type = source.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if source_type == "base64" {
                            let media_type = source
                                .get("media_type")
                                .and_then(|m| m.as_str())
                                .unwrap_or("image/png");
                            let data = source.get("data").and_then(|d| d.as_str()).unwrap_or("");
                            parts.push(IrContentPart::ImageBase64 {
                                media_type: Cow::Owned(media_type.to_string()),
                                data: Cow::Owned(data.to_string()),
                            });
                        } else if source_type == "url" {
                            let url = source.get("url").and_then(|u| u.as_str()).unwrap_or("");
                            parts.push(IrContentPart::ImageUrl {
                                url: Cow::Owned(url.to_string()),
                                detail: None,
                            });
                        } else {
                            parts.push(IrContentPart::Opaque {
                                provider: Cow::Borrowed("anthropic"),
                                payload: block.clone(),
                            });
                        }
                    } else {
                        parts.push(IrContentPart::Opaque {
                            provider: Cow::Borrowed("anthropic"),
                            payload: block.clone(),
                        });
                    }
                }
                "document" => {
                    if let Some(source) = block.get("source") {
                        let source_type = source.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if source_type == "file" {
                            let file_id =
                                source.get("file_id").and_then(|f| f.as_str()).unwrap_or("");
                            parts.push(IrContentPart::FileRef {
                                file_id: Cow::Owned(file_id.to_string()),
                            });
                        } else if source_type == "base64" {
                            let media_type = source
                                .get("media_type")
                                .and_then(|m| m.as_str())
                                .unwrap_or("application/pdf");
                            let data = source.get("data").and_then(|d| d.as_str()).unwrap_or("");
                            let doc_title = block
                                .get("title")
                                .and_then(|t| t.as_str())
                                .map(|s| Cow::Owned(s.to_string()));
                            parts.push(IrContentPart::Document {
                                media_type: Cow::Owned(media_type.to_string()),
                                data: Cow::Owned(data.to_string()),
                                filename: doc_title,
                            });
                        } else {
                            parts.push(IrContentPart::Opaque {
                                provider: Cow::Borrowed("anthropic"),
                                payload: block.clone(),
                            });
                        }
                    } else {
                        parts.push(IrContentPart::Opaque {
                            provider: Cow::Borrowed("anthropic"),
                            payload: block.clone(),
                        });
                    }
                }
                "thinking" => {
                    let text = block.get("thinking").and_then(|t| t.as_str()).unwrap_or("");
                    let signature = block
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .map(|s| Cow::Owned(s.to_string()));
                    parts.push(IrContentPart::Reasoning {
                        text: Cow::Owned(text.to_string()),
                        signature,
                    });
                }
                "redacted_thinking" => {
                    let data = block.get("data").and_then(|d| d.as_str()).unwrap_or("");
                    parts.push(IrContentPart::RedactedReasoning {
                        data: Cow::Owned(data.to_string()),
                    });
                }
                "tool_use" => {
                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_default();
                    parts.push(IrContentPart::FunctionCall {
                        id: Cow::Owned(id.to_string()),
                        name: Cow::Owned(name.to_string()),
                        arguments: Cow::Owned(args),
                    });
                }
                "tool_result" => {
                    // 内嵌 tool_result（未走消息级拆分路径）→ FunctionResponse 部件，
                    // 保留 tool_use_id 关联
                    let id = block
                        .get("tool_use_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("");
                    let response = block
                        .get("content")
                        .cloned()
                        .unwrap_or(serde_json::Value::String(String::new()));
                    parts.push(IrContentPart::FunctionResponse {
                        id: Cow::Owned(id.to_string()),
                        name: Cow::Owned(String::new()),
                        response,
                    });
                }
                _ => {
                    parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("anthropic"),
                        payload: block.clone(),
                    });
                }
            }
        }

        let tc: Option<Vec<IrToolCall<'static>>> = None;

        let content = match parts.len() {
            0 => IrContent::Text(Cow::Owned(String::new())),
            1 => {
                if let IrContentPart::Text { ref text, .. } = parts[0] {
                    IrContent::Text(Cow::Owned(text.to_string()))
                } else {
                    IrContent::Parts(parts)
                }
            }
            _ => IrContent::Parts(parts),
        };

        (content, tc)
    }
}

impl DecodeRequest for AnthropicShim {
    fn decode_request<'a>(&self, body: &'a [u8]) -> Result<IrRequest<'a>, CodecError> {
        let raw_json: serde_json::Value = serde_json::from_slice(body)?;
        let preserved = super::extract_preserved(&raw_json);
        let req: AntRequestIn<'a> = serde_json::from_slice(body)?;

        let mut messages: Vec<IrMessage<'_>> = Vec::new();

        // system prompt
        if let Some(ref system) = req.system {
            let sys_text = if let Some(s) = system.as_str() {
                s.to_string()
            } else if let Some(arr) = system.as_array() {
                arr.iter()
                    .filter_map(|block| block.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                String::new()
            };
            // Extract cache_control from last system block (if array form)
            let sys_cache_control = system.as_array().and_then(|arr| {
                arr.last().and_then(|block| {
                    block.get("cache_control").and_then(|cc| {
                        let cc_type = cc
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("ephemeral");
                        if cc_type == "ephemeral" {
                            Some(IrCacheControl {
                                r#type: CacheControlType::Ephemeral,
                                ttl: cc
                                    .get("ttl")
                                    .and_then(|t| t.as_str())
                                    .map(|s| Cow::Owned(s.to_string())),
                            })
                        } else {
                            None
                        }
                    })
                })
            });
            if !sys_text.is_empty() {
                messages.push(IrMessage {
                    role: Role::System,
                    content: IrContent::Text(Cow::Owned(sys_text)),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls: None,
                    cache_control: sys_cache_control,
                    refusal: None,
                });
            }
        }

        // messages
        for m in &req.messages {
            let role = match m.role.as_ref() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::User,
            };

            // 检查是否有 tool_result 块 — 如果有，拆分为 Tool 消息；
            // 同消息内的其他块（text/image 等）保留为并列的 User 消息，不丢弃
            if role == Role::User {
                if let Some(arr) = m.content.as_array() {
                    let has_tool_result = arr
                        .iter()
                        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"));
                    if has_tool_result {
                        let mut sibling_blocks: Vec<serde_json::Value> = Vec::new();
                        for block in arr {
                            if block.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                                let tool_use_id = block
                                    .get("tool_use_id")
                                    .and_then(|i| i.as_str())
                                    .unwrap_or("");
                                // tool_result.content：字符串→Text；数组→按块解析
                                // （保留 image/document，与 EncodeRequest 对称）
                                let content = match block.get("content") {
                                    Some(c) if c.is_string() => IrContent::Text(Cow::Owned(
                                        c.as_str().unwrap_or("").to_string(),
                                    )),
                                    Some(c) if c.is_array() => {
                                        let (parsed, _) = Self::parse_ant_content(c);
                                        parsed
                                    }
                                    _ => IrContent::Text(Cow::Owned(String::new())),
                                };
                                let block_cc = block.get("cache_control").and_then(|cc| {
                                    let cc_type = cc
                                        .get("type")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("ephemeral");
                                    if cc_type == "ephemeral" {
                                        Some(IrCacheControl {
                                            r#type: CacheControlType::Ephemeral,
                                            ttl: cc
                                                .get("ttl")
                                                .and_then(|t| t.as_str())
                                                .map(|s| Cow::Owned(s.to_string())),
                                        })
                                    } else {
                                        None
                                    }
                                });
                                messages.push(IrMessage {
                                    role: Role::Tool,
                                    content,
                                    tool_call_id: Some(Cow::Owned(tool_use_id.to_string())),
                                    tool_name: None,
                                    tool_calls: None,
                                    cache_control: block_cc,
                                    refusal: None,
                                });
                            } else {
                                sibling_blocks.push(block.clone());
                            }
                        }
                        // 兄弟块 → 追加一条 User 消息
                        if !sibling_blocks.is_empty() {
                            let sibling_val = serde_json::Value::Array(sibling_blocks);
                            let (content, tool_calls) = Self::parse_ant_content(&sibling_val);
                            messages.push(IrMessage {
                                role: Role::User,
                                content,
                                tool_call_id: None,
                                tool_name: None,
                                tool_calls,
                                cache_control: None,
                                refusal: None,
                            });
                        }
                        continue;
                    }
                }
            }

            let (content, tool_calls) = Self::parse_ant_content(&m.content);
            let cache_control = m
                .cache_control
                .as_ref()
                .or_else(|| {
                    m.content
                        .as_array()
                        .and_then(|arr| arr.iter().rev().find_map(|b| b.get("cache_control")))
                })
                .and_then(|cc| {
                    let cc_type = cc
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("ephemeral");
                    if cc_type == "ephemeral" {
                        Some(IrCacheControl {
                            r#type: CacheControlType::Ephemeral,
                            ttl: cc
                                .get("ttl")
                                .and_then(|t| t.as_str())
                                .map(|s| Cow::Owned(s.to_string())),
                        })
                    } else {
                        None
                    }
                });
            messages.push(IrMessage {
                role,
                content,
                tool_call_id: None,
                tool_name: None,
                tool_calls,
                cache_control,
                refusal: None,
            });
        }

        super::backfill_tool_names(&mut messages);

        let raw_tools = raw_json.get("tools").and_then(|v| v.as_array());
        let tools: Option<Vec<IrTool<'_>>> = req.tools.as_ref().map(|ts| {
            ts.iter()
                .enumerate()
                .map(|(i, t)| {
                    let cache_control = t.cache_control.as_ref().and_then(|cc| {
                        let cc_type = cc
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("ephemeral");
                        if cc_type == "ephemeral" {
                            Some(IrCacheControl {
                                r#type: CacheControlType::Ephemeral,
                                ttl: cc
                                    .get("ttl")
                                    .and_then(|v| v.as_str())
                                    .map(|s| Cow::Owned(s.to_string())),
                            })
                        } else {
                            None
                        }
                    });
                    // type → IrToolType（版本无关前缀匹配；bash/custom/无 → Function）
                    let tool_type = match t.r#type.as_deref() {
                        Some(s) if s.starts_with("computer") => IrToolType::ComputerUse,
                        Some(s) if s.starts_with("text_editor") => IrToolType::TextEditor,
                        Some(s) if s.starts_with("web_search") => IrToolType::WebSearch,
                        Some(s) if s.starts_with("code_execution") => IrToolType::CodeInterpreter,
                        Some("mcp") => IrToolType::Mcp,
                        _ => IrToolType::Function,
                    };
                    // extra：保留 type + 工具专有字段（display_width_px 等），
                    // 供 EncodeRequest 精确重建特殊工具的有线格式
                    let extra = raw_tools
                        .and_then(|arr| arr.get(i))
                        .and_then(|v| v.as_object())
                        .map(|obj| {
                            let mut m = serde_json::Map::new();
                            for (k, v) in obj {
                                if matches!(
                                    k.as_str(),
                                    "name" | "description" | "input_schema" | "cache_control"
                                ) {
                                    continue;
                                }
                                m.insert(k.clone(), v.clone());
                            }
                            m
                        })
                        .filter(|m| !m.is_empty())
                        .map(serde_json::Value::Object);
                    IrTool {
                        tool_type,
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone().unwrap_or(serde_json::json!({})),
                        cache_control,
                        extra,
                    }
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(Self::parse_ant_tool_choice);

        // parallel_tool_calls from tool_choice.disable_parallel_tool_use
        let parallel_tool_calls = req.tool_choice.as_ref().and_then(|tc| {
            tc.get("disable_parallel_tool_use")
                .and_then(|v| v.as_bool())
                .map(|disabled| !disabled)
        });

        // thinking → reasoning config
        let reasoning = req.thinking.as_ref().and_then(|t| {
            let type_str = t.get("type").and_then(|v| v.as_str()).unwrap_or("disabled");
            if type_str == "disabled" {
                None
            } else {
                Some(ReasoningConfig {
                    mode: ReasoningMode::Enabled,
                    budget_tokens: t
                        .get("budget_tokens")
                        .and_then(|b| b.as_u64())
                        .map(|b| b as u32),
                    effort: None,
                })
            }
        });

        // metadata.user_id
        let user_id = req
            .metadata
            .as_ref()
            .and_then(|m| m.get("user_id"))
            .and_then(|u| u.as_str())
            .map(|s| Cow::Owned(s.to_string()));

        let metadata = if user_id.is_some() || req.service_tier.is_some() {
            Some(IrMetadata {
                user_id,
                service_tier: req.service_tier.clone(),
            })
        } else {
            None
        };

        let mut provider_metadata: Option<Box<serde_json::Value>> = None;
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        Ok(IrRequest {
            model: req.model,
            messages,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: req.max_tokens,
            stop: req.stop_sequences,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            stream: req.stream,
            store: None,
            modalities: None,
            tools,
            tool_choice,
            parallel_tool_calls,
            reasoning,
            response_format: None,
            previous_response_id: None,
            truncation: None,
            metadata,
            provider_metadata,
            metadata_mode: MetadataMode::default(),
        })
    }
}

// ─── EncodeResponse ─────────────────────────────────────────────────

impl EncodeResponse for AnthropicShim {
    fn encode_response(&self, ir: &IrResponse<'_>) -> Result<Vec<u8>, CodecError> {
        for choice in &ir.choices {
            super::validate_tool_arguments(std::slice::from_ref(&choice.message))?;
        }
        let choice = ir.choices.first();
        let msg = choice.map(|c| &c.message);

        // 无损保留：收集 Anthropic 不支持的 IR 内容部件
        let mut preserved: Vec<serde_json::Value> = Vec::new();

        // content blocks
        let mut content_blocks: Vec<serde_json::Value> = Vec::new();

        if let Some(m) = msg {
            match &m.content {
                IrContent::Text(s) => {
                    if !s.is_empty() {
                        content_blocks.push(serde_json::json!({
                            "type": "text",
                            "text": s,
                        }));
                    }
                }
                IrContent::Parts(parts) => {
                    for p in parts {
                        match p {
                            IrContentPart::Text { text, citations } => {
                                let mut block = serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                });
                                if let Some(ref cits) = citations {
                                    let cit_arr: Vec<serde_json::Value> = cits
                                        .iter()
                                        .map(|c| {
                                            let mut obj = serde_json::json!({
                                                "type": c.r#type,
                                            });
                                            if let Some(ref url) = c.url {
                                                obj["url"] =
                                                    serde_json::Value::String(url.to_string());
                                            }
                                            if let Some(ref title) = c.title {
                                                obj["title"] =
                                                    serde_json::Value::String(title.to_string());
                                            }
                                            if let Some(ref ct) = c.cited_text {
                                                obj["cited_text"] =
                                                    serde_json::Value::String(ct.to_string());
                                            }
                                            if let Some(ref ei) = c.encrypted_index {
                                                obj["encrypted_index"] =
                                                    serde_json::Value::String(ei.to_string());
                                            }
                                            obj
                                        })
                                        .collect();
                                    block["citations"] = serde_json::Value::Array(cit_arr);
                                }
                                content_blocks.push(block);
                            }
                            IrContentPart::Reasoning { text, signature } => {
                                let mut block = serde_json::json!({
                                    "type": "thinking",
                                    "thinking": text,
                                });
                                if let Some(sig) = signature {
                                    block["signature"] = serde_json::Value::String(sig.to_string());
                                }
                                content_blocks.push(block);
                            }
                            IrContentPart::RedactedReasoning { data } => {
                                content_blocks.push(serde_json::json!({
                                    "type": "redacted_thinking",
                                    "data": data,
                                }));
                            }
                            IrContentPart::FunctionCall {
                                ref id,
                                ref name,
                                ref arguments,
                            } => {
                                let input: serde_json::Value = serde_json::from_str(arguments)
                                    .unwrap_or(serde_json::json!({}));
                                content_blocks.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": input,
                                }));
                            }
                            // 同源 Opaque → 直接回写为原生块
                            IrContentPart::Opaque {
                                provider, payload, ..
                            } if provider == "anthropic" => {
                                content_blocks.push(payload.clone());
                            }
                            // Anthropic 响应不支持的变体 — 无损保留
                            IrContentPart::ImageUrl { .. }
                            | IrContentPart::ImageBase64 { .. }
                            | IrContentPart::Document { .. }
                            | IrContentPart::FileRef { .. }
                            | IrContentPart::Audio { .. }
                            | IrContentPart::Video { .. }
                            | IrContentPart::Opaque { .. }
                            | IrContentPart::FunctionResponse { .. } => {
                                if let Ok(v) = serde_json::to_value(p) {
                                    preserved.push(v);
                                }
                            }
                        }
                    }
                }
            }

            // tool_calls → tool_use blocks
            if let Some(ref tcs) = m.tool_calls {
                for tc in tcs {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    content_blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input,
                    }));
                }
            }
        }

        // stop_reason
        let stop_reason = choice.and_then(|c| c.finish_reason).map(|fr| match fr {
            IrFinishReason::Stop => "end_turn",
            IrFinishReason::StopSequence => "stop_sequence",
            IrFinishReason::Length => "max_tokens",
            IrFinishReason::ToolCalls => "tool_use",
            IrFinishReason::PauseTurn => "pause_turn",
            // 安全拦截/审核 → refusal，不得伪装成正常完成
            IrFinishReason::ContentFilter | IrFinishReason::Safety | IrFinishReason::Recitation => {
                "refusal"
            }
            IrFinishReason::MalformedFunctionCall => "end_turn",
        });

        let mut resp = serde_json::json!({
            "id": ir.id,
            "type": "message",
            "role": "assistant",
            "model": ir.model,
            "content": content_blocks,
        });

        if let Some(sr) = stop_reason {
            resp["stop_reason"] = serde_json::Value::String(sr.to_string());
        } else {
            resp["stop_reason"] = serde_json::Value::Null;
        }

        if let Some(ref usage) = ir.usage {
            resp["usage"] = serde_json::json!({
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            });
            if let Some(cr) = usage.cache_read_tokens {
                resp["usage"]["cache_read_input_tokens"] = serde_json::json!(cr);
            }
            if let Some(cc) = usage.cache_creation_tokens {
                resp["usage"]["cache_creation_input_tokens"] = serde_json::json!(cc);
            }
        }

        // service_tier & stop_sequence from provider_metadata
        if let Some(ref pm) = ir.provider_metadata {
            if let Some(st) = pm.get("service_tier").and_then(|v| v.as_str()) {
                resp["service_tier"] = serde_json::Value::String(st.to_string());
            }
            if let Some(ss) = pm.get("stop_sequence") {
                resp["stop_sequence"] = ss.clone();
            }
        }

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        super::attach_preserved(&mut resp, preserved);
        serde_json::to_vec(&resp).map_err(CodecError::from)
    }
}

// ─── EncodeStream ───────────────────────────────────────────────────

/// 编码器侧的已打开块类型
#[derive(Clone, Copy, PartialEq, Eq)]
enum AntOutBlockKind {
    Text,
    Thinking,
    ToolUse,
}

/// Anthropic 流式编码器 — 每条流一个实例。
///
/// Anthropic 有线协议要求每个内容块被 content_block_start / content_block_stop
/// 包裹、块索引在整条消息内单调递增，且 stop_reason 与 usage 在同一条
/// message_delta 中。上游 IR 事件流不保证提供这些结构（如 OpenAI 源没有
/// 块级 start 信号，且 reasoning/content/tool 各自独立编号），编码器有状态地补齐：
/// - 首个 delta 自动发 content_block_start（IR index + 块类型 → 重映射到
///   单调递增的输出索引；类型切换时先 stop 旧块再以新索引 start）
/// - ReasoningDone 携带的 signature 先编码为 signature_delta 再 stop
/// - ChoiceFinish 关闭所有块、缓存 stop_reason，与 Usage 合并为单条 message_delta
/// - Done 时关闭所有未闭合块、补发未发送的 message_delta
pub struct AntStreamEncoder {
    /// 已打开的块: (IR 事件 index, 输出块 index, kind)
    open_blocks: Vec<(u32, u32, AntOutBlockKind)>,
    /// 下一个输出块索引（Anthropic 要求全消息单调递增）
    next_out_index: u32,
    /// 缓存的 stop_reason（等待与 usage 合并）
    pending_stop_reason: Option<&'static str>,
    /// 缓存的最新累计 usage（上游可能多次发出，后到取代先到）
    pending_usage: Option<IrUsage>,
    /// Anthropic 不支持的流事件（如 Logprobs）无损缓存，
    /// 在终态 message_delta 的 metadata 中随 _openxlate_preserved 透传
    preserved_events: Vec<serde_json::Value>,
}

impl AntStreamEncoder {
    pub fn new() -> Self {
        Self {
            open_blocks: Vec::new(),
            next_out_index: 0,
            pending_stop_reason: None,
            pending_usage: None,
            preserved_events: Vec::new(),
        }
    }

    fn sse_frame(event_name: &str, payload: &serde_json::Value) -> Result<Vec<u8>, CodecError> {
        Ok(format!(
            "event: {event_name}\ndata: {}\n\n",
            serde_json::to_string(payload)?
        )
        .into_bytes())
    }

    fn block_stop_frame(out_index: u32) -> Result<Vec<u8>, CodecError> {
        Self::sse_frame(
            "content_block_stop",
            &serde_json::json!({ "type": "content_block_stop", "index": out_index }),
        )
    }

    /// 确保 IR index 处有 kind 类型的打开块，返回其输出索引；必要时生成 stop/start 帧。
    /// Text/Thinking 共用 choice 索引空间，二者在同 IR index 上切换时先关旧块；
    /// ToolUse 使用独立的 tool 索引空间，与 Text/Thinking 互不冲突。
    fn ensure_block(
        &mut self,
        out: &mut Vec<u8>,
        ir_index: u32,
        kind: AntOutBlockKind,
        start_block: serde_json::Value,
    ) -> Result<u32, CodecError> {
        if let Some(pos) = self
            .open_blocks
            .iter()
            .position(|(i, _, k)| *i == ir_index && *k == kind)
        {
            return Ok(self.open_blocks[pos].1);
        }
        // Text ↔ Thinking 类型切换：关闭同 IR index 的另一类块
        if kind != AntOutBlockKind::ToolUse {
            if let Some(pos) = self
                .open_blocks
                .iter()
                .position(|(i, _, k)| *i == ir_index && *k != AntOutBlockKind::ToolUse)
            {
                let (_, old_out, _) = self.open_blocks.remove(pos);
                out.extend(Self::block_stop_frame(old_out)?);
            }
        }
        let out_index = self.next_out_index;
        self.next_out_index += 1;
        out.extend(Self::sse_frame(
            "content_block_start",
            &serde_json::json!({
                "type": "content_block_start",
                "index": out_index,
                "content_block": start_block,
            }),
        )?);
        self.open_blocks.push((ir_index, out_index, kind));
        Ok(out_index)
    }

    /// 查询 (IR index, 块类型) 对应的已打开输出块索引。
    /// IR 中 content 的 index 是 choice 索引、tool 的 index 是 tool 索引，
    /// 两个编号空间可能重叠（都从 0 起），必须按类型区分。
    fn out_index_of(&self, ir_index: u32, kind: AntOutBlockKind) -> Option<u32> {
        self.open_blocks
            .iter()
            .find(|(i, _, k)| *i == ir_index && *k == kind)
            .map(|(_, o, _)| *o)
    }

    fn close_block(
        &mut self,
        out: &mut Vec<u8>,
        ir_index: u32,
        kind: AntOutBlockKind,
    ) -> Result<(), CodecError> {
        if let Some(pos) = self
            .open_blocks
            .iter()
            .position(|(i, _, k)| *i == ir_index && *k == kind)
        {
            let (_, out_index, _) = self.open_blocks.remove(pos);
            out.extend(Self::block_stop_frame(out_index)?);
        }
        Ok(())
    }

    fn close_all(&mut self, out: &mut Vec<u8>) -> Result<(), CodecError> {
        for (_, out_index, _) in std::mem::take(&mut self.open_blocks) {
            out.extend(Self::block_stop_frame(out_index)?);
        }
        Ok(())
    }

    fn ir_finish_to_stop_reason(fr: IrFinishReason) -> &'static str {
        match fr {
            IrFinishReason::Stop => "end_turn",
            IrFinishReason::StopSequence => "stop_sequence",
            IrFinishReason::Length => "max_tokens",
            IrFinishReason::ToolCalls => "tool_use",
            IrFinishReason::PauseTurn => "pause_turn",
            // 安全拦截/审核 → refusal，不得伪装成正常完成
            IrFinishReason::ContentFilter | IrFinishReason::Safety | IrFinishReason::Recitation => {
                "refusal"
            }
            IrFinishReason::MalformedFunctionCall => "end_turn",
        }
    }
}

impl Default for AntStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeStream for AntStreamEncoder {
    fn encode_sse_event(&mut self, event: &IrStreamEvent<'_>) -> Result<Vec<u8>, CodecError> {
        let mut out = Vec::new();
        match event {
            IrStreamEvent::Start { id, model, usage } => {
                let mut message = serde_json::json!({
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                });
                if let Some(u) = usage {
                    let mut usage_json = serde_json::json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                    });
                    if let Some(cr) = u.cache_read_tokens {
                        usage_json["cache_read_input_tokens"] = serde_json::json!(cr);
                    }
                    if let Some(cc) = u.cache_creation_tokens {
                        usage_json["cache_creation_input_tokens"] = serde_json::json!(cc);
                    }
                    message["usage"] = usage_json;
                }
                out.extend(Self::sse_frame(
                    "message_start",
                    &serde_json::json!({ "type": "message_start", "message": message }),
                )?);
            }
            IrStreamEvent::ContentDelta { index, delta } => {
                let out_index = self.ensure_block(
                    &mut out,
                    *index,
                    AntOutBlockKind::Text,
                    serde_json::json!({ "type": "text", "text": "" }),
                )?;
                out.extend(Self::sse_frame(
                    "content_block_delta",
                    &serde_json::json!({
                        "type": "content_block_delta",
                        "index": out_index,
                        "delta": { "type": "text_delta", "text": delta },
                    }),
                )?);
            }
            IrStreamEvent::ReasoningDelta { index, delta } => {
                let out_index = self.ensure_block(
                    &mut out,
                    *index,
                    AntOutBlockKind::Thinking,
                    serde_json::json!({ "type": "thinking", "thinking": "" }),
                )?;
                out.extend(Self::sse_frame(
                    "content_block_delta",
                    &serde_json::json!({
                        "type": "content_block_delta",
                        "index": out_index,
                        "delta": { "type": "thinking_delta", "thinking": delta },
                    }),
                )?);
            }
            IrStreamEvent::RedactedReasoning { index, data } => {
                // redacted_thinking：完整数据在 start 一次性给出，随即关闭
                self.ensure_block(
                    &mut out,
                    *index,
                    AntOutBlockKind::Thinking,
                    serde_json::json!({ "type": "redacted_thinking", "data": data }),
                )?;
                self.close_block(&mut out, *index, AntOutBlockKind::Thinking)?;
            }
            IrStreamEvent::ToolCallStart {
                index, id, name, ..
            } => {
                self.ensure_block(
                    &mut out,
                    *index,
                    AntOutBlockKind::ToolUse,
                    serde_json::json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {},
                    }),
                )?;
            }
            IrStreamEvent::ToolCallDelta {
                index,
                arguments_delta,
                ..
            } => {
                let out_index = match self.out_index_of(*index, AntOutBlockKind::ToolUse) {
                    Some(i) => i,
                    None => self.ensure_block(
                        &mut out,
                        *index,
                        AntOutBlockKind::ToolUse,
                        serde_json::json!({
                            "type": "tool_use",
                            "id": "",
                            "name": "",
                            "input": {},
                        }),
                    )?,
                };
                out.extend(Self::sse_frame(
                    "content_block_delta",
                    &serde_json::json!({
                        "type": "content_block_delta",
                        "index": out_index,
                        "delta": { "type": "input_json_delta", "partial_json": arguments_delta },
                    }),
                )?);
            }
            IrStreamEvent::ContentDone { index } => {
                self.close_block(&mut out, *index, AntOutBlockKind::Text)?;
            }
            IrStreamEvent::ReasoningDone { index, signature } => {
                if let Some(sig) = signature {
                    let out_index = self
                        .out_index_of(*index, AntOutBlockKind::Thinking)
                        .unwrap_or(*index);
                    out.extend(Self::sse_frame(
                        "content_block_delta",
                        &serde_json::json!({
                            "type": "content_block_delta",
                            "index": out_index,
                            "delta": { "type": "signature_delta", "signature": sig },
                        }),
                    )?);
                }
                self.close_block(&mut out, *index, AntOutBlockKind::Thinking)?;
            }
            IrStreamEvent::ToolCallDone {
                index,
                id,
                name,
                arguments,
                ..
            } => {
                // 上游未流式宣告过该 tool_call（如 Gemini 一次性完整下发）→ 补发完整块
                if self
                    .out_index_of(*index, AntOutBlockKind::ToolUse)
                    .is_none()
                {
                    let out_index = self.ensure_block(
                        &mut out,
                        *index,
                        AntOutBlockKind::ToolUse,
                        serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": {},
                        }),
                    )?;
                    if !arguments.is_empty() {
                        out.extend(Self::sse_frame(
                            "content_block_delta",
                            &serde_json::json!({
                                "type": "content_block_delta",
                                "index": out_index,
                                "delta": { "type": "input_json_delta", "partial_json": arguments },
                            }),
                        )?);
                    }
                }
                self.close_block(&mut out, *index, AntOutBlockKind::ToolUse)?;
            }
            IrStreamEvent::ChoiceFinish { finish_reason, .. } => {
                // 内容已终结：关闭所有未闭合块（保证 content_block_stop 先于 message_delta）
                self.close_all(&mut out)?;
                // 缓存 stop_reason，等待与 Usage 合并成单条 message_delta
                self.pending_stop_reason = Some(Self::ir_finish_to_stop_reason(*finish_reason));
            }
            IrStreamEvent::Usage(usage) => {
                // Usage 是累计值且可能多次到达（如 Gemini 每 chunk 回传）—
                // 只缓存最新值，在 Done 时随 stop_reason 合并为终态 message_delta，
                // 避免在内容块中间发出 message_delta（违反 Anthropic 事件顺序）
                self.pending_usage = Some(usage.clone());
            }
            IrStreamEvent::Logprobs { .. } => {
                // Anthropic 有线协议无 logprobs 表示 —— 无损缓存，
                // 在 Done 的终态 message_delta metadata 中透传（同 RedactedReasoning 保留模式）
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
            }
            IrStreamEvent::Done => {
                // 关闭所有未闭合块
                self.close_all(&mut out)?;
                // 终态 message_delta：stop_reason 与最新 usage 合并；
                // 上游未发 ChoiceFinish（截断/无 finish_reason）时缺省 end_turn，
                // 保证 Anthropic 客户端总能拿到 stop_reason
                let sr = self.pending_stop_reason.take().or(Some("end_turn"));
                let u = self.pending_usage.take();
                {
                    let mut payload = serde_json::json!({
                        "type": "message_delta",
                        "delta": {},
                    });
                    if let Some(sr) = sr {
                        payload["delta"]["stop_reason"] = serde_json::json!(sr);
                    }
                    if let Some(u) = u {
                        let mut usage_json = serde_json::json!({
                            "input_tokens": u.prompt_tokens,
                            "output_tokens": u.completion_tokens,
                        });
                        if let Some(cr) = u.cache_read_tokens {
                            usage_json["cache_read_input_tokens"] = serde_json::json!(cr);
                        }
                        if let Some(cc) = u.cache_creation_tokens {
                            usage_json["cache_creation_input_tokens"] = serde_json::json!(cc);
                        }
                        payload["usage"] = usage_json;
                    }
                    if !self.preserved_events.is_empty() {
                        let preserved = std::mem::take(&mut self.preserved_events);
                        super::attach_preserved(&mut payload, preserved);
                    }
                    out.extend(Self::sse_frame("message_delta", &payload)?);
                }
                out.extend(Self::sse_frame(
                    "message_stop",
                    &serde_json::json!({ "type": "message_stop" }),
                )?);
            }
            IrStreamEvent::Citation { index, citation } => {
                let out_index = self.ensure_block(
                    &mut out,
                    *index,
                    AntOutBlockKind::Text,
                    serde_json::json!({ "type": "text", "text": "" }),
                )?;
                out.extend(Self::sse_frame(
                    "content_block_delta",
                    &serde_json::json!({
                        "type": "content_block_delta",
                        "index": out_index,
                        "delta": {
                            "type": "citations_delta",
                            "citation": citation,
                        },
                    }),
                )?);
            }
            IrStreamEvent::OpaqueBlock { .. } => {
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
            }
            IrStreamEvent::Error { message } => {
                out.extend(Self::sse_frame(
                    "error",
                    &serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": message },
                    }),
                )?);
            }
        }
        Ok(out)
    }
}
