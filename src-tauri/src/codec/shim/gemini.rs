//! Google Gemini shim — IR ↔ Gemini generateContent API
//!
//! 覆盖 Google AI Studio (generativelanguage.googleapis.com) 和 Vertex AI。
//! 零拷贝：encode → to_vec, decode → from_slice + Cow::Borrowed

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::{
    DecodeRequest, DecodeResponse, DecodeStream, EncodeRequest, EncodeResponse, EncodeStream,
};
use crate::codec::error::CodecError;
use crate::codec::ir::*;

// ─── Gemini 有线格式（编码）────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemRequest<'a> {
    contents: Vec<GemContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GemContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GemGenConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GemTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GemToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_content: Option<&'a str>,
}

#[derive(Serialize, Deserialize)]
struct GemContent<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    parts: Vec<GemPart<'a>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemPart<'a> {
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    text: Option<Cow<'a, str>>,
    /// Gemini 线格式：思考部件为 {"text": "...", "thought": true} —
    /// thought 是布尔标志，思考文本仍在 text 字段
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    /// Gemini 思考签名（decode 保留）
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    thought_signature: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GemInlineData<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GemFunctionCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GemFunctionResponse<'a>>,
    /// Files API URI / 远程 URL 引用
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<GemFileData<'a>>,
    /// executableCode / codeExecutionResult 等未显式建模的字段
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemFileData<'a> {
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    mime_type: Option<Cow<'a, str>>,
    #[serde(borrow)]
    file_uri: Cow<'a, str>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemInlineData<'a> {
    #[serde(borrow)]
    mime_type: Cow<'a, str>,
    #[serde(borrow)]
    data: Cow<'a, str>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemFunctionCall<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemFunctionResponse<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemGenConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GemThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_logprobs: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemTool<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    function_declarations: Option<Vec<GemFuncDecl<'a>>>,
    /// codeExecution 工具（IrToolType::CodeInterpreter），值通常为 {}
    #[serde(skip_serializing_if = "Option::is_none")]
    code_execution: Option<Cow<'a, serde_json::Value>>,
    /// googleSearchRetrieval 工具（IrToolType::WebSearch），可带 dynamicRetrievalConfig
    #[serde(skip_serializing_if = "Option::is_none")]
    google_search_retrieval: Option<Cow<'a, serde_json::Value>>,
}

#[derive(Serialize)]
struct GemFuncDecl<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemToolConfig {
    function_calling_config: GemFunctionCallingConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GemFunctionCallingConfig {
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

// ─── 反序列化（响应）────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemResponse<'a> {
    #[serde(default, borrow)]
    candidates: Vec<GemCandidate<'a>>,
    #[serde(default)]
    usage_metadata: Option<GemUsage>,
    #[serde(borrow, default)]
    model_version: Option<Cow<'a, str>>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    prompt_feedback: Option<serde_json::Value>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemCandidate<'a> {
    #[serde(borrow, default)]
    content: Option<GemContent<'a>>,
    #[serde(borrow, default)]
    finish_reason: Option<Cow<'a, str>>,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    safety_ratings: Option<serde_json::Value>,
    #[serde(default)]
    grounding_metadata: Option<serde_json::Value>,
    #[serde(default)]
    search_entry_point: Option<serde_json::Value>,
    #[serde(default)]
    citation_metadata: Option<serde_json::Value>,
    #[serde(default, rename = "avgLogprobs")]
    avg_logprobs: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemUsage {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
    total_token_count: Option<u32>,
    cached_content_token_count: Option<u32>,
    #[serde(default)]
    thoughts_token_count: Option<u32>,
}

// ─── 实现 ──────────────────────────────────────────────────────────

/// 解析 `data:<mime>;base64,<data>` URL，返回 (mime, data)
fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64").unwrap_or(meta);
    Some((
        if mime.is_empty() {
            "application/octet-stream"
        } else {
            mime
        },
        data,
    ))
}

pub struct GeminiShim;

impl GeminiShim {
    fn ir_role_to_gem(role: Role) -> &'static str {
        match role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "model",
            Role::System | Role::Developer => "user", // system 走 system_instruction，不应到这里
        }
    }

    fn ir_content_to_parts<'a>(
        msg: &'a IrMessage<'a>,
        preserved: &mut Vec<serde_json::Value>,
    ) -> Vec<GemPart<'a>> {
        let mut parts = Vec::new();

        // tool result → functionResponse
        if msg.role == Role::Tool {
            let text = msg.content.text_concat();
            let resp_val =
                serde_json::from_str(&text).unwrap_or(serde_json::json!({ "result": text }));
            parts.push(GemPart {
                text: None,
                thought: None,
                thought_signature: None,
                inline_data: None,
                function_call: None,
                function_response: Some(GemFunctionResponse {
                    // Gemini 用函数名关联工具结果；tool_name 缺失时退回 call id
                    name: msg
                        .tool_name
                        .clone()
                        .or_else(|| msg.tool_call_id.clone())
                        .unwrap_or(Cow::Borrowed("")),
                    response: resp_val,
                }),
                file_data: None,
                extra: serde_json::Map::new(),
            });
            return parts;
        }

        // tool calls → functionCall parts
        if let Some(ref tcs) = msg.tool_calls {
            for tc in tcs {
                let args: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                parts.push(GemPart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    function_call: Some(GemFunctionCall {
                        name: tc.name.clone(),
                        args,
                    }),
                    function_response: None,
                    file_data: None,
                    extra: serde_json::Map::new(),
                });
            }
        }

        match &msg.content {
            IrContent::Text(s) => {
                if !s.is_empty() || parts.is_empty() {
                    parts.push(GemPart {
                        text: Some(s.clone()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        file_data: None,
                        extra: serde_json::Map::new(),
                    });
                }
            }
            IrContent::Parts(content_parts) => {
                for cp in content_parts {
                    match cp {
                        IrContentPart::Text { text, citations } => {
                            if let Some(ref cits) = citations {
                                if let Ok(v) = serde_json::to_value(cits) {
                                    preserved.push(serde_json::json!({"text_citations": v}));
                                }
                            }
                            parts.push(GemPart {
                                text: Some(text.clone()),
                                thought: None,
                                thought_signature: None,
                                inline_data: None,
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::ImageBase64 {
                            media_type, data, ..
                        } => {
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: Some(GemInlineData {
                                    mime_type: media_type.clone(),
                                    data: data.clone(),
                                }),
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::Audio {
                            media_type, data, ..
                        } => {
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: Some(GemInlineData {
                                    mime_type: media_type.clone(),
                                    data: data.clone(),
                                }),
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::Video {
                            media_type, data, ..
                        } => {
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: Some(GemInlineData {
                                    mime_type: media_type.clone(),
                                    data: data.clone(),
                                }),
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::Document {
                            media_type,
                            data,
                            filename,
                        } => {
                            if let Some(ref f) = filename {
                                preserved.push(serde_json::json!({"document_filename": f}));
                            }
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: Some(GemInlineData {
                                    mime_type: media_type.clone(),
                                    data: data.clone(),
                                }),
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::Reasoning { text, signature } => {
                            parts.push(GemPart {
                                text: Some(text.clone()),
                                thought: Some(true),
                                thought_signature: signature.clone(),
                                inline_data: None,
                                function_call: None,
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::ImageUrl { url, detail } => {
                            if let Some(ref d) = detail {
                                preserved.push(serde_json::json!({"image_url_detail": d}));
                            }
                            // data URL → inlineData；普通 URL → fileData（Vertex AI 支持
                            // http(s)/gs URI；AI Studio 需 Files API URI，由上游保证）
                            if let Some((media_type, data)) = parse_data_url(url) {
                                parts.push(GemPart {
                                    text: None,
                                    thought: None,
                                    thought_signature: None,
                                    inline_data: Some(GemInlineData {
                                        mime_type: Cow::Owned(media_type.to_string()),
                                        data: Cow::Owned(data.to_string()),
                                    }),
                                    function_call: None,
                                    function_response: None,
                                    file_data: None,
                                    extra: serde_json::Map::new(),
                                });
                            } else {
                                parts.push(GemPart {
                                    text: None,
                                    thought: None,
                                    thought_signature: None,
                                    inline_data: None,
                                    function_call: None,
                                    function_response: None,
                                    file_data: Some(GemFileData {
                                        mime_type: None,
                                        file_uri: url.clone(),
                                    }),
                                    extra: serde_json::Map::new(),
                                });
                            }
                        }
                        IrContentPart::FileRef { file_id } => {
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: None,
                                function_call: None,
                                function_response: None,
                                file_data: Some(GemFileData {
                                    mime_type: None,
                                    file_uri: file_id.clone(),
                                }),
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::FunctionCall {
                            name, arguments, ..
                        } => {
                            let args: serde_json::Value =
                                serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: None,
                                function_call: Some(GemFunctionCall {
                                    name: name.clone(),
                                    args,
                                }),
                                function_response: None,
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        IrContentPart::FunctionResponse { name, response, .. } => {
                            parts.push(GemPart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                inline_data: None,
                                function_call: None,
                                function_response: Some(GemFunctionResponse {
                                    name: name.clone(),
                                    response: response.clone(),
                                }),
                                file_data: None,
                                extra: serde_json::Map::new(),
                            });
                        }
                        // 同源 Opaque → 通过 extra 字段回写为原生 GemPart
                        IrContentPart::Opaque {
                            provider, payload, ..
                        } if provider == "google" => {
                            if let serde_json::Value::Object(map) = payload {
                                parts.push(GemPart {
                                    text: None,
                                    thought: None,
                                    thought_signature: None,
                                    inline_data: None,
                                    function_call: None,
                                    function_response: None,
                                    file_data: None,
                                    extra: map.clone(),
                                });
                            }
                        }
                        IrContentPart::RedactedReasoning { .. } | IrContentPart::Opaque { .. } => {
                            if let Ok(v) = serde_json::to_value(cp) {
                                preserved.push(v);
                            }
                        }
                    }
                }
            }
        }

        if parts.is_empty() {
            parts.push(GemPart {
                text: Some(Cow::Borrowed("")),
                thought: None,
                thought_signature: None,
                inline_data: None,
                function_call: None,
                function_response: None,
                file_data: None,
                extra: serde_json::Map::new(),
            });
        }

        parts
    }

    fn parse_finish_reason(s: &str) -> IrFinishReason {
        match s {
            "STOP" | "stop" => IrFinishReason::Stop,
            "MAX_TOKENS" | "max_tokens" => IrFinishReason::Length,
            "MALFORMED_FUNCTION_CALL" => IrFinishReason::MalformedFunctionCall,
            "RECITATION" => IrFinishReason::Recitation,
            "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => IrFinishReason::Safety,
            _ => IrFinishReason::Stop,
        }
    }

    fn gem_parts_to_ir_message<'a>(
        parts: &[GemPart<'a>],
        candidate: &GemCandidate<'a>,
    ) -> (
        IrContent<'a>,
        Vec<IrToolCall<'a>>,
        Option<Box<serde_json::Value>>,
    ) {
        let mut content_parts: Vec<IrContentPart<'a>> = Vec::new();

        for (i, part) in parts.iter().enumerate() {
            if let Some(ref text) = part.text {
                if part.thought == Some(true) {
                    content_parts.push(IrContentPart::Reasoning {
                        text: text.clone(),
                        signature: part.thought_signature.clone(),
                    });
                } else {
                    content_parts.push(IrContentPart::Text {
                        text: text.clone(),
                        citations: None,
                    });
                }
            }
            if let Some(ref fc) = part.function_call {
                // functionCall 部件可能携带 thoughtSignature（Gemini 2.5+）—
                // 回填到最后一个无签名的 Reasoning 部件；无匹配则合成空 Reasoning
                if let Some(ref sig) = part.thought_signature {
                    let mut backfilled = false;
                    for cp in content_parts.iter_mut().rev() {
                        if let IrContentPart::Reasoning { signature, .. } = cp {
                            if signature.is_none() {
                                *signature = Some(sig.clone());
                                backfilled = true;
                                break;
                            }
                        }
                    }
                    if !backfilled {
                        content_parts.push(IrContentPart::Reasoning {
                            text: Cow::Borrowed(""),
                            signature: Some(sig.clone()),
                        });
                    }
                }
                let args_str = serde_json::to_string(&fc.args).unwrap_or_default();
                let cand_idx = candidate.index.unwrap_or(0);
                content_parts.push(IrContentPart::FunctionCall {
                    id: Cow::Owned(format!("call_gemini_c{cand_idx}_{i}")),
                    name: fc.name.clone(),
                    arguments: Cow::Owned(args_str),
                });
            }
            if let Some(ref inline) = part.inline_data {
                // inlineData in response → 根据 mime_type 类型化解码
                let mime = inline.mime_type.as_ref();
                if mime.starts_with("audio/") {
                    content_parts.push(IrContentPart::Audio {
                        media_type: inline.mime_type.clone(),
                        data: inline.data.clone(),
                    });
                } else if mime.starts_with("video/") {
                    content_parts.push(IrContentPart::Video {
                        media_type: inline.mime_type.clone(),
                        data: inline.data.clone(),
                    });
                } else if mime.starts_with("image/") {
                    content_parts.push(IrContentPart::ImageBase64 {
                        media_type: inline.mime_type.clone(),
                        data: inline.data.clone(),
                    });
                } else if mime == "application/pdf" || mime.starts_with("text/") {
                    content_parts.push(IrContentPart::Document {
                        media_type: inline.mime_type.clone(),
                        data: inline.data.clone(),
                        filename: None,
                    });
                } else {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("google"),
                        payload: serde_json::json!({
                            "inline_data": {
                                "mime_type": inline.mime_type,
                                "data": inline.data,
                            }
                        }),
                    });
                }
            }
            if let Some(ref fd) = part.file_data {
                content_parts.push(IrContentPart::FileRef {
                    file_id: fd.file_uri.clone(),
                });
            }
            // executableCode / codeExecutionResult 等未建模字段 → Opaque
            if !part.extra.is_empty()
                && part.text.is_none()
                && part.inline_data.is_none()
                && part.function_call.is_none()
                && part.function_response.is_none()
                && part.file_data.is_none()
            {
                content_parts.push(IrContentPart::Opaque {
                    provider: Cow::Borrowed("google"),
                    payload: serde_json::Value::Object(part.extra.clone()),
                });
            }
        }

        // provider_metadata: 保留 safetyRatings, groundingMetadata
        let mut pm = serde_json::Map::new();
        if let Some(ref sr) = candidate.safety_ratings {
            pm.insert("safety_ratings".into(), sr.clone());
        }
        if let Some(ref gm) = candidate.grounding_metadata {
            pm.insert("grounding_metadata".into(), gm.clone());
        }
        if let Some(ref sep) = candidate.search_entry_point {
            pm.insert("search_entry_point".into(), sep.clone());
        }
        if let Some(ref cm) = candidate.citation_metadata {
            pm.insert("citation_metadata".into(), cm.clone());
        }
        let provider_metadata = if pm.is_empty() {
            None
        } else {
            Some(Box::new(serde_json::Value::Object(pm)))
        };

        let content = match content_parts.len() {
            0 => IrContent::Text(Cow::Borrowed("")),
            1 => {
                if let IrContentPart::Text { ref text, .. } = content_parts[0] {
                    IrContent::Text(text.clone())
                } else {
                    IrContent::Parts(content_parts)
                }
            }
            _ => IrContent::Parts(content_parts),
        };

        (content, Vec::new(), provider_metadata)
    }

    fn convert_usage(u: &GemUsage) -> IrUsage {
        let prompt = u.prompt_token_count.unwrap_or(0);
        let completion = u.candidates_token_count.unwrap_or(0);
        IrUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: u
                .total_token_count
                .unwrap_or_else(|| prompt.saturating_add(completion)),
            cache_read_tokens: u.cached_content_token_count,
            cache_creation_tokens: None,
            reasoning_tokens: u.thoughts_token_count,
            audio_tokens: None,
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }
}

impl EncodeRequest for GeminiShim {
    fn encode_request(&self, ir: &IrRequest<'_>) -> Result<Vec<u8>, CodecError> {
        super::validate_tool_arguments(&ir.messages)?;
        let mut system_parts: Vec<GemPart<'_>> = Vec::new();
        let mut contents: Vec<GemContent<'_>> = Vec::new();
        let mut preserved: Vec<serde_json::Value> = Vec::new();

        for msg in &ir.messages {
            if msg.role == Role::System || msg.role == Role::Developer {
                let text = msg.content.text_concat();
                system_parts.push(GemPart {
                    text: Some(Cow::Owned(text)),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                    file_data: None,
                    extra: serde_json::Map::new(),
                });
                continue;
            }

            let parts = Self::ir_content_to_parts(msg, &mut preserved);
            contents.push(GemContent {
                role: Some(Self::ir_role_to_gem(msg.role)),
                parts,
            });
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(GemContent {
                role: None,
                parts: system_parts,
            })
        };

        // generationConfig
        let has_gen_config = ir.temperature.is_some()
            || ir.top_p.is_some()
            || ir.top_k.is_some()
            || ir.max_tokens.is_some()
            || ir.stop.is_some()
            || ir.n.is_some()
            || ir.response_format.is_some()
            || ir.reasoning.is_some()
            || ir.frequency_penalty.is_some()
            || ir.presence_penalty.is_some()
            || ir.seed.is_some()
            || ir.logprobs.is_some();

        let generation_config = if has_gen_config {
            // 结构化输出 → responseMimeType + responseSchema
            let (response_mime_type, response_schema) = match ir.response_format {
                Some(ref rf) => match rf.r#type {
                    ResponseFormatType::Text => (None, None),
                    ResponseFormatType::JsonObject => (Some("application/json"), None),
                    ResponseFormatType::JsonSchema => {
                        // Gemini 的 responseSchema 只承载纯 JSON Schema，无处安放
                        // name/strict — 经保留通道透传避免静默丢失
                        let mut rf_meta = serde_json::Map::new();
                        if let Some(ref name) = rf.name {
                            rf_meta.insert("name".into(), serde_json::json!(name));
                        }
                        if let Some(strict) = rf.strict {
                            rf_meta.insert("strict".into(), serde_json::json!(strict));
                        }
                        if !rf_meta.is_empty() {
                            preserved.push(serde_json::json!({ "response_format": rf_meta }));
                        }
                        (Some("application/json"), rf.schema.as_ref())
                    }
                },
                None => (None, None),
            };

            // ReasoningConfig → thinkingConfig
            let thinking_config = ir.reasoning.as_ref().map(|r| {
                if r.mode == ReasoningMode::Disabled {
                    GemThinkingConfig {
                        thinking_budget: Some(0),
                    }
                } else {
                    GemThinkingConfig {
                        thinking_budget: r.budget_tokens,
                    }
                }
            });

            Some(GemGenConfig {
                temperature: ir.temperature,
                top_p: ir.top_p,
                top_k: ir.top_k,
                max_output_tokens: ir.max_tokens,
                stop_sequences: ir
                    .stop
                    .as_ref()
                    .map(|s| s.iter().map(|v| v.to_string()).collect()),
                candidate_count: ir.n,
                response_mime_type,
                response_schema,
                thinking_config,
                frequency_penalty: ir.frequency_penalty,
                presence_penalty: ir.presence_penalty,
                seed: ir.seed,
                response_logprobs: ir.logprobs,
            })
        } else {
            None
        };

        // 工具分三类：function → functionDeclarations（合并为单个 GemTool），
        // CodeInterpreter → codeExecution，WebSearch → googleSearchRetrieval
        // （各自独立 GemTool，config 取自 extra）
        let tools: Option<Vec<GemTool<'_>>> = ir.tools.as_ref().and_then(|ts| {
            let mut func_decls: Vec<GemFuncDecl<'_>> = Vec::new();
            let mut extra_tools: Vec<GemTool<'_>> = Vec::new();
            for t in ts.iter() {
                match t.tool_type {
                    IrToolType::Function => {
                        func_decls.push(GemFuncDecl {
                            name: &t.name,
                            description: t.description.as_deref(),
                            parameters: &t.parameters,
                        });
                    }
                    IrToolType::CodeInterpreter => {
                        extra_tools.push(GemTool {
                            function_declarations: None,
                            code_execution: Some(
                                t.extra
                                    .as_ref()
                                    .map(Cow::Borrowed)
                                    .unwrap_or_else(|| Cow::Owned(serde_json::json!({}))),
                            ),
                            google_search_retrieval: None,
                        });
                    }
                    IrToolType::WebSearch => {
                        extra_tools.push(GemTool {
                            function_declarations: None,
                            code_execution: None,
                            google_search_retrieval: Some(
                                t.extra
                                    .as_ref()
                                    .map(Cow::Borrowed)
                                    .unwrap_or_else(|| Cow::Owned(serde_json::json!({}))),
                            ),
                        });
                    }
                    _ => {
                        if let Ok(v) = serde_json::to_value(t) {
                            preserved.push(serde_json::json!({
                                "type": "unsupported_tool",
                                "tool": v,
                            }));
                        }
                    }
                }
            }
            let mut gem_tools: Vec<GemTool<'_>> = Vec::new();
            if !func_decls.is_empty() {
                gem_tools.push(GemTool {
                    function_declarations: Some(func_decls),
                    code_execution: None,
                    google_search_retrieval: None,
                });
            }
            gem_tools.extend(extra_tools);
            if gem_tools.is_empty() {
                None
            } else {
                Some(gem_tools)
            }
        });

        // safetySettings 从 provider_metadata 中提取（Strip 模式下跳过）
        let safety_settings = if ir.metadata_mode == MetadataMode::Strip {
            None
        } else {
            ir.provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("safety_settings"))
                .map(|v| v as &serde_json::Value)
        };

        // cachedContent 从 provider_metadata 透传（Strip 模式下跳过）
        let cached_content = if ir.metadata_mode == MetadataMode::Strip {
            None
        } else {
            ir.provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("cached_content"))
                .and_then(|v| v.as_str())
        };

        // toolConfig from tool_choice / parallel_tool_calls
        let tool_config = ir.tool_choice.as_ref().map(|tc| {
            let (mode, allowed) = match tc {
                IrToolChoice::Auto => ("AUTO", None),
                IrToolChoice::None => ("NONE", None),
                IrToolChoice::Required => ("ANY", None),
                IrToolChoice::Specific { name } => ("ANY", Some(vec![name.to_string()])),
            };
            GemToolConfig {
                function_calling_config: GemFunctionCallingConfig {
                    mode,
                    allowed_function_names: allowed,
                },
            }
        });

        let req = GemRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
            safety_settings,
            tool_config,
            cached_content,
        };

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        let mut out = serde_json::to_value(&req).map_err(CodecError::from)?;
        super::attach_preserved(&mut out, preserved);
        serde_json::to_vec(&out).map_err(CodecError::from)
    }

    fn endpoint(&self, base_url: &str) -> String {
        format!("{base_url}:generateContent")
    }

    fn headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![
            ("Content-Type", "application/json".into()),
            ("x-goog-api-key", api_key.to_string()),
        ]
    }
}

impl DecodeResponse for GeminiShim {
    fn decode_response<'a>(&self, body: &'a [u8]) -> Result<IrResponse<'a>, CodecError> {
        let gem: GemResponse<'a> = serde_json::from_slice(body)?;

        // 收集候选级 provider_metadata
        let mut response_pm = serde_json::Map::new();

        let choices: Vec<IrChoice<'a>> = gem
            .candidates
            .iter()
            .enumerate()
            .map(|(i, cand)| {
                let (content, tool_calls, cand_pm) = cand
                    .content
                    .as_ref()
                    .map(|c| Self::gem_parts_to_ir_message(&c.parts, cand))
                    .unwrap_or((IrContent::Text(Cow::Borrowed("")), Vec::new(), None));

                // 合并候选级 pm 到 response 级
                if let Some(pm) = cand_pm {
                    if let serde_json::Value::Object(map) = *pm {
                        for (k, v) in map {
                            response_pm.insert(k, v);
                        }
                    }
                }

                let message = IrMessage {
                    role: Role::Assistant,
                    content,
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    cache_control: None,
                    refusal: None,
                };

                IrChoice {
                    index: cand.index.unwrap_or(i as u32),
                    message,
                    finish_reason: cand.finish_reason.as_deref().map(Self::parse_finish_reason),
                    logprobs: cand.avg_logprobs.map(|avg| IrLogprobs {
                        content: None,
                        refusal: None,
                        avg_logprob: Some(avg),
                    }),
                }
            })
            .collect();

        let usage = gem.usage_metadata.as_ref().map(Self::convert_usage);

        if let Some(ref pf) = gem.prompt_feedback {
            response_pm.insert("prompt_feedback".into(), pf.clone());
        }
        if let Some(ref meta) = gem.metadata {
            response_pm.insert("metadata".into(), meta.clone());
        }

        let mut provider_metadata = if response_pm.is_empty() {
            None
        } else {
            Some(Box::new(serde_json::Value::Object(response_pm)))
        };

        // 从响应体提取无损保留部件
        let raw: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
        let preserved = super::extract_preserved(&raw);
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        let id_str = gem.response_id.unwrap_or_default();
        Ok(IrResponse {
            id: Cow::Owned(id_str),
            model: gem.model_version.unwrap_or(Cow::Borrowed("")),
            choices,
            usage,
            provider_metadata,
        })
    }
}

/// Gemini 流式解码器 — 每条流一个实例。
/// Gemini 流协议无显式生命周期信号：解码器在首 chunk 合成 Start，
/// 跟踪 thought/text 状态在 finish 时合成 ReasoningDone/ContentDone，
/// 为每条流内的 function_call 分配全局递增 id。
struct GemCandidateState {
    index: u32,
    reasoning_open: bool,
    content_open: bool,
    finished: bool,
    pending_thought_signature: Option<String>,
}

pub struct GemStreamDecoder {
    started: bool,
    done_sent: bool,
    /// 每个 candidate 独立维护块生命周期和思考签名，避免 n>1 串流污染。
    seen: Vec<GemCandidateState>,
    /// 流内 function_call 计数，保证 id 全流唯一
    tool_seq: u32,
    metadata_seq: u32,
    /// 流式 candidate 级 provider_metadata 缓冲（groundingMetadata /
    /// searchEntryPoint / citationMetadata）。IR 流事件无对应变体，缓冲到
    /// 解码器状态，聚合器在流结束后经 take_provider_metadata 取出附加到重建响应。
    pending_metadata: serde_json::Map<String, serde_json::Value>,
}

impl GemStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            done_sent: false,
            seen: Vec::new(),
            tool_seq: 0,
            metadata_seq: 0,
            pending_metadata: serde_json::Map::new(),
        }
    }

    /// 取出流式过程中缓冲的 candidate 级 provider_metadata。
    /// IR 流事件模型无携带 provider_metadata 的变体 —— 聚合器在流结束后调用，
    /// 将结果并入重建 IrResponse 的 provider_metadata。
    pub fn take_provider_metadata(&mut self) -> Option<Box<serde_json::Value>> {
        if self.pending_metadata.is_empty() {
            None
        } else {
            Some(Box::new(serde_json::Value::Object(std::mem::take(
                &mut self.pending_metadata,
            ))))
        }
    }

    fn seen_pos(&mut self, index: u32) -> usize {
        match self.seen.iter().position(|s| s.index == index) {
            Some(pos) => pos,
            None => {
                self.seen.push(GemCandidateState {
                    index,
                    reasoning_open: false,
                    content_open: false,
                    finished: false,
                    pending_thought_signature: None,
                });
                self.seen.len() - 1
            }
        }
    }

    fn finalize(&mut self) -> Vec<IrStreamEvent<'static>> {
        if self.done_sent {
            return Vec::new();
        }
        self.done_sent = true;
        let mut events = Vec::new();
        for state in &mut self.seen {
            if state.reasoning_open {
                state.reasoning_open = false;
                events.push(IrStreamEvent::ReasoningDone {
                    index: state.index,
                    signature: state.pending_thought_signature.take().map(Cow::Owned),
                });
            }
            if state.content_open {
                state.content_open = false;
                events.push(IrStreamEvent::ContentDone { index: state.index });
            }
            if !state.finished {
                state.finished = true;
                events.push(IrStreamEvent::ChoiceFinish {
                    index: state.index,
                    finish_reason: IrFinishReason::Stop,
                });
            }
        }
        events.push(IrStreamEvent::Done);
        events
    }
}

impl Default for GemStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeStream for GemStreamDecoder {
    fn decode_sse_data<'a>(
        &mut self,
        data: &'a [u8],
    ) -> Result<Vec<IrStreamEvent<'a>>, CodecError> {
        if self.done_sent {
            return Err(CodecError::InvalidState(
                "Gemini stream received data after completion".to_string(),
            ));
        }
        let gem: GemResponse<'a> = serde_json::from_slice(data)?;

        // 流式错误快路径：Gemini 以 {"error":{"message":"..."}} 形式下发错误，
        // 因全部字段可选，反序列化会成功但不产出任何事件，导致错误被静默吞掉。
        if let Some(ref err) = gem.error {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            self.done_sent = true;
            return Ok(vec![IrStreamEvent::Error {
                message: Cow::Owned(msg.to_string()),
            }]);
        }

        let mut events = Vec::with_capacity(4);

        // Gemini 无 Start 信号 — 首 chunk 合成（id 为空，modelVersion 可用则带上）
        if !self.started {
            self.started = true;
            events.push(IrStreamEvent::Start {
                id: Cow::Borrowed(""),
                model: gem.model_version.clone().unwrap_or(Cow::Borrowed("")),
                usage: None,
            });
        }

        for cand in &gem.candidates {
            let idx = cand.index.unwrap_or(0);
            // candidate 级元数据缓冲（末次到达者覆盖，Gemini 通常在末 chunk 回传）
            if let Some(ref gm) = cand.grounding_metadata {
                self.pending_metadata
                    .insert("grounding_metadata".into(), gm.clone());
            }
            if let Some(ref sep) = cand.search_entry_point {
                self.pending_metadata
                    .insert("search_entry_point".into(), sep.clone());
            }
            if let Some(ref cm) = cand.citation_metadata {
                self.pending_metadata
                    .insert("citation_metadata".into(), cm.clone());
            }
            if let Some(ref sr) = cand.safety_ratings {
                self.pending_metadata
                    .insert("safety_ratings".into(), sr.clone());
            }
            if let Some(avg) = cand.avg_logprobs {
                self.pending_metadata
                    .insert("avg_logprobs".into(), serde_json::json!(avg));
            }
            if let Some(ref content) = cand.content {
                for part in content.parts.iter() {
                    if let Some(ref text) = part.text {
                        if !text.is_empty() {
                            let pos = self.seen_pos(idx);
                            if part.thought == Some(true) {
                                // text→thought 切换：关闭已打开的正文块
                                if self.seen[pos].content_open {
                                    self.seen[pos].content_open = false;
                                    events.push(IrStreamEvent::ContentDone { index: idx });
                                }
                                self.seen[pos].reasoning_open = true;
                                if part.thought_signature.is_some() {
                                    self.seen[pos].pending_thought_signature =
                                        part.thought_signature.as_ref().map(|s| s.to_string());
                                }
                                events.push(IrStreamEvent::ReasoningDelta {
                                    index: idx,
                                    delta: text.clone(),
                                });
                            } else {
                                // thought → text 切换：合成 ReasoningDone（Gemini 思考先于正文）
                                if self.seen[pos].reasoning_open {
                                    self.seen[pos].reasoning_open = false;
                                    events.push(IrStreamEvent::ReasoningDone {
                                        index: idx,
                                        signature: self.seen[pos]
                                            .pending_thought_signature
                                            .take()
                                            .map(Cow::Owned),
                                    });
                                }
                                self.seen[pos].content_open = true;
                                events.push(IrStreamEvent::ContentDelta {
                                    index: idx,
                                    delta: text.clone(),
                                });
                            }
                        }
                    }
                    if let Some(ref fc) = part.function_call {
                        let pos = self.seen_pos(idx);
                        // text→tool 切换：关闭已打开的正文块
                        if self.seen[pos].content_open {
                            self.seen[pos].content_open = false;
                            events.push(IrStreamEvent::ContentDone { index: idx });
                        }
                        // thought→tool 切换 或 无思考但带 thoughtSignature
                        if self.seen[pos].reasoning_open || part.thought_signature.is_some() {
                            self.seen[pos].reasoning_open = false;
                            events.push(IrStreamEvent::ReasoningDone {
                                index: idx,
                                signature: part.thought_signature.clone().or_else(|| {
                                    self.seen[pos]
                                        .pending_thought_signature
                                        .take()
                                        .map(Cow::Owned)
                                }),
                            });
                        }
                        let args_str = serde_json::to_string(&fc.args).unwrap_or_default();
                        let tool_idx = self.tool_seq;
                        let call_id = format!("call_gemini_{tool_idx}");
                        self.tool_seq += 1;
                        // Gemini 流式 function_call 是完整的，一次性发
                        // Start + Delta(全量) + Done — 下游增量式编码器
                        // （OpenAI/Anthropic）依赖 Delta 携带 arguments
                        events.push(IrStreamEvent::ToolCallStart {
                            index: tool_idx,
                            choice_index: idx,
                            id: Cow::Owned(call_id.clone()),
                            name: fc.name.clone(),
                        });
                        if !args_str.is_empty() {
                            events.push(IrStreamEvent::ToolCallDelta {
                                index: tool_idx,
                                choice_index: idx,
                                arguments_delta: Cow::Owned(args_str.clone()),
                            });
                        }
                        events.push(IrStreamEvent::ToolCallDone {
                            index: tool_idx,
                            choice_index: idx,
                            id: Cow::Owned(call_id),
                            name: fc.name.clone(),
                            arguments: Cow::Owned(args_str),
                        });
                    }
                    // 流式 inline_data / file_data → IR 无原生流事件，缓冲到 pending_metadata
                    if let Some(ref inline) = part.inline_data {
                        let key = format!("_stream_inline_data_{}", self.metadata_seq);
                        self.metadata_seq += 1;
                        self.pending_metadata.insert(
                            key,
                            serde_json::json!({
                                "mime_type": inline.mime_type,
                                "data": inline.data,
                            }),
                        );
                    }
                    if let Some(ref fd) = part.file_data {
                        let key = format!("_stream_file_data_{}", self.metadata_seq);
                        self.metadata_seq += 1;
                        self.pending_metadata.insert(
                            key,
                            serde_json::json!({
                                "file_uri": fd.file_uri,
                            }),
                        );
                    }
                }
            }
            if let Some(ref fr) = cand.finish_reason {
                let pos = self.seen_pos(idx);
                if self.seen[pos].reasoning_open {
                    self.seen[pos].reasoning_open = false;
                    let sig = cand.content.as_ref().and_then(|c| {
                        c.parts
                            .iter()
                            .rev()
                            .find(|p| p.thought == Some(true))
                            .and_then(|p| p.thought_signature.clone())
                    });
                    events.push(IrStreamEvent::ReasoningDone {
                        index: idx,
                        signature: sig.or_else(|| {
                            self.seen[pos]
                                .pending_thought_signature
                                .take()
                                .map(Cow::Owned)
                        }),
                    });
                }
                if self.seen[pos].content_open {
                    self.seen[pos].content_open = false;
                    events.push(IrStreamEvent::ContentDone { index: idx });
                }
                if !self.seen[pos].finished {
                    self.seen[pos].finished = true;
                    events.push(IrStreamEvent::ChoiceFinish {
                        index: idx,
                        finish_reason: GeminiShim::parse_finish_reason(fr),
                    });
                }
            }
        }

        if let Some(ref u) = gem.usage_metadata {
            events.push(IrStreamEvent::Usage(GeminiShim::convert_usage(u)));
        }

        if !self.done_sent && !self.seen.is_empty() && self.seen.iter().all(|s| s.finished) {
            self.done_sent = true;
            events.push(IrStreamEvent::Done);
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<IrStreamEvent<'static>>, CodecError> {
        Ok(self.finalize())
    }
}

// ─── DecodeRequest ──────────────────────────────────────────────────

/// Gemini generateContent 请求 — 反序列化用
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemRequestIn<'a> {
    #[serde(borrow, default)]
    contents: Vec<GemContent<'a>>,
    #[serde(borrow, default)]
    system_instruction: Option<GemContent<'a>>,
    #[serde(default)]
    generation_config: Option<GemGenConfigIn<'a>>,
    #[serde(default)]
    tools: Option<Vec<GemToolIn<'a>>>,
    #[serde(default)]
    safety_settings: Option<serde_json::Value>,
    #[serde(default)]
    tool_config: Option<GemToolConfigIn>,
    #[serde(borrow, default)]
    cached_content: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemGenConfigIn<'a> {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(borrow, default)]
    stop_sequences: Option<Vec<Cow<'a, str>>>,
    #[serde(default)]
    candidate_count: Option<u32>,
    #[serde(borrow, default)]
    response_mime_type: Option<Cow<'a, str>>,
    #[serde(default)]
    response_schema: Option<serde_json::Value>,
    #[serde(default)]
    thinking_config: Option<GemThinkingConfigIn>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    response_logprobs: Option<bool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemThinkingConfigIn {
    thinking_budget: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemToolIn<'a> {
    #[serde(borrow, default)]
    function_declarations: Vec<GemFuncDeclIn<'a>>,
    /// codeExecution 工具（无 name/description/parameters）
    #[serde(default)]
    code_execution: Option<serde_json::Value>,
    /// googleSearchRetrieval 工具（可带 dynamicRetrievalConfig）
    #[serde(default)]
    google_search_retrieval: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct GemFuncDeclIn<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(borrow, default)]
    description: Option<Cow<'a, str>>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemToolConfigIn {
    #[serde(default)]
    function_calling_config: Option<GemFunctionCallingConfigIn>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GemFunctionCallingConfigIn {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    allowed_function_names: Option<Vec<String>>,
}

impl GeminiShim {
    fn gem_role_to_ir(role: &str) -> Role {
        match role {
            "user" => Role::User,
            "model" => Role::Assistant,
            _ => Role::User,
        }
    }

    fn gem_parts_to_ir_content_request<'a>(
        parts: &[GemPart<'a>],
        call_counter: &mut u32,
    ) -> (
        IrContent<'static>,
        Option<Cow<'static, str>>,
        Option<Vec<IrToolCall<'static>>>,
    ) {
        let mut content_parts: Vec<IrContentPart<'static>> = Vec::new();
        let mut tool_call_id: Option<Cow<'static, str>> = None;

        for part in parts.iter() {
            if let Some(ref text) = part.text {
                if part.thought == Some(true) {
                    content_parts.push(IrContentPart::Reasoning {
                        text: Cow::Owned(text.to_string()),
                        signature: part
                            .thought_signature
                            .as_ref()
                            .map(|s| Cow::Owned(s.to_string())),
                    });
                } else {
                    content_parts.push(IrContentPart::Text {
                        text: Cow::Owned(text.to_string()),
                        citations: None,
                    });
                }
            }
            if let Some(ref inline) = part.inline_data {
                let mime = inline.mime_type.as_ref();
                if mime.starts_with("image/") {
                    content_parts.push(IrContentPart::ImageBase64 {
                        media_type: Cow::Owned(inline.mime_type.to_string()),
                        data: Cow::Owned(inline.data.to_string()),
                    });
                } else if mime.starts_with("audio/") {
                    content_parts.push(IrContentPart::Audio {
                        media_type: Cow::Owned(inline.mime_type.to_string()),
                        data: Cow::Owned(inline.data.to_string()),
                    });
                } else if mime.starts_with("video/") {
                    content_parts.push(IrContentPart::Video {
                        media_type: Cow::Owned(inline.mime_type.to_string()),
                        data: Cow::Owned(inline.data.to_string()),
                    });
                } else if mime == "application/pdf" || mime.starts_with("text/") {
                    content_parts.push(IrContentPart::Document {
                        media_type: Cow::Owned(inline.mime_type.to_string()),
                        data: Cow::Owned(inline.data.to_string()),
                        filename: None,
                    });
                } else {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("google"),
                        payload: serde_json::json!({
                            "inline_data": {
                                "mime_type": inline.mime_type,
                                "data": inline.data,
                            }
                        }),
                    });
                }
            }
            if let Some(ref fd) = part.file_data {
                content_parts.push(IrContentPart::FileRef {
                    file_id: Cow::Owned(fd.file_uri.to_string()),
                });
            }
            if let Some(ref fc) = part.function_call {
                if let Some(ref sig) = part.thought_signature {
                    let mut backfilled = false;
                    for cp in content_parts.iter_mut().rev() {
                        if let IrContentPart::Reasoning { signature, .. } = cp {
                            if signature.is_none() {
                                *signature = Some(Cow::Owned(sig.to_string()));
                                backfilled = true;
                                break;
                            }
                        }
                    }
                    if !backfilled {
                        content_parts.push(IrContentPart::Reasoning {
                            text: Cow::Borrowed(""),
                            signature: Some(Cow::Owned(sig.to_string())),
                        });
                    }
                }
                let args = serde_json::to_string(&fc.args).unwrap_or_default();
                let cid = *call_counter;
                *call_counter += 1;
                content_parts.push(IrContentPart::FunctionCall {
                    id: Cow::Owned(format!("call_gemini_{cid}")),
                    name: Cow::Owned(fc.name.to_string()),
                    arguments: Cow::Owned(args),
                });
            }
            if let Some(ref fr) = part.function_response {
                // function_response → Tool 消息，role 处理在调用方
                tool_call_id = Some(Cow::Owned(fr.name.to_string()));
                let text = serde_json::to_string(&fr.response).unwrap_or_default();
                content_parts.push(IrContentPart::Text {
                    text: Cow::Owned(text),
                    citations: None,
                });
            }
            if !part.extra.is_empty()
                && part.text.is_none()
                && part.inline_data.is_none()
                && part.function_call.is_none()
                && part.function_response.is_none()
                && part.file_data.is_none()
            {
                content_parts.push(IrContentPart::Opaque {
                    provider: Cow::Borrowed("google"),
                    payload: serde_json::Value::Object(
                        part.extra
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    ),
                });
            }
        }

        let tc: Option<Vec<IrToolCall<'static>>> = None;

        let content = match content_parts.len() {
            0 => IrContent::Text(Cow::Owned(String::new())),
            1 => {
                if let IrContentPart::Text { ref text, .. } = content_parts[0] {
                    IrContent::Text(Cow::Owned(text.to_string()))
                } else {
                    IrContent::Parts(content_parts)
                }
            }
            _ => IrContent::Parts(content_parts),
        };

        (content, tool_call_id, tc)
    }

    fn finish_reason_to_str(fr: IrFinishReason) -> &'static str {
        match fr {
            IrFinishReason::Stop => "STOP",
            IrFinishReason::Length => "MAX_TOKENS",
            IrFinishReason::MalformedFunctionCall => "MALFORMED_FUNCTION_CALL",
            IrFinishReason::Recitation => "RECITATION",
            IrFinishReason::Safety | IrFinishReason::ContentFilter => "SAFETY",
            IrFinishReason::ToolCalls
            | IrFinishReason::PauseTurn
            | IrFinishReason::StopSequence => "STOP",
        }
    }
}

impl DecodeRequest for GeminiShim {
    fn decode_request<'a>(&self, body: &'a [u8]) -> Result<IrRequest<'a>, CodecError> {
        let req: GemRequestIn<'a> = serde_json::from_slice(body)?;

        let mut messages: Vec<IrMessage<'_>> = Vec::new();

        // system_instruction
        if let Some(ref si) = req.system_instruction {
            let text: String = si
                .parts
                .iter()
                .filter_map(|p| p.text.as_ref().map(|t| t.as_ref().to_string()))
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                messages.push(IrMessage {
                    role: Role::System,
                    content: IrContent::Text(Cow::Owned(text)),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls: None,
                    cache_control: None,
                    refusal: None,
                });
            }
        }

        let mut gem_call_counter: u32 = 0;

        // contents
        for content in &req.contents {
            let role_str = content.role.unwrap_or("user");
            let role = Self::gem_role_to_ir(role_str);

            // 检查是否有 function_response — 这些需要变成 Tool 消息；
            // 同 content 内的其他部件（text 等）保留为并列消息
            let has_function_response = content.parts.iter().any(|p| p.function_response.is_some());
            if has_function_response {
                let sibling_parts: Vec<&GemPart<'_>> = content
                    .parts
                    .iter()
                    .filter(|p| p.function_response.is_none())
                    .collect();
                for part in &content.parts {
                    if let Some(ref fr) = part.function_response {
                        let text = serde_json::to_string(&fr.response).unwrap_or_default();
                        messages.push(IrMessage {
                            role: Role::Tool,
                            content: IrContent::Text(Cow::Owned(text)),
                            tool_call_id: Some(Cow::Owned(fr.name.to_string())),
                            tool_name: Some(Cow::Owned(fr.name.to_string())),
                            tool_calls: None,
                            cache_control: None,
                            refusal: None,
                        });
                    }
                }
                // 兄弟部件（text/inline_data 等）→ 并列消息，不丢弃
                if !sibling_parts.is_empty() {
                    let mut ir_parts: Vec<IrContentPart<'static>> = Vec::new();
                    for part in &sibling_parts {
                        if let Some(ref text) = part.text {
                            if part.thought == Some(true) {
                                ir_parts.push(IrContentPart::Reasoning {
                                    text: Cow::Owned(text.to_string()),
                                    signature: part
                                        .thought_signature
                                        .as_ref()
                                        .map(|s| Cow::Owned(s.to_string())),
                                });
                            } else {
                                ir_parts.push(IrContentPart::Text {
                                    text: Cow::Owned(text.to_string()),
                                    citations: None,
                                });
                            }
                        }
                        if let Some(ref inline) = part.inline_data {
                            let mime = inline.mime_type.as_ref();
                            let part = if mime.starts_with("audio/") {
                                IrContentPart::Audio {
                                    media_type: Cow::Owned(inline.mime_type.to_string()),
                                    data: Cow::Owned(inline.data.to_string()),
                                }
                            } else if mime.starts_with("video/") {
                                IrContentPart::Video {
                                    media_type: Cow::Owned(inline.mime_type.to_string()),
                                    data: Cow::Owned(inline.data.to_string()),
                                }
                            } else if mime.starts_with("image/") {
                                IrContentPart::ImageBase64 {
                                    media_type: Cow::Owned(inline.mime_type.to_string()),
                                    data: Cow::Owned(inline.data.to_string()),
                                }
                            } else if mime == "application/pdf" || mime.starts_with("text/") {
                                IrContentPart::Document {
                                    media_type: Cow::Owned(inline.mime_type.to_string()),
                                    data: Cow::Owned(inline.data.to_string()),
                                    filename: None,
                                }
                            } else {
                                IrContentPart::Opaque {
                                    provider: Cow::Borrowed("google"),
                                    payload: serde_json::json!({
                                        "inline_data": {
                                            "mime_type": inline.mime_type,
                                            "data": inline.data,
                                        }
                                    }),
                                }
                            };
                            ir_parts.push(part);
                        }
                        if let Some(ref fd) = part.file_data {
                            ir_parts.push(IrContentPart::FileRef {
                                file_id: Cow::Owned(fd.file_uri.to_string()),
                            });
                        }
                        if let Some(ref fc) = part.function_call {
                            let args = serde_json::to_string(&fc.args).unwrap_or_default();
                            let cid = gem_call_counter;
                            gem_call_counter += 1;
                            ir_parts.push(IrContentPart::FunctionCall {
                                id: Cow::Owned(format!("call_gemini_{cid}")),
                                name: Cow::Owned(fc.name.to_string()),
                                arguments: Cow::Owned(args),
                            });
                        }
                        if !part.extra.is_empty()
                            && part.text.is_none()
                            && part.inline_data.is_none()
                            && part.function_call.is_none()
                            && part.function_response.is_none()
                            && part.file_data.is_none()
                        {
                            ir_parts.push(IrContentPart::Opaque {
                                provider: Cow::Borrowed("google"),
                                payload: serde_json::Value::Object(part.extra.clone()),
                            });
                        }
                    }
                    if !ir_parts.is_empty() {
                        messages.push(IrMessage {
                            role,
                            content: IrContent::Parts(ir_parts),
                            tool_call_id: None,
                            tool_name: None,
                            tool_calls: None,
                            cache_control: None,
                            refusal: None,
                        });
                    }
                }
                continue;
            }

            let (content_ir, _tool_call_id, tool_calls) =
                Self::gem_parts_to_ir_content_request(&content.parts, &mut gem_call_counter);

            messages.push(IrMessage {
                role,
                content: content_ir,
                tool_call_id: None,
                tool_name: None,
                tool_calls,
                cache_control: None,
                refusal: None,
            });
        }

        // tools：functionDeclarations → Function；codeExecution → CodeInterpreter；
        // googleSearchRetrieval → WebSearch（后两者无 name/params，config 存 extra）
        let tools: Option<Vec<IrTool<'_>>> = req.tools.as_ref().map(|ts| {
            let mut out: Vec<IrTool<'_>> = Vec::new();
            for t in ts.iter() {
                for fd in t.function_declarations.iter() {
                    out.push(IrTool {
                        tool_type: IrToolType::Function,
                        name: fd.name.clone(),
                        description: fd.description.clone(),
                        parameters: fd.parameters.clone().unwrap_or(serde_json::json!({})),
                        cache_control: None,
                        extra: None,
                    });
                }
                if let Some(ref ce) = t.code_execution {
                    out.push(IrTool {
                        tool_type: IrToolType::CodeInterpreter,
                        name: Cow::Borrowed("code_execution"),
                        description: None,
                        parameters: serde_json::json!({}),
                        cache_control: None,
                        extra: Some(ce.clone()),
                    });
                }
                if let Some(ref gs) = t.google_search_retrieval {
                    out.push(IrTool {
                        tool_type: IrToolType::WebSearch,
                        name: Cow::Borrowed("google_search_retrieval"),
                        description: None,
                        parameters: serde_json::json!({}),
                        cache_control: None,
                        extra: Some(gs.clone()),
                    });
                }
            }
            out
        });

        // generation_config
        let gc = req.generation_config.as_ref();
        let temperature = gc.and_then(|g| g.temperature);
        let top_p = gc.and_then(|g| g.top_p);
        let top_k = gc.and_then(|g| g.top_k);
        let max_tokens = gc.and_then(|g| g.max_output_tokens);
        let n = gc.and_then(|g| g.candidate_count);
        let stop = gc.and_then(|g| g.stop_sequences.clone());
        let frequency_penalty = gc.and_then(|g| g.frequency_penalty);
        let presence_penalty = gc.and_then(|g| g.presence_penalty);
        let seed = gc.and_then(|g| g.seed);
        let logprobs = gc.and_then(|g| g.response_logprobs);

        // response_format from responseMimeType + responseSchema
        let response_format = gc.and_then(|g| {
            let mime = g.response_mime_type.as_deref();
            match mime {
                Some("application/json") => {
                    if g.response_schema.is_some() {
                        Some(IrResponseFormat {
                            r#type: ResponseFormatType::JsonSchema,
                            schema: g.response_schema.clone(),
                            name: None,
                            strict: None,
                        })
                    } else {
                        Some(IrResponseFormat {
                            r#type: ResponseFormatType::JsonObject,
                            schema: None,
                            name: None,
                            strict: None,
                        })
                    }
                }
                _ => None,
            }
        });

        // reasoning from thinkingConfig
        let reasoning = gc.and_then(|g| {
            g.thinking_config.as_ref().map(|tc| ReasoningConfig {
                mode: ReasoningMode::Enabled,
                budget_tokens: tc.thinking_budget,
                effort: None,
            })
        });

        // tool_choice from toolConfig
        let tool_choice = req.tool_config.as_ref().and_then(|tc| {
            tc.function_calling_config.as_ref().map(|fcc| {
                let mode = fcc.mode.as_deref().unwrap_or("AUTO");
                match mode {
                    "NONE" => IrToolChoice::None,
                    "ANY" => {
                        if let Some(ref names) = fcc.allowed_function_names {
                            if names.len() == 1 {
                                IrToolChoice::Specific {
                                    name: Cow::Owned(names[0].clone()),
                                }
                            } else {
                                IrToolChoice::Required
                            }
                        } else {
                            IrToolChoice::Required
                        }
                    }
                    _ => IrToolChoice::Auto, // AUTO 及未知模式
                }
            })
        });

        // 从请求体提取无损保留部件
        let raw: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
        let preserved = super::extract_preserved(&raw);

        let mut pm_map = serde_json::Map::new();
        if let Some(ss) = req.safety_settings {
            pm_map.insert("safety_settings".into(), ss);
        }
        if let Some(cc) = req.cached_content {
            pm_map.insert(
                "cached_content".into(),
                serde_json::Value::String(cc.into_owned()),
            );
        }
        let mut provider_metadata = if pm_map.is_empty() {
            None
        } else {
            Some(Box::new(serde_json::Value::Object(pm_map)))
        };
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        // 回填 Tool 消息的 tool_name（Gemini functionResponse 仅带函数名，
        // 跨格式多轮工具对话需据此关联）
        super::backfill_tool_names(&mut messages);

        // Gemini 按名称关联工具结果，但 IR 按 id 关联。
        // 将 Tool 消息的 tool_call_id 从函数名替换为对应 FunctionCall 的合成 id。
        for i in 0..messages.len() {
            if messages[i].role == Role::Tool {
                if let Some(ref tname) = messages[i].tool_name {
                    let target = tname.to_string();
                    for j in (0..i).rev() {
                        let mut found = None;
                        if let IrContent::Parts(ref parts) = messages[j].content {
                            for p in parts {
                                if let IrContentPart::FunctionCall {
                                    ref id, ref name, ..
                                } = p
                                {
                                    if name.as_ref() == target {
                                        found = Some(id.to_string());
                                        break;
                                    }
                                }
                            }
                        }
                        if let Some(fid) = found {
                            messages[i].tool_call_id = Some(Cow::Owned(fid));
                            break;
                        }
                    }
                }
            }
        }

        Ok(IrRequest {
            model: Cow::Owned(String::new()), // Gemini 的 model 在 URL 中，不在请求体
            messages,
            temperature,
            top_p,
            top_k,
            max_tokens,
            stop,
            frequency_penalty,
            presence_penalty,
            seed,
            n,
            logprobs,
            top_logprobs: None,
            stream: false, // Gemini 用 URL 区分流式（streamGenerateContent）
            store: None,
            modalities: None,
            tools,
            tool_choice,
            parallel_tool_calls: None,
            reasoning,
            response_format,
            previous_response_id: None,
            truncation: None,
            metadata: None,
            provider_metadata,
            metadata_mode: MetadataMode::default(),
        })
    }
}

// ─── EncodeResponse ─────────────────────────────────────────────────

impl EncodeResponse for GeminiShim {
    fn encode_response(&self, ir: &IrResponse<'_>) -> Result<Vec<u8>, CodecError> {
        for choice in &ir.choices {
            super::validate_tool_arguments(std::slice::from_ref(&choice.message))?;
        }
        let mut preserved: Vec<serde_json::Value> = Vec::new();
        let candidates: Vec<serde_json::Value> = ir
            .choices
            .iter()
            .map(|c| {
                let mut parts: Vec<serde_json::Value> = Vec::new();

                match &c.message.content {
                    IrContent::Text(s) => {
                        if !s.is_empty() {
                            parts.push(serde_json::json!({ "text": s }));
                        }
                    }
                    IrContent::Parts(content_parts) => {
                        for p in content_parts {
                            match p {
                                IrContentPart::Text { text, citations } => {
                                    if let Some(ref cits) = citations {
                                        if let Ok(v) = serde_json::to_value(cits) {
                                            preserved
                                                .push(serde_json::json!({"text_citations": v}));
                                        }
                                    }
                                    parts.push(serde_json::json!({ "text": text }));
                                }
                                IrContentPart::Reasoning { text, signature } => {
                                    let mut part =
                                        serde_json::json!({ "text": text, "thought": true });
                                    if let Some(sig) = signature {
                                        part["thoughtSignature"] =
                                            serde_json::Value::String(sig.to_string());
                                    }
                                    parts.push(part);
                                }
                                IrContentPart::ImageBase64 {
                                    media_type, data, ..
                                }
                                | IrContentPart::Audio {
                                    media_type, data, ..
                                }
                                | IrContentPart::Video {
                                    media_type, data, ..
                                } => {
                                    parts.push(serde_json::json!({
                                        "inlineData": {
                                            "mimeType": media_type,
                                            "data": data,
                                        }
                                    }));
                                }
                                IrContentPart::Document {
                                    media_type,
                                    data,
                                    filename,
                                } => {
                                    if let Some(ref f) = filename {
                                        preserved.push(serde_json::json!({"document_filename": f}));
                                    }
                                    parts.push(serde_json::json!({
                                        "inlineData": {
                                            "mimeType": media_type,
                                            "data": data,
                                        }
                                    }));
                                }
                                IrContentPart::FunctionCall {
                                    name, arguments, ..
                                } => {
                                    let args: serde_json::Value = serde_json::from_str(arguments)
                                        .unwrap_or(serde_json::json!({}));
                                    parts.push(serde_json::json!({
                                        "functionCall": { "name": name, "args": args }
                                    }));
                                }
                                IrContentPart::FileRef { file_id } => {
                                    parts.push(serde_json::json!({
                                        "fileData": { "fileUri": file_id }
                                    }));
                                }
                                // 同源 Opaque → 直接回写为原生 part
                                IrContentPart::Opaque {
                                    provider, payload, ..
                                } if provider == "google" => {
                                    parts.push(payload.clone());
                                }
                                IrContentPart::FunctionResponse { .. }
                                | IrContentPart::ImageUrl { .. }
                                | IrContentPart::RedactedReasoning { .. }
                                | IrContentPart::Opaque { .. } => {
                                    if let Ok(v) = serde_json::to_value(p) {
                                        preserved.push(v);
                                    }
                                }
                            }
                        }
                    }
                }

                // tool_calls → functionCall parts
                if let Some(ref tcs) = c.message.tool_calls {
                    for tc in tcs {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                        parts.push(serde_json::json!({
                            "functionCall": {
                                "name": tc.name,
                                "args": args,
                            }
                        }));
                    }
                }

                let mut candidate = serde_json::json!({
                    "content": {
                        "role": "model",
                        "parts": parts,
                    },
                    "index": c.index,
                });

                if let Some(fr) = c.finish_reason {
                    candidate["finishReason"] =
                        serde_json::Value::String(Self::finish_reason_to_str(fr).to_string());
                }

                if let Some(ref lp) = c.logprobs {
                    if let Some(avg) = lp.avg_logprob {
                        candidate["avgLogprobs"] = serde_json::json!(avg);
                    }
                }

                if let Some(ref pm) = ir.provider_metadata {
                    if let Some(sr) = pm.get("safety_ratings") {
                        candidate["safetyRatings"] = sr.clone();
                    }
                    if let Some(gm) = pm.get("grounding_metadata") {
                        candidate["groundingMetadata"] = gm.clone();
                    }
                    if let Some(sep) = pm.get("search_entry_point") {
                        candidate["searchEntryPoint"] = sep.clone();
                    }
                    if let Some(cm) = pm.get("citation_metadata") {
                        candidate["citationMetadata"] = cm.clone();
                    }
                }

                candidate
            })
            .collect();

        let mut resp = serde_json::json!({ "candidates": candidates });

        if !ir.id.is_empty() {
            resp["responseId"] = serde_json::Value::String(ir.id.to_string());
        }
        if let Some(ref pm) = ir.provider_metadata {
            if let Some(pf) = pm.get("prompt_feedback") {
                resp["promptFeedback"] = pf.clone();
            }
            if let Some(meta) = pm.get("metadata") {
                resp["metadata"] = meta.clone();
            }
        }

        if let Some(ref usage) = ir.usage {
            resp["usageMetadata"] = serde_json::json!({
                "promptTokenCount": usage.prompt_tokens,
                "candidatesTokenCount": usage.completion_tokens,
                "totalTokenCount": usage.total_tokens,
            });
            if let Some(cr) = usage.cache_read_tokens {
                resp["usageMetadata"]["cachedContentTokenCount"] = serde_json::json!(cr);
            }
            if let Some(r) = usage.reasoning_tokens {
                resp["usageMetadata"]["thoughtsTokenCount"] = serde_json::json!(r);
            }
        }

        if !ir.model.is_empty() {
            resp["modelVersion"] = serde_json::Value::String(ir.model.to_string());
        }

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        super::attach_preserved(&mut resp, preserved);

        serde_json::to_vec(&resp).map_err(CodecError::from)
    }
}

// ─── EncodeStream ───────────────────────────────────────────────────

/// Gemini 流式编码器 — 每条流一个实例。
///
/// Gemini 有线格式期望 finishReason 与 usageMetadata 在最终 chunk 合并；
/// tool call 需要完整的 functionCall（不支持增量 args）。编码器缓存
/// ChoiceFinish 等待 Usage 合并，为无 ToolCallDone 的上游累积 arguments
/// 并在 ChoiceFinish/Done 时补发完整 functionCall。
pub struct GemStreamEncoder {
    /// 各 candidate 缓存的 finishReason，等待与 Usage 合并。
    pending_finishes: Vec<(u32, &'static str)>,
    /// 累积中的 tool_calls: (工具序号, candidate 序号, name, args, 已发出)
    tools: Vec<(u32, u32, String, String, bool)>,
    preserved_events: Vec<serde_json::Value>,
    finished: bool,
}

impl GemStreamEncoder {
    pub fn new() -> Self {
        Self {
            pending_finishes: Vec::new(),
            tools: Vec::new(),
            preserved_events: Vec::new(),
            finished: false,
        }
    }

    fn parse_tool_arguments(arguments: &str) -> Result<serde_json::Value, CodecError> {
        let value: serde_json::Value =
            serde_json::from_str(arguments).map_err(|error| CodecError::InvalidInput {
                context: "stream.tool_call.arguments",
                message: error.to_string(),
            })?;
        if !value.is_object() {
            return Err(CodecError::InvalidInput {
                context: "stream.tool_call.arguments",
                message: "Gemini functionCall args must be a JSON object".to_string(),
            });
        }
        Ok(value)
    }

    /// 将指定 candidate 中尚未发出的 tool_calls 编为 functionCall parts。
    fn flush_tools_for_choice(
        &mut self,
        choice_index: u32,
        out: &mut Vec<serde_json::Value>,
    ) -> Result<(), CodecError> {
        for (_, _, name, args, emitted) in self
            .tools
            .iter_mut()
            .filter(|tool| tool.1 == choice_index && !tool.4)
        {
            let args_val = Self::parse_tool_arguments(args)?;
            *emitted = true;
            out.push(serde_json::json!({
                "functionCall": { "name": name, "args": args_val }
            }));
        }
        Ok(())
    }

    fn drain_pending_candidates(&mut self) -> Result<Vec<serde_json::Value>, CodecError> {
        let mut grouped: Vec<(u32, Vec<serde_json::Value>, Option<&'static str>)> = Vec::new();
        for (_, choice_index, name, args, emitted) in self.tools.iter_mut().filter(|tool| !tool.4) {
            let args_val = Self::parse_tool_arguments(args)?;
            *emitted = true;
            let pos = grouped
                .iter()
                .position(|candidate| candidate.0 == *choice_index)
                .unwrap_or_else(|| {
                    grouped.push((*choice_index, Vec::new(), None));
                    grouped.len() - 1
                });
            grouped[pos].1.push(serde_json::json!({
                "functionCall": { "name": name, "args": args_val }
            }));
        }
        for (index, reason) in self.pending_finishes.drain(..) {
            let pos = grouped
                .iter()
                .position(|candidate| candidate.0 == index)
                .unwrap_or_else(|| {
                    grouped.push((index, Vec::new(), None));
                    grouped.len() - 1
                });
            grouped[pos].2 = Some(reason);
        }

        Ok(grouped
            .into_iter()
            .map(|(index, parts, reason)| {
                let mut candidate = serde_json::json!({ "index": index });
                if !parts.is_empty() {
                    candidate["content"] = serde_json::json!({
                        "role": "model",
                        "parts": parts,
                    });
                }
                if let Some(reason) = reason {
                    candidate["finishReason"] = serde_json::json!(reason);
                }
                candidate
            })
            .collect())
    }
}

impl Default for GemStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeStream for GemStreamEncoder {
    fn encode_sse_event(&mut self, event: &IrStreamEvent<'_>) -> Result<Vec<u8>, CodecError> {
        if self.finished {
            return if matches!(event, IrStreamEvent::Done) {
                Ok(Vec::new())
            } else {
                Err(CodecError::InvalidState(
                    "Gemini stream received an event after completion".to_string(),
                ))
            };
        }
        // Gemini 流式响应是 JSON 数组中的独立 generateContent 对象
        match event {
            IrStreamEvent::ContentDelta { index, delta } => {
                let chunk = serde_json::json!({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{ "text": delta }],
                        },
                        "index": index,
                    }],
                });
                serde_json::to_vec(&chunk).map_err(CodecError::from)
            }
            IrStreamEvent::ReasoningDelta { index, delta } => {
                let chunk = serde_json::json!({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{ "text": delta, "thought": true }],
                        },
                        "index": index,
                    }],
                });
                serde_json::to_vec(&chunk).map_err(CodecError::from)
            }
            IrStreamEvent::ReasoningDone { index, signature } => {
                // 签名以独立 thought part 发出（thoughtSignature 挂在 part 上）
                if let Some(sig) = signature {
                    let chunk = serde_json::json!({
                        "candidates": [{
                            "content": {
                                "role": "model",
                                "parts": [{ "text": "", "thought": true, "thoughtSignature": sig }],
                            },
                            "index": index,
                        }],
                    });
                    return serde_json::to_vec(&chunk).map_err(CodecError::from);
                }
                Ok(Vec::new())
            }
            IrStreamEvent::ToolCallStart {
                index,
                choice_index,
                id: _,
                name,
            } => {
                // Gemini 不支持增量 functionCall — 开始累积，Done/Finish 时发出
                self.tools.push((
                    *index,
                    *choice_index,
                    name.to_string(),
                    String::new(),
                    false,
                ));
                Ok(Vec::new())
            }
            IrStreamEvent::ToolCallDelta {
                index,
                arguments_delta,
                ..
            } => {
                if let Some(t) = self
                    .tools
                    .iter_mut()
                    .rev()
                    .find(|tool| tool.0 == *index && !tool.4)
                {
                    t.3.push_str(arguments_delta);
                }
                Ok(Vec::new())
            }
            IrStreamEvent::ToolCallDone {
                index,
                choice_index,
                name,
                arguments,
                ..
            } => {
                // 标记累积项为已发出（若存在），用 Done 的权威数据编码
                if let Some(t) = self
                    .tools
                    .iter_mut()
                    .rev()
                    .find(|tool| tool.0 == *index && !tool.4)
                {
                    t.4 = true;
                }
                let args = Self::parse_tool_arguments(arguments)?;
                let chunk = serde_json::json!({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": name,
                                    "args": args,
                                }
                            }],
                        },
                        "index": choice_index,
                    }],
                });
                serde_json::to_vec(&chunk).map_err(CodecError::from)
            }
            IrStreamEvent::ChoiceFinish {
                index,
                finish_reason,
            } => {
                // 未发出的累积 tool_calls 先补发，finishReason 缓存等待 Usage 合并
                let mut parts = Vec::new();
                self.flush_tools_for_choice(*index, &mut parts)?;
                let reason = GeminiShim::finish_reason_to_str(*finish_reason);
                if let Some(pending) = self
                    .pending_finishes
                    .iter_mut()
                    .find(|pending| pending.0 == *index)
                {
                    pending.1 = reason;
                } else {
                    self.pending_finishes.push((*index, reason));
                }
                if parts.is_empty() {
                    Ok(Vec::new())
                } else {
                    let chunk = serde_json::json!({
                        "candidates": [{
                            "content": { "role": "model", "parts": parts },
                            "index": index,
                        }],
                    });
                    serde_json::to_vec(&chunk).map_err(CodecError::from)
                }
            }
            IrStreamEvent::Usage(usage) => {
                let mut chunk = serde_json::json!({
                    "usageMetadata": {
                        "promptTokenCount": usage.prompt_tokens,
                        "candidatesTokenCount": usage.completion_tokens,
                        "totalTokenCount": usage.total_tokens,
                    }
                });
                if let Some(r) = usage.reasoning_tokens {
                    chunk["usageMetadata"]["thoughtsTokenCount"] = serde_json::json!(r);
                }
                if let Some(cr) = usage.cache_read_tokens {
                    chunk["usageMetadata"]["cachedContentTokenCount"] = serde_json::json!(cr);
                }
                // finishReason 与 usageMetadata 合并到最终 chunk
                if !self.pending_finishes.is_empty() {
                    let candidates = self
                        .pending_finishes
                        .drain(..)
                        .map(|(index, reason)| {
                            serde_json::json!({
                                "index": index,
                                "finishReason": reason,
                            })
                        })
                        .collect::<Vec<_>>();
                    chunk["candidates"] = serde_json::Value::Array(candidates);
                }
                if !self.preserved_events.is_empty() {
                    let preserved = std::mem::take(&mut self.preserved_events);
                    super::attach_preserved(&mut chunk, preserved);
                }
                serde_json::to_vec(&chunk).map_err(CodecError::from)
            }
            IrStreamEvent::Done => {
                self.finished = true;
                let candidates = self.drain_pending_candidates()?;
                let mut chunk = if candidates.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::json!({ "candidates": candidates })
                };
                if !self.preserved_events.is_empty() {
                    let preserved = std::mem::take(&mut self.preserved_events);
                    super::attach_preserved(&mut chunk, preserved);
                }
                if chunk.as_object().is_some_and(serde_json::Map::is_empty) {
                    Ok(Vec::new())
                } else {
                    serde_json::to_vec(&chunk).map_err(CodecError::from)
                }
            }
            // 不需要在 Gemini 格式中产生输出的事件
            IrStreamEvent::Start { .. } | IrStreamEvent::ContentDone { .. } => Ok(Vec::new()),
            // Gemini 流式无 redacted reasoning / per-token logprobs 对应 —
            // 序列化整事件缓冲到 preserved_events，于 Usage/Done 时随保留通道发出
            IrStreamEvent::RedactedReasoning { .. }
            | IrStreamEvent::Logprobs { .. }
            | IrStreamEvent::Citation { .. }
            | IrStreamEvent::OpaqueBlock { .. } => {
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
                Ok(Vec::new())
            }
            IrStreamEvent::Error { message } => {
                self.finished = true;
                let chunk = serde_json::json!({
                    "error": { "message": message },
                });
                serde_json::to_vec(&chunk).map_err(CodecError::from)
            }
        }
    }
}
