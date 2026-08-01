//! OpenAI shim — 覆盖 OpenAI / DeepSeek / 所有 OpenAI 兼容 API
//!
//! 零拷贝策略：
//! - encode: 直接用 serde_json::to_vec 序列化，跳过中间 String
//! - decode: serde_json::from_slice 直接从 &[u8] 反序列化，Cow::Borrowed 引用原始字节

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::{
    DecodeRequest, DecodeResponse, DecodeStream, EncodeRequest, EncodeResponse, EncodeStream,
};
use crate::codec::error::CodecError;
use crate::codec::ir::*;

// ─── OpenAI 有线格式（序列化：IR → OpenAI 请求）──────────────────

#[derive(Serialize)]
struct OaiRequest<'a> {
    model: &'a str,
    messages: Vec<OaiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [Cow<'a, str>]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OaiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<&'a [Cow<'a, str>]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct OaiStreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OaiMessage<'a> {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiToolCallOut<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<&'a str>,
}

#[derive(Serialize)]
struct OaiToolCallOut<'a> {
    id: &'a str,
    r#type: &'static str,
    function: OaiFunctionOut<'a>,
}

#[derive(Serialize)]
struct OaiFunctionOut<'a> {
    name: &'a str,
    arguments: &'a str,
}

// ─── 反序列化（OpenAI 响应 → IR）────────────────────────────────

#[derive(Deserialize)]
struct OaiResponse<'a> {
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow, default)]
    model: Cow<'a, str>,
    #[serde(default)]
    choices: Vec<OaiChoiceIn<'a>>,
    usage: Option<OaiUsageIn>,
    #[serde(default)]
    created: Option<i64>,
    #[serde(borrow, default)]
    system_fingerprint: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OaiChoiceIn<'a> {
    index: u32,
    message: Option<OaiMsgIn<'a>>,
    delta: Option<OaiMsgIn<'a>>,
    #[serde(borrow)]
    finish_reason: Option<Cow<'a, str>>,
    #[serde(default)]
    logprobs: Option<OaiLogprobsIn>,
}

#[derive(Deserialize)]
struct OaiLogprobsIn {
    #[serde(default)]
    content: Option<Vec<OaiTokenLogprobIn>>,
    #[serde(default)]
    refusal: Option<Vec<OaiTokenLogprobIn>>,
}

#[derive(Deserialize)]
struct OaiTokenLogprobIn {
    token: String,
    logprob: f64,
    #[serde(default)]
    bytes: Option<Vec<u8>>,
    #[serde(default)]
    top_logprobs: Option<Vec<OaiTopLogprobIn>>,
}

#[derive(Deserialize)]
struct OaiTopLogprobIn {
    token: String,
    logprob: f64,
    #[serde(default)]
    bytes: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct OaiMsgIn<'a> {
    #[serde(borrow, default)]
    role: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    content: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    reasoning_content: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    reasoning_signature: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    refusal: Option<Cow<'a, str>>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallIn<'a>>>,
    #[serde(default)]
    audio: Option<OaiAudioOut<'a>>,
    /// web search URL 引文（OpenAI Chat Completions `message.annotations`）
    #[serde(default)]
    annotations: Option<Vec<serde_json::Value>>,
}

/// 音频响应（OpenAI audio model output）
#[derive(Deserialize)]
struct OaiAudioOut<'a> {
    #[serde(borrow, default)]
    #[allow(dead_code)]
    id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    data: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    transcript: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct OaiToolCallIn<'a> {
    #[serde(default)]
    index: Option<u32>,
    #[serde(borrow, default)]
    id: Option<Cow<'a, str>>,
    function: Option<OaiFunctionIn<'a>>,
}

#[derive(Deserialize)]
struct OaiFunctionIn<'a> {
    #[serde(borrow, default)]
    name: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    arguments: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct OaiUsageIn {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens_details: Option<OaiCompletionDetails>,
    #[serde(default)]
    prompt_tokens_details: Option<OaiPromptDetails>,
}

#[derive(Deserialize)]
struct OaiCompletionDetails {
    reasoning_tokens: Option<u32>,
    audio_tokens: Option<u32>,
    accepted_prediction_tokens: Option<u32>,
    rejected_prediction_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct OaiPromptDetails {
    cached_tokens: Option<u32>,
}

// ─── 实现 ────────────────────────────────────────────────────────

pub struct OpenAiShim;

impl OpenAiShim {
    fn ir_role_to_str(role: Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    fn ir_content_to_json(
        content: &IrContent<'_>,
        preserved: &mut Vec<serde_json::Value>,
    ) -> Option<serde_json::Value> {
        match content {
            IrContent::Text(s) => Some(serde_json::Value::String(s.to_string())),
            IrContent::Parts(parts) => {
                let arr: Vec<serde_json::Value> = parts
                    .iter()
                    .filter_map(|p| match p {
                        IrContentPart::Text { text, .. } => Some(serde_json::json!({
                            "type": "text",
                            "text": text,
                        })),
                        IrContentPart::ImageUrl { url, detail, .. } => {
                            let mut img = serde_json::json!({ "url": url });
                            if let Some(d) = detail {
                                img["detail"] = serde_json::Value::String(d.to_string());
                            }
                            Some(serde_json::json!({
                                "type": "image_url",
                                "image_url": img,
                            }))
                        }
                        IrContentPart::ImageBase64 {
                            media_type, data, ..
                        } => {
                            let data_url = format!("data:{media_type};base64,{data}");
                            Some(serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": data_url },
                            }))
                        }
                        IrContentPart::Audio {
                            media_type, data, ..
                        } => {
                            // OpenAI input_audio: 从 media_type 提取格式（如 "audio/wav" → "wav"）
                            let format = media_type
                                .strip_prefix("audio/")
                                .unwrap_or(media_type.as_ref());
                            Some(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": data,
                                    "format": format,
                                },
                            }))
                        }
                        IrContentPart::Document {
                            media_type,
                            data,
                            filename,
                        } => {
                            let mut file = serde_json::json!({
                                "file_data": format!("data:{media_type};base64,{data}"),
                            });
                            if let Some(f) = filename {
                                file["filename"] = serde_json::Value::String(f.to_string());
                            }
                            Some(serde_json::json!({ "type": "file", "file": file }))
                        }
                        IrContentPart::FileRef { file_id } => Some(serde_json::json!({
                            "type": "file",
                            "file": { "file_id": file_id },
                        })),
                        IrContentPart::FunctionCall { .. } => None,
                        IrContentPart::FunctionResponse { .. } => {
                            if let Ok(v) = serde_json::to_value(p) {
                                preserved.push(v);
                            }
                            None
                        }
                        // OpenAI 不支持的内容类型 → 保留到 preserved
                        IrContentPart::Video { .. }
                        | IrContentPart::Reasoning { .. }
                        | IrContentPart::RedactedReasoning { .. }
                        | IrContentPart::Opaque { .. } => {
                            if let Ok(v) = serde_json::to_value(p) {
                                preserved.push(v);
                            }
                            None
                        }
                    })
                    .collect();
                if arr.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Array(arr))
                }
            }
        }
    }

    fn ir_tool_choice_to_json(tc: &IrToolChoice<'_>) -> serde_json::Value {
        match tc {
            IrToolChoice::Auto => serde_json::json!("auto"),
            IrToolChoice::None => serde_json::json!("none"),
            IrToolChoice::Required => serde_json::json!("required"),
            IrToolChoice::Specific { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }

    fn ir_response_format_to_json(rf: &IrResponseFormat) -> serde_json::Value {
        match rf.r#type {
            ResponseFormatType::Text => serde_json::json!({ "type": "text" }),
            ResponseFormatType::JsonObject => serde_json::json!({ "type": "json_object" }),
            ResponseFormatType::JsonSchema => {
                let mut json_schema = serde_json::Map::new();
                if let Some(ref name) = rf.name {
                    json_schema.insert("name".into(), serde_json::Value::String(name.clone()));
                }
                if let Some(ref schema) = rf.schema {
                    json_schema.insert("schema".into(), schema.clone());
                }
                if let Some(strict) = rf.strict {
                    json_schema.insert("strict".into(), serde_json::Value::Bool(strict));
                }
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": json_schema,
                })
            }
        }
    }

    fn convert_logprobs(lp: &OaiLogprobsIn) -> IrLogprobs {
        fn convert_tokens(tokens: &[OaiTokenLogprobIn]) -> Vec<IrTokenLogprob> {
            tokens
                .iter()
                .map(|t| IrTokenLogprob {
                    token: t.token.clone(),
                    logprob: t.logprob,
                    bytes: t.bytes.clone(),
                    top_logprobs: t.top_logprobs.as_ref().map(|tl| {
                        tl.iter()
                            .map(|tp| IrTopLogprob {
                                token: tp.token.clone(),
                                logprob: tp.logprob,
                                bytes: tp.bytes.clone(),
                            })
                            .collect()
                    }),
                })
                .collect()
        }
        IrLogprobs {
            content: lp.content.as_ref().map(|c| convert_tokens(c)),
            refusal: lp.refusal.as_ref().map(|r| convert_tokens(r)),
            avg_logprob: None,
        }
    }

    fn ir_tool_type_to_str(t: &IrToolType) -> &'static str {
        match t {
            IrToolType::Function => "function",
            IrToolType::WebSearch => "web_search",
            IrToolType::CodeInterpreter => "code_interpreter",
            IrToolType::FileSearch => "file_search",
            IrToolType::ComputerUse => "computer_use",
            IrToolType::TextEditor => "text_editor",
            IrToolType::Mcp => "mcp",
        }
    }

    /// OpenAI `message.annotations`（web search URL 引文）→ IrCitation
    fn annotations_to_citations<'a>(anns: &[serde_json::Value]) -> Option<Vec<IrCitation<'a>>> {
        let cits: Vec<IrCitation<'a>> = anns
            .iter()
            .filter_map(|a| {
                let type_str = a
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("url_citation");
                let uc = a.get("url_citation");
                let url = uc
                    .and_then(|u| u.get("url"))
                    .or_else(|| a.get("url"))
                    .and_then(|v| v.as_str())
                    .map(|s| Cow::Owned(s.to_string()));
                let title = uc
                    .and_then(|u| u.get("title"))
                    .or_else(|| a.get("title"))
                    .and_then(|v| v.as_str())
                    .map(|s| Cow::Owned(s.to_string()));
                let cited_text = uc
                    .and_then(|u| u.get("content"))
                    .and_then(|v| v.as_str())
                    .map(|s| Cow::Owned(s.to_string()));
                if url.is_none() && title.is_none() {
                    return None;
                }
                Some(IrCitation {
                    r#type: Cow::Owned(type_str.to_string()),
                    url,
                    title,
                    cited_text,
                    encrypted_index: None,
                })
            })
            .collect();
        if cits.is_empty() {
            None
        } else {
            Some(cits)
        }
    }

    fn parse_finish_reason(s: &str) -> IrFinishReason {
        match s {
            "stop" => IrFinishReason::Stop,
            "length" => IrFinishReason::Length,
            "tool_calls" => IrFinishReason::ToolCalls,
            "content_filter" => IrFinishReason::ContentFilter,
            _ => IrFinishReason::Stop,
        }
    }

    fn oai_msg_to_ir<'a>(msg: &OaiMsgIn<'a>, role_hint: Role) -> IrMessage<'a> {
        let role = msg
            .role
            .as_deref()
            .map(|r| match r {
                "system" => Role::System,
                "developer" => Role::Developer,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => role_hint,
            })
            .unwrap_or(role_hint);

        // 构建 content parts
        let mut parts: Vec<IrContentPart<'a>> = Vec::new();

        if let Some(ref reasoning) = msg.reasoning_content {
            if !reasoning.is_empty() {
                parts.push(IrContentPart::Reasoning {
                    text: reasoning.clone(),
                    signature: msg.reasoning_signature.clone(),
                });
            }
        }

        // 音频响应
        if let Some(ref audio) = msg.audio {
            if let Some(ref data) = audio.data {
                parts.push(IrContentPart::Audio {
                    media_type: Cow::Borrowed("audio/wav"),
                    data: data.clone(),
                });
            }
            if let Some(ref transcript) = audio.transcript {
                if !transcript.is_empty() {
                    parts.push(IrContentPart::Text {
                        text: transcript.clone(),
                        citations: None,
                    });
                }
            }
        }

        // web search 引文 → 附着到文本内容
        let citations = msg
            .annotations
            .as_ref()
            .and_then(|anns| Self::annotations_to_citations(anns));

        let content = if let Some(ref text) = msg.content {
            if parts.is_empty() && citations.is_none() {
                IrContent::Text(text.clone())
            } else {
                parts.push(IrContentPart::Text {
                    text: text.clone(),
                    citations,
                });
                IrContent::Parts(parts)
            }
        } else if citations.is_some() {
            parts.push(IrContentPart::Text {
                text: Cow::Borrowed(""),
                citations,
            });
            IrContent::Parts(parts)
        } else if !parts.is_empty() {
            IrContent::Parts(parts)
        } else {
            IrContent::Text(Cow::Borrowed(""))
        };

        let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
            tcs.iter()
                .filter_map(|tc| {
                    let func = tc.function.as_ref()?;
                    Some(IrToolCall {
                        id: tc.id.clone().unwrap_or(Cow::Borrowed("")),
                        name: func.name.clone().unwrap_or(Cow::Borrowed("")),
                        arguments: func.arguments.clone().unwrap_or(Cow::Borrowed("")),
                    })
                })
                .collect()
        });

        IrMessage {
            role,
            content,
            tool_call_id: None,
            tool_name: None,
            tool_calls,
            cache_control: None, // OpenAI 无 cache_control
            refusal: msg.refusal.clone(),
        }
    }

    fn oai_usage_to_ir(u: &OaiUsageIn) -> IrUsage {
        IrUsage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            cache_read_tokens: u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
            cache_creation_tokens: None,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            audio_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.audio_tokens),
            accepted_prediction_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.accepted_prediction_tokens),
            rejected_prediction_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.rejected_prediction_tokens),
        }
    }

    fn make_provider_metadata(
        system_fingerprint: &Option<Cow<'_, str>>,
    ) -> Option<Box<serde_json::Value>> {
        system_fingerprint.as_ref().map(|fp| {
            Box::new(serde_json::json!({
                "system_fingerprint": fp,
            }))
        })
    }
}

// ─── EncodeRequest ───────────────────────────────────────────────

impl EncodeRequest for OpenAiShim {
    fn encode_request(&self, ir: &IrRequest<'_>) -> Result<Vec<u8>, CodecError> {
        let mut preserved = Vec::new();
        let messages: Vec<OaiMessage<'_>> = ir
            .messages
            .iter()
            .map(|m| {
                let content = Self::ir_content_to_json(&m.content, &mut preserved);
                let tool_calls = {
                    let mut from_content: Vec<OaiToolCallOut<'_>> = Vec::new();
                    if let IrContent::Parts(ref parts) = m.content {
                        for p in parts {
                            if let IrContentPart::FunctionCall {
                                id,
                                name,
                                arguments,
                            } = p
                            {
                                from_content.push(OaiToolCallOut {
                                    id,
                                    r#type: "function",
                                    function: OaiFunctionOut { name, arguments },
                                });
                            }
                        }
                    }
                    if !from_content.is_empty() {
                        Some(from_content)
                    } else {
                        m.tool_calls.as_ref().map(|tcs| {
                            tcs.iter()
                                .map(|tc| OaiToolCallOut {
                                    id: &tc.id,
                                    r#type: "function",
                                    function: OaiFunctionOut {
                                        name: &tc.name,
                                        arguments: &tc.arguments,
                                    },
                                })
                                .collect()
                        })
                    }
                };
                OaiMessage {
                    role: Self::ir_role_to_str(m.role),
                    content,
                    tool_call_id: m.tool_call_id.as_deref(),
                    tool_calls,
                    refusal: m.refusal.as_deref(),
                }
            })
            .collect();

        // function 工具走标准 { type: function, function: {...} }；
        // 非 function（web_search 等）用 extra 承载供应商原始表示、回填 type
        let tools: Option<Vec<serde_json::Value>> = ir.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| match t.tool_type {
                    IrToolType::Function => {
                        let mut func = serde_json::Map::new();
                        func.insert("name".into(), serde_json::Value::String(t.name.to_string()));
                        if let Some(ref d) = t.description {
                            func.insert(
                                "description".into(),
                                serde_json::Value::String(d.to_string()),
                            );
                        }
                        func.insert("parameters".into(), t.parameters.clone());
                        if let Some(serde_json::Value::Object(extra)) = t.extra.as_ref() {
                            for (k, v) in extra {
                                func.insert(k.clone(), v.clone());
                            }
                        }
                        serde_json::json!({ "type": "function", "function": func })
                    }
                    _ => {
                        let mut v = match t.extra {
                            Some(ref e) if e.is_object() => e.clone(),
                            _ => serde_json::json!({}),
                        };
                        v["type"] = serde_json::Value::String(
                            Self::ir_tool_type_to_str(&t.tool_type).to_string(),
                        );
                        v
                    }
                })
                .collect()
        });

        let tool_choice = ir.tool_choice.as_ref().map(Self::ir_tool_choice_to_json);

        let response_format = ir
            .response_format
            .as_ref()
            .map(Self::ir_response_format_to_json);

        // metadata.user_id → "user" 顶层字段
        let user = ir.metadata.as_ref().and_then(|m| m.user_id.as_deref());

        // reasoning → reasoning_effort（优先原始 effort 字符串，往返不漂移）
        let reasoning_effort = ir.reasoning.as_ref().map(|r| {
            r.effort.as_deref().unwrap_or(match r.mode {
                ReasoningMode::Disabled => "none",
                ReasoningMode::Auto => "medium",
                ReasoningMode::Enabled => match r.budget_tokens {
                    Some(b) if b < 2000 => "low",
                    _ => "high",
                },
            })
        });

        // service_tier
        let service_tier = ir.metadata.as_ref().and_then(|m| m.service_tier.as_deref());

        // prediction / audio 配置从 provider_metadata 透传（Strip 模式下跳过）
        let (prediction, audio) = if ir.metadata_mode == MetadataMode::Strip {
            (None, None)
        } else {
            let prediction = ir
                .provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("prediction").cloned());
            let audio = ir
                .provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("audio").cloned());
            (prediction, audio)
        };

        let req = OaiRequest {
            model: &ir.model,
            messages,
            temperature: ir.temperature,
            top_p: ir.top_p,
            max_completion_tokens: ir.max_tokens,
            stop: ir.stop.as_deref(),
            frequency_penalty: ir.frequency_penalty,
            presence_penalty: ir.presence_penalty,
            seed: ir.seed,
            n: ir.n,
            logprobs: ir.logprobs,
            top_logprobs: ir.top_logprobs,
            stream: ir.stream,
            tools,
            tool_choice,
            parallel_tool_calls: ir.parallel_tool_calls,
            response_format,
            stream_options: if ir.stream {
                Some(OaiStreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            user,
            reasoning_effort,
            service_tier,
            store: ir.store,
            modalities: ir.modalities.as_deref(),
            prediction,
            audio,
        };

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        if preserved.is_empty() {
            serde_json::to_vec(&req).map_err(CodecError::from)
        } else {
            let mut req_json = serde_json::to_value(&req).map_err(CodecError::from)?;
            super::attach_preserved(&mut req_json, preserved);
            serde_json::to_vec(&req_json).map_err(CodecError::from)
        }
    }

    fn endpoint(&self, base_url: &str) -> String {
        format!("{base_url}/v1/chat/completions")
    }

    fn headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![
            ("Authorization", format!("Bearer {api_key}")),
            ("Content-Type", "application/json".into()),
        ]
    }
}

// ─── DecodeResponse ──────────────────────────────────────────────

impl DecodeResponse for OpenAiShim {
    fn decode_response<'a>(&self, body: &'a [u8]) -> Result<IrResponse<'a>, CodecError> {
        let oai: OaiResponse<'a> = serde_json::from_slice(body)?;

        let choices = oai
            .choices
            .into_iter()
            .map(|c| {
                let msg = c
                    .message
                    .map(|m| Self::oai_msg_to_ir(&m, Role::Assistant))
                    .unwrap_or(IrMessage {
                        role: Role::Assistant,
                        content: IrContent::Text(Cow::Borrowed("")),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls: None,
                        cache_control: None,
                        refusal: None,
                    });
                IrChoice {
                    index: c.index,
                    message: msg,
                    finish_reason: c.finish_reason.as_deref().map(Self::parse_finish_reason),
                    logprobs: c.logprobs.as_ref().map(Self::convert_logprobs),
                }
            })
            .collect();

        let usage = oai.usage.as_ref().map(Self::oai_usage_to_ir);
        let mut provider_metadata = Self::make_provider_metadata(&oai.system_fingerprint);

        // Merge created, service_tier, error, metadata into provider_metadata
        if let Some(created) = oai.created {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            pm["created"] = serde_json::json!(created);
        }
        if let Some(ref st) = oai.service_tier {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            pm["service_tier"] = serde_json::Value::String(st.to_string());
        }
        if let Some(ref err) = oai.error {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            pm["error"] = err.clone();
        }
        if let Some(ref meta) = oai.metadata {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            pm["metadata"] = meta.clone();
        }

        // 从原始 JSON 提取 preserved parts
        let resp_json: serde_json::Value = serde_json::from_slice(body)?;
        let preserved = super::extract_preserved(&resp_json);
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        Ok(IrResponse {
            id: oai.id,
            model: oai.model,
            choices,
            usage,
            provider_metadata,
        })
    }
}

// ─── DecodeStream ────────────────────────────────────────────────
// OpenAI Chat Completions 流式协议不提供块级完成信号（tool call 的
// id/name/arguments 分散在多个 delta 中）。解码器有状态地跟踪块生命
// 周期，合成 Start / ReasoningDone / ContentDone / ToolCallDone。

/// OpenAI 流式解码器 — 每条 SSE 流一个实例
pub struct OaiStreamDecoder {
    started: bool,
    finished: bool,
    /// choice index → (reasoning 块打开, content 块打开, 累积的 reasoning 签名)
    open_blocks: Vec<(u32, bool, bool, Option<String>)>,
    /// 活跃 tool_calls: (choice 索引, 线上 tc 索引, 全局工具序号, id, name, args 累积)。
    /// n>1 时不同 choice 的线上 tc 索引会重叠，必须按 (choice, tc_index) 定位；
    /// IR 工具事件的 index 输出全局序号（见 ir.rs 规范）
    active_tools: Vec<(u32, u32, u32, String, String, String)>,
    /// 全局工具序号计数
    tool_seq: u32,
    finished_choices: Vec<u32>,
}

impl OaiStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            finished: false,
            open_blocks: Vec::new(),
            active_tools: Vec::new(),
            tool_seq: 0,
            finished_choices: Vec::new(),
        }
    }

    fn entry_pos(&mut self, index: u32) -> usize {
        match self.open_blocks.iter().position(|b| b.0 == index) {
            Some(pos) => pos,
            None => {
                self.open_blocks.push((index, false, false, None));
                self.open_blocks.len() - 1
            }
        }
    }

    fn finalize(&mut self) -> Vec<IrStreamEvent<'static>> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();
        for (index, reasoning_open, content_open, signature) in &mut self.open_blocks {
            if *reasoning_open {
                *reasoning_open = false;
                events.push(IrStreamEvent::ReasoningDone {
                    index: *index,
                    signature: signature.take().map(Cow::Owned),
                });
            }
            if *content_open {
                *content_open = false;
                events.push(IrStreamEvent::ContentDone { index: *index });
            }
        }
        for (choice_index, _, seq, id, name, arguments) in self.active_tools.drain(..) {
            events.push(IrStreamEvent::ToolCallDone {
                index: seq,
                choice_index,
                id: Cow::Owned(id),
                name: Cow::Owned(name),
                arguments: Cow::Owned(arguments),
            });
        }
        for (index, _, _, _) in &self.open_blocks {
            if !self.finished_choices.contains(index) {
                self.finished_choices.push(*index);
                events.push(IrStreamEvent::ChoiceFinish {
                    index: *index,
                    finish_reason: IrFinishReason::Stop,
                });
            }
        }
        events.push(IrStreamEvent::Done);
        events
    }
}

impl Default for OaiStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeStream for OaiStreamDecoder {
    fn decode_sse_data<'a>(
        &mut self,
        data: &'a [u8],
    ) -> Result<Vec<IrStreamEvent<'a>>, CodecError> {
        if data == b"[DONE]" {
            return Ok(self.finalize());
        }
        if self.finished {
            return Err(CodecError::InvalidState(
                "OpenAI stream received data after completion".to_string(),
            ));
        }

        let chunk: OaiResponse<'a> = serde_json::from_slice(data)?;

        // 流式错误快路径：OpenAI 以 {"error":{"message":"..."}} 形式下发错误，
        // 此时 choices 为空、无 usage，若不拦截会静默丢弃。
        if let Some(ref err) = chunk.error {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            self.finished = true;
            return Ok(vec![IrStreamEvent::Error {
                message: Cow::Owned(msg.to_string()),
            }]);
        }

        let mut events = Vec::with_capacity(4);

        // 首个 chunk 携带 id/model → 合成 Start
        if !self.started {
            self.started = true;
            events.push(IrStreamEvent::Start {
                id: chunk.id.clone(),
                model: chunk.model.clone(),
                usage: None,
            });
        }

        for choice in &chunk.choices {
            self.entry_pos(choice.index);
            if let Some(ref delta) = choice.delta {
                // reasoning delta
                if let Some(ref r) = delta.reasoning_content {
                    if !r.is_empty() {
                        let pos = self.entry_pos(choice.index);
                        self.open_blocks[pos].1 = true;
                        events.push(IrStreamEvent::ReasoningDelta {
                            index: choice.index,
                            delta: r.clone(),
                        });
                    }
                }

                // reasoning 签名 — 在块生命周期内某个 chunk 到达，缓存至 ReasoningDone 发出
                if let Some(ref sig) = delta.reasoning_signature {
                    if !sig.is_empty() {
                        let pos = self.entry_pos(choice.index);
                        self.open_blocks[pos].3 = Some(sig.to_string());
                    }
                }

                // content delta — reasoning → content 切换时合成 ReasoningDone
                if let Some(ref c) = delta.content {
                    if !c.is_empty() {
                        let pos = self.entry_pos(choice.index);
                        if self.open_blocks[pos].1 {
                            self.open_blocks[pos].1 = false;
                            let signature = self.open_blocks[pos].3.take().map(Cow::Owned);
                            events.push(IrStreamEvent::ReasoningDone {
                                index: choice.index,
                                signature,
                            });
                        }
                        self.open_blocks[pos].2 = true;
                        events.push(IrStreamEvent::ContentDelta {
                            index: choice.index,
                            delta: c.clone(),
                        });
                    }
                }

                // refusal delta — OpenAI 安全拒答替代 content，路由到 ContentDelta
                if let Some(ref refusal) = delta.refusal {
                    if !refusal.is_empty() {
                        let pos = self.entry_pos(choice.index);
                        self.open_blocks[pos].2 = true;
                        events.push(IrStreamEvent::ContentDelta {
                            index: choice.index,
                            delta: refusal.clone(),
                        });
                    }
                }

                // audio delta — transcript 路由到 ContentDelta
                if let Some(ref audio) = delta.audio {
                    if let Some(ref transcript) = audio.transcript {
                        if !transcript.is_empty() {
                            let pos = self.entry_pos(choice.index);
                            self.open_blocks[pos].2 = true;
                            events.push(IrStreamEvent::ContentDelta {
                                index: choice.index,
                                delta: transcript.clone(),
                            });
                        }
                    }
                }

                // 流式 annotations → Citation 事件
                if let Some(ref anns) = delta.annotations {
                    for ann in anns {
                        events.push(IrStreamEvent::Citation {
                            index: choice.index,
                            citation: ann.clone(),
                        });
                    }
                }

                // tool call delta — 按 id 有无区分 Start / Delta；
                // 定位键为 (choice 索引, 线上 tc 索引)，事件输出全局工具序号
                if let Some(ref tcs) = delta.tool_calls {
                    for tc in tcs {
                        let tc_index = tc.index.unwrap_or(0);
                        let choice_idx = choice.index;
                        let existing_seq = self
                            .active_tools
                            .iter()
                            .find(|c| c.0 == choice_idx && c.1 == tc_index)
                            .map(|c| c.2);
                        if tc.id.is_some() && existing_seq.is_none() {
                            let id = tc.id.as_ref().unwrap();
                            // 首个 chunk：携带 id + name → ToolCallStart
                            let seq = self.tool_seq;
                            self.tool_seq += 1;
                            let name = tc
                                .function
                                .as_ref()
                                .and_then(|f| f.name.clone())
                                .unwrap_or(Cow::Borrowed(""));
                            self.active_tools.push((
                                choice_idx,
                                tc_index,
                                seq,
                                id.to_string(),
                                name.to_string(),
                                String::new(),
                            ));
                            events.push(IrStreamEvent::ToolCallStart {
                                index: seq,
                                choice_index: choice_idx,
                                id: id.clone(),
                                name,
                            });
                            // 首个 chunk 若同时携带非空 arguments，追加一个 Delta
                            if let Some(ref func) = tc.function {
                                if let Some(ref args) = func.arguments {
                                    if !args.is_empty() {
                                        if let Some(call) = self
                                            .active_tools
                                            .iter_mut()
                                            .find(|c| c.0 == choice_idx && c.1 == tc_index)
                                        {
                                            call.5.push_str(args);
                                        }
                                        events.push(IrStreamEvent::ToolCallDelta {
                                            index: seq,
                                            choice_index: choice_idx,
                                            arguments_delta: args.clone(),
                                        });
                                    }
                                }
                            }
                        } else if let Some(ref func) = tc.function {
                            // 后续 chunk：无 id（或重复 id）→ ToolCallDelta
                            let seq = match existing_seq {
                                Some(s) => s,
                                None => continue,
                            };
                            let args = func.arguments.clone().unwrap_or(Cow::Borrowed(""));
                            if let Some(call) = self
                                .active_tools
                                .iter_mut()
                                .find(|c| c.0 == choice_idx && c.1 == tc_index)
                            {
                                call.5.push_str(&args);
                            }
                            if !args.is_empty() {
                                events.push(IrStreamEvent::ToolCallDelta {
                                    index: seq,
                                    choice_index: choice_idx,
                                    arguments_delta: args,
                                });
                            }
                        }
                    }
                }
            }

            // 流式 logprobs — OpenAI 在携带内容的同一 chunk 上回传 choice.logprobs
            if let Some(ref lp) = choice.logprobs {
                events.push(IrStreamEvent::Logprobs {
                    index: choice.index,
                    logprobs: OpenAiShim::convert_logprobs(lp),
                });
            }

            // finish_reason 到达：关闭打开的块 → 合成 ToolCallDone → ChoiceFinish
            if let Some(ref fr) = choice.finish_reason {
                if self.finished_choices.contains(&choice.index) {
                    continue;
                }
                let reason = OpenAiShim::parse_finish_reason(fr);
                let pos = self.entry_pos(choice.index);
                if self.open_blocks[pos].1 {
                    self.open_blocks[pos].1 = false;
                    let signature = self.open_blocks[pos].3.take().map(Cow::Owned);
                    events.push(IrStreamEvent::ReasoningDone {
                        index: choice.index,
                        signature,
                    });
                }
                if self.open_blocks[pos].2 {
                    self.open_blocks[pos].2 = false;
                    events.push(IrStreamEvent::ContentDone {
                        index: choice.index,
                    });
                }
                // 无论 finish_reason 为何，只要该 choice 有活跃 tool call 就补发
                // Done — 部分兼容供应商发了 tool_calls 却以 "stop" 结束；
                // 只 drain 本 choice 的工具（n>1 时其他 choice 可能还在流）
                let mut i = 0;
                while i < self.active_tools.len() {
                    if self.active_tools[i].0 == choice.index {
                        let (_, _, seq, id, name, args) = self.active_tools.remove(i);
                        events.push(IrStreamEvent::ToolCallDone {
                            index: seq,
                            choice_index: choice.index,
                            id: Cow::Owned(id),
                            name: Cow::Owned(name),
                            arguments: Cow::Owned(args),
                        });
                    } else {
                        i += 1;
                    }
                }
                events.push(IrStreamEvent::ChoiceFinish {
                    index: choice.index,
                    finish_reason: reason,
                });
                self.finished_choices.push(choice.index);
            }
        }

        // usage（stream_options.include_usage 产生的尾部 chunk）
        if let Some(ref usage) = chunk.usage {
            events.push(IrStreamEvent::Usage(OpenAiShim::oai_usage_to_ir(usage)));
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<IrStreamEvent<'static>>, CodecError> {
        Ok(self.finalize())
    }
}

// ─── DecodeRequest（请求体反序列化）────────────────────────────────

/// OpenAI Chat Completions 请求 — 反序列化用
#[derive(Deserialize)]
struct OaiRequestIn<'a> {
    #[serde(borrow)]
    model: Cow<'a, str>,
    #[serde(default)]
    messages: Vec<OaiMessageIn<'a>>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(borrow, default)]
    stop: Option<OaiStop<'a>>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<OaiToolIn<'a>>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    #[serde(borrow, default)]
    user: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    reasoning_effort: Option<Cow<'a, str>>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(borrow, default)]
    modalities: Option<Vec<Cow<'a, str>>>,
    #[serde(default)]
    prediction: Option<serde_json::Value>,
    #[serde(default)]
    audio: Option<serde_json::Value>,
}

/// stop 可以是字符串或字符串数组
#[derive(Deserialize)]
#[serde(untagged)]
enum OaiStop<'a> {
    #[serde(borrow)]
    Single(Cow<'a, str>),
    #[serde(borrow)]
    Multiple(Vec<Cow<'a, str>>),
}

#[derive(Deserialize)]
struct OaiMessageIn<'a> {
    #[serde(borrow, default)]
    role: Cow<'a, str>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(borrow, default)]
    tool_call_id: Option<Cow<'a, str>>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallIn<'a>>>,
    #[serde(borrow, default)]
    refusal: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct OaiToolIn<'a> {
    #[serde(borrow, default)]
    r#type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    function: Option<OaiToolFunctionIn<'a>>,
    #[serde(flatten)]
    extra_fields: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct OaiToolFunctionIn<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(borrow, default)]
    description: Option<Cow<'a, str>>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
    #[serde(default)]
    strict: Option<bool>,
}

impl OpenAiShim {
    fn parse_role(s: &str) -> Role {
        match s {
            "system" => Role::System,
            "developer" => Role::Developer,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        }
    }

    fn parse_tool_choice(v: &serde_json::Value) -> IrToolChoice<'static> {
        if let Some(s) = v.as_str() {
            match s {
                "auto" => IrToolChoice::Auto,
                "none" => IrToolChoice::None,
                "required" => IrToolChoice::Required,
                _ => IrToolChoice::Auto,
            }
        } else if let Some(name) = v
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        {
            IrToolChoice::Specific {
                name: Cow::Owned(name.to_string()),
            }
        } else {
            IrToolChoice::Auto
        }
    }

    fn parse_response_format(v: &serde_json::Value) -> IrResponseFormat {
        let type_str = v.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match type_str {
            "json_object" => IrResponseFormat {
                r#type: ResponseFormatType::JsonObject,
                schema: None,
                name: None,
                strict: None,
            },
            "json_schema" => {
                let json_schema = v.get("json_schema");
                IrResponseFormat {
                    r#type: ResponseFormatType::JsonSchema,
                    schema: json_schema.and_then(|js| js.get("schema")).cloned(),
                    name: json_schema
                        .and_then(|js| js.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string()),
                    strict: json_schema
                        .and_then(|js| js.get("strict"))
                        .and_then(|s| s.as_bool()),
                }
            }
            _ => IrResponseFormat {
                r#type: ResponseFormatType::Text,
                schema: None,
                name: None,
                strict: None,
            },
        }
    }

    fn json_content_to_ir<'a>(content: &serde_json::Value) -> IrContent<'a> {
        if let Some(s) = content.as_str() {
            IrContent::Text(Cow::Owned(s.to_string()))
        } else if let Some(arr) = content.as_array() {
            let parts: Vec<IrContentPart<'a>> = arr
                .iter()
                .filter_map(|item| {
                    let t = item.get("type")?.as_str()?;
                    match t {
                        "text" => {
                            let text = item.get("text")?.as_str()?;
                            Some(IrContentPart::Text {
                                text: Cow::Owned(text.to_string()),
                                citations: None,
                            })
                        }
                        "image_url" => {
                            let url = item
                                .get("image_url")
                                .and_then(|iu| iu.get("url"))
                                .and_then(|u| u.as_str())
                                .unwrap_or("");
                            if let Some((mime, data)) = url
                                .strip_prefix("data:")
                                .and_then(|rest| rest.split_once(','))
                                .map(|(meta, data)| {
                                    (meta.strip_suffix(";base64").unwrap_or(meta), data)
                                })
                                .filter(|(mime, _)| mime.starts_with("image/"))
                            {
                                Some(IrContentPart::ImageBase64 {
                                    media_type: Cow::Owned(mime.to_string()),
                                    data: Cow::Owned(data.to_string()),
                                })
                            } else {
                                let detail = item
                                    .get("image_url")
                                    .and_then(|iu| iu.get("detail"))
                                    .and_then(|d| d.as_str())
                                    .map(|s| Cow::Owned(s.to_string()));
                                Some(IrContentPart::ImageUrl {
                                    url: Cow::Owned(url.to_string()),
                                    detail,
                                })
                            }
                        }
                        "input_audio" => {
                            let ia = item.get("input_audio")?;
                            let data = ia.get("data")?.as_str()?;
                            let format = ia.get("format").and_then(|f| f.as_str()).unwrap_or("wav");
                            Some(IrContentPart::Audio {
                                media_type: Cow::Owned(format!("audio/{format}")),
                                data: Cow::Owned(data.to_string()),
                            })
                        }
                        "file" => {
                            let file = item.get("file")?;
                            if let Some(file_id) = file.get("file_id").and_then(|v| v.as_str()) {
                                Some(IrContentPart::FileRef {
                                    file_id: Cow::Owned(file_id.to_string()),
                                })
                            } else if let Some(fd) = file.get("file_data").and_then(|v| v.as_str())
                            {
                                // data URL 形式 → Document
                                let rest = fd.strip_prefix("data:")?;
                                let (meta, data) = rest.split_once(',')?;
                                let mime = meta.strip_suffix(";base64").unwrap_or(meta);
                                Some(IrContentPart::Document {
                                    media_type: Cow::Owned(mime.to_string()),
                                    data: Cow::Owned(data.to_string()),
                                    filename: file
                                        .get("filename")
                                        .and_then(|v| v.as_str())
                                        .map(|s| Cow::Owned(s.to_string())),
                                })
                            } else {
                                None
                            }
                        }
                        _ => Some(IrContentPart::Opaque {
                            provider: Cow::Borrowed("openai"),
                            payload: item.clone(),
                        }),
                    }
                })
                .collect();
            if parts.is_empty() {
                IrContent::Text(Cow::Owned(String::new()))
            } else {
                IrContent::Parts(parts)
            }
        } else {
            IrContent::Text(Cow::Owned(String::new()))
        }
    }
}

impl DecodeRequest for OpenAiShim {
    fn decode_request<'a>(&self, body: &'a [u8]) -> Result<IrRequest<'a>, CodecError> {
        let req: OaiRequestIn<'a> = serde_json::from_slice(body)?;

        let messages: Vec<IrMessage<'_>> = req
            .messages
            .iter()
            .map(|m| {
                let role = Self::parse_role(&m.role);
                let content = m
                    .content
                    .as_ref()
                    .map(Self::json_content_to_ir)
                    .unwrap_or(IrContent::Text(Cow::Owned(String::new())));

                let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .filter_map(|tc| {
                            let func = tc.function.as_ref()?;
                            Some(IrToolCall {
                                id: tc.id.clone().unwrap_or(Cow::Borrowed("")),
                                name: func.name.clone().unwrap_or(Cow::Borrowed("")),
                                arguments: func.arguments.clone().unwrap_or(Cow::Borrowed("")),
                            })
                        })
                        .collect()
                });

                IrMessage {
                    role,
                    content,
                    tool_call_id: m.tool_call_id.clone(),
                    tool_name: None,
                    tool_calls,
                    cache_control: None,
                    refusal: m.refusal.clone(),
                }
            })
            .collect();

        let mut messages = messages;
        super::backfill_tool_names(&mut messages);

        let tools: Option<Vec<IrTool<'_>>> = req.tools.as_ref().map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    if let Some(ref f) = t.function {
                        let extra = f.strict.map(|s| serde_json::json!({"strict": s}));
                        Some(IrTool {
                            tool_type: IrToolType::Function,
                            name: f.name.clone(),
                            description: f.description.clone(),
                            parameters: f.parameters.clone().unwrap_or(serde_json::json!({})),
                            cache_control: None,
                            extra,
                        })
                    } else {
                        let tool_type = match t.r#type.as_deref() {
                            Some("web_search") => IrToolType::WebSearch,
                            Some("code_interpreter") => IrToolType::CodeInterpreter,
                            Some("file_search") => IrToolType::FileSearch,
                            Some("computer_use") => IrToolType::ComputerUse,
                            Some("text_editor") => IrToolType::TextEditor,
                            Some("mcp") => IrToolType::Mcp,
                            _ => return None,
                        };
                        let type_str = t.r#type.as_deref().unwrap_or("unknown");
                        let mut raw = serde_json::Value::Object(t.extra_fields.clone());
                        raw["type"] = serde_json::json!(type_str);
                        Some(IrTool {
                            tool_type,
                            name: Cow::Owned(type_str.to_string()),
                            description: None,
                            parameters: serde_json::json!({}),
                            cache_control: None,
                            extra: Some(raw),
                        })
                    }
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(Self::parse_tool_choice);
        let response_format = req
            .response_format
            .as_ref()
            .map(Self::parse_response_format);
        let max_tokens = req.max_tokens.or(req.max_completion_tokens);

        let stop = req.stop.map(|s| match s {
            OaiStop::Single(s) => vec![s],
            OaiStop::Multiple(v) => v,
        });

        let metadata = if req.user.is_some() || req.service_tier.is_some() {
            Some(IrMetadata {
                user_id: req.user.as_ref().map(|u| Cow::Owned(u.to_string())),
                service_tier: req.service_tier.clone(),
            })
        } else {
            None
        };

        // reasoning_effort → ReasoningConfig（effort 原样保留，往返不漂移）
        let reasoning = req.reasoning_effort.as_deref().map(|effort| {
            let mode = match effort {
                "none" => ReasoningMode::Disabled,
                "high" => ReasoningMode::Enabled,
                _ => ReasoningMode::Auto,
            };
            ReasoningConfig {
                mode,
                budget_tokens: None,
                effort: Some(effort.to_string()),
            }
        });

        // 从原始 JSON 提取 preserved parts
        let req_json: serde_json::Value = serde_json::from_slice(body)?;
        let preserved = super::extract_preserved(&req_json);
        let mut provider_metadata: Option<Box<serde_json::Value>> = None;
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved);

        // prediction / audio 配置 → provider_metadata 透传（编码时回填）
        if req.prediction.is_some() || req.audio.is_some() {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            if let Some(ref p) = req.prediction {
                pm["prediction"] = p.clone();
            }
            if let Some(ref a) = req.audio {
                pm["audio"] = a.clone();
            }
        }

        Ok(IrRequest {
            model: req.model,
            messages,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: None,
            max_tokens,
            stop,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            seed: req.seed,
            n: req.n,
            logprobs: req.logprobs,
            top_logprobs: req.top_logprobs,
            stream: req.stream,
            store: req.store,
            modalities: req.modalities,
            tools,
            tool_choice,
            parallel_tool_calls: req.parallel_tool_calls,
            reasoning,
            response_format,
            previous_response_id: None,
            truncation: None,
            metadata,
            provider_metadata,
            metadata_mode: MetadataMode::default(),
        })
    }
}

// ─── EncodeResponse ─────────────────────────────────────────────────

impl EncodeResponse for OpenAiShim {
    fn encode_response(&self, ir: &IrResponse<'_>) -> Result<Vec<u8>, CodecError> {
        let mut resp = serde_json::json!({
            "id": ir.id,
            "object": "chat.completion",
            "model": ir.model,
            "created": ir.provider_metadata.as_ref()
                .and_then(|pm| pm.get("created"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
        });

        let mut preserved = Vec::new();

        // choices
        let choices: Vec<serde_json::Value> = ir
            .choices
            .iter()
            .map(|c| {
                let mut message = serde_json::Map::new();
                message.insert("role".into(), serde_json::json!("assistant"));

                // 分离 reasoning_content 和 text content
                let mut reasoning_content: Option<String> = None;
                let mut reasoning_signature: Option<String> = None;
                let mut audio_data: Option<String> = None;
                let mut annotations: Vec<serde_json::Value> = Vec::new();
                let text_content: Option<String>;

                match &c.message.content {
                    IrContent::Text(s) => {
                        text_content = Some(s.to_string());
                    }
                    IrContent::Parts(parts) => {
                        let mut reasoning_parts = Vec::new();
                        let mut text_parts = Vec::new();
                        for p in parts {
                            match p {
                                IrContentPart::Reasoning { text, signature } => {
                                    reasoning_parts.push(text.as_ref().to_string());
                                    if reasoning_signature.is_none() {
                                        if let Some(sig) = signature {
                                            reasoning_signature = Some(sig.as_ref().to_string());
                                        }
                                    }
                                }
                                IrContentPart::Text { text, citations } => {
                                    text_parts.push(text.as_ref().to_string());
                                    if let Some(cits) = citations {
                                        for cit in cits {
                                            annotations.push(Self::citation_to_annotation(cit));
                                        }
                                    }
                                }
                                IrContentPart::Audio { data, .. } => {
                                    // OpenAI 音频响应 → message.audio.data
                                    audio_data = Some(data.as_ref().to_string());
                                }
                                IrContentPart::FunctionCall { .. } => {}
                                IrContentPart::FunctionResponse { .. } => {
                                    if let Ok(v) = serde_json::to_value(p) {
                                        preserved.push(v);
                                    }
                                }
                                // OpenAI 响应不支持的内容类型 → 保留
                                IrContentPart::ImageUrl { .. }
                                | IrContentPart::ImageBase64 { .. }
                                | IrContentPart::Video { .. }
                                | IrContentPart::Document { .. }
                                | IrContentPart::FileRef { .. }
                                | IrContentPart::RedactedReasoning { .. }
                                | IrContentPart::Opaque { .. } => {
                                    if let Ok(v) = serde_json::to_value(p) {
                                        preserved.push(v);
                                    }
                                }
                            }
                        }
                        if !reasoning_parts.is_empty() {
                            reasoning_content = Some(reasoning_parts.join(""));
                        }
                        text_content = Some(text_parts.join(""));
                    }
                }

                if let Some(ref rc) = reasoning_content {
                    message.insert(
                        "reasoning_content".into(),
                        serde_json::Value::String(rc.clone()),
                    );
                }
                if let Some(ref sig) = reasoning_signature {
                    message.insert(
                        "reasoning_signature".into(),
                        serde_json::Value::String(sig.clone()),
                    );
                }
                if let Some(ref ad) = audio_data {
                    message.insert("audio".into(), serde_json::json!({ "data": ad }));
                }
                message.insert(
                    "content".into(),
                    text_content
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );

                // refusal（拒答文本）
                if let Some(ref refusal) = c.message.refusal {
                    message.insert(
                        "refusal".into(),
                        serde_json::Value::String(refusal.to_string()),
                    );
                }

                // tool_calls：优先从 content 的 FunctionCall 部件提取（保留交错源的数据），
                // 无 FunctionCall 时 fallback 到 tool_calls 字段
                {
                    let mut tc_arr: Vec<serde_json::Value> = Vec::new();
                    if let IrContent::Parts(ref parts) = c.message.content {
                        for p in parts {
                            if let IrContentPart::FunctionCall {
                                id,
                                name,
                                arguments,
                            } = p
                            {
                                tc_arr.push(serde_json::json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    }
                                }));
                            }
                        }
                    }
                    if tc_arr.is_empty() {
                        if let Some(ref tcs) = c.message.tool_calls {
                            for tc in tcs {
                                tc_arr.push(serde_json::json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }
                                }));
                            }
                        }
                    }
                    if !tc_arr.is_empty() {
                        message.insert("tool_calls".into(), serde_json::Value::Array(tc_arr));
                    }
                }

                // web search URL 引文 → message.annotations
                if !annotations.is_empty() {
                    message.insert("annotations".into(), serde_json::Value::Array(annotations));
                }

                let mut choice = serde_json::json!({
                    "index": c.index,
                    "message": serde_json::Value::Object(message),
                });
                choice["finish_reason"] = c
                    .finish_reason
                    .map(|fr| serde_json::Value::String(Self::finish_reason_to_str(fr).to_string()))
                    .unwrap_or(serde_json::Value::Null);
                if let Some(ref lp) = c.logprobs {
                    choice["logprobs"] = Self::ir_logprobs_to_json(lp);
                }
                choice
            })
            .collect();

        resp["choices"] = serde_json::Value::Array(choices);

        // usage
        if let Some(ref usage) = ir.usage {
            resp["usage"] = Self::ir_usage_to_json(usage);
        }

        // system_fingerprint, service_tier, error, metadata from provider_metadata
        if let Some(ref pm) = ir.provider_metadata {
            if let Some(fp) = pm.get("system_fingerprint").and_then(|v| v.as_str()) {
                resp["system_fingerprint"] = serde_json::Value::String(fp.to_string());
            }
            if let Some(st) = pm.get("service_tier").and_then(|v| v.as_str()) {
                resp["service_tier"] = serde_json::Value::String(st.to_string());
            }
            if let Some(err) = pm.get("error") {
                resp["error"] = err.clone();
            }
            if let Some(meta) = pm.get("metadata") {
                resp["metadata"] = meta.clone();
            }
        }

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        super::attach_preserved(&mut resp, preserved);

        serde_json::to_vec(&resp).map_err(CodecError::from)
    }
}

impl OpenAiShim {
    /// IrCitation → OpenAI `message.annotations` 项（web search URL 引文）
    fn citation_to_annotation(cit: &IrCitation<'_>) -> serde_json::Value {
        let mut uc = serde_json::Map::new();
        if let Some(ref url) = cit.url {
            uc.insert("url".into(), serde_json::Value::String(url.to_string()));
        }
        if let Some(ref title) = cit.title {
            uc.insert("title".into(), serde_json::Value::String(title.to_string()));
        }
        if let Some(ref content) = cit.cited_text {
            uc.insert(
                "content".into(),
                serde_json::Value::String(content.to_string()),
            );
        }
        serde_json::json!({
            "type": cit.r#type,
            "url_citation": serde_json::Value::Object(uc),
        })
    }

    fn finish_reason_to_str(fr: IrFinishReason) -> &'static str {
        match fr {
            IrFinishReason::Stop | IrFinishReason::StopSequence | IrFinishReason::PauseTurn => {
                "stop"
            }
            IrFinishReason::Length => "length",
            IrFinishReason::ToolCalls => "tool_calls",
            // 安全拦截/审核类 → content_filter，不得伪装成正常完成
            IrFinishReason::ContentFilter | IrFinishReason::Safety | IrFinishReason::Recitation => {
                "content_filter"
            }
            // OpenAI 无对应值，保守取 stop
            IrFinishReason::MalformedFunctionCall => "stop",
        }
    }

    fn ir_logprobs_to_json(lp: &IrLogprobs) -> serde_json::Value {
        fn tokens_to_json(tokens: &[IrTokenLogprob]) -> serde_json::Value {
            serde_json::Value::Array(
                tokens
                    .iter()
                    .map(|t| {
                        let mut obj = serde_json::json!({
                            "token": t.token,
                            "logprob": t.logprob,
                        });
                        if let Some(ref b) = t.bytes {
                            obj["bytes"] = serde_json::json!(b);
                        } else {
                            obj["bytes"] = serde_json::Value::Null;
                        }
                        if let Some(ref tl) = t.top_logprobs {
                            obj["top_logprobs"] = serde_json::Value::Array(
                                tl.iter()
                                    .map(|tp| {
                                        let mut tobj = serde_json::json!({
                                            "token": tp.token,
                                            "logprob": tp.logprob,
                                        });
                                        if let Some(ref b) = tp.bytes {
                                            tobj["bytes"] = serde_json::json!(b);
                                        } else {
                                            tobj["bytes"] = serde_json::Value::Null;
                                        }
                                        tobj
                                    })
                                    .collect(),
                            );
                        }
                        obj
                    })
                    .collect(),
            )
        }
        let mut obj = serde_json::Map::new();
        if let Some(ref c) = lp.content {
            obj.insert("content".into(), tokens_to_json(c));
        } else {
            obj.insert("content".into(), serde_json::Value::Null);
        }
        if let Some(ref r) = lp.refusal {
            obj.insert("refusal".into(), tokens_to_json(r));
        } else {
            obj.insert("refusal".into(), serde_json::Value::Null);
        }
        serde_json::Value::Object(obj)
    }

    fn ir_usage_to_json(u: &IrUsage) -> serde_json::Value {
        let mut usage = serde_json::json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        });

        // completion_tokens_details
        let mut details = serde_json::Map::new();
        if let Some(r) = u.reasoning_tokens {
            details.insert("reasoning_tokens".into(), serde_json::json!(r));
        }
        if let Some(a) = u.audio_tokens {
            details.insert("audio_tokens".into(), serde_json::json!(a));
        }
        if let Some(a) = u.accepted_prediction_tokens {
            details.insert("accepted_prediction_tokens".into(), serde_json::json!(a));
        }
        if let Some(r) = u.rejected_prediction_tokens {
            details.insert("rejected_prediction_tokens".into(), serde_json::json!(r));
        }
        if !details.is_empty() {
            usage["completion_tokens_details"] = serde_json::Value::Object(details);
        }

        // prompt_tokens_details
        if let Some(c) = u.cache_read_tokens {
            usage["prompt_tokens_details"] = serde_json::json!({ "cached_tokens": c });
        }

        usage
    }
}

// ─── EncodeStream ───────────────────────────────────────────────────

/// OpenAI 流式编码器 — 记住 Start 的 id/model，为每个 chunk 补齐（OpenAI
/// 有线格式要求每个 chunk 携带 id/model）。跟踪已 Start 的 tool_call，
/// 对未见过 Start 的 ToolCallDone 补发完整 tool_call 帧。
pub struct OaiStreamEncoder {
    id: String,
    model: String,
    /// 已通过 ToolCallStart 宣告过的 tool index
    announced_tools: Vec<u32>,
    /// 缓存的最新累计 usage（多次到达时后到取代先到，Done 前发出一次）
    pending_usage: Option<IrUsage>,
    preserved_events: Vec<serde_json::Value>,
}

impl OaiStreamEncoder {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            announced_tools: Vec::new(),
            pending_usage: None,
            preserved_events: Vec::new(),
        }
    }

    fn frame(&self, mut chunk: serde_json::Value) -> Result<Vec<u8>, CodecError> {
        chunk["id"] = serde_json::Value::String(self.id.clone());
        chunk["object"] = serde_json::json!("chat.completion.chunk");
        chunk["model"] = serde_json::Value::String(self.model.clone());
        Ok(format!("data: {}\n\n", serde_json::to_string(&chunk)?).into_bytes())
    }
}

impl Default for OaiStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeStream for OaiStreamEncoder {
    fn encode_sse_event(&mut self, event: &IrStreamEvent<'_>) -> Result<Vec<u8>, CodecError> {
        match event {
            IrStreamEvent::Start { id, model, usage } => {
                self.id = id.to_string();
                self.model = model.to_string();
                self.pending_usage = usage.clone();
                self.frame(serde_json::json!({
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant" },
                    }],
                }))
            }
            IrStreamEvent::ContentDelta { index, delta } => self.frame(serde_json::json!({
                "choices": [{
                    "index": index,
                    "delta": { "content": delta },
                }],
            })),
            IrStreamEvent::Logprobs { index, logprobs } => {
                // OpenAI 在与内容同一 chunk 上回传 logprobs（choice.logprobs）
                self.frame(serde_json::json!({
                    "choices": [{
                        "index": index,
                        "delta": {},
                        "logprobs": OpenAiShim::ir_logprobs_to_json(logprobs),
                    }],
                }))
            }
            IrStreamEvent::ReasoningDelta { index, delta } => self.frame(serde_json::json!({
                "choices": [{
                    "index": index,
                    "delta": { "reasoning_content": delta },
                }],
            })),
            IrStreamEvent::ToolCallStart {
                index,
                choice_index,
                id,
                name,
            } => {
                self.announced_tools.push(*index);
                self.frame(serde_json::json!({
                    "choices": [{
                        "index": choice_index,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": { "name": name, "arguments": "" },
                            }]
                        },
                    }],
                }))
            }
            IrStreamEvent::ToolCallDelta {
                index,
                choice_index,
                arguments_delta,
            } => self.frame(serde_json::json!({
                "choices": [{
                    "index": choice_index,
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "function": { "arguments": arguments_delta },
                        }]
                    },
                }],
            })),
            IrStreamEvent::ToolCallDone {
                index,
                choice_index,
                id,
                name,
                arguments,
            } => {
                // 已流式发送过 Start+Delta 的 tool_call 无需重复；
                // 未宣告过的（如 Gemini 一次性完整 function_call 且上游未发 Start）补发完整帧
                if self.announced_tools.contains(index) {
                    Ok(Vec::new())
                } else {
                    self.announced_tools.push(*index);
                    self.frame(serde_json::json!({
                        "choices": [{
                            "index": choice_index,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "id": id,
                                    "type": "function",
                                    "function": { "name": name, "arguments": arguments },
                                }]
                            },
                        }],
                    }))
                }
            }
            IrStreamEvent::ChoiceFinish {
                index,
                finish_reason,
            } => {
                let mut chunk = serde_json::json!({
                    "choices": [{
                        "index": index,
                        "delta": {},
                        "finish_reason": OpenAiShim::finish_reason_to_str(*finish_reason),
                    }],
                });
                if !self.preserved_events.is_empty() {
                    let preserved = std::mem::take(&mut self.preserved_events);
                    super::attach_preserved(&mut chunk, preserved);
                }
                self.frame(chunk)
            }
            IrStreamEvent::Usage(usage) => {
                // 累计值可能多次到达（如 Gemini 每 chunk 回传）— 缓存最新，
                // Done 前发出一次尾部 usage chunk（OpenAI 惯例）
                self.pending_usage = Some(usage.clone());
                Ok(Vec::new())
            }
            IrStreamEvent::Done => {
                let mut out = Vec::new();
                if let Some(u) = self.pending_usage.take() {
                    out.extend(self.frame(serde_json::json!({
                        "choices": [],
                        "usage": OpenAiShim::ir_usage_to_json(&u),
                    }))?);
                }
                out.extend(b"data: [DONE]\n\n");
                Ok(out)
            }
            IrStreamEvent::ReasoningDone { index, signature } => {
                // 签名在块完成时到达 → 补发一个携带 reasoning_signature 的 chunk
                if let Some(sig) = signature {
                    self.frame(serde_json::json!({
                        "choices": [{
                            "index": index,
                            "delta": { "reasoning_signature": sig },
                        }],
                    }))
                } else {
                    Ok(Vec::new())
                }
            }
            IrStreamEvent::ContentDone { .. } => Ok(Vec::new()),
            IrStreamEvent::Citation { index, citation } => self.frame(serde_json::json!({
                "choices": [{
                    "index": index,
                    "delta": { "annotations": [citation] },
                }],
            })),
            IrStreamEvent::RedactedReasoning { .. } | IrStreamEvent::OpaqueBlock { .. } => {
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
                Ok(Vec::new())
            }
            IrStreamEvent::Error { message } => {
                let chunk = serde_json::json!({
                    "error": { "message": message },
                });
                Ok(format!("data: {}\n\n", serde_json::to_string(&chunk)?).into_bytes())
            }
        }
    }
}
