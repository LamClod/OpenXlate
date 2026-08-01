//! OpenAI Responses API shim — IR ↔ POST /v1/responses
//!
//! 与 Chat Completions 的关键差异：
//! - input/output 是类型化 Item 数组（message / function_call / reasoning 等）
//! - system prompt 用顶层 `instructions` 字段
//! - 支持 `previous_response_id` 服务端状态管理
//! - 结构化输出用 `text.format` 而非 `response_format`
//! - assistant 消息的 tool_calls → function_call input items

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::{
    DecodeRequest, DecodeResponse, DecodeStream, EncodeRequest, EncodeResponse, EncodeStream,
};
use crate::codec::error::CodecError;
use crate::codec::ir::*;

// ─── Responses API 有线格式（编码）─────────────────────────────────

#[derive(Serialize)]
struct RspRequest<'a> {
    model: &'a str,
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

// ─── 反序列化（响应）────────────────────────────────────────────────

#[derive(Deserialize)]
struct RspResponse<'a> {
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow, default)]
    model: Cow<'a, str>,
    #[serde(default)]
    output: Vec<RspOutputItem<'a>>,
    #[serde(default)]
    usage: Option<RspUsage>,
    #[serde(borrow, default)]
    status: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspOutputItem<'a> {
    Message {
        #[serde(borrow, default)]
        #[allow(dead_code)]
        role: Option<Cow<'a, str>>,
        #[serde(default)]
        content: Vec<RspContentBlock<'a>>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(borrow)]
        #[allow(dead_code)]
        id: Option<Cow<'a, str>>,
        #[serde(borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(borrow)]
        name: Cow<'a, str>,
        #[serde(borrow)]
        arguments: Cow<'a, str>,
    },
    Reasoning {
        #[serde(default)]
        content: Vec<RspReasoningBlock<'a>>,
    },
    /// 内置工具调用（web_search_call 等）
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "computer_call")]
    ComputerCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspContentBlock<'a> {
    OutputText {
        #[serde(borrow)]
        text: Cow<'a, str>,
        #[serde(default)]
        annotations: Option<Vec<serde_json::Value>>,
    },
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspReasoningBlock<'a> {
    ReasoningText {
        #[serde(borrow)]
        text: Cow<'a, str>,
        #[serde(borrow, default)]
        signature: Option<Cow<'a, str>>,
    },
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

#[derive(Deserialize)]
struct RspUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
    #[serde(default)]
    output_tokens_details: Option<RspOutputTokenDetails>,
    #[serde(default)]
    input_tokens_details: Option<RspInputTokenDetails>,
}

#[derive(Deserialize)]
struct RspOutputTokenDetails {
    reasoning_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct RspInputTokenDetails {
    cached_tokens: Option<u32>,
}

// ─── SSE 流事件 ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RspStreamEvent<'a> {
    #[serde(borrow)]
    r#type: Cow<'a, str>,
    #[serde(default)]
    item: Option<serde_json::Value>,
    #[serde(default)]
    delta: Option<serde_json::Value>,
    /// function_call_arguments.done 事件的顶层完整 arguments
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    response: Option<RspStreamResponse<'a>>,
    #[serde(default)]
    output_index: Option<u32>,
}

#[derive(Deserialize)]
struct RspStreamResponse<'a> {
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow, default)]
    model: Cow<'a, str>,
    usage: Option<RspUsage>,
    #[serde(borrow, default)]
    status: Option<Cow<'a, str>>,
}

/// 高频文本 delta 事件的快路径解析 — 零拷贝
#[derive(Deserialize)]
struct RspTextDeltaFast<'a> {
    #[serde(borrow)]
    r#type: Cow<'a, str>,
    #[serde(borrow, default)]
    delta: Option<Cow<'a, str>>,
}

// ─── 实现 ──────────────────────────────────────────────────────────

pub struct OpenAiResponsesShim;

impl OpenAiResponsesShim {
    fn convert_usage(u: &RspUsage) -> IrUsage {
        let input = u.input_tokens.unwrap_or(0);
        let output = u.output_tokens.unwrap_or(0);
        IrUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: u
                .total_tokens
                .unwrap_or_else(|| input.saturating_add(output)),
            cache_read_tokens: u
                .input_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
            cache_creation_tokens: None,
            reasoning_tokens: u
                .output_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            audio_tokens: None,
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }

    /// Responses API tool `type` 字符串 → IrToolType
    fn parse_tool_type(s: &str) -> IrToolType {
        match s {
            "function" => IrToolType::Function,
            "web_search_preview" | "web_search" => IrToolType::WebSearch,
            "file_search" => IrToolType::FileSearch,
            "code_interpreter" => IrToolType::CodeInterpreter,
            "computer_use_preview" | "computer_use" => IrToolType::ComputerUse,
            "text_editor" => IrToolType::TextEditor,
            "mcp" => IrToolType::Mcp,
            _ => IrToolType::Function,
        }
    }

    /// IrToolType → Responses API tool `type` 字符串
    fn tool_type_to_wire(t: &IrToolType) -> &'static str {
        match t {
            IrToolType::Function => "function",
            IrToolType::WebSearch => "web_search_preview",
            IrToolType::FileSearch => "file_search",
            IrToolType::CodeInterpreter => "code_interpreter",
            IrToolType::ComputerUse => "computer_use_preview",
            IrToolType::TextEditor => "text_editor",
            IrToolType::Mcp => "mcp",
        }
    }

    fn ir_tool_choice_to_json(tc: &IrToolChoice<'_>) -> serde_json::Value {
        match tc {
            IrToolChoice::Auto => serde_json::json!("auto"),
            IrToolChoice::None => serde_json::json!("none"),
            IrToolChoice::Required => serde_json::json!("required"),
            IrToolChoice::Specific { name } => serde_json::json!({
                "type": "function",
                "name": name
            }),
        }
    }

    fn ir_content_to_rsp_content(
        content: &IrContent<'_>,
        role: &str,
        preserved: &mut Vec<serde_json::Value>,
    ) -> serde_json::Value {
        match content {
            IrContent::Text(s) => serde_json::json!(s),
            IrContent::Parts(parts) => {
                let arr: Vec<serde_json::Value> = parts
                    .iter()
                    .filter_map(|p| match p {
                        IrContentPart::Text { text, .. } => {
                            let type_name = if role == "user" {
                                "input_text"
                            } else {
                                "output_text"
                            };
                            Some(serde_json::json!({
                                "type": type_name,
                                "text": text,
                            }))
                        }
                        IrContentPart::ImageUrl { url, detail, .. } => {
                            let mut v = serde_json::json!({
                                "type": "input_image",
                                "image_url": url,
                            });
                            if let Some(d) = detail {
                                v["detail"] = serde_json::Value::String(d.to_string());
                            }
                            Some(v)
                        }
                        IrContentPart::ImageBase64 {
                            media_type, data, ..
                        } => {
                            let data_url = format!("data:{media_type};base64,{data}");
                            Some(serde_json::json!({
                                "type": "input_image",
                                "image_url": data_url,
                            }))
                        }
                        IrContentPart::Document {
                            media_type,
                            data,
                            filename,
                        } => {
                            let mut f = serde_json::json!({
                                "type": "input_file",
                                "file_data": format!("data:{media_type};base64,{data}"),
                            });
                            if let Some(name) = filename {
                                f["filename"] = serde_json::Value::String(name.to_string());
                            }
                            Some(f)
                        }
                        IrContentPart::FileRef { file_id } => Some(serde_json::json!({
                            "type": "input_file",
                            "file_id": file_id,
                        })),
                        IrContentPart::Audio { media_type, data } => {
                            let format = media_type
                                .strip_prefix("audio/")
                                .unwrap_or(media_type.as_ref());
                            Some(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": { "data": data, "format": format },
                            }))
                        }
                        // FunctionCall 由 assistant 消息处理器提取为 function_call
                        // input item（见 Role::Assistant 分支），此处静默跳过避免重复
                        IrContentPart::FunctionCall { .. } => None,
                        // 不支持的类型 → 无损保留
                        IrContentPart::Video { .. }
                        | IrContentPart::Reasoning { .. }
                        | IrContentPart::RedactedReasoning { .. }
                        | IrContentPart::FunctionResponse { .. }
                        | IrContentPart::Opaque { .. } => {
                            if let Ok(v) = serde_json::to_value(p) {
                                preserved.push(v);
                            }
                            None
                        }
                    })
                    .collect();
                if arr.len() == 1 {
                    // 单文本快路径
                    if let Some(text) = arr[0].get("text") {
                        return text.clone();
                    }
                }
                serde_json::Value::Array(arr)
            }
        }
    }

    fn parse_status(s: &str) -> Option<IrFinishReason> {
        match s {
            "completed" => Some(IrFinishReason::Stop),
            "incomplete" => Some(IrFinishReason::Length),
            "failed" => Some(IrFinishReason::ContentFilter),
            "cancelled" => Some(IrFinishReason::ContentFilter),
            // 非终态：queued / in_progress → 无 finish_reason
            "queued" | "in_progress" => None,
            _ => Some(IrFinishReason::Stop),
        }
    }
}

impl EncodeRequest for OpenAiResponsesShim {
    fn encode_request(&self, ir: &IrRequest<'_>) -> Result<Vec<u8>, CodecError> {
        let mut instructions_acc: Option<String> = None;
        let mut input: Vec<serde_json::Value> = Vec::new();
        let mut preserved: Vec<serde_json::Value> = Vec::new();

        for msg in &ir.messages {
            match msg.role {
                Role::System | Role::Developer => {
                    // 多条 system/developer 消息拼接（覆盖会丢前面的）
                    let text = msg.content.text_concat();
                    if !text.is_empty() {
                        match instructions_acc {
                            Some(ref mut acc) => {
                                acc.push('\n');
                                acc.push_str(&text);
                            }
                            None => instructions_acc = Some(text),
                        }
                    }
                }
                Role::User => {
                    let content =
                        Self::ir_content_to_rsp_content(&msg.content, "user", &mut preserved);
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::Assistant => {
                    // assistant 消息：text → message, tool_calls → function_call items, reasoning → reasoning item
                    // 先输出 reasoning content（如果有）
                    if let IrContent::Parts(ref parts) = msg.content {
                        let reasoning_parts: Vec<serde_json::Value> = parts
                            .iter()
                            .filter_map(|p| match p {
                                IrContentPart::Reasoning { text, signature } => {
                                    let mut block = serde_json::json!({
                                        "type": "reasoning_text",
                                        "text": text,
                                    });
                                    if let Some(sig) = signature {
                                        block["signature"] =
                                            serde_json::Value::String(sig.to_string());
                                    }
                                    Some(block)
                                }
                                IrContentPart::RedactedReasoning { data } => {
                                    Some(serde_json::json!({
                                        "type": "redacted_reasoning",
                                        "data": data,
                                    }))
                                }
                                _ => None,
                            })
                            .collect();
                        if !reasoning_parts.is_empty() {
                            input.push(serde_json::json!({
                                "type": "reasoning",
                                "content": reasoning_parts,
                            }));
                        }
                    }

                    // 输出 text content（过滤掉 reasoning 部件）
                    let text_content: String = match &msg.content {
                        IrContent::Text(s) => s.to_string(),
                        IrContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| {
                                if let IrContentPart::Text { text, .. } = p {
                                    Some(text.as_ref())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    };
                    if !text_content.is_empty() {
                        input.push(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "content": text_content,
                        }));
                    }

                    // tool_calls → function_call items
                    // 优先从 content 的 FunctionCall 部件提取（保留交错源的数据），
                    // 无 FunctionCall 时 fallback 到 tool_calls 字段
                    {
                        let mut found_fc = false;
                        if let IrContent::Parts(ref parts) = msg.content {
                            for p in parts {
                                if let IrContentPart::FunctionCall {
                                    id,
                                    name,
                                    arguments,
                                } = p
                                {
                                    input.push(serde_json::json!({
                                        "type": "function_call",
                                        "call_id": id,
                                        "name": name,
                                        "arguments": arguments,
                                    }));
                                    found_fc = true;
                                }
                            }
                        }
                        if !found_fc {
                            if let Some(ref tcs) = msg.tool_calls {
                                for tc in tcs {
                                    input.push(serde_json::json!({
                                        "type": "function_call",
                                        "call_id": tc.id,
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }));
                                }
                            }
                        }
                    }
                    // preserved fallback: assistant content 中未被上面处理的部件
                    if let IrContent::Parts(ref parts) = msg.content {
                        for p in parts {
                            match p {
                                IrContentPart::Text { .. }
                                | IrContentPart::Reasoning { .. }
                                | IrContentPart::RedactedReasoning { .. }
                                | IrContentPart::FunctionCall { .. } => {}
                                other => {
                                    if let Ok(val) = serde_json::to_value(other) {
                                        preserved.push(serde_json::json!({
                                            "type": "unhandled_assistant_content",
                                            "part": val,
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }
                Role::Tool => {
                    let text = msg.content.text_concat();
                    let call_id = msg.tool_call_id.as_deref().unwrap_or("");
                    input.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": text,
                    }));
                }
            }
        }

        let tools: Option<Vec<serde_json::Value>> = ir.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| match t.tool_type {
                    IrToolType::Function => {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".into(), serde_json::json!("function"));
                        obj.insert("name".into(), serde_json::json!(t.name));
                        if let Some(ref d) = t.description {
                            obj.insert("description".into(), serde_json::json!(d));
                        }
                        obj.insert("parameters".into(), t.parameters.clone());
                        if let Some(serde_json::Value::Object(extra)) = &t.extra {
                            for (k, v) in extra {
                                obj.insert(k.clone(), v.clone());
                            }
                        }
                        serde_json::Value::Object(obj)
                    }
                    // 内置工具（web_search_preview / file_search / ...）：
                    // {"type": <wire>, ...extra}
                    _ => {
                        let mut obj = serde_json::Map::new();
                        obj.insert(
                            "type".into(),
                            serde_json::json!(Self::tool_type_to_wire(&t.tool_type)),
                        );
                        if let Some(serde_json::Value::Object(extra)) = &t.extra {
                            for (k, v) in extra {
                                obj.insert(k.clone(), v.clone());
                            }
                        }
                        serde_json::Value::Object(obj)
                    }
                })
                .collect()
        });

        let tool_choice = ir.tool_choice.as_ref().map(Self::ir_tool_choice_to_json);

        // previous_response_id 从 IR 读取
        let previous_response_id = ir.previous_response_id.as_deref();

        // 结构化输出 → text.format
        let text = ir.response_format.as_ref().and_then(|rf| match rf.r#type {
            ResponseFormatType::Text => None,
            ResponseFormatType::JsonObject => {
                Some(serde_json::json!({ "format": { "type": "json_object" } }))
            }
            ResponseFormatType::JsonSchema => {
                let mut schema_obj = serde_json::Map::new();
                schema_obj.insert("type".into(), serde_json::json!("json_schema"));
                if let Some(ref name) = rf.name {
                    schema_obj.insert("name".into(), serde_json::json!(name));
                }
                if let Some(ref schema) = rf.schema {
                    schema_obj.insert("schema".into(), schema.clone());
                }
                if let Some(strict) = rf.strict {
                    schema_obj.insert("strict".into(), serde_json::json!(strict));
                }
                Some(serde_json::json!({ "format": schema_obj }))
            }
        });

        // ReasoningConfig → reasoning: { effort }
        // Disabled 不发 reasoning（"low" 仍会开启推理，语义相反）；
        // 保留原始 effort 字符串以保证 OpenAI↔Responses 往返不漂移
        let reasoning = ir.reasoning.as_ref().and_then(|r| {
            if r.mode == ReasoningMode::Disabled {
                return None;
            }
            let effort = r.effort.as_deref().unwrap_or(match r.mode {
                ReasoningMode::Auto => "medium",
                ReasoningMode::Enabled => "high",
                ReasoningMode::Disabled => unreachable!(),
            });
            Some(serde_json::json!({ "effort": effort }))
        });

        // truncation from IR
        let truncation = ir.truncation.as_ref().map(|t| match t.r#type {
            TruncationType::Auto => serde_json::json!({ "type": "auto" }),
            TruncationType::Disabled => serde_json::json!({ "type": "disabled" }),
        });

        // include / background 从 provider_metadata 透传（Strip 模式下跳过）
        let (include, background) = if ir.metadata_mode == MetadataMode::Strip {
            (None, None)
        } else {
            let include = ir
                .provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("include").cloned());
            let background = ir
                .provider_metadata
                .as_ref()
                .and_then(|pm| pm.get("background"))
                .and_then(|v| v.as_bool());
            (include, background)
        };

        let instructions = instructions_acc.as_deref();
        let req = RspRequest {
            model: &ir.model,
            input,
            instructions,
            temperature: ir.temperature,
            top_p: ir.top_p,
            max_output_tokens: ir.max_tokens,
            stream: ir.stream,
            tools,
            tool_choice,
            previous_response_id,
            text,
            reasoning,
            parallel_tool_calls: ir.parallel_tool_calls,
            truncation,
            service_tier: ir.metadata.as_ref().and_then(|m| m.service_tier.as_deref()),
            store: ir.store,
            include,
            background,
            user: ir.metadata.as_ref().and_then(|m| m.user_id.as_deref()),
        };

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        let mut json = serde_json::to_value(&req)?;
        super::attach_preserved(&mut json, preserved);
        serde_json::to_vec(&json).map_err(CodecError::from)
    }

    fn endpoint(&self, base_url: &str) -> String {
        format!("{base_url}/v1/responses")
    }

    fn headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![
            ("Authorization", format!("Bearer {api_key}")),
            ("Content-Type", "application/json".into()),
        ]
    }
}

impl DecodeResponse for OpenAiResponsesShim {
    fn decode_response<'a>(&self, body: &'a [u8]) -> Result<IrResponse<'a>, CodecError> {
        let rsp: RspResponse<'a> = serde_json::from_slice(body)?;

        let mut content_parts: Vec<IrContentPart<'a>> = Vec::new();
        let mut tool_calls: Vec<IrToolCall<'a>> = Vec::new();

        for item in &rsp.output {
            match item {
                RspOutputItem::Message { content, .. } => {
                    for block in content {
                        match block {
                            RspContentBlock::OutputText { text, annotations } => {
                                let citations = annotations.as_ref().and_then(|anns| {
                                    let cits: Vec<IrCitation<'_>> = anns
                                        .iter()
                                        .map(|a| IrCitation {
                                            r#type: a
                                                .get("type")
                                                .and_then(|v| v.as_str())
                                                .map(|s| Cow::Owned(s.to_string()))
                                                .unwrap_or(Cow::Borrowed("")),
                                            url: a
                                                .get("url")
                                                .and_then(|v| v.as_str())
                                                .map(|s| Cow::Owned(s.to_string())),
                                            title: a
                                                .get("title")
                                                .and_then(|v| v.as_str())
                                                .map(|s| Cow::Owned(s.to_string())),
                                            cited_text: None,
                                            encrypted_index: None,
                                        })
                                        .collect();
                                    if cits.is_empty() {
                                        None
                                    } else {
                                        Some(cits)
                                    }
                                });
                                content_parts.push(IrContentPart::Text {
                                    text: text.clone(),
                                    citations,
                                });
                            }
                            RspContentBlock::Unknown(raw) => {
                                content_parts.push(IrContentPart::Opaque {
                                    provider: Cow::Borrowed("openai_responses"),
                                    payload: raw.clone(),
                                });
                            }
                        }
                    }
                }
                RspOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    tool_calls.push(IrToolCall {
                        id: call_id.clone().unwrap_or(Cow::Borrowed("")),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                }
                RspOutputItem::Reasoning { content } => {
                    for block in content {
                        match block {
                            RspReasoningBlock::ReasoningText { text, signature } => {
                                content_parts.push(IrContentPart::Reasoning {
                                    text: text.clone(),
                                    signature: signature.clone(),
                                });
                            }
                            RspReasoningBlock::Unknown(raw) => {
                                content_parts.push(IrContentPart::Opaque {
                                    provider: Cow::Borrowed("openai_responses"),
                                    payload: raw.clone(),
                                });
                            }
                        }
                    }
                }
                // 内置工具调用 → Opaque
                RspOutputItem::WebSearchCall { payload } => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: {
                            let mut v = payload.clone();
                            v["type"] = serde_json::json!("web_search_call");
                            v
                        },
                    });
                }
                RspOutputItem::FileSearchCall { payload } => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: {
                            let mut v = payload.clone();
                            v["type"] = serde_json::json!("file_search_call");
                            v
                        },
                    });
                }
                RspOutputItem::ComputerCall { payload } => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: {
                            let mut v = payload.clone();
                            v["type"] = serde_json::json!("computer_call");
                            v
                        },
                    });
                }
                RspOutputItem::CodeInterpreterCall { payload } => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: {
                            let mut v = payload.clone();
                            v["type"] = serde_json::json!("code_interpreter_call");
                            v
                        },
                    });
                }
                RspOutputItem::ImageGenerationCall { payload } => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: {
                            let mut v = payload.clone();
                            v["type"] = serde_json::json!("image_generation_call");
                            v
                        },
                    });
                }
                RspOutputItem::Unknown(raw) => {
                    content_parts.push(IrContentPart::Opaque {
                        provider: Cow::Borrowed("openai_responses"),
                        payload: raw.clone(),
                    });
                }
            }
        }

        let content = match content_parts.len() {
            0 => IrContent::Text(Cow::Borrowed("")),
            1 => {
                if let IrContentPart::Text {
                    ref text,
                    citations: None,
                } = content_parts[0]
                {
                    IrContent::Text(text.clone())
                } else {
                    IrContent::Parts(content_parts)
                }
            }
            _ => IrContent::Parts(content_parts),
        };

        let finish_reason = rsp
            .status
            .as_deref()
            .and_then(Self::parse_status)
            .map(|fr| {
                if fr == IrFinishReason::Stop && !tool_calls.is_empty() {
                    IrFinishReason::ToolCalls
                } else {
                    fr
                }
            });

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

        let usage = rsp.usage.as_ref().map(Self::convert_usage);

        let mut provider_metadata = rsp
            .service_tier
            .as_ref()
            .map(|st| Box::new(serde_json::json!({ "service_tier": st })));

        // 提取无损保留的 content parts
        let raw: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
        let preserved_parts = super::extract_preserved(&raw);
        super::merge_preserved_into_metadata(&mut provider_metadata, preserved_parts);

        Ok(IrResponse {
            id: rsp.id,
            model: rsp.model,
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

/// Responses API 流式解码器 — 每条流一个实例。
/// 跟踪 function_call item 的 call_id/name（output_index → 信息），供
/// arguments.done 事件缺失 item 字段时兜底合成完整 ToolCallDone。
pub struct RspStreamDecoder {
    /// output_index → (IR 全局工具序号, call_id, name, args 累积)
    tools: Vec<(u32, u32, String, String, String)>,
    /// 流内是否出现过工具调用（tools 会在 done 时 remove，不能用 is_empty 判断）
    saw_tool_call: bool,
    next_tool_seq: u32,
    started: bool,
    finished: bool,
    choice_finished: bool,
    content_open: bool,
    reasoning_open: bool,
}

impl RspStreamDecoder {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            saw_tool_call: false,
            next_tool_seq: 0,
            started: false,
            finished: false,
            choice_finished: false,
            content_open: false,
            reasoning_open: false,
        }
    }

    fn close_open_blocks(&mut self) -> Vec<IrStreamEvent<'static>> {
        let mut events = Vec::new();
        if self.reasoning_open {
            self.reasoning_open = false;
            events.push(IrStreamEvent::ReasoningDone {
                index: 0,
                signature: None,
            });
        }
        if self.content_open {
            self.content_open = false;
            events.push(IrStreamEvent::ContentDone { index: 0 });
        }
        for (_, seq, id, name, arguments) in self.tools.drain(..) {
            events.push(IrStreamEvent::ToolCallDone {
                index: seq,
                choice_index: 0,
                id: Cow::Owned(id),
                name: Cow::Owned(name),
                arguments: Cow::Owned(arguments),
            });
        }
        events
    }

    fn finalize(&mut self) -> Vec<IrStreamEvent<'static>> {
        if self.finished {
            return Vec::new();
        }
        let mut events = self.close_open_blocks();
        if self.started && !self.choice_finished {
            self.choice_finished = true;
            events.push(IrStreamEvent::ChoiceFinish {
                index: 0,
                finish_reason: if self.saw_tool_call {
                    IrFinishReason::ToolCalls
                } else {
                    IrFinishReason::Stop
                },
            });
        }
        self.finished = true;
        events.push(IrStreamEvent::Done);
        events
    }
}

impl Default for RspStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeStream for RspStreamDecoder {
    fn decode_sse_data<'a>(
        &mut self,
        data: &'a [u8],
    ) -> Result<Vec<IrStreamEvent<'a>>, CodecError> {
        if self.finished {
            let duplicate_terminal = serde_json::from_slice::<serde_json::Value>(data)
                .ok()
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(|kind| kind.as_str())
                        .map(str::to_owned)
                })
                .is_some_and(|kind| {
                    matches!(
                        kind.as_str(),
                        "response.completed" | "response.incomplete" | "response.failed"
                    )
                });
            return if duplicate_terminal {
                Ok(Vec::new())
            } else {
                Err(CodecError::InvalidState(
                    "Responses stream received data after completion".to_string(),
                ))
            };
        }

        // 零拷贝快路径：翻译场景 99% 是 response.output_text.delta
        if let Ok(fast) = serde_json::from_slice::<RspTextDeltaFast<'a>>(data) {
            if fast.r#type.as_ref() == "response.output_text.delta" {
                if let Some(delta) = fast.delta {
                    if !delta.is_empty() {
                        self.started = true;
                        self.content_open = true;
                        return Ok(vec![IrStreamEvent::ContentDelta { index: 0, delta }]);
                    }
                }
                return Ok(Vec::new());
            }
        }

        // 慢路径：完整解析
        let evt: RspStreamEvent<'a> = serde_json::from_slice(data)?;
        let mut events = Vec::with_capacity(2);

        match evt.r#type.as_ref() {
            "response.created" => {
                self.started = true;
                if let Some(ref resp) = evt.response {
                    events.push(IrStreamEvent::Start {
                        id: resp.id.clone(),
                        model: resp.model.clone(),
                        usage: None,
                    });
                }
            }
            "response.output_item.added" => {
                // 检查是否为 function_call → ToolCallStart
                if let Some(ref item) = evt.item {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let idx = evt.output_index.unwrap_or(0);
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        self.saw_tool_call = true;
                        self.started = true;
                        let seq = self.next_tool_seq;
                        self.next_tool_seq += 1;
                        self.tools.push((
                            idx,
                            seq,
                            call_id.to_string(),
                            name.to_string(),
                            String::new(),
                        ));
                        events.push(IrStreamEvent::ToolCallStart {
                            index: seq,
                            choice_index: 0,
                            id: Cow::Owned(call_id.to_string()),
                            name: Cow::Owned(name.to_string()),
                        });
                    }
                }
            }
            "response.output_text.delta" => {
                // 通常由快路径处理；仅在快路径解析失败时到达此处
                if let Some(ref delta) = evt.delta {
                    if let Some(text) = delta.as_str() {
                        if !text.is_empty() {
                            self.started = true;
                            self.content_open = true;
                            events.push(IrStreamEvent::ContentDelta {
                                index: 0,
                                delta: Cow::Owned(text.to_string()),
                            });
                        }
                    }
                }
            }
            "response.output_text.done" => {
                if self.content_open {
                    self.content_open = false;
                    events.push(IrStreamEvent::ContentDone { index: 0 });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(ref delta) = evt.delta {
                    if let Some(args) = delta.as_str() {
                        let idx = evt.output_index.unwrap_or(0);
                        let seq = if let Some(t) = self.tools.iter_mut().find(|t| t.0 == idx) {
                            t.4.push_str(args);
                            t.1
                        } else {
                            let seq = self.next_tool_seq;
                            self.next_tool_seq += 1;
                            self.tools.push((
                                idx,
                                seq,
                                String::new(),
                                String::new(),
                                args.to_string(),
                            ));
                            seq
                        };
                        self.saw_tool_call = true;
                        self.started = true;
                        events.push(IrStreamEvent::ToolCallDelta {
                            index: seq,
                            choice_index: 0,
                            arguments_delta: Cow::Owned(args.to_string()),
                        });
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let idx = evt.output_index.unwrap_or(0);
                // 优先从 item 提取完整信息；缺失时用解码器累积状态兜底
                let tracked = self
                    .tools
                    .iter()
                    .position(|t| t.0 == idx)
                    .map(|pos| self.tools.remove(pos));
                let from_item = |key: &str| -> Option<String> {
                    evt.item
                        .as_ref()
                        .and_then(|i| i.get(key))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                };
                // 顶层 arguments 字段（标准事件形态）优先；delta 字符串兜底
                let top_args = evt.arguments.clone().or_else(|| {
                    evt.delta
                        .as_ref()
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string())
                });
                let (seq, call_id, name, arguments) = match tracked {
                    Some((_, seq, tid, tname, targs)) => (
                        seq,
                        from_item("call_id").unwrap_or(tid),
                        from_item("name").unwrap_or(tname),
                        from_item("arguments").or(top_args).unwrap_or(targs),
                    ),
                    None => {
                        let seq = self.next_tool_seq;
                        self.next_tool_seq += 1;
                        (
                            seq,
                            from_item("call_id").unwrap_or_default(),
                            from_item("name").unwrap_or_default(),
                            from_item("arguments").or(top_args).unwrap_or_default(),
                        )
                    }
                };
                events.push(IrStreamEvent::ToolCallDone {
                    index: seq,
                    choice_index: 0,
                    id: Cow::Owned(call_id),
                    name: Cow::Owned(name),
                    arguments: Cow::Owned(arguments),
                });
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(ref delta) = evt.delta {
                    if let Some(text) = delta.as_str() {
                        if !text.is_empty() {
                            self.started = true;
                            self.reasoning_open = true;
                            events.push(IrStreamEvent::ReasoningDelta {
                                index: 0,
                                delta: Cow::Owned(text.to_string()),
                            });
                        }
                    }
                }
            }
            "response.reasoning_summary_text.done" => {
                if self.reasoning_open {
                    self.reasoning_open = false;
                    events.push(IrStreamEvent::ReasoningDone {
                        index: 0,
                        signature: None,
                    });
                }
            }
            "response.completed" => {
                events.extend(self.close_open_blocks());
                let mut finish_reason = if self.saw_tool_call {
                    IrFinishReason::ToolCalls
                } else {
                    IrFinishReason::Stop
                };
                if let Some(ref resp) = evt.response {
                    // 完整 status 映射；tool_calls 非空时覆盖为 ToolCalls
                    if let Some(ref status) = resp.status {
                        if let Some(mut fr) = OpenAiResponsesShim::parse_status(status) {
                            if fr == IrFinishReason::Stop && self.saw_tool_call {
                                fr = IrFinishReason::ToolCalls;
                            }
                            finish_reason = fr;
                        }
                    }
                }
                if !self.choice_finished {
                    self.choice_finished = true;
                    events.push(IrStreamEvent::ChoiceFinish {
                        index: 0,
                        finish_reason,
                    });
                }
                if let Some(ref resp) = evt.response {
                    if let Some(ref u) = resp.usage {
                        events.push(IrStreamEvent::Usage(OpenAiResponsesShim::convert_usage(u)));
                    }
                }
                self.finished = true;
                events.push(IrStreamEvent::Done);
            }
            "response.incomplete" => {
                events.extend(self.close_open_blocks());
                events.push(IrStreamEvent::ChoiceFinish {
                    index: 0,
                    finish_reason: IrFinishReason::Length,
                });
                self.choice_finished = true;
                if let Some(ref resp) = evt.response {
                    if let Some(ref u) = resp.usage {
                        events.push(IrStreamEvent::Usage(OpenAiResponsesShim::convert_usage(u)));
                    }
                }
                self.finished = true;
                events.push(IrStreamEvent::Done);
            }
            "response.failed" => {
                events.extend(self.close_open_blocks());
                if let Some(ref resp) = evt.response {
                    if let Some(ref u) = resp.usage {
                        events.push(IrStreamEvent::Usage(OpenAiResponsesShim::convert_usage(u)));
                    }
                }
                events.push(IrStreamEvent::Error {
                    message: Cow::Borrowed("response failed"),
                });
                self.finished = true;
                events.push(IrStreamEvent::Done);
            }
            _ => {} // response.output_item.done, response.content_part.added 等
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<IrStreamEvent<'static>>, CodecError> {
        Ok(self.finalize())
    }
}

// ─── DecodeRequest ──────────────────────────────────────────────────

/// Responses API 请求 — 反序列化用
#[derive(Deserialize)]
struct RspRequestIn<'a> {
    #[serde(borrow)]
    model: Cow<'a, str>,
    #[serde(default)]
    input: serde_json::Value,
    #[serde(borrow, default)]
    instructions: Option<Cow<'a, str>>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<RspToolIn>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(borrow, default)]
    previous_response_id: Option<Cow<'a, str>>,
    #[serde(default)]
    text: Option<serde_json::Value>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    truncation: Option<serde_json::Value>,
    #[serde(borrow, default)]
    service_tier: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    user: Option<Cow<'a, str>>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    include: Option<serde_json::Value>,
    #[serde(default)]
    background: Option<bool>,
}

#[derive(Deserialize)]
struct RspToolIn {
    #[serde(default, rename = "type")]
    r#type: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
    /// 工具类型专有配置（vector_store_ids / server_url / container 等），
    /// flatten 收集除已命名字段外的全部字段
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl OpenAiResponsesShim {
    fn parse_rsp_tool_choice(v: &serde_json::Value) -> IrToolChoice<'static> {
        if let Some(s) = v.as_str() {
            match s {
                "auto" => IrToolChoice::Auto,
                "none" => IrToolChoice::None,
                "required" => IrToolChoice::Required,
                _ => IrToolChoice::Auto,
            }
        } else if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
            IrToolChoice::Specific {
                name: Cow::Owned(name.to_string()),
            }
        } else {
            IrToolChoice::Auto
        }
    }

    fn parse_rsp_input(input: &serde_json::Value) -> Vec<IrMessage<'static>> {
        let mut messages = Vec::new();

        // input 可以是字符串（快路径）或 item 数组
        if let Some(s) = input.as_str() {
            messages.push(IrMessage {
                role: Role::User,
                content: IrContent::Text(Cow::Owned(s.to_string())),
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                cache_control: None,
                refusal: None,
            });
            return messages;
        }

        let items = match input.as_array() {
            Some(arr) => arr,
            None => return messages,
        };

        let mut pending_tool_calls: Vec<IrToolCall<'static>> = Vec::new();

        for item in items {
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

            if item_type != "function_call" && !pending_tool_calls.is_empty() {
                messages.push(IrMessage {
                    role: Role::Assistant,
                    content: IrContent::Text(Cow::Owned(String::new())),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls: Some(std::mem::take(&mut pending_tool_calls)),
                    cache_control: None,
                    refusal: None,
                });
            }

            match item_type {
                "message" => {
                    let role_str = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let role = match role_str {
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "system" | "developer" => Role::System,
                        _ => Role::User,
                    };

                    let content = item.get("content");
                    let ir_content = if let Some(c) = content {
                        if let Some(s) = c.as_str() {
                            IrContent::Text(Cow::Owned(s.to_string()))
                        } else if let Some(arr) = c.as_array() {
                            let parts: Vec<IrContentPart<'static>> = arr
                                .iter()
                                .filter_map(|block| {
                                    let t = block.get("type")?.as_str()?;
                                    match t {
                                        "input_text" | "output_text" => {
                                            let text = block.get("text")?.as_str()?;
                                            Some(IrContentPart::Text {
                                                text: Cow::Owned(text.to_string()),
                                                citations: None,
                                            })
                                        }
                                        "input_image" => {
                                            let url = block.get("image_url")?.as_str()?;
                                            if let Some((mime, data)) = url
                                                .strip_prefix("data:")
                                                .and_then(|rest| rest.split_once(','))
                                                .map(|(meta, data)| {
                                                    (
                                                        meta.strip_suffix(";base64")
                                                            .unwrap_or(meta),
                                                        data,
                                                    )
                                                })
                                                .filter(|(mime, _)| mime.starts_with("image/"))
                                            {
                                                Some(IrContentPart::ImageBase64 {
                                                    media_type: Cow::Owned(mime.to_string()),
                                                    data: Cow::Owned(data.to_string()),
                                                })
                                            } else {
                                                let detail = block
                                                    .get("detail")
                                                    .and_then(|d| d.as_str())
                                                    .map(|d| Cow::Owned(d.to_string()));
                                                Some(IrContentPart::ImageUrl {
                                                    url: Cow::Owned(url.to_string()),
                                                    detail,
                                                })
                                            }
                                        }
                                        "input_audio" => {
                                            let ia = block.get("input_audio")?;
                                            let data = ia.get("data")?.as_str()?;
                                            let format = ia
                                                .get("format")
                                                .and_then(|f| f.as_str())
                                                .unwrap_or("wav");
                                            Some(IrContentPart::Audio {
                                                media_type: Cow::Owned(format!("audio/{format}")),
                                                data: Cow::Owned(data.to_string()),
                                            })
                                        }
                                        "input_file" => {
                                            if let Some(file_id) =
                                                block.get("file_id").and_then(|v| v.as_str())
                                            {
                                                Some(IrContentPart::FileRef {
                                                    file_id: Cow::Owned(file_id.to_string()),
                                                })
                                            } else if let Some(fd) =
                                                block.get("file_data").and_then(|v| v.as_str())
                                            {
                                                // data URL 形式 → Document
                                                let rest = fd.strip_prefix("data:")?;
                                                let (meta, data) = rest.split_once(',')?;
                                                let mime =
                                                    meta.strip_suffix(";base64").unwrap_or(meta);
                                                Some(IrContentPart::Document {
                                                    media_type: Cow::Owned(mime.to_string()),
                                                    data: Cow::Owned(data.to_string()),
                                                    filename: block
                                                        .get("filename")
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| Cow::Owned(s.to_string())),
                                                })
                                            } else {
                                                None
                                            }
                                        }
                                        // 未知类型 → 无损保留为 Opaque，避免静默丢弃
                                        _ => Some(IrContentPart::Opaque {
                                            provider: Cow::Borrowed("openai_responses"),
                                            payload: block.clone(),
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
                    } else {
                        IrContent::Text(Cow::Owned(String::new()))
                    };

                    messages.push(IrMessage {
                        role,
                        content: ir_content,
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls: None,
                        cache_control: None,
                        refusal: None,
                    });
                }
                "function_call" => {
                    let call_id = item.get("call_id").and_then(|c| c.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let arguments = item
                        .get("arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}");

                    pending_tool_calls.push(IrToolCall {
                        id: Cow::Owned(call_id.to_string()),
                        name: Cow::Owned(name.to_string()),
                        arguments: Cow::Owned(arguments.to_string()),
                    });
                }
                "function_call_output" => {
                    let call_id = item.get("call_id").and_then(|c| c.as_str()).unwrap_or("");
                    let output = item.get("output").and_then(|o| o.as_str()).unwrap_or("");
                    messages.push(IrMessage {
                        role: Role::Tool,
                        content: IrContent::Text(Cow::Owned(output.to_string())),
                        tool_call_id: Some(Cow::Owned(call_id.to_string())),
                        tool_name: None,
                        tool_calls: None,
                        cache_control: None,
                        refusal: None,
                    });
                }
                "reasoning" => {
                    // reasoning items — 附加到 assistant
                    if let Some(content_arr) = item.get("content").and_then(|c| c.as_array()) {
                        let parts: Vec<IrContentPart<'static>> = content_arr
                            .iter()
                            .filter_map(|block| {
                                let t = block.get("type")?.as_str()?;
                                if t == "reasoning_text" {
                                    let text = block.get("text")?.as_str()?;
                                    let sig = block
                                        .get("signature")
                                        .and_then(|s| s.as_str())
                                        .map(|s| Cow::Owned(s.to_string()));
                                    Some(IrContentPart::Reasoning {
                                        text: Cow::Owned(text.to_string()),
                                        signature: sig,
                                    })
                                } else if t == "redacted_reasoning" {
                                    let data = block.get("data")?.as_str()?;
                                    Some(IrContentPart::RedactedReasoning {
                                        data: Cow::Owned(data.to_string()),
                                    })
                                } else {
                                    Some(IrContentPart::Opaque {
                                        provider: Cow::Owned("openai_responses".to_string()),
                                        payload: block.clone(),
                                    })
                                }
                            })
                            .collect();
                        if !parts.is_empty() {
                            messages.push(IrMessage {
                                role: Role::Assistant,
                                content: IrContent::Parts(parts),
                                tool_call_id: None,
                                tool_name: None,
                                tool_calls: None,
                                cache_control: None,
                                refusal: None,
                            });
                        }
                    }
                }
                _ => {
                    messages.push(IrMessage {
                        role: Role::Assistant,
                        content: IrContent::Parts(vec![IrContentPart::Opaque {
                            provider: Cow::Owned("openai_responses".to_string()),
                            payload: item.clone(),
                        }]),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls: None,
                        cache_control: None,
                        refusal: None,
                    });
                }
            }
        }

        if !pending_tool_calls.is_empty() {
            messages.push(IrMessage {
                role: Role::Assistant,
                content: IrContent::Text(Cow::Owned(String::new())),
                tool_call_id: None,
                tool_name: None,
                tool_calls: Some(pending_tool_calls),
                cache_control: None,
                refusal: None,
            });
        }

        messages
    }
}

impl DecodeRequest for OpenAiResponsesShim {
    fn decode_request<'a>(&self, body: &'a [u8]) -> Result<IrRequest<'a>, CodecError> {
        let req: RspRequestIn<'a> = serde_json::from_slice(body)?;

        let mut messages = Vec::new();

        // instructions → system message
        if let Some(ref instructions) = req.instructions {
            messages.push(IrMessage {
                role: Role::System,
                content: IrContent::Text(Cow::Owned(instructions.to_string())),
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                cache_control: None,
                refusal: None,
            });
        }

        // input → messages
        messages.extend(Self::parse_rsp_input(&req.input));
        super::backfill_tool_names(&mut messages);

        let tools: Option<Vec<IrTool<'_>>> = req.tools.as_ref().map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    let tool_type = t
                        .r#type
                        .as_deref()
                        .map(Self::parse_tool_type)
                        .unwrap_or(IrToolType::Function);
                    // function 工具必须有 name；内置工具无 name 时用空串占位
                    let name = match &t.name {
                        Some(n) => Cow::Owned(n.clone()),
                        None if tool_type == IrToolType::Function => return None,
                        None => Cow::Owned(String::new()),
                    };
                    let extra = if t.extra.is_empty() {
                        None
                    } else {
                        Some(serde_json::Value::Object(t.extra.clone()))
                    };
                    Some(IrTool {
                        tool_type,
                        name,
                        description: t.description.clone().map(Cow::Owned),
                        parameters: t.parameters.clone().unwrap_or(serde_json::json!({})),
                        cache_control: None,
                        extra,
                    })
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(Self::parse_rsp_tool_choice);

        // text.format → response_format
        let response_format = req.text.as_ref().and_then(|t| {
            let format = t.get("format")?;
            let type_str = format.get("type")?.as_str()?;
            match type_str {
                "json_object" => Some(IrResponseFormat {
                    r#type: ResponseFormatType::JsonObject,
                    schema: None,
                    name: None,
                    strict: None,
                }),
                "json_schema" => Some(IrResponseFormat {
                    r#type: ResponseFormatType::JsonSchema,
                    schema: format.get("schema").cloned(),
                    name: format
                        .get("name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string()),
                    strict: format.get("strict").and_then(|s| s.as_bool()),
                }),
                _ => None,
            }
        });

        // reasoning → ReasoningConfig（effort 原样保留；low 是低推理而非关闭）
        let reasoning = req.reasoning.as_ref().map(|r| {
            let effort = r.get("effort").and_then(|e| e.as_str()).unwrap_or("medium");
            let mode = match effort {
                "high" => ReasoningMode::Enabled,
                _ => ReasoningMode::Auto,
            };
            ReasoningConfig {
                mode,
                budget_tokens: None,
                effort: Some(effort.to_string()),
            }
        });

        // truncation
        let truncation = req.truncation.as_ref().and_then(|t| {
            let type_str = t.get("type")?.as_str()?;
            Some(IrTruncation {
                r#type: match type_str {
                    "auto" => TruncationType::Auto,
                    _ => TruncationType::Disabled,
                },
            })
        });

        let metadata = if req.user.is_some() || req.service_tier.is_some() {
            Some(IrMetadata {
                user_id: req.user.as_ref().map(|u| Cow::Owned(u.to_string())),
                service_tier: req
                    .service_tier
                    .as_ref()
                    .map(|st| Cow::Owned(st.to_string())),
            })
        } else {
            None
        };

        // 提取无损保留的 content parts
        let raw: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
        let preserved_parts = super::extract_preserved(&raw);
        let mut provider_metadata: Option<Box<serde_json::Value>> = None;

        // include / background → provider_metadata 透传
        if req.include.is_some() || req.background.is_some() {
            let pm = provider_metadata.get_or_insert_with(|| Box::new(serde_json::json!({})));
            if let serde_json::Value::Object(map) = pm.as_mut() {
                if let Some(ref inc) = req.include {
                    map.insert("include".to_string(), inc.clone());
                }
                if let Some(bg) = req.background {
                    map.insert("background".to_string(), serde_json::json!(bg));
                }
            }
        }

        super::merge_preserved_into_metadata(&mut provider_metadata, preserved_parts);

        Ok(IrRequest {
            model: req.model,
            messages,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: None,
            max_tokens: req.max_output_tokens,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            stream: req.stream,
            store: req.store,
            modalities: None,
            tools,
            tool_choice,
            parallel_tool_calls: req.parallel_tool_calls,
            reasoning,
            response_format,
            previous_response_id: req.previous_response_id,
            truncation,
            metadata,
            provider_metadata,
            metadata_mode: MetadataMode::default(),
        })
    }
}

// ─── EncodeResponse ─────────────────────────────────────────────────

impl OpenAiResponsesShim {
    fn finish_reason_to_status(fr: IrFinishReason) -> &'static str {
        match fr {
            IrFinishReason::Stop
            | IrFinishReason::StopSequence
            | IrFinishReason::PauseTurn
            | IrFinishReason::ToolCalls => "completed",
            IrFinishReason::Length => "incomplete",
            // 安全拦截/审核/畸形调用 → failed，不得伪装成正常完成
            IrFinishReason::ContentFilter
            | IrFinishReason::Safety
            | IrFinishReason::Recitation
            | IrFinishReason::MalformedFunctionCall => "failed",
        }
    }
}

impl EncodeResponse for OpenAiResponsesShim {
    fn encode_response(&self, ir: &IrResponse<'_>) -> Result<Vec<u8>, CodecError> {
        let choice = ir.choices.first();
        let msg = choice.map(|c| &c.message);

        let mut output: Vec<serde_json::Value> = Vec::new();
        let mut preserved: Vec<serde_json::Value> = Vec::new();

        if let Some(m) = msg {
            // reasoning items
            let mut reasoning_blocks: Vec<serde_json::Value> = Vec::new();
            let mut text_blocks: Vec<serde_json::Value> = Vec::new();
            // content 中是否已有 FunctionCall 部件（交错源）；有则跳过 tool_calls 字段避免重复
            let mut found_fc_in_content = false;

            match &m.content {
                IrContent::Text(s) => {
                    if !s.is_empty() {
                        text_blocks.push(serde_json::json!({
                            "type": "output_text",
                            "text": s,
                        }));
                    }
                }
                IrContent::Parts(parts) => {
                    for p in parts {
                        match p {
                            IrContentPart::Text { text, citations } => {
                                let mut block = serde_json::json!({
                                    "type": "output_text",
                                    "text": text,
                                });
                                if let Some(ref cits) = citations {
                                    let ann_arr: Vec<serde_json::Value> = cits
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
                                    block["annotations"] = serde_json::Value::Array(ann_arr);
                                }
                                text_blocks.push(block);
                            }
                            IrContentPart::Reasoning { text, signature } => {
                                let mut block = serde_json::json!({
                                    "type": "reasoning_text",
                                    "text": text,
                                });
                                if let Some(sig) = signature {
                                    block["signature"] = serde_json::Value::String(sig.to_string());
                                }
                                reasoning_blocks.push(block);
                            }
                            IrContentPart::RedactedReasoning { data } => {
                                reasoning_blocks.push(serde_json::json!({
                                    "type": "redacted_reasoning",
                                    "data": data,
                                }));
                            }
                            IrContentPart::Opaque { payload, .. } => {
                                output.push(payload.clone());
                            }
                            IrContentPart::FunctionCall {
                                id,
                                name,
                                arguments,
                            } => {
                                output.push(serde_json::json!({
                                    "type": "function_call",
                                    "call_id": id,
                                    "name": name,
                                    "arguments": arguments,
                                }));
                                found_fc_in_content = true;
                            }
                            // 不支持的类型 → 无损保留
                            IrContentPart::ImageUrl { .. }
                            | IrContentPart::ImageBase64 { .. }
                            | IrContentPart::Audio { .. }
                            | IrContentPart::Video { .. }
                            | IrContentPart::Document { .. }
                            | IrContentPart::FileRef { .. }
                            | IrContentPart::FunctionResponse { .. } => {
                                if let Ok(v) = serde_json::to_value(p) {
                                    preserved.push(v);
                                }
                            }
                        }
                    }
                }
            }

            if !reasoning_blocks.is_empty() {
                output.push(serde_json::json!({
                    "type": "reasoning",
                    "content": reasoning_blocks,
                }));
            }

            if !text_blocks.is_empty() {
                output.push(serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": text_blocks,
                }));
            }

            // tool_calls → function_call items
            // 仅在 content 无 FunctionCall 部件时发射，避免与上面提取的交错工具调用重复
            if !found_fc_in_content {
                if let Some(ref tcs) = m.tool_calls {
                    for tc in tcs {
                        output.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }));
                    }
                }
            }
        }

        let status = choice
            .and_then(|c| c.finish_reason)
            .map(Self::finish_reason_to_status)
            .unwrap_or("completed");

        let mut resp = serde_json::json!({
            "id": ir.id,
            "object": "response",
            "model": ir.model,
            "output": output,
            "status": status,
        });

        if let Some(ref usage) = ir.usage {
            let input = usage.prompt_tokens;
            let output_tokens = usage.completion_tokens;
            let mut usage_json = serde_json::json!({
                "input_tokens": input,
                "output_tokens": output_tokens,
                "total_tokens": usage.total_tokens,
            });
            if let Some(cr) = usage.cache_read_tokens {
                usage_json["input_tokens_details"] = serde_json::json!({ "cached_tokens": cr });
            }
            if let Some(r) = usage.reasoning_tokens {
                usage_json["output_tokens_details"] = serde_json::json!({ "reasoning_tokens": r });
            }
            resp["usage"] = usage_json;
        }

        // service_tier from provider_metadata
        if let Some(ref pm) = ir.provider_metadata {
            if let Some(st) = pm.get("service_tier").and_then(|v| v.as_str()) {
                resp["service_tier"] = serde_json::Value::String(st.to_string());
            }
        }

        preserved.extend(super::collect_provider_preserved(&ir.provider_metadata));
        super::attach_preserved(&mut resp, preserved);
        serde_json::to_vec(&resp).map_err(CodecError::from)
    }
}

// ─── EncodeStream ───────────────────────────────────────────────────

/// Responses API 流式编码器 — 每条流一个实例。
///
/// Responses 协议要求：每个 output item 有 output_item.added → (delta...) →
/// output_item.done 生命周期，且终态 response.completed 携带完整 output 数组。
/// 上游 IR 流不提供这些结构，编码器有状态重建：
/// - 首个 ContentDelta/ReasoningDelta 前补发 output_item.added + content_part.added
/// - 累积全部文本/推理/工具内容，在 completed 时重建 output 数组
/// - ChoiceFinish/Usage 缓存，Done 时合并为单条权威 response.completed
pub struct RspStreamEncoder {
    id: String,
    model: String,
    pending_status: Option<&'static str>,
    pending_usage: Option<IrUsage>,
    completed_sent: bool,
    /// 下一个 output item 索引
    next_output_index: u32,
    /// 已打开的 message item（文本）: output_index
    message_item: Option<u32>,
    /// 已打开的 reasoning item: output_index
    reasoning_item: Option<u32>,
    /// 文本累积（completed 时重建 output）
    text_acc: String,
    /// 推理文本累积
    reasoning_acc: String,
    /// 推理签名（ReasoningDone 时获取）
    reasoning_signature: Option<String>,
    /// 已完成的 function_call items: (call_id, name, arguments)
    done_tools: Vec<(String, String, String)>,
    /// 进行中的 tool: IR tool index → output_index
    open_tools: Vec<(u32, u32)>,
    /// 无损保留事件（logprobs 等），completed 时附加到 response
    preserved_events: Vec<serde_json::Value>,
    /// redacted reasoning 数据累积
    redacted_data: Vec<String>,
    /// 已通过 output_item.done 关闭的 message/reasoning items（供 build_output 使用）
    done_output: Vec<serde_json::Value>,
}

impl RspStreamEncoder {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            pending_status: None,
            pending_usage: None,
            completed_sent: false,
            next_output_index: 0,
            message_item: None,
            reasoning_item: None,
            text_acc: String::new(),
            reasoning_acc: String::new(),
            reasoning_signature: None,
            redacted_data: Vec::new(),
            done_tools: Vec::new(),
            open_tools: Vec::new(),
            preserved_events: Vec::new(),
            done_output: Vec::new(),
        }
    }

    fn frame(evt: &serde_json::Value) -> Result<Vec<u8>, CodecError> {
        Ok(format!("data: {}\n\n", serde_json::to_string(evt)?).into_bytes())
    }

    fn alloc_index(&mut self) -> u32 {
        let i = self.next_output_index;
        self.next_output_index += 1;
        i
    }

    /// 确保 message item 已宣告，返回其 output_index
    fn ensure_message_item(&mut self, out: &mut Vec<u8>) -> Result<u32, CodecError> {
        if let Some(idx) = self.message_item {
            return Ok(idx);
        }
        let idx = self.alloc_index();
        self.message_item = Some(idx);
        out.extend(Self::frame(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": idx,
            "item": { "type": "message", "role": "assistant", "content": [] },
        }))?);
        out.extend(Self::frame(&serde_json::json!({
            "type": "response.content_part.added",
            "output_index": idx,
            "content_index": 0,
            "part": { "type": "output_text", "text": "" },
        }))?);
        Ok(idx)
    }

    fn ensure_reasoning_item(&mut self, out: &mut Vec<u8>) -> Result<u32, CodecError> {
        if let Some(idx) = self.reasoning_item {
            return Ok(idx);
        }
        let idx = self.alloc_index();
        self.reasoning_item = Some(idx);
        out.extend(Self::frame(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": idx,
            "item": { "type": "reasoning", "content": [] },
        }))?);
        Ok(idx)
    }

    /// 从累积状态重建 output 数组
    fn build_output(&self) -> Vec<serde_json::Value> {
        let mut output: Vec<serde_json::Value> = self.done_output.clone();
        // redacted reasoning（不通过 Done 生命周期，单独收集）
        if !self.redacted_data.is_empty() {
            let content: Vec<serde_json::Value> = self
                .redacted_data
                .iter()
                .map(|data| serde_json::json!({ "type": "redacted_reasoning", "data": data }))
                .collect();
            output.push(serde_json::json!({
                "type": "reasoning",
                "content": content,
            }));
        }
        // 未关闭的 reasoning/text（流中断场景）
        if !self.reasoning_acc.is_empty() {
            let mut block =
                serde_json::json!({ "type": "reasoning_text", "text": self.reasoning_acc });
            if let Some(ref sig) = self.reasoning_signature {
                block["signature"] = serde_json::Value::String(sig.clone());
            }
            output.push(serde_json::json!({
                "type": "reasoning",
                "content": [block],
            }));
        }
        if !self.text_acc.is_empty() {
            output.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": self.text_acc }],
            }));
        }
        for (call_id, name, arguments) in &self.done_tools {
            output.push(serde_json::json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }));
        }
        output
    }

    fn completed_frame(&mut self) -> Result<Vec<u8>, CodecError> {
        self.completed_sent = true;
        let status = self.pending_status.take().unwrap_or("completed");
        let mut response = serde_json::json!({
            "id": self.id,
            "object": "response",
            "model": self.model,
            "status": status,
            "output": self.build_output(),
        });
        if let Some(u) = self.pending_usage.take() {
            let mut usage = serde_json::json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            });
            if let Some(cr) = u.cache_read_tokens {
                usage["input_tokens_details"] = serde_json::json!({ "cached_tokens": cr });
            }
            if let Some(r) = u.reasoning_tokens {
                usage["output_tokens_details"] = serde_json::json!({ "reasoning_tokens": r });
            }
            response["usage"] = usage;
        }
        if !self.preserved_events.is_empty() {
            let preserved = std::mem::take(&mut self.preserved_events);
            super::attach_preserved(&mut response, preserved);
        }
        Self::frame(&serde_json::json!({
            "type": "response.completed",
            "response": response,
        }))
    }
}

impl Default for RspStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeStream for RspStreamEncoder {
    fn encode_sse_event(&mut self, event: &IrStreamEvent<'_>) -> Result<Vec<u8>, CodecError> {
        let mut out = Vec::new();
        match event {
            IrStreamEvent::Start { id, model, usage } => {
                self.id = id.to_string();
                self.model = model.to_string();
                self.pending_usage = usage.clone();
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.created",
                    "response": {
                        "id": id,
                        "object": "response",
                        "model": model,
                        "status": "in_progress",
                        "output": [],
                    }
                }))?);
            }
            IrStreamEvent::ContentDelta { delta, .. } => {
                let idx = self.ensure_message_item(&mut out)?;
                self.text_acc.push_str(delta);
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": idx,
                    "content_index": 0,
                    "delta": delta,
                }))?);
            }
            IrStreamEvent::ContentDone { .. } => {
                if let Some(idx) = self.message_item.take() {
                    let item = serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": self.text_acc }],
                    });
                    out.extend(Self::frame(&serde_json::json!({
                        "type": "response.output_text.done",
                        "output_index": idx,
                        "content_index": 0,
                        "text": self.text_acc,
                    }))?);
                    out.extend(Self::frame(&serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": idx,
                        "item": item,
                    }))?);
                    self.done_output.push(item);
                    self.text_acc.clear();
                }
            }
            IrStreamEvent::ReasoningDelta { delta, .. } => {
                let idx = self.ensure_reasoning_item(&mut out)?;
                self.reasoning_acc.push_str(delta);
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": idx,
                    "delta": delta,
                }))?);
            }
            IrStreamEvent::ReasoningDone { signature, .. } => {
                if let Some(ref sig) = signature {
                    self.reasoning_signature = Some(sig.to_string());
                }
                if let Some(idx) = self.reasoning_item.take() {
                    out.extend(Self::frame(&serde_json::json!({
                        "type": "response.reasoning_summary_text.done",
                        "output_index": idx,
                        "text": self.reasoning_acc,
                    }))?);
                    let mut reasoning_block = serde_json::json!({
                        "type": "reasoning_text",
                        "text": self.reasoning_acc,
                    });
                    if let Some(ref sig) = self.reasoning_signature {
                        reasoning_block["signature"] = serde_json::Value::String(sig.clone());
                    }
                    let item = serde_json::json!({
                        "type": "reasoning",
                        "content": [reasoning_block],
                    });
                    out.extend(Self::frame(&serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": idx,
                        "item": item,
                    }))?);
                    self.done_output.push(item);
                    self.reasoning_acc.clear();
                    self.reasoning_signature = None;
                }
            }
            IrStreamEvent::ToolCallStart {
                index, id, name, ..
            } => {
                let out_idx = self.alloc_index();
                self.open_tools.push((*index, out_idx));
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": out_idx,
                    "item": {
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": "",
                    }
                }))?);
            }
            IrStreamEvent::ToolCallDelta {
                index,
                arguments_delta,
                ..
            } => {
                let out_idx = self
                    .open_tools
                    .iter()
                    .find(|(i, _)| i == index)
                    .map(|(_, o)| *o)
                    .unwrap_or(*index);
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.function_call_arguments.delta",
                    "output_index": out_idx,
                    "delta": arguments_delta,
                }))?);
            }
            IrStreamEvent::ToolCallDone {
                index,
                id,
                name,
                arguments,
                ..
            } => {
                // 未宣告过的（上游一次性下发）→ 先补 added
                let out_idx = match self.open_tools.iter().position(|(i, _)| i == index) {
                    Some(pos) => self.open_tools.remove(pos).1,
                    None => {
                        let out_idx = self.alloc_index();
                        out.extend(Self::frame(&serde_json::json!({
                            "type": "response.output_item.added",
                            "output_index": out_idx,
                            "item": {
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": "",
                            }
                        }))?);
                        out_idx
                    }
                };
                self.done_tools
                    .push((id.to_string(), name.to_string(), arguments.to_string()));
                let item = serde_json::json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                });
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": out_idx,
                    "arguments": arguments,
                    "item": item,
                }))?);
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": item,
                }))?);
            }
            IrStreamEvent::RedactedReasoning { data, .. } => {
                self.redacted_data.push(data.to_string());
            }
            IrStreamEvent::ChoiceFinish { finish_reason, .. } => {
                self.pending_status =
                    Some(OpenAiResponsesShim::finish_reason_to_status(*finish_reason));
            }
            IrStreamEvent::Usage(usage) => {
                self.pending_usage = Some(usage.clone());
            }
            // Responses API 无 per-token logprobs；缓冲后于 completed 附加到 response
            IrStreamEvent::Logprobs { .. } => {
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
            }
            IrStreamEvent::Done => {
                if !self.completed_sent {
                    out.extend(self.completed_frame()?);
                }
            }
            IrStreamEvent::Citation { .. } | IrStreamEvent::OpaqueBlock { .. } => {
                if let Ok(val) = serde_json::to_value(event) {
                    self.preserved_events.push(val);
                }
            }
            IrStreamEvent::Error { message } => {
                self.completed_sent = true;
                let mut response = serde_json::json!({
                    "id": self.id,
                    "object": "response",
                    "model": self.model,
                    "status": "failed",
                });
                if !self.preserved_events.is_empty() {
                    let pe = std::mem::take(&mut self.preserved_events);
                    super::attach_preserved(&mut response, pe);
                }
                out.extend(Self::frame(&serde_json::json!({
                    "type": "response.failed",
                    "response": response,
                    "error": { "message": message },
                }))?);
            }
        }
        Ok(out)
    }
}
