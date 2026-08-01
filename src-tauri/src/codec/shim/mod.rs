//! 供应商 Shim —— IR ↔ 供应商格式双向转换
//!
//! ## 无状态 trait（请求/响应，一次一发）
//! EncodeRequest / DecodeResponse / DecodeRequest / EncodeResponse
//!
//! ## 有状态 trait（流式，每条流一个实例）
//! DecodeStream / EncodeStream 取 `&mut self`：解码器跨事件跟踪块生命周期，
//! 合成底层协议缺失的信号（Start / ReasoningDone / ToolCallDone）；编码器
//! 维护块索引分配、arguments 补发、stop_reason 与 usage 合并等目标协议要求。
//! 每条 SSE 流必须使用独立的 `*StreamDecoder` / `*StreamEncoder` 实例。

pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod openai_responses;

use crate::codec::error::CodecError;
use crate::codec::ir::*;

// ─── Outbound（客户端 → 上游供应商）────────────────────────────────

/// Outbound: IR → 供应商 HTTP 请求体（零拷贝，返回序列化字节）
pub trait EncodeRequest {
    fn encode_request(&self, ir: &IrRequest<'_>) -> Result<Vec<u8>, CodecError>;
    fn endpoint(&self, base_url: &str) -> String;
    fn headers(&self, api_key: &str) -> Vec<(&'static str, String)>;
}

/// Inbound: 供应商响应体 → IR（零拷贝反序列化）
pub trait DecodeResponse {
    fn decode_response<'a>(&self, body: &'a [u8]) -> Result<IrResponse<'a>, CodecError>;
}

/// 流式解码: 供应商 SSE chunk → IR 流事件（有状态，每条流一个实例）
///
/// Anthropic / OpenAI Chat Completions 的流式路径可以实现部分零拷贝
/// （serde 从 &[u8] 直接借用字符串字段）。合成事件（ToolCallDone 等）
/// 的字段来自解码器内部累积状态，为 Cow::Owned。
pub trait DecodeStream: Send {
    /// 解析单条 SSE data 行，返回零或多个 IR 事件
    fn decode_sse_data<'a>(&mut self, data: &'a [u8])
        -> Result<Vec<IrStreamEvent<'a>>, CodecError>;

    /// 上游连接结束时刷新内部状态。具体解码器应补齐尚未闭合的块；默认只发 Done。
    fn finish(&mut self) -> Result<Vec<IrStreamEvent<'static>>, CodecError> {
        Ok(vec![IrStreamEvent::Done])
    }
}

// ─── Inbound（下游客户端 → 网关）──────────────────────────────────

/// Inbound 请求解码：供应商格式请求体 → IR
/// 网关用：接收客户端的 OpenAI/Anthropic/Gemini 格式请求，解码为统一 IR
pub trait DecodeRequest {
    fn decode_request<'a>(&self, body: &'a [u8]) -> Result<IrRequest<'a>, CodecError>;
}

/// Outbound 响应编码：IR → 供应商格式响应体
/// 网关用：把从上游供应商解码得到的 IR 响应，编码为客户端期望的格式
pub trait EncodeResponse {
    fn encode_response(&self, ir: &IrResponse<'_>) -> Result<Vec<u8>, CodecError>;
}

/// 流式响应编码：IR 流事件 → 供应商格式 SSE data 行（有状态，每条流一个实例）
///
/// 一个 IR 事件可能产出零个、一个或多个目标格式 SSE 帧（例如 Anthropic
/// 编码器在切换块类型时需要先 content_block_stop 再 content_block_start）。
pub trait EncodeStream: Send {
    fn encode_sse_event(&mut self, event: &IrStreamEvent<'_>) -> Result<Vec<u8>, CodecError>;
}

// ─── 无损保留 ─────────────────────────────────────────────────────

pub const PRESERVED_KEY: &str = "_openxlate_preserved";

pub fn attach_preserved(json: &mut serde_json::Value, parts: Vec<serde_json::Value>) {
    if parts.is_empty() {
        return;
    }
    let metadata = json
        .as_object_mut()
        .expect("attach_preserved requires JSON object")
        .entry("metadata")
        .or_insert_with(|| serde_json::json!({}));
    if !metadata.is_object() {
        *metadata = serde_json::json!({});
    }
    metadata[PRESERVED_KEY] = serde_json::Value::Array(parts);
}

pub fn extract_preserved(json: &serde_json::Value) -> Vec<serde_json::Value> {
    if let Some(metadata) = json.get("metadata") {
        if let Some(val) = metadata.get(PRESERVED_KEY) {
            if let Some(arr) = val.as_array() {
                return arr.clone();
            }
            if let Some(s) = val.as_str() {
                if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(s) {
                    return arr;
                }
            }
        }
    }
    json.get(PRESERVED_KEY)
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

pub fn merge_preserved_into_metadata(
    existing: &mut Option<Box<serde_json::Value>>,
    preserved: Vec<serde_json::Value>,
) {
    if preserved.is_empty() {
        return;
    }
    let pm = existing.get_or_insert_with(|| Box::new(serde_json::json!({})));
    if let serde_json::Value::Object(map) = pm.as_mut() {
        map.insert(
            PRESERVED_KEY.to_string(),
            serde_json::Value::Array(preserved),
        );
    }
}

/// 从 provider_metadata 中提取先前 decode 存入的 _openxlate_preserved 条目，
/// 供 encode 路径合并到输出，避免多跳链路丢失侧信道数据。
pub fn collect_provider_preserved(pm: &Option<Box<serde_json::Value>>) -> Vec<serde_json::Value> {
    if let Some(ref pm) = pm {
        if let Some(arr) = pm.get(PRESERVED_KEY).and_then(|v| v.as_array()) {
            return arr.clone();
        }
    }
    Vec::new()
}

// ─── 公共工具 ─────────────────────────────────────────────────────

/// 为 Tool 消息回填 `tool_name`：从同请求内先前 assistant 消息的 tool_calls
/// 中按 `tool_call_id` 反查函数名。
///
/// OpenAI/Anthropic/Responses 的工具结果只携带 call id，而 Gemini 用函数名
/// 关联工具结果 — 缺失 tool_name 会使跨格式多轮工具对话断链。各 DecodeRequest
/// 在构建完 messages 后调用此函数。
pub fn backfill_tool_names(messages: &mut [IrMessage<'_>]) {
    // 收集 (id, name) 对 — 借用冲突规避：先复制成 owned
    let mut id_to_name: Vec<(String, String)> = messages
        .iter()
        .filter_map(|m| m.tool_calls.as_ref())
        .flatten()
        .map(|tc| (tc.id.to_string(), tc.name.to_string()))
        .collect();

    // M10: 交错源把工具调用放在 content 的 FunctionCall 部件里
    for m in messages.iter() {
        if let IrContent::Parts(ref parts) = m.content {
            for p in parts {
                if let IrContentPart::FunctionCall { id, name, .. } = p {
                    id_to_name.push((id.to_string(), name.to_string()));
                }
            }
        }
    }

    for msg in messages.iter_mut() {
        if msg.role == Role::Tool && msg.tool_name.is_none() {
            if let Some(ref id) = msg.tool_call_id {
                if let Some((_, name)) = id_to_name.iter().find(|(i, _)| i == id.as_ref()) {
                    msg.tool_name = Some(std::borrow::Cow::Owned(name.clone()));
                }
            }
        }
    }
}

/// Validate tool arguments before entering provider encoders that require an object.
/// Some legacy serialization branches retain an empty-object fallback for internal
/// construction, so every public Anthropic/Gemini encode path calls this first.
pub(crate) fn validate_tool_arguments(messages: &[IrMessage<'_>]) -> Result<(), CodecError> {
    for message in messages {
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                validate_tool_argument_object(&tool_call.arguments)?;
            }
        }
        if let IrContent::Parts(parts) = &message.content {
            for part in parts {
                if let IrContentPart::FunctionCall { arguments, .. } = part {
                    validate_tool_argument_object(arguments)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_tool_argument_object(arguments: &str) -> Result<(), CodecError> {
    let value: serde_json::Value =
        serde_json::from_str(arguments).map_err(|error| CodecError::InvalidInput {
            context: "tool_call.arguments",
            message: error.to_string(),
        })?;
    if !value.is_object() {
        return Err(CodecError::InvalidInput {
            context: "tool_call.arguments",
            message: "target provider requires a JSON object".to_string(),
        });
    }
    Ok(())
}
