//! # IR Core — 统一中间表示
//!
//! 参考 LLM-Rosetta 论文 (arXiv:2604.09360) 的 12 类内容部件 + 16 类流事件 schema。
//! 所有结构体使用 `Cow<'a, str>` 实现零拷贝反序列化。

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

// ─── 供应商 ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Openai,
    OpenaiResponses,
    Anthropic,
    Deepseek,
    Google,
}

// ─── 顶层请求 ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrRequest<'a> {
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    pub messages: Vec<IrMessage<'a>>,

    // ── 采样参数 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<Cow<'a, str>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    pub stream: bool,

    // ── 存储 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    // ── 输出模态 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Cow<'a, str>>>,

    // ── 工具 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<IrTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<IrToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // ── 推理/思考控制 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    // ── 结构化输出 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<IrResponseFormat>,

    // ── 状态续接 (Responses API) ──
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub previous_response_id: Option<Cow<'a, str>>,

    // ── Truncation (Responses API) ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<IrTruncation>,

    // ── 元数据 ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<IrMetadata<'a>>,

    /// 供应商专有字段，preserve 模式保留、strip 模式丢弃
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<Box<serde_json::Value>>,

    /// 元数据模式：Preserve 保留 provider_metadata，Strip 丢弃
    #[serde(default)]
    pub metadata_mode: MetadataMode,
}

// ─── 推理配置 ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub mode: ReasoningMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// OpenAI 系 effort 字符串（"minimal"/"low"/"medium"/"high"），原样保留以
    /// 保证 OpenAI↔Responses 往返不漂移；Anthropic/Gemini 编码时忽略（用 budget）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    Auto,
    Enabled,
    Disabled,
}

// ─── 结构化输出 ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrResponseFormat {
    pub r#type: ResponseFormatType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

// ─── 元数据 ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMetadata<'a> {
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub user_id: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub service_tier: Option<Cow<'a, str>>,
}

// ─── 元数据模式 ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    Preserve,
    Strip,
}

impl Default for MetadataMode {
    fn default() -> Self {
        Self::Preserve
    }
}

// ─── Truncation ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTruncation {
    pub r#type: TruncationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationType {
    Auto,
    Disabled,
}

// ─── 引文 ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrCitation<'a> {
    #[serde(borrow)]
    pub r#type: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub url: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub title: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub cited_text: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub encrypted_index: Option<Cow<'a, str>>,
}

// ─── 缓存控制 ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrCacheControl<'a> {
    pub r#type: CacheControlType,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub ttl: Option<Cow<'a, str>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    Ephemeral,
}

// ─── 顶层响应 ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrResponse<'a> {
    #[serde(borrow)]
    pub id: Cow<'a, str>,
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    #[serde(borrow)]
    pub choices: Vec<IrChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<IrUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<Box<serde_json::Value>>,
}

// ─── 消息 ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMessage<'a> {
    pub role: Role,
    #[serde(borrow)]
    pub content: IrContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub tool_call_id: Option<Cow<'a, str>>,
    /// Tool 消息对应的函数名。OpenAI/Anthropic 用 id 关联工具结果、Gemini 用
    /// 函数名关联 — 两者必须都保留，否则跨格式多轮工具对话断链。
    /// 为 None 时，Gemini 编码器从同请求内先前 assistant 消息的 tool_calls
    /// 中按 id 反查函数名。
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub tool_name: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub tool_calls: Option<Vec<IrToolCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<IrCacheControl<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub refusal: Option<Cow<'a, str>>,
}

// ─── 12 类内容部件（完整）──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IrContent<'a> {
    /// 纯文本快路径（翻译场景 99% 命中，零分配）
    #[serde(borrow)]
    Text(Cow<'a, str>),
    /// 多部件内容
    Parts(Vec<IrContentPart<'a>>),
}

impl<'a> IrContent<'a> {
    #[inline]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Parts(parts) if parts.len() == 1 => {
                if let IrContentPart::Text { text, .. } = &parts[0] {
                    Some(text)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// 收集所有纯文本段拼接
    pub fn text_concat(&self) -> String {
        match self {
            Self::Text(s) => s.to_string(),
            Self::Parts(parts) => parts
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
        }
    }
}

/// 内容部件 — 完整 12 类内容部件
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IrContentPart<'a> {
    // ── 1. 文本 ──
    Text {
        #[serde(borrow)]
        text: Cow<'a, str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<IrCitation<'a>>>,
    },
    // ── 2. 图片（URL 引用）──
    ImageUrl {
        #[serde(borrow)]
        url: Cow<'a, str>,
        #[serde(skip_serializing_if = "Option::is_none", borrow)]
        detail: Option<Cow<'a, str>>,
    },
    // ── 3. 图片（base64 内联）──
    ImageBase64 {
        #[serde(borrow)]
        media_type: Cow<'a, str>,
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    // ── 4. 音频 ──
    Audio {
        #[serde(borrow)]
        media_type: Cow<'a, str>,
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    // ── 5. 视频 ──
    Video {
        #[serde(borrow)]
        media_type: Cow<'a, str>,
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    // ── 6. 文件/文档 ──
    Document {
        #[serde(borrow)]
        media_type: Cow<'a, str>,
        #[serde(borrow)]
        data: Cow<'a, str>,
        #[serde(skip_serializing_if = "Option::is_none", borrow)]
        filename: Option<Cow<'a, str>>,
    },
    // ── 7. 文件引用 ──
    FileRef {
        #[serde(borrow)]
        file_id: Cow<'a, str>,
    },
    // ── 8. 推理/思考 ──
    Reasoning {
        #[serde(borrow)]
        text: Cow<'a, str>,
        #[serde(skip_serializing_if = "Option::is_none", borrow)]
        signature: Option<Cow<'a, str>>,
    },
    // ── 8b. 已加密的推理（Anthropic redacted_thinking）──
    RedactedReasoning {
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    // ── 9. 内联函数调用（保留交错顺序）──
    FunctionCall {
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        name: Cow<'a, str>,
        #[serde(borrow)]
        arguments: Cow<'a, str>,
    },
    // ── 9b. 内联函数响应 ──
    FunctionResponse {
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        name: Cow<'a, str>,
        response: serde_json::Value,
    },
    /// 供应商专有内容块，原样保留
    Opaque {
        #[serde(borrow)]
        provider: Cow<'a, str>,
        payload: serde_json::Value,
    },
}

// ─── 工具 ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrToolType {
    Function,
    WebSearch,
    CodeInterpreter,
    FileSearch,
    ComputerUse,
    TextEditor,
    Mcp,
}

impl Default for IrToolType {
    fn default() -> Self {
        Self::Function
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTool<'a> {
    #[serde(default)]
    pub tool_type: IrToolType,
    #[serde(borrow)]
    pub name: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub description: Option<Cow<'a, str>>,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<IrCacheControl<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IrToolChoice<'a> {
    Auto,
    None,
    Required,
    Specific {
        #[serde(borrow)]
        name: Cow<'a, str>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrToolCall<'a> {
    #[serde(borrow)]
    pub id: Cow<'a, str>,
    #[serde(borrow)]
    pub name: Cow<'a, str>,
    /// JSON 字符串形式的参数，延迟解析
    #[serde(borrow)]
    pub arguments: Cow<'a, str>,
}

// ─── Logprobs ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<IrTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<IrTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTokenLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<IrTopLogprob>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTopLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

// ─── Choice / FinishReason ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrChoice<'a> {
    pub index: u32,
    #[serde(borrow)]
    pub message: IrMessage<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<IrFinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<IrLogprobs>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    /// Anthropic: pause_turn
    PauseTurn,
    /// Anthropic: stop_sequence（具体值通过 provider_metadata.stop_sequence 保留）
    StopSequence,
    /// Gemini: MALFORMED_FUNCTION_CALL
    MalformedFunctionCall,
    /// Gemini: RECITATION
    Recitation,
    /// Gemini: SAFETY / BLOCKLIST / PROHIBITED_CONTENT / SPII
    Safety,
}

// ─── Usage ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IrUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

// ─── 流式事件 IR（完整 16 类流事件 schema）─────────────────────
//
// ## index 语义规范（解码器必须遵守、编码器可以依赖）
//
// - ContentDelta / ContentDone / ReasoningDelta / ReasoningDone /
//   RedactedReasoning / ChoiceFinish 的 `index` 是 **choice/candidate 索引**
//   （单候选供应商如 Anthropic 恒为 0）。内容块的细分结构由编码器按目标
//   协议自行重建，不通过 index 传递。
// - ToolCallStart / ToolCallDelta / ToolCallDone 的 `index` 是 **全流唯一的
//   工具序号**（0 起递增），与 choice 索引是独立编号空间。`choice_index`
//   记录该工具事件所属的 choice/candidate（单候选供应商恒为 0）。
//
// ## Usage 语义
//
// Usage 事件携带 **累计总量**（非增量）。上游可能多次发出（如 Gemini 每
// chunk 回传累计 usageMetadata），后到的事件取代先到的。只发一次 usage 的
// 目标协议（OpenAI 尾部 chunk / Anthropic message_delta / Responses
// completed）的编码器必须缓存最新值、在终态合并发出，不得逐条转发。

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IrStreamEvent<'a> {
    /// 流开始
    Start {
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        model: Cow<'a, str>,
        /// 初始 usage（Anthropic message_start 携带 input_tokens；其余供应商为 None）
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<IrUsage>,
    },
    /// 文本 delta
    ContentDelta {
        index: u32,
        #[serde(borrow)]
        delta: Cow<'a, str>,
    },
    /// 文本块完成
    ContentDone { index: u32 },
    /// reasoning/thinking delta
    ReasoningDelta {
        index: u32,
        #[serde(borrow)]
        delta: Cow<'a, str>,
    },
    /// reasoning 块完成
    ReasoningDone {
        index: u32,
        /// 思考签名（Anthropic signature_delta / Gemini thoughtSignature），
        /// 在块完成时到达，用于跨供应商流式保留签名
        #[serde(skip_serializing_if = "Option::is_none", borrow)]
        signature: Option<Cow<'a, str>>,
    },
    /// redacted reasoning 块（Anthropic redacted_thinking，数据在块生命周期内一次性到达）
    RedactedReasoning {
        index: u32,
        #[serde(borrow)]
        data: Cow<'a, str>,
    },
    /// tool call 开始（携带 id + name）
    ToolCallStart {
        index: u32,
        /// 所属 choice/candidate 索引（单候选供应商恒为 0）
        #[serde(default)]
        choice_index: u32,
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        name: Cow<'a, str>,
    },
    /// tool call arguments delta
    ToolCallDelta {
        index: u32,
        #[serde(default)]
        choice_index: u32,
        #[serde(borrow)]
        arguments_delta: Cow<'a, str>,
    },
    /// tool call 完成
    ToolCallDone {
        index: u32,
        #[serde(default)]
        choice_index: u32,
        #[serde(borrow)]
        id: Cow<'a, str>,
        #[serde(borrow)]
        name: Cow<'a, str>,
        #[serde(borrow)]
        arguments: Cow<'a, str>,
    },
    /// 流式 logprobs（per-token，伴随 ContentDelta 到达）
    Logprobs { index: u32, logprobs: IrLogprobs },
    /// 单个 choice 完成
    ChoiceFinish {
        index: u32,
        finish_reason: IrFinishReason,
    },
    /// usage 更新
    Usage(IrUsage),
    /// 流结束
    Done,
    /// 流式引文（Anthropic citations_delta）
    Citation {
        index: u32,
        citation: serde_json::Value,
    },
    /// 不透明内容块（目标格式无原生流式表示的块，如 Anthropic server_tool_use）
    OpaqueBlock {
        index: u32,
        #[serde(borrow)]
        provider: Cow<'a, str>,
        payload: serde_json::Value,
    },
    /// 错误
    Error {
        #[serde(borrow)]
        message: Cow<'a, str>,
    },
}
