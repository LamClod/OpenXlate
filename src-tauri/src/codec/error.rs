//! 译码器错误类型

use std::fmt;

#[derive(Debug)]
pub enum CodecError {
    /// JSON 序列化/反序列化错误
    Json(serde_json::Error),
    /// 供应商 API 返回的错误
    Api { status: u16, body: String },
    /// 不支持的供应商或格式
    Unsupported(String),
    /// 必需字段缺失
    MissingField(&'static str),
    /// SSE 解析错误
    Sse(String),
    /// 输入在语法上可解析，但不满足 IR 或目标协议约束
    InvalidInput {
        context: &'static str,
        message: String,
    },
    /// 单次载荷或流缓冲区超过配置上限
    LimitExceeded {
        resource: &'static str,
        limit: usize,
        actual: usize,
    },
    /// 有状态流在终态后继续接收事件，或事件顺序不合法
    InvalidState(String),
    /// 严格保真模式检测到目标协议无法保留的 IR 字段
    LossyConversion { target: String, paths: Vec<String> },
    /// CLI / 网关边界上的 I/O 错误
    Io(std::io::Error),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "JSON 错误: {e}"),
            Self::Api { status, body } => write!(f, "API 错误 ({status}): {body}"),
            Self::Unsupported(msg) => write!(f, "不支持: {msg}"),
            Self::MissingField(field) => write!(f, "缺少字段: {field}"),
            Self::Sse(msg) => write!(f, "SSE 错误: {msg}"),
            Self::InvalidInput { context, message } => {
                write!(f, "无效输入 ({context}): {message}")
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(
                f,
                "资源超限 ({resource}): 上限 {limit} 字节，实际 {actual} 字节"
            ),
            Self::InvalidState(message) => write!(f, "无效流状态: {message}"),
            Self::LossyConversion { target, paths } => write!(
                f,
                "目标格式 {target} 无法无损表示字段: {}",
                paths.join(", ")
            ),
            Self::Io(error) => write!(f, "I/O 错误: {error}"),
        }
    }
}

impl std::error::Error for CodecError {}

impl From<serde_json::Error> for CodecError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<std::io::Error> for CodecError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
