use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum XlateError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),
    #[error("provider transport error: {0}")]
    Transport(String),
    #[error("provider returned an error: status={status:?} message={message}")]
    Provider { status: Option<u16>, message: String },
    #[error("stream idle timeout after {0}ms without effective content")]
    IdleTimeout(u64),
    #[error("stream canceled")]
    Canceled,
    #[error("internal kernel error: {0}")]
    Internal(String),
    #[error("stream backpressure: kernel overloaded")]
    Overloaded,
    #[error("rate limited: {0}")]
    RateLimited(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XlateErrorPayload {
    pub kind: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
}

impl From<&XlateError> for XlateErrorPayload {
    fn from(err: &XlateError) -> Self {
        let (kind, http_status) = match err {
            XlateError::InvalidRequest(_) => ("invalid_request", None),
            XlateError::UnsupportedProvider(_) => ("unsupported_provider", None),
            XlateError::Transport(_) => ("transport", None),
            XlateError::Provider { status, .. } => ("provider", *status),
            XlateError::IdleTimeout(_) => ("idle_timeout", None),
            XlateError::Canceled => ("canceled", None),
            XlateError::Internal(_) => ("internal", None),
            XlateError::Overloaded => ("overloaded", None),
            XlateError::RateLimited(_) => ("rate_limited", None),
        };
        Self {
            kind: kind.to_string(),
            message: err.to_string(),
            http_status,
        }
    }
}
