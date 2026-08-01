//! # 译码器 (Codec)
//!
//! LLM API 格式转译引擎，参考 LLM-Rosetta 的 hub-and-spoke IR 设计，
//! 用 Rust 零拷贝重写。所有供应商格式经由统一 IR 互转。
//!
//! ## 架构
//!
//! ```text
//! OpenAI ──┐              ┌── OpenAI
//! Anthropic ┤→  IR Core  ←├── Anthropic
//! DeepSeek ──┘              └── DeepSeek
//! ```

pub mod error;
pub mod fidelity;
pub mod format;
pub mod ir;
pub mod shim;
pub mod sse;

pub use error::CodecError;
pub use fidelity::{
    AuditedTranscode, ConversionKind, FidelityIssue, FidelityLevel, FidelityReport,
    FIDELITY_CONTRACT_VERSION,
};
pub use format::{Codec, CodecFormat, CodecLimits, SseStreamTranscoder, StreamTranscoder};
pub use ir::*;
