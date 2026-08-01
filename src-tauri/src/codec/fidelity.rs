//! Auditable semantic-fidelity contract for cross-provider conversion.
//!
//! The contract covers fields represented by the OpenXlate IR. JSON whitespace,
//! object key order, and original SSE chunk boundaries are intentionally outside
//! the lossless scope and are reported as normalization.
//! "Near-lossless" means that a report contains no `Dropped` finding. Values
//! marked `Preserved` remain in an OpenXlate side channel and are not guaranteed
//! to be interpreted natively by the target provider.

use serde::{Deserialize, Serialize};

use super::error::CodecError;
use super::format::CodecFormat;
use super::ir::*;

pub const FIDELITY_CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionKind {
    Request,
    Response,
    Stream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityLevel {
    /// The modeled value is represented directly by the target protocol.
    Exact,
    /// Semantics are retained but wire representation or granularity changes.
    Normalized,
    /// The value is carried through `_openxlate_preserved` instead of natively.
    Preserved,
    /// The source protocol has no stable identity and the decoder creates one.
    Synthesized,
    /// The target path cannot retain the modeled value.
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityIssue {
    pub path: String,
    pub level: FidelityLevel,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityReport {
    pub contract_version: u32,
    pub source: CodecFormat,
    pub target: CodecFormat,
    pub kind: ConversionKind,
    pub issues: Vec<FidelityIssue>,
}

impl FidelityReport {
    pub fn is_near_lossless(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|issue| issue.level == FidelityLevel::Dropped)
    }

    pub fn dropped_paths(&self) -> Vec<String> {
        self.issues
            .iter()
            .filter(|issue| issue.level == FidelityLevel::Dropped)
            .map(|issue| issue.path.clone())
            .collect()
    }

    pub fn require_near_lossless(&self) -> Result<(), CodecError> {
        let paths = self.dropped_paths();
        if paths.is_empty() {
            Ok(())
        } else {
            Err(CodecError::LossyConversion {
                target: self.target.to_string(),
                paths,
            })
        }
    }

    fn new(source: CodecFormat, target: CodecFormat, kind: ConversionKind) -> Self {
        Self {
            contract_version: FIDELITY_CONTRACT_VERSION,
            source,
            target,
            kind,
            issues: vec![FidelityIssue {
                path: "$wire".to_string(),
                level: FidelityLevel::Normalized,
                detail: "JSON formatting and field order are normalized through the IR".to_string(),
            }],
        }
    }

    fn issue(&mut self, path: impl Into<String>, level: FidelityLevel, detail: &str) {
        let path = path.into();
        if self
            .issues
            .iter()
            .any(|issue| issue.path == path && issue.level == level && issue.detail == detail)
        {
            return;
        }
        self.issues.push(FidelityIssue {
            path,
            level,
            detail: detail.to_string(),
        });
    }

    fn dropped_if(&mut self, condition: bool, path: &str, detail: &str) {
        if condition {
            self.issue(path, FidelityLevel::Dropped, detail);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditedTranscode {
    pub body: Vec<u8>,
    pub fidelity: FidelityReport,
}

impl AuditedTranscode {
    pub fn into_body(self) -> Vec<u8> {
        self.body
    }
}

pub(crate) fn audit_request(
    source: CodecFormat,
    target: CodecFormat,
    request: &IrRequest<'_>,
) -> FidelityReport {
    let mut report = FidelityReport::new(source, target, ConversionKind::Request);

    if target != CodecFormat::Gemini {
        report.dropped_if(
            request.model.trim().is_empty(),
            "request.model",
            "the target protocol requires a model in the JSON body",
        );
    }

    match target {
        CodecFormat::OpenAi => {
            drop_request_fields(
                &mut report,
                [
                    (request.top_k.is_some(), "request.top_k"),
                    (
                        request.previous_response_id.is_some(),
                        "request.previous_response_id",
                    ),
                    (request.truncation.is_some(), "request.truncation"),
                ],
                "OpenAI Chat Completions has no native field",
            );
        }
        CodecFormat::OpenAiResponses => {
            drop_request_fields(
                &mut report,
                [
                    (request.top_k.is_some(), "request.top_k"),
                    (request.stop.is_some(), "request.stop"),
                    (
                        request.frequency_penalty.is_some(),
                        "request.frequency_penalty",
                    ),
                    (
                        request.presence_penalty.is_some(),
                        "request.presence_penalty",
                    ),
                    (request.seed.is_some(), "request.seed"),
                    (request.n.is_some(), "request.n"),
                    (request.logprobs.is_some(), "request.logprobs"),
                    (request.top_logprobs.is_some(), "request.top_logprobs"),
                    (request.modalities.is_some(), "request.modalities"),
                ],
                "OpenAI Responses has no native field in this codec",
            );
        }
        CodecFormat::Anthropic => {
            drop_request_fields(
                &mut report,
                [
                    (
                        request.frequency_penalty.is_some(),
                        "request.frequency_penalty",
                    ),
                    (
                        request.presence_penalty.is_some(),
                        "request.presence_penalty",
                    ),
                    (request.seed.is_some(), "request.seed"),
                    (request.n.is_some(), "request.n"),
                    (request.logprobs.is_some(), "request.logprobs"),
                    (request.top_logprobs.is_some(), "request.top_logprobs"),
                    (request.store.is_some(), "request.store"),
                    (request.modalities.is_some(), "request.modalities"),
                    (request.response_format.is_some(), "request.response_format"),
                    (
                        request.previous_response_id.is_some(),
                        "request.previous_response_id",
                    ),
                    (request.truncation.is_some(), "request.truncation"),
                ],
                "Anthropic Messages has no native field",
            );
            if request.max_tokens.is_none() {
                report.issue(
                    "request.max_tokens",
                    FidelityLevel::Synthesized,
                    "Anthropic requires max_tokens; the encoder uses 4096",
                );
            }
        }
        CodecFormat::Gemini => {
            drop_request_fields(
                &mut report,
                [
                    (!request.model.trim().is_empty(), "request.model"),
                    (request.stream, "request.stream"),
                    (request.top_logprobs.is_some(), "request.top_logprobs"),
                    (request.store.is_some(), "request.store"),
                    (request.modalities.is_some(), "request.modalities"),
                    (
                        request.parallel_tool_calls.is_some(),
                        "request.parallel_tool_calls",
                    ),
                    (
                        request.previous_response_id.is_some(),
                        "request.previous_response_id",
                    ),
                    (request.truncation.is_some(), "request.truncation"),
                    (request.metadata.is_some(), "request.metadata"),
                ],
                "Gemini generateContent has no equivalent field",
            );
        }
    }

    audit_reasoning(request, target, &mut report);
    audit_messages(
        &request.messages,
        source,
        target,
        ConversionKind::Request,
        request.metadata_mode,
        &mut report,
    );
    audit_tools(request, target, &mut report);

    audit_provider_metadata(
        request.provider_metadata.as_deref(),
        target,
        ConversionKind::Request,
        request.metadata_mode,
        &mut report,
    );

    report
}

pub(crate) fn audit_response(
    source: CodecFormat,
    target: CodecFormat,
    response: &IrResponse<'_>,
) -> FidelityReport {
    let mut report = FidelityReport::new(source, target, ConversionKind::Response);

    report.dropped_if(
        response.choices.len() > 1
            && matches!(
                target,
                CodecFormat::Anthropic | CodecFormat::OpenAiResponses
            ),
        "response.choices[1..]",
        "the target response encoder supports a single choice",
    );
    report.dropped_if(
        target != CodecFormat::Gemini && response.id.trim().is_empty(),
        "response.id",
        "the target response requires a stable identifier",
    );

    for (position, choice) in response.choices.iter().enumerate() {
        let prefix = format!("response.choices[{position}]");
        audit_finish_reason(choice.finish_reason, target, &prefix, &mut report);
        audit_response_logprobs(choice, target, &prefix, &mut report);
        audit_messages(
            std::slice::from_ref(&choice.message),
            source,
            target,
            ConversionKind::Response,
            MetadataMode::Preserve,
            &mut report,
        );
        report.dropped_if(
            choice.index != 0
                && matches!(
                    target,
                    CodecFormat::Anthropic | CodecFormat::OpenAiResponses
                ),
            &format!("{prefix}.index"),
            "the single-choice target cannot retain a non-zero choice index",
        );
    }

    if let Some(usage) = &response.usage {
        audit_usage(usage, target, "response.usage", &mut report);
    }
    audit_provider_metadata(
        response.provider_metadata.as_deref(),
        target,
        ConversionKind::Response,
        MetadataMode::Preserve,
        &mut report,
    );

    report
}

pub(crate) fn audit_stream(source: CodecFormat, target: CodecFormat) -> FidelityReport {
    let mut report = FidelityReport::new(source, target, ConversionKind::Stream);
    report.issue(
        "stream.chunk_boundaries",
        FidelityLevel::Normalized,
        "target framing and delta boundaries are reconstructed from IR events",
    );
    if source == CodecFormat::Gemini {
        report.issue(
            "stream.tool_call.id",
            FidelityLevel::Synthesized,
            "Gemini does not provide stable streaming function-call identifiers",
        );
    }
    if target == CodecFormat::Gemini {
        report.issue(
            "stream.tool_call.arguments_delta",
            FidelityLevel::Normalized,
            "Gemini requires complete function arguments, so deltas are buffered",
        );
    }
    report
}

pub(crate) fn audit_stream_event(report: &mut FidelityReport, event: &IrStreamEvent<'_>) {
    let target = report.target;
    if matches!(
        target,
        CodecFormat::Anthropic | CodecFormat::OpenAiResponses
    ) {
        let choice_index = match event {
            IrStreamEvent::ContentDelta { index, .. }
            | IrStreamEvent::ContentDone { index }
            | IrStreamEvent::ReasoningDelta { index, .. }
            | IrStreamEvent::ReasoningDone { index, .. }
            | IrStreamEvent::RedactedReasoning { index, .. }
            | IrStreamEvent::Logprobs { index, .. }
            | IrStreamEvent::ChoiceFinish { index, .. }
            | IrStreamEvent::Citation { index, .. }
            | IrStreamEvent::OpaqueBlock { index, .. } => Some(*index),
            IrStreamEvent::ToolCallStart { choice_index, .. }
            | IrStreamEvent::ToolCallDelta { choice_index, .. }
            | IrStreamEvent::ToolCallDone { choice_index, .. } => Some(*choice_index),
            IrStreamEvent::Start { .. }
            | IrStreamEvent::Usage(_)
            | IrStreamEvent::Done
            | IrStreamEvent::Error { .. } => None,
        };
        report.dropped_if(
            choice_index.is_some_and(|index| index != 0),
            "stream.choice_index",
            "the target stream supports one choice and would merge additional choices",
        );
    }

    match event {
        IrStreamEvent::Start {
            id, model, usage, ..
        } => {
            if target == CodecFormat::Gemini {
                report.dropped_if(
                    !id.is_empty(),
                    "stream.start.id",
                    "the Gemini stream encoder does not emit the source response ID",
                );
                report.dropped_if(
                    !model.is_empty(),
                    "stream.start.model",
                    "the Gemini stream encoder does not emit the source model version",
                );
            }
            if let Some(usage) = usage {
                audit_usage(usage, target, "stream.usage", report);
            }
        }
        IrStreamEvent::Usage(usage) => {
            audit_usage(usage, target, "stream.usage", report);
        }
        IrStreamEvent::ChoiceFinish { finish_reason, .. } => {
            audit_finish_reason(Some(*finish_reason), target, "stream", report);
        }
        IrStreamEvent::Logprobs { .. } if target != CodecFormat::OpenAi => report.issue(
            "stream.logprobs",
            FidelityLevel::Preserved,
            "per-token log probabilities use the stream preservation channel",
        ),
        IrStreamEvent::Citation { .. } if target != CodecFormat::Anthropic => report.issue(
            "stream.citation",
            FidelityLevel::Preserved,
            "citations use the stream preservation channel",
        ),
        IrStreamEvent::RedactedReasoning { .. }
            if !matches!(
                target,
                CodecFormat::Anthropic | CodecFormat::OpenAiResponses
            ) =>
        {
            report.issue(
                "stream.redacted_reasoning",
                FidelityLevel::Preserved,
                "redacted reasoning uses the stream preservation channel",
            );
        }
        IrStreamEvent::OpaqueBlock { .. } => report.issue(
            "stream.opaque_block",
            FidelityLevel::Preserved,
            "opaque blocks use the stream preservation channel",
        ),
        _ => {}
    }
}

fn drop_request_fields<const N: usize>(
    report: &mut FidelityReport,
    fields: [(bool, &'static str); N],
    detail: &str,
) {
    for (present, path) in fields {
        report.dropped_if(present, path, detail);
    }
}

fn audit_reasoning(request: &IrRequest<'_>, target: CodecFormat, report: &mut FidelityReport) {
    let Some(reasoning) = &request.reasoning else {
        return;
    };
    match target {
        CodecFormat::OpenAi | CodecFormat::OpenAiResponses => {
            if reasoning.budget_tokens.is_some() {
                report.issue(
                    "request.reasoning.budget_tokens",
                    FidelityLevel::Normalized,
                    "the token budget is mapped to a target effort level",
                );
            }
        }
        CodecFormat::Anthropic | CodecFormat::Gemini => {
            if reasoning.effort.is_some() {
                report.issue(
                    "request.reasoning.effort",
                    FidelityLevel::Normalized,
                    "the effort label is mapped to a target token budget",
                );
            }
        }
    }
}

fn audit_messages(
    messages: &[IrMessage<'_>],
    source: CodecFormat,
    target: CodecFormat,
    kind: ConversionKind,
    metadata_mode: MetadataMode,
    report: &mut FidelityReport,
) {
    for (message_index, message) in messages.iter().enumerate() {
        let prefix = format!("messages[{message_index}]");
        if matches!(
            target,
            CodecFormat::Anthropic | CodecFormat::Gemini | CodecFormat::OpenAiResponses
        ) && matches!(message.role, Role::System | Role::Developer)
        {
            report.issue(
                format!("{prefix}.role"),
                FidelityLevel::Normalized,
                "instruction messages are represented by the target instruction container",
            );
        }
        if matches!(target, CodecFormat::Anthropic | CodecFormat::Gemini)
            && message.role == Role::Tool
        {
            report.issue(
                format!("{prefix}.role"),
                FidelityLevel::Normalized,
                "tool results are represented as user-role content blocks",
            );
        }
        report.dropped_if(
            message.cache_control.is_some()
                && (kind == ConversionKind::Response || target != CodecFormat::Anthropic),
            &format!("{prefix}.cache_control"),
            "the target encoder does not emit message cache controls",
        );
        report.dropped_if(
            message.refusal.is_some() && target != CodecFormat::OpenAi,
            &format!("{prefix}.refusal"),
            "only OpenAI Chat Completions has a native refusal field in this codec",
        );
        if message.role == Role::Tool
            && message.tool_name.is_some()
            && target != CodecFormat::Gemini
        {
            report.issue(
                format!("{prefix}.tool_name"),
                FidelityLevel::Normalized,
                "the target associates tool results by call ID instead of function name",
            );
        }
        if target == CodecFormat::Gemini
            && (message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
                || contains_function_call(&message.content))
        {
            report.issue(
                format!("{prefix}.tool_call.id"),
                FidelityLevel::Synthesized,
                "Gemini associates function calls by name and does not retain call IDs",
            );
        }
        if source == CodecFormat::Gemini
            && (message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
                || contains_function_call(&message.content))
        {
            report.issue(
                format!("{prefix}.tool_call.id"),
                FidelityLevel::Synthesized,
                "the source Gemini call ID was synthesized during decode",
            );
        }
        audit_content(
            &message.content,
            &prefix,
            message.role,
            target,
            kind,
            metadata_mode,
            report,
        );
        if contains_opaque(&message.content) {
            report.issue(
                format!("{prefix}.content.opaque"),
                if metadata_mode == MetadataMode::Preserve {
                    FidelityLevel::Preserved
                } else {
                    FidelityLevel::Dropped
                },
                "opaque content uses the provider preservation channel",
            );
        }
    }
}

fn audit_content(
    content: &IrContent<'_>,
    prefix: &str,
    role: Role,
    target: CodecFormat,
    kind: ConversionKind,
    metadata_mode: MetadataMode,
    report: &mut FidelityReport,
) {
    let IrContent::Parts(parts) = content else {
        return;
    };
    for (part_index, part) in parts.iter().enumerate() {
        let path = format!("{prefix}.content[{part_index}]");
        if kind == ConversionKind::Request
            && matches!(role, Role::System | Role::Developer)
            && matches!(target, CodecFormat::OpenAiResponses | CodecFormat::Gemini)
        {
            match part {
                IrContentPart::Text {
                    citations: None, ..
                } => {}
                IrContentPart::Text {
                    citations: Some(_), ..
                } => report.issue(
                    format!("{path}.citations"),
                    FidelityLevel::Dropped,
                    "the target instruction container retains text but not citations",
                ),
                _ => report.issue(
                    path,
                    FidelityLevel::Dropped,
                    "the target instruction container retains only text content",
                ),
            }
            continue;
        }
        match part {
            IrContentPart::Text {
                citations: Some(_), ..
            } if kind == ConversionKind::Request && target != CodecFormat::Gemini => {
                report.issue(
                    format!("{path}.citations"),
                    FidelityLevel::Dropped,
                    "the target request encoder emits text but not citation annotations",
                );
            }
            IrContentPart::Text {
                citations: Some(_), ..
            } if target == CodecFormat::Gemini => {
                report.issue(
                    format!("{path}.citations"),
                    if metadata_mode == MetadataMode::Preserve {
                        FidelityLevel::Preserved
                    } else {
                        FidelityLevel::Dropped
                    },
                    "Gemini carries text citations through the preservation channel",
                );
            }
            IrContentPart::Document {
                filename: Some(_), ..
            } if target == CodecFormat::Gemini => report.issue(
                format!("{path}.filename"),
                if metadata_mode == MetadataMode::Preserve {
                    FidelityLevel::Preserved
                } else {
                    FidelityLevel::Dropped
                },
                "Gemini carries document filenames through the preservation channel",
            ),
            _ => {}
        }
    }
}

fn audit_tools(request: &IrRequest<'_>, target: CodecFormat, report: &mut FidelityReport) {
    let Some(tools) = &request.tools else {
        return;
    };
    for (index, tool) in tools.iter().enumerate() {
        let prefix = format!("request.tools[{index}]");
        report.dropped_if(
            tool.cache_control.is_some() && target != CodecFormat::Anthropic,
            &format!("{prefix}.cache_control"),
            "only the Anthropic request encoder emits tool cache controls",
        );
        report.dropped_if(
            target == CodecFormat::Gemini
                && tool.tool_type == IrToolType::Function
                && tool.extra.is_some(),
            &format!("{prefix}.extra"),
            "Gemini function declarations have no field for provider-specific tool options",
        );
    }
}

fn audit_provider_metadata(
    metadata: Option<&serde_json::Value>,
    target: CodecFormat,
    kind: ConversionKind,
    mode: MetadataMode,
    report: &mut FidelityReport,
) {
    let Some(metadata) = metadata else {
        return;
    };
    let prefix = match kind {
        ConversionKind::Request => "request.provider_metadata",
        ConversionKind::Response => "response.provider_metadata",
        ConversionKind::Stream => "stream.provider_metadata",
    };
    let Some(map) = metadata.as_object() else {
        report.issue(
            prefix,
            FidelityLevel::Dropped,
            "provider metadata must be an object to be mapped or preserved",
        );
        return;
    };
    for key in map.keys() {
        let path = format!("{prefix}.{key}");
        if mode == MetadataMode::Strip {
            report.issue(
                path,
                FidelityLevel::Dropped,
                "metadata stripping was requested",
            );
        } else if key == super::shim::PRESERVED_KEY {
            report.issue(
                path,
                FidelityLevel::Preserved,
                "the value is carried through the OpenXlate preservation channel",
            );
        } else if provider_metadata_key_is_native(target, kind, key) {
            report.issue(
                path,
                FidelityLevel::Normalized,
                "the provider metadata key is emitted by the target encoder",
            );
        } else {
            report.issue(
                path,
                FidelityLevel::Dropped,
                "the target encoder neither emits nor sidecar-preserves this metadata key",
            );
        }
    }
}

fn provider_metadata_key_is_native(target: CodecFormat, kind: ConversionKind, key: &str) -> bool {
    match (kind, target) {
        (ConversionKind::Request, CodecFormat::OpenAi) => {
            matches!(key, "prediction" | "audio")
        }
        (ConversionKind::Request, CodecFormat::OpenAiResponses) => {
            matches!(key, "include" | "background")
        }
        (ConversionKind::Request, CodecFormat::Anthropic) => false,
        (ConversionKind::Request, CodecFormat::Gemini) => {
            matches!(key, "safety_settings" | "cached_content")
        }
        (ConversionKind::Response, CodecFormat::OpenAi) => matches!(
            key,
            "created" | "system_fingerprint" | "service_tier" | "error" | "metadata"
        ),
        (ConversionKind::Response, CodecFormat::OpenAiResponses) => key == "service_tier",
        (ConversionKind::Response, CodecFormat::Anthropic) => {
            matches!(key, "service_tier" | "stop_sequence")
        }
        (ConversionKind::Response, CodecFormat::Gemini) => matches!(
            key,
            "safety_ratings"
                | "grounding_metadata"
                | "search_entry_point"
                | "citation_metadata"
                | "prompt_feedback"
                | "metadata"
        ),
        (ConversionKind::Stream, _) => false,
    }
}

fn audit_usage(usage: &IrUsage, target: CodecFormat, prefix: &str, report: &mut FidelityReport) {
    match target {
        CodecFormat::OpenAi => {
            drop_usage_fields(usage, report, prefix, ["cache_creation_tokens"]);
        }
        CodecFormat::OpenAiResponses => {
            drop_usage_fields(
                usage,
                report,
                prefix,
                [
                    "cache_creation_tokens",
                    "audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                ],
            );
        }
        CodecFormat::Anthropic => {
            drop_usage_fields(
                usage,
                report,
                prefix,
                [
                    "reasoning_tokens",
                    "audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                ],
            );
            report.dropped_if(
                usage.total_tokens != usage.prompt_tokens.saturating_add(usage.completion_tokens),
                &format!("{prefix}.total_tokens"),
                "Anthropic reconstructs total tokens as input plus output tokens",
            );
        }
        CodecFormat::Gemini => {
            drop_usage_fields(
                usage,
                report,
                prefix,
                [
                    "cache_creation_tokens",
                    "audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                ],
            );
        }
    }
}

fn drop_usage_fields<const N: usize>(
    usage: &IrUsage,
    report: &mut FidelityReport,
    prefix: &str,
    fields: [&str; N],
) {
    for field in fields {
        let present = match field {
            "cache_creation_tokens" => usage.cache_creation_tokens.is_some(),
            "reasoning_tokens" => usage.reasoning_tokens.is_some(),
            "audio_tokens" => usage.audio_tokens.is_some(),
            "accepted_prediction_tokens" => usage.accepted_prediction_tokens.is_some(),
            "rejected_prediction_tokens" => usage.rejected_prediction_tokens.is_some(),
            _ => false,
        };
        report.dropped_if(
            present,
            &format!("{prefix}.{field}"),
            "the target usage schema has no equivalent field in this codec",
        );
    }
}

fn contains_function_call(content: &IrContent<'_>) -> bool {
    matches!(
        content,
        IrContent::Parts(parts)
            if parts
                .iter()
                .any(|part| matches!(part, IrContentPart::FunctionCall { .. }))
    )
}

fn contains_opaque(content: &IrContent<'_>) -> bool {
    matches!(
        content,
        IrContent::Parts(parts)
            if parts
                .iter()
                .any(|part| matches!(part, IrContentPart::Opaque { .. }))
    )
}

fn audit_response_logprobs(
    choice: &IrChoice<'_>,
    target: CodecFormat,
    prefix: &str,
    report: &mut FidelityReport,
) {
    let Some(logprobs) = &choice.logprobs else {
        return;
    };
    match target {
        CodecFormat::OpenAi => {}
        CodecFormat::Gemini if logprobs.content.is_none() && logprobs.refusal.is_none() => {
            report.issue(
                format!("{prefix}.logprobs"),
                FidelityLevel::Normalized,
                "Gemini represents only the average log probability",
            );
        }
        CodecFormat::Gemini => report.issue(
            format!("{prefix}.logprobs.tokens"),
            FidelityLevel::Dropped,
            "Gemini has no per-token response logprobs representation",
        ),
        CodecFormat::Anthropic | CodecFormat::OpenAiResponses => report.issue(
            format!("{prefix}.logprobs"),
            FidelityLevel::Dropped,
            "the target response protocol has no static logprobs representation",
        ),
    }
}

fn audit_finish_reason(
    reason: Option<IrFinishReason>,
    target: CodecFormat,
    prefix: &str,
    report: &mut FidelityReport,
) {
    let Some(reason) = reason else {
        return;
    };
    let collapsed = match target {
        CodecFormat::OpenAi => matches!(
            reason,
            IrFinishReason::StopSequence
                | IrFinishReason::PauseTurn
                | IrFinishReason::Safety
                | IrFinishReason::Recitation
                | IrFinishReason::MalformedFunctionCall
        ),
        CodecFormat::OpenAiResponses => matches!(
            reason,
            IrFinishReason::StopSequence
                | IrFinishReason::PauseTurn
                | IrFinishReason::ToolCalls
                | IrFinishReason::ContentFilter
                | IrFinishReason::Safety
                | IrFinishReason::Recitation
                | IrFinishReason::MalformedFunctionCall
        ),
        CodecFormat::Anthropic => matches!(
            reason,
            IrFinishReason::Safety
                | IrFinishReason::Recitation
                | IrFinishReason::MalformedFunctionCall
        ),
        CodecFormat::Gemini => matches!(
            reason,
            IrFinishReason::ToolCalls
                | IrFinishReason::PauseTurn
                | IrFinishReason::StopSequence
                | IrFinishReason::ContentFilter
        ),
    };
    report.dropped_if(
        collapsed,
        &format!("{prefix}.finish_reason"),
        "the target collapses this finish reason into a broader category",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Codec;

    const BASIC_OPENAI_REQUEST: &[u8] = br#"{
        "model":"gpt-test",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":64
    }"#;

    #[test]
    fn common_text_request_has_no_unreported_drop() {
        let result = Codec::default()
            .transcode_request_near_lossless(
                CodecFormat::OpenAi,
                CodecFormat::Anthropic,
                BASIC_OPENAI_REQUEST,
            )
            .unwrap();

        assert!(result.fidelity.is_near_lossless());
        assert!(!result.body.is_empty());
        assert!(result
            .fidelity
            .issues
            .iter()
            .all(|issue| issue.level != FidelityLevel::Dropped));
    }

    #[test]
    fn unsupported_request_field_is_reported_and_strictly_rejected() {
        let body = br#"{
            "model":"gpt-test",
            "messages":[{"role":"user","content":"hello"}],
            "max_tokens":64,
            "seed":7
        }"#;
        let codec = Codec::default();
        let audited = codec
            .transcode_request_audited(CodecFormat::OpenAi, CodecFormat::Anthropic, body)
            .unwrap();

        assert!(!audited.fidelity.is_near_lossless());
        assert_eq!(audited.fidelity.dropped_paths(), ["request.seed"]);
        assert!(matches!(
            codec.transcode_request_near_lossless(
                CodecFormat::OpenAi,
                CodecFormat::Anthropic,
                body,
            ),
            Err(CodecError::LossyConversion { paths, .. }) if paths == ["request.seed"]
        ));
    }

    #[test]
    fn single_choice_target_rejects_multi_choice_response() {
        let body = br#"{
            "id":"chatcmpl-1",
            "model":"gpt-test",
            "choices":[
                {"index":0,"message":{"role":"assistant","content":"one"},"finish_reason":"stop"},
                {"index":1,"message":{"role":"assistant","content":"two"},"finish_reason":"stop"}
            ]
        }"#;
        let error = Codec::default()
            .transcode_response_near_lossless(CodecFormat::OpenAi, CodecFormat::Anthropic, body)
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"response.choices[1..]".to_string())
        ));
    }

    #[test]
    fn collapsed_finish_reason_is_not_silently_called_lossless() {
        let body = br#"{
            "id":"msg-1",
            "type":"message",
            "role":"assistant",
            "model":"claude-test",
            "content":[{"type":"text","text":"partial"}],
            "stop_reason":"pause_turn",
            "stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":1}
        }"#;
        let error = Codec::default()
            .transcode_response_near_lossless(CodecFormat::Anthropic, CodecFormat::OpenAi, body)
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths == ["response.choices[0].finish_reason"]
        ));
    }

    #[test]
    fn streaming_contract_reports_normalization_without_drops() {
        let transcoder = Codec::default()
            .stream_transcoder_near_lossless(CodecFormat::OpenAi, CodecFormat::Gemini)
            .unwrap();
        let report = transcoder.fidelity_report();

        assert!(report.is_near_lossless());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.path == "stream.chunk_boundaries"));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.path == "stream.tool_call.arguments_delta"));
    }

    #[test]
    fn message_level_cache_control_is_not_silently_dropped() {
        let body = br#"{
            "model":"claude-test",
            "max_tokens":64,
            "messages":[{
                "role":"user",
                "content":[{
                    "type":"text",
                    "text":"hello",
                    "cache_control":{"type":"ephemeral"}
                }]
            }]
        }"#;
        let error = Codec::default()
            .transcode_request_near_lossless(CodecFormat::Anthropic, CodecFormat::OpenAi, body)
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"messages[0].cache_control".to_string())
        ));
    }

    #[test]
    fn unknown_provider_metadata_is_not_claimed_as_preserved() {
        let body = br#"{
            "id":"chatcmpl-1",
            "created":123,
            "model":"gpt-test",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"hello"},
                "finish_reason":"stop"
            }]
        }"#;
        let error = Codec::default()
            .transcode_response_near_lossless(CodecFormat::OpenAi, CodecFormat::Anthropic, body)
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"response.provider_metadata.created".to_string())
        ));
    }

    #[test]
    fn usage_details_are_part_of_the_strict_contract() {
        let body = br#"{
            "id":"chatcmpl-1",
            "model":"gpt-test",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"hello"},
                "finish_reason":"stop"
            }],
            "usage":{
                "prompt_tokens":1,
                "completion_tokens":1,
                "total_tokens":2,
                "completion_tokens_details":{"audio_tokens":1}
            }
        }"#;
        let error = Codec::default()
            .transcode_response_near_lossless(CodecFormat::OpenAi, CodecFormat::Anthropic, body)
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"response.usage.audio_tokens".to_string())
        ));
    }

    #[test]
    fn strict_stream_rejects_a_late_lossy_event_before_encoding_it() {
        let mut transcoder = Codec::default()
            .stream_transcoder_near_lossless(CodecFormat::OpenAi, CodecFormat::Anthropic)
            .unwrap();
        let error = transcoder
            .transcode_data(
                br#"{"id":"c1","model":"m","choices":[{"index":1,"delta":{"content":"hi"}}]}"#,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"stream.choice_index".to_string())
        ));
        assert!(!transcoder.fidelity_report().is_near_lossless());
    }

    #[test]
    fn gemini_body_conversion_does_not_claim_to_retain_url_fields() {
        let error = Codec::default()
            .transcode_request_near_lossless(
                CodecFormat::OpenAi,
                CodecFormat::Gemini,
                BASIC_OPENAI_REQUEST,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"request.model".to_string())
        ));
    }

    #[test]
    fn instruction_only_targets_reject_non_text_instruction_parts() {
        let body = br#"{
            "model":"gpt-test",
            "messages":[{
                "role":"system",
                "content":[{
                    "type":"image_url",
                    "image_url":{"url":"https://example.invalid/image.png"}
                }]
            },{
                "role":"user",
                "content":"hello"
            }]
        }"#;
        let error = Codec::default()
            .transcode_request_near_lossless(
                CodecFormat::OpenAi,
                CodecFormat::OpenAiResponses,
                body,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"messages[0].content[0]".to_string())
        ));
    }

    #[test]
    fn gemini_stream_reports_start_identity_loss_when_observed() {
        let mut transcoder = Codec::default()
            .stream_transcoder_near_lossless(CodecFormat::OpenAi, CodecFormat::Gemini)
            .unwrap();
        let error = transcoder
            .transcode_data(
                br#"{"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            CodecError::LossyConversion { paths, .. }
                if paths.contains(&"stream.start.id".to_string())
                    && paths.contains(&"stream.start.model".to_string())
        ));
    }
}
