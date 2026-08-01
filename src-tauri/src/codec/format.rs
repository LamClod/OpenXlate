//! 稳定的格式转换门面。
//!
//! Provider shim 保持专注于线格式映射；本模块统一负责格式选择、资源限制、
//! IR 语义校验和跨格式转换，供桌面应用、CLI 与未来网关复用。

use std::fmt;
use std::str::FromStr;

use super::error::CodecError;
use super::fidelity::{
    audit_request, audit_response, audit_stream, audit_stream_event, AuditedTranscode,
    FidelityReport,
};
use super::ir::*;
use super::shim::anthropic::{AntStreamDecoder, AntStreamEncoder, AnthropicShim};
use super::shim::gemini::{GemStreamDecoder, GemStreamEncoder, GeminiShim};
use super::shim::openai::{OaiStreamDecoder, OaiStreamEncoder, OpenAiShim};
use super::shim::openai_responses::{OpenAiResponsesShim, RspStreamDecoder, RspStreamEncoder};
use super::shim::{
    DecodeRequest, DecodeResponse, DecodeStream, EncodeRequest, EncodeResponse, EncodeStream,
};
use super::sse::SseParser;

/// Codec 支持的线格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CodecFormat {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
}

impl CodecFormat {
    pub const ALL: [Self; 4] = [
        Self::OpenAi,
        Self::OpenAiResponses,
        Self::Anthropic,
        Self::Gemini,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenAiResponses => "responses",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }

    pub fn endpoint(self, base_url: &str) -> String {
        match self {
            Self::OpenAi => OpenAiShim.endpoint(base_url),
            Self::OpenAiResponses => OpenAiResponsesShim.endpoint(base_url),
            Self::Anthropic => AnthropicShim.endpoint(base_url),
            Self::Gemini => GeminiShim.endpoint(base_url),
        }
    }

    pub fn headers(self, api_key: &str) -> Vec<(&'static str, String)> {
        match self {
            Self::OpenAi => OpenAiShim.headers(api_key),
            Self::OpenAiResponses => OpenAiResponsesShim.headers(api_key),
            Self::Anthropic => AnthropicShim.headers(api_key),
            Self::Gemini => GeminiShim.headers(api_key),
        }
    }

    pub fn stream_decoder(self) -> Box<dyn DecodeStream> {
        match self {
            Self::OpenAi => Box::new(OaiStreamDecoder::new()),
            Self::OpenAiResponses => Box::new(RspStreamDecoder::new()),
            Self::Anthropic => Box::new(AntStreamDecoder::new()),
            Self::Gemini => Box::new(GemStreamDecoder::new()),
        }
    }

    pub fn stream_encoder(self) -> Box<dyn EncodeStream> {
        match self {
            Self::OpenAi => Box::new(OaiStreamEncoder::new()),
            Self::OpenAiResponses => Box::new(RspStreamEncoder::new()),
            Self::Anthropic => Box::new(AntStreamEncoder::new()),
            Self::Gemini => Box::new(GemStreamEncoder::new()),
        }
    }
}

impl fmt::Display for CodecFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for CodecFormat {
    type Err = CodecError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" | "chat-completions" | "chat_completions" => Ok(Self::OpenAi),
            "responses" | "openai-responses" | "openai_responses" => Ok(Self::OpenAiResponses),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "gemini" | "google" => Ok(Self::Gemini),
            other => Err(CodecError::Unsupported(format!(
                "未知格式: {other}; 支持 openai, responses, anthropic, gemini"
            ))),
        }
    }
}

/// 对不可信载荷的默认资源上限。
#[derive(Debug, Clone)]
pub struct CodecLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_stream_event_bytes: usize,
}

impl Default for CodecLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 16 * 1024 * 1024,
            max_output_bytes: 32 * 1024 * 1024,
            max_stream_event_bytes: 2 * 1024 * 1024,
        }
    }
}

/// 可复用、无内部可变状态的请求/响应转换器。
#[derive(Debug, Clone, Default)]
pub struct Codec {
    limits: CodecLimits,
}

impl Codec {
    pub fn new(limits: CodecLimits) -> Result<Self, CodecError> {
        if limits.max_input_bytes == 0
            || limits.max_output_bytes == 0
            || limits.max_stream_event_bytes == 0
        {
            return Err(CodecError::InvalidInput {
                context: "codec limits",
                message: "所有资源上限必须大于 0".to_string(),
            });
        }
        Ok(Self { limits })
    }

    pub fn limits(&self) -> &CodecLimits {
        &self.limits
    }

    pub fn decode_request<'a>(
        &self,
        format: CodecFormat,
        body: &'a [u8],
    ) -> Result<IrRequest<'a>, CodecError> {
        self.check_input("request body", body)?;
        let request = match format {
            CodecFormat::OpenAi => OpenAiShim.decode_request(body),
            CodecFormat::OpenAiResponses => OpenAiResponsesShim.decode_request(body),
            CodecFormat::Anthropic => AnthropicShim.decode_request(body),
            CodecFormat::Gemini => GeminiShim.decode_request(body),
        }?;
        validate_request(&request)?;
        Ok(request)
    }

    pub fn encode_request(
        &self,
        format: CodecFormat,
        request: &IrRequest<'_>,
    ) -> Result<Vec<u8>, CodecError> {
        validate_request(request)?;
        validate_target_arguments_in_messages(&request.messages, format)?;
        let output = match format {
            CodecFormat::OpenAi => OpenAiShim.encode_request(request),
            CodecFormat::OpenAiResponses => OpenAiResponsesShim.encode_request(request),
            CodecFormat::Anthropic => AnthropicShim.encode_request(request),
            CodecFormat::Gemini => GeminiShim.encode_request(request),
        }?;
        self.check_output(output)
    }

    pub fn transcode_request(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<Vec<u8>, CodecError> {
        Ok(self.transcode_request_audited(source, target, body)?.body)
    }

    /// Audit a decoded request before encoding it for a target protocol.
    pub fn request_fidelity(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        request: &IrRequest<'_>,
    ) -> FidelityReport {
        audit_request(source, target, request)
    }

    /// Convert a request and return a field-level fidelity report.
    pub fn transcode_request_audited(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<AuditedTranscode, CodecError> {
        let request = self.decode_request(source, body)?;
        let fidelity = audit_request(source, target, &request);
        let body = self.encode_request(target, &request)?;
        Ok(AuditedTranscode { body, fidelity })
    }

    /// Convert only when every modeled IR field is exact, normalized,
    /// synthesized, or carried by the preservation channel.
    pub fn transcode_request_near_lossless(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<AuditedTranscode, CodecError> {
        let request = self.decode_request(source, body)?;
        let fidelity = audit_request(source, target, &request);
        fidelity.require_near_lossless()?;
        let body = self.encode_request(target, &request)?;
        Ok(AuditedTranscode { body, fidelity })
    }

    pub fn decode_response<'a>(
        &self,
        format: CodecFormat,
        body: &'a [u8],
    ) -> Result<IrResponse<'a>, CodecError> {
        self.check_input("response body", body)?;
        let response = match format {
            CodecFormat::OpenAi => OpenAiShim.decode_response(body),
            CodecFormat::OpenAiResponses => OpenAiResponsesShim.decode_response(body),
            CodecFormat::Anthropic => AnthropicShim.decode_response(body),
            CodecFormat::Gemini => GeminiShim.decode_response(body),
        }?;
        validate_response(&response)?;
        Ok(response)
    }

    pub fn encode_response(
        &self,
        format: CodecFormat,
        response: &IrResponse<'_>,
    ) -> Result<Vec<u8>, CodecError> {
        validate_response(response)?;
        for choice in &response.choices {
            validate_target_arguments_in_message(&choice.message, format)?;
        }
        let output = match format {
            CodecFormat::OpenAi => OpenAiShim.encode_response(response),
            CodecFormat::OpenAiResponses => OpenAiResponsesShim.encode_response(response),
            CodecFormat::Anthropic => AnthropicShim.encode_response(response),
            CodecFormat::Gemini => GeminiShim.encode_response(response),
        }?;
        self.check_output(output)
    }

    pub fn transcode_response(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<Vec<u8>, CodecError> {
        Ok(self.transcode_response_audited(source, target, body)?.body)
    }

    /// Audit a decoded response before encoding it for a target protocol.
    pub fn response_fidelity(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        response: &IrResponse<'_>,
    ) -> FidelityReport {
        audit_response(source, target, response)
    }

    /// Return the baseline stream contract. Event-specific findings are added
    /// to the transcoder's report as stream data arrives.
    pub fn stream_fidelity(&self, source: CodecFormat, target: CodecFormat) -> FidelityReport {
        audit_stream(source, target)
    }

    /// Convert a response and return a field-level fidelity report.
    pub fn transcode_response_audited(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<AuditedTranscode, CodecError> {
        let response = self.decode_response(source, body)?;
        let fidelity = audit_response(source, target, &response);
        let body = self.encode_response(target, &response)?;
        Ok(AuditedTranscode { body, fidelity })
    }

    /// Convert a response only when the target retains all modeled semantics.
    pub fn transcode_response_near_lossless(
        &self,
        source: CodecFormat,
        target: CodecFormat,
        body: &[u8],
    ) -> Result<AuditedTranscode, CodecError> {
        let response = self.decode_response(source, body)?;
        let fidelity = audit_response(source, target, &response);
        fidelity.require_near_lossless()?;
        let body = self.encode_response(target, &response)?;
        Ok(AuditedTranscode { body, fidelity })
    }

    pub fn stream_transcoder(&self, source: CodecFormat, target: CodecFormat) -> StreamTranscoder {
        StreamTranscoder {
            decoder: source.stream_decoder(),
            encoder: target.stream_encoder(),
            max_event_bytes: self.limits.max_stream_event_bytes,
            max_output_bytes: self.limits.max_output_bytes,
            terminal: false,
            fidelity: audit_stream(source, target),
            strict_fidelity: false,
        }
    }

    pub fn stream_transcoder_near_lossless(
        &self,
        source: CodecFormat,
        target: CodecFormat,
    ) -> Result<StreamTranscoder, CodecError> {
        let mut transcoder = self.stream_transcoder(source, target);
        transcoder.fidelity.require_near_lossless()?;
        transcoder.strict_fidelity = true;
        Ok(transcoder)
    }

    /// 构造接收任意 SSE 网络分片的端到端流转换器。
    pub fn sse_stream_transcoder(
        &self,
        source: CodecFormat,
        target: CodecFormat,
    ) -> Result<SseStreamTranscoder, CodecError> {
        Ok(SseStreamTranscoder {
            parser: SseParser::new(self.limits.max_stream_event_bytes)?,
            transcoder: self.stream_transcoder(source, target),
            max_output_bytes: self.limits.max_output_bytes,
        })
    }

    pub fn sse_stream_transcoder_near_lossless(
        &self,
        source: CodecFormat,
        target: CodecFormat,
    ) -> Result<SseStreamTranscoder, CodecError> {
        let transcoder = self.stream_transcoder_near_lossless(source, target)?;
        Ok(SseStreamTranscoder {
            parser: SseParser::new(self.limits.max_stream_event_bytes)?,
            transcoder,
            max_output_bytes: self.limits.max_output_bytes,
        })
    }

    fn check_input(&self, resource: &'static str, body: &[u8]) -> Result<(), CodecError> {
        if body.is_empty() {
            return Err(CodecError::InvalidInput {
                context: resource,
                message: "载荷不能为空".to_string(),
            });
        }
        check_limit(resource, body.len(), self.limits.max_input_bytes)
    }

    fn check_output(&self, output: Vec<u8>) -> Result<Vec<u8>, CodecError> {
        check_limit("encoded output", output.len(), self.limits.max_output_bytes)?;
        Ok(output)
    }
}

/// 单条上游 data 载荷到目标流帧的有状态转换器。
pub struct StreamTranscoder {
    decoder: Box<dyn DecodeStream>,
    encoder: Box<dyn EncodeStream>,
    max_event_bytes: usize,
    max_output_bytes: usize,
    terminal: bool,
    fidelity: FidelityReport,
    strict_fidelity: bool,
}

/// 原始 SSE 字节流到目标供应商帧的增量转换器。
pub struct SseStreamTranscoder {
    parser: SseParser,
    transcoder: StreamTranscoder,
    max_output_bytes: usize,
}

impl SseStreamTranscoder {
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<u8>, CodecError> {
        let payloads = self.parser.push(chunk)?;
        self.transcode_payloads(payloads)
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        let mut output = Vec::new();
        if let Some(payload) = self.parser.finish()? {
            output.extend(self.transcoder.transcode_data(&payload)?);
        }
        output.extend(self.transcoder.finish()?);
        check_limit("stream output", output.len(), self.max_output_bytes)?;
        Ok(output)
    }

    pub fn buffered_input_bytes(&self) -> usize {
        self.parser.buffered_len()
    }

    pub fn fidelity_report(&self) -> &FidelityReport {
        self.transcoder.fidelity_report()
    }

    fn transcode_payloads(&mut self, payloads: Vec<Vec<u8>>) -> Result<Vec<u8>, CodecError> {
        let mut output = Vec::new();
        for payload in payloads {
            output.extend(self.transcoder.transcode_data(&payload)?);
            check_limit("stream output", output.len(), self.max_output_bytes)?;
        }
        Ok(output)
    }
}

impl StreamTranscoder {
    pub fn fidelity_report(&self) -> &FidelityReport {
        &self.fidelity
    }

    pub fn transcode_data(&mut self, data: &[u8]) -> Result<Vec<u8>, CodecError> {
        check_limit("stream event", data.len(), self.max_event_bytes)?;
        if self.terminal {
            if data == b"[DONE]" {
                return Ok(Vec::new());
            }
            return Err(CodecError::InvalidState(
                "终态后收到新的上游流事件".to_string(),
            ));
        }

        let events = self.decoder.decode_sse_data(data)?;
        self.encode_events(events)
    }

    /// 上游连接结束时刷新解码器内部尚未闭合的块。
    pub fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if self.terminal {
            return Ok(Vec::new());
        }
        let events = self.decoder.finish()?;
        self.encode_events(events)
    }

    fn encode_events(&mut self, events: Vec<IrStreamEvent<'_>>) -> Result<Vec<u8>, CodecError> {
        let mut output = Vec::new();
        for event in events {
            audit_stream_event(&mut self.fidelity, &event);
            if self.strict_fidelity {
                if let Err(error) = self.fidelity.require_near_lossless() {
                    self.terminal = true;
                    return Err(error);
                }
            }
            let is_terminal = matches!(event, IrStreamEvent::Done | IrStreamEvent::Error { .. });
            output.extend(self.encoder.encode_sse_event(&event)?);
            if is_terminal {
                self.terminal = true;
            }
            check_limit("stream output", output.len(), self.max_output_bytes)?;
        }
        Ok(output)
    }
}

fn check_limit(resource: &'static str, actual: usize, limit: usize) -> Result<(), CodecError> {
    if actual > limit {
        Err(CodecError::LimitExceeded {
            resource,
            limit,
            actual,
        })
    } else {
        Ok(())
    }
}

fn validate_request(request: &IrRequest<'_>) -> Result<(), CodecError> {
    if request.messages.is_empty() {
        return Err(CodecError::MissingField("messages"));
    }
    if request.max_tokens == Some(0) {
        return Err(CodecError::InvalidInput {
            context: "request.max_tokens",
            message: "必须大于 0".to_string(),
        });
    }
    for message in &request.messages {
        validate_message(message)?;
    }
    if let Some(tools) = &request.tools {
        for tool in tools {
            if tool.tool_type == IrToolType::Function && tool.name.trim().is_empty() {
                return Err(CodecError::MissingField("tools[].name"));
            }
        }
    }
    Ok(())
}

fn validate_response(response: &IrResponse<'_>) -> Result<(), CodecError> {
    for choice in &response.choices {
        validate_message(&choice.message)?;
    }
    Ok(())
}

fn validate_message(message: &IrMessage<'_>) -> Result<(), CodecError> {
    if message.role == Role::Tool
        && message
            .tool_call_id
            .as_deref()
            .unwrap_or_default()
            .is_empty()
        && message.tool_name.as_deref().unwrap_or_default().is_empty()
    {
        return Err(CodecError::MissingField(
            "tool message tool_call_id/tool_name",
        ));
    }
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            validate_tool_identity(&tool_call.id, &tool_call.name)?;
        }
    }
    if let IrContent::Parts(parts) = &message.content {
        for part in parts {
            if let IrContentPart::FunctionCall { id, name, .. } = part {
                validate_tool_identity(id, name)?;
            }
        }
    }
    Ok(())
}

fn validate_tool_identity(id: &str, name: &str) -> Result<(), CodecError> {
    if id.trim().is_empty() {
        return Err(CodecError::MissingField("tool_call.id"));
    }
    if name.trim().is_empty() {
        return Err(CodecError::MissingField("tool_call.name"));
    }
    Ok(())
}

fn validate_target_arguments_in_messages(
    messages: &[IrMessage<'_>],
    target: CodecFormat,
) -> Result<(), CodecError> {
    for message in messages {
        validate_target_arguments_in_message(message, target)?;
    }
    Ok(())
}

fn validate_target_arguments_in_message(
    message: &IrMessage<'_>,
    target: CodecFormat,
) -> Result<(), CodecError> {
    if !matches!(target, CodecFormat::Anthropic | CodecFormat::Gemini) {
        return Ok(());
    }
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            validate_json_object(&tool_call.arguments, "tool_call.arguments")?;
        }
    }
    if let IrContent::Parts(parts) = &message.content {
        for part in parts {
            if let IrContentPart::FunctionCall { arguments, .. } = part {
                validate_json_object(arguments, "function_call.arguments")?;
            }
        }
    }
    Ok(())
}

fn validate_json_object(value: &str, context: &'static str) -> Result<(), CodecError> {
    let parsed: serde_json::Value =
        serde_json::from_str(value).map_err(|error| CodecError::InvalidInput {
            context,
            message: error.to_string(),
        })?;
    if !parsed.is_object() {
        return Err(CodecError::InvalidInput {
            context,
            message: "目标协议要求 JSON object".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_aliases_are_stable() {
        assert_eq!(
            "openai-responses".parse::<CodecFormat>().unwrap(),
            CodecFormat::OpenAiResponses
        );
        assert_eq!(
            "claude".parse::<CodecFormat>().unwrap(),
            CodecFormat::Anthropic
        );
        assert!("unknown".parse::<CodecFormat>().is_err());
    }

    #[test]
    fn serde_names_match_the_public_protocol_contract() {
        assert_eq!(
            serde_json::to_string(&CodecFormat::OpenAi).unwrap(),
            "\"openai\""
        );
        assert_eq!(
            serde_json::to_string(&CodecFormat::OpenAiResponses).unwrap(),
            "\"responses\""
        );
        assert!(serde_json::from_str::<CodecFormat>("\"open_ai\"").is_err());
        assert!(serde_json::from_str::<CodecFormat>("\"open_ai_responses\"").is_err());
    }

    #[test]
    fn input_limit_is_enforced_before_json_parsing() {
        let codec = Codec::new(CodecLimits {
            max_input_bytes: 2,
            ..CodecLimits::default()
        })
        .unwrap();
        let error = codec
            .decode_request(CodecFormat::OpenAi, b"{} ")
            .unwrap_err();
        assert!(matches!(error, CodecError::LimitExceeded { .. }));
    }

    #[test]
    fn malformed_arguments_are_not_silently_replaced() {
        let codec = Codec::default();
        let request = IrRequest {
            model: "m".into(),
            messages: vec![IrMessage {
                role: Role::Assistant,
                content: IrContent::Text("".into()),
                tool_call_id: None,
                tool_name: None,
                tool_calls: Some(vec![IrToolCall {
                    id: "call_1".into(),
                    name: "f".into(),
                    arguments: "{".into(),
                }]),
                cache_control: None,
                refusal: None,
            }],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: Some(10),
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            stream: false,
            store: None,
            modalities: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            response_format: None,
            previous_response_id: None,
            truncation: None,
            metadata: None,
            provider_metadata: None,
            metadata_mode: MetadataMode::Preserve,
        };

        assert!(matches!(
            codec.encode_request(CodecFormat::Anthropic, &request),
            Err(CodecError::InvalidInput { .. })
        ));
        assert!(matches!(
            AnthropicShim.encode_request(&request),
            Err(CodecError::InvalidInput { .. })
        ));
        assert!(matches!(
            GeminiShim.encode_request(&request),
            Err(CodecError::InvalidInput { .. })
        ));
    }

    #[test]
    fn fragmented_sse_is_transcoded_end_to_end() {
        let codec = Codec::default();
        let mut stream = codec
            .sse_stream_transcoder(CodecFormat::OpenAi, CodecFormat::Anthropic)
            .unwrap();
        let first = stream
            .push(b"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{")
            .unwrap();
        assert!(first.is_empty());
        let second = stream
            .push(b"\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
            .unwrap();
        assert!(String::from_utf8(second).unwrap().contains("message_start"));
        assert!(stream.finish().unwrap().is_empty());
    }

    #[test]
    fn openai_disconnect_flushes_open_blocks_once() {
        let mut decoder = CodecFormat::OpenAi.stream_decoder();
        decoder
            .decode_sse_data(
                concat!(
                    r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"#,
                    r#""content":"hi","#,
                    r#""tool_calls":[{"index":4,"id":"call_1","function":{"name":"lookup","#,
                    r#""arguments":"{\"q\":"}}]}}]}"#,
                )
                .as_bytes(),
            )
            .unwrap();

        let events = decoder.finish().unwrap();
        assert!(events
            .iter()
            .any(|event| matches!(event, IrStreamEvent::ContentDone { index: 0 })));
        assert!(events.iter().any(|event| matches!(
            event,
            IrStreamEvent::ToolCallDone {
                index: 0,
                choice_index: 0,
                arguments,
                ..
            } if arguments == "{\"q\":"
        )));
        assert!(matches!(events.last(), Some(IrStreamEvent::Done)));
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn anthropic_disconnect_flushes_tool_arguments() {
        let mut decoder = CodecFormat::Anthropic.stream_decoder();
        decoder
            .decode_sse_data(
                concat!(
                    r#"{"type":"message_start","#,
                    r#""message":{"id":"m1","model":"claude","usage":{"input_tokens":1,"output_tokens":0}}}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        decoder
            .decode_sse_data(
                concat!(
                    r#"{"type":"content_block_start","index":9,"content_block":{"type":"tool_use","#,
                    r#""id":"tool_1","name":"lookup","input":{}}}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        decoder
            .decode_sse_data(
                concat!(
                    r#"{"type":"content_block_delta","index":9,"delta":{"type":"input_json_delta","#,
                    r#""partial_json":"{\"q\":1}"}}"#,
                )
                .as_bytes(),
            )
            .unwrap();

        let events = decoder.finish().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            IrStreamEvent::ToolCallDone { arguments, .. } if arguments == "{\"q\":1}"
        )));
        assert!(matches!(events.last(), Some(IrStreamEvent::Done)));
    }

    #[test]
    fn responses_tool_indices_are_global_not_output_indices() {
        let mut decoder = CodecFormat::OpenAiResponses.stream_decoder();
        let first = decoder
            .decode_sse_data(
                concat!(
                    r#"{"type":"response.output_item.added","output_index":8,"item":{"type":"function_call","#,
                    r#""call_id":"a","name":"one"}}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        let second = decoder
            .decode_sse_data(
                concat!(
                    r#"{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","#,
                    r#""call_id":"b","name":"two"}}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        let delta = decoder
            .decode_sse_data(
                r#"{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{}"}"#
                    .as_bytes(),
            )
            .unwrap();

        assert!(matches!(
            first.as_slice(),
            [IrStreamEvent::ToolCallStart { index: 0, .. }]
        ));
        assert!(matches!(
            second.as_slice(),
            [IrStreamEvent::ToolCallStart { index: 1, .. }]
        ));
        assert!(matches!(
            delta.as_slice(),
            [IrStreamEvent::ToolCallDelta { index: 1, .. }]
        ));
    }

    #[test]
    fn gemini_thought_signatures_are_isolated_per_candidate() {
        let mut decoder = CodecFormat::Gemini.stream_decoder();
        decoder
            .decode_sse_data(
                concat!(
                    r#"{"candidates":[{"index":0,"content":{"parts":[{"text":"r0","thought":true,"#,
                    r#""thoughtSignature":"sig0"}]}},{"index":1,"content":{"parts":[{"text":"r1","thought":true,"#,
                    r#""thoughtSignature":"sig1"}]}}]}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        let events = decoder
            .decode_sse_data(
                concat!(
                    r#"{"candidates":[{"index":0,"content":{"parts":[{"text":"a0"}]}},{"index":1,"#,
                    r#""content":{"parts":[{"text":"a1"}]}}]}"#,
                )
                .as_bytes(),
            )
            .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            IrStreamEvent::ReasoningDone { index: 0, signature: Some(signature) }
                if signature == "sig0"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            IrStreamEvent::ReasoningDone { index: 1, signature: Some(signature) }
                if signature == "sig1"
        )));
    }

    #[test]
    fn gemini_encoder_keeps_all_candidate_finishes() {
        let mut encoder = CodecFormat::Gemini.stream_encoder();
        encoder
            .encode_sse_event(&IrStreamEvent::ChoiceFinish {
                index: 0,
                finish_reason: IrFinishReason::Stop,
            })
            .unwrap();
        encoder
            .encode_sse_event(&IrStreamEvent::ChoiceFinish {
                index: 1,
                finish_reason: IrFinishReason::Length,
            })
            .unwrap();
        let bytes = encoder
            .encode_sse_event(&IrStreamEvent::Usage(IrUsage::default()))
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let candidates = value["candidates"].as_array().unwrap();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0]["index"], 0);
        assert_eq!(candidates[1]["index"], 1);
    }
}
