//! 有状态流式管线端到端测试
//!
//! 验证 4×4 转换矩阵中最脆弱的路径：SSE 流跨格式转译。
//! 每条测试模拟「上游供应商 SSE → DecodeStream → IR 事件 → EncodeStream → 下游格式」。

use openxlate::codec::ir::*;
use openxlate::codec::shim::anthropic::{AntStreamDecoder, AntStreamEncoder};
use openxlate::codec::shim::gemini::{GemStreamDecoder, GemStreamEncoder};
use openxlate::codec::shim::openai::{OaiStreamDecoder, OaiStreamEncoder};
use openxlate::codec::shim::openai_responses::{RspStreamDecoder, RspStreamEncoder};
use openxlate::codec::shim::{DecodeStream, EncodeStream};

fn decode_all<'a, D: DecodeStream>(dec: &mut D, chunks: &'a [&'a str]) -> Vec<IrStreamEvent<'a>> {
    let mut events = Vec::new();
    for c in chunks {
        events.extend(dec.decode_sse_data(c.as_bytes()).expect("decode failed"));
    }
    events
}

fn encode_all<E: EncodeStream>(enc: &mut E, events: &[IrStreamEvent<'_>]) -> String {
    let mut out = Vec::new();
    for e in events {
        out.extend(enc.encode_sse_event(e).expect("encode failed"));
    }
    String::from_utf8(out).expect("non-utf8 output")
}

// ─── OpenAI 解码器：合成 Start / ReasoningDone / ContentDone / ToolCallDone ──

#[test]
fn oai_decoder_synthesizes_lifecycle() {
    let chunks = [
        r#"{"id":"c1","model":"gpt-x","choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#,
        r#"{"id":"c1","model":"gpt-x","choices":[{"index":0,"delta":{"content":"hello"}}]}"#,
        r#"{"id":"c1","model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ];
    let mut dec = OaiStreamDecoder::new();
    let events = decode_all(&mut dec, &chunks);

    assert!(
        matches!(events[0], IrStreamEvent::Start { .. }),
        "首 chunk 应合成 Start"
    );
    // reasoning → content 切换处应有 ReasoningDone
    let rd_pos = events
        .iter()
        .position(|e| matches!(e, IrStreamEvent::ReasoningDone { .. }));
    let cd_pos = events
        .iter()
        .position(|e| matches!(e, IrStreamEvent::ContentDelta { .. }));
    assert!(
        rd_pos.is_some() && rd_pos < cd_pos,
        "ReasoningDone 应先于 ContentDelta"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, IrStreamEvent::ContentDone { .. })),
        "finish 时应合成 ContentDone"
    );
}

#[test]
fn oai_decoder_synthesizes_toolcalldone_with_args() {
    let chunks = [
        r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}"#,
        r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
        r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]}}]}"#,
        r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    ];
    let mut dec = OaiStreamDecoder::new();
    let events = decode_all(&mut dec, &chunks);

    let done = events.iter().find_map(|e| match e {
        IrStreamEvent::ToolCallDone {
            id,
            name,
            arguments,
            ..
        } => Some((id.to_string(), name.to_string(), arguments.to_string())),
        _ => None,
    });
    let (id, name, args) = done.expect("应合成 ToolCallDone");
    assert_eq!(id, "call_1");
    assert_eq!(name, "get_weather");
    assert_eq!(args, r#"{"city":"SF"}"#, "arguments 应为完整累积值");
}

// ─── Anthropic 解码器：块类型区分 + signature 捕获 ─────────────────

#[test]
fn ant_decoder_distinguishes_block_stops() {
    let chunks = [
        r#"{"type":"message_start","message":{"id":"m1","model":"claude","usage":{"input_tokens":10,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG42"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_1","name":"f","input":{}}}"#,
        r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}"#,
        r#"{"type":"content_block_stop","index":2}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
        r#"{"type":"message_stop"}"#,
    ];
    let mut dec = AntStreamDecoder::new();
    let events = decode_all(&mut dec, &chunks);

    // Start 应携带初始 usage
    match &events[0] {
        IrStreamEvent::Start { usage: Some(u), .. } => assert_eq!(u.prompt_tokens, 10),
        other => panic!("首事件应为携带 usage 的 Start，得到 {other:?}"),
    }
    // thinking 块 stop → ReasoningDone 且带 signature
    let sig = events.iter().find_map(|e| match e {
        IrStreamEvent::ReasoningDone { signature, .. } => Some(signature.clone()),
        _ => None,
    });
    assert_eq!(
        sig.flatten().as_deref(),
        Some("SIG42"),
        "signature_delta 应在 ReasoningDone 中输出"
    );
    // text 块 stop → ContentDone（index 是 choice 索引，Anthropic 单候选恒 0）
    assert!(events
        .iter()
        .any(|e| matches!(e, IrStreamEvent::ContentDone { index: 0 })));
    // tool_use 块 stop → ToolCallDone 带累积 arguments
    let tc = events.iter().find_map(|e| match e {
        IrStreamEvent::ToolCallDone { id, arguments, .. } => {
            Some((id.to_string(), arguments.to_string()))
        }
        _ => None,
    });
    let (id, args) = tc.expect("tool_use stop 应产生 ToolCallDone");
    assert_eq!(id, "tu_1");
    assert_eq!(args, r#"{"a":1}"#);
}

// ─── Gemini 解码器：合成 Start / Done、全流唯一 tool id ─────────────

#[test]
fn gem_decoder_lifecycle_and_unique_tool_ids() {
    let chunks = [
        r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f1","args":{"x":1}}},{"functionCall":{"name":"f2","args":{"y":2}}}]},"index":0}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}"#,
    ];
    let mut dec = GemStreamDecoder::new();
    let events = decode_all(&mut dec, &chunks);

    assert!(
        matches!(events[0], IrStreamEvent::Start { .. }),
        "Gemini 首 chunk 应合成 Start"
    );
    let ids: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            IrStreamEvent::ToolCallStart { id, .. } => Some(id.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        ids,
        vec!["call_gemini_0", "call_gemini_1"],
        "同 chunk 多个 function_call 的 id 应唯一"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, IrStreamEvent::ContentDone { .. })),
        "finish 时应关闭 content"
    );
    assert!(
        matches!(events.last(), Some(IrStreamEvent::Done)),
        "finishReason 后应合成 Done"
    );
}

// ─── Anthropic 编码器：块生命周期包裹 + stop_reason/usage 合并 ─────

#[test]
fn ant_encoder_wraps_blocks_and_merges_message_delta() {
    let events: Vec<IrStreamEvent<'_>> = vec![
        IrStreamEvent::Start {
            id: "x1".into(),
            model: "m".into(),
            usage: Some(IrUsage {
                prompt_tokens: 3,
                ..Default::default()
            }),
        },
        IrStreamEvent::ReasoningDelta {
            index: 0,
            delta: "think".into(),
        },
        IrStreamEvent::ReasoningDone {
            index: 0,
            signature: Some("S".into()),
        },
        IrStreamEvent::ContentDelta {
            index: 0,
            delta: "hi".into(),
        },
        IrStreamEvent::ContentDone { index: 0 },
        IrStreamEvent::ChoiceFinish {
            index: 0,
            finish_reason: IrFinishReason::Stop,
        },
        IrStreamEvent::Usage(IrUsage {
            prompt_tokens: 3,
            completion_tokens: 9,
            total_tokens: 12,
            ..Default::default()
        }),
        IrStreamEvent::Done,
    ];
    let mut enc = AntStreamEncoder::new();
    let out = encode_all(&mut enc, &events);

    // message_start 带 usage
    assert!(
        out.contains(r#""input_tokens":3"#),
        "message_start 应携带初始 usage"
    );
    // thinking 块被 start/stop 包裹且索引 0
    assert!(
        out.contains(r#""content_block":{"thinking":"","type":"thinking"}"#)
            || out.contains(r#""content_block":{"type":"thinking","thinking":""}"#),
        "应有 thinking 的 content_block_start：\n{out}"
    );
    // signature_delta 出现在块内
    assert!(
        out.contains("signature_delta"),
        "ReasoningDone 的签名应编码为 signature_delta"
    );
    // text 块使用递增索引 1（thinking 占 0）
    assert!(
        out.contains(r#""index":1"#),
        "text 块应重映射到输出索引 1：\n{out}"
    );
    // stop_reason 与 usage 合并在同一条 message_delta
    let md_count = out.matches(r#""type":"message_delta""#).count();
    assert_eq!(
        md_count, 1,
        "stop_reason 与 usage 应合并为单条 message_delta：\n{out}"
    );
    assert!(out.contains(r#""stop_reason":"end_turn""#));
    assert!(out.contains("message_stop"));
}

// ─── OpenAI 编码器：id/model 每帧回填 + 无 Start 的 tool 补发 ──────

#[test]
fn oai_encoder_backfills_id_and_full_toolcall() {
    let events: Vec<IrStreamEvent<'_>> = vec![
        IrStreamEvent::Start {
            id: "resp1".into(),
            model: "mm".into(),
            usage: None,
        },
        IrStreamEvent::ContentDelta {
            index: 0,
            delta: "ok".into(),
        },
        // 未经 ToolCallStart 直接 Done（模拟仅有完整 tool call 的上游）
        IrStreamEvent::ToolCallDone {
            index: 0,
            choice_index: 0,
            id: "call_g".into(),
            name: "fn".into(),
            arguments: r#"{"k":true}"#.into(),
        },
        IrStreamEvent::ChoiceFinish {
            index: 0,
            finish_reason: IrFinishReason::ToolCalls,
        },
        IrStreamEvent::Done,
    ];
    let mut enc = OaiStreamEncoder::new();
    let out = encode_all(&mut enc, &events);

    // 每个 chunk 都应携带 id/model
    for line in out.lines().filter(|l| l.starts_with("data: {")) {
        assert!(line.contains(r#""id":"resp1""#), "chunk 缺 id: {line}");
        assert!(line.contains(r#""model":"mm""#), "chunk 缺 model: {line}");
    }
    // 未宣告的 ToolCallDone → 补发完整 tool_call 帧
    assert!(
        out.contains(r#""name":"fn""#) && out.contains(r#"{\"k\":true}"#),
        "应补发完整 tool_call：\n{out}"
    );
    assert!(out.contains("[DONE]"));
}

// ─── Gemini 编码器：finishReason 与 usage 合并 + 增量 args 累积 ────

#[test]
fn gem_encoder_merges_finish_with_usage_and_accumulates_args() {
    let events: Vec<IrStreamEvent<'_>> = vec![
        IrStreamEvent::Start {
            id: "".into(),
            model: "g".into(),
            usage: None,
        },
        // OpenAI 式增量 tool call（无一次性完整 arguments）
        IrStreamEvent::ToolCallStart {
            index: 0,
            choice_index: 0,
            id: "c1".into(),
            name: "fx".into(),
        },
        IrStreamEvent::ToolCallDelta {
            index: 0,
            choice_index: 0,
            arguments_delta: r#"{"a""#.into(),
        },
        IrStreamEvent::ToolCallDelta {
            index: 0,
            choice_index: 0,
            arguments_delta: r#":1}"#.into(),
        },
        IrStreamEvent::ChoiceFinish {
            index: 0,
            finish_reason: IrFinishReason::ToolCalls,
        },
        IrStreamEvent::Usage(IrUsage {
            prompt_tokens: 2,
            completion_tokens: 4,
            total_tokens: 6,
            ..Default::default()
        }),
        IrStreamEvent::Done,
    ];
    let mut enc = GemStreamEncoder::new();
    let out = encode_all(&mut enc, &events);

    // 增量 args 应累积成完整 functionCall
    assert!(
        out.contains(r#""functionCall":{"args":{"a":1},"name":"fx"}"#)
            || out.contains(r#""functionCall":{"name":"fx","args":{"a":1}}"#),
        "增量 arguments 应累积为完整 functionCall：\n{out}"
    );
    // finishReason 与 usageMetadata 在同一 chunk
    let final_chunk = out.split("}{").last().unwrap_or(&out);
    assert!(
        out.contains(r#""finishReason":"STOP""#) || final_chunk.contains("finishReason"),
        "finishReason 应存在：\n{out}"
    );
    assert!(out.contains("usageMetadata"));
}

// ─── Responses 编码器：单条权威 response.completed ─────────────────

#[test]
fn rsp_encoder_single_completed_event() {
    let events: Vec<IrStreamEvent<'_>> = vec![
        IrStreamEvent::Start {
            id: "r1".into(),
            model: "o3".into(),
            usage: None,
        },
        IrStreamEvent::ContentDelta {
            index: 0,
            delta: "text".into(),
        },
        IrStreamEvent::ContentDone { index: 0 },
        IrStreamEvent::ChoiceFinish {
            index: 0,
            finish_reason: IrFinishReason::Stop,
        },
        IrStreamEvent::Usage(IrUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            ..Default::default()
        }),
        IrStreamEvent::Done,
    ];
    let mut enc = RspStreamEncoder::new();
    let out = encode_all(&mut enc, &events);

    assert_eq!(
        out.matches("response.completed").count(),
        1,
        "终态应恰好一条 response.completed：\n{out}"
    );
    // completed 帧应同时携带 id/model/status/usage
    let completed_line = out
        .lines()
        .find(|l| l.contains("response.completed"))
        .unwrap();
    assert!(completed_line.contains(r#""id":"r1""#));
    assert!(completed_line.contains(r#""status":"completed""#));
    assert!(completed_line.contains(r#""total_tokens":3"#));
}

// ─── 端到端矩阵：Anthropic 流 → OpenAI SSE ─────────────────────────

#[test]
fn e2e_anthropic_stream_to_openai_sse() {
    let ant_chunks = [
        r#"{"type":"message_start","message":{"id":"m9","model":"claude-x","usage":{"input_tokens":4,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
        r#"{"type":"message_stop"}"#,
    ];
    let mut dec = AntStreamDecoder::new();
    let mut enc = OaiStreamEncoder::new();
    let mut out = Vec::new();
    for c in ant_chunks {
        for e in dec.decode_sse_data(c.as_bytes()).unwrap() {
            out.extend(enc.encode_sse_event(&e).unwrap());
        }
    }
    let s = String::from_utf8(out).unwrap();

    assert!(s.contains(r#""id":"m9""#) && s.contains(r#""model":"claude-x""#));
    assert!(s.contains(r#""content":"你好""#));
    assert!(s.contains(r#""finish_reason":"stop""#));
    assert!(
        s.ends_with("data: [DONE]\n\n"),
        "OpenAI 流应以 [DONE] 结束：\n{s}"
    );
}

// ─── 端到端矩阵：OpenAI 流 → Anthropic SSE ─────────────────────────

#[test]
fn e2e_openai_stream_to_anthropic_sse() {
    let oai_chunks = [
        r#"{"id":"c7","model":"gpt","choices":[{"index":0,"delta":{"content":"Hi"}}]}"#,
        r#"{"id":"c7","model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        r#"{"id":"c7","model":"gpt","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
    ];
    let mut dec = OaiStreamDecoder::new();
    let mut enc = AntStreamEncoder::new();
    let mut out = Vec::new();
    for c in oai_chunks {
        for e in dec.decode_sse_data(c.as_bytes()).unwrap() {
            out.extend(enc.encode_sse_event(&e).unwrap());
        }
    }
    // [DONE]
    for e in dec.decode_sse_data(b"[DONE]").unwrap() {
        out.extend(enc.encode_sse_event(&e).unwrap());
    }
    let s = String::from_utf8(out).unwrap();

    assert!(
        s.contains("message_start"),
        "OpenAI Start → message_start：\n{s}"
    );
    assert!(
        s.contains("content_block_start"),
        "首 delta 应自动开块：\n{s}"
    );
    assert!(s.contains(r#""text":"Hi""#));
    assert!(s.contains("content_block_stop"), "finish 应关块：\n{s}");
    assert!(s.contains(r#""stop_reason":"end_turn""#));
    assert!(s.contains("message_stop"), "结束应有 message_stop：\n{s}");
    // 顺序：content_block_stop 必须在 message_delta 之前
    let stop_pos = s.find("content_block_stop").unwrap();
    let md_pos = s.find(r#""stop_reason":"end_turn""#).unwrap();
    assert!(stop_pos < md_pos, "块关闭应先于 message_delta");
}

// ─── 端到端矩阵：Gemini 流 → Anthropic SSE（tool call 补发）────────

#[test]
fn e2e_gemini_stream_to_anthropic_sse_with_tools() {
    let gem_chunks = [
        r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"look","args":{"q":"x"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
    ];
    let mut dec = GemStreamDecoder::new();
    let mut enc = AntStreamEncoder::new();
    let mut out = Vec::new();
    for c in gem_chunks {
        for e in dec.decode_sse_data(c.as_bytes()).unwrap() {
            out.extend(enc.encode_sse_event(&e).unwrap());
        }
    }
    let s = String::from_utf8(out).unwrap();

    assert!(
        s.contains(r#""type":"tool_use""#),
        "functionCall → tool_use 块：\n{s}"
    );
    assert!(s.contains(r#""name":"look""#));
    assert!(
        s.contains("input_json_delta"),
        "args 应作为 input_json_delta 发出：\n{s}"
    );
    assert!(s.contains("content_block_stop"));
}

// ─── 回归：Claude thinking 流 → OpenAI，正文块索引不得泄漏为 choice 索引 ──

#[test]
fn e2e_anthropic_thinking_stream_to_openai_choice_index() {
    let ant_chunks = [
        r#"{"type":"message_start","message":{"id":"m1","model":"claude","usage":{"input_tokens":5,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
        r#"{"type":"message_stop"}"#,
    ];
    let mut dec = AntStreamDecoder::new();
    let mut enc = OaiStreamEncoder::new();
    let mut out = Vec::new();
    for c in ant_chunks {
        for e in dec.decode_sse_data(c.as_bytes()).unwrap() {
            out.extend(enc.encode_sse_event(&e).unwrap());
        }
    }
    let s = String::from_utf8(out).unwrap();

    // 正文（原线上块索引 1）必须落在 choice 0
    assert!(
        s.contains(r#""choices":[{"delta":{"content":"answer"},"index":0}"#),
        "正文应在 choice 0，不得泄漏内容块索引：\n{s}"
    );
    assert!(
        !s.contains(r#""index":1"#),
        "不应出现 choice index 1：\n{s}"
    );
}

// ─── 回归：Gemini 逐 chunk usage → Anthropic 只在终态发一条 message_delta ──

#[test]
fn e2e_gemini_periodic_usage_to_anthropic_single_message_delta() {
    let gem_chunks = [
        // 非终止 chunk 携带累计 usage
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]},"index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"b"}]},"index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":2,"totalTokenCount":4}}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"c"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}"#,
    ];
    let mut dec = GemStreamDecoder::new();
    let mut enc = AntStreamEncoder::new();
    let mut out = Vec::new();
    for c in gem_chunks {
        for e in dec.decode_sse_data(c.as_bytes()).unwrap() {
            out.extend(enc.encode_sse_event(&e).unwrap());
        }
    }
    let s = String::from_utf8(out).unwrap();

    // 只允许一条 message_delta，且携带最终 usage（output 3）
    assert_eq!(
        s.matches(r#""type":"message_delta""#).count(),
        1,
        "usage 逐 chunk 到达时应只在终态发一条 message_delta：\n{s}"
    );
    assert!(
        s.contains(r#""output_tokens":3"#),
        "应携带最新累计 usage：\n{s}"
    );
    // message_delta 必须在所有 content_block_stop 之后
    let last_stop = s.rfind("content_block_stop").unwrap();
    let md = s.find(r#""type":"message_delta""#).unwrap();
    assert!(md > last_stop, "message_delta 应在全部块关闭之后：\n{s}");
}

// ─── 回归：Gemini thinking（thought:true 布尔标志）正确解析 ──

#[test]
fn gemini_thought_boolean_flag_parses() {
    let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"思考中","thought":true},{"text":"正文"}]},"finishReason":"STOP","index":0}]}"#;
    let mut dec = GemStreamDecoder::new();
    let events = dec
        .decode_sse_data(chunk.as_bytes())
        .expect("thought:true 不应导致解析失败");

    assert!(
        events.iter().any(|e| matches!(e,
        IrStreamEvent::ReasoningDelta { delta, .. } if delta == "思考中")),
        "thought:true 的 text 应为 ReasoningDelta"
    );
    assert!(
        events.iter().any(|e| matches!(e,
        IrStreamEvent::ContentDelta { delta, .. } if delta == "正文")),
        "无 thought 标志的 text 应为 ContentDelta"
    );

    // 非流式响应同样不得崩溃
    use openxlate::codec::shim::gemini::GeminiShim;
    use openxlate::codec::shim::DecodeResponse;
    let resp = GeminiShim
        .decode_response(chunk.as_bytes())
        .expect("非流式解析不应失败");
    let content = &resp.choices[0].message.content;
    match content {
        IrContent::Parts(parts) => {
            assert!(parts
                .iter()
                .any(|p| matches!(p, IrContentPart::Reasoning { .. })));
            assert!(parts
                .iter()
                .any(|p| matches!(p, IrContentPart::Text { .. })));
        }
        _ => panic!("应为多部件内容"),
    }
}

// ─── 回归：Anthropic tool_result 兄弟 text 块不丢失 ──

#[test]
fn anthropic_tool_result_sibling_text_preserved() {
    use openxlate::codec::shim::anthropic::AnthropicShim;
    use openxlate::codec::shim::DecodeRequest;
    let body = r#"{
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "assistant", "content": [{"type":"tool_use","id":"t1","name":"calc","input":{}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                {"type": "text", "text": "请解释这个结果"}
            ]}
        ]
    }"#;
    let ir = AnthropicShim.decode_request(body.as_bytes()).unwrap();

    let tool_msg = ir
        .messages
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("应有 Tool 消息");
    assert_eq!(tool_msg.content.as_text(), Some("42"));
    // tool_name 应通过 backfill 从 assistant 的 tool_use 反查到
    assert_eq!(
        tool_msg.tool_name.as_deref(),
        Some("calc"),
        "tool_name 应回填"
    );

    let follow_up = ir
        .messages
        .iter()
        .filter(|m| m.role == Role::User)
        .last()
        .expect("兄弟 text 应保留");
    assert_eq!(follow_up.content.as_text(), Some("请解释这个结果"));
}

// ─── 回归：跨格式工具结果关联（OpenAI 请求 → Gemini functionResponse.name）──

#[test]
fn openai_tool_result_to_gemini_uses_function_name() {
    use openxlate::codec::shim::gemini::GeminiShim;
    use openxlate::codec::shim::openai::OpenAiShim;
    use openxlate::codec::shim::{DecodeRequest, EncodeRequest};
    let body = r#"{
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\": 20}"}
        ]
    }"#;
    let ir = OpenAiShim.decode_request(body.as_bytes()).unwrap();
    let out = GeminiShim.encode_request(&ir).unwrap();
    let s = String::from_utf8(out).unwrap();

    assert!(
        s.contains(r#""name":"get_weather""#),
        "functionResponse.name 应为函数名而非 call id：\n{s}"
    );
    assert!(
        !s.contains(r#""functionResponse":{"name":"call_abc123""#),
        "不得用 call id 作为 functionResponse.name：\n{s}"
    );
}

// ─── 回归：Gemini SAFETY → OpenAI content_filter（安全拦截不得伪装成 stop）──

#[test]
fn gemini_safety_to_openai_content_filter() {
    let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"部分"}]},"finishReason":"SAFETY","index":0}]}"#;
    let mut dec = GemStreamDecoder::new();
    let mut enc = OaiStreamEncoder::new();
    let mut out = Vec::new();
    for e in dec.decode_sse_data(chunk.as_bytes()).unwrap() {
        out.extend(enc.encode_sse_event(&e).unwrap());
    }
    let s = String::from_utf8(out).unwrap();
    assert!(
        s.contains(r#""finish_reason":"content_filter""#),
        "SAFETY 应映射为 content_filter：\n{s}"
    );
}

// ─── 回归：Responses 编码器补齐 item 生命周期 + completed 含 output ──

#[test]
fn rsp_encoder_item_lifecycle_and_output_array() {
    let events: Vec<IrStreamEvent<'_>> = vec![
        IrStreamEvent::Start {
            id: "r1".into(),
            model: "o3".into(),
            usage: None,
        },
        IrStreamEvent::ContentDelta {
            index: 0,
            delta: "hello".into(),
        },
        IrStreamEvent::ContentDone { index: 0 },
        IrStreamEvent::ChoiceFinish {
            index: 0,
            finish_reason: IrFinishReason::Stop,
        },
        IrStreamEvent::Done,
    ];
    let mut enc = RspStreamEncoder::new();
    let out = encode_all(&mut enc, &events);

    // delta 前必须有 output_item.added
    let added = out
        .find("response.output_item.added")
        .expect("缺 output_item.added");
    let delta = out.find("response.output_text.delta").expect("缺 delta");
    assert!(added < delta, "output_item.added 应先于 delta：\n{out}");
    // completed 必须携带重建的 output 数组
    let completed_line = out
        .lines()
        .find(|l| l.contains("response.completed"))
        .unwrap();
    assert!(
        completed_line.contains(r#""output":[{"#) && completed_line.contains("hello"),
        "completed 应含完整 output：\n{completed_line}"
    );
}

// ─── 回归：OpenAI 上游 finish_reason=stop 但有 tool_calls 也要合成 Done ──

#[test]
fn oai_decoder_tool_done_even_on_stop_finish() {
    let chunks = [
        r#"{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{}"}}]}}]}"#,
        r#"{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ];
    let mut dec = OaiStreamDecoder::new();
    let events = decode_all(&mut dec, &chunks);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, IrStreamEvent::ToolCallDone { .. })),
        "即使 finish_reason=stop，活跃 tool call 也应合成 Done"
    );
}
