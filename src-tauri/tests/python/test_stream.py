"""流式编解码端到端测试

用真实 SSE 载荷验证 4 格式的 DecodeStream/EncodeStream：
- 纯文本流
- 工具调用流
- 推理/思考流
- 跨格式管线（A SSE → IR → B SSE → IR → 语义比较）
- n>1 多候选
- RedactedReasoning 保留
"""

import json
import pytest
from conftest import (
    decode_stream,
    encode_stream_raw,
    extract_text,
    extract_reasoning,
    extract_tool_calls,
    extract_finish,
    extract_usage,
)

# ═══════════════════════════════════════════════════════════════════
# SSE 载荷 fixtures — 模拟真实供应商返回
# ═══════════════════════════════════════════════════════════════════

OPENAI_TEXT_SSE = """\
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}

data: [DONE]

"""

ANTHROPIC_TEXT_SSE = """\
data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

data: {"type":"message_stop"}

"""

GEMINI_TEXT_SSE = """\
data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}

"""

RESPONSES_TEXT_SSE = """\
data: {"type":"response.created","response":{"id":"resp_x","object":"response","model":"gpt-4o","status":"in_progress","output":[]}}

data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}

data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}

data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}

data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":" world"}

data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello world"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}}

data: {"type":"response.completed","response":{"id":"resp_x","object":"response","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}

"""

# ─── 工具调用流 ──────────────────────────────────────────────────

OPENAI_TOOL_SSE = """\
data: {"id":"chatcmpl-t","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"NYC\\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":15,"total_tokens":35}}

data: [DONE]

"""

ANTHROPIC_TOOL_SSE = """\
data: {"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":20,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"get_weather","input":{}}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"ci"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ty\\":\\"NYC\\"}"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}

data: {"type":"message_stop"}

"""

# ─── 推理/思考流 ─────────────────────────────────────────────────

ANTHROPIC_THINKING_SSE = """\
data: {"type":"message_start","message":{"id":"msg_th","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}

data: {"type":"content_block_stop","index":0}

data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}

data: {"type":"content_block_stop","index":1}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}

data: {"type":"message_stop"}

"""

ANTHROPIC_REDACTED_SSE = """\
data: {"type":"message_start","message":{"id":"msg_rd","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"ENCRYPTED_DATA_BLOCK_1"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Result."}}

data: {"type":"content_block_stop","index":1}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

data: {"type":"message_stop"}

"""

# ─── n>1 多候选流 ────────────────────────────────────────────────

OPENAI_N2_SSE = """\
data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":1,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Choice A"},"finish_reason":null}]}

data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":1,"delta":{"content":"Choice B"},"finish_reason":null}]}

data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-n2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":1,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}

data: [DONE]

"""


# ═══════════════════════════════════════════════════════════════════
# 1. 各格式文本流解码
# ═══════════════════════════════════════════════════════════════════


class TestDecodeStream:
    def test_openai_text(self):
        events = decode_stream("openai", OPENAI_TEXT_SSE)
        assert extract_text(events) == "Hello world"
        assert extract_finish(events) == "stop"
        types = [e["type"] for e in events]
        assert "start" in types
        assert "done" in types

    def test_anthropic_text(self):
        events = decode_stream("anthropic", ANTHROPIC_TEXT_SSE)
        assert extract_text(events) == "Hello world"
        assert extract_finish(events) == "stop"

    def test_gemini_text(self):
        events = decode_stream("gemini", GEMINI_TEXT_SSE)
        assert extract_text(events) == "Hello world"
        assert extract_finish(events) == "stop"

    def test_responses_text(self):
        events = decode_stream("responses", RESPONSES_TEXT_SSE)
        assert extract_text(events) == "Hello world"
        assert extract_finish(events) == "stop"

    def test_openai_tool_call(self):
        events = decode_stream("openai", OPENAI_TOOL_SSE)
        tcs = extract_tool_calls(events)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "get_weather"
        assert tcs[0]["id"] == "call_1"
        args = json.loads(tcs[0]["arguments"])
        assert args["city"] == "NYC"
        assert extract_finish(events) == "tool_calls"

    def test_anthropic_tool_call(self):
        events = decode_stream("anthropic", ANTHROPIC_TOOL_SSE)
        tcs = extract_tool_calls(events)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "get_weather"
        assert tcs[0]["id"] == "call_1"
        assert extract_finish(events) == "tool_calls"

    def test_anthropic_thinking(self):
        events = decode_stream("anthropic", ANTHROPIC_THINKING_SSE)
        assert extract_reasoning(events) == "Let me think..."
        assert extract_text(events) == "The answer is 42."
        assert extract_finish(events) == "stop"
        reasoning_done = [e for e in events if e["type"] == "reasoning_done"]
        assert len(reasoning_done) >= 1

    def test_anthropic_redacted_reasoning(self):
        events = decode_stream("anthropic", ANTHROPIC_REDACTED_SSE)
        redacted = [e for e in events if e["type"] == "redacted_reasoning"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "ENCRYPTED_DATA_BLOCK_1"
        assert extract_text(events) == "Result."

    def test_openai_n2_multi_candidate(self):
        events = decode_stream("openai", OPENAI_N2_SSE)
        deltas_0 = [e for e in events if e["type"] == "content_delta" and e["index"] == 0]
        deltas_1 = [e for e in events if e["type"] == "content_delta" and e["index"] == 1]
        text_0 = "".join(e["delta"] for e in deltas_0)
        text_1 = "".join(e["delta"] for e in deltas_1)
        assert text_0 == "Choice A"
        assert text_1 == "Choice B"
        finishes = [e for e in events if e["type"] == "choice_finish"]
        assert len(finishes) == 2
        indices = {e["index"] for e in finishes}
        assert indices == {0, 1}


# ═══════════════════════════════════════════════════════════════════
# 2. 流式 round-trip（SSE → IR → SSE → IR）
# ═══════════════════════════════════════════════════════════════════


class TestStreamRoundTrip:
    def _round_trip(self, fmt, sse):
        events_a = decode_stream(fmt, sse)
        sse_b = encode_stream_raw(fmt, events_a)
        events_b = decode_stream(fmt, sse_b)
        return events_a, events_b

    def test_openai_text_rt(self):
        a, b = self._round_trip("openai", OPENAI_TEXT_SSE)
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_anthropic_text_rt(self):
        a, b = self._round_trip("anthropic", ANTHROPIC_TEXT_SSE)
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_gemini_text_rt(self):
        a, b = self._round_trip("gemini", GEMINI_TEXT_SSE)
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_responses_text_rt(self):
        a, b = self._round_trip("responses", RESPONSES_TEXT_SSE)
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_openai_tool_rt(self):
        a, b = self._round_trip("openai", OPENAI_TOOL_SSE)
        tca = extract_tool_calls(a)
        tcb = extract_tool_calls(b)
        assert len(tca) == len(tcb)
        assert tca[0]["name"] == tcb[0]["name"]
        assert tca[0]["id"] == tcb[0]["id"]

    def test_anthropic_thinking_rt(self):
        a, b = self._round_trip("anthropic", ANTHROPIC_THINKING_SSE)
        assert extract_reasoning(a) == extract_reasoning(b)
        assert extract_text(a) == extract_text(b)


# ═══════════════════════════════════════════════════════════════════
# 3. 跨格式流管线（A SSE → IR → B SSE → IR → 语义比较）
# ═══════════════════════════════════════════════════════════════════


class TestStreamCrossFormat:
    def _cross(self, src_fmt, sse, dst_fmt):
        ir_a = decode_stream(src_fmt, sse)
        sse_b = encode_stream_raw(dst_fmt, ir_a)
        ir_b = decode_stream(dst_fmt, sse_b)
        return ir_a, ir_b

    def test_openai_to_anthropic(self):
        a, b = self._cross("openai", OPENAI_TEXT_SSE, "anthropic")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_openai_to_gemini(self):
        a, b = self._cross("openai", OPENAI_TEXT_SSE, "gemini")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_openai_to_responses(self):
        a, b = self._cross("openai", OPENAI_TEXT_SSE, "responses")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_anthropic_to_openai(self):
        a, b = self._cross("anthropic", ANTHROPIC_TEXT_SSE, "openai")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_anthropic_to_gemini(self):
        a, b = self._cross("anthropic", ANTHROPIC_TEXT_SSE, "gemini")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_anthropic_to_responses(self):
        a, b = self._cross("anthropic", ANTHROPIC_TEXT_SSE, "responses")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_gemini_to_openai(self):
        a, b = self._cross("gemini", GEMINI_TEXT_SSE, "openai")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_gemini_to_anthropic(self):
        a, b = self._cross("gemini", GEMINI_TEXT_SSE, "anthropic")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_responses_to_openai(self):
        a, b = self._cross("responses", RESPONSES_TEXT_SSE, "openai")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_responses_to_anthropic(self):
        a, b = self._cross("responses", RESPONSES_TEXT_SSE, "anthropic")
        assert extract_text(a) == extract_text(b)
        assert extract_finish(a) == extract_finish(b)

    def test_tool_openai_to_anthropic(self):
        a, b = self._cross("openai", OPENAI_TOOL_SSE, "anthropic")
        tca = extract_tool_calls(a)
        tcb = extract_tool_calls(b)
        assert len(tca) == len(tcb)
        assert tca[0]["name"] == tcb[0]["name"]
        assert extract_finish(a) == extract_finish(b)

    def test_tool_anthropic_to_openai(self):
        a, b = self._cross("anthropic", ANTHROPIC_TOOL_SSE, "openai")
        tca = extract_tool_calls(a)
        tcb = extract_tool_calls(b)
        assert len(tca) == len(tcb)
        assert tca[0]["name"] == tcb[0]["name"]

    def test_thinking_anthropic_to_openai(self):
        a, b = self._cross("anthropic", ANTHROPIC_THINKING_SSE, "openai")
        assert extract_text(a) == extract_text(b)
        assert extract_reasoning(a) == extract_reasoning(b)

    def test_thinking_anthropic_to_responses(self):
        a, b = self._cross("anthropic", ANTHROPIC_THINKING_SSE, "responses")
        assert extract_text(a) == extract_text(b)
        assert extract_reasoning(a) == extract_reasoning(b)


# ═══════════════════════════════════════════════════════════════════
# 4. RedactedReasoning 跨格式保留
# ═══════════════════════════════════════════════════════════════════


class TestRedactedReasoningPreservation:
    def test_redacted_through_responses_encoder(self):
        """Anthropic redacted_thinking → IR → Responses SSE → IR: data 保留"""
        ir_a = decode_stream("anthropic", ANTHROPIC_REDACTED_SSE)
        redacted_a = [e for e in ir_a if e["type"] == "redacted_reasoning"]
        assert len(redacted_a) == 1

        sse_rsp = encode_stream_raw("responses", ir_a)
        ir_b = decode_stream("responses", sse_rsp)
        assert extract_text(ir_b) == "Result."

    def test_redacted_through_openai_encoder(self):
        """Anthropic redacted_thinking → IR → OpenAI SSE → IR: 文本保留"""
        ir_a = decode_stream("anthropic", ANTHROPIC_REDACTED_SSE)
        sse_oai = encode_stream_raw("openai", ir_a)
        ir_b = decode_stream("openai", sse_oai)
        assert extract_text(ir_b) == "Result."


# ═══════════════════════════════════════════════════════════════════
# 5. 全链路管线（A → IR → B → IR → C → IR → 验证）
# ═══════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_three_hop_text(self):
        """OpenAI → Anthropic → Gemini → OpenAI: 文本不丢失"""
        ir_0 = decode_stream("openai", OPENAI_TEXT_SSE)
        text_0 = extract_text(ir_0)

        sse_ant = encode_stream_raw("anthropic", ir_0)
        ir_1 = decode_stream("anthropic", sse_ant)
        assert extract_text(ir_1) == text_0

        sse_gem = encode_stream_raw("gemini", ir_1)
        ir_2 = decode_stream("gemini", sse_gem)
        assert extract_text(ir_2) == text_0

        sse_oai = encode_stream_raw("openai", ir_2)
        ir_3 = decode_stream("openai", sse_oai)
        assert extract_text(ir_3) == text_0

    def test_three_hop_tools(self):
        """OpenAI tools → Anthropic → Gemini → OpenAI: 工具调用不丢失"""
        ir_0 = decode_stream("openai", OPENAI_TOOL_SSE)
        tc_0 = extract_tool_calls(ir_0)
        assert len(tc_0) == 1

        sse_ant = encode_stream_raw("anthropic", ir_0)
        ir_1 = decode_stream("anthropic", sse_ant)
        tc_1 = extract_tool_calls(ir_1)
        assert len(tc_1) == 1
        assert tc_1[0]["name"] == tc_0[0]["name"]

        sse_gem = encode_stream_raw("gemini", ir_1)
        ir_2 = decode_stream("gemini", sse_gem)
        tc_2 = extract_tool_calls(ir_2)
        assert len(tc_2) == 1
        assert tc_2[0]["name"] == tc_0[0]["name"]

    def test_four_hop_text(self):
        """OpenAI → Anthropic → Gemini → Responses → OpenAI"""
        ir = decode_stream("openai", OPENAI_TEXT_SSE)
        text = extract_text(ir)

        for fmt in ["anthropic", "gemini", "responses", "openai"]:
            sse = encode_stream_raw(fmt, ir)
            ir = decode_stream(fmt, sse)
            assert extract_text(ir) == text, f"文本在 → {fmt} 后丢失"
