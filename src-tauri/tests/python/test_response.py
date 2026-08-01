"""响应编解码端到端测试

覆盖 4 种格式的 DecodeResponse/EncodeResponse：
- 纯文本响应
- 带工具调用的响应
- usage 字段保留
- finish_reason 映射
- 跨格式 round-trip
"""

import json
import pytest
from conftest import decode_response, encode_response

# ═══════════════════════════════════════════════════════════════════
# 原始载荷
# ═══════════════════════════════════════════════════════════════════

OPENAI_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 12,
        "total_tokens": 27,
    },
}

ANTHROPIC_RESPONSE = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [{"type": "text", "text": "你好！有什么可以帮助你的？"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 15, "output_tokens": 12},
}

GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "你好！有什么可以帮助你的？"}],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 15,
        "candidatesTokenCount": 12,
        "totalTokenCount": 27,
    },
}

RESPONSES_RESPONSE = {
    "id": "resp_abc123",
    "object": "response",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "你好！有什么可以帮助你的？"}],
        }
    ],
    "status": "completed",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 12,
        "total_tokens": 27,
    },
}

OPENAI_RESPONSE_TOOLS = {
    "id": "chatcmpl-tools",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"LA"}',
                        },
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
}

ANTHROPIC_RESPONSE_TOOLS = {
    "id": "msg_tools",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        },
        {
            "type": "tool_use",
            "id": "call_2",
            "name": "get_weather",
            "input": {"city": "LA"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 30},
}


# ═══════════════════════════════════════════════════════════════════
# 1. 各格式解码验证
# ═══════════════════════════════════════════════════════════════════


class TestDecodeResponse:
    def test_openai(self):
        ir = decode_response("openai", OPENAI_RESPONSE)
        assert ir["model"] == "gpt-4o"
        assert len(ir["choices"]) == 1
        assert "你好" in ir["choices"][0]["message"]["content"]
        assert ir["choices"][0]["finish_reason"] == "stop"
        assert ir["usage"]["prompt_tokens"] == 15
        assert ir["usage"]["completion_tokens"] == 12

    def test_anthropic(self):
        ir = decode_response("anthropic", ANTHROPIC_RESPONSE)
        assert ir["model"] == "claude-sonnet-4-20250514"
        assert len(ir["choices"]) == 1
        assert ir["choices"][0]["finish_reason"] == "stop"
        assert ir["usage"]["prompt_tokens"] == 15

    def test_gemini(self):
        ir = decode_response("gemini", GEMINI_RESPONSE)
        assert len(ir["choices"]) == 1
        assert ir["choices"][0]["finish_reason"] == "stop"
        assert ir["usage"]["prompt_tokens"] == 15

    def test_responses(self):
        ir = decode_response("responses", RESPONSES_RESPONSE)
        assert ir["model"] == "gpt-4o"
        assert len(ir["choices"]) == 1
        assert ir["choices"][0]["finish_reason"] == "stop"

    def test_openai_tools(self):
        ir = decode_response("openai", OPENAI_RESPONSE_TOOLS)
        tc = ir["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["name"] == "get_weather"
        assert ir["choices"][0]["finish_reason"] == "tool_calls"

    def test_anthropic_tools(self):
        ir = decode_response("anthropic", ANTHROPIC_RESPONSE_TOOLS)
        # M10: Anthropic 解码器将 tool_use 放入 content FunctionCall 部件
        content = ir["choices"][0]["message"]["content"]
        tc = [p for p in content if p["type"] == "function_call"]
        assert len(tc) == 2
        assert tc[0]["name"] == "get_weather"
        assert ir["choices"][0]["finish_reason"] == "tool_calls"


# ═══════════════════════════════════════════════════════════════════
# 2. 响应 round-trip（A → IR → A）
# ═══════════════════════════════════════════════════════════════════


class TestResponseRoundTrip:
    def _round_trip(self, fmt, payload):
        ir = decode_response(fmt, payload)
        out = encode_response(fmt, ir)
        ir2 = decode_response(fmt, out)
        return ir, ir2

    def test_openai(self):
        ir1, ir2 = self._round_trip("openai", OPENAI_RESPONSE)
        assert ir1["choices"][0]["message"]["content"] == ir2["choices"][0]["message"]["content"]
        assert ir1["choices"][0]["finish_reason"] == ir2["choices"][0]["finish_reason"]
        assert ir1["usage"]["prompt_tokens"] == ir2["usage"]["prompt_tokens"]

    def test_anthropic(self):
        ir1, ir2 = self._round_trip("anthropic", ANTHROPIC_RESPONSE)
        c1 = ir1["choices"][0]["message"]["content"]
        c2 = ir2["choices"][0]["message"]["content"]
        # content 可能是 string 或 parts，提取文本比较
        t1 = c1 if isinstance(c1, str) else "".join(
            p.get("text", "") for p in c1 if isinstance(p, dict)
        )
        t2 = c2 if isinstance(c2, str) else "".join(
            p.get("text", "") for p in c2 if isinstance(p, dict)
        )
        assert t1 == t2

    def test_gemini(self):
        ir1, ir2 = self._round_trip("gemini", GEMINI_RESPONSE)
        assert ir1["choices"][0]["finish_reason"] == ir2["choices"][0]["finish_reason"]

    def test_responses(self):
        ir1, ir2 = self._round_trip("responses", RESPONSES_RESPONSE)
        assert ir1["choices"][0]["finish_reason"] == ir2["choices"][0]["finish_reason"]

    def test_openai_tools_round_trip(self):
        ir1, ir2 = self._round_trip("openai", OPENAI_RESPONSE_TOOLS)
        tc1 = ir1["choices"][0]["message"]["tool_calls"]
        tc2 = ir2["choices"][0]["message"]["tool_calls"]
        assert len(tc1) == len(tc2)
        for a, b in zip(tc1, tc2):
            assert a["name"] == b["name"]
            assert a["id"] == b["id"]


# ═══════════════════════════════════════════════════════════════════
# 3. 跨格式响应转换
# ═══════════════════════════════════════════════════════════════════


class TestResponseCrossFormat:
    def _cross(self, src_fmt, src_payload, dst_fmt):
        ir = decode_response(src_fmt, src_payload)
        encoded = encode_response(dst_fmt, ir)
        ir2 = decode_response(dst_fmt, encoded)
        return ir, ir2

    def _get_text(self, ir):
        content = ir["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return ""

    def test_openai_to_anthropic(self):
        ir1, ir2 = self._cross("openai", OPENAI_RESPONSE, "anthropic")
        assert self._get_text(ir1) == self._get_text(ir2)
        assert ir1["choices"][0]["finish_reason"] == ir2["choices"][0]["finish_reason"]

    def test_openai_to_gemini(self):
        ir1, ir2 = self._cross("openai", OPENAI_RESPONSE, "gemini")
        assert self._get_text(ir1) == self._get_text(ir2)

    def test_openai_to_responses(self):
        ir1, ir2 = self._cross("openai", OPENAI_RESPONSE, "responses")
        assert self._get_text(ir1) == self._get_text(ir2)

    def test_anthropic_to_openai(self):
        ir1, ir2 = self._cross("anthropic", ANTHROPIC_RESPONSE, "openai")
        assert self._get_text(ir1) == self._get_text(ir2)

    def test_gemini_to_anthropic(self):
        ir1, ir2 = self._cross("gemini", GEMINI_RESPONSE, "anthropic")
        assert self._get_text(ir1) == self._get_text(ir2)

    def test_tools_openai_to_anthropic(self):
        ir1, ir2 = self._cross("openai", OPENAI_RESPONSE_TOOLS, "anthropic")
        tc1 = ir1["choices"][0]["message"]["tool_calls"]
        # M10: Anthropic 解码器将 tool_use 放入 content FunctionCall 部件
        content2 = ir2["choices"][0]["message"]["content"]
        tc2 = [p for p in content2 if p["type"] == "function_call"]
        assert len(tc1) == len(tc2)
        for a, b in zip(tc1, tc2):
            assert a["name"] == b["name"]

    def test_tools_anthropic_to_openai(self):
        ir1, ir2 = self._cross("anthropic", ANTHROPIC_RESPONSE_TOOLS, "openai")
        # M10: ir1 来自 Anthropic 解码 → FunctionCall 在 content 中
        content1 = ir1["choices"][0]["message"]["content"]
        tc1 = [p for p in content1 if p["type"] == "function_call"]
        tc2 = ir2["choices"][0]["message"]["tool_calls"]
        assert len(tc1) == len(tc2)


# ═══════════════════════════════════════════════════════════════════
# 4. Logprobs 测试
# ═══════════════════════════════════════════════════════════════════

OPENAI_RESPONSE_LOGPROBS = {
    "id": "chatcmpl-lp",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "Hello",
                        "logprob": -0.5,
                        "bytes": [72, 101, 108, 108, 111],
                        "top_logprobs": [
                            {"token": "Hello", "logprob": -0.5, "bytes": [72, 101, 108, 108, 111]},
                            {"token": "Hi", "logprob": -1.2, "bytes": [72, 105]},
                        ],
                    }
                ],
                "refusal": None,
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

GEMINI_RESPONSE_LOGPROBS = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "Hello"}],
            },
            "finishReason": "STOP",
            "index": 0,
            "avgLogprobs": -0.75,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 1,
        "totalTokenCount": 6,
    },
}


class TestLogprobs:
    def test_openai_decode_logprobs(self):
        ir = decode_response("openai", OPENAI_RESPONSE_LOGPROBS)
        lp = ir["choices"][0]["logprobs"]
        assert lp is not None
        assert len(lp["content"]) == 1
        tok = lp["content"][0]
        assert tok["token"] == "Hello"
        assert tok["logprob"] == -0.5
        assert tok["bytes"] == [72, 101, 108, 108, 111]
        assert len(tok["top_logprobs"]) == 2
        assert tok["top_logprobs"][1]["token"] == "Hi"

    def test_openai_logprobs_round_trip(self):
        ir = decode_response("openai", OPENAI_RESPONSE_LOGPROBS)
        out = encode_response("openai", ir)
        ir2 = decode_response("openai", out)
        lp1 = ir["choices"][0]["logprobs"]
        lp2 = ir2["choices"][0]["logprobs"]
        assert lp1["content"][0]["token"] == lp2["content"][0]["token"]
        assert lp1["content"][0]["logprob"] == lp2["content"][0]["logprob"]
        assert len(lp1["content"][0]["top_logprobs"]) == len(lp2["content"][0]["top_logprobs"])

    def test_openai_no_logprobs(self):
        ir = decode_response("openai", OPENAI_RESPONSE)
        assert ir["choices"][0].get("logprobs") is None

    def test_gemini_decode_avg_logprobs(self):
        ir = decode_response("gemini", GEMINI_RESPONSE_LOGPROBS)
        lp = ir["choices"][0]["logprobs"]
        assert lp is not None
        assert lp["avg_logprob"] == -0.75

    def test_gemini_avg_logprobs_round_trip(self):
        ir = decode_response("gemini", GEMINI_RESPONSE_LOGPROBS)
        out = encode_response("gemini", ir)
        ir2 = decode_response("gemini", out)
        assert ir2["choices"][0]["logprobs"]["avg_logprob"] == -0.75

    def test_openai_logprobs_encode_format(self):
        """EncodeResponse 编码的 logprobs 结构应符合 OpenAI 格式"""
        ir = decode_response("openai", OPENAI_RESPONSE_LOGPROBS)
        out = encode_response("openai", ir)
        lp = out["choices"][0]["logprobs"]
        assert "content" in lp
        assert lp["content"][0]["token"] == "Hello"
        assert "top_logprobs" in lp["content"][0]

    def test_logprobs_cross_format_preserved_as_none(self):
        """Anthropic 不支持 logprobs，OpenAI logprobs 跨格式后丢失"""
        ir = decode_response("openai", OPENAI_RESPONSE_LOGPROBS)
        out = encode_response("anthropic", ir)
        ir2 = decode_response("anthropic", out)
        assert ir2["choices"][0].get("logprobs") is None
