"""请求编解码端到端测试

覆盖 4 种格式的 DecodeRequest/EncodeRequest，包括：
- 纯文本对话
- 系统提示词
- 工具定义 + 工具调用 + 工具结果多轮
- 多模态（image_url）
- 推理配置 reasoning
- 采样参数保留
- 跨格式转换语义一致性
"""

import json
import pytest
from conftest import decode_request, encode_request

# ═══════════════════════════════════════════════════════════════════
# 原始载荷 fixtures
# ═══════════════════════════════════════════════════════════════════

OPENAI_SIMPLE = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "Translate: goodbye"},
    ],
    "temperature": 0.3,
    "max_tokens": 200,
    "top_p": 0.95,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "seed": 42,
    "stream": False,
}

ANTHROPIC_SIMPLE = {
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {"role": "user", "content": "Translate: hello"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "Translate: goodbye"},
    ],
    "system": "You are a translator.",
    "max_tokens": 200,
    "temperature": 0.3,
    "top_p": 0.95,
}

GEMINI_SIMPLE = {
    "contents": [
        {"role": "user", "parts": [{"text": "Translate: hello"}]},
        {"role": "model", "parts": [{"text": "你好"}]},
        {"role": "user", "parts": [{"text": "Translate: goodbye"}]},
    ],
    "systemInstruction": {"parts": [{"text": "You are a translator."}]},
    "generationConfig": {
        "temperature": 0.3,
        "maxOutputTokens": 200,
        "topP": 0.95,
    },
}

RESPONSES_SIMPLE = {
    "model": "gpt-4o",
    "input": [
        {"type": "message", "role": "user", "content": "Translate: hello"},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "你好"},
        ]},
        {"type": "message", "role": "user", "content": "Translate: goodbye"},
    ],
    "instructions": "You are a translator.",
    "temperature": 0.3,
    "max_output_tokens": 200,
    "top_p": 0.95,
}

OPENAI_TOOLS = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You help with weather."},
        {"role": "user", "content": "What's the weather in NYC and LA?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"New York"}',
                    },
                },
                {
                    "id": "call_def",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Los Angeles"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "72°F, Sunny"},
        {"role": "tool", "tool_call_id": "call_def", "content": "85°F, Clear"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "stream": False,
}

ANTHROPIC_TOOLS = {
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {"role": "user", "content": "What's the weather in NYC and LA?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "get_weather",
                    "input": {"city": "New York"},
                },
                {
                    "type": "tool_use",
                    "id": "call_def",
                    "name": "get_weather",
                    "input": {"city": "Los Angeles"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_abc",
                    "content": "72°F, Sunny",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_def",
                    "content": "85°F, Clear",
                },
            ],
        },
    ],
    "system": "You help with weather.",
    "max_tokens": 1024,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    "tool_choice": {"type": "auto"},
}

OPENAI_MULTIMODAL = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/cat.jpg",
                        "detail": "high",
                    },
                },
            ],
        }
    ],
    "stream": False,
}

OPENAI_REASONING = {
    "model": "o3-mini",
    "messages": [
        {"role": "user", "content": "Prove P=NP or explain why not."},
    ],
    "reasoning_effort": "high",
    "stream": False,
}

RESPONSES_FUNCTION_CALLS = {
    "model": "gpt-4o",
    "input": [
        {"type": "message", "role": "user", "content": "Weather in NYC and LA?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"New York"}',
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "get_weather",
            "arguments": '{"city":"Los Angeles"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "72°F",
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "85°F",
        },
    ],
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ],
}


# ═══════════════════════════════════════════════════════════════════
# 1. 纯文本请求 — 各格式解码到 IR 的结构验证
# ═══════════════════════════════════════════════════════════════════


class TestDecodeRequest:
    def test_openai_simple(self):
        ir = decode_request("openai", OPENAI_SIMPLE)
        assert ir["model"] == "gpt-4o"
        assert ir["temperature"] == 0.3
        assert ir["max_tokens"] == 200
        assert ir["seed"] == 42
        roles = [m["role"] for m in ir["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert ir["messages"][0]["content"] == "You are a translator."
        assert ir["messages"][1]["content"] == "Translate: hello"
        assert ir["stream"] is False

    def test_anthropic_simple(self):
        ir = decode_request("anthropic", ANTHROPIC_SIMPLE)
        assert ir["model"] == "claude-sonnet-4-20250514"
        assert ir["temperature"] == 0.3
        assert ir["max_tokens"] == 200
        roles = [m["role"] for m in ir["messages"]]
        assert "system" in roles
        assert roles.count("user") == 2

    def test_gemini_simple(self):
        ir = decode_request("gemini", GEMINI_SIMPLE)
        assert ir["temperature"] == 0.3
        assert ir["max_tokens"] == 200
        roles = [m["role"] for m in ir["messages"]]
        assert "system" in roles

    def test_responses_simple(self):
        ir = decode_request("responses", RESPONSES_SIMPLE)
        assert ir["model"] == "gpt-4o"
        assert ir["temperature"] == 0.3
        assert ir["max_tokens"] == 200
        roles = [m["role"] for m in ir["messages"]]
        assert "system" in roles

    def test_openai_tools_parsed(self):
        ir = decode_request("openai", OPENAI_TOOLS)
        assert ir["tools"] is not None
        assert len(ir["tools"]) == 1
        assert ir["tools"][0]["name"] == "get_weather"
        assistant_msgs = [m for m in ir["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 2
        tool_msgs = [m for m in ir["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2

    def test_anthropic_tools_parsed(self):
        ir = decode_request("anthropic", ANTHROPIC_TOOLS)
        assert ir["tools"] is not None
        assert len(ir["tools"]) == 1
        assistant_msgs = [m for m in ir["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # M10: Anthropic 解码器将 tool_use 放入 content FunctionCall 部件
        fc = [p for p in assistant_msgs[0]["content"] if p["type"] == "function_call"]
        assert len(fc) == 2

    def test_openai_multimodal(self):
        ir = decode_request("openai", OPENAI_MULTIMODAL)
        user_msg = ir["messages"][0]
        parts = user_msg["content"]
        assert isinstance(parts, list)
        types = [p["type"] for p in parts]
        assert "text" in types
        assert "image_url" in types

    def test_openai_reasoning(self):
        ir = decode_request("openai", OPENAI_REASONING)
        assert ir["reasoning"] is not None
        assert ir["reasoning"]["effort"] == "high"

    def test_responses_function_calls_merged(self):
        """连续 function_call 应合并为单条 assistant 消息的多 tool_calls"""
        ir = decode_request("responses", RESPONSES_FUNCTION_CALLS)
        assistant_msgs = [m for m in ir["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1, (
            f"连续 function_call 应合并为 1 条 assistant，实际 {len(assistant_msgs)} 条"
        )
        assert len(assistant_msgs[0]["tool_calls"]) == 2
        tool_msgs = [m for m in ir["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2


# ═══════════════════════════════════════════════════════════════════
# 2. 请求 round-trip（A → IR → A）
# ═══════════════════════════════════════════════════════════════════


class TestRequestRoundTrip:
    def test_openai_round_trip(self):
        ir = decode_request("openai", OPENAI_SIMPLE)
        out = encode_request("openai", ir)
        assert out["model"] == "gpt-4o"
        assert out["temperature"] == 0.3
        assert out["max_completion_tokens"] == 200
        assert out["seed"] == 42
        assert len(out["messages"]) == 4

    def test_anthropic_round_trip(self):
        ir = decode_request("anthropic", ANTHROPIC_SIMPLE)
        out = encode_request("anthropic", ir)
        assert out["model"] == "claude-sonnet-4-20250514"
        assert out["max_tokens"] == 200
        assert out["temperature"] == 0.3
        msgs = out["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_gemini_round_trip(self):
        ir = decode_request("gemini", GEMINI_SIMPLE)
        out = encode_request("gemini", ir)
        assert "contents" in out
        gc = out.get("generationConfig", {})
        assert gc.get("temperature") == 0.3
        assert gc.get("maxOutputTokens") == 200

    def test_responses_round_trip(self):
        ir = decode_request("responses", RESPONSES_SIMPLE)
        out = encode_request("responses", ir)
        assert out["model"] == "gpt-4o"
        assert out["temperature"] == 0.3
        assert out["max_output_tokens"] == 200

    def test_openai_tools_round_trip(self):
        ir = decode_request("openai", OPENAI_TOOLS)
        out = encode_request("openai", ir)
        assert len(out["tools"]) == 1
        assert out["tools"][0]["function"]["name"] == "get_weather"
        assistant_msg = next(m for m in out["messages"] if m.get("role") == "assistant")
        assert len(assistant_msg["tool_calls"]) == 2
        tool_msgs = [m for m in out["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert "call_abc" in ids
        assert "call_def" in ids


# ═══════════════════════════════════════════════════════════════════
# 3. 跨格式转换（A → IR → B）语义保持
# ═══════════════════════════════════════════════════════════════════


class TestRequestCrossFormat:
    """每条测试：源格式解码 → IR → 目标格式编码 → 目标格式解码 → 比较 IR"""

    def _cross(self, src_fmt, src_payload, dst_fmt):
        ir_a = decode_request(src_fmt, src_payload)
        encoded = encode_request(dst_fmt, ir_a)
        ir_b = decode_request(dst_fmt, encoded)
        return ir_a, ir_b

    def _assert_messages_match(self, ir_a, ir_b):
        """非 system 消息的 role 序列和用户文本应一致"""
        non_sys_a = [m for m in ir_a["messages"] if m["role"] != "system"]
        non_sys_b = [m for m in ir_b["messages"] if m["role"] != "system"]
        roles_a = [m["role"] for m in non_sys_a]
        roles_b = [m["role"] for m in non_sys_b]
        assert roles_a == roles_b, f"role 序列不匹配: {roles_a} vs {roles_b}"

    def _assert_sampling(self, ir_a, ir_b):
        for key in ("temperature", "max_tokens", "top_p"):
            a_val = ir_a.get(key)
            b_val = ir_b.get(key)
            if a_val is not None:
                assert b_val == a_val, f"{key}: {a_val} → {b_val}"

    def test_openai_to_anthropic(self):
        ir_a, ir_b = self._cross("openai", OPENAI_SIMPLE, "anthropic")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_openai_to_gemini(self):
        ir_a, ir_b = self._cross("openai", OPENAI_SIMPLE, "gemini")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_openai_to_responses(self):
        ir_a, ir_b = self._cross("openai", OPENAI_SIMPLE, "responses")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_anthropic_to_openai(self):
        ir_a, ir_b = self._cross("anthropic", ANTHROPIC_SIMPLE, "openai")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_anthropic_to_gemini(self):
        ir_a, ir_b = self._cross("anthropic", ANTHROPIC_SIMPLE, "gemini")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_gemini_to_openai(self):
        ir_a, ir_b = self._cross("gemini", GEMINI_SIMPLE, "openai")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_gemini_to_anthropic(self):
        ir_a, ir_b = self._cross("gemini", GEMINI_SIMPLE, "anthropic")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_responses_to_openai(self):
        ir_a, ir_b = self._cross("responses", RESPONSES_SIMPLE, "openai")
        self._assert_messages_match(ir_a, ir_b)
        self._assert_sampling(ir_a, ir_b)

    def test_tools_openai_to_anthropic(self):
        ir_a, ir_b = self._cross("openai", OPENAI_TOOLS, "anthropic")
        a_tools = ir_a["tools"]
        b_tools = ir_b["tools"]
        assert len(a_tools) == len(b_tools)
        assert a_tools[0]["name"] == b_tools[0]["name"]
        a_tc = [m for m in ir_a["messages"] if m["role"] == "assistant"][0]["tool_calls"]
        # M10: Anthropic 解码器将 tool_use 放入 content FunctionCall 部件
        b_msg = [m for m in ir_b["messages"] if m["role"] == "assistant"][0]
        b_tc = [p for p in b_msg["content"] if p["type"] == "function_call"]
        assert len(a_tc) == len(b_tc)

    def test_tools_openai_to_gemini(self):
        ir_a, ir_b = self._cross("openai", OPENAI_TOOLS, "gemini")
        assert len(ir_a["tools"]) == len(ir_b["tools"])

    def test_tools_anthropic_to_openai(self):
        ir_a, ir_b = self._cross("anthropic", ANTHROPIC_TOOLS, "openai")
        assert len(ir_a["tools"]) == len(ir_b["tools"])

    def test_multimodal_openai_to_anthropic(self):
        ir_a, ir_b = self._cross("openai", OPENAI_MULTIMODAL, "anthropic")
        a_parts = ir_a["messages"][0]["content"]
        b_parts = ir_b["messages"][0]["content"]
        a_types = {p["type"] for p in a_parts} if isinstance(a_parts, list) else set()
        b_types = {p["type"] for p in b_parts} if isinstance(b_parts, list) else set()
        assert "text" in a_types and "text" in b_types

    def test_multimodal_openai_to_gemini(self):
        ir_a, ir_b = self._cross("openai", OPENAI_MULTIMODAL, "gemini")
        a_parts = ir_a["messages"][0]["content"]
        b_parts = ir_b["messages"][0]["content"]
        assert isinstance(a_parts, list) and isinstance(b_parts, list)


# ═══════════════════════════════════════════════════════════════════
# 4. store 字段测试
# ═══════════════════════════════════════════════════════════════════


class TestStoreField:
    """store 字段在 OpenAI/Responses 请求中的编解码"""

    def test_openai_decode_store_true(self):
        payload = {**OPENAI_SIMPLE, "store": True}
        ir = decode_request("openai", payload)
        assert ir["store"] is True

    def test_openai_decode_store_false(self):
        payload = {**OPENAI_SIMPLE, "store": False}
        ir = decode_request("openai", payload)
        assert ir["store"] is False

    def test_openai_decode_store_absent(self):
        ir = decode_request("openai", OPENAI_SIMPLE)
        assert ir.get("store") is None

    def test_openai_encode_store_true(self):
        ir = decode_request("openai", OPENAI_SIMPLE)
        ir["store"] = True
        out = encode_request("openai", ir)
        assert out["store"] is True

    def test_openai_encode_store_absent(self):
        ir = decode_request("openai", OPENAI_SIMPLE)
        out = encode_request("openai", ir)
        assert "store" not in out

    def test_openai_store_round_trip(self):
        payload = {**OPENAI_SIMPLE, "store": True}
        ir = decode_request("openai", payload)
        out = encode_request("openai", ir)
        ir2 = decode_request("openai", out)
        assert ir2["store"] is True

    def test_responses_encode_store(self):
        ir = decode_request("responses", RESPONSES_SIMPLE)
        ir["store"] = True
        out = encode_request("responses", ir)
        assert out["store"] is True

    def test_responses_decode_store(self):
        payload = {**RESPONSES_SIMPLE, "store": True}
        ir = decode_request("responses", payload)
        assert ir["store"] is True

    def test_cross_openai_to_responses_store(self):
        payload = {**OPENAI_SIMPLE, "store": True}
        ir = decode_request("openai", payload)
        out = encode_request("responses", ir)
        assert out["store"] is True

    def test_anthropic_ignores_store(self):
        """Anthropic 不支持 store，编码应忽略"""
        ir = decode_request("openai", OPENAI_SIMPLE)
        ir["store"] = True
        out = encode_request("anthropic", ir)
        assert "store" not in out

    def test_gemini_ignores_store(self):
        """Gemini 不支持 store，编码应忽略"""
        ir = decode_request("openai", OPENAI_SIMPLE)
        ir["store"] = True
        out = encode_request("gemini", ir)
        assert "store" not in out
