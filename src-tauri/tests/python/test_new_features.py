"""新特性端到端测试

覆盖新实现的编解码能力：
1. 工具类型 IrToolType（function / web_search / code_interpreter / file_search /
   computer_use / text_editor / mcp）在 4 种格式下的解码、编码、跨格式与往返。
2. Modalities（仅 OpenAI Chat Completions 支持）。
3. Refusal（OpenAI 响应 message.refusal）。
4. 流式 Logprobs 事件（IrStreamEvent type="logprobs"）。
5. Annotations（OpenAI web search 引文 → 文本部件 citations）。
6. 流式推理签名（reasoning_signature → ReasoningDone.signature）。
7. P2 特性：Responses include/background、Gemini cachedContent、OpenAI prediction/audio
   通过 provider_metadata 往返。

测试模式约定见 conftest.py：
- decode_request(fmt, raw)   → IR dict
- encode_request(fmt, ir)    → vendor dict
- decode_response/encode_response 同理
- decode_stream(fmt, sse)    → [IR event dict]
- encode_stream_raw(fmt, [ir event]) → SSE 文本
"""

import json
import pytest

from conftest import (
    decode_request,
    encode_request,
    decode_response,
    encode_response,
    decode_stream,
    encode_stream_raw,
)

# ═══════════════════════════════════════════════════════════════════
# 最小载荷构造器 — 只包含各格式必需字段
# ═══════════════════════════════════════════════════════════════════


def openai_req(**extra):
    return {
        "model": "gpt-4o",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def anthropic_req(**extra):
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def gemini_req(**extra):
    return {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **extra,
    }


def responses_req(**extra):
    return {
        "model": "gpt-4o",
        "input": "hi",
        **extra,
    }


def openai_resp(message, **extra):
    return {
        "id": "chatcmpl-nf",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        **extra,
    }


def _text_parts(content):
    """把 message.content 归一化为部件列表（可能是 str 或 list）"""
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
    return []


# ═══════════════════════════════════════════════════════════════════
# 1. 工具类型 IrToolType
# ═══════════════════════════════════════════════════════════════════


class TestToolTypes:
    # ── OpenAI Chat Completions ──

    def test_openai_function_decode(self):
        """OpenAI function 工具 → IR tool_type=function"""
        ir = decode_request(
            "openai",
            openai_req(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
        )
        assert ir["tools"] is not None
        assert len(ir["tools"]) == 1
        assert ir["tools"][0]["tool_type"] == "function"
        assert ir["tools"][0]["name"] == "get_weather"

    def test_openai_encode_web_search(self):
        """IR tool_type=web_search → OpenAI 输出 {"type":"web_search"}"""
        ir = openai_req()
        ir["tools"] = [{"tool_type": "web_search", "name": "web_search", "parameters": {}}]
        out = encode_request("openai", ir)
        assert len(out["tools"]) == 1
        assert out["tools"][0]["type"] == "web_search"

    # ── Anthropic ──

    def test_anthropic_computer_use_decode(self):
        """Anthropic computer_use → IR tool_type=computer_use，版本与尺寸落在 extra"""
        ir = decode_request(
            "anthropic",
            anthropic_req(
                tools=[
                    {
                        "type": "computer_20241022",
                        "name": "computer",
                        "display_width_px": 1024,
                        "display_height_px": 768,
                    }
                ]
            ),
        )
        tool = ir["tools"][0]
        assert tool["tool_type"] == "computer_use"
        assert tool["name"] == "computer"
        # 版本字符串保留在 extra.type，尺寸保留原始键名
        assert tool["extra"]["type"] == "computer_20241022"
        assert tool["extra"]["display_width_px"] == 1024
        assert tool["extra"]["display_height_px"] == 768

    def test_anthropic_computer_use_round_trip(self):
        """computer_use decode → encode → decode 保留 tool_type 与 extra 全字段"""
        payload = anthropic_req(
            tools=[
                {
                    "type": "computer_20241022",
                    "name": "computer",
                    "display_width_px": 1024,
                    "display_height_px": 768,
                }
            ]
        )
        ir = decode_request("anthropic", payload)
        out = encode_request("anthropic", ir)
        ir2 = decode_request("anthropic", out)
        tool = ir2["tools"][0]
        assert tool["tool_type"] == "computer_use"
        assert tool["extra"]["type"] == "computer_20241022"
        assert tool["extra"]["display_width_px"] == 1024
        assert tool["extra"]["display_height_px"] == 768

    def test_anthropic_text_editor_decode(self):
        """Anthropic text_editor → IR tool_type=text_editor，版本落在 extra"""
        ir = decode_request(
            "anthropic",
            anthropic_req(
                tools=[{"type": "text_editor_20241022", "name": "str_replace_editor"}]
            ),
        )
        tool = ir["tools"][0]
        assert tool["tool_type"] == "text_editor"
        assert tool["name"] == "str_replace_editor"
        assert tool["extra"]["type"] == "text_editor_20241022"

    def test_anthropic_text_editor_round_trip(self):
        """text_editor decode → encode → decode 保留 tool_type 与版本"""
        payload = anthropic_req(
            tools=[{"type": "text_editor_20241022", "name": "str_replace_editor"}]
        )
        ir = decode_request("anthropic", payload)
        out = encode_request("anthropic", ir)
        ir2 = decode_request("anthropic", out)
        tool = ir2["tools"][0]
        assert tool["tool_type"] == "text_editor"
        assert tool["extra"]["type"] == "text_editor_20241022"

    # ── Gemini ──

    def test_gemini_code_execution_decode(self):
        """Gemini codeExecution → IR tool_type=code_interpreter"""
        ir = decode_request("gemini", gemini_req(tools=[{"codeExecution": {}}]))
        assert len(ir["tools"]) == 1
        assert ir["tools"][0]["tool_type"] == "code_interpreter"

    def test_gemini_google_search_retrieval_decode(self):
        """Gemini googleSearchRetrieval → IR tool_type=web_search"""
        ir = decode_request("gemini", gemini_req(tools=[{"googleSearchRetrieval": {}}]))
        assert len(ir["tools"]) == 1
        assert ir["tools"][0]["tool_type"] == "web_search"

    def test_gemini_encode_code_interpreter(self):
        """IR tool_type=code_interpreter → Gemini 输出 {"codeExecution":{}}"""
        ir = gemini_req()
        ir = decode_request("gemini", ir)  # 归一化到 IR
        ir["tools"] = [
            {"tool_type": "code_interpreter", "name": "code_execution", "parameters": {}}
        ]
        out = encode_request("gemini", ir)
        assert len(out["tools"]) == 1
        assert "codeExecution" in out["tools"][0]

    # ── Responses API ──

    def test_responses_web_search_decode(self):
        """Responses web_search_preview → IR tool_type=web_search"""
        ir = decode_request(
            "responses", responses_req(tools=[{"type": "web_search_preview"}])
        )
        assert len(ir["tools"]) == 1
        assert ir["tools"][0]["tool_type"] == "web_search"

    def test_responses_web_search_round_trip(self):
        """web_search 工具 decode → encode → decode 保留 tool_type"""
        ir = decode_request(
            "responses", responses_req(tools=[{"type": "web_search_preview"}])
        )
        out = encode_request("responses", ir)
        # 编码回 Responses 时规范化为 web_search_preview
        assert out["tools"][0]["type"] == "web_search_preview"
        ir2 = decode_request("responses", out)
        assert ir2["tools"][0]["tool_type"] == "web_search"

    # ── 跨格式 ──

    def test_cross_openai_function_to_anthropic(self):
        """OpenAI function 工具 → Anthropic：仍为 function 工具"""
        ir = decode_request(
            "openai",
            openai_req(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
        )
        ant = encode_request("anthropic", ir)
        # Anthropic function 工具形态：有 name + input_schema，无 type
        assert ant["tools"][0]["name"] == "get_weather"
        assert "input_schema" in ant["tools"][0]
        assert "type" not in ant["tools"][0]
        ir_b = decode_request("anthropic", ant)
        assert ir_b["tools"][0]["tool_type"] == "function"

    def test_cross_gemini_code_exec_to_openai(self):
        """Gemini codeExecution → OpenAI：编码为 code_interpreter 类型"""
        ir = decode_request("gemini", gemini_req(tools=[{"codeExecution": {}}]))
        oai = encode_request("openai", ir)
        assert oai["tools"][0]["type"] == "code_interpreter"


# ═══════════════════════════════════════════════════════════════════
# 2. Modalities（仅 OpenAI Chat Completions）
# ═══════════════════════════════════════════════════════════════════


class TestModalities:
    def test_openai_decode(self):
        ir = decode_request("openai", openai_req(modalities=["text", "audio"]))
        assert ir["modalities"] == ["text", "audio"]

    def test_openai_encode(self):
        ir = openai_req()
        ir["modalities"] = ["text", "audio"]
        out = encode_request("openai", ir)
        assert out["modalities"] == ["text", "audio"]

    def test_openai_round_trip(self):
        ir = decode_request("openai", openai_req(modalities=["text", "audio"]))
        out = encode_request("openai", ir)
        ir2 = decode_request("openai", out)
        assert ir2["modalities"] == ["text", "audio"]

    def test_openai_absent(self):
        ir = decode_request("openai", openai_req())
        assert ir.get("modalities") is None

    def test_anthropic_ignores_modalities(self):
        """Anthropic 不支持 modalities，编码应忽略"""
        ir = decode_request("openai", openai_req(modalities=["text", "audio"]))
        out = encode_request("anthropic", ir)
        assert "modalities" not in out

    def test_gemini_ignores_modalities(self):
        """Gemini 不支持 modalities，编码应忽略"""
        ir = decode_request("openai", openai_req(modalities=["text", "audio"]))
        out = encode_request("gemini", ir)
        assert "modalities" not in out


# ═══════════════════════════════════════════════════════════════════
# 3. Refusal（OpenAI 响应 message.refusal）
# ═══════════════════════════════════════════════════════════════════


class TestRefusal:
    def test_openai_decode(self):
        ir = decode_response(
            "openai",
            openai_resp(
                {"role": "assistant", "refusal": "I cannot help with that"}
            ),
        )
        assert ir["choices"][0]["message"]["refusal"] == "I cannot help with that"

    def test_openai_encode(self):
        ir = decode_response(
            "openai",
            openai_resp(
                {"role": "assistant", "refusal": "I cannot help with that"}
            ),
        )
        out = encode_response("openai", ir)
        assert out["choices"][0]["message"]["refusal"] == "I cannot help with that"

    def test_openai_round_trip(self):
        ir = decode_response(
            "openai",
            openai_resp(
                {"role": "assistant", "refusal": "I cannot help with that"}
            ),
        )
        out = encode_response("openai", ir)
        ir2 = decode_response("openai", out)
        assert ir2["choices"][0]["message"]["refusal"] == "I cannot help with that"


# ═══════════════════════════════════════════════════════════════════
# 4. 流式 Logprobs 事件
# ═══════════════════════════════════════════════════════════════════

OPENAI_LOGPROBS_SSE = """\
data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.1,"bytes":[72,105],"top_logprobs":[{"token":"Hi","logprob":-0.1,"bytes":[72,105]},{"token":"Hello","logprob":-1.5,"bytes":[72,101]}]}]},"finish_reason":null}]}

data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}

data: [DONE]

"""


def _parse_openai_sse_chunks(raw):
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


class TestStreamLogprobs:
    def test_openai_decode_logprobs_event(self):
        """带 logprobs 的 OpenAI 流 → 产出 type=logprobs 的 IR 事件"""
        events = decode_stream("openai", OPENAI_LOGPROBS_SSE)
        lp_events = [e for e in events if e["type"] == "logprobs"]
        assert len(lp_events) == 1
        lp = lp_events[0]["logprobs"]
        assert lp["content"][0]["token"] == "Hi"
        assert lp["content"][0]["logprob"] == -0.1
        assert len(lp["content"][0]["top_logprobs"]) == 2

    def test_openai_encode_logprobs(self):
        """logprobs IR 事件 → OpenAI SSE：某个 chunk 的 choice 携带 logprobs"""
        events = decode_stream("openai", OPENAI_LOGPROBS_SSE)
        raw = encode_stream_raw("openai", events)
        chunks = _parse_openai_sse_chunks(raw)
        lp_chunks = [
            c
            for c in chunks
            if any("logprobs" in ch for ch in c.get("choices", []))
        ]
        assert len(lp_chunks) >= 1
        # 取出携带 logprobs 的 choice 验证 token
        lp = None
        for c in lp_chunks:
            for ch in c["choices"]:
                if "logprobs" in ch:
                    lp = ch["logprobs"]
                    break
            if lp:
                break
        assert lp["content"][0]["token"] == "Hi"


# ═══════════════════════════════════════════════════════════════════
# 5. Annotations（OpenAI web search 引文）
# ═══════════════════════════════════════════════════════════════════


class TestAnnotations:
    def test_openai_decode_citations(self):
        """OpenAI 响应 message.annotations → 文本部件的 citations"""
        ir = decode_response(
            "openai",
            openai_resp(
                {
                    "role": "assistant",
                    "content": "Hello",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com",
                                "title": "Example",
                            },
                        }
                    ],
                }
            ),
        )
        content = ir["choices"][0]["message"]["content"]
        # 有 citations 时 content 被展开为部件列表
        assert isinstance(content, list)
        text_parts = _text_parts(content)
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "Hello"
        citations = text_parts[0]["citations"]
        assert len(citations) == 1
        assert citations[0]["type"] == "url_citation"
        assert citations[0]["url"] == "https://example.com"
        assert citations[0]["title"] == "Example"


# ═══════════════════════════════════════════════════════════════════
# 6. 流式推理签名（reasoning_signature → ReasoningDone.signature）
# ═══════════════════════════════════════════════════════════════════

OPENAI_REASONING_SIG_SSE = """\
data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}

data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"reasoning_signature":"sig_abc123"},"finish_reason":null}]}

data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}

data: {"id":"c","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

"""


class TestStreamReasoningSignature:
    def test_openai_reasoning_done_signature(self):
        """reasoning_content 结束时携带的 reasoning_signature → ReasoningDone.signature"""
        events = decode_stream("openai", OPENAI_REASONING_SIG_SSE)
        done = [e for e in events if e["type"] == "reasoning_done"]
        assert len(done) >= 1
        assert done[0]["signature"] == "sig_abc123"


# ═══════════════════════════════════════════════════════════════════
# 7. P2 特性：provider_metadata 往返
# ═══════════════════════════════════════════════════════════════════


class TestP2ProviderMetadata:
    # ── Responses include / background ──

    def test_responses_include_background_decode(self):
        ir = decode_request(
            "responses",
            responses_req(include=["reasoning.encrypted_content"], background=True),
        )
        pm = ir.get("provider_metadata", {})
        assert pm.get("include") == ["reasoning.encrypted_content"]
        assert pm.get("background") is True

    def test_responses_include_background_round_trip(self):
        ir = decode_request(
            "responses",
            responses_req(include=["reasoning.encrypted_content"], background=True),
        )
        out = encode_request("responses", ir)
        assert out.get("include") == ["reasoning.encrypted_content"]
        assert out.get("background") is True

    # ── Gemini cachedContent ──

    def test_gemini_cached_content_decode(self):
        ir = decode_request(
            "gemini", gemini_req(cachedContent="cachedContents/abc123")
        )
        pm = ir.get("provider_metadata", {})
        # IR 侧使用 snake_case 键名
        assert pm.get("cached_content") == "cachedContents/abc123"

    def test_gemini_cached_content_round_trip(self):
        ir = decode_request(
            "gemini", gemini_req(cachedContent="cachedContents/abc123")
        )
        out = encode_request("gemini", ir)
        assert out.get("cachedContent") == "cachedContents/abc123"

    # ── OpenAI prediction / audio ──

    def test_openai_prediction_audio_decode(self):
        ir = decode_request(
            "openai",
            openai_req(
                prediction={"type": "content", "content": "predicted"},
                audio={"voice": "alloy", "format": "wav"},
            ),
        )
        pm = ir.get("provider_metadata", {})
        assert pm.get("prediction") == {"type": "content", "content": "predicted"}
        assert pm.get("audio") == {"voice": "alloy", "format": "wav"}

    def test_openai_prediction_audio_round_trip(self):
        ir = decode_request(
            "openai",
            openai_req(
                prediction={"type": "content", "content": "predicted"},
                audio={"voice": "alloy", "format": "wav"},
            ),
        )
        out = encode_request("openai", ir)
        assert out.get("prediction") == {"type": "content", "content": "predicted"}
        assert out.get("audio") == {"voice": "alloy", "format": "wav"}
