"""_openxlate_preserved 无损保留机制测试

验证不支持的 IrContentPart 变体：
- EncodeRequest: 不支持的 part → 输出 JSON metadata 含 _openxlate_preserved
- DecodeRequest: 输入含 metadata._openxlate_preserved → IR provider_metadata 含保留数据
- EncodeResponse: 同上
- DecodeResponse: 同上
- 往返完整性: encode → 保留字段出现 → decode → metadata 含原始数据
- 跨格式保留: 保留数据跨格式透传不丢失
"""

import json
import pytest

from conftest import (
    decode_request,
    encode_request,
    decode_response,
    encode_response,
)

PRESERVED_KEY = "_openxlate_preserved"

# ── 不支持的内容部件样本 ──────────────────────────────────────────

VIDEO_PART = {"type": "video", "media_type": "video/mp4", "data": "dmlkZW9kYXRh"}
OPAQUE_PART = {"type": "opaque", "provider": "custom", "payload": {"key": "value"}}
REDACTED_PART = {"type": "redacted_reasoning", "data": "c2VjcmV0"}
REASONING_PART = {"type": "reasoning", "text": "Let me think..."}
FUNC_CALL_PART = {
    "type": "function_call",
    "id": "call_99",
    "name": "get_weather",
    "arguments": '{"city":"Tokyo"}',
}
FUNC_RESP_PART = {
    "type": "function_response",
    "id": "call_99",
    "name": "get_weather",
    "response": {"temp": 25},
}


def _ir_request_with_parts(*parts):
    return {
        "model": "test-model",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, *parts],
            }
        ],
    }


def _ir_response_with_parts(*parts):
    return {
        "id": "ir-test-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}, *parts],
                },
                "finish_reason": "stop",
            }
        ],
    }


def _get_preserved(out):
    """从 metadata._openxlate_preserved 提取保留数据"""
    return out["metadata"][PRESERVED_KEY]


def _has_preserved(out):
    """检查 metadata._openxlate_preserved 是否存在"""
    return "metadata" in out and PRESERVED_KEY in out.get("metadata", {})


def _find_preserved_by_type(preserved, type_name):
    return [p for p in preserved if p.get("type") == type_name]


# ═══════════════════════════════════════════════════════════════════
# EncodeRequest: 不支持的 part → metadata._openxlate_preserved 出现
# ═══════════════════════════════════════════════════════════════════


class TestEncodeRequestPreserved:
    """IR 含不支持的 part → encode_request 输出 metadata 携带 _openxlate_preserved"""

    def test_openai_video_preserved(self):
        ir = _ir_request_with_parts(VIDEO_PART)
        out = encode_request("openai", ir)
        assert _has_preserved(out)
        preserved = _get_preserved(out)
        assert len(preserved) >= 1
        videos = _find_preserved_by_type(preserved, "video")
        assert len(videos) == 1
        assert videos[0]["media_type"] == "video/mp4"

    def test_anthropic_video_preserved(self):
        ir = _ir_request_with_parts(VIDEO_PART)
        out = encode_request("anthropic", ir)
        assert _has_preserved(out)
        videos = _find_preserved_by_type(_get_preserved(out), "video")
        assert len(videos) == 1

    def test_gemini_redacted_preserved(self):
        ir = _ir_request_with_parts(REDACTED_PART)
        out = encode_request("gemini", ir)
        assert _has_preserved(out)
        redacted = _find_preserved_by_type(_get_preserved(out), "redacted_reasoning")
        assert len(redacted) == 1
        assert redacted[0]["data"] == "c2VjcmV0"

    def test_responses_video_preserved(self):
        ir = _ir_request_with_parts(VIDEO_PART)
        out = encode_request("responses", ir)
        assert _has_preserved(out)
        videos = _find_preserved_by_type(_get_preserved(out), "video")
        assert len(videos) == 1

    def test_opaque_preserved_all_formats(self):
        """Opaque 在所有 4 种格式中都应被保留"""
        ir = _ir_request_with_parts(OPAQUE_PART)
        for fmt in ("openai", "anthropic", "gemini", "responses"):
            out = encode_request(fmt, ir)
            assert _has_preserved(out), f"{fmt}: missing preserved"
            opaque = _find_preserved_by_type(_get_preserved(out), "opaque")
            assert len(opaque) == 1, f"{fmt}: expected 1 opaque, got {len(opaque)}"
            assert opaque[0]["payload"] == {"key": "value"}, f"{fmt}: payload mismatch"

    def test_multiple_unsupported_parts(self):
        """多个不支持的 part 应全部保留"""
        ir = _ir_request_with_parts(VIDEO_PART, OPAQUE_PART, REASONING_PART)
        out = encode_request("openai", ir)
        preserved = _get_preserved(out)
        assert len(preserved) == 3
        types = {p["type"] for p in preserved}
        assert types == {"video", "opaque", "reasoning"}

    def test_supported_part_not_preserved(self):
        """已支持的 part 不应出现在 _openxlate_preserved 中"""
        ir = _ir_request_with_parts(VIDEO_PART)
        out = encode_request("openai", ir)
        preserved = _get_preserved(out)
        text_parts = _find_preserved_by_type(preserved, "text")
        assert len(text_parts) == 0, "text should NOT be preserved"

    def test_no_unsupported_no_metadata_preserved(self):
        """全部 part 都支持时，不应出现 metadata._openxlate_preserved"""
        ir = _ir_request_with_parts()  # only text part
        out = encode_request("openai", ir)
        assert not _has_preserved(out)


# ═══════════════════════════════════════════════════════════════════
# EncodeResponse: 不支持的 part → metadata._openxlate_preserved 出现
# ═══════════════════════════════════════════════════════════════════


class TestEncodeResponsePreserved:
    """IR Response 含不支持的 part → encode_response 输出 metadata 携带 _openxlate_preserved"""

    def test_openai_video_in_response(self):
        ir = _ir_response_with_parts(VIDEO_PART)
        out = encode_response("openai", ir)
        assert _has_preserved(out)
        videos = _find_preserved_by_type(_get_preserved(out), "video")
        assert len(videos) == 1

    def test_anthropic_video_in_response(self):
        ir = _ir_response_with_parts(VIDEO_PART)
        out = encode_response("anthropic", ir)
        assert _has_preserved(out)

    def test_gemini_redacted_in_response(self):
        ir = _ir_response_with_parts(REDACTED_PART)
        out = encode_response("gemini", ir)
        assert _has_preserved(out)

    def test_responses_video_in_response(self):
        ir = _ir_response_with_parts(VIDEO_PART)
        out = encode_response("responses", ir)
        assert _has_preserved(out)


# ═══════════════════════════════════════════════════════════════════
# DecodeRequest: metadata._openxlate_preserved → IR provider_metadata
# ═══════════════════════════════════════════════════════════════════


class TestDecodeRequestPreserved:
    """格式 JSON 含 metadata._openxlate_preserved → decode 后 IR provider_metadata 含保留数据"""

    def test_openai_decode_preserves(self):
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {PRESERVED_KEY: [VIDEO_PART]},
        }
        ir = decode_request("openai", payload)
        assert ir.get("provider_metadata") is not None
        pm = ir["provider_metadata"]
        assert PRESERVED_KEY in pm
        assert len(pm[PRESERVED_KEY]) == 1
        assert pm[PRESERVED_KEY][0]["type"] == "video"

    def test_anthropic_decode_preserves(self):
        payload = {
            "model": "claude-3",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {PRESERVED_KEY: [OPAQUE_PART]},
        }
        ir = decode_request("anthropic", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert pm[PRESERVED_KEY][0]["type"] == "opaque"

    def test_gemini_decode_preserves(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "metadata": {PRESERVED_KEY: [REDACTED_PART]},
        }
        ir = decode_request("gemini", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert pm[PRESERVED_KEY][0]["type"] == "redacted_reasoning"

    def test_responses_decode_preserves(self):
        payload = {
            "model": "gpt-4",
            "input": "hello",
            "metadata": {PRESERVED_KEY: [VIDEO_PART, OPAQUE_PART]},
        }
        ir = decode_request("responses", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert len(pm[PRESERVED_KEY]) == 2

    def test_backward_compat_toplevel(self):
        """旧格式（顶层 _openxlate_preserved）仍可被解码"""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            PRESERVED_KEY: [VIDEO_PART],
        }
        ir = decode_request("openai", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert pm[PRESERVED_KEY][0]["type"] == "video"


# ═══════════════════════════════════════════════════════════════════
# DecodeResponse: metadata._openxlate_preserved → IR provider_metadata
# ═══════════════════════════════════════════════════════════════════


class TestDecodeResponsePreserved:
    """响应 JSON 含 metadata._openxlate_preserved → decode 后 IR provider_metadata 含保留数据"""

    def test_openai_response_decode(self):
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "metadata": {PRESERVED_KEY: [VIDEO_PART]},
        }
        ir = decode_response("openai", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert pm[PRESERVED_KEY][0]["type"] == "video"

    def test_anthropic_response_decode(self):
        payload = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "metadata": {PRESERVED_KEY: [OPAQUE_PART]},
        }
        ir = decode_response("anthropic", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm

    def test_gemini_response_decode(self):
        payload = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "hi"}]},
                    "finishReason": "STOP",
                }
            ],
            "metadata": {PRESERVED_KEY: [REDACTED_PART]},
        }
        ir = decode_response("gemini", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm

    def test_responses_response_decode(self):
        payload = {
            "id": "resp-1",
            "object": "response",
            "status": "completed",
            "model": "gpt-4",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
            "metadata": {PRESERVED_KEY: [VIDEO_PART]},
        }
        ir = decode_response("responses", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm

    def test_backward_compat_toplevel(self):
        """旧格式（顶层 _openxlate_preserved）仍可被解码"""
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            PRESERVED_KEY: [VIDEO_PART],
        }
        ir = decode_response("openai", payload)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm


# ═══════════════════════════════════════════════════════════════════
# 往返保留: encode → metadata._openxlate_preserved → decode → 完整
# ═══════════════════════════════════════════════════════════════════


class TestPreservedRoundTrip:
    """encode(IR with unsupported) → decode → provider_metadata 含保留数据，数据完整"""

    def _round_trip_request(self, fmt, unsupported_parts):
        ir = _ir_request_with_parts(*unsupported_parts)
        encoded = encode_request(fmt, ir)
        assert _has_preserved(encoded), f"encode_request({fmt}) missing preserved"
        decoded = decode_request(fmt, encoded)
        pm = decoded.get("provider_metadata", {})
        assert PRESERVED_KEY in pm, f"decode_request({fmt}) missing preserved in metadata"
        return pm[PRESERVED_KEY]

    def _round_trip_response(self, fmt, unsupported_parts):
        ir = _ir_response_with_parts(*unsupported_parts)
        encoded = encode_response(fmt, ir)
        assert _has_preserved(encoded), f"encode_response({fmt}) missing preserved"
        decoded = decode_response(fmt, encoded)
        pm = decoded.get("provider_metadata", {})
        assert PRESERVED_KEY in pm, f"decode_response({fmt}) missing preserved in metadata"
        return pm[PRESERVED_KEY]

    def test_openai_request_video_roundtrip(self):
        preserved = self._round_trip_request("openai", [VIDEO_PART])
        assert len(preserved) == 1
        v = preserved[0]
        assert v["type"] == "video"
        assert v["media_type"] == "video/mp4"
        assert v["data"] == "dmlkZW9kYXRh"

    def test_anthropic_request_video_roundtrip(self):
        preserved = self._round_trip_request("anthropic", [VIDEO_PART])
        assert len(preserved) == 1
        assert preserved[0]["type"] == "video"

    def test_gemini_request_redacted_roundtrip(self):
        preserved = self._round_trip_request("gemini", [REDACTED_PART])
        assert len(preserved) == 1
        assert preserved[0]["type"] == "redacted_reasoning"
        assert preserved[0]["data"] == "c2VjcmV0"

    def test_responses_request_multi_roundtrip(self):
        preserved = self._round_trip_request(
            "responses", [VIDEO_PART, REASONING_PART, OPAQUE_PART]
        )
        assert len(preserved) == 3
        types = {p["type"] for p in preserved}
        assert types == {"video", "reasoning", "opaque"}

    def test_openai_response_video_roundtrip(self):
        preserved = self._round_trip_response("openai", [VIDEO_PART])
        assert len(preserved) == 1
        assert preserved[0]["type"] == "video"

    def test_anthropic_response_opaque_roundtrip(self):
        preserved = self._round_trip_response("anthropic", [OPAQUE_PART])
        opaque = [p for p in preserved if p["type"] == "opaque"]
        assert len(opaque) >= 1
        assert opaque[0]["payload"] == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════
# 跨格式保留: A → IR → B，不支持的 part 不丢失
# ═══════════════════════════════════════════════════════════════════


class TestPreservedCrossFormat:
    """格式 A 的 IR 含不支持 part → encode 到格式 B → preserved 随载荷透传"""

    def test_openai_to_anthropic_video(self):
        """OpenAI IR (含 Video) → encode to Anthropic → Video 保留"""
        ir = _ir_request_with_parts(VIDEO_PART)
        ant = encode_request("anthropic", ir)
        assert _has_preserved(ant)
        videos = _find_preserved_by_type(_get_preserved(ant), "video")
        assert len(videos) == 1

    def test_openai_to_gemini_opaque(self):
        """Opaque (所有格式不支持) → encode to Gemini → 保留"""
        ir = _ir_request_with_parts(OPAQUE_PART)
        gem = encode_request("gemini", ir)
        assert _has_preserved(gem)
        opaque = _find_preserved_by_type(_get_preserved(gem), "opaque")
        assert len(opaque) == 1

    def test_anthropic_to_openai_reasoning(self):
        """Reasoning → encode to OpenAI → 保留 (OpenAI 不原生支持 reasoning content part)"""
        ir = _ir_request_with_parts(REASONING_PART)
        oai = encode_request("openai", ir)
        assert _has_preserved(oai)
        reasoning = _find_preserved_by_type(_get_preserved(oai), "reasoning")
        assert len(reasoning) == 1
        assert reasoning[0]["text"] == "Let me think..."

    def test_full_gateway_trip_request(self):
        """完整网关路径: client(OpenAI) → decode → IR(+Video) → encode(Anthropic) → preserved 存在"""
        client_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        ir = decode_request("openai", client_payload)
        ir["messages"][0]["content"] = [
            {"type": "text", "text": "hello"},
            VIDEO_PART,
        ]
        ant_encoded = encode_request("anthropic", ir)
        assert _has_preserved(ant_encoded)

    def test_full_gateway_trip_response(self):
        """完整网关路径: upstream(Anthropic) → decode → IR(+Video in preserved) → encode(OpenAI)"""
        upstream_resp = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "metadata": {PRESERVED_KEY: [VIDEO_PART]},
        }
        ir = decode_response("anthropic", upstream_resp)
        pm = ir.get("provider_metadata", {})
        assert PRESERVED_KEY in pm
        assert pm[PRESERVED_KEY][0]["type"] == "video"


# ═══════════════════════════════════════════════════════════════════
# 数据完整性: 序列化 → 反序列化 数据一致
# ═══════════════════════════════════════════════════════════════════


class TestPreservedDataIntegrity:
    """保留数据在 encode/decode 全链路后，字段完整、值不变"""

    def test_function_call_extracted_to_tool_calls(self):
        """M10: FunctionCall 在 assistant 消息中被提取为 tool_calls，不走保留"""
        ir = {
            "model": "test-model",
            "stream": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling tool"},
                        FUNC_CALL_PART,
                    ],
                }
            ],
        }
        out = encode_request("openai", ir)
        msgs = out["messages"]
        assistant = [m for m in msgs if m["role"] == "assistant"][0]
        assert len(assistant["tool_calls"]) == 1
        tc = assistant["tool_calls"][0]
        assert tc["id"] == "call_99"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city":"Tokyo"}'

    def test_reasoning_fields_intact(self):
        """Reasoning 部件应走保留通道，字段完整"""
        ir = _ir_request_with_parts(REASONING_PART)
        out = encode_request("openai", ir)
        preserved = _get_preserved(out)
        r = _find_preserved_by_type(preserved, "reasoning")[0]
        assert r["text"] == "Let me think..."

    def test_opaque_payload_preserved_exactly(self):
        complex_payload = {
            "nested": {"array": [1, 2, 3], "null": None, "bool": True},
            "string": "hello 世界",
        }
        opaque = {
            "type": "opaque",
            "provider": "test_provider",
            "payload": complex_payload,
        }
        ir = _ir_request_with_parts(opaque)
        out = encode_request("openai", ir)
        preserved = _get_preserved(out)
        op = _find_preserved_by_type(preserved, "opaque")[0]
        assert op["provider"] == "test_provider"
        assert op["payload"] == complex_payload

    def test_unicode_preserved(self):
        video = {"type": "video", "media_type": "video/mp4", "data": "数据🎬"}
        ir = _ir_request_with_parts(video)
        out = encode_request("openai", ir)
        preserved = _get_preserved(out)
        v = _find_preserved_by_type(preserved, "video")[0]
        assert v["data"] == "数据🎬"


# ═══════════════════════════════════════════════════════════════════
# 流式保留: EncodeStream 不支持的事件附到 ChoiceFinish 帧
# ═══════════════════════════════════════════════════════════════════


class TestStreamPreserved:
    """EncodeStream 遇到不支持事件时缓存，在 ChoiceFinish 时附到帧的 metadata 里"""

    def test_openai_redacted_in_choicefinish(self):
        """RedactedReasoning 通过 OpenAI EncodeStream 后，附在 ChoiceFinish 帧的 metadata 中"""
        from conftest import encode_stream_raw
        events = [
            {"type": "start", "id": "c-1", "model": "m"},
            {"type": "content_delta", "index": 0, "delta": "hi"},
            {"type": "redacted_reasoning", "index": 0, "data": "secret123"},
            {"type": "content_done", "index": 0},
            {"type": "choice_finish", "index": 0, "finish_reason": "stop"},
            {"type": "done"},
        ]
        raw = encode_stream_raw("openai", events)
        # 解析 SSE 帧找到含 finish_reason 的帧
        finish_frame = None
        for line in raw.splitlines():
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                for c in choices:
                    if c.get("finish_reason"):
                        finish_frame = chunk
                        break
        assert finish_frame is not None, "no finish frame found"
        assert "metadata" in finish_frame, "finish frame missing metadata"
        assert PRESERVED_KEY in finish_frame["metadata"]
        preserved = finish_frame["metadata"][PRESERVED_KEY]
        assert any(
            p.get("type") == "redacted_reasoning" and p.get("data") == "secret123"
            for p in preserved
        )

    def test_gemini_redacted_preserved_in_stream(self):
        """RedactedReasoning 通过 Gemini EncodeStream 后，附在最终帧的 metadata 中"""
        from conftest import encode_stream_raw
        events = [
            {"type": "content_delta", "index": 0, "delta": "hi"},
            {"type": "redacted_reasoning", "index": 0, "data": "gem_secret"},
            {"type": "content_done", "index": 0},
            {"type": "choice_finish", "index": 0, "finish_reason": "stop"},
            {"type": "usage", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"type": "done"},
        ]
        raw = encode_stream_raw("gemini", events)
        # Gemini 编码器输出裸 JSON 行，codec_cli 会包上 data: 前缀
        found = False
        for line in raw.splitlines():
            text = line.strip()
            if text.startswith("data: "):
                text = text[6:]
            if not text:
                continue
            try:
                chunk = json.loads(text)
            except json.JSONDecodeError:
                continue
            if "metadata" in chunk and PRESERVED_KEY in chunk.get("metadata", {}):
                preserved = chunk["metadata"][PRESERVED_KEY]
                if any(
                    p.get("type") == "redacted_reasoning" and p.get("data") == "gem_secret"
                    for p in preserved
                ):
                    found = True
                    break
        assert found, "RedactedReasoning not found in any Gemini stream frame"

    def test_no_unsupported_no_stream_metadata(self):
        """没有不支持事件时，ChoiceFinish 帧不应有 metadata._openxlate_preserved"""
        from conftest import encode_stream_raw
        events = [
            {"type": "start", "id": "c-1", "model": "m"},
            {"type": "content_delta", "index": 0, "delta": "hi"},
            {"type": "content_done", "index": 0},
            {"type": "choice_finish", "index": 0, "finish_reason": "stop"},
            {"type": "done"},
        ]
        raw = encode_stream_raw("openai", events)
        for line in raw.splitlines():
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                meta = chunk.get("metadata", {})
                assert PRESERVED_KEY not in meta
