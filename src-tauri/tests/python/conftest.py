"""codec_cli 桥接层 — pytest 公共 fixtures"""

import json
import os
import subprocess
import sys

import pytest

_EXT = ".exe" if sys.platform == "win32" else ""
_CLI = os.path.join(
    os.path.dirname(__file__), "..", "..", "target", "debug", f"codec_cli{_EXT}"
)
CODEC_CLI = os.environ.get("CODEC_CLI", _CLI)


def _run(op: str, fmt: str, data) -> bytes:
    if isinstance(data, (dict, list)):
        raw = json.dumps(data, ensure_ascii=False).encode()
    elif isinstance(data, str):
        raw = data.encode()
    else:
        raw = data
    r = subprocess.run(
        [CODEC_CLI, op, fmt], input=raw, capture_output=True, timeout=15
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"codec_cli {op} {fmt} exit={r.returncode}\n{r.stderr.decode()}"
        )
    return r.stdout


# ── 快捷函数 ──────────────────────────────────────────────────────


def decode_request(fmt, payload) -> dict:
    return json.loads(_run("decode-request", fmt, payload))


def encode_request(fmt, ir) -> dict:
    return json.loads(_run("encode-request", fmt, ir))


def decode_response(fmt, payload) -> dict:
    return json.loads(_run("decode-response", fmt, payload))


def encode_response(fmt, ir) -> dict:
    return json.loads(_run("encode-response", fmt, ir))


def decode_stream(fmt, sse_text) -> list[dict]:
    raw = _run("decode-stream", fmt, sse_text)
    return [json.loads(line) for line in raw.decode().strip().splitlines() if line.strip()]


def encode_stream_raw(fmt, events: list[dict]) -> str:
    jsonl = "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
    return _run("encode-stream", fmt, jsonl).decode()


def encode_stream(fmt, events: list[dict]) -> list[dict]:
    """编码后再解码回来，方便做跨格式语义对比"""
    sse = encode_stream_raw(fmt, events)
    return decode_stream(fmt, sse)


# ── 语义提取 ─────────────────────────────────────────────────────


def extract_text(events: list[dict]) -> str:
    return "".join(
        e.get("delta", "") for e in events if e.get("type") == "content_delta"
    )


def extract_reasoning(events: list[dict]) -> str:
    return "".join(
        e.get("delta", "") for e in events if e.get("type") == "reasoning_delta"
    )


def extract_tool_calls(events: list[dict]) -> list[dict]:
    done = [e for e in events if e.get("type") == "tool_call_done"]
    return [{"id": d["id"], "name": d["name"], "arguments": d["arguments"]} for d in done]


def extract_finish(events: list[dict]) -> str | None:
    for e in events:
        if e.get("type") == "choice_finish":
            return e.get("finish_reason")
    return None


def extract_usage(events: list[dict]) -> dict | None:
    for e in reversed(events):
        if e.get("type") == "usage":
            return e
    return None


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def ensure_binary():
    if not os.path.isfile(CODEC_CLI):
        pytest.skip(f"codec_cli not found at {CODEC_CLI}, run `cargo build --bin codec_cli` first")
