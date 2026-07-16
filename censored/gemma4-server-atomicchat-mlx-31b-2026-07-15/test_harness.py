#!/usr/bin/env python3
"""Standalone harness resilience tests for AtomicChat Gemma 4 + gemma4_kilo_proxy.

Runs *outside* Kilo against the OpenAI-compatible public API (default :8080).

Usage:
  ./2_start_mlx.sh                 # proxy + engine already up
  python3 test_harness.py
  python3 test_harness.py --base http://127.0.0.1:8080 --strict
  python3 test_harness.py --unit-only   # pure proxy helpers, no network
  python3 test_harness.py --live-only
  python3 test_harness.py --quick       # skip slower multi-turn live tests

Exit codes:
  0  all required checks passed
  1  one or more required checks failed
  2  connectivity failure (no healthy live endpoint)
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import gemma4_kilo_proxy as proxy

DEFAULT_BASE = "http://127.0.0.1:8080"
MODEL = "gemma-4-31b-it-atomicchat-mlx-4bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by pattern",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]

SUMMARIZE_BAIT_SYSTEM = (
    "Do not re-summarize the conversation history. Preserve key information. "
    "Create a concise summary only if the user asks. agent=compaction is not active."
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    soft: bool = False


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", soft: bool = False) -> None:
        self.results.append(CheckResult(name, ok, detail, soft))
        status = "PASS" if ok else ("WARN" if soft else "FAIL")
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def check(
        self,
        name: str,
        cond: bool,
        detail: str = "",
        soft: bool = False,
    ) -> bool:
        self.add(name, cond, detail, soft=soft)
        return cond


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_json(
    base: str,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 180.0,
) -> tuple[int, Any, float]:
    url = base.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.time() - t0
            if not raw:
                return resp.status, None, elapsed
            try:
                return resp.status, json.loads(raw.decode("utf-8")), elapsed
            except json.JSONDecodeError:
                return resp.status, raw.decode("utf-8", errors="replace")[:200], elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        raw = e.read()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = raw.decode("utf-8", errors="replace")[:200]
        return e.code, parsed, elapsed
    except urllib.error.URLError as e:
        elapsed = time.time() - t0
        raise ConnectionError(str(e.reason if hasattr(e, "reason") else e)) from e


def _chat(
    base: str,
    messages: list[dict],
    *,
    tools: list | None = None,
    tool_choice: Any = None,
    max_tokens: int = 128,
    stream: bool = False,
    temperature: float = 0.35,
    model: str = MODEL,
    timeout: float = 180.0,
    extra: dict | None = None,
) -> tuple[int, dict | Any, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if extra:
        body.update(extra)
    return _http_json(base, "POST", "/v1/chat/completions", body, timeout=timeout)


def _stream_raw(base: str, body: dict, timeout: float = 120.0) -> tuple[int, str, float]:
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    raw_body = json.dumps(body).encode("utf-8")
    t0 = time.time()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=raw_body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    elapsed = time.time() - t0
    status = resp.status
    conn.close()
    return status, data, elapsed


def _msg(data: dict) -> dict:
    return (data.get("choices") or [{}])[0].get("message") or {}


def _finish(data: dict) -> Any:
    return (data.get("choices") or [{}])[0].get("finish_reason")


def _tool_names(msg: dict) -> list[str]:
    names: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _content(msg: dict) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else ""


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def run_unit_tests(report: Report) -> None:
    print("\n== Unit: compaction detection ==")

    body = {
        "messages": [
            {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
            {"role": "user", "content": "list files with tools"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    report.check(
        "unit: agent turn with summarize-bait system is NOT compaction",
        proxy._looks_like_compaction(body) is False,
    )

    body_user_summary = {
        "messages": [
            {
                "role": "user",
                "content": "Please summarize the conversation and preserve key information.",
            }
        ],
        "max_tokens": 2048,
    }
    report.check(
        "unit: user summary wording (no tools) IS compaction",
        proxy._looks_like_compaction(body_user_summary) is True,
    )

    body_none = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "tool_choice": "none",
    }
    report.check(
        "unit: tool_choice=none IS compaction",
        proxy._looks_like_compaction(body_none) is True,
    )

    body_user_summary_with_tools = {
        "messages": [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "Write a summary of main.py using tools"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    report.check(
        "unit: tools+auto never compaction even if user says summary",
        proxy._looks_like_compaction(body_user_summary_with_tools) is False,
    )

    body_history = {
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Please summarize the conversation history from earlier.",
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now list the repo with tools"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    report.check(
        "unit: only latest user message used for text compaction hints",
        proxy._looks_like_compaction(body_history) is False,
    )

    print("\n== Unit: prepare_body policy ==")

    body = {
        "messages": [
            {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
            {"role": "user", "content": "use tools"},
        ],
        "tools": list(TOOLS),
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.0,
    }
    tr = proxy._prepare_body(body)
    report.check(
        "unit: prepare keeps tools_out == tools_in for agent",
        tr["compaction"] is False and tr["tools_out"] == len(TOOLS),
        f"trace={ {k: tr[k] for k in ('compaction','tools_in','tools_out','nudged_multi_step')} }",
    )
    report.check(
        "unit: prepare forces enable_thinking=false",
        body.get("chat_template_kwargs", {}).get("enable_thinking") is False,
    )
    report.check(
        "unit: prepare multi-step nudge on agentic turn",
        tr.get("nudged_multi_step") is True
        and "[Harness] Multi-step tasks:" in body["messages"][0]["content"],
    )
    report.check(
        "unit: prepare raises agentic temperature floor from 0.0",
        float(body.get("temperature", 0)) >= 0.35,
        f"temp={body.get('temperature')}",
    )

    body_c = {
        "messages": [
            {
                "role": "user",
                "content": "Please summarize the conversation and preserve key information.",
            }
        ],
        "tools": list(TOOLS),
        "tool_choice": "auto",  # text path: no tools means user summary can still compact
        "max_tokens": 8192,
    }
    # Without tools list empty - compaction from user text
    body_c2 = {
        "messages": body_c["messages"],
        "max_tokens": 8192,
    }
    tr_c = proxy._prepare_body(body_c2)
    report.check(
        "unit: compaction caps max_tokens",
        tr_c["compaction"] is True
        and int(body_c2.get("max_tokens") or 0) <= proxy._COMPACTION_MAX_TOKENS_CEILING,
        f"max_tokens={body_c2.get('max_tokens')}",
    )

    body_strip = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": list(TOOLS),
        "tool_choice": "none",
        "max_tokens": 4096,
    }
    tr_s = proxy._prepare_body(body_strip)
    report.check(
        "unit: compaction strips tools from body",
        tr_s["compaction"] is True and "tools" not in body_strip,
    )

    print("\n== Unit: empty tool recovery ==")

    report.check("unit: empty string is empty tool", proxy._is_empty_tool_content(""))
    report.check(
        "unit: (no output) is empty tool",
        proxy._is_empty_tool_content("(no output)"),
    )
    report.check(
        "unit: useful text is NOT empty tool",
        not proxy._is_empty_tool_content(
            "commands.cpp has DownloadHandler and FlashHandler"
        ),
    )
    report.check(
        "unit: long text containing '404' is NOT empty tool",
        not proxy._is_empty_tool_content(
            "error 404 on secondary link but here is useful data xyz"
        ),
    )

    msgs_empty = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find husky"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "bash"}}],
        },
        {"role": "tool", "content": "(no output)"},
    ]
    report.check(
        "unit: empty tool streak == 1",
        proxy._recent_empty_tool_streak(msgs_empty) == 1,
        f"streak={proxy._recent_empty_tool_streak(msgs_empty)}",
    )

    msgs_two = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "(no output)"},
        {"role": "tool", "content": ""},
    ]
    report.check(
        "unit: empty tool streak == 2",
        proxy._recent_empty_tool_streak(msgs_two) == 2,
    )

    body_rec = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "find sources"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "(no output)"},
        ],
        "tools": list(TOOLS),
        "tool_choice": "auto",
    }
    tr_rec = proxy._prepare_body(body_rec)
    report.check(
        "unit: prepare empty_tool_recovery=True",
        tr_rec.get("empty_tool_recovery") is True
        and "[Harness] EMPTY TOOL RESULT:" in body_rec["messages"][0]["content"],
        f"streak={tr_rec.get('empty_tool_streak')}",
    )

    print("\n== Unit: truncation / flatten / remap ==")

    big = "x" * 30_000
    msgs_big = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": big},
    ]
    n = proxy._truncate_tool_messages(msgs_big)
    report.check(
        "unit: truncates large tool messages",
        n == 1 and len(msgs_big[1]["content"]) < 30_000,
        f"len={len(msgs_big[1]['content'])}",
    )
    report.check(
        "unit: truncation marker present",
        "truncated" in msgs_big[1]["content"].lower(),
    )

    msg_tc = {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"' + ("a" * 500) + '"}',
                }
            }
        ],
    }
    flat_n = proxy._flatten_tool_calls_in_history([msg_tc])
    report.check(
        "unit: flatten removes tool_calls key",
        flat_n == 1 and "tool_calls" not in msg_tc,
    )
    report.check(
        "unit: flatten keeps prior tool call text",
        "prior tool call" in (msg_tc.get("content") or "").lower()
        or "bash" in (msg_tc.get("content") or ""),
    )

    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "HELLO",
                }
            }
        ]
    }
    report.check(
        "unit: remap reasoning → content (message)",
        proxy._remap_reasoning_in_completion_payload(data)
        and data["choices"][0]["message"]["content"] == "HELLO"
        and "reasoning" not in data["choices"][0]["message"],
    )

    data_delta = {
        "choices": [{"delta": {"reasoning": "OK"}, "finish_reason": None}]
    }
    report.check(
        "unit: remap reasoning → content (delta)",
        proxy._remap_reasoning_in_completion_payload(data_delta)
        and data_delta["choices"][0]["delta"].get("content") == "OK",
    )

    line = b'data: {"choices":[{"delta":{"reasoning":"Z"}}]}\n'
    out = proxy._rewrite_sse_chunk(line)
    report.check(
        "unit: SSE rewrite maps reasoning to content",
        b'"content"' in out and b"reasoning" not in out,
        out[:120].decode("utf-8", errors="replace"),
    )
    report.check(
        "unit: SSE rewrite passes [DONE]",
        proxy._rewrite_sse_chunk(b"data: [DONE]\n") == b"data: [DONE]\n",
    )
    report.check(
        "unit: SSE rewrite passes non-data lines",
        proxy._rewrite_sse_chunk(b": keepalive\n") == b": keepalive\n",
    )

    print("\n== Unit: harness summary helpers ==")

    summary = proxy._summarize_completion_json(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    report.check(
        "unit: completion summary includes finish + tool name",
        "tool_calls" in summary and "bash" in summary and "finish=" in summary,
        summary,
    )

    stats = proxy.StreamHarnessStats()
    stats.observe_line(
        b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"read"}}]}}]}\n'
    )
    stats.observe_line(
        b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n'
    )
    report.check(
        "unit: StreamHarnessStats tracks tools + finish + content",
        stats.finish == "stop"
        and "read" in stats.tool_names
        and stats.content_chars >= 2,
        stats.summary(),
    )

    report.check(
        "unit: tool_schema_names extracts names",
        proxy._tool_schema_names({"tools": TOOLS}) == [
            "bash",
            "read",
            "glob",
            "grep",
        ],
    )
    report.check(
        "unit: message_role_counts",
        "user:1" in proxy._message_role_counts(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        and "assistant:1"
        in proxy._message_role_counts(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        ),
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def run_live_tests(
    base: str,
    report: Report,
    *,
    strict: bool,
    quick: bool,
) -> bool:
    """Returns False if connectivity failed entirely."""
    print(f"\n== Live contract ({base}) ==")
    connected = False

    try:
        code, data, elapsed = _http_json(base, "GET", "/healthz", timeout=5)
        ok = code == 200 and isinstance(data, dict) and data.get("ok") is True
        report.check(
            "live: GET /healthz",
            ok,
            f"status={code} body={data!r} {elapsed:.2f}s",
        )
        if not ok:
            return False
        connected = True
    except Exception as e:
        report.check("live: GET /healthz", False, str(e))
        return False

    try:
        code, data, elapsed = _http_json(base, "GET", "/v1/models", timeout=10)
        report.check(
            "live: GET /v1/models",
            code == 200 and isinstance(data, dict) and "data" in data,
            f"status={code} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: GET /v1/models", False, str(e))

    # Engine direct (optional — may differ host)
    try:
        eng = "http://127.0.0.1:8090"
        code, data, elapsed = _http_json(eng, "GET", "/v1/models", timeout=5)
        report.check(
            "live: engine :8090 /v1/models reachable",
            code == 200,
            f"status={code} {elapsed:.2f}s",
            soft=True,  # proxy-only deploys ok
        )
    except Exception as e:
        report.check("live: engine :8090 /v1/models reachable", False, str(e), soft=True)

    # Thinking off
    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word PONG."}],
            max_tokens=16,
            tools=None,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        report.check(
            "live: thinking off → non-empty content",
            code == 200 and bool(content) and "PONG" in content.upper(),
            f"finish={_finish(data)!r} content={content[:40]!r} "
            f"has_reasoning={bool(reasoning)} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: thinking off → non-empty content", False, str(e))

    # Tools not stripped (summarize bait)
    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Using tools only, run bash with command exactly: echo harness_ok. "
                        "Do not answer in plain text first."
                    ),
                },
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=128,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        hard_ok = code == 200 and isinstance(data, dict)
        soft_ok = hard_ok and (finish == "tool_calls" or "bash" in names)
        if not hard_ok:
            report.check(
                "live: tools kept (summarize bait) → tool_calls",
                False,
                f"status={code}",
            )
        else:
            report.check(
                "live: tools kept (summarize bait) → tool_calls",
                soft_ok,
                f"finish={finish!r} tools={names} {elapsed:.2f}s",
                soft=not soft_ok,
            )
    except Exception as e:
        report.check("live: tools kept (summarize bait) → tool_calls", False, str(e))

    # Explicit tool_choice function force (if server supports)
    try:
        code, data, elapsed = _chat(
            base,
            [
                {
                    "role": "user",
                    "content": "Call bash with command: echo forced",
                }
            ],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "bash"}},
            max_tokens=64,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        # Some servers ignore forced tool_choice — soft
        hard_ok = code == 200
        soft_ok = hard_ok and ("bash" in names or _finish(data) == "tool_calls")
        report.check(
            "live: tool_choice function force bash",
            soft_ok if hard_ok else False,
            f"status={code} finish={_finish(data)!r} tools={names} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check("live: tool_choice function force bash", False, str(e), soft=True)

    # Empty tool recovery
    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": "You are a coding agent."},
                {
                    "role": "user",
                    "content": "Find local source files in the workspace using tools.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {
                                        "command": (
                                            "curl -sL https://example.com/ | grep -i husky"
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "(no output)",
                },
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=128,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        hard_ok = code == 200 and isinstance(data, dict)
        used_tool = finish == "tool_calls" or bool(names)
        soft_ok = hard_ok and used_tool
        report.check(
            "live: empty tool → continues with tool_calls (not plan-only stop)",
            soft_ok,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check(
            "live: empty tool → continues with tool_calls (not plan-only stop)",
            False,
            str(e),
        )

    # Stream content
    try:
        status, raw, elapsed = _stream_raw(
            base,
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 12,
                "temperature": 0.35,
                "stream": True,
            },
        )
        has_content = '"content"' in raw and (
            '"content": null' not in raw or raw.count('"content"') > raw.count(
                '"content": null'
            )
        )
        # Prefer seeing actual token content
        ok = status == 200 and (
            '"content": "' in raw or '"content":"' in raw
        )
        report.check(
            "live: stream has content deltas",
            ok,
            f"status={status} has_content_field={has_content} {elapsed:.2f}s",
        )
        # Reasoning-only stream would be a hard fail
        only_reasoning = (
            '"reasoning"' in raw
            and '"content": "' not in raw
            and '"content":"' not in raw
        )
        report.check(
            "live: stream is not reasoning-only",
            status == 200 and not only_reasoning,
            f"only_reasoning={only_reasoning}",
        )
    except Exception as e:
        report.check("live: stream has content deltas", False, str(e))
        report.check("live: stream is not reasoning-only", False, str(e))

    # Stream + tools
    try:
        status, raw, elapsed = _stream_raw(
            base,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                    {
                        "role": "user",
                        "content": "Call bash with command: echo stream_tool",
                    },
                ],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 96,
                "temperature": 0.35,
                "stream": True,
            },
        )
        has_tool = "tool_calls" in raw or "bash" in raw
        hard_ok = status == 200
        report.check(
            "live: stream+tools returns tool_calls or bash name",
            hard_ok and has_tool,
            f"status={status} has_tool_marker={has_tool} {elapsed:.2f}s",
            soft=hard_ok and not has_tool,
        )
    except Exception as e:
        report.check(
            "live: stream+tools returns tool_calls or bash name",
            False,
            str(e),
            soft=True,
        )

    # Model id aliases
    for mid in (MODEL, "default_model"):
        try:
            code, data, elapsed = _chat(
                base,
                [{"role": "user", "content": "Say HI"}],
                model=mid,
                max_tokens=8,
                tools=None,
            )
            msg = _msg(data) if isinstance(data, dict) else {}
            ok = code == 200 and bool(_content(msg).strip() or msg.get("tool_calls"))
            report.check(
                f"live: model id accepts {mid!r}",
                ok,
                f"status={code} finish={_finish(data)!r} {elapsed:.2f}s",
                soft=(mid == "default_model" and not ok and code != 200),
            )
        except Exception as e:
            report.check(f"live: model id accepts {mid!r}", False, str(e), soft=True)

    # Compaction-style text request still 200
    try:
        code, data, elapsed = _chat(
            base,
            [
                {
                    "role": "user",
                    "content": (
                        "Please summarize the conversation and preserve key information. "
                        "Prior context: user asked about health check. Reply briefly."
                    ),
                }
            ],
            tools=None,
            tool_choice=None,
            max_tokens=64,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        report.check(
            "live: summary-style request returns content",
            code == 200 and bool(_content(msg).strip()),
            f"status={code} chars={len(_content(msg))} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: summary-style request returns content", False, str(e))

    if quick:
        return connected

    print("\n== Live multi-turn (slower) ==")

    # Multi-turn: tool then continue
    try:
        code1, data1, e1 = _chat(
            base,
            [
                {
                    "role": "user",
                    "content": "Using tools, run bash: echo step1_harness",
                }
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
        )
        msg1 = _msg(data1) if isinstance(data1, dict) else {}
        names1 = _tool_names(msg1)
        if not names1:
            report.check(
                "live: multi-turn step1 tool_calls",
                False,
                f"finish={_finish(data1)!r} no tools {e1:.2f}s",
                soft=True,
            )
        else:
            report.check(
                "live: multi-turn step1 tool_calls",
                True,
                f"tools={names1} {e1:.2f}s",
            )
            # Simulate tool result and ask for step 2
            tc = (msg1.get("tool_calls") or [])[0]
            tc_id = tc.get("id") or "call_mt"
            code2, data2, e2 = _chat(
                base,
                [
                    {
                        "role": "user",
                        "content": (
                            "Do 2 steps with tools: 1) echo step1_harness "
                            "2) echo step2_harness. Continue."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": msg1.get("content") or "",
                        "tool_calls": msg1.get("tool_calls"),
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "step1_harness",
                    },
                ],
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=96,
            )
            msg2 = _msg(data2) if isinstance(data2, dict) else {}
            names2 = _tool_names(msg2)
            finish2 = _finish(data2) if isinstance(data2, dict) else None
            soft_ok = code2 == 200 and (
                finish2 == "tool_calls" or bool(names2)
            )
            report.check(
                "live: multi-turn step2 still uses tools",
                soft_ok,
                f"finish={finish2!r} tools={names2} {e2:.2f}s",
                soft=code2 == 200 and not soft_ok,
            )
    except Exception as e:
        report.check("live: multi-turn step1 tool_calls", False, str(e), soft=True)

    # Two empty tools in a row still recover
    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": "Coding agent. Prefer local tools."},
                {"role": "user", "content": "Locate fastboot sources under this machine."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"curl -s https://a/ | grep x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "(no output)"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"curl -s https://b/ | grep y"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": ""},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=128,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        soft_ok = code == 200 and (finish == "tool_calls" or bool(names))
        report.check(
            "live: double empty tool still tool_calls",
            soft_ok,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=code == 200 and not soft_ok,
        )
    except Exception as e:
        report.check("live: double empty tool still tool_calls", False, str(e), soft=True)

    # Latency smoke (soft)
    try:
        times = []
        for _ in range(3):
            _, _, elapsed = _chat(
                base,
                [{"role": "user", "content": "Say X"}],
                max_tokens=4,
                tools=None,
            )
            times.append(elapsed)
        avg = sum(times) / len(times)
        report.check(
            "live: short completion latency median-ish < 30s",
            avg < 30.0,
            f"times={[round(t,2) for t in times]} avg={avg:.2f}s",
            soft=True,
        )
    except Exception as e:
        report.check(
            "live: short completion latency median-ish < 30s",
            False,
            str(e),
            soft=True,
        )

    return connected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AtomicChat Kilo harness smoke tests")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Public API base (proxy)")
    ap.add_argument("--unit-only", action="store_true", help="No network")
    ap.add_argument("--live-only", action="store_true", help="Skip unit tests")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip slower multi-turn live tests",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat soft (model behavior) failures as hard failures",
    )
    args = ap.parse_args(argv)

    print("AtomicChat harness tests")
    print(
        f"  base={args.base}  strict={args.strict}  "
        f"unit_only={args.unit_only} live_only={args.live_only} quick={args.quick}"
    )

    report = Report()
    if not args.live_only:
        run_unit_tests(report)

    connected = True
    if not args.unit_only:
        connected = run_live_tests(
            args.base, report, strict=args.strict, quick=args.quick
        )

    hard = [r for r in report.results if not r.ok and not r.soft]
    soft = [r for r in report.results if not r.ok and r.soft]
    passed = [r for r in report.results if r.ok]

    print("\n== Summary ==")
    print(
        f"  total={len(report.results)}  passed={len(passed)}  "
        f"hard_fail={len(hard)}  soft_fail={len(soft)}"
    )
    if hard:
        print("  HARD FAILURES:")
        for r in hard:
            print(f"    - {r.name}: {r.detail}")
    if soft:
        label = " [strict → fail]" if args.strict else " [ignored]"
        print(f"  SOFT (model) FAILURES:{label}")
        for r in soft:
            print(f"    - {r.name}: {r.detail}")

    if hard:
        return 1
    if args.strict and soft:
        return 1
    if not args.unit_only and not connected:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
