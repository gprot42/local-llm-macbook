#!/usr/bin/env python3
"""Standalone harness tests for Gemma 4 31B Heretic + gemma4_mlx_kilo_proxy.

Runs *outside* Kilo against the OpenAI-compatible public API (default :8080).

Usage:
  ./2_start_mlx.sh                 # proxy + engine already up
  python3 test_harness.py
  python3 test_harness.py --base http://127.0.0.1:8080 --strict
  python3 test_harness.py --unit-only   # pure proxy helpers, no network
  python3 test_harness.py --live-only
  python3 test_harness.py --quick       # skip slower multi-turn live tests
  python3 test_harness.py --gate        # post-start gate (unit + critical live)

We do NOT fully emulate Kilo (no session DB, compaction UI, permissions).

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
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gemma4_mlx_kilo_proxy as proxy  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:8080"
MODEL = "gemma-4-31b-heretic-mlx-4bit"
FIXTURES = ROOT / "tests" / "fixtures"

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
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
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
    model: str | None = None,
    timeout: float = 180.0,
    extra: dict | None = None,
) -> tuple[int, dict | Any, float]:
    body: dict[str, Any] = {
        "model": model or MODEL,
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


def _stream_raw(base: str, body: dict, timeout: float = 180.0) -> tuple[int, str, float]:
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


def _parse_sse_fixture(name: str) -> list[dict]:
    events: list[dict] = []
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload and payload != "[DONE]":
            events.append(json.loads(payload))
    return events


def _sse_text_and_tools(raw: str) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    names: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in ev.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            msg = choice.get("message") or {}
            if isinstance(msg.get("content"), str):
                text_parts.append(msg["content"])
            for tc in (delta.get("tool_calls") or []) + (msg.get("tool_calls") or []):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or ""
                if name:
                    names.append(str(name))
    return "".join(text_parts), names


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
        proxy._is_compaction_request(body) is False,
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
        proxy._is_compaction_request(body_user_summary) is True,
    )

    body_none = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "tool_choice": "none",
    }
    report.check(
        "unit: tool_choice=none IS compaction",
        proxy._is_compaction_request(body_none) is True,
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
        proxy._is_compaction_request(body_user_summary_with_tools) is False,
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
        proxy._is_compaction_request(body_history) is False,
    )

    print("\n== Unit: compaction prepare ==")

    body_c = {
        "messages": [
            {
                "role": "user",
                "content": "Please summarize the conversation and preserve key information.",
            }
        ],
        "max_tokens": 8192,
        "tool_choice": "none",
        "tools": list(TOOLS),
    }
    proxy._prepare_compaction_request(body_c)
    report.check(
        "unit: compaction strips tools from body",
        "tools" not in body_c and body_c.get("tool_choice") == "none",
    )
    report.check(
        "unit: compaction caps max_tokens",
        int(body_c.get("max_tokens") or 0) <= proxy._COMPACTION_MAX_TOKENS_CEILING,
        f"max_tokens={body_c.get('max_tokens')}",
    )

    body_hist = {
        "tool_choice": "none",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo hi"}',
                        }
                    }
                ],
            }
        ],
    }
    proxy._prepare_compaction_request(body_hist)
    assistant = next(m for m in body_hist["messages"] if m.get("role") == "assistant")
    report.check(
        "unit: compaction flattens history tool_calls",
        "tool_calls" not in assistant and "bash" in (assistant.get("content") or ""),
    )

    print("\n== Unit: agentic settings ==")

    body_a = {
        "tools": list(TOOLS),
        "temperature": 0.0,
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "foo": "bar"},
    }
    proxy._force_agentic_settings(body_a)
    report.check(
        "unit: prepare raises agentic temperature floor from 0.0",
        float(body_a.get("temperature", 0)) >= 0.35,
        f"temp={body_a.get('temperature')}",
    )
    report.check(
        "unit: prepare forces enable_thinking=false",
        body_a.get("enable_thinking") is False
        and body_a.get("chat_template_kwargs", {}).get("enable_thinking") is False,
    )
    report.check(
        "unit: preserves other chat_template_kwargs keys",
        body_a.get("chat_template_kwargs", {}).get("foo") == "bar",
    )
    report.check(
        "unit: harmony logit_bias injected",
        body_a.get("logit_bias", {}).get("100") == -100.0
        and body_a.get("logit_bias", {}).get("98") == -100.0,
    )

    body_bias = {"tools": [{}], "logit_bias": {"100": 5.0}}
    proxy._force_agentic_settings(body_bias)
    report.check(
        "unit: caller logit_bias wins",
        body_bias["logit_bias"]["100"] == 5.0,
    )

    body_plain = {
        "enable_thinking": True,
        "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    proxy._disable_thinking(body_plain)
    report.check(
        "unit: thinking forced off even without tools",
        body_plain.get("enable_thinking") is False
        and body_plain.get("chat_template_kwargs", {}).get("enable_thinking") is False
        and body_plain.get("thinking", {}).get("type") == "disabled",
    )

    print("\n== Unit: tool repair + SSE fixtures ==")

    writers = {
        **proxy._DEFAULT_WRITERS,
        "write_name": "Write",
        "write_path_field": "path",
        "write_content_field": "content",
        "write_available": True,
        "tool_names": frozenset({"Write", "StrReplace", "TodoWrite"}),
    }
    name, args = proxy.repair_tool_call(
        "write",
        json.dumps({"filePath": "/tmp/a.html", "content": "<html/>"}),
        writers,
    )
    parsed = json.loads(args)
    report.check(
        "unit: gemma write case-fold + filePath remap",
        name == "Write" and parsed.get("path") == "/tmp/a.html" and "filePath" not in parsed,
        f"name={name} args={parsed}",
    )

    events = _parse_sse_fixture("bash_tool_call.sse")
    calls = proxy._reassemble_tool_calls(events)
    bash_ok = False
    try:
        bash_ok = json.loads(calls[0]["arguments"])["command"] == "ls -la"
    except Exception:
        bash_ok = False
    report.check(
        "unit: bash SSE fixture reassembles",
        bool(calls) and calls[0].get("name") == "bash" and bash_ok,
    )

    hw_events = _parse_sse_fixture("hallucinated_write_tool.sse")
    chunks = proxy._emit_repaired_stream(hw_events, writers)
    hw_payload = json.loads(chunks[0].decode().split("data: ", 1)[1])
    hw_tc = hw_payload["choices"][0]["delta"]["tool_calls"][0]
    hw_args = json.loads(hw_tc["function"]["arguments"])
    report.check(
        "unit: hallucinated write_to_file remapped to Write",
        hw_tc["function"]["name"] == "Write"
        and hw_args.get("path") == "/tmp/hello.txt",
        f"name={hw_tc['function']['name']} args={hw_args}",
    )

    print("\n== Unit: native Gemma tool markup leak ==")

    leaked = (
        "<tool_call|>\n"
        "→Read vendor/fastboot/fastboot.cpp\n"
        '<tool_call|>call:read{filePath:<|"|>/tmp/x.cpp<|"|>}'
    )
    report.check(
        "unit: broken Gemma tool markup is a leak",
        proxy.content_leaks_native_tools(leaked) is True,
    )
    report.check(
        "unit: well-formed native call is a leak in content",
        proxy.content_leaks_native_tools(
            '<|tool_call>call:bash{"command":"echo hi"}'
        )
        is True,
    )
    report.check(
        "unit: normal prose is not a leak",
        proxy.content_leaks_native_tools("I will list the files next.") is False,
    )

    print("\n== Unit: planning strip / write-name ==")
    report.check(
        "unit: todowrite is not a write tool",
        proxy._is_write_tool_name("todowrite") is False
        and proxy._is_write_tool_name("todo_write") is False,
    )
    body_strip = {
        "tools": [
            {"type": "function", "function": {"name": "todowrite"}},
            {"type": "function", "function": {"name": "write"}},
        ],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "todowrite", "arguments": "{}"}}
                ],
            }
        ],
    }
    proxy._strip_planning_tools_if_stuck(body_strip)
    names = [(t.get("function") or {}).get("name") for t in body_strip["tools"]]
    report.check(
        "unit: strips todowrite after plan with no write",
        "todowrite" not in names and "write" in names,
        f"names={names}",
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def run_live_tests(
    base: str, report: Report, *, strict: bool, quick: bool
) -> bool:
    print(f"\n== Live ({base}) ==")
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
    except Exception as e:
        report.check("live: GET /healthz", False, str(e))
        return False

    try:
        code, data, elapsed = _http_json(base, "GET", "/v1/models", timeout=10)
        report.check(
            "live: GET /v1/models",
            code == 200 and isinstance(data, dict),
            f"status={code} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: GET /v1/models", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word PONG."}],
            max_tokens=16,
            tools=None,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        report.check(
            "live: thinking off → non-empty content",
            code == 200 and "PONG" in content.upper(),
            f"content={content[:40]!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: thinking off → non-empty content", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Using tools only, run bash with command exactly: echo gate_ok."
                    ),
                },
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        content = _content(msg)
        hard_ok = code == 200 and not proxy.content_leaks_native_tools(content)
        soft_ok = hard_ok and (finish == "tool_calls" or "bash" in names)
        report.check(
            "live: tools kept (summarize bait) → tool_calls",
            soft_ok,
            f"finish={finish!r} tools={names} leak={proxy.content_leaks_native_tools(content)} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
        report.check(
            "live: tools kept — no native Gemma markup in content",
            hard_ok,
            f"content={content[:80]!r}",
        )
    except Exception as e:
        report.check("live: tools kept (summarize bait) → tool_calls", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Find files with tools."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "g1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo empty_probe"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "g1", "content": "(no output)"},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        content = _content(msg)
        hard_ok = code == 200 and not proxy.content_leaks_native_tools(content)
        soft_ok = hard_ok and (finish == "tool_calls" or bool(names))
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

    try:
        code, raw, elapsed = _stream_raw(
            base,
            {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Reply with exactly the word ZAP."}
                ],
                "max_tokens": 16,
                "temperature": 0.35,
                "stream": True,
            },
        )
        text, _ = _sse_text_and_tools(raw)
        report.check(
            "live: stream has content deltas",
            code == 200 and bool(text.strip()),
            f"text={text[:40]!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: stream has content deltas", False, str(e))

    try:
        code, raw, elapsed = _stream_raw(
            base,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                    {
                        "role": "user",
                        "content": "Using tools only, run bash with command exactly: echo stream_ok.",
                    },
                ],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 96,
                "temperature": 0.35,
                "stream": True,
            },
        )
        text, names = _sse_text_and_tools(raw)
        leak = proxy.content_leaks_native_tools(text)
        hard_ok = code == 200 and not leak
        soft_ok = hard_ok and (bool(names) or "bash" in text.lower())
        report.check(
            "live: stream+tools returns tool_calls or bash name",
            soft_ok,
            f"tools={names} leak={leak} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
        report.check(
            "live: stream+tools — no native Gemma markup in content",
            hard_ok,
            f"text={text[:80]!r}",
        )
    except Exception as e:
        report.check(
            "live: stream+tools returns tool_calls or bash name", False, str(e)
        )

    if not quick:
        try:
            code, data, elapsed = _chat(
                base,
                [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "List files, then read README.md."},
                ],
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=96,
            )
            msg = _msg(data) if isinstance(data, dict) else {}
            names = _tool_names(msg)
            content = _content(msg)
            hard_ok = code == 200 and not proxy.content_leaks_native_tools(content)
            soft_ok = hard_ok and bool(names)
            report.check(
                "live: multi-turn step1 tool_calls",
                soft_ok,
                f"tools={names} {elapsed:.2f}s",
                soft=hard_ok and not soft_ok,
            )
        except Exception as e:
            report.check("live: multi-turn step1 tool_calls", False, str(e), soft=True)

    _ = strict  # reserved: caller maps soft→hard in summary
    return True


def run_gate_tests(base: str, report: Report) -> bool:
    """Fast post-start gate: unit + critical live only."""
    print("\n== Gate mode (post-start) ==")
    run_unit_tests(report)

    print(f"\n== Gate live ({base}) ==")
    try:
        code, data, elapsed = _http_json(base, "GET", "/healthz", timeout=5)
        ok = code == 200 and isinstance(data, dict) and data.get("ok") is True
        report.check(
            "gate: GET /healthz",
            ok,
            f"status={code} body={data!r} {elapsed:.2f}s",
        )
        if not ok:
            return False
    except Exception as e:
        report.check("gate: GET /healthz", False, str(e))
        return False

    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word PONG."}],
            max_tokens=16,
            tools=None,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        report.check(
            "gate: thinking off → content",
            code == 200 and "PONG" in content.upper(),
            f"content={content[:40]!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("gate: thinking off → content", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Using tools only, run bash with command exactly: echo gate_ok."
                    ),
                },
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        content = _content(msg)
        leak = proxy.content_leaks_native_tools(content)
        hard_ok = code == 200 and not leak
        soft_ok = hard_ok and (finish == "tool_calls" or "bash" in names)
        report.check(
            "gate: tools not stripped → tool_calls",
            soft_ok,
            f"finish={finish!r} tools={names} leak={leak} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
        report.check(
            "gate: no native Gemma tool markup in content",
            hard_ok,
            f"content={content[:80]!r}",
        )
    except Exception as e:
        report.check("gate: tools not stripped → tool_calls", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Find files with tools."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "g1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo empty_probe"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "g1", "content": "(no output)"},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        content = _content(msg)
        hard_ok = code == 200 and not proxy.content_leaks_native_tools(content)
        soft_ok = hard_ok and (finish == "tool_calls" or bool(names))
        report.check(
            "gate: empty tool → continues with tools",
            soft_ok,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check("gate: empty tool → continues with tools", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word ZAP."}],
            max_tokens=16,
            tools=None,
            extra={
                "enable_thinking": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        report.check(
            "gate: client enable_thinking forced off",
            code == 200 and bool(content),
            f"content={content[:40]!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("gate: client enable_thinking forced off", False, str(e))

    return True


def main(argv: list[str] | None = None) -> int:
    global MODEL
    ap = argparse.ArgumentParser(description="Heretic Gemma 4 Kilo harness smoke tests")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Public API base (proxy)")
    ap.add_argument("--unit-only", action="store_true", help="No network")
    ap.add_argument("--live-only", action="store_true", help="Skip unit tests")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Post-start gate: unit + critical live only",
    )
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
    ap.add_argument("--model", default=MODEL, help="Model id to send")
    args = ap.parse_args(argv)
    MODEL = args.model

    print("Heretic Gemma 4 harness tests")
    print(
        f"  base={args.base}  strict={args.strict}  "
        f"unit_only={args.unit_only} live_only={args.live_only} "
        f"gate={args.gate} quick={args.quick}"
    )

    report = Report()
    connected = True

    if args.gate:
        connected = run_gate_tests(args.base, report)
    else:
        if not args.live_only:
            run_unit_tests(report)
        if not args.unit_only:
            try:
                connected = run_live_tests(
                    args.base, report, strict=args.strict, quick=args.quick
                )
            except ConnectionError as e:
                report.check("live: proxy reachable", False, str(e))
                connected = False

    hard = [r for r in report.results if not r.ok and not r.soft]
    soft = [r for r in report.results if not r.ok and r.soft]
    if args.strict:
        hard.extend(soft)
        soft = []
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
        print("  SOFT (model) FAILURES: [ignored]")
        for r in soft:
            print(f"    - {r.name}: {r.detail}")

    if not connected and not args.unit_only:
        return 2
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
