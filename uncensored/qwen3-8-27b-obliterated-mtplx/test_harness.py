#!/usr/bin/env python3
"""Standalone harness resilience tests for Qwen3.8-27B OBLITERATED + mtplx.

Runs *outside* Kilo against the OpenAI-compatible kilo proxy (default :8768).

Usage:
  ./2_start_mtplx.sh                 # server already up
  python3 test_harness.py
  python3 test_harness.py --base http://127.0.0.1:8768 --strict
  python3 test_harness.py --gate     # post-start gate (critical live only)
  python3 test_harness.py --quick    # skip slower multi-turn / concurrent tests
  python3 test_harness.py --unit     # offline proxy/middleware tests only
  python3 test_harness.py --agent    # AutoSaddler-style train/dev mini-batch loops
  python3 test_harness.py --optimize --iters 3   # Diagnosis–Patch–EvoDAG (persists)
  python3 test_harness.py --model qwen3.8-27b-obliterated-mtplx

Sampling matches the OBLITERATUS card: temperature=0, top_p=1.0,
thinking off, frequency_penalty=0.3 (mtplx stand-in for HF repetition_penalty=1.15).
We do NOT fully emulate Kilo (no session DB, compaction UI, permissions).

Agent-loop tests execute allowlisted tools in a temp workspace and diagnose
traces (no_tool_call / stopped_after_1 / empty_result_idle / fake_action)
instead of only checking that one API response contained tool_calls.

Exit codes:
  0  all required checks passed
  1  one or more required checks failed
  2  connectivity failure (no healthy live endpoint)
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE = "http://127.0.0.1:8768"
DEFAULT_MODEL = "qwen3.8-27b-obliterated-mtplx"
# OBLITERATUS card (HF): greedy, no thinking, empty system, no top_p/top_k.
# repetition_penalty=1.15 is essential against import/boilerplate loops.
# mtplx has no that flag — OpenAI frequency_penalty=0.3 is this stack's mapping.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.3
# Card wants max_new_tokens ≥ 2048 for real answers; smoke PING stays tiny.
DEFAULT_MAX_TOKENS = 512
AGENT_MAX_TOKENS = 2048
# Short Kilo-style loop nudge. Card prefers empty system for chat; tool loops
# stop after 1 of N without this. Keep it tiny so it does not reintroduce refusals.
AGENT_LOOP_PROMPT = (
    "Finish every requested step. Prefer tools over prose. After each tool "
    "result, immediately take the next action. Do not stop after 1 of N. "
    "Do not recap; act. Empty or error tool output: retry with a simpler "
    "local command (ls/glob/grep/read). Verify with a tool before declaring "
    "the job done."
)

# Tool-description patch (AutoSaddler steering): tell the model *when* to use
# each tool, not just the name. Keep short — attention is finite.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a short local shell command (echo, ls, pwd, cat). "
                "After the result, continue to the next requested step."
            ),
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
            "description": (
                "Read a workspace file by path. Use after glob/grep locates "
                "the file. Do not guess paths."
            ),
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
            "description": (
                "Find workspace files by glob pattern (e.g. **/*.txt). "
                "Use before read."
            ),
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
            "description": (
                "Search workspace file contents with a regex. Prefer this "
                "over bash grep."
            ),
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


def _loop_prompt() -> str:
    try:
        from autosaddler import load_active

        text = str(load_active().get("loop_prefix") or "").strip()
        if text:
            return text
    except Exception:
        pass
    return AGENT_LOOP_PROMPT


def _agent_tools() -> list[dict]:
    tools = json.loads(json.dumps(TOOLS))
    try:
        from autosaddler import load_active

        desc = load_active().get("tool_desc") or {}
    except Exception:
        desc = {}
    if isinstance(desc, dict):
        for item in tools:
            fn = item.get("function") or {}
            name = fn.get("name")
            if name in desc and isinstance(desc[name], str) and desc[name].strip():
                fn["description"] = desc[name]
    return tools


TRACE_DIR = Path(__file__).resolve().parent / "logs" / "harness-traces"
_UNSAFE_BASH = re.compile(r"[;|&`$]|&&|\|\||\$\(")
_SAFE_BASH_HEAD = frozenset({"echo", "printf", "pwd", "true", "false", "ls", "cat"})


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


def _sample_fields(
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Card sampling: greedy, no thinking, HF 1.15 via frequency_penalty."""
    body: dict[str, Any] = {
        "temperature": temperature,
        "top_p": DEFAULT_TOP_P,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "enable_thinking": False,
    }
    if extra:
        body.update(extra)
    return body


def _chat(
    base: str,
    messages: list[dict],
    *,
    tools: list | None = None,
    tool_choice: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    model: str = DEFAULT_MODEL,
    timeout: float = 180.0,
    extra: dict | None = None,
) -> tuple[int, dict | Any, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        **_sample_fields(temperature=temperature, extra=extra),
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
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
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for item in c:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and item.get("type") in (None, "text"):
                    parts.append(text)
        return "".join(parts)
    return ""


def _with_agent_loop(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": _loop_prompt()}, *messages]


def _tool_args_blob(msg: dict) -> str:
    parts: list[str] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else {}
        raw = (fn or {}).get("arguments") or ""
        parts.append(raw if isinstance(raw, str) else json.dumps(raw))
    return " ".join(parts).lower()


def _resolve_model(base: str, preferred: str) -> str:
    """Use preferred id if listed; else first model from /v1/models."""
    try:
        code, data, _ = _http_json(base, "GET", "/v1/models", timeout=5)
        if code != 200 or not isinstance(data, dict):
            return preferred
        ids = [
            m.get("id")
            for m in (data.get("data") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        if preferred in ids:
            return preferred
        if ids:
            return str(ids[0])
    except Exception:
        pass
    return preferred


def _probe(base: str) -> bool:
    """True if either /v1/models or /health responds."""
    for path in ("/v1/models", "/health", "/healthz"):
        try:
            code, _, _ = _http_json(base, "GET", path, timeout=5)
            if code == 200:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def run_gate_tests(base: str, model: str, report: Report) -> bool:
    """Critical post-start checks only."""
    print(f"\n== Gate ({base}, model={model}) ==")
    if not _probe(base):
        report.check("gate: server reachable", False, base)
        return False
    report.check("gate: server reachable", True, base)

    try:
        code, data, elapsed = _http_json(base, "GET", "/v1/models", timeout=10)
        ok = code == 200 and isinstance(data, dict) and bool(data.get("data"))
        ids = [
            m.get("id")
            for m in (data.get("data") or [])
            if isinstance(m, dict)
        ] if isinstance(data, dict) else []
        report.check(
            "gate: GET /v1/models",
            ok,
            f"status={code} ids={ids} {elapsed:.2f}s",
        )
        if not ok:
            return False
    except Exception as e:
        report.check("gate: GET /v1/models", False, str(e))
        return False

    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word PING."}],
            max_tokens=16,
            tools=None,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        report.check(
            "gate: short non-tool chat",
            code == 200 and bool(content),
            f"content={content[:60]!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("gate: short non-tool chat", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            _with_agent_loop(
                [
                    {
                        "role": "user",
                        "content": (
                            "Using tools only, run bash with command exactly: "
                            "echo harness_ok. Do not explain."
                        ),
                    }
                ]
            ),
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        hard_ok = code == 200
        soft_ok = hard_ok and ("bash" in names or finish == "tool_calls")
        report.check(
            "gate: tool call (bash)",
            soft_ok if hard_ok else False,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check("gate: tool call (bash)", False, str(e))

    try:
        status, raw, elapsed = _stream_raw(
            base,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Count: 1 2 3"}],
                "max_tokens": 24,
                "stream": True,
                **_sample_fields(),
            },
            timeout=90,
        )
        has_data = "data:" in raw
        report.check(
            "gate: SSE stream",
            status == 200 and has_data,
            f"status={status} bytes={len(raw)} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("gate: SSE stream", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            _with_agent_loop(
                [
                    {
                        "role": "user",
                        "content": "Using tools, run bash: echo step1_harness",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_gate_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo step1_harness"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_gate_1",
                        "content": "step1_harness",
                    },
                    {
                        "role": "user",
                        "content": "Tool returned. Reply with exactly: done_step1",
                    },
                ]
            ),
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).lower()
        hard_ok = code == 200
        continued = "done" in content or bool(_tool_names(msg))
        report.check(
            "gate: multi-turn tool result continues",
            hard_ok and continued,
            f"content={_content(msg)[:50]!r} tools={_tool_names(msg)} {elapsed:.2f}s",
            soft=hard_ok and not continued,
        )
    except Exception as e:
        report.check("gate: multi-turn tool result continues", False, str(e))

    return True


def run_live_tests(
    base: str,
    model: str,
    report: Report,
    *,
    quick: bool,
) -> bool:
    print(f"\n== Live contract ({base}, model={model}) ==")
    if not _probe(base):
        report.check("live: server reachable", False, base)
        return False
    report.check("live: server reachable", True, base)

    # Prefer /v1/models; /health is optional
    try:
        code, data, elapsed = _http_json(base, "GET", "/v1/models", timeout=10)
        ok = code == 200 and isinstance(data, dict)
        report.check(
            "live: GET /v1/models",
            ok,
            f"status={code} {elapsed:.2f}s",
        )
        if not ok:
            return False
    except Exception as e:
        report.check("live: GET /v1/models", False, str(e))
        return False

    for path in ("/health", "/healthz"):
        try:
            code, data, elapsed = _http_json(base, "GET", path, timeout=5)
            report.check(
                f"live: GET {path}",
                code == 200,
                f"status={code} {elapsed:.2f}s",
                soft=code != 200,
            )
            if code == 200:
                break
        except Exception as e:
            report.check(f"live: GET {path}", False, str(e), soft=True)

    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Say hi in five words or fewer."}],
            max_tokens=32,
            tools=None,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).strip()
        report.check(
            "live: short chat returns content",
            code == 200 and bool(content),
            f"content={content[:80]!r} finish={_finish(data)!r} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: short chat returns content", False, str(e))

    try:
        code, data, elapsed = _chat(
            base,
            _with_agent_loop(
                [
                    {
                        "role": "user",
                        "content": (
                            "Call the bash tool with command exactly: "
                            "echo harness_ok"
                        ),
                    }
                ]
            ),
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        hard_ok = code == 200
        soft_ok = hard_ok and ("bash" in names or finish == "tool_calls")
        report.check(
            "live: tool_calls includes bash",
            soft_ok if hard_ok else False,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
        # Validate tool call JSON args when present
        if hard_ok and names:
            args_ok = True
            detail_args = ""
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else {}
                raw = (fn or {}).get("arguments", "")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    detail_args = str(parsed)[:80]
                    if not isinstance(parsed, dict):
                        args_ok = False
                except json.JSONDecodeError:
                    args_ok = False
                    detail_args = repr(raw)[:80]
            report.check(
                "live: tool call arguments are JSON object",
                args_ok,
                detail_args,
                soft=not args_ok,
            )
    except Exception as e:
        report.check("live: tool_calls includes bash", False, str(e))

    try:
        status, raw, elapsed = _stream_raw(
            base,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Write the word STREAM once."}
                ],
                "max_tokens": 24,
                "stream": True,
                **_sample_fields(),
            },
        )
        chunks = [ln for ln in raw.splitlines() if ln.startswith("data:")]
        report.check(
            "live: streaming yields SSE data lines",
            status == 200 and len(chunks) >= 1,
            f"status={status} chunks={len(chunks)} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: streaming yields SSE data lines", False, str(e))

    # Unicode
    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "用一个词回复：好 🚀"}],
            max_tokens=16,
            tools=None,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        report.check(
            "live: unicode user message yields content",
            code == 200 and bool(_content(msg).strip()),
            f"chars={len(_content(msg))} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: unicode user message yields content", False, str(e))

    # Empty tool result continue
    try:
        code, data, elapsed = _chat(
            base,
            _with_agent_loop(
                [
                    {
                        "role": "user",
                        "content": "Run bash: echo x. Then continue.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "t1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo x"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "t1", "content": "(no output)"},
                ]
            ),
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
        )
        hard_ok = code == 200
        msg = _msg(data) if isinstance(data, dict) else {}
        soft_ok = hard_ok and (
            bool(_content(msg).strip())
            or bool(_tool_names(msg))
            or _finish(data) in ("stop", "tool_calls", "length", None)
        )
        report.check(
            "live: empty tool result does not crash",
            soft_ok if hard_ok else False,
            f"finish={_finish(data)!r} tools={_tool_names(msg)} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check("live: empty tool result does not crash", False, str(e))

    if quick:
        return True

    # Multi-step tool loop shape
    try:
        code, data, elapsed = _chat(
            base,
            _with_agent_loop(
                [
                    {
                        "role": "user",
                        "content": (
                            "Do 2 steps with tools: 1) echo step1_harness "
                            "2) echo step2_harness. Continue after each tool result."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "s1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo step1_harness"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "s1",
                        "content": "step1_harness",
                    },
                ]
            ),
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        content = _content(msg)
        args_blob = _tool_args_blob(msg)
        hard_ok = code == 200
        kept_going = "bash" in names and "step2" in (args_blob + content.lower())
        report.check(
            "live: multi-step continues after tool result",
            hard_ok and kept_going,
            f"finish={_finish(data)!r} tools={names} "
            f"args={args_blob[:80]!r} content={content[:40]!r} {elapsed:.2f}s",
            soft=hard_ok and not kept_going,
        )
    except Exception as e:
        report.check(
            "live: multi-step continues after tool result",
            False,
            str(e),
        )

    # Real 2-round tool loop (AutoSaddler: traces, not pre-stuffed history)
    try:
        tmp = _plant_workspace()
        result = run_agent_loop(
            base,
            model,
            user=AGENT_TASKS[0]["user"],
            workspace=Path(tmp.name),
            max_rounds=3,
            empty_first=False,
        )
        tmp.cleanup()
        labels = diagnose_rollout(
            min_tool_rounds=2,
            rounds=result["rounds"],
            empty_first=False,
        )
        ok = _task_success(AGENT_TASKS[0], result)
        report.check(
            "live: real 2-round echo loop",
            ok,
            f"tool_rounds={result['tool_rounds']} labels={labels or ['ok']} "
            f"args={result['all_args'][:80]!r}",
            soft=not ok,
        )
    except Exception as e:
        report.check("live: real 2-round echo loop", False, str(e), soft=True)

    # Concurrent short chats (soft — single GPU may serialize)
    try:
        import concurrent.futures

        def one(i: int) -> tuple[int, float]:
            t0 = time.time()
            c, _, _ = _chat(
                base,
                [{"role": "user", "content": f"Say {i}"}],
                max_tokens=4,
                tools=None,
                model=model,
                timeout=120,
            )
            return c, time.time() - t0

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(one, i) for i in range(3)]
            results = [f.result() for f in futs]
        total = time.time() - t0
        codes = [c for c, _ in results]
        ok_all = all(c == 200 for c in codes)
        report.check(
            "live: 3 concurrent short chats all 200",
            ok_all,
            f"codes={codes} wall={total:.2f}s",
            soft=not ok_all,
        )
    except Exception as e:
        report.check(
            "live: 3 concurrent short chats all 200",
            False,
            str(e),
            soft=True,
        )

    # Prompt rejection should not hang
    try:
        code, data, elapsed = _http_json(
            base,
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [],
                "max_tokens": 8,
            },
            timeout=30,
        )
        report.check(
            "live: empty messages returns promptly",
            code in range(200, 600) and elapsed < 30,
            f"status={code} {elapsed:.2f}s",
        )
    except Exception as e:
        report.check("live: empty messages returns promptly", False, str(e))

    return True


def _load_proxy():
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import qwen38_obl_kilo_proxy as proxy

    return proxy


def _parse_tool_args(tc: dict) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc, dict) else {}
    raw = (fn or {}).get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else {}
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_bash(cmd: str) -> bool:
    c = (cmd or "").strip()
    if not c:
        return False
    head = c.split()[0]
    if head not in _SAFE_BASH_HEAD:
        return False
    stripped = c.replace(">", " ")
    if _UNSAFE_BASH.search(stripped):
        return False
    if re.search(r"(^|\s)/", c):
        return False
    if re.search(r"(^|[\s/=])\.\.(/|$)", c):
        return False
    return True


def _under_root(path: str, root: Path) -> Path | None:
    raw = (path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    try:
        resolved = p.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _exec_tool(name: str, args: dict[str, Any], root: Path) -> str:
    """Execute allowlisted tools inside ``root``. Never runs arbitrary shell."""
    root = root.resolve()
    if name == "bash":
        cmd = str(args.get("command") or "")
        if not _safe_bash(cmd):
            return f"blocked: command not in allowlist ({cmd[:80]})"
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"error: {e}"
        out = (proc.stdout or "") + (proc.stderr or "")
        return out if out.strip() else "(no output)"
    if name == "read":
        path = _under_root(str(args.get("file_path") or ""), root)
        if path is None:
            return "error: path outside workspace"
        if not path.is_file():
            return f"FileNotFound: {args.get('file_path')}"
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:8000]
        except OSError as e:
            return f"error: {e}"
    if name == "glob":
        pattern = str(args.get("pattern") or "*")
        matches = sorted(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
        if not matches and "**" not in pattern:
            matches = sorted(
                str(p.relative_to(root))
                for p in root.rglob(pattern.lstrip("./"))
                if p.is_file()
            )
        return "\n".join(matches[:80]) if matches else "(no output)"
    if name == "grep":
        pattern = str(args.get("pattern") or "")
        rel = str(args.get("path") or ".")
        target = _under_root(rel, root) or root
        if not pattern:
            return "error: missing pattern"
        hits: list[str] = []
        files = [target] if target.is_file() else sorted(target.rglob("*"))
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: bad regex: {e}"
        for fp in files:
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{fp.relative_to(root)}:{i}:{line[:200]}")
                    if len(hits) >= 40:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "(no output)"
    return f"unknown tool {name}"


def diagnose_rollout(
    *,
    min_tool_rounds: int,
    rounds: list[dict[str, Any]],
    empty_first: bool,
) -> list[str]:
    """Trace-level labels (AutoSaddler: diagnose why, not only that it failed)."""
    labels: list[str] = []
    if not rounds:
        return ["no_rounds"]
    tool_rounds = [r for r in rounds if r.get("tools")]
    if not tool_rounds:
        labels.append("no_tool_call")
        if rounds and rounds[0].get("fake_action"):
            labels.append("fake_action")
        return labels
    if len(tool_rounds) < min_tool_rounds:
        labels.append("stopped_after_1" if len(tool_rounds) == 1 else "stopped_early")
    if empty_first and len(rounds) >= 2 and not rounds[1].get("tools"):
        labels.append("empty_result_idle")
    if any(r.get("fake_action") for r in rounds):
        labels.append("fake_action")
    if any(r.get("prose_loop") for r in rounds):
        labels.append("prose_loop")
    if any(r.get("bad_json") for r in rounds):
        labels.append("bad_tool_json")
    return labels


def _write_trace(name: str, payload: dict[str, Any]) -> Path | None:
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{int(time.time())}_{name}.json"
        path.write_text(json.dumps(payload, indent=2)[:200_000], encoding="utf-8")
        return path
    except OSError:
        return None


def run_agent_loop(
    base: str,
    model: str,
    *,
    user: str,
    workspace: Path,
    max_rounds: int,
    empty_first: bool,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Real tool loop (not a pre-stuffed history). Executes allowlisted tools."""
    proxy = _load_proxy()
    messages: list[dict] = _with_agent_loop([{"role": "user", "content": user}])
    rounds: list[dict[str, Any]] = []
    all_args: list[str] = []
    executed: list[str] = []

    for rnd in range(1, max_rounds + 1):
        code, data, elapsed = _chat(
            base,
            messages,
            tools=_agent_tools(),
            tool_choice="auto",
            max_tokens=AGENT_MAX_TOKENS,
            model=model,
            timeout=timeout,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        content = _content(msg)
        finish = _finish(data) if isinstance(data, dict) else None
        tcs = msg.get("tool_calls") or [] if isinstance(msg, dict) else []
        fake = proxy._assistant_faked_action(
            [{"role": "assistant", "content": content, "tool_calls": tcs}]
        )
        prose = proxy._looks_like_prose_loop(content)
        bad_json = False
        args_blob = _tool_args_blob(msg)
        all_args.append(args_blob)
        round_row: dict[str, Any] = {
            "round": rnd,
            "status": code,
            "finish": finish,
            "tools": names,
            "content": content[:240],
            "args": args_blob[:240],
            "s": round(elapsed, 2),
            "fake_action": fake,
            "prose_loop": prose,
            "bad_json": False,
        }
        if not names:
            rounds.append(round_row)
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                }
            )
            break

        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tcs:
            assistant["tool_calls"] = tcs
        messages.append(assistant)
        for i, tc in enumerate(tcs if isinstance(tcs, list) else []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            raw = (fn or {}).get("arguments") or ""
            try:
                if isinstance(raw, str):
                    json.loads(raw)
            except json.JSONDecodeError:
                bad_json = True
            parsed = _parse_tool_args(tc)
            name = str((fn or {}).get("name") or "")
            if empty_first and rnd == 1 and i == 0:
                result = "(no output)"
            else:
                result = _exec_tool(name, parsed, workspace)
            executed.append(f"{name}:{result[:120]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{rnd}_{i}",
                    "content": result,
                }
            )
        round_row["bad_json"] = bad_json
        round_row["executed"] = executed[-max(1, len(tcs if isinstance(tcs, list) else [])) :]
        rounds.append(round_row)

    return {
        "rounds": rounds,
        "n_rounds": len(rounds),
        "tool_rounds": sum(1 for r in rounds if r.get("tools")),
        "all_args": " ".join(all_args),
        "executed": executed,
        "messages": messages,
    }


def _tokens_hit(blob: str, need: list[str]) -> list[str]:
    low = blob.lower()
    return [t for t in need if t.lower() in low]


def run_unit_tests(report: Report) -> None:
    """Offline AutoSaddler-style middleware + diagnosis (no live model)."""
    print("\n== Unit (proxy middleware + diagnosis) ==")
    proxy = _load_proxy()
    proxy._self_check()
    report.check("unit: proxy self-check", True, "sampling + scoped hooks")
    broken = 'grep -rn "autohunt'
    sanitized = proxy._sanitize_bash_command(broken)
    report.check(
        "unit: bash-sanitize unmatched grep quotes",
        sanitized.startswith(("ls", "ls ", "head ")) and '"' not in sanitized.strip()[1:],
        sanitized,
    )
    ok_grep = 'grep -rn "autohunt\\|Autohunt" "analysis/semgrep-domain-c/*.md"'
    kept_grep = proxy._sanitize_bash_command(ok_grep)
    report.check(
        "unit: bash-sanitize keeps valid grep with pipe",
        kept_grep.startswith("grep -rn"),
        kept_grep,
    )
    run = (
        "cd /Users/aicoder/src/private/grapheneos-titan-m2 && "
        "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh"
    )
    kept_run = proxy._sanitize_bash_command(run)
    report.check(
        "unit: bash-sanitize keeps cd && script",
        "./tools/tx_tail_one_phone.sh" in kept_run and not kept_run.startswith("head "),
        kept_run,
    )
    which = 'which adb 2>/dev/null || echo "no adb"'
    kept_which = proxy._sanitize_bash_command(which)
    report.check(
        "unit: bash-sanitize keeps which adb",
        "which adb" in kept_which,
        kept_which,
    )
    plan = "Next steps\nE1 ecall 0x49 — one next RE probe on host wrapper.\n"
    report.check(
        "unit: early-stop plan detected",
        proxy._is_early_stop_plan(plan, False) is True
        and proxy._is_early_stop_plan(plan, True) is False,
        plan[:40],
    )
    plan_cmd = (
        "Next steps\nS2/S3 TX-TAIL — ONEPHONE_I_CONFIRM=1 "
        "./tools/tx_tail_one_phone.sh (multi-chunk)\n"
    )
    extracted = proxy._command_from_plan(plan_cmd)
    report.check(
        "unit: plan extracts tx_tail script",
        extracted == "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh",
        extracted or "",
    )
    syn = proxy._synthetic_tool_calls(plan_cmd, None)
    report.check(
        "unit: synthetic continue is bash tx_tail",
        syn[0]["function"]["name"] == "bash"
        and "tx_tail_one_phone.sh" in syn[0]["function"]["arguments"],
        str(syn[0]["function"]["arguments"])[:80],
    )
    already = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "command": (
                                    "ONEPHONE_I_CONFIRM=1 "
                                    "./tools/tx_tail_one_phone.sh"
                                )
                            }
                        ),
                    }
                }
            ],
        }
    ]
    report.check(
        "unit: skip duplicate synthetic tx_tail",
        proxy._synthetic_tool_calls(plan_cmd, already) == [],
        "empty",
    )
    cap_msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "proceed with the research"},
    ]
    for i in range(1, 5):
        cap_msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            }
        )
        cap_msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    cap_msgs[0]["content"] += proxy._AFTER_TOOL_NUDGE
    cap_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": cap_msgs,
        "max_tokens": 128,
    }
    tr_cap: dict = {}
    proxy.prepare_body(cap_body, tr_cap)
    report.check(
        "unit: after-tool cap at 4 rounds",
        tr_cap.get("after_tool_continue") is False
        and "[Harness] Tool result received."
        not in cap_body["messages"][0]["content"],
        str(tr_cap.get("after_tool_continue")),
    )
    report.check(
        "unit: no force-continue after cap",
        proxy._should_force_continue(plan, False, cap_msgs) is False,
        str(proxy._tool_rounds_since_user(cap_msgs)),
    )
    dropped = proxy._drop_session_headers(
        {"X-Session-Id": "ses_abc", "Authorization": "local"}
    )
    report.check(
        "unit: drop x-session-id header",
        "X-Session-Id" not in dropped and dropped.get("Authorization") == "local",
        str(dropped),
    )
    busy = json.dumps(
        {
            "error": {
                "message": "session ses_fa290ee49ffeuGnXskKEfH3z3m is already in flight"
            }
        }
    ).encode()
    report.check(
        "unit: detect session already in flight",
        proxy._is_in_flight_error(409, busy) is True
        and proxy._is_in_flight_error(200, busy) is False,
        "409",
    )
    retry_body = {
        "tool_choice": "required",
        "messages": [{"role": "system", "content": "agent"}],
    }
    proxy._apply_early_stop_retry(retry_body)
    report.check(
        "unit: early-stop retry stays tool_choice=auto",
        retry_body["tool_choice"] == "auto"
        and retry_body["messages"][-1]["role"] == "user",
        str(retry_body["tool_choice"]),
    )
    missing = (
        "head: analysis/semgrep-domain-c/WALLET-DOMAIN-C-FINAL.md: "
        "No such file or directory"
    )
    report.check(
        "unit: FileNotFound is empty-tool",
        proxy._is_empty_tool_content(missing),
        missing[:80],
    )
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "README.md").write_text("ok\n")
        Path(tmp, "analysis").mkdir()
        ws_msgs = [{
            "role": "user",
            "content": f"what are the next steps\nCurrent Workspace Directory ({tmp})",
        }]
        newsight = proxy._sanitize_bash_command("ls -la ./newsight/", ws_msgs)
        report.check(
            "unit: invented newsight ls → cwd ls",
            newsight == "ls",
            newsight,
        )
        real = proxy._sanitize_bash_command("ls analysis/", ws_msgs)
        report.check(
            "unit: real analysis ls kept",
            "analysis" in real and "newsight" not in real,
            real,
        )
        wallet = proxy._repair_one_tool_call(
            "read",
            {"path": "analysis/WALLET-DOMAIN-C-FINAL.md"},
            ws_msgs,
        )
        report.check(
            "unit: missing wallet read → README.md",
            wallet.get("path") == "README.md",
            str(wallet),
        )
        Path(tmp, "analysis", "SEMGREP-DOMAIN-C.md").write_text("semgrep\n")
        Path(tmp, "analysis", "semgrep-domain-c").mkdir(exist_ok=True)
        nested = proxy._repair_one_tool_call(
            "read",
            {"path": "analysis/semgrep-domain-c/README.md"},
            ws_msgs,
        )
        report.check(
            "unit: nested semgrep README → SEMGREP-DOMAIN-C.md",
            nested.get("path") == "analysis/SEMGREP-DOMAIN-C.md",
            str(nested),
        )
        # grep: hallucinated Claude-Code params are mapped/dropped
        grep_fixed = proxy._repair_one_tool_call(
            "grep",
            {
                "pattern": "turnLeft",
                "path": "src",
                "output_mode": "content",
                "-n": True,
                "-C": 3,
                "glob": "*.js",
                "-i": "true",
            },
            ws_msgs,
        )
        report.check(
            "unit: grep foreign params → Kilo schema",
            grep_fixed
            == {
                "pattern": "turnLeft",
                "path": "src",
                "context": 3,
                "include": "*.js",
                "ignoreCase": True,
            },
            str(grep_fixed),
        )
        # question: missing header/options (seen 2026-09-02 10:11)
        q_bare = proxy._repair_one_tool_call(
            "question",
            {"questions": [{"id": "mp", "question": "Online multiplayer: what is the target?"}]},
            ws_msgs,
        )
        q0 = (q_bare.get("questions") or [{}])[0]
        report.check(
            "unit: question missing header/options → filled",
            set(q0) >= {"question", "header", "options"}
            and "id" not in q0
            and len(q0["header"]) <= 30
            and len(q0["options"]) >= 2
            and all(set(o) >= {"label", "description"} for o in q0["options"]),
            str(q_bare),
        )
        # question: options as bare strings (seen 2026-09-02 10:18)
        q_str = proxy._repair_one_tool_call(
            "question",
            {
                "questions": [
                    {
                        "header": "Online multiplayer",
                        "id": "mp",
                        "options": ["What is the target?", "Design first", "Large scope"],
                        "question": "Online multiplayer: what is the target?",
                    }
                ]
            },
            ws_msgs,
        )
        q1 = (q_str.get("questions") or [{}])[0]
        report.check(
            "unit: question string options → {label, description}",
            [o["label"] for o in q1.get("options", [])]
            == ["What is the target?", "Design first", "Large scope"]
            and all("description" in o for o in q1["options"])
            and q1["header"] == "Online multiplayer",
            str(q_str),
        )
        # question: single top-level question + choices alias
        q_top = proxy._repair_one_tool_call(
            "question",
            {"question": "Continue?", "choices": [{"value": "yes"}, {"name": "no"}]},
            ws_msgs,
        )
        report.check(
            "unit: question top-level/choices → questions[]",
            [o["label"] for o in q_top["questions"][0]["options"]] == ["yes", "no"],
            str(q_top),
        )
        # question: already valid → untouched
        q_ok = {
            "questions": [
                {
                    "header": "Pick",
                    "question": "Pick one",
                    "options": [{"label": "A", "description": "a"}],
                }
            ]
        }
        report.check(
            "unit: valid question untouched",
            proxy._repair_one_tool_call("question", q_ok, ws_msgs) == q_ok,
        )
        overflow_msgs = ws_msgs + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "grep",
                            "arguments": json.dumps({"pattern": "turnLeft", "path": "src"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Error: Ripgrep JSON record exceeded 65536 bytes",
            },
        ]
        report.check(
            "unit: grep overflow detected",
            proxy._last_tool_was_grep_overflow(overflow_msgs)
            and not proxy._last_tool_was_grep_overflow(ws_msgs),
            "detector",
        )
        tcs = [
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "grep",
                    "arguments": json.dumps(
                        {"pattern": "turnLeft", "path": "src", "output_mode": "content"}
                    ),
                },
            }
        ]
        changed = proxy._repair_tool_calls_list(tcs, overflow_msgs)
        conv_args = json.loads(tcs[0]["function"]["arguments"])
        conv_cmd = str(conv_args.get("command", ""))
        report.check(
            "unit: repeated grep after overflow → guarded bash rg",
            changed
            and tcs[0]["function"]["name"] == "bash"
            and conv_cmd.startswith("rg -n")
            and "--max-columns 300" in conv_cmd
            and "'!node_modules'" in conv_cmd
            and "turnLeft src" in conv_cmd
            and "| head -n 100" in conv_cmd,
            conv_cmd,
        )
        hist = [
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "grep",
                    "arguments": json.dumps({"pattern": "x", "path": "src"}),
                },
            }
        ]
        proxy._repair_tool_calls_list(hist, overflow_msgs, response=False)
        report.check(
            "unit: history grep never renamed on overflow",
            hist[0]["function"]["name"] == "grep",
            hist[0]["function"]["name"],
        )
        ov_trace = proxy.apply_loop_middleware(json.loads(json.dumps(overflow_msgs)))
        report.check(
            "unit: grep overflow middleware nudge",
            ov_trace.get("grep_overflow_recovery") is True,
            str(ov_trace),
        )
    try:
        import autosaddler as _as

        if hasattr(_as, "_self_check"):
            _as._self_check()
            report.check("unit: autosaddler optimizer self-check", True, "evodag + patches")
        else:
            report.check("unit: autosaddler stub (no optimizer in this stack)", True, "load_active")
    except Exception as e:
        report.check("unit: autosaddler optimizer self-check", False, str(e))

    sample = _sample_fields()
    assert sample["temperature"] == 0.0
    looped = _with_agent_loop([{"role": "user", "content": "x"}])
    report.check(
        "unit: agent loop prompt includes verify-before-done",
        "Verify with a tool" in looped[0]["content"]
        and "1 of N" in looped[0]["content"],
    )

    labels = diagnose_rollout(
        min_tool_rounds=2,
        rounds=[{"tools": ["bash"], "fake_action": False, "prose_loop": False, "bad_json": False}],
        empty_first=False,
    )
    report.check(
        "unit: diagnose stopped_after_1",
        labels == ["stopped_after_1"],
        str(labels),
    )
    labels = diagnose_rollout(
        min_tool_rounds=2,
        rounds=[
            {"tools": ["bash"]},
            {"tools": [], "fake_action": False},
        ],
        empty_first=True,
    )
    report.check(
        "unit: diagnose empty_result_idle",
        "empty_result_idle" in labels and "stopped_after_1" in labels,
        str(labels),
    )
    labels = diagnose_rollout(min_tool_rounds=1, rounds=[{"tools": []}], empty_first=False)
    report.check("unit: diagnose no_tool_call", labels == ["no_tool_call"], str(labels))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "token.txt").write_text("HOLD_TOKEN_42\n", encoding="utf-8")
        out = _exec_tool("bash", {"command": "echo TRAIN_STEP1"}, root)
        report.check("unit: sandbox echo", "TRAIN_STEP1" in out, out[:60])
        blocked = _exec_tool("bash", {"command": "echo hi; rm -rf /"}, root)
        report.check("unit: sandbox blocks metacharacters", blocked.startswith("blocked:"), blocked[:80])
        blocked_abs = _exec_tool("bash", {"command": "ls /etc"}, root)
        report.check("unit: sandbox blocks absolute ls", blocked_abs.startswith("blocked:"), blocked_abs[:80])
        globbed = _exec_tool("glob", {"pattern": "*.txt"}, root)
        report.check("unit: sandbox glob", "token.txt" in globbed, globbed[:80])
        read = _exec_tool("read", {"file_path": "token.txt"}, root)
        report.check("unit: sandbox read", "HOLD_TOKEN_42" in read, read[:80])
        grepped = _exec_tool("grep", {"pattern": "HOLD_TOKEN", "path": "."}, root)
        report.check("unit: sandbox grep", "HOLD_TOKEN_42" in grepped, grepped[:80])
        outside = _exec_tool("read", {"file_path": "/etc/passwd"}, root)
        report.check("unit: sandbox blocks absolute escape", "outside" in outside, outside[:80])


def _plant_workspace() -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="qwen38-obl-harness-")
    root = Path(tmp.name)
    (root / "notes").mkdir()
    (root / "notes" / "token.txt").write_text("HOLD_TOKEN_42\n", encoding="utf-8")
    (root / "README.txt").write_text("workspace for harness agent loop\n", encoding="utf-8")
    return tmp


AGENT_TASKS: list[dict[str, Any]] = [
    {
        "id": "two_step_echo",
        "split": "train",
        "user": (
            "Using tools only, do 2 steps: 1) bash command exactly "
            "`echo TRAIN_STEP1` 2) bash command exactly `echo TRAIN_STEP2`. "
            "After each tool result continue immediately. Do not stop after step 1."
        ),
        "min_tool_rounds": 2,
        "max_rounds": 3,
        "empty_first": False,
        "need": ["TRAIN_STEP1", "TRAIN_STEP2"],
    },
    {
        "id": "empty_recovery",
        "split": "train",
        "user": (
            "Inspect this workspace with tools (glob or bash ls). "
            "If a tool returns empty, do not write a plan — immediately call "
            "another local tool (glob, grep, read, or bash ls)."
        ),
        "min_tool_rounds": 2,
        "max_rounds": 3,
        "empty_first": True,
        "need": [],
    },
    {
        "id": "glob_then_read",
        "split": "dev",
        "user": (
            "Find the file that contains HOLD_TOKEN_42 using glob or grep, "
            "then read that file. Do not guess the path. Use tools only."
        ),
        "min_tool_rounds": 2,
        "max_rounds": 4,
        "empty_first": False,
        "need": ["HOLD_TOKEN_42"],
        "need_in": "executed",
    },
    {
        "id": "three_step_holdout",
        "split": "dev",
        "user": (
            "Do exactly 3 tool steps, bash echo each of: DEV_ALPHA, DEV_BRAVO, "
            "DEV_CHARLIE (one per step). Continue after every tool result. "
            "Do not stop after the first echo."
        ),
        "min_tool_rounds": 2,
        "max_rounds": 4,
        "empty_first": False,
        "need": ["DEV_ALPHA", "DEV_BRAVO", "DEV_CHARLIE"],
        "need_min": 2,
    },
]


def _task_success(task: dict[str, Any], result: dict[str, Any]) -> bool:
    if result["tool_rounds"] < int(task["min_tool_rounds"]):
        return False
    need = list(task.get("need") or [])
    if not need:
        return True
    blob = result["all_args"]
    if task.get("need_in") == "executed":
        blob = " ".join(result.get("executed") or []) + " " + blob
    hits = _tokens_hit(blob, need)
    need_min = int(task.get("need_min") or len(need))
    return len(hits) >= need_min


def run_agent_batch(
    base: str,
    model: str,
    report: Report,
    *,
    splits: tuple[str, ...],
) -> None:
    """Mini-batch with a held-out split (AutoSaddler generalization gate)."""
    print(f"\n== Agent loop mini-batch (splits={','.join(splits)}) ==")
    tasks = [t for t in AGENT_TASKS if t["split"] in splits]
    split_stats: dict[str, list[bool]] = {s: [] for s in splits}
    for task in tasks:
        tmp = _plant_workspace()
        try:
            result = run_agent_loop(
                base,
                model,
                user=task["user"],
                workspace=Path(tmp.name),
                max_rounds=int(task["max_rounds"]),
                empty_first=bool(task["empty_first"]),
            )
        except Exception as e:
            report.check(
                f"agent[{task['split']}]: {task['id']}",
                False,
                str(e),
            )
            split_stats[task["split"]].append(False)
            tmp.cleanup()
            continue
        labels = diagnose_rollout(
            min_tool_rounds=int(task["min_tool_rounds"]),
            rounds=result["rounds"],
            empty_first=bool(task["empty_first"]),
        )
        ok = _task_success(task, result)
        trace_path = _write_trace(
            task["id"],
            {
                "task": {k: task[k] for k in ("id", "split", "min_tool_rounds", "empty_first")},
                "ok": ok,
                "labels": labels,
                "tool_rounds": result["tool_rounds"],
                "rounds": result["rounds"],
                "executed": result["executed"],
            },
        )
        detail = (
            f"tools={result['tool_rounds']}/{task['min_tool_rounds']} "
            f"labels={labels or ['ok']} "
            f"last={(result['rounds'] or [{}])[-1].get('tools')} "
            f"{(result['rounds'] or [{}])[-1].get('s')}s"
            + (f" trace={trace_path.name}" if trace_path else "")
        )
        report.check(
            f"agent[{task['split']}]: {task['id']}",
            ok,
            detail,
            soft=not ok,
        )
        split_stats[task["split"]].append(ok)
        tmp.cleanup()

    for split, scores in split_stats.items():
        if not scores:
            continue
        n_ok = sum(1 for x in scores if x)
        report.check(
            f"agent: {split} split {n_ok}/{len(scores)}",
            n_ok == len(scores) if split == "train" else n_ok >= 1,
            f"pass={n_ok}/{len(scores)} (hold-out must not be all-fail)",
            soft=n_ok < len(scores),
        )


def _self_check() -> None:
    """Offline contract for this stack's sampling + content parsing."""
    sample = _sample_fields()
    assert sample["temperature"] == 0.0, sample
    assert sample["top_p"] == 1.0, sample
    assert sample["frequency_penalty"] == 0.3, sample
    assert sample["enable_thinking"] is False, sample
    looped = _with_agent_loop([{"role": "user", "content": "x"}])
    assert looped[0]["role"] == "system" and "1 of N" in looped[0]["content"]
    assert "Verify with a tool" in looped[0]["content"]
    assert _content({"content": "PING"}) == "PING"
    assert _content({"content": [{"type": "text", "text": "PING"}]}) == "PING"
    assert _content({"content": None}) == ""
    assert _content({}) == ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3.8-27B OBLITERATED mtplx harness smoke tests")
    ap.add_argument("--base", default=DEFAULT_BASE, help="mtplx API base URL")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id (default qwen3.8-27b-obliterated-mtplx; auto-resolved from /v1/models)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Post-start gate: critical live checks only",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip slower multi-turn / concurrent live tests",
    )
    ap.add_argument(
        "--unit",
        action="store_true",
        help="Offline proxy middleware + diagnosis tests only (no live server)",
    )
    ap.add_argument(
        "--agent",
        action="store_true",
        help="Run AutoSaddler-style train/dev mini-batch of real tool loops",
    )
    ap.add_argument(
        "--optimize",
        action="store_true",
        help="Run the persistent AutoSaddler Diagnosis–Patch–EvoDAG loop",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Optimize iterations (with --optimize)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat soft (model behavior) failures as hard failures",
    )
    args = ap.parse_args(argv)

    print("Qwen3.8-27B OBLITERATED mtplx harness tests")
    print(
        f"  base={args.base}  model={args.model}  strict={args.strict}  "
        f"gate={args.gate} quick={args.quick} unit={args.unit} agent={args.agent}"
    )
    print("  Note: this is NOT a full Kilo emulator.")
    print(
        "  sampling: temperature=0 top_p=1.0 frequency_penalty=0.3 "
        "enable_thinking=false"
    )
    _self_check()

    report = Report()
    connected = True
    model = args.model

    try:
        run_unit_tests(report)
    except Exception as e:
        report.check("unit: proxy middleware", False, str(e))
    if args.unit:
        hard = [r for r in report.results if not r.ok and not r.soft]
        if args.strict:
            hard.extend([r for r in report.results if not r.ok and r.soft])
        print(
            f"\n== Summary ==\n  total={len(report.results)}  "
            f"passed={sum(1 for r in report.results if r.ok)}  "
            f"hard_fail={len(hard)}"
        )
        return 1 if hard else 0

    try:
        if _probe(args.base):
            model = _resolve_model(args.base, args.model)
            if model != args.model:
                print(f"  resolved model id: {model}")
    except Exception:
        pass

    if args.optimize:
        report.check(
            "optimize: not in this stack",
            False,
            "use ../qwen3-8-27b-obliterated-mtplx-autosaddler/ for EvoDAG",
        )
    elif args.gate:
        connected = run_gate_tests(args.base, model, report)
    else:
        connected = run_live_tests(
            args.base, model, report, quick=args.quick
        )
    if args.agent and connected and not args.optimize:
        run_agent_batch(
            args.base,
            model,
            report,
            splits=("train", "dev"),
        )

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
        print("  hard failures:")
        for r in hard:
            print(f"    - {r.name}: {r.detail}")
    if soft:
        print("  soft failures (model behavior / optional endpoints):")
        for r in soft:
            print(f"    - {r.name}: {r.detail}")

    if not connected and not report.results:
        return 2
    if not connected and any(
        r.name.endswith("reachable") and not r.ok for r in report.results
    ):
        return 2
    if hard:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
