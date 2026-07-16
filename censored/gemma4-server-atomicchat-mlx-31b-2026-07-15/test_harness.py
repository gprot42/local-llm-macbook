#!/usr/bin/env python3
"""Standalone harness resilience tests for AtomicChat Gemma 4 + gemma4_kilo_proxy.

Runs *outside* Kilo against the OpenAI-compatible public API (default :8080).

Usage:
  ./2_start_mlx.sh                 # proxy + engine already up
  python3 test_harness.py
  python3 test_harness.py --base http://127.0.0.1:8080 --strict
  python3 test_harness.py --unit-only   # pure proxy helpers, no network

Exit codes:
  0  all required checks passed
  1  one or more required checks failed
  2  soft/connectivity failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Local pure-function tests (no server required)
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
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    soft: bool = False  # soft failures only fail under --strict


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", soft: bool = False) -> None:
        self.results.append(CheckResult(name, ok, detail, soft))
        status = "PASS" if ok else ("WARN" if soft else "FAIL")
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def hard_failed(self) -> bool:
        return any(not r.ok and not r.soft for r in self.results)

    def soft_failed(self) -> bool:
        return any(not r.ok and r.soft for r in self.results)


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


def _chat(
    base: str,
    messages: list[dict],
    *,
    tools: list | None = None,
    tool_choice: Any = None,
    max_tokens: int = 128,
    stream: bool = False,
    temperature: float = 0.35,
    timeout: float = 180.0,
) -> tuple[int, dict | Any, float]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return _http_json(base, "POST", "/v1/chat/completions", body, timeout=timeout)


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


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


def run_unit_tests(report: Report) -> None:
    print("\n== Unit (gemma4_kilo_proxy helpers) ==")

    # Compaction false positive on system prompt
    body = {
        "messages": [
            {
                "role": "system",
                "content": "Do not re-summarize the conversation history unless asked.",
            },
            {"role": "user", "content": "list local files with tools"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    try:
        assert proxy._looks_like_compaction(body) is False
        tr = proxy._prepare_body(body)
        assert "tools" in body and body["tool_choice"] == "auto"
        assert tr["compaction"] is False
        assert tr["tools_out"] == len(TOOLS)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        report.add("unit: tools not stripped on system 'summarize' text", True)
    except AssertionError as e:
        report.add("unit: tools not stripped on system 'summarize' text", False, str(e))

    # Compaction on tool_choice none
    body2 = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "tool_choice": "none",
        "max_tokens": 8192,
    }
    try:
        assert proxy._looks_like_compaction(body2) is True
        tr2 = proxy._prepare_body(body2)
        assert "tools" not in body2
        assert tr2["compaction"] is True
        report.add("unit: tool_choice=none → compaction strip", True)
    except AssertionError as e:
        report.add("unit: tool_choice=none → compaction strip", False, str(e))

    # Empty tool recovery
    body3 = {
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
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    try:
        assert proxy._recent_empty_tool_streak(body3["messages"]) >= 1
        tr3 = proxy._prepare_body(body3)
        assert tr3["empty_tool_recovery"] is True
        assert "[Harness] EMPTY TOOL RESULT:" in body3["messages"][0]["content"]
        report.add("unit: empty tool recovery nudge", True)
    except AssertionError as e:
        report.add("unit: empty tool recovery nudge", False, str(e))

    # Reasoning remap
    try:
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "OK",
                    }
                }
            ]
        }
        assert proxy._remap_reasoning_in_completion_payload(data)
        assert data["choices"][0]["message"]["content"] == "OK"
        report.add("unit: reasoning → content remap", True)
    except AssertionError as e:
        report.add("unit: reasoning → content remap", False, str(e))


# ---------------------------------------------------------------------------
# Live contract tests
# ---------------------------------------------------------------------------


def run_live_tests(base: str, report: Report, *, strict: bool) -> None:
    print(f"\n== Live contract ({base}) ==")

    # Health
    try:
        code, data, elapsed = _http_json(base, "GET", "/healthz", timeout=5)
        ok = code == 200 and isinstance(data, dict) and data.get("ok") is True
        report.add(
            "live: GET /healthz",
            ok,
            f"status={code} body={data!r} {elapsed:.2f}s",
        )
        if not ok:
            return
    except Exception as e:
        report.add("live: GET /healthz", False, str(e))
        return

    # Models
    try:
        code, data, elapsed = _http_json(base, "GET", "/v1/models", timeout=10)
        ok = code == 200 and isinstance(data, dict) and "data" in data
        report.add("live: GET /v1/models", ok, f"status={code} {elapsed:.2f}s")
    except Exception as e:
        report.add("live: GET /v1/models", False, str(e))

    # Content not reasoning-only (thinking off)
    try:
        code, data, elapsed = _chat(
            base,
            [{"role": "user", "content": "Reply with exactly the word PONG."}],
            max_tokens=16,
            tools=None,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        ok = (
            code == 200
            and isinstance(data, dict)
            and bool(content)
            and (not reasoning or content)
        )
        report.add(
            "live: thinking off → non-empty content",
            ok,
            f"finish={_finish(data)!r} content={content[:40]!r} "
            f"has_reasoning={bool(reasoning)} {elapsed:.2f}s",
        )
    except Exception as e:
        report.add("live: thinking off → non-empty content", False, str(e))

    # Tools preserved despite system summarize wording
    try:
        code, data, elapsed = _chat(
            base,
            [
                {
                    "role": "system",
                    "content": (
                        "Do not re-summarize the conversation history. "
                        "Preserve key information. Create a concise summary only if asked."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Using tools only, run bash with command exactly: echo harness_ok. "
                        "Do not answer in plain text."
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
        # Hard: request succeeded and did not crash. Prefer tool_calls.
        hard_ok = code == 200 and isinstance(data, dict)
        soft_ok = hard_ok and (finish == "tool_calls" or "bash" in names)
        report.add(
            "live: tools kept (system summarize bait) → tool_calls preferred",
            soft_ok if not strict else soft_ok,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=not soft_ok and hard_ok,
        )
        if not hard_ok:
            report.results[-1].soft = False
            report.results[-1].ok = False
    except Exception as e:
        report.add(
            "live: tools kept (system summarize bait) → tool_calls preferred",
            False,
            str(e),
        )

    # Empty tool recovery → local tool preferred (soft if model ignores nudge)
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
        # Success: any tool call (bash/read/glob) rather than long plan-only stop
        used_tool = finish == "tool_calls" or bool(names)
        # Prefer non-curl local tools when possible
        localish = any(n in ("glob", "read", "grep") for n in names) or (
            "bash" in names
        )
        hard_ok = code == 200 and isinstance(data, dict)
        soft_ok = hard_ok and used_tool and localish
        report.add(
            "live: empty tool result → continues with tools (not plan-only)",
            soft_ok,
            f"finish={finish!r} tools={names} {elapsed:.2f}s",
            soft=not soft_ok and hard_ok,
        )
        if not hard_ok:
            report.results[-1].soft = False
            report.results[-1].ok = False
    except Exception as e:
        report.add(
            "live: empty tool result → continues with tools (not plan-only)",
            False,
            str(e),
        )

    # Stream: content deltas present (not reasoning-only)
    try:
        import http.client
        from urllib.parse import urlparse

        u = urlparse(base)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
        body = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 12,
                "temperature": 0.35,
                "stream": True,
            }
        ).encode()
        t0 = time.time()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        elapsed = time.time() - t0
        conn.close()
        has_content = '"content"' in raw
        has_only_reasoning = (
            '"reasoning"' in raw and '"content"' not in raw.replace('"content": null', "")
        )
        # After proxy rewrite we expect content fields
        ok = resp.status == 200 and has_content and not (
            has_only_reasoning and "content" not in raw
        )
        # Simpler: at least one non-null content token in stream
        ok = resp.status == 200 and (
            '"content": "OK"' in raw
            or '"content":"OK"' in raw
            or '"content": "O"' in raw
            or '"content":"' in raw
        )
        report.add(
            "live: stream has content deltas",
            ok,
            f"status={resp.status} content_marker={has_content} {elapsed:.2f}s",
        )
    except Exception as e:
        report.add("live: stream has content deltas", False, str(e))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AtomicChat Kilo harness smoke tests")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Public API base (proxy)")
    ap.add_argument(
        "--unit-only",
        action="store_true",
        help="Only pure-function unit tests (no server)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat soft (model behavior) failures as hard failures",
    )
    args = ap.parse_args(argv)

    print("AtomicChat harness tests")
    print(f"  base={args.base}  strict={args.strict}  unit_only={args.unit_only}")

    report = Report()
    run_unit_tests(report)

    if not args.unit_only:
        run_live_tests(args.base, report, strict=args.strict)

    hard = [r for r in report.results if not r.ok and not r.soft]
    soft = [r for r in report.results if not r.ok and r.soft]
    passed = [r for r in report.results if r.ok]

    print("\n== Summary ==")
    print(f"  passed={len(passed)}  hard_fail={len(hard)}  soft_fail={len(soft)}")
    if hard:
        print("  HARD FAILURES:")
        for r in hard:
            print(f"    - {r.name}: {r.detail}")
    if soft:
        print("  SOFT (model) FAILURES:" + (" [strict]" if args.strict else " [ignored]"))
        for r in soft:
            print(f"    - {r.name}: {r.detail}")

    if hard:
        return 1
    if args.strict and soft:
        return 1
    if not args.unit_only and not any(
        r.name.startswith("live:") and r.ok for r in report.results
    ):
        # connectivity: no live passes at all
        if any(r.name.startswith("live:") for r in report.results):
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
