#!/usr/bin/env python3
"""Standalone harness resilience tests for Qwen3.8-27B OBLITERATED + mtplx.

Runs *outside* Kilo against the OpenAI-compatible mtplx API (default :8767).

Usage:
  ./2_start_mtplx.sh                 # server already up
  python3 test_harness.py
  python3 test_harness.py --base http://127.0.0.1:8767 --strict
  python3 test_harness.py --gate     # post-start gate (critical live only)
  python3 test_harness.py --quick    # skip slower multi-turn / concurrent tests
  python3 test_harness.py --model qwen3.8-27b-obliterated-mtplx

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
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE = "http://127.0.0.1:8767"
DEFAULT_MODEL = "qwen3.8-27b-obliterated-mtplx"

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
    temperature: float = 0.6,
    model: str = DEFAULT_MODEL,
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
            temperature=0.2,
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
            [
                {
                    "role": "user",
                    "content": (
                        "Using tools only, run bash with command exactly: "
                        "echo harness_ok. Do not explain."
                    ),
                }
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
            model=model,
            temperature=0.3,
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
                "temperature": 0.2,
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
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=48,
            model=model,
            temperature=0.3,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        content = _content(msg).lower()
        hard_ok = code == 200
        # Accept either a short reply or another tool call — both prove multi-turn works.
        soft_ok = hard_ok and (
            "done" in content
            or "step1" in content
            or bool(_tool_names(msg))
            or bool(content.strip())
        )
        report.check(
            "gate: multi-turn tool result continues",
            soft_ok if hard_ok else False,
            f"content={_content(msg)[:50]!r} tools={_tool_names(msg)} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
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
            [
                {
                    "role": "user",
                    "content": (
                        "Call the bash tool with command exactly: "
                        "echo harness_ok"
                    ),
                }
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
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
                "temperature": 0.2,
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
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=64,
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
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=96,
            model=model,
        )
        msg = _msg(data) if isinstance(data, dict) else {}
        names = _tool_names(msg)
        content = _content(msg)
        hard_ok = code == 200
        soft_ok = hard_ok and (
            "bash" in names
            or "step2" in content.lower()
            or bool(content.strip())
        )
        report.check(
            "live: multi-step continues after tool result",
            soft_ok if hard_ok else False,
            f"finish={_finish(data)!r} tools={names} "
            f"content={content[:40]!r} {elapsed:.2f}s",
            soft=hard_ok and not soft_ok,
        )
    except Exception as e:
        report.check(
            "live: multi-step continues after tool result",
            False,
            str(e),
        )

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
        "--strict",
        action="store_true",
        help="Treat soft (model behavior) failures as hard failures",
    )
    args = ap.parse_args(argv)

    print("Qwen3.8-27B OBLITERATED mtplx harness tests")
    print(
        f"  base={args.base}  model={args.model}  strict={args.strict}  "
        f"gate={args.gate} quick={args.quick}"
    )
    print("  Note: this is NOT a full Kilo emulator.")

    report = Report()
    connected = True
    model = args.model

    if not args.gate:
        # Resolve model id early when server is up
        try:
            if _probe(args.base):
                model = _resolve_model(args.base, args.model)
                if model != args.model:
                    print(f"  resolved model id: {model}")
        except Exception:
            pass

    if args.gate:
        try:
            if _probe(args.base):
                model = _resolve_model(args.base, args.model)
        except Exception:
            pass
        connected = run_gate_tests(args.base, model, report)
    else:
        connected = run_live_tests(
            args.base, model, report, quick=args.quick
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
