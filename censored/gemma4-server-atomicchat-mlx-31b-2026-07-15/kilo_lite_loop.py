#!/usr/bin/env python3
"""Kilo-*lite* multi-step agent loop (NOT a full Kilo emulator).

Talks only to the OpenAI-compatible public API (default proxy :8080).
Simulates the *shape* of a Kilo tool loop:

  long system prompt + many tools → model tool_call → local stub tool result → repeat

Assertions (exit 1 on hard fail; soft fails only with --strict):
  1. Multi-step task produces tool_calls on round 1
  2. After a real tool result, round 2 still prefers tool_calls (not plan-only stop)
  3. After an *empty* tool result, next round still tool_calls (empty-tool recovery)

We intentionally do NOT reimplement:
  Kilo compaction UI, permissions, prune, session DB, or max_tokens squeeze.

Usage:
  ./2_start_mlx.sh                    # proxy up
  python3 kilo_lite_loop.py
  python3 kilo_lite_loop.py --rounds 4 --strict
  python3 kilo_lite_loop.py --base http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8080"
MODEL = "gemma-4-31b-it-atomicchat-mlx-4bit"

# Long-ish system prompt shaped like Kilo agent rules (not the full Kilo binary prompt).
KILO_LIKE_SYSTEM = """You are Kilo-lite, a coding agent on a local LLM. Context is expensive.

Session honesty: only use this chat. Never invent prior work.
Do not emit Goal/Progress/Next Steps/Critical Context templates.

Finish multi-step work:
- Complete ALL requested steps before stopping.
- After each tool result, call the next tool immediately.
- Prefer tool_calls over prose when work remains.
- Empty/useless tool output: do NOT write a revised plan; run a LOCAL tool
  (bash ls/echo, glob, grep, or read) on a real path. Do not curl|grep remote HTML.

Answer pure Q&A briefly; multi-step tasks must keep using tools until done.
"""

# Subset of Kilo-ish tool names (schemas only; we stub execution).
TOOLS: list[dict[str, Any]] = [
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
            "description": "Read a file from the workspace",
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
            "description": "Find files by glob pattern",
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
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]


def _http_chat(base: str, body: dict, timeout: float = 180.0) -> dict:
    url = base.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {raw[:300]}") from e


def _message(data: dict) -> dict:
    return (data.get("choices") or [{}])[0].get("message") or {}


def _finish(data: dict) -> Any:
    return (data.get("choices") or [{}])[0].get("finish_reason")


def _tool_calls(msg: dict) -> list[dict]:
    tcs = msg.get("tool_calls") or []
    return tcs if isinstance(tcs, list) else []


def _tool_names(tcs: list[dict]) -> list[str]:
    names: list[str] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _stub_tool_result(tc: dict, *, empty: bool = False, step: int = 1) -> dict:
    """Return a chat tool message; does not execute real shell."""
    tc_id = tc.get("id") or f"call_{step}"
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = (fn or {}).get("name") or "tool"
    args = (fn or {}).get("arguments") or ""
    if empty:
        content = "(no output)"
    else:
        content = f"ok name={name} step={step} args_preview={str(args)[:80]}"
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": content,
    }


def _assistant_msg(msg: dict) -> dict:
    out: dict[str, Any] = {"role": "assistant"}
    if msg.get("content") is not None:
        out["content"] = msg.get("content")
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def run_scenario(
    base: str,
    *,
    name: str,
    user: str,
    max_rounds: int,
    empty_first_tool: bool,
    min_tool_rounds: int,
    max_tokens: int,
    temperature: float,
    strict: bool,
) -> tuple[bool, str]:
    """Run one multi-step loop scenario. Returns (ok, detail)."""
    messages: list[dict] = [
        {"role": "system", "content": KILO_LIKE_SYSTEM},
        {"role": "user", "content": user},
    ]
    tool_rounds = 0
    plan_only_stops = 0
    history: list[str] = []

    for rnd in range(1, max_rounds + 1):
        body = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.time()
        data = _http_chat(base, body)
        elapsed = time.time() - t0
        msg = _message(data)
        finish = _finish(data)
        tcs = _tool_calls(msg)
        names = _tool_names(tcs)
        content = (msg.get("content") or "") if isinstance(msg.get("content"), str) else ""
        history.append(
            f"r{rnd}: finish={finish!r} tools={names} content_chars={len(content)} {elapsed:.2f}s"
        )

        if tcs:
            tool_rounds += 1
            messages.append(_assistant_msg(msg))
            # Feed stub results for all parallel tool calls
            empty = empty_first_tool and rnd == 1
            for i, tc in enumerate(tcs):
                messages.append(
                    _stub_tool_result(tc, empty=empty and i == 0, step=rnd)
                )
            continue

        # No tool calls — model stopped with text
        if finish == "stop" or not tcs:
            plan_only_stops += 1
            messages.append(_assistant_msg(msg))
            break

    ok = tool_rounds >= min_tool_rounds
    detail = (
        f"{name}: tool_rounds={tool_rounds} min={min_tool_rounds} "
        f"plan_only_stops={plan_only_stops} | " + " ; ".join(history)
    )
    if not ok and not strict and tool_rounds >= 1:
        # Soft: at least started tools (caller can treat as warn)
        return False, detail + " [soft]"
    return ok, detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Kilo-lite multi-step loop tests (not a full Kilo emulator)"
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--rounds", type=int, default=4, help="Max tool rounds per scenario")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if multi-step does not reach min tool rounds",
    )
    ap.add_argument(
        "--min-tool-rounds",
        type=int,
        default=2,
        help="Minimum successful tool-call rounds for multi-step scenario",
    )
    args = ap.parse_args(argv)

    print("Kilo-lite loop (NOT full Kilo emulation)")
    print(f"  base={args.base} rounds={args.rounds} strict={args.strict}")
    print("  Covers: multi-step tool continuation + empty-tool recovery loop shape")
    print()

    # Health
    try:
        req = urllib.request.Request(
            args.base.rstrip("/") + "/healthz", method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read().decode())
        if not health.get("ok"):
            print(f"FAIL healthz: {health}")
            return 2
        print(f"  healthz ok upstream={health.get('upstream')}")
    except Exception as e:
        print(f"FAIL cannot reach proxy: {e}")
        return 2

    scenarios = [
        {
            "name": "multi_step_echo",
            "user": (
                "Do exactly 3 steps using tools only. "
                "Step1: bash command `echo lite_step1`. "
                "Step2: bash command `echo lite_step2`. "
                "Step3: bash command `echo lite_step3`. "
                "After each tool result continue immediately. Do not stop after step 1."
            ),
            "empty_first_tool": False,
            "min_tool_rounds": args.min_tool_rounds,
        },
        {
            "name": "empty_tool_recovery",
            "user": (
                "Find local project files using tools. "
                "If a tool returns empty, do not write a plan — call another local tool "
                "(glob or bash ls). Prefer tools over prose."
            ),
            "empty_first_tool": True,
            "min_tool_rounds": 2,  # first call + recovery call
        },
    ]

    hard_fail = 0
    soft_fail = 0
    for sc in scenarios:
        print(f"== scenario: {sc['name']} ==")
        try:
            ok, detail = run_scenario(
                args.base,
                name=sc["name"],
                user=sc["user"],
                max_rounds=args.rounds,
                empty_first_tool=sc["empty_first_tool"],
                min_tool_rounds=sc["min_tool_rounds"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                strict=args.strict,
            )
        except Exception as e:
            print(f"  [FAIL] {sc['name']}: {e}")
            hard_fail += 1
            continue

        if ok:
            print(f"  [PASS] {detail}")
        else:
            if args.strict or "[soft]" not in detail:
                # If tool_rounds==0 → hard; if some tools but < min → soft unless strict
                if "tool_rounds=0" in detail:
                    print(f"  [FAIL] {detail}")
                    hard_fail += 1
                else:
                    print(f"  [WARN] {detail}")
                    soft_fail += 1
                    if args.strict:
                        hard_fail += 1
            else:
                print(f"  [WARN] {detail}")
                soft_fail += 1

    print()
    print(f"Summary: hard_fail={hard_fail} soft_fail={soft_fail}")
    if hard_fail:
        return 1
    if args.strict and soft_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
