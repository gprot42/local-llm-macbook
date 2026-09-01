#!/usr/bin/env python3
"""Kilo steering proxy for Qwen3.8-27B OBLITERATED + mtplx.

Kilo's *global* agent config is Muse Glimmer (temperature 1.0, long prompt).
That overrides mtplx ``--default-temperature 0`` and the OBLITERATUS card
(greedy + repetition_penalty + thinking off). Agents then look lazy or loop.

This proxy sits in front of mtplx and:

  1. Forces card sampling: temperature=0, top_p=1.0, frequency_penalty=0.3
     (mtplx stand-in for HF repetition_penalty=1.15), enable_thinking=false
  2. Floors max_tokens at 2048 on agentic (tools) turns
  3. Prepends a short finish-the-job loop nudge on tool turns
  4. Strips tools on Kilo compaction / summary
  5. Soft-repairs truncated tool-call argument JSON
  6. Capability middleware (scoped — not always-on):
     empty-tool recovery, fake-action recovery, prose-loop recovery,
     just-in-time continue after a tool result, cap huge tool outputs
  7. Auto-fix tool calls (same as Ornith): bash-sanitize *broken* commands
     only (unmatched quotes, invented paths, command substitution, missing
     cat/head files). Do not rewrite cd/&&, env=value ./script, which, pipes,
     or 2>/dev/null — those were causing the agent to stop after dummy ls/head
  8. Response-side early-stop: after a tool result, if the model dumps a
     Next-steps/plan with no tool_calls, retry once with a user nudge
     (mtplx only accepts tool_choice auto|none — do not send required).
     If it still will not call a tool, synthesize one from the plan
     (e.g. ./tools/*.sh) so Kilo cannot end the turn on a recap.


Usage:
  python3 qwen38_obl_kilo_proxy.py --upstream http://127.0.0.1:8767 --port 8768
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
import threading
from collections import Counter
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autosaddler import (  # noqa: E402
    AS_LOOP_END,
    AS_LOOP_START,
    append_live_event,
    load_active,
)

log = logging.getLogger("qwen38_obl_kilo_proxy")
LOG_FILE = (
    Path(__file__).resolve().parent / ".qwen38_obl_proxy.log"
)

CARD_TEMPERATURE = 0.0
CARD_TOP_P = 1.0
CARD_FREQUENCY_PENALTY = 0.3
AGENT_MAX_TOKENS_FLOOR = 2048
LOOP_MARKER = "Do not stop after 1 of N."
# Keep this short (finite attention). Extra recovery lives in scoped middleware
# below — AutoSaddler: capability/loop logic first, then steering; prefer
# replacement over stacking more rules.
LOOP_PREFIX = (
    "Finish every requested step. Prefer tools over prose. After each tool "
    "result, immediately take the next action. Do not stop after 1 of N. "
    "Do not recap; act. Empty or error tool output: retry with a simpler "
    "local command (ls/glob/grep/read). Verify with a tool before declaring "
    "the job done.\n\n"
)

# AutoSaddler TB2 infra patch: raise tool-output room, still cap context.
TOOL_RESULT_MAX_CHARS = 30_000
TOOL_RESULT_KEEP_HEAD = 12_000
TOOL_RESULT_KEEP_TAIL = 8_000

_EMPTY_TOOL_EXACT = frozenset(
    {
        "(no output)",
        "no output",
        "(empty)",
        "empty",
        "empty response",
        "null",
        "none",
        "[]",
        "{}",
        "''",
        '""',
        "no matches",
        "no matches found",
        "(no matches)",
    }
)

_FAKE_ACTION_RE = re.compile(
    r"(?is)"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:next steps?|to-?dos?)\b"
    r"|(?:^|\n)\s*(?:executing|running|probing|fetching)\b"
    r"|\bi(?:'| a)?m\s+(?:going\s+to|about\s+to)\s+"
    r"|\bi(?:'| wi)?ll\s+(?:start|probe|check|fetch|run|search|use|look|verify)\b"
    r"|\blet\s+me\s+(?:probe|check|fetch|run|search|verify|start)\b"
    r"|\bnext[,:]?\s+i\s+will\b"
    r"|\bi\s+will\s+(?:start|probe|check|fetch|run|search)\b"
)
_EARLY_STOP_PLAN_RE = re.compile(
    r"(?is)"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:next steps?|to-?dos?|plan)\b"
    r"|\bi(?:'| a)?m\s+(?:going\s+to|about\s+to)\s+"
    r"|\bi(?:'| wi)?ll\s+(?:start|probe|check|fetch|run|search|use|look|verify|read|list)\b"
    r"|\blet\s+me\s+(?:probe|check|fetch|run|search|verify|start|read|list)\b"
    r"|\bnext[,:]?\s+i\s+will\b"
)

_EMPTY_TOOL_NUDGE = (
    "\n\n[Harness] EMPTY TOOL RESULT: the latest tool output was empty or "
    "useless. Do not write a revised plan. Next message MUST be a local tool "
    "(bash ls/echo, glob, grep, or read) on a real path. If FileNotFound, "
    "list the parent directory — do not idle."
)
_MISSING_PATH_NUDGE = (
    "\n\n[Harness] MISSING PATH: that file or directory is not in this "
    "workspace. Do not invent a project. Next message MUST read README.md. "
    "Do not ls or cat the missing name again. Do not continue a fake task."
)
_FAKE_ACTION_NUDGE = (
    "\n\n[Harness] FAKE ACTION: the last assistant message claimed to act "
    "('Executing…', 'I will check…') but emitted NO tool_calls. Next message "
    "MUST include tool_calls — run the command now. No plans."
)
_PROSE_LOOP_NUDGE = (
    "\n\n[Harness] PROSE LOOP: the last assistant turn repeated itself "
    "without tool_calls. Stop monologuing. Next message MUST be tool_calls "
    "only — take the next unfinished step."
)
_AFTER_TOOL_NUDGE = (
    "\n\n[Harness] Tool result received. If any requested step is unfinished, "
    "emit the next tool_call now. Do not recap. Verify before stopping."
)
_EARLY_STOP_NUDGE = (
    "\n\n[Harness] EARLY STOP: that was a plan/next-steps dump, not a finished "
    "job. Next message MUST be tool_calls for the next unfinished step. "
    "Do not recap."
)
_RETRY_USER = (
    "[Harness] EARLY STOP: emit tool_calls for the next unfinished step now. "
    "No Next steps list. No recap. Run the next command."
)
_USER_ACTION_RE = re.compile(
    r"(?is)\b(?:continue|keep going|do it|run it|go ahead|"
    r"keep (?:working|going)|finish|research)\b"
)
_PLAN_SCRIPT_RE = re.compile(
    r"(?:ONEPHONE_I_CONFIRM=1\s+)?(?:\./)?tools/[\w.-]+\.sh"
)
_PROXY_EMPTY_HINT = "[proxy] Command printed nothing"
_CMD_SUB_RE = re.compile(r"`|\$\(")
_SAFE_BASH_STARTS = (
    "cat ", "head ", "tail ", "less ", "ls ", "tree ", "pwd",
    "echo ", "wc ", "grep ", "rg ", "ag ", "jq ", "file ",
    "diff ", "du ", "df ", "date", "uname", "whoami",
    "which ", "cd ", "adb ",
)
_CAT_PATH_RE = re.compile(
    r"(?:^|[\s;|&])(?:cat|head|tail|less)\s+(?:-[nN]\s+\d+\s+)?['\"]?"
    r"((?:/|~/|\./)?[\w./~-]+\.[A-Za-z0-9]+)",
    re.I,
)
_QUOTED_FILE_RE = re.compile(
    r"""["']((?:[\w./~-]+)\.[A-Za-z0-9]+)["']"""
)
_FILENAME_RE = re.compile(
    r"(?<![\w.~-])(/?[\w./~-]+\.(?:sh|py|js|ts|tsx|jsx|json|md|txt|yaml|yml|toml|rs|go|rb|java|kt|swift|c|h|cpp|hpp|mk))\b",
    re.IGNORECASE,
)
_NEXT_STEP_MAX_READS = 4


def _active_harness() -> dict[str, Any]:
    try:
        return load_active()
    except Exception:
        return {}


def _loop_prefix() -> str:
    text = str(_active_harness().get("loop_prefix") or LOOP_PREFIX).strip()
    return text + "\n\n"


def _empty_tool_nudge() -> str:
    return str(_active_harness().get("empty_tool_nudge") or _EMPTY_TOOL_NUDGE)


def _fake_action_nudge() -> str:
    return str(_active_harness().get("fake_action_nudge") or _FAKE_ACTION_NUDGE)


def _prose_loop_nudge() -> str:
    return str(_active_harness().get("prose_loop_nudge") or _PROSE_LOOP_NUDGE)

_COMPACTION_HINTS = (
    "summarize the conversation",
    "generate a brief summary",
    "compact the conversation",
    "create a concise summary",
    "conversation summary",
    "agent=compaction",
)


def _looks_like_compaction(body: dict) -> bool:
    tc = body.get("tool_choice")
    if tc == "none" or (isinstance(tc, dict) and tc.get("type") == "none"):
        return True
    # Kilo agent system prompts often mention "summarize the conversation".
    # That is not compaction. Never strip tools on a real tool turn.
    tools = body.get("tools") or body.get("functions") or []
    if isinstance(tools, list) and tools:
        return False
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("system", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            content = "\n".join(parts)
        if not isinstance(content, str):
            continue
        low = content.lower()
        if any(h in low for h in _COMPACTION_HINTS):
            return True
    return False


def _message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _set_message_text(msg: dict, text: str) -> None:
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text"):
                block["text"] = text
                return
        content.append({"type": "text", "text": text})
        return
    msg["content"] = text


def _truncate_tool_text(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    head = text[:TOOL_RESULT_KEEP_HEAD]
    tail = text[-TOOL_RESULT_KEEP_TAIL:]
    skipped = len(text) - TOOL_RESULT_KEEP_HEAD - TOOL_RESULT_KEEP_TAIL
    return (
        f"{head}\n... [proxy: truncated {skipped} chars of tool output] ...\n{tail}"
    )


def _truncate_tool_messages(messages: list[dict]) -> int:
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("tool", "function"):
            continue
        text = _message_text(msg)
        if not text or len(text) <= TOOL_RESULT_MAX_CHARS:
            continue
        _set_message_text(msg, _truncate_tool_text(text))
        n += 1
    return n


def _tool_result_is_error_or_missing(text: str) -> bool:
    low = (text or "").lower()
    return any(
        needle in low
        for needle in (
            "no such file",
            "filenotfound",
            "file not found",
            "unexpected eof while looking for matching",
            "syntax error: unexpected end of file",
        )
    )


def _is_empty_tool_content(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _PROXY_EMPTY_HINT in t:
        return True
    if _tool_result_is_error_or_missing(t):
        return True
    low = t.lower()
    if low in _EMPTY_TOOL_EXACT:
        return True
    if not any(ch.isalnum() for ch in t):
        return True
    return False


def _recent_empty_tool_streak(messages: list[dict] | None) -> int:
    if not messages:
        return 0
    empty = 0
    saw_tool = False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            saw_tool = True
            if _is_empty_tool_content(_message_text(msg)):
                empty += 1
            else:
                break
            continue
        if role == "assistant":
            if saw_tool:
                break
            if not msg.get("tool_calls") and not _is_empty_tool_content(
                _message_text(msg)
            ):
                break
            continue
        if role == "user":
            break
        if saw_tool:
            break
    return empty


def _looks_like_prose_loop(text: str) -> bool:
    """True when assistant prose is looping (line or 8-gram overlap)."""
    t = (text or "").strip()
    if len(t) < 400:
        return False
    lines = [ln.strip() for ln in t.splitlines() if len(ln.strip()) >= 16]
    if len(lines) >= 4:
        top = Counter(lines).most_common(1)[0][1]
        if top >= 4 or top / len(lines) >= 0.35:
            return True
    words = t.split()
    if len(words) >= 24:
        grams = [" ".join(words[i : i + 8]) for i in range(len(words) - 7)]
        if grams:
            top = Counter(grams).most_common(1)[0][1]
            if top >= 5 or top / len(grams) >= 0.35:
                return True
    return False


def _assistant_prose_loop(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return False
        if role == "assistant":
            if msg.get("tool_calls"):
                return False
            return _looks_like_prose_loop(_message_text(msg))
        if role == "user":
            return False
    return False


def _assistant_faked_action(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return False
        if role == "assistant":
            if msg.get("tool_calls"):
                return False
            text = _message_text(msg)
            if len(text.strip()) < 24:
                return False
            return bool(_FAKE_ACTION_RE.search(text))
        if role == "user":
            return False
    return False


def _last_non_system_role(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role and role != "system":
            return str(role)
    return None


def _nudge_system(messages: list[dict], marker: str, nudge: str, log_tag: str) -> bool:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if marker in text:
            return True
        _set_message_text(msg, text + nudge)
        log.info("[agent] %s", log_tag)
        return True
    messages.insert(0, {"role": "system", "content": nudge.strip()})
    log.info("[agent] %s (new system)", log_tag)
    return True


def apply_loop_middleware(messages: list[dict]) -> dict[str, Any]:
    """Scoped AutoSaddler middleware: empty-tool, fake-action, prose-loop, JIT continue.

    Hooks fire only on matching conversation state so they do not regress
    unrelated turns (AutoSaddler: over-broad PreToolUse hooks overfit).
    """
    trace: dict[str, Any] = {
        "empty_tool_streak": 0,
        "empty_tool_recovery": False,
        "fake_action": False,
        "fake_action_recovery": False,
        "prose_loop": False,
        "prose_loop_recovery": False,
        "after_tool_continue": False,
        "truncated_tool_msgs": 0,
        "missing_path_recovery": False,
    }
    if not messages:
        return trace
    truncated = _truncate_tool_messages(messages)
    trace["truncated_tool_msgs"] = truncated
    streak = _recent_empty_tool_streak(messages)
    fake = _assistant_faked_action(messages)
    prose = _assistant_prose_loop(messages)
    last_role = _last_non_system_role(messages)
    trace["empty_tool_streak"] = streak
    trace["fake_action"] = fake
    trace["prose_loop"] = prose
    cfg = _active_harness()
    streak_min = int(cfg.get("empty_streak_min") or 1)
    if streak >= streak_min and cfg.get("mw_empty_tool", True):
        if _last_tool_was_missing_path(messages) and _nudge_system(
            messages,
            "[Harness] MISSING PATH:",
            _MISSING_PATH_NUDGE,
            f"missing-path recovery (streak={streak})",
        ):
            trace["empty_tool_recovery"] = True
            trace["missing_path_recovery"] = True
        elif not _last_tool_was_missing_path(messages) and _nudge_system(
            messages,
            "[Harness] EMPTY TOOL RESULT:",
            _empty_tool_nudge(),
            f"empty-tool recovery (streak={streak})",
        ):
            trace["empty_tool_recovery"] = True
    elif (
        last_role in ("tool", "function")
        and cfg.get("mw_after_tool", True)
        and _nudge_system(
            messages,
            "[Harness] Tool result received.",
            _AFTER_TOOL_NUDGE,
            "after-tool continue",
        )
    ):
        trace["after_tool_continue"] = True
    if fake and cfg.get("mw_fake_action", True) and _nudge_system(
        messages,
        "[Harness] FAKE ACTION:",
        _fake_action_nudge(),
        "fake-action recovery",
    ):
        trace["fake_action_recovery"] = True
    if prose and cfg.get("mw_prose_loop", True) and _nudge_system(
        messages,
        "[Harness] PROSE LOOP:",
        _prose_loop_nudge(),
        "prose-loop recovery",
    ):
        trace["prose_loop_recovery"] = True
    if (
        trace.get("empty_tool_recovery")
        or trace.get("fake_action_recovery")
        or trace.get("prose_loop_recovery")
    ):
        append_live_event(
            {
                "empty_tool_recovery": trace.get("empty_tool_recovery"),
                "fake_action_recovery": trace.get("fake_action_recovery"),
                "prose_loop_recovery": trace.get("prose_loop_recovery"),
                "after_tool_continue": trace.get("after_tool_continue"),
                "fake_action": fake,
                "prose_loop": prose,
                "empty_tool_streak": streak,
            }
        )
    return trace


def _apply_card_sampling(body: dict) -> None:
    body["temperature"] = CARD_TEMPERATURE
    body["top_p"] = CARD_TOP_P
    body["frequency_penalty"] = CARD_FREQUENCY_PENALTY
    body["enable_thinking"] = False
    body.pop("top_k", None)


def _ensure_max_tokens(body: dict, *, floor: int) -> None:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0 or value < floor:
        body["max_tokens"] = floor


def _wrap_loop_prefix(prefix: str) -> str:
    return f"{AS_LOOP_START}\n{prefix.strip()}\n{AS_LOOP_END}\n\n"


def _apply_loop_prefix_text(text: str, prefix: str) -> str:
    wrapped = _wrap_loop_prefix(prefix)
    if AS_LOOP_START in text:
        return re.sub(
            r"\[AS_LOOP\].*?\[/AS_LOOP\]\s*",
            wrapped,
            text,
            count=1,
            flags=re.S,
        )
    if prefix.strip() in text:
        return text
    default = LOOP_PREFIX.strip()
    if default in text and prefix.strip() != default:
        return text.replace(default, prefix.strip(), 1)
    if LOOP_MARKER in text and prefix.strip() == default:
        return text
    return wrapped + text


def _inject_loop_prompt(body: dict) -> None:
    if not body.get("tools"):
        return
    prefix = _loop_prefix()
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        body["messages"] = [{"role": "system", "content": prefix.strip()}]
        return
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        content = msgs[0].get("content")
        if isinstance(content, str):
            msgs[0]["content"] = _apply_loop_prefix_text(content, prefix)
            return
        if isinstance(content, list):
            text = ""
            idx = None
            for i, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text") or "")
                    if idx is None:
                        idx = i
            new = _apply_loop_prefix_text(text, prefix)
            if idx is not None:
                content[idx]["text"] = new
            else:
                content.insert(0, {"type": "text", "text": new})
            return
    msgs.insert(0, {"role": "system", "content": prefix.strip()})


def summarize_request(body: dict) -> str:
    msgs = body.get("messages") or []
    roles = [
        m.get("role") if isinstance(m, dict) else type(m).__name__ for m in msgs
    ]
    tools = body.get("tools") or body.get("functions") or []
    last = ""
    for msg in reversed(msgs):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            last = content.replace("\n", " ")[:120]
        elif isinstance(content, list):
            bits = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    bits.append(str(block.get("text") or ""))
            last = " ".join(bits).replace("\n", " ")[:120]
        break
    return (
        f"keys={sorted(body.keys())} nmsg={len(msgs)} roles={roles} "
        f"ntools={len(tools) if isinstance(tools, list) else 0} "
        f"tool_choice={body.get('tool_choice')!r} stream={bool(body.get('stream'))} "
        f"user={last!r}"
    )


def prepare_body(body: dict, trace: dict | None = None) -> dict:
    """Mutate a chat-completions body to the OBLITERATUS + agent contract.

    Optional ``trace`` is filled with middleware flags for tests / logs.
    """
    tr: dict[str, Any] = trace if trace is not None else {}
    tr.clear()
    tr.update(
        {
            "compaction": False,
            "empty_tool_streak": 0,
            "empty_tool_recovery": False,
            "fake_action": False,
            "fake_action_recovery": False,
            "prose_loop": False,
            "prose_loop_recovery": False,
            "after_tool_continue": False,
            "truncated_tool_msgs": 0,
            "loop_prompt": False,
        }
    )
    if _looks_like_compaction(body):
        body.pop("tools", None)
        body["tool_choice"] = "none"
        _apply_card_sampling(body)
        _ensure_max_tokens(body, floor=512)
        tr["compaction"] = True
        return body

    _apply_card_sampling(body)
    if body.get("tools"):
        _ensure_max_tokens(body, floor=AGENT_MAX_TOKENS_FLOOR)
        _inject_loop_prompt(body)
        tr["loop_prompt"] = True
        msgs = body.get("messages")
        if isinstance(msgs, list):
            tr.update(apply_loop_middleware(msgs))
            _repair_history_tool_calls(msgs)
    return body


def _path_basename(path: str) -> str:
    return (path or "").rstrip("/").rsplit("/", 1)[-1]


def _unmatched_quotes(text: str) -> bool:
    s = text or ""
    return (s.count('"') - s.count('\\"')) % 2 == 1 or (
        s.count("'") - s.count("\\'")
    ) % 2 == 1


def _parent_dir_of_path(path: str) -> str:
    p = (path or "").strip().strip("'\"")
    p = re.sub(r"[*?].*$", "", p).rstrip("/")
    if not p or p in {".", ".."}:
        return "."
    if "/" not in p:
        return "."
    return p.rsplit("/", 1)[0] or "."


def _is_pathless_head(command: str) -> bool:
    return bool(re.match(r"^(?:head|tail)(?:\s+-\d+)?\s*$", command or ""))


def _restore_abs_prefix(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    if p.startswith(("Users/", "home/", "tmp/", "var/", "opt/", "private/")):
        return "/" + p
    return p


def _dedupe_repeated_path_prefix(path: str) -> str:
    abs_path = path.startswith("/")
    parts = [p for p in path.split("/") if p]
    n = len(parts)
    if n < 4:
        return path
    for seq_len in range(2, n // 2 + 1):
        seq = parts[:seq_len]
        repeats = 0
        i = 0
        while i + seq_len <= n and parts[i : i + seq_len] == seq:
            repeats += 1
            i += seq_len
        if repeats >= 2:
            rebuilt = seq + parts[i:]
            out = "/".join(rebuilt)
            return "/" + out if abs_path or rebuilt[0] in {"Users", "home"} else out
    return path


def _prefer_workspace_relative(path: str, workspace: str | None) -> str:
    if not path or not workspace:
        return path
    p = path.replace("\\", "/")
    ws = workspace.replace("\\", "/").rstrip("/")
    if p == ws:
        return "."
    if p.startswith(ws + "/"):
        return p[len(ws) + 1 :]
    bare = ws.lstrip("/")
    if p == bare:
        return "."
    if p.startswith(bare + "/"):
        return p[len(bare) + 1 :]
    return path


def _extract_workspace_dir(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    patterns = (
        r"Current Workspace Directory\s*\(([^)]+)\)",
        r"Current Working Directory[:\s]+([^\s\n]+)",
        r"Workspace Directory[:\s]+([^\s\n]+)",
        r"Working directory:\s*([^\s\n]+)",
        r"<cwd>\s*([^<\s]+)\s*</cwd>",
    )
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        text = _message_text(msg)
        if not text:
            continue
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                wd = m.group(1).strip().strip("()").rstrip("/")
                if wd:
                    return wd
    return None


def _last_tool_was_missing_path(messages: list[dict] | None) -> bool:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return _tool_result_is_error_or_missing(_message_text(msg))
        if role == "user":
            return False
    return False


def _workspace_entry_exists(
    path: str, messages: list[dict] | None
) -> bool | None:
    ws = _extract_workspace_dir(messages)
    if not ws:
        return None
    raw = (path or "").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = _normalize_fs_path(raw, messages)
    if not raw or raw in {".", ".."}:
        return True
    cand = Path(raw) if Path(raw).is_absolute() else Path(ws) / raw
    try:
        return cand.exists()
    except OSError:
        return None


def _ls_target_dir(command: str) -> str | None:
    m = re.match(
        r"^(?:ls|tree)\s+(?:-[a-zA-Z0-9]+\s+)*['\"]?([^'\"\n]+)",
        (command or "").strip(),
        re.I,
    )
    if not m:
        return None
    target = re.split(r"[\s;|&<>]", m.group(1).strip(), maxsplit=1)[0]
    target = target.strip().strip("'\"").rstrip("/")
    if not target or target.startswith("-"):
        return None
    return target


def _is_invented_workspace_path(
    path: str, messages: list[dict] | None
) -> bool:
    exists = _workspace_entry_exists(path, messages)
    if exists is not False:
        return False
    raw = (path or "").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    first = raw.split("/")[0].lower()
    common = {
        "analysis", "tools", "src", "lib", "docs", "firmware", "sources",
        "third_party", "tests", "test", "scripts", ".pi", ".grok", "include",
        "app", "pkg", "cmd", "internal",
    }
    if first in common and "/" not in raw:
        return False
    return True


def _normalize_fs_path(path: str, messages: list[dict] | None = None) -> str:
    raw = (path or "").strip().strip("'\"")
    if not raw or raw in {".", ".."}:
        return raw
    raw = _restore_abs_prefix(raw)
    workspace = _extract_workspace_dir(messages)
    if workspace:
        raw = _dedupe_repeated_path_prefix(raw)
        ws = workspace.replace("\\", "/").rstrip("/")
        token = ws.lstrip("/")
        if token and token in raw.replace("\\", "/"):
            collapsed = raw.replace("\\", "/")
            while f"{token}/{token}" in collapsed:
                collapsed = collapsed.replace(f"{token}/{token}", token, 1)
            while f"{ws}/{token}" in collapsed:
                collapsed = collapsed.replace(f"{ws}/{token}", ws, 1)
            raw = collapsed
        raw = _prefer_workspace_relative(raw, workspace)
    else:
        raw = _dedupe_repeated_path_prefix(raw)
    return raw


def _recent_tool_result_paths(messages: list[dict] | None, limit_msgs: int = 8) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    count = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        count += 1
        if count > limit_msgs:
            break
        blob = _message_text(msg)
        if _tool_result_is_error_or_missing(blob):
            continue
        for match in _FILENAME_RE.finditer(blob):
            path = _normalize_fs_path(match.group(1), messages)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _analysis_dir_markdown(
    path: str, messages: list[dict] | None
) -> str | None:
    """analysis/semgrep-domain-c/README.md → analysis/SEMGREP-DOMAIN-C.md."""
    raw = _normalize_fs_path((path or "").strip().strip("'\""), messages)
    parts = [p for p in raw.replace("\\", "/").split("/") if p and p != "."]
    if len(parts) < 2 or parts[0].lower() != "analysis":
        return None
    slug = parts[1]
    if "." in slug and slug.lower().rsplit(".", 1)[-1] in {"md", "txt", "json"}:
        return None
    want = re.sub(r"[^a-z0-9]+", "", slug.lower())
    if not want:
        return None
    ws = _extract_workspace_dir(messages)
    if ws:
        analysis_dir = Path(ws) / "analysis"
        try:
            names = list(analysis_dir.iterdir())
        except OSError:
            names = []
        for entry in names:
            stem = entry.stem if entry.suffix else entry.name
            if re.sub(r"[^a-z0-9]+", "", stem.lower()) != want:
                continue
            if entry.is_file():
                return f"analysis/{entry.name}"
    guess = f"analysis/{slug.upper()}.md"
    if guess != raw:
        return guess
    return None


def _resolve_read_path(path: str, messages: list[dict] | None) -> str | None:
    raw = _normalize_fs_path((path or "").strip().strip("'\""), messages)
    if not raw:
        return None
    if _workspace_entry_exists(raw, messages) is True:
        return raw
    sibling = _analysis_dir_markdown(raw, messages)
    if sibling and sibling != raw and _workspace_entry_exists(raw, messages) is not True:
        if _workspace_entry_exists(sibling, messages) is not False:
            log.info("[read-resolve] %s → %s", raw, sibling)
            return sibling
    known = _recent_tool_result_paths(messages, limit_msgs=24)
    extras = ["README.md", "AGENTS.md", "COMPLETE.md"]
    pool = list(dict.fromkeys(known + extras))
    pool.sort(key=lambda p: (len(p), p.count("/"), p))
    if raw in pool:
        return raw
    base = _path_basename(raw).lower()
    for item in pool:
        if _path_basename(item).lower() == base:
            return item
    names = [_path_basename(item) for item in pool]
    close = difflib.get_close_matches(_path_basename(raw), names, n=1, cutoff=0.72)
    if close:
        want = close[0].lower()
        for item in pool:
            if _path_basename(item).lower() == want:
                return item
    return raw if raw in extras else None


def _history_paths_by_basename(messages: list[dict] | None) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in reversed(_recent_tool_result_paths(messages, limit_msgs=16)):
        name = _path_basename(path)
        if name and name not in found:
            found[name] = path
    return found


def _rewrite_missing_path_command(command: str, messages: list[dict] | None) -> str | None:
    known = _history_paths_by_basename(messages)
    if not known or not command:
        return None
    rewritten = command
    changed = False
    for match in _CAT_PATH_RE.finditer(command):
        raw = match.group(1)
        name = _path_basename(raw)
        alt = known.get(name)
        if alt and alt != raw:
            rewritten = rewritten.replace(raw, alt)
            changed = True
    return rewritten if changed else None


def _paths_in_bash_command(command: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for rx in (_QUOTED_FILE_RE, _CAT_PATH_RE, _FILENAME_RE):
        for match in rx.finditer(command or ""):
            path = match.group(1)
            if path and path not in seen:
                seen.add(path)
                found.append(path)
    return found


def _safe_head_command(paths: list[str], messages: list[dict] | None = None) -> str:
    cleaned: list[str] = []
    parents: list[str] = []
    for p in paths:
        if not p or re.search(r"[\n|;|&`$<>]", p):
            continue
        n = _normalize_fs_path(p, messages)
        resolved = _resolve_read_path(n, messages) if messages is not None else n
        if resolved:
            if resolved not in cleaned:
                cleaned.append(resolved)
            continue
        known = _recent_tool_result_paths(messages) if messages else []
        if known:
            parent = _parent_dir_of_path(n)
            if parent not in parents:
                parents.append(parent)
        elif n not in cleaned:
            cleaned.append(n)
    if cleaned:
        return "head -80 " + " ".join(cleaned[:_NEXT_STEP_MAX_READS])
    if parents:
        return "ls " + " ".join(parents[:4])
    return "ls"


def _sanitize_unmatched_quotes(
    cmd: str, messages: list[dict] | None
) -> str:
    paths = _paths_in_bash_command(cmd)
    for piece in re.split(r"[\n;|&]+", cmd):
        s = piece.strip()
        if not s or _unmatched_quotes(s):
            continue
        if _is_pathless_head(s):
            continue
        if s.startswith(_SAFE_BASH_STARTS):
            return s
    if paths:
        return _safe_head_command(paths, messages)
    for match in re.finditer(
        r"""['"]((?:[\w./~-]+/)+)\*[^'"]*['"]|(?:^|[\s\"'])((?:[\w./~-]+/)+)""",
        cmd,
    ):
        directory = (match.group(1) or match.group(2) or "").rstrip("/")
        if directory and directory not in {".", ".."}:
            return "ls " + directory
    return "ls"


def _sanitize_bash_command(command: str, messages: list[dict] | None = None) -> str:
    """Fix *broken* bash. Keep real work (cd && script, which, pipes, env=)."""
    cmd = (command or "").strip()
    if not cmd:
        return "ls"
    rewritten = _rewrite_missing_path_command(cmd, messages)
    if rewritten:
        cmd = rewritten.strip()
    ls_dir = _ls_target_dir(cmd)
    if ls_dir and _is_invented_workspace_path(ls_dir, messages):
        return "ls"
    if cmd in {"ls", "pwd", "date", "uname", "whoami"}:
        return cmd
    if _is_pathless_head(cmd):
        return "ls"
    if _unmatched_quotes(cmd):
        return _sanitize_unmatched_quotes(cmd, messages)
    if "\n" in cmd:
        first = next((ln.strip() for ln in cmd.splitlines() if ln.strip()), "")
        if first and first != cmd:
            return _sanitize_bash_command(first, messages)
    if _CMD_SUB_RE.search(cmd):
        paths = _paths_in_bash_command(cmd)
        if paths:
            return _safe_head_command(paths, messages)
        return "ls"
    paths = _paths_in_bash_command(cmd)
    if paths and messages and re.match(r"^(?:cat|head|tail|less|grep|rg)\b", cmd):
        known = _recent_tool_result_paths(messages)
        if known and not any(
            _resolve_read_path(_normalize_fs_path(p, messages), messages) for p in paths
        ):
            return _safe_head_command(paths, messages)
    return cmd


def _is_early_stop_plan(text: str, has_tools: bool) -> bool:
    if has_tools:
        return False
    blob = (text or "").strip()
    if len(blob) < 24:
        return False
    return bool(_EARLY_STOP_PLAN_RE.search(blob))


def _sse_text_and_tools(raw: bytes) -> tuple[str, bool]:
    parts: list[str] = []
    saw_tool = False
    for event in _iter_sse_json(raw):
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("tool_calls"):
                saw_tool = True
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return "".join(parts), saw_tool


def _json_text_and_tools(payload: bytes) -> tuple[str, bool]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", False
    if not isinstance(data, dict):
        return "", False
    parts: list[str] = []
    saw_tool = False
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            continue
        if msg.get("tool_calls"):
            saw_tool = True
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return "".join(parts), saw_tool


def _last_user_wants_action(messages: list[dict] | None) -> bool:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        return bool(_USER_ACTION_RE.search(_message_text(msg)[:500]))
    return False


def _should_force_continue(
    text: str, has_tools: bool, messages: list[dict] | None
) -> bool:
    if has_tools:
        return False
    if _is_early_stop_plan(text, False):
        return True
    if _last_user_wants_action(messages) and len((text or "").strip()) >= 24:
        return True
    return not (text or "").strip()


def _command_from_plan(text: str) -> str | None:
    blob = text or ""
    m = re.search(r"ONEPHONE_I_CONFIRM=1\s+\./tools/[\w.-]+\.sh", blob)
    if m:
        return m.group(0)
    m = _PLAN_SCRIPT_RE.search(blob)
    if m:
        cmd = m.group(0).strip()
        if not cmd.startswith("./") and cmd.startswith("tools/"):
            cmd = "./" + cmd
        return cmd
    return None


def _make_tool_call(name: str, args: dict) -> dict:
    return {
        "id": f"call_saddle_{name}",
        "type": "function",
        "index": 0,
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _synthetic_tool_calls(
    text: str, messages: list[dict] | None
) -> list[dict]:
    cmd = _command_from_plan(text)
    if cmd:
        cmd = _sanitize_bash_command(cmd, messages)
        return [_make_tool_call("bash", {"command": cmd})]
    for path in ("AGENTS.md", "analysis/CONTINUE.md", "README.md", "COMPLETE.md"):
        if _workspace_entry_exists(path, messages) is True:
            return [_make_tool_call("bash", {"command": f"head -80 {path}"})]
    return [_make_tool_call("bash", {"command": "ls"})]


def _payload_ids(raw: bytes, *, sse: bool) -> tuple[str, str]:
    model = "qwen3.8-27b-obliterated-mtplx"
    response_id = "chatcmpl-saddle"
    if sse:
        for event in _iter_sse_json(raw):
            if not isinstance(event, dict):
                continue
            model = str(event.get("model") or model)
            response_id = str(event.get("id") or response_id)
        return model, response_id
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return model, response_id
    if isinstance(data, dict):
        model = str(data.get("model") or model)
        response_id = str(data.get("id") or response_id)
    return model, response_id


def _sse_tool_payload(tcs: list[dict], model: str, response_id: str) -> bytes:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "tool_calls": tcs},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _json_tool_payload(tcs: list[dict], model: str, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tcs,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _apply_early_stop_retry(body: dict) -> None:
    """mtplx accepts tool_choice auto|none only — nudge via a user turn."""
    body["tool_choice"] = "auto"
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    _nudge_system(
        msgs,
        "[Harness] EARLY STOP:",
        _EARLY_STOP_NUDGE,
        "early-stop plan retry",
    )
    if any(
        isinstance(m, dict)
        and m.get("role") == "user"
        and "[Harness] EARLY STOP:" in _message_text(m)
        for m in msgs
    ):
        return
    msgs.append({"role": "user", "content": _RETRY_USER})


def _path_from_args(args: dict) -> str:
    for key in ("path", "filePath", "file_path", "target_file"):
        val = args.get(key)
        if val:
            return str(val)
    return ""


def _repair_one_tool_call(
    name: str,
    args: dict,
    messages: list[dict] | None,
) -> dict:
    lname = (name or "").lower().replace("-", "_")
    if lname in {"bash", "shell", "execute_command"}:
        raw_cmd = str(args.get("command") or "")
        if raw_cmd:
            safe = _sanitize_bash_command(raw_cmd, messages)
            ls_dir = _ls_target_dir(safe) or _ls_target_dir(raw_cmd)
            if ls_dir and _is_invented_workspace_path(ls_dir, messages):
                safe = "ls"
            if safe != raw_cmd:
                log.info("[bash-sanitize] %r → %r", raw_cmd[:80], safe[:80])
                return {**args, "command": safe}
        return args
    if lname in {"read", "readfile", "read_file", "fs_read"}:
        path = _path_from_args(args)
        resolved = _resolve_read_path(path, messages) or _normalize_fs_path(path, messages)
        if _workspace_entry_exists(resolved or path, messages) is False:
            sibling = _analysis_dir_markdown(path, messages)
            resolved = sibling or "README.md"
        if resolved and resolved != path:
            log.info("[read-resolve] %r → %s", path, resolved)
            field = "path" if "path" in args else next(
                (k for k in ("filePath", "file_path", "target_file") if k in args),
                "path",
            )
            return {**args, field: resolved}
    return args


def _repair_tool_calls_list(tcs: list, messages: list[dict] | None) -> bool:
    changed = False
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        parsed: dict | None = None
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
                if isinstance(loaded, dict):
                    parsed = loaded
            except json.JSONDecodeError:
                fixed = _try_close_json(raw_args)
                if fixed:
                    try:
                        loaded = json.loads(fixed)
                        if isinstance(loaded, dict):
                            parsed = loaded
                            fn["arguments"] = fixed
                            changed = True
                    except json.JSONDecodeError:
                        parsed = None
        if not parsed:
            continue
        repaired = _repair_one_tool_call(name, parsed, messages)
        if repaired != parsed:
            fn["arguments"] = json.dumps(repaired, ensure_ascii=False)
            changed = True
    return changed


def _repair_history_tool_calls(messages: list[dict]) -> None:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                _repair_tool_calls_list(tcs, messages)


def _try_close_json(s: str) -> str | None:
    s = s.strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    if s.count('"') % 2 == 1:
        s = s + '"'
    s = re.sub(r",\s*$", "", s)
    closes_needed_obj = 0
    closes_needed_arr = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            closes_needed_obj += 1
        elif ch == "}":
            closes_needed_obj = max(0, closes_needed_obj - 1)
        elif ch == "[":
            closes_needed_arr += 1
        elif ch == "]":
            closes_needed_arr = max(0, closes_needed_arr - 1)
    candidate = s + ("]" * closes_needed_arr) + ("}" * closes_needed_obj)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def _repair_tool_calls_in_message(
    msg: dict, messages: list[dict] | None = None
) -> bool:
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list):
        return False
    return _repair_tool_calls_list(tcs, messages)


def _repair_response_payload(
    payload: bytes, messages: list[dict] | None = None
) -> bytes:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload
    if not isinstance(data, dict):
        return payload
    changed = False
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if isinstance(msg, dict) and _repair_tool_calls_in_message(msg, messages):
            changed = True
    if not changed:
        return payload
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def _iter_sse_json(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _repair_sse_payload(raw: bytes, messages: list[dict] | None) -> bytes:
    assembled: dict[int, dict] = {}
    model = ""
    response_id = "chatcmpl-saddle"
    saw_tool = False
    for event in _iter_sse_json(raw):
        if not isinstance(event, dict):
            continue
        model = str(event.get("model") or model)
        response_id = str(event.get("id") or response_id)
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                saw_tool = True
                idx = int(tc.get("index") or 0)
                slot = assembled.setdefault(
                    idx,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += str(fn["arguments"])
    if not saw_tool:
        return raw
    tcs = [assembled[i] for i in sorted(assembled)]
    if not _repair_tool_calls_list(tcs, messages):
        return raw
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "tool_calls": tcs},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


class ProxyState:
    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        parsed = urlparse(self.upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise SystemExit(f"Invalid upstream URL: {upstream}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)


def _connect(state: ProxyState) -> HTTPConnection:
    if state.scheme == "https":
        return HTTPSConnection(state.host, state.port, timeout=900)
    return HTTPConnection(state.host, state.port, timeout=900)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: ProxyState

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _send_json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/healthz", "/health"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "upstream": self.state.upstream,
                    "sampling": {
                        "temperature": CARD_TEMPERATURE,
                        "top_p": CARD_TOP_P,
                        "frequency_penalty": CARD_FREQUENCY_PENALTY,
                        "enable_thinking": False,
                    },
                },
            )
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, x-api-key",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _upstream_once(
        self, path: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], str, bytes]:
        conn = _connect(self.state)
        try:
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_headers = {k: v for k, v in resp.getheaders()}
            content_type = resp_headers.get("Content-Type", "")
            chunks: list[bytes] = []
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            return resp.status, resp_headers, content_type, b"".join(chunks)
        finally:
            conn.close()

    def _proxy(self) -> None:
        body = self._read_body()
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower()
            not in (
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "accept-encoding",
            )
        }

        path = self.path
        stream = False
        steer_messages: list[dict] | None = None
        parsed: dict | None = None
        steer_trace: dict[str, Any] = {}
        if body and "application/json" in (self.headers.get("Content-Type") or ""):
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                parsed = data
                stream = bool(data.get("stream"))
                if path.rstrip("/").endswith("/chat/completions") or path.rstrip(
                    "/"
                ).endswith("/messages"):
                    log.info("[in] %s", summarize_request(data))
                    prepare_body(data, steer_trace)
                    msgs = data.get("messages")
                    if isinstance(msgs, list):
                        steer_messages = msgs
                    log.info(
                        "[steer] temp=%s think=%s fp=%s ntools=%s max_tokens=%s "
                        "compaction=%s empty_rec=%s fake=%s prose=%s "
                        "after_tool=%s trunc=%s",
                        data.get("temperature"),
                        data.get("enable_thinking"),
                        data.get("frequency_penalty"),
                        len(data.get("tools") or []),
                        data.get("max_tokens"),
                        steer_trace.get("compaction"),
                        steer_trace.get("empty_tool_recovery"),
                        steer_trace.get("fake_action_recovery"),
                        steer_trace.get("prose_loop_recovery"),
                        steer_trace.get("after_tool_continue"),
                        steer_trace.get("truncated_tool_msgs"),
                    )
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"

        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{self.state.host}:{self.state.port}"
        headers["Connection"] = "close"

        try:
            status, resp_headers, content_type, payload = self._upstream_once(
                path, body, headers
            )
            is_sse = stream or "text/event-stream" in content_type
            if is_sse:
                text, has_tools = _sse_text_and_tools(payload)
                if (
                    parsed is not None
                    and steer_trace.get("after_tool_continue")
                    and _active_harness().get("mw_early_stop", True)
                    and _should_force_continue(text, has_tools, steer_messages)
                ):
                    log.info("[agent] early-stop plan retry")
                    _apply_early_stop_retry(parsed)
                    body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                    headers["Content-Length"] = str(len(body))
                    status, resp_headers, content_type, payload = self._upstream_once(
                        path, body, headers
                    )
                    is_sse = stream or "text/event-stream" in content_type
                    text, has_tools = _sse_text_and_tools(payload)
                    if _should_force_continue(text, has_tools, steer_messages):
                        tcs = _synthetic_tool_calls(text, steer_messages)
                        model, rid = _payload_ids(payload, sse=is_sse)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        payload = _sse_tool_payload(tcs, model, rid)
                    else:
                        payload = _repair_sse_payload(payload, steer_messages)
                else:
                    payload = _repair_sse_payload(payload, steer_messages)
            elif "application/json" in content_type:
                payload = _repair_response_payload(payload, steer_messages)
                text, has_tools = _json_text_and_tools(payload)
                if (
                    parsed is not None
                    and steer_trace.get("after_tool_continue")
                    and _active_harness().get("mw_early_stop", True)
                    and _should_force_continue(text, has_tools, steer_messages)
                ):
                    log.info("[agent] early-stop plan retry")
                    _apply_early_stop_retry(parsed)
                    body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                    headers["Content-Length"] = str(len(body))
                    status, resp_headers, content_type, payload = self._upstream_once(
                        path, body, headers
                    )
                    if "application/json" in content_type:
                        payload = _repair_response_payload(payload, steer_messages)
                    text, has_tools = _json_text_and_tools(payload)
                    if _should_force_continue(text, has_tools, steer_messages):
                        tcs = _synthetic_tool_calls(text, steer_messages)
                        model, rid = _payload_ids(payload, sse=False)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        payload = _json_tool_payload(tcs, model, rid)

            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() in (
                    "transfer-encoding",
                    "connection",
                    "content-length",
                    "content-encoding",
                ):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            log.exception("upstream error: %s", exc)
            try:
                self._send_json(
                    502, {"error": {"message": str(exc), "type": "proxy_error"}}
                )
            except Exception:
                pass


def _self_check() -> None:
    kilo_like = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 128,
        "enable_thinking": True,
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "fix it"},
        ],
    }
    tr: dict[str, Any] = {}
    prepare_body(kilo_like, tr)
    assert kilo_like["temperature"] == 0.0, kilo_like
    assert kilo_like["top_p"] == 1.0, kilo_like
    assert kilo_like["frequency_penalty"] == CARD_FREQUENCY_PENALTY, kilo_like
    assert kilo_like["enable_thinking"] is False, kilo_like
    assert kilo_like["max_tokens"] == AGENT_MAX_TOKENS_FLOOR, kilo_like
    assert LOOP_MARKER in kilo_like["messages"][0]["content"]
    assert tr.get("loop_prompt") is True
    # First user turn must NOT get empty-tool / after-tool hooks (over-broad).
    sys0 = kilo_like["messages"][0]["content"]
    assert "[Harness] EMPTY TOOL RESULT:" not in sys0
    assert "[Harness] Tool result received." not in sys0
    assert tr.get("empty_tool_recovery") is False
    assert tr.get("after_tool_continue") is False

    compact = {
        "tool_choice": "none",
        "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "compact the conversation"}],
        "temperature": 1.0,
    }
    prepare_body(compact)
    assert "tools" not in compact
    assert compact["tool_choice"] == "none"

    kilo_agent_with_summary_boilerplate = {
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "list_files"}}],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent. You may summarize the conversation "
                    "when asked. Prefer tools."
                ),
            },
            {"role": "user", "content": "review work and next steps"},
        ],
        "max_tokens": 32000,
    }
    tr_kilo: dict[str, Any] = {}
    prepare_body(kilo_agent_with_summary_boilerplate, tr_kilo)
    assert tr_kilo.get("compaction") is False, tr_kilo
    assert kilo_agent_with_summary_boilerplate.get("tools")
    assert kilo_agent_with_summary_boilerplate["tool_choice"] == "auto"

    empty_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "(no output)"},
        ],
        "max_tokens": 128,
    }
    tr_empty: dict[str, Any] = {}
    prepare_body(empty_body, tr_empty)
    assert tr_empty.get("empty_tool_recovery") is True, tr_empty
    assert "[Harness] EMPTY TOOL RESULT:" in empty_body["messages"][0]["content"]
    # Empty recovery supersedes generic after-tool continue.
    assert tr_empty.get("after_tool_continue") is False, tr_empty

    useful_tool = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "echo hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t2",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo hi"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t2", "content": "hi"},
        ],
        "max_tokens": 128,
    }
    tr_useful: dict[str, Any] = {}
    prepare_body(useful_tool, tr_useful)
    assert tr_useful.get("empty_tool_recovery") is False, tr_useful
    assert tr_useful.get("after_tool_continue") is True, tr_useful
    assert "[Harness] Tool result received." in useful_tool["messages"][0]["content"]
    assert "[Harness] EMPTY TOOL RESULT:" not in useful_tool["messages"][0]["content"]

    fake_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "run tests"},
            {
                "role": "assistant",
                "content": "I will check the test suite and then run pytest.",
            },
        ],
        "max_tokens": 128,
    }
    tr_fake: dict[str, Any] = {}
    prepare_body(fake_body, tr_fake)
    assert tr_fake.get("fake_action") is True, tr_fake
    assert tr_fake.get("fake_action_recovery") is True, tr_fake
    assert "[Harness] FAKE ACTION:" in fake_body["messages"][0]["content"]

    loop_line = "import os\nimport sys\nimport os\nimport sys\n"
    loop_text = loop_line * 20
    prose_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "write parse_ini"},
            {"role": "assistant", "content": loop_text},
        ],
        "max_tokens": 128,
    }
    tr_prose: dict[str, Any] = {}
    prepare_body(prose_body, tr_prose)
    assert tr_prose.get("prose_loop") is True, tr_prose
    assert tr_prose.get("prose_loop_recovery") is True, tr_prose

    huge = "x" * (TOOL_RESULT_MAX_CHARS + 5000)
    trunc_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "dump"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t3",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t3", "content": huge},
        ],
        "max_tokens": 128,
    }
    tr_trunc: dict[str, Any] = {}
    prepare_body(trunc_body, tr_trunc)
    assert tr_trunc.get("truncated_tool_msgs") == 1, tr_trunc
    assert len(trunc_body["messages"][-1]["content"]) < len(huge)

    broken = 'grep -rn "autohunt'
    fixed = _sanitize_bash_command(broken)
    assert not _unmatched_quotes(fixed)
    assert fixed.startswith(("ls", "ls ", "head "))
    ok_grep = 'grep -rn "autohunt\\|Autohunt" "analysis/semgrep-domain-c/*.md"'
    kept_grep = _sanitize_bash_command(ok_grep)
    assert kept_grep.startswith("grep -rn"), kept_grep
    run = (
        "cd /Users/aicoder/src/private/grapheneos-titan-m2 && "
        "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh"
    )
    kept_run = _sanitize_bash_command(run)
    assert "./tools/tx_tail_one_phone.sh" in kept_run, kept_run
    assert not kept_run.startswith("head "), kept_run
    which = 'which adb 2>/dev/null || echo "no adb"'
    kept_which = _sanitize_bash_command(which)
    assert "which adb" in kept_which, kept_which
    plan = "Next steps\nE1 ecall 0x49 — one next RE probe on host wrapper.\n"
    assert _is_early_stop_plan(plan, False)
    assert _is_early_stop_plan(plan, True) is False
    assert _is_early_stop_plan("verified README.md and tests passed", False) is False
    plan_cmd = (
        "Next steps\nS2/S3 TX-TAIL — ONEPHONE_I_CONFIRM=1 "
        "./tools/tx_tail_one_phone.sh (multi-chunk)\n"
    )
    assert _command_from_plan(plan_cmd) == (
        "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh"
    )
    syn = _synthetic_tool_calls(plan_cmd, None)
    assert syn[0]["function"]["name"] == "bash"
    assert "tx_tail_one_phone.sh" in syn[0]["function"]["arguments"]
    retry_body = {
        "tool_choice": "auto",
        "messages": [{"role": "system", "content": "agent"}],
    }
    _apply_early_stop_retry(retry_body)
    assert retry_body["tool_choice"] == "auto"
    assert retry_body["messages"][-1]["role"] == "user"
    assert "EARLY STOP" in retry_body["messages"][-1]["content"]
    assert _should_force_continue(plan, False, None) is True
    assert _should_force_continue(plan, True, None) is False
    miss = (
        "head: analysis/semgrep-domain-c/WALLET-DOMAIN-C-FINAL.md: "
        "No such file or directory"
    )
    assert _is_empty_tool_content(miss)
    ws = "/Users/aicoder/src/private/grapheneos-titan-m2"
    rel = _normalize_fs_path(
        "Users/aicoder/src/private/grapheneos-titan-m2/README.md",
        [{"role": "user", "content": f"Current Workspace Directory ({ws})"}],
    )
    assert rel == "README.md"
    junk = "/".join(["Users/aicoder/src/private/grapheneos-titan-m2"] * 4) + "/README.md"
    assert _normalize_fs_path(
        junk,
        [{"role": "user", "content": f"Current Workspace Directory ({ws})"}],
    ) == "README.md"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Qwen3.8 OBLITERATED ↔ Kilo agent harness proxy"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument(
        "--upstream",
        default="http://127.0.0.1:8767",
        help="raw mtplx base (no trailing /v1)",
    )
    p.add_argument("--self-check", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _self_check()
    if args.self_check:
        print("qwen38_obl_kilo_proxy self-check ok")
        return 0

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log.info("debug log file %s", LOG_FILE)

    state = ProxyState(args.upstream)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(
        "OBLITERATED kilo proxy on http://%s:%d → %s (greedy, thinking off)",
        args.host,
        args.port,
        state.upstream,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    _ = threading
    sys.exit(main())
