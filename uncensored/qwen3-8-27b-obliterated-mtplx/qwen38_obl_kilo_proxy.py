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
     Next-steps/plan with no tool_calls, retry once (mtplx: tool_choice
     auto|none only). Synthesize a plan script at most once; never repeat
     a command already in history. After-tool JIT nudges cap at 3 tool
     rounds since the last user message, then the nudge is stripped so
     the model can recap and stop.
  9. Serialize chat generations and drop client session-affinity headers.
     mtplx raises 409 "session … already in flight" when Kilo retries the
     same sticky session while a stream is still running; retry that 409.


Usage:
  python3 qwen38_obl_kilo_proxy.py --upstream http://127.0.0.1:8767 --port 8768
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import queue
import re
import select
import socket
import sys
import threading
import time
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


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


# Stability: a greedy 27B that starts looping at max_tokens=32000 holds the
# engine for 20+ minutes. Kilo's turn loop re-prompts anyway, so cap one
# completion here (the card only asks for >= 2048).
AGENT_MAX_TOKENS_CAP = _env_int("QWEN38_OBL_MAX_TOKENS", 8192)
# No-tool chat (title, plain Q&A) and compaction decode at ~5 tok/s once the
# prompt is large: 8192 tokens is 25+ minutes. Measured compaction summaries
# are ~600 tokens; 1536 leaves headroom and bounds the worst case to ~5 min.
CHAT_MAX_TOKENS_CAP = _env_int("QWEN38_OBL_CHAT_MAX_TOKENS", 2048)
COMPACTION_MAX_TOKENS_CAP = _env_int("QWEN38_OBL_COMPACTION_MAX_TOKENS", 1536)
# Prompt-size guard. mtplx decode collapses from ~25 tok/s at 30k prompt
# tokens to 2-5 tok/s past ~50k (measured 2026-09-02). Kilo's own compaction
# is the primary control (see limit.context in kilo.jsonc); this is the
# backstop: oldest non-system turns are dropped in tool-call-consistent groups
# until the history fits. ~4 chars/token.
CONTEXT_CHAR_BUDGET = _env_int("QWEN38_OBL_CONTEXT_CHARS", 160_000)
_TRIM_NOTE = "[proxy: {n} older messages dropped to fit the local context window]"
# Cross-turn repetition guard. A greedy model that re-issues the SAME tool
# call (or the same prose) turn after turn never converges: measured 17+
# identical 38-token completions while the prompt grew 2.8k tokens a step.
# In-message loop detection (_looks_like_prose_loop) cannot see this because
# each completion is short. Consecutive identical assistant turns:
#   >= REPEAT_NUDGE_MIN  -> system nudge
#   >= REPEAT_NO_TOOLS   -> nudge + tools removed (forces a final text answer,
#                           which ends Kilo's turn loop)
#   >= REPEAT_HARD_STOP  -> synthetic final answer, upstream not called
REPEAT_NUDGE_MIN = 2
REPEAT_NO_TOOLS = 3
REPEAT_HARD_STOP = 4
_REPEAT_NUDGE = (
    "\n\n[Harness] REPEATED ACTION: your last {n} assistant turns were "
    "identical. That tool result is already in the conversation above. Do NOT "
    "issue the same tool call or text again. Take a DIFFERENT next step, or "
    "answer the user now with what you already know."
)
_REPEAT_STOP_TEXT = (
    "[Harness] Stopped: the model repeated the same action {n} times without "
    "progress ({what}). The last tool result is above. Rephrase the request "
    "or name the exact file/command to continue."
)
LOOP_MARKER = "Do not stop after 1 of N."
# Keep this short (finite attention). Extra recovery lives in scoped middleware
# below — AutoSaddler: capability/loop logic first, then steering; prefer
# replacement over stacking more rules.
LOOP_PREFIX = (
    "Finish every requested step. Prefer tools over prose. After each tool "
    "result, immediately take the next action. Do not stop after 1 of N. "
    "Do not recap; act. Empty or error tool output: retry with a simpler "
    "local command (ls/glob/grep/read). Verify with a tool before declaring "
    "the job done. Tool calls must be real tool_calls, never prose like "
    "'[Calling tool: ...]'. Never repeat an identical tool call; its result "
    "is already above. For 'what does X do' questions read only the file "
    "header (read with limit=150), not the whole file.\n\n"
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
_PLAN_SCRIPT_RE = re.compile(
    r"(?:ONEPHONE_I_CONFIRM=1\s+)?(?:\./)?tools/[\w.-]+\.sh"
)
# Cap runaway tool loops: JIT continue only for the first N tools after
# the last real user message; force a Next-steps retry at most once.
AFTER_TOOL_NUDGE_MAX = 3
EARLY_STOP_FORCE_MAX = 1
_UPSTREAM_GEN_LOCK = threading.RLock()
# Mirrors the lock for /healthz ("engine_busy") without touching the lock.
_UPSTREAM_GEN_LOCK_BUSY = threading.Event()
_IN_FLIGHT_RE = re.compile(r"already in flight", re.I)
_SESSION_HEADER_DROP = frozenset(
    {
        "x-mtplx-session-id",
        "x-session-affinity",
        "x-session-id",
        "x-openwebui-chat-id",
        "x-openwebui-user-id",
    }
)
_SESSION_BODY_DROP = (
    "user",
    "session_id",
    "mtplx_session_id",
    "chat_id",
    "conversation_id",
)
_IN_FLIGHT_RETRIES = 4
# Per-read socket timeout toward mtplx. Only trips if the engine is truly
# dead: the client side is kept alive separately with SSE keepalive comments.
UPSTREAM_TIMEOUT = _env_int("QWEN38_OBL_UPSTREAM_TIMEOUT", 300)
# One generation at a time on the engine. A waiter that cannot get the lock
# within this budget gets a clean 503 (Kilo shows an error and can retry)
# instead of a silent multi-minute "Thinking" spinner.
LOCK_WAIT_TIMEOUT = _env_int("QWEN38_OBL_LOCK_WAIT", 900)
# SSE comment lines are ignored by every OpenAI-compatible parser; sending one
# every few seconds keeps Kilo's headerTimeout/chunkTimeout from tripping
# while we buffer a response for the early-stop middleware, and lets us notice
# a client that already gave up (write fails -> abort upstream, free the lock).
SSE_KEEPALIVE_SECS = 3.0
_SSE_KEEPALIVE = b": keepalive\n\n"
_SSE_DONE = b"data: [DONE]\n\n"
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
    "compress conversation context",
    "compress the conversation",
    "agent=compaction",
)
# A tool-less request carrying this many turns of history is a compaction
# even when the prompt text does not match a hint (Kilo's compaction agent
# prompt is user-configurable). Plain chat never arrives with a long history.
_COMPACTION_MIN_MSGS = 16


def _looks_like_compaction(body: dict) -> bool:
    tc = body.get("tool_choice")
    if tc == "none" or (isinstance(tc, dict) and tc.get("type") == "none"):
        return True
    # Kilo agent system prompts often mention "summarize the conversation".
    # That is not compaction. Never strip tools on a real tool turn.
    tools = body.get("tools") or body.get("functions") or []
    if isinstance(tools, list) and tools:
        return False
    msgs = body.get("messages") or []
    if isinstance(msgs, list) and len(msgs) >= _COMPACTION_MIN_MSGS:
        return True
    for msg in msgs:
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


def _assistant_signature(msg: dict) -> str | None:
    """Normalised identity of an assistant turn: tool calls (name+args) + text.

    None when the turn is too small to mean anything (a bare 'Done.').
    """
    tcs = msg.get("tool_calls")
    sig_tools: list[str] = []
    if isinstance(tcs, list):
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    pass
            if isinstance(args, dict):
                args = json.dumps(args, sort_keys=True, ensure_ascii=False)
            sig_tools.append(f"{name}({args})")
    text = re.sub(r"\s+", " ", _message_text(msg)).strip().lower()
    if not sig_tools and len(text) < 16:
        return None
    return "|".join(sorted(sig_tools)) + "#" + text


def _assistant_repeat_count(messages: list[dict] | None) -> tuple[int, str]:
    """How many CONSECUTIVE assistant turns (ignoring interleaved tool/user
    messages) are identical to the most recent one. Returns (count, sig)."""
    if not messages:
        return 0, ""
    last_sig: str | None = None
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        sig = _assistant_signature(msg)
        if last_sig is None:
            if sig is None:
                return 0, ""
            last_sig = sig
            count = 1
            continue
        if sig == last_sig:
            count += 1
        else:
            break
    return count, last_sig or ""


def _repeat_summary(sig: str) -> str:
    head = sig.split("#", 1)[0]
    if head:
        return head[:120]
    return "identical text: " + sig.split("#", 1)[-1][:80]


def _message_chars(msg: dict) -> int:
    n = len(_message_text(msg))
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        n += len(json.dumps(tcs, ensure_ascii=False))
    return n


def _trim_history_to_budget(messages: list[dict], budget: int = CONTEXT_CHAR_BUDGET) -> int:
    """Drop the oldest non-system messages until the history fits ``budget``.

    Drops whole groups so the OpenAI contract holds: an assistant message with
    tool_calls is removed together with the tool results that answer it. The
    leading system message(s) and the last 4 messages are never touched.
    Returns the number of messages dropped.
    """
    total = sum(_message_chars(m) for m in messages if isinstance(m, dict))
    if total <= budget:
        return 0
    first = 0
    while first < len(messages) and isinstance(messages[first], dict) and messages[first].get("role") == "system":
        first += 1
    keep_tail = 4
    dropped = 0
    while total > budget and len(messages) - first > keep_tail:
        msg = messages[first]
        group_end = first + 1
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            while (
                group_end < len(messages)
                and isinstance(messages[group_end], dict)
                and messages[group_end].get("role") in ("tool", "function")
            ):
                group_end += 1
        if group_end > len(messages) - keep_tail:
            break
        for m in messages[first:group_end]:
            if isinstance(m, dict):
                total -= _message_chars(m)
        del messages[first:group_end]
        dropped += group_end - first
    # Never leave an orphaned tool result at the head of the history.
    while (
        len(messages) - first > keep_tail
        and isinstance(messages[first], dict)
        and messages[first].get("role") in ("tool", "function")
    ):
        total -= _message_chars(messages[first])
        del messages[first]
        dropped += 1
    if dropped:
        note = _TRIM_NOTE.format(n=dropped)
        if first > 0:
            sys_msg = messages[first - 1]
            _set_message_text(sys_msg, _message_text(sys_msg) + "\n\n" + note)
        else:
            messages.insert(0, {"role": "system", "content": note})
        log.info("[agent] context trim: dropped=%s remaining_chars=%s", dropped, total)
    return dropped


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


def _strip_nudge(messages: list[dict], nudge: str) -> None:
    needle = (nudge or "").strip()
    if not needle:
        return
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if needle not in text and nudge not in text:
            continue
        cleaned = text.replace(nudge, "").replace(needle, "")
        _set_message_text(msg, cleaned)


def _is_harness_user(msg: dict) -> bool:
    text = _message_text(msg)
    return "[Harness] EARLY STOP:" in text


def _tool_rounds_since_user(messages: list[dict] | None) -> int:
    """Tool results after the last real (non-harness) user message."""
    n = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            if _is_harness_user(msg):
                continue
            break
        if role in ("tool", "function"):
            n += 1
    return n


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
        "repeat_count": 0,
        "repeat_recovery": False,
        "repeat_no_tools": False,
        "repeat_stop": False,
        "repeat_what": "",
        "trimmed_msgs": 0,
    }
    if not messages:
        return trace
    truncated = _truncate_tool_messages(messages)
    trace["truncated_tool_msgs"] = truncated
    cfg = _active_harness()
    if cfg.get("mw_context_trim", True):
        budget = int(cfg.get("context_char_budget") or CONTEXT_CHAR_BUDGET)
        trace["trimmed_msgs"] = _trim_history_to_budget(messages, budget)
    streak = _recent_empty_tool_streak(messages)
    fake = _assistant_faked_action(messages)
    prose = _assistant_prose_loop(messages)
    last_role = _last_non_system_role(messages)
    trace["empty_tool_streak"] = streak
    trace["fake_action"] = fake
    trace["prose_loop"] = prose
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
    elif last_role in ("tool", "function") and cfg.get("mw_after_tool", True):
        rounds = _tool_rounds_since_user(messages)
        after_max = int(cfg.get("after_tool_nudge_max") or AFTER_TOOL_NUDGE_MAX)
        if rounds <= after_max:
            _nudge_system(
                messages,
                "[Harness] Tool result received.",
                _AFTER_TOOL_NUDGE,
                "after-tool continue",
            )
            trace["after_tool_continue"] = True
        else:
            _strip_nudge(messages, _AFTER_TOOL_NUDGE)
            log.info("[agent] after-tool cap rounds=%s max=%s", rounds, after_max)
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
    if cfg.get("mw_repeat_guard", True):
        rep, sig = _assistant_repeat_count(messages)
        trace["repeat_count"] = rep
        nudge_min = int(cfg.get("repeat_nudge_min") or REPEAT_NUDGE_MIN)
        no_tools_min = int(cfg.get("repeat_no_tools") or REPEAT_NO_TOOLS)
        stop_min = int(cfg.get("repeat_hard_stop") or REPEAT_HARD_STOP)
        if rep >= nudge_min:
            trace["repeat_what"] = _repeat_summary(sig)
            # Escalate: the marker is stable, the count in the text is not, so
            # replace an older nudge instead of stacking a second one.
            _strip_nudge_prefix(messages, "[Harness] REPEATED ACTION:")
            _nudge_system(
                messages,
                "[Harness] REPEATED ACTION:",
                _REPEAT_NUDGE.format(n=rep),
                f"repeat recovery (n={rep} what={trace['repeat_what']!r})",
            )
            trace["repeat_recovery"] = True
            trace["repeat_no_tools"] = rep >= no_tools_min
            trace["repeat_stop"] = rep >= stop_min
    if (
        trace.get("empty_tool_recovery")
        or trace.get("fake_action_recovery")
        or trace.get("prose_loop_recovery")
        or trace.get("repeat_recovery")
    ):
        append_live_event(
            {
                "empty_tool_recovery": trace.get("empty_tool_recovery"),
                "fake_action_recovery": trace.get("fake_action_recovery"),
                "prose_loop_recovery": trace.get("prose_loop_recovery"),
                "repeat_recovery": trace.get("repeat_recovery"),
                "repeat_count": trace.get("repeat_count"),
                "repeat_stop": trace.get("repeat_stop"),
                "after_tool_continue": trace.get("after_tool_continue"),
                "fake_action": fake,
                "prose_loop": prose,
                "empty_tool_streak": streak,
            }
        )
    return trace


def _strip_nudge_prefix(messages: list[dict], marker: str) -> None:
    """Remove a previously injected nudge paragraph that starts with ``marker``."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        i = text.find(marker)
        if i < 0:
            continue
        j = text.find("\n\n", i)
        cleaned = (text[:i] + (text[j:] if j >= 0 else "")).rstrip()
        _set_message_text(msg, cleaned)


def _apply_card_sampling(body: dict) -> None:
    body["temperature"] = CARD_TEMPERATURE
    body["top_p"] = CARD_TOP_P
    body["frequency_penalty"] = CARD_FREQUENCY_PENALTY
    body["enable_thinking"] = False
    body.pop("top_k", None)


def _ensure_max_tokens(body: dict, *, floor: int, cap: int | None = None) -> None:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0 or value < floor:
        value = floor
    if cap is not None and value > cap:
        value = max(cap, floor)
    body["max_tokens"] = value
    # mtplx honours max_tokens; drop the alias so the two never disagree.
    body.pop("max_completion_tokens", None)


def _cap_max_tokens(body: dict, *, cap: int) -> None:
    """Clamp an explicit max_tokens without inventing one."""
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    if raw is None:
        return
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return
    if value > cap:
        body["max_tokens"] = cap
        body.pop("max_completion_tokens", None)


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
            "repeat_count": 0,
            "repeat_recovery": False,
            "repeat_no_tools": False,
            "repeat_stop": False,
            "repeat_what": "",
            "trimmed_msgs": 0,
        }
    )
    if _looks_like_compaction(body):
        body.pop("tools", None)
        body["tool_choice"] = "none"
        _apply_card_sampling(body)
        _ensure_max_tokens(body, floor=512, cap=COMPACTION_MAX_TOKENS_CAP)
        msgs = body.get("messages")
        if isinstance(msgs, list):
            # A compaction prompt is the one request guaranteed to arrive with
            # the whole history; keep it inside the window the engine can
            # actually decode at a usable speed.
            tr["truncated_tool_msgs"] = _truncate_tool_messages(msgs)
            tr["trimmed_msgs"] = _trim_history_to_budget(msgs)
        tr["compaction"] = True
        return body

    _apply_card_sampling(body)
    if body.get("tools"):
        _ensure_max_tokens(
            body, floor=AGENT_MAX_TOKENS_FLOOR, cap=AGENT_MAX_TOKENS_CAP
        )
        _inject_loop_prompt(body)
        tr["loop_prompt"] = True
        msgs = body.get("messages")
        if isinstance(msgs, list):
            tr.update(apply_loop_middleware(msgs))
            _repair_history_tool_calls(msgs)
            if tr.get("repeat_no_tools"):
                # The model keeps re-issuing one tool call. Take the tools
                # away for this turn so it has to answer in text, which ends
                # Kilo's tool loop instead of burning another engine slot.
                body.pop("tools", None)
                body.pop("functions", None)
                body["tool_choice"] = "none"
                _cap_max_tokens(body, cap=CHAT_MAX_TOKENS_CAP)
                log.info("[agent] repeat guard: tools removed (n=%s)", tr.get("repeat_count"))
    else:
        # Plain chat / title generation: still bound a runaway greedy loop,
        # but leave an absent max_tokens to the engine default.
        _cap_max_tokens(body, cap=CHAT_MAX_TOKENS_CAP)
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


def _should_force_continue(
    text: str, has_tools: bool, messages: list[dict] | None
) -> bool:
    if has_tools:
        return False
    rounds = _tool_rounds_since_user(messages)
    max_force = int(
        _active_harness().get("early_stop_force_max") or EARLY_STOP_FORCE_MAX
    )
    if rounds > max_force:
        return False
    if _is_early_stop_plan(text, False):
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


def _recent_bash_commands(messages: list[dict] | None) -> set[str]:
    found: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            if not isinstance(fn, dict):
                continue
            raw = fn.get("arguments") or "{}"
            parsed: dict | None = None
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = None
            if parsed and parsed.get("command"):
                found.add(str(parsed["command"]).strip())
    return found


def _synthetic_tool_calls(
    text: str, messages: list[dict] | None
) -> list[dict]:
    cmd = _command_from_plan(text)
    if not cmd:
        return []
    cmd = _sanitize_bash_command(cmd, messages)
    if cmd in _recent_bash_commands(messages):
        log.info("[agent] skip duplicate synthetic %s", cmd[:80])
        return []
    return [_make_tool_call("bash", {"command": cmd})]


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


def _sse_text_payload(text: str, model: str, response_id: str) -> bytes:
    """A complete SSE body carrying one final assistant text (finish=stop)."""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _json_text_payload(text: str, model: str, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _is_chat_path(path: str) -> bool:
    p = (path or "").rstrip("/")
    return p.endswith("/chat/completions") or p.endswith("/messages")


def _drop_session_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _SESSION_HEADER_DROP
    }


def _scrub_session_body(data: dict) -> bool:
    changed = False
    for key in _SESSION_BODY_DROP:
        if key in data:
            data.pop(key, None)
            changed = True
    meta = data.get("metadata")
    if isinstance(meta, dict):
        for key in _SESSION_BODY_DROP:
            if key in meta:
                meta.pop(key, None)
                changed = True
    return changed


def _is_in_flight_error(status: int, payload: bytes) -> bool:
    if status not in {409, 429, 500, 503}:
        return False
    try:
        blob = payload.decode("utf-8", errors="replace")
    except Exception:
        return False
    return bool(_IN_FLIGHT_RE.search(blob))


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
        return HTTPSConnection(state.host, state.port, timeout=UPSTREAM_TIMEOUT)
    return HTTPConnection(state.host, state.port, timeout=UPSTREAM_TIMEOUT)


def _needs_buffered_sse(steer_trace: dict[str, Any]) -> bool:
    return bool(
        steer_trace.get("after_tool_continue")
        and _active_harness().get("mw_early_stop", True)
    )


class _ClientGone(Exception):
    """The Kilo side closed its socket; stop generating for it."""


def _sse_error_payload(status: int, detail: bytes | str) -> bytes:
    """Wrap an upstream/proxy failure as a terminal SSE event.

    Once we have committed to ``text/event-stream`` we cannot change the HTTP
    status, so the OpenAI-style error object is delivered in-band. Every
    OpenAI-compatible client surfaces ``{"error": ...}`` chunks as a failure.
    """
    text = detail.decode("utf-8", "replace") if isinstance(detail, bytes) else detail
    message = text.strip() or f"upstream HTTP {status}"
    stripped = text.lstrip()
    if stripped.startswith("data:") or '"chat.completion.chunk"' in stripped[:600]:
        # A partial SSE body (the engine died or restarted mid-stream). Kilo
        # would otherwise show 2 KB of raw chunk JSON as the error text.
        message = (
            f"upstream stream aborted mid-response (HTTP {status}); the engine "
            "probably restarted — retry the request"
        )
    else:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                message = str(parsed["error"].get("message") or message)
        except (ValueError, TypeError):
            pass
    obj = {
        "error": {
            "message": message[:600],
            "type": "proxy_error",
            "code": int(status),
        }
    }
    return (
        f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8") + _SSE_DONE
    )


def _sse_at_event_boundary(buf: bytes) -> bool:
    """True when it is safe to interleave an SSE comment line."""
    return not buf or buf.endswith(b"\n\n") or buf.endswith(b"\r\n\r\n")


class _UpstreamFetch:
    """One upstream request, read in a worker thread.

    The handler thread never blocks on the engine socket. It polls the chunk
    queue with a short timeout so it can emit keepalives to the client (and
    notice when the client is gone) even while mtplx is still prefilling.
    """

    def __init__(
        self,
        state: ProxyState,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.content_type = ""
        self.error: BaseException | None = None
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._ready = threading.Event()
        self._aborted = False
        self._conn: HTTPConnection | None = None
        self._thread = threading.Thread(
            target=self._run,
            args=(state, method, path, body, headers),
            name="upstream-fetch",
            daemon=True,
        )
        self._thread.start()

    def _run(
        self,
        state: ProxyState,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        conn: HTTPConnection | None = None
        try:
            conn = _connect(state)
            self._conn = conn
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            self.status = resp.status
            self.headers = {k: v for k, v in resp.getheaders()}
            # uvicorn/mtplx send lowercase names; http.client.getheader is
            # case-insensitive. A dict .get("Content-Type") misses them and
            # the passthrough path then wraps a good SSE stream as
            # {"error": ...} — Kilo reports that as "cannot connect to API".
            self.content_type = resp.getheader("Content-Type") or ""
            self._ready.set()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self._q.put(chunk)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the handler
            if not self._aborted:
                self.error = exc
        finally:
            self._ready.set()
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._q.put(None)

    def abort(self) -> None:
        """Tear down the engine socket so mtplx cancels the generation."""
        self._aborted = True
        conn = self._conn
        if conn is None:
            return
        sock = getattr(conn, "sock", None)
        try:
            if sock is not None:
                sock.shutdown(2)  # SHUT_RDWR
        except OSError:
            pass
        try:
            conn.close()
        except Exception:
            pass

    def wait_ready(self, on_tick, tick: float = SSE_KEEPALIVE_SECS) -> None:
        while not self._ready.wait(tick):
            on_tick()
        if self.error is not None:
            raise self.error

    def iter_chunks(self, on_tick, tick: float = SSE_KEEPALIVE_SECS):
        while True:
            try:
                chunk = self._q.get(timeout=tick)
            except queue.Empty:
                on_tick()
                continue
            if chunk is None:
                if self.error is not None:
                    raise self.error
                return
            yield chunk

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)


def _noop() -> None:
    return None


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
                    "limits": {
                        "max_tokens_cap": AGENT_MAX_TOKENS_CAP,
                        "upstream_timeout": UPSTREAM_TIMEOUT,
                        "lock_wait_timeout": LOCK_WAIT_TIMEOUT,
                        "sse_keepalive_secs": SSE_KEEPALIVE_SECS,
                    },
                    "engine_busy": _UPSTREAM_GEN_LOCK_BUSY.is_set(),
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

    # ---- client-side helpers -------------------------------------------

    def _client_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise _ClientGone(str(exc)) from exc

    def _sse_begin(self, extra_headers: dict[str, str] | None = None) -> None:
        """Commit to a 200 text/event-stream response right away.

        Kilo's ``headerTimeout`` starts at request time; the engine can take
        minutes to prefill a long history, so headers go out before we know
        anything about the upstream result.
        """
        self.send_response(200)
        for k, v in (extra_headers or {}).items():
            if k.lower() in (
                "transfer-encoding",
                "connection",
                "content-length",
                "content-encoding",
                "content-type",
            ):
                continue
            self.send_header(k, v)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        try:
            self.end_headers()
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise _ClientGone(str(exc)) from exc
        self._sse_sent = b""

    def _client_closed(self) -> bool:
        """Passive EOF probe: True once Kilo has closed its side.

        The request body is fully consumed before we get here, so any readable
        state on the socket is either EOF (b"") or a reset. Works even while an
        SSE event is half-sent, when a keepalive write is not allowed.
        """
        sock = getattr(self, "connection", None)
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return False
            return sock.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _sse_tick(self) -> None:
        """Keepalive between events only (never inside a partial event)."""
        if self._client_closed():
            raise _ClientGone("client closed socket")
        if _sse_at_event_boundary(getattr(self, "_sse_sent", b"")):
            self._client_write(_SSE_KEEPALIVE)

    def _sse_send(self, data: bytes) -> None:
        self._client_write(data)
        # Remember only the tail; we just need to know if an event is open.
        self._sse_sent = data[-4:]

    def _sse_fail(self, status: int, detail: bytes | str) -> None:
        log.info("[agent] sse in-band error %s", status)
        self._sse_send(_sse_error_payload(status, detail))

    # ---- engine lock -----------------------------------------------------

    def _acquire_gen_lock(self, on_tick) -> bool:
        """Serialize engine use; keep the client alive while queued."""
        deadline = time.monotonic() + LOCK_WAIT_TIMEOUT
        waited = False
        while True:
            if _UPSTREAM_GEN_LOCK.acquire(timeout=SSE_KEEPALIVE_SECS):
                _UPSTREAM_GEN_LOCK_BUSY.set()
                if waited:
                    log.info("[agent] engine lock acquired after wait")
                return True
            if not waited:
                log.info("[agent] engine busy; queuing request")
                waited = True
            on_tick()
            if time.monotonic() >= deadline:
                log.info("[agent] engine lock wait exceeded %ss", LOCK_WAIT_TIMEOUT)
                return False

    @staticmethod
    def _release_gen_lock() -> None:
        _UPSTREAM_GEN_LOCK_BUSY.clear()
        try:
            _UPSTREAM_GEN_LOCK.release()
        except RuntimeError:
            pass

    # ---- upstream --------------------------------------------------------

    def _fetch_buffered(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        on_tick=_noop,
    ) -> tuple[int, dict[str, str], str, bytes]:
        """Full upstream response, retrying mtplx 409 session-in-flight.

        Caller holds the generation lock. ``on_tick`` runs every few seconds
        while we wait so a streaming client can be kept alive / detected gone.
        """
        chat = _is_chat_path(path)
        attempts = _IN_FLIGHT_RETRIES if chat else 1
        last: tuple[int, dict[str, str], str, bytes] | None = None
        for attempt in range(attempts):
            fetch = _UpstreamFetch(self.state, self.command, path, body, headers)
            try:
                fetch.wait_ready(on_tick)
                chunks = list(fetch.iter_chunks(on_tick))
            except _ClientGone:
                fetch.abort()
                fetch.join()
                raise
            finally:
                fetch.join()
            last = (
                int(fetch.status or 502),
                fetch.headers,
                fetch.content_type,
                b"".join(chunks),
            )
            if chat and _is_in_flight_error(last[0], last[3]):
                log.info(
                    "[agent] session already in flight (try %s/%s)",
                    attempt + 1,
                    attempts,
                )
                # Back off, but keep the client alive while doing so.
                end = time.monotonic() + 0.4 * (attempt + 1)
                while time.monotonic() < end:
                    time.sleep(min(0.2, max(0.0, end - time.monotonic())))
                on_tick()
                continue
            return last
        assert last is not None
        return last

    def _stream_passthrough(
        self, path: str, body: bytes, headers: dict[str, str]
    ) -> None:
        """Forward SSE as it arrives; headers + keepalives go out immediately."""
        chat = _is_chat_path(path)
        self._sse_begin()
        if chat and not self._acquire_gen_lock(self._sse_tick):
            self._sse_fail(503, "engine busy: another generation is still running")
            return
        fetch: _UpstreamFetch | None = None
        try:
            fetch = _UpstreamFetch(self.state, self.command, path, body, headers)
            fetch.wait_ready(self._sse_tick)
            status = int(fetch.status or 502)
            if status != 200 or "text/event-stream" not in fetch.content_type:
                payload = b"".join(fetch.iter_chunks(self._sse_tick))
                if status == 200 and "application/json" in fetch.content_type:
                    # Upstream ignored stream=true; hand the JSON over as-is
                    # inside an SSE event so the client still gets an answer.
                    payload = _repair_response_payload(payload, None)
                    self._sse_send(b"data: " + payload + b"\n\n" + _SSE_DONE)
                    return
                self._sse_fail(status, payload)
                return
            for chunk in fetch.iter_chunks(self._sse_tick):
                self._sse_send(chunk)
            if not getattr(self, "_sse_sent", b"").endswith(b"\n\n"):
                self._sse_send(b"\n\n")
        except _ClientGone:
            log.info("client disconnected mid-stream; aborting upstream")
            if fetch is not None:
                fetch.abort()
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error (stream): %s", exc)
            try:
                self._sse_fail(502, str(exc))
            except _ClientGone:
                pass
        finally:
            if fetch is not None:
                fetch.join()
            if chat:
                self._release_gen_lock()

    # ---- middleware decision -------------------------------------------

    def _post_process(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
        parsed: dict | None,
        steer_messages: list[dict] | None,
        steer_trace: dict[str, Any],
        stream: bool,
        on_tick,
    ) -> tuple[int, dict[str, str], str, bytes]:
        """Fetch once, apply the early-stop retry / synthetic continue."""
        status, resp_headers, content_type, payload = self._fetch_buffered(
            path, body, headers, on_tick
        )
        if status != 200:
            return status, resp_headers, content_type, payload
        is_sse = stream or "text/event-stream" in content_type
        want_retry = (
            parsed is not None
            and steer_trace.get("after_tool_continue")
            and _active_harness().get("mw_early_stop", True)
        )
        if is_sse:
            text, has_tools = _sse_text_and_tools(payload)
            if want_retry and _should_force_continue(text, has_tools, steer_messages):
                log.info("[agent] early-stop plan retry")
                _apply_early_stop_retry(parsed)  # type: ignore[arg-type]
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                status, resp_headers, content_type, payload = self._fetch_buffered(
                    path, body, headers, on_tick
                )
                if status != 200:
                    return status, resp_headers, content_type, payload
                is_sse = stream or "text/event-stream" in content_type
                text, has_tools = _sse_text_and_tools(payload)
                if _should_force_continue(text, has_tools, steer_messages):
                    tcs = _synthetic_tool_calls(text, steer_messages)
                    if tcs:
                        model, rid = _payload_ids(payload, sse=is_sse)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        return status, resp_headers, content_type, _sse_tool_payload(
                            tcs, model, rid
                        )
                    log.info("[agent] early-stop cap: letting recap through")
            return (
                status,
                resp_headers,
                content_type,
                _repair_sse_payload(payload, steer_messages),
            )
        if "application/json" in content_type:
            payload = _repair_response_payload(payload, steer_messages)
            text, has_tools = _json_text_and_tools(payload)
            if want_retry and _should_force_continue(text, has_tools, steer_messages):
                log.info("[agent] early-stop plan retry")
                _apply_early_stop_retry(parsed)  # type: ignore[arg-type]
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                status, resp_headers, content_type, payload = self._fetch_buffered(
                    path, body, headers, on_tick
                )
                if status != 200:
                    return status, resp_headers, content_type, payload
                if "application/json" in content_type:
                    payload = _repair_response_payload(payload, steer_messages)
                text, has_tools = _json_text_and_tools(payload)
                if _should_force_continue(text, has_tools, steer_messages):
                    tcs = _synthetic_tool_calls(text, steer_messages)
                    if tcs:
                        model, rid = _payload_ids(payload, sse=False)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        payload = _json_tool_payload(tcs, model, rid)
                    else:
                        log.info("[agent] early-stop cap: letting recap through")
        return status, resp_headers, content_type, payload

    # ---- request entry ---------------------------------------------------

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
            and k.lower() not in _SESSION_HEADER_DROP
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
                if _scrub_session_body(data):
                    log.info("[agent] stripped client session affinity")
                stream = bool(data.get("stream"))
                if _is_chat_path(path):
                    log.info("[in] %s", summarize_request(data))
                    prepare_body(data, steer_trace)
                    msgs = data.get("messages")
                    if isinstance(msgs, list):
                        steer_messages = msgs
                    log.info(
                        "[steer] temp=%s think=%s fp=%s ntools=%s max_tokens=%s "
                        "compaction=%s empty_rec=%s fake=%s prose=%s "
                        "after_tool=%s trunc=%s repeat=%s trimmed=%s nmsg=%s",
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
                        steer_trace.get("repeat_count"),
                        steer_trace.get("trimmed_msgs"),
                        len(msgs) if isinstance(msgs, list) else 0,
                    )
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"

        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{self.state.host}:{self.state.port}"
        headers["Connection"] = "close"
        chat = _is_chat_path(path)

        if chat and steer_trace.get("repeat_stop") and parsed is not None:
            # Hard stop: N identical assistant turns in a row. Do not spend
            # another engine slot; end the turn with a visible explanation.
            n = int(steer_trace.get("repeat_count") or 0)
            what = str(steer_trace.get("repeat_what") or "same action")
            text = _REPEAT_STOP_TEXT.format(n=n, what=what)
            model = str(parsed.get("model") or "qwen3.8-27b-obliterated-mtplx")
            rid = f"chatcmpl-proxy-repeat-{int(time.time())}"
            log.warning("[agent] repeat guard: hard stop n=%s what=%r", n, what)
            append_live_event({"repeat_hard_stop": True, "repeat_count": n, "what": what})
            try:
                if stream:
                    self._sse_begin()
                    self._sse_send(_sse_text_payload(text, model, rid))
                else:
                    payload = _json_text_payload(text, model, rid)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except (_ClientGone, BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            if stream and not _needs_buffered_sse(steer_trace):
                log.info("[agent] sse passthrough")
                self._stream_passthrough(path, body, headers)
                return
            if stream:
                self._proxy_buffered_sse(
                    path, headers, body, parsed, steer_messages, steer_trace
                )
                return

            # Non-streaming (harness / curl): plain JSON in, plain JSON out.
            if chat and not self._acquire_gen_lock(_noop):
                self._send_json(
                    503,
                    {
                        "error": {
                            "message": "engine busy: another generation is still running",
                            "type": "proxy_busy",
                        }
                    },
                )
                return
            try:
                status, resp_headers, content_type, payload = self._post_process(
                    path=path,
                    headers=headers,
                    body=body,
                    parsed=parsed,
                    steer_messages=steer_messages,
                    steer_trace=steer_trace,
                    stream=False,
                    on_tick=_noop,
                )
            finally:
                if chat:
                    self._release_gen_lock()
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
        except _ClientGone:
            log.info("client disconnected (gone before response)")
        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected (broken pipe)")
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error: %s", exc)
            try:
                self._send_json(
                    502, {"error": {"message": str(exc), "type": "proxy_error"}}
                )
            except Exception:
                pass

    def _proxy_buffered_sse(
        self,
        path: str,
        headers: dict[str, str],
        body: bytes,
        parsed: dict | None,
        steer_messages: list[dict] | None,
        steer_trace: dict[str, Any],
    ) -> None:
        """Buffered SSE (early-stop middleware) with the client kept alive.

        Headers are committed first; ``: keepalive`` comments flow while the
        engine works (including a possible retry). The final payload is a
        complete SSE body, so nothing is ever retracted from the client.
        """
        log.info("[agent] sse buffered (early-stop middleware)")
        self._sse_begin()
        if not self._acquire_gen_lock(self._sse_tick):
            self._sse_fail(503, "engine busy: another generation is still running")
            return
        try:
            status, _resp_headers, content_type, payload = self._post_process(
                path=path,
                headers=headers,
                body=body,
                parsed=parsed,
                steer_messages=steer_messages,
                steer_trace=steer_trace,
                stream=True,
                on_tick=self._sse_tick,
            )
            if status != 200:
                self._sse_fail(status, payload)
                return
            if "text/event-stream" not in content_type and payload.lstrip().startswith(
                b"{"
            ):
                self._sse_send(b"data: " + payload.strip() + b"\n\n" + _SSE_DONE)
                return
            if b"[DONE]" not in payload[-64:]:
                payload = payload.rstrip(b"\r\n") + b"\n\n" + _SSE_DONE
            self._sse_send(payload)
        except _ClientGone:
            log.info("client disconnected while buffering; upstream aborted")
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error (buffered sse): %s", exc)
            try:
                self._sse_fail(502, str(exc))
            except _ClientGone:
                pass
        finally:
            self._release_gen_lock()


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

    cap_msgs: list[dict] = [
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
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            }
        )
        cap_msgs.append(
            {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}
        )
    cap_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": cap_msgs,
        "max_tokens": 128,
    }
    # First inject the JIT nudge (as if round 1 already happened), then
    # prepare the 4-tool history so the cap strips it.
    cap_msgs[0]["content"] += _AFTER_TOOL_NUDGE
    tr_cap: dict[str, Any] = {}
    prepare_body(cap_body, tr_cap)
    assert tr_cap.get("after_tool_continue") is False, tr_cap
    assert "[Harness] Tool result received." not in cap_body["messages"][0]["content"]
    assert _tool_rounds_since_user(cap_msgs) == 4
    plan = "Next steps\nE1 ecall 0x49 — one next RE probe on host wrapper.\n"
    assert _should_force_continue(plan, False, cap_msgs) is False

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
    assert _synthetic_tool_calls(plan_cmd, already) == []
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
    dropped = _drop_session_headers(
        {"X-Session-Id": "ses_abc", "Authorization": "local", "Content-Type": "application/json"}
    )
    assert "X-Session-Id" not in dropped
    assert dropped.get("Authorization") == "local"
    body_sid = {
        "user": "kilo",
        "session_id": "ses_abc",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"session_id": "ses_abc"},
    }
    assert _scrub_session_body(body_sid) is True
    assert "user" not in body_sid
    assert "session_id" not in body_sid
    assert "session_id" not in body_sid["metadata"]
    assert body_sid["messages"][0]["content"] == "hi"
    busy = json.dumps(
        {"error": {"message": "session ses_fa290ee49ffeuGnXskKEfH3z3m is already in flight"}}
    ).encode()
    assert _is_in_flight_error(409, busy) is True
    assert _is_in_flight_error(200, busy) is False
    assert _is_chat_path("/v1/chat/completions") is True
    assert UPSTREAM_TIMEOUT >= 120
    assert _needs_buffered_sse({"after_tool_continue": False}) is False
    assert _needs_buffered_sse({"after_tool_continue": True}) is True

    # Stability: runaway greedy completions are capped, floors still apply.
    cap_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 32000,
    }
    prepare_body(cap_body)
    assert cap_body["max_tokens"] == AGENT_MAX_TOKENS_CAP, cap_body["max_tokens"]
    small_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 16,
    }
    prepare_body(small_body)
    assert small_body["max_tokens"] == AGENT_MAX_TOKENS_FLOOR
    alias_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_completion_tokens": 100000,
    }
    prepare_body(alias_body)
    assert alias_body["max_tokens"] == AGENT_MAX_TOKENS_CAP
    assert "max_completion_tokens" not in alias_body
    plain_body = {"messages": [{"role": "user", "content": "hi"}]}
    prepare_body(plain_body)
    assert "max_tokens" not in plain_body  # engine default, nothing invented
    plain_big = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}
    prepare_body(plain_big)
    assert plain_big["max_tokens"] == CHAT_MAX_TOKENS_CAP
    ping_body = {"messages": [{"role": "user", "content": "PING"}], "max_tokens": 16}
    prepare_body(ping_body)
    assert ping_body["max_tokens"] == 16

    # Compaction: detected by history length even without a prompt hint, and
    # its decode is capped.
    long_hist = [{"role": "system", "content": "Compress conversation context."}]
    for i in range(20):
        long_hist.append({"role": "user", "content": f"q{i}"})
        long_hist.append({"role": "assistant", "content": f"a{i} " * 8})
    comp_body = {"messages": long_hist, "max_tokens": 32768}
    tr_comp: dict[str, Any] = {}
    prepare_body(comp_body, tr_comp)
    assert tr_comp["compaction"] is True
    assert comp_body["max_tokens"] == COMPACTION_MAX_TOKENS_CAP
    hint_body = {"messages": [{"role": "user", "content": "compress conversation context"}]}
    assert _looks_like_compaction(hint_body) is True

    # Repeat guard: same tool call three turns running -> nudge, tools gone.
    rep_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"filePath": "notes.md"}'},
    }
    rep_msgs: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for _ in range(3):
        rep_msgs.append({"role": "assistant", "content": "Let me check the notes.", "tool_calls": [dict(rep_call)]})
        rep_msgs.append({"role": "tool", "tool_call_id": "c1", "content": "# notes\nline"})
    n_rep, sig_rep = _assistant_repeat_count(rep_msgs)
    assert n_rep == 3 and sig_rep.startswith("read(")
    rep_body = {
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": [dict(m) for m in rep_msgs],
    }
    tr_rep: dict[str, Any] = {}
    prepare_body(rep_body, tr_rep)
    assert tr_rep["repeat_count"] == 3 and tr_rep["repeat_recovery"] is True
    assert tr_rep["repeat_no_tools"] is True and tr_rep["repeat_stop"] is False
    assert "tools" not in rep_body and rep_body["tool_choice"] == "none"
    assert "[Harness] REPEATED ACTION" in rep_body["messages"][0]["content"]
    assert rep_body["messages"][0]["content"].count("[Harness] REPEATED ACTION") == 1
    # Four in a row -> hard stop flag (the handler answers without upstream).
    rep_msgs.append({"role": "assistant", "content": "Let me check the notes.", "tool_calls": [dict(rep_call)]})
    rep_msgs.append({"role": "tool", "tool_call_id": "c1", "content": "# notes\nline"})
    tr_rep4: dict[str, Any] = {}
    prepare_body({"tools": [{"type": "function", "function": {"name": "read"}}], "messages": rep_msgs}, tr_rep4)
    assert tr_rep4["repeat_stop"] is True and tr_rep4["repeat_count"] == 4
    # A different call in between breaks the chain (git status before/after a
    # commit is legitimate).
    mixed = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [dict(rep_call)]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "git commit"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [dict(rep_call)]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    assert _assistant_repeat_count(mixed)[0] == 1
    # Identical prose (no tools) also counts; tiny acks do not.
    prose_rep = [{"role": "user", "content": "a"}] + [
        {"role": "assistant", "content": "I will check the directory and report back shortly."}
    ] * 3
    assert _assistant_repeat_count(prose_rep)[0] == 3
    assert _assistant_repeat_count([{"role": "assistant", "content": "Done."}] * 5)[0] == 0
    stop_txt = _sse_text_payload("halt", "m", "rid")
    assert b'"finish_reason": "stop"' in stop_txt and stop_txt.endswith(_SSE_DONE)
    assert json.loads(_json_text_payload("halt", "m", "rid"))["choices"][0]["message"]["content"] == "halt"

    # Context trim: oldest groups go first, tool_call/tool pairs stay intact,
    # the tail is untouched, a note is appended to the system prompt.
    trim_msgs: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for i in range(6):
        trim_msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        trim_msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 5000})
    trim_msgs.append({"role": "user", "content": "final question"})
    dropped = _trim_history_to_budget(trim_msgs, budget=16000)
    # 'start' user message + whole assistant/tool pairs -> odd count, >= 3.
    assert dropped >= 3 and dropped % 2 == 1, dropped
    assert trim_msgs[-1]["content"] == "final question"
    roles_after = [m["role"] for m in trim_msgs]
    assert roles_after[0] == "system" and "dropped" in trim_msgs[0]["content"]
    for i, m in enumerate(trim_msgs):
        if m["role"] == "tool":
            assert trim_msgs[i - 1]["role"] == "assistant" and trim_msgs[i - 1].get("tool_calls")
    assert _trim_history_to_budget([{"role": "user", "content": "short"}], budget=1000) == 0

    # SSE plumbing: in-band error is a terminal, parseable event.
    err = _sse_error_payload(503, b'{"error":{"message":"busy"}}')
    assert err.startswith(b"data: ") and err.endswith(_SSE_DONE)
    evt = json.loads(err.split(b"\n\n")[0][len(b"data: "):])
    assert evt["error"]["message"] == "busy" and evt["error"]["code"] == 503
    err2 = _sse_error_payload(502, "socket closed")
    assert b"socket closed" in err2
    # A half-streamed upstream body must not be echoed back as the error text.
    err3 = _sse_error_payload(502, b'data: {"id":"x","object":"chat.completion.chunk","choices":[]}\n\n: keep-alive\n\n')
    evt3 = json.loads(err3.split(b"\n\n")[0][len(b"data: "):])
    assert "aborted mid-response" in evt3["error"]["message"]
    assert "chat.completion.chunk" not in evt3["error"]["message"]
    assert _sse_at_event_boundary(b"") is True
    assert _sse_at_event_boundary(b"data: {}\n\n") is True
    assert _sse_at_event_boundary(b"data: {\"cho") is False
    assert _SSE_KEEPALIVE.startswith(b":") and _SSE_KEEPALIVE.endswith(b"\n\n")

    # Engine lock: RLock with timeout works and busy flag mirrors it.
    assert _UPSTREAM_GEN_LOCK.acquire(timeout=0.01) is True
    _UPSTREAM_GEN_LOCK.release()
    assert _UPSTREAM_GEN_LOCK_BUSY.is_set() is False
    assert LOCK_WAIT_TIMEOUT > 0 and _env_int("QWEN38_OBL_UNSET_TEST", 900) == 900


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
