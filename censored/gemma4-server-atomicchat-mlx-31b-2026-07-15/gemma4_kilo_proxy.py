#!/usr/bin/env python3
"""Lean Kilo harness proxy for Gemma 4 AtomicChat + mlx_lm/mlx_vlm.

Sits between Kilo and the MLX OpenAI server:

  Kilo  →  :8080 this proxy  →  :8090 mlx_lm.server / mlx_vlm.server

Why this exists
---------------
1. **Thinking / empty content (critical)**
   mlx_lm + Gemma 4 often emits all tokens in ``message.reasoning`` with
   ``content: null``. Kilo then sees blank assistant turns, never parses
   tool calls, and the agent "bombs out". Fix: force
   ``chat_template_kwargs.enable_thinking=false`` and remap any leftover
   reasoning → content (JSON + SSE).

2. **Compaction**
   Kilo compaction turns often still ship ``tools`` + ``tool_choice=auto``.
   Gemma explores instead of summarizing → ContextOverflowError after 3
   failed compactions. Fix: strip tools on summary turns; cap max_tokens.

3. **Context bloat**
   Truncate large ``role=tool`` message bodies before they hit the model.

Usage:
  python3 gemma4_kilo_proxy.py --upstream http://127.0.0.1:8090 --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("gemma4_kilo_proxy")

# Targeted harness traces (no message bodies). Disable with --no-harness-log.
HARNESS_LOG = True

# Keep compaction summaries short — long Goal/Progress dumps re-inflate context.
_COMPACTION_MAX_TOKENS = 1024
_COMPACTION_MAX_TOKENS_CEILING = 2048

# Truncate old tool results before they hit the model (chars, not tokens).
_TOOL_RESULT_MAX_CHARS = 12_000
_TOOL_RESULT_KEEP_HEAD = 8_000
_TOOL_RESULT_KEEP_TAIL = 2_000

_COMPACTION_HINTS = (
    "summarize the conversation",
    "generate a brief summary",
    "compact the conversation",
    "create a concise summary",
    "conversation summary",
    "agent=compaction",
    "preserve key information",
    "concise summary",
    "session summary",
    "summarize this conversation",
    "summarize the session",
)

_COMPACTION_NUDGE = (
    "\n\nRespond with plain text only. Do not call tools or emit tool_calls. "
    "Max ~40 short lines. No Goal/Progress/Next Steps templates. "
    "No long file trees or code dumps — paths and one-line facts only."
)


def _message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


def _set_message_text(msg: dict, text: str) -> None:
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = text
                return
        content.append({"type": "text", "text": text})
        return
    msg["content"] = text


def _tool_choice_disallows_tools(tool_choice: Any) -> bool:
    if tool_choice == "none":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


def _has_tools(body: dict) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _looks_like_compaction(body: dict) -> bool:
    """Detect Kilo session-compaction turns — must NOT false-positive agent turns.

    Critical bug (fixed): scanning the full system prompt matched phrases like
    "Do not re-summarize the conversation history" from agent harness rules,
    so *every* agent request was treated as compaction and tools were stripped.
    Kilo then could never tool-call.

    Rules:
      1. If tools are present and tool_choice is not none → agentic, never compact.
      2. tool_choice=none alone is a strong signal (Kilo compaction / text-only).
      3. Otherwise only inspect the *latest user* message for summary wording
         (never the system prompt).
    """
    choice_none = _tool_choice_disallows_tools(body.get("tool_choice"))
    if _has_tools(body) and not choice_none:
        return False

    if choice_none:
        return True

    # Text-only / no tools: only the latest user message (avoid system false positives).
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return False
    blob = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            blob = _message_text(msg).lower()
            break
    if not blob:
        return False
    if any(h in blob for h in _COMPACTION_HINTS):
        return True
    return bool(
        re.search(r"summariz(?:e|ing)\b.*\b(?:conversation|session|history|context|chat)\b", blob)
        or re.search(r"\bcompact(?:ion|ing)?\b.*\b(?:context|session|conversation|history)\b", blob)
        or re.search(r"\bagent\s*=\s*compaction\b", blob)
    )


def _nudge_compaction_system(messages: list[dict]) -> None:
    for msg in messages:
        if msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if _COMPACTION_NUDGE.strip() not in text:
            _set_message_text(msg, text + _COMPACTION_NUDGE)
        return
    messages.insert(0, {"role": "system", "content": _COMPACTION_NUDGE.strip()})


def _cap_compaction_tokens(body: dict) -> None:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0 or value > _COMPACTION_MAX_TOKENS_CEILING:
        body["max_tokens"] = _COMPACTION_MAX_TOKENS
        log.info("[compaction] max_tokens → %d", _COMPACTION_MAX_TOKENS)
    elif value > _COMPACTION_MAX_TOKENS:
        body["max_tokens"] = _COMPACTION_MAX_TOKENS
        log.info("[compaction] max_tokens %d → %d", value, _COMPACTION_MAX_TOKENS)


def _truncate_tool_text(text: str) -> str:
    if len(text) <= _TOOL_RESULT_MAX_CHARS:
        return text
    head = text[:_TOOL_RESULT_KEEP_HEAD]
    tail = text[-_TOOL_RESULT_KEEP_TAIL:]
    skipped = len(text) - _TOOL_RESULT_KEEP_HEAD - _TOOL_RESULT_KEEP_TAIL
    return (
        f"{head}\n... [proxy: truncated {skipped} chars of tool output to save context; "
        f"ask for a specific section if needed] ...\n{tail}"
    )


def _truncate_tool_messages(messages: list[dict]) -> int:
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("tool", "function"):
            continue
        text = _message_text(msg)
        if not text or len(text) <= _TOOL_RESULT_MAX_CHARS:
            continue
        _set_message_text(msg, _truncate_tool_text(text))
        n += 1
    return n


def _flatten_tool_calls_in_history(messages: list[dict]) -> int:
    """On compaction, drop residual tool_calls so the model can't re-issue them."""
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        tcs = msg.pop("tool_calls", None)
        if not tcs:
            continue
        n += 1
        snippets: list[str] = []
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or "tool"
                args = fn.get("arguments") or ""
                if isinstance(args, str) and len(args) > 400:
                    args = args[:400] + "…"
                snippets.append(f"[prior tool call: {name}({args})]")
        extra = "\n".join(snippets)
        if extra:
            prev = _message_text(msg)
            _set_message_text(msg, (prev + "\n" + extra).strip() if prev else extra)
    return n


def _disable_thinking(body: dict) -> None:
    """Force Gemma chat template out of thinking mode (agent-safe default).

    Without this, mlx_lm streams tokens into ``delta.reasoning`` and leaves
    ``content`` empty — Kilo shows blank turns and never gets tool_calls.
    """
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    else:
        kwargs = dict(kwargs)
    if kwargs.get("enable_thinking") is not False:
        kwargs["enable_thinking"] = False
        body["chat_template_kwargs"] = kwargs
        log.info("[think] chat_template_kwargs.enable_thinking=false")
    else:
        body["chat_template_kwargs"] = kwargs
    # Common client aliases (ignored by mlx if unknown; harmless)
    body["enable_thinking"] = False
    if isinstance(body.get("thinking"), dict):
        body["thinking"] = {"type": "disabled"}
    elif "thinking" not in body:
        body["thinking"] = {"type": "disabled"}


def _content_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _remap_reasoning_to_content_in_message(msg: dict) -> bool:
    """If content is empty and reasoning is set, move reasoning → content."""
    if not isinstance(msg, dict):
        return False
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        return False
    if not _content_empty(msg.get("content")):
        return False
    msg["content"] = reasoning
    msg.pop("reasoning", None)
    msg.pop("reasoning_content", None)
    return True


def _remap_reasoning_in_completion_payload(data: dict) -> bool:
    changed = False
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if isinstance(msg, dict) and _remap_reasoning_to_content_in_message(msg):
            changed = True
        delta = choice.get("delta")
        if isinstance(delta, dict):
            # Stream chunk shape: delta.reasoning without delta.content
            reasoning = delta.get("reasoning")
            if reasoning is None:
                reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                if _content_empty(delta.get("content")):
                    delta["content"] = reasoning
                    delta.pop("reasoning", None)
                    delta.pop("reasoning_content", None)
                    changed = True
    return changed


def _rewrite_sse_chunk(line: bytes) -> bytes:
    """Map reasoning→content inside one SSE ``data:`` line (pass-through otherwise)."""
    if not line.startswith(b"data:"):
        return line
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return line
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return line
    if not isinstance(data, dict):
        return line
    if not _remap_reasoning_in_completion_payload(data):
        return line
    return b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n"


_AGENT_CONTINUE_NUDGE = (
    "\n\n[Harness] Multi-step tasks: keep using tools until every requested step is done. "
    "Do not stop after 1 of N. Prefer tool_calls over plans/status when work remains. "
    "Do not emit Goal/Progress/Next Steps templates."
)

_EMPTY_TOOL_NUDGE = (
    "\n\n[Harness] EMPTY TOOL RESULT: the latest tool output was empty or useless. "
    "Do NOT write a revised plan, Goal/Progress block, or status essay. "
    "Your next action MUST be a local workspace tool (bash ls/find, glob, grep, or read) "
    "on a real project path. Prefer paths already in this chat or under the workspace root. "
    "Do not curl|grep remote HTML unless the user only asked for web docs. "
    "If a path FileNotFound, list its parent directory next — do not invent a new research plan."
)

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
    }
)


def _is_empty_tool_content(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if low in _EMPTY_TOOL_EXACT:
        return True
    # Kilo sometimes returns only a short empty-marker line (exact-ish)
    if len(t) <= 40 and low in _EMPTY_TOOL_EXACT | {
        "no matches",
        "no matches found",
        "(no matches)",
    }:
        return True
    # Whitespace / punctuation only
    if not any(ch.isalnum() for ch in t):
        return True
    return False


def _recent_empty_tool_streak(messages: list[dict] | None) -> int:
    """Count consecutive empty tool results at the tail (after last user/assistant text)."""
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
            # Skip pure assistant tool-call shells; stop if we already counted tools
            if saw_tool:
                break
            # If assistant has substantial content and no trailing tools, streak is 0
            if not msg.get("tool_calls") and not _is_empty_tool_content(_message_text(msg)):
                break
            continue
        if role == "user":
            break
        # system / other
        if saw_tool:
            break
    return empty


def _nudge_agent_multi_step(messages: list[dict]) -> None:
    """Append a short multi-step continue rule to the first system message."""
    marker = "[Harness] Multi-step tasks:"
    for msg in messages:
        if msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if marker in text:
            return
        _set_message_text(msg, text + _AGENT_CONTINUE_NUDGE)
        return
    messages.insert(0, {"role": "system", "content": _AGENT_CONTINUE_NUDGE.strip()})


def _nudge_empty_tool_recovery(messages: list[dict]) -> bool:
    """If last tool result(s) empty, inject a local-tool-only recovery system line."""
    streak = _recent_empty_tool_streak(messages)
    if streak < 1:
        return False
    marker = "[Harness] EMPTY TOOL RESULT:"
    # Prefer appending to last system message; avoid duplicating
    for msg in messages:
        if msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if marker in text:
            # refresh: keep single copy
            return True
        _set_message_text(msg, text + _EMPTY_TOOL_NUDGE)
        log.info("[agent] empty-tool recovery nudge (streak=%d)", streak)
        return True
    messages.insert(0, {"role": "system", "content": _EMPTY_TOOL_NUDGE.strip()})
    log.info("[agent] empty-tool recovery nudge (streak=%d, new system)", streak)
    return True


def _tool_schema_names(body: dict) -> list[str]:
    names: list[str] = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _tool_choice_repr(tool_choice: Any) -> str:
    if tool_choice is None:
        return "unset"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return json.dumps(tool_choice, separators=(",", ":"), ensure_ascii=False)[:80]
    return type(tool_choice).__name__


def _message_role_counts(messages: Any) -> str:
    if not isinstance(messages, list):
        return "msgs=0"
    counts: dict[str, int] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "?")
        counts[role] = counts.get(role, 0) + 1
    if not counts:
        return "msgs=0"
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return "msgs=" + ",".join(parts)


def _harness_log(msg: str, *args: Any) -> None:
    if HARNESS_LOG:
        log.info(msg, *args)


def _prepare_body(body: dict) -> dict[str, Any]:
    """Mutate body for agent safety. Returns a small trace dict (no bodies)."""
    trace: dict[str, Any] = {
        "compaction": False,
        "tools_in": len(body.get("tools") or []) if isinstance(body.get("tools"), list) else 0,
        "tool_names_in": _tool_schema_names(body),
        "tool_choice_in": _tool_choice_repr(body.get("tool_choice")),
        "truncated_tool_msgs": 0,
        "nudged_multi_step": False,
        "empty_tool_recovery": False,
        "empty_tool_streak": 0,
        "thinking_off": True,
    }

    # Always disable thinking for Kilo — empty content breaks the agent loop.
    _disable_thinking(body)

    messages = body.get("messages")
    if isinstance(messages, list):
        truncated = _truncate_tool_messages(messages)
        if truncated:
            trace["truncated_tool_msgs"] = truncated
            log.info("[truncate] shortened %d tool message(s)", truncated)

    if _looks_like_compaction(body):
        body.pop("tools", None)
        body["tool_choice"] = "none"
        if isinstance(messages, list):
            _nudge_compaction_system(messages)
            flat = _flatten_tool_calls_in_history(messages)
            if flat:
                log.info("[compaction] flattened tool_calls on %d message(s)", flat)
        _cap_compaction_tokens(body)
        body.setdefault("temperature", 0.2)
        trace["compaction"] = True
        trace["tools_out"] = 0
        trace["tool_choice_out"] = "none"
        log.info("[mode] compaction → tools stripped, short max_tokens")
        return trace

    # Agentic turns: keep tools; multi-step + empty-tool recovery nudges; temp floor.
    if _has_tools(body) and isinstance(messages, list):
        _nudge_agent_multi_step(messages)
        trace["nudged_multi_step"] = True
        streak = _recent_empty_tool_streak(messages)
        trace["empty_tool_streak"] = streak
        if streak >= 1 and _nudge_empty_tool_recovery(messages):
            trace["empty_tool_recovery"] = True

    temp = body.get("temperature")
    try:
        t = float(temp) if temp is not None else None
    except (TypeError, ValueError):
        t = None
    if t is not None and t < 0.2 and body.get("tools"):
        body["temperature"] = 0.35
        log.info("[temp] raised agentic temperature %.2f → 0.35", t)

    trace["tools_out"] = len(body.get("tools") or []) if isinstance(body.get("tools"), list) else 0
    trace["tool_choice_out"] = _tool_choice_repr(body.get("tool_choice"))
    return trace


def _log_request_harness(body: dict, trace: dict[str, Any], *, stream: bool) -> None:
    mt = body.get("max_tokens")
    if mt is None:
        mt = body.get("max_completion_tokens")
    names = trace.get("tool_names_in") or []
    names_s = ",".join(names[:12])
    if len(names) > 12:
        names_s += f",+{len(names) - 12}"
    _harness_log(
        "[harness] req compaction=%s stream=%s %s tools_in=%s tools_out=%s "
        "tool_choice_in=%s tool_choice_out=%s tool_names=[%s] max_tokens=%s "
        "multi_step_nudge=%s empty_tool_recovery=%s empty_tool_streak=%s "
        "truncate_tools=%s model=%s",
        trace.get("compaction"),
        stream,
        _message_role_counts(body.get("messages")),
        trace.get("tools_in"),
        trace.get("tools_out"),
        trace.get("tool_choice_in"),
        trace.get("tool_choice_out"),
        names_s,
        mt,
        trace.get("nudged_multi_step"),
        trace.get("empty_tool_recovery"),
        trace.get("empty_tool_streak"),
        trace.get("truncated_tool_msgs"),
        body.get("model"),
    )


def _tool_call_names_from_message(msg: dict) -> list[str]:
    names: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(str(name))
        elif tc.get("name"):
            names.append(str(tc["name"]))
    return names


def _summarize_completion_json(data: dict) -> str:
    """One-line response summary; no content text."""
    finish = None
    tool_names: list[str] = []
    content_len = 0
    has_reasoning = False
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") is not None:
            finish = choice.get("finish_reason")
        msg = choice.get("message")
        if isinstance(msg, dict):
            tool_names.extend(_tool_call_names_from_message(msg))
            c = msg.get("content")
            if isinstance(c, str):
                content_len += len(c)
            if msg.get("reasoning") or msg.get("reasoning_content"):
                has_reasoning = True
        delta = choice.get("delta")
        if isinstance(delta, dict):
            tool_names.extend(_tool_call_names_from_message(delta))
            # partial tool call deltas (name in function)
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if isinstance(fn, dict) and fn.get("name"):
                    n = str(fn["name"])
                    if n not in tool_names:
                        tool_names.append(n)
            c = delta.get("content")
            if isinstance(c, str):
                content_len += len(c)
            if delta.get("reasoning") or delta.get("reasoning_content"):
                has_reasoning = True
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return (
        f"finish={finish!r} tool_calls=[{','.join(tool_names)}] "
        f"content_chars={content_len} has_reasoning_field={has_reasoning} "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )


class StreamHarnessStats:
    """Accumulate finish_reason / tool names from SSE without storing content."""

    def __init__(self) -> None:
        self.finish: Any = None
        self.tool_names: list[str] = []
        self.content_chars = 0
        self.reasoning_chars = 0
        self.chunks = 0
        self.remap_events = 0

    def observe_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        self.chunks += 1
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                self.finish = choice.get("finish_reason")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            for name in _tool_call_names_from_message(delta):
                if name not in self.tool_names:
                    self.tool_names.append(name)
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if isinstance(fn, dict) and fn.get("name"):
                    n = str(fn["name"])
                    if n not in self.tool_names:
                        self.tool_names.append(n)
            c = delta.get("content")
            if isinstance(c, str):
                self.content_chars += len(c)
            r = delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(r, str):
                self.reasoning_chars += len(r)

    def summary(self) -> str:
        return (
            f"finish={self.finish!r} tool_calls=[{','.join(self.tool_names)}] "
            f"content_chars={self.content_chars} reasoning_chars={self.reasoning_chars} "
            f"sse_chunks={self.chunks}"
        )


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
            self._send_json(200, {"ok": True, "upstream": self.state.upstream})
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
        is_chat = False
        if body and "application/json" in (self.headers.get("Content-Type") or ""):
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                stream = bool(data.get("stream"))
                if path.rstrip("/").endswith("/chat/completions") or path.rstrip(
                    "/"
                ).endswith("/messages"):
                    is_chat = True
                    trace = _prepare_body(data)
                    _log_request_harness(data, trace, stream=stream)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"

        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{self.state.host}:{self.state.port}"
        headers["Connection"] = "close"

        conn = _connect(self.state)
        try:
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
            # Header names are case-insensitive; mlx may send Content-type vs Content-Type.
            resp_headers = {k: v for k, v in resp.getheaders()}
            resp_headers_l = {k.lower(): v for k, v in resp_headers.items()}
            content_type = (resp_headers_l.get("content-type") or "").lower()

            self.send_response(resp.status)
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

            if stream or "text/event-stream" in content_type:
                self.end_headers()
                # Line-buffer SSE so we can remap reasoning → content per event.
                buf = b""
                stats = StreamHarnessStats() if is_chat else None
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        raw_line = line + b"\n"
                        out = _rewrite_sse_chunk(raw_line)
                        if stats is not None:
                            stats.observe_line(out)
                        self.wfile.write(out)
                        self.wfile.flush()
                if buf:
                    out = _rewrite_sse_chunk(buf)
                    if stats is not None:
                        stats.observe_line(out)
                    self.wfile.write(out)
                    self.wfile.flush()
                if stats is not None:
                    _harness_log("[harness] resp stream status=%s %s", resp.status, stats.summary())
            else:
                payload = resp.read()
                if "application/json" in content_type or is_chat:
                    try:
                        data = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        data = None
                    if isinstance(data, dict):
                        if _remap_reasoning_in_completion_payload(data):
                            log.info(
                                "[think] remapped reasoning → content in non-stream response"
                            )
                            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                        if is_chat:
                            _harness_log(
                                "[harness] resp json status=%s %s",
                                resp.status,
                                _summarize_completion_json(data),
                            )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except Exception as exc:
            log.exception("upstream error: %s", exc)
            _harness_log("[harness] resp error %s", exc)
            try:
                self._send_json(502, {"error": {"message": str(exc), "type": "proxy_error"}})
            except Exception:
                pass
        finally:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    global HARNESS_LOG
    p = argparse.ArgumentParser(description="Gemma4 AtomicChat ↔ Kilo harness proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--upstream",
        default="http://127.0.0.1:8090",
        help="mlx_lm/mlx_vlm base URL (no trailing /v1)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (still no full message bodies in harness traces)",
    )
    p.add_argument(
        "--harness-log",
        action="store_true",
        default=True,
        help="One-line req/resp harness traces (default: on)",
    )
    p.add_argument(
        "--no-harness-log",
        action="store_true",
        help="Disable [harness] one-line traces",
    )
    args = p.parse_args(argv)
    HARNESS_LOG = not args.no_harness_log

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state = ProxyState(args.upstream)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(
        "gemma4 kilo proxy on http://%s:%d → %s "
        "(thinking OFF + compaction tool-strip ON; harness_log=%s)",
        args.host,
        args.port,
        state.upstream,
        HARNESS_LOG,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
