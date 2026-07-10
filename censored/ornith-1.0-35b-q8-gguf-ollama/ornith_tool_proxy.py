#!/usr/bin/env python3
"""OpenAI-compatible tool-call proxy for the Ornith Ollama harness.

Repairs malformed tool calls that local agentic models often emit when used with
OpenCode, Kilo Code, or Grok:

  AskQuestion:
    - ``questions`` passed as a JSON-encoded string instead of an array
    - wrong field names (``header``, ``question``, ``description``, ``multiple``)
    - root argument is a bare array instead of ``{"questions": [...]}``

  TodoWrite / todowrite:
    - ``todos`` passed as a JSON-encoded string instead of an array
    - wrong field names (``task``, ``description``, ``text``, ``state``)
    - root argument is a bare array instead of ``{"todos": [...]}``
    - tool name aliases (``todowrite``, ``TodoWrite``, ``update_todo_list``)

  Write / StrReplace (and other tools):
    - Cline/Roo names (``write_to_file``, ``replace_in_file``) remapped to the
      client's actual tool names (``Write``, ``StrReplace``, ``write``, ``edit``)
    - argument keys remapped (``path`` → ``filePath``, ``old_str`` → ``old_string``)
    - fuzzy ``old_string`` / ``oldString`` repair when the model's edit block doesn't
      match exactly (indent drift, flexible line windows, suffix-biased deletion)
    - edit tool streams buffered briefly for fuzzy repair; Write streams pass through
    - long tool-result messages truncated before upstream to reduce thinking time

  Kilo compaction / summary:
    - Detects Kilo ``agent=compaction``-style requests (``tool_choice: none`` first; prompt
      heuristics only on system + latest user message to avoid history false positives)
    - Strips tools upstream so the model emits plain text instead of hallucinated tool calls
    - Converts any residual ``tool_calls`` in the response to ``content`` (Kilo rejects tool
      calls during summary generation with "Tool call not allowed while generating summary")

  Reliability / token output:
    - Content streams buffered when text→tool recovery is active (no dual SSE completions)
    - Upstream stream errors always close with finish_reason + ``[DONE]``
    - ``tool_choice=required`` only on agentic user turns (plain Q&A stays text)
    - Honors client ``think`` / ``enable_thinking``; defaults ``max_tokens`` when omitted
    - Defaults ``reasoning_effort=none`` so Ollama /v1 actually disables thinking (``think:false``
      alone is ignored on the OpenAI-compatible endpoint — otherwise Ornith fills ``reasoning``
      until max_tokens and returns empty ``content`` with finish_reason=length)
    - Fuzzy edit repair runs in a worker thread with file/size caps
    - Large tool results keep head + tail rather than head-only

Clients connect to this proxy; the proxy forwards to Ollama upstream.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("ornith_tool_proxy")

ASK_TOOL_NAMES = frozenset({
    "AskQuestion",
    "ask_question",
    "ask_user_question",
    "question",
})

TODO_TOOL_NAMES = frozenset({
    "TodoWrite",
    "todowrite",
    "todo_write",
    "todoWrite",
    "update_todo_list",
    "updateTodoList",
})

_VALID_TODO_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
_VALID_TODO_PRIORITIES = frozenset({"high", "medium", "low"})

_DEFAULT_TODO_INFO: dict[str, Any] = {
    "client_todo_name": None,
    "todo_item_fields": frozenset({"id", "content", "status"}),
    "has_merge": True,
}

# Kilo/Grok clients close SSE connections after ~60–90s of silence.
_KEEPALIVE_INTERVAL_S = 15.0
_KEEPALIVE_LINE = b": keepalive\n\n"

# Fuzzy old_string repair — SequenceMatcher ratio threshold (0–1).
_FUZZY_THRESHOLD = 0.85
_FUZZY_THRESHOLD_DELETE = 0.80
_FUZZY_WINDOW_SLACK = 4
# Keep fuzzy repair off the hot path for huge files (event-loop safety).
_FUZZY_MAX_FILE_BYTES = 2_000_000
_FUZZY_MAX_FILE_LINES = 20_000
_FUZZY_MAX_OLD_LINES = 200

# Cap tool-result size forwarded to the model (speeds up long read/grep turns).
# Keep head + tail so stack traces and log endings survive truncation.
_TOOL_RESULT_MAX_LINES = 300
_TOOL_RESULT_HEAD_LINES = 150
_TOOL_RESULT_TAIL_LINES = 150
_TOOL_RESULT_TRUNCATION_MSG = (
    "\n... [proxy: truncated to first {head} + last {tail} of {total} lines "
    "({skipped} hidden to save context). Ask for specific sections if needed.] ..."
)

# When the client omits max_tokens, avoid Ollama short defaults cutting streams.
_DEFAULT_MAX_TOKENS = 32_768

# User turns that likely need tools (force tool_choice=required only then).
# Avoid bare words like "add", "change", "update", "test" — they fire on Q&A.
_AGENTIC_TURN_RE = re.compile(
    r"(?:"
    r"\b(?:implement|refactor|debug|patch|apply|"
    r"write|edit|create|delete|remove|rename|move|"
    r"execute|install|build|commit|fix)\b"
    r"|"
    # Narrow forms of previously too-broad verbs
    r"\b(?:add|change|modify|update)\s+(?:a\s+|the\s+|an\s+)?"
    r"(?:file|function|class|method|test|script|module|code|bug|issue|"
    r"endpoint|route|config|handler|component|tool)\b"
    r"|"
    r"\b(?:write|create|generate)\s+(?:a\s+|the\s+|an\s+)?"
    r"(?:file|function|class|script|test|module|code|patch|tool)\b"
    r"|"
    r"\brun\s+(?:the\s+)?(?:tests?|build|script|command|server|app)\b"
    r"|"
    r"\b(?:unit|integration)\s+tests?\b"
    r"|"
    r"\btests?\s+(?:for|failing|passing)\b"
    r")",
    re.I,
)

# Gemma 4 was trained on Cline/Roo traces. Clients use different tool vocabularies.
_HALLUCINATED_WRITE_NAMES = frozenset({
    "write_to_file", "writeToFile", "write_file", "create_file",
    "createFile", "new_file", "newFile",
})
_HALLUCINATED_EDIT_NAMES = frozenset({
    "replace_in_file", "replaceInFile", "edit_file", "editFile",
    "apply_diff", "applyDiff", "str_replace_editor", "str_replace",
})
_PATH_FIELD_VARIANTS = ("filePath", "file_path", "path", "target_file", "filepath")
_CONTENT_FIELD_VARIANTS = ("content", "fileContent", "text", "body", "newContent")
_OLD_FIELD_VARIANTS = ("old_string", "oldString", "old_str")
_NEW_FIELD_VARIANTS = ("new_string", "newString", "new_str")

_DEFAULT_WRITERS: dict[str, Any] = {
    "write_name": "Write",
    "write_path_field": "path",
    "write_content_field": "content",
    "edit_name": "StrReplace",
    "edit_path_field": "path",
    "edit_old_field": "old_string",
    "edit_new_field": "new_string",
    "write_available": False,
    "edit_available": False,
    "tool_names": frozenset(),
}


def _get_message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


_COMPACTION_HINTS = (
    re.compile(
        r"summariz(?:e|ing)\b.*\b(?:conversation|session|history|context|chat|messages?)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(?:conversation|session|chat) (?:history|transcript)\b.*\bsummar",
        re.I | re.S,
    ),
    re.compile(
        r"\bcompact(?:ion|ing)?\b.*\b(?:context|session|conversation|history)\b",
        re.I | re.S,
    ),
    re.compile(r"\bconcise summary\b", re.I),
    re.compile(r"\bpreserve key information\b", re.I),
    re.compile(r"\bgenerate (?:a )?(?:concise )?summary\b", re.I),
)

_COMPACTION_NUDGE = (
    "\n\nRespond with plain text only. Do not call tools or emit tool_calls."
)


def _tool_choice_disallows_tools(tool_choice: Any) -> bool:
    if tool_choice == "none":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


def _message_blob(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    return "\n".join(_get_message_text(msg) for msg in messages)


def _compaction_probe_blob(messages: list[dict] | None) -> str:
    """Text used for compaction heuristics — system + latest user only.

    Scanning the full transcript false-positives when older turns mention
    "summarize the conversation".
    """
    if not messages:
        return ""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            text = _get_message_text(msg)
            if text:
                parts.append(text)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _get_message_text(msg)
            if text:
                parts.append(text)
            break
    return "\n".join(parts)


def _is_compaction_request(body: dict) -> bool:
    """True when Kilo/Grok is generating a session summary (no tools allowed)."""
    # Structured signal is authoritative.
    if _tool_choice_disallows_tools(body.get("tool_choice")):
        return True
    blob = _compaction_probe_blob(body.get("messages"))
    return bool(blob) and any(pattern.search(blob) for pattern in _COMPACTION_HINTS)


def _last_message_role(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    return messages[-1].get("role")


def _last_user_text(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _get_message_text(msg)
    return ""


def _user_turn_looks_agentic(messages: list[dict] | None) -> bool:
    """Heuristic: only force tools when the user likely wants code actions."""
    text = _last_user_text(messages).strip()
    if not text:
        return False
    if _AGENTIC_TURN_RE.search(text):
        return True
    if re.search(r"```", text):
        return True
    if re.search(
        r"\b[\w./~-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|swift|c|h|cpp|hpp|sh|md|json|yaml|yml|toml)\b",
        text,
        re.I,
    ):
        return True
    return False


def _prepare_tool_choice(body: dict, messages: list[dict] | None) -> None:
    """Ornith often answers with reasoning-only text when tool_choice is auto.

    Force tools only on agentic user turns so plain Q&A can still stream text.
    Explicit client tool_choice (other than auto) is always respected.
    """
    if not body.get("tools"):
        return
    if _tool_choice_disallows_tools(body.get("tool_choice")):
        return
    existing = body.get("tool_choice")
    if existing is not None and existing != "auto":
        return
    if _last_message_role(messages) == "user" and _user_turn_looks_agentic(messages):
        body["tool_choice"] = "required"
        log.info("[tools] agentic user turn — forcing tool_choice=required")


def _resolve_think_flag(body: dict) -> bool | None:
    """Map client think / enable_thinking into Ollama's ``think`` field.

    Returns None when the client did not express a preference (leave unset).
    """
    if "think" in body:
        return bool(body["think"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    options = body.get("options")
    if isinstance(options, dict) and "enable_thinking" in options:
        return bool(options["enable_thinking"])
    return None


_EFFORT_OFF = frozenset({"none", "off", "false", "0", "minimal"})
_EFFORT_ON = frozenset({"low", "medium", "high", "max", "true", "1"})


def _prepare_think_for_ollama(body: dict) -> None:
    """Disable or enable thinking in a way Ollama's OpenAI /v1 endpoint honors.

    Background: Ornith (qwen35moe) is a thinking model. On ``/api/chat``, ``think:false``
    works. On ``/v1/chat/completions``, ``think:false`` is ignored and the model emits
    only ``reasoning`` until ``max_tokens``, then ``finish_reason=length`` with empty
    ``content`` — Kilo/OpenCode show an abrupt end (thinking dump, no answer/tools).

    ``reasoning_effort: "none"`` is honored on /v1 and restores normal content.
    """
    effort_raw = body.get("reasoning_effort")
    if effort_raw is not None:
        effort = str(effort_raw).strip().lower()
        if effort in _EFFORT_OFF:
            body["reasoning_effort"] = "none"
            body["think"] = False
            log.info("[think] reasoning_effort=%r → disabled", effort_raw)
            return
        if effort in _EFFORT_ON or effort:
            body["think"] = True
            log.info("[think] reasoning_effort=%r → enabled", effort_raw)
            return

    think = _resolve_think_flag(body)
    if think is True:
        body["think"] = True
        # Leave effort unset so Ollama uses its default thinking level.
        body.pop("reasoning_effort", None)
        log.info("[think] client preference → enabled")
        return

    # think is False, or unset: default OFF for reliable agentic content/tool calls.
    body["think"] = False
    body["reasoning_effort"] = "none"
    if think is False:
        log.info("[think] client preference → disabled (reasoning_effort=none)")
    else:
        log.info("[think] defaulting to disabled (reasoning_effort=none)")


def _ensure_max_tokens(body: dict) -> None:
    """Guarantee a high enough generation budget for long tool args / patches."""
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        body["max_tokens"] = _DEFAULT_MAX_TOKENS
        log.info("[tokens] defaulting max_tokens=%d", _DEFAULT_MAX_TOKENS)


def _prepare_upstream_request(body: dict, messages: list[dict] | None) -> None:
    """Tune upstream requests for agentic clients talking to Ollama."""
    _prepare_tool_choice(body, messages)
    _ensure_max_tokens(body)
    _prepare_think_for_ollama(body)


def _prepare_compaction_request(body: dict) -> None:
    """Force text-only upstream generation for compaction turns."""
    body.pop("tools", None)
    body["tool_choice"] = "none"
    body["think"] = False
    body["reasoning_effort"] = "none"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if _COMPACTION_NUDGE.strip() not in content:
                msg["content"] = content + _COMPACTION_NUDGE
            return
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if _COMPACTION_NUDGE.strip() not in text:
                        block["text"] = text + _COMPACTION_NUDGE
                    return
    messages.insert(0, {"role": "system", "content": _COMPACTION_NUDGE.strip()})


def _tool_call_to_plain_text(tc: dict) -> str:
    func = tc.get("function") or {}
    name = func.get("name") or "tool"
    args = func.get("arguments") or ""
    try:
        parsed = json.loads(args) if args else {}
        if isinstance(parsed, dict) and parsed:
            args = json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass
    return f"[{name}]\n{args}".strip()


_HEREDOC_CAT_RE = re.compile(
    r"cat\s+>\s*(\S+)\s+<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\2\b",
    re.DOTALL,
)
_MALFORMED_TOOL_TAG_RE = re.compile(
    r"<(?:tool_call|invoke|function_call)[^>]*>\s*(\w+)",
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(
    r"\b([\w./~-]+\.(?:sh|py|js|ts|tsx|jsx|json|md|txt|yaml|yml|toml|rs|go|rb|java|kt|swift|c|h|cpp|hpp))\b",
    re.IGNORECASE,
)
_SHELL_LINE_PREFIXES = (
    "#!/",
    "cat ",
    "chmod ",
    "mkdir ",
    "echo ",
    "printf ",
    "touch ",
    "cp ",
    "mv ",
    "rm ",
    "ls ",
    "cd ",
    "git ",
    "npm ",
    "npx ",
    "python",
    "pip ",
    "cargo ",
    "make ",
    "curl ",
    "sed ",
    "awk ",
    "export ",
    "source ",
    "./",
)


def _bash_tool_available(writers: dict[str, Any]) -> bool:
    tool_names = writers.get("tool_names") or frozenset()
    return "bash" in tool_names or "Bash" in tool_names or "shell" in tool_names


def _write_tool_available(writers: dict[str, Any]) -> bool:
    return bool(writers.get("write_available"))


def _looks_like_shell_block(content: str) -> bool:
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    hits = 0
    for ln in lines:
        if ln.startswith(_SHELL_LINE_PREFIXES) or "<<" in ln or ln in {"EOF", "DONE"}:
            hits += 1
    return hits >= max(1, len(lines) // 2)


def _infer_path_from_messages(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if msg.get("role") not in ("user", "assistant"):
            continue
        text = _get_message_text(msg)
        if not text:
            continue
        match = _FILENAME_RE.search(text)
        if match:
            return match.group(1)
        match = re.search(
            r"(?:called|named?|create|write)\s+[`'\"]?([\w./~-]+)[`'\"]?",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return None


def _infer_file_content_from_messages(messages: list[dict] | None, path: str) -> str | None:
    if not messages:
        return None
    blob = _message_blob(messages)
    if re.search(r"echo\s+[`'\"]hello world[`'\"]", blob, re.IGNORECASE):
        if path.endswith(".sh"):
            return '#!/bin/bash\necho "hello world"\n'
        return 'echo "hello world"\n'
    if re.search(r"hello world", blob, re.IGNORECASE) and path.endswith(".sh"):
        return '#!/bin/bash\necho "hello world"\n'
    return None


def _make_tool_call(name: str, args: dict[str, Any]) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _content_is_mostly_shell(content: str, *, min_ratio: float = 0.85) -> bool:
    """True when almost every non-empty line looks like a shell command (not prose)."""
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    hits = 0
    for ln in lines:
        if ln.startswith(_SHELL_LINE_PREFIXES) or "<<" in ln or ln in {"EOF", "DONE"}:
            hits += 1
    return hits >= max(1, int(len(lines) * min_ratio + 0.999))


def _extract_tool_calls_from_text(
    content: str,
    writers: dict[str, Any],
    messages: list[dict] | None = None,
    *,
    aggressive: bool = False,
) -> list[dict]:
    """Recover structured tool_calls when the model prints shell/XML instead.

    By default only high-confidence shapes (heredoc cat, malformed tool tags) are
    recovered. Free-form shell blocks require ``aggressive=True`` (tool_choice was
    forced required) so explain-the-script answers stay plain text.
    """
    if not content or not content.strip():
        return []

    text = content.strip()
    tool_calls: list[dict] = []

    if _write_tool_available(writers):
        heredoc = _HEREDOC_CAT_RE.search(text)
        if heredoc:
            path, _marker, body = heredoc.group(1), heredoc.group(2), heredoc.group(3)
            write_name = writers.get("write_name", _DEFAULT_WRITERS["write_name"])
            path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
            content_field = writers.get(
                "write_content_field", _DEFAULT_WRITERS["write_content_field"]
            )
            tool_calls.append(
                _make_tool_call(write_name, {path_field: path, content_field: body.rstrip() + "\n"})
            )
            remainder = text[heredoc.end() :].strip()
            if remainder and _bash_tool_available(writers) and _looks_like_shell_block(remainder):
                tool_calls.append(_make_tool_call("bash", {"command": remainder}))
            if tool_calls:
                log.info("[text-repair] heredoc cat → %s (%s)", write_name, path)
                return tool_calls

    malformed = _MALFORMED_TOOL_TAG_RE.search(text)
    if malformed and _write_tool_available(writers):
        hinted = malformed.group(1).lower()
        if hinted in {"write", "edit", "strreplace", "apply_patch"}:
            path = _infer_path_from_messages(messages)
            if path:
                body = _infer_file_content_from_messages(messages, path)
                if body is not None:
                    write_name = writers.get("write_name", _DEFAULT_WRITERS["write_name"])
                    path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
                    content_field = writers.get(
                        "write_content_field", _DEFAULT_WRITERS["write_content_field"]
                    )
                    log.info("[text-repair] malformed <%s> tag → %s (%s)", hinted, write_name, path)
                    return [
                        _make_tool_call(
                            write_name,
                            {path_field: path, content_field: body},
                        )
                    ]

    if _bash_tool_available(writers) and _looks_like_shell_block(text):
        # Free-form prose + a few shell lines stays text unless forced/mostly shell.
        if aggressive or _content_is_mostly_shell(text):
            log.info("[text-repair] shell block → bash (aggressive=%s)", aggressive)
            return [_make_tool_call("bash", {"command": text})]

    return []


def _repair_text_tool_calls_in_message(
    msg: dict,
    writers: dict[str, Any],
    messages: list[dict] | None = None,
    *,
    aggressive: bool = False,
) -> bool:
    if msg.get("tool_calls"):
        return False
    content = _get_message_text(msg)
    recovered = _extract_tool_calls_from_text(
        content, writers, messages, aggressive=aggressive
    )
    if not recovered:
        return False
    for tc in recovered:
        func = tc.get("function", {})
        func["name"], func["arguments"] = repair_tool_call(
            func.get("name", ""),
            func.get("arguments", "{}"),
            writers,
            messages,
        )
    msg["tool_calls"] = recovered
    msg["content"] = ""
    return True


def _repair_text_tool_calls_in_response(
    data: dict,
    writers: dict[str, Any],
    messages: list[dict] | None = None,
    *,
    aggressive: bool = False,
) -> bool:
    changed = False
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        if _repair_text_tool_calls_in_message(
            msg, writers, messages, aggressive=aggressive
        ):
            choice["finish_reason"] = "tool_calls"
            changed = True
    return changed


def _event_has_content_delta(ev: dict) -> bool:
    for choice in ev.get("choices", []):
        piece = choice.get("delta", {}).get("content")
        if isinstance(piece, str) and piece:
            return True
    return False


def _stream_content_from_events(events: list[dict]) -> str:
    parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                parts.append(piece)
            msg = choice.get("message", {})
            piece = msg.get("content")
            if isinstance(piece, str) and piece:
                parts.append(piece)
    return "".join(parts)


def _stream_reasoning_from_events(events: list[dict]) -> str:
    parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            for key in ("reasoning_content", "reasoning"):
                piece = delta.get(key)
                if isinstance(piece, str) and piece:
                    parts.append(piece)
            msg = choice.get("message", {})
            for key in ("reasoning_content", "reasoning"):
                piece = msg.get(key)
                if isinstance(piece, str) and piece:
                    parts.append(piece)
    return "".join(parts)


def _stream_meta(events: list[dict]) -> tuple[str, str]:
    response_id = (
        events[0].get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        if events
        else f"chatcmpl-{uuid.uuid4().hex[:12]}"
    )
    model_name = events[0].get("model", "") if events else ""
    return response_id, model_name


def _emit_content_stream_chunks(
    events: list[dict],
    *,
    finish_reason: str = "stop",
) -> list[bytes]:
    """Replay buffered content (and reasoning) as a clean single SSE completion."""
    response_id, model_name = _stream_meta(events)
    content = _stream_content_from_events(events)
    reasoning = _stream_reasoning_from_events(events)
    chunks: list[bytes] = []
    if content or reasoning:
        delta: dict[str, Any] = {"role": "assistant"}
        if content:
            delta["content"] = content
        if reasoning:
            delta["reasoning_content"] = reasoning
        chunks.append(_encode_sse_event({
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        }))
    chunks.append(_encode_sse_event({
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def _emit_stream_error(response_id: str = "", model_name: str = "", message: str = "") -> list[bytes]:
    """Close an SSE stream cleanly after an upstream failure."""
    rid = response_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    err_text = message or "upstream stream error"
    return [
        _encode_sse_event({
            "id": rid,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": f"\n\n[proxy error: {err_text}]"},
                "finish_reason": None,
            }],
        }),
        _encode_sse_event({
            "id": rid,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }),
        b"data: [DONE]\n\n",
    ]


def _emit_text_recovered_tool_stream(
    events: list[dict],
    writers: dict[str, Any],
    messages: list[dict] | None = None,
    *,
    aggressive: bool = False,
) -> list[bytes] | None:
    content = _stream_content_from_events(events).strip()
    recovered = _extract_tool_calls_from_text(
        content, writers, messages, aggressive=aggressive
    )
    if not recovered:
        return None

    response_id, model_name = _stream_meta(events)

    for tc in recovered:
        func = tc.get("function", {})
        func["name"], func["arguments"] = repair_tool_call(
            func.get("name", ""),
            func.get("arguments", "{}"),
            writers,
            messages,
        )

    # Single coherent stream: tool_calls only (no prior live content was sent).
    delta_event = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(recovered)
                ],
            },
            "finish_reason": None,
        }],
    }
    finish_event = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return [
        f"data: {json.dumps(delta_event)}\n\n".encode(),
        f"data: {json.dumps(finish_event)}\n\n".encode(),
        b"data: [DONE]\n\n",
    ]


def _flatten_tool_calls_in_message(msg: dict) -> None:
    tool_calls = msg.pop("tool_calls", None)
    if not tool_calls:
        return
    pieces = [
        _tool_call_to_plain_text(tc)
        for tc in tool_calls
        if isinstance(tc, dict)
    ]
    extra = "\n\n".join(piece for piece in pieces if piece)
    content = msg.get("content") or ""
    if extra:
        msg["content"] = f"{content}\n\n{extra}".strip() if content else extra


def _flatten_tool_calls_in_response(data: dict) -> None:
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        _flatten_tool_calls_in_message(msg)
        if choice.get("finish_reason") == "tool_calls":
            choice["finish_reason"] = "stop"


def _tool_delta_to_content_event(ev: dict) -> dict | None:
    """Rewrite a streaming chunk's tool_call deltas as plain content."""
    choices = ev.get("choices") or []
    if not choices:
        return None
    parts: list[str] = []
    for choice in choices:
        for tc in choice.get("delta", {}).get("tool_calls", []) or []:
            text = _tool_call_to_plain_text({
                "function": {
                    "name": (tc.get("function") or {}).get("name", ""),
                    "arguments": (tc.get("function") or {}).get("arguments", ""),
                }
            })
            if text:
                parts.append(text)
    if not parts:
        return None
    patched = json.loads(json.dumps(ev))
    patched["choices"] = [{
        "index": 0,
        "delta": {"content": "\n\n".join(parts)},
        "finish_reason": None,
    }]
    return patched


def _emit_text_only_stream(events: list[dict]) -> list[bytes]:
    """Collapse a buffered stream (content + tool_calls) into text-only SSE."""
    response_id = events[0].get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}") if events else (
        f"chatcmpl-{uuid.uuid4().hex[:12]}"
    )
    model_name = events[0].get("model", "") if events else ""
    content_parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
    tool_calls = _reassemble_tool_calls(events)
    for tc in tool_calls.values():
        content_parts.append(_tool_call_to_plain_text({
            "function": {"name": tc["name"], "arguments": tc["arguments"]},
        }))
    text = "".join(content_parts)
    chunks: list[bytes] = []
    if text:
        chunks.append(_encode_sse_event({
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }],
        }))
    chunks.append(_encode_sse_event({
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


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
        text = _get_message_text(msg)
        if not text:
            continue
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                wd = m.group(1).strip().strip("()").rstrip("/")
                if wd:
                    return wd
    return None


def _resolve_file_path(file_path: str, messages: list[dict] | None) -> str:
    if not file_path:
        return file_path
    if os.path.isabs(file_path) and os.path.isfile(file_path):
        return file_path
    candidates: list[str] = []
    workspace = _extract_workspace_dir(messages)
    if workspace:
        candidates.append(os.path.join(workspace, file_path))
    candidates.append(os.path.join(os.getcwd(), file_path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0] if candidates else file_path


def _normalize_block_lines(lines: list[str]) -> list[str]:
    return [line.rstrip() for line in lines]


def _relative_indent_lines(lines: list[str]) -> list[str]:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return [line.strip() for line in lines]
    base_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    normalized: list[str] = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue
        indent = len(line) - len(line.lstrip())
        dedent = max(0, indent - base_indent)
        normalized.append((" " * dedent) + line.lstrip())
    return normalized


def _block_similarity(file_window: list[str], old_lines: list[str]) -> float:
    file_norm = _normalize_block_lines(file_window)
    old_norm = _normalize_block_lines(old_lines)
    stripped_ratio = difflib.SequenceMatcher(
        None,
        [line.strip() for line in file_norm],
        [line.strip() for line in old_norm],
        autojunk=False,
    ).ratio()
    relative_ratio = difflib.SequenceMatcher(
        None,
        _relative_indent_lines(file_norm),
        _relative_indent_lines(old_norm),
        autojunk=False,
    ).ratio()
    return max(stripped_ratio, relative_ratio)


def _fuzzy_find_in_lines(
    file_lines: list[str],
    old_lines: list[str],
    *,
    prefer_suffix: bool = False,
    threshold: float = _FUZZY_THRESHOLD,
) -> tuple[int, int, float] | None:
    """Return (start, end_exclusive, ratio) for the best fuzzy block match."""
    if not old_lines or not file_lines:
        return None

    target = len(old_lines)
    min_window = max(1, target - _FUZZY_WINDOW_SLACK)
    max_window = target + _FUZZY_WINDOW_SLACK

    if prefer_suffix:
        search_start = max(0, len(file_lines) - (target + 15))
    else:
        search_start = 0

    best_ratio = 0.0
    best_span: tuple[int, int] | None = None

    for start in range(search_start, len(file_lines)):
        for window in range(min_window, max_window + 1):
            end = start + window
            if end > len(file_lines):
                break
            ratio = _block_similarity(file_lines[start:end], old_lines)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, end)

    if best_span is None or best_ratio < threshold:
        return None
    start, end = best_span
    return start, end, best_ratio


def _fuzzy_find(
    file_path: str,
    old_string: str,
    *,
    prefer_suffix: bool = False,
) -> str | None:
    if not os.path.isfile(file_path):
        return None
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    if size > _FUZZY_MAX_FILE_BYTES:
        log.warning(
            "[fuzzy] skip %s — file too large (%d bytes > %d)",
            os.path.basename(file_path),
            size,
            _FUZZY_MAX_FILE_BYTES,
        )
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            file_content = fh.read()
    except OSError:
        return None

    if old_string in file_content:
        return old_string

    file_lines = file_content.splitlines()
    old_lines = old_string.splitlines()
    if not old_lines:
        return None
    if len(old_lines) > _FUZZY_MAX_OLD_LINES:
        log.warning(
            "[fuzzy] skip %s — old_string too long (%d lines > %d)",
            os.path.basename(file_path),
            len(old_lines),
            _FUZZY_MAX_OLD_LINES,
        )
        return None
    if len(file_lines) > _FUZZY_MAX_FILE_LINES:
        log.warning(
            "[fuzzy] skip %s — file too many lines (%d > %d)",
            os.path.basename(file_path),
            len(file_lines),
            _FUZZY_MAX_FILE_LINES,
        )
        return None

    threshold = _FUZZY_THRESHOLD_DELETE if prefer_suffix else _FUZZY_THRESHOLD
    match = _fuzzy_find_in_lines(
        file_lines,
        old_lines,
        prefer_suffix=prefer_suffix,
        threshold=threshold,
    )
    if match is None:
        log.warning(
            "[fuzzy] %s could not repair old_string (%d lines, suffix=%s)",
            os.path.basename(file_path),
            len(old_lines),
            prefer_suffix,
        )
        return None

    start, end, ratio = match
    matched = "\n".join(file_lines[start:end])
    log.info(
        "[fuzzy] %s ratio=%.2f fixed old_string (%d→%d lines, suffix=%s)",
        os.path.basename(file_path),
        ratio,
        len(old_lines),
        end - start,
        prefer_suffix,
    )
    return matched


def _normalize_edit_arg_keys(args: dict[str, Any], writers: dict[str, Any]) -> dict[str, Any]:
    """Rename edit argument keys to the client's schema (e.g. oldString → old_string)."""
    if not writers.get("edit_available"):
        return args

    path_field = writers.get("edit_path_field", _DEFAULT_WRITERS["edit_path_field"])
    old_field = writers.get("edit_old_field", _DEFAULT_WRITERS["edit_old_field"])
    new_field = writers.get("edit_new_field", _DEFAULT_WRITERS["edit_new_field"])

    for src in _PATH_FIELD_VARIANTS:
        if src in args and src != path_field:
            args[path_field] = args.pop(src)
            break

    for src in _OLD_FIELD_VARIANTS:
        if src in args and src != old_field:
            args[old_field] = args.pop(src)
            break

    for src in _NEW_FIELD_VARIANTS:
        if src in args and src != new_field:
            args[new_field] = args.pop(src)
            break

    return args


def _truncate_tool_results(messages: list[dict]) -> list[dict]:
    """Keep head + tail of large tool results so stack traces still reach the model."""
    patched: list[dict] = []
    head_n = _TOOL_RESULT_HEAD_LINES
    tail_n = _TOOL_RESULT_TAIL_LINES
    max_lines = max(_TOOL_RESULT_MAX_LINES, head_n + tail_n)
    for msg in messages:
        if msg.get("role") != "tool":
            patched.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            patched.append(msg)
            continue
        lines = content.splitlines()
        if len(lines) <= max_lines:
            patched.append(msg)
            continue
        # Avoid overlapping head/tail when the budget is tight.
        if head_n + tail_n >= len(lines):
            patched.append(msg)
            continue
        skipped = len(lines) - head_n - tail_n
        middle = _TOOL_RESULT_TRUNCATION_MSG.format(
            head=head_n,
            tail=tail_n,
            total=len(lines),
            skipped=skipped,
        )
        kept = lines[:head_n] + [middle.strip()] + lines[-tail_n:]
        patched.append({**msg, "content": "\n".join(kept)})
        log.info(
            "[truncate] tool result: %d → %d lines (head=%d tail=%d, %d hidden)",
            len(lines),
            head_n + tail_n,
            head_n,
            tail_n,
            skipped,
        )
    return patched


def _is_edit_tool_name(name: str, writers: dict[str, Any]) -> bool:
    edit_name = writers.get("edit_name", _DEFAULT_WRITERS["edit_name"])
    return name == edit_name or name in _HALLUCINATED_EDIT_NAMES


def _slug(text: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:48] or fallback


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def _normalize_option(opt: Any, index: int) -> dict[str, str] | None:
    if not isinstance(opt, dict):
        return None
    label = opt.get("label") or opt.get("description") or opt.get("text") or f"Option {index + 1}"
    opt_id = opt.get("id") or _slug(str(label), f"opt_{index}")
    return {"id": str(opt_id), "label": str(label)}


def _normalize_question(q: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(q, dict):
        return None
    prompt = q.get("prompt") or q.get("question") or q.get("text") or f"Question {index + 1}"
    qid = q.get("id") or _slug(str(prompt), f"q_{index}")
    options_raw = q.get("options") or []
    if isinstance(options_raw, str):
        options_raw = _parse_json_value(options_raw) or []
    options = []
    if isinstance(options_raw, list):
        for i, opt in enumerate(options_raw):
            normalized = _normalize_option(opt, i)
            if normalized:
                options.append(normalized)
    if len(options) < 2:
        return None
    allow_multiple = bool(q.get("allow_multiple", q.get("multiple", False)))
    return {
        "id": str(qid),
        "prompt": str(prompt),
        "options": options,
        "allow_multiple": allow_multiple,
    }


def repair_ask_question_args(args: Any) -> dict[str, Any]:
    """Normalize AskQuestion tool arguments to Kilo/Grok schema."""
    parsed = _parse_json_value(args) if not isinstance(args, dict) else dict(args)

    if isinstance(parsed, list):
        parsed = {"questions": parsed}

    if not isinstance(parsed, dict):
        parsed = {}

    title = parsed.get("title") or parsed.get("header")
    questions_raw = parsed.get("questions")
    if questions_raw is None and "question" in parsed:
        questions_raw = [parsed]
    questions_raw = _parse_json_value(questions_raw)
    if isinstance(questions_raw, dict):
        questions_raw = [questions_raw]

    questions: list[dict[str, Any]] = []
    if isinstance(questions_raw, list):
        for i, q in enumerate(questions_raw):
            normalized = _normalize_question(q, i)
            if normalized:
                questions.append(normalized)
                if not title and isinstance(q, dict):
                    title = q.get("header") or q.get("title")

    if not questions:
        raise ValueError("no valid questions after normalization")

    if not title:
        title = "Questions"

    return {"title": str(title), "questions": questions}


def _normalize_todo_status(status: Any) -> str:
    if not isinstance(status, str):
        return "pending"
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "inprogress": "in_progress",
        "in_prog": "in_progress",
        "wip": "in_progress",
        "done": "completed",
        "complete": "completed",
        "canceled": "cancelled",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _VALID_TODO_STATUSES else "pending"


def _normalize_todo_item(
    item: Any,
    index: int,
    fields: frozenset[str],
) -> dict[str, Any] | None:
    if isinstance(item, str) and item.strip():
        parsed_item = _parse_json_value(item.strip())
        item = parsed_item if isinstance(parsed_item, dict) else {"content": item}
    if not isinstance(item, dict):
        return None

    content = (
        item.get("content")
        or item.get("task")
        or item.get("description")
        or item.get("text")
        or item.get("title")
        or item.get("name")
        or ""
    )
    if not str(content).strip():
        return None

    out: dict[str, Any] = {"content": str(content).strip()}
    if "id" in fields:
        out["id"] = str(item.get("id") or _slug(str(content), f"todo_{index}"))

    out["status"] = _normalize_todo_status(item.get("status") or item.get("state"))

    if "priority" in fields:
        priority = str(item.get("priority", "medium")).strip().lower()
        out["priority"] = priority if priority in _VALID_TODO_PRIORITIES else "medium"

    return out


def repair_todo_write_args(
    args: Any,
    todo_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize TodoWrite/todowrite tool arguments to the client schema."""
    info = todo_info or dict(_DEFAULT_TODO_INFO)
    fields = info.get("todo_item_fields") or _DEFAULT_TODO_INFO["todo_item_fields"]

    parsed = _parse_json_value(args) if not isinstance(args, dict) else dict(args)
    if isinstance(parsed, list):
        parsed = {"todos": parsed}
    if not isinstance(parsed, dict):
        parsed = {}

    todos_raw = parsed.get("todos")
    if todos_raw is None and any(k in parsed for k in ("content", "task", "description", "text")):
        todos_raw = [parsed]
    todos_raw = _parse_json_value(todos_raw)
    if isinstance(todos_raw, dict):
        todos_raw = [todos_raw]

    todos: list[dict[str, Any]] = []
    if isinstance(todos_raw, list):
        for i, item in enumerate(todos_raw):
            normalized = _normalize_todo_item(item, i, fields)
            if normalized:
                todos.append(normalized)

    if not todos:
        raise ValueError("no valid todos after normalization")

    result: dict[str, Any] = {"todos": todos}
    if info.get("has_merge") or "merge" in parsed:
        merge = parsed.get("merge", False)
        if isinstance(merge, str):
            merge = merge.strip().lower() in ("true", "1", "yes")
        result["merge"] = bool(merge)
    return result


def _discover_todo_info(tools: list[dict] | None) -> dict[str, Any]:
    info: dict[str, Any] = dict(_DEFAULT_TODO_INFO)
    if not tools:
        return info

    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name", "")
        if name not in TODO_TOOL_NAMES:
            continue
        info["client_todo_name"] = name
        props = ((fn.get("parameters") or {}).get("properties") or {})
        items = ((props.get("todos") or {}).get("items") or {})
        item_props = (items.get("properties") or {})
        if item_props:
            info["todo_item_fields"] = frozenset(item_props.keys())
        info["has_merge"] = "merge" in props
        break
    return info


def _is_todo_tool_name(name: str, writers: dict[str, Any]) -> bool:
    if name in TODO_TOOL_NAMES:
        return True
    client_name = (writers.get("todo_info") or {}).get("client_todo_name")
    return bool(client_name and name == client_name)


def _remap_todo_tool_name(name: str, writers: dict[str, Any]) -> str:
    if name not in TODO_TOOL_NAMES:
        return name
    client_name = (writers.get("todo_info") or {}).get("client_todo_name")
    if client_name and name != client_name:
        log.info("[remap] %s → %s", name, client_name)
        return client_name
    return name


def _discover_writers(tools: list[dict] | None) -> dict[str, Any]:
    """Identify the client's write/edit tools and parameter field names."""
    if not tools:
        return dict(_DEFAULT_WRITERS)

    info: dict[str, Any] = dict(_DEFAULT_WRITERS)
    info["tool_names"] = frozenset(
        (t.get("function", {}) or {}).get("name", "")
        for t in tools
        if isinstance(t, dict)
    )

    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        lname = name.lower()
        params = ((fn.get("parameters") or {}).get("properties") or {})

        is_write = (
            ("write" in lname and "rewrite" not in lname and "overwrite" not in lname)
            or "create_file" in lname or "createfile" in lname
            or "new_file" in lname or "newfile" in lname
        )
        if is_write and not info["write_available"]:
            has_path = any(k in params for k in _PATH_FIELD_VARIANTS)
            has_content = any(k in params for k in _CONTENT_FIELD_VARIANTS)
            if has_path and has_content:
                info["write_name"] = name
                info["write_path_field"] = next(
                    (k for k in _PATH_FIELD_VARIANTS if k in params),
                    _DEFAULT_WRITERS["write_path_field"],
                )
                info["write_content_field"] = next(
                    (k for k in _CONTENT_FIELD_VARIANTS if k in params),
                    _DEFAULT_WRITERS["write_content_field"],
                )
                info["write_available"] = True

        is_edit = (
            ("edit" in lname and "credit" not in lname)
            or "replace" in lname
            or "apply_diff" in lname or "applydiff" in lname
            or "str_replace" in lname or "strreplace" in lname
        )
        if is_edit and not info["edit_available"]:
            has_path = any(k in params for k in _PATH_FIELD_VARIANTS)
            if has_path:
                info["edit_name"] = name
                info["edit_path_field"] = next(
                    (k for k in _PATH_FIELD_VARIANTS if k in params),
                    _DEFAULT_WRITERS["edit_path_field"],
                )
                info["edit_old_field"] = next(
                    (k for k in _OLD_FIELD_VARIANTS if k in params),
                    _DEFAULT_WRITERS["edit_old_field"],
                )
                info["edit_new_field"] = next(
                    (k for k in _NEW_FIELD_VARIANTS if k in params),
                    _DEFAULT_WRITERS["edit_new_field"],
                )
                info["edit_available"] = True

    info["todo_info"] = _discover_todo_info(tools)
    return info


def _remap_tool_call_name_and_args(
    name: str, args_str: str, writers: dict[str, Any]
) -> tuple[str, str]:
    """Remap Cline/Roo tool names and argument keys to the client's schema."""
    tool_names = writers.get("tool_names") or frozenset()
    target_name: str | None = None
    target_path_field: str | None = None
    target_content_field: str | None = None
    target_old_field: str | None = None
    target_new_field: str | None = None

    if name in _HALLUCINATED_WRITE_NAMES and name not in tool_names:
        target_name = writers.get("write_name", _DEFAULT_WRITERS["write_name"])
        target_path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
        target_content_field = writers.get(
            "write_content_field", _DEFAULT_WRITERS["write_content_field"]
        )
    elif name in _HALLUCINATED_EDIT_NAMES and name not in tool_names:
        target_name = writers.get("edit_name", _DEFAULT_WRITERS["edit_name"])
        target_path_field = writers.get("edit_path_field", _DEFAULT_WRITERS["edit_path_field"])
        target_old_field = writers.get("edit_old_field", _DEFAULT_WRITERS["edit_old_field"])
        target_new_field = writers.get("edit_new_field", _DEFAULT_WRITERS["edit_new_field"])
    else:
        return name, args_str

    if not args_str or not args_str.strip():
        log.info("[remap] %s → %s (streaming name only)", name, target_name)
        return target_name, args_str

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        log.warning("[remap] %s → %s: args JSON parse failed, name only", name, target_name)
        return target_name, _remap_argument_fragment(
            name, args_str, writers, target_name=target_name
        )

    if not isinstance(args, dict):
        return target_name, args_str

    for src in _PATH_FIELD_VARIANTS:
        if src in args and src != target_path_field:
            args[target_path_field] = args.pop(src)
            break

    if target_content_field is not None:
        for src in _CONTENT_FIELD_VARIANTS:
            if src in args and src != target_content_field:
                args[target_content_field] = args.pop(src)
                break

    if target_old_field is not None:
        for src in _OLD_FIELD_VARIANTS:
            if src in args and src != target_old_field:
                args[target_old_field] = args.pop(src)
                break

    if target_new_field is not None:
        for src in _NEW_FIELD_VARIANTS:
            if src in args and src != target_new_field:
                args[target_new_field] = args.pop(src)
                break

    log.info(
        "[remap] %s → %s (%s)",
        name, target_name, ",".join(sorted(args.keys())),
    )
    return target_name, json.dumps(args)


def _replace_json_key_name(fragment: str, old_key: str, new_key: str) -> str:
    """Rename a JSON object key without touching string values that contain the word."""
    if old_key == new_key or not old_key:
        return fragment
    # Match "old_key" only when it is a key (followed by optional space and colon).
    pattern = re.compile(rf'"{re.escape(old_key)}"(\s*:)')
    return pattern.sub(rf'"{new_key}"\1', fragment)


def _remap_argument_fragment(
    name: str,
    args_fragment: str,
    writers: dict[str, Any],
    *,
    target_name: str | None = None,
) -> str:
    """Best-effort field-key remap inside a streaming JSON args fragment."""
    if not args_fragment:
        return args_fragment

    tool_names = writers.get("tool_names") or frozenset()
    if name in _HALLUCINATED_WRITE_NAMES and name not in tool_names:
        path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
        content_field = writers.get("write_content_field", _DEFAULT_WRITERS["write_content_field"])
        for variant in _PATH_FIELD_VARIANTS:
            if variant != path_field:
                args_fragment = _replace_json_key_name(args_fragment, variant, path_field)
        for variant in _CONTENT_FIELD_VARIANTS:
            if variant != content_field:
                args_fragment = _replace_json_key_name(args_fragment, variant, content_field)
    elif name in _HALLUCINATED_EDIT_NAMES and name not in tool_names:
        path_field = writers.get("edit_path_field", _DEFAULT_WRITERS["edit_path_field"])
        old_field = writers.get("edit_old_field", _DEFAULT_WRITERS["edit_old_field"])
        new_field = writers.get("edit_new_field", _DEFAULT_WRITERS["edit_new_field"])
        for variant in _PATH_FIELD_VARIANTS:
            if variant != path_field:
                args_fragment = _replace_json_key_name(args_fragment, variant, path_field)
        for variant in _OLD_FIELD_VARIANTS:
            if variant != old_field:
                args_fragment = _replace_json_key_name(args_fragment, variant, old_field)
        for variant in _NEW_FIELD_VARIANTS:
            if variant != new_field:
                args_fragment = _replace_json_key_name(args_fragment, variant, new_field)
    return args_fragment


def _repair_edit_tool_args(
    name: str,
    args_str: str,
    writers: dict[str, Any],
    messages: list[dict] | None,
) -> str:
    if not _is_edit_tool_name(name, writers) or not (args_str or "").strip():
        return args_str
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return args_str
    if not isinstance(args, dict):
        return args_str

    args = _normalize_edit_arg_keys(args, writers)

    old_key = writers.get("edit_old_field", _DEFAULT_WRITERS["edit_old_field"])
    path_key = writers.get("edit_path_field", _DEFAULT_WRITERS["edit_path_field"])
    new_key = writers.get("edit_new_field", _DEFAULT_WRITERS["edit_new_field"])
    if old_key not in args or path_key not in args:
        return json.dumps(args)

    old_val = args.get(old_key)
    new_val = args.get(new_key) if new_key in args else None
    if isinstance(old_val, str) and isinstance(new_val, str) and old_val == new_val:
        log.warning("[fuzzy] skipping edit — old_string equals new_string")
        return json.dumps(args)

    if isinstance(old_val, str) and old_val:
        resolved = _resolve_file_path(str(args[path_key]), messages)
        deleting = isinstance(new_val, str) and not new_val.strip()
        fixed = _fuzzy_find(resolved, old_val, prefer_suffix=deleting)
        if fixed is not None and fixed != old_val:
            args[old_key] = fixed

    return json.dumps(args)


def repair_tool_call(
    name: str,
    args_str: str,
    writers: dict[str, Any] | None = None,
    messages: list[dict] | None = None,
) -> tuple[str, str]:
    writers = writers or dict(_DEFAULT_WRITERS)
    if "todo_info" not in writers:
        writers = {**writers, "todo_info": _discover_todo_info(None)}
    name, args_str = _remap_tool_call_name_and_args(name, args_str, writers)
    name = _remap_todo_tool_name(name, writers)
    args_str = _repair_edit_tool_args(name, args_str, writers, messages)

    if not (args_str or "").strip():
        return name, args_str

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return name, args_str

    if _is_todo_tool_name(name, writers):
        try:
            fixed = repair_todo_write_args(args, writers.get("todo_info"))
        except ValueError as exc:
            log.warning("[todo-repair] could not repair %s: %s", name, exc)
            return name, args_str
        log.info("[todo-repair] repaired %s (%d todo(s))", name, len(fixed["todos"]))
        return name, json.dumps(fixed)

    if name not in ASK_TOOL_NAMES:
        return name, args_str
    try:
        fixed = repair_ask_question_args(args)
    except ValueError as exc:
        log.warning("[ask-repair] could not repair %s: %s", name, exc)
        return name, args_str
    log.info("[ask-repair] repaired %s (%d question(s))", name, len(fixed["questions"]))
    return name, json.dumps(fixed)


def _patch_tool_call_delta(
    tc: dict,
    writers: dict[str, Any],
    remapped_sources: dict[int, str],
) -> None:
    func = tc.get("function") or {}
    raw_name = func.get("name") or ""
    idx = tc.get("index", 0)

    if raw_name:
        new_name, new_args = _remap_tool_call_name_and_args(
            raw_name, func.get("arguments") or "", writers
        )
        if new_name != raw_name:
            remapped_sources[idx] = raw_name
        func["name"] = new_name
        func["arguments"] = new_args
    elif idx in remapped_sources:
        func["arguments"] = _remap_argument_fragment(
            remapped_sources[idx], func.get("arguments") or "", writers
        )


def _patch_tool_call_event(ev: dict, writers: dict[str, Any], remapped_sources: dict[int, str]) -> bool:
    changed = False
    for choice in ev.get("choices", []):
        for tc in choice.get("delta", {}).get("tool_calls", []):
            before = json.dumps(tc, sort_keys=True)
            _patch_tool_call_delta(tc, writers, remapped_sources)
            if json.dumps(tc, sort_keys=True) != before:
                changed = True
    return changed


def _encode_sse_event(ev: dict) -> bytes:
    return f"data: {json.dumps(ev)}\n\n".encode()


def _encode_sse_line(line: str, writers: dict[str, Any], remapped_sources: dict[int, str]) -> bytes:
    payload = line[5:].strip()
    try:
        ev = json.loads(payload)
    except json.JSONDecodeError:
        return f"{line}\n\n".encode()
    _patch_tool_call_event(ev, writers, remapped_sources)
    return _encode_sse_event(ev)


def _tool_names_from_event(ev: dict) -> list[str]:
    names: list[str] = []
    for choice in ev.get("choices", []):
        for tc in choice.get("delta", {}).get("tool_calls", []):
            name = (tc.get("function") or {}).get("name") or ""
            if name:
                names.append(name)
    return names


def _stream_buffer_mode(events: list[dict], writers: dict[str, Any]) -> str | None:
    """Return ``ask``, ``todo``, or ``edit`` when a stream must be buffered for repair."""
    for ev in events:
        for name in _tool_names_from_event(ev):
            if name in ASK_TOOL_NAMES:
                return "ask"
            if _is_todo_tool_name(name, writers):
                return "todo"
            if _is_edit_tool_name(name, writers):
                return "edit"
    return None


def _reassemble_tool_calls(events: list[dict]) -> dict[int, dict]:
    tool_calls: dict[int, dict] = {}
    for ev in events:
        for choice in ev.get("choices", []):
            for tc in choice.get("delta", {}).get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "type": "function", "name": "", "arguments": ""}
                slot = tool_calls[idx]
                slot["id"] = slot["id"] or tc.get("id", "")
                slot["type"] = slot["type"] or tc.get("type", "function")
                func = tc.get("function", {})
                slot["name"] = slot["name"] or func.get("name", "")
                slot["arguments"] += func.get("arguments") or ""
    return tool_calls


def _emit_repaired_stream(
    events: list[dict],
    writers: dict[str, Any],
    messages: list[dict] | None = None,
) -> list[bytes]:
    tool_calls = _reassemble_tool_calls(events)
    if not tool_calls:
        return [f"data: {json.dumps(ev)}\n\n".encode() for ev in events] + [b"data: [DONE]\n\n"]

    response_id, model_name = _stream_meta(events)

    for tc in tool_calls.values():
        tc["name"], tc["arguments"] = repair_tool_call(
            tc["name"], tc["arguments"], writers, messages
        )

    chunks: list[bytes] = []
    # Preserve any content / reasoning tokens that accompanied the tool stream.
    content = _stream_content_from_events(events)
    reasoning = _stream_reasoning_from_events(events)
    if content or reasoning:
        pre_delta: dict[str, Any] = {"role": "assistant"}
        if content:
            pre_delta["content"] = content
        if reasoning:
            pre_delta["reasoning_content"] = reasoning
        chunks.append(_encode_sse_event({
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": pre_delta,
                "finish_reason": None,
            }],
        }))

    delta_event = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant" if not (content or reasoning) else "",
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc["id"] or f"call_{uuid.uuid4().hex[:12]}",
                        "type": tc["type"],
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for i, tc in sorted(tool_calls.items())
                ],
            },
            "finish_reason": None,
        }],
    }
    # Empty role string confuses some clients — omit when content already set role.
    if content or reasoning:
        delta_event["choices"][0]["delta"].pop("role", None)
    finish_event = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    chunks.append(f"data: {json.dumps(delta_event)}\n\n".encode())
    chunks.append(f"data: {json.dumps(finish_event)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


async def _emit_repaired_stream_async(
    events: list[dict],
    writers: dict[str, Any],
    messages: list[dict] | None = None,
) -> list[bytes]:
    """Run CPU-heavy tool repair (fuzzy file match) off the event loop."""
    return await asyncio.to_thread(_emit_repaired_stream, events, writers, messages)


async def _repair_tool_call_async(
    name: str,
    args_str: str,
    writers: dict[str, Any] | None = None,
    messages: list[dict] | None = None,
) -> tuple[str, str]:
    return await asyncio.to_thread(repair_tool_call, name, args_str, writers, messages)


def _parse_upstream(upstream: str) -> str:
    return upstream.rstrip("/")


def _upstream_api_path(upstream: str, subpath: str) -> str:
    """Build httpx path for OpenAI routes when base URL may already end in ``/v1``.

    With ``http://host:11434/v1``, ``chat/completions`` must be requested as
    ``/chat/completions`` — not ``/v1/chat/completions`` (which 404s on Ollama).
    llama-server is typically configured the same way in this repo.
    """
    base = _parse_upstream(upstream)
    subpath = subpath.lstrip("/")
    if subpath.startswith("v1/"):
        subpath = subpath[3:]
    if base.endswith("/v1"):
        return f"/{subpath}" if subpath else "/"
    return f"/v1/{subpath}" if subpath else "/v1"


def create_app(upstream: str) -> FastAPI:
    upstream_base = _parse_upstream(upstream)
    client = httpx.AsyncClient(base_url=upstream_base, timeout=httpx.Timeout(900.0, connect=30.0))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="ornith-tool-proxy", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        if messages:
            body["messages"] = _truncate_tool_results(messages)
            messages = body["messages"]
        compaction = _is_compaction_request(body)
        if compaction:
            _prepare_compaction_request(body)
            _ensure_max_tokens(body)
            log.info("[compaction] text-only mode — stripped tools for summary generation")
        else:
            _prepare_upstream_request(body, messages)
        writers = _discover_writers(body.get("tools"))
        stream = bool(body.get("stream"))
        tools_forced = body.get("tool_choice") == "required"
        text_repair_enabled = (
            not compaction
            and bool(body.get("tools"))
            and (_bash_tool_available(writers) or _write_tool_available(writers))
        )

        try:
            if stream:
                chat_path = _upstream_api_path(upstream_base, "chat/completions")
                req = client.build_request("POST", chat_path, json=body)
                resp = await client.send(req, stream=True)
            else:
                resp = await client.post(
                    _upstream_api_path(upstream_base, "chat/completions"), json=body
                )
        except httpx.HTTPError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "proxy_error"}}, status_code=502)

        if resp.status_code != 200:
            content = await resp.aread()
            return Response(
                content=content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

        if not stream:
            try:
                raw = await resp.aread()
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                snippet = (raw[:200] if isinstance(raw, (bytes, bytearray)) else b"")
                log.warning(
                    "[upstream] non-JSON chat completion body (%s): %r",
                    exc,
                    snippet,
                )
                return JSONResponse(
                    {
                        "error": {
                            "message": f"upstream returned non-JSON body: {exc}",
                            "type": "proxy_upstream_decode_error",
                        }
                    },
                    status_code=502,
                )
            if not isinstance(data, dict):
                return JSONResponse(
                    {
                        "error": {
                            "message": "upstream JSON was not an object",
                            "type": "proxy_upstream_decode_error",
                        }
                    },
                    status_code=502,
                )
            if compaction:
                _flatten_tool_calls_in_response(data)
            else:
                for choice in data.get("choices", []):
                    msg = choice.get("message", {})
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        name, args = await _repair_tool_call_async(
                            func.get("name", ""),
                            func.get("arguments", "{}"),
                            writers,
                            messages,
                        )
                        func["name"], func["arguments"] = name, args
                await asyncio.to_thread(
                    _repair_text_tool_calls_in_response,
                    data,
                    writers,
                    messages,
                    aggressive=tools_forced,
                )
            return JSONResponse(data)

        async def stream_gen():
            # AskQuestion, TodoWrite, and edit/StrReplace streams are buffered for repair.
            # Write streams pass through in real time so large payloads are not delayed.
            # When text-repair is active, content is buffered until the stream ends so we
            # never dual-emit free text + recovered tool_calls.
            events: list[dict] = []
            content_events: list[dict] = []
            pending_tool_lines: list[str] = []
            saw_tool_call = False
            buffer_mode: str | None = None
            remapped_sources: dict[int, str] = {}
            compaction_response_id = ""
            compaction_model = ""
            stream_response_id = ""
            stream_model = ""
            line_iter = resp.aiter_lines().__aiter__()
            done_sent = False
            try:
                while True:
                    if await request.is_disconnected():
                        log.info("[stream] client disconnected — aborting upstream read")
                        break
                    try:
                        line = await asyncio.wait_for(
                            line_iter.__anext__(),
                            timeout=_KEEPALIVE_INTERVAL_S,
                        )
                    except asyncio.TimeoutError:
                        yield _KEEPALIVE_LINE
                        continue
                    except StopAsyncIteration:
                        break

                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    stream_response_id = stream_response_id or ev.get("id", "")
                    stream_model = stream_model or ev.get("model", "")

                    has_tool_delta = any(
                        choice.get("delta", {}).get("tool_calls")
                        for choice in ev.get("choices", [])
                    )
                    if compaction:
                        compaction_response_id = compaction_response_id or ev.get("id", "")
                        compaction_model = compaction_model or ev.get("model", "")
                        if has_tool_delta:
                            patched = _tool_delta_to_content_event(ev)
                            if patched:
                                yield _encode_sse_event(patched)
                            continue
                        for choice in ev.get("choices", []):
                            if choice.get("finish_reason") == "tool_calls":
                                choice["finish_reason"] = "stop"
                        yield _encode_sse_event(ev)
                        continue

                    if has_tool_delta:
                        saw_tool_call = True
                        # Flush any content we were holding for text-repair before tools.
                        if content_events:
                            for cev in content_events:
                                yield _encode_sse_event(cev)
                            content_events.clear()
                        if buffer_mode is None:
                            pending_tool_lines.append(line)
                            events.append(ev)
                            names = _tool_names_from_event(ev)
                            if names:
                                buffer_mode = _stream_buffer_mode(events, writers)
                                if buffer_mode:
                                    log.info("[stream] buffering %s stream for repair", buffer_mode)
                                else:
                                    for pending in pending_tool_lines:
                                        yield _encode_sse_line(pending, writers, remapped_sources)
                                    pending_tool_lines.clear()
                                    events.clear()
                            continue

                    if buffer_mode is not None:
                        events.append(ev)
                        continue

                    if pending_tool_lines:
                        for pending in pending_tool_lines:
                            yield _encode_sse_line(pending, writers, remapped_sources)
                        pending_tool_lines.clear()
                        events.clear()

                    # Buffer pure-content streams for optional text→tool recovery.
                    # Do NOT live-yield: recovery replaces the whole completion.
                    if text_repair_enabled and not saw_tool_call:
                        content_events.append(ev)
                        continue

                    yield _encode_sse_line(line, writers, remapped_sources)

                if compaction:
                    yield _encode_sse_event({
                        "id": compaction_response_id or stream_response_id
                        or f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "model": compaction_model or stream_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                    yield b"data: [DONE]\n\n"
                    done_sent = True
                elif buffer_mode is not None:
                    for chunk in await _emit_repaired_stream_async(events, writers, messages):
                        yield chunk
                    done_sent = True
                else:
                    recovered = None
                    if text_repair_enabled and content_events and not saw_tool_call:
                        recovered = await asyncio.to_thread(
                            _emit_text_recovered_tool_stream,
                            content_events,
                            writers,
                            messages,
                            aggressive=tools_forced,
                        )
                    if recovered:
                        for chunk in recovered:
                            yield chunk
                        done_sent = True
                    elif content_events and not saw_tool_call:
                        for chunk in _emit_content_stream_chunks(content_events):
                            yield chunk
                        done_sent = True
                    else:
                        for pending in pending_tool_lines:
                            yield _encode_sse_line(pending, writers, remapped_sources)
                        yield b"data: [DONE]\n\n"
                        done_sent = True
            except Exception as exc:
                log.exception("[stream] upstream/proxy failure: %s", exc)
                if not done_sent:
                    for chunk in _emit_stream_error(
                        stream_response_id or compaction_response_id,
                        stream_model or compaction_model,
                        str(exc),
                    ):
                        yield chunk
                    done_sent = True
            finally:
                if not done_sent:
                    try:
                        yield b"data: [DONE]\n\n"
                    except Exception:
                        pass
                await resp.aclose()

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, request: Request):
        try:
            resp = await client.request(
                request.method,
                _upstream_api_path(upstream_base, path),
                content=await request.body(),
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() in ("content-type", "authorization")
                },
            )
        except httpx.HTTPError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=502)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @app.get("/healthz")
    async def healthz():
        """Probe upstream readiness.

        When ``upstream`` ends in ``/v1`` (Ollama + llama-server in this repo),
        httpx resolves ``/health`` to ``/v1/health`` which Ollama does not serve.
        ``/models`` via ``_upstream_api_path`` is the reliable check.
        """
        ok = False
        try:
            r = await client.get(_upstream_api_path(upstream_base, "models"), timeout=5.0)
            ok = r.status_code == 200
            if not ok and not upstream_base.endswith("/v1"):
                r = await client.get("/health", timeout=5.0)
                ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        return JSONResponse({"ok": ok, "upstream": upstream_base}, status_code=200 if ok else 503)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-call proxy for Ornith Ollama harness")
    parser.add_argument("--host", default="0.0.0.0")
    # Match 2_start_ollama.sh defaults (proxy :18082 → Ollama :11434).
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    uvicorn.run(create_app(args.upstream), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()