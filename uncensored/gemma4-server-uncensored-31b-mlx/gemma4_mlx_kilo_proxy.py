#!/usr/bin/env python3
"""Lean Kilo Code proxy for Gemma 4 + vllm-mlx (31B Heretic).

OpenAI-compatible middleware between Kilo/Continue and vllm-mlx.

Tool repair
  • AskQuestion / TodoWrite argument normalization
  • Cline/Roo tool names remapped to the client's schema
  • Fuzzy old_string repair for StrReplace-style edits
  • Large tool results truncated before upstream

Kilo compaction
  • Detects summary turns; strips tools; flattens residual tool_calls to text

MLX / Heretic reliability (kept from the old 10k-line proxy)
  • Harmony token logit_bias (ids 98/100/101) on agentic turns
  • Agentic temperature floor 0.35; thinking disabled
  • Empty-delta abort when vllm-mlx spins empty SSE chunks
  • Token-stall abort with graceful finish_reason=stop (no Kilo retry storm)
  • Tool-args size cap (repetition loops)
  • Single-flight lock (one active generation — MLX state is not multi-stream safe)
  • Optional model-name rewrite (--model)

Architecture with ``./2_start_mlx.sh --proxy``::

    Kilo  →  :8080 this proxy  →  :8090 vllm-mlx
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

# Runtime import required: with ``from __future__ import annotations``, FastAPI
# resolves parameter annotations from this module's globals.  A TYPE_CHECKING-only
# import leaves ``Request`` unresolved, so routes treat ``request`` as a required
# *query* param and every /v1/* call returns HTTP 422.
from fastapi import Request

log = logging.getLogger("gemma4_mlx_kilo_proxy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

PLANNING_TOOL_NAMES = frozenset({
    "TodoWrite",
    "todowrite",
    "todo_write",
    "todoWrite",
    "update_todo_list",
    "updateTodoList",
    "TodoRead",
    "todoread",
    "todo_read",
})

_VALID_TODO_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
_VALID_TODO_PRIORITIES = frozenset({"high", "medium", "low"})

_DEFAULT_TODO_INFO: dict[str, Any] = {
    "client_todo_name": None,
    "todo_item_fields": frozenset({"id", "content", "status"}),
    "has_merge": True,
}

_KEEPALIVE_INTERVAL_S = 15.0
_KEEPALIVE_LINE = b": keepalive\n\n"

_FUZZY_THRESHOLD = 0.85
_FUZZY_THRESHOLD_DELETE = 0.80
_FUZZY_WINDOW_SLACK = 4
_FUZZY_MAX_FILE_BYTES = 2_000_000
_FUZZY_MAX_FILE_LINES = 20_000

_TOOL_RESULT_MAX_LINES = 300
_TOOL_RESULT_TRUNCATION_MSG = (
    "\n... [proxy: truncated to {kept} lines of {total} — "
    "remaining {skipped} lines hidden to save context. "
    "Ask for specific sections if needed.] ..."
)

# Heretic distillation sometimes emits Harmony / CoT control tokens that
# derail the gemma4 tool parser. Ban them at the sampler when tools are present.
# Token ids from gemma-4 heretic tokenizer.json added_tokens.
_HARMONY_LOGIT_BIAS: dict[str, float] = {
    "100": -100.0,  # <|channel>
    "101": -100.0,  # <channel|>
    "98": -100.0,   # <|think|>
}

_AGENT_TEMP_MIN = 0.35
_AGENT_TEMP_MAX = 0.55

# Stream guards (vllm-mlx MLLM-path empty-delta loops, etc.)
_EMPTY_DELTA_STREAK = 100
_STALL_ABORT_S = 90.0
_STALL_ABORT_AGENTIC_S = 45.0
_ARGS_CAP_CHARS = 8192

_HALLUCINATED_WRITE_NAMES = frozenset({
    "write",  # Gemma native short name (Kilo expects Write)
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

# One generation at a time — concurrent vllm-mlx streams can corrupt MLX state.
# Lazily created so import does not bind a loop before uvicorn starts.
_singleflight_lock: asyncio.Lock | None = None


def _singleflight() -> asyncio.Lock:
    global _singleflight_lock
    if _singleflight_lock is None:
        _singleflight_lock = asyncio.Lock()
    return _singleflight_lock


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

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
    """System + latest user only (full history false-positives on 'summarize')."""
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
    if _tool_choice_disallows_tools(body.get("tool_choice")):
        return True
    blob = _compaction_probe_blob(body.get("messages"))
    return bool(blob) and any(pattern.search(blob) for pattern in _COMPACTION_HINTS)


def _prepare_compaction_request(body: dict) -> None:
    body.pop("tools", None)
    body["tool_choice"] = "none"
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


# ---------------------------------------------------------------------------
# Path / fuzzy edit repair
# ---------------------------------------------------------------------------

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
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                path = m.group(1).strip().rstrip(")'\"")
                if path and os.path.isdir(path):
                    return path
    return None


def _resolve_file_path(file_path: str, messages: list[dict] | None) -> str:
    if not file_path:
        return file_path
    if os.path.isabs(file_path):
        return file_path
    workspace = _extract_workspace_dir(messages)
    if workspace:
        candidate = os.path.join(workspace, file_path)
        if os.path.exists(candidate):
            return candidate
        return candidate
    return file_path


def _normalize_block_lines(lines: list[str]) -> list[str]:
    return [ln.rstrip() for ln in lines]


def _relative_indent_lines(lines: list[str]) -> list[str]:
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return lines
    indents = [len(ln) - len(ln.lstrip(" \t")) for ln in non_empty]
    min_indent = min(indents) if indents else 0
    if min_indent <= 0:
        return lines
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        else:
            out.append(ln[min_indent:] if len(ln) >= min_indent else ln.lstrip())
    return out


def _block_similarity(file_window: list[str], old_lines: list[str]) -> float:
    a = "\n".join(_normalize_block_lines(file_window))
    b = "\n".join(_normalize_block_lines(old_lines))
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    a2 = "\n".join(_normalize_block_lines(_relative_indent_lines(file_window)))
    b2 = "\n".join(_normalize_block_lines(_relative_indent_lines(old_lines)))
    ratio2 = difflib.SequenceMatcher(None, a2, b2).ratio()
    return max(ratio, ratio2)


def _fuzzy_find_in_lines(
    file_lines: list[str],
    old_lines: list[str],
    *,
    prefer_suffix: bool = False,
    threshold: float = _FUZZY_THRESHOLD,
) -> tuple[int, int, float] | None:
    target = len(old_lines)
    if target == 0 or not file_lines:
        return None
    min_window = max(1, target - _FUZZY_WINDOW_SLACK)
    max_window = target + _FUZZY_WINDOW_SLACK
    search_start = max(0, len(file_lines) - (target + 15)) if prefer_suffix else 0

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
        if size > _FUZZY_MAX_FILE_BYTES:
            log.warning("[fuzzy] skip %s — file too large (%d bytes)", file_path, size)
            return None
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            file_content = fh.read()
    except OSError:
        return None

    if old_string in file_content:
        return old_string

    file_lines = file_content.splitlines()
    if len(file_lines) > _FUZZY_MAX_FILE_LINES:
        log.warning("[fuzzy] skip %s — too many lines (%d)", file_path, len(file_lines))
        return None
    old_lines = old_string.splitlines()
    if not old_lines:
        return None

    threshold = _FUZZY_THRESHOLD_DELETE if prefer_suffix else _FUZZY_THRESHOLD
    match = _fuzzy_find_in_lines(
        file_lines, old_lines, prefer_suffix=prefer_suffix, threshold=threshold,
    )
    if match is None:
        log.warning(
            "[fuzzy] %s could not repair old_string (%d lines)",
            os.path.basename(file_path), len(old_lines),
        )
        return None

    start, end, ratio = match
    matched = "\n".join(file_lines[start:end])
    log.info(
        "[fuzzy] %s ratio=%.2f fixed old_string (%d→%d lines)",
        os.path.basename(file_path), ratio, len(old_lines), end - start,
    )
    return matched


# ---------------------------------------------------------------------------
# Tool schema discovery + repair
# ---------------------------------------------------------------------------

def _normalize_edit_arg_keys(args: dict[str, Any], writers: dict[str, Any]) -> dict[str, Any]:
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
    patched: list[dict] = []
    for msg in messages:
        if msg.get("role") != "tool":
            patched.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            patched.append(msg)
            continue
        lines = content.splitlines()
        if len(lines) <= _TOOL_RESULT_MAX_LINES:
            patched.append(msg)
            continue
        kept = lines[:_TOOL_RESULT_MAX_LINES]
        skipped = len(lines) - _TOOL_RESULT_MAX_LINES
        suffix = _TOOL_RESULT_TRUNCATION_MSG.format(
            kept=_TOOL_RESULT_MAX_LINES, total=len(lines), skipped=skipped,
        )
        patched.append({**msg, "content": "\n".join(kept) + suffix})
        log.info(
            "[truncate] tool result: %d → %d lines (%d hidden)",
            len(lines), _TOOL_RESULT_MAX_LINES, skipped,
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
    item: Any, index: int, fields: frozenset[str],
) -> dict[str, Any] | None:
    if isinstance(item, str) and item.strip():
        parsed_item = _parse_json_value(item.strip())
        item = parsed_item if isinstance(parsed_item, dict) else {"content": item}
    if not isinstance(item, dict):
        return None
    content = (
        item.get("content") or item.get("task") or item.get("description")
        or item.get("text") or item.get("title") or item.get("name") or ""
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
    args: Any, todo_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if not tools:
        return dict(_DEFAULT_WRITERS)
    info: dict[str, Any] = dict(_DEFAULT_WRITERS)
    info["tool_names"] = frozenset(
        (t.get("function", {}) or {}).get("name", "")
        for t in tools if isinstance(t, dict)
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


def _tool_names_lower(tool_names: Any) -> set[str]:
    if not tool_names:
        return set()
    return {str(n).lower() for n in tool_names}


def _remap_tool_call_name_and_args(
    name: str, args_str: str, writers: dict[str, Any],
) -> tuple[str, str]:
    tool_names = writers.get("tool_names") or frozenset()
    names_lower = _tool_names_lower(tool_names)
    # Exact match wins (Kilo often exposes Write/StrReplace with that casing).
    if name in tool_names:
        return name, args_str
    # Case-insensitive match → rewrite to the client's real tool name.
    for real in tool_names:
        if str(real).lower() == name.lower():
            if real != name:
                log.info("[remap] case-fold %s → %s", name, real)
            return str(real), args_str

    target_name: str | None = None
    target_path_field: str | None = None
    target_content_field: str | None = None
    target_old_field: str | None = None
    target_new_field: str | None = None

    lname = name.lower()
    if (name in _HALLUCINATED_WRITE_NAMES or lname in _HALLUCINATED_WRITE_NAMES) and (
        lname not in names_lower
    ):
        target_name = writers.get("write_name", _DEFAULT_WRITERS["write_name"])
        target_path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
        target_content_field = writers.get(
            "write_content_field", _DEFAULT_WRITERS["write_content_field"],
        )
    elif (name in _HALLUCINATED_EDIT_NAMES or lname in _HALLUCINATED_EDIT_NAMES) and (
        lname not in names_lower
    ):
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
        return target_name, _remap_argument_fragment(name, args_str, writers)

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

    log.info("[remap] %s → %s (%s)", name, target_name, ",".join(sorted(args.keys())))
    return target_name, json.dumps(args)


def _remap_argument_fragment(
    name: str, args_fragment: str, writers: dict[str, Any],
) -> str:
    if not args_fragment:
        return args_fragment
    tool_names = writers.get("tool_names") or frozenset()
    if name in _HALLUCINATED_WRITE_NAMES and name not in tool_names:
        path_field = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
        content_field = writers.get("write_content_field", _DEFAULT_WRITERS["write_content_field"])
        for variant in _PATH_FIELD_VARIANTS:
            if variant != path_field:
                args_fragment = args_fragment.replace(f'"{variant}"', f'"{path_field}"')
        for variant in _CONTENT_FIELD_VARIANTS:
            if variant != content_field:
                args_fragment = args_fragment.replace(f'"{variant}"', f'"{content_field}"')
    elif name in _HALLUCINATED_EDIT_NAMES and name not in tool_names:
        path_field = writers.get("edit_path_field", _DEFAULT_WRITERS["edit_path_field"])
        old_field = writers.get("edit_old_field", _DEFAULT_WRITERS["edit_old_field"])
        new_field = writers.get("edit_new_field", _DEFAULT_WRITERS["edit_new_field"])
        for variant in _PATH_FIELD_VARIANTS:
            if variant != path_field:
                args_fragment = args_fragment.replace(f'"{variant}"', f'"{path_field}"')
        for variant in _OLD_FIELD_VARIANTS:
            if variant != old_field:
                args_fragment = args_fragment.replace(f'"{variant}"', f'"{old_field}"')
        for variant in _NEW_FIELD_VARIANTS:
            if variant != new_field:
                args_fragment = args_fragment.replace(f'"{variant}"', f'"{new_field}"')
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


def _normalize_write_arg_keys(args: dict[str, Any], writers: dict[str, Any]) -> dict[str, Any]:
    """Map filePath/content aliases onto the client's Write schema fields."""
    path_key = writers.get("write_path_field", _DEFAULT_WRITERS["write_path_field"])
    content_key = writers.get("write_content_field", _DEFAULT_WRITERS["write_content_field"])
    out = dict(args)
    if path_key not in out:
        for src in _PATH_FIELD_VARIANTS:
            if src in out:
                out[path_key] = out.pop(src)
                break
    if content_key not in out:
        for src in _CONTENT_FIELD_VARIANTS:
            if src in out:
                out[content_key] = out.pop(src)
                break
    return out


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

    # Gemma4 parser often emits filePath; Kilo Write expects path.
    if _is_write_tool_name(name) or name == writers.get("write_name"):
        if isinstance(args, dict):
            fixed_keys = _normalize_write_arg_keys(args, writers)
            if fixed_keys != args:
                log.info("[remap] write arg keys → %s", ",".join(sorted(fixed_keys)))
                args = fixed_keys
                args_str = json.dumps(args)

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


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _patch_tool_call_delta(
    tc: dict, writers: dict[str, Any], remapped_sources: dict[int, str],
) -> None:
    func = tc.get("function") or {}
    raw_name = func.get("name") or ""
    idx = tc.get("index", 0)
    if raw_name:
        new_name, new_args = _remap_tool_call_name_and_args(
            raw_name, func.get("arguments") or "", writers,
        )
        if new_name != raw_name:
            remapped_sources[idx] = raw_name
        func["name"] = new_name
        func["arguments"] = new_args
    elif idx in remapped_sources:
        func["arguments"] = _remap_argument_fragment(
            remapped_sources[idx], func.get("arguments") or "", writers,
        )


def _patch_tool_call_event(
    ev: dict, writers: dict[str, Any], remapped_sources: dict[int, str],
) -> bool:
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
                    tool_calls[idx] = {
                        "id": "", "type": "function", "name": "", "arguments": "",
                    }
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

    response_id = events[0].get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    model_name = events[0].get("model", "")

    for tc in tool_calls.values():
        tc["name"], tc["arguments"] = repair_tool_call(
            tc["name"], tc["arguments"], writers, messages,
        )

    delta_event = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
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


def _graceful_stop_chunk(response_id: str, model_name: str) -> list[bytes]:
    """Normal completion end so Kilo does not retry (errors re-queue ghosts)."""
    return [
        _encode_sse_event({
            "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }),
        b"data: [DONE]\n\n",
    ]


def _completion_tokens(ev: dict) -> int | None:
    usage = ev.get("usage")
    if not isinstance(usage, dict):
        return None
    val = usage.get("completion_tokens")
    return int(val) if isinstance(val, (int, float)) else None


def _delta_is_empty(ev: dict) -> bool:
    for choice in ev.get("choices", []):
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
            return False
        if choice.get("finish_reason"):
            return False
    return True


# ---------------------------------------------------------------------------
# Agentic request shaping (MLX / Heretic)
# ---------------------------------------------------------------------------

def _assistant_tool_names(messages: list[dict] | None) -> set[str]:
    names: set[str] = set()
    if not messages:
        return names
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = ((tc.get("function") or {}).get("name") or "").lower()
            if name:
                names.add(name)
    return names


def _is_write_tool_name(name: str) -> bool:
    """True for file-write tools — not TodoWrite / todowrite (planning)."""
    lname = name.lower()
    if lname in {p.lower() for p in PLANNING_TOOL_NAMES}:
        return False
    if lname in {n.lower() for n in _HALLUCINATED_WRITE_NAMES}:
        return True
    if "create_file" in lname or "createfile" in lname:
        return True
    # Require write as a path segment, not a substring of "todowrite".
    tokens = re.split(r"[_\s-]+", lname)
    return "write" in tokens and "rewrite" not in tokens


def _strip_planning_tools_if_stuck(body: dict) -> None:
    """After a planning tool ran with no write yet, hide todowrite so the model must write."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    messages = body.get("messages") or []
    names = _assistant_tool_names(messages)
    if not names:
        return
    planned = any(n in {p.lower() for p in PLANNING_TOOL_NAMES} for n in names)
    wrote = any(_is_write_tool_name(n) for n in names)
    if not planned or wrote:
        return
    kept = []
    removed = []
    for t in tools:
        fn = (t.get("function") or {}) if isinstance(t, dict) else {}
        name = fn.get("name", "")
        if name in PLANNING_TOOL_NAMES or name.lower() in {p.lower() for p in PLANNING_TOOL_NAMES}:
            removed.append(name)
        else:
            kept.append(t)
    if removed and kept:
        body["tools"] = kept
        log.info("[strip-planning] removed %s (no write yet)", removed)


def _force_agentic_settings(body: dict) -> None:
    """Temperature floor, thinking off, Harmony logit bias for tool-using turns."""
    if not body.get("tools"):
        return
    old_temp = float(body.get("temperature") or 0)
    body["temperature"] = min(max(old_temp, _AGENT_TEMP_MIN), _AGENT_TEMP_MAX)
    body.setdefault("top_p", 0.95)
    body["enable_thinking"] = False
    ctk = body.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    ctk["enable_thinking"] = False
    body["chat_template_kwargs"] = ctk

    bias = dict(_HARMONY_LOGIT_BIAS)
    existing = body.get("logit_bias")
    if isinstance(existing, dict):
        # Caller-supplied keys win.
        bias.update({str(k): float(v) for k, v in existing.items()})
    body["logit_bias"] = bias
    log.info(
        "[settings] agentic — temp=%.2f logit_bias Harmony suppress keys=%s",
        body["temperature"], list(_HARMONY_LOGIT_BIAS),
    )


def _parse_upstream(upstream: str) -> str:
    return upstream.rstrip("/")


def _upstream_api_path(upstream: str, subpath: str) -> str:
    base = _parse_upstream(upstream)
    subpath = subpath.lstrip("/")
    if subpath.startswith("v1/"):
        subpath = subpath[3:]
    if base.endswith("/v1"):
        return f"/{subpath}" if subpath else "/"
    return f"/v1/{subpath}" if subpath else "/v1"


# ---------------------------------------------------------------------------
# Model rewrite middleware
# ---------------------------------------------------------------------------

def _make_model_rewrite_middleware(model_name: str) -> Any:
    """Force ``model`` field on chat/completions bodies to a fixed id."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from fastapi import Request

    class ModelRewriteMiddleware(BaseHTTPMiddleware):
        def __init__(self, app: Any):
            super().__init__(app)
            self.model_name = model_name

        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            if (
                request.url.path.rstrip("/").endswith("/chat/completions")
                and request.method == "POST"
            ):
                try:
                    body = await request.body()
                    data = json.loads(body) if body else {}
                    if isinstance(data, dict) and data.get("model") != self.model_name:
                        log.debug(
                            "[model-rewrite] %r → %r",
                            data.get("model"), self.model_name,
                        )
                        data["model"] = self.model_name
                        body = json.dumps(data).encode()
                    headers = [
                        (k, v) for k, v in request.scope.get("headers", [])
                        if k.lower() != b"content-length"
                    ]
                    headers.append((b"content-length", str(len(body)).encode()))
                    request.scope["headers"] = headers

                    async def receive() -> dict:
                        return {"type": "http.request", "body": body, "more_body": False}

                    request = Request(request.scope, receive)
                except Exception as exc:
                    log.warning("[model-rewrite] failed: %s", exc)
            return await call_next(request)

    return ModelRewriteMiddleware


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(upstream: str, model_override: str | None = None) -> Any:
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    upstream_base = _parse_upstream(upstream)
    client = httpx.AsyncClient(
        base_url=upstream_base,
        timeout=httpx.Timeout(900.0, connect=30.0),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="gemma4-mlx-kilo-proxy", lifespan=lifespan)

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
            log.info("[compaction] text-only mode — stripped tools for summary generation")
        else:
            _strip_planning_tools_if_stuck(body)
            _force_agentic_settings(body)

        writers = _discover_writers(body.get("tools"))
        stream = bool(body.get("stream"))
        is_agentic = bool(body.get("tools"))

        if not stream:
            async with _singleflight():
                try:
                    resp = await client.post(
                        _upstream_api_path(upstream_base, "chat/completions"),
                        json=body,
                    )
                except httpx.HTTPError as exc:
                    return JSONResponse(
                        {"error": {"message": str(exc), "type": "proxy_error"}},
                        status_code=502,
                    )
                if resp.status_code != 200:
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )
                data = resp.json()
            if compaction:
                _flatten_tool_calls_in_response(data)
            else:
                for choice in data.get("choices", []):
                    msg = choice.get("message", {})
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        func["name"], func["arguments"] = repair_tool_call(
                            func.get("name", ""),
                            func.get("arguments", "{}"),
                            writers,
                            messages,
                        )
            return JSONResponse(data)

        async def stream_gen():
            # Hold singleflight only while the generator is live so a never-
            # consumed StreamingResponse cannot deadlock the lock.
            await _singleflight().acquire()
            events: list[dict] = []
            pending_tool_lines: list[str] = []
            buffer_mode: str | None = None
            remapped_sources: dict[int, str] = {}
            compaction_response_id = ""
            compaction_model = ""
            response_id = ""
            model_name = ""
            empty_streak = 0
            last_token_t = time.monotonic()
            last_tokens: int | None = None
            saw_useful = False
            args_chars = 0
            aborted = False
            stall_limit = _STALL_ABORT_AGENTIC_S if is_agentic else _STALL_ABORT_S
            resp = None

            try:
                chat_path = _upstream_api_path(upstream_base, "chat/completions")
                req = client.build_request("POST", chat_path, json=body)
                try:
                    resp = await client.send(req, stream=True)
                except httpx.HTTPError as exc:
                    err = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"[proxy error] {exc}"},
                            "finish_reason": "stop",
                        }],
                    }
                    yield _encode_sse_event(err)
                    yield b"data: [DONE]\n\n"
                    return

                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    yield body_bytes
                    return

                line_iter = resp.aiter_lines().__aiter__()
                while True:
                    try:
                        line = await asyncio.wait_for(
                            line_iter.__anext__(),
                            timeout=_KEEPALIVE_INTERVAL_S,
                        )
                    except asyncio.TimeoutError:
                        # Keepalive; also check wall-clock stall.
                        if saw_useful and (time.monotonic() - last_token_t) > stall_limit:
                            log.warning(
                                "[stall-abort] no token change for %.1fs — graceful stop",
                                time.monotonic() - last_token_t,
                            )
                            for chunk in _graceful_stop_chunk(response_id, model_name):
                                yield chunk
                            aborted = True
                            break
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

                    response_id = response_id or ev.get("id", "")
                    model_name = model_name or ev.get("model", "")

                    # Track completion tokens for stall / empty-delta guards.
                    tokens = _completion_tokens(ev)
                    if tokens is not None and tokens != last_tokens:
                        last_tokens = tokens
                        last_token_t = time.monotonic()
                        empty_streak = 0
                    elif _delta_is_empty(ev) and saw_useful:
                        empty_streak += 1
                        if empty_streak >= _EMPTY_DELTA_STREAK:
                            log.warning(
                                "[empty-delta-abort] %d empty deltas after useful content "
                                "(tokens=%s) — graceful stop",
                                empty_streak, last_tokens,
                            )
                            for chunk in _graceful_stop_chunk(response_id, model_name):
                                yield chunk
                            aborted = True
                            break
                    elif not _delta_is_empty(ev):
                        empty_streak = 0
                        last_token_t = time.monotonic()

                    if saw_useful and (time.monotonic() - last_token_t) > stall_limit:
                        log.warning(
                            "[stall-abort] no token change for %.1fs — graceful stop",
                            time.monotonic() - last_token_t,
                        )
                        for chunk in _graceful_stop_chunk(response_id, model_name):
                            yield chunk
                        aborted = True
                        break

                    # Drop reasoning_content on agentic turns — Kilo should only
                    # see tools/content. Server-side gemma4 reasoning parser puts
                    # <|channel>thought… into reasoning_content; without this
                    # strip the monologue still paints the chat UI.
                    if is_agentic and not compaction:
                        for choice in ev.get("choices", []):
                            delta = choice.get("delta")
                            if isinstance(delta, dict) and delta.get("reasoning_content"):
                                delta.pop("reasoning_content", None)

                    has_tool_delta = any(
                        choice.get("delta", {}).get("tool_calls")
                        for choice in ev.get("choices", [])
                    )
                    has_content = any(
                        choice.get("delta", {}).get("content")
                        for choice in ev.get("choices", [])
                    )
                    if has_tool_delta or has_content:
                        saw_useful = True

                    if has_tool_delta:
                        for choice in ev.get("choices", []):
                            for tc in choice.get("delta", {}).get("tool_calls", []) or []:
                                args_chars += len((tc.get("function") or {}).get("arguments") or "")
                        if args_chars > _ARGS_CAP_CHARS:
                            log.warning(
                                "[args-cap] tool arguments exceeded %d chars — graceful stop",
                                _ARGS_CAP_CHARS,
                            )
                            for chunk in _graceful_stop_chunk(response_id, model_name):
                                yield chunk
                            aborted = True
                            break

                    if compaction:
                        compaction_response_id = compaction_response_id or response_id
                        compaction_model = compaction_model or model_name
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
                                        yield _encode_sse_line(
                                            pending, writers, remapped_sources,
                                        )
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

                    yield _encode_sse_line(line, writers, remapped_sources)

                if aborted:
                    return
                if compaction:
                    yield _encode_sse_event({
                        "id": compaction_response_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "model": compaction_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                    yield b"data: [DONE]\n\n"
                elif buffer_mode is not None:
                    for chunk in _emit_repaired_stream(events, writers, messages):
                        yield chunk
                else:
                    for pending in pending_tool_lines:
                        yield _encode_sse_line(pending, writers, remapped_sources)
                    yield b"data: [DONE]\n\n"
            finally:
                if resp is not None:
                    await resp.aclose()
                try:
                    _singleflight().release()
                except RuntimeError:
                    pass

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
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
        ok = False
        try:
            r = await client.get(_upstream_api_path(upstream_base, "models"), timeout=5.0)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        return JSONResponse(
            {"ok": ok, "upstream": upstream_base},
            status_code=200 if ok else 503,
        )

    if model_override:
        Middleware = _make_model_rewrite_middleware(model_override)
        return Middleware(app)
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Lean Kilo proxy for Gemma 4 + vllm-mlx",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--model", default=None,
        help="Rewrite request model field to this id (local weight dir name)",
    )
    parser.add_argument("--debug", action="store_true")
    # Accepted for CLI compatibility with older 2_start_mlx.sh flags.
    parser.add_argument("--debug-stream", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-thinking", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-guards", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info(
        "starting lean proxy → %s  model=%s  port=%d",
        args.upstream, args.model or "passthrough", args.port,
    )
    uvicorn.run(
        create_app(args.upstream, args.model),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
