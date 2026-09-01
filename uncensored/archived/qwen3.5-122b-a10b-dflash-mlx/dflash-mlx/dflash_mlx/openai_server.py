from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mlx.core as mx

log = logging.getLogger("dflash-openai")

# Agent clients (Kilo etc.) often omit max_tokens or send huge budgets.
# 256 is too small for tool-call JSON; uncapped 32k+ can run for many minutes.
# Tool-heavy agent turns rarely need more than a few hundred tokens of XML;
# keep defaults moderate so a single turn cannot run 10+ minutes.
DEFAULT_MAX_TOKENS = 4096
MAX_TOKENS_CEILING = 8192
# When tools are present, cap even lower. Kilo often sends max_tokens=8192
# explicitly — without a tools ceiling that burns many minutes on 122B.
DEFAULT_TOOLS_MAX_TOKENS = 1536
TOOLS_MAX_TOKENS_CEILING = 2048
# Reliability caps for 122B on Apple Silicon (prefill dominates wall time).
# Empirically: ~10k prompt ≈ 25–30s prefill; ~18k prompt ≈ 3–6+ minutes.
# Oversized agent histories are auto-trimmed to this budget (not hard-failed).
MAX_PROMPT_TOKENS = 12288
# Per-message content cap (tool dumps dominate Kilo context growth).
MAX_MESSAGE_CHARS = 8000
GENERATION_WALL_S = 120.0  # hard stop even if max_tokens not reached
TOOLS_GENERATION_WALL_S = 90.0


def adaptive_max_new_tokens(
    n_prompt: int,
    requested: int,
    *,
    tools: bool,
) -> int:
    """Shrink decode budget as prompt grows so turns stay interactive."""
    if n_prompt > 10000:
        cap = 256 if tools else 512
    elif n_prompt > 8000:
        cap = 512 if tools else 768
    elif n_prompt > 5000:
        cap = 768 if tools else 1024
    elif n_prompt > 3000:
        cap = 1024 if tools else 2048
    else:
        cap = requested
    return max(64, min(int(requested), int(cap)))


def _truncate_text(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 32
    return (
        text[:head]
        + "\n…[truncated for local 122B context budget]…\n"
        + text[-tail:]
    )


def shrink_message_contents(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = MAX_MESSAGE_CHARS,
) -> list[dict[str, Any]]:
    """Cap huge tool/user dumps that blow up agent context."""
    out: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        content = entry.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            entry["content"] = _truncate_text(content, max_chars)
        out.append(entry)
    return out


def trim_messages_to_budget(
    messages: list[dict[str, Any]],
    *,
    tools: list[Any] | None,
    count_tokens,
    max_tokens: int = MAX_PROMPT_TOKENS,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Keep system + newest turns until the rendered prompt fits.

    ``count_tokens(messages, tools) -> int`` measures the full chat-template size.
    Returns (messages, token_count, stats).
    """
    normalized = normalize_openai_messages(messages)
    shrunk = shrink_message_contents(normalized)
    content_truncated = any(
        (a.get("content") or "") != (b.get("content") or "")
        for a, b in zip(shrunk, normalized)
    )
    stats: dict[str, Any] = {
        "original_messages": len(normalized),
        "content_truncated": content_truncated,
    }

    n = count_tokens(shrunk, tools)
    if n <= max_tokens:
        stats.update({"trimmed_messages": 0, "final_tokens": n, "trimmed": False})
        return shrunk, n, stats

    systems = [m for m in shrunk if m.get("role") == "system"]
    rest = [m for m in shrunk if m.get("role") != "system"]
    dropped = 0

    # Drop oldest non-system messages until under budget (keep ≥1).
    while len(rest) > 1:
        n = count_tokens(systems + rest, tools)
        if n <= max_tokens:
            break
        rest.pop(0)
        dropped += 1

    fitted = systems + rest
    n = count_tokens(fitted, tools)

    # Still over: aggressively shrink remaining contents.
    if n > max_tokens:
        aggressive = shrink_message_contents(fitted, max_chars=max(1500, MAX_MESSAGE_CHARS // 4))
        fitted = aggressive
        n = count_tokens(fitted, tools)
        stats["aggressive_content_trim"] = True

    # Last resort: keep system + final message only, hard-trim content.
    if n > max_tokens and fitted:
        last = dict(fitted[-1])
        content = last.get("content") or ""
        if isinstance(content, str):
            # Binary-ish shrink until under or tiny
            budget_chars = 4000
            while budget_chars >= 500:
                trial_last = dict(last)
                trial_last["content"] = _truncate_text(content, budget_chars)
                trial = systems + [trial_last]
                n_trial = count_tokens(trial, tools)
                if n_trial <= max_tokens:
                    fitted = trial
                    n = n_trial
                    stats["last_message_chars"] = budget_chars
                    break
                budget_chars //= 2
            else:
                trial_last = dict(last)
                trial_last["content"] = _truncate_text(content, 400)
                fitted = systems + [trial_last]
                n = count_tokens(fitted, tools)

    stats.update(
        {
            "trimmed_messages": dropped,
            "final_messages": len(fitted),
            "final_tokens": n,
            "trimmed": dropped > 0 or stats.get("content_truncated") or n <= max_tokens,
        }
    )
    if n > max_tokens:
        # Only now fail — single turn still exceeds budget (tools schema alone, etc.)
        raise ValueError(
            f"Prompt still too large after auto-trim: {n} tokens "
            f"(max {max_tokens}). Tool schemas + system prompt may exceed the "
            f"budget alone; reduce tools or system prompt size."
        )
    return fitted, n, stats

# Qwen3.5 / Qwen3 tool XML emitted inside assistant content.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=([^>\s]+)\s*>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_health_response() -> dict[str, str]:
    return {"status": "ok"}


def build_models_response(model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "dflash-mlx",
            }
        ],
    }


def strip_think_blocks(text: str) -> str:
    """Remove Qwen think blocks (including empty ones from enable_thinking=false)."""
    cleaned = _THINK_RE.sub("", text or "")
    return cleaned.strip()


def _coerce_arg_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _openai_tool_call_entry(
    *,
    name: str,
    arguments: Any,
    call_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(arguments, str):
        args_str = arguments
    else:
        args_str = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_str,
        },
    }


def _normalize_parsed_tool_call_list(raw_list: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in raw_list:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        args = fn.get("arguments", tc.get("arguments", {}))
        out.append(
            _openai_tool_call_entry(
                name=str(name),
                arguments=args,
                call_id=tc.get("id") if isinstance(tc.get("id"), str) else None,
            )
        )
    return out


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse Qwen ``<tool_call>`` XML into OpenAI tool_calls.

    Returns (remaining_visible_content, tool_calls).
    """
    if not text or "<tool_call>" not in text.lower():
        return strip_think_blocks(text or ""), []

    tool_calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_BLOCK_RE.finditer(text):
        inner = block.group(1)
        for fn_match in _FUNCTION_BLOCK_RE.finditer(inner):
            name = fn_match.group(1).strip()
            params_blob = fn_match.group(2)
            args: dict[str, Any] = {}
            for pm in _PARAMETER_RE.finditer(params_blob):
                args[pm.group(1).strip()] = _coerce_arg_value(pm.group(2))
            # If no <parameter> tags, try whole function body as JSON
            if not args and params_blob.strip():
                try:
                    parsed = json.loads(params_blob.strip())
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        args = {"value": parsed}
                except json.JSONDecodeError:
                    args = {"raw": params_blob.strip()}
            tool_calls.append(_openai_tool_call_entry(name=name, arguments=args))

    remainder = _TOOL_CALL_BLOCK_RE.sub("", text)
    remainder = strip_think_blocks(remainder)
    return remainder, tool_calls


_BRACKET_TOOL_CALLS_RE = re.compile(
    r"\[tool_calls\]\s*(\[[\s\S]*\])\s*$",
    re.IGNORECASE,
)


def parse_bracket_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse legacy/plain ``[tool_calls] [{...}]`` dumps the model sometimes emits.

    This format was previously used in our plain-prompt fallback and can appear
    in assistant text; Kilo cannot execute it unless converted to real tool_calls.
    """
    cleaned = strip_think_blocks(text or "")
    if not cleaned or "[tool_calls]" not in cleaned.lower():
        return cleaned, []

    match = _BRACKET_TOOL_CALLS_RE.search(cleaned)
    if not match:
        # Mid-string variant (trailing prose after JSON is rare)
        match = re.search(
            r"\[tool_calls\]\s*(\[[\s\S]*?\])\s*(?:$|\n)",
            cleaned,
            re.IGNORECASE,
        )
    if not match:
        return cleaned, []

    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        # Best-effort close truncated arrays/objects
        candidate = raw_json
        if candidate.count("[") > candidate.count("]"):
            candidate += "]" * (candidate.count("[") - candidate.count("]"))
        if candidate.count("{") > candidate.count("}"):
            candidate += "}" * (candidate.count("{") - candidate.count("}"))
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return cleaned, []

    if not isinstance(parsed, list) or not parsed:
        return cleaned, []
    tool_calls = _normalize_parsed_tool_call_list(parsed)
    if not tool_calls:
        return cleaned, []
    remainder = (cleaned[: match.start()] + cleaned[match.end() :]).strip()
    return remainder, tool_calls


def parse_json_array_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse a bare JSON array of OpenAI-style tool_calls as the whole message."""
    cleaned = strip_think_blocks(text or "").strip()
    if not cleaned.startswith("["):
        return cleaned, []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned, []
    if not isinstance(parsed, list) or not parsed:
        return cleaned, []
    if not isinstance(parsed[0], dict):
        return cleaned, []
    if not (
        parsed[0].get("type") == "function"
        or isinstance(parsed[0].get("function"), dict)
        or parsed[0].get("name")
    ):
        return cleaned, []
    tool_calls = _normalize_parsed_tool_call_list(parsed)
    if not tool_calls:
        return cleaned, []
    return "", tool_calls


def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract tool calls from model text (Qwen XML or OpenAI-ish dumps)."""
    cleaned = strip_think_blocks(text or "")
    content, tool_calls = parse_qwen_tool_calls(cleaned)
    if tool_calls:
        return content, tool_calls
    content, tool_calls = parse_bracket_tool_calls(cleaned)
    if tool_calls:
        return content, tool_calls
    content, tool_calls = parse_json_array_tool_calls(cleaned)
    if tool_calls:
        return content, tool_calls
    return cleaned, []


def finalize_assistant_message(
    text: str,
    *,
    tools_requested: bool,
) -> tuple[str | None, list[dict[str, Any]] | None, str]:
    """Turn raw model text into OpenAI message fields.

    Returns (content, tool_calls_or_none, finish_reason).

    Always converts recognized tool-call dumps (XML or ``[tool_calls]``) when
    present so agents execute them instead of printing JSON as chat text.
    """
    cleaned = strip_think_blocks(text or "")
    content, tool_calls = parse_tool_calls_from_text(cleaned)
    if tool_calls:
        content_out: str | None = content if content else None
        return content_out, tool_calls, "tool_calls"
    if tools_requested:
        return cleaned or None, None, "stop"
    return cleaned, None, "stop"


def tool_calls_to_qwen_xml(tool_calls: list[dict[str, Any]]) -> str:
    """Render OpenAI tool_calls as Qwen chat-template XML for history fallback."""
    parts: list[str] = []
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or "unknown"
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args_obj = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args_obj = {"raw": args}
        elif isinstance(args, dict):
            args_obj = args
        else:
            args_obj = {"value": args}
        block = [f"<tool_call>\n<function={name}>"]
        if isinstance(args_obj, dict):
            for key, val in args_obj.items():
                if isinstance(val, (dict, list)):
                    rendered = json.dumps(val, ensure_ascii=False)
                else:
                    rendered = str(val)
                block.append(f"<parameter={key}>\n{rendered}\n</parameter>")
        block.append("</function>\n</tool_call>")
        parts.append("\n".join(block))
    return "\n".join(parts)


def build_chat_response(
    *,
    model: str,
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    created = int(time.time())
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_chat_stream_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def clamp_max_tokens(
    raw: Any,
    *,
    default: int = DEFAULT_MAX_TOKENS,
    ceiling: int = MAX_TOKENS_CEILING,
) -> int:
    """Normalize client max_tokens for agent use.

    - Missing / null → default (enough for tool JSON)
    - Below 1 → 1
    - Above ceiling → ceiling (prevents multi-hour generations)
    """
    if raw is None:
        return max(1, int(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_tokens must be an integer, got {raw!r}") from exc
    return max(1, min(value, int(ceiling)))


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Unsupported message content format for text-only server.")
            item_type = item.get("type", "text")
            if item_type != "text":
                raise ValueError("This DFlash OpenAI-compatible server is text-only.")
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("Text content parts must include a string 'text' field.")
            parts.append(text)
        return "\n".join(part for part in parts if part)
    raise ValueError("Unsupported message content format for text-only server.")


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    """Keep OpenAI tool_calls shape; ensure function.arguments is a dict for Jinja."""
    if not tool_calls:
        return None
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list.")
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            raise ValueError("Each tool_call must be an object.")
        entry = dict(tc)
        fn = entry.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    parsed = {"raw": args}
                fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
            elif args is None:
                fn["arguments"] = {}
            elif not isinstance(args, dict):
                fn["arguments"] = {"value": args}
            entry["function"] = fn
        normalized.append(entry)
    return normalized


def normalize_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Normalize client messages for Qwen chat templates (null content, tools)."""
    if not messages:
        raise ValueError("messages must not be empty")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must include a string role.")
        entry: dict[str, Any] = {"role": role}
        # content may be null on assistant tool-call turns
        if "content" in message:
            content = message.get("content")
            if content is None:
                entry["content"] = ""
            elif isinstance(content, str):
                entry["content"] = content
            else:
                entry["content"] = _extract_text_content(content)
        else:
            entry["content"] = ""

        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        if tool_calls is not None:
            entry["tool_calls"] = tool_calls
        if message.get("tool_call_id") is not None:
            entry["tool_call_id"] = message["tool_call_id"]
        if message.get("name") is not None:
            entry["name"] = message["name"]
        if message.get("reasoning_content") is not None:
            entry["reasoning_content"] = message["reasoning_content"]
        normalized.append(entry)
    return normalized


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Plain-text fallback when no chat template is available."""
    normalized = normalize_openai_messages(messages)
    lines: list[str] = []
    for message in normalized:
        role = message["role"]
        content = message.get("content") or ""
        if message.get("tool_calls"):
            # Use Qwen XML — never the `[tool_calls] [...]` dump (models copy it).
            xml = tool_calls_to_qwen_xml(message["tool_calls"])
            content = f"{content}\n{xml}".strip() if content else xml
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "tool"
            lines.append(f"Tool ({name}): {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[Any] | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render messages with the model chat template (no double-wrap)."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(messages, **kwargs)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


@dataclass
class GenerationChunk:
    delta: str
    text: str
    completion_tokens: int
    prompt_tokens: int = 0
    finish_reason: str | None = None
    finished: bool = False


class RunnerProtocol:
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        tools: list[Any] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        tools: list[Any] | None = None,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError


class DFlashRunner(RunnerProtocol):
    """Thread-safe wrapper: one MLX generation at a time (prevents Metal hangs)."""

    def __init__(
        self,
        *,
        target_model: str,
        draft_model: str,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        seed: int = 0,
        enable_thinking: bool = False,
    ):
        from .api import DFlashGenerator

        self.generator = DFlashGenerator(
            target_model=target_model,
            draft_model=draft_model,
            seed=seed,
        )
        self.speculative_tokens = speculative_tokens
        self.verify_mode = verify_mode
        self.verify_chunk_size = verify_chunk_size
        self.enable_thinking = enable_thinking
        self._lock = threading.Lock()
        self._busy = False
        self._busy_since: float | None = None
        self._busy_meta: dict[str, Any] = {}

    @property
    def busy(self) -> bool:
        return self._busy

    def status(self) -> dict[str, Any]:
        meta = dict(self._busy_meta)
        if self._busy and self._busy_since is not None:
            meta["busy_for_s"] = round(time.time() - self._busy_since, 2)
        return {"busy": self._busy, **meta}

    def _count_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None,
    ) -> int:
        """Token count only (for budget fitting)."""
        _, _, n = self._prompt_tokens(messages, tools, already_normalized=True)
        return n

    def _prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None,
        *,
        already_normalized: bool = False,
    ) -> tuple[mx.array, str, int]:
        """Build token ids via chat template — avoid double-wrapping as a single user msg."""
        tokenizer = self.generator.target.tokenizer
        normalized = messages if already_normalized else normalize_openai_messages(messages)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = apply_chat_template_text(
                    tokenizer,
                    normalized,
                    tools=tools,
                    enable_thinking=self.enable_thinking,
                )
                token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                return (
                    mx.array(token_ids, dtype=mx.uint32),
                    prompt_text,
                    len(token_ids),
                )
            except Exception as exc:
                log.warning("chat_template failed (%s); using plain prompt join", exc)
        # Fallback: plain join then adapter single-user wrap
        plain = messages_to_prompt(normalized)
        tokens = self.generator.encode_prompt(plain)
        return tokens, plain, int(tokens.shape[0])

    def _prepare_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None,
        max_new_tokens: int,
    ) -> tuple[mx.array, str, int, int]:
        """Tokenize, auto-trim agent history to budget, adapt decode budget.

        Returns (prompt_tokens, prompt_text, n_prompt, max_new_tokens_adapted).
        """
        fitted, measured, trim_stats = trim_messages_to_budget(
            messages,
            tools=tools,
            count_tokens=self._count_prompt_tokens,
            max_tokens=MAX_PROMPT_TOKENS,
        )
        if trim_stats.get("trimmed_messages") or trim_stats.get("content_truncated"):
            log.warning(
                "auto-trimmed context: msgs %s→%s dropped=%s final_tokens=%s "
                "(cap=%s) — oldest history/tool dumps removed for reliability",
                trim_stats.get("original_messages"),
                trim_stats.get("final_messages"),
                trim_stats.get("trimmed_messages"),
                trim_stats.get("final_tokens"),
                MAX_PROMPT_TOKENS,
            )
        prompt_tokens, prompt_text, n_prompt = self._prompt_tokens(
            fitted, tools, already_normalized=True
        )
        n_prompt = int(prompt_tokens.shape[0])
        del measured  # counted during fit; tensor length is authoritative

        adapted = adaptive_max_new_tokens(
            n_prompt, max_new_tokens, tools=bool(tools)
        )
        if adapted < max_new_tokens:
            log.info(
                "adapted max_new_tokens %d → %d (prompt_tokens=%d tools=%s)",
                max_new_tokens,
                adapted,
                n_prompt,
                bool(tools),
            )
        self._busy_meta = {
            **self._busy_meta,
            "context_trimmed": bool(
                trim_stats.get("trimmed_messages") or trim_stats.get("content_truncated")
            ),
            "trim": {
                k: trim_stats[k]
                for k in (
                    "original_messages",
                    "final_messages",
                    "trimmed_messages",
                    "final_tokens",
                )
                if k in trim_stats
            },
        }
        return prompt_tokens, prompt_text, n_prompt, adapted

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        tools: list[Any] | None = None,
    ) -> GenerationResult:
        acquired = self._lock.acquire(blocking=True)
        if not acquired:  # pragma: no cover
            raise RuntimeError("Failed to acquire generation lock")
        self._busy = True
        self._busy_since = time.time()
        t0 = time.perf_counter()
        try:
            prompt_tokens, prompt_text, n_prompt, max_new_tokens = (
                self._prepare_generation(messages, tools, max_new_tokens)
            )
            wall = TOOLS_GENERATION_WALL_S if tools else GENERATION_WALL_S
            self._busy_meta = {
                "mode": "generate",
                "max_new_tokens": max_new_tokens,
                "prompt_tokens": n_prompt,
                "prompt_chars": len(prompt_text),
                "messages": len(messages),
                "wall_s": wall,
            }
            log.info(
                "generate start msgs=%d prompt_tokens=%d max_new_tokens=%d temp=%.2f wall=%.0fs",
                len(messages),
                n_prompt,
                max_new_tokens,
                temperature,
                wall,
            )
            # Non-stream path: no mid-gen cancel; keep budget tight via adaptive cap.
            result = self.generator.generate_from_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                speculative_tokens=self.speculative_tokens,
                verify_mode=self.verify_mode,
                verify_chunk_size=self.verify_chunk_size,
                skip_special_tokens=True,
            )
            prompt_tokens_n = int(result.metrics.get("num_input_tokens", n_prompt))
            completion_tokens = len(result.generated_tokens)
            finish_reason = str(result.metrics.get("finish_reason", "stop"))
            if finish_reason == "max_tokens":
                finish_reason = "length"
            elapsed = time.perf_counter() - t0
            log.info(
                "generate done completion_tokens=%d finish=%s elapsed=%.2fs",
                completion_tokens,
                finish_reason,
                elapsed,
            )
            return GenerationResult(
                text=result.text,
                prompt_tokens=prompt_tokens_n,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
        except ValueError:
            # Client errors (prompt too large, etc.) — no stack spam
            raise
        except Exception:
            log.exception(
                "generate failed after %.2fs", time.perf_counter() - t0
            )
            raise
        finally:
            self._busy = False
            self._busy_since = None
            self._busy_meta = {}
            self._lock.release()

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        tools: list[Any] | None = None,
    ) -> Iterator[GenerationChunk]:
        self._lock.acquire()
        self._busy = True
        self._busy_since = time.time()
        t0 = time.perf_counter()
        try:
            prompt_tokens, prompt_text, n_prompt, max_new_tokens = (
                self._prepare_generation(messages, tools, max_new_tokens)
            )
            wall = TOOLS_GENERATION_WALL_S if tools else GENERATION_WALL_S
            self._busy_meta = {
                "mode": "stream",
                "max_new_tokens": max_new_tokens,
                "prompt_tokens": n_prompt,
                "prompt_chars": len(prompt_text),
                "messages": len(messages),
                "wall_s": wall,
            }
            log.info(
                "stream start msgs=%d prompt_tokens=%d max_new_tokens=%d temp=%.2f wall=%.0fs",
                len(messages),
                n_prompt,
                max_new_tokens,
                temperature,
                wall,
            )
            last_text = ""
            last_completion = 0
            for event in self.generator.stream_from_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                speculative_tokens=self.speculative_tokens,
                verify_mode=self.verify_mode,
                verify_chunk_size=self.verify_chunk_size,
                skip_special_tokens=True,
            ):
                elapsed = time.perf_counter() - t0
                if event.finished:
                    if event.delta:
                        yield GenerationChunk(
                            delta=event.delta,
                            text=event.text,
                            completion_tokens=len(event.generated_tokens),
                            prompt_tokens=n_prompt,
                        )
                    metrics = event.metrics or {}
                    prompt_tokens_n = int(metrics.get("num_input_tokens", n_prompt))
                    finish_reason = str(metrics.get("finish_reason", "stop"))
                    if finish_reason == "max_tokens":
                        finish_reason = "length"
                    yield GenerationChunk(
                        delta="",
                        text=event.text,
                        prompt_tokens=prompt_tokens_n,
                        completion_tokens=len(event.generated_tokens),
                        finish_reason=finish_reason,
                        finished=True,
                    )
                    break
                last_text = event.text
                last_completion = len(event.generated_tokens)
                self._busy_meta["completion_tokens"] = last_completion
                self._busy_meta["elapsed_s"] = round(elapsed, 1)
                yield GenerationChunk(
                    delta=event.delta,
                    text=event.text,
                    completion_tokens=last_completion,
                    prompt_tokens=n_prompt,
                )
                # Wall-clock guard: return whatever we have so agents unstick.
                if elapsed >= wall:
                    log.warning(
                        "wall-clock stop after %.1fs (limit=%.0fs) "
                        "completion_tokens=%d prompt_tokens=%d",
                        elapsed,
                        wall,
                        last_completion,
                        n_prompt,
                    )
                    yield GenerationChunk(
                        delta="",
                        text=last_text,
                        prompt_tokens=n_prompt,
                        completion_tokens=last_completion,
                        finish_reason="length",
                        finished=True,
                    )
                    break
            log.info("stream done elapsed=%.2fs", time.perf_counter() - t0)
        except GeneratorExit:
            log.info(
                "stream closed after %.2fs (early-stop or client cancel)",
                time.perf_counter() - t0,
            )
            raise
        except Exception:
            log.exception("stream failed after %.2fs", time.perf_counter() - t0)
            raise
        finally:
            self._busy = False
            self._busy_since = None
            self._busy_meta = {}
            self._lock.release()


@dataclass
class ServerConfig:
    host: str
    port: int
    model_id: str
    runner: RunnerProtocol
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    max_tokens_ceiling: int = MAX_TOKENS_CEILING


def make_handler(config: ServerConfig):
    class OpenAIHandler(BaseHTTPRequestHandler):
        server_version = "dflash-mlx-openai/0.2"
        protocol_version = "HTTP/1.1"

        def handle(self) -> None:
            # Keep-alive clients often RST after cancel; don't spam the log.
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                log.info("client disconnected while sending JSON status=%s", status)

        def _send_sse_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _write_sse(self, payload: dict[str, Any] | str) -> None:
            data = payload if isinstance(payload, str) else json.dumps(payload)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Invalid Content-Length.")
            # Guard against runaway bodies (agents can send large histories; 64 MiB cap)
            if length > 64 * 1024 * 1024:
                raise ValueError("Request body too large (max 64 MiB).")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                raise ValueError("Request body is required.")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                body = build_health_response()
                runner = config.runner
                if hasattr(runner, "status"):
                    body = {**body, **runner.status()}  # type: ignore[operator]
                self._send_json(HTTPStatus.OK, body)
                return
            if path == "/v1/models":
                self._send_json(HTTPStatus.OK, build_models_response(config.model_id))
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": "Not found", "type": "not_found_error"}},
            )

        def _emit_tool_call_stream(
            self,
            *,
            chunk_id: str,
            created: int,
            model: str,
            content: str | None,
            tool_calls: list[dict[str, Any]],
            usage: dict[str, int],
        ) -> None:
            """Emit OpenAI-style streaming tool_calls (agents need this, not raw XML)."""
            self._write_sse(
                build_chat_stream_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"role": "assistant"},
                )
            )
            if content:
                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={"content": content},
                    )
                )
            for index, tc in enumerate(tool_calls):
                fn = tc.get("function") or {}
                # First delta: id + name
                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                    )
                )
                # Second delta: arguments payload
                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {
                                        "arguments": fn.get("arguments", "{}"),
                                    },
                                }
                            ]
                        },
                    )
                )
            self._write_sse(
                build_chat_stream_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={},
                    finish_reason="tool_calls",
                    usage=usage,
                )
            )
            self._write_sse("[DONE]")

        def _send_streaming_chat(
            self,
            *,
            messages: list[dict[str, Any]],
            model: str,
            max_new_tokens: int,
            temperature: float,
            tools: list[Any] | None,
        ) -> None:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            tools_requested = bool(tools)
            self._send_sse_headers()
            try:
                # When tools are present, buffer text then convert Qwen XML →
                # OpenAI tool_calls. Live-streaming raw <tool_call> breaks Kilo.
                # Early-stop as soon as a complete </tool_call> appears so we do
                # not burn the full max_tokens budget after the model already
                # emitted a usable tool call (common agent failure mode).
                if tools_requested:
                    final_chunk: GenerationChunk | None = None
                    last_chunk: GenerationChunk | None = None
                    stream_iter = config.runner.stream(
                        messages=messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        tools=tools,
                    )
                    try:
                        for chunk in stream_iter:
                            last_chunk = chunk
                            if chunk.finished:
                                final_chunk = chunk
                                break
                            # Complete tool call(s) already in the text (XML or dump)
                            lower = chunk.text.lower()
                            if (
                                "</tool_call>" in lower
                                or "[tool_calls]" in lower
                            ):
                                content_chk, tool_calls_chk, _ = (
                                    finalize_assistant_message(
                                        chunk.text, tools_requested=True
                                    )
                                )
                                if tool_calls_chk:
                                    log.info(
                                        "early-stop stream after complete tool_call "
                                        "(%d calls, %d completion tokens so far)",
                                        len(tool_calls_chk),
                                        chunk.completion_tokens,
                                    )
                                    final_chunk = GenerationChunk(
                                        delta="",
                                        text=chunk.text,
                                        prompt_tokens=chunk.prompt_tokens,
                                        completion_tokens=chunk.completion_tokens,
                                        finish_reason="tool_calls",
                                        finished=True,
                                    )
                                    break
                    finally:
                        # Closing the generator releases the MLX lock promptly
                        # (GeneratorExit in DFlashRunner.stream).
                        close = getattr(stream_iter, "close", None)
                        if callable(close):
                            close()
                    if final_chunk is None:
                        if last_chunk is not None:
                            final_chunk = GenerationChunk(
                                delta="",
                                text=last_chunk.text,
                                prompt_tokens=last_chunk.prompt_tokens,
                                completion_tokens=last_chunk.completion_tokens,
                                finish_reason="stop",
                                finished=True,
                            )
                        else:
                            raise RuntimeError(
                                "Streaming generation did not produce a final chunk."
                            )
                    content, tool_calls, finish_reason = finalize_assistant_message(
                        final_chunk.text,
                        tools_requested=True,
                    )
                    usage = {
                        "prompt_tokens": final_chunk.prompt_tokens,
                        "completion_tokens": final_chunk.completion_tokens,
                        "total_tokens": (
                            final_chunk.prompt_tokens + final_chunk.completion_tokens
                        ),
                    }
                    if tool_calls:
                        log.info(
                            "stream converted %d Qwen tool_call(s) → OpenAI tool_calls",
                            len(tool_calls),
                        )
                        self._emit_tool_call_stream(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            content=content,
                            tool_calls=tool_calls,
                            usage=usage,
                        )
                        return
                    # No tool calls after all — stream text like a normal reply
                    self._write_sse(
                        build_chat_stream_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta={"role": "assistant"},
                        )
                    )
                    if content:
                        self._write_sse(
                            build_chat_stream_chunk(
                                chunk_id=chunk_id,
                                created=created,
                                model=model,
                                delta={"content": content},
                            )
                        )
                    self._write_sse(
                        build_chat_stream_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta={},
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                    )
                    self._write_sse("[DONE]")
                    return

                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={"role": "assistant"},
                    )
                )

                final_chunk = None
                for chunk in config.runner.stream(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=tools,
                ):
                    if chunk.finished:
                        final_chunk = chunk
                        continue
                    if not chunk.delta:
                        continue
                    # Strip think markers from live stream if they appear
                    delta_text = chunk.delta
                    if "<think>" in delta_text or "</think>" in delta_text:
                        continue
                    self._write_sse(
                        build_chat_stream_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta={"content": delta_text},
                        )
                    )
                if final_chunk is None:
                    raise RuntimeError("Streaming generation did not produce a final chunk.")
                usage = {
                    "prompt_tokens": final_chunk.prompt_tokens,
                    "completion_tokens": final_chunk.completion_tokens,
                    "total_tokens": final_chunk.prompt_tokens + final_chunk.completion_tokens,
                }
                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={},
                        finish_reason=final_chunk.finish_reason or "stop",
                        usage=usage,
                    )
                )
                self._write_sse("[DONE]")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                log.info("client disconnected during stream id=%s", chunk_id)
            except Exception as exc:  # pragma: no cover - sent after headers
                log.exception("stream handler error")
                try:
                    self._write_sse(
                        {
                            "error": {
                                "message": str(exc),
                                "type": "server_error",
                            }
                        }
                    )
                    self._write_sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/v1/chat/completions":
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "Not found", "type": "not_found_error"}},
                )
                return
            try:
                payload = self._read_json()
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("'messages' must be a list.")
                # Validate/normalize early so bad agent payloads fail fast
                normalize_openai_messages(messages)

                model = str(payload.get("model") or config.model_id)
                tools = payload.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError("'tools' must be a list when provided.")
                raw_max = payload.get("max_tokens")
                if raw_max is None:
                    raw_max = payload.get("max_completion_tokens")
                # Tool turns: tight default + hard ceiling even when Kilo sends 8192.
                # A single tool_call XML is usually <100 tokens; long monologues after
                # that waste wall-clock on large prompts (15k–20k tokens).
                default_max = config.default_max_tokens
                ceiling = config.max_tokens_ceiling
                if tools:
                    default_max = min(default_max, DEFAULT_TOOLS_MAX_TOKENS)
                    ceiling = min(ceiling, TOOLS_MAX_TOKENS_CEILING)
                max_new_tokens = clamp_max_tokens(
                    raw_max,
                    default=default_max,
                    ceiling=ceiling,
                )
                temperature = float(payload.get("temperature", 0.0))
                stream = bool(payload.get("stream"))

                if (
                    hasattr(config.runner, "busy")
                    and config.runner.busy  # type: ignore[attr-defined]
                ):
                    # Still queue behind the lock; log so operators see backlog.
                    log.info(
                        "request queued while generation busy: %s",
                        getattr(config.runner, "status", lambda: {})(),
                    )

                log.info(
                    "chat completion: msgs=%d max_tokens=%d temp=%.2f stream=%s tools=%s",
                    len(messages),
                    max_new_tokens,
                    temperature,
                    stream,
                    len(tools) if tools else 0,
                )

                if stream:
                    self._send_streaming_chat(
                        messages=messages,
                        model=model,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        tools=tools,
                    )
                    return
                generation = config.runner.generate(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=tools,
                )
                content, tool_calls, finish_reason = finalize_assistant_message(
                    generation.text,
                    tools_requested=bool(tools),
                )
                if tool_calls:
                    log.info(
                        "converted %d Qwen tool_call(s) → OpenAI tool_calls",
                        len(tool_calls),
                    )
                    # Prefer tool_calls finish over length/stop from decoder
                    if generation.finish_reason == "length":
                        finish_reason = "length"
                elif generation.finish_reason == "length":
                    finish_reason = "length"
                response = build_chat_response(
                    model=model,
                    content=content if content is not None else ("" if not tool_calls else None),
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )
                self._send_json(HTTPStatus.OK, response)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(exc), "type": "invalid_request_error"}},
                )
            except Exception as exc:  # pragma: no cover - safety net for runtime errors
                log.exception("request failed")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": str(exc), "type": "server_error"}},
                )

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("[dflash-openai] " + format % args + "\n")

    return OpenAIHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible HTTP server for dflash-mlx.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--model-id", default="dflash-mlx")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-speculative-tokens", type=int, default=None)
    parser.add_argument(
        "--verify-mode",
        choices=["stream", "chunked", "parallel-replay", "parallel-lazy-logits", "parallel-greedy-argmax"],
        default="parallel-replay",
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Used when client omits max_tokens (agent-friendly default).",
    )
    parser.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=MAX_TOKENS_CEILING,
        help="Hard cap even if client requests more.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave Qwen thinking mode open (slower; not recommended for agents).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log.info(
        "Loading DFlash runner target=%s draft=%s (max_tokens default=%d ceiling=%d)",
        args.target_model,
        args.draft_model,
        args.default_max_tokens,
        args.max_tokens_ceiling,
    )
    runner = DFlashRunner(
        target_model=args.target_model,
        draft_model=args.draft_model,
        speculative_tokens=args.max_speculative_tokens,
        verify_mode=args.verify_mode,
        verify_chunk_size=args.verify_chunk_size,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        runner=runner,
        default_max_tokens=args.default_max_tokens,
        max_tokens_ceiling=args.max_tokens_ceiling,
    )
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    # One-shot requests; avoid idle keep-alive stalls with cancelled agent clients
    server.request_queue_size = 32
    print(
        f"Serving DFlash OpenAI-compatible API on http://{config.host}:{config.port} "
        f"(serialized generation, max_tokens default={config.default_max_tokens} "
        f"ceiling={config.max_tokens_ceiling})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
