#!/usr/bin/env python3
"""Lean Kilo harness proxy for Gemma 4 AtomicChat + mlx_lm/mlx_vlm.

Sits between Kilo and the MLX OpenAI server:

  Kilo  →  :8080 this proxy  →  :8090 mlx_lm.server / mlx_vlm.server

Why this exists
---------------
Without a proxy, Kilo compaction turns often still ship ``tools`` +
``tool_choice=auto``. Gemma then *explores with tools* instead of writing a
short summary. Context never shrinks (logs: ``pruned=0``), auto-compaction
retries thrice, then:

  ContextOverflowError: Compaction exhausted: context still exceeds model limits

This proxy:

  1. Detects compaction / summary turns and strips tools (tool_choice none).
  2. Caps compaction ``max_tokens`` so summaries stay short.
  3. Truncates large ``role=tool`` message bodies in the outbound prompt
     (extra safety beyond Kilo ``tool_output`` caps).
  4. Passes through ``/v1/*`` including streaming SSE.

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


def _looks_like_compaction(body: dict) -> bool:
    if _tool_choice_disallows_tools(body.get("tool_choice")):
        return True
    # Prefer system + latest user only (full history false-positives on "summarize").
    parts: list[str] = []
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            t = _message_text(msg)
            if t:
                parts.append(t)
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            t = _message_text(msg)
            if t:
                parts.append(t)
            break
    blob = "\n".join(parts).lower()
    if not blob:
        return False
    if any(h in blob for h in _COMPACTION_HINTS):
        return True
    # Regex fallbacks for slight prompt wording drift
    return bool(
        re.search(r"summariz(?:e|ing)\b.*\b(?:conversation|session|history|context|chat)\b", blob)
        or re.search(r"\bcompact(?:ion|ing)?\b.*\b(?:context|session|conversation|history)\b", blob)
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


def _prepare_body(body: dict) -> None:
    messages = body.get("messages")
    if isinstance(messages, list):
        truncated = _truncate_tool_messages(messages)
        if truncated:
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
        # Lower temp for denser summaries
        body.setdefault("temperature", 0.2)
        log.info("[mode] compaction → tools stripped, short max_tokens")
        return

    # Agentic turns: leave tools alone; optional light temp floor (mlx_lm default is 0.35).
    temp = body.get("temperature")
    try:
        t = float(temp) if temp is not None else None
    except (TypeError, ValueError):
        t = None
    if t is not None and t < 0.2 and body.get("tools"):
        body["temperature"] = 0.35
        log.info("[temp] raised agentic temperature %.2f → 0.35", t)


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
                    _prepare_body(data)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"

        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{self.state.host}:{self.state.port}"
        headers["Connection"] = "close"

        conn = _connect(self.state)
        try:
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_headers = {k: v for k, v in resp.getheaders()}
            content_type = resp_headers.get("Content-Type", "")

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
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                payload = resp.read()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except Exception as exc:
            log.exception("upstream error: %s", exc)
            try:
                self._send_json(502, {"error": {"message": str(exc), "type": "proxy_error"}})
            except Exception:
                pass
        finally:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gemma4 AtomicChat ↔ Kilo harness proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--upstream",
        default="http://127.0.0.1:8090",
        help="mlx_lm/mlx_vlm base URL (no trailing /v1)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state = ProxyState(args.upstream)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(
        "gemma4 kilo proxy on http://%s:%d → %s (compaction tool-strip ON)",
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
    sys.exit(main())
