#!/usr/bin/env python3
"""OpenAI-compatible harness proxy for ds4-server + Kilo/OpenCode.

Why this exists
---------------
ds4-server defaults DeepSeek chat requests to **high-effort thinking**. In
thinking mode the model can burn the generation budget on chain-of-thought,
truncate tool-call JSON (Kilo: "JSON Parse error: Expected '}'"), or end the
stream without a clean finish_reason — agents "diagnose" bugs then abort mid-fix.

This proxy sits between Kilo and ds4-server and:

  1. Defaults **thinking off** for reliable tool calls
     (``thinking: {type: disabled}``, ``think: false``).
     Opt back in with ``"think": true`` or ``"reasoning_effort": "high"``.
  2. Ensures a floor on ``max_tokens`` so tool argument JSON can finish.
  3. Strips tools on Kilo compaction / summary turns (tool_choice none).
  4. Soft-repairs truncated tool ``function.arguments`` JSON by closing braces.
  5. Passes through ``/v1/*`` (including streaming SSE).

Usage:
  python3 ds4_kilo_proxy.py --upstream http://127.0.0.1:18083 --port 8083
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("ds4_kilo_proxy")

DEFAULT_MAX_TOKENS = 8192
MIN_MAX_TOKENS = 4096

_EFFORT_OFF = frozenset({"none", "off", "false", "0", "minimal", "disabled"})
_EFFORT_ON = frozenset({"low", "medium", "high", "max", "true", "1"})

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
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("system", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
            content = "\n".join(parts)
        if not isinstance(content, str):
            continue
        low = content.lower()
        if any(h in low for h in _COMPACTION_HINTS):
            return True
    return False


def _client_wants_thinking(body: dict) -> bool | None:
    """True / False if client expressed a preference; None if unset."""
    if "think" in body:
        return bool(body["think"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        t = str(thinking.get("type", "")).lower()
        if t in ("disabled", "none", "off"):
            return False
        if t in ("enabled", "high", "max", "auto"):
            return True
    effort = body.get("reasoning_effort")
    if effort is not None:
        e = str(effort).strip().lower()
        if e in _EFFORT_OFF:
            return False
        if e in _EFFORT_ON or e:
            return True
    return None


def _disable_thinking(body: dict) -> None:
    body["think"] = False
    body["thinking"] = {"type": "disabled"}
    body["reasoning_effort"] = "none"
    # Some DeepSeek clients use model alias for non-thinking
    # Leave model id alone — ds4 maps both flash/pro to the loaded GGUF.


def _enable_thinking(body: dict) -> None:
    body["think"] = True
    body.pop("thinking", None)
    if "reasoning_effort" not in body:
        body["reasoning_effort"] = "high"


def _ensure_max_tokens(body: dict) -> None:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        body["max_tokens"] = DEFAULT_MAX_TOKENS
        log.info("[tokens] default max_tokens=%d", DEFAULT_MAX_TOKENS)
    elif value < MIN_MAX_TOKENS:
        body["max_tokens"] = MIN_MAX_TOKENS
        log.info("[tokens] raised max_tokens %d → %d", value, MIN_MAX_TOKENS)


def _prepare_body(body: dict) -> None:
    if _looks_like_compaction(body):
        body.pop("tools", None)
        body["tool_choice"] = "none"
        _disable_thinking(body)
        _ensure_max_tokens(body)
        log.info("[mode] compaction → tools stripped, thinking off")
        return

    _ensure_max_tokens(body)

    want = _client_wants_thinking(body)
    if want is True:
        _enable_thinking(body)
        log.info("[think] client → enabled")
    else:
        _disable_thinking(body)
        if want is False:
            log.info("[think] client → disabled")
        else:
            log.info("[think] default → disabled (agent-safe)")


def _try_close_json(s: str) -> str | None:
    """Best-effort repair of truncated JSON objects/arrays for tool arguments."""
    s = s.strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Drop trailing incomplete string if cut mid-value
    if s.count('"') % 2 == 1:
        s = s + '"'

    # Remove trailing comma before we close
    s = re.sub(r",\s*$", "", s)

    opens = 0
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
            opens += 1
            closes_needed_obj += 1
        elif ch == "}":
            closes_needed_obj = max(0, closes_needed_obj - 1)
        elif ch == "[":
            opens += 1
            closes_needed_arr += 1
        elif ch == "]":
            closes_needed_arr = max(0, closes_needed_arr - 1)

    candidate = s + ("]" * closes_needed_arr) + ("}" * closes_needed_obj)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def _repair_tool_calls_in_message(msg: dict) -> bool:
    changed = False
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list):
        return False
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        if not isinstance(args, str):
            continue
        try:
            json.loads(args)
            continue
        except json.JSONDecodeError:
            pass
        fixed = _try_close_json(args)
        if fixed is not None and fixed != args:
            fn["arguments"] = fixed
            changed = True
            log.info(
                "[repair] tool %s arguments closed (%d→%d chars)",
                fn.get("name", "?"),
                len(args),
                len(fixed),
            )
    return changed


def _repair_response_payload(payload: bytes) -> bytes:
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
        if isinstance(msg, dict) and _repair_tool_calls_in_message(msg):
            changed = True
    if not changed:
        return payload
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


class ProxyState:
    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        parsed = urlparse(self.upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise SystemExit(f"Invalid upstream URL: {upstream}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")  # usually empty or /v1 stripped


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
        # Kilo talks to baseURL .../v1 — forward path as-is.
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
                if "application/json" in content_type:
                    payload = _repair_response_payload(payload)
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
    p = argparse.ArgumentParser(description="ds4 ↔ Kilo agent harness proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument(
        "--upstream",
        default="http://127.0.0.1:18083",
        help="ds4-server base (no trailing /v1)",
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
        "ds4 kilo proxy on http://%s:%d → %s (thinking default OFF)",
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
    # Silence unused import warning for threading — ThreadingHTTPServer uses it.
    _ = threading
    sys.exit(main())
