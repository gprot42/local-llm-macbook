#!/usr/bin/env python3
"""Stop an OpenCode turn that is not making progress.

OpenCode's doom_loop check only looks at tool calls inside one assistant
message, and one call per step never trips it. Alternating two commands
never trips it either. This proxy sits in front of TensorFold and, after
the latest user message, refuses to call the model when:

- the same tool call has returned the same result twice with no write/edit in
  between (nothing changed, so repeating it cannot help; an edit between two
  identical `node --check` runs is a normal verify loop and is forwarded)
- the same tool call has been issued three times, even with different output
- the turn has already run 48 tool calls

A new user message starts the count over. Upstream bytes are relayed as they
arrive (read1), so streamed tokens reach the client immediately.

  python3 loop_proxy.py --self-test
  python3 loop_proxy.py --listen 8769 --upstream 127.0.0.1:8779
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import http.client
import uuid

STOP_AFTER = 2
CYCLE_REPEAT = 3
MAX_TOOL_ROUNDS = 48
# Tools that change files. An identical call repeated after one of these is a
# legitimate re-verification, not a loop.
STATE_CHANGING = {"write", "edit", "patch", "multiedit", "apply_patch"}

STOP_TEXT = (
    "[Harness] Stopped: the same tool call already ran "
    "{n} times with the same result. That result is above. "
    "Do not run it again. Change the approach, or report what it printed."
)
CYCLE_TEXT = (
    "[Harness] Stopped: the same tool call was issued {n} times this turn, "
    "mixed with other calls. That is a loop. Change the approach, or report "
    "what the last result printed."
)
CAP_TEXT = (
    "[Harness] Stopped: this turn already ran {n} tool calls without "
    "finishing. Report what is done and what is blocked. Do not call another tool."
)


def _canonical_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return str(arguments).strip()


def _tool_sig(call: dict[str, Any]) -> tuple[str, str]:
    fn = call.get("function") or {}
    name = str(fn.get("name") or call.get("name") or "")
    return str(call.get("id") or ""), name + "\n" + _canonical_args(fn.get("arguments", call.get("arguments", "")))


def _norm_result(content: Any) -> str:
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    return " ".join(content.split())


def _rounds_after_last_user(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Tool calls after the latest user message, in order, paired with results."""

    last_user = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = index
    pending: dict[str, str] = {}
    rounds: list[tuple[str, str]] = []
    for message in messages[last_user + 1:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id, sig = _tool_sig(call)
                if call_id:
                    pending[call_id] = sig
                else:
                    rounds.append((sig, ""))
        elif role == "tool":
            sig = pending.pop(str(message.get("tool_call_id") or ""), None)
            if sig is None:
                continue
            rounds.append((sig, _norm_result(message.get("content") or "")))
    return rounds


def stop_reason(messages: list[dict[str, Any]]) -> str | None:
    """Why this request should not reach the model, or None to forward it.

    Only the turns after the latest user message count. An older repeated
    scan stays in the history, and a new prompt must still reach the model.
    """

    rounds = _rounds_after_last_user(messages)
    if len(rounds) >= MAX_TOOL_ROUNDS:
        return CAP_TEXT.format(n=len(rounds))
    by_sig: dict[str, list[int]] = {}
    for index, (sig, _) in enumerate(rounds):
        by_sig.setdefault(sig, []).append(index)
    for indices in by_sig.values():
        if len(indices) < STOP_AFTER:
            continue
        prev, last = indices[-2], indices[-1]
        same_result = rounds[prev][1] == rounds[last][1]
        changed_between = any(_tool_name(rounds[k][0]) in STATE_CHANGING for k in range(prev + 1, last))
        if same_result and not changed_between:
            return STOP_TEXT.format(n=len(indices))
    for indices in by_sig.values():
        if len(indices) >= CYCLE_REPEAT:
            return CYCLE_TEXT.format(n=len(indices))
    return None


def _tool_name(sig: str) -> str:
    return sig.split("\n", 1)[0]


def _completion(text: str, *, model: str, stream: bool) -> bytes:
    completion_id = "chatcmpl-loopstop-" + uuid.uuid4().hex[:12]
    if not stream:
        body = {
            "id": completion_id,
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }
        return json.dumps(body).encode()
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    payload = b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
    return payload + b"data: [DONE]\n\n"


class Proxy(BaseHTTPRequestHandler):
    upstream_host = "127.0.0.1"
    upstream_port = 8779
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[loop-proxy] " + (fmt % args) + "\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._forward(b"")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length else b""
        if self.path.split("?", 1)[0].rstrip("/").endswith("/chat/completions") and body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                reason = stop_reason(payload.get("messages") or [])
                if reason:
                    self._send_stop(payload, reason)
                    return
        self._forward(body)

    def _send_stop(self, payload: dict[str, Any], reason: str) -> None:
        stream = bool(payload.get("stream"))
        model = str(payload.get("model") or "local-model")
        raw = _completion(reason, model=model, stream=stream)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()
        self.log_message("stopped repeated tool call (%d bytes)", len(raw))

    def _forward(self, body: bytes) -> None:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
        if body:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=3600)
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            message = json.dumps({"error": {"message": f"tensorfold upstream unavailable: {exc}", "type": "server_error"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        # read1 returns as soon as any bytes are available; read(n) would block
        # until n bytes or EOF and turn a token stream into one late flush.
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()


def _self_test() -> None:
    call = lambda n, args, cid: {  # noqa: E731
        "role": "assistant",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": n, "arguments": args}}],
    }
    result = lambda cid, text: {"role": "tool", "tool_call_id": cid, "content": text}  # noqa: E731
    same = json.dumps({"command": "python3 scan.py"})
    once = [call("bash", same, "a"), result("a", "total symbols: 268")]
    assert stop_reason(once) is None
    twice = once + [call("bash", same, "b"), result("b", "total symbols: 268")]
    assert stop_reason(twice) is not None
    # A new user prompt after an old loop must be forwarded.
    assert stop_reason(twice + [{"role": "user", "content": "try something else"}]) is None
    resumed = twice + [{"role": "user", "content": "try something else"}, call("bash", same, "c"), result("c", "total symbols: 268")]
    assert stop_reason(resumed) is None
    # Alternating commands: neither pair is adjacent, but one call recurs.
    other = json.dumps({"command": "curl localhost"})
    alternating = [{"role": "user", "content": "go"}]
    for i, args in enumerate((same, other, same, other, same)):
        alternating += [call("bash", args, f"x{i}"), result(f"x{i}", f"out-{i}")]
    assert stop_reason(alternating) is not None
    # Distinct calls are fine until the hard cap.
    unique = [{"role": "user", "content": "go"}]
    for i in range(MAX_TOOL_ROUNDS - 1):
        args = json.dumps({"command": f"step {i}"})
        unique += [call("bash", args, f"u{i}"), result(f"u{i}", f"ok {i}")]
    assert stop_reason(unique) is None
    unique += [call("bash", json.dumps({"command": "last"}), "uZ"), result("uZ", "ok")]
    assert stop_reason(unique) is not None
    changed = once + [call("bash", same, "b"), result("b", "total symbols: 269")]
    assert stop_reason(changed) is None
    other = once + [call("bash", json.dumps({"command": "ls"}), "b"), result("b", "src")]
    assert stop_reason(other) is None
    # A healthy verify loop: edit -> node --check -> another edit -> node --check.
    # Both checks print the same (empty) output, but a file changed in between,
    # so the second check is legitimate and must be forwarded.
    check = json.dumps({"command": "node --check game.js"})
    healthy = [
        {"role": "user", "content": "build"},
        call("write", json.dumps({"filePath": "game.js", "content": "v1"}), "w1"), result("w1", "Wrote file successfully."),
        call("bash", check, "c1"), result("c1", ""),
        call("edit", json.dumps({"filePath": "game.js", "oldString": "v1", "newString": "v2"}), "e1"), result("e1", "Edit applied successfully."),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert stop_reason(healthy) is None, "edit between identical checks must not stop the turn"
    # Only a read between the two identical checks: nothing changed, still a loop.
    looping = healthy[:5] + [
        call("read", json.dumps({"filePath": "game.js"}), "r1"), result("r1", "v1"),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert stop_reason(looping) is not None, "identical repeat with no file change must stop"
    _stream_self_test()
    print("loop_proxy self-test ok")


def _stream_self_test() -> None:
    """The proxy must relay a chunked SSE stream as it arrives, not at the end."""

    import threading
    import time

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or "0"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for text, delay in (("data: first\n\n", 0.0), ("data: second\n\n", 0.8)):
                time.sleep(delay)
                raw = text.encode()
                self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    saved = (Proxy.upstream_host, Proxy.upstream_port)
    Proxy.upstream_host, Proxy.upstream_port = "127.0.0.1", upstream.server_address[1]
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        started = time.time()
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        first = resp.read1(4096)
        first_at = time.time() - started
        rest = b""
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            rest += chunk
        assert b"data: first" in first, f"unexpected first chunk: {first!r}"
        assert first_at < 0.5, f"stream was buffered: first chunk arrived after {first_at:.2f}s"
        assert b"data: second" in rest
    finally:
        proxy.shutdown()
        upstream.shutdown()
        Proxy.upstream_host, Proxy.upstream_port = saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="8769")
    parser.add_argument("--upstream", default="127.0.0.1:8779")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    host, _, port = args.listen.rpartition(":")
    if not port:
        host, port = "127.0.0.1", args.listen
    upstream_host, _, upstream_port = args.upstream.rpartition(":")
    Proxy.upstream_host = upstream_host or "127.0.0.1"
    Proxy.upstream_port = int(upstream_port)
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Proxy)
    print(f"[loop-proxy] {host or '127.0.0.1'}:{port} -> {Proxy.upstream_host}:{Proxy.upstream_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
