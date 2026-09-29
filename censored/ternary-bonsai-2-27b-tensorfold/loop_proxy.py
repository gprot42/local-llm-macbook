#!/usr/bin/env python3
"""Keep an OpenCode turn from looping — without interrupting the user.

OpenCode's doom_loop check only looks at tool calls inside one assistant
message, and one call per step never trips it. Alternating two commands
never trips it either. This proxy sits in front of the engine and looks at
the tool calls after the latest user message. It answers in two tiers:

Nudge (the turn continues): when the model has *just* run a call that
returned the same result a 2nd or 3rd time with no write/edit in between
(an edit between two identical `node --check` runs is a normal verify loop
and is left alone), or has just issued the same call a 3rd time even with
different output, the proxy appends a "[Harness] ... do not run it again"
note to that tool result and forwards the request. The model corrects
itself in place; the user is not asked to do anything.

Stop (the turn ends, the model is not called): only when the identical
call comes back a 4th time, the same call is issued a 6th time, or the turn
has run 200 tool calls (--max-rounds / LOOP_MAX_ROUNDS — a backstop for a
turn that never repeats and never finishes; it must sit far above real
work, since a 48-call exploration turn was once cut off by a cap of 48).

Stopping at the second identical result was tried first: it halted one
session three times in an hour while the model retried a broken grep after
every "continue". Every stop is an interruption; the nudge tier is what
keeps the user out of the loop.

A new user message starts the counts over. Upstream bytes are relayed as
they arrive (read1), so streamed tokens reach the client immediately.

  python3 loop_proxy.py --self-test
  python3 loop_proxy.py --listen 8091 --upstream 127.0.0.1:8101
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import http.client
import uuid

NUDGE_SAME = 2      # identical result, nothing changed: nudge in place
STOP_SAME = 4       # identical result keeps coming back: end the turn
NUDGE_CYCLE = 3     # same call issued again (any output): nudge in place
STOP_CYCLE = 6      # same call issued again and again: end the turn
MAX_TOOL_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "200"))
# Tools that change files. An identical call repeated after one of these is a
# legitimate re-verification, not a loop.
STATE_CHANGING = {"write", "edit", "patch", "multiedit", "apply_patch"}

DIRECTIVE_AT = 3    # from the 3rd repeat, also inject a user-role directive
OUTPUT_QUOTE = 600  # chars of the original output quoted inside a refusal

# A repeat's result is REPLACED with a refusal (an appended note was ignored by
# a model that ran the same grep five times in two minutes). The original
# output is quoted inside it, so nothing is lost.
REFUSAL_TEXT = (
    "[Harness] REFUSED: this exact command already ran {n} times in this turn "
    "with identical output, so it was not run again. That output is final and "
    "is repeated here:\n\n{output}\n\n"
    "Do not run this command again. Use this output, run a different command, "
    "edit a file, or report what it shows."
)
CYCLE_REFUSAL_TEXT = (
    "[Harness] REFUSED: this exact command has now been issued {n} times in "
    "this turn. Its latest output is:\n\n{output}\n\n"
    "Do not issue it again. Change the approach, or report what this shows."
)
# From the 3rd repeat a user-role message is added as well: the strongest
# signal a chat model has, and the last step before the turn is ended.
DIRECTIVE_TEXT = (
    "[Harness] Stop re-running that command. It has been run {n} times with the "
    "same result, which is shown above and will not change. Take a different "
    "action now: run a different command, edit a file, or report what you found."
)
# The stop messages are read by the USER (they end the turn), so they say
# what the model is stuck on and what to type instead of "continue".
STOP_TEXT = (
    "[Harness] Stopped: the model is stuck re-running the same command "
    "({n} times, identical output, ignored two corrections):\n\n"
    "    {command}\n\n"
    "Replying \"continue\" will repeat it. Reply with a specific instruction "
    "instead: what to run, which file to look at, or what to do with that output."
)
CYCLE_TEXT = (
    "[Harness] Stopped: the model keeps issuing the same command "
    "({n} times this turn, mixed with other calls, ignored the corrections):\n\n"
    "    {command}\n\n"
    "Replying \"continue\" will likely repeat it. Reply with a specific "
    "instruction instead."
)
CAP_TEXT = (
    "[Harness] Stopped: this turn already ran {n} tool calls without "
    "finishing. Report what is done and what is blocked. Do not call another tool. "
    "(If the work really is this long, reply to continue — the count resets on a "
    "new message — or start the proxy with a higher --max-rounds.)"
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


def decide(messages: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    """(action, text, directive).

    ("stop", text, None): end the turn with `text` (the model is not called).
    ("nudge", refusal, directive): replace the latest tool result with
    `refusal`, append a user-role `directive` when not None, then forward.
    (None, None, None): forward untouched.

    Only the rounds after the latest user message count, and a nudge or stop
    is issued only when the offending call is the latest round — a turn that
    repeated something earlier and then moved on is never touched.
    """

    rounds = _rounds_after_last_user(messages)
    if len(rounds) >= MAX_TOOL_ROUNDS:
        return "stop", CAP_TEXT.format(n=len(rounds)), None
    if not rounds:
        return None, None, None
    by_sig: dict[str, list[int]] = {}
    for index, (sig, _) in enumerate(rounds):
        by_sig.setdefault(sig, []).append(index)
    sig, output = rounds[-1]
    indices = by_sig[sig]
    n = len(indices)
    quoted = output[:OUTPUT_QUOTE] + (" …" if len(output) > OUTPUT_QUOTE else "")
    directive = DIRECTIVE_TEXT.format(n=n) if n >= DIRECTIVE_AT else None
    if n >= NUDGE_SAME:
        prev, last = indices[-2], indices[-1]
        same_result = rounds[prev][1] == rounds[last][1]
        changed_between = any(_tool_name(rounds[k][0]) in STATE_CHANGING for k in range(prev + 1, last))
        if same_result and not changed_between:
            if n >= STOP_SAME:
                return "stop", STOP_TEXT.format(n=n, command=_display_command(sig)), None
            return "nudge", REFUSAL_TEXT.format(n=n, output=quoted), directive
    if n >= STOP_CYCLE:
        return "stop", CYCLE_TEXT.format(n=n, command=_display_command(sig)), None
    if n >= NUDGE_CYCLE:
        return "nudge", CYCLE_REFUSAL_TEXT.format(n=n, output=quoted), directive
    return None, None, None


def _display_command(sig: str) -> str:
    """The command / path inside a signature, for the user-facing stop text."""

    _, _, args = sig.partition("\n")
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("command", "filePath", "pattern", "path"):
            if parsed.get(key):
                args = str(parsed[key])
                break
    args = " ".join(args.split())
    return args if len(args) <= 200 else args[:200] + " …"


def apply_nudge(messages: list[dict[str, Any]], refusal: str, directive: str | None) -> bool:
    """Replace the latest tool result with the refusal; add the directive as a
    user turn when given. Only the forwarded copy changes — the client's own
    transcript keeps the real output."""

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        message["content"] = refusal
        if directive:
            messages.append({"role": "user", "content": directive})
        return True
    return False


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
                messages = payload.get("messages") or []
                action, text, directive = decide(messages)
                if action == "stop":
                    self._send_stop(payload, text or "")
                    return
                if action == "nudge" and text and apply_nudge(messages, text, directive):
                    body = json.dumps(payload).encode()
                    self.log_message("refused repeat%s: %s", " + directive" if directive else "", text[len("[Harness] "):80])
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
        self.log_message("stopped turn: %s", reason[len("[Harness] Stopped: "):76])

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
    action = lambda msgs: decide(msgs)[0]  # noqa: E731
    same = json.dumps({"command": "python3 scan.py"})

    def repeated(k: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(k):
            msgs += [call("bash", same, f"r{i}"), result(f"r{i}", "total symbols: 268")]
        return msgs

    assert action(repeated(1)) is None
    assert action(repeated(2)) == "nudge", "2nd identical result must nudge, not stop"
    assert action(repeated(3)) == "nudge"
    assert action(repeated(4)) == "stop", "4th identical result must end the turn"
    # The model moved on after a repeat: leave the turn alone.
    moved_on = repeated(2) + [call("bash", json.dumps({"command": "ls"}), "m"), result("m", "src")]
    assert action(moved_on) is None, "a turn that moved on must not be touched"
    # A new user prompt after an old loop must be forwarded untouched.
    assert action(repeated(4) + [{"role": "user", "content": "try something else"}]) is None
    resumed = repeated(4) + [{"role": "user", "content": "try something else"}, call("bash", same, "c"), result("c", "total symbols: 268")]
    assert action(resumed) is None
    # Alternating commands: the same call recurs, each time with different output.
    other = json.dumps({"command": "curl localhost"})

    def alternating(k: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(k):
            msgs += [call("bash", same if i % 2 == 0 else other, f"x{i}"), result(f"x{i}", f"out-{i}")]
        return msgs

    assert action(alternating(3)) is None, "issued twice with different output is not yet a loop"
    assert action(alternating(5)) == "nudge", "issued a 3rd time must nudge"
    assert action(alternating(11)) == "stop", "issued a 6th time must end the turn"
    # Distinct calls are fine until the hard cap.
    unique = [{"role": "user", "content": "go"}]
    for i in range(MAX_TOOL_ROUNDS - 1):
        unique += [call("bash", json.dumps({"command": f"step {i}"}), f"u{i}"), result(f"u{i}", f"ok {i}")]
    assert action(unique) is None
    unique += [call("bash", json.dumps({"command": "last"}), "uZ"), result("uZ", "ok")]
    assert action(unique) == "stop"
    changed = repeated(1) + [call("bash", same, "b"), result("b", "total symbols: 269")]
    assert action(changed) is None, "a different result is progress"
    # A healthy verify loop: edit -> node --check -> another edit -> node --check.
    # Both checks print the same (empty) output, but a file changed in between.
    check = json.dumps({"command": "node --check game.js"})
    healthy = [
        {"role": "user", "content": "build"},
        call("write", json.dumps({"filePath": "game.js", "content": "v1"}), "w1"), result("w1", "Wrote file successfully."),
        call("bash", check, "c1"), result("c1", ""),
        call("edit", json.dumps({"filePath": "game.js", "oldString": "v1", "newString": "v2"}), "e1"), result("e1", "Edit applied successfully."),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert action(healthy) is None, "edit between identical checks must be left alone"
    # Only a read between the two identical checks: nothing changed, nudge.
    looping = healthy[:5] + [
        call("read", json.dumps({"filePath": "game.js"}), "r1"), result("r1", "v1"),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert action(looping) == "nudge", "identical repeat with no file change must nudge"
    # 2nd repeat: the duplicate's result is replaced by a refusal quoting the
    # original output; no user directive yet.
    msgs = repeated(2)
    _, refusal, directive = decide(msgs)
    assert refusal and refusal.startswith("[Harness] REFUSED") and "total symbols: 268" in refusal
    assert directive is None
    assert apply_nudge(msgs, refusal, directive)
    assert msgs[-1]["role"] == "tool" and msgs[-1]["content"] == refusal
    # 3rd repeat: refusal plus a user-role directive appended after it.
    msgs = repeated(3)
    _, refusal, directive = decide(msgs)
    assert directive and directive.startswith("[Harness] Stop re-running")
    assert apply_nudge(msgs, refusal, directive)
    assert msgs[-2]["role"] == "tool" and msgs[-2]["content"] == refusal
    assert msgs[-1] == {"role": "user", "content": directive}
    # The stop text names the command the model is stuck on, for the user.
    _, stop_text, _ = decide(repeated(4))
    assert "python3 scan.py" in stop_text and '"continue" will repeat it' in stop_text
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
    global MAX_TOOL_ROUNDS  # declared first: the name is read below before it may be assigned
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="8091")
    parser.add_argument("--upstream", default="127.0.0.1:8101")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help=f"tool calls allowed per user turn (default {MAX_TOOL_ROUNDS}; env LOOP_MAX_ROUNDS)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.max_rounds is not None:
        MAX_TOOL_ROUNDS = args.max_rounds
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
