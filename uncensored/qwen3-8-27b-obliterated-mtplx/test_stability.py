"""End-to-end stability test for qwen38_obl_kilo_proxy against a fake slow engine.

Run:  python3 test_stability.py          (offline, ~40s, no model needed)

No 27B weights needed. Proves:
  1. passthrough SSE: headers + keepalive comments arrive while the "engine"
     is still prefilling; content is forwarded intact.
  2. buffered SSE (early-stop middleware round): headers within ~1s, keepalives
     during the multi-second buffer, complete payload ending in [DONE].
  3. client disconnect while the engine is busy aborts the upstream request
     and releases the engine lock (next request served promptly).
  4. lock wait timeout -> in-band 503 (stream) / HTTP 503 (json), never a hang.
  5. max_tokens clamp visible upstream.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.environ.get("PROXY_PY") or os.path.join(HERE, "qwen38_obl_kilo_proxy.py")
UP_PORT = 18767
PX_PORT = 18768

SEEN: list[dict] = []
ABORTED = threading.Event()
HOLD = threading.Event()  # when set, upstream blocks until cleared


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        raw = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(body)
        delay = float(self.headers.get("X-Test-Delay") or body.get("_delay", 0))
        text = body.get("_text", "hello from fake")
        # simulate prefill: silence
        t0 = time.time()
        while time.time() - t0 < delay:
            if HOLD.is_set():
                pass
            time.sleep(0.05)
        while HOLD.is_set():
            time.sleep(0.05)
        self.send_response(200)
        # Match uvicorn/mtplx: lowercase header names. Title-case hid a
        # proxy bug that wrapped live SSE as {"error": ...}.
        self.send_header("content-type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i, tok in enumerate(text.split(" ")):
                ev = {
                    "id": "chatcmpl-x",
                    "object": "chat.completion.chunk",
                    "model": "fake",
                    "choices": [
                        {"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}
                    ],
                }
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            fin = {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "model": "fake",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
        except OSError:
            ABORTED.set()


def raw_http(port: int, body: dict, *, read_secs: float, close_after: float | None = None) -> tuple[bytes, float]:
    """Send one POST over a raw socket; return (bytes, header_latency)."""
    payload = json.dumps(body).encode()
    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + payload
    s = socket.create_connection(("127.0.0.1", port), timeout=read_secs + 5)
    t0 = time.time()
    s.sendall(req)
    out = b""
    hdr_at = None
    deadline = time.time() + read_secs
    s.settimeout(0.5)
    while time.time() < deadline:
        if close_after is not None and time.time() - t0 >= close_after:
            break
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        out += chunk
        if hdr_at is None and b"\r\n\r\n" in out:
            hdr_at = time.time() - t0
    s.close()
    return out, (hdr_at if hdr_at is not None else -1.0)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    up = ThreadingHTTPServer(("127.0.0.1", UP_PORT), Fake)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    env = dict(os.environ, QWEN38_OBL_LOCK_WAIT="6", QWEN38_OBL_MAX_TOKENS="4096",
               QWEN38_OBL_THINKING="1", QWEN38_OBL_REASONING_EFFORT="medium",
               QWEN38_OBL_THINKING_BUDGET="1024")  # floor 3072 < cap 4096
    px = subprocess.Popen(
        [sys.executable, PROXY, "--port", str(PX_PORT), "--upstream", f"http://127.0.0.1:{UP_PORT}"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(PROXY),
    )
    ok = True
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", PX_PORT), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise SystemExit("proxy never listened")

        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]

        # 1. passthrough with slow prefill (6s) -> headers fast, keepalives present
        body = {"model": "fake", "stream": True, "max_tokens": 32000, "tools": tools,
                "messages": [{"role": "user", "content": "hi"}], "_delay": 7}
        out, hdr = raw_http(PX_PORT, body, read_secs=12)
        ok &= check("passthrough: headers before prefill finished", 0 <= hdr < 2, f"{hdr:.2f}s")
        ok &= check("passthrough: keepalive comments while waiting", out.count(b": keepalive") >= 1, str(out.count(b": keepalive")))
        ok &= check("passthrough: content forwarded", b"hello" in out and b"[DONE]" in out)
        ok &= check(
            "passthrough: not wrapped as error (uvicorn lowercase content-type)",
            b'"error"' not in out,
            out[:180].decode("utf-8", "replace"),
        )
        ok &= check("max_tokens clamped upstream", SEEN[-1].get("max_tokens") == 4096, str(SEEN[-1].get("max_tokens")))
        ok &= check("card sampling forced", SEEN[-1].get("temperature") == 0 and SEEN[-1].get("top_p") == 1.0)
        ok &= check("thinking ON for tool turn", SEEN[-1].get("enable_thinking") is True and SEEN[-1].get("reasoning_effort") == "medium", str(SEEN[-1].get("reasoning_effort")))

        # 2. buffered SSE round (after a tool result within first 3 rounds)
        SEEN.clear()
        body = {"model": "fake", "stream": True, "max_tokens": 2048, "tools": tools,
                "messages": [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "do two things"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "README.md\n"},
                ], "_delay": 7, "_text": "Done: listed the files."}
        out, hdr = raw_http(PX_PORT, body, read_secs=14)
        ok &= check("buffered: headers immediately", 0 <= hdr < 2, f"{hdr:.2f}s")
        ok &= check("buffered: keepalives during buffer", out.count(b": keepalive") >= 1, str(out.count(b": keepalive")))
        ok &= check("buffered: complete payload with [DONE]", b"Done:" in out and out.rstrip().endswith(b"data: [DONE]"))
        ok &= check("buffered: exactly one upstream call (no spurious retry)", len(SEEN) == 1, str(len(SEEN)))

        # 3. client disconnect while engine busy -> abort upstream, free lock
        SEEN.clear(); ABORTED.clear()
        body = {"model": "fake", "stream": True, "tools": tools,
                "messages": [{"role": "user", "content": "long"}], "_delay": 4,
                "_text": " ".join(["tok"] * 200)}
        t0 = time.time()
        raw_http(PX_PORT, body, read_secs=10, close_after=1.0)  # client gives up after 1s
        # Immediately send another request; must be served before the abandoned one would have finished (~4s+10s)
        body2 = {"model": "fake", "stream": True, "tools": tools,
                 "messages": [{"role": "user", "content": "quick"}], "_delay": 0, "_text": "second ok"}
        out2, hdr2 = raw_http(PX_PORT, body2, read_secs=12)
        el = time.time() - t0
        ok &= check("disconnect: follow-up request served", b"second " in out2 and b"ok " in out2 and b"[DONE]" in out2, f"{el:.1f}s total")
        ok &= check("disconnect: lock released promptly (<9s)", el < 9, f"{el:.1f}s")
        ok &= check("disconnect: upstream saw abort", ABORTED.wait(6))

        # 4. lock wait timeout -> 503 in-band (stream) and HTTP 503 (json)
        HOLD.set()
        holder = threading.Thread(target=raw_http, args=(PX_PORT, {"model": "fake", "stream": True,
                                  "messages": [{"role": "user", "content": "hold"}]}), kwargs={"read_secs": 30}, daemon=True)
        holder.start()
        time.sleep(0.8)
        out, hdr = raw_http(PX_PORT, {"model": "fake", "stream": True, "messages": [{"role": "user", "content": "x"}]}, read_secs=10)
        ok &= check("busy: stream gets in-band 503 error", b'"code": 503' in out and b"[DONE]" in out, f"hdr {hdr:.2f}s")
        outj, _ = raw_http(PX_PORT, {"model": "fake", "stream": False, "messages": [{"role": "user", "content": "x"}]}, read_secs=10)
        ok &= check("busy: json gets HTTP 503", outj.startswith(b"HTTP/1.1 503"), outj.split(b"\r\n")[0].decode())
        HOLD.clear()
        holder.join(10)

        # 5. healthz reports limits
        import urllib.request
        h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PX_PORT}/healthz", timeout=3).read())
        ok &= check("healthz: limits exposed", h.get("limits", {}).get("max_tokens_cap") == 4096 and h.get("engine_busy") is False)
    finally:
        px.terminate()
        up.shutdown()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
