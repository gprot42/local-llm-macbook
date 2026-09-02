#!/usr/bin/env python3
"""Offline e2e: a prose <bash>…</bash> reply from the engine must reach the
client as real tool_calls (2026-09-02 live-trace regression). No model needed.
Run: python3 test_prose_toolcall.py"""
import json, os, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.getcwd()
UP, PX = 18867, 18868
class Up(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(b'{"data":[{"id":"m"}]}')
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n))
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.end_headers()
        def ev(delta, fin=None):
            return ("data: " + json.dumps({"id":"x","model":"m","choices":[{"index":0,"delta":delta,"finish_reason":fin}]}) + "\n\n").encode()
        self.wfile.write(ev({"role":"assistant","reasoning_content":"think a bit"}))
        self.wfile.write(ev({"content":"<bash>\n\nls -la /tmp/x/ 2>/dev/null; echo ---TOOLS---\n\n\n"}))
        self.wfile.write(ev({}, "stop")); self.wfile.write(b"data: [DONE]\n\n")
up = ThreadingHTTPServer(("127.0.0.1", UP), Up); threading.Thread(target=up.serve_forever, daemon=True).start()
px = subprocess.Popen([sys.executable, "qwen38_obl_kilo_proxy.py", "--port", str(PX), "--upstream", f"http://127.0.0.1:{UP}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, QWEN38_OBL_THINKING="1"))
for _ in range(50):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PX}/healthz", timeout=1); break
    except Exception: time.sleep(0.2)
body = {"model":"m","stream":True,"max_tokens":4096,
        "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
        "messages":[{"role":"system","content":"agent"},{"role":"user","content":"continue the research"}]}
req = urllib.request.Request(f"http://127.0.0.1:{PX}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
out = urllib.request.urlopen(req, timeout=30).read().decode()
px.terminate(); up.shutdown()
print(out[:600])
print("RESULT:", "PASS" if '"tool_calls"' in out and '"finish_reason": "tool_calls"' in out and "ls -la /tmp/x/" in out else "FAIL")
