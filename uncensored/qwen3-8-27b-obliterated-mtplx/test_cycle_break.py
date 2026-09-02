#!/usr/bin/env python3
"""Offline e2e: when the model keeps emitting a command that already recurred
CYCLE_BREAK_MIN times, the proxy bans it, retries once, and (if still repeated)
ends the turn with a visible [Harness] Stopped message instead of forwarding
the loop. No model needed. Run: python3 test_cycle_break.py"""
import json, os, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP, PX = 18967, 18968
CMD = "tail -c 2000 /x/JOURNAL.md"
def ev(delta, fin=None):
    return ("data: " + json.dumps({"id":"x","model":"m","choices":[{"index":0,"delta":delta,"finish_reason":fin}]}) + "\n\n").encode()
class Up(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(b'{"data":[{"id":"m"}]}')
    def do_POST(self):
        n=int(self.headers.get("Content-Length") or 0); json.loads(self.rfile.read(n))
        # Always return the SAME banned command (a stubborn looping model).
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.end_headers()
        self.wfile.write(ev({"role":"assistant","tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"bash","arguments":json.dumps({"command":CMD})}}]}))
        self.wfile.write(ev({}, "tool_calls")); self.wfile.write(b"data: [DONE]\n\n")
up=ThreadingHTTPServer(("127.0.0.1",UP),Up); threading.Thread(target=up.serve_forever,daemon=True).start()
px=subprocess.Popen([sys.executable,"qwen38_obl_kilo_proxy.py","--port",str(PX),"--upstream",f"http://127.0.0.1:{UP}"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for _ in range(50):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PX}/healthz",timeout=1); break
    except Exception: time.sleep(0.2)
# History: a NON-consecutive cycle (CMD interspersed with `ls`) so CMD recurs
# >= CYCLE_BREAK_MIN times in the window but the consecutive repeat-guard's
# hard stop does NOT fire (history ends on a different command). This exercises
# the cycle-break path specifically.
def asst(cmd):
    return {"role":"assistant","content":"Let me look.","tool_calls":[{"id":"c","type":"function","function":{"name":"bash","arguments":json.dumps({"command":cmd})}}]}
LS="ls -la /x/"
msgs=[{"role":"system","content":"agent"},{"role":"user","content":"continue the research"}]
for cmd in [CMD,CMD,CMD,LS,CMD,CMD,CMD,LS]:
    msgs.append(asst(cmd)); msgs.append({"role":"tool","tool_call_id":"c","content":"(nothing new)"})
body={"model":"m","stream":True,"max_tokens":4096,
      "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
      "messages":msgs}
req=urllib.request.Request(f"http://127.0.0.1:{PX}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
out=urllib.request.urlopen(req,timeout=30).read().decode()
px.terminate(); up.shutdown()
stopped = "[Harness] Stopped: stuck in a read-only loop" in out and '"tool_calls"' not in out
print(out[:400])
print("RESULT:", "PASS" if stopped else "FAIL")
sys.exit(0 if stopped else 1)
