#!/usr/bin/env python3
"""Offline e2e: a cycle of ANY period must end the turn.

The consecutive repeat guard needs identical ADJACENT turns and the cycle-break
ban needs one signature >= CYCLE_BREAK_MIN times in an 8-turn window, so a 2-,
3- or 4-command cycle satisfies neither and used to be nudged forever (live log
2026-09-02: 26 of 37 cycle detections had no exit). The novelty-starvation
counter is the exit: N tool turns with no NEW action -> tools removed, then a
visible [Harness] Stopped message, without spending an engine slot.

No model needed. Run: python3 test_cycle_stale.py
"""
import json, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP, PX = 18969, 18970
STOP_TEXT = "[Harness] Stopped: stuck in a read-only loop"
hits = []


def ev(delta, fin=None):
    return ("data: " + json.dumps({"id": "x", "model": "m", "choices": [
        {"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n").encode()


class Up(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(b'{"data":[{"id":"m"}]}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); json.loads(self.rfile.read(n))
        hits.append(1)
        # A brand-new command: if the proxy ever reaches upstream, the turn
        # continues normally -- which is what the negative case asserts.
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(ev({"role": "assistant", "tool_calls": [{"index": 0, "id": "c",
            "type": "function", "function": {"name": "bash",
            "arguments": json.dumps({"command": "pytest -q tests/"})}}]}))
        self.wfile.write(ev({}, "tool_calls")); self.wfile.write(b"data: [DONE]\n\n")


up = ThreadingHTTPServer(("127.0.0.1", UP), Up)
threading.Thread(target=up.serve_forever, daemon=True).start()
px = subprocess.Popen([sys.executable, "qwen38_obl_kilo_proxy.py", "--port", str(PX),
                       "--upstream", f"http://127.0.0.1:{UP}"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(50):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PX}/healthz", timeout=1); break
    except Exception:
        time.sleep(0.2)


def asst(cmd):
    return {"role": "assistant", "content": "Let me look.", "tool_calls": [{"id": "c",
            "type": "function", "function": {"name": "bash",
            "arguments": json.dumps({"command": cmd})}}]}


def ask(cmds, rounds):
    msgs = [{"role": "system", "content": "agent"},
            {"role": "user", "content": "continue the research"}]
    for i in range(rounds):
        msgs.append(asst(cmds[i % len(cmds)]))
        msgs.append({"role": "tool", "tool_call_id": "c", "content": "(nothing new)"})
    body = {"model": "m", "stream": True, "max_tokens": 4096,
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {
                "type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}}}],
            "messages": msgs}
    req = urllib.request.Request(f"http://127.0.0.1:{PX}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


CYCLES = {
    "2-cycle": ["ls -la /x/", "tail -c 2000 /x/JOURNAL.md"],
    "3-cycle": ["ls -la /x/", "tail -c 2000 /x/JOURNAL.md", "cat /x/SUMMARY.md"],
    "4-cycle": ["ls -la /x/", "tail -c 2000 /x/JOURNAL.md", "cat /x/SUMMARY.md",
                "head -40 /x/NOTES.md"],
}
results = []
for name, cmds in CYCLES.items():
    before = len(hits)
    out = ask(cmds, 20)
    stopped = STOP_TEXT in out and '"tool_calls"' not in out
    # The stop is decided request-side: no engine slot is spent.
    no_upstream = len(hits) == before
    results.append((name, stopped and no_upstream,
                    f"stopped={stopped} upstream_calls={len(hits) - before}"))

# Negative: a varied, progressing run is untouched and still gets its tools.
before = len(hits)
out = ask(["ls -la /x/", "grep -rn TODO /x/", "sed -n 1,40p /x/a.py",
           "sed -n 1,40p /x/b.py", "python3 -m pytest -q"], 5)
varied_ok = '"tool_calls"' in out and STOP_TEXT not in out and len(hits) > before
results.append(("varied run untouched", varied_ok,
                f"tool_calls={'\"tool_calls\"' in out} upstream_calls={len(hits) - before}"))

px.terminate(); up.shutdown()
for name, ok, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} {detail}")
allok = all(ok for _, ok, _ in results)
print("RESULT:", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
