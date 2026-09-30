#!/usr/bin/env python3
"""End-to-end check of the whole harness: OpenCode (headless `opencode run`)
→ loop_proxy.py (:8092) → TensorFold (:8102), on a real coding task in a
fresh directory. The default prompt is the task the abliterated Gemma 4 stack
failed on 2026-09-30 ("create a tetris game in folder named z": prose instead
of tool calls, then a six-minute slide into LaTeX soup); it is a good stress
test for any local model behind this harness.

  python3 test_e2e_opencode.py                       # the tetris prompt, 15 min cap
  python3 test_e2e_opencode.py --prompt "..." --timeout 600
  python3 test_e2e_opencode.py --keep                # keep the work dir

Checks: the run ends inside the cap; at least one write/edit ran; every text
part the model produced passes harness_judge (no soup, loops, narration or
leaked envelopes reached the transcript); the project dir has new files and
every .js file passes `node --check`; and it reports how often the proxy had
to retry or stop during the run (from /harness/health).

Exit 0 on success, 1 on a failed check, 2 when OpenCode or the server is
unavailable. The model comes from ~/.config/opencode/opencode.json by default,
so this exercises the live agent prompt and permissions too.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_judge import judge_prose  # noqa: E402

DEFAULT_MODEL = "gemma4-tensorfold/gemma-4-26b-a4b-tensorfold"
DEFAULT_PROMPT = "create a tetris game in folder named z"
DEFAULT_BASE = "http://127.0.0.1:8092"


def health(base: str) -> dict:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/harness/health", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # progress shows up in a redirected log as it happens
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenCode provider/model id")
    ap.add_argument("--agent", default="build")
    ap.add_argument("--base", default=DEFAULT_BASE, help="proxy base URL, for /harness/health")
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock cap in seconds")
    ap.add_argument("--dir", default=None, help="work dir (default: a fresh temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    args = ap.parse_args()

    if not shutil.which("opencode"):
        print("opencode is not on PATH")
        return 2
    before = health(args.base)
    if "error" in before:
        print(f"harness proxy not reachable at {args.base}: {before['error']} — start ./2_start_tensorfold.sh")
        return 2
    work = args.dir or tempfile.mkdtemp(prefix="harness-e2e-")
    os.makedirs(work, exist_ok=True)
    existing = {p for p in _walk(work)}
    cmd = ["opencode", "run", "--dir", work, "--format", "json", "--agent", args.agent, "--auto",
           "-m", args.model, args.prompt]
    print(f"== e2e: {' '.join(cmd[:8])} … {args.prompt!r}")
    print(f"   work dir: {work}   cap: {args.timeout:.0f}s   proxy counts before: {before.get('counts')}")
    started = time.time()
    stderr_path = os.path.join(tempfile.gettempdir(), f"harness-e2e-{os.getpid()}.stderr")
    stderr_file = open(stderr_path, "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True, cwd=work)
    events: list[dict] = []
    texts: list[str] = []
    tools: list[tuple[str, str]] = []
    finishes: list[str] = []
    errors: list[str] = []
    timed_out = False
    assert proc.stdout is not None
    import selectors

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    buf = ""
    eof = False
    while not eof:
        if time.time() - started > args.timeout:
            timed_out = True
            proc.kill()
            break
        for key, _ in sel.select(0.5):
            line = key.fileobj.readline()
            if not line:          # EOF: opencode closed stdout (it has exited or is exiting)
                eof = True
                break
            buf += line
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("{"):
                if line:
                    print("   |", line[:200])
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print("   |", line[:200])
                continue
            events.append(event)
            _consume(event, texts, tools, finishes, errors, started)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    stderr_file.close()
    try:
        with open(stderr_path) as handle:
            stderr = handle.read()
        os.remove(stderr_path)
    except OSError:
        stderr = ""
    elapsed = time.time() - started
    after = health(args.base)
    print(f"\n== result after {elapsed:.0f}s: exit={proc.returncode} timed_out={timed_out} events={len(events)}")

    failures: list[str] = []
    if timed_out:
        failures.append(f"did not finish inside {args.timeout:.0f}s")
    if proc.returncode not in (0, None) and not timed_out:
        failures.append(f"opencode exited {proc.returncode}: {stderr.strip()[-400:]}")
    if errors:
        failures.append("errors: " + "; ".join(errors)[:400])
    write_like = [t for t, _ in tools if t in ("write", "edit", "patch", "multiedit", "apply_patch", "bash")]
    print(f"   tool calls: {len(tools)} ({', '.join(t for t, _ in tools)[:200]})")
    if not any(t in ("write", "edit", "patch", "multiedit", "apply_patch") for t, _ in tools):
        failures.append("no write/edit tool call ran")
    for text in texts:
        verdict = judge_prose(text, agentic=True, finished=True)
        if not verdict.ok:
            failures.append(f"transcript text failed the judge: {verdict.reason} — {verdict.detail}")
            break
    print(f"   text parts: {len(texts)} ({sum(len(t) for t in texts)} chars) — last: {texts[-1][:160]!r}" if texts else "   text parts: 0")
    new_files = sorted(p for p in _walk(work) if p not in existing)
    print(f"   new files: {new_files[:12]}")
    if not new_files:
        failures.append("no files were created")
    node = shutil.which("node")
    for path in new_files:
        if path.endswith(".js") and node:
            check = subprocess.run([node, "--check", os.path.join(work, path)], capture_output=True, text=True)
            status = "ok" if check.returncode == 0 else f"FAIL: {check.stderr.strip()[:200]}"
            print(f"   node --check {path}: {status}")
            if check.returncode != 0:
                failures.append(f"node --check failed for {path}")
    counts_before, counts_after = before.get("counts") or {}, after.get("counts") or {}
    delta = {k: counts_after.get(k, 0) - counts_before.get(k, 0) for k in ("steps", "retries", "stops", "cuts", "repeat_nudges", "repeat_stops", "stalls", "upstream_errors")}
    print(f"   proxy during run: {delta}   failures by kind (total): {after.get('failures')}")
    print(f"   step finishes: {finishes}")
    if failures:
        print("\n== FAIL")
        for f in failures:
            print("   -", f)
    else:
        print("\n== PASS")
    if not args.keep and not args.dir and not failures:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"   kept work dir: {work}")
    return 1 if failures else 0


def _walk(root: str) -> list[str]:
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".opencode")]
        for name in files:
            out.append(os.path.relpath(os.path.join(base, name), root))
    return out


def _consume(event: dict, texts: list[str], tools: list[tuple[str, str]], finishes: list[str], errors: list[str], started: float) -> None:
    kind = event.get("type")
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    stamp = f"{time.time() - started:6.1f}s"
    if kind == "text":
        text = part.get("text") or ""
        texts.append(text)
        print(f"   {stamp} text {len(text)}ch: {text[:100]!r}")
    elif kind in ("tool", "tool_use", "tool_call", "tool-invocation"):
        tool = str(part.get("tool") or part.get("name") or "?")
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = str(state.get("status") or "")
        inputs = state.get("input") if isinstance(state.get("input"), dict) else {}
        target = inputs.get("filePath") or inputs.get("command") or inputs.get("pattern") or ""
        tools.append((tool, status))
        print(f"   {stamp} tool {tool} {status} {str(target)[:80]}")
        if status == "error":
            errors.append(f"{tool}: {str(state.get('error') or '')[:120]}")
    elif kind == "step_finish":
        finishes.append(str(part.get("reason")))
        tokens = part.get("tokens") or {}
        print(f"   {stamp} step_finish reason={part.get('reason')} out_tokens={tokens.get('output')}")
    elif kind == "error":
        errors.append(json.dumps(event.get("error") or event)[:200])
        print(f"   {stamp} ERROR {json.dumps(event)[:200]}")
    elif kind not in ("step_start",):
        print(f"   {stamp} {kind} {json.dumps(event)[:120]}")


if __name__ == "__main__":
    sys.exit(main())
