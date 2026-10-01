#!/usr/bin/env python3
"""Reliability proxy between OpenCode / Kilo and the TensorFold engine.

Three layers. Each one acts on its own and, when it works, the user sees
nothing but a step that succeeded.

1. Repeat guard. OpenCode's doom_loop check only looks inside one assistant
   message. This proxy looks at the tool calls after the latest user message:
   a call that returns the same result a 2nd/3rd time with no write in
   between, or is issued a 3rd time, gets its result replaced by a "[Harness]
   REFUSED" note (from the 3rd repeat also a user-role directive) and the turn
   continues; a 4th identical result, a 6th issue, or 200 calls in one turn
   end the turn with a "[Harness] Stopped" message. An edit identical to one
   already applied this turn is caught before the client runs it (a repeat
   corrupts the file) and the step is retried. The same file rewritten 4
   times in one turn gets a directive to verify or edit instead; at 6 the
   tools are withheld for one reply so the model sums up and the turn ends
   with its own report; 10 times ends the turn with a stop message.

2. Step judge + retry (harness_judge.py). A request that offers tools is an
   agentic step: its reply is buffered, the text is judged while it streams
   and the whole reply at the end. Loops, symbol/LaTeX soup, thousands of
   chars of "let's go / the first action is" with no call, leaked
   <|tool_call>/<|channel> envelopes, replies cut at the token cap and tool
   calls with unusable arguments are failures. The generation is cancelled at
   the engine (closing the upstream socket cancels the job within one decode
   round), the failed text is saved under .harness_failures/ and never reaches
   the transcript, and the step is retried with a fresh seed, a lower
   temperature, min_p and a corrective user message; the last attempt adds a
   repetition penalty for loops. After --max-attempts the turn ends with a
   "[Harness] Stopped" message that says why. A request without tools (plain
   chat) streams live; only a loop or a leak cuts it, with a note, no retry.

   Why buffer: TensorFold emits tool calls as structured deltas only after the
   reply is complete, so an agentic reply's streamed text is prose only and is
   short when the step is healthy. Discarding a bad attempt is only possible
   before the client has seen it, and text the client has recorded is fed back
   to the model on every later step, where garbage begets garbage.

3. Request hygiene. Sampling defaults for fields the client omits (OpenCode
   sends no temperature unless the model entry says "temperature": true, and
   the engine's default is 1.0 — the abliterated pack degenerates there), a
   token cap per agentic step (--step-max-tokens; a runaway at 45 tok/s ran
   six minutes to 16384 without it), include_usage upstream so truncation is
   visible, a stall timeout, waiting out an engine restart instead of failing
   the turn, and GET /harness/health with counters.

  python3 loop_proxy.py --self-test
  python3 loop_proxy.py --listen 8094 --upstream 127.0.0.1:8104
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import queue
import re
import random
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_judge import PASS, Verdict, judge_prose, judge_reply, repair_path_args  # noqa: E402

# ---- repeat guard -------------------------------------------------------------
NUDGE_SAME = 2      # identical result, nothing changed: nudge in place
STOP_SAME = 4       # identical result keeps coming back: end the turn
NUDGE_CYCLE = 3     # same call issued again (any output): nudge in place
STOP_CYCLE = 6      # same call issued again and again: end the turn
MAX_TOOL_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "200"))
STATE_CHANGING = {"write", "edit", "patch", "multiedit", "apply_patch"}
DIRECTIVE_AT = 3
OUTPUT_QUOTE = 600
WRITE_CHURN_NUDGE = 4    # the same file written this many times in one turn: directive
WRITE_CHURN_FINISH = 6   # ... this many times: tools are withheld for one reply, so the model sums up
WRITE_CHURN_STOP = 10    # ... this many times: end the turn
NUDGE_TEMPERATURE = 0.6   # a nudge with a directive also samples at least this wide

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
REAPPLIED_TEXT = (
    "[Harness] This exact {tool} has now been applied {n} times in this turn, and each application changes "
    "the file, so {path} may now contain the change more than once. Its latest result was:\n\n{output}\n\n"
    "Read the file and repair anything duplicated or broken. Do not repeat this {tool}."
)
DIRECTIVE_TEXT = (
    "[Harness] Stop re-running that command. It has been run {n} times with the "
    "same result, which is shown above and will not change. Take a different "
    "action now: run a different command, edit a file, or report what you found."
)
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
CHURN_DIRECTIVE_TEXT = (
    "[Harness] You have now written {path} {n} times in this turn. Do not write it again unless a check "
    "failed. Verify it (for example `node --check` or `python3 -m py_compile`), edit only the part that is "
    "wrong, or report that the work is done."
)
CHURN_FINISH_TEXT = (
    "[Harness] You have now written {path} {n} times in this turn and ignored the request to stop. Tools "
    "are withheld for this reply. Reply with a short summary in plain text: what was built, how it was "
    "verified, and how the user runs it. Do not write code."
)
CHURN_CHECK_TEXT = (
    "[Harness] You have now written {path} {n} times in this turn. The write tool is withheld from now on. "
    "Run `{command}` with the bash tool; if it reports an error, fix it with a small edit; then report in a few "
    "plain lines what works and what does not."
)
CHURN_STOP_TEXT = (
    "[Harness] Stopped: the model rewrote the same file {n} times in this turn without finishing:\n\n"
    "    {path}\n\n"
    "Open the file and reply with what is wrong with it, or ask for a specific change. Replying "
    "\"continue\" will most likely rewrite it again."
)
CAP_TEXT = (
    "[Harness] Stopped: this turn already ran {n} tool calls without "
    "finishing. Report what is done and what is blocked. Do not call another tool. "
    "(If the work really is this long, reply to continue — the count resets on a "
    "new message — or start the proxy with a higher --max-rounds.)"
)

# ---- step judge + retry -------------------------------------------------------
JUDGE_EVERY = 200        # judge the streamed prose every this many new chars
KEEPALIVE_S = 15.0       # SSE comment to the client while a buffered reply is pending
UPSTREAM_WAIT_S = 90.0   # wait this long for an engine that is (re)starting
RETRY_TEMPERATURES = (0.3, 0.2)            # degeneration (loops, soup, runaways): sample tighter
BEHAVIOR_TEMPERATURES = (0.6, 0.8)         # a model set on one wrong action: sample wider so it tries another
BEHAVIORAL = {"noop-edit", "repeat-edit", "bad-args", "missing-args", "shrinking-rewrite", "unverified-claim",
              "unchecked-change", "narration", "partial-rewrite", "regressed-file", "unchecked-final", "code-in-report",
              "sibling-copy"}
QUARANTINE_EDIT = {"noop-edit", "repeat-edit"}   # the retry is offered no edit tool at all
RETRY_SAMPLING = {"top_p": 0.9, "top_k": 40, "min_p": 0.05}
RETRY_PENALTY = 1.15     # last attempt only, and only for loop-shaped failures (it drops drafting)
NON_IDEMPOTENT = {"edit", "patch", "multiedit", "apply_patch", "str_replace", "str_replace_editor"}
# Commands a final report may claim as verification. A claim is only checked when one of these is
# quoted in backticks next to claim wording and no bash call in the turn ran it.
CHECK_COMMANDS = ("node --check", "python3 -m py_compile", "python -m py_compile", "python3 -m pytest", "python -m pytest",
                  "pytest", "npm test", "npm run test", "npx tsc", "tsc", "cargo test", "cargo check", "go test", "go vet",
                  "ruff", "eslint", "make test", "bash -n", "shellcheck", "mypy")
CLAIM_WORDS = ("verif", "no syntax error", "passed", "passes", "succeed", "confirmed", "checked", "no errors", "clean")
FAILED_RESULT = ("error", "not found", "invalid", "failed", "no changes to apply", "could not", "does not exist")
SHRINK_MIN_PREVIOUS = 1500   # chars: only a real file can be "lost"
SHRINK_RATIO = 0.35          # a rewrite below this share of the largest earlier version is a stub (hard)
SHRINK_SOFT_RATIO = 0.60     # below this: retried once (live: "restart the whole file" rewrites at 39-48%)
RETRY_WHY = {
    "loop": "it repeated the same text over and over ({detail}); write each thing once, without running commentary",
    "gibberish": "it turned into unreadable symbol soup",
    "narration": "it described what you were going to do instead of doing it",
    "runaway": "it wrote pages of text without ever making the tool call",
    "leak": "it contained a raw tool-call / channel envelope instead of a parsed tool call, so nothing ran",
    "truncated": "it ran past the {max_tokens}-token limit of one step; do less per step: one file or one command at a time, and never restate a file",
    "silent": "it spent the whole step reasoning silently and never answered; do not deliberate, answer directly with the tool call or one plain sentence",
    "bad-args": "its tool call had unusable arguments ({detail})",
    "shrinking-rewrite": "{detail}; a write replaces the whole file, so it must contain the complete file — use edit for a partial change",
    "repeat-edit": "{detail}; applying the same edit twice changes the file twice and corrupts it — read the file or run a check (node --check, py_compile), or make a different change",
    "missing-args": "{detail}; call the tool again with every required argument",
    "noop-edit": "its edit had identical old and new text, so it changed nothing: the file already says that. Check the file (run its syntax check) or report that the work is done",
    "unchecked-change": "{detail}",
    "partial-rewrite": "{detail}; a write replaces the whole file, so write the complete file, every part of it, or use edit for a partial change",
    "regressed-file": "{detail}; restore the complete file (every part the earlier version had), check it, then report",
    "unchecked-final": "{detail}",
    "sibling-copy": "{detail}",
    "code-in-report": "{detail}",
    "unverified-claim": "{detail}; run it now with the bash tool and report what it prints, or leave the claim out",
}
RETRY_DIRECTIVE = (
    "[Harness] Your previous reply to this step was discarded: {why}. Redo the step now. "
    "If the step needs a tool, answer with the tool call only, no text before it. "
    "If the work is finished, answer in at most five plain lines. No math notation."
)
RETRY_DIRECTIVE_FINAL = (
    "[Harness] Your previous replies to this step were discarded again: {why}. This is the last attempt. "
    "Answer with exactly one tool call and nothing else, or with one plain sentence that says what is done."
)
EXHAUSTED_TEXT = (
    "[Harness] Stopped: the model failed this step {n} times ({reasons}). The failed replies were discarded, "
    "not shown, and saved under {where}. Reply with a smaller, more specific instruction: one file or one "
    "command at a time. Replying \"continue\" will most likely fail the same way."
)
CUT_TEXT = "\n\n[Harness] Output cut: {detail}. Ask again with a narrower question."


@dataclass
class Config:
    defaults: dict[str, float] = field(default_factory=lambda: {"temperature": 0.45, "top_p": 0.9, "top_k": 40, "min_p": 0.05})
    step_max_tokens: int = 6144
    stall_seconds: float = 240.0
    max_attempts: int = 3
    failures_dir: str = ".harness_failures"
    failures_keep: int = 60
    stream_prose: bool = False   # True: stream agentic prose live (cut only, no retry)
    upstream_wait: float = UPSTREAM_WAIT_S
    keepalive: float = KEEPALIVE_S


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = time.time()
        self.counts: dict[str, int] = {"requests": 0, "steps": 0, "retries": 0, "stops": 0, "cuts": 0,
                                       "upstream_errors": 0, "stalls": 0, "repeat_nudges": 0, "repeat_stops": 0, "churn_nudges": 0, "churn_finishes": 0,
                                       "unverified_claims": 0, "unchecked_changes": 0, "check_nudges": 0,
                                       "repaired_calls": 0, "partial_rewrites": 0, "regressed_files": 0,
                                       "unchecked_finals": 0, "code_in_reports": 0, "sibling_copies": 0}
        self.failures: dict[str, int] = {}
        self.last_failure: dict[str, Any] | None = None

    def bump(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def failed(self, verdict: Verdict, attempt: int, where: str) -> None:
        with self.lock:
            self.failures[verdict.reason] = self.failures.get(verdict.reason, 0) + 1
            self.last_failure = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": verdict.reason,
                                 "detail": verdict.detail, "attempt": attempt, "saved": where}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"uptime_s": round(time.time() - self.started), "counts": dict(self.counts),
                    "failures": dict(self.failures), "last_failure": self.last_failure}


STATS = Stats()


# ---- repeat guard (unchanged logic) --------------------------------------------

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
    """Tool calls after the latest user message, in order, paired with results. The proxy's own
    "[Harness] …" directives are user-role messages in the forwarded copy only; they do not start a
    turn (counting them hid the whole turn from every check after a nudge)."""

    last_user = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user" and not _is_harness(message):
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


def _is_harness(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and content.startswith("[Harness]")


def decide(messages: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    """(action, text, directive): ("stop", text, None) ends the turn; ("nudge",
    refusal, directive) replaces the latest tool result and forwards; (None,
    None, None) forwards untouched. Only rounds after the latest user message
    count, and only when the offending call is the latest round."""

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
            if _tool_name(sig) in NON_IDEMPOTENT:
                # the client already ran it again: say so instead of "it was not run again"
                return "nudge", REAPPLIED_TEXT.format(tool=_tool_name(sig), n=n, path=_display_command(sig), output=quoted), directive
            return "nudge", REFUSAL_TEXT.format(n=n, output=quoted), directive
    # The same call re-issued after a file changed is verification (write -> node --check -> write ->
    # node --check), which the check nudge asks for; only a re-issue with no change in between is a cycle.
    reissued_after_change = n >= 2 and any(_tool_name(rounds[k][0]) in STATE_CHANGING
                                           for k in range(indices[-2] + 1, indices[-1]))
    if n >= STOP_CYCLE and not reissued_after_change:
        return "stop", CYCLE_TEXT.format(n=n, command=_display_command(sig)), None
    if n >= NUDGE_CYCLE and not reissued_after_change:
        return "nudge", CYCLE_REFUSAL_TEXT.format(n=n, output=quoted), directive
    # Churn: the same file rewritten again and again (each write differs, so
    # the repeat checks never see it). Live: z/tetris.js written 7 times in
    # 50 s with no check in between. The result stays; a directive is added.
    path = _write_path(sig)
    if path:
        writes = sum(1 for other, _ in rounds if _write_path(other) == path)
        if writes >= WRITE_CHURN_STOP:
            return "stop", CHURN_STOP_TEXT.format(n=writes, path=path), None
        if writes >= WRITE_CHURN_FINISH:
            return "churn-finish", None, CHURN_FINISH_TEXT.format(n=writes, path=path)
        if writes >= WRITE_CHURN_NUDGE:
            return "churn", None, CHURN_DIRECTIVE_TEXT.format(n=writes, path=path)
    return None, None, None


def _write_path(sig: str) -> str | None:
    """The file a `write` signature targets, else None."""

    name, _, args = sig.partition("\n")
    if name != "write":
        return None
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("filePath", "file_path", "path"):
        if isinstance(parsed.get(key), str) and parsed[key]:
            return parsed[key]
    return None


def _display_command(sig: str) -> str:
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


# ---- synthetic completions -------------------------------------------------------

def _chunk(completion_id: str, model: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    body = {"id": completion_id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n".encode()


def _completion(text: str, *, model: str, stream: bool) -> bytes:
    completion_id = "chatcmpl-harness-" + uuid.uuid4().hex[:12]
    if not stream:
        body = {"id": completion_id, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        return json.dumps(body).encode()
    return (_chunk(completion_id, model, {"role": "assistant", "content": text})
            + _chunk(completion_id, model, {}, "stop") + b"data: [DONE]\n\n")


# ---- request hygiene -------------------------------------------------------------

def prepare_request(payload: dict[str, Any], cfg: Config, *, agentic: bool | None = None) -> dict[str, Any]:
    """Sampling defaults for omitted fields, the per-step token cap for agentic
    requests, include_usage on streams. Returns a summary for the log.
    `agentic` overrides the tools-present test (a churn-finish reply has had
    its tools withheld but is still an agentic step)."""

    if agentic is None:
        agentic = bool(payload.get("tools"))
    for key, value in cfg.defaults.items():
        if payload.get(key) is None:
            payload[key] = value
    if agentic and cfg.step_max_tokens:
        asked = payload.get("max_tokens") or payload.get("max_completion_tokens")
        try:
            asked = int(asked) if asked is not None else None
        except (TypeError, ValueError):
            asked = None
        payload["max_tokens"] = min(asked, cfg.step_max_tokens) if asked else cfg.step_max_tokens
        payload.pop("max_completion_tokens", None)
    if payload.get("stream"):
        options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
        options.setdefault("include_usage", True)
        payload["stream_options"] = options
    return {"agentic": agentic, "max_tokens": payload.get("max_tokens"), "temperature": payload.get("temperature")}


def turn_checks(payload: dict[str, Any], tool_calls: list[dict[str, Any]]) -> Verdict:
    """Checks that need the request as well as the reply, run before the client executes anything:
    required arguments against the offered schemas, an edit already applied this turn, and a write
    that replaces an earlier file with a stub."""

    for check in (missing_required(payload.get("tools") or [], tool_calls),
                  repeated_edit(payload.get("messages") or [], tool_calls),
                  shrinking_rewrite(payload.get("messages") or [], tool_calls)):
        if not check.ok:
            return check
    return PASS


def unverified_claim(messages: list[dict[str, Any]], content: str, tool_calls: list[dict[str, Any]]) -> Verdict:
    """A final report that cites a check command as verification although no bash call in this turn ran
    it. Live: "Final verification: `node --check z/tetris.js` (No syntax errors found)" after a turn of
    one mkdir and five writes. Only final replies (no tool calls) with claim wording are checked."""

    if tool_calls or not content:
        return PASS
    lowered = content.lower()
    if not any(word in lowered for word in CLAIM_WORDS):
        return PASS
    ran = []
    for sig, _ in _rounds_after_last_user(messages):
        name, _, args = sig.partition("\n")
        if name == "bash":
            try:
                ran.append(" ".join(str(json.loads(args).get("command", "")).split()))
            except (json.JSONDecodeError, AttributeError):
                ran.append(args)
    for span in re.findall(r"`([^`\n]{3,160})`", content):
        command = re.sub(r"^\s*(?:bash:\s*|\$\s*)", "", span).strip()
        check = next((c for c in CHECK_COMMANDS if command == c or command.startswith(c + " ")), None)
        if check and not any(check in done for done in ran):
            return Verdict(False, "unverified-claim", f"the report cites `{command}` as verification, but no such command ran in this turn")
    return PASS


SYNTAX_CHECKS = {".js": "node --check {path}", ".mjs": "node --check {path}", ".cjs": "node --check {path}",
                 ".py": "python3 -m py_compile {path}", ".sh": "bash -n {path}", ".json": "python3 -m json.tool {path}"}


CHECK_NUDGE_TEXT = (
    "[Harness] {path} just changed. Run `{command}` with the bash tool now, and fix what it reports before "
    "changing the file again. If it passes and the work is done, report."
)


def pending_check(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    """(path, command) when the latest round changed a code file successfully: the next request is steered
    to its syntax check before the model generates anything, which costs nothing when it complies.
    unchecked_change() is the backstop when it does not."""

    rounds = _rounds_after_last_user(messages)
    if not rounds:
        return None
    sig, result = rounds[-1]
    name, _, raw = sig.partition("\n")
    if name not in STATE_CHANGING or any(w in result.lower() for w in FAILED_RESULT):
        return None
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None
    path = next((args[k] for k in ("filePath", "file_path", "path") if isinstance(args.get(k), str)), "") if isinstance(args, dict) else ""
    check = SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower())
    return (path, check.format(path=path)) if check else None


REPORT_CODE_MIN = 1500      # chars of fenced code in a final report that belong in a file instead


def _change_state(messages: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    """(round of the last successful change, round of the last look) per path in this turn. A look is a
    read of the path or a bash command naming its file."""

    changed_at: dict[str, int] = {}
    inspected_at: dict[str, int] = {}
    for index, (sig, result) in enumerate(_rounds_after_last_user(messages)):
        name, _, raw = sig.partition("\n")
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            continue
        path = next((args[k] for k in ("filePath", "file_path", "path") if isinstance(args.get(k), str)), "")
        if name in STATE_CHANGING and path and not any(w in result.lower() for w in FAILED_RESULT):
            changed_at[path] = index
        elif name == "read" and path:
            inspected_at[path] = index
        elif name == "bash":
            command = str(args.get("command") or "")
            for known in list(changed_at):
                if os.path.basename(known) in command:
                    inspected_at[known] = index
    return changed_at, inspected_at


def unchecked_final(messages: list[dict[str, Any]], content: str, tool_calls: list[dict[str, Any]]) -> Verdict:
    """A final report while a code file changed after its last look. Live: the last write of tetris.js was
    never checked, the report said the game was done, and the file failed `node --check`."""

    if tool_calls or not content:
        return PASS
    changed_at, inspected_at = _change_state(messages)
    pending = [(path, SYNTAX_CHECKS[os.path.splitext(path)[1].lower()].format(path=path))
               for path, at in changed_at.items()
               if SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower()) and inspected_at.get(path, -1) < at]
    if not pending:
        return PASS
    shown = pending[:8]          # one retry covers every unchecked file, not one retry per file
    command = " && ".join(c for _, c in shown)
    names = ", ".join(p for p, _ in shown) + (f" and {len(pending) - len(shown)} more" if len(pending) > len(shown) else "")
    return Verdict(False, "unchecked-final", f"{names} changed after the last check; run `{command}` with the bash tool "
                                            "before reporting, fix what it reports, then report")


_SIBLING = re.compile(r"^(?P<stem>.+?)[-_. ](?:v\d+|new|final|fixed|fix|copy|clean|backup|old|\d+)$", re.I)


def sibling_copy(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> Verdict:
    """A write to a versioned copy of a file this turn already wrote (tetris.js -> tetris_v2.js,
    tetris_final.js, tetris_new.js). Live: blocked from replacing tetris.js with stubs, the model wrote
    them to new names instead, where none of the size checks look, and none of the copies was the game."""

    changed_at, _ = _change_state(messages)
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if fn.get("name") != "write":
            continue
        path, _ = _write_target(fn.get("arguments"))
        if not path or path in changed_at:
            continue
        folder, name = os.path.split(path)
        stem, ext = os.path.splitext(name)
        match = _SIBLING.match(stem)
        if not match:
            continue
        original = os.path.join(folder, match.group("stem") + ext)
        if original in changed_at:
            return Verdict(False, "sibling-copy", f"it wrote {path}, a copy of {original} which this turn already wrote; "
                                                  f"put the complete file in {original} itself (the user runs that one), "
                                                  "or fix it there with an edit")
    return PASS


def code_in_report(messages: list[dict[str, Any]], content: str, tool_calls: list[dict[str, Any]]) -> Verdict:
    """A final report that carries a file's worth of code while the turn wrote code files: the code belongs
    in the file. Live: a 6,006-char report with the complete game in a code block, over a broken tetris.js."""

    if tool_calls or not content:
        return PASS
    code = sum(len(block) for block in re.findall(r"```[^\n]*\n(.*?)```", content, flags=re.S))
    if code < REPORT_CODE_MIN:
        return PASS
    changed_at, _ = _change_state(messages)
    written = [p for p in changed_at if SYNTAX_CHECKS.get(os.path.splitext(p)[1].lower())]
    if not written:
        return PASS
    return Verdict(False, "code-in-report", f"the report carries {code} chars of code instead of putting it in a file; "
                                            f"if it is the complete {written[-1]}, write it there with the write tool and "
                                            "run its check, then report in a few lines without code")


def unchecked_change(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> Verdict:
    """A change to a code file that already changed in this turn and has not been looked at since (no
    bash command naming it, no read of it). Live: two blind edits in a row broke tetris.js, then the
    model got stuck. One retry asks for the syntax check first; the change is allowed after that."""

    changed_at: dict[str, int] = {}
    inspected_at: dict[str, int] = {}
    rounds = _rounds_after_last_user(messages)
    for index, (sig, result) in enumerate(rounds):
        name, _, raw = sig.partition("\n")
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            continue
        path = next((args[k] for k in ("filePath", "file_path", "path") if isinstance(args.get(k), str)), "")
        if name in STATE_CHANGING and path and not any(w in result.lower() for w in FAILED_RESULT):
            changed_at[path] = index
        elif name == "read" and path:
            inspected_at[path] = index
        elif name == "bash":
            command = str(args.get("command") or "")
            for known in list(changed_at):
                if os.path.basename(known) in command:
                    inspected_at[known] = index
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if fn.get("name") not in STATE_CHANGING:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        path = next((args[k] for k in ("filePath", "file_path", "path") if isinstance(args.get(k), str)), "") if isinstance(args, dict) else ""
        check = SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower())
        if check and path in changed_at and inspected_at.get(path, -1) < changed_at[path]:
            command = check.format(path=path)
            return Verdict(False, "unchecked-change",
                           f"{path} already changed in this turn and has not been checked since; run `{command}` "
                           "with the bash tool first, then fix what it reports")
    return PASS


def missing_required(tools: list[Any], tool_calls: list[dict[str, Any]]) -> Verdict:
    """A call that lacks a key its tool's own JSON schema requires. Live: `edit` without `oldString`,
    which the client rejects after a wasted step."""

    required: dict[str, list[str]] = {}
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else {}
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        if fn.get("name") and isinstance(params.get("required"), list):
            required[str(fn["name"])] = [str(k) for k in params["required"]]
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        if name not in required:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
        except json.JSONDecodeError:
            continue   # unparseable arguments are the judge's bad-args case
        if not isinstance(args, dict):
            continue
        missing = [key for key in required[name] if key not in args]
        if missing:
            return Verdict(False, "missing-args", f"the {name} call lacks the required argument(s) {', '.join(missing)}")
    return PASS


def repeated_edit(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> Verdict:
    """An edit identical to one already applied successfully in this turn. The client would apply it
    again: live, OpenCode's loose matching reported "Edit applied successfully" four times for one
    edit and left the file unparseable. Only a successful earlier application counts: an edit that
    failed may legitimately succeed after the file changed."""

    applied: set[str] = set()
    for sig, result in _rounds_after_last_user(messages):
        if _tool_name(sig) in NON_IDEMPOTENT and not any(word in result.lower() for word in FAILED_RESULT):
            applied.add(sig)
    for call in tool_calls:
        _, sig = _tool_sig(call)
        if _tool_name(sig) in NON_IDEMPOTENT and sig in applied:
            path = _display_command(sig)
            return Verdict(False, "repeat-edit", f"it repeated an {_tool_name(sig)} of {path} that was already applied in this turn")
    return PASS


def _largest_writes(messages: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    """(largest, latest) content size per path written in this turn."""

    largest: dict[str, int] = {}
    latest: dict[str, int] = {}
    for sig, result in _rounds_after_last_user(messages):
        name, _, raw = sig.partition("\n")
        if name != "write" or any(w in result.lower() for w in FAILED_RESULT):
            continue
        path, size = _write_target(raw)
        if path:
            largest[path] = max(largest.get(path, 0), size)
            latest[path] = size
    return largest, latest


def partial_rewrite(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> Verdict:
    """A write between the stub floor and SHRINK_SOFT_RATIO of the largest earlier version: soft, one retry."""

    largest, _ = _largest_writes(messages)
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if fn.get("name") != "write":
            continue
        path, size = _write_target(fn.get("arguments"))
        before = largest.get(path or "", 0)
        if path and before >= SHRINK_MIN_PREVIOUS and before * SHRINK_RATIO <= size < before * SHRINK_SOFT_RATIO:
            return Verdict(False, "partial-rewrite", f"it rewrote {path} with {size} chars where the version written earlier this turn had {before}")
    return PASS


def regressed_file(messages: list[dict[str, Any]], content: str, tool_calls: list[dict[str, Any]]) -> Verdict:
    """A final report while a code file this turn wrote is below SHRINK_SOFT_RATIO of its own largest version.
    Live: a report claiming a finished, verified game over a 2,989-char tetris.js that had been 7,543."""

    if tool_calls or not content:
        return PASS
    largest, latest = _largest_writes(messages)
    for path, size in latest.items():
        peak = largest.get(path, 0)
        if SYNTAX_CHECKS.get(os.path.splitext(path)[1].lower()) and peak >= SHRINK_MIN_PREVIOUS and size < peak * SHRINK_SOFT_RATIO:
            return Verdict(False, "regressed-file", f"{path} is now {size} chars, down from {peak} earlier in this turn, "
                                                   "so the latest version is missing parts of the earlier one")
    return PASS


def shrinking_rewrite(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> Verdict:
    """A `write` whose content is a fraction of what this turn wrote to the same
    path before is a stub replacing a full file (live: the last of ten rewrites
    of z/tetris.js was 542 bytes over a complete game). Fail it so the step is
    retried with the rule spelled out."""

    largest: dict[str, int] = {}
    last_user = max((i for i, m in enumerate(messages)
                     if isinstance(m, dict) and m.get("role") == "user" and not _is_harness(m)), default=-1)
    for message in messages[last_user + 1:]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
            if fn.get("name") != "write":
                continue
            path, size = _write_target(fn.get("arguments"))
            if path:
                largest[path] = max(largest.get(path, 0), size)
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if fn.get("name") != "write":
            continue
        path, size = _write_target(fn.get("arguments"))
        before = largest.get(path or "", 0)
        if path and before >= SHRINK_MIN_PREVIOUS and size < before * SHRINK_RATIO:
            return Verdict(False, "shrinking-rewrite",
                           f"it rewrote {path} with {size} chars where the version written earlier this turn had {before}")
    return PASS


def _write_target(arguments: Any) -> tuple[str | None, int]:
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return None, 0
    if not isinstance(parsed, dict):
        return None, 0
    path = next((parsed[k] for k in ("filePath", "file_path", "path") if isinstance(parsed.get(k), str)), None)
    content = parsed.get("content")
    return path, len(content) if isinstance(content, str) else 0


def retry_payload(original: dict[str, Any], attempt: int, verdict: Verdict, max_attempts: int) -> dict[str, Any]:
    """The request for attempt `attempt` (2, 3, …): a corrective user message,
    a fresh seed (an omitted seed is keyed to the prompt, so the same prompt
    reproduces the same failure), a lower temperature, min_p; the last attempt
    adds a repetition penalty when the failure was loop-shaped."""

    payload = json.loads(json.dumps(original))
    why = RETRY_WHY.get(verdict.reason, verdict.detail or verdict.reason).format(
        max_tokens=original.get("max_tokens"), detail=verdict.detail)
    text = (RETRY_DIRECTIVE_FINAL if attempt >= max_attempts else RETRY_DIRECTIVE).format(why=why)
    payload["messages"] = list(original.get("messages") or []) + [{"role": "user", "content": text}]
    payload["seed"] = random.SystemRandom().randrange(1, 2**31)
    schedule = BEHAVIOR_TEMPERATURES if verdict.reason in BEHAVIORAL else RETRY_TEMPERATURES
    payload["temperature"] = schedule[min(attempt - 2, len(schedule) - 1)]
    payload.update(RETRY_SAMPLING)
    if verdict.reason in QUARANTINE_EDIT and isinstance(payload.get("tools"), list):
        kept = [t for t in payload["tools"]
                if not (isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name") in NON_IDEMPOTENT)]
        if kept and len(kept) < len(payload["tools"]):
            payload["tools"] = kept
            choice = payload.get("tool_choice")
            if isinstance(choice, dict) and (choice.get("function") or {}).get("name") in NON_IDEMPOTENT:
                payload.pop("tool_choice", None)
            payload["messages"][-1]["content"] += " The edit tool is not available for this retry."
    if attempt >= max_attempts and verdict.reason in ("loop", "gibberish", "runaway"):
        payload["repetition_penalty"] = RETRY_PENALTY
    return payload


# ---- one upstream attempt --------------------------------------------------------

class Reply:
    """What one upstream attempt has produced so far, parsed from its SSE
    events (or from its JSON body when not streaming)."""

    def __init__(self) -> None:
        self.events: list[bytes] = []
        self.content = ""
        self.reasoning = ""      # reasoning_content deltas: relayed as-is, kept for the judge and the dump
        self.calls: dict[int, dict[str, str]] = {}
        self.finish: str | None = None
        self.usage: dict[str, Any] | None = None
        self.completion_id: str | None = None
        self.model: str | None = None
        self.error: Any = None
        self.done = False
        self.judged_at = 0
        self.repaired: list[str] = []     # "write.filePath" … fixed by the proxy; the reply is rebuilt on relay

    def feed_event(self, raw: bytes) -> None:
        self.events.append(raw)
        for line in raw.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                self.done = True
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            self.feed_object(obj, streaming=True)

    def feed_object(self, obj: dict[str, Any], *, streaming: bool) -> None:
        if not isinstance(obj, dict):
            return
        if "error" in obj:
            self.error = obj["error"]
            return
        self.completion_id = obj.get("id") or self.completion_id
        self.model = obj.get("model") or self.model
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            part = choice.get("delta") if streaming else choice.get("message")
            part = part if isinstance(part, dict) else {}
            if isinstance(part.get("content"), str):
                self.content += part["content"]
            for key in ("reasoning_content", "reasoning"):
                if isinstance(part.get(key), str):
                    self.reasoning += part[key]
            for call in part.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", len(self.calls)) if streaming else len(self.calls)
                slot = self.calls.setdefault(int(index), {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = str(call["id"])
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                if fn.get("name"):
                    slot["name"] = str(fn["name"])
                if fn.get("arguments"):
                    slot["arguments"] += str(fn["arguments"])
            if choice.get("finish_reason"):
                self.finish = str(choice["finish_reason"])

    def repair_paths(self) -> bool:
        """Strip backticks / quotes around path arguments in place; True when anything changed."""

        for slot in self.calls.values():
            fixed = repair_path_args(slot["arguments"])
            if fixed:
                slot["arguments"] = fixed[0]
                self.repaired += [f"{slot['name']}.{key}" for key in fixed[1]]
        return bool(self.repaired)

    def tool_calls(self) -> list[dict[str, Any]]:
        return [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for _, c in sorted(self.calls.items())]

    def completion_tokens(self) -> int | None:
        if isinstance(self.usage, dict) and isinstance(self.usage.get("completion_tokens"), int):
            return self.usage["completion_tokens"]
        return None

    def judge_now(self) -> Verdict:
        self.judged_at = len(self.content)
        return judge_prose(self.content, agentic=True, finished=False)


@dataclass
class Attempt:
    kind: str                     # pass | fail | cut | error | client-gone | passthrough
    reply: Reply | None = None
    verdict: Verdict = PASS
    status: int = 200
    body: bytes = b""             # non-stream body to relay / error body
    detail: str = ""
    retryable: bool = False
    headers: list[tuple[str, str]] = field(default_factory=list)


def _reader(resp: http.client.HTTPResponse, out: queue.Queue) -> None:
    """Upstream SSE bytes → whole events on the queue. Runs in its own thread so
    the handler can send keepalives and notice a client that left."""

    buf = b""
    try:
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                out.put(("event", event + b"\n\n"))
        if buf.strip():
            out.put(("event", buf))
        out.put(("eof", None))
    except Exception as exc:  # socket timeout (stall), reset, cancelled by us
        out.put(("error", exc))


class Proxy(BaseHTTPRequestHandler):
    upstream_host = "127.0.0.1"
    upstream_port = 8104
    protocol_version = "HTTP/1.1"
    cfg = Config()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[harness] " + (fmt % args) + "\n")
        sys.stderr.flush()

    def handle(self) -> None:
        # A client that aborts (Esc, timeout, our own synthetic stop) drops the
        # socket; the next read/write raises inside http.server. End quietly.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # -- routing ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/harness/health":
            self._send_health()
            return
        self._passthrough(b"")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length else b""
        STATS.bump("requests")
        if not (self.path.split("?", 1)[0].rstrip("/").endswith("/chat/completions") and body):
            self._passthrough(body)
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            self._passthrough(body)
            return
        action, text, directive = decide(payload["messages"])
        if action == "stop":
            STATS.bump("repeat_stops")
            self._send_synthetic(payload, text or "", headers_sent=False)
            self.log_message("stopped turn: %s", (text or "")[len("[Harness] Stopped: "):76])
            return
        agentic_override = None
        if action in ("churn", "churn-finish") and directive:
            STATS.bump("churn_nudges")
            payload["messages"].append({"role": "user", "content": directive})
            payload["seed"] = random.SystemRandom().randrange(1, 2**31)
            if action == "churn-finish":
                STATS.bump("churn_finishes")
                changed_at, inspected_at = _change_state(payload["messages"])
                unchecked = [(p, SYNTAX_CHECKS[os.path.splitext(p)[1].lower()].format(path=p)) for p, at in changed_at.items()
                             if SYNTAX_CHECKS.get(os.path.splitext(p)[1].lower()) and inspected_at.get(p, -1) < at]
                tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
                kept = [t for t in tools if not (isinstance(t, dict) and (t.get("function") or {}).get("name") == "write")]
                if unchecked and kept:
                    # The file changed since its last check: withhold only write, so the model can still
                    # check it, fix a syntax error with a small edit, and report honestly. (Withholding
                    # every tool here sent a report over a broken tetris.js the model could not check.)
                    payload["tools"] = kept
                    payload["messages"][-1]["content"] = CHURN_CHECK_TEXT.format(
                        path=unchecked[0][0], n=directive.split(" times", 1)[0].rsplit(" ", 1)[-1] if " times" in directive else "several",
                        command=" && ".join(c for _, c in unchecked[:8]))
                else:
                    # Everything is checked: withhold the tools for this one reply so the model sums up in
                    # prose (the turn ends with its own report instead of a stop message).
                    agentic_override = bool(payload.get("tools"))
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    payload.pop("parallel_tool_calls", None)
            self.log_message("%s: %s", action, directive[len("[Harness] "):90])
        if action is None:
            pending = pending_check(payload["messages"])
            if pending:
                STATS.bump("check_nudges")
                payload["messages"].append({"role": "user", "content": CHECK_NUDGE_TEXT.format(path=pending[0], command=pending[1])})
        if action == "nudge" and text and apply_nudge(payload["messages"], text, directive):
            STATS.bump("repeat_nudges")
            # Rewording the history alone did not break a loop of identical
            # edits at temperature 0.35, so a nudge also re-seeds the draw and,
            # from the directive tier, samples wider; the judge still guards it.
            payload["seed"] = random.SystemRandom().randrange(1, 2**31)
            if directive:
                payload["temperature"] = max(float(payload.get("temperature") or 0.0), NUDGE_TEMPERATURE)
            self.log_message("refused repeat%s: %s", " + directive" if directive else "", text[len("[Harness] "):80])
        self._chat(payload, agentic_override=agentic_override)

    # -- health -------------------------------------------------------------------

    def _send_health(self) -> None:
        upstream_ok, models = False, []
        try:
            conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=3)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            raw = resp.read()
            upstream_ok = resp.status == 200
            if upstream_ok:
                models = [m.get("id") for m in (json.loads(raw).get("data") or []) if isinstance(m, dict)]
            conn.close()
        except (OSError, ValueError):
            upstream_ok = False
        cfg = self.cfg
        body = {"ok": upstream_ok, "upstream": f"{self.upstream_host}:{self.upstream_port}", "upstream_ok": upstream_ok,
                "models": models, **STATS.snapshot(),
                "config": {"defaults": cfg.defaults, "step_max_tokens": cfg.step_max_tokens,
                           "stall_seconds": cfg.stall_seconds, "max_attempts": cfg.max_attempts,
                           "failures_dir": cfg.failures_dir, "stream_prose": cfg.stream_prose,
                           "max_tool_rounds": MAX_TOOL_ROUNDS}}
        raw = json.dumps(body, indent=1).encode()
        self.send_response(200 if upstream_ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    # -- plain relay (GET, non-chat POST) ------------------------------------------

    def _connect(self, body: bytes, *, timeout: float) -> tuple[http.client.HTTPConnection | None, http.client.HTTPResponse | None, str, str]:
        """Send the request upstream. Returns (conn, resp, error, kind); kind is
        "" on success, "down" when the engine refused for cfg.upstream_wait
        seconds (it is (re)starting: keep trying instead of failing the turn),
        "stall" when it accepted the request but sent no headers in `timeout`."""

        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
        if body:
            headers["Content-Length"] = str(len(body))
        deadline = time.time() + self.cfg.upstream_wait
        last = ""
        while True:
            conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=timeout)
            try:
                conn.request(self.command, self.path, body=body or None, headers=headers)
                return conn, conn.getresponse(), "", ""
            except socket.timeout as exc:
                conn.close()
                return None, None, f"no response headers from the engine in {timeout:.0f}s ({exc})", "stall"
            except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                conn.close()
                if time.time() >= deadline:
                    return None, None, last, "down"
                time.sleep(2.0)
            except OSError as exc:
                conn.close()
                return None, None, f"{type(exc).__name__}: {exc}", "down"

    def _passthrough(self, body: bytes) -> None:
        conn, resp, error, _kind = self._connect(body, timeout=self.cfg.stall_seconds)
        if resp is None or conn is None:
            STATS.bump("upstream_errors")
            self._send_json({"error": {"message": f"tensorfold upstream unavailable: {error}", "type": "server_error"}}, 503)
            return
        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()

    def _send_json(self, body: dict[str, Any], status: int) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    # -- chat completions: judge + retry -------------------------------------------

    def _chat(self, payload: dict[str, Any], *, agentic_override: bool | None = None) -> None:
        cfg = self.cfg
        summary = prepare_request(payload, cfg, agentic=agentic_override)
        agentic, stream = summary["agentic"], bool(payload.get("stream"))
        model = str(payload.get("model") or "local-model")
        live = stream and (not agentic or cfg.stream_prose)
        if agentic:
            STATS.bump("steps")
        started = time.time()
        self._sse_open = False      # opened by _attempt once the engine answers 200 (see _open_sse)
        verdicts: list[Verdict] = []
        attempt = 0
        while True:
            attempt += 1
            # A judged failure retries with a directive and fresh sampling; a
            # transient upstream error just resends the same request.
            request = retry_payload(payload, attempt, verdicts[-1], cfg.max_attempts) if verdicts else payload
            body = json.dumps(request).encode()
            result = self._attempt(body, agentic=agentic, stream=stream, live=live, max_tokens=request.get("max_tokens"))
            reply = result.reply
            calls = [c["function"]["name"] for c in reply.tool_calls()] if reply else []
            self.log_message(
                "%s attempt %d/%d: %s finish=%s calls=%s text=%dch tokens=%s %.1fs%s",
                "step" if agentic else "chat", attempt, cfg.max_attempts, result.kind,
                reply.finish if reply else None, calls, len(reply.content) if reply else 0,
                reply.completion_tokens() if reply else None, time.time() - started,
                f" [{result.verdict.reason}: {result.verdict.detail}]" if result.kind == "fail" else
                (f" [{result.detail}]" if result.detail else ""),
            )
            if result.kind == "pass" and agentic and reply is not None:
                turn = turn_checks(payload, reply.tool_calls())
                if turn.ok:
                    # Soft checks: one retry to get a verification step in; never a reason to stop the turn.
                    # Most severe first: the work is not in the file, the file lost parts, it was not checked,
                    # the report claims a check that never ran; then the two checks on tool calls.
                    history = payload.get("messages") or []
                    for soft, counter in ((code_in_report(history, reply.content, reply.tool_calls()), "code_in_reports"),
                                          (regressed_file(history, reply.content, reply.tool_calls()), "regressed_files"),
                                          (unchecked_final(history, reply.content, reply.tool_calls()), "unchecked_finals"),
                                          (unverified_claim(history, reply.content, reply.tool_calls()), "unverified_claims"),
                                          (sibling_copy(history, reply.tool_calls()), "sibling_copies"),
                                          (partial_rewrite(history, reply.tool_calls()), "partial_rewrites"),
                                          (unchecked_change(history, reply.tool_calls()), "unchecked_changes")):
                        if soft.ok:
                            continue
                        STATS.bump(counter)
                        if attempt == 1:
                            turn = soft
                        else:
                            self.log_message("step attempt %d/%d: passing despite [%s]", attempt, cfg.max_attempts, soft.detail)
                        break
                if not turn.ok:
                    result = Attempt("fail", reply=reply, verdict=turn)
                    self.log_message("step attempt %d/%d: fail [%s: %s]", attempt, cfg.max_attempts, turn.reason, turn.detail)
            if result.kind == "pass":
                self._relay_pass(result, stream=stream, headers_sent=self._sse_open)
                return
            if result.kind in ("cut", "client-gone", "passthrough"):
                return
            if result.kind == "error":
                if result.retryable and attempt < cfg.max_attempts:
                    time.sleep(1.0)
                    continue
                STATS.bump("upstream_errors")
                self._send_error(result, stream=stream, headers_sent=self._sse_open)
                return
            # kind == "fail": a judged failure of this attempt
            verdicts.append(result.verdict)
            where = self._save_failure(request, reply, result.verdict, attempt, summary)
            STATS.failed(result.verdict, attempt, where)
            if attempt >= cfg.max_attempts:
                STATS.bump("stops")
                reasons = ", ".join(dict.fromkeys(v.reason for v in verdicts))
                text = EXHAUSTED_TEXT.format(n=attempt, reasons=reasons, where=cfg.failures_dir or "(not saved)")
                self._send_synthetic({"stream": stream, "model": model}, text, headers_sent=self._sse_open)
                return
            STATS.bump("retries")

    def _attempt(self, body: bytes, *, agentic: bool, stream: bool, live: bool, max_tokens: Any) -> Attempt:
        cfg = self.cfg
        timeout = cfg.stall_seconds if stream else cfg.stall_seconds * 3
        conn, resp, error, kind = self._connect(body, timeout=timeout)
        if resp is None or conn is None:
            if kind == "stall":
                STATS.bump("stalls")
                return Attempt("error", detail=f"stalled: {error}", status=504, retryable=True,
                               body=json.dumps({"error": {"message": f"tensorfold upstream stalled: {error}", "type": "server_error"}}).encode())
            return Attempt("error", detail=f"upstream unavailable: {error}", status=503, retryable=False,
                           body=json.dumps({"error": {"message": f"tensorfold upstream unavailable: {error}", "type": "server_error"}}).encode())
        if resp.status != 200:
            raw = resp.read()
            conn.close()
            return Attempt("error", status=resp.status, body=raw, retryable=resp.status >= 500,
                           detail=f"upstream HTTP {resp.status}: {raw[:160]!r}",
                           headers=[(k, v) for k, v in resp.getheaders() if k.lower() not in ("transfer-encoding", "connection", "content-length")])
        if stream:
            self._open_sse()
        reply = Reply()
        try:
            if not stream:
                raw = resp.read()
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    return Attempt("passthrough", body=raw, status=200)
                reply.feed_object(obj, streaming=False)
                reply.events = [raw]
                verdict = judge_reply(reply.content, reply.tool_calls(), reply.finish, agentic=agentic,
                                      completion_tokens=reply.completion_tokens(), max_tokens=_int_or_none(max_tokens),
                                      reasoning=reply.reasoning)
                if not verdict.ok and verdict.reason == "bad-args" and reply.repair_paths():
                    verdict = judge_reply(reply.content, reply.tool_calls(), reply.finish, agentic=agentic,
                                          completion_tokens=reply.completion_tokens(), max_tokens=_int_or_none(max_tokens),
                                          reasoning=reply.reasoning)
                    if verdict.ok:
                        STATS.bump("repaired_calls")
                        self.log_message("repaired %s instead of failing the step", ", ".join(reply.repaired))
                return Attempt("pass" if verdict.ok else "fail", reply=reply, verdict=verdict, body=raw)
            return self._stream_attempt(resp, reply, agentic=agentic, live=live, max_tokens=_int_or_none(max_tokens))
        except socket.timeout:
            STATS.bump("stalls")
            return Attempt("error", detail=f"no bytes from the engine for {timeout:.0f}s", retryable=True, status=504,
                           body=json.dumps({"error": {"message": "tensorfold upstream stalled", "type": "server_error"}}).encode())
        except (ConnectionResetError, BrokenPipeError, http.client.HTTPException) as exc:
            return Attempt("error", detail=f"upstream connection failed: {exc}", retryable=True, status=502,
                           body=json.dumps({"error": {"message": f"tensorfold upstream connection failed: {exc}", "type": "server_error"}}).encode())
        finally:
            conn.close()

    def _stream_attempt(self, resp: http.client.HTTPResponse, reply: Reply, *, agentic: bool, live: bool, max_tokens: int | None) -> Attempt:
        cfg = self.cfg
        events: queue.Queue = queue.Queue()
        threading.Thread(target=_reader, args=(resp, events), daemon=True).start()
        while True:
            try:
                kind, item = events.get(timeout=cfg.keepalive)
            except queue.Empty:
                if not live:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        resp.close()
                        return Attempt("client-gone")
                continue
            if kind == "error":
                if isinstance(item, socket.timeout):
                    raise item
                return Attempt("error", detail=f"upstream stream broke: {item}", retryable=True, status=502,
                               body=json.dumps({"error": {"message": f"tensorfold stream broke: {item}", "type": "server_error"}}).encode())
            if kind == "eof":
                break
            reply.feed_event(item)
            if reply.error is not None:
                return Attempt("error", detail=f"engine error: {json.dumps(reply.error)[:200]}", retryable=False, status=502,
                               body=json.dumps({"error": reply.error}).encode())
            if live:
                try:
                    self.wfile.write(item)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    resp.close()
                    return Attempt("client-gone")
            if len(reply.content) - reply.judged_at >= JUDGE_EVERY:
                verdict = (reply.judge_now() if agentic else
                           judge_prose(reply.content, agentic=False, finished=False))
                if not verdict.ok:
                    resp.close()   # the engine sees the disconnect and cancels the job
                    if live:
                        STATS.bump("cuts")
                        cid = reply.completion_id or "chatcmpl-harness-cut"
                        model = reply.model or "local-model"
                        self._write(_chunk(cid, model, {"content": CUT_TEXT.format(detail=verdict.detail)})
                                    + _chunk(cid, model, {}, "stop") + b"data: [DONE]\n\n")
                        self.log_message("cut live reply: %s", verdict.detail)
                        return Attempt("cut", reply=reply, verdict=verdict)
                    return Attempt("fail", reply=reply, verdict=verdict)
            if reply.done:
                break
        if live:
            if not reply.done:
                self._write(b"data: [DONE]\n\n")
            return Attempt("passthrough", reply=reply)
        verdict = judge_reply(reply.content, reply.tool_calls(), reply.finish, agentic=agentic,
                              completion_tokens=reply.completion_tokens(), max_tokens=max_tokens,
                              reasoning=reply.reasoning)
        if not verdict.ok and verdict.reason == "bad-args" and reply.repair_paths():
            verdict = judge_reply(reply.content, reply.tool_calls(), reply.finish, agentic=agentic,
                                  completion_tokens=reply.completion_tokens(), max_tokens=max_tokens,
                                  reasoning=reply.reasoning)
            if verdict.ok:
                STATS.bump("repaired_calls")
                self.log_message("repaired %s instead of failing the step", ", ".join(reply.repaired))
        return Attempt("pass" if verdict.ok else "fail", reply=reply, verdict=verdict)

    def _open_sse(self) -> None:
        """Send the client's event-stream headers, once. Deferred until the
        engine has answered 200, so a refusal the engine sends before its own
        stream (a 400, a 503 capacity refusal) reaches the client with its real
        status and body instead of as an error event inside a 200 stream: a
        client retries a 503 with its own backoff."""

        if getattr(self, "_sse_open", False):
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._sse_open = True

    def _write(self, raw: bytes) -> None:
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _relay_pass(self, result: Attempt, *, stream: bool, headers_sent: bool) -> None:
        reply = result.reply
        if reply is not None and reply.repaired:
            self._relay_rebuilt(result, stream=stream)
            return
        if stream:
            raw = b"".join(reply.events if reply else [])
            if not (reply and reply.done):
                raw += b"data: [DONE]\n\n"
            self._write(raw)
            return
        raw = result.body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._write(raw)

    def _relay_rebuilt(self, result: Attempt, *, stream: bool) -> None:
        """Relay a reply the proxy repaired: rebuilt from what was parsed, with the fixed arguments."""

        reply = result.reply
        assert reply is not None
        calls = reply.tool_calls()
        if not stream:
            try:
                body = json.loads(result.body)
                message = body["choices"][0]["message"]
                message["tool_calls"] = [{**old, "function": {**old.get("function", {}), "arguments": new["function"]["arguments"]}}
                                         for old, new in zip(message.get("tool_calls") or [], calls)]
                raw = json.dumps(body).encode()
            except (ValueError, KeyError, IndexError, TypeError):
                raw = result.body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self._write(raw)
            return
        cid, model = reply.completion_id or "chatcmpl-harness-repaired", reply.model or "local-model"
        out = [_chunk(cid, model, {"role": "assistant"})]
        if reply.reasoning:
            out.append(_chunk(cid, model, {"reasoning_content": reply.reasoning}))
        if reply.content:
            out.append(_chunk(cid, model, {"content": reply.content}))
        for index, call in enumerate(calls):
            out.append(_chunk(cid, model, {"tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                                             "function": {"name": call["function"]["name"], "arguments": ""}}]}))
            out.append(_chunk(cid, model, {"tool_calls": [{"index": index, "function": {"arguments": call["function"]["arguments"]}}]}))
        final = {"id": cid, "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": reply.finish or ("tool_calls" if calls else "stop")}]}
        if reply.usage:
            final["usage"] = reply.usage
        out.append(f"data: {json.dumps(final)}\n\n".encode())
        out.append(b"data: [DONE]\n\n")
        self._write(b"".join(out))

    def _send_synthetic(self, payload: dict[str, Any], text: str, *, headers_sent: bool) -> None:
        stream = bool(payload.get("stream"))
        model = str(payload.get("model") or "local-model")
        raw = _completion(text, model=model, stream=stream)
        if stream and headers_sent:
            self._write(raw)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._write(raw)

    def _send_error(self, result: Attempt, *, stream: bool, headers_sent: bool) -> None:
        self.log_message("upstream error: %s", result.detail)
        if stream and headers_sent:
            try:
                error = json.loads(result.body).get("error") or {"message": result.detail}
            except (json.JSONDecodeError, AttributeError):
                error = {"message": result.detail or f"upstream HTTP {result.status}", "type": "server_error"}
            self._write(f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n".encode())
            return
        raw = result.body or json.dumps({"error": {"message": result.detail, "type": "server_error"}}).encode()
        self.send_response(result.status or 502)
        sent_type = False
        for key, value in result.headers:
            if key.lower() == "content-type":
                sent_type = True
            self.send_header(key, value)
        if not sent_type:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._write(raw)

    def _save_failure(self, request: dict[str, Any], reply: Reply | None, verdict: Verdict, attempt: int, summary: dict[str, Any]) -> str:
        directory = self.cfg.failures_dir
        if not directory:
            return ""
        try:
            os.makedirs(directory, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(directory, f"{stamp}_{verdict.reason}_attempt{attempt}_{uuid.uuid4().hex[:6]}.txt")
            last_user = next((m.get("content") for m in reversed(request.get("messages") or [])
                              if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)), "")
            head = {"time": stamp, "reason": verdict.reason, "detail": verdict.detail, "attempt": attempt,
                    "model": request.get("model"), "tools": len(request.get("tools") or []),
                    "messages": len(request.get("messages") or []), "temperature": request.get("temperature"),
                    "seed": request.get("seed"), "max_tokens": request.get("max_tokens"),
                    "finish_reason": reply.finish if reply else None,
                    "completion_tokens": reply.completion_tokens() if reply else None,
                    "last_user_message": str(last_user)[:300]}
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(head, indent=1) + "\n\n--- content ---\n")
                handle.write(reply.content if reply else "")
                if reply and reply.reasoning:
                    handle.write("\n\n--- reasoning (hidden from the client) ---\n")
                    handle.write(reply.reasoning[-20000:])
                handle.write("\n\n--- tool_calls ---\n")
                handle.write(json.dumps(reply.tool_calls() if reply else [], indent=1)[:200000])
            self._prune_failures(directory)
            return path
        except OSError as exc:
            self.log_message("could not save failure: %s", exc)
            return ""

    def _prune_failures(self, directory: str) -> None:
        keep = self.cfg.failures_keep
        try:
            files = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))
        except OSError:
            return
        for name in files[:-keep] if len(files) > keep else []:
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---- self-test -------------------------------------------------------------------

def _self_test() -> None:
    from harness_judge import selftest as judge_selftest

    judge_selftest()
    _repeat_guard_self_test()
    _stream_self_test()
    _handle_guard_self_test()
    _hygiene_self_test()
    _judge_retry_self_test()
    print("loop_proxy self-test ok")


def _repeat_guard_self_test() -> None:
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
    moved_on = repeated(2) + [call("bash", json.dumps({"command": "ls"}), "m"), result("m", "src")]
    assert action(moved_on) is None, "a turn that moved on must not be touched"
    assert action(repeated(4) + [{"role": "user", "content": "try something else"}]) is None
    resumed = repeated(4) + [{"role": "user", "content": "try something else"}, call("bash", same, "c"), result("c", "total symbols: 268")]
    assert action(resumed) is None
    other = json.dumps({"command": "curl localhost"})

    def alternating(k: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(k):
            msgs += [call("bash", same if i % 2 == 0 else other, f"x{i}"), result(f"x{i}", f"out-{i}")]
        return msgs

    # write -> check -> write -> check ... is verification, never a cycle, however many rounds
    verify: list[dict[str, Any]] = [{"role": "user", "content": "build"}]
    for i in range(8):
        verify += [call("write", json.dumps({"filePath": f"f{i}.js", "content": f"v{i}"}), f"w{i}"), result(f"w{i}", "Wrote file successfully."),
                   call("bash", json.dumps({"command": "node --check game.js"}), f"k{i}"), result(f"k{i}", f"out {i}")]
    assert action(verify) is None, "a check re-run after each change is not a cycle"
    assert action(alternating(3)) is None
    assert action(alternating(5)) == "nudge"
    assert action(alternating(11)) == "stop"
    unique = [{"role": "user", "content": "go"}]
    for i in range(MAX_TOOL_ROUNDS - 1):
        unique += [call("bash", json.dumps({"command": f"step {i}"}), f"u{i}"), result(f"u{i}", f"ok {i}")]
    assert action(unique) is None
    unique += [call("bash", json.dumps({"command": "last"}), "uZ"), result("uZ", "ok")]
    assert action(unique) == "stop"
    changed = repeated(1) + [call("bash", same, "b"), result("b", "total symbols: 269")]
    assert action(changed) is None
    # Churn: the same file written again and again with different content.
    def churn(k: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "build"}]
        for i in range(k):
            msgs += [call("write", json.dumps({"filePath": "z/tetris.js", "content": f"v{i}"}), f"w{i}"), result(f"w{i}", "Wrote file successfully.")]
        return msgs

    assert action(churn(3)) is None, "three rewrites are normal iteration"
    kind, text, directive = decide(churn(4))
    assert kind == "churn" and text is None and directive and "z/tetris.js 4 times" in directive
    assert action(churn(5)) == "churn"
    kind, text, directive = decide(churn(WRITE_CHURN_FINISH))
    assert kind == "churn-finish" and "Tools are withheld" in directive
    assert action(churn(WRITE_CHURN_STOP - 1)) == "churn-finish"
    assert action(churn(WRITE_CHURN_STOP)) == "stop"
    assert "rewrote the same file 10 times" in decide(churn(WRITE_CHURN_STOP))[1]
    two_files = [{"role": "user", "content": "build"}]
    for i in range(6):
        two_files += [call("write", json.dumps({"filePath": f"f{i % 2}.js", "content": f"v{i}"}), f"t{i}"), result(f"t{i}", "ok")]
    assert action(two_files) is None, "three writes each to two files is not churn"
    moved = churn(4) + [call("bash", json.dumps({"command": "node --check z/tetris.js"}), "chk"), result("chk", "")]
    assert action(moved) is None, "only the latest round is judged"
    check = json.dumps({"command": "node --check game.js"})
    healthy = [
        {"role": "user", "content": "build"},
        call("write", json.dumps({"filePath": "game.js", "content": "v1"}), "w1"), result("w1", "Wrote file successfully."),
        call("bash", check, "c1"), result("c1", ""),
        call("edit", json.dumps({"filePath": "game.js", "oldString": "v1", "newString": "v2"}), "e1"), result("e1", "Edit applied successfully."),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert action(healthy) is None, "edit between identical checks must be left alone"
    looping = healthy[:5] + [
        call("read", json.dumps({"filePath": "game.js"}), "r1"), result("r1", "v1"),
        call("bash", check, "c2"), result("c2", ""),
    ]
    assert action(looping) == "nudge"
    # A "[Harness]" directive in the history does not start a new turn.
    with_directive = repeated(2) + [{"role": "user", "content": "[Harness] Stop re-running that command."}]
    assert len(_rounds_after_last_user(with_directive)) == 2, "the proxy's own directive is not a turn boundary"
    assert _rounds_after_last_user(repeated(2) + [{"role": "user", "content": "new request"}]) == []
    # A repeated edit that the client already applied again gets the truthful text, not "not run again".
    same_edit = json.dumps({"filePath": "z/t.js", "oldString": "a", "newString": "b"})
    twice = [{"role": "user", "content": "go"},
             call("edit", same_edit, "e1"), result("e1", "Edit applied successfully."),
             call("edit", same_edit, "e2"), result("e2", "Edit applied successfully.")]
    kind, text, _ = decide(twice)
    assert kind == "nudge" and text.startswith("[Harness] This exact edit has now been applied 2 times") and "z/t.js" in text
    msgs = repeated(2)
    _, refusal, directive = decide(msgs)
    assert refusal and refusal.startswith("[Harness] REFUSED") and "total symbols: 268" in refusal
    assert directive is None
    assert apply_nudge(msgs, refusal, directive)
    assert msgs[-1]["role"] == "tool" and msgs[-1]["content"] == refusal
    msgs = repeated(3)
    _, refusal, directive = decide(msgs)
    assert directive and directive.startswith("[Harness] Stop re-running")
    assert apply_nudge(msgs, refusal, directive)
    assert msgs[-2]["role"] == "tool" and msgs[-2]["content"] == refusal
    assert msgs[-1] == {"role": "user", "content": directive}
    _, stop_text, _ = decide(repeated(4))
    assert "python3 scan.py" in stop_text and '"continue" will repeat it' in stop_text


def _hygiene_self_test() -> None:
    cfg = Config()
    agentic = {"model": "m", "stream": True, "max_tokens": 16384, "tools": [{"type": "function", "function": {"name": "write"}}],
               "messages": [{"role": "user", "content": "go"}]}
    summary = prepare_request(agentic, cfg)
    assert summary["agentic"] and agentic["temperature"] == 0.45 and agentic["min_p"] == 0.05
    assert agentic["max_tokens"] == 6144, "the step cap applies to agentic requests"
    assert agentic["stream_options"] == {"include_usage": True}
    explicit = {"model": "m", "temperature": 0.35, "top_p": 0.9, "max_completion_tokens": 2000, "tools": [{}],
                "messages": []}
    prepare_request(explicit, cfg)
    assert explicit["temperature"] == 0.35 and explicit["top_p"] == 0.9, "client values are kept"
    assert explicit["max_tokens"] == 2000 and "max_completion_tokens" not in explicit
    greedy = {"model": "m", "temperature": 0, "messages": [], "max_tokens": 20000}
    prepare_request(greedy, cfg)
    assert greedy["temperature"] == 0 and greedy["max_tokens"] == 20000, "chat is not capped; zero is not 'absent'"
    retry = retry_payload(agentic, 2, Verdict(False, "loop", "x"), 3)
    assert retry["messages"][-1]["role"] == "user" and retry["messages"][-1]["content"].startswith("[Harness]")
    assert retry["temperature"] == 0.3 and retry["min_p"] == 0.05 and retry["seed"] and "repetition_penalty" not in retry
    assert agentic["messages"][-1]["role"] == "user" and len(agentic["messages"]) == 1, "the original is not mutated"
    final = retry_payload(agentic, 3, Verdict(False, "loop", "x"), 3)
    assert final["repetition_penalty"] == RETRY_PENALTY and final["temperature"] == 0.2
    assert "last attempt" in final["messages"][-1]["content"]
    final_narration = retry_payload(agentic, 3, Verdict(False, "narration", "x"), 3)
    assert "repetition_penalty" not in final_narration, "the penalty is only for loop-shaped failures"
    truncated = retry_payload(agentic, 2, Verdict(False, "truncated", "x"), 3)
    assert "6144-token limit" in truncated["messages"][-1]["content"]


class FakeEngine(BaseHTTPRequestHandler):
    """A scripted TensorFold: each request pops the next script entry.
    ("text", s) streams s in 40-char pieces; ("tool", name, args) emits a
    structured call; ("length", n) a truncated reply; ("hang", secs) sleeps;
    ("status", code) an HTTP error. Records every request body."""

    protocol_version = "HTTP/1.1"
    script: list[tuple[Any, ...]] = []
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def handle(self) -> None:
        # The proxy cancels a generation by closing the socket, as the real engine
        # expects; reading the next request off that socket then resets. Quietly end.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        raw = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or "0")) or b"{}")
        with self.lock:
            self.requests.append(body)
            entry = self.script.pop(0) if self.script else ("text", "fallback")
        kind = entry[0]
        if kind == "status":
            raw = json.dumps({"error": {"message": "scripted failure"}}).encode()
            self.send_response(entry[1])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if kind == "hang":
            time.sleep(entry[1])
            kind, entry = "text", ("text", "late")
        stream = bool(body.get("stream"))
        cid, model = "chatcmpl-fake", "fake"
        content, calls, finish = "", [], "stop"
        if kind == "text":
            content = entry[1]
        elif kind == "tool":
            calls = [{"id": "call_1", "type": "function", "function": {"name": entry[1], "arguments": entry[2]}}]
            finish = "tool_calls"
        elif kind == "length":
            content, finish = entry[1], "length"
        tokens = entry[3] if len(entry) > 3 else max(1, (len(content) + sum(len(c["function"]["arguments"]) for c in calls)) // 4)
        usage = {"prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": 10 + tokens}
        if not stream:
            message: dict[str, Any] = {"role": "assistant", "content": content or None}
            if calls:
                message["tool_calls"] = calls
            raw = json.dumps({"id": cid, "object": "chat.completion", "model": model, "usage": usage,
                              "choices": [{"index": 0, "message": message, "finish_reason": finish}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(chunk: bytes) -> None:
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()

        try:
            emit(_chunk(cid, model, {"role": "assistant"}))
            for i in range(0, len(content), 40):
                emit(_chunk(cid, model, {"content": content[i:i + 40]}))
                if len(entry) > 2 and kind == "text" and entry[2]:
                    time.sleep(entry[2])
            for index, call in enumerate(calls):
                emit(_chunk(cid, model, {"tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                                         "function": {"name": call["function"]["name"], "arguments": ""}}]}))
                emit(_chunk(cid, model, {"tool_calls": [{"index": index, "function": {"arguments": call["function"]["arguments"]}}]}))
            final = {"id": cid, "object": "chat.completion.chunk", "model": model, "usage": usage,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            emit(f"data: {json.dumps(final)}\n\n".encode())
            emit(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # the proxy cancelled us


def _with_fake_engine(script: list[tuple[Any, ...]], cfg: Config):
    """Start a FakeEngine with `script` and a Proxy in front; return (proxy_port, stop)."""

    FakeEngine.script = list(script)
    FakeEngine.requests = []
    engine = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    saved = (Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg)
    Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg = "127.0.0.1", engine.server_address[1], cfg
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    def stop() -> None:
        proxy.shutdown()
        engine.shutdown()
        Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg = saved

    return proxy.server_address[1], stop


def _client(port: int, body: dict[str, Any], *, path: str = "/v1/chat/completions", method: str = "POST",
            timeout: float = 30) -> tuple[int, bytes, float]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    started = time.time()
    conn.request(method, path, body=json.dumps(body) if body else None, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    first_at = None
    raw = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        if first_at is None:
            first_at = time.time() - started
        raw += chunk
    conn.close()
    return resp.status, raw, first_at or 0.0


def _sse_content(raw: bytes) -> tuple[str, list[dict[str, Any]], str | None]:
    reply = Reply()
    for event in raw.split(b"\n\n"):
        if event.strip() and not event.startswith(b":"):
            reply.feed_event(event + b"\n\n")
    return reply.content, reply.tool_calls(), reply.finish


def _judge_retry_self_test() -> None:
    import shutil
    import tempfile

    soup = "".join(random.Random(3).choice("$-(){}[]|:\\^_.,;!$-$xm0 ") for _ in range(2400))
    good_call = ("tool", "write", json.dumps({"filePath": "z/index.html", "content": "<html></html>"}))
    tools = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}}]
    agentic_body = {"model": "m", "stream": True, "max_tokens": 16384, "tools": tools,
                    "messages": [{"role": "user", "content": "create a tetris game in folder z"}]}
    tmp = tempfile.mkdtemp(prefix="harness-failures-")
    cfg = Config(failures_dir=tmp, keepalive=0.2, upstream_wait=3.0, stall_seconds=5.0)
    try:
        # 1. soup mid-stream → cancelled, retried with a directive + fresh sampling; client sees only the good call
        port, stop = _with_fake_engine([("text", "Sure, the game is ready. " * 8 + soup, 0.0), good_call], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            content, calls, finish = _sse_content(raw)
            assert status == 200 and finish == "tool_calls" and [c["function"]["name"] for c in calls] == ["write"], (status, finish, calls)
            assert "$" not in content and "Sure" not in content, f"failed text leaked to the client: {content[:80]!r}"
            assert len(FakeEngine.requests) == 2, len(FakeEngine.requests)
            first, second = FakeEngine.requests
            assert first["temperature"] == 0.45 and first["max_tokens"] == 6144 and first["stream_options"] == {"include_usage": True}
            assert second["messages"][-1]["role"] == "user" and "[Harness]" in second["messages"][-1]["content"]
            assert "symbol soup" in second["messages"][-1]["content"]
            assert second["temperature"] == 0.3 and second["seed"] and second["min_p"] == 0.05
            assert len(os.listdir(tmp)) == 1 and "_gibberish_attempt1_" in os.listdir(tmp)[0]
            saved = open(os.path.join(tmp, os.listdir(tmp)[0])).read()
            assert '"reason": "gibberish"' in saved and "Sure, the game is ready." in saved
        finally:
            stop()
        # 2. truncated tool call (finish=length) → retried
        port, stop = _with_fake_engine([("length", "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>", 1, 6144), good_call], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            content, calls, finish = _sse_content(raw)
            assert finish == "tool_calls" and len(calls) == 1 and "<|tool_call>" not in content
            assert "6144-token limit" in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        # 3. three failures → a [Harness] Stopped message, the turn ends cleanly, files saved
        loop = "</td></tr></tbody></table>" * 60
        port, stop = _with_fake_engine([("text", loop), ("text", loop), ("text", loop), good_call], cfg)
        try:
            before = len(os.listdir(tmp))
            status, raw, _ = _client(port, agentic_body)
            content, calls, finish = _sse_content(raw)
            assert finish == "stop" and not calls and content.startswith("[Harness] Stopped: the model failed this step 3 times (loop)"), content[:120]
            assert "</td>" not in content
            assert len(FakeEngine.requests) == 3, "no fourth attempt"
            assert FakeEngine.requests[2]["repetition_penalty"] == RETRY_PENALTY, "last attempt against a loop adds the penalty"
            assert len(os.listdir(tmp)) == before + 3
        finally:
            stop()
        # 4. a healthy step passes straight through, with buffering: exactly one upstream request, same bytes
        port, stop = _with_fake_engine([good_call], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            content, calls, finish = _sse_content(raw)
            assert finish == "tool_calls" and calls[0]["function"]["arguments"].startswith('{"filePath"')
            assert len(FakeEngine.requests) == 1 and raw.endswith(b"data: [DONE]\n\n")
        finally:
            stop()
        # 5. a healthy final report passes too (prose, no call), narration-shaped talk with a call passes
        port, stop = _with_fake_engine([("text", "Done: wrote z/index.html and z/tetris.js; open z/index.html to play.")], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            content, calls, finish = _sse_content(raw)
            assert finish == "stop" and content.startswith("Done:") and len(FakeEngine.requests) == 1
        finally:
            stop()
        # 6. plain chat streams live and a loop is cut with a note, never retried
        chat_body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        port, stop = _with_fake_engine([("text", "Here is a table:\n" + loop, 0.01), ("text", "never")], cfg)
        try:
            status, raw, first_at = _client(port, chat_body)
            content, calls, finish = _sse_content(raw)
            assert content.startswith("Here is a table:") and "[Harness] Output cut" in content and finish == "stop"
            assert content.count("</td></tr></tbody></table>") < 60, "the loop was cut short"
            assert len(FakeEngine.requests) == 1, "chat is never retried"
            assert raw.endswith(b"data: [DONE]\n\n")
        finally:
            stop()
        # 6b. a repeat nudge re-seeds the forwarded request; the directive tier also samples wider
        same = json.dumps({"command": "node --check game.js"})
        call = lambda cid: {"role": "assistant", "tool_calls": [{"id": cid, "type": "function", "function": {"name": "bash", "arguments": same}}]}  # noqa: E731
        result = lambda cid: {"role": "tool", "tool_call_id": cid, "content": "same output"}  # noqa: E731
        repeated = {**agentic_body, "temperature": 0.35, "messages": [{"role": "user", "content": "go"}, call("a"), result("a"), call("b"), result("b")]}
        port, stop = _with_fake_engine([good_call, good_call], cfg)
        try:
            _client(port, repeated)
            sent = FakeEngine.requests[0]
            assert sent["seed"] and sent["temperature"] == 0.35 and sent["messages"][-1]["content"].startswith("[Harness] REFUSED")
            _client(port, {**repeated, "messages": repeated["messages"] + [call("c"), result("c")]})
            sent = FakeEngine.requests[1]
            assert sent["seed"] and sent["temperature"] == NUDGE_TEMPERATURE and sent["messages"][-1]["role"] == "user"
            assert sent["messages"][-1]["content"].startswith("[Harness] Stop re-running")
        finally:
            stop()
        # 6c. churn-finish: the forwarded request has no tools but is still judged as a step (capped, buffered)
        writes = [{"role": "user", "content": "build"}]
        for i in range(WRITE_CHURN_FINISH):
            writes += [{"role": "assistant", "tool_calls": [{"id": f"w{i}", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": f"v{i}"})}}]},
                       {"role": "tool", "tool_call_id": f"w{i}", "content": "Wrote file successfully."}]
        port, stop = _with_fake_engine([("text", soup), ("text", "Built z/index.html and z/t.js; open the html file.")], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": writes})
            content, calls, finish = _sse_content(raw)
            assert "tools" not in FakeEngine.requests[0] and FakeEngine.requests[0]["max_tokens"] == 6144
            assert FakeEngine.requests[0]["messages"][-1]["content"].startswith("[Harness] You have now written z/t.js 6 times")
            assert content.startswith("Built z/") and "$" not in content, "soup was judged and retried even without tools"
            assert len(FakeEngine.requests) == 2
        finally:
            stop()
        # 6d. a stub rewrite of a file written earlier in the turn is failed and retried
        full = "// full game\n" + "\n".join(f"function f{i}() {{ return {i}; }}" for i in range(120))
        history = [{"role": "user", "content": "build"},
                   {"role": "assistant", "tool_calls": [{"id": "w0", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": full})}}]},
                   {"role": "tool", "tool_call_id": "w0", "content": "Wrote file successfully."}]
        stub = ("tool", "write", json.dumps({"filePath": "z/t.js", "content": "// TODO: game goes here\n"}))
        port, stop = _with_fake_engine([stub, good_call], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": history})
            content, calls, finish = _sse_content(raw)
            assert finish == "tool_calls" and calls[0]["function"]["arguments"].startswith('{"filePath": "z/index.html"')
            assert len(FakeEngine.requests) == 2 and "must contain the complete file" in FakeEngine.requests[1]["messages"][-1]["content"]
            assert shrinking_rewrite(history, [{"function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": full[:2000]})}}]).ok, "a modest shrink is fine"
            assert shrinking_rewrite(history, [{"function": {"name": "write", "arguments": json.dumps({"filePath": "z/other.js", "content": "x"})}}]).ok, "a new file is fine"
        finally:
            stop()
        # 6e. an edit already applied this turn never reaches the client again; a missing required key is retried
        edit_args = json.dumps({"filePath": "z/t.js", "oldString": "a();", "newString": "a();\nb();"})
        edit_tools = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "required": ["filePath", "oldString", "newString"]}}},
                      {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "required": ["filePath", "content"]}}}]
        applied = [{"role": "user", "content": "fix it"},
                   {"role": "assistant", "tool_calls": [{"id": "e1", "type": "function", "function": {"name": "edit", "arguments": edit_args}}]},
                   {"role": "tool", "tool_call_id": "e1", "content": "Edit applied successfully."}]
        port, stop = _with_fake_engine([("tool", "edit", edit_args), good_call], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "tools": edit_tools, "messages": applied})
            content, calls, finish = _sse_content(raw)
            assert [c["function"]["name"] for c in calls] == ["write"], "the repeated edit must not be relayed"
            assert len(FakeEngine.requests) == 2 and "applying the same edit twice" in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        failed_before = applied[:2] + [{"role": "tool", "tool_call_id": "e1", "content": "Error: oldString not found in content"}]
        assert repeated_edit(failed_before, [{"id": "x", "function": {"name": "edit", "arguments": edit_args}}]).ok, "a failed edit may be retried"
        assert not repeated_edit(applied, [{"id": "x", "function": {"name": "edit", "arguments": edit_args}}]).ok
        other = json.dumps({"filePath": "z/t.js", "oldString": "b();", "newString": "c();"})
        assert repeated_edit(applied, [{"id": "x", "function": {"name": "edit", "arguments": other}}]).ok, "a different edit is fine"
        no_old = json.dumps({"filePath": "z/t.js", "newString": "c();"})
        v = missing_required(edit_tools, [{"function": {"name": "edit", "arguments": no_old}}])
        assert v.reason == "missing-args" and "oldString" in v.detail, v
        assert missing_required(edit_tools, [{"function": {"name": "edit", "arguments": edit_args}}]).ok
        assert missing_required([], [{"function": {"name": "edit", "arguments": no_old}}]).ok, "no schema, no check"
        # 6f. a final report that claims an unrun check is retried once; the second claim passes through
        wrote = [{"role": "user", "content": "build"},
                 {"role": "assistant", "tool_calls": [{"id": "w", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": "x"})}}]},
                 {"role": "tool", "tool_call_id": "w", "content": "Wrote file successfully."}]
        claim = "Done. Final verification: `node --check z/t.js` (No syntax errors found)."
        port, stop = _with_fake_engine([("text", claim), ("tool", "bash", json.dumps({"command": "node --check z/t.js"}))], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": wrote})
            content, calls, finish = _sse_content(raw)
            assert [c["function"]["name"] for c in calls] == ["bash"] and len(FakeEngine.requests) == 2
            retry_text = FakeEngine.requests[1]["messages"][-1]["content"]
            assert "`node --check z/t.js`" in retry_text, retry_text        # unchecked-final or unverified-claim: both ask for the check
        finally:
            stop()
        port, stop = _with_fake_engine([("text", claim), ("text", claim)], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": wrote})
            content, calls, finish = _sse_content(raw)
            assert content == claim and finish == "stop" and len(FakeEngine.requests) == 2, "never stops the turn"
        finally:
            stop()
        checked = wrote + [{"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "node --check z/t.js"})}}]},
                           {"role": "tool", "tool_call_id": "c", "content": ""}]
        assert unverified_claim(checked, claim, []).ok, "a check that ran may be cited"
        assert unverified_claim(wrote, "Open `z/index.html` in a browser.", []).ok, "no check command, no claim"
        assert unverified_claim(wrote, "Wrote it. Run `node --check z/t.js` yourself.", []).ok, "no claim wording"
        assert unverified_claim(wrote, claim, [{"function": {"name": "bash"}}]).ok, "only final replies are checked"
        # 6g. behavioural failures retry wider, and a no-op / repeated edit retries without the edit tool
        edit_tool = {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}}
        base = {**agentic_body, "tools": [edit_tool, *agentic_body["tools"]]}
        r = retry_payload(base, 2, Verdict(False, "noop-edit", "x"), 3)
        assert r["temperature"] == BEHAVIOR_TEMPERATURES[0] and [t["function"]["name"] for t in r["tools"]] == ["write"]
        assert "not available for this retry" in r["messages"][-1]["content"]
        assert retry_payload(base, 3, Verdict(False, "repeat-edit", "x"), 3)["temperature"] == BEHAVIOR_TEMPERATURES[1]
        r = retry_payload(base, 2, Verdict(False, "loop", "x"), 3)
        assert r["temperature"] == RETRY_TEMPERATURES[0] and len(r["tools"]) == 2, "degeneration keeps every tool, samples tighter"
        only_edit = {**agentic_body, "tools": [edit_tool]}
        assert len(retry_payload(only_edit, 2, Verdict(False, "noop-edit", "x"), 3)["tools"]) == 1, "never strips the last tool"
        # 6h. a second change to a code file with no look at it since is retried once for the check
        w1 = {"id": "w1", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": "a"})}}
        e1 = {"id": "e1", "type": "function", "function": {"name": "edit", "arguments": json.dumps({"filePath": "z/t.js", "oldString": "a", "newString": "b"})}}
        h = [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [w1]}, {"role": "tool", "tool_call_id": "w1", "content": "Wrote file successfully."}]
        v = unchecked_change(h, [e1])
        assert v.reason == "unchecked-change" and "node --check z/t.js" in v.detail, v
        chk = {"id": "c", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "node --check z/t.js"})}}
        assert unchecked_change(h + [{"role": "assistant", "tool_calls": [chk]}, {"role": "tool", "tool_call_id": "c", "content": ""}], [e1]).ok
        rd = {"id": "r", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "z/t.js"})}}
        assert unchecked_change(h + [{"role": "assistant", "tool_calls": [rd]}, {"role": "tool", "tool_call_id": "r", "content": "a"}], [e1]).ok, "a read counts"
        assert unchecked_change(h, [{"id": "x", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/other.js", "content": "b"})}}]).ok
        assert unchecked_change(h, [{"id": "x", "function": {"name": "write", "arguments": json.dumps({"filePath": "notes.md", "content": "b"})}}]).ok
        md = [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [{"id": "m", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "README.md", "content": "a"})}}]},
              {"role": "tool", "tool_call_id": "m", "content": "Wrote file successfully."}]
        assert unchecked_change(md, [{"id": "x", "function": {"name": "write", "arguments": json.dumps({"filePath": "README.md", "content": "b"})}}]).ok, "no check for markdown"
        port, stop = _with_fake_engine([("tool", "edit", e1["function"]["arguments"]), ("tool", "edit", e1["function"]["arguments"])], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "tools": [edit_tool, *agentic_body["tools"]], "messages": h})
            content, calls, finish = _sse_content(raw)
            assert [c["function"]["name"] for c in calls] == ["edit"] and len(FakeEngine.requests) == 2, "second attempt passes: soft check"
            assert "node --check z/t.js" in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        # 6i. after a code file changes, the next request is steered to its syntax check before generation
        assert pending_check(h) == ("z/t.js", "node --check z/t.js")
        assert pending_check(md) is None, "no check for markdown"
        failed_write = h[:2] + [{"role": "tool", "tool_call_id": "w1", "content": "Error: permission denied"}]
        assert pending_check(failed_write) is None
        after_check = h + [{"role": "assistant", "tool_calls": [chk]}, {"role": "tool", "tool_call_id": "c", "content": ""}]
        assert pending_check(after_check) is None
        port, stop = _with_fake_engine([("tool", "bash", json.dumps({"command": "node --check z/t.js"}))], cfg)
        try:
            _client(port, {**agentic_body, "messages": h})
            sent = FakeEngine.requests[0]["messages"]
            assert sent[-1]["role"] == "user" and "Run `node --check z/t.js`" in sent[-1]["content"]
            assert len(h) == 3, "the client's own history is not touched"
        finally:
            stop()
        # 6j. a complete write whose path carries stray backticks is repaired and relayed, not retried
        ticked = ("tool", "write", json.dumps({"filePath": "z/tetris.js``", "content": "const x = 1;"}))
        for stream_mode in (True, False):
            port, stop = _with_fake_engine([ticked], cfg)
            try:
                status, raw, _ = _client(port, {**agentic_body, "stream": stream_mode})
                if stream_mode:
                    _, calls, finish = _sse_content(raw)
                    args = json.loads(calls[0]["function"]["arguments"])
                    assert finish == "tool_calls" and raw.endswith(b"data: [DONE]\n\n")
                else:
                    args = json.loads(json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
                assert args == {"filePath": "z/tetris.js", "content": "const x = 1;"} and len(FakeEngine.requests) == 1
            finally:
                stop()
        inside = ("tool", "write", json.dumps({"filePath": "z/te`tris.js", "content": "x"}))
        port, stop = _with_fake_engine([inside, good_call], cfg)
        try:
            _client(port, agentic_body)
            assert len(FakeEngine.requests) == 2, "junk inside a path is still a retry"
        finally:
            stop()
        # 6k. a 35-60% rewrite is retried once; a final report over a regressed file is retried once
        big = "// full\n" + "\n".join(f"function g{i}() {{ return {i}; }}" for i in range(200))
        hist = [{"role": "user", "content": "build"},
                {"role": "assistant", "tool_calls": [{"id": "w0", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": big})}}]},
                {"role": "tool", "tool_call_id": "w0", "content": "Wrote file successfully."}]
        half = json.dumps({"filePath": "z/t.js", "content": big[: len(big) // 2]})
        assert partial_rewrite(hist, [{"function": {"name": "write", "arguments": half}}]).reason == "partial-rewrite"
        assert partial_rewrite(hist, [{"function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": big[:int(len(big) * .8)]})}}]).ok
        assert shrinking_rewrite(hist, [{"function": {"name": "write", "arguments": half}}]).ok, "50% is soft, not the hard floor"
        regressed = hist + [{"role": "assistant", "tool_calls": [{"id": "w1", "type": "function", "function": {"name": "write", "arguments": half}}]},
                            {"role": "tool", "tool_call_id": "w1", "content": "Wrote file successfully."}]
        v = regressed_file(regressed, "Done, the game is complete.", [])
        assert v.reason == "regressed-file" and "down from" in v.detail, v
        assert regressed_file(hist, "Done.", []).ok and regressed_file(regressed, "", [{"function": {"name": "bash"}}]).ok
        port, stop = _with_fake_engine([("text", "Done, the game is complete."), ("text", "Done, the game is complete.")], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": regressed})
            content, _, finish = _sse_content(raw)
            assert content == "Done, the game is complete." and len(FakeEngine.requests) == 2, "soft: one retry, then it passes"
            assert "down from" in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        # 6l. a final report over an unchecked change, or one carrying the file's code, is retried once
        wr = {"id": "w", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": "x"})}}
        wrote_js = [{"role": "user", "content": "build"}, {"role": "assistant", "tool_calls": [wr]},
                    {"role": "tool", "tool_call_id": "w", "content": "Wrote file successfully."}]
        assert unchecked_final(wrote_js, "Done.", []).reason == "unchecked-final"
        ck = {"id": "k", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "node --check z/t.js"})}}
        checked_js = wrote_js + [{"role": "assistant", "tool_calls": [ck]}, {"role": "tool", "tool_call_id": "k", "content": ""}]
        assert unchecked_final(checked_js, "Done.", []).ok and unchecked_final(wrote_js, "", [ck]).ok
        two = wrote_js + [{"role": "assistant", "tool_calls": [{"id": "w2", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/u.py", "content": "x"})}}]},
                          {"role": "tool", "tool_call_id": "w2", "content": "Wrote file successfully."}]
        v = unchecked_final(two, "Done.", [])
        assert "`node --check z/t.js && python3 -m py_compile z/u.py`" in v.detail, v.detail
        big_report = "Done. The final version:\n```javascript\n" + "const a = 1;\n" * 200 + "```\nOpen the page."
        assert code_in_report(checked_js, big_report, []).reason == "code-in-report"
        assert code_in_report(checked_js, "Done.\n```bash\nnode --check z/t.js\n```", []).ok, "a short snippet is fine"
        assert code_in_report([{"role": "user", "content": "show me"}], big_report, []).ok, "no file written: showing code is the answer"
        port, stop = _with_fake_engine([("text", "Done, the game works."), ("tool", "bash", json.dumps({"command": "node --check z/t.js"}))], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "messages": wrote_js})
            _, calls, _ = _sse_content(raw)
            assert [c["function"]["name"] for c in calls] == ["bash"] and "before reporting" in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        # 6m. churn-finish over an unchecked file keeps bash/read/edit and withholds only write
        churned = [{"role": "user", "content": "build"}]
        for i in range(WRITE_CHURN_FINISH):
            churned += [{"role": "assistant", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "write", "arguments": json.dumps({"filePath": "z/t.js", "content": f"v{i}"})}}]},
                        {"role": "tool", "tool_call_id": f"c{i}", "content": "Wrote file successfully."}]
        three_tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in ("write", "edit", "bash")]
        port, stop = _with_fake_engine([("tool", "bash", json.dumps({"command": "node --check z/t.js"}))], cfg)
        try:
            _client(port, {**agentic_body, "tools": three_tools, "messages": churned})
            sent = FakeEngine.requests[0]
            assert [t["function"]["name"] for t in sent["tools"]] == ["edit", "bash"], sent.get("tools")
            assert "write tool is withheld" in sent["messages"][-1]["content"] and "`node --check z/t.js`" in sent["messages"][-1]["content"]
        finally:
            stop()
        checked_churn = churned + [{"role": "assistant", "tool_calls": [{"id": "k", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "node --check z/t.js"})}}]},
                                   {"role": "tool", "tool_call_id": "k", "content": ""}]
        # (the latest round is now the check, so decide() sees no churn action for this request; build the
        # finish case directly: a churned history whose last write was followed by a check)
        assert not [p for p, at in _change_state(checked_churn)[0].items() if _change_state(checked_churn)[1].get(p, -1) < at]
        # 6n. a versioned copy of a file already written this turn is retried once
        for copy in ("z/t_v2.js", "z/t_final.js", "z/t-new.js", "z/t.2.js"):
            v = sibling_copy(wrote_js, [{"function": {"name": "write", "arguments": json.dumps({"filePath": copy, "content": "x"})}}])
            assert v.reason == "sibling-copy" and "z/t.js" in v.detail, (copy, v)
        for other in ("z/other.js", "z/t.js", "z/t_v2.py", "z/utils.js"):
            assert sibling_copy(wrote_js, [{"function": {"name": "write", "arguments": json.dumps({"filePath": other, "content": "x"})}}]).ok, other
        assert sibling_copy([{"role": "user", "content": "go"}], [{"function": {"name": "write", "arguments": json.dumps({"filePath": "z/t_v2.js", "content": "x"})}}]).ok, "no original in this turn"
        # 7. chat without temperature gets the defaults; chat max_tokens is not capped
        port, stop = _with_fake_engine([("text", "pong")], cfg)
        try:
            _client(port, {"model": "m", "stream": False, "max_tokens": 20000, "messages": [{"role": "user", "content": "ping"}]})
            sent = FakeEngine.requests[0]
            assert sent["temperature"] == 0.45 and sent["top_k"] == 40 and sent["max_tokens"] == 20000
        finally:
            stop()
        # 8. non-stream agentic: soup → retried → JSON pass
        port, stop = _with_fake_engine([("text", soup), good_call], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "stream": False})
            body = json.loads(raw)
            assert status == 200 and body["choices"][0]["finish_reason"] == "tool_calls"
            assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "write"
            assert len(FakeEngine.requests) == 2
        finally:
            stop()
        # 9. transient upstream 503 → retried without a directive; a 400 passes through
        port, stop = _with_fake_engine([("status", 503), good_call], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            _, calls, finish = _sse_content(raw)
            assert finish == "tool_calls" and len(FakeEngine.requests) == 2
            assert "[Harness]" not in FakeEngine.requests[1]["messages"][-1]["content"]
        finally:
            stop()
        port, stop = _with_fake_engine([("status", 400)], cfg)
        try:
            status, raw, _ = _client(port, {**agentic_body, "stream": False})
            assert status == 400 and b"scripted failure" in raw
        finally:
            stop()
        # 9b. a streaming request keeps the engine's real status for a refusal sent before its stream
        port, stop = _with_fake_engine([("status", 400)], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            assert status == 400 and b"scripted failure" in raw and b"data:" not in raw, (status, raw[:120])
        finally:
            stop()
        port, stop = _with_fake_engine([("status", 503), ("status", 503), ("status", 503)], cfg)
        try:
            status, raw, _ = _client(port, agentic_body)
            assert status == 503 and len(FakeEngine.requests) == 3, (status, len(FakeEngine.requests))
        finally:
            stop()
        # 10. a stalled engine → the attempt errors after stall_seconds and is retried
        port, stop = _with_fake_engine([("hang", 2.5), good_call], Config(failures_dir=tmp, keepalive=0.2, upstream_wait=3.0, stall_seconds=1.0))
        try:
            status, raw, _ = _client(port, agentic_body, timeout=30)
            _, calls, finish = _sse_content(raw)
            assert finish == "tool_calls" and len(FakeEngine.requests) == 2, (finish, len(FakeEngine.requests))
            assert STATS.counts["stalls"] >= 1
        finally:
            stop()
        # 11. engine down → 503 JSON after upstream_wait, not a hang
        saved = (Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg)
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        dead_port = dead.getsockname()[1]
        dead.close()
        Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg = "127.0.0.1", dead_port, Config(upstream_wait=1.0)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            started = time.time()
            status, raw, _ = _client(proxy.server_address[1], {**agentic_body, "stream": False})
            assert status == 503 and b"unavailable" in raw and time.time() - started < 10
            status, raw, _ = _client(proxy.server_address[1], {}, path="/harness/health", method="GET")
            assert status == 503 and json.loads(raw)["upstream_ok"] is False
        finally:
            proxy.shutdown()
            Proxy.upstream_host, Proxy.upstream_port, Proxy.cfg = saved
        # 12. health endpoint with a live engine
        port, stop = _with_fake_engine([], cfg)
        try:
            status, raw, _ = _client(port, {}, path="/harness/health", method="GET")
            health = json.loads(raw)
            assert status == 200 and health["ok"] and health["models"] == ["fake"]
            assert health["counts"]["retries"] >= 3 and health["failures"]["gibberish"] >= 2 and health["counts"]["stops"] >= 1
            assert health["config"]["step_max_tokens"] == 6144 and health["last_failure"]["reason"]
        finally:
            stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stream_self_test() -> None:
    """A plain chat stream is relayed as it arrives, not at the end."""

    port, stop = _with_fake_engine([("text", "first chunk here, then a pause; " * 2, 0.8)], Config(failures_dir=""))
    try:
        started = time.time()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        first, first_at = b"", None
        rest = b""
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            if first_at is None:
                first += chunk
                if b"first chunk" in first:
                    first_at = time.time() - started
            else:
                rest += chunk
        assert first_at is not None, f"first text never arrived: {first!r}"
        assert first_at < 0.6, f"stream was buffered: first chunk arrived after {first_at:.2f}s"
        assert b"data: [DONE]" in rest, "the pause-delayed second half and [DONE] must follow"
    finally:
        stop()


def _handle_guard_self_test() -> None:
    import http.server

    base = http.server.BaseHTTPRequestHandler
    original = base.handle
    try:
        for exc in (BrokenPipeError(32, "Broken pipe"), ConnectionResetError()):
            def raising(self, _exc=exc):
                raise _exc

            base.handle = raising
            handler = Proxy.__new__(Proxy)
            handler.close_connection = False
            handler.handle()
            assert handler.close_connection is True
    finally:
        base.handle = original


# ---- main --------------------------------------------------------------------------

def _parse_defaults(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, _, value = item.partition("=")
        out[key.strip()] = int(value) if key.strip() == "top_k" else float(value)
    return out


def main() -> None:
    global MAX_TOOL_ROUNDS
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--listen", default="8094")
    parser.add_argument("--upstream", default="127.0.0.1:8104")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help=f"tool calls allowed per user turn (default {MAX_TOOL_ROUNDS}; env LOOP_MAX_ROUNDS)")
    parser.add_argument("--defaults", default=env("HARNESS_DEFAULTS", "temperature=0.45,top_p=0.9,top_k=40,min_p=0.05"),
                        help="sampling for fields the client omits (env HARNESS_DEFAULTS; empty = none)")
    parser.add_argument("--step-max-tokens", type=int, default=int(env("HARNESS_STEP_MAX_TOKENS", "6144")),
                        help="token cap per agentic step, 0 = client's value (env HARNESS_STEP_MAX_TOKENS)")
    parser.add_argument("--stall-seconds", type=float, default=float(env("HARNESS_STALL_SECONDS", "240")),
                        help="no bytes from the engine for this long fails the attempt (env HARNESS_STALL_SECONDS)")
    parser.add_argument("--max-attempts", type=int, default=int(env("HARNESS_MAX_ATTEMPTS", "3")),
                        help="attempts per agentic step before the turn is stopped (env HARNESS_MAX_ATTEMPTS)")
    parser.add_argument("--failures-dir", default=env("HARNESS_FAILURES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".harness_failures")),
                        help="where discarded replies are saved (env HARNESS_FAILURES_DIR; empty = don't)")
    parser.add_argument("--stream-prose", action="store_true", default=env("HARNESS_STREAM_PROSE", "") == "1",
                        help="stream agentic prose live (a failure is then cut with a note, not retried)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.max_rounds is not None:
        MAX_TOOL_ROUNDS = args.max_rounds
    if args.self_test:
        _self_test()
        return
    Proxy.cfg = Config(defaults=_parse_defaults(args.defaults), step_max_tokens=max(0, args.step_max_tokens),
                       stall_seconds=args.stall_seconds, max_attempts=max(1, args.max_attempts),
                       failures_dir=args.failures_dir, stream_prose=args.stream_prose)
    host, _, port = args.listen.rpartition(":")
    if not port:
        host, port = "127.0.0.1", args.listen
    upstream_host, _, upstream_port = args.upstream.rpartition(":")
    Proxy.upstream_host = upstream_host or "127.0.0.1"
    Proxy.upstream_port = int(upstream_port)
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Proxy)
    server.daemon_threads = True
    cfg = Proxy.cfg
    print(f"[harness] {host or '127.0.0.1'}:{port} -> {Proxy.upstream_host}:{Proxy.upstream_port} "
          f"defaults={cfg.defaults} step_max_tokens={cfg.step_max_tokens} stall={cfg.stall_seconds:.0f}s "
          f"attempts={cfg.max_attempts} failures={cfg.failures_dir or '-'} rounds={MAX_TOOL_ROUNDS}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
