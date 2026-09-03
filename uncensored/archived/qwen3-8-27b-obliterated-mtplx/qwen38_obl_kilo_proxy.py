#!/usr/bin/env python3
"""Kilo steering proxy for Qwen3.8-27B OBLITERATED + mtplx.

Kilo's *global* agent config is Muse Glimmer (temperature 1.0, long prompt).
That overrides mtplx ``--default-temperature 0`` and the OBLITERATUS card
(greedy + repetition_penalty + thinking off). Agents then look lazy or loop.

This proxy sits in front of mtplx and:

  1. Forces card sampling: temperature=0, top_p=1.0, frequency_penalty=0.3
     (mtplx stand-in for HF repetition_penalty=1.15). Thinking is ON for
     tool turns (reasoning_effort medium by default, bounded by the engine's
     MTPLX_THINKING_BUDGET guard) and OFF for compaction / title / plain chat.
     See THINKING / REASONING_EFFORT / THINKING_BUDGET below.
  2. Floors max_tokens at 2048 (+ the thinking budget) on agentic turns
  3. Prepends a short finish-the-job loop nudge on tool turns
  4. Strips tools on Kilo compaction / summary
  5. Soft-repairs truncated tool-call argument JSON
  6. Capability middleware (scoped — not always-on):
     empty-tool recovery, fake-action recovery, prose-loop recovery,
     just-in-time continue after a tool result, cap huge tool outputs
  7. Auto-fix tool calls (same as Ornith): bash-sanitize *broken* commands
     only (unmatched quotes, invented paths, command substitution, missing
     cat/head files). Do not rewrite cd/&&, env=value ./script, which, pipes,
     or 2>/dev/null — those were causing the agent to stop after dummy ls/head
   8. Response-side early-stop: after a tool result, if the model dumps a
      Next-steps/plan with no tool_calls, retry once (mtplx: tool_choice
      auto|none only). Synthesize a plan script at most once; never repeat
      a command already in history. After-tool JIT nudges cap at 3 tool
      rounds since the last user message, then the nudge is stripped so
      the model can recap and stop. The forced retry covers every nudged
      round (EARLY_STOP_FORCE_MAX == AFTER_TOOL_NUDGE_MAX).
   9. Serialize chat generations and drop client session-affinity headers.
      mtplx raises 409 "session … already in flight" when Kilo retries the
      same sticky session while a stream is still running; retry that 409.
  10. Language enforcement (2026-09-02): the OBLITERATED fine-tune drifts
      into Chinese on English prompts (measured: full Chinese session
      recaps). An English-only directive rides every chat request, and a
      reply that comes back CJK is regenerated once with a hard nudge —
      early-aborted on the SSE path (nothing forwarded yet), plain re-fetch
      on the buffered / non-stream paths. Kill switch: QWEN38_OBL_ENGLISH=0.
  11. First-turn guard (2026-09-02): the very first reply after a user
      prompt on an agent (tools) turn is buffered and run through the same
      early-stop check as tool rounds. Before this, an "I'll start by… /
      Next steps:" dump with no tool_calls ended Kilo's turn and the task
      sat idle until a human typed "continue". Plain answers pass through
      unchanged (one upstream call). Harness flag: mw_first_turn.


Usage:
  python3 qwen38_obl_kilo_proxy.py --upstream http://127.0.0.1:8767 --port 8768
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import queue
import re
import select
import shlex
import socket
import sys
import threading
import time
from collections import Counter
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autosaddler import (  # noqa: E402
    AS_LOOP_END,
    AS_LOOP_START,
    append_live_event,
    load_active,
)

log = logging.getLogger("qwen38_obl_kilo_proxy")
LOG_FILE = (
    Path(__file__).resolve().parent / ".qwen38_obl_proxy.log"
)

CARD_TEMPERATURE = 0.0
CARD_TOP_P = 1.0
CARD_FREQUENCY_PENALTY = 0.3


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


# Thinking on agent turns (2026-09-02). Live Titan M2 sessions: with thinking
# forced OFF every step was a greedy no-deliberation reply -- the model read a
# file, re-read the same file until the repeat guard took its tools away, or
# handed back a recap after a few rounds, and a human had to type "continue".
# Capability before steering (AutoSaddler): give the model room to reason
# through the whole task before it acts. Qwen3.8's template accepts exactly
# low / medium / xhigh; the reasoning segment is bounded by mtplx's thinking
# guard (MTPLX_THINKING_BUDGET, exported by 2_start_mtplx.sh), so a self-doubt loop
# cannot hold the engine. Scope: requests that carry tools. Compaction, title
# generation, plain chat and the repeat-guard "answer in text" turn stay
# thinking-off (no budget guard applies there; a tiny max_tokens would be
# eaten by <think>). Kill switch: QWEN38_OBL_THINKING=0.
THINKING = (
    os.environ.get("QWEN38_OBL_THINKING", "1").strip().lower()
    not in ("0", "false", "off")
)
_REASONING_EFFORT_CHOICES = ("low", "medium", "xhigh")


def _env_effort(default: str = "medium") -> str:
    raw = os.environ.get("QWEN38_OBL_REASONING_EFFORT", "").strip().lower()
    return raw if raw in _REASONING_EFFORT_CHOICES else default


REASONING_EFFORT = _env_effort()
# Must match the engine's MTPLX_THINKING_BUDGET (the start script exports
# the same env var to both). Tokens of <think> per agent turn.
THINKING_BUDGET = _env_int("QWEN38_OBL_THINKING_BUDGET", 4096)
# The card asks for >= 2048 visible tokens; with thinking on, the reasoning
# segment comes out of the same max_tokens, so the floor grows by the budget.
AGENT_MAX_TOKENS_FLOOR = (2048 + THINKING_BUDGET) if THINKING else 2048


# Stability: a greedy 27B that starts looping at max_tokens=32000 holds the
# engine for 20+ minutes. Kilo's turn loop re-prompts anyway, so cap one
# completion here (the card only asks for >= 2048).
AGENT_MAX_TOKENS_CAP = max(_env_int("QWEN38_OBL_MAX_TOKENS", 8192), AGENT_MAX_TOKENS_FLOOR)
# No-tool chat (title, plain Q&A) and compaction decode at ~5 tok/s once the
# prompt is large: 8192 tokens is 25+ minutes. Measured compaction summaries
# are ~600 tokens; 1536 leaves headroom and bounds the worst case to ~5 min.
CHAT_MAX_TOKENS_CAP = _env_int("QWEN38_OBL_CHAT_MAX_TOKENS", 2048)
COMPACTION_MAX_TOKENS_CAP = _env_int("QWEN38_OBL_COMPACTION_MAX_TOKENS", 1536)
# Prompt-size guard. mtplx decode collapses from ~25 tok/s at 30k prompt
# tokens to 2-5 tok/s past ~50k (measured 2026-09-02). Kilo's own compaction
# is the primary control (see limit.context in kilo.jsonc); this is the
# backstop: oldest non-system turns are dropped in tool-call-consistent groups
# until the history fits. ~4 chars/token.
CONTEXT_CHAR_BUDGET = _env_int("QWEN38_OBL_CONTEXT_CHARS", 160_000)
_TRIM_NOTE = "[proxy: {n} older messages dropped to fit the local context window]"
# Cross-turn repetition guard. A greedy model that re-issues the SAME tool
# call (or the same prose) turn after turn never converges: measured 17+
# identical 38-token completions while the prompt grew 2.8k tokens a step.
# In-message loop detection (_looks_like_prose_loop) cannot see this because
# each completion is short. Consecutive identical assistant turns:
#   >= REPEAT_NUDGE_MIN  -> system nudge
#   >= REPEAT_NO_TOOLS   -> nudge + tools removed (forces a final text answer,
#                           which ends Kilo's turn loop)
#   >= REPEAT_HARD_STOP  -> synthetic final answer, upstream not called
# 2026-09-02: 3/4 -> 5/6. Live session with thinking on: the model re-issued
# one `cat ...` three times, the guard took its tools away at n=3 and the
# forced text answer ENDED Kilo's turn nine rounds into a research task --
# the guard was the thing quitting. The response-side duplicate retry below
# (_apply_duplicate_retry) now catches an identical call before it is even
# forwarded; tool removal is the last resort, not the second step.
REPEAT_NUDGE_MIN = 2
REPEAT_NO_TOOLS = 5
REPEAT_HARD_STOP = 6
_REPEAT_NUDGE = (
    "\n\n[Harness] REPEATED ACTION: your last {n} assistant turns were "
    "identical. That tool result is already in the conversation above. Do NOT "
    "issue the same tool call or text again. Take a DIFFERENT next step, or "
    "answer the user now with what you already know."
)
_REPEAT_STOP_TEXT = (
    "[Harness] Stopped: the model repeated the same action {n} times without "
    "progress ({what}). The last tool result is above. Rephrase the request "
    "or name the exact file/command to continue."
)
LOOP_MARKER = "Do not stop after 1 of N."
# Keep this short (finite attention). Extra recovery lives in scoped middleware
# below — AutoSaddler: capability/loop logic first, then steering; prefer
# replacement over stacking more rules.
LOOP_PREFIX = (
    "Finish every requested step. Prefer tools over prose. After each tool "
    "result, immediately take the next action. Do not stop after 1 of N. "
    "Do not recap; act. Empty or error tool output: retry with a simpler "
    "local command (ls/glob/grep/read). Do not write a final answer while any "
    "requested step is unfinished or unverified: after editing a file, run the "
    "tests/build or read it back with a tool first. Never end with a question "
    "or 'let me know if…'. Tool calls must be real tool_calls, never prose like "
    "'[Calling tool: ...]'. Never repeat an identical tool call; its result "
    "is already above. For 'what does X do' questions read only the file "
    "header (read with limit=150), not the whole file. Reply in English "
    "only — never Chinese/CJK prose; quote foreign text only verbatim.\n\n"
)

# Language enforcement. The OBLITERATED fine-tune is Chinese-heavy and
# answers English prompts in Chinese (2026-09-02: whole session recaps in
# Chinese at 28 tok/s). Belt and braces: (1) an English-only system
# directive on every chat request, (2) a CJK reply is regenerated once
# with a hard nudge — early-abort on the SSE path (the first window of
# content is held back, so nothing was forwarded), re-fetch on the
# buffered / non-stream paths. Kill switch: QWEN38_OBL_ENGLISH=0.
ENGLISH_GUARD = (
    os.environ.get("QWEN38_OBL_ENGLISH", "1").strip().lower()
    not in ("0", "false", "off")
)
_ENGLISH_ONLY_DIRECTIVE = (
    "\n\n[Harness] ENGLISH ONLY: reply in English exclusively. Never write "
    "Chinese, Japanese, or Korean. Every heading, list, summary, question, "
    "and explanation must be in English. Keep another language only inside "
    "verbatim quotes from files or command output."
)
_LANG_RETRY_NUDGE = (
    "\n\n[Harness] LANGUAGE FAILURE: your previous reply was in Chinese "
    "(CJK). Write the SAME answer again, entirely in English. English only — "
    "no Chinese words or characters outside verbatim quotes."
)
_CJK_RE = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
_CJK_MIN_CHARS = 8             # fewer CJK chars is quoting, not a language flip
_CJK_MIN_LEN = 24             # do not judge tiny interjections
_CJK_RATIO = 0.30             # CJK share of letters that counts as a flip
CJK_CANCEL_WINDOW_CHARS = 64  # streamed prefix held back while deciding


def _is_mostly_cjk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < _CJK_MIN_LEN:
        return False
    cjk = len(_CJK_RE.findall(t))
    if cjk < _CJK_MIN_CHARS:
        return False
    letters = sum(1 for ch in t if ch.isalpha())
    return letters > 0 and cjk / letters >= _CJK_RATIO


def _inject_english_directive(body: dict) -> None:
    if not ENGLISH_GUARD:
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    _nudge_system(
        msgs,
        "[Harness] ENGLISH ONLY:",
        _ENGLISH_ONLY_DIRECTIVE,
        "english-only directive",
    )


def _apply_language_retry(body: dict) -> None:
    """Second chance after a CJK reply: same history + hard English nudge."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    _nudge_system(
        msgs,
        "[Harness] LANGUAGE FAILURE:",
        _LANG_RETRY_NUDGE,
        "language retry",
    )


def _split_sse_events(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete SSE events (each ends with a blank line) + leftover bytes."""
    events: list[bytes] = []
    rest = buf
    while True:
        i = rest.find(b"\n\n")
        j = rest.find(b"\r\n\r\n")
        if i < 0 and j < 0:
            break
        if j >= 0 and (i < 0 or j < i):
            events.append(rest[: j + 4])
            rest = rest[j + 4 :]
        else:
            events.append(rest[: i + 2])
            rest = rest[i + 2 :]
    return events, rest

# AutoSaddler TB2 infra patch: raise tool-output room, still cap context.
TOOL_RESULT_MAX_CHARS = 30_000
TOOL_RESULT_KEEP_HEAD = 12_000
TOOL_RESULT_KEEP_TAIL = 8_000

_EMPTY_TOOL_EXACT = frozenset(
    {
        "(no output)",
        "no output",
        "(empty)",
        "empty",
        "empty response",
        "null",
        "none",
        "[]",
        "{}",
        "''",
        '""',
        "no matches",
        "no matches found",
        "(no matches)",
    }
)

_FAKE_ACTION_RE = re.compile(
    r"(?is)"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:next steps?|to-?dos?)\b"
    r"|(?:^|\n)\s*(?:executing|running|probing|fetching)\b"
    r"|\bi(?:'| a)?m\s+(?:going\s+to|about\s+to)\s+"
    r"|\bi(?:'| wi)?ll\s+(?:start|probe|check|fetch|run|search|use|look|verify)\b"
    r"|\blet\s+me\s+(?:probe|check|fetch|run|search|verify|start)\b"
    r"|\bnext[,:]?\s+i\s+will\b"
    r"|\bi\s+will\s+(?:start|probe|check|fetch|run|search)\b"
)
_EARLY_STOP_PLAN_RE = re.compile(
    r"(?is)"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:next steps?|to-?dos?|plan|remaining (?:work|steps|tasks|items)|pending|follow-?ups?)\b"
    r"|\bi(?:'| a)?m\s+(?:going\s+to|about\s+to)\s+"
    r"|\bi(?:'| wi)?ll\s+(?:start|probe|check|fetch|run|search|use|look|verify|read|list|now|then|next)\b"
    r"|\blet\s+me\s+(?:probe|check|fetch|run|search|verify|start|read|list|know\s+if)\b"
    r"|\bnext[,:]?\s+i\s+will\b"
    # Hand-back endings: the model knows work remains and asks permission.
    r"|\b(?:would|do)\s+you\s+(?:like|want)\s+me\s+to\b"
    r"|\bshall\s+i\s+(?:proceed|continue|go\s+ahead|run|start)\b"
    r"|\bshould\s+i\s+(?:proceed|continue|go\s+ahead)\b"
    r"|\b(?:still|yet)\s+(?:to\s+be\s+|to\s+)?(?:done|verified|checked|tested|implemented)\b"
    r"|\bnot\s+yet\s+(?:verified|tested|checked|run|implemented|done)\b"
)

_EMPTY_TOOL_NUDGE = (
    "\n\n[Harness] EMPTY TOOL RESULT: the latest tool output was empty or "
    "useless. Do not write a revised plan. Next message MUST be a local tool "
    "(bash ls/echo, glob, grep, or read) on a real path. If FileNotFound, "
    "list the parent directory — do not idle."
)
_MISSING_PATH_NUDGE = (
    "\n\n[Harness] MISSING PATH: that file or directory is not in this "
    "workspace. Do not invent a project. Next message MUST read README.md. "
    "Do not ls or cat the missing name again. Do not continue a fake task."
)
_FAKE_ACTION_NUDGE = (
    "\n\n[Harness] FAKE ACTION: the last assistant message claimed to act "
    "('Executing…', 'I will check…') but emitted NO tool_calls. Next message "
    "MUST include tool_calls — run the command now. No plans."
)
_PROSE_LOOP_NUDGE = (
    "\n\n[Harness] PROSE LOOP: the last assistant turn repeated itself "
    "without tool_calls. Stop monologuing. Next message MUST be tool_calls "
    "only — take the next unfinished step."
)
_CYCLE_NUDGE = (
    "\n\n[Harness] LOOP DETECTED: you are cycling over the same read-only "
    "commands ({what}) without making progress — their output is already in "
    "the conversation above. Re-reading status/summary/journal files will not "
    "advance the task. STOP re-observing. Take a DIFFERENT, forward action "
    "that changes state or produces a NEW finding (run an analysis/build/test, "
    "edit a file, or inspect something not yet seen). If — and only if — the "
    "mission is genuinely complete, state the final conclusion with the "
    "evidence. Do not issue any of the repeated commands again."
)
_CYCLE_BREAK_USER = (
    "[Harness] BANNED COMMAND: you have run `{what}` {n} times this turn with "
    "no new result. It is now BANNED — running it again will END the turn. "
    "Its output is already above:\n---\n{head}\n---\n"
    "Take a COMPLETELY DIFFERENT action that moves the mission forward — "
    "analyse code, edit or create a file, run a build/test, or inspect "
    "something not yet seen — or, only if the mission is truly complete, give "
    "the final conclusion with its evidence."
)
_CYCLE_STOP_TEXT = (
    "[Harness] Stopped: stuck in a read-only loop. `{what}` ran repeatedly "
    "with no new information and the model did not break out when redirected. "
    "The last tool result is above. To continue, reply with a concrete next "
    "action or a narrower goal — e.g. name the specific file to analyse or the "
    "finding to produce."
)
_AFTER_TOOL_NUDGE = (
    "\n\n[Harness] Tool result received. If any requested step is unfinished, "
    "emit the next tool_call now. Do not recap. Verify before stopping."
)
_EARLY_STOP_NUDGE = (
    "\n\n[Harness] EARLY STOP: that was a plan/next-steps dump, not a finished "
    "job. Next message MUST be tool_calls for the next unfinished step. "
    "Do not recap."
)
_RETRY_USER = (
    "[Harness] EARLY STOP: emit tool_calls for the next unfinished step now. "
    "No Next steps list. No recap. Run the next command."
)
# Verification gate: a final text answer whose most recent action was a file
# edit/write (nothing run or read back since) is not a finished job.
_VERIFY_GATE_NUDGE = (
    "\n\n[Harness] VERIFY GATE: you changed files and then answered without "
    "verifying. Before any final answer, run the project's tests/build or "
    "read back the changed file with a tool. Next message MUST be that "
    "verification tool_call. Do not recap."
)
_VERIFY_RETRY_USER = (
    "[Harness] VERIFY GATE: run the verification now (tests/build, or read the "
    "file you changed). tool_calls only. The final answer comes after the "
    "verification result."
)
_EDIT_TOOL_NAMES = frozenset(
    {"edit", "write", "multiedit", "apply_patch", "apply_diff", "write_to_file",
     "replace_in_file", "insert_content", "search_and_replace"}
)
# Kilo's grep tool parses `rg --json`; a matched line > 64KB (minified dist
# bundle, package-lock.json, .map) aborts the whole search with this error.
# Qwen then re-issues the identical grep until the repeat guard fires.
_GREP_OVERFLOW_NEEDLES = (
    "ripgrep json record exceeded",
    "record exceeded 65536 bytes",
)
_GREP_OVERFLOW_NUDGE = (
    "\n\n[Harness] GREP OVERFLOW: the last grep hit a >64KB line (minified "
    "bundle / lockfile / node_modules / dist). Do NOT repeat the same grep. "
    "Either narrow `path` to a source dir and set `include` (e.g. "
    "\"*.{js,ts,py}\"), or run bash: rg -n --max-columns 300 "
    "-g '!node_modules' -g '!dist' -g '!*lock*' -g '!*.min.*' PATTERN PATH."
)
# Kilo grep schema: pattern, path, include, ignoreCase, context, limit, literal.
_GREP_TOOL_NAMES = frozenset({"grep", "grep_search", "search_files", "ripgrep"})
_GREP_KNOWN_KEYS = frozenset(
    {"pattern", "path", "include", "ignoreCase", "context", "limit", "literal"}
)
# Claude-Code / Cursor style parameter names the model hallucinates.
_GREP_KEY_ALIASES = {
    "glob": "include",
    "file_pattern": "include",
    "filePattern": "include",
    "regex": "pattern",
    "query": "pattern",
    "-i": "ignoreCase",
    "case_insensitive": "ignoreCase",
    "caseInsensitive": "ignoreCase",
    "-C": "context",
    "-A": "context",
    "-B": "context",
    "head_limit": "limit",
    "max_results": "limit",
    "maxResults": "limit",
    "fixed_strings": "literal",
    "-F": "literal",
}
_RG_EXCLUDE_GLOBS = (
    "!node_modules",
    "!dist",
    "!build",
    "!out",
    "!.git",
    "!*lock*",
    "!*.min.*",
    "!*.map",
    "!*.wasm",
)
_RG_MAX_COLUMNS = 300
_RG_DEFAULT_LIMIT = 100
_PLAN_SCRIPT_RE = re.compile(
    r"(?:ONEPHONE_I_CONFIRM=1\s+)?(?:\./)?tools/[\w.-]+\.sh"
)
# Cap runaway tool loops: JIT continue only for the first N tools after
# the last real user message. A plan/next-steps dump (no tool_calls) is
# retried once per turn while the tool round count is <= EARLY_STOP_FORCE_MAX;
# it tracks AFTER_TOOL_NUDGE_MAX so every nudged round can also be forced.
# Round 0 is the first reply after the user prompt (first-turn guard).
# 2026-09-02: 3 -> 8. Live Titan M2 session: the model recapped and stopped
# at round 3, exactly where the nudge was stripped; a human had to type
# "continue". Kilo's own step budget (kilo.json steps=50) is the hard cap.
# 2026-09-02 (2): 8 -> 40. With thinking on, a research task ran 9 rounds in
# 90 s and the cap stripped the nudge exactly when the model was mid-task;
# past the cap the stream is passthrough and NO response-side check runs,
# so a recap or a duplicate call ends the turn unchallenged. The nudge is
# what keeps the loop going; a real final answer (not a plan, not an
# unverified edit) still passes at any round. Kilo steps=50 is the hard cap.
AFTER_TOOL_NUDGE_MAX = _env_int("QWEN38_OBL_AFTER_TOOL_MAX", 40)
EARLY_STOP_FORCE_MAX = AFTER_TOOL_NUDGE_MAX
_UPSTREAM_GEN_LOCK = threading.RLock()
# Mirrors the lock for /healthz ("engine_busy") without touching the lock.
_UPSTREAM_GEN_LOCK_BUSY = threading.Event()
_IN_FLIGHT_RE = re.compile(r"already in flight", re.I)
_SESSION_HEADER_DROP = frozenset(
    {
        "x-mtplx-session-id",
        "x-session-affinity",
        "x-session-id",
        "x-openwebui-chat-id",
        "x-openwebui-user-id",
    }
)
_SESSION_BODY_DROP = (
    "user",
    "session_id",
    "mtplx_session_id",
    "chat_id",
    "conversation_id",
)
_IN_FLIGHT_RETRIES = 4
# Per-read socket timeout toward mtplx. Only trips if the engine is truly
# dead: the client side is kept alive separately with SSE keepalive comments.
UPSTREAM_TIMEOUT = _env_int("QWEN38_OBL_UPSTREAM_TIMEOUT", 300)
# One generation at a time on the engine. A waiter that cannot get the lock
# within this budget gets a clean 503 (Kilo shows an error and can retry)
# instead of a silent multi-minute "Thinking" spinner.
LOCK_WAIT_TIMEOUT = _env_int("QWEN38_OBL_LOCK_WAIT", 900)
# SSE comment lines are ignored by every OpenAI-compatible parser; sending one
# every few seconds keeps Kilo's headerTimeout/chunkTimeout from tripping
# while we buffer a response for the early-stop middleware, and lets us notice
# a client that already gave up (write fails -> abort upstream, free the lock).
SSE_KEEPALIVE_SECS = 3.0
_SSE_KEEPALIVE = b": keepalive\n\n"
_SSE_DONE = b"data: [DONE]\n\n"
_PROXY_EMPTY_HINT = "[proxy] Command printed nothing"
_CMD_SUB_RE = re.compile(r"`|\$\(")
_SAFE_BASH_STARTS = (
    "cat ", "head ", "tail ", "less ", "ls ", "tree ", "pwd",
    "echo ", "wc ", "grep ", "rg ", "ag ", "jq ", "file ",
    "diff ", "du ", "df ", "date", "uname", "whoami",
    "which ", "cd ", "adb ",
)
_CAT_PATH_RE = re.compile(
    r"(?:^|[\s;|&])(?:cat|head|tail|less)\s+(?:-[nN]\s+\d+\s+)?['\"]?"
    r"((?:/|~/|\./)?[\w./~-]+\.[A-Za-z0-9]+)",
    re.I,
)
_QUOTED_FILE_RE = re.compile(
    r"""["']((?:[\w./~-]+)\.[A-Za-z0-9]+)["']"""
)
_FILENAME_RE = re.compile(
    r"(?<![\w.~-])(/?[\w./~-]+\.(?:sh|py|js|ts|tsx|jsx|json|md|txt|yaml|yml|toml|rs|go|rb|java|kt|swift|c|h|cpp|hpp|mk))\b",
    re.IGNORECASE,
)
_NEXT_STEP_MAX_READS = 4


def _active_harness() -> dict[str, Any]:
    try:
        return load_active()
    except Exception:
        return {}


def _loop_prefix() -> str:
    text = str(_active_harness().get("loop_prefix") or LOOP_PREFIX).strip()
    return text + "\n\n"


def _empty_tool_nudge() -> str:
    return str(_active_harness().get("empty_tool_nudge") or _EMPTY_TOOL_NUDGE)


def _fake_action_nudge() -> str:
    return str(_active_harness().get("fake_action_nudge") or _FAKE_ACTION_NUDGE)


def _prose_loop_nudge() -> str:
    return str(_active_harness().get("prose_loop_nudge") or _PROSE_LOOP_NUDGE)

_COMPACTION_HINTS = (
    "summarize the conversation",
    "generate a brief summary",
    "compact the conversation",
    "create a concise summary",
    "conversation summary",
    "compress conversation context",
    "compress the conversation",
    "agent=compaction",
)
# A tool-less request carrying this many turns of history is a compaction
# even when the prompt text does not match a hint (Kilo's compaction agent
# prompt is user-configurable). Plain chat never arrives with a long history.
_COMPACTION_MIN_MSGS = 16


def _looks_like_compaction(body: dict) -> bool:
    tc = body.get("tool_choice")
    if tc == "none" or (isinstance(tc, dict) and tc.get("type") == "none"):
        return True
    # Kilo agent system prompts often mention "summarize the conversation".
    # That is not compaction. Never strip tools on a real tool turn.
    tools = body.get("tools") or body.get("functions") or []
    if isinstance(tools, list) and tools:
        return False
    msgs = body.get("messages") or []
    if isinstance(msgs, list) and len(msgs) >= _COMPACTION_MIN_MSGS:
        return True
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("system", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            content = "\n".join(parts)
        if not isinstance(content, str):
            continue
        low = content.lower()
        if any(h in low for h in _COMPACTION_HINTS):
            return True
    return False


def _message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _set_message_text(msg: dict, text: str) -> None:
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text"):
                block["text"] = text
                return
        content.append({"type": "text", "text": text})
        return
    msg["content"] = text


def _truncate_tool_text(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    head = text[:TOOL_RESULT_KEEP_HEAD]
    tail = text[-TOOL_RESULT_KEEP_TAIL:]
    skipped = len(text) - TOOL_RESULT_KEEP_HEAD - TOOL_RESULT_KEEP_TAIL
    return (
        f"{head}\n... [proxy: truncated {skipped} chars of tool output] ...\n{tail}"
    )


def _truncate_tool_messages(messages: list[dict]) -> int:
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("tool", "function"):
            continue
        text = _message_text(msg)
        if not text or len(text) <= TOOL_RESULT_MAX_CHARS:
            continue
        _set_message_text(msg, _truncate_tool_text(text))
        n += 1
    return n


def _tool_result_is_error_or_missing(text: str) -> bool:
    low = (text or "").lower()
    return any(
        needle in low
        for needle in (
            "no such file",
            "filenotfound",
            "file not found",
            "unexpected eof while looking for matching",
            "syntax error: unexpected end of file",
        )
    )


def _is_empty_tool_content(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _PROXY_EMPTY_HINT in t:
        return True
    if _tool_result_is_error_or_missing(t):
        return True
    low = t.lower()
    if low in _EMPTY_TOOL_EXACT:
        return True
    if not any(ch.isalnum() for ch in t):
        return True
    return False


def _recent_empty_tool_streak(messages: list[dict] | None) -> int:
    if not messages:
        return 0
    empty = 0
    saw_tool = False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            saw_tool = True
            if _is_empty_tool_content(_message_text(msg)):
                empty += 1
            else:
                break
            continue
        if role == "assistant":
            if saw_tool:
                break
            if not msg.get("tool_calls") and not _is_empty_tool_content(
                _message_text(msg)
            ):
                break
            continue
        if role == "user":
            break
        if saw_tool:
            break
    return empty


def _looks_like_prose_loop(text: str) -> bool:
    """True when assistant prose is looping (line or 8-gram overlap)."""
    t = (text or "").strip()
    if len(t) < 400:
        return False
    lines = [ln.strip() for ln in t.splitlines() if len(ln.strip()) >= 16]
    if len(lines) >= 4:
        top = Counter(lines).most_common(1)[0][1]
        if top >= 4 or top / len(lines) >= 0.35:
            return True
    words = t.split()
    if len(words) >= 24:
        grams = [" ".join(words[i : i + 8]) for i in range(len(words) - 7)]
        if grams:
            top = Counter(grams).most_common(1)[0][1]
            if top >= 5 or top / len(grams) >= 0.35:
                return True
    return False


def _assistant_prose_loop(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return False
        if role == "assistant":
            if msg.get("tool_calls"):
                return False
            return _looks_like_prose_loop(_message_text(msg))
        if role == "user":
            return False
    return False


def _assistant_faked_action(messages: list[dict] | None) -> bool:
    if not messages:
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return False
        if role == "assistant":
            if msg.get("tool_calls"):
                return False
            text = _message_text(msg)
            if len(text.strip()) < 24:
                return False
            return bool(_FAKE_ACTION_RE.search(text))
        if role == "user":
            return False
    return False


def _assistant_signature(msg: dict) -> str | None:
    """Normalised identity of an assistant turn: tool calls (name+args) + text.

    None when the turn is too small to mean anything (a bare 'Done.').
    """
    tcs = msg.get("tool_calls")
    sig_tools: list[str] = []
    if isinstance(tcs, list):
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    pass
            if isinstance(args, dict):
                args = json.dumps(args, sort_keys=True, ensure_ascii=False)
            sig_tools.append(f"{name}({args})")
    text = re.sub(r"\s+", " ", _message_text(msg)).strip().lower()
    if not sig_tools and len(text) < 16:
        return None
    return "|".join(sorted(sig_tools)) + "#" + text


def _assistant_repeat_count(messages: list[dict] | None) -> tuple[int, str]:
    """How many CONSECUTIVE assistant turns (ignoring interleaved tool/user
    messages) are identical to the most recent one. Returns (count, sig)."""
    if not messages:
        return 0, ""
    last_sig: str | None = None
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        sig = _assistant_signature(msg)
        if last_sig is None:
            if sig is None:
                return 0, ""
            last_sig = sig
            count = 1
            continue
        if sig == last_sig:
            count += 1
        else:
            break
    return count, last_sig or ""


def _repeat_summary(sig: str) -> str:
    head = sig.split("#", 1)[0]
    if head:
        return head[:120]
    return "identical text: " + sig.split("#", 1)[-1][:80]


# Non-consecutive cycle detector (2026-09-02, live trace 19:16-19:18). The
# model looped read(SUMMARY) -> tail(JOURNAL) -> ls(recent) -> read(SUMMARY)
# ... for 12 rounds, re-reading the same three status files and never doing
# the research. The consecutive repeat guard saw repeat_count=1 the whole
# time because no TWO ADJACENT turns are identical. This looks at the tool
# signatures of the last CYCLE_WINDOW assistant turns and reports how heavily
# they cycle: the max repeat of any one signature, how many distinct ones,
# and a human summary of the repeated ones.
CYCLE_WINDOW = _env_int("QWEN38_OBL_CYCLE_WINDOW", 8)
# Fire when one signature recurs at least this many times in the window.
CYCLE_REPEAT_MIN = _env_int("QWEN38_OBL_CYCLE_MIN", 3)


def _tool_signature(msg: dict) -> str | None:
    """Just the tool-call half of _assistant_signature (name+args), no text.

    Read-only tools that only re-observe state are what cycle; a signature of
    tools alone means 'ls a; cat b' and 'ls a; cat b' match even if the model
    wrapped them in different prose each time.
    """
    full = _assistant_signature(msg)
    if full is None:
        return None
    head = full.split("#", 1)[0]
    return head or None


def _assistant_cycle(
    messages: list[dict] | None, window: int = CYCLE_WINDOW
) -> tuple[int, int, str]:
    """(max_recurrence, distinct_sigs, summary) over the last ``window``
    assistant TOOL turns. max_recurrence == 1 means no cycle."""
    sigs: list[str] = []
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        sig = _tool_signature(msg)
        if sig is None:
            continue
        sigs.append(sig)
        if len(sigs) >= window:
            break
    if not sigs:
        return 1, 0, ""
    counts = Counter(sigs)
    top_sig, top_n = counts.most_common(1)[0]
    repeated = [sig for sig, n in counts.items() if n >= 2]
    summary = "; ".join(_repeat_summary(sig + "#") for sig in repeated[:3])
    return top_n, len(counts), summary


# When a single command has recurred this many times in the window the nudge
# has demonstrably failed (live: cycle guard fired 4x, model kept re-running
# the same `tail`). At this point the proxy stops asking and forces the break
# response-side: ban the command, and if the model STILL emits it, end the
# turn with a visible message rather than spin. Above CYCLE_REPEAT_MIN (nudge)
# but reached only by a genuinely stuck run.
CYCLE_BREAK_MIN = _env_int("QWEN38_OBL_CYCLE_BREAK", 6)


def _cycled_signatures(
    messages: list[dict] | None, min_count: int, window: int = CYCLE_WINDOW
) -> set[str]:
    """Tool signatures that recur >= ``min_count`` times in the last
    ``window`` assistant tool turns (the ones to ban)."""
    counts: Counter[str] = Counter()
    n = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        sig = _tool_signature(msg)
        if sig is None:
            continue
        counts[sig] += 1
        n += 1
        if n >= window:
            break
    return {sig for sig, c in counts.items() if c >= min_count}


# Novelty starvation: the missing exit for MULTI-command cycles (2026-09-02).
# A period-N cycle tops out at CYCLE_WINDOW/N recurrences of any one signature,
# so a 2- or 3-cycle can never reach CYCLE_BREAK_MIN and only ever gets nudged
# -- forever. Live log: 26 of 37 cycle detections had max<6 and no exit at all;
# termination fell through to Kilo's own steps=50 budget. What every cycle
# shares, whatever its period, is that no NEW tool signature ever appears.
# Count assistant tool turns since one last produced an unseen signature:
# genuine progress resets it to 0, a cycle of ANY period drives it up without
# bound. Thresholds sit inside AFTER_TOOL_NUDGE_MAX (40) and kilo steps (50)
# so the proxy stops the loop before the client's budget does.
CYCLE_STALE_NO_TOOLS = _env_int("QWEN38_OBL_CYCLE_STALE_NO_TOOLS", 10)
CYCLE_STALE_STOP = _env_int("QWEN38_OBL_CYCLE_STALE_STOP", 12)


def _turns_since_new_signature(messages: list[dict] | None, cap: int = 64) -> int:
    """Assistant tool turns since one last produced an unseen tool signature.

    Truncating the scan at ``cap`` can only UNDER-count staleness (older
    signatures are forgotten and read as new), so it never causes a false stop.
    """
    sigs: list[str] = []
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        sig = _tool_signature(msg)
        if sig is None:
            continue
        sigs.append(sig)
        if len(sigs) >= cap:
            break
    sigs.reverse()  # oldest -> newest
    seen: set[str] = set()
    stale = 0
    for sig in sigs:
        if sig in seen:
            stale += 1
        else:
            seen.add(sig)
            stale = 0
    return stale


def _message_chars(msg: dict) -> int:
    n = len(_message_text(msg))
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        n += len(json.dumps(tcs, ensure_ascii=False))
    return n


def _trim_history_to_budget(messages: list[dict], budget: int = CONTEXT_CHAR_BUDGET) -> int:
    """Drop the oldest non-system messages until the history fits ``budget``.

    Drops whole groups so the OpenAI contract holds: an assistant message with
    tool_calls is removed together with the tool results that answer it. The
    leading system message(s) and the last 4 messages are never touched.
    Returns the number of messages dropped.
    """
    total = sum(_message_chars(m) for m in messages if isinstance(m, dict))
    if total <= budget:
        return 0
    first = 0
    while first < len(messages) and isinstance(messages[first], dict) and messages[first].get("role") == "system":
        first += 1
    keep_tail = 4
    dropped = 0
    while total > budget and len(messages) - first > keep_tail:
        msg = messages[first]
        group_end = first + 1
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            while (
                group_end < len(messages)
                and isinstance(messages[group_end], dict)
                and messages[group_end].get("role") in ("tool", "function")
            ):
                group_end += 1
        if group_end > len(messages) - keep_tail:
            break
        for m in messages[first:group_end]:
            if isinstance(m, dict):
                total -= _message_chars(m)
        del messages[first:group_end]
        dropped += group_end - first
    # Never leave an orphaned tool result at the head of the history.
    while (
        len(messages) - first > keep_tail
        and isinstance(messages[first], dict)
        and messages[first].get("role") in ("tool", "function")
    ):
        total -= _message_chars(messages[first])
        del messages[first]
        dropped += 1
    if dropped:
        note = _TRIM_NOTE.format(n=dropped)
        if first > 0:
            sys_msg = messages[first - 1]
            _set_message_text(sys_msg, _message_text(sys_msg) + "\n\n" + note)
        else:
            messages.insert(0, {"role": "system", "content": note})
        log.info("[agent] context trim: dropped=%s remaining_chars=%s", dropped, total)
    return dropped


def _last_non_system_role(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role and role != "system":
            return str(role)
    return None


def _nudge_system(messages: list[dict], marker: str, nudge: str, log_tag: str) -> bool:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if marker in text:
            return True
        _set_message_text(msg, text + nudge)
        log.info("[agent] %s", log_tag)
        return True
    messages.insert(0, {"role": "system", "content": nudge.strip()})
    log.info("[agent] %s (new system)", log_tag)
    return True


def _strip_nudge(messages: list[dict], nudge: str) -> None:
    needle = (nudge or "").strip()
    if not needle:
        return
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        if needle not in text and nudge not in text:
            continue
        cleaned = text.replace(nudge, "").replace(needle, "")
        _set_message_text(msg, cleaned)


_HARNESS_USER_MARKERS = (
    "[Harness] EARLY STOP:",
    "[Harness] VERIFY GATE:",
    "[Harness] REPEATED ACTION:",
    "[Harness] BANNED COMMAND:",
)


def _is_harness_user(msg: dict) -> bool:
    text = _message_text(msg)
    return any(m in text for m in _HARNESS_USER_MARKERS)


def _tool_rounds_since_user(messages: list[dict] | None) -> int:
    """Tool results after the last real (non-harness) user message."""
    n = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            if _is_harness_user(msg):
                continue
            break
        if role in ("tool", "function"):
            n += 1
    return n


def apply_loop_middleware(messages: list[dict]) -> dict[str, Any]:
    """Scoped AutoSaddler middleware: empty-tool, fake-action, prose-loop, JIT continue.

    Hooks fire only on matching conversation state so they do not regress
    unrelated turns (AutoSaddler: over-broad PreToolUse hooks overfit).
    """
    trace: dict[str, Any] = {
        "empty_tool_streak": 0,
        "empty_tool_recovery": False,
        "fake_action": False,
        "fake_action_recovery": False,
        "prose_loop": False,
        "prose_loop_recovery": False,
        "after_tool_continue": False,
        "first_turn_guard": False,
        "truncated_tool_msgs": 0,
        "missing_path_recovery": False,
        "repeat_count": 0,
        "repeat_recovery": False,
        "repeat_no_tools": False,
        "repeat_stop": False,
        "repeat_what": "",
        "cycle_max": 1,
        "cycle_distinct": 0,
        "cycle_recovery": False,
        "cycle_what": "",
        "cycle_stale": 0,
        "cycle_stale_no_tools": False,
        "cycle_stale_stop": False,
        "trimmed_msgs": 0,
        "grep_overflow_recovery": False,
    }
    if not messages:
        return trace
    truncated = _truncate_tool_messages(messages)
    trace["truncated_tool_msgs"] = truncated
    cfg = _active_harness()
    if cfg.get("mw_context_trim", True):
        budget = int(cfg.get("context_char_budget") or CONTEXT_CHAR_BUDGET)
        trace["trimmed_msgs"] = _trim_history_to_budget(messages, budget)
    streak = _recent_empty_tool_streak(messages)
    fake = _assistant_faked_action(messages)
    prose = _assistant_prose_loop(messages)
    last_role = _last_non_system_role(messages)
    trace["empty_tool_streak"] = streak
    trace["fake_action"] = fake
    trace["prose_loop"] = prose
    streak_min = int(cfg.get("empty_streak_min") or 1)
    grep_overflow = _last_tool_was_grep_overflow(messages)
    if grep_overflow and cfg.get("mw_grep_overflow", True):
        _strip_nudge(messages, _AFTER_TOOL_NUDGE)
        _nudge_system(
            messages,
            "[Harness] GREP OVERFLOW:",
            _GREP_OVERFLOW_NUDGE,
            "grep-overflow recovery",
        )
        trace["grep_overflow_recovery"] = True
    elif streak >= streak_min and cfg.get("mw_empty_tool", True):
        if _last_tool_was_missing_path(messages) and _nudge_system(
            messages,
            "[Harness] MISSING PATH:",
            _MISSING_PATH_NUDGE,
            f"missing-path recovery (streak={streak})",
        ):
            trace["empty_tool_recovery"] = True
            trace["missing_path_recovery"] = True
        elif not _last_tool_was_missing_path(messages) and _nudge_system(
            messages,
            "[Harness] EMPTY TOOL RESULT:",
            _empty_tool_nudge(),
            f"empty-tool recovery (streak={streak})",
        ):
            trace["empty_tool_recovery"] = True
    elif last_role in ("tool", "function") and cfg.get("mw_after_tool", True):
        rounds = _tool_rounds_since_user(messages)
        after_max = int(cfg.get("after_tool_nudge_max") or AFTER_TOOL_NUDGE_MAX)
        if rounds <= after_max:
            _nudge_system(
                messages,
                "[Harness] Tool result received.",
                _AFTER_TOOL_NUDGE,
                "after-tool continue",
            )
            trace["after_tool_continue"] = True
        else:
            _strip_nudge(messages, _AFTER_TOOL_NUDGE)
            log.info("[agent] after-tool cap rounds=%s max=%s", rounds, after_max)
    elif last_role == "user" and cfg.get("mw_first_turn", True):
        # First reply after the user prompt. Kilo ends the turn if it comes
        # back as prose, so a plan/next-steps dump here strands the task
        # until a human types "continue". Arm the early-stop middleware so
        # that reply is buffered, checked, and retried once with tool_calls.
        trace["first_turn_guard"] = True
        log.info("[agent] first-turn guard armed")
    if fake and cfg.get("mw_fake_action", True) and _nudge_system(
        messages,
        "[Harness] FAKE ACTION:",
        _fake_action_nudge(),
        "fake-action recovery",
    ):
        trace["fake_action_recovery"] = True
    if prose and cfg.get("mw_prose_loop", True) and _nudge_system(
        messages,
        "[Harness] PROSE LOOP:",
        _prose_loop_nudge(),
        "prose-loop recovery",
    ):
        trace["prose_loop_recovery"] = True
    if cfg.get("mw_repeat_guard", True):
        rep, sig = _assistant_repeat_count(messages)
        trace["repeat_count"] = rep
        nudge_min = int(cfg.get("repeat_nudge_min") or REPEAT_NUDGE_MIN)
        no_tools_min = int(cfg.get("repeat_no_tools") or REPEAT_NO_TOOLS)
        stop_min = int(cfg.get("repeat_hard_stop") or REPEAT_HARD_STOP)
        if rep >= nudge_min:
            trace["repeat_what"] = _repeat_summary(sig)
            # Escalate: the marker is stable, the count in the text is not, so
            # replace an older nudge instead of stacking a second one.
            _strip_nudge_prefix(messages, "[Harness] REPEATED ACTION:")
            _nudge_system(
                messages,
                "[Harness] REPEATED ACTION:",
                _REPEAT_NUDGE.format(n=rep),
                f"repeat recovery (n={rep} what={trace['repeat_what']!r})",
            )
            trace["repeat_recovery"] = True
            trace["repeat_no_tools"] = rep >= no_tools_min
            trace["repeat_stop"] = rep >= stop_min
    if cfg.get("mw_cycle_guard", True):
        cyc_max, cyc_distinct, cyc_what = _assistant_cycle(messages)
        trace["cycle_max"] = cyc_max
        trace["cycle_distinct"] = cyc_distinct
        cyc_min = int(cfg.get("cycle_repeat_min") or CYCLE_REPEAT_MIN)
        # Fire only when the CONSECUTIVE repeat guard did not already handle
        # it (that one removes tools; this one keeps them and redirects).
        if cyc_max >= cyc_min and not trace.get("repeat_recovery"):
            trace["cycle_what"] = cyc_what
            _strip_nudge_prefix(messages, "[Harness] LOOP DETECTED:")
            _nudge_system(
                messages,
                "[Harness] LOOP DETECTED:",
                _CYCLE_NUDGE.format(what=cyc_what or "the same few commands"),
                f"cycle recovery (max={cyc_max} distinct={cyc_distinct} what={cyc_what!r})",
            )
            trace["cycle_recovery"] = True
        # Period-independent exit. Runs even when the consecutive repeat guard
        # owns this turn: that one has its own ladder and fires first, and a
        # stale count only rises while nothing new is being tried.
        stale = _turns_since_new_signature(messages)
        trace["cycle_stale"] = stale
        no_tools_at = int(cfg.get("cycle_stale_no_tools") or CYCLE_STALE_NO_TOOLS)
        stop_at = int(cfg.get("cycle_stale_stop") or CYCLE_STALE_STOP)
        if stale >= no_tools_at:
            trace["cycle_stale_no_tools"] = True
            trace["cycle_stale_stop"] = stale >= stop_at
            if not trace.get("cycle_what"):
                trace["cycle_what"] = _assistant_cycle(messages)[2]
            log.info(
                "[agent] cycle stale: %s turns with no new action (no_tools=%s stop=%s)",
                stale, no_tools_at, trace["cycle_stale_stop"],
            )
    if (
        trace.get("empty_tool_recovery")
        or trace.get("fake_action_recovery")
        or trace.get("prose_loop_recovery")
        or trace.get("repeat_recovery")
        or trace.get("cycle_recovery")
    ):
        append_live_event(
            {
                "empty_tool_recovery": trace.get("empty_tool_recovery"),
                "fake_action_recovery": trace.get("fake_action_recovery"),
                "prose_loop_recovery": trace.get("prose_loop_recovery"),
                "repeat_recovery": trace.get("repeat_recovery"),
                "repeat_count": trace.get("repeat_count"),
                "repeat_stop": trace.get("repeat_stop"),
                "cycle_recovery": trace.get("cycle_recovery"),
                "cycle_max": trace.get("cycle_max"),
                "after_tool_continue": trace.get("after_tool_continue"),
                "fake_action": fake,
                "prose_loop": prose,
                "empty_tool_streak": streak,
            }
        )
    return trace


def _strip_nudge_prefix(messages: list[dict], marker: str) -> None:
    """Remove a previously injected nudge paragraph that starts with ``marker``."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        text = _message_text(msg)
        i = text.find(marker)
        if i < 0:
            continue
        j = text.find("\n\n", i)
        cleaned = (text[:i] + (text[j:] if j >= 0 else "")).rstrip()
        _set_message_text(msg, cleaned)


def _apply_card_sampling(body: dict, *, thinking: bool = False) -> None:
    body["temperature"] = CARD_TEMPERATURE
    body["top_p"] = CARD_TOP_P
    body["frequency_penalty"] = CARD_FREQUENCY_PENALTY
    body.pop("top_k", None)
    _set_thinking(body, thinking)


def _set_thinking(body: dict, on: bool) -> None:
    """Thinking on/off for this request, explicitly, on every field mtplx reads.

    Top-level ``enable_thinking`` wins in mtplx; ``chat_template_kwargs`` is
    kept in agreement so the two can never disagree. ``reasoning_effort`` is
    only meaningful with thinking on (the template raises on anything but
    low/medium/xhigh, so the value is validated at import).
    """
    on = bool(on and THINKING)
    body["enable_thinking"] = on
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        ctk["enable_thinking"] = on
    if on:
        body["reasoning_effort"] = REASONING_EFFORT
    else:
        body.pop("reasoning_effort", None)


def _ensure_max_tokens(body: dict, *, floor: int, cap: int | None = None) -> None:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0 or value < floor:
        value = floor
    if cap is not None and value > cap:
        value = max(cap, floor)
    body["max_tokens"] = value
    # mtplx honours max_tokens; drop the alias so the two never disagree.
    body.pop("max_completion_tokens", None)


def _cap_max_tokens(body: dict, *, cap: int) -> None:
    """Clamp an explicit max_tokens without inventing one."""
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    if raw is None:
        return
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return
    if value > cap:
        body["max_tokens"] = cap
        body.pop("max_completion_tokens", None)


def _wrap_loop_prefix(prefix: str) -> str:
    return f"{AS_LOOP_START}\n{prefix.strip()}\n{AS_LOOP_END}\n\n"


def _apply_loop_prefix_text(text: str, prefix: str) -> str:
    wrapped = _wrap_loop_prefix(prefix)
    if AS_LOOP_START in text:
        return re.sub(
            r"\[AS_LOOP\].*?\[/AS_LOOP\]\s*",
            wrapped,
            text,
            count=1,
            flags=re.S,
        )
    if prefix.strip() in text:
        return text
    default = LOOP_PREFIX.strip()
    if default in text and prefix.strip() != default:
        return text.replace(default, prefix.strip(), 1)
    if LOOP_MARKER in text and prefix.strip() == default:
        return text
    return wrapped + text


def _inject_loop_prompt(body: dict) -> None:
    if not body.get("tools"):
        return
    prefix = _loop_prefix()
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        body["messages"] = [{"role": "system", "content": prefix.strip()}]
        return
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        content = msgs[0].get("content")
        if isinstance(content, str):
            msgs[0]["content"] = _apply_loop_prefix_text(content, prefix)
            return
        if isinstance(content, list):
            text = ""
            idx = None
            for i, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text") or "")
                    if idx is None:
                        idx = i
            new = _apply_loop_prefix_text(text, prefix)
            if idx is not None:
                content[idx]["text"] = new
            else:
                content.insert(0, {"type": "text", "text": new})
            return
    msgs.insert(0, {"role": "system", "content": prefix.strip()})


def summarize_request(body: dict) -> str:
    msgs = body.get("messages") or []
    roles = [
        m.get("role") if isinstance(m, dict) else type(m).__name__ for m in msgs
    ]
    tools = body.get("tools") or body.get("functions") or []
    last = ""
    for msg in reversed(msgs):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            last = content.replace("\n", " ")[:120]
        elif isinstance(content, list):
            bits = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    bits.append(str(block.get("text") or ""))
            last = " ".join(bits).replace("\n", " ")[:120]
        break
    return (
        f"keys={sorted(body.keys())} nmsg={len(msgs)} roles={roles} "
        f"ntools={len(tools) if isinstance(tools, list) else 0} "
        f"tool_choice={body.get('tool_choice')!r} stream={bool(body.get('stream'))} "
        f"user={last!r}"
    )


def prepare_body(body: dict, trace: dict | None = None) -> dict:
    """Mutate a chat-completions body to the OBLITERATUS + agent contract.

    Optional ``trace`` is filled with middleware flags for tests / logs.
    """
    tr: dict[str, Any] = trace if trace is not None else {}
    tr.clear()
    tr.update(
        {
            "compaction": False,
            "empty_tool_streak": 0,
            "empty_tool_recovery": False,
            "fake_action": False,
            "fake_action_recovery": False,
            "prose_loop": False,
            "prose_loop_recovery": False,
            "after_tool_continue": False,
            "truncated_tool_msgs": 0,
            "loop_prompt": False,
            "repeat_count": 0,
            "repeat_recovery": False,
            "repeat_no_tools": False,
            "repeat_stop": False,
            "repeat_what": "",
            "trimmed_msgs": 0,
        }
    )
    if _looks_like_compaction(body):
        body.pop("tools", None)
        body["tool_choice"] = "none"
        _apply_card_sampling(body)
        _ensure_max_tokens(body, floor=512, cap=COMPACTION_MAX_TOKENS_CAP)
        msgs = body.get("messages")
        if isinstance(msgs, list):
            # A compaction prompt is the one request guaranteed to arrive with
            # the whole history; keep it inside the window the engine can
            # actually decode at a usable speed.
            tr["truncated_tool_msgs"] = _truncate_tool_messages(msgs)
            tr["trimmed_msgs"] = _trim_history_to_budget(msgs)
        _inject_english_directive(body)
        tr["compaction"] = True
        return body

    _apply_card_sampling(body, thinking=bool(body.get("tools")))
    if body.get("tools"):
        _ensure_max_tokens(
            body, floor=AGENT_MAX_TOKENS_FLOOR, cap=AGENT_MAX_TOKENS_CAP
        )
        _inject_loop_prompt(body)
        _inject_english_directive(body)
        tr["loop_prompt"] = True
        msgs = body.get("messages")
        if isinstance(msgs, list):
            tr.update(apply_loop_middleware(msgs))
            _repair_history_tool_calls(msgs)
            # Self-healing: when a loop is detected, give THIS turn maximum
            # deliberation so the model can reason its way out instead of
            # re-emitting the same greedy action (live: 49 reasoning chars on
            # looping turns at medium). Capability before steering.
            if (
                tr.get("cycle_recovery")
                and THINKING
                and body.get("enable_thinking")
                and body.get("reasoning_effort") != "xhigh"
            ):
                body["reasoning_effort"] = "xhigh"
                tr["cycle_escalate_effort"] = True
                log.info("[agent] cycle: reasoning_effort -> xhigh for this turn")
            if tr.get("repeat_no_tools") or tr.get("cycle_stale_no_tools"):
                # The model keeps re-issuing one tool call. Take the tools
                # away for this turn so it has to answer in text, which ends
                # Kilo's tool loop instead of burning another engine slot.
                body.pop("tools", None)
                body.pop("functions", None)
                body["tool_choice"] = "none"
                # No tools -> no engine thinking budget; a <think> block would
                # eat the whole (now small) max_tokens and return no answer.
                _set_thinking(body, False)
                _cap_max_tokens(body, cap=CHAT_MAX_TOKENS_CAP)
                log.info(
                    "[agent] tools removed (repeat n=%s cycle_stale=%s)",
                    tr.get("repeat_count"), tr.get("cycle_stale"),
                )
    else:
        # Plain chat / title generation: still bound a runaway greedy loop,
        # but leave an absent max_tokens to the engine default.
        _inject_english_directive(body)
        _cap_max_tokens(body, cap=CHAT_MAX_TOKENS_CAP)
    return body


def _path_basename(path: str) -> str:
    return (path or "").rstrip("/").rsplit("/", 1)[-1]


def _unmatched_quotes(text: str) -> bool:
    s = text or ""
    return (s.count('"') - s.count('\\"')) % 2 == 1 or (
        s.count("'") - s.count("\\'")
    ) % 2 == 1


def _parent_dir_of_path(path: str) -> str:
    p = (path or "").strip().strip("'\"")
    p = re.sub(r"[*?].*$", "", p).rstrip("/")
    if not p or p in {".", ".."}:
        return "."
    if "/" not in p:
        return "."
    return p.rsplit("/", 1)[0] or "."


def _is_pathless_head(command: str) -> bool:
    return bool(re.match(r"^(?:head|tail)(?:\s+-\d+)?\s*$", command or ""))


def _restore_abs_prefix(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    if p.startswith(("Users/", "home/", "tmp/", "var/", "opt/", "private/")):
        return "/" + p
    return p


def _dedupe_repeated_path_prefix(path: str) -> str:
    abs_path = path.startswith("/")
    parts = [p for p in path.split("/") if p]
    n = len(parts)
    if n < 4:
        return path
    for seq_len in range(2, n // 2 + 1):
        seq = parts[:seq_len]
        repeats = 0
        i = 0
        while i + seq_len <= n and parts[i : i + seq_len] == seq:
            repeats += 1
            i += seq_len
        if repeats >= 2:
            rebuilt = seq + parts[i:]
            out = "/".join(rebuilt)
            return "/" + out if abs_path or rebuilt[0] in {"Users", "home"} else out
    return path


def _prefer_workspace_relative(path: str, workspace: str | None) -> str:
    if not path or not workspace:
        return path
    p = path.replace("\\", "/")
    ws = workspace.replace("\\", "/").rstrip("/")
    if p == ws:
        return "."
    if p.startswith(ws + "/"):
        return p[len(ws) + 1 :]
    bare = ws.lstrip("/")
    if p == bare:
        return "."
    if p.startswith(bare + "/"):
        return p[len(bare) + 1 :]
    return path


def _extract_workspace_dir(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    patterns = (
        r"Current Workspace Directory\s*\(([^)]+)\)",
        r"Current Working Directory[:\s]+([^\s\n]+)",
        r"Workspace Directory[:\s]+([^\s\n]+)",
        r"Working directory:\s*([^\s\n]+)",
        r"<cwd>\s*([^<\s]+)\s*</cwd>",
    )
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        text = _message_text(msg)
        if not text:
            continue
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                wd = m.group(1).strip().strip("()").rstrip("/")
                if wd:
                    return wd
    return None


def _last_tool_was_missing_path(messages: list[dict] | None) -> bool:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return _tool_result_is_error_or_missing(_message_text(msg))
        if role == "user":
            return False
    return False


def _tool_result_is_grep_overflow(text: str) -> bool:
    low = (text or "").lower()
    return any(needle in low for needle in _GREP_OVERFLOW_NEEDLES)


def _last_tool_was_grep_overflow(messages: list[dict] | None) -> bool:
    """True when the most recent tool result (since the last user turn) is the
    Kilo grep ``Ripgrep JSON record exceeded 65536 bytes`` failure."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            return _tool_result_is_grep_overflow(_message_text(msg))
        if role == "user":
            return False
    return False


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _coerce_int(val: Any, default: int | None = None) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _repair_grep_args(args: dict) -> dict:
    """Map hallucinated Claude-Code/Cursor grep params onto Kilo's schema and
    drop the unknown ones (``output_mode``, ``-n``, ``multiline``, ...)."""
    out: dict = {}
    for key, val in args.items():
        k = _GREP_KEY_ALIASES.get(key, key)
        if k == "type" and isinstance(val, str) and val.strip():
            k, val = "include", f"*.{val.strip().lstrip('.')}"
        if k not in _GREP_KNOWN_KEYS:
            continue
        if k in ("ignoreCase", "literal"):
            val = _coerce_bool(val)
        elif k in ("context", "limit"):
            val = _coerce_int(val)
            if val is None:
                continue
        elif k == "path" and (val is None or str(val).strip() in ("", ".")):
            continue
        out[k] = val
    if "pattern" not in out and "pattern" in args:
        out["pattern"] = args["pattern"]
    return out


_QUESTION_ITEM_KEYS = frozenset({"question", "header", "options", "multiple"})
_QUESTION_OPTION_KEYS = frozenset({"label", "description"})
_QUESTION_HEADER_MAX = 30


def _question_option(item, idx: int) -> dict | None:
    """Coerce one option into Kilo's ``{label, description}`` shape.
    Small models emit bare strings, ``{value,text}`` or ``{name}`` dicts."""
    if isinstance(item, str):
        label = item.strip()
        desc = ""
    elif isinstance(item, dict):
        label = str(
            item.get("label")
            or item.get("value")
            or item.get("name")
            or item.get("title")
            or item.get("text")
            or ""
        ).strip()
        desc = str(item.get("description") or item.get("desc") or "").strip()
    else:
        label = str(item).strip() if item is not None else ""
        desc = ""
    if not label:
        return None
    out = {"label": label[:200], "description": desc or label}
    if isinstance(item, dict):
        for k in ("mode", "labelKey", "descriptionKey"):
            if isinstance(item.get(k), str):
                out[k] = item[k]
    return out


def _question_item(item, idx: int) -> dict | None:
    if isinstance(item, str):
        item = {"question": item}
    if not isinstance(item, dict):
        return None
    text = str(
        item.get("question") or item.get("prompt") or item.get("text") or ""
    ).strip()
    raw_opts = item.get("options")
    if raw_opts is None:
        raw_opts = item.get("choices") or item.get("answers") or []
    if isinstance(raw_opts, (str, dict)):
        raw_opts = [raw_opts]
    if not isinstance(raw_opts, list):
        raw_opts = []
    opts = [o for o in (_question_option(o, i) for i, o in enumerate(raw_opts)) if o]
    seen: set[str] = set()
    opts = [o for o in opts if not (o["label"] in seen or seen.add(o["label"]))]
    if not text and not opts:
        return None
    if not text:
        text = "Which option?"
    if not opts:
        # Kilo requires options; a bare free-text question is still useful.
        opts = [
            {"label": "Yes", "description": "Confirm"},
            {"label": "No", "description": "Decline"},
        ]
    header = str(item.get("header") or item.get("title") or "").strip()
    if not header:
        # Prefer the "Topic:" prefix, else cut at a word boundary.
        lead = text.split(":", 1)[0].strip() if ":" in text else ""
        header = lead if 0 < len(lead) <= _QUESTION_HEADER_MAX else text
        header = re.sub(r"[?:.!]+$", "", header).strip() or "Question"
    if len(header) > _QUESTION_HEADER_MAX:
        cut = header[:_QUESTION_HEADER_MAX]
        header = (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:-")
    out: dict = {"question": text, "header": header, "options": opts}
    if "multiple" in item:
        out["multiple"] = bool(_coerce_bool(item.get("multiple")))
    for k in ("headerKey", "questionKey"):
        if isinstance(item.get(k), str):
            out[k] = item[k]
    return out


def _repair_question_args(args: dict) -> dict:
    """Normalize a ``question`` tool call onto Kilo's schema:
    ``{questions: [{question, header<=30, options: [{label, description}]}]}``.
    Handles the recurring local-model failures: missing header/options,
    options as bare strings, a single top-level ``question`` string, ``id``
    and other unknown keys."""
    raw = args.get("questions")
    if raw is None:
        if any(k in args for k in ("question", "prompt", "text", "options", "choices")):
            raw = [args]
        else:
            raw = []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    items = [q for q in (_question_item(q, i) for i, q in enumerate(raw)) if q]
    if not items:
        return args
    return {"questions": items}


def _grep_to_rg_command(args: dict) -> str | None:
    """Build a bash ``rg`` command equivalent to a Kilo grep call, with the
    globs/columns guards that avoid the 64KB JSON record abort."""
    pattern = str(args.get("pattern") or "")
    if not pattern:
        return None
    parts = [
        "rg",
        "-n",
        "--no-heading",
        "--max-columns",
        str(_RG_MAX_COLUMNS),
        "--max-columns-preview",
    ]
    if _coerce_bool(args.get("ignoreCase", False)):
        parts.append("-i")
    if _coerce_bool(args.get("literal", False)):
        parts.append("-F")
    ctx = _coerce_int(args.get("context"), 0) or 0
    if ctx > 0:
        parts += ["-C", str(ctx)]
    include = str(args.get("include") or "").strip()
    if include:
        parts += ["-g", shlex.quote(include)]
    for g in _RG_EXCLUDE_GLOBS:
        parts += ["-g", shlex.quote(g)]
    parts.append("--")
    parts.append(shlex.quote(pattern))
    path = str(args.get("path") or "").strip()
    if path:
        parts.append(shlex.quote(path))
    limit = _coerce_int(args.get("limit"), _RG_DEFAULT_LIMIT) or _RG_DEFAULT_LIMIT
    return " ".join(parts) + f" | head -n {limit}"


def _workspace_entry_exists(
    path: str, messages: list[dict] | None
) -> bool | None:
    ws = _extract_workspace_dir(messages)
    if not ws:
        return None
    raw = (path or "").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = _normalize_fs_path(raw, messages)
    if not raw or raw in {".", ".."}:
        return True
    cand = Path(raw) if Path(raw).is_absolute() else Path(ws) / raw
    try:
        return cand.exists()
    except OSError:
        return None


def _ls_target_dir(command: str) -> str | None:
    m = re.match(
        r"^(?:ls|tree)\s+(?:-[a-zA-Z0-9]+\s+)*['\"]?([^'\"\n]+)",
        (command or "").strip(),
        re.I,
    )
    if not m:
        return None
    target = re.split(r"[\s;|&<>]", m.group(1).strip(), maxsplit=1)[0]
    target = target.strip().strip("'\"").rstrip("/")
    if not target or target.startswith("-"):
        return None
    return target


def _is_invented_workspace_path(
    path: str, messages: list[dict] | None
) -> bool:
    exists = _workspace_entry_exists(path, messages)
    if exists is not False:
        return False
    raw = (path or "").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    first = raw.split("/")[0].lower()
    common = {
        "analysis", "tools", "src", "lib", "docs", "firmware", "sources",
        "third_party", "tests", "test", "scripts", ".pi", ".grok", "include",
        "app", "pkg", "cmd", "internal",
    }
    if first in common and "/" not in raw:
        return False
    return True


def _normalize_fs_path(path: str, messages: list[dict] | None = None) -> str:
    raw = (path or "").strip().strip("'\"")
    if not raw or raw in {".", ".."}:
        return raw
    raw = _restore_abs_prefix(raw)
    workspace = _extract_workspace_dir(messages)
    if workspace:
        raw = _dedupe_repeated_path_prefix(raw)
        ws = workspace.replace("\\", "/").rstrip("/")
        token = ws.lstrip("/")
        if token and token in raw.replace("\\", "/"):
            collapsed = raw.replace("\\", "/")
            while f"{token}/{token}" in collapsed:
                collapsed = collapsed.replace(f"{token}/{token}", token, 1)
            while f"{ws}/{token}" in collapsed:
                collapsed = collapsed.replace(f"{ws}/{token}", ws, 1)
            raw = collapsed
        raw = _prefer_workspace_relative(raw, workspace)
    else:
        raw = _dedupe_repeated_path_prefix(raw)
    return raw


def _recent_tool_result_paths(messages: list[dict] | None, limit_msgs: int = 8) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    count = 0
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        count += 1
        if count > limit_msgs:
            break
        blob = _message_text(msg)
        if _tool_result_is_error_or_missing(blob):
            continue
        for match in _FILENAME_RE.finditer(blob):
            path = _normalize_fs_path(match.group(1), messages)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _analysis_dir_markdown(
    path: str, messages: list[dict] | None
) -> str | None:
    """analysis/semgrep-domain-c/README.md → analysis/SEMGREP-DOMAIN-C.md."""
    raw = _normalize_fs_path((path or "").strip().strip("'\""), messages)
    parts = [p for p in raw.replace("\\", "/").split("/") if p and p != "."]
    if len(parts) < 2 or parts[0].lower() != "analysis":
        return None
    slug = parts[1]
    if "." in slug and slug.lower().rsplit(".", 1)[-1] in {"md", "txt", "json"}:
        return None
    want = re.sub(r"[^a-z0-9]+", "", slug.lower())
    if not want:
        return None
    ws = _extract_workspace_dir(messages)
    if ws:
        analysis_dir = Path(ws) / "analysis"
        try:
            names = list(analysis_dir.iterdir())
        except OSError:
            names = []
        for entry in names:
            stem = entry.stem if entry.suffix else entry.name
            if re.sub(r"[^a-z0-9]+", "", stem.lower()) != want:
                continue
            if entry.is_file():
                return f"analysis/{entry.name}"
    guess = f"analysis/{slug.upper()}.md"
    if guess != raw:
        return guess
    return None


def _resolve_read_path(path: str, messages: list[dict] | None) -> str | None:
    raw = _normalize_fs_path((path or "").strip().strip("'\""), messages)
    if not raw:
        return None
    if _workspace_entry_exists(raw, messages) is True:
        return raw
    sibling = _analysis_dir_markdown(raw, messages)
    if sibling and sibling != raw and _workspace_entry_exists(raw, messages) is not True:
        if _workspace_entry_exists(sibling, messages) is not False:
            log.info("[read-resolve] %s → %s", raw, sibling)
            return sibling
    known = _recent_tool_result_paths(messages, limit_msgs=24)
    extras = ["README.md", "AGENTS.md", "COMPLETE.md"]
    pool = list(dict.fromkeys(known + extras))
    pool.sort(key=lambda p: (len(p), p.count("/"), p))
    if raw in pool:
        return raw
    base = _path_basename(raw).lower()
    for item in pool:
        if _path_basename(item).lower() == base:
            return item
    names = [_path_basename(item) for item in pool]
    close = difflib.get_close_matches(_path_basename(raw), names, n=1, cutoff=0.72)
    if close:
        want = close[0].lower()
        for item in pool:
            if _path_basename(item).lower() == want:
                return item
    return raw if raw in extras else None


def _history_paths_by_basename(messages: list[dict] | None) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in reversed(_recent_tool_result_paths(messages, limit_msgs=16)):
        name = _path_basename(path)
        if name and name not in found:
            found[name] = path
    return found


_LEADING_CD_RE = re.compile(r"^\s*cd\s+['\"]?([^'\"\s;&|]+)['\"]?\s*(?:&&|;)")


def _leading_cd_dir(command: str) -> str | None:
    m = _LEADING_CD_RE.match(command or "")
    return m.group(1) if m else None


def _bash_path_exists(
    path: str, cd_dir: str | None, messages: list[dict] | None
) -> bool | None:
    """Does ``path`` exist as the shell would resolve it? None = unknown."""
    raw = (path or "").strip().strip("'\"")
    if not raw:
        return None
    if raw.startswith("~/"):
        raw = str(Path.home() / raw[2:])
    try:
        if Path(raw).is_absolute():
            return Path(raw).exists()
        if cd_dir:
            base = Path(cd_dir)
            if not base.is_absolute():
                ws = _extract_workspace_dir(messages)
                if not ws:
                    return None
                base = Path(ws) / base
            return (base / raw).exists()
    except OSError:
        return None
    return _workspace_entry_exists(raw, messages)


def _rewrite_missing_path_command(command: str, messages: list[dict] | None) -> str | None:
    """Point cat/head at the path history actually showed — for MISSING paths.

    A path that exists (as the shell resolves it, honouring a leading
    ``cd <dir> &&``) is real work and is left alone. 2026-09-02 Titan M2
    session: a correct ``cd repo && cat analysis/.../autohunt/STATE.json``
    was rewritten to the bare ``STATE.json`` an earlier ls had listed; that
    did not exist from the repo root, the model re-asked, the rewrite hit
    again, and the repeat guard ended the task after three identical turns.
    A directory-qualified path is also never downgraded to a bare basename.
    """
    known = _history_paths_by_basename(messages)
    if not known or not command:
        return None
    cd_dir = _leading_cd_dir(command)
    rewritten = command
    changed = False
    for match in _CAT_PATH_RE.finditer(command):
        raw = match.group(1)
        if _bash_path_exists(raw, cd_dir, messages) is True:
            continue
        name = _path_basename(raw)
        alt = known.get(name)
        if not alt or alt == raw:
            continue
        if "/" in raw.strip("./") and "/" not in alt:
            continue
        if _bash_path_exists(alt, cd_dir, messages) is False:
            continue
        rewritten = rewritten.replace(raw, alt)
        changed = True
    return rewritten if changed else None


def _paths_in_bash_command(command: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for rx in (_QUOTED_FILE_RE, _CAT_PATH_RE, _FILENAME_RE):
        for match in rx.finditer(command or ""):
            path = match.group(1)
            if path and path not in seen:
                seen.add(path)
                found.append(path)
    return found


def _safe_head_command(paths: list[str], messages: list[dict] | None = None) -> str:
    cleaned: list[str] = []
    parents: list[str] = []
    for p in paths:
        if not p or re.search(r"[\n|;|&`$<>]", p):
            continue
        n = _normalize_fs_path(p, messages)
        resolved = _resolve_read_path(n, messages) if messages is not None else n
        if resolved:
            if resolved not in cleaned:
                cleaned.append(resolved)
            continue
        known = _recent_tool_result_paths(messages) if messages else []
        if known:
            parent = _parent_dir_of_path(n)
            if parent not in parents:
                parents.append(parent)
        elif n not in cleaned:
            cleaned.append(n)
    if cleaned:
        return "head -80 " + " ".join(cleaned[:_NEXT_STEP_MAX_READS])
    if parents:
        return "ls " + " ".join(parents[:4])
    return "ls"


def _sanitize_unmatched_quotes(
    cmd: str, messages: list[dict] | None
) -> str:
    paths = _paths_in_bash_command(cmd)
    for piece in re.split(r"[\n;|&]+", cmd):
        s = piece.strip()
        if not s or _unmatched_quotes(s):
            continue
        if _is_pathless_head(s):
            continue
        if s.startswith(_SAFE_BASH_STARTS):
            return s
    if paths:
        return _safe_head_command(paths, messages)
    for match in re.finditer(
        r"""['"]((?:[\w./~-]+/)+)\*[^'"]*['"]|(?:^|[\s\"'])((?:[\w./~-]+/)+)""",
        cmd,
    ):
        directory = (match.group(1) or match.group(2) or "").rstrip("/")
        if directory and directory not in {".", ".."}:
            return "ls " + directory
    return "ls"


def _sanitize_bash_command(command: str, messages: list[dict] | None = None) -> str:
    """Fix *broken* bash. Keep real work (cd && script, which, pipes, env=)."""
    cmd = (command or "").strip()
    if not cmd:
        return "ls"
    rewritten = _rewrite_missing_path_command(cmd, messages)
    if rewritten:
        cmd = rewritten.strip()
    ls_dir = _ls_target_dir(cmd)
    if ls_dir and _is_invented_workspace_path(ls_dir, messages):
        return "ls"
    if cmd in {"ls", "pwd", "date", "uname", "whoami"}:
        return cmd
    if _is_pathless_head(cmd):
        return "ls"
    if _unmatched_quotes(cmd):
        return _sanitize_unmatched_quotes(cmd, messages)
    if "\n" in cmd:
        first = next((ln.strip() for ln in cmd.splitlines() if ln.strip()), "")
        if first and first != cmd:
            return _sanitize_bash_command(first, messages)
    if _CMD_SUB_RE.search(cmd):
        paths = _paths_in_bash_command(cmd)
        if paths:
            return _safe_head_command(paths, messages)
        return "ls"
    paths = _paths_in_bash_command(cmd)
    if paths and messages and re.match(r"^(?:cat|head|tail|less|grep|rg)\b", cmd):
        known = _recent_tool_result_paths(messages)
        if known and not any(
            _resolve_read_path(_normalize_fs_path(p, messages), messages) for p in paths
        ):
            return _safe_head_command(paths, messages)
    return cmd


def _is_early_stop_plan(text: str, has_tools: bool) -> bool:
    if has_tools:
        return False
    blob = (text or "").strip()
    if len(blob) < 24:
        return False
    return bool(_EARLY_STOP_PLAN_RE.search(blob))


def _last_tool_names_since_user(messages: list[dict] | None) -> list[str]:
    """Tool names of the most recent assistant tool_calls turn since the last
    real user message; [] when the last action was not a tool call."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and not _is_harness_user(msg):
            return []
        if role != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            return []
        names: list[str] = []
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = (fn or {}).get("name") if isinstance(fn, dict) else None
            if name:
                names.append(str(name))
        return names
    return []


def _unverified_edit(messages: list[dict] | None) -> bool:
    """True when the most recent action in this user turn was a file edit or
    write — nothing has been run or read back since, so a final answer now
    is unverified."""
    names = _last_tool_names_since_user(messages)
    return bool(names) and all(n in _EDIT_TOOL_NAMES for n in names)


def _sse_text_and_tools(raw: bytes) -> tuple[str, bool]:
    parts: list[str] = []
    saw_tool = False
    for event in _iter_sse_json(raw):
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("tool_calls"):
                saw_tool = True
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return "".join(parts), saw_tool


def _json_text_and_tools(payload: bytes) -> tuple[str, bool]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", False
    if not isinstance(data, dict):
        return "", False
    parts: list[str] = []
    saw_tool = False
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            continue
        if msg.get("tool_calls"):
            saw_tool = True
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return "".join(parts), saw_tool


def _payload_tool_calls(raw: bytes, *, sse: bool) -> list[dict]:
    """Assembled tool calls [{name, arguments}] from an SSE or JSON payload."""
    out: list[dict] = []
    if sse:
        assembled: dict[int, dict] = {}
        for event in _iter_sse_json(raw):
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    idx = int(tc.get("index") or 0)
                    slot = assembled.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    if fn.get("name"):
                        slot["name"] = str(fn["name"])
                    if fn.get("arguments"):
                        slot["arguments"] += str(fn["arguments"])
        return [assembled[i] for i in sorted(assembled)]
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for choice in data.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        for tc in (msg or {}).get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            out.append(
                {"name": str(fn.get("name") or ""), "arguments": str(fn.get("arguments") or "")}
            )
    return out


def _tool_calls_sig(tcs: list[dict]) -> str:
    """Same normalisation as the tools half of _assistant_signature."""
    sig: list[str] = []
    for tc in tcs:
        args: Any = tc.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass
        if isinstance(args, dict):
            args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        sig.append(f"{tc.get('name', '')}({args})")
    return "|".join(sorted(sig))


def _last_assistant_tool_sig(messages: list[dict] | None) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            full = _assistant_signature(msg) or ""
            return full.split("#", 1)[0]
    return ""


def _last_tool_result_head(messages: list[dict] | None, n: int = 400) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") in ("tool", "function"):
            return _message_text(msg)[:n]
    return ""


def _payload_reasoning_chars(raw: bytes, *, sse: bool) -> int:
    n = 0
    if sse:
        for event in _iter_sse_json(raw):
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or choice.get("message") or {}
                if not isinstance(delta, dict):
                    continue
                for key in ("reasoning_content", "reasoning"):
                    v = delta.get(key)
                    if isinstance(v, str):
                        n += len(v)
        return n
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    for choice in (data.get("choices") or []) if isinstance(data, dict) else []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        for key in ("reasoning_content", "reasoning"):
            v = (msg or {}).get(key)
            if isinstance(v, str):
                n += len(v)
    return n


def _payload_finish_reason(raw: bytes, *, sse: bool) -> str:
    reason = ""
    if sse:
        for event in _iter_sse_json(raw):
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    reason = str(choice["finish_reason"])
        return reason
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    for choice in (data.get("choices") or []) if isinstance(data, dict) else []:
        if isinstance(choice, dict) and choice.get("finish_reason"):
            reason = str(choice["finish_reason"])
    return reason


# Live rollout trace (AutoSaddler: diagnose traces, not symptoms). One JSON
# line per model response on agent turns: what the model was shown last
# (tool result head), what it produced (reasoning size, content head, tool
# calls), and which middleware fired. Read with:
#   tail -n 20 logs/live-trace.jsonl | python3 -m json.tool
LIVE_TRACE_FILE = Path(__file__).resolve().parent / "logs" / "live-trace.jsonl"
_LIVE_TRACE_LOCK = threading.Lock()


def _write_live_trace(record: dict) -> None:
    try:
        LIVE_TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _LIVE_TRACE_LOCK, LIVE_TRACE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        log.warning("[trace] live trace write failed: %s", exc)


_DUPLICATE_RETRY_USER = (
    "[Harness] REPEATED ACTION: you just issued the SAME tool call as your "
    "previous turn ({what}). It already ran; its result is in the "
    "conversation above and begins:\n---\n{head}\n---\n"
    "Do not run it again. Use that result: take the next DIFFERENT step "
    "toward the goal now (tool_calls), or if the goal is met give the "
    "final conclusion."
)


def _apply_duplicate_retry(body: dict, what: str, head: str) -> None:
    """Response-side duplicate: ask once for a different step, with the
    result the model apparently did not read quoted back at it."""
    body["tool_choice"] = "auto"
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    text = _DUPLICATE_RETRY_USER.format(what=what[:160], head=head.strip()[:400] or "(empty)")
    msgs.append({"role": "user", "content": text})


def _should_force_continue(
    text: str, has_tools: bool, messages: list[dict] | None
) -> bool:
    if has_tools:
        return False
    rounds = _tool_rounds_since_user(messages)
    max_force = int(
        _active_harness().get("early_stop_force_max") or EARLY_STOP_FORCE_MAX
    )
    if rounds > max_force:
        return False
    if _is_early_stop_plan(text, False):
        return True
    if _active_harness().get("mw_verify_gate", True) and _unverified_edit(messages):
        return True
    return not (text or "").strip()


def _command_from_plan(text: str) -> str | None:
    blob = text or ""
    m = re.search(r"ONEPHONE_I_CONFIRM=1\s+\./tools/[\w.-]+\.sh", blob)
    if m:
        return m.group(0)
    m = _PLAN_SCRIPT_RE.search(blob)
    if m:
        cmd = m.group(0).strip()
        if not cmd.startswith("./") and cmd.startswith("tools/"):
            cmd = "./" + cmd
        return cmd
    return None


def _make_tool_call(name: str, args: dict) -> dict:
    # Unique per call: two synthetic calls in one session must not share a
    # tool_call_id, or the client can pair a result with the wrong call.
    return {
        "id": f"call_saddle_{name}_{int(time.time() * 1000) % 100_000_000:x}",
        "type": "function",
        "index": 0,
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _recent_bash_commands(messages: list[dict] | None) -> set[str]:
    found: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            if not isinstance(fn, dict):
                continue
            raw = fn.get("arguments") or "{}"
            parsed: dict | None = None
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = None
            if parsed and parsed.get("command"):
                found.add(str(parsed["command"]).strip())
    return found


# Prose tool calls (2026-09-02, live trace): the model answered a research
# turn with `<bash>\n\nls -la …; ls … | head -60\n\n` as plain CONTENT —
# an XML-shaped tool call that never became tool_calls (mtplx's parser only
# knows the Qwen <tool_call>{json}</tool_call> form). The early-stop regex
# did not match, so the text passed as a final answer and Kilo ended the turn.
# Capability patch: recognise the shapes the model actually emits and turn
# them into real tool_calls before the client sees them. No retry, no
# extra engine round.
_TOOL_TAG_RE = re.compile(r"(?s)<([A-Za-z_][\w-]*)>\s*(.*?)\s*(?:</\1>|$)")
_TOOL_CALL_JSON_RE = re.compile(r"(?s)<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)")
_FENCE_ONLY_RE = re.compile(r"(?s)^\s*```(?:bash|sh|shell|zsh)?\s*\n(.*?)\n?```\s*$")
# The bracketed prose form the LOOP_PREFIX warns against and the model still
# emits: "[Tool calls: bash({json})]" / "[Calling tool: read({json})]".
# 2026-09-02 live trace round 12 ended a turn this way (finish=stop).
_BRACKET_TOOL_RE = re.compile(
    r"(?is)\[\s*(?:tool\s*calls?|calling\s*tool)\s*[:\-]?\s*"
    r"([A-Za-z_][\w-]*)\s*\(\s*(\{.*?\})\s*\)\s*\]"
)
_INNER_KV_RE = re.compile(r"(?s)<([A-Za-z_][\w-]*)>\s*(.*?)\s*</\1>")


def _request_tool_schemas(body: dict | None) -> dict[str, dict]:
    """name -> parameters schema for the tools declared on the request."""
    out: dict[str, dict] = {}
    for tool in (body or {}).get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name") or "")
        if name:
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            out[name] = params or {}
    return out


def _primary_arg_name(schema: dict, fallback: str) -> str:
    req = schema.get("required")
    if isinstance(req, list) and req and isinstance(req[0], str):
        return req[0]
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        return next(iter(props))
    return fallback


def _pseudo_tool_calls(text: str, body: dict | None) -> list[dict]:
    """Tool calls the model wrote as prose, as real tool_calls (or [])."""
    blob = (text or "").strip()
    if not blob or len(blob) > 6000:
        return []
    schemas = _request_tool_schemas(body)
    if not schemas:
        return []
    # 1. Qwen JSON form that the engine parser missed (e.g. unclosed tag).
    m = _TOOL_CALL_JSON_RE.search(blob)
    if m:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and str(obj.get("name") or "") in schemas:
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            return [_make_tool_call(str(obj["name"]), args)]
    # 2. <toolname> … </toolname> (Kilo/Cline XML style), possibly unclosed.
    m = _TOOL_TAG_RE.search(blob)
    if m and m.group(1) in schemas:
        name = m.group(1)
        inner = m.group(2).strip()
        schema = schemas[name]
        kv = {k: v.strip() for k, v in _INNER_KV_RE.findall(inner)}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if kv and any(k in props for k in kv):
            args: dict[str, Any] = {}
            for k, v in kv.items():
                if k not in props:
                    continue
                typ = (props.get(k) or {}).get("type") if isinstance(props.get(k), dict) else None
                if typ == "integer":
                    try:
                        args[k] = int(v)
                        continue
                    except ValueError:
                        pass
                if typ == "boolean":
                    args[k] = v.lower() in ("1", "true", "yes")
                    continue
                args[k] = v
            if args:
                return [_make_tool_call(name, args)]
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                return [_make_tool_call(name, obj)]
        # Bare inner text -> primary arg. Only when the tag actually looks
        # like an invocation: properly closed, or opening the reply (the
        # unclosed live shape "<bash>\n\nls -la ...", 2026-09-02). A tag named
        # mid-sentence is prose -- "I'll use the <bash> tool to list files"
        # would otherwise become bash({command: "tool to list files"}).
        closed = m.group(0).rstrip().endswith(f"</{name}>")
        if inner and not kv and (closed or m.start() == 0):
            key = _primary_arg_name(schema, "command" if name == "bash" else "input")
            return [_make_tool_call(name, {key: inner})]
    # 3. "[Tool calls: name({json})]" / "[Calling tool: name({json})]".
    m = _BRACKET_TOOL_RE.search(blob)
    if m and m.group(1) in schemas:
        try:
            obj = json.loads(m.group(2))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return [_make_tool_call(m.group(1), obj)]
    # 4. A reply that is nothing but a shell code fence.
    if "bash" in schemas:
        m = _FENCE_ONLY_RE.match(blob)
        if m and m.group(1).strip():
            return [_make_tool_call("bash", {"command": m.group(1).strip()})]
    return []


def _synthetic_tool_calls(
    text: str, messages: list[dict] | None
) -> list[dict]:
    cmd = _command_from_plan(text)
    if not cmd:
        return []
    cmd = _sanitize_bash_command(cmd, messages)
    if cmd in _recent_bash_commands(messages):
        log.info("[agent] skip duplicate synthetic %s", cmd[:80])
        return []
    return [_make_tool_call("bash", {"command": cmd})]


def _payload_ids(raw: bytes, *, sse: bool) -> tuple[str, str]:
    model = "qwen3.8-27b-obliterated-mtplx"
    response_id = "chatcmpl-saddle"
    if sse:
        for event in _iter_sse_json(raw):
            if not isinstance(event, dict):
                continue
            model = str(event.get("model") or model)
            response_id = str(event.get("id") or response_id)
        return model, response_id
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return model, response_id
    if isinstance(data, dict):
        model = str(data.get("model") or model)
        response_id = str(data.get("id") or response_id)
    return model, response_id


def _sse_tool_payload(tcs: list[dict], model: str, response_id: str) -> bytes:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "tool_calls": tcs},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _sse_text_payload(text: str, model: str, response_id: str) -> bytes:
    """A complete SSE body carrying one final assistant text (finish=stop)."""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _json_text_payload(text: str, model: str, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _is_chat_path(path: str) -> bool:
    p = (path or "").rstrip("/")
    return p.endswith("/chat/completions") or p.endswith("/messages")


def _drop_session_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _SESSION_HEADER_DROP
    }


def _scrub_session_body(data: dict) -> bool:
    changed = False
    for key in _SESSION_BODY_DROP:
        if key in data:
            data.pop(key, None)
            changed = True
    meta = data.get("metadata")
    if isinstance(meta, dict):
        for key in _SESSION_BODY_DROP:
            if key in meta:
                meta.pop(key, None)
                changed = True
    return changed


def _is_in_flight_error(status: int, payload: bytes) -> bool:
    if status not in {409, 429, 500, 503}:
        return False
    try:
        blob = payload.decode("utf-8", errors="replace")
    except Exception:
        return False
    return bool(_IN_FLIGHT_RE.search(blob))


def _json_tool_payload(tcs: list[dict], model: str, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tcs,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _apply_early_stop_retry(body: dict) -> None:
    """mtplx accepts tool_choice auto|none only — nudge via a user turn.

    Picks the VERIFY GATE wording when the stop followed an unverified
    edit/write, otherwise the generic EARLY STOP wording. One retry per turn
    either way: the system marker and the user nudge are both idempotent.
    """
    body["tool_choice"] = "auto"
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    verify = _unverified_edit(msgs)
    marker = "[Harness] VERIFY GATE:" if verify else "[Harness] EARLY STOP:"
    nudge = _VERIFY_GATE_NUDGE if verify else _EARLY_STOP_NUDGE
    user_text = _VERIFY_RETRY_USER if verify else _RETRY_USER
    _nudge_system(
        msgs,
        marker,
        nudge,
        "verify-gate retry" if verify else "early-stop plan retry",
    )
    if any(
        isinstance(m, dict)
        and m.get("role") == "user"
        and ("[Harness] EARLY STOP:" in _message_text(m) or "[Harness] VERIFY GATE:" in _message_text(m))
        for m in msgs
    ):
        return
    msgs.append({"role": "user", "content": user_text})


def _path_from_args(args: dict) -> str:
    for key in ("path", "filePath", "file_path", "target_file"):
        val = args.get(key)
        if val:
            return str(val)
    return ""


def _repair_one_tool_call(
    name: str,
    args: dict,
    messages: list[dict] | None,
) -> dict:
    lname = (name or "").lower().replace("-", "_")
    if lname in {"bash", "shell", "execute_command"}:
        raw_cmd = str(args.get("command") or "")
        if raw_cmd:
            safe = _sanitize_bash_command(raw_cmd, messages)
            ls_dir = _ls_target_dir(safe) or _ls_target_dir(raw_cmd)
            if ls_dir and _is_invented_workspace_path(ls_dir, messages):
                safe = "ls"
            if safe != raw_cmd:
                log.info("[bash-sanitize] %r → %r", raw_cmd[:80], safe[:80])
                return {**args, "command": safe}
        return args
    if lname in {"read", "readfile", "read_file", "fs_read"}:
        path = _path_from_args(args)
        resolved = _resolve_read_path(path, messages) or _normalize_fs_path(path, messages)
        if _workspace_entry_exists(resolved or path, messages) is False:
            sibling = _analysis_dir_markdown(path, messages)
            resolved = sibling or "README.md"
        if resolved and resolved != path:
            log.info("[read-resolve] %r → %s", path, resolved)
            field = "path" if "path" in args else next(
                (k for k in ("filePath", "file_path", "target_file") if k in args),
                "path",
            )
            return {**args, field: resolved}
    if lname in _GREP_TOOL_NAMES:
        fixed = _repair_grep_args(args)
        if fixed != args:
            dropped = sorted(set(args) - set(fixed))
            log.info("[grep-repair] keys %s → %s", dropped, sorted(fixed))
            return fixed
    if lname in {"question", "ask_question", "ask_followup_question"}:
        fixed = _repair_question_args(args)
        if fixed != args:
            log.info(
                "[question-repair] %d question(s), keys %s → schema",
                len(fixed.get("questions") or []),
                sorted(args),
            )
            return fixed
    return args


def _convert_overflow_grep(
    name: str, args: dict, messages: list[dict] | None
) -> tuple[str, dict] | None:
    """After a Kilo grep died on a >64KB line, re-issue the same search as a
    guarded bash ``rg`` (excludes node_modules/dist/lockfiles, caps columns)
    instead of letting the model repeat the failing grep."""
    lname = (name or "").lower().replace("-", "_")
    if lname not in _GREP_TOOL_NAMES:
        return None
    if not _last_tool_was_grep_overflow(messages):
        return None
    cmd = _grep_to_rg_command(args)
    if not cmd:
        return None
    log.info("[grep-overflow] grep → bash %r", cmd[:120])
    return "bash", {"command": cmd, "description": "rg search (grep overflow guard)"}


def _repair_tool_calls_list(
    tcs: list, messages: list[dict] | None, *, response: bool = True
) -> bool:
    """Repair tool-call arguments in place. ``response=False`` for history
    rewrites: only argument fixes apply there, never tool renames."""
    changed = False
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        parsed: dict | None = None
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
                if isinstance(loaded, dict):
                    parsed = loaded
            except json.JSONDecodeError:
                fixed = _try_close_json(raw_args)
                if fixed:
                    try:
                        loaded = json.loads(fixed)
                        if isinstance(loaded, dict):
                            parsed = loaded
                            fn["arguments"] = fixed
                            changed = True
                    except json.JSONDecodeError:
                        parsed = None
        if not parsed:
            continue
        repaired = _repair_one_tool_call(name, parsed, messages)
        if repaired != parsed:
            fn["arguments"] = json.dumps(repaired, ensure_ascii=False)
            changed = True
        converted = _convert_overflow_grep(name, repaired, messages) if response else None
        if converted:
            fn["name"], new_args = converted
            fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
            changed = True
    return changed


def _repair_history_tool_calls(messages: list[dict]) -> None:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                _repair_tool_calls_list(tcs, messages, response=False)


def _try_close_json(s: str) -> str | None:
    s = s.strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    if s.count('"') % 2 == 1:
        s = s + '"'
    s = re.sub(r",\s*$", "", s)
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
            closes_needed_obj += 1
        elif ch == "}":
            closes_needed_obj = max(0, closes_needed_obj - 1)
        elif ch == "[":
            closes_needed_arr += 1
        elif ch == "]":
            closes_needed_arr = max(0, closes_needed_arr - 1)
    candidate = s + ("]" * closes_needed_arr) + ("}" * closes_needed_obj)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def _repair_tool_calls_in_message(
    msg: dict, messages: list[dict] | None = None
) -> bool:
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list):
        return False
    return _repair_tool_calls_list(tcs, messages)


def _repair_response_payload(
    payload: bytes, messages: list[dict] | None = None
) -> bytes:
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
        if isinstance(msg, dict) and _repair_tool_calls_in_message(msg, messages):
            changed = True
    if not changed:
        return payload
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def _iter_sse_json(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _repair_sse_payload(raw: bytes, messages: list[dict] | None) -> bytes:
    assembled: dict[int, dict] = {}
    model = ""
    response_id = "chatcmpl-saddle"
    saw_tool = False
    for event in _iter_sse_json(raw):
        if not isinstance(event, dict):
            continue
        model = str(event.get("model") or model)
        response_id = str(event.get("id") or response_id)
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                saw_tool = True
                idx = int(tc.get("index") or 0)
                slot = assembled.setdefault(
                    idx,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += str(fn["arguments"])
    if not saw_tool:
        return raw
    tcs = [assembled[i] for i in sorted(assembled)]
    if not _repair_tool_calls_list(tcs, messages):
        return raw
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "tool_calls": tcs},
                "finish_reason": None,
            }
        ],
    }
    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    return (
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


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
        return HTTPSConnection(state.host, state.port, timeout=UPSTREAM_TIMEOUT)
    return HTTPConnection(state.host, state.port, timeout=UPSTREAM_TIMEOUT)


def _early_stop_armed(steer_trace: dict[str, Any]) -> bool:
    """Early-stop middleware applies to nudged tool rounds and the first turn.

    Not when the repeat guard already took the tools away: there is nothing
    to force a tool_call with, the model is meant to answer in text.
    """
    return bool(
        (steer_trace.get("after_tool_continue") or steer_trace.get("first_turn_guard"))
        and not steer_trace.get("repeat_no_tools")
        and not steer_trace.get("cycle_stale_no_tools")
        and _active_harness().get("mw_early_stop", True)
    )


def _needs_buffered_sse(steer_trace: dict[str, Any]) -> bool:
    return _early_stop_armed(steer_trace)


class _ClientGone(Exception):
    """The Kilo side closed its socket; stop generating for it."""


def _sse_error_payload(status: int, detail: bytes | str) -> bytes:
    """Wrap an upstream/proxy failure as a terminal SSE event.

    Once we have committed to ``text/event-stream`` we cannot change the HTTP
    status, so the OpenAI-style error object is delivered in-band. Every
    OpenAI-compatible client surfaces ``{"error": ...}`` chunks as a failure.
    """
    text = detail.decode("utf-8", "replace") if isinstance(detail, bytes) else detail
    message = text.strip() or f"upstream HTTP {status}"
    stripped = text.lstrip()
    if stripped.startswith("data:") or '"chat.completion.chunk"' in stripped[:600]:
        # A partial SSE body (the engine died or restarted mid-stream). Kilo
        # would otherwise show 2 KB of raw chunk JSON as the error text.
        message = (
            f"upstream stream aborted mid-response (HTTP {status}); the engine "
            "probably restarted — retry the request"
        )
    else:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                message = str(parsed["error"].get("message") or message)
        except (ValueError, TypeError):
            pass
    obj = {
        "error": {
            "message": message[:600],
            "type": "proxy_error",
            "code": int(status),
        }
    }
    return (
        f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8") + _SSE_DONE
    )


def _sse_at_event_boundary(buf: bytes) -> bool:
    """True when it is safe to interleave an SSE comment line."""
    return not buf or buf.endswith(b"\n\n") or buf.endswith(b"\r\n\r\n")


class _UpstreamFetch:
    """One upstream request, read in a worker thread.

    The handler thread never blocks on the engine socket. It polls the chunk
    queue with a short timeout so it can emit keepalives to the client (and
    notice when the client is gone) even while mtplx is still prefilling.
    """

    def __init__(
        self,
        state: ProxyState,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.content_type = ""
        self.error: BaseException | None = None
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._ready = threading.Event()
        self._aborted = False
        self._conn: HTTPConnection | None = None
        self._thread = threading.Thread(
            target=self._run,
            args=(state, method, path, body, headers),
            name="upstream-fetch",
            daemon=True,
        )
        self._thread.start()

    def _run(
        self,
        state: ProxyState,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        conn: HTTPConnection | None = None
        try:
            conn = _connect(state)
            self._conn = conn
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            self.status = resp.status
            self.headers = {k: v for k, v in resp.getheaders()}
            # uvicorn/mtplx send lowercase names; http.client.getheader is
            # case-insensitive. A dict .get("Content-Type") misses them and
            # the passthrough path then wraps a good SSE stream as
            # {"error": ...} — Kilo reports that as "cannot connect to API".
            self.content_type = resp.getheader("Content-Type") or ""
            self._ready.set()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self._q.put(chunk)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the handler
            if not self._aborted:
                self.error = exc
        finally:
            self._ready.set()
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._q.put(None)

    def abort(self) -> None:
        """Tear down the engine socket so mtplx cancels the generation."""
        self._aborted = True
        conn = self._conn
        if conn is None:
            return
        sock = getattr(conn, "sock", None)
        try:
            if sock is not None:
                sock.shutdown(2)  # SHUT_RDWR
        except OSError:
            pass
        try:
            conn.close()
        except Exception:
            pass

    def wait_ready(self, on_tick, tick: float = SSE_KEEPALIVE_SECS) -> None:
        while not self._ready.wait(tick):
            on_tick()
        if self.error is not None:
            raise self.error

    def iter_chunks(self, on_tick, tick: float = SSE_KEEPALIVE_SECS):
        while True:
            try:
                chunk = self._q.get(timeout=tick)
            except queue.Empty:
                on_tick()
                continue
            if chunk is None:
                if self.error is not None:
                    raise self.error
                return
            yield chunk

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)


def _noop() -> None:
    return None


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
            self._send_json(
                200,
                {
                    "ok": True,
                    "upstream": self.state.upstream,
                    "sampling": {
                        "temperature": CARD_TEMPERATURE,
                        "top_p": CARD_TOP_P,
                        "frequency_penalty": CARD_FREQUENCY_PENALTY,
                        "enable_thinking": THINKING,
                        "thinking_scope": "tool turns only" if THINKING else "off",
                        "reasoning_effort": REASONING_EFFORT if THINKING else None,
                        "thinking_budget": THINKING_BUDGET if THINKING else 0,
                    },
                    "english_only": ENGLISH_GUARD,
                    "limits": {
                        "max_tokens_cap": AGENT_MAX_TOKENS_CAP,
                        "upstream_timeout": UPSTREAM_TIMEOUT,
                        "lock_wait_timeout": LOCK_WAIT_TIMEOUT,
                        "sse_keepalive_secs": SSE_KEEPALIVE_SECS,
                    },
                    "engine_busy": _UPSTREAM_GEN_LOCK_BUSY.is_set(),
                },
            )
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

    # ---- client-side helpers -------------------------------------------

    def _client_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise _ClientGone(str(exc)) from exc

    def _sse_begin(self, extra_headers: dict[str, str] | None = None) -> None:
        """Commit to a 200 text/event-stream response right away.

        Kilo's ``headerTimeout`` starts at request time; the engine can take
        minutes to prefill a long history, so headers go out before we know
        anything about the upstream result.
        """
        self.send_response(200)
        for k, v in (extra_headers or {}).items():
            if k.lower() in (
                "transfer-encoding",
                "connection",
                "content-length",
                "content-encoding",
                "content-type",
            ):
                continue
            self.send_header(k, v)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        try:
            self.end_headers()
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise _ClientGone(str(exc)) from exc
        self._sse_sent = b""

    def _client_closed(self) -> bool:
        """Passive EOF probe: True once Kilo has closed its side.

        The request body is fully consumed before we get here, so any readable
        state on the socket is either EOF (b"") or a reset. Works even while an
        SSE event is half-sent, when a keepalive write is not allowed.
        """
        sock = getattr(self, "connection", None)
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return False
            return sock.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _sse_tick(self) -> None:
        """Keepalive between events only (never inside a partial event)."""
        if self._client_closed():
            raise _ClientGone("client closed socket")
        if _sse_at_event_boundary(getattr(self, "_sse_sent", b"")):
            self._client_write(_SSE_KEEPALIVE)

    def _sse_send(self, data: bytes) -> None:
        self._client_write(data)
        # Remember only the tail; we just need to know if an event is open.
        self._sse_sent = data[-4:]

    def _sse_fail(self, status: int, detail: bytes | str) -> None:
        log.info("[agent] sse in-band error %s", status)
        self._sse_send(_sse_error_payload(status, detail))

    # ---- engine lock -----------------------------------------------------

    def _acquire_gen_lock(self, on_tick) -> bool:
        """Serialize engine use; keep the client alive while queued."""
        deadline = time.monotonic() + LOCK_WAIT_TIMEOUT
        waited = False
        while True:
            if _UPSTREAM_GEN_LOCK.acquire(timeout=SSE_KEEPALIVE_SECS):
                _UPSTREAM_GEN_LOCK_BUSY.set()
                if waited:
                    log.info("[agent] engine lock acquired after wait")
                return True
            if not waited:
                log.info("[agent] engine busy; queuing request")
                waited = True
            on_tick()
            if time.monotonic() >= deadline:
                log.info("[agent] engine lock wait exceeded %ss", LOCK_WAIT_TIMEOUT)
                return False

    @staticmethod
    def _release_gen_lock() -> None:
        _UPSTREAM_GEN_LOCK_BUSY.clear()
        try:
            _UPSTREAM_GEN_LOCK.release()
        except RuntimeError:
            pass

    # ---- upstream --------------------------------------------------------

    def _fetch_buffered(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        on_tick=_noop,
    ) -> tuple[int, dict[str, str], str, bytes]:
        """Full upstream response, retrying mtplx 409 session-in-flight.

        Caller holds the generation lock. ``on_tick`` runs every few seconds
        while we wait so a streaming client can be kept alive / detected gone.
        """
        chat = _is_chat_path(path)
        attempts = _IN_FLIGHT_RETRIES if chat else 1
        last: tuple[int, dict[str, str], str, bytes] | None = None
        for attempt in range(attempts):
            fetch = _UpstreamFetch(self.state, self.command, path, body, headers)
            try:
                fetch.wait_ready(on_tick)
                chunks = list(fetch.iter_chunks(on_tick))
            except _ClientGone:
                fetch.abort()
                fetch.join()
                raise
            finally:
                fetch.join()
            last = (
                int(fetch.status or 502),
                fetch.headers,
                fetch.content_type,
                b"".join(chunks),
            )
            if chat and _is_in_flight_error(last[0], last[3]):
                log.info(
                    "[agent] session already in flight (try %s/%s)",
                    attempt + 1,
                    attempts,
                )
                # Back off, but keep the client alive while doing so.
                end = time.monotonic() + 0.4 * (attempt + 1)
                while time.monotonic() < end:
                    time.sleep(min(0.2, max(0.0, end - time.monotonic())))
                on_tick()
                continue
            return last
        assert last is not None
        return last

    def _stream_passthrough(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        parsed: dict | None = None,
        steer_trace: dict[str, Any] | None = None,
    ) -> None:
        """Forward SSE as it arrives; headers + keepalives go out immediately.

        A reply that STARTS in CJK is aborted (the cancel window is held
        back, nothing forwarded) and regenerated once with the English-only
        nudge.
        """
        chat = _is_chat_path(path)
        self._sse_begin()
        if chat and not self._acquire_gen_lock(self._sse_tick):
            self._sse_fail(503, "engine busy: another generation is still running")
            return
        fetch: _UpstreamFetch | None = None
        try:
            for attempt in range(2):
                fetch = _UpstreamFetch(self.state, self.command, path, body, headers)
                fetch.wait_ready(self._sse_tick)
                status = int(fetch.status or 502)
                if status != 200 or "text/event-stream" not in fetch.content_type:
                    payload = b"".join(fetch.iter_chunks(self._sse_tick))
                    if status == 200 and "application/json" in fetch.content_type:
                        # Upstream ignored stream=true; hand the JSON over as-is
                        # inside an SSE event so the client still gets an answer.
                        payload = _repair_response_payload(payload, None)
                        self._sse_send(b"data: " + payload + b"\n\n" + _SSE_DONE)
                        return
                    self._sse_fail(status, payload)
                    return
                verdict = self._relay_sse_with_cjk_guard(
                    fetch,
                    allow_retry=(
                        attempt == 0
                        and ENGLISH_GUARD
                        and chat
                        and parsed is not None
                        and not (steer_trace or {}).get("lang_retried")
                    ),
                )
                if verdict != "retry":
                    return
                log.info("[agent] CJK stream start: regenerating in English")
                fetch.abort()
                fetch.join()
                fetch = None
                _apply_language_retry(parsed)  # type: ignore[arg-type]
                if steer_trace is not None:
                    steer_trace["lang_retried"] = True
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
        except _ClientGone:
            log.info("client disconnected mid-stream; aborting upstream")
            if fetch is not None:
                fetch.abort()
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error (stream): %s", exc)
            try:
                self._sse_fail(502, str(exc))
            except _ClientGone:
                pass
        finally:
            if fetch is not None:
                fetch.join()
            if chat:
                self._release_gen_lock()

    def _relay_sse_with_cjk_guard(
        self, fetch: _UpstreamFetch, *, allow_retry: bool
    ) -> str:
        """Stream upstream SSE to the client, event by event.

        While the answer is still inside the cancel window (no tool_calls
        seen, under ``CJK_CANCEL_WINDOW_CHARS`` of content) events are held
        back. If the text is CJK there, return "retry" — the client has only
        seen keepalives, nothing to retract. Otherwise flush the held events
        and stream the rest straight through. Returns "done".
        """
        pending = b""
        held = b""
        acc = ""
        saw_tools = False
        decided = not allow_retry
        for chunk in fetch.iter_chunks(self._sse_tick):
            pending += chunk
            events, pending = _split_sse_events(pending)
            for event in events:
                if decided:
                    self._sse_send(event)
                    continue
                text, tools = _sse_text_and_tools(event)
                saw_tools = saw_tools or tools
                acc += text
                held += event
                if saw_tools or len(acc) >= CJK_CANCEL_WINDOW_CHARS:
                    decided = True
                    if not saw_tools and _is_mostly_cjk(acc):
                        return "retry"
                    self._sse_send(held)
                    held = b""
        if not decided:
            if _is_mostly_cjk(acc):
                return "retry"
            if held:
                self._sse_send(held)
        if pending:
            self._sse_send(pending)
        if not getattr(self, "_sse_sent", b"").endswith(b"\n\n"):
            self._sse_send(b"\n\n")
        return "done"

    # ---- middleware decision -------------------------------------------

    def _post_process(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
        parsed: dict | None,
        steer_messages: list[dict] | None,
        steer_trace: dict[str, Any],
        stream: bool,
        on_tick,
    ) -> tuple[int, dict[str, str], str, bytes]:
        """Early-stop / verify-gate retry, then duplicate-call retry, then trace."""
        t0 = time.monotonic()
        status, resp_headers, content_type, payload = self._post_process_inner(
            path=path,
            headers=headers,
            body=body,
            parsed=parsed,
            steer_messages=steer_messages,
            steer_trace=steer_trace,
            stream=stream,
            on_tick=on_tick,
        )
        if status != 200 or parsed is None or not isinstance(parsed, dict):
            return status, resp_headers, content_type, payload
        is_sse = stream or "text/event-stream" in content_type
        if not (is_sse or "application/json" in content_type):
            return status, resp_headers, content_type, payload
        tcs = _payload_tool_calls(payload, sse=is_sse)
        # Duplicate tool call: identical to the previous assistant turn. Kilo
        # would run it again for nothing and the request-side repeat guard
        # would count one more toward taking the tools away. Ask once for a
        # different step, quoting the result the model evidently skipped.
        if (
            tcs
            and parsed.get("tools")
            and _active_harness().get("mw_duplicate_retry", True)
            and not steer_trace.get("dup_retried")
            and not steer_trace.get("repeat_no_tools")
            and not steer_trace.get("cycle_stale_no_tools")
        ):
            sig = _tool_calls_sig(tcs)
            if sig and sig == _last_assistant_tool_sig(steer_messages):
                steer_trace["dup_retried"] = True
                what = _repeat_summary(sig)
                log.info("[agent] duplicate tool call: retrying once (%s)", what)
                _apply_duplicate_retry(
                    parsed, what, _last_tool_result_head(steer_messages)
                )
                body2 = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body2))
                st2, rh2, ct2, pl2 = self._fetch_buffered(path, body2, headers, on_tick)
                if st2 == 200:
                    is_sse2 = stream or "text/event-stream" in ct2
                    if is_sse2:
                        pl2 = _repair_sse_payload(pl2, steer_messages)
                    elif "application/json" in ct2:
                        pl2 = _repair_response_payload(pl2, steer_messages)
                    tcs2 = _payload_tool_calls(pl2, sse=is_sse2)
                    if tcs2 and _tool_calls_sig(tcs2) == sig:
                        log.info("[agent] duplicate tool call: still identical, forwarding")
                    status, resp_headers, content_type, payload = st2, rh2, ct2, pl2
                    is_sse = is_sse2
                    tcs = tcs2
        # Cycle break: the model keeps re-issuing a command that already
        # recurred CYCLE_BREAK_MIN times (the nudge + xhigh escalation did not
        # work). Ban it: one hard retry, and if it STILL comes back, end the
        # turn with a visible message instead of letting Kilo spin forever.
        if (
            tcs
            and parsed.get("tools")
            and _active_harness().get("mw_cycle_break", True)
            and not steer_trace.get("cycle_broke")
        ):
            banned = _cycled_signatures(steer_messages, CYCLE_BREAK_MIN)
            sig = _tool_calls_sig(tcs)
            if sig and sig in banned:
                steer_trace["cycle_broke"] = True
                what = _repeat_summary(sig)
                head = _last_tool_result_head(steer_messages)
                log.info("[agent] cycle break: banning %s", what)
                _strip_nudge_prefix(parsed.get("messages") or [], "[Harness] BANNED COMMAND:")
                msgs2 = parsed.get("messages")
                if isinstance(msgs2, list):
                    msgs2.append({
                        "role": "user",
                        "content": _CYCLE_BREAK_USER.format(
                            what=what[:160], n=CYCLE_BREAK_MIN, head=(head.strip()[:400] or "(empty)")
                        ),
                    })
                parsed["tool_choice"] = "auto"
                body3 = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body3))
                st3, rh3, ct3, pl3 = self._fetch_buffered(path, body3, headers, on_tick)
                if st3 == 200:
                    is_sse3 = stream or "text/event-stream" in ct3
                    if is_sse3:
                        pl3 = _repair_sse_payload(pl3, steer_messages)
                    elif "application/json" in ct3:
                        pl3 = _repair_response_payload(pl3, steer_messages)
                    tcs3 = _payload_tool_calls(pl3, sse=is_sse3)
                    if tcs3 and _tool_calls_sig(tcs3) in banned:
                        # Still the banned command -> visible stop.
                        steer_trace["cycle_stop"] = True
                        model, rid = _payload_ids(pl3, sse=is_sse3)
                        stop_text = _CYCLE_STOP_TEXT.format(what=what[:160])
                        log.info("[agent] cycle break: model repeated banned cmd -> stopping turn")
                        payload = (
                            _sse_text_payload(stop_text, model, rid)
                            if is_sse3
                            else _json_text_payload(stop_text, model, rid)
                        )
                        return status, resp_headers, content_type, payload
                    log.info("[agent] cycle break: model took a different action")
                    status, resp_headers, content_type, payload = st3, rh3, ct3, pl3
                    is_sse = is_sse3
                    tcs = tcs3
        text, _has = (
            _sse_text_and_tools(payload) if is_sse else _json_text_and_tools(payload)
        )
        reasoning_chars = _payload_reasoning_chars(payload, sse=is_sse)
        finish = _payload_finish_reason(payload, sse=is_sse)
        log.info(
            "[out] %.1fs reasoning=%dch content=%dch tools=%s finish=%s%s",
            time.monotonic() - t0,
            reasoning_chars,
            len(text),
            [f"{tc['name']}({tc['arguments'][:80]})" for tc in tcs] or "-",
            finish or "?",
            (" dup_retry" if steer_trace.get("dup_retried") else "")
            + (" cycle_break" if steer_trace.get("cycle_broke") else "")
            + (" cycle_stop" if steer_trace.get("cycle_stop") else "")
            + (" xhigh" if steer_trace.get("cycle_escalate_effort") else ""),
        )
        _write_live_trace(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": round(time.monotonic() - t0, 1),
                "nmsg": len(steer_messages or []),
                "rounds_since_user": _tool_rounds_since_user(steer_messages),
                "last_tool_head": _last_tool_result_head(steer_messages, 600),
                "reasoning_chars": reasoning_chars,
                "content_head": text[:800],
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc["arguments"][:400]} for tc in tcs
                ],
                "finish_reason": finish,
                "cycle_max": steer_trace.get("cycle_max"),
                "flags": {
                    k: v
                    for k, v in steer_trace.items()
                    if v and k not in ("repeat_what",)
                },
                "repeat_what": steer_trace.get("repeat_what") or "",
            }
        )
        return status, resp_headers, content_type, payload

    def _post_process_inner(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
        parsed: dict | None,
        steer_messages: list[dict] | None,
        steer_trace: dict[str, Any],
        stream: bool,
        on_tick,
    ) -> tuple[int, dict[str, str], str, bytes]:
        """Fetch once, apply the early-stop retry / synthetic continue."""
        status, resp_headers, content_type, payload = self._fetch_buffered(
            path, body, headers, on_tick
        )
        if status != 200:
            return status, resp_headers, content_type, payload
        is_sse = stream or "text/event-stream" in content_type
        if (
            ENGLISH_GUARD
            and parsed is not None
            and _is_chat_path(path)
            and not steer_trace.get("lang_retried")
        ):
            if is_sse:
                text, has_tools = _sse_text_and_tools(payload)
            elif "application/json" in content_type:
                text, has_tools = _json_text_and_tools(payload)
            else:
                text, has_tools = "", True
            if not has_tools and _is_mostly_cjk(text):
                log.info(
                    "[agent] CJK reply (%d chars): regenerating in English",
                    len(text),
                )
                _apply_language_retry(parsed)
                steer_trace["lang_retried"] = True
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                status, resp_headers, content_type, payload = self._fetch_buffered(
                    path, body, headers, on_tick
                )
                if status != 200:
                    return status, resp_headers, content_type, payload
                is_sse = stream or "text/event-stream" in content_type
        want_retry = parsed is not None and _early_stop_armed(steer_trace)
        # Prose tool call -> real tool_calls (both transports, no retry).
        if parsed is not None and parsed.get("tools"):
            if is_sse:
                text0, has_tools0 = _sse_text_and_tools(payload)
            elif "application/json" in content_type:
                text0, has_tools0 = _json_text_and_tools(payload)
            else:
                text0, has_tools0 = "", True
            if not has_tools0:
                pseudo = _pseudo_tool_calls(text0, parsed)
                if pseudo:
                    _repair_tool_calls_list(pseudo, steer_messages)
                    model, rid = _payload_ids(payload, sse=is_sse)
                    log.info(
                        "[agent] prose tool call -> tool_calls %s %s",
                        pseudo[0]["function"]["name"],
                        pseudo[0]["function"]["arguments"][:100],
                    )
                    steer_trace["prose_tool_call"] = True
                    if is_sse:
                        return status, resp_headers, content_type, _sse_tool_payload(pseudo, model, rid)
                    return status, resp_headers, content_type, _json_tool_payload(pseudo, model, rid)
        if is_sse:
            text, has_tools = _sse_text_and_tools(payload)
            if want_retry and _should_force_continue(text, has_tools, steer_messages):
                log.info("[agent] early-stop plan retry")
                _apply_early_stop_retry(parsed)  # type: ignore[arg-type]
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                status, resp_headers, content_type, payload = self._fetch_buffered(
                    path, body, headers, on_tick
                )
                if status != 200:
                    return status, resp_headers, content_type, payload
                is_sse = stream or "text/event-stream" in content_type
                text, has_tools = _sse_text_and_tools(payload)
                if not has_tools:
                    pseudo = _pseudo_tool_calls(text, parsed)
                    if pseudo:
                        _repair_tool_calls_list(pseudo, steer_messages)
                        model, rid = _payload_ids(payload, sse=True)
                        log.info("[agent] prose tool call (after retry) -> tool_calls %s", pseudo[0]["function"]["name"])
                        steer_trace["prose_tool_call"] = True
                        return status, resp_headers, content_type, _sse_tool_payload(pseudo, model, rid)
                if _should_force_continue(text, has_tools, steer_messages):
                    tcs = _synthetic_tool_calls(text, steer_messages)
                    if tcs:
                        model, rid = _payload_ids(payload, sse=is_sse)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        return status, resp_headers, content_type, _sse_tool_payload(
                            tcs, model, rid
                        )
                    log.info("[agent] early-stop cap: letting recap through")
            return (
                status,
                resp_headers,
                content_type,
                _repair_sse_payload(payload, steer_messages),
            )
        if "application/json" in content_type:
            payload = _repair_response_payload(payload, steer_messages)
            text, has_tools = _json_text_and_tools(payload)
            if want_retry and _should_force_continue(text, has_tools, steer_messages):
                log.info("[agent] early-stop plan retry")
                _apply_early_stop_retry(parsed)  # type: ignore[arg-type]
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                status, resp_headers, content_type, payload = self._fetch_buffered(
                    path, body, headers, on_tick
                )
                if status != 200:
                    return status, resp_headers, content_type, payload
                if "application/json" in content_type:
                    payload = _repair_response_payload(payload, steer_messages)
                text, has_tools = _json_text_and_tools(payload)
                if not has_tools:
                    pseudo = _pseudo_tool_calls(text, parsed)
                    if pseudo:
                        _repair_tool_calls_list(pseudo, steer_messages)
                        model, rid = _payload_ids(payload, sse=False)
                        log.info("[agent] prose tool call (after retry) -> tool_calls %s", pseudo[0]["function"]["name"])
                        steer_trace["prose_tool_call"] = True
                        return status, resp_headers, content_type, _json_tool_payload(pseudo, model, rid)
                if _should_force_continue(text, has_tools, steer_messages):
                    tcs = _synthetic_tool_calls(text, steer_messages)
                    if tcs:
                        model, rid = _payload_ids(payload, sse=False)
                        log.info(
                            "[agent] synthetic continue %s %s",
                            tcs[0]["function"]["name"],
                            tcs[0]["function"]["arguments"][:80],
                        )
                        payload = _json_tool_payload(tcs, model, rid)
                    else:
                        log.info("[agent] early-stop cap: letting recap through")
        return status, resp_headers, content_type, payload

    # ---- request entry ---------------------------------------------------

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
            and k.lower() not in _SESSION_HEADER_DROP
        }
        # The proxy owns every control it sends (sampling, enable_thinking,
        # reasoning_effort). mtplx honours anonymous body controls by default;
        # this header keeps that true if an operator ever sets
        # MTPLX_CLIENT_CONTROLS_DEFAULT=hints on the engine.
        headers["X-MTPLX-Allow-Client-Controls"] = "1"

        path = self.path
        stream = False
        steer_messages: list[dict] | None = None
        parsed: dict | None = None
        steer_trace: dict[str, Any] = {}
        if body and "application/json" in (self.headers.get("Content-Type") or ""):
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                parsed = data
                if _scrub_session_body(data):
                    log.info("[agent] stripped client session affinity")
                stream = bool(data.get("stream"))
                if _is_chat_path(path):
                    log.info("[in] %s", summarize_request(data))
                    prepare_body(data, steer_trace)
                    msgs = data.get("messages")
                    if isinstance(msgs, list):
                        steer_messages = msgs
                    log.info(
                        "[steer] temp=%s think=%s/%s fp=%s ntools=%s max_tokens=%s "
                        "compaction=%s empty_rec=%s fake=%s prose=%s "
                        "after_tool=%s first_turn=%s trunc=%s repeat=%s cycle=%s trimmed=%s nmsg=%s",
                        data.get("temperature"),
                        data.get("enable_thinking"),
                        data.get("reasoning_effort") or "-",
                        data.get("frequency_penalty"),
                        len(data.get("tools") or []),
                        data.get("max_tokens"),
                        steer_trace.get("compaction"),
                        steer_trace.get("empty_tool_recovery"),
                        steer_trace.get("fake_action_recovery"),
                        steer_trace.get("prose_loop_recovery"),
                        steer_trace.get("after_tool_continue"),
                        steer_trace.get("first_turn_guard"),
                        steer_trace.get("truncated_tool_msgs"),
                        steer_trace.get("repeat_count"),
                        steer_trace.get("cycle_max"),
                        steer_trace.get("trimmed_msgs"),
                        len(msgs) if isinstance(msgs, list) else 0,
                    )
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"

        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{self.state.host}:{self.state.port}"
        headers["Connection"] = "close"
        chat = _is_chat_path(path)

        stale_stop = bool(steer_trace.get("cycle_stale_stop"))
        if chat and (steer_trace.get("repeat_stop") or stale_stop) and parsed is not None:
            # Hard stop: N identical assistant turns in a row, or a cycle of any
            # period that has gone N turns without a new action. Do not spend
            # another engine slot; end the turn with a visible explanation.
            n = int(steer_trace.get("repeat_count") or 0)
            if stale_stop and not steer_trace.get("repeat_stop"):
                stale = int(steer_trace.get("cycle_stale") or 0)
                what = str(steer_trace.get("cycle_what") or "the same few commands")
                text = _CYCLE_STOP_TEXT.format(what=what[:160])
                rid = f"chatcmpl-proxy-cycle-{int(time.time())}"
                log.warning("[agent] cycle guard: hard stop stale=%s what=%r", stale, what)
                append_live_event(
                    {"cycle_stale_stop": True, "cycle_stale": stale, "what": what}
                )
            else:
                what = str(steer_trace.get("repeat_what") or "same action")
                text = _REPEAT_STOP_TEXT.format(n=n, what=what)
                rid = f"chatcmpl-proxy-repeat-{int(time.time())}"
                log.warning("[agent] repeat guard: hard stop n=%s what=%r", n, what)
                append_live_event({"repeat_hard_stop": True, "repeat_count": n, "what": what})
            model = str(parsed.get("model") or "qwen3.8-27b-obliterated-mtplx")
            try:
                if stream:
                    self._sse_begin()
                    self._sse_send(_sse_text_payload(text, model, rid))
                else:
                    payload = _json_text_payload(text, model, rid)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except (_ClientGone, BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            if stream and not _needs_buffered_sse(steer_trace):
                log.info("[agent] sse passthrough")
                self._stream_passthrough(path, body, headers, parsed, steer_trace)
                return
            if stream:
                self._proxy_buffered_sse(
                    path, headers, body, parsed, steer_messages, steer_trace
                )
                return

            # Non-streaming (harness / curl): plain JSON in, plain JSON out.
            if chat and not self._acquire_gen_lock(_noop):
                self._send_json(
                    503,
                    {
                        "error": {
                            "message": "engine busy: another generation is still running",
                            "type": "proxy_busy",
                        }
                    },
                )
                return
            try:
                status, resp_headers, content_type, payload = self._post_process(
                    path=path,
                    headers=headers,
                    body=body,
                    parsed=parsed,
                    steer_messages=steer_messages,
                    steer_trace=steer_trace,
                    stream=False,
                    on_tick=_noop,
                )
            finally:
                if chat:
                    self._release_gen_lock()
            self.send_response(status)
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
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except _ClientGone:
            log.info("client disconnected (gone before response)")
        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected (broken pipe)")
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error: %s", exc)
            try:
                self._send_json(
                    502, {"error": {"message": str(exc), "type": "proxy_error"}}
                )
            except Exception:
                pass

    def _proxy_buffered_sse(
        self,
        path: str,
        headers: dict[str, str],
        body: bytes,
        parsed: dict | None,
        steer_messages: list[dict] | None,
        steer_trace: dict[str, Any],
    ) -> None:
        """Buffered SSE (early-stop middleware) with the client kept alive.

        Headers are committed first; ``: keepalive`` comments flow while the
        engine works (including a possible retry). The final payload is a
        complete SSE body, so nothing is ever retracted from the client.
        """
        log.info("[agent] sse buffered (early-stop middleware)")
        self._sse_begin()
        if not self._acquire_gen_lock(self._sse_tick):
            self._sse_fail(503, "engine busy: another generation is still running")
            return
        try:
            status, _resp_headers, content_type, payload = self._post_process(
                path=path,
                headers=headers,
                body=body,
                parsed=parsed,
                steer_messages=steer_messages,
                steer_trace=steer_trace,
                stream=True,
                on_tick=self._sse_tick,
            )
            if status != 200:
                self._sse_fail(status, payload)
                return
            if "text/event-stream" not in content_type and payload.lstrip().startswith(
                b"{"
            ):
                self._sse_send(b"data: " + payload.strip() + b"\n\n" + _SSE_DONE)
                return
            if b"[DONE]" not in payload[-64:]:
                payload = payload.rstrip(b"\r\n") + b"\n\n" + _SSE_DONE
            self._sse_send(payload)
        except _ClientGone:
            log.info("client disconnected while buffering; upstream aborted")
        except Exception as exc:  # noqa: BLE001
            log.exception("upstream error (buffered sse): %s", exc)
            try:
                self._sse_fail(502, str(exc))
            except _ClientGone:
                pass
        finally:
            self._release_gen_lock()


def _self_check() -> None:
    kilo_like = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 128,
        "enable_thinking": True,
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "fix it"},
        ],
    }
    tr: dict[str, Any] = {}
    prepare_body(kilo_like, tr)
    assert kilo_like["temperature"] == 0.0, kilo_like
    assert kilo_like["top_p"] == 1.0, kilo_like
    assert kilo_like["frequency_penalty"] == CARD_FREQUENCY_PENALTY, kilo_like
    # Tool turn: thinking ON (bounded by the engine budget), effort validated.
    assert kilo_like["enable_thinking"] is THINKING, kilo_like
    if THINKING:
        assert kilo_like["reasoning_effort"] in _REASONING_EFFORT_CHOICES, kilo_like
        assert AGENT_MAX_TOKENS_FLOOR >= 2048 + THINKING_BUDGET
    else:
        assert "reasoning_effort" not in kilo_like, kilo_like
    assert kilo_like["max_tokens"] == AGENT_MAX_TOKENS_FLOOR, kilo_like
    assert AGENT_MAX_TOKENS_CAP >= AGENT_MAX_TOKENS_FLOOR
    assert LOOP_MARKER in kilo_like["messages"][0]["content"]
    assert tr.get("loop_prompt") is True
    # First user turn must NOT get empty-tool / after-tool hooks (over-broad).
    sys0 = kilo_like["messages"][0]["content"]
    assert "[Harness] EMPTY TOOL RESULT:" not in sys0
    assert "[Harness] Tool result received." not in sys0
    assert tr.get("empty_tool_recovery") is False
    assert tr.get("after_tool_continue") is False

    compact = {
        "tool_choice": "none",
        "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "compact the conversation"}],
        "temperature": 1.0,
    }
    # Duplicate-call detection: response sig must match the history sig.
    hist = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"cat a.md\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "A CONTENT"},
    ]
    def _ev(delta: dict, finish=None) -> str:
        return "data: " + json.dumps(
            {"id": "x", "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        ) + "\n\n"
    sse = (
        _ev({"tool_calls": [{"index": 0, "id": "c2", "type": "function",
                             "function": {"name": "bash", "arguments": '{"comm'}}]})
        + _ev({"tool_calls": [{"index": 0, "function": {"arguments": 'and": "cat a.md"}'}}]})
        + _ev({}, "tool_calls")
        + "data: [DONE]\n\n"
    ).encode()
    tcs = _payload_tool_calls(sse, sse=True)
    assert tcs == [{"name": "bash", "arguments": '{"command": "cat a.md"}'}], tcs
    assert _tool_calls_sig(tcs) == _last_assistant_tool_sig(hist), (_tool_calls_sig(tcs), _last_assistant_tool_sig(hist))
    assert _payload_finish_reason(sse, sse=True) == "tool_calls"
    assert _last_tool_result_head(hist) == "A CONTENT"
    dup_body = {"messages": list(hist), "tools": [{"type": "function"}]}
    _apply_duplicate_retry(dup_body, "bash(cat a.md)", "A CONTENT")
    assert dup_body["messages"][-1]["role"] == "user" and "A CONTENT" in dup_body["messages"][-1]["content"]
    assert _tool_rounds_since_user(dup_body["messages"]) == 1  # harness user msg is skipped
    assert REPEAT_NO_TOOLS > 3 and AFTER_TOOL_NUDGE_MAX >= 20

    # Prose tool calls -> real tool_calls (the 19:08 live trace shape and kin).
    tools_body = {"tools": [
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}},
    ]}
    live = "<bash>\n\nls -la /tmp/x/ 2>/dev/null; echo ---TOOLS---; ls /tmp/y/ | head -60\n\n\n"
    pt = _pseudo_tool_calls(live, tools_body)
    assert pt and pt[0]["function"]["name"] == "bash", pt
    assert json.loads(pt[0]["function"]["arguments"])["command"].startswith("ls -la /tmp/x/"), pt
    pt = _pseudo_tool_calls("<read>\n<filePath>README.md</filePath>\n<limit>150</limit>\n</read>", tools_body)
    assert pt and json.loads(pt[0]["function"]["arguments"]) == {"filePath": "README.md", "limit": 150}, pt
    pt = _pseudo_tool_calls('<tool_call>{"name": "read", "arguments": {"filePath": "a.md"}}', tools_body)
    assert pt and pt[0]["function"]["name"] == "read", pt
    pt = _pseudo_tool_calls("```bash\ngrep -rn foo src | head\n```", tools_body)
    assert pt and "grep -rn foo" in pt[0]["function"]["arguments"], pt
    assert _pseudo_tool_calls("The build passes; nothing else to do.", tools_body) == []
    assert _pseudo_tool_calls("Use <b>bold</b> here.", tools_body) == []  # unknown tag
    # A declared tool name mentioned in prose is NOT an invocation.
    assert _pseudo_tool_calls("I'll use the <bash> tool to list files.", tools_body) == []
    assert _pseudo_tool_calls("Next I will <read> the config file.", tools_body) == []
    # ... but a closed tag mid-reply still is.
    pt = _pseudo_tool_calls("Let me check.\n<bash>ls -la</bash>", tools_body)
    assert pt and json.loads(pt[0]["function"]["arguments"]) == {"command": "ls -la"}, pt
    assert _pseudo_tool_calls(live, {"tools": []}) == []
    pt = _pseudo_tool_calls('[Tool calls: bash({"command": "tail -60 J.md"})]', tools_body)
    assert pt and json.loads(pt[0]["function"]["arguments"])["command"] == "tail -60 J.md", pt
    pt = _pseudo_tool_calls('Let me check.\n[Calling tool: read({"filePath": "a.md"})]', tools_body)
    assert pt and pt[0]["function"]["name"] == "read", pt

    # Non-consecutive cycle: A B C A B C A -> max recurrence 3, no consecutive.
    def _asst(cmd):
        return {"role": "assistant", "content": "Let me look.",
                "tool_calls": [{"id": "x", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]}
    cyc_msgs = [{"role": "user", "content": "go"}]
    for cmd in ["ls recent", "read SUMMARY", "tail JOURNAL", "ls recent", "read SUMMARY", "tail JOURNAL", "ls recent"]:
        cyc_msgs.append(_asst(cmd))
        cyc_msgs.append({"role": "tool", "content": "out"})
    cyc_max, cyc_distinct, cyc_what = _assistant_cycle(cyc_msgs)
    assert cyc_max == 3 and cyc_distinct == 3, (cyc_max, cyc_distinct)
    assert _assistant_repeat_count(cyc_msgs)[0] == 1  # invisible to consecutive guard
    cyc_body = {"tools": [{"type": "function", "function": {"name": "bash"}}], "messages": [dict(m) for m in cyc_msgs]}
    tr_cyc: dict[str, Any] = {}
    prepare_body(cyc_body, tr_cyc)
    assert tr_cyc["cycle_recovery"] is True and tr_cyc["cycle_max"] == 3, tr_cyc
    assert "tools" in cyc_body  # cycle keeps tools (redirect, not stop)
    assert "[Harness] LOOP DETECTED:" in cyc_body["messages"][0]["content"]
    if THINKING:
        assert cyc_body.get("reasoning_effort") == "xhigh" and tr_cyc.get("cycle_escalate_effort") is True, cyc_body
    # A healthy varied run does not trip it.
    ok_msgs = [{"role": "user", "content": "go"}]
    for cmd in ["ls", "read a", "grep b", "edit c", "test d"]:
        ok_msgs.append(_asst(cmd)); ok_msgs.append({"role": "tool", "content": "out"})
    assert _assistant_cycle(ok_msgs)[0] == 1
    # Cycle-break: a signature seen >= CYCLE_BREAK_MIN times is banned; effort
    # escalates to xhigh on cycle detection.
    ban_msgs = [{"role": "user", "content": "go"}]
    for _ in range(CYCLE_BREAK_MIN):
        ban_msgs.append(_asst("tail -c 2000 J.md"))
        ban_msgs.append({"role": "tool", "content": "same"})
    banned = _cycled_signatures(ban_msgs, CYCLE_BREAK_MIN)
    assert 'bash({"command": "tail -c 2000 J.md"})' in banned, banned
    assert _cycled_signatures(ban_msgs, CYCLE_BREAK_MIN + 1) == set()
    # Below the break threshold the command is not yet banned.
    assert _cycled_signatures(ban_msgs[:-4], CYCLE_BREAK_MIN) == set()

    # Novelty starvation: the exit for cycles of ANY period. A 2-, 3- or
    # 4-cycle is invisible to both the consecutive guard and the ban list, so
    # the stale counter is the only thing that can end those turns.
    def _cycle_msgs(period_cmds, rounds):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(rounds):
            msgs.append(_asst(period_cmds[i % len(period_cmds)]))
            msgs.append({"role": "tool", "content": "nothing new"})
        return msgs
    for cmds in (["A"], ["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]):
        loop_msgs = _cycle_msgs(cmds, 20)
        stale = _turns_since_new_signature(loop_msgs)
        assert stale == 20 - len(cmds), (cmds, stale)
        assert stale >= CYCLE_STALE_STOP, (cmds, stale)
    # Neither the consecutive guard nor the ban list can see a 3-cycle.
    three = _cycle_msgs(["A", "B", "C"], 20)
    assert _assistant_repeat_count(three)[0] == 1
    assert _cycled_signatures(three, CYCLE_BREAK_MIN) == set()
    # Progress resets it: a new signature at the end means stale == 0.
    assert _turns_since_new_signature(three + [_asst("E"), {"role": "tool", "content": "new"}]) == 0
    # A varied run never accumulates staleness.
    assert _turns_since_new_signature(ok_msgs) == 0
    # End-to-end: the 3-cycle turn loses its tools and is marked for a stop.
    stale_body = {"tools": [{"type": "function", "function": {"name": "bash"}}],
                  "messages": [dict(m) for m in three]}
    tr_stale: dict[str, Any] = {}
    prepare_body(stale_body, tr_stale)
    assert tr_stale["cycle_stale"] == 17, tr_stale
    assert tr_stale["cycle_stale_no_tools"] is True and tr_stale["cycle_stale_stop"] is True, tr_stale
    assert "tools" not in stale_body, stale_body  # forced to answer in text
    assert tr_stale.get("cycle_what"), tr_stale

    prepare_body(compact)
    assert "tools" not in compact
    assert compact["enable_thinking"] is False and "reasoning_effort" not in compact, compact
    plain = {"messages": [{"role": "user", "content": "title please"}], "max_tokens": 32}
    prepare_body(plain)
    assert plain["enable_thinking"] is False and "reasoning_effort" not in plain, plain
    assert compact["tool_choice"] == "none"

    kilo_agent_with_summary_boilerplate = {
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "list_files"}}],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent. You may summarize the conversation "
                    "when asked. Prefer tools."
                ),
            },
            {"role": "user", "content": "review work and next steps"},
        ],
        "max_tokens": 32000,
    }
    tr_kilo: dict[str, Any] = {}
    prepare_body(kilo_agent_with_summary_boilerplate, tr_kilo)
    assert tr_kilo.get("compaction") is False, tr_kilo
    assert kilo_agent_with_summary_boilerplate.get("tools")
    assert kilo_agent_with_summary_boilerplate["tool_choice"] == "auto"

    empty_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "(no output)"},
        ],
        "max_tokens": 128,
    }
    tr_empty: dict[str, Any] = {}
    prepare_body(empty_body, tr_empty)
    assert tr_empty.get("empty_tool_recovery") is True, tr_empty
    assert "[Harness] EMPTY TOOL RESULT:" in empty_body["messages"][0]["content"]
    # Empty recovery supersedes generic after-tool continue.
    assert tr_empty.get("after_tool_continue") is False, tr_empty

    useful_tool = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "echo hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t2",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo hi"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t2", "content": "hi"},
        ],
        "max_tokens": 128,
    }
    tr_useful: dict[str, Any] = {}
    prepare_body(useful_tool, tr_useful)
    assert tr_useful.get("empty_tool_recovery") is False, tr_useful
    assert tr_useful.get("after_tool_continue") is True, tr_useful
    assert "[Harness] Tool result received." in useful_tool["messages"][0]["content"]
    assert "[Harness] EMPTY TOOL RESULT:" not in useful_tool["messages"][0]["content"]

    cap_msgs: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "proceed with the research"},
    ]
    for i in range(1, AFTER_TOOL_NUDGE_MAX + 2):
        cap_msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            }
        )
        cap_msgs.append(
            {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}
        )
    cap_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": cap_msgs,
        "max_tokens": 128,
    }
    # First inject the JIT nudge (as if round 1 already happened), then
    # prepare the (cap+1)-tool history so the cap strips it.
    cap_msgs[0]["content"] += _AFTER_TOOL_NUDGE
    tr_cap: dict[str, Any] = {}
    prepare_body(cap_body, tr_cap)
    assert tr_cap.get("after_tool_continue") is False, tr_cap
    assert "[Harness] Tool result received." not in cap_body["messages"][0]["content"]
    assert _tool_rounds_since_user(cap_msgs) == AFTER_TOOL_NUDGE_MAX + 1
    plan = "Next steps\nE1 ecall 0x49 — one next RE probe on host wrapper.\n"
    assert _should_force_continue(plan, False, cap_msgs) is False

    fake_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "run tests"},
            {
                "role": "assistant",
                "content": "I will check the test suite and then run pytest.",
            },
        ],
        "max_tokens": 128,
    }
    tr_fake: dict[str, Any] = {}
    prepare_body(fake_body, tr_fake)
    assert tr_fake.get("fake_action") is True, tr_fake
    assert tr_fake.get("fake_action_recovery") is True, tr_fake
    assert "[Harness] FAKE ACTION:" in fake_body["messages"][0]["content"]

    loop_line = "import os\nimport sys\nimport os\nimport sys\n"
    loop_text = loop_line * 20
    prose_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "write parse_ini"},
            {"role": "assistant", "content": loop_text},
        ],
        "max_tokens": 128,
    }
    tr_prose: dict[str, Any] = {}
    prepare_body(prose_body, tr_prose)
    assert tr_prose.get("prose_loop") is True, tr_prose
    assert tr_prose.get("prose_loop_recovery") is True, tr_prose

    huge = "x" * (TOOL_RESULT_MAX_CHARS + 5000)
    trunc_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "dump"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t3",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t3", "content": huge},
        ],
        "max_tokens": 128,
    }
    tr_trunc: dict[str, Any] = {}
    prepare_body(trunc_body, tr_trunc)
    assert tr_trunc.get("truncated_tool_msgs") == 1, tr_trunc
    assert len(trunc_body["messages"][-1]["content"]) < len(huge)

    broken = 'grep -rn "autohunt'
    fixed = _sanitize_bash_command(broken)
    assert not _unmatched_quotes(fixed)
    assert fixed.startswith(("ls", "ls ", "head "))
    ok_grep = 'grep -rn "autohunt\\|Autohunt" "analysis/semgrep-domain-c/*.md"'
    kept_grep = _sanitize_bash_command(ok_grep)
    assert kept_grep.startswith("grep -rn"), kept_grep
    run = (
        "cd /Users/aicoder/src/private/grapheneos-titan-m2 && "
        "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh"
    )
    kept_run = _sanitize_bash_command(run)
    assert "./tools/tx_tail_one_phone.sh" in kept_run, kept_run
    assert not kept_run.startswith("head "), kept_run
    which = 'which adb 2>/dev/null || echo "no adb"'
    kept_which = _sanitize_bash_command(which)
    assert "which adb" in kept_which, kept_which
    plan = "Next steps\nE1 ecall 0x49 — one next RE probe on host wrapper.\n"
    assert _is_early_stop_plan(plan, False)
    assert _is_early_stop_plan(plan, True) is False
    assert _is_early_stop_plan("verified README.md and tests passed", False) is False
    plan_cmd = (
        "Next steps\nS2/S3 TX-TAIL — ONEPHONE_I_CONFIRM=1 "
        "./tools/tx_tail_one_phone.sh (multi-chunk)\n"
    )
    assert _command_from_plan(plan_cmd) == (
        "ONEPHONE_I_CONFIRM=1 ./tools/tx_tail_one_phone.sh"
    )
    syn = _synthetic_tool_calls(plan_cmd, None)
    assert syn[0]["function"]["name"] == "bash"
    assert "tx_tail_one_phone.sh" in syn[0]["function"]["arguments"]
    already = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "command": (
                                    "ONEPHONE_I_CONFIRM=1 "
                                    "./tools/tx_tail_one_phone.sh"
                                )
                            }
                        ),
                    }
                }
            ],
        }
    ]
    assert _synthetic_tool_calls(plan_cmd, already) == []
    retry_body = {
        "tool_choice": "auto",
        "messages": [{"role": "system", "content": "agent"}],
    }
    _apply_early_stop_retry(retry_body)
    assert retry_body["tool_choice"] == "auto"
    assert retry_body["messages"][-1]["role"] == "user"
    assert "EARLY STOP" in retry_body["messages"][-1]["content"]
    assert _should_force_continue(plan, False, None) is True
    assert _should_force_continue(plan, True, None) is False
    miss = (
        "head: analysis/semgrep-domain-c/WALLET-DOMAIN-C-FINAL.md: "
        "No such file or directory"
    )
    assert _is_empty_tool_content(miss)
    ws = "/Users/aicoder/src/private/grapheneos-titan-m2"
    rel = _normalize_fs_path(
        "Users/aicoder/src/private/grapheneos-titan-m2/README.md",
        [{"role": "user", "content": f"Current Workspace Directory ({ws})"}],
    )
    assert rel == "README.md"
    junk = "/".join(["Users/aicoder/src/private/grapheneos-titan-m2"] * 4) + "/README.md"
    assert _normalize_fs_path(
        junk,
        [{"role": "user", "content": f"Current Workspace Directory ({ws})"}],
    ) == "README.md"
    dropped = _drop_session_headers(
        {"X-Session-Id": "ses_abc", "Authorization": "local", "Content-Type": "application/json"}
    )
    assert "X-Session-Id" not in dropped
    assert dropped.get("Authorization") == "local"
    body_sid = {
        "user": "kilo",
        "session_id": "ses_abc",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"session_id": "ses_abc"},
    }
    assert _scrub_session_body(body_sid) is True
    assert "user" not in body_sid
    assert "session_id" not in body_sid
    assert "session_id" not in body_sid["metadata"]
    assert body_sid["messages"][0]["content"] == "hi"
    busy = json.dumps(
        {"error": {"message": "session ses_fa290ee49ffeuGnXskKEfH3z3m is already in flight"}}
    ).encode()
    assert _is_in_flight_error(409, busy) is True
    assert _is_in_flight_error(200, busy) is False
    assert _is_chat_path("/v1/chat/completions") is True
    assert UPSTREAM_TIMEOUT >= 120
    assert _needs_buffered_sse({"after_tool_continue": False}) is False
    assert _needs_buffered_sse({"after_tool_continue": True}) is True
    # First-turn guard: the reply right after a user prompt is buffered and
    # a plan dump there is forced into tool_calls; tool rounds are unchanged.
    assert _needs_buffered_sse({"first_turn_guard": True}) is True
    assert _needs_buffered_sse({"first_turn_guard": True, "repeat_no_tools": True}) is False
    ft_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "review this repo and run the tests"},
        ],
        "max_tokens": 128,
    }
    tr_ft: dict[str, Any] = {}
    prepare_body(ft_body, tr_ft)
    assert tr_ft.get("first_turn_guard") is True
    assert tr_ft.get("after_tool_continue") is False
    assert "[Harness] Tool result received." not in ft_body["messages"][0]["content"]
    ft_plan = (
        "I'll start by reading the README to understand the layout.\n\n"
        "Next steps:\n1. Read README.md\n2. Run the tests"
    )
    assert _should_force_continue(ft_plan, False, ft_body["messages"]) is True
    assert _should_force_continue("", False, ft_body["messages"]) is True
    assert _should_force_continue("The tests pass; 40/40 green.", False, ft_body["messages"]) is False
    assert _should_force_continue(ft_plan, True, ft_body["messages"]) is False
    ft_tool = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "t1", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "README.md"},
        ],
        "max_tokens": 128,
    }
    tr_ft2: dict[str, Any] = {}
    prepare_body(ft_tool, tr_ft2)
    assert tr_ft2.get("first_turn_guard") is False
    assert tr_ft2.get("after_tool_continue") is True
    # Forced retry now covers every nudged round (1..AFTER_TOOL_NUDGE_MAX).
    rounds_msgs = list(ft_tool["messages"])
    for i in range(2, AFTER_TOOL_NUDGE_MAX + 1):
        rounds_msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"t{i}", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}
                ],
            }
        )
        rounds_msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "ok"})
    assert _tool_rounds_since_user(rounds_msgs) == AFTER_TOOL_NUDGE_MAX
    assert _should_force_continue(ft_plan, False, rounds_msgs) is True
    rounds_msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tx", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}],
        }
    )
    rounds_msgs.append({"role": "tool", "tool_call_id": "tx", "content": "ok"})
    assert _should_force_continue(ft_plan, False, rounds_msgs) is False

    # Stop condition: hand-back endings count as an early stop.
    for hand_back in (
        "I've read both reports. Let me know if you want me to continue with the memcpy sites.",
        "Summary so far: 8 sites triaged.\n\nRemaining work: the 95 Semgrep findings.",
        "Would you like me to proceed with the ecall49 analysis?",
        "The patch is written but not yet tested against the harness.",
    ):
        assert _is_early_stop_plan(hand_back, False) is True, hand_back
    for final in (
        "All 40 unit checks pass; the self-check is green and the README row is updated.",
        "Root cause: the sanitizer rewrote an existing path to its bare basename.",
    ):
        assert _is_early_stop_plan(final, False) is False, final

    # Verification gate: edit then answer (no run/read since) is forced once.
    vg_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "v1", "function": {"name": "read", "arguments": '{"filePath":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "v1", "content": "def f(): pass"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "v2", "function": {"name": "edit", "arguments": '{"filePath":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "v2", "content": "Edit applied successfully."},
    ]
    assert _unverified_edit(vg_msgs) is True
    assert _should_force_continue("Fixed: f() now returns 1.", False, vg_msgs) is True
    vg_body = {"messages": list(vg_msgs)}
    _apply_early_stop_retry(vg_body)
    assert "[Harness] VERIFY GATE:" in vg_body["messages"][0]["content"]
    assert vg_body["messages"][-1]["role"] == "user"
    assert "[Harness] VERIFY GATE:" in vg_body["messages"][-1]["content"]
    assert _tool_rounds_since_user(vg_body["messages"]) == 2  # harness user ignored
    _apply_early_stop_retry(vg_body)
    assert vg_body["messages"][0]["content"].count("[Harness] VERIFY GATE:") == 1
    assert sum(1 for m in vg_body["messages"] if m["role"] == "user") == 2
    vg_msgs.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "v3", "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'}}]})
    vg_msgs.append({"role": "tool", "tool_call_id": "v3", "content": "3 passed"})
    assert _unverified_edit(vg_msgs) is False
    assert _should_force_continue("Fixed: f() now returns 1. 3 tests pass.", False, vg_msgs) is False

    # Bash sanitizer: an existing path is never rewritten to a bare basename
    # (the 2026-09-02 Titan M2 repeat-guard loop).
    import tempfile

    with tempfile.TemporaryDirectory() as ws_dir:
        nested = Path(ws_dir) / "analysis" / "autohunt"
        nested.mkdir(parents=True)
        (nested / "STATE.json").write_text("{}")
        (nested / "JOURNAL.md").write_text("# j")
        san_msgs = [
            {"role": "system", "content": f"Current Workspace Directory ({ws_dir})"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "s1", "function": {"name": "bash", "arguments": '{"command":"ls analysis/autohunt"}'}}]},
            {"role": "tool", "tool_call_id": "s1", "content": "STATE.json\nJOURNAL.md\n"},
        ]
        good = f"cd {ws_dir} && cat analysis/autohunt/STATE.json; echo ---; head -60 analysis/autohunt/JOURNAL.md"
        assert _rewrite_missing_path_command(good, san_msgs) is None, _rewrite_missing_path_command(good, san_msgs)
        assert _sanitize_bash_command(good, san_msgs) == good
        rel_good = "cat analysis/autohunt/STATE.json"
        assert _sanitize_bash_command(rel_good, san_msgs) == rel_good
        # A genuinely missing path with a known qualified sibling is still fixed.
        san_msgs[-1]["content"] = "analysis/autohunt/STATE.json\n"
        assert _rewrite_missing_path_command("cat STATE.json", san_msgs) == "cat analysis/autohunt/STATE.json"
        assert _leading_cd_dir("cd /tmp && ls") == "/tmp"
        assert _leading_cd_dir("ls /tmp") is None

    # Stability: runaway greedy completions are capped, floors still apply.
    cap_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 32000,
    }
    prepare_body(cap_body)
    assert cap_body["max_tokens"] == AGENT_MAX_TOKENS_CAP, cap_body["max_tokens"]
    small_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 16,
    }
    prepare_body(small_body)
    assert small_body["max_tokens"] == AGENT_MAX_TOKENS_FLOOR
    alias_body = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [{"role": "user", "content": "go"}],
        "max_completion_tokens": 100000,
    }
    prepare_body(alias_body)
    assert alias_body["max_tokens"] == AGENT_MAX_TOKENS_CAP
    assert "max_completion_tokens" not in alias_body
    plain_body = {"messages": [{"role": "user", "content": "hi"}]}
    prepare_body(plain_body)
    assert "max_tokens" not in plain_body  # engine default, nothing invented
    plain_big = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}
    prepare_body(plain_big)
    assert plain_big["max_tokens"] == CHAT_MAX_TOKENS_CAP
    ping_body = {"messages": [{"role": "user", "content": "PING"}], "max_tokens": 16}
    prepare_body(ping_body)
    assert ping_body["max_tokens"] == 16

    # Compaction: detected by history length even without a prompt hint, and
    # its decode is capped.
    long_hist = [{"role": "system", "content": "Compress conversation context."}]
    for i in range(20):
        long_hist.append({"role": "user", "content": f"q{i}"})
        long_hist.append({"role": "assistant", "content": f"a{i} " * 8})
    comp_body = {"messages": long_hist, "max_tokens": 32768}
    tr_comp: dict[str, Any] = {}
    prepare_body(comp_body, tr_comp)
    assert tr_comp["compaction"] is True
    assert comp_body["max_tokens"] == COMPACTION_MAX_TOKENS_CAP
    hint_body = {"messages": [{"role": "user", "content": "compress conversation context"}]}
    assert _looks_like_compaction(hint_body) is True

    # Language: English-only directive on every chat request; CJK detection.
    lang_plain = {"messages": [{"role": "user", "content": "总结这个项目"}]}
    tr_lang: dict[str, Any] = {}
    prepare_body(lang_plain, tr_lang)
    assert "[Harness] ENGLISH ONLY:" in lang_plain["messages"][0]["content"]
    prepare_body(lang_plain, tr_lang)  # idempotent (Kilo retries same body)
    assert lang_plain["messages"][0]["content"].count("[Harness] ENGLISH ONLY:") == 1
    lang_agent = {
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "go"},
        ],
        "max_tokens": 128,
    }
    tr_lang2: dict[str, Any] = {}
    prepare_body(lang_agent, tr_lang2)
    assert "[Harness] ENGLISH ONLY:" in lang_agent["messages"][0]["content"]
    comp_lang = {
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "compress conversation context"}],
    }
    prepare_body(comp_lang)
    assert "[Harness] ENGLISH ONLY:" in comp_lang["messages"][0]["content"]
    assert _is_mostly_cjk(
        "工作回顾：这是 Titan M2 研究环境的现状总结，包含全部关键信息与下一步。"
    ) is True
    assert _is_mostly_cjk(
        "Work review: the Titan M2 harness is healthy and all gates pass."
    ) is False
    assert _is_mostly_cjk("Status: 完成 (done) — see REPORT.md for the details.") is False
    lang_retry_body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
    }
    _apply_language_retry(lang_retry_body)
    _apply_language_retry(lang_retry_body)
    assert (
        lang_retry_body["messages"][0]["content"].count(
            "[Harness] LANGUAGE FAILURE:"
        )
        == 1
    )
    evs, rest = _split_sse_events(b'data: {"a":1}\n\ndata: {"b"')
    assert len(evs) == 1 and evs[0].endswith(b"\n\n") and rest == b'data: {"b"'
    evs, rest = _split_sse_events(b"data: 1\n\ndata: 2\n\n")
    assert len(evs) == 2 and rest == b""
    evs, rest = _split_sse_events(b"data: 1\r\n\r\ndata: 2\n\n")
    assert len(evs) == 2 and rest == b""

    # Repeat guard: same tool call REPEAT_NO_TOOLS turns running -> nudge,
    # tools gone; below that the nudge alone (tools kept: the response-side
    # duplicate retry gets its chance first).
    rep_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"filePath": "notes.md"}'},
    }
    rep_msgs: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for _ in range(3):
        rep_msgs.append({"role": "assistant", "content": "Let me check the notes.", "tool_calls": [dict(rep_call)]})
        rep_msgs.append({"role": "tool", "tool_call_id": "c1", "content": "# notes\nline"})
    n_rep, sig_rep = _assistant_repeat_count(rep_msgs)
    assert n_rep == 3 and sig_rep.startswith("read(")
    rep_body3 = {
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": [dict(m) for m in rep_msgs],
    }
    tr_rep3: dict[str, Any] = {}
    prepare_body(rep_body3, tr_rep3)
    assert tr_rep3["repeat_count"] == 3 and tr_rep3["repeat_recovery"] is True
    assert tr_rep3["repeat_no_tools"] is False and "tools" in rep_body3, tr_rep3
    assert rep_body3["enable_thinking"] is THINKING
    while _assistant_repeat_count(rep_msgs)[0] < REPEAT_NO_TOOLS:
        rep_msgs.append({"role": "assistant", "content": "Let me check the notes.", "tool_calls": [dict(rep_call)]})
        rep_msgs.append({"role": "tool", "tool_call_id": "c1", "content": "# notes\nline"})
    rep_body = {
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": [dict(m) for m in rep_msgs],
    }
    tr_rep: dict[str, Any] = {}
    prepare_body(rep_body, tr_rep)
    assert tr_rep["repeat_count"] == REPEAT_NO_TOOLS and tr_rep["repeat_recovery"] is True
    assert tr_rep["repeat_no_tools"] is True and tr_rep["repeat_stop"] is False
    assert "tools" not in rep_body and rep_body["tool_choice"] == "none"
    assert rep_body["enable_thinking"] is False  # no tools -> no budget -> no think
    assert "[Harness] REPEATED ACTION" in rep_body["messages"][0]["content"]
    assert rep_body["messages"][0]["content"].count("[Harness] REPEATED ACTION") == 1
    # One more -> hard stop flag (the handler answers without upstream).
    rep_msgs.append({"role": "assistant", "content": "Let me check the notes.", "tool_calls": [dict(rep_call)]})
    rep_msgs.append({"role": "tool", "tool_call_id": "c1", "content": "# notes\nline"})
    tr_rep4: dict[str, Any] = {}
    prepare_body({"tools": [{"type": "function", "function": {"name": "read"}}], "messages": rep_msgs}, tr_rep4)
    assert tr_rep4["repeat_stop"] is True and tr_rep4["repeat_count"] == REPEAT_HARD_STOP
    # A different call in between breaks the chain (git status before/after a
    # commit is legitimate).
    mixed = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [dict(rep_call)]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "git commit"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [dict(rep_call)]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    assert _assistant_repeat_count(mixed)[0] == 1
    # Identical prose (no tools) also counts; tiny acks do not.
    prose_rep = [{"role": "user", "content": "a"}] + [
        {"role": "assistant", "content": "I will check the directory and report back shortly."}
    ] * 3
    assert _assistant_repeat_count(prose_rep)[0] == 3
    assert _assistant_repeat_count([{"role": "assistant", "content": "Done."}] * 5)[0] == 0
    stop_txt = _sse_text_payload("halt", "m", "rid")
    assert b'"finish_reason": "stop"' in stop_txt and stop_txt.endswith(_SSE_DONE)
    assert json.loads(_json_text_payload("halt", "m", "rid"))["choices"][0]["message"]["content"] == "halt"

    # Context trim: oldest groups go first, tool_call/tool pairs stay intact,
    # the tail is untouched, a note is appended to the system prompt.
    trim_msgs: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for i in range(6):
        trim_msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        trim_msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 5000})
    trim_msgs.append({"role": "user", "content": "final question"})
    dropped = _trim_history_to_budget(trim_msgs, budget=16000)
    # 'start' user message + whole assistant/tool pairs -> odd count, >= 3.
    assert dropped >= 3 and dropped % 2 == 1, dropped
    assert trim_msgs[-1]["content"] == "final question"
    roles_after = [m["role"] for m in trim_msgs]
    assert roles_after[0] == "system" and "dropped" in trim_msgs[0]["content"]
    for i, m in enumerate(trim_msgs):
        if m["role"] == "tool":
            assert trim_msgs[i - 1]["role"] == "assistant" and trim_msgs[i - 1].get("tool_calls")
    assert _trim_history_to_budget([{"role": "user", "content": "short"}], budget=1000) == 0

    # SSE plumbing: in-band error is a terminal, parseable event.
    err = _sse_error_payload(503, b'{"error":{"message":"busy"}}')
    assert err.startswith(b"data: ") and err.endswith(_SSE_DONE)
    evt = json.loads(err.split(b"\n\n")[0][len(b"data: "):])
    assert evt["error"]["message"] == "busy" and evt["error"]["code"] == 503
    err2 = _sse_error_payload(502, "socket closed")
    assert b"socket closed" in err2
    # A half-streamed upstream body must not be echoed back as the error text.
    err3 = _sse_error_payload(502, b'data: {"id":"x","object":"chat.completion.chunk","choices":[]}\n\n: keep-alive\n\n')
    evt3 = json.loads(err3.split(b"\n\n")[0][len(b"data: "):])
    assert "aborted mid-response" in evt3["error"]["message"]
    assert "chat.completion.chunk" not in evt3["error"]["message"]
    assert _sse_at_event_boundary(b"") is True
    assert _sse_at_event_boundary(b"data: {}\n\n") is True
    assert _sse_at_event_boundary(b"data: {\"cho") is False
    assert _SSE_KEEPALIVE.startswith(b":") and _SSE_KEEPALIVE.endswith(b"\n\n")

    # Engine lock: RLock with timeout works and busy flag mirrors it.
    assert _UPSTREAM_GEN_LOCK.acquire(timeout=0.01) is True
    _UPSTREAM_GEN_LOCK.release()
    assert _UPSTREAM_GEN_LOCK_BUSY.is_set() is False
    assert LOCK_WAIT_TIMEOUT > 0 and _env_int("QWEN38_OBL_UNSET_TEST", 900) == 900


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Qwen3.8 OBLITERATED ↔ Kilo agent harness proxy"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument(
        "--upstream",
        default="http://127.0.0.1:8767",
        help="raw mtplx base (no trailing /v1)",
    )
    p.add_argument("--self-check", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _self_check()
    if args.self_check:
        print("qwen38_obl_kilo_proxy self-check ok")
        return 0

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log.info("debug log file %s", LOG_FILE)

    state = ProxyState(args.upstream)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(
        "OBLITERATED kilo proxy on http://%s:%d → %s (greedy, thinking %s)",
        args.host,
        args.port,
        state.upstream,
        f"{REASONING_EFFORT} budget={THINKING_BUDGET} on tool turns"
        if THINKING
        else "off",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    _ = threading
    sys.exit(main())
