#!/usr/bin/env python3
"""Judge one reply from the local model for the failure shapes seen in long
agentic sessions (abliterated Gemma 4 26B-A4B on TensorFold; OpenCode
transcripts of 2026-09-30). Pure functions, no I/O. loop_proxy.py calls
`judge_prose` on the text as it streams and `judge_reply` on the finished
reply, then cancels the generation at the engine and retries the step.

Every shape below is taken from a real transcript:

  loop        `</td></tr></tbody></table>` x 700 (18k chars): one unit
              repeated back to back. The drafter's acceptance collapses and
              the step crawls (45 tok/s) to the token cap.
  gibberish   prose that turns into `$-$-$(-x[m|size(-$` LaTeX / punctuation
              soup: the symbol share of a 600-char window climbs from <0.2 to
              0.35-0.57 and `$` appears 40+ times a window.
  narration   3,000+ chars of "Let's go. / The first action is bash / Wait,
              / Actually," and no tool call: the model talks about acting
              instead of acting (45k chars, "Let's go." x 98).
  leak        a raw `<|tool_call>call:write{...}` envelope or a
              `<|channel>thought` block in the answer text: the engine could
              not parse it, so the client shows markup and nothing runs.
  truncated   finish_reason "length" (or completion_tokens == max_tokens): the
              step hit the token cap, so a tool call is partial.
  bad-args    a tool call whose arguments are not JSON, whose path is a bare
              degenerate token (`mapsto_path_..._now`), whose string value is
              itself a loop, an edit whose old and new strings are equal (a
              no-op the model re-issued five times in a row, live), or a bash
              command that is only comments (the model "checking" in prose).

Thresholds were calibrated on 90 assistant messages from this model (37 clean
final replies, 49 tool-call steps, 4 failures): healthy prose never exceeded a
0.17 symbol share per window, 2 `$` in a whole reply, 5 contiguous repeats of
a unit, or 1 duplicated line, and long clean code compressed no tighter than
0.16 (a 20-row matrix literal is 20 identical lines, so tool arguments
need 40 repeats). The failures sit far past every threshold (see findings.md).

Ported to the Gemma 4 base stack (censored/gemma4-26b-a4b-mlx-tensorfold)
on 2026-09-30 and replayed over the OpenCode transcripts of the local
TensorFold models (Gemma 4 base: 13 assistant messages, 11 text parts, 4
tool calls; as extra evidence from the same engine family, Ternary Bonsai 2:
263 / 103 / 313 and Qwen3.8-27B: 1063 / 120 / 1109). Two changes came out
of that:

  - Markdown table rows are exempt from the duplicated-line rule. A healthy
    11k-char Qwen exploration report repeated `| File | Lines |` once per
    crate section, six times, which is a per-section header, not a loop;
    the loop rule still catches contiguous repeats. No other line (headings
    included) repeated more than once in any healthy reply, against 98 for
    the abliterated "Let's go." failure, so the threshold stays at 6.
  - The leak rule also knows the Hermes `<tool_call>{...}` envelope and a
    `<think>` opener at the start of the answer (the shapes the other
    TensorFold lanes use). None of these strings occurs in any healthy
    transcript.

Nothing else moved: the healthy maxima (symbol share 0.11 / 0.12 per window
on Bonsai / Gemma base, 0.21 on Qwen; one `$` per reply; no contiguous
repeats; at most one narration phrase per reply) sit well under the
thresholds above. The three
Gemma 4 base trips are real `<|channel>thought` leaks from before the engine's
channel fix.
"""

from __future__ import annotations

import collections
import functools
import json
import re
import zlib
from dataclasses import dataclass
from typing import Any

# ---- calibration ------------------------------------------------------------
WINDOW = 600             # chars per window for the symbol-share test
WINDOW_STEP = 300
SYMBOL_HARD = 0.40       # one window at/above this: gibberish (healthy prose max 0.17)
SYMBOL_SOFT = 0.30       # two consecutive windows at/above this
DOLLARS_PER_WINDOW = 12  # `$` per window: LaTeX soup (healthy replies: <= 2 in total)
LOOP_REPS_PROSE = 12     # contiguous repeats of one unit in prose (healthy code max 5)
LOOP_REPS_ARGS = 40      # the same inside a tool argument: a 20-row matrix literal is 20 identical lines
LOOP_UNIT_MIN = 3
LOOP_UNIT_MAX = 160
LOOP_TAIL = 2400         # chars of tail scanned while streaming
LOOP_ZLIB_RATIO = 0.08   # a tail compressing tighter than this repeats a unit longer than LOOP_UNIT_MAX
DUP_LINE_MIN_LEN = 8
DUP_LINES = 6            # one line this many times (healthy max 1)
NARRATION_MIN_CHARS = 3000
NARRATION_MIN_HITS = 8
NARRATION_PER_KCHAR = 1.5
PROSE_CAP = 16000        # chars of prose and still no tool call in an agentic step
PATH_MAX = 400
PATH_KEYS = {"filepath", "file_path", "path", "notebook_path", "directory", "dir"}
STRING_ARG_LOOP_MIN = 200

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t:|-]{3,}\|?[ \t]*$", re.M)   # | --- | :-- |
_PUNCT_LINE = re.compile(r"^[ \t]*[-=*_~#]{3,}[ \t]*$", re.M)          # ----- / =====
_LEAK_START = re.compile(r"\A\s*(?:thought\n|<\|channel>|<think>)")
_LEAK_TOOL = re.compile(   # Gemma <|tool_call>call:NAME{, Hermes <tool_call>{json}, Nemotron <TOOLCALL>[json]
    r"<\|tool_call>\s*(?:call)?:\s*[A-Za-z_][\w.-]*\s*\{|<tool_call>\s*\{|<TOOLCALL>\s*\[")
_LEAK_CHANNEL = re.compile(r"<\|channel>\s*thought")
_TABLE_ROW = re.compile(r"^[ \t]*\|")   # a table header repeats once per section of a report
_INTENT = re.compile(
    r"(?:let'?s (?:go|do it|do this|start|begin|write|proceed|make|run|do the)|the first action"
    r"|i'?ll now|i will now|will be written|is being written|are being written|ready to"
    r"|the write calls?|let the action|will run|will replace"
    r"|\bwait[,.!]|\bactually[,.]|\bokay[,.!]|\bok[,.!]|\bhmm[,.])",
    re.I,
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""    # "", loop, gibberish, narration, leak, truncated, bad-args, runaway
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


PASS = Verdict(True)


def strip_code(text: str) -> str:
    """Prose without fenced blocks, inline code, table rules and punctuation
    rules: code and tables are symbol-heavy on purpose and must not count."""

    text = _FENCE.sub(" ", text)
    text = _INLINE.sub(" ", text)
    text = _TABLE_RULE.sub("", text)
    return _PUNCT_LINE.sub("", text)


@functools.lru_cache(maxsize=8)
def _loop_pattern(reps: int, unit_min: int) -> re.Pattern[str]:
    return re.compile(r"(.{%d,%d}?)\1{%d,}" % (unit_min, LOOP_UNIT_MAX, reps - 1), re.S)


def find_loop(text: str, *, reps: int = LOOP_REPS_PROSE, tail: int = LOOP_TAIL) -> str | None:
    """The unit that `text` (its last `tail` chars; 0 = all) repeats `reps`+
    times back to back, or None. A run of one character (`-----`, `=====`)
    is a rule, not a loop, and is ignored."""

    part = text[-tail:] if tail else text
    if len(part) < LOOP_UNIT_MIN * reps:
        return None
    pattern = _loop_pattern(reps, 4 if reps == LOOP_REPS_ARGS else LOOP_UNIT_MIN)
    for match in pattern.finditer(part):
        unit = match.group(1)
        if len(set(unit)) > 1:
            return unit
    if tail and len(part) >= 800:
        packed = len(zlib.compress(part.encode("utf-8", "replace"), 6))
        if packed / len(part) < LOOP_ZLIB_RATIO and len(set(part.strip())) > 1:
            return part[-80:]
    return None


def duplicated_line(text: str, *, times: int = DUP_LINES) -> tuple[str, int] | None:
    counts = collections.Counter(
        line.strip() for line in text.splitlines()
        if len(line.strip()) >= DUP_LINE_MIN_LEN and not _TABLE_ROW.match(line)
    )
    if not counts:
        return None
    line, n = counts.most_common(1)[0]
    return (line, n) if n >= times else None


def gibberish(stripped: str, *, tail_only: bool) -> str | None:
    """Symbol soup / LaTeX soup in `stripped` (see strip_code)."""

    scan = stripped[-(WINDOW * 2 + WINDOW_STEP):] if tail_only else stripped
    prev_soft = False
    for start in range(0, max(1, len(scan) - WINDOW // 2 + 1), WINDOW_STEP):
        window = scan[start:start + WINDOW]
        if len(window) < WINDOW // 2:
            break
        n = len(window)
        symbols = sum(1 for c in window if not c.isalnum() and not c.isspace())
        share = symbols / n
        dollars = window.count("$")
        if share >= SYMBOL_HARD:
            return f"symbol share {share:.2f} in a {WINDOW}-char window (prose is below 0.2)"
        if dollars >= DOLLARS_PER_WINDOW:
            return f"{dollars} `$` signs in a {WINDOW}-char window (LaTeX soup)"
        if share >= SYMBOL_SOFT and prev_soft:
            return f"symbol share {share:.2f} in two consecutive {WINDOW}-char windows"
        prev_soft = share >= SYMBOL_SOFT
    return None


def narration(stripped: str) -> str | None:
    if len(stripped) < NARRATION_MIN_CHARS:
        return None
    hits = len(_INTENT.findall(stripped))
    per_kchar = hits / (len(stripped) / 1000)
    if hits >= NARRATION_MIN_HITS and per_kchar >= NARRATION_PER_KCHAR:
        return f"{hits} 'let's go / the first action is / wait, actually' phrases in {len(stripped)} chars and no tool call"
    return None


def leak(raw: str, stripped: str) -> str | None:
    if _LEAK_START.search(raw):
        return "the reply opens with a raw thought / think channel"
    if _LEAK_TOOL.search(stripped):
        return "a raw tool-call envelope sits in the answer text, so the call never ran"
    if _LEAK_CHANNEL.search(stripped):
        return "a raw <|channel>thought block sits in the answer text"
    return None


def judge_prose(content: str, *, agentic: bool, finished: bool, with_tool_calls: bool = False) -> Verdict:
    """Judge answer text. `agentic`: the request offered tools (a coding-agent
    step), which enables the gibberish, narration and runaway rules; a plain
    chat may legitimately be math or ASCII art. `finished`: judge all of it,
    else only the tail that streaming just added. `with_tool_calls`: the
    finished reply also carries tool calls, so talk before them is tolerated."""

    if not content:
        return PASS
    stripped = strip_code(content)
    why = leak(content, stripped)
    if why:
        return Verdict(False, "leak", why)
    unit = find_loop(content)
    if unit is not None:
        return Verdict(False, "loop", f"repeats {unit[:40]!r} {LOOP_REPS_PROSE}+ times back to back")
    dup = duplicated_line(stripped)
    if dup:
        return Verdict(False, "loop", f"the line {dup[0][:60]!r} appears {dup[1]} times")
    if agentic:
        why = gibberish(stripped, tail_only=not finished)
        if why:
            return Verdict(False, "gibberish", why)
        if not with_tool_calls:
            why = narration(stripped)
            if why:
                return Verdict(False, "narration", why)
            if len(content) >= PROSE_CAP:
                return Verdict(False, "runaway", f"{len(content)} chars of prose and still no tool call")
    return PASS


def judge_tool_call(name: str, arguments: Any) -> Verdict:
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return Verdict(False, "bad-args", f"{name}: arguments are not JSON ({exc.msg} at {exc.pos})")
    else:
        args = arguments
    if not isinstance(args, dict):
        return Verdict(False, "bad-args", f"{name}: arguments are not an object")
    # An edit whose old and new strings match can never change the file. Live,
    # the model issued the same no-op edit five times in a row (OpenCode: "No
    # changes to apply: oldString and newString are identical").
    old, new = args.get("oldString", args.get("old_string")), args.get("newString", args.get("new_string"))
    if name.lower() in ("edit", "str_replace", "str_replace_editor") and isinstance(old, str) and old == new:
        return Verdict(False, "bad-args", f"{name}: oldString and newString are identical, a no-op edit; change the text or stop editing")
    command = args.get("command")
    if name.lower() in ("bash", "shell", "run_terminal_cmd", "execute_command") and isinstance(command, str):
        live = [line for line in command.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not live:
            return Verdict(False, "bad-args", f"{name}: the command is empty or only comments (a no-op); run a real check or stop")
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        if key.lower() in PATH_KEYS:
            if len(value) > PATH_MAX or "\n" in value or "\r" in value:
                return Verdict(False, "bad-args", f"{name}.{key} is {len(value)} chars or spans lines")
            if "/" not in value and "." not in value and len(value) > 40:
                return Verdict(False, "bad-args", f"{name}.{key} is a bare token, not a path: {value[:60]!r}")
        if len(value) >= STRING_ARG_LOOP_MIN:
            unit = find_loop(value, reps=LOOP_REPS_ARGS, tail=0)
            if unit is not None:
                return Verdict(False, "loop", f"{name}.{key} repeats {unit[:40]!r} {LOOP_REPS_ARGS}+ times back to back")
    return PASS


def judge_reply(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str | None,
    *,
    agentic: bool,
    completion_tokens: int | None = None,
    max_tokens: int | None = None,
) -> Verdict:
    """The finished reply: truncation (agentic only: a chat that asked for 100
    tokens is meant to stop there), every tool call's arguments, then the text."""

    calls = tool_calls or []
    if agentic and (finish_reason == "length" or (completion_tokens and max_tokens and completion_tokens >= max_tokens)):
        what = "a tool call" if calls else "the reply"
        return Verdict(False, "truncated", f"{what} ran into the {max_tokens or completion_tokens}-token limit of one step")
    for call in calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or call.get("name") or "?")
        verdict = judge_tool_call(name, fn.get("arguments", call.get("arguments", "")))
        if not verdict.ok:
            return verdict
    return judge_prose(content or "", agentic=agentic, finished=True, with_tool_calls=bool(calls))


# ---- self-test ---------------------------------------------------------------

def selftest() -> None:
    import random

    rng = random.Random(7)   # soup that does not repeat a unit, so the loop rule stays quiet
    soup = "".join(rng.choice("$-(){}[]|:\\^_.,;!$-$xm0 ") for _ in range(1400))
    table_loop = "</td></tr></tbody></table>" * 40
    healthy_code = "\n".join(
        f"    if (arena[y][x] === {i}) {{\n        continue outer;\n    }}" for i in range(5)
    )
    narrated = "".join(
        f"Let's go. Actually, wait, the first action {i} is a bash call. Ready to run step {i} now.\n"
        for i in range(60)
    )
    report = (
        "Wrote two files and verified them.\n\n| file | check |\n| --- | --- |\n| z/index.html | opens |\n"
        "| z/tetris.js | `node --check` clean |\n\nRun it with `PORT=$PORT ./serve.sh` or open the .html file.\n"
        "----------------------------------------------------------------------------------------\n"
        "```js\nconst $ = document.querySelector.bind(document); $('#a'); $('#b'); $('#c'); $('#d');\n```\n"
    )
    # loops
    assert judge_prose(table_loop, agentic=True, finished=False).reason == "loop"
    assert judge_prose(table_loop, agentic=False, finished=True).reason == "loop", "a loop is a failure even in chat"
    assert judge_prose(healthy_code, agentic=True, finished=True).ok
    assert judge_prose("-" * 300 + "\nrule above\n", agentic=True, finished=True).ok, "a rule is not a loop"
    assert judge_prose("Let's go.\n" * 8, agentic=True, finished=True).reason == "loop", "one line repeated 6+ times"
    # gibberish only counts in agentic steps
    assert judge_prose(report + soup, agentic=True, finished=False).reason == "gibberish"
    assert judge_prose(report + soup, agentic=True, finished=True).reason == "gibberish"
    assert judge_prose(soup, agentic=False, finished=True).ok, "chat may be math / ASCII art"
    assert judge_prose(report, agentic=True, finished=True).ok, "tables, rules, inline `$`, fenced code are fine"
    assert judge_prose("The task ran past the $12 \\times 20$ grid.", agentic=True, finished=True).ok, "a little math is fine"
    # narration
    assert judge_prose(narrated, agentic=True, finished=True).reason == "narration"
    assert judge_prose(narrated, agentic=True, finished=True, with_tool_calls=True).ok, "talk before a real call passes"
    assert judge_prose(narrated, agentic=False, finished=True).ok
    assert judge_prose(narrated[:2000], agentic=True, finished=True).ok, "short talk is not judged"
    # leaks
    raw_call = '<|tool_call>call:write{content:<|"|><!DOCTYPE html><html>…'
    assert judge_prose(raw_call, agentic=True, finished=True).reason == "leak"
    assert judge_prose("The parser missed `<|tool_call>call:write{…}` here.", agentic=True, finished=True).ok
    assert judge_prose("```\n" + raw_call + "\n```\nis what leaked.", agentic=True, finished=True).ok
    assert judge_prose("thought\nThe glob output showed a candidate.", agentic=True, finished=False).reason == "leak"
    assert judge_prose("<|channel>thought\nplanning<channel|>Answer", agentic=False, finished=True).reason == "leak"
    hermes = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
    assert judge_prose(hermes, agentic=True, finished=True).reason == "leak", "a raw Hermes envelope (Qwen / Bonsai)"
    assert judge_prose('<TOOLCALL>[{"name": "bash", "arguments": {"command": "ls"}}]</TOOLCALL>', agentic=True, finished=False).reason == "leak"
    assert judge_prose("<think>\nplan first\n</think>\nDone.", agentic=True, finished=False).reason == "leak"
    assert judge_prose("Wrap reasoning in `<think>` tags and calls in `<tool_call>{...}`.", agentic=True, finished=True).ok
    sectioned = "".join(
        f"### crates/{name}/ ({i * 100} lines)\n| File | Lines |\n|---|---|\n| src/lib.rs | {i * 100} |\n\n"
        for i, name in enumerate(("quic", "h3", "qpack", "scenarios", "cli", "afl", "fuzz", "docs"))
    )
    assert judge_prose(sectioned, agentic=True, finished=True).ok, "a table header once per section is not a loop"
    assert judge_prose(sectioned + "Let's go.\n" * 6, agentic=True, finished=True).reason == "loop", "other lines still count"
    # runaway
    prose = " ".join(f"token{i}" for i in range(2400))   # non-repeating, symbol-free, 20k+ chars
    assert judge_prose(prose, agentic=True, finished=False).reason == "runaway"
    assert judge_prose(prose, agentic=False, finished=True).ok
    # tool calls
    assert judge_tool_call("write", '{"filePath": "z/index.html", "content": "<html></html>"}').ok
    assert judge_tool_call("write", '{"filePath": "z/index.html", "content": "<html>').reason == "bad-args"
    assert judge_tool_call("write", '{"filePath": "mapsto_path_for_the_next_write_after_fixing_the_logic_mess", "content": "x"}').reason == "bad-args"
    assert judge_tool_call("write", json.dumps({"filePath": "a.html", "content": table_loop * 2})).reason == "loop"
    assert judge_tool_call("write", json.dumps({"filePath": "a.py", "content": "# " + "-" * 86 + "\n" + "\n".join(f"x{i} = {i}" for i in range(30))})).ok
    matrix = "const arena = [\n" + "  [0,0,0,0,0,0,0,0,0,0,0,0],\n" * 20 + "];\n"
    assert judge_tool_call("write", json.dumps({"filePath": "a.js", "content": matrix})).ok, "a 20-row literal is not a loop"
    assert judge_tool_call("write", json.dumps({"filePath": "a.js", "content": "  [0,0,0,0,0,0,0,0,0,0,0,0],\n" * 60})).reason == "loop"
    assert judge_tool_call("bash", '{"command": "ls -la"}').ok
    assert judge_tool_call("bash", json.dumps({"command": "# Since it is a browser game we cannot run it\n# Let's check syntax instead"})).reason == "bad-args"
    assert judge_tool_call("bash", json.dumps({"command": "# check syntax\nnode --check z/tetris.js"})).ok
    assert judge_tool_call("edit", json.dumps({"filePath": "z/t.js", "oldString": "a();", "newString": "a();"})).reason == "bad-args"
    assert judge_tool_call("edit", json.dumps({"filePath": "z/t.js", "oldString": "a();", "newString": "b();"})).ok
    assert judge_tool_call("read", '{"filePath": "README"}').ok
    call = {"function": {"name": "write", "arguments": '{"filePath": "a", "content": "b"}'}}
    # replies
    assert judge_reply("", [call], "tool_calls", agentic=True).ok
    assert judge_reply("", [call], "length", agentic=True).reason == "truncated"
    assert judge_reply("", [call], "tool_calls", agentic=True, completion_tokens=6144, max_tokens=6144).reason == "truncated"
    assert judge_reply("cut off", [], "length", agentic=False).ok, "a chat that asked for N tokens stops at N"
    assert judge_reply(soup, [call], "tool_calls", agentic=True).reason == "gibberish", "soup next to a call still fails"
    assert judge_reply(narrated, [call], "tool_calls", agentic=True).ok
    assert judge_reply("Done: wrote z/index.html and z/tetris.js.", [], "stop", agentic=True).ok
    print("harness_judge self-test ok")


if __name__ == "__main__":
    selftest()
