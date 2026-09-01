#!/usr/bin/env python3
"""AutoSaddler optimizer for the Qwen3.8 OBLITERATED mtplx harness (Park et al., arXiv:2608.23041).

Offline mini-batch loop over a *named component map* (prompts, tool descriptions,
middleware flags) — not unconstrained repo edits:

  1. Evaluate H_n on a train mini-batch (real tool loops)
  2. Diagnose failed traces → one structured patch (capability first, then steering)
  3. Verify on the same mini-batch (must not regress; prefer strict train gain)
  4. If that passes, evaluate D_dev; keep only if hold-out does not regress
  5. Reflect (fixed / regressed / still-failing / still-passing) → lessons
  6. Record node+edge in EvoDAG (append-only events)
  7. Evolve H_{n+1} from the DAG (best-dev parent, optional component merge)

State is durable under ``.autosaddler/`` so later ``--optimize`` runs resume.
The proxy reloads ``.autosaddler/active.json`` by mtime (no restart needed).

Usage:
  python3 autosaddler.py --iters 3
  python3 test_harness.py --optimize --iters 3
  ./2_start_mtplx.sh optimize
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".autosaddler"
ACTIVE_PATH = STATE_DIR / "active.json"
EVODAG_PATH = STATE_DIR / "evodag.json"
EVENTS_PATH = STATE_DIR / "events.jsonl"
LIVE_EVENTS_PATH = STATE_DIR / "live-events.jsonl"
CANDIDATES_DIR = STATE_DIR / "candidates"
LOCK_PATH = STATE_DIR / "optimize.lock"
DAEMON_STATE_PATH = STATE_DIR / "daemon-state.json"
DAEMON_PID_PATH = STATE_DIR / "daemon.pid"
DAEMON_LOG_PATH = STATE_DIR / "daemon.log"

AS_LOOP_START = "[AS_LOOP]"
AS_LOOP_END = "[/AS_LOOP]"
LOOP_MARKER = "Do not stop after 1 of N."

# Built-in base harness (capability + steering). Matches qwen38_obl_kilo_proxy defaults.
DEFAULT_HARNESS: dict[str, Any] = {
    "loop_prefix": (
        "Finish every requested step. Prefer tools over prose. After each tool "
        "result, immediately take the next action. Do not stop after 1 of N. "
        "Do not recap; act. Empty or error tool output: retry with a simpler "
        "local command (ls/glob/grep/read). Verify with a tool before declaring "
        "the job done."
    ),
    "empty_tool_nudge": (
        "\n\n[Harness] EMPTY TOOL RESULT: the latest tool output was empty or "
        "useless. Do not write a revised plan. Next message MUST be a local tool "
        "(bash ls/echo, glob, grep, or read) on a real path. If FileNotFound, "
        "list the parent directory — do not idle."
    ),
    "fake_action_nudge": (
        "\n\n[Harness] FAKE ACTION: the last assistant message claimed to act "
        "('Executing…', 'I will check…') but emitted NO tool_calls. Next message "
        "MUST include tool_calls — run the command now. No plans."
    ),
    "prose_loop_nudge": (
        "\n\n[Harness] PROSE LOOP: the last assistant turn repeated itself "
        "without tool_calls. Stop monologuing. Next message MUST be tool_calls "
        "only — take the next unfinished step."
    ),
    "tool_desc": {
        "bash": (
            "Run a short local shell command (echo, ls, pwd, cat). "
            "After the result, continue to the next requested step."
        ),
        "read": (
            "Read a workspace file by path. Use after glob/grep locates "
            "the file. Do not guess paths."
        ),
        "glob": (
            "Find workspace files by glob pattern (e.g. **/*.txt). "
            "Use before read."
        ),
        "grep": (
            "Search workspace file contents with a regex. Prefer this "
            "over bash grep."
        ),
    },
    "mw_empty_tool": True,
    "mw_fake_action": True,
    "mw_prose_loop": True,
    "mw_after_tool": True,
    "mw_force_tools_on_fake_prose": True,
    "empty_streak_min": 1,
}

# Paper Table 1: Capability (C) vs Steering (S).
CAPABILITY_TYPES = frozenset(
    {"agent_loop_logic", "infrastructure", "tool_implementation"}
)
STEERING_TYPES = frozenset(
    {"prompt_rule_modify", "prompt_rule_add", "tool_description", "pretool_hook"}
)

_CACHE: dict[str, Any] = {"mtime": None, "data": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def harness_id(harness: dict[str, Any]) -> str:
    blob = json.dumps(normalize_harness(harness), sort_keys=True, ensure_ascii=False)
    return "h_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def normalize_harness(raw: dict[str, Any] | None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    out = deepcopy(DEFAULT_HARNESS)
    for key in (
        "loop_prefix",
        "empty_tool_nudge",
        "fake_action_nudge",
        "prose_loop_nudge",
    ):
        val = src.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val
    td = src.get("tool_desc")
    if isinstance(td, dict):
        merged = dict(out["tool_desc"])
        for name, desc in td.items():
            if isinstance(desc, str) and desc.strip() and name in merged:
                merged[name] = desc
        out["tool_desc"] = merged
    for key in (
        "mw_empty_tool",
        "mw_fake_action",
        "mw_prose_loop",
        "mw_after_tool",
        "mw_force_tools_on_fake_prose",
    ):
        if key in src:
            out[key] = bool(src[key])
    if "empty_streak_min" in src:
        try:
            out["empty_streak_min"] = max(1, int(src["empty_streak_min"]))
        except (TypeError, ValueError):
            pass
    return out


def load_active() -> dict[str, Any]:
    """Live overlay. Proxy/harness call this; reloads when active.json changes."""
    path = ACTIVE_PATH
    if not path.is_file():
        return deepcopy(DEFAULT_HARNESS)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return deepcopy(DEFAULT_HARNESS)
    if _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
        return deepcopy(_CACHE["data"])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_HARNESS)
    if isinstance(data, dict) and isinstance(data.get("harness"), dict):
        data = data["harness"]
    merged = normalize_harness(data if isinstance(data, dict) else {})
    _CACHE["mtime"] = mtime
    _CACHE["data"] = merged
    return deepcopy(merged)


def save_active(harness: dict[str, Any]) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    h = normalize_harness(harness)
    hid = harness_id(h)
    payload = {"id": hid, "harness": h, "updated_at": _now()}
    cand = CANDIDATES_DIR / f"{hid}.json"
    tmp = ACTIVE_PATH.with_suffix(".json.tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    cand.write_text(text, encoding="utf-8")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(ACTIVE_PATH)
    _CACHE["mtime"] = None
    _CACHE["data"] = None
    return ACTIVE_PATH


def append_event(kind: str, payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": _now(), "kind": kind, **payload}
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False)[:20_000] + "\n")


def append_live_event(payload: dict[str, Any]) -> None:
    """Proxy writes recovery firings so the next optimize sees real-use evidence."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        row = {"ts": _now(), **payload}
        with LIVE_EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False)[:4_000] + "\n")
    except OSError:
        pass


_FAILURE_EVENT_KEYS = (
    "empty_tool_recovery",
    "fake_action_recovery",
    "prose_loop_recovery",
    "fake_action",
    "prose_loop",
)


def live_failure_signal(events: list[dict[str, Any]]) -> bool:
    return any(any(ev.get(k) for k in _FAILURE_EVENT_KEYS) for ev in events)


def load_daemon_state() -> dict[str, Any]:
    if not DAEMON_STATE_PATH.is_file():
        return {"live_bytes": 0}
    try:
        data = json.loads(DAEMON_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"live_bytes": 0}
    return data if isinstance(data, dict) else {"live_bytes": 0}


def save_daemon_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DAEMON_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(DAEMON_STATE_PATH)


def _live_size() -> int:
    try:
        return int(LIVE_EVENTS_PATH.stat().st_size)
    except OSError:
        return 0


def unread_live_events(cursor: int) -> tuple[list[dict[str, Any]], int]:
    size = _live_size()
    if size <= cursor or not LIVE_EVENTS_PATH.is_file():
        return [], size
    try:
        raw = LIVE_EVENTS_PATH.read_bytes()[max(0, cursor) :]
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return [], size
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events, size


def acquire_optimize_lock():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        return None
    return fd


def load_live_events(limit: int = 40) -> list[dict[str, Any]]:
    if not LIVE_EVENTS_PATH.is_file():
        return []
    try:
        lines = LIVE_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def empty_evodag() -> dict[str, Any]:
    base = normalize_harness(DEFAULT_HARNESS)
    hid = harness_id(base)
    node = {
        "id": hid,
        "parent_id": None,
        "patch": None,
        "harness": base,
        "train": None,
        "dev": None,
        "accepted": True,
        "outcomes": {},
        "lessons": "Base Qwen3.8 OBLITERATED AutoSaddler harness (hand-applied defaults).",
        "created_at": _now(),
    }
    return {
        "version": 1,
        "active_id": hid,
        "best_dev_id": hid,
        "iteration": 0,
        "nodes": {hid: node},
        "edges": [],
    }


def load_evodag() -> dict[str, Any]:
    if not EVODAG_PATH.is_file():
        dag = empty_evodag()
        save_evodag(dag)
        if not ACTIVE_PATH.is_file():
            save_active(dag["nodes"][dag["active_id"]]["harness"])
        return dag
    try:
        dag = json.loads(EVODAG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_evodag()
    if not isinstance(dag, dict) or "nodes" not in dag:
        return empty_evodag()
    return dag


def save_evodag(dag: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EVODAG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dag, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(EVODAG_PATH)


def apply_patch(harness: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply one structured patch. Replace, do not stack named components."""
    h = normalize_harness(harness)
    target = str(patch.get("target") or "")
    action = str(patch.get("action") or "replace")
    value = patch.get("value")
    if target in (
        "loop_prefix",
        "empty_tool_nudge",
        "fake_action_nudge",
        "prose_loop_nudge",
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"patch {target} needs non-empty text")
        if target == "loop_prefix" and LOOP_MARKER not in value:
            value = value.rstrip() + " Do not stop after 1 of N."
        if action == "append" and isinstance(h.get(target), str):
            # Paper: replace, don't stack. Treat append as "merge into one paragraph".
            cur = h[target].rstrip()
            extra = value.strip()
            if extra not in cur:
                h[target] = cur + " " + extra
        else:
            h[target] = value.strip() if target == "loop_prefix" else value
    elif target.startswith("tool_desc."):
        name = target.split(".", 1)[1]
        if name not in h["tool_desc"]:
            raise ValueError(f"unknown tool {name}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tool description must be text")
        h["tool_desc"][name] = value.strip()
    elif target.startswith("mw_"):
        if target not in h:
            raise ValueError(f"unknown middleware flag {target}")
        h[target] = bool(value)
    elif target == "empty_streak_min":
        h["empty_streak_min"] = max(1, int(value))
    else:
        raise ValueError(f"refusing unknown patch target {target!r}")
    return normalize_harness(h)


def _score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for r in rows if r.get("ok")) / len(rows)


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        for lab in row.get("labels") or []:
            counts[str(lab)] = counts.get(str(lab), 0) + 1
    return counts


def _live_hint_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events:
        for key in (
            "empty_tool_recovery",
            "fake_action_recovery",
            "prose_loop_recovery",
            "after_tool_continue",
            "fake_action",
            "prose_loop",
        ):
            if ev.get(key):
                counts[key] = counts.get(key, 0) + 1
    return counts


def diagnose(
    train_rows: list[dict[str, Any]],
    *,
    live_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evidence-grounded diagnosis from traces (not 'it failed, add a rule')."""
    failed = [r for r in train_rows if not r.get("ok")]
    labels = _label_counts(train_rows)
    live = _live_hint_counts(live_events or [])
    evidence: list[str] = []
    for row in failed:
        last = (row.get("rounds") or [{}])[-1]
        evidence.append(
            f"{row.get('id')}: labels={row.get('labels')} "
            f"tools={last.get('tools')} finish={last.get('finish')} "
            f"fake={last.get('fake_action')} prose={last.get('prose_loop')} "
            f"content={(last.get('content') or '')[:80]!r}"
        )
    root = "unknown"
    if labels.get("empty_result_idle"):
        root = "empty_result_idle"
    elif labels.get("stopped_after_1") or labels.get("stopped_early"):
        root = "stopped_after_1"
    elif labels.get("fake_action") or labels.get("no_tool_call"):
        root = "no_tool_call"
    elif labels.get("prose_loop"):
        root = "prose_loop"
    elif labels.get("bad_tool_json"):
        root = "bad_tool_json"
    elif live.get("empty_tool_recovery"):
        root = "empty_result_idle"
    elif live.get("fake_action") or live.get("fake_action_recovery"):
        root = "no_tool_call"
    elif live.get("prose_loop") or live.get("prose_loop_recovery"):
        root = "prose_loop"
    return {
        "root_cause": root,
        "n_fail": len(failed),
        "n": len(train_rows),
        "labels": labels,
        "live": live,
        "evidence": evidence,
    }


def _phase_for_iter(i: int, n_iters: int) -> str:
    """Phased patch scheduling: capability first, then steering."""
    cut = max(1, (n_iters + 1) // 2)
    return "capability" if i < cut else "steering"


def propose_patch(
    harness: dict[str, Any],
    diagnosis: dict[str, Any],
    *,
    phase: str,
    tried: set[str],
) -> dict[str, Any] | None:
    """One structured patch. Capability before steering. Skip already-tried keys."""
    h = normalize_harness(harness)
    root = diagnosis.get("root_cause") or "unknown"
    labels = diagnosis.get("labels") or {}

    def _ok(patch: dict[str, Any]) -> dict[str, Any] | None:
        key = f"{patch['type']}:{patch['target']}:{patch.get('action')}"
        if key in tried:
            return None
        patch["key"] = key
        patch["root_cause"] = root
        return patch

    capability_candidates: list[dict[str, Any]] = []
    steering_candidates: list[dict[str, Any]] = []

    if root in {"empty_result_idle"} or labels.get("empty_result_idle"):
        if not h.get("mw_empty_tool"):
            capability_candidates.append(
                {
                    "type": "agent_loop_logic",
                    "category": "middleware",
                    "target": "mw_empty_tool",
                    "action": "replace",
                    "value": True,
                    "rationale": "Empty tool results idle the agent; enable empty-tool recovery.",
                }
            )
        if int(h.get("empty_streak_min") or 1) > 1:
            capability_candidates.append(
                {
                    "type": "infrastructure",
                    "category": "middleware",
                    "target": "empty_streak_min",
                    "action": "replace",
                    "value": 1,
                    "rationale": "Fire empty-tool recovery on the first empty result, not later.",
                }
            )
        steering_candidates.append(
            {
                "type": "pretool_hook",
                "category": "middleware",
                "target": "empty_tool_nudge",
                "action": "replace",
                "value": (
                    "\n\n[Harness] EMPTY TOOL RESULT: output was empty. Do not plan. "
                    "Next message MUST be a local tool (glob, grep, read, or bash ls/echo) "
                    "on a real workspace path. If FileNotFound, list the parent. "
                    "Do not idle."
                ),
                "rationale": "Strengthen just-in-time empty-result reminder (scoped hook).",
            }
        )

    if root in {"stopped_after_1"} or labels.get("stopped_after_1") or labels.get(
        "stopped_early"
    ):
        if not h.get("mw_after_tool"):
            capability_candidates.append(
                {
                    "type": "agent_loop_logic",
                    "category": "middleware",
                    "target": "mw_after_tool",
                    "action": "replace",
                    "value": True,
                    "rationale": "Agent stops after 1 of N; enable after-tool continue hook.",
                }
            )
        steering_candidates.append(
            {
                "type": "prompt_rule_modify",
                "category": "prompt",
                "target": "loop_prefix",
                "action": "replace",
                "value": (
                    "Finish every requested step. Prefer tools over prose. After each tool "
                    "result, immediately emit the next tool_call. Do not stop after 1 of N. "
                    "Do not recap; act. Empty or error tool output: retry with a simpler "
                    "local command (ls/glob/grep/read). Verify with a tool before declaring "
                    "the job done. A turn that only says what you will do next is a failure."
                ),
                "rationale": "Replace finish-the-job rule so step 2 is a tool call, not prose.",
            }
        )

    if root in {"no_tool_call"} or labels.get("no_tool_call") or labels.get("fake_action"):
        if not h.get("mw_fake_action"):
            capability_candidates.append(
                {
                    "type": "agent_loop_logic",
                    "category": "middleware",
                    "target": "mw_fake_action",
                    "action": "replace",
                    "value": True,
                    "rationale": "Assistant claimed to act with no tool_calls; enable fake-action recovery.",
                }
            )
        if not h.get("mw_force_tools_on_fake_prose"):
            capability_candidates.append(
                {
                    "type": "agent_loop_logic",
                    "category": "middleware",
                    "target": "mw_force_tools_on_fake_prose",
                    "action": "replace",
                    "value": True,
                    "rationale": "Force tool_choice=required after a fake-action / prose-loop turn.",
                }
            )
        steering_candidates.append(
            {
                "type": "pretool_hook",
                "category": "middleware",
                "target": "fake_action_nudge",
                "action": "replace",
                "value": (
                    "\n\n[Harness] FAKE ACTION: you described a next step but emitted no "
                    "tool_calls. Next message MUST be tool_calls only — run bash/glob/grep/read now."
                ),
                "rationale": "Replace fake-action hook text; do not stack extra rules.",
            }
        )
        steering_candidates.append(
            {
                "type": "tool_description",
                "category": "tool",
                "target": "tool_desc.bash",
                "action": "replace",
                "value": (
                    "Run a short local shell command (echo, ls, pwd, cat). "
                    "Use this instead of saying you will run a command. "
                    "After the result, continue to the next requested step."
                ),
                "rationale": "Tell the model *when* to use bash so it does not narrate instead.",
            }
        )

    if root == "prose_loop" or labels.get("prose_loop"):
        if not h.get("mw_prose_loop"):
            capability_candidates.append(
                {
                    "type": "agent_loop_logic",
                    "category": "middleware",
                    "target": "mw_prose_loop",
                    "action": "replace",
                    "value": True,
                    "rationale": "Repeated prose without tools; enable prose-loop recovery.",
                }
            )
        steering_candidates.append(
            {
                "type": "pretool_hook",
                "category": "middleware",
                "target": "prose_loop_nudge",
                "action": "replace",
                "value": (
                    "\n\n[Harness] PROSE LOOP: repeated text with no tool_calls. Stop. "
                    "Next message MUST be tool_calls for the next unfinished step only."
                ),
                "rationale": "Replace prose-loop hook; keep it scoped to matching state.",
            }
        )

    if labels.get("bad_tool_json"):
        steering_candidates.append(
            {
                "type": "tool_description",
                "category": "tool",
                "target": "tool_desc.bash",
                "action": "replace",
                "value": (
                    "Run a short local shell command (echo, ls, pwd, cat). "
                    "arguments MUST be a JSON object {\"command\":\"...\"}, complete and short. "
                    "After the result, continue to the next requested step."
                ),
                "rationale": "Tool JSON was truncated/invalid; describe the arguments contract.",
            }
        )

    pool = capability_candidates if phase == "capability" else steering_candidates
    if not pool:
        pool = capability_candidates + steering_candidates
    for cand in pool:
        got = _ok(cand)
        if got is not None:
            return got
    for cand in capability_candidates + steering_candidates:
        got = _ok(cand)
        if got is not None:
            return got
    return None


def reflect(
    parent_rows: list[dict[str, Any]],
    child_rows: list[dict[str, Any]],
    patch: dict[str, Any] | None,
    train_improved: bool,
    dev_ok: bool | None,
) -> tuple[dict[str, list[str]], str]:
    parent = {r["id"]: r for r in parent_rows}
    outcomes = {
        "fixed": [],
        "regressed": [],
        "still_failing": [],
        "still_passing": [],
    }
    ids = [r["id"] for r in child_rows]
    for tid in ids:
        before = bool((parent.get(tid) or {}).get("ok"))
        after = bool(next((r.get("ok") for r in child_rows if r["id"] == tid), False))
        if not before and after:
            outcomes["fixed"].append(tid)
        elif before and not after:
            outcomes["regressed"].append(tid)
        elif before and after:
            outcomes["still_passing"].append(tid)
        else:
            outcomes["still_failing"].append(tid)
    bits = [
        f"patch={patch.get('type') if patch else None}:{patch.get('target') if patch else None}",
        f"train_improved={train_improved} dev_ok={dev_ok}",
        f"fixed={outcomes['fixed'] or '-'} regressed={outcomes['regressed'] or '-'}",
        f"still_failing={outcomes['still_failing'] or '-'} still_passing={outcomes['still_passing'] or '-'}",
    ]
    if outcomes["regressed"]:
        bits.append(
            "Lesson: patch over-scoped or trajectory-specific — do not keep as active."
        )
    elif outcomes["fixed"] and not outcomes["regressed"]:
        bits.append("Lesson: targeted hook/rule fixed the diagnosed failure without collateral.")
    elif outcomes["still_failing"] and not outcomes["fixed"]:
        bits.append("Lesson: this layer did not address the root cause; try another category.")
    return outcomes, " ".join(bits)


def compose_from_dag(dag: dict[str, Any]) -> dict[str, Any]:
    """Evolution: recombine accepted components across lineages (not only last child)."""
    nodes = list((dag.get("nodes") or {}).values())
    scored = [
        n
        for n in nodes
        if isinstance(n, dict) and n.get("accepted") and (n.get("dev") or n.get("train"))
    ]
    if not scored:
        return normalize_harness(DEFAULT_HARNESS)
    def _dev(n: dict[str, Any]) -> float:
        d = n.get("dev") or {}
        t = n.get("train") or {}
        return float(d.get("score") if d.get("score") is not None else t.get("score") or 0.0)

    best = max(scored, key=_dev)
    composed = normalize_harness(best.get("harness"))
    for n in scored:
        if n.get("id") == best.get("id"):
            continue
        patch = n.get("patch") or {}
        if not patch.get("target"):
            continue
        if _dev(n) + 1e-9 < _dev(best):
            continue
        try:
            composed = apply_patch(composed, _component_patch_from_node(n))
        except (ValueError, TypeError):
            continue
    return normalize_harness(composed)


def _component_patch_from_node(node: dict[str, Any]) -> dict[str, Any]:
    patch = node.get("patch") or {}
    target = str(patch.get("target") or "")
    h = normalize_harness(node.get("harness"))
    if target.startswith("tool_desc."):
        name = target.split(".", 1)[1]
        value: Any = h["tool_desc"].get(name)
    else:
        value = h.get(target)
    return {"target": target, "action": "replace", "value": value, "type": patch.get("type")}


def _rows_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r.get("ok")),
        "score": _score(rows),
        "tasks": {
            r["id"]: {"ok": r.get("ok"), "labels": r.get("labels")} for r in rows
        },
    }


# ---------------------------------------------------------------------------
# Optimizer (lazy-imports test_harness to avoid import cycles with the proxy)
# ---------------------------------------------------------------------------


def _eval_split(base: str, model: str, split: str) -> list[dict[str, Any]]:
    import test_harness as th

    tasks = [t for t in th.AGENT_TASKS if t["split"] == split]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        tmp = th._plant_workspace()
        try:
            result = th.run_agent_loop(
                base,
                model,
                user=task["user"],
                workspace=Path(tmp.name),
                max_rounds=int(task["max_rounds"]),
                empty_first=bool(task["empty_first"]),
            )
            labels = th.diagnose_rollout(
                min_tool_rounds=int(task["min_tool_rounds"]),
                rounds=result["rounds"],
                empty_first=bool(task["empty_first"]),
            )
            ok = th._task_success(task, result)
            rows.append(
                {
                    "id": task["id"],
                    "split": split,
                    "ok": ok,
                    "labels": labels,
                    "tool_rounds": result["tool_rounds"],
                    "rounds": result["rounds"],
                    "executed": result.get("executed"),
                }
            )
            th._write_trace(
                f"opt_{task['id']}",
                {
                    "task": task["id"],
                    "split": split,
                    "ok": ok,
                    "labels": labels,
                    "tool_rounds": result["tool_rounds"],
                    "rounds": result["rounds"],
                },
            )
        except Exception as exc:
            rows.append(
                {
                    "id": task["id"],
                    "split": split,
                    "ok": False,
                    "labels": ["eval_error"],
                    "error": str(exc),
                    "rounds": [],
                    "tool_rounds": 0,
                }
            )
        finally:
            tmp.cleanup()
    return rows


def optimize(
    *,
    base: str,
    model: str,
    iters: int = 3,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run AutoSaddler iterations. Resumes EvoDAG. Promotes best-dev to active.json."""
    lock = acquire_optimize_lock()
    if lock is None:
        print("optimize already running — skip")
        return load_evodag()
    try:
        return _optimize_locked(base=base, model=model, iters=iters, use_llm=use_llm)
    finally:
        try:
            fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            pass
        lock.close()


def _optimize_locked(
    *,
    base: str,
    model: str,
    iters: int,
    use_llm: bool,
) -> dict[str, Any]:
    dag = load_evodag()
    active_id = dag.get("active_id") or next(iter(dag["nodes"]))
    current = normalize_harness(dag["nodes"][active_id]["harness"])
    save_active(current)
    tried: set[str] = set()
    for node in (dag.get("nodes") or {}).values():
        patch = (node or {}).get("patch") or {}
        if patch.get("key"):
            tried.add(str(patch["key"]))

    print(
        f"AutoSaddler optimize  iters={iters}  active={active_id}  "
        f"nodes={len(dag['nodes'])}  base={base}  model={model}"
    )
    parent_train = _eval_split(base, model, "train")
    parent_dev = _eval_split(base, model, "dev")
    dag["nodes"][active_id]["train"] = _rows_payload(parent_train)
    dag["nodes"][active_id]["dev"] = _rows_payload(parent_dev)
    save_evodag(dag)
    append_event(
        "eval",
        {
            "id": active_id,
            "train": dag["nodes"][active_id]["train"],
            "dev": dag["nodes"][active_id]["dev"],
        },
    )
    print(
        f"  H_n train={_score(parent_train):.2f} ({sum(r['ok'] for r in parent_train)}/{len(parent_train)})  "
        f"dev={_score(parent_dev):.2f} ({sum(r['ok'] for r in parent_dev)}/{len(parent_dev)})"
    )

    for i in range(iters):
        phase = _phase_for_iter(i, iters)
        dag["iteration"] = int(dag.get("iteration") or 0) + 1
        diagnosis = diagnose(parent_train, live_events=load_live_events())
        print(
            f"\n== iter {dag['iteration']} phase={phase} root={diagnosis['root_cause']} "
            f"fail={diagnosis['n_fail']}/{diagnosis['n']} labels={diagnosis['labels']}"
        )
        if diagnosis["root_cause"] == "unknown" and diagnosis["n_fail"] == 0:
            print("  no failure signal — skip patch")
            break
        patch = propose_patch(current, diagnosis, phase=phase, tried=tried)
        if patch is None and use_llm:
            patch = _llm_propose_patch(base, model, current, diagnosis, tried)
        if patch is None:
            print("  no unused structured patch; evolving composed candidate")
            composed = compose_from_dag(dag)
            if harness_id(composed) == harness_id(current):
                print("  evolution produced the same harness — stopping")
                break
            patch = {
                "type": "evolution_compose",
                "category": "evolution",
                "target": "loop_prefix",
                "action": "replace",
                "value": composed["loop_prefix"],
                "key": f"evolve:{harness_id(composed)}",
                "rationale": "EvoDAG composition of accepted lineage components.",
                "composed": True,
            }
            candidate = composed
        else:
            tried.add(str(patch["key"]))
            try:
                candidate = apply_patch(current, patch)
            except ValueError as exc:
                print(f"  skip invalid patch: {exc}")
                append_event("patch_invalid", {"error": str(exc), "patch": patch})
                continue

        cid = harness_id(candidate)
        print(
            f"  patch {patch.get('type')} → {patch.get('target')} "
            f"({patch.get('rationale', '')[:80]})"
        )
        save_active(candidate)
        child_train = _eval_split(base, model, "train")
        train_parent_s = _score(parent_train)
        train_child_s = _score(child_train)
        train_ok = train_child_s > train_parent_s + 1e-9 or (
            train_parent_s >= 1.0 - 1e-9 and train_child_s + 1e-9 >= train_parent_s
        )
        child_dev: list[dict[str, Any]] | None = None
        dev_ok: bool | None = None
        if train_ok:
            child_dev = _eval_split(base, model, "dev")
            dev_parent_s = _score(parent_dev)
            dev_child_s = _score(child_dev)
            dev_ok = dev_child_s + 1e-9 >= dev_parent_s
        else:
            print(
                f"  train did not improve ({train_child_s:.2f} ≰ {train_parent_s:.2f}) — "
                "skip D_dev (mini-batch gate)"
            )

        outcomes, lessons = reflect(
            parent_train,
            child_train,
            patch,
            train_improved=train_ok,
            dev_ok=dev_ok,
        )
        accepted = bool(train_ok and dev_ok)
        node = {
            "id": cid,
            "parent_id": harness_id(current),
            "patch": {
                k: patch.get(k)
                for k in (
                    "type",
                    "category",
                    "target",
                    "action",
                    "value",
                    "key",
                    "rationale",
                    "root_cause",
                )
            },
            "harness": candidate,
            "train": _rows_payload(child_train),
            "dev": _rows_payload(child_dev) if child_dev is not None else None,
            "accepted": accepted,
            "outcomes": outcomes,
            "lessons": lessons,
            "created_at": _now(),
        }
        dag["nodes"][cid] = node
        dag["edges"].append(
            {
                "from": node["parent_id"],
                "to": cid,
                "delta": patch.get("target"),
                "accepted": accepted,
            }
        )
        append_event("candidate", {"id": cid, "accepted": accepted, "lessons": lessons})
        print(f"  train {train_child_s:.2f}  accepted={accepted}  {lessons}")

        if accepted:
            current = candidate
            parent_train = child_train
            if child_dev is not None:
                parent_dev = child_dev
            dag["active_id"] = cid
            best_id = dag.get("best_dev_id") or cid
            best_dev = ((dag["nodes"].get(best_id) or {}).get("dev") or {}).get("score")
            this_dev = (node.get("dev") or {}).get("score")
            if this_dev is not None and (best_dev is None or this_dev >= float(best_dev)):
                dag["best_dev_id"] = cid
        else:
            # Stay on parent; restore active to best-dev / parent.
            restore = dag["nodes"].get(dag.get("best_dev_id") or node["parent_id"])
            if restore:
                current = normalize_harness(restore["harness"])
                dag["active_id"] = restore["id"]
            save_active(current)

        save_evodag(dag)

    best_id = dag.get("best_dev_id") or dag.get("active_id")
    best = dag["nodes"].get(best_id) or dag["nodes"][dag["active_id"]]
    save_active(best["harness"])
    dag["active_id"] = best["id"]
    save_evodag(dag)
    print(
        f"\nAutoSaddler done. active={best['id']} "
        f"train={(best.get('train') or {}).get('score')} "
        f"dev={(best.get('dev') or {}).get('score')} "
        f"state={STATE_DIR}"
    )
    return dag


def run_daemon(
    *,
    base: str,
    model: str,
    poll: int = 10,
    idle: int = 90,
    iters: int = 1,
    use_llm: bool = True,
) -> int:
    """Hands-off loop: after Kilo is quiet, optimize from live failure events."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DAEMON_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    def _on_stop(_signum: int, _frame: Any) -> None:
        print("daemon: stop", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)
    print(
        f"AutoSaddler daemon pid={os.getpid()}  idle={idle}s  poll={poll}s  "
        f"iters={iters}  base={base}  model={model}",
        flush=True,
    )
    print("  Hands-off: use Kilo; after a quiet gap a new saddle is written if needed.", flush=True)
    while True:
        state = load_daemon_state()
        cursor = int(state.get("live_bytes") or 0)
        size = _live_size()
        if size < cursor:
            cursor = 0
        events, size = unread_live_events(cursor)
        if not live_failure_signal(events):
            if size > cursor:
                state["live_bytes"] = size
                save_daemon_state(state)
            time.sleep(max(1, poll))
            continue
        try:
            age = time.time() - LIVE_EVENTS_PATH.stat().st_mtime
        except OSError:
            age = idle
        if age < idle:
            time.sleep(min(max(1, poll), idle - age + 0.4))
            continue
        print(
            f"daemon: {len(events)} failure event(s), idle {age:.0f}s — optimize {iters} iter(s)",
            flush=True,
        )
        try:
            optimize(base=base, model=model, iters=max(1, iters), use_llm=use_llm)
            state["live_bytes"] = _live_size()
            state["last_run"] = _now()
            state["last_reason"] = "live_failure_idle"
            save_daemon_state(state)
        except Exception as exc:
            print(f"daemon: optimize failed: {exc}", flush=True)
            time.sleep(max(5, poll * 2))
        time.sleep(max(1, poll))


def _llm_propose_patch(
    base: str,
    model: str,
    harness: dict[str, Any],
    diagnosis: dict[str, Any],
    tried: set[str],
) -> dict[str, Any] | None:
    """Optional in-depth diagnosis via the task model; validated against the taxonomy."""
    import test_harness as th

    allowed_targets = [
        "loop_prefix",
        "empty_tool_nudge",
        "fake_action_nudge",
        "prose_loop_nudge",
        "tool_desc.bash",
        "tool_desc.read",
        "tool_desc.glob",
        "tool_desc.grep",
        "mw_empty_tool",
        "mw_fake_action",
        "mw_prose_loop",
        "mw_after_tool",
        "mw_force_tools_on_fake_prose",
        "empty_streak_min",
    ]
    prompt = (
        "You optimize an LLM agent HARNESS (prompts, tool descriptions, middleware), "
        "not task answers. Return ONE JSON object only, no markdown:\n"
        '{"type":"prompt_rule_modify|pretool_hook|tool_description|agent_loop_logic|infrastructure",'
        '"target":"<one allowed target>","action":"replace","value":"<new text or bool or int>",'
        '"rationale":"<why this root cause>"}\n'
        f"Allowed targets: {allowed_targets}\n"
        "Replace, do not stack rules. loop_prefix MUST contain: Do not stop after 1 of N.\n"
        f"Diagnosis: {json.dumps(diagnosis, ensure_ascii=False)[:2500]}\n"
        f"Current loop_prefix: {harness.get('loop_prefix')!r}\n"
    )
    try:
        code, data, _ = th._chat(
            base,
            [{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=400,
            model=model,
            timeout=90.0,
        )
    except Exception:
        return None
    if code != 200:
        return None
    content = th._content(th._msg(data) if isinstance(data, dict) else {})
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    target = str(raw.get("target") or "")
    if target not in allowed_targets:
        return None
    ptype = str(raw.get("type") or "prompt_rule_modify")
    value = raw.get("value")
    patch = {
        "type": ptype,
        "category": "prompt" if "prompt" in ptype else (
            "tool" if target.startswith("tool_desc") else "middleware"
        ),
        "target": target,
        "action": "replace",
        "value": value,
        "rationale": str(raw.get("rationale") or "llm diagnosis")[:400],
        "root_cause": diagnosis.get("root_cause"),
    }
    key = f"{patch['type']}:{patch['target']}:{patch['action']}"
    if key in tried:
        return None
    patch["key"] = key
    try:
        apply_patch(harness, patch)
    except (ValueError, TypeError):
        return None
    return patch


def _self_check() -> None:
    h = normalize_harness(None)
    assert LOOP_MARKER in h["loop_prefix"]
    hid = harness_id(h)
    assert hid == harness_id(DEFAULT_HARNESS)
    patched = apply_patch(
        h,
        {
            "type": "agent_loop_logic",
            "target": "mw_empty_tool",
            "action": "replace",
            "value": False,
        },
    )
    assert patched["mw_empty_tool"] is False
    assert harness_id(patched) != hid
    diag = diagnose(
        [
            {
                "id": "a",
                "ok": False,
                "labels": ["stopped_after_1"],
                "rounds": [{"tools": ["bash"], "content": "done", "fake_action": False}],
            },
            {
                "id": "b",
                "ok": True,
                "labels": [],
                "rounds": [{"tools": ["bash", "bash"]}],
            },
        ]
    )
    assert diag["root_cause"] == "stopped_after_1"
    p = propose_patch(h, diag, phase="capability", tried=set())
    assert p is not None
    p2 = propose_patch(h, diag, phase="steering", tried=set())
    assert p2 is not None
    outcomes, lesson = reflect(
        [{"id": "a", "ok": False}, {"id": "b", "ok": True}],
        [{"id": "a", "ok": True}, {"id": "b", "ok": True}],
        p2,
        True,
        True,
    )
    assert outcomes["fixed"] == ["a"]
    assert "targeted" in lesson.lower() or "fixed" in lesson.lower()
    live_only = diagnose(
        [{"id": "ok", "ok": True, "labels": [], "rounds": []}],
        live_events=[{"empty_tool_recovery": True}],
    )
    assert live_only["root_cause"] == "empty_result_idle"
    assert live_failure_signal([{"after_tool_continue": True}]) is False
    assert live_failure_signal([{"fake_action_recovery": True}]) is True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AutoSaddler harness optimizer (Qwen3.8 OBLITERATED)")
    ap.add_argument("--base", default="http://127.0.0.1:8768")
    ap.add_argument("--model", default="qwen3.8-27b-obliterated-mtplx")
    ap.add_argument("--iters", type=int, default=3, help="Optimization iterations")
    ap.add_argument("--no-llm", action="store_true", help="Templates only (no extra model calls)")
    ap.add_argument("--status", action="store_true", help="Print EvoDAG status and exit")
    ap.add_argument(
        "--daemon",
        action="store_true",
        help="Hands-off: watch live-events.jsonl and optimize after a quiet gap",
    )
    ap.add_argument("--idle", type=int, default=90, help="Daemon: seconds of Kilo quiet before optimize")
    ap.add_argument("--poll", type=int, default=10, help="Daemon poll interval seconds")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)
    _self_check()
    if args.self_check:
        print("autosaddler self-check ok")
        return 0
    if args.daemon:
        return run_daemon(
            base=args.base,
            model=args.model,
            poll=max(2, int(args.poll)),
            idle=max(5, int(args.idle)),
            iters=max(1, int(args.iters)),
            use_llm=not args.no_llm,
        )
    if args.status:
        dag = load_evodag()
        active = dag.get("active_id")
        node = (dag.get("nodes") or {}).get(active) or {}
        print(f"active={active}")
        print(f"best_dev={dag.get('best_dev_id')}  iters={dag.get('iteration')}  nodes={len(dag.get('nodes') or {})}")
        print(f"train={node.get('train')}  dev={node.get('dev')}")
        print(f"active.json={ACTIVE_PATH} exists={ACTIVE_PATH.is_file()}")
        dpid = ""
        if DAEMON_PID_PATH.is_file():
            dpid = DAEMON_PID_PATH.read_text(encoding="utf-8").strip()
        alive = False
        if dpid.isdigit():
            try:
                os.kill(int(dpid), 0)
                alive = True
            except OSError:
                alive = False
        print(f"daemon: pid={dpid or '-'} running={alive} state={DAEMON_STATE_PATH}")
        return 0
    optimize(
        base=args.base,
        model=args.model,
        iters=max(1, int(args.iters)),
        use_llm=not args.no_llm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
