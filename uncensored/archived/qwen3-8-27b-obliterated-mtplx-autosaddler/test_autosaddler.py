#!/usr/bin/env python3
"""Offline AutoSaddler optimizer tests (no live model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autosaddler as asd  # noqa: E402


def test_self_check() -> None:
    asd._self_check()


def test_live_events_drive_diagnosis_when_batch_is_green() -> None:
    diag = asd.diagnose(
        [{"id": "two_step_echo", "ok": True, "labels": [], "rounds": []}],
        live_events=[{"fake_action_recovery": True, "fake_action": True}],
    )
    assert diag["root_cause"] == "no_tool_call"
    assert asd.live_failure_signal([{"after_tool_continue": True}]) is False


def test_patch_replace_not_unknown() -> None:
    h = asd.normalize_harness(None)
    nxt = asd.apply_patch(
        h,
        {
            "type": "prompt_rule_modify",
            "target": "loop_prefix",
            "action": "replace",
            "value": "Act with tools. Do not stop after 1 of N.",
        },
    )
    assert "Act with tools" in nxt["loop_prefix"]
    assert asd.LOOP_MARKER in nxt["loop_prefix"]
    try:
        asd.apply_patch(h, {"target": "rm -rf", "value": "x"})
        raise AssertionError("unknown target must fail")
    except ValueError:
        pass


def test_diagnose_and_phase() -> None:
    diag = asd.diagnose(
        [
            {
                "id": "empty_recovery",
                "ok": False,
                "labels": ["stopped_after_1", "empty_result_idle"],
                "rounds": [
                    {"tools": ["bash"]},
                    {"tools": [], "content": "I will glob next"},
                ],
            }
        ]
    )
    assert diag["root_cause"] == "empty_result_idle"
    p = asd.propose_patch(asd.DEFAULT_HARNESS, diag, phase="capability", tried=set())
    assert p is not None
    p2 = asd.propose_patch(asd.DEFAULT_HARNESS, diag, phase="steering", tried=set())
    assert p2 is not None
    assert p2["type"] in asd.STEERING_TYPES or p["type"] in asd.CAPABILITY_TYPES


def test_accept_requires_train_then_dev() -> None:
    parent = [{"id": "a", "ok": False}, {"id": "b", "ok": True}]
    child_train_gain = [{"id": "a", "ok": True}, {"id": "b", "ok": True}]
    child_train_reg = [{"id": "a", "ok": False}, {"id": "b", "ok": False}]
    oc, lesson = asd.reflect(parent, child_train_gain, {"type": "x", "target": "y"}, True, True)
    assert oc["fixed"] == ["a"]
    assert not oc["regressed"]
    oc2, lesson2 = asd.reflect(parent, child_train_reg, {"type": "x", "target": "y"}, False, None)
    assert oc2["regressed"] == ["b"]
    assert "over-scoped" in lesson2.lower() or "regressed" in lesson2.lower()
    assert asd._score(child_train_gain) > asd._score(parent)
    assert asd._score(child_train_reg) < asd._score(parent)


def test_evodag_persist_and_compose(tmp_path: Path | None = None) -> None:
    td = Path(tempfile.mkdtemp(prefix="as-dag-"))
    old_state = asd.STATE_DIR
    old_active = asd.ACTIVE_PATH
    old_evo = asd.EVODAG_PATH
    old_events = asd.EVENTS_PATH
    old_cand = asd.CANDIDATES_DIR
    asd.STATE_DIR = td
    asd.ACTIVE_PATH = td / "active.json"
    asd.EVODAG_PATH = td / "evodag.json"
    asd.EVENTS_PATH = td / "events.jsonl"
    asd.CANDIDATES_DIR = td / "candidates"
    try:
        dag = asd.empty_evodag()
        base_id = dag["active_id"]
        h2 = asd.apply_patch(
            dag["nodes"][base_id]["harness"],
            {
                "type": "agent_loop_logic",
                "target": "mw_after_tool",
                "action": "replace",
                "value": False,
            },
        )
        n2 = {
            "id": asd.harness_id(h2),
            "parent_id": base_id,
            "patch": {"type": "agent_loop_logic", "target": "mw_after_tool", "value": False},
            "harness": h2,
            "train": {"score": 1.0},
            "dev": {"score": 0.5},
            "accepted": True,
            "outcomes": {},
            "lessons": "ok",
        }
        dag["nodes"][n2["id"]] = n2
        dag["edges"].append({"from": base_id, "to": n2["id"], "delta": "mw_after_tool", "accepted": True})
        dag["best_dev_id"] = n2["id"]
        asd.save_evodag(dag)
        asd.save_active(h2)
        loaded = asd.load_evodag()
        assert loaded["best_dev_id"] == n2["id"]
        live = asd.load_active()
        assert live["mw_after_tool"] is False
        composed = asd.compose_from_dag(loaded)
        assert composed["mw_after_tool"] is False
    finally:
        asd.STATE_DIR = old_state
        asd.ACTIVE_PATH = old_active
        asd.EVODAG_PATH = old_evo
        asd.EVENTS_PATH = old_events
        asd.CANDIDATES_DIR = old_cand


def main() -> int:
    tests = [
        test_self_check,
        test_live_events_drive_diagnosis_when_batch_is_green,
        test_patch_replace_not_unknown,
        test_diagnose_and_phase,
        test_accept_requires_train_then_dev,
        test_evodag_persist_and_compose,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
