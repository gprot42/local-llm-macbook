#!/usr/bin/env python3
"""Unit tests for pure functions in the lean gemma4_mlx_kilo_proxy.

No LLM, no running server. Run with:
    python3 -m pytest tests/test_pure_functions.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_proxy():
    root = Path(__file__).resolve().parent.parent
    src = root / "gemma4_mlx_kilo_proxy.py"
    spec = importlib.util.spec_from_file_location("gemma4_mlx_kilo_proxy", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_mlx_kilo_proxy"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ag = _load_proxy()


# ---------------------------------------------------------------------------
# Fuzzy find
# ---------------------------------------------------------------------------

class TestFuzzyFind:
    def _write(self, tmp: str, name: str, content: str) -> str:
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_exact_match_returns_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = "def foo():\n    return 42\n"
            path = self._write(tmp, "f.py", content)
            result = ag._fuzzy_find(path, "def foo():\n    return 42")
            assert result == "def foo():\n    return 42"

    def test_whitespace_difference_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_content = "def bar():\n    x = 1\n    return x\n"
            path = self._write(tmp, "b.py", file_content)
            old_string = "def bar():\n  x = 1\n  return x"
            result = ag._fuzzy_find(path, old_string)
            assert result is not None
            assert "    x = 1" in result

    def test_below_threshold_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "x.py", "import os\nimport sys\n")
            result = ag._fuzzy_find(
                path, "class Totally_Different:\n    pass\n    pass\n    pass",
            )
            assert result is None

    def test_missing_file_returns_none(self):
        assert ag._fuzzy_find("/no/such/file.py", "x = 1") is None


# ---------------------------------------------------------------------------
# AskQuestion / TodoWrite repair
# ---------------------------------------------------------------------------

MALFORMED_ASK = json.dumps([{
    "header": "Target platform",
    "options": [
        {"description": "Web", "label": "Web version"},
        {"description": "CLI", "label": "Terminal version"},
    ],
    "multiple": False,
    "question": "Which platform?",
}])


class TestAskQuestionRepair:
    def test_stringified_questions(self):
        fixed = ag.repair_ask_question_args({"questions": MALFORMED_ASK})
        assert fixed["title"] == "Target platform"
        assert len(fixed["questions"]) == 1
        assert fixed["questions"][0]["prompt"] == "Which platform?"
        assert len(fixed["questions"][0]["options"]) == 2

    def test_bare_array_root(self):
        fixed = ag.repair_ask_question_args(json.loads(MALFORMED_ASK))
        assert fixed["title"] == "Target platform"


class TestTodoWriteRepair:
    def test_stringified_todos(self):
        todos = json.dumps([
            {"task": "Plan", "state": "wip"},
            {"description": "Implement", "status": "done"},
        ])
        fixed = ag.repair_todo_write_args({"todos": todos})
        assert len(fixed["todos"]) == 2
        assert fixed["todos"][0]["status"] == "in_progress"
        assert fixed["todos"][1]["status"] == "completed"
        assert fixed["todos"][0]["content"] == "Plan"

    def test_bare_array(self):
        fixed = ag.repair_todo_write_args([{"content": "A", "status": "pending"}])
        assert fixed["todos"][0]["content"] == "A"


# ---------------------------------------------------------------------------
# Tool remap
# ---------------------------------------------------------------------------

class TestToolRemap:
    def test_write_to_file_remapped(self):
        writers = {
            **ag._DEFAULT_WRITERS,
            "write_name": "write",
            "write_path_field": "path",
            "write_content_field": "content",
            "write_available": True,
            "tool_names": frozenset({"write", "StrReplace"}),
        }
        name, args = ag._remap_tool_call_name_and_args(
            "write_to_file",
            json.dumps({"path": "a.py", "content": "hi"}),
            writers,
        )
        assert name == "write"
        parsed = json.loads(args)
        assert parsed["path"] == "a.py"
        assert parsed["content"] == "hi"

    def test_gemma_write_casefold_to_Write(self):
        writers = {
            **ag._DEFAULT_WRITERS,
            "write_name": "Write",
            "write_path_field": "path",
            "write_content_field": "content",
            "tool_names": frozenset({"Write", "StrReplace", "TodoWrite"}),
        }
        name, args = ag.repair_tool_call(
            "write",
            json.dumps({"filePath": "/tmp/a.html", "content": "<html/>"}),
            writers,
        )
        assert name == "Write"
        parsed = json.loads(args)
        assert parsed["path"] == "/tmp/a.html"
        assert parsed["content"] == "<html/>"
        assert "filePath" not in parsed

    def test_repair_tool_call_todo(self):
        name, args = ag.repair_tool_call(
            "todowrite",
            json.dumps({"todos": [{"task": "x", "state": "pending"}]}),
        )
        assert name == "todowrite"
        parsed = json.loads(args)
        assert parsed["todos"][0]["content"] == "x"


# ---------------------------------------------------------------------------
# Compaction / agentic settings
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_tool_choice_none_is_compaction(self):
        assert ag._is_compaction_request({"tool_choice": "none"}) is True

    def test_prepare_strips_tools(self):
        body = {
            "tools": [{"type": "function", "function": {"name": "write"}}],
            "messages": [{"role": "system", "content": "You are helpful."}],
            "tool_choice": "auto",
        }
        # Force compaction path via tool_choice
        body["tool_choice"] = "none"
        ag._prepare_compaction_request(body)
        assert "tools" not in body
        assert body["tool_choice"] == "none"
        assert "plain text only" in body["messages"][0]["content"]


class TestAgenticSettings:
    def test_harmony_bias_injected(self):
        body = {
            "tools": [{"type": "function", "function": {"name": "write"}}],
            "temperature": 0,
        }
        ag._force_agentic_settings(body)
        assert body["temperature"] >= 0.35
        assert body["enable_thinking"] is False
        assert body["logit_bias"]["100"] == -100.0
        assert body["logit_bias"]["98"] == -100.0

    def test_non_agentic_untouched(self):
        body = {"temperature": 0.1}
        ag._force_agentic_settings(body)
        assert body["temperature"] == 0.1
        assert "logit_bias" not in body

    def test_caller_logit_bias_wins(self):
        body = {
            "tools": [{}],
            "logit_bias": {"100": 5.0},
        }
        ag._force_agentic_settings(body)
        assert body["logit_bias"]["100"] == 5.0


class TestStripPlanning:
    def test_strips_after_todo_without_write(self):
        body = {
            "tools": [
                {"type": "function", "function": {"name": "todowrite"}},
                {"type": "function", "function": {"name": "write"}},
            ],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {"name": "todowrite", "arguments": "{}"},
                    }],
                },
            ],
        }
        ag._strip_planning_tools_if_stuck(body)
        names = [(t.get("function") or {}).get("name") for t in body["tools"]]
        assert "todowrite" not in names
        assert "write" in names

    def test_keeps_planning_if_write_done(self):
        body = {
            "tools": [
                {"type": "function", "function": {"name": "todowrite"}},
                {"type": "function", "function": {"name": "write"}},
            ],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "todowrite", "arguments": "{}"}},
                        {"function": {"name": "write", "arguments": "{}"}},
                    ],
                },
            ],
        }
        ag._strip_planning_tools_if_stuck(body)
        names = [(t.get("function") or {}).get("name") for t in body["tools"]]
        assert "todowrite" in names


class TestTruncate:
    def test_long_tool_result_truncated(self):
        lines = "\n".join(f"line {i}" for i in range(500))
        msgs = [{"role": "tool", "content": lines, "tool_call_id": "1"}]
        out = ag._truncate_tool_results(msgs)
        assert len(out[0]["content"].splitlines()) < 500
        assert "truncated" in out[0]["content"]


class TestUpstreamPath:
    def test_with_v1_suffix(self):
        assert ag._upstream_api_path("http://127.0.0.1:8090/v1", "models") == "/models"

    def test_without_v1_suffix(self):
        assert ag._upstream_api_path("http://127.0.0.1:8090", "chat/completions") == (
            "/v1/chat/completions"
        )


class TestGuards:
    def test_delta_is_empty(self):
        assert ag._delta_is_empty({
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }) is True
        assert ag._delta_is_empty({
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        }) is False

    def test_graceful_stop_ends_cleanly(self):
        chunks = ag._graceful_stop_chunk("id1", "model1")
        assert chunks[-1] == b"data: [DONE]\n\n"
        payload = json.loads(chunks[0].decode().split("data: ", 1)[1])
        assert payload["choices"][0]["finish_reason"] == "stop"


if __name__ == "__main__":
    # Minimal runner without pytest.
    import traceback

    failures = 0
    for name, obj in list(globals().items()):
        if not (isinstance(obj, type) and name.startswith("Test")):
            continue
        inst = obj()
        for method_name in dir(inst):
            if not method_name.startswith("test_"):
                continue
            try:
                getattr(inst, method_name)()
                print(f"PASS  {name}.{method_name}")
            except Exception:
                failures += 1
                print(f"FAIL  {name}.{method_name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
