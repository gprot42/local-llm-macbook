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

SUMMARIZE_BAIT_SYSTEM = (
    "Do not re-summarize the conversation history. Preserve key information. "
    "Create a concise summary only if the user asks. agent=compaction is not active."
)

TOOLS_MIN = [
    {"type": "function", "function": {"name": "bash"}},
    {"type": "function", "function": {"name": "write"}},
]


class TestCompaction:
    def test_tool_choice_none_is_compaction(self):
        assert ag._is_compaction_request({"tool_choice": "none"}) is True

    def test_tool_choice_dict_none_is_compaction(self):
        assert ag._is_compaction_request({"tool_choice": {"type": "none"}}) is True

    def test_summarize_bait_system_with_tools_is_not_compaction(self):
        body = {
            "messages": [
                {"role": "system", "content": SUMMARIZE_BAIT_SYSTEM},
                {"role": "user", "content": "list files with tools"},
            ],
            "tools": TOOLS_MIN,
            "tool_choice": "auto",
        }
        assert ag._is_compaction_request(body) is False

    def test_user_summary_without_tools_is_compaction(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "Please summarize the conversation and preserve key information.",
                }
            ],
            "max_tokens": 2048,
        }
        assert ag._is_compaction_request(body) is True

    def test_tools_auto_never_compaction_even_if_user_says_summary(self):
        body = {
            "messages": [
                {"role": "system", "content": "agent"},
                {"role": "user", "content": "Write a summary of main.py using tools"},
            ],
            "tools": TOOLS_MIN,
            "tool_choice": "auto",
        }
        assert ag._is_compaction_request(body) is False

    def test_only_latest_user_message_used_for_text_hints(self):
        body = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": "Please summarize the conversation history from earlier.",
                },
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now list the repo with tools"},
            ],
            "tools": TOOLS_MIN,
            "tool_choice": "auto",
        }
        assert ag._is_compaction_request(body) is False

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

    def test_prepare_caps_max_tokens(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "Please summarize the conversation and preserve key information.",
                }
            ],
            "max_tokens": 8192,
            "tool_choice": "none",
        }
        ag._prepare_compaction_request(body)
        assert int(body["max_tokens"]) <= ag._COMPACTION_MAX_TOKENS_CEILING

    def test_prepare_flattens_history_tool_calls(self):
        body = {
            "tool_choice": "none",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo hi"}',
                            }
                        }
                    ],
                }
            ],
        }
        ag._prepare_compaction_request(body)
        assistant = next(m for m in body["messages"] if m.get("role") == "assistant")
        assert "tool_calls" not in assistant
        assert "bash" in assistant["content"]


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
        assert body["enable_thinking"] is False

    def test_thinking_disabled_without_tools(self):
        body = {
            "enable_thinking": True,
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": True, "foo": "bar"},
        }
        ag._disable_thinking(body)
        assert body["enable_thinking"] is False
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert body["chat_template_kwargs"]["foo"] == "bar"
        assert body["thinking"]["type"] == "disabled"

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


def _parse_sse_fixture(name: str) -> list[dict]:
    path = Path(__file__).resolve().parent / "fixtures" / name
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload and payload != "[DONE]":
            events.append(json.loads(payload))
    return events


class TestSseFixtures:
    def test_bash_stream_reassembles(self):
        events = _parse_sse_fixture("bash_tool_call.sse")
        calls = ag._reassemble_tool_calls(events)
        assert calls[0]["name"] == "bash"
        args = json.loads(calls[0]["arguments"])
        assert args["command"] == "ls -la"

    def test_hallucinated_write_remapped_on_repaired_stream(self):
        events = _parse_sse_fixture("hallucinated_write_tool.sse")
        writers = {
            **ag._DEFAULT_WRITERS,
            "write_name": "Write",
            "write_path_field": "path",
            "write_content_field": "content",
            "write_available": True,
            "tool_names": frozenset({"Write", "StrReplace"}),
        }
        chunks = ag._emit_repaired_stream(events, writers)
        payload = json.loads(chunks[0].decode().split("data: ", 1)[1])
        tc = payload["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "Write"
        args = json.loads(tc["function"]["arguments"])
        assert args["path"] == "/tmp/hello.txt"
        assert args["content"] == "Hello world"

    def test_text_response_has_no_tool_calls(self):
        events = _parse_sse_fixture("text_response.sse")
        assert ag._reassemble_tool_calls(events) == {}
        assert ag._stream_buffer_mode(events, ag._DEFAULT_WRITERS) is None


class TestNativeToolLeak:
    def test_broken_gemma_markup_is_a_leak(self):
        leaked = (
            '<tool_call|>\n'
            '→Read vendor/fastboot/fastboot.cpp\n'
            '<tool_call|>call:read{filePath:<|"|>/tmp/x.cpp<|"|>}'
        )
        assert ag.content_leaks_native_tools(leaked) is True

    def test_well_formed_native_call_is_a_leak(self):
        assert ag.content_leaks_native_tools(
            "<|tool_call>call:bash{\"command\":\"echo hi\"}"
        ) is True

    def test_normal_prose_is_not_a_leak(self):
        assert ag.content_leaks_native_tools("I will list the files next.") is False
        assert ag.content_leaks_native_tools("") is False


class TestWriteName:
    def test_todowrite_is_not_a_write(self):
        assert ag._is_write_tool_name("todowrite") is False
        assert ag._is_write_tool_name("todo_write") is False
        assert ag._is_write_tool_name("TodoWrite") is False

    def test_write_and_create_file_are_writes(self):
        assert ag._is_write_tool_name("write") is True
        assert ag._is_write_tool_name("Write") is True
        assert ag._is_write_tool_name("create_file") is True


class TestMessageText:
    def test_multimodal_text_blocks(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "world"},
            ],
        }
        assert ag._get_message_text(msg) == "hello\nworld"

    def test_latest_user_skips_trailing_assistant(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "later"},
        ]
        assert ag._latest_user_text(messages) == "second"


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


# ---------------------------------------------------------------------------
# Turn-stop: Gemma end-of-turn as a stop sequence + leaked-token strip
# ---------------------------------------------------------------------------

class TestTurnStop:
    def test_adds_stop_when_absent(self):
        body = {}
        ag._add_turn_stop(body)
        assert body["stop"] == ["<turn|>"]

    def test_appends_to_existing_list(self):
        body = {"stop": ["\n\n"]}
        ag._add_turn_stop(body)
        assert body["stop"] == ["\n\n", "<turn|>"]

    def test_idempotent(self):
        body = {"stop": ["<turn|>"]}
        ag._add_turn_stop(body)
        assert body["stop"] == ["<turn|>"]

    def test_promotes_string_stop(self):
        body = {"stop": "###"}
        ag._add_turn_stop(body)
        assert body["stop"] == ["###", "<turn|>"]

    def test_strips_leaked_turn_token_from_content(self):
        ev = {"choices": [{"index": 0, "delta": {"content": "Hi there!<turn|>"}}]}
        ag._strip_turn_stop(ev)
        assert ev["choices"][0]["delta"]["content"] == "Hi there!"

    def test_strip_leaves_tool_deltas_alone(self):
        ev = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}}]}
        ag._strip_turn_stop(ev)
        assert ev["choices"][0]["delta"] == {"tool_calls": [{"index": 0}]}

    def test_graceful_stop_reason_param(self):
        chunks = ag._graceful_stop_chunk("id1", "m", "tool_calls")
        assert b'"finish_reason": "tool_calls"' in chunks[0]
        assert chunks[-1] == b"data: [DONE]\n\n"

    def test_graceful_stop_default_reason_unchanged(self):
        chunks = ag._graceful_stop_chunk("id1", "m")
        assert b'"finish_reason": "stop"' in chunks[0]

    def test_strips_leaked_turn_token_from_nonstream_message(self):
        data = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG<turn|>"}}]}
        ag._strip_turn_stop_response(data)
        assert data["choices"][0]["message"]["content"] == "PONG"


# ---------------------------------------------------------------------------
# Text-collapse loop detector
# ---------------------------------------------------------------------------

_LOOP_BLOCK = [
    "<details>",
    "<summary>Plan</summary>",
    "1. Search for build files (Makefile, CMakeLists.txt, etc.).",
    "</details>",
    "Actually, I'll just use `glob` to find them.",
]


class TestTextRepeats:
    def test_detects_observed_planning_loop(self):
        # The real failure: a 5-line block repeated; 3 copies must trigger.
        lines = ["Intro line."] + _LOOP_BLOCK * 3
        assert ag._text_repeats(lines) == (5, 3)

    def test_two_copies_do_not_trigger(self):
        lines = ["Intro line."] + _LOOP_BLOCK * 2
        assert ag._text_repeats(lines) is None

    def test_repeated_closing_braces_in_code_are_not_a_loop(self):
        # Short/punctuation lines repeat legitimately in code.
        lines = ["int main() {", "  if (a) {", "  }", "}", "}", "}", "}", "}"]
        assert ag._text_repeats(lines) is None

    def test_single_line_needs_five_copies_and_substance(self):
        line = "This exact sentence keeps being repeated by the model again."
        assert ag._text_repeats([line] * 4) is None
        assert ag._text_repeats([line] * 5) == (1, 5)

    def test_normal_prose_is_clean(self):
        lines = [f"Point {i}: something different about file {i}.c" for i in range(20)]
        assert ag._text_repeats(lines) is None

    def test_ignores_blank_lines_between_copies(self):
        lines = []
        for _ in range(3):
            lines += _LOOP_BLOCK + ["", "   "]
        assert ag._text_repeats(lines) == (5, 3)


class TestTurnCeiling:
    def test_fresh_turn_is_fine(self):
        assert ag._turn_ceiling_reason(1.0, saw_useful=False, is_agentic=True) is None

    def test_prefill_window_allowed_before_no_output(self):
        # Big-context prefill: no useful output yet at 60s must not abort.
        assert ag._turn_ceiling_reason(60.0, saw_useful=False, is_agentic=True) is None

    def test_no_output_hang_caught(self):
        assert ag._turn_ceiling_reason(
            ag._NO_OUTPUT_MAX_S + 1, saw_useful=False, is_agentic=True
        ) == "no-output"

    def test_useful_output_survives_no_output_window(self):
        # Once useful content streamed, only the hard ceiling applies.
        assert ag._turn_ceiling_reason(
            ag._NO_OUTPUT_MAX_S + 1, saw_useful=True, is_agentic=True
        ) is None

    def test_hard_ceiling_agentic(self):
        assert ag._turn_ceiling_reason(
            ag._TURN_MAX_AGENTIC_S + 1, saw_useful=True, is_agentic=True
        ) == "hard"

    def test_hard_ceiling_plain_is_higher(self):
        # A plain-chat turn between the two hard ceilings is not yet hard-capped.
        mid = (ag._TURN_MAX_AGENTIC_S + ag._TURN_MAX_S) / 2
        assert ag._turn_ceiling_reason(mid, saw_useful=True, is_agentic=True) == "hard"
        assert ag._turn_ceiling_reason(mid, saw_useful=True, is_agentic=False) is None


class TestCollapseRepeats:
    def test_collapses_observed_indecisive_loop(self):
        block = (
            "Actually, I'll use `bash` to run `make` in `vendor/fastboot/exploit`.\n"
            "Actually, I'll check if they want to compile the actual fastboot.\n"
            "I'll just do it."
        )
        looped = "\n".join([block] * 5)
        assert ag._collapse_repeats(looped) == block

    def test_leading_text_then_loop(self):
        out = ag._collapse_repeats("Intro.\nA\nB\nA\nB\nA\nB")
        assert out == "Intro.\nA\nB"

    def test_no_repetition_is_unchanged(self):
        txt = "Line one.\nLine two.\nLine three."
        assert ag._collapse_repeats(txt) == txt

    def test_single_line_run_collapses(self):
        assert ag._collapse_repeats("go\ngo\ngo\ngo") == "go"

    def test_empty_string(self):
        assert ag._collapse_repeats("") == ""
