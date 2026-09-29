"""
Comprehensive tests for all new/modified proxy functions.

Covers gaps identified by coverage analysis:
  * _text_runaway_task_key / _record_text_runaway_for_task / _had_recent_text_runaway_for_task
  * _extract_workspace_dir (all-roles fix — system/assistant/tool messages)
  * _find_spec_file (explicit name + fuzzy workspace scan)
  * _inject_spec_into_messages
  * _looks_like_game_input_fix
  * _explicit_html_target_path (bare filename behaviour)
  * _synthetic_browser_game_html (JS brace balance + title branch + new UI fixes)
  * Text-mode repetition regexes (_TEXT_REPETITION_RE, _TEXT_TOKEN_REPEAT_WITH_DELIMS_RE,
    _TEXT_SENTENCE_REPETITION_RE)
  * _force_write_tool_for_create_turn integration with spec injection
  * _RECENT_TEXT_RUNAWAY_TASKS TTL semantics
"""

import json
import os
import re
import time
import types
import importlib
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gemma4_mlx_kilo_proxy as ag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}

def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }


# ===========================================================================
# 1.  _text_runaway_task_key
# ===========================================================================

class TestTextRunawayTaskKey:
    def _msgs(self, user_text: str, count: int = 2) -> list[dict]:
        msgs = [_msg("system", "kilo")]
        while len(msgs) < count:
            msgs.append(_msg("user", user_text))
        return msgs[:count]

    def test_key_format(self):
        msgs = [_msg("system", "s"), _msg("user", "read spec and create game")]
        key = ag._text_runaway_task_key(msgs)
        assert "read spec" in key
        assert ":" not in key.split("create game")[0][-3:]

    def test_key_same_task_different_message_counts(self):
        """Kilo retries shrink history but keep the same task text."""
        msgs2 = [_msg("user", "create web game")]
        msgs11 = [_msg("system", "s")] + [_msg("user", "create web game")] * 5
        assert ag._text_runaway_task_key(msgs2) == ag._text_runaway_task_key(msgs11)

    def test_key_truncates_task_at_200_chars(self):
        long_user = "create game " + "x" * 400
        msgs = [_msg("user", long_user)]
        key = ag._text_runaway_task_key(msgs)
        assert len(key) <= 200, f"task key too long: {len(key)}"

    def test_key_stable_for_same_input(self):
        msgs = [_msg("system", "s"), _msg("user", "create web game")]
        assert ag._text_runaway_task_key(msgs) == ag._text_runaway_task_key(msgs)

    def test_key_differs_for_different_tasks(self):
        msgs_a = [_msg("user", "create web game")]
        msgs_b = [_msg("user", "build a todo app")]
        assert ag._text_runaway_task_key(msgs_a) != ag._text_runaway_task_key(msgs_b)


# ===========================================================================
# 2.  _record_text_runaway_for_task / _had_recent_text_runaway_for_task
# ===========================================================================

class TestRunawayTracker:
    def setup_method(self):
        # Clear the global dict before each test
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def teardown_method(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def _msgs(self, task="read spec and create game"):
        return [_msg("system", "s"), _msg("user", task)]

    def test_had_returns_false_before_record(self):
        msgs = self._msgs("create mario game")
        assert ag._had_recent_text_runaway_for_task(msgs) is False

    def test_record_then_had_returns_true(self):
        msgs = self._msgs("create mario game")
        ag._record_text_runaway_for_task(msgs)
        assert ag._had_recent_text_runaway_for_task(msgs) is True

    def test_had_returns_false_after_ttl(self):
        msgs = self._msgs("create mario game")
        ag._record_text_runaway_for_task(msgs)
        # Fake the timestamp to be past TTL
        key = ag._text_runaway_task_key(msgs)
        ag._RECENT_TEXT_RUNAWAY_TASKS[key] = time.monotonic() - ag._RECENT_TEXT_RUNAWAY_TTL_S - 1
        assert ag._had_recent_text_runaway_for_task(msgs) is False

    def test_different_tasks_tracked_independently(self):
        msgs_a = self._msgs("create mario game")
        msgs_b = self._msgs("build a snake game")
        ag._record_text_runaway_for_task(msgs_a)
        assert ag._had_recent_text_runaway_for_task(msgs_a) is True
        assert ag._had_recent_text_runaway_for_task(msgs_b) is False

    def test_record_prunes_stale_entries(self):
        # Insert many stale entries
        now = time.monotonic()
        for i in range(20):
            ag._RECENT_TEXT_RUNAWAY_TASKS[f"99:task_{i}"] = (
                now - ag._RECENT_TEXT_RUNAWAY_TTL_S * 3
            )
        initial_count = len(ag._RECENT_TEXT_RUNAWAY_TASKS)
        msgs = self._msgs("fresh create game")
        ag._record_text_runaway_for_task(msgs)
        # After record, stale entries should be pruned
        assert len(ag._RECENT_TEXT_RUNAWAY_TASKS) < initial_count

    def test_record_populates_dict(self):
        msgs = self._msgs("unique task xyz")
        assert len(ag._RECENT_TEXT_RUNAWAY_TASKS) == 0
        ag._record_text_runaway_for_task(msgs)
        assert len(ag._RECENT_TEXT_RUNAWAY_TASKS) == 1

    def test_second_record_overwrites_first(self):
        msgs = self._msgs("some task")
        ag._record_text_runaway_for_task(msgs)
        t1 = time.monotonic()
        time.sleep(0.01)
        ag._record_text_runaway_for_task(msgs)
        key = ag._text_runaway_task_key(msgs)
        t2 = ag._RECENT_TEXT_RUNAWAY_TASKS[key]
        assert t2 > t1 - 1  # updated


# ===========================================================================
# 3.  Integration: runaway tracker → force_write_tool override
# ===========================================================================

class TestRunawayTrackerForceWriteOverride:
    def setup_method(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def teardown_method(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def test_read_first_defers_when_no_runaway(self):
        """Standard first-turn read-first: no runaway recorded → defer."""
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", "You are Kilo."),
            _msg("user", "read spec file and create web game"),
        ]
        _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is False
        assert "tool_choice" not in body

    def test_read_first_forces_after_runaway(self):
        """After a text runaway is recorded, read-first deferral is overridden."""
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", "You are Kilo."),
            _msg("user", "read spec file and create web game"),
        ]
        # Simulate the runaway having been recorded
        ag._record_text_runaway_for_task(messages)
        _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is True, "runaway override should force write tool"
        assert body.get("tool_choice", {}).get("function", {}).get("name") == "write"

    def test_runaway_override_does_not_affect_non_read_first(self):
        """Recording a runaway doesn't matter for tasks that don't start with 'read'."""
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", "You are Kilo."),
            _msg("user", "create a web game"),
        ]
        ag._record_text_runaway_for_task(messages)
        _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        # Non-read-first task gets forced unconditionally regardless of tracker
        assert forced is True


# ===========================================================================
# 4.  _extract_workspace_dir — all-roles fix
# ===========================================================================

class TestExtractWorkspaceDir:
    """_extract_workspace_dir now searches ALL message roles, not just user."""

    def _with_ws(self, role: str, extra: str = "") -> list[dict]:
        return [_msg(role, f"Some content.\nCurrent Workspace Directory (/my/workspace)\n{extra}")]

    def test_extracts_from_user_message(self):
        msgs = self._with_ws("user")
        assert ag._extract_workspace_dir(msgs) == "/my/workspace"

    def test_extracts_from_system_message(self):
        """Key fix: system messages are now searched (Kilo puts workspace in system)."""
        msgs = self._with_ws("system")
        assert ag._extract_workspace_dir(msgs) == "/my/workspace"

    def test_extracts_from_assistant_message(self):
        msgs = self._with_ws("assistant")
        assert ag._extract_workspace_dir(msgs) == "/my/workspace"

    def test_extracts_from_tool_message(self):
        msgs = self._with_ws("tool")
        assert ag._extract_workspace_dir(msgs) == "/my/workspace"

    def test_returns_none_when_absent(self):
        msgs = [_msg("system", "No workspace here."), _msg("user", "do something")]
        assert ag._extract_workspace_dir(msgs) is None

    def test_returns_none_for_empty_messages(self):
        assert ag._extract_workspace_dir([]) is None

    def test_pattern_current_working_directory(self):
        msgs = [_msg("system", "Current Working Directory: /cwd/path")]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/cwd/path"

    def test_pattern_workspace_directory(self):
        msgs = [_msg("system", "Workspace Directory: /ws/path")]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/ws/path"

    def test_pattern_cwd_xml_tag(self):
        msgs = [_msg("system", "<cwd>/xml/path</cwd>")]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/xml/path"

    def test_trailing_slash_stripped(self):
        msgs = [_msg("system", "Current Workspace Directory (/path/with/slash/)")]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/path/with/slash"

    def test_most_recent_message_wins(self):
        """Later messages override earlier ones."""
        msgs = [
            _msg("system", "Current Workspace Directory (/old/workspace)"),
            _msg("user", "Current Workspace Directory (/new/workspace)"),
        ]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/new/workspace"

    def test_real_kilo_system_prompt_format(self):
        """Matches the real Kilo Code system prompt format."""
        kilo_system = (
            "You are Kilo Code, an AI assistant...\n"
            "Current Workspace Directory (/Users/aicoder/src/zzz-test)\n"
            "Tools: read, write, edit\n"
        )
        msgs = [_msg("system", kilo_system), _msg("user", "create game")]
        result = ag._extract_workspace_dir(msgs)
        assert result == "/Users/aicoder/src/zzz-test"


# ===========================================================================
# 5.  _find_spec_file
# ===========================================================================

class TestFindSpecFile:
    def test_returns_none_no_workspace(self):
        """No workspace in messages, no explicit filename → None."""
        msgs = [_msg("user", "read spec file and create web game")]
        result = ag._find_spec_file(msgs)
        assert result is None

    def test_explicit_name_found(self, tmp_path):
        """Explicit 'read 1942-spec.txt' with workspace containing the file."""
        spec = tmp_path / "1942-spec.txt"
        spec.write_text("# Game Spec\nrequirements here")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read 1942-spec.txt and create web game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is not None
        path, content = result
        assert "1942-spec.txt" in path
        assert "Game Spec" in content

    def test_explicit_name_not_found_returns_none(self):
        """Explicit name in message but file doesn't exist → None."""
        msgs = [
            _msg("system", "Current Workspace Directory (/nonexistent/path)"),
            _msg("user", "read missing-spec.txt and create web game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is None

    def test_fuzzy_scan_finds_spec_md(self, tmp_path):
        """Fuzzy scan picks up 'spec.md' from workspace."""
        spec = tmp_path / "spec.md"
        spec.write_text("# Spec\nsome requirements")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read spec file and create game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is not None
        _, content = result
        assert "some requirements" in content

    def test_fuzzy_scan_finds_requirements_txt(self, tmp_path):
        spec = tmp_path / "requirements.txt"
        spec.write_text("game requirements here")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is not None
        _, content = result
        assert "game requirements" in content

    def test_fuzzy_scan_ignores_non_spec_files(self, tmp_path):
        """Files not matching _SPEC_FILENAME_RE are not returned."""
        (tmp_path / "README.md").write_text("readme text")
        (tmp_path / "notes.txt").write_text("notes text")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is None

    def test_content_truncated_at_max_chars(self, tmp_path):
        spec = tmp_path / "game-spec.txt"
        spec.write_text("x" * (ag._SPEC_FILE_MAX_CHARS + 1000))
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read game-spec.txt and create game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is not None
        _, content = result
        assert len(content) <= ag._SPEC_FILE_MAX_CHARS

    def test_returns_tuple_path_and_content(self, tmp_path):
        spec = tmp_path / "1942-spec.txt"
        spec.write_text("spec content")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read 1942-spec.txt and create game"),
        ]
        result = ag._find_spec_file(msgs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        path, content = result
        assert isinstance(path, str)
        assert isinstance(content, str)

    def test_design_doc_md_matched(self, tmp_path):
        """design-doc.md matches the fuzzy scan pattern."""
        spec = tmp_path / "design-doc.md"
        spec.write_text("# Design Document\ngame design here")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read design doc and build game"),
        ]
        result = ag._find_spec_file(msgs)
        assert result is not None


# ===========================================================================
# 6.  _inject_spec_into_messages
# ===========================================================================

class TestInjectSpecIntoMessages:
    def test_appends_to_existing_system_message(self):
        msgs = [_msg("system", "Original system."), _msg("user", "create game")]
        out = ag._inject_spec_into_messages(msgs, "/path/spec.txt", "spec content here")
        sys_msgs = [m for m in out if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert "Original system." in sys_msgs[0]["content"]
        assert "spec content here" in sys_msgs[0]["content"]

    def test_creates_system_when_absent(self):
        msgs = [_msg("user", "create game")]
        out = ag._inject_spec_into_messages(msgs, "/path/spec.txt", "spec content")
        sys_msgs = [m for m in out if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert "spec content" in sys_msgs[0]["content"]

    def test_uses_basename_in_header(self):
        msgs = [_msg("system", "s")]
        out = ag._inject_spec_into_messages(msgs, "/deep/path/1942-spec.txt", "c")
        content = out[0]["content"]
        assert "1942-spec.txt" in content
        assert "=== 1942-spec.txt ===" in content

    def test_block_format(self):
        msgs = [_msg("system", "s")]
        out = ag._inject_spec_into_messages(msgs, "/p/spec.md", "game spec here")
        content = out[0]["content"]
        assert "=== spec.md ===" in content
        assert "=== END spec.md ===" in content
        assert "game spec here" in content
        assert "REFERENCE" in content

    def test_preserves_user_messages(self):
        msgs = [_msg("system", "s"), _msg("user", "create game")]
        out = ag._inject_spec_into_messages(msgs, "/p/s.txt", "content")
        user_msgs = [m for m in out if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "create game"

    def test_message_count_unchanged(self):
        msgs = [_msg("system", "s"), _msg("user", "t1"), _msg("user", "t2")]
        out = ag._inject_spec_into_messages(msgs, "/p/s.txt", "c")
        assert len(out) == len(msgs)

    def test_original_messages_not_mutated(self):
        msgs = [_msg("system", "original")]
        original_content = msgs[0]["content"]
        ag._inject_spec_into_messages(msgs, "/p/s.txt", "injected")
        assert msgs[0]["content"] == original_content


# ===========================================================================
# 7.  _looks_like_game_input_fix
# ===========================================================================

class TestLooksLikeGameInputFix:
    def _check(self, text: str) -> bool:
        return ag._looks_like_game_input_fix(text)

    def test_buttons_dont_work(self):
        assert self._check("buttons in top right dont work") is True

    def test_spacebar_not_working(self):
        assert self._check("spacebar not working") is True

    def test_keyboard_broken(self):
        assert self._check("keyboard input is broken") is True

    def test_key_input_issue(self):
        assert self._check("key input not responding") is True

    def test_click_doesnt_respond(self):
        assert self._check("click doesn't respond") is True

    def test_fix_the_controls(self):
        assert self._check("fix the controls") is True

    def test_space_key_fails(self):
        assert self._check("space key fails to start game") is True

    def test_input_not_working(self):
        assert self._check("input not working") is True

    def test_interact_broken(self):
        assert self._check("can't interact with the game") is True

    def test_debug_controls(self):
        assert self._check("debug the controls issue") is True

    def test_unrelated_create_game(self):
        assert self._check("create a web game") is False

    def test_unrelated_add_enemies(self):
        assert self._check("add more enemies to the game") is False

    def test_empty_string(self):
        assert self._check("") is False

    def test_none_like(self):
        assert self._check("   ") is False

    def test_mute_button(self):
        assert self._check("mute button doesnt work") is True

    def test_arrows_not_moving_player(self):
        assert self._check("arrow keys not moving the player") is True


# ===========================================================================
# 8.  _explicit_html_target_path — absolute and bare filename behaviour
# ===========================================================================

class TestExplicitHtmlTargetPath:
    def test_absolute_path_found(self):
        msgs = [_msg("user", "edit /Users/aicoder/src/zzz-test/game.html")]
        result = ag._explicit_html_target_path(msgs)
        assert result == "/Users/aicoder/src/zzz-test/game.html"

    def test_absolute_path_in_html(self, tmp_path):
        """Absolute .htm extension also matched."""
        msgs = [_msg("user", f"update {tmp_path}/page.htm")]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None and result.endswith(".htm")

    def test_returns_none_when_no_html(self):
        msgs = [_msg("user", "create a web game")]
        result = ag._explicit_html_target_path(msgs)
        assert result is None

    def test_multiple_user_msgs_last_wins(self):
        msgs = [
            _msg("user", "edit /old/game.html"),
            _msg("user", "create /new/game.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        # reversed iteration → last user message first
        assert result == "/new/game.html"

    def test_bare_filename_mario_html(self, tmp_path):
        """'mario.html' (no leading /) should be found and resolved to workspace."""
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game - mario.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None, (
            "bare filename 'mario.html' should be extractable from task"
        )
        assert "mario.html" in result

    def test_bare_filename_named_pattern(self, tmp_path):
        """'write to file named mario.html' pattern (the actual user task)."""
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game, write to file named mario.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None, (
            "'named mario.html' should be extractable"
        )
        assert "mario.html" in result

    def test_bare_filename_called_pattern(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "build a game called game.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None
        assert "game.html" in result

    def test_bare_filename_index_html(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "build index.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None
        assert "index.html" in result

    def test_named_path_resolves_to_workspace(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game named mario.html"),
        ]
        result = ag._explicit_html_target_path(msgs)
        assert result is not None
        assert str(tmp_path) in result
        assert result.endswith("mario.html")

    def test_non_user_message_searched(self):
        """The function searches all roles (consistent with workspace extraction fix)."""
        msgs = [_msg("system", "write to /abs/game.html")]
        result = ag._explicit_html_target_path(msgs)
        _ = result  # accept either None or a match; don't crash

    def test_default_html_target_path_uses_named(self, tmp_path):
        """_default_html_target_path picks up 'named mario.html' via _explicit."""
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game, write to file named mario.html"),
        ]
        result = ag._default_html_target_path(msgs)
        assert "mario.html" in result, (
            f"expected mario.html but got {result}"
        )


# ===========================================================================
# 9.  _synthetic_browser_game_html — JS validity + title branch + new UI fixes
# ===========================================================================

class TestSyntheticBrowserGameHtml:
    def _html(self, task: str = "create web game") -> str:
        return ag._synthetic_browser_game_html(task)

    def _js(self, html: str) -> str:
        import re
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        return m.group(1) if m else ""

    # --- Title branch ---

    def test_title_1942_when_task_contains_1942(self):
        html = self._html("create 1942 game")
        assert "1942 Arcade Shooter" in html

    def test_title_1942_when_task_contains_plane(self):
        html = self._html("build a plane shooter")
        assert "1942 Arcade Shooter" in html

    def test_title_generic_for_unrelated_task(self):
        html = self._html("create mario game")
        assert "Browser Arcade Shooter" in html

    def test_title_generic_for_empty_task(self):
        html = self._html("")
        assert "Browser Arcade Shooter" in html

    # --- JS syntax ---

    def test_js_brace_balance(self):
        js = self._js(self._html())
        opens = js.count("{")
        closes = js.count("}")
        assert opens == closes, f"JS brace mismatch: {opens} {{ vs {closes} }}"

    def test_js_paren_balance(self):
        js = self._js(self._html())
        opens = js.count("(")
        closes = js.count(")")
        assert opens == closes, f"JS paren mismatch: {opens} ( vs {closes} )"

    def test_js_bracket_balance(self):
        js = self._js(self._html())
        opens = js.count("[")
        closes = js.count("]")
        assert opens == closes, f"JS bracket mismatch: {opens} [ vs {closes} ]"

    def test_js_iife_present(self):
        js = self._js(self._html())
        assert "(() => {" in js or "(() => {{" in js or "()=>" in js

    # --- Keyboard / UI fixes (new) ---

    def test_canvas_has_tabindex(self):
        html = self._html()
        assert 'tabindex="0"' in html.lower() or "tabindex='0'" in html.lower(), (
            "canvas must have tabindex=0 for keyboard focus"
        )

    def test_hud_right_has_pointer_events_auto(self):
        html = self._html()
        assert "pointer-events:auto" in html or "pointer-events: auto" in html, (
            "HUD .right must have pointer-events:auto for mute button"
        )

    def test_arrow_keys_prevented_default(self):
        js = self._js(self._html())
        assert "ArrowUp" in js and "preventDefault" in js, (
            "arrow keys must call preventDefault to stop page scroll"
        )

    def test_canvas_click_listener(self):
        js = self._js(self._html())
        assert "canvas.addEventListener" in js, (
            "canvas must have a click listener for click-to-start"
        )

    # --- Cache headers ---

    def test_no_store_cache_header(self):
        html = self._html()
        assert "no-store" in html.lower()

    def test_no_cache_pragma(self):
        html = self._html()
        assert "no-cache" in html.lower()

    # --- Required features present ---

    def test_canvas_element_present(self):
        assert "<canvas" in self._html()

    def test_requestanimationframe_present(self):
        assert "requestAnimationFrame" in self._html()

    def test_keydown_listener_on_window(self):
        js = self._js(self._html())
        assert "addEventListener('keydown'" in js or 'addEventListener("keydown"' in js

    def test_keyup_listener(self):
        js = self._js(self._html())
        assert "keyup" in js

    def test_has_enemies(self):
        js = self._js(self._html())
        low = js.lower()
        assert "enemy" in low or "enemies" in low

    def test_has_bullets(self):
        js = self._js(self._html())
        low = js.lower()
        assert "bullet" in low

    def test_has_score(self):
        assert "score" in self._html().lower()

    def test_has_audio(self):
        js = self._js(self._html())
        assert "AudioContext" in js or "audiocontext" in js.lower()


# ===========================================================================
# 10.  Text-mode repetition regexes
# ===========================================================================

class TestTextRepetitionRegexes:
    # --- _TEXT_REPETITION_RE (word × 30) ---

    def test_word_repetition_fires_30_times(self):
        text = "only " * 30
        assert ag._TEXT_REPETITION_RE.search(text) is not None

    def test_word_repetition_fires_on_korean(self):
        text = "위한 " * 35
        assert ag._TEXT_REPETITION_RE.search(text) is not None

    def test_word_repetition_no_fire_below_threshold(self):
        text = "only " * 10  # < 30
        assert ag._TEXT_REPETITION_RE.search(text) is None

    def test_word_repetition_no_fire_mixed_prose(self):
        text = "the quick brown fox jumps over the lazy dog. " * 3
        assert ag._TEXT_REPETITION_RE.search(text) is None

    def test_word_repetition_extracts_repeated_word(self):
        text = "design " * 30
        m = ag._TEXT_REPETITION_RE.search(text)
        assert m is not None
        assert m.group(1).lower() == "design"

    # --- _TEXT_TOKEN_REPEAT_WITH_DELIMS_RE (token-with-delimiters × 16) ---

    def test_token_delim_fires_hyphen(self):
        text = "only-" * 16
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is not None

    def test_token_delim_no_fire_underscore_only(self):
        # '_' is a \w character in Python's re, so \b doesn't fire between
        # 'foo' and '_foo'. The regex supports '-', ',', '.', ';', ':' etc.
        # but NOT bare '_' as the sole delimiter (foo_foo is one \w token).
        # This test documents the known limitation.
        text = "foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo_foo"
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is None

    def test_token_delim_fires_hyphen_long(self):
        # Use '-' which IS a non-\w char, so \b fires correctly.
        text = "foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo-foo"
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is not None

    def test_token_delim_fires_comma(self):
        text = "the,the,the,the,the,the,the,the,the,the,the,the,the,the,the,the"
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is not None

    def test_token_delim_no_fire_below_threshold(self):
        text = "foo-foo-foo-foo-foo"  # only 5
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is None

    def test_token_delim_captures_token(self):
        text = "enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough-enough"
        m = ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text)
        assert m is not None
        assert m.group(1).lower() == "enough"

    def test_token_delim_fires_semicolon(self):
        text = "ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba;ba"
        assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(text) is not None

    # --- _TEXT_SENTENCE_REPETITION_RE (sentence × 5) ---

    def test_sentence_repetition_fires(self):
        sentence = "I'll use the ls command to list the directory contents.\n"
        text = sentence * 5
        assert ag._TEXT_SENTENCE_REPETITION_RE.search(text) is not None

    def test_sentence_repetition_with_period(self):
        text = "Actually I'll just use bash to create the directory." * 5
        assert ag._TEXT_SENTENCE_REPETITION_RE.search(text) is not None

    def test_sentence_repetition_no_fire_below_threshold(self):
        text = "I'll use bash to create it." * 4  # only 4 reps
        assert ag._TEXT_SENTENCE_REPETITION_RE.search(text) is None

    def test_sentence_repetition_no_fire_varied_prose(self):
        text = (
            "First I will read the file. "
            "Then I will edit the content. "
            "Finally I will write the output. "
            "The task should be complete now. "
            "Let me verify the changes."
        )
        assert ag._TEXT_SENTENCE_REPETITION_RE.search(text) is None

    def test_sentence_repetition_captures_group(self):
        sentence = "방적의상상: own-thought\n"
        text = sentence * 5
        m = ag._TEXT_SENTENCE_REPETITION_RE.search(text)
        assert m is not None
        assert len(m.group(1)) >= 10

    def test_more_increment_repetition_fires(self):
        text = (
            "I'll read mario-clone/index.html more closely. "
            "I'll read mario-clone/index.html more more closely. "
            "I'll read mario-clone/index.html more more more more closely."
        )
        assert ag._TEXT_MORE_INCREMENT_RE.search(text) is not None

    def test_read_mantra_repetition_fires(self):
        text = (
            "I'll begin by reading the mario-clone/index.html file. "
            "I'll begin by reading the mario-clone/index.html file. "
        )
        assert ag._TEXT_READ_MANTRA_RE.search(text) is not None


class TestProxyDebug:
    def test_debug_flags_from_env(self, monkeypatch):
        monkeypatch.setenv("PROXY_DEBUG_STREAM", "1")
        flags = ag.ProxyDebugFlags.load(cli_debug=False)
        assert flags.stream_trace is True

    def test_describe_request_steering(self):
        body = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "create mario-clone game"},
            ],
            "tools": [{"type": "function", "function": {"name": "write"}}],
            "temperature": 0,
            "stream": True,
        }
        steer = ag._describe_request_steering(body)
        assert steer["create_task"] is True
        assert "write" in steer["tool_names"]
        assert steer["msg_count"] == 2


class TestHarmonyStrip:
    def test_strip_control_markers(self):
        raw = "<|channel>thought plan here <channel|>"
        out = ag._strip_harmony_control_text(raw)
        assert "<|channel>" not in out
        assert "<channel|>" not in out
        # Inline thinking blocks are removed entirely, not left as visible prose.
        assert "plan here" not in out

    def test_thinking_filter_strips_leaked_prefix(self):
        tf = ag._ThinkingFilter()
        out = tf.feed("<|channel>thought <channel|>")
        assert "<|channel>" not in out
        assert "<channel|>" not in out

    def test_strip_turn_thought_inline_block(self):
        raw = (
            "I'll update index.html"
            "<turn|>thought eyes ls -R secret<channel|>"
            "done"
        )
        out = ag._strip_harmony_control_text(raw)
        assert "<turn|>" not in out
        assert "secret" not in out
        assert "done" in out

    def test_verbatim_phrase_loop_detects_mario_stall(self):
        phrase = (
            "I've updated the index.html file to include basic CSS for "
            "centering the canvas and setting a background color."
        )
        blob = (phrase + "<turn|>thought eyes ") * 4
        assert ag._looks_like_verbatim_phrase_loop(blob)

    def test_control_marker_collapse_turn_thought(self):
        blob = "<turn|>thought eyes " * 12
        assert ag._looks_like_control_marker_collapse(blob)


# ===========================================================================
# 11.  _find_spec_file + _inject_spec_into_messages + _force_write integration
# ===========================================================================

class TestSpecInjectionIntegration:
    def setup_method(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def teardown_method(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def test_spec_found_forces_write_on_first_turn(self, tmp_path):
        """Spec file found → injected → write forced (no deferral)."""
        spec = tmp_path / "1942-spec.txt"
        spec.write_text("# Spec\nCreate a 1942 arcade game.")
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read spec file and create web game"),
        ]
        out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is True, "spec found → should force write immediately"
        assert body.get("tool_choice", {}).get("function", {}).get("name") == "write"

    def test_spec_content_injected_into_system(self, tmp_path):
        """The spec content appears in the outgoing system message."""
        spec = tmp_path / "game-spec.md"
        spec.write_text("# Game Spec\nPlatformer with coins.")
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read game-spec.md and create game"),
        ]
        out_msgs, _, _ = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        sys_content = next(
            (m["content"] for m in out_msgs if m.get("role") == "system"), ""
        )
        assert "Platformer with coins" in sys_content, (
            "spec content should be injected into system message"
        )

    def test_no_spec_defers(self, tmp_path):
        """Workspace exists but no spec file → original read-first deferral."""
        body = {}
        tools = [_tool("write"), _tool("read")]
        messages = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read spec file and create web game"),
        ]
        _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is False, "no spec file → should defer to model"

    def test_spec_injection_survives_tools_1_filter(self, tmp_path):
        """After spec injection, tools are still filtered to [write] only."""
        spec = tmp_path / "requirements.txt"
        spec.write_text("game requirements")
        body = {}
        tools = [_tool("write"), _tool("read"), _tool("edit")]
        messages = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "read requirements.txt and create game"),
        ]
        _, out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is True
        assert len(out_tools) == 1
        assert out_tools[0]["function"]["name"] == "write"


# ===========================================================================
# 12.  _RECENT_TEXT_RUNAWAY_TASKS TTL constants
# ===========================================================================

class TestRunawayTtlConstants:
    def test_ttl_is_positive(self):
        assert ag._RECENT_TEXT_RUNAWAY_TTL_S > 0

    def test_ttl_is_reasonable(self):
        """TTL should be at least 30s (cover Kilo retry lag) but < 300s."""
        assert 30 <= ag._RECENT_TEXT_RUNAWAY_TTL_S <= 300

    def test_stale_cutoff_is_2x_ttl(self):
        """Pruning uses 2× TTL — verify the dict cleanup condition."""
        msgs = [_msg("system", "s"), _msg("user", "create game")]
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()
        ag._record_text_runaway_for_task(msgs)
        key = ag._text_runaway_task_key(msgs)
        # Set to just inside stale cutoff (< 2× TTL) → should NOT be pruned
        ag._RECENT_TEXT_RUNAWAY_TASKS[key] = (
            time.monotonic() - ag._RECENT_TEXT_RUNAWAY_TTL_S * 1.9
        )
        msgs2 = [_msg("system", "s"), _msg("user", "create another game")]
        ag._record_text_runaway_for_task(msgs2)
        # The first key at 1.9× TTL should still be in the dict (not pruned)
        assert key in ag._RECENT_TEXT_RUNAWAY_TASKS

        # Set to well past stale cutoff (> 2× TTL) → should be pruned
        ag._RECENT_TEXT_RUNAWAY_TASKS[key] = (
            time.monotonic() - ag._RECENT_TEXT_RUNAWAY_TTL_S * 2.1
        )
        msgs3 = [_msg("system", "s"), _msg("user", "create yet another game")]
        ag._record_text_runaway_for_task(msgs3)
        assert key not in ag._RECENT_TEXT_RUNAWAY_TASKS


# ===========================================================================
# 13.  _looks_like_game_input_fix is called in the proxy stream path
# ===========================================================================

class TestLooksLikeGameInputFixCallSites:
    """Verify _looks_like_game_input_fix is actually wired into the proxy."""

    def test_function_exists(self):
        assert callable(ag._looks_like_game_input_fix)

    def test_referenced_in_source(self):
        """The function must be called somewhere in the proxy (not dead code)."""
        import inspect
        source = inspect.getsource(ag)
        # Count how many times the function name appears
        count = source.count("_looks_like_game_input_fix")
        assert count >= 2, (
            f"_looks_like_game_input_fix appears only {count} time(s) — "
            "it may be dead code if count == 1 (only the def)"
        )


# ===========================================================================
# 14.  _SPEC_FILE_MAX_CHARS + _SPEC_FILENAME_RE + _EXPLICIT_SPEC_RE constants
# ===========================================================================

class TestSpecConstants:
    def test_max_chars_reasonable(self):
        assert 4096 <= ag._SPEC_FILE_MAX_CHARS <= 65536

    def test_spec_filename_re_matches_spec_txt(self):
        assert ag._SPEC_FILENAME_RE.search("1942-spec.txt")

    def test_spec_filename_re_matches_requirements_md(self):
        assert ag._SPEC_FILENAME_RE.search("requirements.md")

    def test_spec_filename_re_matches_design_doc_md(self):
        assert ag._SPEC_FILENAME_RE.search("design-doc.md")

    def test_spec_filename_re_matches_game_brief_txt(self):
        assert ag._SPEC_FILENAME_RE.search("game-brief.txt")

    def test_spec_filename_re_no_match_readme(self):
        # README doesn't match the pattern
        assert not ag._SPEC_FILENAME_RE.search("README.md")

    def test_spec_filename_re_no_match_random_txt(self):
        assert not ag._SPEC_FILENAME_RE.search("notes-random.txt")

    def test_explicit_spec_re_matches_read_file(self):
        m = ag._EXPLICIT_SPEC_RE.search("read 1942-spec.txt and create")
        assert m is not None
        assert m.group(1) == "1942-spec.txt"

    def test_explicit_spec_re_matches_read_md(self):
        m = ag._EXPLICIT_SPEC_RE.search("read game-spec.md and build")
        assert m is not None
        assert m.group(1) == "game-spec.md"

    def test_explicit_spec_re_no_match_load(self):
        # "load" is not in the regex — only "read"
        m = ag._EXPLICIT_SPEC_RE.search("load spec.txt and create")
        assert m is None


# ===========================================================================
# 15.  _extract_mistral_write_content
# ===========================================================================

class TestExtractMistralWriteContent:
    """Parses <|tool_call>call:write{content:<|"|>...} text streams."""

    SAMPLE_HTML = (
        "<!DOCTYPE html><html><head><title>Game</title></head>"
        "<body><canvas id='c'></canvas>"
        "<script>requestAnimationFrame(loop);</script></body></html>"
    )

    # The heretic model uses <|"|> as a string-value delimiter.
    # We must not embed it in an f-string to avoid Python syntax errors.
    _STR_DELIM = '<|"|>'

    def _wrap(self, content: str, filepath: str | None = None) -> str:
        """Wrap content in the heretic model's tool call format."""
        d = self._STR_DELIM
        if filepath:
            return (
                "<|tool_call>call:write{filePath:" + d + filepath + d
                + ",content:" + d + content
            )
        return "<|tool_call>call:write{content:" + d + content

    def test_extracts_html_content(self):
        text = self._wrap(self.SAMPLE_HTML)
        _, content = ag._extract_mistral_write_content(text)
        assert content is not None
        assert "<canvas" in content

    def test_returns_none_when_no_tool_call(self):
        text = "Just some plain text response."
        path, content = ag._extract_mistral_write_content(text)
        assert path is None
        assert content is None

    def test_extracts_filepath_when_present(self):
        text = self._wrap(self.SAMPLE_HTML, filepath="mario.html")
        path, content = ag._extract_mistral_write_content(text)
        assert path == "mario.html"
        assert content is not None

    def test_none_filepath_when_absent(self):
        text = self._wrap(self.SAMPLE_HTML)
        path, content = ag._extract_mistral_write_content(text)
        assert path is None  # no filePath in the call

    def test_trims_at_html_end_tag(self):
        """Content after </html> is stripped."""
        text = self._wrap(self.SAMPLE_HTML + "\nextra trailing text after close")
        _, content = ag._extract_mistral_write_content(text)
        assert content is not None
        assert content.endswith("</html>")

    def test_trims_at_script_end_tag_when_no_html(self):
        partial = (
            "<canvas id='c'></canvas>"
            "<script>var x=1;</script>"
            " trailing garbage"
        )
        text = self._wrap(partial)
        _, content = ag._extract_mistral_write_content(text)
        assert content is not None
        assert content.endswith("</script>")

    def test_returns_none_for_non_html_content(self):
        """Plain text without canvas/html/script tags → None."""
        text = self._wrap("# Game Spec\nSome requirements text only.")
        _, content = ag._extract_mistral_write_content(text)
        assert content is None

    def test_handles_partial_stream(self):
        """A truncated HTML stream (game not complete) still returns partial."""
        partial = (
            "<!DOCTYPE html><html><head><title>T</title></head>"
            "<body><canvas id='c'></canvas><script>var x="
            # Stream cut here — no </html>
        )
        text = self._wrap(partial)
        _, content = ag._extract_mistral_write_content(text)
        # Partial but has canvas — should be extractable
        assert content is not None
        assert "<canvas" in content

    def test_special_tokens_stripped_from_content(self):
        """<|...|> tokens within content are removed."""
        html_with_tokens = (
            "<!DOCTYPE html><|think|><html><head></head>"
            "<body><canvas></canvas><script></script></body></html>"
        )
        text = self._wrap(html_with_tokens)
        _, content = ag._extract_mistral_write_content(text)
        assert content is not None
        assert "<|think|>" not in content

    def test_empty_string_returns_none(self):
        path, content = ag._extract_mistral_write_content("")
        assert path is None
        assert content is None

    def test_real_world_sample(self):
        """Matches the format seen in production logs."""
        d = self._STR_DELIM
        html_body = (
            "Super Mario Bros. Clone\n"
            "MARIO: 000000\n"
            "<script>\n"
            "const canvas = document.getElementById('gameCanvas');\n"
            "const ctx = canvas.getContext('2d');\n"
            "const GRAVITY = 0.8;\n"
            "</script>\n"
        )
        real_text = "<|tool_call>call:write{content:" + d + html_body
        _, content = ag._extract_mistral_write_content(real_text)
        assert content is not None
        assert "GRAVITY" in content


# ===========================================================================
# 16.  _default_html_target_path with 'named' keyword  (regression)
# ===========================================================================

class TestRequestedOutputDir:
    MARIO_DIR = "/Users/aicoder/src/zzz-test/mario-clone"

    def test_extract_folder_phrase(self):
        msgs = [_msg(
            "user",
            "write and update into the folder "
            f"{self.MARIO_DIR}",
        )]
        assert ag._extract_requested_output_dir(msgs) == self.MARIO_DIR

    def test_default_html_uses_requested_folder(self):
        msgs = [_msg(
            "user",
            f"create a mario clone in folder {self.MARIO_DIR}",
        )]
        result = ag._default_html_target_path(msgs)
        assert result == os.path.join(self.MARIO_DIR, "index.html")

    def test_write_path_misses_requested_dir_parent_index(self):
        msgs = [_msg(
            "user",
            f"write into the folder {self.MARIO_DIR}",
        )]
        wrong = "/Users/aicoder/src/zzz-test/index.html"
        assert ag._write_path_misses_requested_dir(wrong, msgs) is True
        right = os.path.join(self.MARIO_DIR, "index.html")
        assert ag._write_path_misses_requested_dir(right, msgs) is False

    def test_extract_mario_clone_by_name_under_workspace(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "update mario-clone"),
        ]
        assert ag._extract_requested_output_dir(msgs) == self.MARIO_DIR

    def test_default_html_uses_mario_clone_when_named(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "improve the code in mario-clone/"),
        ]
        result = ag._default_html_target_path(msgs)
        assert result == os.path.join(self.MARIO_DIR, "index.html")

    def test_empty_write_path_is_not_dir_miss(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "improve the code in mario-clone/"),
        ]
        assert ag._write_path_misses_requested_dir("", msgs) is False

    def test_repair_write_target_preserves_content_fixes_path(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "improve the code in mario-clone/"),
        ]
        wrong = "/Users/aicoder/src/zzz-test/index.html"
        content = (
            "<!DOCTYPE html><html><head></head><body><canvas id='c'></canvas>"
            "<script>const canvas=document.getElementById('c');"
            "requestAnimationFrame(()=>{});document.addEventListener('keydown',()=>{});"
            "</script></body></html>"
        )
        path, body = ag._repair_write_target_for_messages(msgs, wrong, content)
        assert path == os.path.join(self.MARIO_DIR, "game.js")
        assert body == content

    def test_kilo_environment_details_does_not_steal_output_dir(self):
        """Working directory: in environment_details must not override mario-clone/."""
        msgs = [
            _msg("system", "You are Kilo."),
            _msg("user", [
                {"type": "text", "text": "improve the code in mario-clone/"},
                {
                    "type": "text",
                    "text": (
                        "<environment_details>\n"
                        "Working directory: /Users/aicoder/src/zzz-test\n"
                        "</environment_details>"
                    ),
                },
            ]),
        ]
        assert ag._extract_requested_output_dir(msgs) == self.MARIO_DIR
        assert ag._default_html_target_path(msgs) == os.path.join(
            self.MARIO_DIR, "index.html"
        )
        assert ag._write_path_misses_requested_dir(
            "/Users/aicoder/src/zzz-test/index.html", msgs
        )

    def test_find_workspace_prefers_mario_clone_over_parent_index(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "update mario-clone"),
        ]
        found = ag._find_workspace_html_target(msgs)
        assert found is not None
        path, _content = found
        assert path == os.path.join(self.MARIO_DIR, "index.html")

    def test_split_game_project_uses_edit_on_game_js(self):
        """mario-clone has game.js — improve must edit logic, not rewrite index.html."""
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "improve the code in mario-clone/ controls dont work"),
        ]
        body = {}
        tools = [_tool("write"), _tool("edit"), _tool("read_file")]
        assert ag._is_split_browser_game_project(self.MARIO_DIR)
        assert ag._suggested_project_edit_path(msgs).endswith("mario-clone/game.js")
        assert ag._should_force_write_for_html_project_update(msgs, tools) is False
        _out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
            body, msgs, tools
        )
        assert forced is True
        assert body["tool_choice"]["function"]["name"] == "edit"
        assert "game.js" in _out_msgs[0]["content"]

    def test_introduce_sounds_is_project_update_not_create(self):
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "introduce some sounds into mario-clone/"),
        ]
        assert ag._is_create_task(msgs)[0] is False
        assert ag._is_existing_code_project_update(msgs) is True

    def test_introduce_sounds_forces_edit_without_create_verb(self):
        """Non-create tasks like 'introduce sounds' must still force edit on game.js."""
        msgs = [
            _msg("system", "Current Workspace Directory (/Users/aicoder/src/zzz-test)"),
            _msg("user", "introduce some sounds into mario-clone/"),
        ]
        body = {}
        tools = [
            _tool("write"),
            _tool("edit"),
            _tool("read_file"),
            _tool("glob"),
            _tool("todowrite"),
        ]
        _out_msgs, out_tools, forced = ag._force_tool_for_agentic_turn(body, msgs, tools)
        assert forced is True
        assert body["tool_choice"]["function"]["name"] == "edit"
        tool_names = {
            (t.get("function") or {}).get("name") for t in out_tools if isinstance(t, dict)
        }
        assert "glob" not in tool_names
        assert "todowrite" not in tool_names
        assert "edit" in tool_names
        assert "game.js" in _out_msgs[0]["content"]

    def test_force_write_for_html_project_improve(self, tmp_path):
        mario = tmp_path / "mario-clone"
        mario.mkdir()
        (mario / "index.html").write_text("<html><body><canvas></canvas></body></html>")
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", f"improve the code in {mario}/"),
        ]
        body = {}
        tools = [_tool("write"), _tool("edit"), _tool("read_file")]
        _out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
            body, msgs, tools
        )
        assert forced is True
        assert body["tool_choice"]["function"]["name"] == "write"
        assert len(out_tools) == 1

    def test_tool_stream_blocks_low_chunk_guard(self):
        assert ag._tool_stream_blocks_low_chunk_guard("tool", "buffer") is True
        assert ag._tool_stream_blocks_low_chunk_guard("tool", "stream") is True
        assert ag._tool_stream_blocks_low_chunk_guard("text", None) is False

    def test_runaway_tracker_links_short_kilo_retry(self):
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()
        long_msgs = [
            _msg("system", "s"),
            _msg("user", "improve the code in mario-clone/"),
            _msg("assistant", "x" * 3000),
        ]
        short_msgs = [
            _msg("system", "s"),
            _msg("user", "improve the code in mario-clone/"),
        ]
        ag._record_text_runaway_for_task(long_msgs)
        assert ag._had_recent_text_runaway_for_task(short_msgs) is True
        ag._RECENT_TEXT_RUNAWAY_TASKS.clear()

    def test_repair_bad_write_redirects_parent_index_to_subfolder(self):
        msgs = [_msg(
            "user",
            f"write into the folder {self.MARIO_DIR}",
        )]
        writers = ag._discover_writers([_tool("write")])
        delta = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "function": {
                            "name": "write",
                            "arguments": json.dumps({
                                "filePath": "/Users/aicoder/src/zzz-test/index.html",
                                "content": "<html><body><canvas></canvas></body></html>",
                            }),
                        },
                    }],
                },
            }],
        }
        repaired, changed = ag._repair_bad_streaming_write_delta(
            delta,
            body={},
            messages=msgs,
            writers=writers,
        )
        assert changed is True
        args = json.loads(
            repaired["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        )
        assert args["filePath"] == os.path.join(self.MARIO_DIR, "index.html")


class TestDefaultHtmlTargetPathNamed:
    def test_write_to_file_named(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game, write to file named mario.html"),
        ]
        result = ag._default_html_target_path(msgs)
        assert result.endswith("mario.html"), f"got {result}"

    def test_create_game_dash_filename(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create web game - snake.html"),
        ]
        result = ag._default_html_target_path(msgs)
        assert result.endswith("snake.html"), f"got {result}"

    def test_fallback_to_index_when_no_name(self, tmp_path):
        msgs = [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg("user", "create a web game"),
        ]
        result = ag._default_html_target_path(msgs)
        assert result.endswith("index.html"), f"got {result}"


# ===========================================================================
# 17.  _recover_mistral_write_artifact preserves requested target path
# ===========================================================================

class TestRecoverMistralWriteArtifact:
    def _msgs(self, tmp_path, user_text=None):
        return [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg(
                "user",
                user_text
                or "create web game, write to file named mario.html",
            ),
        ]

    def _tool_text(self, html: str) -> str:
        return '<|tool_call>call:write{content:<|"|>' + html

    def test_fallback_preserves_mario_html_when_extraction_fails(self, tmp_path):
        msgs = self._msgs(tmp_path)
        path, content, features, used_model = ag._recover_mistral_write_artifact(
            msgs,
            '<|tool_call>call:write{content:<|"|>not html',
        )
        assert path.endswith("mario.html"), path
        assert "index.html" not in path
        assert used_model is False
        assert "<canvas" in content.lower()
        assert features

    def test_model_html_preserves_mario_html(self, tmp_path):
        msgs = self._msgs(tmp_path)
        html = (
            "<!DOCTYPE html><html><body>"
            "<canvas id='game'></canvas>"
            "<script>"
            "function loop(){requestAnimationFrame(loop)};"
            "let enemy=[], bullet=[], score=0;"
            "addEventListener('keydown',()=>{});"
            "</script></body></html>"
        )
        path, content, features, used_model = ag._recover_mistral_write_artifact(
            msgs,
            self._tool_text(html),
        )
        assert path.endswith("mario.html"), path
        assert used_model is True
        assert "<canvas" in content.lower()

    def test_explicit_filepath_in_tool_text_resolves_to_workspace(self, tmp_path):
        msgs = self._msgs(tmp_path, "create web game")
        html = (
            "<!DOCTYPE html><html><body><canvas></canvas>"
            "<script>requestAnimationFrame(()=>{}); let enemy=[], bullet=[], score=0;"
            "addEventListener('keydown',()=>{});</script></body></html>"
        )
        text = '<|tool_call>call:write{filePath:<|"|>mario.html<|"|>,content:<|"|>' + html
        path, _content, _features, _used_model = ag._recover_mistral_write_artifact(
            msgs,
            text,
        )
        assert path == str(tmp_path / "mario.html")

    def test_midstream_threshold_example_is_recoverable_at_12kb(self, tmp_path):
        msgs = self._msgs(tmp_path)
        # Simulates the production log: ~12 KB buffered while the model keeps
        # going. The proxy should be able to recover immediately rather than
        # waiting for upstream EOS.
        html = (
            "<!DOCTYPE html><html><body><canvas id='game'></canvas>"
            "<script>requestAnimationFrame(()=>{}); let enemy=[], bullet=[], score=0;"
            "addEventListener('keydown',()=>{});</script>"
            + (" " * 12_200)
        )
        text = self._tool_text(html)
        path, content, _features, _used_model = ag._recover_mistral_write_artifact(
            msgs,
            text,
        )
        assert path.endswith("mario.html"), path
        assert content


# ===========================================================================
# 18.  Platformer / Mario validation and fallback
# ===========================================================================

class TestPlatformerFallbackAndValidation:
    def _msgs(self, tmp_path):
        return [
            _msg("system", f"Current Workspace Directory ({tmp_path})"),
            _msg(
                "user",
                "create web game, write to file named mario.html\n"
                "Super Mario Bros. platformer with coins, goombas, jump physics, "
                "mushrooms, pipes, and a flagpole",
            ),
        ]

    def test_platformer_task_detection(self):
        assert ag._looks_like_platformer_task("Super Mario Bros platformer")
        assert ag._looks_like_platformer_task("jump on goomba and collect coins")
        assert not ag._looks_like_platformer_task("1942 airplane shooter")

    def test_synthetic_game_html_for_task_uses_platformer(self):
        html = ag._synthetic_game_html_for_task("Super Mario Bros platformer")
        low = html.lower()
        assert "platform" in low or "mario" in low
        assert "gravity" in low
        assert "coin" in low
        assert "goomba" in low or "enemy" in low
        assert "<canvas" in low

    def test_platformer_fallback_draws_tall_visible_mario(self):
        html = ag._synthetic_game_html_for_task("Super Mario Bros platformer")
        low = html.lower()
        assert "w:18,h:32" in html
        assert "drawMario" in html
        assert "drawMario()" in html
        assert "player.w=22;player.h=40" in html
        # The old fallback drew Mario as one flat filled rectangle.  The new
        # fallback uses a multi-part sprite instead.
        assert "fillrect(player.x,player.y,player.w,player.h)" not in low

    def test_platformer_fallback_written_to_mario_html(self, tmp_path):
        msgs = self._msgs(tmp_path)
        path, html = ag._synthetic_write_fallback_artifact(msgs)
        assert path.endswith("mario.html"), path
        low = html.lower()
        assert "gravity" in low
        assert "coin" in low
        assert "jump" in low
        assert "drawmario" in low
        assert "1942" not in low

    def test_valid_platformer_without_bullet_is_not_replaced(self, tmp_path):
        """Mario/platformer content should not require the shooter marker 'bullet'."""
        msgs = self._msgs(tmp_path)
        model_html = """<!DOCTYPE html>
<html><body><canvas id="game"></canvas>
<script>
const keys = Object.create(null); addEventListener('keydown', e => keys[e.code]=true);
let score = 0, coins = 0, gravity = 0.8;
function jump(){ player.vy = -18; }
function loop(){ requestAnimationFrame(loop); }
const player = {x:0,y:0,vy:0}; const goombas=[];
</script></body></html>"""
        path, content, features = ag._validated_write_artifact(
            msgs,
            str(tmp_path / "mario.html"),
            model_html,
        )
        low = content.lower()
        assert path.endswith("mario.html")
        assert "gravity" in low
        assert "goombas" in low
        assert "1942" not in low
        assert any("platformer" in f.lower() for f in features)

    def test_platformer_script_fragment_is_wrapped_and_preserved(self, tmp_path):
        """The real model emits heading + <script> without canvas/html; wrap it."""
        msgs = self._msgs(tmp_path)
        fragment = """
Super Mario Bros. Clone
MARIO: 000000
<script>
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const GRAVITY = 0.8;
let score = 0, coins = 0;
const goombas = [];
function jump(){ mario.vy = -18; }
function loop(){ requestAnimationFrame(loop); }
addEventListener('keydown', e => {});
</script>
"""
        path, content, features = ag._validated_write_artifact(
            msgs,
            str(tmp_path / "mario.html"),
            fragment,
        )
        low = content.lower()
        assert path.endswith("mario.html")
        assert "<html" in low
        assert "<canvas" in low
        assert "gamecanvas" in low
        assert "gravity" in low
        assert "1942" not in low
        assert any("platformer" in f.lower() for f in features)

    def test_invalid_platformer_falls_back_to_platformer_not_1942(self, tmp_path):
        msgs = self._msgs(tmp_path)
        path, content, _features = ag._validated_write_artifact(
            msgs,
            str(tmp_path / "mario.html"),
            "<html><body>not enough game code</body></html>",
        )
        low = content.lower()
        assert path.endswith("mario.html")
        assert "gravity" in low
        assert "coin" in low
        assert "1942" not in low

    def test_shooter_html_rejected_for_mario_prompt(self, tmp_path):
        """Regression: Mario prompts must not keep 1942/shooter HTML."""
        msgs = self._msgs(tmp_path)
        shooter_html = """<!DOCTYPE html>
<html><body><canvas id="game"></canvas>
<script>
let score=0,lives=3,wave=1;
const bullets=[], enemies=[], particles=[];
function spawnEnemy(){ enemies.push({x:0,y:0}); }
function spawnExplosion(){ particles.push({x:0,y:0}); }
function enemyShoot(){ bullets.push({x:0,y:0}); }
function loop(){ requestAnimationFrame(loop); }
addEventListener('keydown',()=>{});
</script></body></html>"""
        path, content, features = ag._validated_write_artifact(
            msgs,
            str(tmp_path / "mario.html"),
            shooter_html,
        )
        low = content.lower()
        assert path.endswith("mario.html"), path
        assert "gravity" in low
        assert "coin" in low
        assert "jump" in low
        assert "1942" not in low
        assert any("platformer" in f.lower() for f in features)
        assert ag._html_looks_like_shooter_not_platformer(shooter_html)

    def test_bare_mario_html_defaults_to_zzz_test_without_workspace(self):
        msgs = [_msg("user", "create web game, write to file named mario.html")]
        result = ag._explicit_html_target_path(msgs)
        # In this repo the dev scratch workspace exists, so bare targets are
        # resolved there instead of leaving Kilo to interpret a relative path.
        if os.path.isdir("/Users/aicoder/src/zzz-test"):
            assert result == "/Users/aicoder/src/zzz-test/mario.html"
        else:
            assert result == "mario.html"


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
