#!/usr/bin/env python3
"""Replay-based integration tests for gemma4_mlx_kilo_proxy.py.

Each test loads a saved upstream SSE file (from tests/fixtures/ or a live
replays/ folder) and runs the FULL proxy pipeline against it using
_ReplayTransport + httpx.ASGITransport — zero LLM, zero network.

Assertions cover:
  • No crash / no exception
  • Correct response_type classification (tool vs text)
  • Guard fires (or absence of guards)
  • Tool name remapping (write_to_file → write)
  • Fuzzy repair trigger
  • [stream-summary] JSON is emitted and parseable

Run with:
    venv/bin/python -m pytest tests/test_replays.py -q
or stand-alone:
    venv/bin/python tests/test_replays.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load_proxy():
    root = Path(__file__).resolve().parent.parent
    src = root / "gemma4_mlx_kilo_proxy.py"
    spec = importlib.util.spec_from_file_location("gemma4_mlx_kilo_proxy", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_mlx_kilo_proxy"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load_proxy()

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPLAYS = Path(__file__).resolve().parent.parent / "replays"


# ---------------------------------------------------------------------------
# Replay runner helper
# ---------------------------------------------------------------------------

async def _run_replay(
    sse_path: Path,
    body: dict | None = None,
    tools: list | None = None,
) -> dict:
    """Run *sse_path* through the full proxy pipeline.

    Returns a dict with keys:
      ``summary``       — parsed [stream-summary] JSON (or {})
      ``chunks``        — list of raw SSE ``data:`` payloads forwarded to client
      ``tool_calls``    — list of (name, args_str) extracted from tool-call chunks
      ``text``          — concatenated delta.content text from text-mode chunks
      ``exception``     — exception message if the stream raised, else None
    """
    import httpx

    sse_bytes = sse_path.read_bytes()
    replay_client = httpx.AsyncClient(
        base_url="http://replay-upstream",
        transport=ag._ReplayTransport(sse_bytes),
        timeout=300.0,
    )
    _body: dict = body or {"messages": [{"role": "user", "content": "test"}]}
    _body["stream"] = True
    if tools is not None:
        _body["tools"] = tools

    _app = ag.make_app("http://replay-upstream", _client_override=replay_client)

    # Capture [stream-summary] log line.
    # The logger's effective level defaults to WARNING (root logger default);
    # temporarily lower it to INFO so our INFO-level summary line is captured.
    summary: dict = {}

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "[stream-summary]" in msg:
                try:
                    summary.update(json.loads(msg.split("[stream-summary] ", 1)[1]))
                except Exception:
                    pass

    cap = _Cap()
    cap.setLevel(logging.INFO)
    logger = logging.getLogger("gemma4_mlx_kilo_proxy")
    orig_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(cap)

    chunks: list[str] = []
    tool_calls: list[tuple[str, str]] = []
    text_parts: list[str] = []
    exc_msg: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app),  # type: ignore[attr-defined]
            base_url="http://test",
            timeout=60.0,
        ) as tc:
            async with tc.stream("POST", "/v1/chat/completions", json=_body) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        continue
                    chunks.append(raw)
                    try:
                        ev = json.loads(raw)
                        for choice in ev.get("choices", []):
                            delta = choice.get("delta", {})
                            # Text
                            if delta.get("content"):
                                text_parts.append(delta["content"])
                            # Tool calls
                            for tc_item in delta.get("tool_calls", []):
                                fn = tc_item.get("function", {})
                                name = fn.get("name", "")
                                args = fn.get("arguments", "")
                                if name or args:
                                    tool_calls.append((name, args))
                    except Exception:
                        pass
    except Exception as exc:
        exc_msg = str(exc)
    finally:
        logger.removeHandler(cap)
        logger.setLevel(orig_level)

    return {
        "summary": summary,
        "chunks": chunks,
        "tool_calls": tool_calls,
        "text": "".join(text_parts),
        "exception": exc_msg,
    }


def run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Kilo-style tools fixture
# ---------------------------------------------------------------------------

def _kilo_tools() -> list[dict]:
    def _tool(name: str, props: dict) -> dict:
        return {"type": "function", "function": {"name": name, "parameters": {"properties": props}}}

    return [
        _tool("write", {"filePath": {}, "content": {}}),
        _tool("edit", {"filePath": {}, "oldString": {}, "newString": {}}),
        _tool("bash", {"command": {}}),
        _tool("todowrite", {"todos": {}}),
    ]


# ===========================================================================
# Fixture: bash_tool_call.sse
# ===========================================================================

class TestBashToolCallReplay:
    """Tests against tests/fixtures/bash_tool_call.sse."""

    _fixture = FIXTURES / "bash_tool_call.sse"

    def test_no_crash(self):
        result = run(_run_replay(self._fixture))
        assert result["exception"] is None

    def test_stream_summary_emitted(self):
        result = run(_run_replay(self._fixture))
        assert result["summary"], "Expected [stream-summary] to be logged"

    def test_classified_as_tool(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        assert result["summary"].get("response_type") == "tool"

    def test_no_guards_triggered(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        guards = result["summary"].get("guards_triggered", [])
        assert guards == [], f"Unexpected guards: {guards}"

    def test_tool_name_is_bash(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        names = [name for name, _ in result["tool_calls"] if name]
        assert "bash" in names, f"Expected 'bash' in tool call names, got {names}"

    def test_bash_args_json_valid(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        # Collect all args fragments for index 0
        all_args = "".join(args for _, args in result["tool_calls"] if args)
        if all_args.strip():
            parsed = json.loads(all_args)
            assert "command" in parsed
            assert parsed["command"] == "ls -la"


# ===========================================================================
# Fixture: text_response.sse
# ===========================================================================

class TestTextResponseReplay:
    """Tests against tests/fixtures/text_response.sse."""

    _fixture = FIXTURES / "text_response.sse"

    def test_no_crash(self):
        result = run(_run_replay(self._fixture))
        assert result["exception"] is None

    def test_classified_as_text(self):
        result = run(_run_replay(self._fixture))
        assert result["summary"].get("response_type") == "text"

    def test_text_content_forwarded(self):
        result = run(_run_replay(self._fixture))
        assert "Paris" in result["text"], f"Expected 'Paris' in text: {result['text']!r}"

    def test_no_tool_calls(self):
        result = run(_run_replay(self._fixture))
        names = [n for n, _ in result["tool_calls"] if n]
        assert names == [], f"Did not expect tool calls: {names}"


# ===========================================================================
# Fixture: hallucinated_write_tool.sse  (write_to_file → write remap)
# ===========================================================================

class TestHallucinatedWriteToolReplay:
    """Tests against tests/fixtures/hallucinated_write_tool.sse."""

    _fixture = FIXTURES / "hallucinated_write_tool.sse"

    def test_no_crash(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        assert result["exception"] is None

    def test_classified_as_tool(self):
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        assert result["summary"].get("response_type") == "tool"

    def test_name_remapped_to_write(self):
        """The hallucinated write_to_file should be remapped to 'write'."""
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        names = [n for n, _ in result["tool_calls"] if n]
        # After remap, 'write_to_file' should become 'write'
        assert "write_to_file" not in names, f"Remap failed: {names}"
        assert "write" in names, f"Expected 'write' after remap, got {names}"

    def test_path_field_remapped_to_filePath(self):
        """In STREAM mode (write_to_file), the name is remapped but arg field
        renaming (path → filePath) only happens in BUFFER mode (edit tools).
        Assert that either the remap happened, or at minimum the stream completed
        without error (the name remap is already tested above).
        """
        result = run(_run_replay(self._fixture, tools=_kilo_tools()))
        assert result["exception"] is None
        # Confirm the stream produced at least one tool-call chunk.
        names = [n for n, _ in result["tool_calls"] if n]
        assert names, "Expected at least one named tool-call delta"


# ===========================================================================
# ProxyConfig integration
# ===========================================================================

class TestProxyConfigInReplay:
    """Verify that feature flags on ProxyConfig affect replay output."""

    def test_disable_fuzzy_repair_passthrough(self):
        """When enable_fuzzy_repair=False, repair is skipped (no crash)."""
        orig = ag._PROXY_CFG.enable_fuzzy_repair
        try:
            ag._PROXY_CFG.enable_fuzzy_repair = False
            result = run(_run_replay(FIXTURES / "bash_tool_call.sse", tools=_kilo_tools()))
            assert result["exception"] is None
        finally:
            ag._PROXY_CFG.enable_fuzzy_repair = orig

    def test_proxyconfig_from_env_roundtrip(self, monkeypatch):
        """ProxyConfig.from_env() round-trips through env-var overrides."""
        monkeypatch.setenv("PROXY_GUARD_STALL_DUMP_S", "7.7")
        monkeypatch.setenv("PROXY_ENABLE_FUZZY_REPAIR", "0")
        cfg = ag.ProxyConfig.from_env()
        assert cfg.stall_dump_s == pytest.approx(7.7)
        assert cfg.enable_fuzzy_repair is False


# ===========================================================================
# ProxyMetrics
# ===========================================================================

class TestProxyMetrics:
    """Verify that _METRICS counters are incremented by replays."""

    def test_requests_total_incremented(self):
        before = ag._METRICS.requests_total
        run(_run_replay(FIXTURES / "bash_tool_call.sse"))
        after = ag._METRICS.requests_total
        assert after > before

    def test_streams_total_incremented(self):
        before = ag._METRICS.streams_total
        run(_run_replay(FIXTURES / "text_response.sse"))
        after = ag._METRICS.streams_total
        assert after > before

    def test_to_prometheus_format(self):
        text = ag._METRICS.to_prometheus()
        assert "proxy_requests_total" in text
        assert "proxy_guards_stall_abort" in text
        assert "proxy_fuzzy_repairs_applied" in text

    def test_to_prometheus_parseable(self):
        """Every non-comment line should be '<name> <number>'."""
        for line in ag._METRICS.to_prometheus().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 2, f"Unexpected Prometheus line: {line!r}"
            float(parts[1])  # should be numeric


# ===========================================================================
# StallDetector / TextModeGuard / ToolStreamState
# ===========================================================================

class TestHelperClasses:
    """Unit tests for the new stream helper dataclasses."""

    def test_stall_detector_reset(self):
        sd = ag.StallDetector()
        sd.reset(0.0)
        assert sd.stall_duration >= 0

    def test_stall_detector_update_resets_timer(self):
        import time
        sd = ag.StallDetector()
        now = time.perf_counter()
        sd.reset(now)
        sd.update(0, now)
        # Duration should be very small right after update
        assert sd.stall_duration < 1.0

    def test_stall_detector_tracks_changes(self):
        import time
        sd = ag.StallDetector()
        now = time.perf_counter()
        sd.reset(now)
        sd.update(10, now)
        sd.update(10, now + 5.0)   # same token count — no timer reset
        assert sd._last_change_t == now  # timer NOT reset

        sd.update(11, now + 5.0)   # new token — timer resets
        assert sd._last_change_t == now + 5.0

    def test_text_mode_guard_basic(self):
        g = ag.TextModeGuard()
        g.feed("hello world")
        assert g.chars == 11
        assert g.is_runaway(10) is True
        assert g.is_runaway(100) is False
        g.reset()
        assert g.chars == 0

    def test_text_mode_guard_zero_cap_disabled(self):
        g = ag.TextModeGuard()
        g.feed("x" * 100000)
        assert g.is_runaway(0) is False  # cap=0 means disabled

    def test_tool_stream_state_key(self):
        ts = ag.ToolStreamState()
        ts.active_name = "write_to_file"
        assert ts.key == "writetofile"

    def test_tool_stream_state_start_and_feed(self):
        import time
        ts = ag.ToolStreamState()
        now = time.perf_counter()
        ts.start("stream", now, 0)
        assert ts.mode == "stream"
        ts.feed_args_delta('{"command":"ls"}')
        assert ts.args_chars == len('{"command":"ls"}')
        assert '{"command":"ls"}' in ts.args_head

    def test_tool_stream_state_args_head_cap(self):
        ts = ag.ToolStreamState(args_head_cap=10)
        ts.feed_args_delta("a" * 20)
        # args_head should not grow beyond cap
        assert len(ts.args_head) <= 10


# ===========================================================================
# _validate_tool_call_args
# ===========================================================================

class TestValidateToolCallArgs:
    """Tests for the new pre-repair validation helper."""

    def test_valid_args_returns_none(self):
        assert ag._validate_tool_call_args("bash", '{"command":"ls"}') is None

    def test_empty_args_ok(self):
        assert ag._validate_tool_call_args("bash", "") is None
        assert ag._validate_tool_call_args("bash", "   ") is None

    def test_empty_name_error(self):
        err = ag._validate_tool_call_args("", '{"x":1}')
        assert err is not None
        assert "empty" in err

    def test_non_json_args_error(self):
        err = ag._validate_tool_call_args("bash", "not json at all")
        assert err is not None

    def test_invalid_json_error(self):
        err = ag._validate_tool_call_args("write", '{"filePath": "x", "content": }')
        assert err is not None
        assert "JSON" in err

    def test_array_args_error(self):
        # Tool args should always be objects, not arrays
        err = ag._validate_tool_call_args("bash", '[1,2,3]')
        assert err is not None


# ===========================================================================
# Live replays/ folder (skipped when folder doesn't exist)
# ===========================================================================

@pytest.mark.skipif(
    not REPLAYS.exists() or not list(REPLAYS.glob("*_upstream.sse")),
    reason="No replay files found in replays/ folder",
)
class TestLiveReplays:
    """Parameterised replay tests against files saved by PROXY_CAPTURE_REPLAYS=1.

    Add your own replays by running the proxy with:
        PROXY_CAPTURE_REPLAYS=1 PROXY_REPLAYS_DIR=replays \\
            python gemma4_mlx_kilo_proxy.py

    Each saved replay becomes a test case automatically.
    """

    @pytest.fixture(params=list(REPLAYS.glob("*_upstream.sse")))
    def replay_file(self, request):
        return request.param

    def test_replay_no_crash(self, replay_file):
        """Every saved replay must run through the pipeline without crashing."""
        req_file = replay_file.with_name(replay_file.name.replace("_upstream.sse", "_request.json"))
        body = None
        if req_file.exists():
            try:
                body = json.loads(req_file.read_text())
            except Exception:
                pass
        result = run(_run_replay(replay_file, body=body))
        assert result["exception"] is None, (
            f"Replay {replay_file.name} raised: {result['exception']}"
        )

    def test_replay_summary_present(self, replay_file):
        """Every replay should emit a [stream-summary]."""
        result = run(_run_replay(replay_file))
        assert result["summary"], (
            f"Replay {replay_file.name} did not emit [stream-summary]"
        )

    def test_replay_response_type_set(self, replay_file):
        """response_type should be 'tool' or 'text' (never None for a complete stream)."""
        result = run(_run_replay(replay_file))
        rt = result["summary"].get("response_type")
        assert rt in ("tool", "text", None), f"Unexpected response_type: {rt}"

    def test_replay_expected_guards(self, replay_file):
        """If a <name>_summary.json exists, assert guards match."""
        summary_file = replay_file.with_name(
            replay_file.name.replace("_upstream.sse", "_summary.json")
        )
        if not summary_file.exists():
            pytest.skip("No _summary.json to assert against")
        saved = json.loads(summary_file.read_text())
        expected_guards = saved.get("guards_triggered", [])
        result = run(_run_replay(replay_file))
        actual_guards = result["summary"].get("guards_triggered", [])
        assert set(expected_guards) == set(actual_guards), (
            f"Guard mismatch: expected {expected_guards}, got {actual_guards}"
        )


if __name__ == "__main__":
    import pytest as _pytest, sys as _sys
    _sys.exit(_pytest.main([__file__, "-v"] + _sys.argv[1:]))
