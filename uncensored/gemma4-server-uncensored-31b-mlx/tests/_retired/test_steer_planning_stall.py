#!/usr/bin/env python3
"""Regression tests for the Harmony-leak / planning-stall fix.

Covers:
  * `_HARMONY_LOGIT_BIAS` — both tokens (`<|channel>`=100, `<|think|>`=98)
    are biased to -100.0.
  * `_force_xhigh_settings` — merges the Harmony bias into agentic
    requests, lets caller-supplied bias win.
  * `_strip_planning_tools_if_stuck` — only fires when planning has
    happened but no write call has been seen.
  * `_upgrade_stall_nudge_after_strip` — replaces a `PLANNING STALL
    DETECTED` trailing nudge with the harsher `NO_PLANNING` variant
    naming the stripped tools and explicitly forbidding the Gemma-4
    Harmony / CoT special tokens.

Run with: ``venv/bin/python -m pytest tests/test_steer_planning_stall.py -q``
or as a script: ``venv/bin/python tests/test_steer_planning_stall.py``.
"""
from __future__ import annotations

import importlib.util
import base64
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_gemma4_mlx_kilo_proxy():
    """Load `gemma4_mlx_kilo_proxy.py` directly from disk (it's a sibling script,
    not an installed package, and has no `__init__.py`)."""
    root = Path(__file__).resolve().parent.parent
    src = root / "gemma4_mlx_kilo_proxy.py"
    spec = importlib.util.spec_from_file_location("gemma4_mlx_kilo_proxy", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_mlx_kilo_proxy"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load_gemma4_mlx_kilo_proxy()


# ---------------------------------------------------------------------------
# _HARMONY_LOGIT_BIAS
# ---------------------------------------------------------------------------


def test_validate_image_parts_rejects_empty_data_url():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,"},
            },
        ],
    }]
    error = ag._validate_image_parts(messages)
    assert error is not None
    assert "empty base64 payload" in error


def test_validate_image_parts_accepts_png_data_url():
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
    )
    messages = [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(png_1x1).decode()
            },
        }],
    }]
    assert ag._validate_image_parts(messages) is None


def test_attach_referenced_local_images_from_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "planes.png")
        with open(image_path, "wb") as handle:
            handle.write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01"
            )
        messages = [{
            "role": "user",
            "content": (
                "read planes.png\n"
                "<environment_details>\n"
                f"Current Workspace Directory ({tmp})\n"
                "</environment_details>"
            ),
        }]
        patched, attached = ag._attach_referenced_local_images(messages)
    assert attached == [image_path]
    content = patched[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_attach_referenced_absolute_image_path_from_text_parts():
    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "planes.png")
        with open(image_path, "wb") as handle:
            handle.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"describe {image_path}"},
                {"type": "text", "text": "<environment_details>\nCurrent time: now\n</environment_details>"},
            ],
        }]
        patched, attached = ag._attach_referenced_local_images(messages)
    assert attached == [image_path]
    content = patched[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_prepare_multimodal_observation_turn_strips_tools_and_history():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe planes.png"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        },
    ]
    patched, tools, changed = ag._prepare_multimodal_observation_turn(
        messages, [_tool("write")]
    )
    assert changed is True
    assert tools is None
    assert [m["role"] for m in patched] == ["system", "user"]
    assert "Inspect the attached image" in patched[-1]["content"][0]["text"]


def test_prepare_multimodal_observation_turn_skips_create_requests():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "read planes.png and create 5 different types of planes to incorporate onto 1942",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        },
    ]
    patched, tools, changed = ag._prepare_multimodal_observation_turn(
        messages, [_tool("write")]
    )
    assert changed is False
    assert tools == [_tool("write")]
    assert patched == messages


def test_detach_images_for_agentic_write_turn_keeps_tools_text_only():
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "read planes.png and create 5 different types of planes to incorporate onto 1942",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        }],
        "tools": [_tool("write")],
        "logit_bias": {"100": -100.0},
    }
    detached = ag._detach_images_for_agentic_write_turn(body)
    assert detached
    assert body["tools"] == [_tool("write")]
    assert body["messages"][0]["content"].startswith("read planes.png and create")
    assert not ag._has_any_image_part(body["messages"])
    assert body.get("logit_bias") == {"100": -100.0}
    assert body.get("enable_thinking") is False


def test_multimodal_observation_body_disables_thinking_without_logit_bias():
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe planes.png"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }],
        "tools": [_tool("write")],
        "max_tokens": 4096,
    }
    body["messages"], prepared_tools, changed = ag._prepare_multimodal_observation_turn(
        body["messages"], body.get("tools")
    )
    assert changed is True
    if changed:
        ag._apply_multimodal_observation_settings(body)
    assert body["tools"] == []
    assert body["enable_thinking"] is False
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert "logit_bias" not in body
    assert body["max_tokens"] == 768


def test_harmony_logit_bias_has_both_tokens():
    """All known Harmony control tokens relevant to this model
    (channel-open=100, channel-close=101, think=98) MUST be banned."""
    bias = ag._HARMONY_LOGIT_BIAS
    assert "100" in bias, "<|channel> (id=100) must be in bias"
    assert "101" in bias, "<channel|> (id=101) must be in bias"
    assert "98" in bias, "<|think|> (id=98) must be in bias"
    assert bias["100"] == -100.0
    assert bias["101"] == -100.0
    assert bias["98"] == -100.0


def test_harmony_bias_keys_are_strings():
    """OpenAI logit_bias wire format uses string keys — the upstream
    `engine/simple.py` patch coerces these to int.  If we accidentally
    emit int keys here, JSON serialisation still works but the upstream
    schema validator may reject them."""
    bias = ag._HARMONY_LOGIT_BIAS
    for k in bias:
        assert isinstance(k, str), f"key {k!r} must be str, got {type(k).__name__}"


# ---------------------------------------------------------------------------
# _force_xhigh_settings — agentic temperature + Harmony bias injection
# ---------------------------------------------------------------------------

def test_force_xhigh_injects_harmony_bias_on_agentic():
    body = {"tools": [{"type": "function", "function": {"name": "write"}}],
            "temperature": 0.0}
    ag._force_xhigh_settings(body)
    bias = body.get("logit_bias")
    assert isinstance(bias, dict), "logit_bias must be set on agentic requests"
    assert bias["100"] == -100.0
    assert bias["101"] == -100.0
    assert bias["98"] == -100.0
    assert body.get("enable_thinking") is False
    assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False


def test_force_xhigh_no_bias_on_non_agentic():
    """Plain chat completions (no tools) — no Harmony bias should be
    injected.  The Harmony leak only matters when the upstream tool
    parser is in play."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    ag._force_xhigh_settings(body)
    assert "logit_bias" not in body or not body["logit_bias"]


def test_force_xhigh_merges_caller_bias_caller_wins():
    """If the caller already supplied `logit_bias`, their entries must
    override the proxy defaults so debug overrides work without code
    changes."""
    body = {
        "tools": [{"type": "function", "function": {"name": "write"}}],
        "temperature": 0.0,
        "logit_bias": {"100": +5.0, "42": -3.0},  # caller unblocks channel + biases 42
    }
    ag._force_xhigh_settings(body)
    bias = body["logit_bias"]
    assert bias["100"] == +5.0, "caller override must win over proxy default"
    assert bias["98"] == -100.0, "untouched proxy default must survive"
    assert bias["42"] == -3.0, "caller's extra entries must survive"


def test_force_xhigh_agentic_temp_floor():
    """Temperature 0.0 on a tool-bearing request must be lifted to the
    agentic minimum (avoid the Gemma-4 'what would you like me to do?'
    failure mode)."""
    body = {"tools": [{"type": "function", "function": {"name": "write"}}],
            "temperature": 0.0}
    ag._force_xhigh_settings(body)
    assert body["temperature"] >= ag._AGENT_TEMP_MIN


def test_force_xhigh_agentic_temp_cap():
    """Caller-supplied excessive temperature must be clamped to the
    agentic max to prevent runaway sampling."""
    body = {"tools": [{"type": "function", "function": {"name": "write"}}],
            "temperature": 2.0}
    ag._force_xhigh_settings(body)
    assert body["temperature"] <= ag._AGENT_TEMP_MAX


def test_synthetic_tool_call_stream_contains_valid_write_args():
    html = ag._synthetic_browser_game_html(
        "create a 1942 capcom game for a web browser, allow user to shoot the incoming planes"
    )
    chunks = ag._synthetic_tool_call_stream(
        body={"model": "m"},
        tool_name="write",
        file_path="index.html",
        content=html,
    )
    assert chunks[-1] == b"data: [DONE]\n\n"
    first = chunks[0].decode()
    assert '"tool_calls"' in first
    assert '"name": "write"' in first
    assert "index.html" in first
    assert "Space" in first


def test_synthetic_browser_game_html_has_shooting_game_features():
    html = ag._synthetic_browser_game_html(
        "create a 1942 capcom game for a web browser"
    )
    low = html.lower()
    assert "<canvas" in low
    assert "space" in low
    assert "bullets" in low
    assert "enemies" in low
    assert "requestanimationframe" in low
    assert "arrow" in low
    assert "keyw" in low
    assert "enter" in low
    assert "wave" in low
    assert "lives" in low
    assert "score" in low
    assert "mute" in low
    assert "audiocontext" in low
    assert "spawnexplosion" in low
    assert "drawplane" in low
    assert "drawocean" in low
    assert "clouds" in low
    assert "trail" in low
    assert "planeTypes" in html
    assert "Emerald Falcon" in html
    assert "serviceworker" not in low


def test_synthetic_write_fallback_preserves_existing_game_on_followup():
    existing = "<html><body><canvas id='game'></canvas></body></html>"
    messages = [
        {"role": "user", "content": "create a 1942 game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({
                    "filePath": "index.html",
                    "content": existing,
                }),
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "enhance the graphics"},
    ]
    path, content = ag._synthetic_write_fallback_artifact(messages)
    assert path == "index.html"
    assert "<canvas id='game'></canvas>" in content
    assert "agent-steer fallback: preserved existing browser game" in content
    assert "agent-steer-visual-polish" in content
    assert "agent-steer-canvas-graphics-upgrade" in content
    assert "function drawPlane" in content
    assert "ctx.fillRect=function" in content
    assert "strokeRect(-w*.72" in content
    assert "arc(0,-h*.54" in content


def test_synthetic_write_fallback_adds_hit_sound_marker():
    existing = "<html><body><canvas id='game'></canvas></body></html>"
    messages = [
        {"role": "user", "content": "create a 1942 game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({
                    "filePath": "index.html",
                    "content": existing,
                }),
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "add shot sound when bullets hit planes"},
    ]
    _path, content = ag._synthetic_write_fallback_artifact(messages)
    assert "agent-steer-hit-sound" in content


def test_synthetic_write_fallback_uses_latest_cache_request():
    existing = "<html><head></head><body><canvas id='game'></canvas><script src='game.js'></script></body></html>"
    messages = [
        {"role": "user", "content": "read planes.png and create 5 different types of planes"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({
                    "filePath": "index.html",
                    "content": existing,
                }),
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "can we ensure that web caching is disabled when loading"},
    ]
    path, content = ag._synthetic_write_fallback_artifact(messages)
    assert path == "index.html"
    assert "planes.js" not in path
    assert "gemma4-mlx-kilo-cache-bust" in content
    assert 'Cache-Control" content="no-store' in content
    assert "searchParams.set('v', version)" in content
    assert "planes" not in ag._last_create_task_text(messages).lower()


def test_synthetic_write_fallback_cache_preserves_workspace_index():
    existing = (
        "<html><head><title>Core</title></head><body>"
        "<canvas id='game'></canvas><script>function loop(){requestAnimationFrame(loop)}</script>"
        "</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        index_path = os.path.join(tmp, "index.html")
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write(existing)
        original = ag._extract_workspace_dir
        ag._extract_workspace_dir = lambda _messages: tmp
        try:
            messages = [
                {"role": "user", "content": "can we ensure that web caching is disabled when loading"},
            ]
            path, content = ag._synthetic_write_fallback_artifact(messages)
        finally:
            ag._extract_workspace_dir = original
    assert path == index_path
    assert "<canvas id='game'></canvas>" in content
    assert "function loop()" in content
    assert "gemma4-mlx-kilo-cache-bust" in content
    assert "Browser Arcade Game" not in content


def test_synthetic_write_fallback_uses_workspace_index_for_new_file():
    with tempfile.TemporaryDirectory() as tmp:
        original = ag._extract_workspace_dir
        ag._extract_workspace_dir = lambda _messages: tmp
        try:
            messages = [
                {"role": "user", "content": "implement 1,2,3,4,5,7,8"},
            ]
            path, content = ag._synthetic_write_fallback_artifact(messages)
        finally:
            ag._extract_workspace_dir = original
    assert path == os.path.join(tmp, "index.html")
    assert os.path.isabs(path)
    assert "<canvas" in content


def test_synthetic_write_fallback_prefers_existing_1942_html():
    with tempfile.TemporaryDirectory() as tmp:
        path_1942 = os.path.join(tmp, "1942.html")
        index_path = os.path.join(tmp, "index.html")
        Path(path_1942).write_text("<html><body><canvas></canvas>1942</body></html>")
        Path(index_path).write_text("<html><body><canvas></canvas>index</body></html>")
        original = ag._extract_workspace_dir
        ag._extract_workspace_dir = lambda _messages: tmp
        try:
            target = ag._default_html_target_path([
                {"role": "user", "content": "implement 1,2,3,4,5,7,8"},
            ])
            found_path, found_content = ag._find_workspace_html_target([
                {"role": "user", "content": "implement 1,2,3,4,5,7,8"},
            ])
        finally:
            ag._extract_workspace_dir = original
    assert target == path_1942
    assert found_path == path_1942
    assert "1942" in found_content


def test_synthetic_write_fallback_honors_explicit_html_path():
    messages = [
        {"role": "user", "content": "Edit /Users/aicoder/src/zzz-test/index.html directly. Improve graphics."},
    ]
    path, content = ag._synthetic_write_fallback_artifact(messages)
    assert path == "/Users/aicoder/src/zzz-test/index.html"
    assert "<canvas" in content


def test_synthetic_write_fallback_reads_explicit_html_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.html")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("<html><body><canvas id='game'></canvas><p>keep me</p></body></html>")
        messages = [{
            "role": "user",
            "content": f"Edit {path} directly. Improve graphics.",
        }]
        out_path, content = ag._synthetic_write_fallback_artifact(messages)
    assert out_path == path
    assert "keep me" in content
    assert "agent-steer-canvas-graphics-upgrade" in content


def test_synthetic_write_fallback_reapplies_upgrade_marker():
    existing = (
        "<html><body><canvas id='game'></canvas>"
        "<script id=\"agent-steer-canvas-graphics-upgrade\">old</script>"
        "<!-- agent-steer fallback: preserved existing browser game after old -->"
        "</body></html>"
    )
    messages = [
        {"role": "user", "content": "create a 1942 game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({
                    "filePath": "index.html",
                    "content": existing,
                }),
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "improve the game with better graphics"},
    ]
    _path, content = ag._synthetic_write_fallback_artifact(messages)
    assert "agent-steer-canvas-graphics-upgrade" in content
    assert "old</script>" not in content
    assert content.count("agent-steer-canvas-graphics-upgrade") == 1


def test_bad_write_target_detector_catches_ls_output():
    assert ag._looks_like_bad_write_target(
        "/Users/aicoder/src/zzz-test/ls_output.txt", "ls -R"
    )
    assert ag._looks_like_bad_write_target(
        "/Users/aicoder/src/zzz-test/ls.txt", ""
    )
    assert ag._looks_like_bad_write_target(
        "/Users/aicoder/src/zzz-test/planes_config.json",
        '{"planes":[{"type":"fighter"}]}',
    )
    assert ag._looks_like_bad_write_target(
        "/Users/aicoder/src/zzz-test/planes.js",
        "// This is a placeholder for planes",
    )
    assert not ag._looks_like_bad_write_target("index.html", "<canvas></canvas>")


def test_noop_write_detector_catches_read_first_content():
    messages = [{"role": "user", "content": "improve the game with better graphics"}]
    assert ag._looks_like_noop_write(
        "index.html",
        "// No changes needed to the file if it's empty, but I need to read it first",
        messages,
    )


def test_repair_bad_streaming_write_delta_replaces_ls_output():
    existing = "<html><body><canvas id='game'></canvas></body></html>"
    messages = [
        {"role": "user", "content": "create a 1942 game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({
                    "filePath": "index.html",
                    "content": existing,
                }),
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "improve the game with better graphics"},
    ]
    d = {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "id": "bad",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": "/Users/aicoder/src/zzz-test/ls_output.txt",
                "content": "ls -R",
            }),
        },
    }]}}]}
    repaired, changed = ag._repair_bad_streaming_write_delta(
        d,
        body={"messages": messages},
        messages=messages,
        writers={
            "write_name": "write",
            "write_path_field": "filePath",
            "write_content_field": "content",
        },
    )
    assert changed is True
    args = json.loads(
        repaired["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["filePath"].endswith(("1942.html", "index.html"))
    assert "agent-steer-visual-polish" in args["content"]
    assert "ls_output.txt" not in args["filePath"]


def test_repair_bad_streaming_write_delta_replaces_planes_json():
    messages = [
        {
            "role": "user",
            "content": "read planes.png and create 5 different types of planes to incorporate onto 1942",
        },
    ]
    d = {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "id": "bad",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": "/Users/aicoder/src/zzz-test/planes_config.json",
                "content": '{"planes":[{"type":"fighter"}]}',
            }),
        },
    }]}}]}
    repaired, changed = ag._repair_bad_streaming_write_delta(
        d,
        body={"messages": messages},
        messages=messages,
        writers={
            "write_name": "write",
            "write_path_field": "filePath",
            "write_content_field": "content",
        },
    )
    assert changed is True
    args = json.loads(
        repaired["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["filePath"].endswith(("1942.html", "index.html"))
    assert "planeTypes" in args["content"]
    assert "Emerald Falcon" in args["content"]


def test_repair_bad_streaming_write_delta_replaces_noop_write():
    messages = [
        {"role": "user", "content": "Edit /Users/aicoder/src/zzz-test/index.html directly. Improve graphics."},
    ]
    d = {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "id": "bad",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": "/Users/aicoder/src/zzz-test/index.html",
                "content": "// No changes needed to the file if it's empty, but I need to read it first",
            }),
        },
    }]}}]}
    repaired, changed = ag._repair_bad_streaming_write_delta(
        d,
        body={"messages": messages},
        messages=messages,
        writers={
            "write_name": "write",
            "write_path_field": "filePath",
            "write_content_field": "content",
        },
    )
    assert changed is True
    args = json.loads(
        repaired["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["filePath"] == "/Users/aicoder/src/zzz-test/index.html"
    assert "<canvas" in args["content"]


def test_repair_empty_ls_txt_cache_write_uses_index_html():
    messages = [
        {"role": "user", "content": "can we ensure that web caching is disabled when loading"},
    ]
    d = {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "id": "bad",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": "/Users/aicoder/src/zzz-test/ls.txt",
                "content": "",
            }),
        },
    }]}}]}
    repaired, changed = ag._repair_bad_streaming_write_delta(
        d,
        body={"messages": messages},
        messages=messages,
        writers={
            "write_name": "write",
            "write_path_field": "filePath",
            "write_content_field": "content",
        },
    )
    assert changed is True
    args = json.loads(
        repaired["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["filePath"].endswith(("1942.html", "index.html"))
    assert "gemma4-mlx-kilo-cache-bust" in args["content"]
    assert "Cache-Control" in args["content"]


# ---------------------------------------------------------------------------
# _strip_planning_tools_if_stuck
# ---------------------------------------------------------------------------

def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _asst_call(name: str, args: str = "{}") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": name, "arguments": args},
        }],
    }


def test_strip_planning_noop_when_no_planning_call():
    """First turn — no assistant history.  Don't strip anything."""
    tools = [_tool("write"), _tool("todowrite")]
    msgs = [{"role": "user", "content": "create arkanoid"}]
    kept, removed = ag._strip_planning_tools_if_stuck(msgs, tools)
    assert removed == []
    assert kept is tools  # same reference, untouched


def test_strip_planning_noop_after_write_call():
    """Once the model has called a write tool, planning tools may stay
    — the model has shown forward progress and may legitimately update
    its todo list."""
    tools = [_tool("write"), _tool("todowrite")]
    msgs = [
        {"role": "user", "content": "create arkanoid"},
        _asst_call("write", '{"filePath":"index.html","content":"..."}'),
        {"role": "tool", "content": "ok"},
        _asst_call("todowrite"),  # legitimate post-write planning
    ]
    kept, removed = ag._strip_planning_tools_if_stuck(msgs, tools)
    assert removed == []


def test_strip_planning_fires_after_todowrite_only():
    """Canonical case from production logs: model called `todowrite`,
    nothing else.  Strip planning tools to force a write."""
    tools = [_tool("write"), _tool("edit"), _tool("todowrite"),
             _tool("todoread")]
    msgs = [
        {"role": "user", "content": "create arkanoid"},
        _asst_call("todowrite"),
        {"role": "tool", "content": ""},
    ]
    kept, removed = ag._strip_planning_tools_if_stuck(msgs, tools)
    assert "todowrite" in removed
    assert "todoread" in removed
    kept_names = {(t.get("function") or {}).get("name") for t in kept}
    assert "write" in kept_names
    assert "edit" in kept_names
    assert "todowrite" not in kept_names
    assert "todoread" not in kept_names


def test_strip_planning_handles_empty_tools_list():
    kept, removed = ag._strip_planning_tools_if_stuck([], [])
    assert removed == []


def test_strip_planning_handles_none_tools():
    kept, removed = ag._strip_planning_tools_if_stuck([], None)
    assert removed == []


# ---------------------------------------------------------------------------
# _force_write_tool_for_create_turn
# ---------------------------------------------------------------------------

def test_force_write_tool_first_create_turn_filters_and_sets_choice():
    body = {}
    tools = [_tool("write"), _tool("edit"), _tool("todowrite")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
    ]
    out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is True
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "write"},
    }
    assert len(out_tools) == 1
    assert (out_tools[0].get("function") or {}).get("name") == "write"
    assert "MANDATORY TOOL CHOICE" in out_msgs[0]["content"]


def test_force_write_tool_create_turn_forces_after_assistant():
    body = {}
    tools = [_tool("write"), _tool("edit"), _tool("todowrite")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant", "content": "I will start by planning."},
    ]
    out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is True
    assert body["tool_choice"]["function"]["name"] == "write"
    assert len(out_tools) == 1


def test_force_write_tool_first_create_turn_noop_non_create():
    body = {}
    tools = [_tool("write"), _tool("edit")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "explain this repo"},
    ]
    _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is False
    assert "tool_choice" not in body


def test_force_write_tool_for_improve_prompt():
    body = {}
    tools = [_tool("write"), _tool("edit")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "improve the game with better graphics"},
    ]
    _out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is True
    assert body["tool_choice"]["function"]["name"] == "write"
    assert len(out_tools) == 1


def test_force_write_tool_read_first_no_prior_assistant_noop():
    """'read spec.txt and create …' on the first turn must NOT force write.

    The model needs to read the file before it can write anything.
    Forcing tool_choice=write strips all other tools (including read) and
    causes the model to write garbage without ever reading the spec.
    """
    body = {}
    tools = [_tool("write"), _tool("edit"), _tool("read")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "read 1942-spec.txt and create this web game"},
    ]
    _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is False, (
        "read-first task on first turn should NOT force tool_choice=write "
        f"(got forced={forced}, body={body})"
    )
    assert "tool_choice" not in body


def test_force_write_tool_read_first_with_prior_assistant_forces():
    """After the model has already had a turn (read the spec), force write."""
    body = {}
    tools = [_tool("write"), _tool("edit"), _tool("read")]
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "read 1942-spec.txt and create this web game"},
        {"role": "assistant", "content": "I have read the spec. Now I will write the game."},
    ]
    _out_msgs, out_tools, forced = ag._force_write_tool_for_create_turn(
        body, messages, tools
    )
    assert forced is True, (
        "read-first task AFTER an assistant turn should force tool_choice=write"
    )
    assert body["tool_choice"]["function"]["name"] == "write"
    assert len(out_tools) == 1


def test_force_write_tool_other_read_variants_noop():
    """Other read-first verbs (load, open, review) also defer write forcing."""
    variants = [
        "load config.json and build the app",
        "open the spec and implement the feature",
        "review README.md and update the code",
        "check the existing file and fix the bug",
    ]
    for text in variants:
        body = {}
        tools = [_tool("write"), _tool("edit"), _tool("read")]
        messages = [
            {"role": "user", "content": text},
        ]
        _out_msgs, _out_tools, forced = ag._force_write_tool_for_create_turn(
            body, messages, tools
        )
        assert forced is False, (
            f"read-first variant {text!r} should NOT force write on first turn"
        )


def test_get_forced_tool_name_openai_shape():
    assert ag._get_forced_tool_name({
        "type": "function",
        "function": {"name": "write"},
    }) == "write"
    assert ag._get_forced_tool_name("auto") == ""
    assert ag._get_forced_tool_name({}) == ""


def test_short_final_after_write_detects_write_tool_result():
    messages = [
        {"role": "user", "content": "create game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "write", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    assert ag._short_final_after_write(messages, [_tool("write")]) is True


def test_short_final_after_write_noop_without_tool_result():
    messages = [
        {"role": "user", "content": "create game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "write", "arguments": "{}"},
        }]},
    ]
    assert ag._short_final_after_write(messages, [_tool("write")]) is False


def test_short_final_after_write_uses_latest_tool_result_only():
    messages = [
        {"role": "user", "content": "create game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {"name": "write", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "r1", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "r1", "content": "contents"},
    ]
    assert ag._short_final_after_write(messages, [_tool("write"), _tool("read_file")]) is False


def test_proxy_warning_marker_strips_proxy_variants():
    text = "ok\n\n---\n[PROXY: internal note]"
    assert text.split(ag._PROXY_WARNING_MARKER, 1)[0] == "ok"


def test_synthetic_post_write_response_body_mentions_file():
    messages = [
        {"role": "user", "content": "create game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": '{"filePath":"index.html","content":"<canvas></canvas>"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
    ]
    body = ag._synthetic_post_write_response_body({
        "model": "m",
        "messages": messages,
        "tools": [_tool("write")],
    })
    assert body is not None
    assert b"Created `index.html`." in body
    assert b"Task: create game" in body
    assert b"Included:" in body
    assert b"HTML5 canvas game" in body
    assert b"<|channel>" not in body


def test_synthetic_post_write_response_uses_latest_user_task():
    messages = [
        {"role": "user", "content": "suggest improvements to 1942 for the user experience"},
        {"role": "assistant", "content": "Suggestions..."},
        {"role": "user", "content": "implement 1,2,3,4,5,7,8"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": '{"filePath":"/Users/aicoder/src/zzz-test/index.html","content":"<canvas></canvas>"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
    ]
    body = ag._synthetic_post_write_response_body({
        "model": "m",
        "messages": messages,
        "tools": [_tool("write")],
    })
    assert body is not None
    assert b"Task: implement 1,2,3,4,5,7,8" in body
    assert b"Task: suggest improvements" not in body


def test_synthetic_post_write_response_skips_failed_tool_result():
    messages = [
        {"role": "user", "content": "create game"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {
                "name": "write",
                "arguments": '{"filePath":"index.html","content":"<canvas></canvas>"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "w1", "content": "no updates were written to disk"},
    ]
    body = ag._synthetic_post_write_response_body({
        "model": "m",
        "messages": messages,
        "tools": [_tool("write")],
    })
    assert body is None


def test_preemptive_deterministic_write_detects_high_risk_tasks():
    assert not ag._should_preemptive_deterministic_write([
        {"role": "user", "content": "can we ensure that web caching is disabled when loading"}
    ])
    assert not ag._should_preemptive_deterministic_write([
        {"role": "user", "content": "Create a complete single-file 1942-inspired arcade shooter for a web browser"}
    ])
    assert not ag._should_preemptive_deterministic_write([
        {"role": "user", "content": "suggest improvements to 1942 for the user experience"},
        {"role": "assistant", "content": (
            "Suggestions for improving the project user experience:\n"
            "3. Tune difficulty progression\n"
            "7. Preserve browser performance by capping particles/enemies and requestAnimationFrame\n"
            "8. Disable stale browser caching"
        )},
        {"role": "user", "content": "implement 3, 4, 7, 8, 2, 1"},
    ])
    assert not ag._should_preemptive_deterministic_write([
        {"role": "user", "content": "what files are in this project?"}
    ])


def test_preemptive_write_branch_removed():
    src = (Path(__file__).resolve().parent.parent / "gemma4_mlx_kilo_proxy.py").read_text()
    assert "[preemptive-write]" not in src
    assert "direct disk write safety net" not in src


def test_validated_write_artifact_repairs_incomplete_game():
    path, content, features = ag._validated_write_artifact(
        [{"role": "user", "content": "Create a complete single-file 1942-inspired arcade shooter for a web browser"}],
        "/tmp/1942.html",
        "<html><body>placeholder</body></html>",
    )
    assert path.endswith(".html")
    assert "drawPlane" in content
    assert "spawnExplosion" in content
    assert "- HTML5 canvas game" in features
    assert "- bullet collisions" in features


def test_validated_write_artifact_adds_cache_busting():
    path, content, features = ag._validated_write_artifact(
        [{"role": "user", "content": "can we ensure that web caching is disabled when loading"}],
        "index.html",
        "<html><head></head><body><canvas></canvas></body></html>",
    )
    assert path == "index.html"
    assert "gemma4-mlx-kilo-cache-bust" in content
    assert "gemma4-mlx-kilo-write-stamp" in content
    assert "- cache disabled" in features


def test_write_file_direct_writes_absolute_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.html")
        ok, msg = ag._write_file_direct(path, "<html>ok</html>")
        assert ok, msg
        assert Path(path).read_text() == "<html>ok</html>"


def test_readonly_synthetic_suggestions_are_concise():
    text = ag._synthetic_readonly_suggestions([
        {"role": "user", "content": "suggest improvements to 1942 for the user experience"}
    ])
    assert "1942 game" in text
    assert "requestAnimationFrame" in text
    assert len(text) < 1400


def test_readonly_intent_uses_latest_real_user_turn():
    assert ag._is_readonly_intent([
        {"role": "user", "content": "suggest improvements to 1942 for the user experience"},
        {"role": "assistant", "content": "Suggestions for improving 1942 game user experience..."},
        {"role": "user", "content": "implement 3, 4, 7, 8, 2, 1"},
    ]) is False


def test_readonly_intent_still_detects_latest_suggestion_request():
    assert ag._is_readonly_intent([
        {"role": "user", "content": "implement the game"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "suggest improvements to 1942 for the user experience"},
    ]) is True


# ---------------------------------------------------------------------------
# _upgrade_stall_nudge_after_strip
# ---------------------------------------------------------------------------

def _trailing_stall_nudge() -> dict:
    """Mimic what `_break_text_stall` appends after a planning stall."""
    return {
        "role": "user",
        "content": (
            "STOP — PLANNING STALL DETECTED.\n\n"
            "Your last 1 assistant turns produced only planning text..."
        ),
    }


def test_upgrade_nudge_replaces_existing_stall():
    """When a `PLANNING STALL DETECTED` user message is already at the
    tail, REPLACE its content with the NO_PLANNING variant rather than
    stacking two stall nudges."""
    tools = [_tool("write"), _tool("edit")]
    msgs = [
        {"role": "user", "content": "create arkanoid game"},
        _asst_call("todowrite"),
        {"role": "tool", "content": ""},
        _trailing_stall_nudge(),
    ]
    out = ag._upgrade_stall_nudge_after_strip(msgs, ["todowrite", "todoread"], tools)
    assert len(out) == len(msgs), "must not stack — total count unchanged"
    assert out[-1]["role"] == "user"
    body = out[-1]["content"]
    # Identifies the stripped tools by name
    assert "`todowrite`" in body
    assert "`todoread`" in body
    # Mentions the user's task verbatim
    assert "arkanoid" in body
    # Forbids the Harmony / CoT prefixes
    assert "<|channel>" in body
    assert "<|think|>" in body
    # Anchors to write tool
    assert "`write`" in body


def test_upgrade_nudge_appends_when_no_existing_stall():
    """If the model's previous turn did not produce a planning stall
    (rare but possible — e.g. the model called `todowrite` and the
    proxy stripped it for the NEXT turn before `_break_text_stall`
    fired), APPEND the harsher nudge as a fresh user message."""
    tools = [_tool("write")]
    msgs = [
        {"role": "user", "content": "create arkanoid game"},
        _asst_call("todowrite"),
        {"role": "tool", "content": ""},
    ]
    out = ag._upgrade_stall_nudge_after_strip(msgs, ["todowrite"], tools)
    assert len(out) == len(msgs) + 1
    assert out[-1]["role"] == "user"
    body = out[-1]["content"]
    assert "arkanoid" in body
    assert "`todowrite`" in body
    assert "<|channel>" in body


def test_upgrade_nudge_noop_when_no_removals():
    """Empty `removed_tool_names` -> messages unchanged."""
    msgs = [{"role": "user", "content": "create a thing"}]
    out = ag._upgrade_stall_nudge_after_strip(msgs, [], None)
    assert out is msgs


def test_upgrade_nudge_noop_when_not_create_task():
    """A non-create user prompt (e.g. read-only question) -> the
    NO_PLANNING nudge template references `task` so it's only emitted
    when we have a concrete user task to plug in."""
    tools = [_tool("write")]
    msgs = [
        {"role": "user", "content": "what is the capital of france"},
        _asst_call("todowrite"),
    ]
    out = ag._upgrade_stall_nudge_after_strip(msgs, ["todowrite"], tools)
    # Either unchanged or no NO_PLANNING nudge appended
    if out is not msgs:
        # If the helper appended anything, it must not be the NO_PLANNING
        # nudge for a non-create task.
        tail = out[-1].get("content", "")
        assert "PLANNING STALL DETECTED (planning tools have been REMOVED)" not in tail


# ---------------------------------------------------------------------------
# _break_text_stall — byte-cap-aborted text collapse detection (Bug 6,
# 2026-05-13).  After the Harmony-leak fix banned `<|channel>`/`<|think|>`
# tokens at the sampler, Gemma-4 distillations sometimes start a plain
# planning checklist that token-level repeats ("break it into pieces of
# code and break it into pieces of code...") until the upstream
# `[text-mode-runaway]` guard byte-caps at `_TEXT_MODE_MAX_CHARS_AGENTIC`
# (~4 KB, ~16s of wasted decode).  Kilo retries by re-sending the
# conversation as `[system, user, assistant_collapse(~4 KB), user_retry]`
# which the legacy walk-back stalls on because the trailing real-user
# message terminates the scan before `consecutive_text` reaches 2.
# ---------------------------------------------------------------------------

def test_collapse_threshold_constant_is_sensible():
    """The collapse threshold must be SMALLER than the byte-cap so a
    truncated runaway always trips it, but LARGER than legit Kilo
    planning prose (typically 200-1000 chars) so we never false-fire."""
    assert ag._TEXT_STALL_COLLAPSE_THRESHOLD < ag._TEXT_MODE_MAX_CHARS_AGENTIC, (
        "collapse threshold must be smaller than the byte-cap, otherwise "
        "a byte-cap-aborted text turn would never trip it"
    )
    assert ag._TEXT_STALL_COLLAPSE_THRESHOLD >= 1024, (
        "collapse threshold should be at least 1 KB to avoid firing on "
        "brief legitimate planning prose"
    )


def _make_collapse_text(size: int) -> str:
    """Generate a string that looks like a byte-cap-aborted runaway:
    a short planning preamble followed by tail-end repetition garbage
    (the exact pattern observed in production 2026-05-13 15:07)."""
    preamble = (
        "I'll start by exploring the directory structure.\n"
        "I'll create a single HTML file.\n"
        "I'll implement the game logic.\n"
    )
    repeat_unit = "I'break into pieces of code and "
    if size <= len(preamble):
        return preamble[:size]
    pad = (size - len(preamble)) // len(repeat_unit) + 1
    body = repeat_unit * pad
    return (preamble + body)[:size]


def test_break_text_stall_fires_on_kilo_retry_with_collapse():
    """Bug 6 regression: production observation 2026-05-13 15:07.

    Conversation tail = [user_orig, assistant_collapse(~4 KB), user_retry].
    The walk-back used to bail at `user_retry` without inspecting the
    preceding 4 KB assistant collapse, so the HARD nudge never fired
    and the model looped indefinitely.  After the fix, the pre-scan
    must catch the collapse and fire the HARD nudge."""
    collapse_text = _make_collapse_text(ag._TEXT_MODE_MAX_CHARS_AGENTIC - 2)
    assert len(collapse_text) >= ag._TEXT_STALL_COLLAPSE_THRESHOLD
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant", "content": collapse_text},
        {"role": "user", "content": "create arkanoid game with powerups"},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert len(out) == len(messages) + 1, (
        "byte-cap-aborted collapse on a create task must trigger the "
        "HARD nudge — pre-scan failed to detect the retry pattern"
    )
    nudge = out[-1]
    assert nudge["role"] == "user"
    assert "PLANNING STALL DETECTED" in nudge["content"]
    assert "create arkanoid game" in nudge["content"]


def test_break_text_stall_fires_on_trailing_collapse_no_retry():
    """Conversation tail = [user_orig, assistant_collapse(~4 KB)] (no
    Kilo retry yet).  Walk-back finds the trailing collapse directly:
    `consecutive_text=1`, `trailing_text_chars>=threshold`,
    `trailing_collapse=True`, so the nudge must fire."""
    collapse_text = _make_collapse_text(ag._TEXT_MODE_MAX_CHARS_AGENTIC - 2)
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant", "content": collapse_text},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert len(out) == len(messages) + 1
    assert "PLANNING STALL DETECTED" in out[-1]["content"]


def test_break_text_stall_skips_short_planning_text():
    """A SHORT single text-only turn ("Let me start by creating the
    file.") is NOT a stall — the model may legitimately be about to
    emit a tool call.  Must not false-fire."""
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant",
         "content": "Let me start by creating the HTML file."},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert out == messages, (
        f"short single text turn must NOT fire — got {len(out)} msgs, "
        f"expected {len(messages)}"
    )


def test_break_text_stall_skips_collapse_when_not_create_task():
    """The HARD nudge is create-task-only — non-create tasks (e.g.,
    "explain X", "what is Y") must not get the create-tailored
    template even if the model produced a collapse."""
    collapse_text = _make_collapse_text(ag._TEXT_MODE_MAX_CHARS_AGENTIC - 2)
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "explain how the proxy works"},
        {"role": "assistant", "content": collapse_text},
        {"role": "user", "content": "explain how the proxy works"},
    ]
    tools = [{"type": "function", "function": {"name": "write"}}]
    out = ag._break_text_stall(messages, tools)
    assert out == messages, (
        "non-create task must not get the HARD `write` nudge even on "
        "a byte-cap-aborted collapse"
    )


def test_break_text_stall_skips_collapse_when_write_already_done():
    """If the model already produced a write call before the collapse,
    we should NOT inject another nudge — the file is on disk, that
    work survives.  The `write_calls > 0` early-bail must take
    precedence over both `trailing_collapse` and `recent_collapse`."""
    collapse_text = _make_collapse_text(ag._TEXT_MODE_MAX_CHARS_AGENTIC - 2)
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "1", "type": "function", "function": {
                "name": "write",
                "arguments": '{"filePath":"index.html","content":"<html></html>"}',
            }}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "assistant", "content": collapse_text},
        {"role": "user", "content": "create arkanoid game with powerups"},
    ]
    tools = [{"type": "function", "function": {"name": "write"}}]
    out = ag._break_text_stall(messages, tools)
    assert out == messages, (
        "write already done — collapse must not retrigger the nudge"
    )


def test_break_text_stall_legacy_two_short_turns_still_fires():
    """Backwards compat: the legacy `consecutive_text >= 2` rule must
    still fire when two short text-only assistant turns are separated
    by a tool message (the original Bug 4 shape) even though neither
    individual turn exceeds the new collapse threshold."""
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant",
         "content": "First, I'll explore the directory."},
        {"role": "assistant",
         "content": "Now let me think about the structure."},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert len(out) == len(messages) + 1, (
        "legacy 2-consecutive-text-turn rule must still fire"
    )


def test_marker_collapse_helper_detects_channel_thought_loop():
    """Short control-marker loops like `<channel|><thought>` repeated
    should be recognized as collapse even when total chars are below
    `_TEXT_STALL_COLLAPSE_THRESHOLD`."""
    txt = "<channel|><thought> " * 12
    assert ag._looks_like_control_marker_collapse(txt)


def test_break_text_stall_fires_on_short_marker_collapse_retry():
    """Regression for 2026-05-13 15:26 pattern:
    sentence-level runaway abort at ~519 chars due to repeated
    `<channel|><thought>`, followed by Kilo retry. This is too short
    for the 2 KB threshold, so marker-collapse detection must fire."""
    collapse_text = (
        "I will start by exploring the directory. "
        + ("<channel|><thought> " * 14)
    )
    assert len(collapse_text) < ag._TEXT_STALL_COLLAPSE_THRESHOLD
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant", "content": collapse_text},
        {"role": "user", "content": "create arkanoid game with powerups"},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert len(out) == len(messages) + 1
    assert "PLANNING STALL DETECTED" in out[-1]["content"]


def test_break_text_stall_skips_short_marker_like_text():
    """Single incidental marker mention should not false-fire."""
    messages = [
        {"role": "system", "content": "You are Kilo."},
        {"role": "user", "content": "create arkanoid game with powerups"},
        {"role": "assistant",
         "content": "I noticed token <channel|> in tokenizer config."},
    ]
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"properties": {
            "filePath": {}, "content": {}}}}}]
    out = ag._break_text_stall(messages, tools)
    assert out == messages


def test_token_repeat_with_delims_detector_matches_the_loop():
    """Regex should catch punctuation-delimited token loops such as
    `the-the-the-...` seen in production."""
    txt = ("the-" * 20) + "done"
    m = ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(txt)
    assert m is not None
    assert m.group(1).lower() == "the"


def test_token_repeat_with_delims_detector_no_false_positive():
    """Natural prose with occasional delimiters must not match."""
    txt = (
        "piecewise-based-logic-engine for the game; "
        "high-speed, robust-enough architecture, and tests."
    )
    assert ag._TEXT_TOKEN_REPEAT_WITH_DELIMS_RE.search(txt) is None


def test_forced_write_prefill_grace_exceeds_agentic_low_chunk_deadline():
    """Forced write prefill needs longer than the normal 10s agentic
    low-chunk-rate deadline, otherwise tool_choice=write gets killed before
    the first real delta arrives."""
    assert ag._LOW_CHUNK_RATE_FORCED_WRITE_PREFILL_S > ag._LOW_CHUNK_RATE_AFTER_AGENTIC_S
    assert ag._LOW_CHUNK_RATE_FORCED_WRITE_PREFILL_S == 45.0


def test_write_tool_slow_rate_grace_exceeds_agentic_stream_grace():
    """A real write stream may pause after its first args delta; it needs
    longer grace than malformed agentic tool loops."""
    assert ag._LOW_TOKEN_RATE_WRITE_TOOL_AFTER_S > ag._LOW_TOKEN_RATE_STREAM_TOOL_AFTER_AGENTIC_S
    assert ag._STALL_ABORT_WRITE_TOOL_S > ag._STALL_ABORT_STREAM_TOOL_S
    assert "write" in ag._WRITE_TOOL_KEYS


def test_format_stream_tps_reports_decode_rate():
    assert ag._format_stream_tps(100, 10.0) == "10.0 tok/s"
    assert ag._format_stream_tps(100, 10.0, 2.0) == (
        "10.0 tok/s total, 12.5 tok/s decode"
    )


# ---------------------------------------------------------------------------
# Sanity / parse-clean
# ---------------------------------------------------------------------------

def test_no_planning_template_has_required_placeholders():
    """The harsher template must accept all the placeholders the helper
    feeds it.  A missing `{removed}` would cause a runtime KeyError."""
    tpl = ag._TEXT_STALL_HARD_CREATE_NO_PLANNING
    formatted = tpl.format(
        task="example task",
        removed="`todowrite`",
        write_tool="write",
        path_field="filePath",
        content_field="content",
        target_path="/tmp/example/index.html",
    )
    assert "example task" in formatted
    assert "`todowrite`" in formatted
    assert "<|channel>" in formatted
    assert "tool_calls" in formatted


def test_gemma4_mlx_kilo_proxy_module_parses_cleanly():
    """Imports above already proved this; assert explicitly so a future
    syntax break is reported as a test failure, not an import error."""
    import ast
    src = (Path(__file__).resolve().parent.parent / "gemma4_mlx_kilo_proxy.py").read_text()
    ast.parse(src)


# ---------------------------------------------------------------------------
# Upstream patch — patches/api/models.py contains the logit_bias field
# ---------------------------------------------------------------------------

def test_patches_api_models_has_logit_bias_field():
    """The in-tree mirror of `api/models.py` MUST declare a `logit_bias`
    field on `ChatCompletionRequest` — otherwise the upstream schema
    rejects the proxy's bias injection."""
    patch_src = (
        Path(__file__).resolve().parent.parent
        / "patches" / "api" / "models.py"
    ).read_text()
    assert "logit_bias" in patch_src, \
        "patches/api/models.py must declare logit_bias on ChatCompletionRequest"


def test_patches_server_forwards_logit_bias():
    """The in-tree mirror of `server.py` MUST forward `logit_bias` from
    the request body into the engine kwargs."""
    patch_src = (
        Path(__file__).resolve().parent.parent / "patches" / "server.py"
    ).read_text()
    assert "logit_bias" in patch_src, \
        "patches/server.py must forward logit_bias into chat kwargs"


def test_patches_engine_simple_uses_logit_bias():
    """The in-tree mirror of `engine/simple.py` MUST pass `logit_bias`
    into `make_logits_processors` (the sampler hook)."""
    patch_src = (
        Path(__file__).resolve().parent.parent
        / "patches" / "engine" / "simple.py"
    ).read_text()
    assert "logit_bias" in patch_src
    assert "make_logits_processors" in patch_src or "logits_processors" in patch_src


def test_patches_engine_simple_rebinds_media_streams():
    """Media MLLM routing must refresh MLX streams before calling mlx_vlm.

    mlx_vlm keeps a module-level generation_stream; after a text-only route
    runs in another worker context, media requests can otherwise fail with
    `There is no Stream(gpu, N) in current thread`.
    """
    patch_src = (
        Path(__file__).resolve().parent.parent
        / "patches" / "engine" / "simple.py"
    ).read_text()
    assert "mx.new_stream(mx.default_device())" in patch_src
    assert "_bind_worker_generation_streams(thread_local=True, clear_cache=True)" in patch_src
    assert "mx.clear_cache()" in patch_src
    media_branch = patch_src[patch_src.index('logger.info("Media request → MLLM path")'):]
    media_branch = media_branch[:media_branch.index("# For LLM, apply chat template")]
    assert "asyncio.to_thread" not in media_branch
    assert "_run_blocking_serialized(run_stream)" not in media_branch


def test_patches_mlx_vlm_generate_refreshes_worker_stream():
    """The in-tree mlx_vlm.generate patch must refresh worker-owned streams."""
    patch_src = (
        Path(__file__).resolve().parent.parent / "patches" / "generate.py"
    ).read_text()
    assert "def _refresh_generation_stream" in patch_src
    assert "generation_stream = mx.new_stream(mx.default_device())" in patch_src
    assert "stream = _refresh_generation_stream()" in patch_src
    assert "with wired_limit(model, [stream])" in patch_src


if __name__ == "__main__":
    # Allow running without pytest.  Each test function is collected by
    # name prefix, executed, and reported.
    import traceback
    failed = []
    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — surface everything
            failed.append((name, traceback.format_exc()))
            print(f"FAIL  {name}: {exc}")
        else:
            passed += 1
            print(f"PASS  {name}")
    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        print("\n=== failure detail ===")
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    import pytest as _pytest, sys as _sys
    _sys.exit(_pytest.main([__file__, "-v"] + _sys.argv[1:]))
