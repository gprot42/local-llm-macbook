#!/usr/bin/env python3
"""Unit tests for openai_proxy conversion helpers (no live Ollama)."""
from __future__ import annotations

import json
import unittest

from openai_proxy import (
    _openai_tool_calls_delta,
    _parse_tool_arguments,
    _sanitize_tool_calls,
    _sanitize_tools,
    alias_model_ids,
    flatten_message_content,
    native_chat_url,
    normalize_messages_for_ollama,
    ollama_chat_to_openai_json,
    ollama_ndjson_to_openai_events,
    openai_chat_to_ollama,
    resolve_think,
    thinking_requested,
)


def _fresh_state(model: str = "glm") -> dict:
    return {
        "id": "chatcmpl-1",
        "created": 1,
        "model": model,
        "role_sent": False,
        "saw_tool": False,
    }


def _parse_sse(events: list[bytes]) -> list:
    out: list = []
    for ev in events:
        text = ev.decode().strip()
        if text == "data: [DONE]":
            out.append("[DONE]")
            continue
        out.append(json.loads(text[len("data:") :].strip()))
    return out


class ThinkResolutionTests(unittest.TestCase):
    def test_default_off_when_unset(self) -> None:
        self.assertIsNone(thinking_requested({}))
        self.assertFalse(resolve_think({}))

    def test_think_true_passthrough(self) -> None:
        self.assertTrue(resolve_think({"think": True}))

    def test_think_low_passthrough(self) -> None:
        self.assertEqual(resolve_think({"think": "low"}), "low")

    def test_enable_thinking_false(self) -> None:
        self.assertFalse(
            resolve_think({"chat_template_kwargs": {"enable_thinking": False}})
        )

    def test_thinking_type_disabled(self) -> None:
        self.assertFalse(resolve_think({"thinking": {"type": "disabled"}}))


class RequestConversionTests(unittest.TestCase):
    def test_alias_model_ids_adds_untagged(self) -> None:
        raw = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": "glm-4.7-flash-heretic-q8:latest",
                        "object": "model",
                        "owned_by": "library",
                    }
                ],
            }
        ).encode()
        data = json.loads(alias_model_ids(raw))
        ids = [m["id"] for m in data["data"]]
        self.assertIn("glm-4.7-flash-heretic-q8:latest", ids)
        self.assertIn("glm-4.7-flash-heretic-q8", ids)

    def test_native_chat_url(self) -> None:
        self.assertEqual(
            native_chat_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/api/chat",
        )

    def test_openai_chat_to_ollama_defaults_think_false(self) -> None:
        native = openai_chat_to_ollama(
            {
                "model": "glm-4.7-flash-heretic-q8",
                "stream": True,
                "max_tokens": 64,
                "temperature": 0.6,
                "messages": [{"role": "user", "content": "pong"}],
            }
        )
        self.assertEqual(native["think"], False)
        self.assertTrue(native["stream"])
        self.assertEqual(native["options"]["num_predict"], 64)
        self.assertEqual(native["options"]["temperature"], 0.6)

    def test_tool_arguments_string_becomes_object(self) -> None:
        msgs = normalize_messages_for_ollama(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                }
            ]
        )
        args = msgs[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, {"path": "a.py"})

    def test_content_part_array_becomes_string(self) -> None:
        text, images = flatten_message_content(
            [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        )
        self.assertEqual(text, "hello\nworld")
        self.assertEqual(images, [])

    def test_kilo_user_message_array_normalized(self) -> None:
        msgs = normalize_messages_for_ollama(
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a coding agent."}]},
                {"role": "user", "content": [{"type": "text", "text": "Fix the hang."}]},
            ]
        )
        self.assertEqual(msgs[0]["content"], "You are a coding agent.")
        self.assertEqual(msgs[1]["content"], "Fix the hang.")
        self.assertIsInstance(msgs[0]["content"], str)
        self.assertIsInstance(msgs[1]["content"], str)
        self.assertNotIn("cache_control", msgs[0])

    def test_reasoning_array_becomes_thinking_string(self) -> None:
        msgs = normalize_messages_for_ollama(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "pong"}],
                    "reasoning_content": [{"type": "text", "text": "think hard"}],
                }
            ]
        )
        self.assertEqual(msgs[0]["content"], "pong")
        self.assertEqual(msgs[0]["thinking"], "think hard")
        self.assertIsInstance(msgs[0]["thinking"], str)

    def test_developer_role_becomes_system(self) -> None:
        msgs = normalize_messages_for_ollama(
            [{"role": "developer", "content": [{"type": "input_text", "text": "rules"}]}]
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "rules")

    def test_native_payload_has_no_openai_extra_keys(self) -> None:
        native = openai_chat_to_ollama(
            {
                "model": "glm",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "strict": True,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        self.assertEqual(set(native["messages"][0]), {"role", "content"})
        self.assertNotIn("strict", native["tools"][0]["function"])

    def test_image_url_extracted(self) -> None:
        text, images = flatten_message_content(
            [
                {"type": "text", "text": "what is this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ]
        )
        self.assertEqual(text, "what is this")
        self.assertEqual(images, ["QUJD"])


class NdjsonConversionTests(unittest.TestCase):
    def _parse(self, events: list[bytes]) -> list[dict | str]:
        out: list[dict | str] = []
        for ev in events:
            text = ev.decode().strip()
            if text == "data: [DONE]":
                out.append("[DONE]")
                continue
            payload = json.loads(text[len("data:") :].strip())
            out.append(payload)
        return out

    def test_content_then_done(self) -> None:
        state = {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "glm",
            "role_sent": False,
            "saw_tool": False,
        }
        events = ollama_ndjson_to_openai_events(
            {"message": {"role": "assistant", "content": "pong", "thinking": ""}},
            state,
        )
        events += ollama_ndjson_to_openai_events(
            {
                "model": "glm",
                "message": {"role": "assistant", "content": "", "thinking": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 2,
            },
            state,
        )
        parsed = self._parse(events)
        self.assertEqual(parsed[0]["choices"][0]["delta"].get("content"), "pong")
        self.assertEqual(parsed[1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(parsed[2]["usage"]["completion_tokens"], 2)
        self.assertEqual(parsed[3], "[DONE]")

    def test_tool_calls_finish_reason(self) -> None:
        state = {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "glm",
            "role_sent": False,
            "saw_tool": False,
        }
        events = ollama_ndjson_to_openai_events(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "a.py"},
                            }
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
            state,
        )
        parsed = self._parse(events)
        delta = parsed[0]["choices"][0]["delta"]
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(delta["tool_calls"][0]["function"]["arguments"]),
            {"path": "a.py"},
        )
        self.assertEqual(parsed[1]["choices"][0]["finish_reason"], "tool_calls")

    def test_nonstream_json(self) -> None:
        body = ollama_chat_to_openai_json(
            {
                "model": "glm",
                "message": {"role": "assistant", "content": "pong", "thinking": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 1,
            }
        )
        self.assertEqual(body["choices"][0]["message"]["content"], "pong")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["total_tokens"], 4)


class RequestShapingTests(unittest.TestCase):
    """openai_chat_to_ollama: sampling/options mapping the upstream depends on."""

    def test_max_tokens_maps_to_num_predict(self) -> None:
        n = openai_chat_to_ollama({"model": "g", "messages": [], "max_tokens": 256})
        self.assertEqual(n["options"]["num_predict"], 256)

    def test_max_completion_tokens_fallback(self) -> None:
        n = openai_chat_to_ollama(
            {"model": "g", "messages": [], "max_completion_tokens": 99}
        )
        self.assertEqual(n["options"]["num_predict"], 99)

    def test_explicit_options_num_predict_wins_over_max_tokens(self) -> None:
        n = openai_chat_to_ollama(
            {"model": "g", "messages": [], "max_tokens": 128, "options": {"num_predict": 999}}
        )
        self.assertEqual(n["options"]["num_predict"], 999)

    def test_sampling_fields_mapped(self) -> None:
        n = openai_chat_to_ollama(
            {
                "model": "g",
                "messages": [],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "stop": ["<x>"],
                "seed": 7,
            }
        )
        o = n["options"]
        self.assertEqual(
            (o["temperature"], o["top_p"], o["top_k"], o["stop"], o["seed"]),
            (0.6, 0.95, 20, ["<x>"], 7),
        )

    def test_explicit_option_wins_over_top_level(self) -> None:
        n = openai_chat_to_ollama(
            {"model": "g", "messages": [], "temperature": 0.6, "options": {"temperature": 0.1}}
        )
        self.assertEqual(n["options"]["temperature"], 0.1)

    def test_no_options_key_when_none_given(self) -> None:
        n = openai_chat_to_ollama({"model": "g", "messages": []})
        self.assertNotIn("options", n)

    def test_tool_choice_required_passed_auto_omitted(self) -> None:
        self.assertEqual(
            openai_chat_to_ollama({"model": "g", "messages": [], "tool_choice": "required"})[
                "tool_choice"
            ],
            "required",
        )
        self.assertNotIn(
            "tool_choice",
            openai_chat_to_ollama({"model": "g", "messages": [], "tool_choice": "auto"}),
        )
        self.assertNotIn(
            "tool_choice", openai_chat_to_ollama({"model": "g", "messages": []})
        )

    def test_keep_alive_passed_through(self) -> None:
        self.assertEqual(
            openai_chat_to_ollama({"model": "g", "messages": [], "keep_alive": -1})[
                "keep_alive"
            ],
            -1,
        )


class ToolSanitizeTests(unittest.TestCase):
    def test_tools_params_string_parsed(self) -> None:
        out = _sanitize_tools(
            [{"type": "function", "function": {"name": "a", "parameters": '{"type": "object"}'}}]
        )
        self.assertEqual(out[0]["function"]["parameters"], {"type": "object"})

    def test_tools_missing_name_skipped(self) -> None:
        self.assertEqual(_sanitize_tools([{"type": "function", "function": {"parameters": {}}}]), [])

    def test_tools_strict_dropped_description_kept(self) -> None:
        out = _sanitize_tools(
            [{"type": "function", "function": {"name": "a", "strict": True, "description": "d"}}]
        )
        self.assertNotIn("strict", out[0]["function"])
        self.assertEqual(out[0]["function"]["description"], "d")

    def test_tools_bad_params_string_defaults(self) -> None:
        out = _sanitize_tools([{"function": {"name": "a", "parameters": "not json"}}])
        self.assertEqual(out[0]["function"]["parameters"], {"type": "object", "properties": {}})

    def test_tools_non_list_returns_empty(self) -> None:
        self.assertEqual(_sanitize_tools(None), [])
        self.assertEqual(_sanitize_tools("nope"), [])

    def test_tool_calls_bad_json_args_become_raw(self) -> None:
        out = _sanitize_tool_calls([{"function": {"name": "x", "arguments": "not json"}}])
        self.assertEqual(out[0]["function"]["arguments"], {"_raw": "not json"})

    def test_tool_calls_id_preserved(self) -> None:
        out = _sanitize_tool_calls(
            [{"id": "call_9", "function": {"name": "x", "arguments": "{}"}}]
        )
        self.assertEqual(out[0]["id"], "call_9")

    def test_parse_tool_arguments_variants(self) -> None:
        self.assertEqual(_parse_tool_arguments(None), {})
        self.assertEqual(_parse_tool_arguments(""), {})
        self.assertEqual(_parse_tool_arguments('{"a": 1}'), {"a": 1})
        self.assertEqual(_parse_tool_arguments("[1, 2]"), {"_raw": [1, 2]})
        self.assertEqual(_parse_tool_arguments("bad"), {"_raw": "bad"})
        self.assertEqual(_parse_tool_arguments({"a": 1}), {"a": 1})

    def test_openai_tool_calls_delta_serializes_object_args(self) -> None:
        out = _openai_tool_calls_delta([{"function": {"name": "f", "arguments": {"a": 1}}}])
        self.assertEqual(out[0]["function"]["arguments"], '{"a": 1}')
        self.assertEqual(out[0]["id"], "call_0")
        self.assertEqual(out[0]["type"], "function")


class ContentFlattenTests(unittest.TestCase):
    def test_output_text_and_input_text(self) -> None:
        self.assertEqual(flatten_message_content([{"type": "output_text", "text": "o"}])[0], "o")
        self.assertEqual(flatten_message_content([{"type": "input_text", "text": "i"}])[0], "i")

    def test_dict_content(self) -> None:
        self.assertEqual(flatten_message_content({"type": "text", "text": "d"})[0], "d")

    def test_bare_text_no_type(self) -> None:
        self.assertEqual(flatten_message_content([{"text": "bare"}])[0], "bare")

    def test_nested_content_list(self) -> None:
        self.assertEqual(
            flatten_message_content([{"content": [{"type": "text", "text": "deep"}]}])[0], "deep"
        )

    def test_plain_string_and_none(self) -> None:
        self.assertEqual(flatten_message_content("plain"), ("plain", []))
        self.assertEqual(flatten_message_content(None), ("", []))

    def test_text_and_image_order_preserved(self) -> None:
        text, images = flatten_message_content(
            [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "text", "text": "b"},
            ]
        )
        self.assertEqual(text, "a\nb")
        self.assertEqual(images, ["QUJD"])

    def test_non_data_url_image_kept_verbatim(self) -> None:
        _, images = flatten_message_content(
            [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
        )
        self.assertEqual(images, ["http://x/y.png"])


class NormalizeMessageTests(unittest.TestCase):
    def test_images_merged_with_existing(self) -> None:
        msgs = normalize_messages_for_ollama(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}}
                    ],
                    "images": ["EXIST"],
                }
            ]
        )
        self.assertEqual(msgs[0]["images"], ["EXIST", "QQ=="])

    def test_tool_role_maps_name_and_drops_tool_call_id(self) -> None:
        msgs = normalize_messages_for_ollama(
            [{"role": "tool", "content": "r", "tool_call_id": "c1", "name": "read_file"}]
        )
        self.assertEqual(msgs[0], {"role": "tool", "content": "r", "tool_name": "read_file"})

    def test_non_dict_messages_skipped(self) -> None:
        self.assertEqual(normalize_messages_for_ollama(["nope", 5, None]), [])

    def test_missing_role_defaults_user(self) -> None:
        self.assertEqual(normalize_messages_for_ollama([{"content": "hi"}])[0]["role"], "user")


class NdjsonStreamTests(unittest.TestCase):
    def test_error_chunk_emits_error_then_done(self) -> None:
        parsed = _parse_sse(ollama_ndjson_to_openai_events({"error": "boom"}, _fresh_state()))
        self.assertEqual(parsed[0]["error"]["message"], "boom")
        self.assertEqual(parsed[0]["choices"], [])
        self.assertEqual(parsed[-1], "[DONE]")

    def test_error_dict_serialized(self) -> None:
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events({"error": {"code": 500}}, _fresh_state())
        )
        self.assertIn("500", parsed[0]["error"]["message"])

    def test_thinking_only_delta_emits_reasoning(self) -> None:
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events({"message": {"thinking": "pondering", "content": ""}}, _fresh_state())
        )
        self.assertEqual(parsed[0]["choices"][0]["delta"]["reasoning"], "pondering")

    def test_content_and_thinking_same_chunk(self) -> None:
        delta = _parse_sse(
            ollama_ndjson_to_openai_events({"message": {"thinking": "t", "content": "c"}}, _fresh_state())
        )[0]["choices"][0]["delta"]
        self.assertEqual(delta["reasoning"], "t")
        self.assertEqual(delta["content"], "c")
        self.assertEqual(delta["role"], "assistant")

    def test_length_done_reason_becomes_length_finish(self) -> None:
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events(
                {"message": {"content": ""}, "done": True, "done_reason": "length"}, _fresh_state()
            )
        )
        self.assertEqual(parsed[0]["choices"][0]["finish_reason"], "length")

    def test_state_model_filled_from_chunk(self) -> None:
        state = _fresh_state(model="")
        ollama_ndjson_to_openai_events({"model": "MOD", "message": {"content": "x"}}, state)
        self.assertEqual(state["model"], "MOD")

    def test_role_only_delta_suppressed_at_done(self) -> None:
        # role-only content delta immediately before done must not be emitted;
        # only finish + usage + [DONE] should come out.
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events(
                {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
                _fresh_state(),
            )
        )
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", parsed[1])
        self.assertEqual(parsed[2], "[DONE]")

    def test_usage_counts_reported(self) -> None:
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events(
                {
                    "message": {"content": "x"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 11,
                    "eval_count": 4,
                },
                _fresh_state(),
            )
        )
        usage = next(p for p in parsed if isinstance(p, dict) and "usage" in p)["usage"]
        self.assertEqual(usage, {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})

    def test_tool_across_chunk_then_done_keeps_tool_finish(self) -> None:
        # tool_calls in one chunk, done (no tools) in the next: saw_tool state
        # must persist so finish stays tool_calls, not stop.
        state = _fresh_state()
        ollama_ndjson_to_openai_events(
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]}}, state
        )
        parsed = _parse_sse(
            ollama_ndjson_to_openai_events(
                {"message": {"content": ""}, "done": True, "done_reason": "stop"}, state
            )
        )
        self.assertEqual(parsed[0]["choices"][0]["finish_reason"], "tool_calls")


class NonStreamJsonTests(unittest.TestCase):
    def test_tool_calls_message_has_null_content_and_tool_finish(self) -> None:
        body = ollama_chat_to_openai_json(
            {
                "model": "g",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                },
                "done": True,
                "done_reason": "stop",
            }
        )
        msg = body["choices"][0]["message"]
        self.assertIsNone(msg["content"])
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "f")
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_reasoning_surfaced_both_fields(self) -> None:
        body = ollama_chat_to_openai_json(
            {"model": "g", "message": {"role": "assistant", "content": "ans", "thinking": "th"}, "done": True}
        )
        msg = body["choices"][0]["message"]
        self.assertEqual(msg["reasoning"], "th")
        self.assertEqual(msg["reasoning_content"], "th")

    def test_length_finish_nonstream(self) -> None:
        body = ollama_chat_to_openai_json(
            {"model": "g", "message": {"content": "x"}, "done": True, "done_reason": "length"}
        )
        self.assertEqual(body["choices"][0]["finish_reason"], "length")


class AliasAndUrlTests(unittest.TestCase):
    def test_alias_noop_without_latest(self) -> None:
        raw = json.dumps({"data": [{"id": "plain"}]}).encode()
        self.assertEqual(json.loads(alias_model_ids(raw))["data"], [{"id": "plain"}])

    def test_alias_invalid_payload_unchanged(self) -> None:
        self.assertEqual(alias_model_ids(b"not json"), b"not json")

    def test_alias_non_list_data_unchanged(self) -> None:
        raw = json.dumps({"data": "oops"}).encode()
        self.assertEqual(alias_model_ids(raw), raw)

    def test_alias_no_duplicate_when_untagged_exists(self) -> None:
        raw = json.dumps(
            {"data": [{"id": "m:latest"}, {"id": "m"}]}
        ).encode()
        ids = [m["id"] for m in json.loads(alias_model_ids(raw))["data"]]
        self.assertEqual(ids.count("m"), 1)

    def test_native_chat_url_strips_v1(self) -> None:
        self.assertEqual(native_chat_url("http://h:1/v1/"), "http://h:1/api/chat")


class ThinkPrecedenceTests(unittest.TestCase):
    def test_think_key_overrides_chat_template_kwargs(self) -> None:
        self.assertTrue(
            resolve_think({"think": True, "chat_template_kwargs": {"enable_thinking": False}})
        )

    def test_enable_thinking_true(self) -> None:
        self.assertTrue(resolve_think({"chat_template_kwargs": {"enable_thinking": True}}))

    def test_thinking_type_enabled(self) -> None:
        self.assertTrue(resolve_think({"thinking": {"type": "enabled"}}))

    def test_custom_default_applied_when_unset(self) -> None:
        self.assertEqual(resolve_think({}, default="low"), "low")


class StreamingNativeIntegrationTests(unittest.TestCase):
    """End-to-end native /api/chat streaming (the path OpenCode uses).

    Drives the real FastAPI app with a fake Ollama upstream, so it exercises
    ndjson buffering across chunk boundaries, finish/usage ordering, the
    reasoning-only recovery, and [DONE] termination — the assembly where the
    hang/blank/repeat bugs lived.
    """

    def _stream(self, ndjson_chunks: list[bytes], body: dict) -> list:
        import httpx
        from fastapi.testclient import TestClient
        import openai_proxy as pp

        class FakeResp:
            status_code = 200
            headers = httpx.Headers({"content-type": "application/x-ndjson"})

            async def aiter_raw(self):
                for c in ndjson_chunks:
                    yield c

            async def aread(self):
                return b"".join(ndjson_chunks)

            async def aclose(self):
                return None

        async def fake_send(self, request, **kwargs):
            return FakeResp()

        orig = httpx.AsyncClient.send
        httpx.AsyncClient.send = fake_send
        try:
            app = pp.build_app(
                "http://127.0.0.1:11434/v1", heartbeat=0, native_chat=True
            )
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=body)
            self.assertEqual(resp.status_code, 200)
            return _parse_sse(
                [
                    (ln + "\n\n").encode()
                    for ln in resp.text.split("\n\n")
                    if ln.strip()
                ]
            )
        finally:
            httpx.AsyncClient.send = orig

    @staticmethod
    def _joined_content(parsed: list) -> str:
        return "".join(
            p["choices"][0]["delta"].get("content", "")
            for p in parsed
            if isinstance(p, dict) and p.get("choices")
        )

    @staticmethod
    def _finish(parsed: list):
        for p in parsed:
            if isinstance(p, dict) and p.get("choices"):
                fr = p["choices"][0].get("finish_reason")
                if fr:
                    return fr
        return None

    def test_content_stream_reassembled_across_chunk_split(self) -> None:
        # "lo" arrives split across two raw chunks — buffering must stitch it.
        chunks = [
            b'{"model":"glm","message":{"role":"assistant","content":"Hel"}}\n'
            b'{"message":{"content":"lo"',
            b'}}\n{"message":{"content":""},"done":true,"done_reason":"stop",'
            b'"prompt_eval_count":5,"eval_count":2}\n',
        ]
        parsed = self._stream(chunks, {"model": "glm", "stream": True, "messages": []})
        self.assertEqual(self._joined_content(parsed), "Hello")
        self.assertEqual(self._finish(parsed), "stop")
        usage = next(p for p in parsed if isinstance(p, dict) and "usage" in p)["usage"]
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(parsed[-1], "[DONE]")

    def test_reasoning_only_turn_recovered_to_content(self) -> None:
        chunks = [
            b'{"message":{"thinking":"I ponder"}}\n'
            b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n',
        ]
        parsed = self._stream(chunks, {"model": "glm", "stream": True, "messages": []})
        # No real content/tool -> the stripped reasoning is surfaced as content.
        self.assertEqual(self._joined_content(parsed), "I ponder")
        self.assertEqual(self._finish(parsed), "stop")
        self.assertEqual(parsed[-1], "[DONE]")

    def test_tool_call_stream_finishes_tool_calls(self) -> None:
        chunks = [
            b'{"message":{"tool_calls":[{"function":{"name":"bash",'
            b'"arguments":{"command":"ls"}}}]}}\n'
            b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n',
        ]
        parsed = self._stream(chunks, {"model": "glm", "stream": True, "messages": []})
        self.assertEqual(self._finish(parsed), "tool_calls")
        names = [
            tc["function"]["name"]
            for p in parsed
            if isinstance(p, dict) and p.get("choices")
            for tc in (p["choices"][0]["delta"].get("tool_calls") or [])
        ]
        self.assertIn("bash", names)
        self.assertEqual(parsed[-1], "[DONE]")

    def test_stream_always_terminates_with_done(self) -> None:
        # Upstream ends without an explicit done chunk: proxy must still emit a
        # finish + [DONE] so the client never hangs waiting.
        chunks = [b'{"message":{"content":"partial"}}\n']
        parsed = self._stream(chunks, {"model": "glm", "stream": True, "messages": []})
        self.assertEqual(parsed[-1], "[DONE]")
        self.assertIsNotNone(self._finish(parsed))


if __name__ == "__main__":
    unittest.main()
