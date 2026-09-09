#!/usr/bin/env python3
"""Unit tests for openai_proxy conversion helpers (no live Ollama)."""
from __future__ import annotations

import json
import unittest

from openai_proxy import (
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


if __name__ == "__main__":
    unittest.main()
