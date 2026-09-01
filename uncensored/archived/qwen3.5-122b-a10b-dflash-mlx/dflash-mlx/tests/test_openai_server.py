from __future__ import annotations

import json


def test_server_module_importable_and_has_main():
    from dflash_mlx import openai_server

    assert callable(openai_server.main)


def test_chat_messages_join_text_segments_into_prompt():
    from dflash_mlx.openai_server import messages_to_prompt

    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this."},
                    {"type": "text", "text": "Keep it short."},
                ],
            },
        ]
    )

    assert "You are helpful." in prompt
    assert "Summarize this." in prompt
    assert "Keep it short." in prompt


def test_chat_messages_reject_non_text_content_parts():
    from dflash_mlx.openai_server import messages_to_prompt

    try:
        messages_to_prompt(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    ],
                }
            ]
        )
    except ValueError as exc:
        assert "text-only" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for non-text content")


def test_build_chat_response_has_openai_shape():
    from dflash_mlx.openai_server import build_chat_response

    payload = build_chat_response(
        model="local/qwen-dflash",
        content="Hello from DFlash.",
        prompt_tokens=12,
        completion_tokens=4,
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "local/qwen-dflash"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "Hello from DFlash."
    assert payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }


def test_build_chat_stream_chunk_has_openai_shape():
    from dflash_mlx.openai_server import build_chat_stream_chunk

    payload = build_chat_stream_chunk(
        chunk_id="chatcmpl-test",
        created=123,
        model="local/qwen-dflash",
        delta={"content": "Hel"},
    )

    assert payload == {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "local/qwen-dflash",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hel"},
                "finish_reason": None,
            }
        ],
    }


def test_models_response_lists_single_configured_model():
    from dflash_mlx.openai_server import build_models_response

    payload = build_models_response(model_id="local/qwen-dflash")

    assert payload == {
        "object": "list",
        "data": [
            {
                "id": "local/qwen-dflash",
                "object": "model",
                "owned_by": "dflash-mlx",
            }
        ],
    }


def test_health_response_shape():
    from dflash_mlx.openai_server import build_health_response

    assert build_health_response() == {"status": "ok"}


def test_clamp_max_tokens_defaults_and_ceiling():
    from dflash_mlx.openai_server import clamp_max_tokens

    assert clamp_max_tokens(None, default=4096, ceiling=8192) == 4096
    assert clamp_max_tokens(32, default=4096, ceiling=8192) == 32
    assert clamp_max_tokens(100_000, default=4096, ceiling=8192) == 8192
    assert clamp_max_tokens(0, default=4096, ceiling=8192) == 1


def test_adaptive_max_new_tokens_shrinks_on_large_prompts():
    from dflash_mlx.openai_server import adaptive_max_new_tokens

    assert adaptive_max_new_tokens(500, 2048, tools=True) == 2048
    assert adaptive_max_new_tokens(11000, 2048, tools=True) == 256
    assert adaptive_max_new_tokens(9000, 2048, tools=False) == 768


def test_trim_messages_to_budget_keeps_recent():
    from dflash_mlx.openai_server import trim_messages_to_budget

    messages = [{"role": "system", "content": "sys"}]
    for i in range(20):
        messages.append({"role": "user", "content": f"msg-{i} " + ("x" * 100)})
        messages.append({"role": "assistant", "content": f"ans-{i}"})

    # Fake counter: 100 tokens per message + 50 for tools overhead
    def count(msgs, tools):
        return 50 + 100 * len(msgs)

    fitted, n, stats = trim_messages_to_budget(
        messages, tools=None, count_tokens=count, max_tokens=500
    )
    assert n <= 500
    assert stats["trimmed_messages"] > 0
    assert fitted[0]["role"] == "system"
    # Newest user content should survive
    assert any("msg-19" in (m.get("content") or "") for m in fitted)


def test_normalize_allows_null_content_and_tool_calls():
    from dflash_mlx.openai_server import normalize_openai_messages

    messages = normalize_openai_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "grep",
                            "arguments": '{"pattern": "Rescue"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "found 2 matches",
            },
        ]
    )

    assert messages[1]["content"] == ""
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "pattern": "Rescue"
    }
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == "found 2 matches"


def test_messages_to_prompt_handles_tool_turns():
    from dflash_mlx.openai_server import messages_to_prompt

    prompt = messages_to_prompt(
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "ls", "content": "a.txt"},
        ]
    )

    assert "search" in prompt
    assert "<tool_call>" in prompt
    assert "function=ls" in prompt
    assert "[tool_calls]" not in prompt  # never teach the dump format
    assert "a.txt" in prompt
    assert prompt.endswith("Assistant:")


def test_parse_qwen_tool_calls_to_openai():
    from dflash_mlx.openai_server import finalize_assistant_message, parse_qwen_tool_calls

    raw = (
        "<tool_call>\n"
        "<function=execute_command>\n"
        "<parameter=command>\n"
        "echo hello_rescue\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, tool_calls = parse_qwen_tool_calls(raw)
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "execute_command"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["command"] == "echo hello_rescue"

    msg_content, msg_tools, reason = finalize_assistant_message(raw, tools_requested=True)
    assert msg_content is None
    assert msg_tools and msg_tools[0]["function"]["name"] == "execute_command"
    assert reason == "tool_calls"


def test_finalize_converts_tool_xml_even_without_tools_flag():
    from dflash_mlx.openai_server import finalize_assistant_message

    raw = "<tool_call><function=x><parameter=a>1</parameter></function></tool_call>"
    content, tools, reason = finalize_assistant_message(raw, tools_requested=False)
    assert tools and tools[0]["function"]["name"] == "x"
    assert reason == "tool_calls"
    assert content is None


def test_parse_bracket_tool_calls_dump():
    from dflash_mlx.openai_server import finalize_assistant_message

    raw = (
        '[tool_calls] [{"id": "call_abc", "type": "function", "function": '
        '{"name": "edit", "arguments": {"filePath": "/tmp/x", "oldText": "a", "newText": "b"}}}]'
    )
    content, tools, reason = finalize_assistant_message(raw, tools_requested=True)
    assert reason == "tool_calls"
    assert content is None
    assert tools and tools[0]["function"]["name"] == "edit"
    args = json.loads(tools[0]["function"]["arguments"])
    assert args["filePath"] == "/tmp/x"
    assert args["newText"] == "b"
