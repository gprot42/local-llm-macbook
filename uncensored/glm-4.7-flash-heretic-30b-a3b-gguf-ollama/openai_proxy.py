#!/usr/bin/env python3
"""OpenAI-compatible reverse proxy for Ollama.

Clients (OpenCode / Kilo) → this proxy → Ollama.

Chat completions go to native ``/api/chat`` (not ``/v1/chat/completions``).
Ollama 0.31.1's OpenAI endpoint ignores ``think`` / ``enable_thinking``, so
GLM always opens ``<think>`` with an unlimited reasoning budget and can burn
the whole ``max_tokens`` on hidden reasoning — which looks like a hung
harness. Native ``/api/chat`` honors ``think=false``.

Other ``/v1/*`` routes still pass through. ``/healthz`` is local.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("glm_openai_proxy")

UPSTREAM = "http://127.0.0.1:11434/v1"
TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)

# Emit an SSE comment line if the upstream stream is quiet for this many seconds.
# Ollama buffers tool_calls and flushes the whole call as one chunk at the end,
# so a large tool call at high context can go minutes with no bytes downstream.
# That trips a client's per-chunk watchdog (Kilo's chunkTimeout). A comment
# (":\n\n") is ignored by SSE parsers but any byte resets that watchdog.
HEARTBEAT_INTERVAL = 15.0

# Promote reasoning to content when a completion ends with reasoning but no
# assistant content and no tool call. GLM-4.7-Flash (a thinking model) routinely
# ends a turn inside the reasoning channel — either it stops right after
# thinking, or its whole output budget went to reasoning — leaving content="".
# Kilo treats such a turn as an "incomplete response" (its replayable() check:
# reasoning-only with no text and no tool → retry), retries twice, then fails
# with "The provider repeatedly ended the response before returning usable
# output." Surfacing the reasoning as content gives the client usable output.
REASONING_FALLBACK = True

# Default for requests that do not set think / enable_thinking. Off keeps the
# Kilo harness from stalling in an unbounded <think> block.
THINK_DEFAULT: bool | str = False


def alias_model_ids(payload: bytes) -> bytes:
    """Expose untagged Ollama ids so Kilo's `glm-4.7-flash-heretic-q8` matches."""
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return payload
    items = data.get("data")
    if not isinstance(items, list):
        return payload
    seen = {m.get("id") for m in items if isinstance(m, dict)}
    extra: list[dict[str, Any]] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or ""
        if isinstance(mid, str) and mid.endswith(":latest"):
            alias = mid[: -len(":latest")]
            if alias and alias not in seen:
                extra.append({**m, "id": alias})
                seen.add(alias)
    if not extra:
        return payload
    data["data"] = items + extra
    return json.dumps(data).encode()


def native_chat_url(upstream_v1: str) -> str:
    base = upstream_v1.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")] + "/api/chat"
    return base + "/api/chat"


def thinking_requested(obj: dict[str, Any]) -> bool | str | None:
    """Return the client's think setting, or None if they did not set one."""
    if "think" in obj:
        return obj["think"]  # bool or "low"/"medium"/"high"
    ctk = obj.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        return bool(ctk["enable_thinking"])
    thinking = obj.get("thinking")
    if isinstance(thinking, dict) and "type" in thinking:
        t = thinking.get("type")
        if t == "disabled":
            return False
        if t == "enabled":
            return True
    return None


def resolve_think(obj: dict[str, Any], default: bool | str = THINK_DEFAULT) -> bool | str:
    requested = thinking_requested(obj)
    return default if requested is None else requested


def _parse_tool_arguments(args: Any) -> Any:
    if args is None or args == "":
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (ValueError, TypeError):
            return {"_raw": args}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    if isinstance(args, dict):
        return args
    return {"_raw": args}


def _image_to_ollama(url: str) -> str:
    if "base64," in url:
        return url.split("base64,", 1)[1]
    return url


def flatten_message_content(content: Any) -> tuple[str, list[str]]:
    """Ollama /api/chat requires messages.content to be a string.

    OpenAI (and Kilo) send content-part arrays:
    ``[{"type": "text", "text": "..."}]``. Passing those through yields
    ``cannot unmarshal array into Go struct field ChatRequest.messages.content``.
    """
    images: list[str] = []
    if content is None:
        return "", images
    if isinstance(content, str):
        return content, images
    if isinstance(content, dict):
        return flatten_message_content([content])
    if not isinstance(content, list):
        return str(content), images
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"image_url", "image"} or "image_url" in part:
            url = part.get("image_url") or part.get("image")
            if isinstance(url, dict):
                url = url.get("url") or url.get("data")
            if isinstance(url, str) and url:
                images.append(_image_to_ollama(url))
            continue
        # OpenAI "text", Responses API "input_text", or a bare {"text": "..."}.
        if ptype in {"text", "input_text", "output_text", None} or "text" in part:
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
                continue
            if isinstance(text, list):
                nested, _ = flatten_message_content(text)
                if nested:
                    texts.append(nested)
                continue
        inner = part.get("content")
        if isinstance(inner, str):
            texts.append(inner)
        elif isinstance(inner, list):
            nested, extra_images = flatten_message_content(inner)
            images.extend(extra_images)
            if nested:
                texts.append(nested)
    return "\n".join(texts), images


def _sanitize_tool_calls(tcs: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or tc.get("name") or ""
        args = _parse_tool_arguments(fn.get("arguments", tc.get("arguments")))
        item: dict[str, Any] = {
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, dict) else {}},
        }
        if tc.get("id"):
            item["id"] = tc["id"]
        converted.append(item)
    return converted


def _sanitize_tools(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") or tool.get("name")
        if not name:
            continue
        params = fn.get("parameters", {"type": "object", "properties": {}})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (ValueError, TypeError):
                params = {"type": "object", "properties": {}}
        cleaned_fn: dict[str, Any] = {"name": name, "parameters": params}
        if fn.get("description"):
            cleaned_fn["description"] = fn["description"]
        out.append({"type": "function", "function": cleaned_fn})
    return out


def normalize_messages_for_ollama(messages: list[Any]) -> list[dict[str, Any]]:
    """Keep only fields Ollama /api/chat can unmarshal.

    OpenAI/Kilo send content-part arrays, extra keys (tool_call_id, reasoning,
    cache_control), and tool `arguments` as JSON strings. Native ChatRequest
    wants ``content``/``thinking`` as strings and ``arguments`` as objects —
    anything else is a 400 and Kilo disables sending.
    """
    out: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role") or "user"
        if role == "developer":
            role = "system"
        text, images = flatten_message_content(raw.get("content"))
        thinking_src = raw.get("thinking") or raw.get("reasoning") or raw.get("reasoning_content")
        thinking, _ = flatten_message_content(thinking_src) if thinking_src else ("", [])
        msg: dict[str, Any] = {"role": role, "content": text}
        if images:
            existing = raw.get("images")
            extra = existing if isinstance(existing, list) else ([existing] if existing else [])
            msg["images"] = extra + images
        if thinking:
            msg["thinking"] = thinking
        tcs = raw.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            msg["tool_calls"] = _sanitize_tool_calls(tcs)
        if role == "tool":
            name = raw.get("name") or raw.get("tool_name")
            if name:
                msg["tool_name"] = name
        out.append(msg)
    return out


def openai_chat_to_ollama(
    obj: dict[str, Any], think_default: bool | str = THINK_DEFAULT
) -> dict[str, Any]:
    native: dict[str, Any] = {
        "model": obj.get("model"),
        "messages": normalize_messages_for_ollama(list(obj.get("messages") or [])),
        "stream": bool(obj.get("stream")),
        "think": resolve_think(obj, think_default),
    }
    if obj.get("tools"):
        native["tools"] = _sanitize_tools(obj["tools"])
    tool_choice = obj.get("tool_choice")
    if tool_choice not in (None, "auto"):
        native["tool_choice"] = tool_choice
    options: dict[str, Any] = {}
    if isinstance(obj.get("options"), dict):
        options.update(obj["options"])
    max_tokens = obj.get("max_tokens", obj.get("max_completion_tokens"))
    if max_tokens is not None and "num_predict" not in options:
        options["num_predict"] = max_tokens
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop", "stop"),
        ("seed", "seed"),
    ):
        if obj.get(src) is not None and dst not in options:
            options[dst] = obj[src]
    if options:
        native["options"] = options
    if obj.get("keep_alive") is not None:
        native["keep_alive"] = obj["keep_alive"]
    return native


def _openai_tool_calls_delta(tool_calls: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or tc.get("name") or ""
        args = fn.get("arguments", {})
        if isinstance(args, (dict, list)):
            args_s = json.dumps(args)
        else:
            args_s = args if isinstance(args, str) else "{}"
        out.append(
            {
                "index": i,
                "id": tc.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args_s},
            }
        )
    return out


def _chunk_payload(state: dict[str, Any], delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": state["id"],
        "object": "chat.completion.chunk",
        "created": state["created"],
        "model": state["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def ollama_ndjson_to_openai_events(chunk: dict[str, Any], state: dict[str, Any]) -> list[bytes]:
    """Turn one native /api/chat NDJSON object into OpenAI SSE events (with \\n\\n)."""
    if chunk.get("error"):
        err = chunk["error"]
        msg = err if isinstance(err, str) else json.dumps(err)
        payload = {
            "id": state["id"],
            "object": "chat.completion.chunk",
            "created": state["created"],
            "model": state["model"],
            "choices": [],
            "error": {"message": msg, "type": "server_error"},
        }
        return [b"data: " + json.dumps(payload).encode() + b"\n\n", b"data: [DONE]\n\n"]

    if chunk.get("model") and not state.get("model"):
        state["model"] = chunk["model"]
    msg = chunk.get("message") or {}
    events: list[bytes] = []
    thinking = msg.get("thinking") or ""
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    delta: dict[str, Any] = {}
    if not state.get("role_sent"):
        delta["role"] = "assistant"
        state["role_sent"] = True
    if thinking:
        delta["reasoning"] = thinking
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = _openai_tool_calls_delta(tool_calls)
        state["saw_tool"] = True
    if len(delta) > 1 or (len(delta) == 1 and "role" not in delta) or (
        len(delta) == 1 and not chunk.get("done")
    ):
        # Skip a role-only delta that is immediately followed by done with no
        # payload — the finish chunk is enough. Still emit role+content/think.
        if not (list(delta.keys()) == ["role"] and chunk.get("done") and not thinking and not content and not tool_calls):
            events.append(b"data: " + json.dumps(_chunk_payload(state, delta)).encode() + b"\n\n")

    if chunk.get("done"):
        reason = chunk.get("done_reason") or "stop"
        if state.get("saw_tool") or tool_calls:
            finish = "tool_calls"
        elif reason == "length":
            finish = "length"
        else:
            finish = "stop"
        events.append(
            b"data: " + json.dumps(_chunk_payload(state, {}, finish)).encode() + b"\n\n"
        )
        prompt_tok = int(chunk.get("prompt_eval_count") or 0)
        completion_tok = int(chunk.get("eval_count") or 0)
        usage = {
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": prompt_tok + completion_tok,
        }
        usage_payload = {
            "id": state["id"],
            "object": "chat.completion.chunk",
            "created": state["created"],
            "model": state["model"],
            "choices": [],
            "usage": usage,
        }
        events.append(b"data: " + json.dumps(usage_payload).encode() + b"\n\n")
        events.append(b"data: [DONE]\n\n")
    return events


def ollama_chat_to_openai_json(chunk: dict[str, Any]) -> dict[str, Any]:
    msg = chunk.get("message") or {}
    thinking = msg.get("thinking") or ""
    content = msg.get("content")
    tool_calls = msg.get("tool_calls") or []
    reason = chunk.get("done_reason") or "stop"
    if tool_calls:
        finish = "tool_calls"
    elif reason == "length":
        finish = "length"
    else:
        finish = "stop"
    message: dict[str, Any] = {
        "role": msg.get("role") or "assistant",
        "content": content if content else None,
    }
    if thinking:
        message["reasoning"] = thinking
        message["reasoning_content"] = thinking
    if tool_calls:
        message["tool_calls"] = _openai_tool_calls_delta(tool_calls)
        message["content"] = content or None
    prompt_tok = int(chunk.get("prompt_eval_count") or 0)
    completion_tok = int(chunk.get("eval_count") or 0)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": chunk.get("model") or "",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": prompt_tok + completion_tok,
        },
    }


def _promote_reasoning_json(payload: bytes) -> bytes:
    """Non-streaming: move reasoning into content when content/tool are empty."""
    try:
        data = json.loads(payload)
        choices = data.get("choices") or []
    except (ValueError, AttributeError):
        return payload
    changed = False
    for ch in choices:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(msg, dict):
            continue
        if msg.get("content") or msg.get("tool_calls"):
            continue
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if reasoning:
            msg["content"] = reasoning
            changed = True
    if not changed:
        return payload
    return json.dumps(data).encode()


def build_app(
    upstream: str,
    heartbeat: float = HEARTBEAT_INTERVAL,
    reasoning_fallback: bool = REASONING_FALLBACK,
    think_default: bool | str = THINK_DEFAULT,
    native_chat: bool = False,
) -> FastAPI:
    app = FastAPI(title="GLM OpenAI proxy", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    base = upstream.rstrip("/")  # e.g. http://127.0.0.1:11434/v1
    chat_url = native_chat_url(base)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "upstream": base, "chat": chat_url, "think_default": think_default}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await healthz()

    def _passthrough_headers(request: Request) -> dict[str, str]:
        return {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in {"host", "content-length", "transfer-encoding", "connection"}
        }

    async def _forward_native_chat(request: Request, obj: dict[str, Any]) -> Response:
        native = openai_chat_to_ollama(obj, think_default=think_default)
        kinds = []
        for m in obj.get("messages") or []:
            if isinstance(m, dict):
                c = m.get("content")
                kinds.append(type(c).__name__ if not isinstance(c, list) else f"list[{len(c)}]")
        log.info(
            "chat → %s think=%r stream=%s model=%s msgs=%d content=%s tools=%d",
            chat_url,
            native.get("think"),
            native.get("stream"),
            native.get("model"),
            len(native.get("messages") or []),
            ",".join(kinds) or "-",
            len(native.get("tools") or []),
        )
        client = httpx.AsyncClient(timeout=TIMEOUT)
        try:
            req = client.build_request(
                "POST",
                chat_url,
                headers=_passthrough_headers(request),
                json=native,
            )
            upstream_resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            log.exception("upstream error")
            return JSONResponse({"error": str(exc)}, status_code=502)

        if upstream_resp.status_code >= 400:
            raw = await upstream_resp.aread()
            status = upstream_resp.status_code
            await upstream_resp.aclose()
            message: Any
            try:
                err_obj = json.loads(raw)
                message = err_obj.get("error", raw.decode("utf-8", "replace"))
            except (ValueError, TypeError, AttributeError):
                message = raw.decode("utf-8", "replace")
            if isinstance(message, dict):
                message = message.get("message") or json.dumps(message)
            log.warning("native chat %s: %s — falling back to /v1", status, message)
            v1_target = f"{base}/chat/completions"
            try:
                req2 = client.build_request(
                    "POST",
                    v1_target,
                    headers=_passthrough_headers(request),
                    json=obj,
                )
                v1_resp = await client.send(req2, stream=True)
            except httpx.HTTPError as exc:
                await client.aclose()
                log.exception("v1 fallback error")
                return JSONResponse(
                    {"error": {"message": str(message), "type": "invalid_request_error"}},
                    status_code=status,
                )
            if v1_resp.status_code >= 400:
                v1_raw = await v1_resp.aread()
                await v1_resp.aclose()
                await client.aclose()
                return JSONResponse(
                    {"error": {"message": str(message), "type": "invalid_request_error"}},
                    status_code=status,
                )

            async def _v1_passthrough():
                try:
                    async for chunk in v1_resp.aiter_raw():
                        yield chunk
                finally:
                    await v1_resp.aclose()
                    await client.aclose()

            media = v1_resp.headers.get("content-type", "text/event-stream")
            return StreamingResponse(
                _v1_passthrough(),
                status_code=200,
                media_type=media.split(";")[0].strip() if media else "text/event-stream",
            )

        if native.get("stream"):
            return _streaming_from_ndjson(upstream_resp, client, native)
        content = await upstream_resp.aread()
        status = upstream_resp.status_code
        await upstream_resp.aclose()
        await client.aclose()
        if status >= 400:
            return Response(content=content, status_code=status, media_type="application/json")
        try:
            chunk = json.loads(content)
        except ValueError:
            return Response(content=content, status_code=status, media_type="application/json")
        if chunk.get("error"):
            return JSONResponse({"error": chunk["error"]}, status_code=502)
        openai_body = ollama_chat_to_openai_json(chunk)
        encoded = json.dumps(openai_body).encode()
        if reasoning_fallback:
            encoded = _promote_reasoning_json(encoded)
        return Response(content=encoded, status_code=200, media_type="application/json")

    def _streaming_from_ndjson(
        upstream_resp: httpx.Response,
        client: httpx.AsyncClient,
        native: dict[str, Any],
    ) -> StreamingResponse:
        async def stream():
            buf = b""
            saw_content = False
            saw_tool = False
            reasoning: list[str] = []
            held_finish: bytes | None = None
            held_usage: bytes | None = None
            done_sent = False
            emitted_finish = False
            meta: dict[str, Any] = {
                "id": f"chatcmpl-{int(time.time())}",
                "created": int(time.time()),
                "model": native.get("model") or "",
            }
            conv_state: dict[str, Any] = {
                "id": meta["id"],
                "created": meta["created"],
                "model": meta["model"],
                "role_sent": False,
                "saw_tool": False,
            }

            def synth_finish(reason: str) -> bytes:
                payload = {
                    "id": meta.get("id", "chatcmpl-proxy"),
                    "object": "chat.completion.chunk",
                    "created": meta.get("created", int(time.time())),
                    "model": meta.get("model", ""),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                }
                return b"data: " + json.dumps(payload).encode() + b"\n\n"

            def synth_content() -> bytes:
                payload = {
                    "id": meta.get("id", "chatcmpl-proxy"),
                    "object": "chat.completion.chunk",
                    "created": meta.get("created", int(time.time())),
                    "model": meta.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "".join(reasoning),
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                log.info(
                    "reasoning-only completion recovered: promoted %d "
                    "reasoning chars to content",
                    len("".join(reasoning)),
                )
                return b"data: " + json.dumps(payload).encode() + b"\n\n"

            def process_event(ev: bytes) -> list[bytes]:
                nonlocal saw_content, saw_tool, held_finish, done_sent
                nonlocal emitted_finish, held_usage
                text = ev.strip()
                if not text or text.startswith(b":"):
                    return [ev + b"\n\n"]
                if not reasoning_fallback:
                    if text == b"data: [DONE]" or text == b"data:[DONE]":
                        done_sent = True
                    return [ev + b"\n\n"]
                if text == b"data: [DONE]" or text == b"data:[DONE]":
                    outs: list[bytes] = []
                    if not saw_content and not saw_tool and reasoning:
                        outs.append(synth_content())
                    if held_finish is not None:
                        outs.append(held_finish)
                        held_finish = None
                        emitted_finish = True
                    if not emitted_finish:
                        outs.append(synth_finish("tool_calls" if saw_tool else "stop"))
                        emitted_finish = True
                    if held_usage is not None:
                        outs.append(held_usage)
                        held_usage = None
                    outs.append(ev + b"\n\n")
                    done_sent = True
                    return outs
                if not text.startswith(b"data:"):
                    return [ev + b"\n\n"]
                try:
                    obj = json.loads(text[len(b"data:") :].strip())
                except ValueError:
                    return [ev + b"\n\n"]
                for key in ("id", "model", "created"):
                    if key in obj and key not in meta:
                        meta[key] = obj[key]
                if obj.get("usage") and not (obj.get("choices") or []):
                    held_usage = ev + b"\n\n"
                    return []
                choices = obj.get("choices") or []
                has_finish = False
                for c in choices:
                    if not isinstance(c, dict):
                        continue
                    delta = c.get("delta") or {}
                    if delta.get("content"):
                        saw_content = True
                    if delta.get("tool_calls"):
                        saw_tool = True
                    r = delta.get("reasoning") or delta.get("reasoning_content")
                    if r:
                        reasoning.append(r)
                    if c.get("finish_reason") is not None:
                        has_finish = True
                if has_finish:
                    # Only hold finish when a reasoning-only fallback might
                    # still need to inject content before it. Otherwise the
                    # client sees finish immediately and does not look hung
                    # waiting for [DONE] / usage.
                    if reasoning_fallback and not saw_content and not saw_tool and reasoning:
                        held_finish = ev + b"\n\n"
                        return []
                    emitted_finish = True
                    return [ev + b"\n\n"]
                return [ev + b"\n\n"]

            raw = upstream_resp.aiter_raw()
            read_task: asyncio.Task | None = None
            stream_error: BaseException | None = None
            events_out = 0
            try:
                while True:
                    if heartbeat and heartbeat > 0:
                        if read_task is None:
                            read_task = asyncio.ensure_future(raw.__anext__())
                        done, _pending = await asyncio.wait({read_task}, timeout=heartbeat)
                        if not done:
                            yield b": keepalive\n\n"
                            continue
                        try:
                            chunk = read_task.result()
                        except StopAsyncIteration:
                            chunk = None
                        except Exception as exc:
                            stream_error = exc
                            chunk = None
                        finally:
                            read_task = None
                    else:
                        try:
                            chunk = await raw.__anext__()
                        except StopAsyncIteration:
                            chunk = None
                        except Exception as exc:
                            stream_error = exc
                            chunk = None
                    if chunk is None:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            native_obj = json.loads(line)
                        except ValueError:
                            log.warning("skipping malformed ndjson: %r", line[:200])
                            continue
                        if not isinstance(native_obj, dict):
                            continue
                        for sse_ev in ollama_ndjson_to_openai_events(native_obj, conv_state):
                            inner = sse_ev[:-2] if sse_ev.endswith(b"\n\n") else sse_ev
                            for out in process_event(inner):
                                yield out
                                events_out += 1
                if buf.strip():
                    try:
                        native_obj = json.loads(buf)
                    except ValueError:
                        native_obj = None
                    if isinstance(native_obj, dict):
                        for sse_ev in ollama_ndjson_to_openai_events(native_obj, conv_state):
                            inner = sse_ev[:-2] if sse_ev.endswith(b"\n\n") else sse_ev
                            for out in process_event(inner):
                                yield out
                if not done_sent:
                    if reasoning_fallback and not saw_content and not saw_tool and reasoning:
                        yield synth_content()
                    if held_finish is not None:
                        yield held_finish
                        held_finish = None
                        emitted_finish = True
                    if not emitted_finish:
                        yield synth_finish("tool_calls" if saw_tool else "stop")
                        emitted_finish = True
                    if held_usage is not None:
                        yield held_usage
                        held_usage = None
                    yield b"data: [DONE]\n\n"
                if stream_error is not None:
                    log.warning("upstream stream ended early: %r", stream_error)
            except (asyncio.CancelledError, GeneratorExit):
                log.warning(
                    "client disconnected / stream cancelled mid-stream after "
                    "%d events (saw_content=%s saw_tool=%s done_sent=%s) — "
                    "closing upstream Ollama connection to abort generation",
                    events_out,
                    saw_content,
                    saw_tool,
                    done_sent,
                )
                raise
            finally:
                if read_task is not None:
                    read_task.cancel()
                await upstream_resp.aclose()
                await client.aclose()

        return StreamingResponse(
            stream(),
            status_code=upstream_resp.status_code,
            media_type="text/event-stream",
        )

    async def _forward(request: Request, target: str) -> Response:
        headers = _passthrough_headers(request)
        body = await request.body()

        if body and request.method == "POST" and "chat/completions" in target:
            try:
                obj = json.loads(body)
            except (ValueError, TypeError):
                obj = None
            else:
                if isinstance(obj, dict) and native_chat:
                    return await _forward_native_chat(request, obj)
                if isinstance(obj, dict) and obj.get("stream") is True:
                    so = obj.get("stream_options")
                    if not isinstance(so, dict):
                        so = {}
                    if not so.get("include_usage"):
                        so["include_usage"] = True
                        obj["stream_options"] = so
                        body = json.dumps(obj).encode()

        client = httpx.AsyncClient(timeout=TIMEOUT)
        try:
            req = client.build_request(
                request.method,
                target,
                headers=headers,
                content=body if body else None,
            )
            upstream_resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            log.exception("upstream error")
            return JSONResponse({"error": str(exc)}, status_code=502)

        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        out_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in excluded
        }
        media = upstream_resp.headers.get("content-type", "")

        if "text/event-stream" in media:

            async def stream():
                buf = b""
                saw_content = False
                saw_tool = False
                reasoning: list[str] = []
                held_finish: bytes | None = None
                held_usage: bytes | None = None
                done_sent = False
                emitted_finish = False
                meta: dict[str, Any] = {}

                def synth_finish(reason: str) -> bytes:
                    payload = {
                        "id": meta.get("id", "chatcmpl-proxy"),
                        "object": "chat.completion.chunk",
                        "created": meta.get("created", int(time.time())),
                        "model": meta.get("model", ""),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                    }
                    return b"data: " + json.dumps(payload).encode() + b"\n\n"

                def synth_content() -> bytes:
                    payload = {
                        "id": meta.get("id", "chatcmpl-proxy"),
                        "object": "chat.completion.chunk",
                        "created": meta.get("created", int(time.time())),
                        "model": meta.get("model", ""),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "".join(reasoning),
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    return b"data: " + json.dumps(payload).encode() + b"\n\n"

                def process_event(ev: bytes) -> list[bytes]:
                    nonlocal saw_content, saw_tool, held_finish, done_sent
                    nonlocal emitted_finish, held_usage
                    text = ev.strip()
                    if not text or text.startswith(b":"):
                        return [ev + b"\n\n"]
                    if not reasoning_fallback:
                        if text == b"data: [DONE]" or text == b"data:[DONE]":
                            done_sent = True
                        return [ev + b"\n\n"]
                    if text == b"data: [DONE]" or text == b"data:[DONE]":
                        outs: list[bytes] = []
                        if not saw_content and not saw_tool and reasoning:
                            outs.append(synth_content())
                        if held_finish is not None:
                            outs.append(held_finish)
                            held_finish = None
                            emitted_finish = True
                        if not emitted_finish:
                            outs.append(synth_finish("tool_calls" if saw_tool else "stop"))
                            emitted_finish = True
                        if held_usage is not None:
                            outs.append(held_usage)
                            held_usage = None
                        outs.append(ev + b"\n\n")
                        done_sent = True
                        return outs
                    if not text.startswith(b"data:"):
                        return [ev + b"\n\n"]
                    try:
                        obj = json.loads(text[len(b"data:") :].strip())
                    except ValueError:
                        return [ev + b"\n\n"]
                    for key in ("id", "model", "created"):
                        if key in obj and key not in meta:
                            meta[key] = obj[key]
                    if obj.get("usage") and not (obj.get("choices") or []):
                        held_usage = ev + b"\n\n"
                        return []
                    choices = obj.get("choices") or []
                    has_finish = False
                    for c in choices:
                        if not isinstance(c, dict):
                            continue
                        delta = c.get("delta") or {}
                        if delta.get("content"):
                            saw_content = True
                        if delta.get("tool_calls"):
                            saw_tool = True
                        r = delta.get("reasoning") or delta.get("reasoning_content")
                        if r:
                            reasoning.append(r)
                        if c.get("finish_reason") is not None:
                            has_finish = True
                    if has_finish:
                        if reasoning_fallback and not saw_content and not saw_tool and reasoning:
                            held_finish = ev + b"\n\n"
                            return []
                        emitted_finish = True
                        return [ev + b"\n\n"]
                    return [ev + b"\n\n"]

                raw = upstream_resp.aiter_raw()
                read_task: asyncio.Task | None = None
                stream_error: BaseException | None = None
                events_out = 0
                try:
                    while True:
                        if heartbeat and heartbeat > 0:
                            if read_task is None:
                                read_task = asyncio.ensure_future(raw.__anext__())
                            done, _pending = await asyncio.wait(
                                {read_task}, timeout=heartbeat
                            )
                            if not done:
                                yield b": keepalive\n\n"
                                continue
                            try:
                                chunk = read_task.result()
                            except StopAsyncIteration:
                                chunk = None
                            except Exception as exc:
                                stream_error = exc
                                chunk = None
                            finally:
                                read_task = None
                        else:
                            try:
                                chunk = await raw.__anext__()
                            except StopAsyncIteration:
                                chunk = None
                            except Exception as exc:
                                stream_error = exc
                                chunk = None
                        if chunk is None:
                            break
                        buf += chunk
                        while b"\n\n" in buf:
                            ev, buf = buf.split(b"\n\n", 1)
                            for out in process_event(ev):
                                yield out
                                events_out += 1
                    if buf.strip():
                        for out in process_event(buf):
                            yield out
                    if not done_sent:
                        if reasoning_fallback and not saw_content and not saw_tool and reasoning:
                            yield synth_content()
                        if held_finish is not None:
                            yield held_finish
                            held_finish = None
                            emitted_finish = True
                        if not emitted_finish:
                            yield synth_finish("tool_calls" if saw_tool else "stop")
                            emitted_finish = True
                        if held_usage is not None:
                            yield held_usage
                            held_usage = None
                        yield b"data: [DONE]\n\n"
                    if stream_error is not None:
                        log.warning("upstream stream ended early: %r", stream_error)
                except (asyncio.CancelledError, GeneratorExit):
                    log.warning(
                        "client disconnected / stream cancelled mid-stream after "
                        "%d events (saw_content=%s saw_tool=%s done_sent=%s) — "
                        "closing upstream Ollama connection to abort generation",
                        events_out,
                        saw_content,
                        saw_tool,
                        done_sent,
                    )
                    raise
                finally:
                    if read_task is not None:
                        read_task.cancel()
                    await upstream_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                stream(),
                status_code=upstream_resp.status_code,
                headers=out_headers,
                media_type=media or None,
            )

        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        if request.method == "GET" and target.split("?", 1)[0].rstrip("/").endswith("/models"):
            content = alias_model_ids(content)
        if reasoning_fallback and "application/json" in (media or ""):
            content = _promote_reasoning_json(content)
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=out_headers,
            media_type=media or None,
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    )
    async def proxy_v1(path: str, request: Request) -> Response:
        if path in {"health", "healthz"}:
            return JSONResponse(await healthz())
        target = f"{base}/{path}" if path else base
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return await _forward(request, target)

    @app.api_route(
        "/v1",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    )
    async def proxy_v1_root(request: Request) -> Response:
        target = base
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return await _forward(request, target)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible Ollama reverse proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument(
        "--upstream",
        default=UPSTREAM,
        help="Ollama OpenAI base, e.g. http://127.0.0.1:11434/v1",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=HEARTBEAT_INTERVAL,
        help="Seconds of upstream silence before sending an SSE keepalive "
        "comment (0 disables). Keep below the client's chunk timeout.",
    )
    parser.add_argument(
        "--no-reasoning-fallback",
        dest="reasoning_fallback",
        action="store_false",
        help="Disable promoting reasoning-only completions to content. When "
        "enabled (default), a turn that ends with reasoning but no content and "
        "no tool call has its reasoning surfaced as content so the client does "
        "not treat it as an empty/incomplete response.",
    )
    parser.set_defaults(reasoning_fallback=REASONING_FALLBACK)
    parser.add_argument(
        "--think-default",
        choices=["off", "on"],
        default="off",
        help="Think setting when using --native-chat and the client omits think.",
    )
    parser.add_argument(
        "--native-chat",
        action="store_true",
        help="Translate chat completions to Ollama /api/chat (think=false). "
        "Off by default: Kilo's OpenAI content-part arrays 400 on native chat, "
        "so /v1 passthrough is the compatible path. Thinking is forced off in "
        "the Modelfile template instead.",
    )
    args = parser.parse_args()
    think_default: bool | str = args.think_default == "on"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info(
        "proxy listening on %s:%s → %s (native-chat %s, heartbeat %.1fs, "
        "reasoning-fallback %s, think-default %s)",
        args.host,
        args.port,
        args.upstream,
        "on" if args.native_chat else "off",
        args.heartbeat,
        "on" if args.reasoning_fallback else "off",
        args.think_default,
    )
    app = build_app(
        args.upstream,
        heartbeat=args.heartbeat,
        reasoning_fallback=args.reasoning_fallback,
        think_default=think_default,
        native_chat=args.native_chat,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
