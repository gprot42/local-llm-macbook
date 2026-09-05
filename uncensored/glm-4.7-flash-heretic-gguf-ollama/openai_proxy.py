#!/usr/bin/env python3
"""Minimal OpenAI-compatible reverse proxy for Ollama.

Clients (OpenCode / Kilo) → this proxy → Ollama /v1.
Provides /healthz and transparent pass-through of /v1/* routes.
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
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("glm_openai_proxy")

UPSTREAM = "http://127.0.0.1:11434/v1"
TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)

# Emit an SSE comment line if the upstream stream is quiet for this many seconds.
# Ollama's OpenAI endpoint buffers tool_calls and flushes the whole call as one
# chunk at the end, so a large tool call at high context can go minutes with no
# bytes downstream. That trips a client's per-chunk watchdog (Kilo's
# chunkTimeout) and aborts an otherwise-healthy generation. A comment line
# (":\n\n") is ignored by SSE parsers but any byte resets that watchdog, so the
# heartbeat must be shorter than the client's chunkTimeout.
HEARTBEAT_INTERVAL = 15.0

# Promote reasoning to content when a completion ends with reasoning but no
# assistant content and no tool call. GLM-4.7-Flash (a thinking model) routinely
# ends a turn inside the reasoning channel — either it stops right after
# thinking, or its whole output budget went to reasoning — leaving content="".
# Kilo treats such a turn as an "incomplete response" (its replayable() check:
# reasoning-only with no text and no tool → retry), retries twice, then fails
# with "The provider repeatedly ended the response before returning usable
# output." Surfacing the reasoning as content gives the client usable output and
# preserves what the model actually produced instead of discarding it.
REASONING_FALLBACK = True


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
) -> FastAPI:
    app = FastAPI(title="GLM OpenAI proxy", docs_url=None, redoc_url=None)
    base = upstream.rstrip("/")  # e.g. http://127.0.0.1:11434/v1

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "upstream": base}

    async def _forward(request: Request, target: str) -> Response:
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "transfer-encoding", "connection"}
        }
        body = await request.body()

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
                # Two jobs on the way through:
                #  1) Heartbeat: race each upstream chunk against the heartbeat
                #     interval and, on a quiet gap, send an SSE comment so a
                #     downstream chunk-watchdog keeps the connection open through
                #     long buffered tool calls. The pending read is kept as a
                #     task and awaited again after each heartbeat — never
                #     cancelled — because cancelling an in-flight httpx read
                #     corrupts the response stream.
                #  2) Reasoning fallback: parse the SSE events, and if the turn
                #     ends with reasoning but no content and no tool call, inject
                #     a content delta carrying the reasoning before the finish
                #     event, so the client sees usable output.
                buf = b""
                saw_content = False
                saw_tool = False
                reasoning: list[str] = []
                held_finish: bytes | None = None
                done_sent = False
                emitted_finish = False
                meta: dict[str, Any] = {}

                def synth_finish(reason: str) -> bytes:
                    payload = {
                        "id": meta.get("id", "chatcmpl-proxy"),
                        "object": "chat.completion.chunk",
                        "created": meta.get("created", int(time.time())),
                        "model": meta.get("model", ""),
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": reason}
                        ],
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
                    # ev is one SSE event without its trailing blank line.
                    nonlocal saw_content, saw_tool, held_finish, done_sent
                    nonlocal emitted_finish
                    text = ev.strip()
                    if not text or text.startswith(b":"):
                        return [ev + b"\n\n"]  # comment / keepalive — pass through
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
                        # Ollama can end a reasoning-only turn with [DONE] but no
                        # finish_reason chunk. Without a finish reason the client
                        # (Kilo) maps the absent value to "other" and shows
                        # "Response ended unexpectedly and may be incomplete."
                        # Guarantee one so the recovered turn reads as a clean stop.
                        if not emitted_finish:
                            outs.append(
                                synth_finish("tool_calls" if saw_tool else "stop")
                            )
                            emitted_finish = True
                        outs.append(ev + b"\n\n")
                        done_sent = True
                        return outs
                    if not text.startswith(b"data:"):
                        return [ev + b"\n\n"]
                    try:
                        obj = json.loads(text[len(b"data:"):].strip())
                    except ValueError:
                        return [ev + b"\n\n"]
                    for key in ("id", "model", "created"):
                        if key in obj and key not in meta:
                            meta[key] = obj[key]
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
                        # Hold the finish event so a fallback content delta can be
                        # emitted before it; released at [DONE] or stream end.
                        held_finish = ev + b"\n\n"
                        return []
                    return [ev + b"\n\n"]

                raw = upstream_resp.aiter_raw()
                read_task: asyncio.Task | None = None
                stream_error: BaseException | None = None
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
                            except Exception as exc:  # upstream dropped mid-stream
                                stream_error = exc
                                chunk = None
                            finally:
                                read_task = None
                        else:
                            try:
                                chunk = await raw.__anext__()
                            except StopAsyncIteration:
                                chunk = None
                            except Exception as exc:  # upstream dropped mid-stream
                                stream_error = exc
                                chunk = None
                        if chunk is None:
                            break
                        buf += chunk
                        while b"\n\n" in buf:
                            ev, buf = buf.split(b"\n\n", 1)
                            for out in process_event(ev):
                                yield out
                    # Flush a trailing partial event, if any.
                    if buf.strip():
                        for out in process_event(buf):
                            yield out
                    # Always terminate the client stream with a finish reason and a
                    # [DONE]. Without this, an upstream drop after the finish chunk
                    # was buffered (or before it arrived at all) — or a clean end
                    # that never carried a finish_reason chunk — would leave the
                    # client with no finish reason. Kilo maps an absent finish
                    # reason to "other" and shows "Response ended unexpectedly and
                    # may be incomplete", so synthesize one whenever none was sent.
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
                        yield b"data: [DONE]\n\n"
                    if stream_error is not None:
                        log.warning("upstream stream ended early: %r", stream_error)
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
        if reasoning_fallback and "application/json" in media:
            content = _promote_reasoning_json(content)
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=out_headers,
            media_type=media or None,
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_v1(path: str, request: Request) -> Response:
        # Client: /v1/chat/completions → upstream: {base}/chat/completions
        target = f"{base}/{path}" if path else base
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return await _forward(request, target)

    @app.api_route(
        "/v1",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info(
        "proxy listening on %s:%s → %s (heartbeat %.1fs, reasoning-fallback %s)",
        args.host,
        args.port,
        args.upstream,
        args.heartbeat,
        "on" if args.reasoning_fallback else "off",
    )
    app = build_app(
        args.upstream,
        heartbeat=args.heartbeat,
        reasoning_fallback=args.reasoning_fallback,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
