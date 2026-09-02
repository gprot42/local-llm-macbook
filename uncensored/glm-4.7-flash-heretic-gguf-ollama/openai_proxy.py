#!/usr/bin/env python3
"""Minimal OpenAI-compatible reverse proxy for Ollama.

Clients (OpenCode / Kilo) → this proxy → Ollama /v1.
Provides /healthz and transparent pass-through of /v1/* routes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
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


def build_app(upstream: str, heartbeat: float = HEARTBEAT_INTERVAL) -> FastAPI:
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
                # Race each upstream chunk against the heartbeat interval; on a
                # quiet gap, send an SSE comment so a downstream chunk-watchdog
                # keeps the connection open through long buffered tool calls.
                #
                # The pending read is kept as a task and awaited again after each
                # heartbeat — never cancelled — because cancelling an in-flight
                # httpx read corrupts the response stream.
                if not heartbeat or heartbeat <= 0:
                    try:
                        async for chunk in upstream_resp.aiter_raw():
                            yield chunk
                    finally:
                        await upstream_resp.aclose()
                        await client.aclose()
                    return

                raw = upstream_resp.aiter_raw()
                read_task: asyncio.Task | None = None
                try:
                    while True:
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
                            break
                        finally:
                            read_task = None
                        yield chunk
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info(
        "proxy listening on %s:%s → %s (heartbeat %.1fs)",
        args.host,
        args.port,
        args.upstream,
        args.heartbeat,
    )
    app = build_app(args.upstream, heartbeat=args.heartbeat)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
