#!/usr/bin/env python3
"""Minimal OpenAI-compatible reverse proxy for Ollama.

Clients (OpenCode / Kilo) → this proxy → Ollama /v1.
Provides /healthz and transparent pass-through of /v1/* routes.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("glm_openai_proxy")

UPSTREAM = "http://127.0.0.1:11434/v1"
TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)


def build_app(upstream: str) -> FastAPI:
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
                try:
                    async for chunk in upstream_resp.aiter_raw():
                        yield chunk
                finally:
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument(
        "--upstream",
        default=UPSTREAM,
        help="Ollama OpenAI base, e.g. http://127.0.0.1:11434/v1",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("proxy listening on %s:%s → %s", args.host, args.port, args.upstream)
    app = build_app(args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
