#!/usr/bin/env python3
"""HTTP API for YuE2-3B (lyra). One generation at a time.

Must be launched via `uv run python` from the yue2-mlx engine checkout so
`lyra` imports resolve. See ./2_start_server.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PIPE = None
PIPE_LOCK = threading.Lock()
BUSY = threading.Event()
ARGS: argparse.Namespace | None = None


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    return status, body


def load_pipeline() -> Any:
    global PIPE
    if PIPE is not None:
        return PIPE
    from lyra import YuE2Pipeline  # type: ignore

    assert ARGS is not None
    kwargs: dict[str, Any] = {
        "vae": ARGS.vae,
        "precision": ARGS.precision,
        "local_files_only": True,
        "progress": True,
    }
    if ARGS.require_ac:
        kwargs["require_ac"] = True
    PIPE = YuE2Pipeline.from_pretrained(ARGS.model, **kwargs)
    return PIPE


def song_payload(song: Any, output_dir: Path) -> dict[str, Any]:
    audio = output_dir / "audio.flac"
    score = output_dir / "score.abc"
    result = output_dir / "result.json"
    truncated = getattr(song, "truncated", None)
    if hasattr(truncated, "__dict__"):
        truncated_out = dict(vars(truncated))
    elif isinstance(truncated, dict):
        truncated_out = truncated
    else:
        truncated_out = truncated
    duration = None
    audio_arr = getattr(song, "audio", None)
    rate = getattr(song, "sample_rate", None) or 48000
    if audio_arr is not None:
        try:
            duration = round(len(audio_arr) / float(rate), 3)
        except TypeError:
            duration = None
    return {
        "id": getattr(song, "id", output_dir.name),
        "model": ARGS.alias if ARGS else "yue2-3b-mlx",
        "output_dir": str(output_dir),
        "audio": str(audio) if audio.is_file() else None,
        "score": str(score) if score.is_file() else None,
        "result": str(result) if result.is_file() else None,
        "truncated": truncated_out,
        "duration_seconds": duration,
        "sample_rate": rate,
        "status": "complete",
    }


def unique_output(base: str) -> Path:
    assert ARGS is not None
    root = Path(ARGS.outputs)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{base}-{stamp}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = root / f"{base}-{stamp}-{n}"
    return candidate


def handle_generate(body: dict[str, Any], plan_only: bool) -> tuple[int, bytes]:
    if BUSY.is_set():
        return _json_bytes(
            {"error": "busy", "detail": "one YuE2 request at a time"}, 409
        )
    request = dict(body)
    request.pop("output", None)
    song_id = str(request.get("id") or ("plan" if plan_only else "song"))
    output_dir = unique_output(song_id)
    BUSY.set()
    try:
        with PIPE_LOCK:
            pipe = load_pipeline()
            if plan_only:
                plan = pipe.plan(**_call_kwargs(request))
                plan.save(str(output_dir))
                score = output_dir / "score.abc"
                return _json_bytes(
                    {
                        "id": song_id,
                        "model": ARGS.alias if ARGS else "yue2-3b-mlx",
                        "output_dir": str(output_dir),
                        "score": str(score) if score.is_file() else None,
                        "status": "complete",
                        "kind": "plan",
                    }
                )
            song = pipe(**_call_kwargs(request))
            song.save_artifacts(str(output_dir))
            return _json_bytes(song_payload(song, output_dir))
    except TypeError as exc:
        return _json_bytes({"error": "bad_request", "detail": str(exc)}, 400)
    except InterruptedError as exc:
        return _json_bytes({"error": "interrupted", "detail": str(exc)}, 503)
    except Exception as exc:
        traceback.print_exc()
        return _json_bytes({"error": "generation_failed", "detail": str(exc)}, 500)
    finally:
        BUSY.clear()


def _call_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "style",
        "lyrics",
        "cot",
        "seed",
        "abc",
        "cfg_scale",
        "id",
        "generation_config",
        "abc_sampling",
        "semantic_sampling",
        "tags",
    }
    return {k: v for k, v in request.items() if k in allowed}


class Handler(BaseHTTPRequestHandler):
    server_version = "yue2-server/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        assert ARGS is not None
        if path in ("/", "/help"):
            payload = {
                "name": "YuE2-3B MLX",
                "model": ARGS.alias,
                "endpoints": {
                    "GET /health": "process + load state",
                    "GET /v1/models": "OpenAI-shaped model list",
                    "POST /generate": "style+lyrics JSON → song artifacts",
                    "POST /plan": "style+lyrics JSON → ABC plan",
                },
                "note": "Not a chat-completions coding model. One request at a time.",
            }
            status, body = _json_bytes(payload)
            self._send(status, body)
            return
        if path == "/health":
            payload = {
                "status": "ok",
                "model": ARGS.alias,
                "precision": ARGS.precision,
                "loaded": PIPE is not None,
                "busy": BUSY.is_set(),
            }
            status, body = _json_bytes(payload)
            self._send(status, body)
            return
        if path in ("/v1/models", "/models"):
            payload = {
                "object": "list",
                "data": [
                    {
                        "id": ARGS.alias,
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            }
            status, body = _json_bytes(payload)
            self._send(status, body)
            return
        self._send(*_json_bytes({"error": "not_found"}, 404))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send(*_json_bytes({"error": "invalid_json", "detail": str(exc)}, 400))
            return
        if not isinstance(body, dict):
            self._send(*_json_bytes({"error": "invalid_json", "detail": "object required"}, 400))
            return
        if path in ("/generate", "/v1/audio/generations"):
            self._send(*handle_generate(body, plan_only=False))
            return
        if path == "/plan":
            self._send(*handle_generate(body, plan_only=True))
            return
        self._send(*_json_bytes({"error": "not_found"}, 404))


def main() -> int:
    global ARGS
    parser = argparse.ArgumentParser(description="YuE2-3B HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--alias", default="yue2-3b-mlx")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--require-ac", action="store_true")
    ARGS = parser.parse_args()

    if not ARGS.lazy:
        print(f"→ Preloading YuE2 pipeline ({ARGS.precision}) ...", flush=True)
        load_pipeline()
        print("→ Pipeline loaded", flush=True)

    httpd = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"→ Listening on http://{ARGS.host}:{ARGS.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("→ Stopping", flush=True)
    finally:
        httpd.server_close()
        if PIPE is not None:
            close = getattr(PIPE, "close", None)
            if callable(close):
                close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
