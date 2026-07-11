#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for DeepSeek V4 Flash (MLX).

Harness goals (for Kilo / agent use):
  - Load + generate on one dedicated MLX worker thread (streams are thread-local)
  - Correct DeepSeek V4 chat template kwargs (thinking_mode)
  - Stop on EOS / next-user tokens (prevents endless monologue)
  - Strong implement-first steering + optional assistant prefill (stops planning loops)
  - Repetition / frequency penalties + loop detector (stops "I'm ready..." spam)
  - Sensible max_tokens defaults for coding agents
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from harness import (
    IMPLEMENT_TEMP_CAP,
    STOP_STRINGS,
    LoopDetector,
    assistant_prefill,
    drop_thrash_assistants,
    is_implement_request,
    prior_assistant_is_thrash,
    steer_messages,
    strip_stop_suffix,
)

log = logging.getLogger("deepseek-openai-server")

MODEL = None
TOKENIZER = None
MODEL_ID = "deepseek-v4-flash-2bit-dq"
MODEL_PATH = ""
DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_THINKING_MODE = "chat"  # "chat" | "thinking"
REPETITION_PENALTY = 1.15
FREQUENCY_PENALTY = 0.5
# Hard ceiling even if client (Kilo) sends 32k — long monologues waste GPU time
MAX_TOKENS_CEILING = 16384

_JOB_Q: queue.Queue = queue.Queue()
_READY = threading.Event()
_LOAD_ERROR: str | None = None


def _messages_to_prompt(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str,
) -> tuple[str, str, bool]:
    """Build chat-template prompt + optional prefill.

    Returns (prompt, prefill, expect_code).
    Prefill is already appended to prompt when non-empty; also returned so the
    stream can emit it to the client as the first content delta.
    """
    # Drop thrash assistant history so Kilo retries get a clean prefill
    # (otherwise mid-thread has no prefill and continues the plan essay).
    cleaned = list(messages)
    if prior_assistant_is_thrash(cleaned):
        log.info("harness: dropping thrash assistant history for clean implement retry")
        cleaned = drop_thrash_assistants(cleaned)

    # Prefill from *client* messages (after thrash drop), before steer suffix
    # so genre labels never see harness text.
    prefill = assistant_prefill(cleaned)
    norm = steer_messages(cleaned)
    expect_code = is_implement_request(norm)

    prompt: str
    if TOKENIZER is not None and hasattr(TOKENIZER, "apply_chat_template"):
        try:
            # DeepSeek V4: thinking_mode 'chat' closes think immediately;
            # 'thinking' opens <think> for extended reasoning.
            prompt = TOKENIZER.apply_chat_template(
                norm,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=thinking_mode,
            )
        except TypeError:
            try:
                prompt = TOKENIZER.apply_chat_template(
                    norm, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                log.warning("chat_template failed (%s); plain join", e)
                prompt = _plain_join(norm)
        except Exception as e:
            log.warning("chat_template failed (%s); plain join", e)
            prompt = _plain_join(norm)
    else:
        prompt = _plain_join(norm)

    if prefill:
        prompt = prompt + prefill
        log.info(
            "harness prefill applied (%d chars) expect_code=%s",
            len(prefill),
            expect_code,
        )
    elif expect_code:
        log.info("harness: implement request steered (no prefill — mid-thread)")

    return prompt, prefill, expect_code


def _plain_join(norm: list[dict[str, str]]) -> str:
    lines = []
    for m in norm:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _configure_tokenizer_stops(tokenizer) -> None:
    """Treat next-turn / EOS markers as end-of-generation."""
    extras = [
        "<｜User｜>",
        "<｜end▁of▁sentence｜>",
        "<｜begin▁of▁sentence｜>",
    ]
    for tok in extras:
        try:
            if hasattr(tokenizer, "add_eos_token"):
                tokenizer.add_eos_token(tok)
            elif hasattr(tokenizer, "eos_token_ids"):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if len(ids) == 1:
                    eids = set(tokenizer.eos_token_ids)
                    eids.add(ids[0])
                    tokenizer.eos_token_ids = list(eids)
        except Exception as e:
            log.debug("could not add eos %r: %s", tok, e)
    try:
        log.info("eos_token_ids=%s", list(tokenizer.eos_token_ids))
    except Exception:
        pass


def _run_generation(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    expect_code: bool = False,
    prefill: str = "",
    stream_out: Optional[queue.Queue] = None,
) -> tuple[str, int, int, str]:
    """
    Run on the MLX worker thread.
    Returns (text, prompt_tokens, completion_tokens, finish_reason).
    If stream_out is set, also puts ("tok", text, None) / ("done", reason, None).
    Prefill (if any) is part of the model prompt; we emit it once as the first
    client-visible content so streaming clients see the forced start.
    """
    prompt_ids = TOKENIZER.encode(prompt)
    # Tokens already "spent" by prefill still count against the budget for
    # generation headroom; max_tokens is model-side continuation length only.
    sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
    # Implement/create turns: stronger anti-repeat (2bit thrash + plan loops)
    rep = REPETITION_PENALTY
    freq = FREQUENCY_PENALTY
    if expect_code:
        rep = max(rep, 1.22)
        freq = max(freq, 0.55)
    processors = make_logits_processors(
        repetition_penalty=rep,
        repetition_context_size=160,
        frequency_penalty=freq,
        frequency_context_size=320,
    )
    detector = LoopDetector(expect_code=expect_code)
    # Seed detector with prefill so code-signal from ```html counts immediately
    if prefill:
        detector.feed(prefill)
        if stream_out is not None:
            stream_out.put(("tok", prefill, None))

    chunks: list[str] = [prefill] if prefill else []
    n_completion = 0
    finish_reason = "length"

    for out in stream_generate(
        model=MODEL,
        tokenizer=TOKENIZER,
        prompt=prompt_ids,
        max_tokens=int(max_tokens),
        sampler=sampler,
        logits_processors=processors,
    ):
        piece = out.text or ""
        if piece:
            chunks.append(piece)
            if stream_out is not None:
                stream_out.put(("tok", piece, None))
        n_completion += 1

        # String stop sequences (multi-token)
        full = "".join(chunks)
        stop_hit = None
        for stop in STOP_STRINGS:
            if stop in full:
                stop_hit = stop
                finish_reason = "stop"
                break
        if stop_hit:
            break

        loop_reason = detector.feed(piece)
        if loop_reason:
            log.warning("loop detector stopped generation: %s", loop_reason)
            finish_reason = "stop"
            break

        if out.finish_reason is not None:
            finish_reason = out.finish_reason or "stop"
            break

    text = strip_stop_suffix("".join(chunks))
    if stream_out is not None:
        stream_out.put(("done", finish_reason, None))
    return text, len(prompt_ids), n_completion, finish_reason


def _worker_loop(model_path: str) -> None:
    global MODEL, TOKENIZER, _LOAD_ERROR
    try:
        stream = mx.default_stream(mx.default_device())
        log.info("MLX worker started; loading %s ...", model_path)
        t0 = time.time()
        with mx.stream(stream):
            MODEL, TOKENIZER = load(model_path)
            _configure_tokenizer_stops(TOKENIZER)
            mx.eval(mx.zeros((1,)))
        log.info(
            "Model loaded in %.1fs — peak mem ~%.1f GB",
            time.time() - t0,
            mx.get_peak_memory() / 1e9,
        )
        _READY.set()
    except Exception as e:
        log.exception("model load failed")
        _LOAD_ERROR = str(e)
        _READY.set()
        return

    while True:
        job = _JOB_Q.get()
        if job is None:
            break
        kind = job["kind"]
        try:
            with mx.stream(stream):
                if kind == "generate":
                    text, n_p, n_c, reason = _run_generation(
                        prompt=job["prompt"],
                        max_tokens=job["max_tokens"],
                        temperature=job["temperature"],
                        top_p=job["top_p"],
                        expect_code=job.get("expect_code", False),
                        prefill=job.get("prefill", ""),
                    )
                    job["result_q"].put(("ok", (text, n_p, n_c, reason)))
                elif kind == "stream":
                    out_q: queue.Queue = job["out_q"]
                    try:
                        text, n_p, n_c, reason = _run_generation(
                            prompt=job["prompt"],
                            max_tokens=job["max_tokens"],
                            temperature=job["temperature"],
                            top_p=job["top_p"],
                            expect_code=job.get("expect_code", False),
                            prefill=job.get("prefill", ""),
                            stream_out=out_q,
                        )
                        job["meta_q"].put(("ok", n_p, n_c, reason, text))
                    except Exception as e:
                        out_q.put(("err", str(e), None))
                        job["meta_q"].put(("err", str(e)))
                else:
                    job.get("result_q", queue.Queue()).put(
                        ("err", f"unknown job {kind}")
                    )
        except Exception as e:
            log.exception("worker job failed")
            if kind == "generate":
                job["result_q"].put(("err", str(e)))
            elif kind == "stream":
                job["out_q"].put(("err", str(e), None))
                job["meta_q"].put(("err", str(e)))


def _generate_text(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    expect_code: bool = False,
    prefill: str = "",
) -> tuple[str, int, int, str]:
    result_q: queue.Queue = queue.Queue()
    _JOB_Q.put(
        {
            "kind": "generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "expect_code": expect_code,
            "prefill": prefill,
            "result_q": result_q,
        }
    )
    status, payload = result_q.get()
    if status != "ok":
        raise RuntimeError(payload)
    return payload  # type: ignore[return-value]


def _stream_sse(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    expect_code: bool = False,
    prefill: str = "",
) -> Iterator[str]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first)}\n\n"

    out_q: queue.Queue = queue.Queue()
    meta_q: queue.Queue = queue.Queue()
    _JOB_Q.put(
        {
            "kind": "stream",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "expect_code": expect_code,
            "prefill": prefill,
            "out_q": out_q,
            "meta_q": meta_q,
        }
    )
    finish_reason = "stop"
    while True:
        kind, text, finish = out_q.get()
        if kind == "err":
            raise RuntimeError(text)
        if kind == "done":
            finish_reason = text or "stop"
            break
        if text:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        if finish is not None:
            finish_reason = finish or "stop"

    end = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(end)}\n\n"
    try:
        meta_q.get(timeout=1)
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            self._send_json(
                200,
                {
                    "status": "ok" if _READY.is_set() and not _LOAD_ERROR else "loading",
                    "model": MODEL_ID,
                    "ready": _READY.is_set() and not _LOAD_ERROR,
                    "thinking_mode_default": DEFAULT_THINKING_MODE,
                    "harness": "steer+prefill+loop-detector",
                },
            )
            return
        if path.startswith("/v1/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        },
                        {
                            "id": MODEL_PATH,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        },
                    ],
                },
            )
            return
        self._send_json(
            404,
            {
                "error": {
                    "message": f"not found: {path}",
                    "type": "invalid_request_error",
                }
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(
                404,
                {
                    "error": {
                        "message": f"not found: {path}",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        if not _READY.is_set():
            self._send_json(
                503,
                {"error": {"message": "model still loading", "type": "server_error"}},
            )
            return
        if _LOAD_ERROR:
            self._send_json(
                500,
                {
                    "error": {
                        "message": f"model failed to load: {_LOAD_ERROR}",
                        "type": "server_error",
                    }
                },
            )
            return

        try:
            body = self._read_json()
        except json.JSONDecodeError as e:
            self._send_json(
                400,
                {
                    "error": {
                        "message": f"invalid json: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        messages = body.get("messages") or []
        if not messages:
            self._send_json(
                400,
                {
                    "error": {
                        "message": "messages required",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        # Kilo often omits max_tokens or sends huge values — clamp sensibly
        raw_max = body.get("max_tokens") or body.get("max_completion_tokens")
        if raw_max is None:
            max_tokens = DEFAULT_MAX_TOKENS
        else:
            max_tokens = max(1, min(int(raw_max), MAX_TOKENS_CEILING))

        temperature = float(body.get("temperature", DEFAULT_TEMP))
        top_p = float(body.get("top_p", DEFAULT_TOP_P))
        stream = bool(body.get("stream", False))

        thinking_mode = DEFAULT_THINKING_MODE
        cta = body.get("chat_template_kwargs") or {}
        if isinstance(cta, dict) and cta.get("thinking_mode"):
            thinking_mode = str(cta["thinking_mode"])
        if body.get("thinking_mode"):
            thinking_mode = str(body["thinking_mode"])
        if thinking_mode not in ("chat", "thinking"):
            thinking_mode = "chat"

        prompt, prefill, expect_code = _messages_to_prompt(
            messages, thinking_mode=thinking_mode
        )
        # Client opt-out of forced assistant prefill (still keeps steer + loop detector)
        if body.get("harness_prefill") is False or body.get("no_prefill"):
            if prefill and prompt.endswith(prefill):
                prompt = prompt[: -len(prefill)]
            prefill = ""

        # 2-bit models thrash hard at temp=1.0 on long code; cap for implement turns
        # unless client explicitly sets harness_temp / no_temp_cap
        if expect_code and not body.get("no_temp_cap"):
            if body.get("harness_temp") is not None:
                temperature = float(body["harness_temp"])
            elif temperature > IMPLEMENT_TEMP_CAP:
                log.info(
                    "harness temp cap: %.2f -> %.2f (implement turn)",
                    temperature,
                    IMPLEMENT_TEMP_CAP,
                )
                temperature = IMPLEMENT_TEMP_CAP

        log.info(
            "chat completion: msgs=%d max_tokens=%d temp=%.2f stream=%s "
            "thinking=%s expect_code=%s prefill=%d prompt_chars=%d",
            len(messages),
            max_tokens,
            temperature,
            stream,
            thinking_mode,
            expect_code,
            len(prefill),
            len(prompt),
        )
        log.debug("prompt tail: %r", prompt[-240:])

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                for line in _stream_sse(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    expect_code=expect_code,
                    prefill=prefill,
                ):
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            text, n_prompt, n_completion, reason = _generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                expect_code=expect_code,
                prefill=prefill,
            )
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model") or MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": reason or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_completion,
                        "total_tokens": n_prompt + n_completion,
                    },
                },
            )
        except Exception as e:
            log.exception("generation failed")
            self._send_json(
                500, {"error": {"message": str(e), "type": "server_error"}}
            )


def main() -> None:
    global MODEL_ID, MODEL_PATH, DEFAULT_TEMP, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS
    global DEFAULT_THINKING_MODE, REPETITION_PENALTY, FREQUENCY_PENALTY

    ap = argparse.ArgumentParser(description="DeepSeek V4 Flash — simple OpenAI server")
    ap.add_argument("--model", required=True, help="Local model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--model-id", default="deepseek-v4-flash-2bit-dq")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--thinking-mode",
        choices=("chat", "thinking"),
        default="chat",
        help="DeepSeek V4 chat template mode (default: chat = no long <think>)",
    )
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--frequency-penalty", type=float, default=0.5)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    MODEL_PATH = args.model
    MODEL_ID = args.model_id
    DEFAULT_TEMP = args.temp
    DEFAULT_TOP_P = args.top_p
    DEFAULT_MAX_TOKENS = args.max_tokens
    DEFAULT_THINKING_MODE = args.thinking_mode
    REPETITION_PENALTY = args.repetition_penalty
    FREQUENCY_PENALTY = args.frequency_penalty

    worker = threading.Thread(
        target=_worker_loop, args=(args.model,), name="mlx-worker", daemon=True
    )
    worker.start()

    log.info(
        "Waiting for model load (thinking_mode=%s rep=%.2f freq=%.2f max_tokens=%d)...",
        DEFAULT_THINKING_MODE,
        REPETITION_PENALTY,
        FREQUENCY_PENALTY,
        DEFAULT_MAX_TOKENS,
    )
    while not _READY.is_set():
        time.sleep(0.2)
    if _LOAD_ERROR:
        raise SystemExit(f"model load failed: {_LOAD_ERROR}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(
        "OpenAI API ready at http://%s:%d/v1  model_id=%s harness=steer+prefill+loop",
        args.host,
        args.port,
        MODEL_ID,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        _JOB_Q.put(None)
        httpd.shutdown()


if __name__ == "__main__":
    main()
