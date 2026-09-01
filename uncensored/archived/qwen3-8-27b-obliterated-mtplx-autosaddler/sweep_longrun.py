#!/usr/bin/env python3
"""Sweep sampling for long-running agent use against raw mtplx (default :8767).

Hits the engine directly so the kilo proxy cannot hide the real sampler.
Scores: tool reliability, multi-step continue, long-decode loopiness, latency.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from itertools import product
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def post(base: str, body: dict, timeout: float = 180.0) -> tuple[int, Any, float]:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.time() - t0
            return resp.status, json.loads(raw.decode("utf-8")), elapsed
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - t0


def msg_of(data: Any) -> dict:
    if not isinstance(data, dict):
        return {}
    return (data.get("choices") or [{}])[0].get("message") or {}


def finish_of(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    return str((data.get("choices") or [{}])[0].get("finish_reason") or "")


def content_of(msg: dict) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else ""


def tool_blob(msg: dict) -> str:
    parts: list[str] = []
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        parts.append(f"{fn.get('name')} {fn.get('arguments')}")
    return " ".join(parts).lower()


def repeat_score(text: str) -> float:
    """0 = unique, 1 = fully looping. Line + 8-gram overlap."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4:
        counts = Counter(lines)
        return max(counts.values()) / len(lines)
    toks = text.split()
    if len(toks) < 16:
        return 0.0
    grams = [" ".join(toks[i : i + 8]) for i in range(len(toks) - 7)]
    if not grams:
        return 0.0
    return max(Counter(grams).values()) / len(grams)


def chat(
    base: str,
    messages: list[dict],
    *,
    temperature: float,
    frequency_penalty: float,
    max_tokens: int,
    tools: bool,
    enable_thinking: bool,
) -> tuple[int, Any, float]:
    body: dict[str, Any] = {
        "model": "qwen3.8-27b-obliterated-mtplx",
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "frequency_penalty": frequency_penalty,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    return post(base, body)


def run_case(base: str, temp: float, fp: float, think: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "temperature": temp,
        "frequency_penalty": fp,
        "enable_thinking": think,
    }
    score = 0.0
    notes: list[str] = []

    code, data, elapsed = chat(
        base,
        [{"role": "user", "content": "Reply with exactly the word PING."}],
        temperature=temp,
        frequency_penalty=fp,
        max_tokens=16,
        tools=False,
        enable_thinking=think,
    )
    content = content_of(msg_of(data)).strip()
    ping_ok = code == 200 and "PING" in content.upper()
    score += 1.0 if ping_ok else 0.0
    row["ping"] = {"ok": ping_ok, "s": round(elapsed, 2), "content": content[:40]}

    code, data, elapsed = chat(
        base,
        [
            {
                "role": "user",
                "content": "Using tools only, run bash with command exactly: echo harness_ok",
            }
        ],
        temperature=temp,
        frequency_penalty=fp,
        max_tokens=256,
        tools=True,
        enable_thinking=think,
    )
    blob = tool_blob(msg_of(data))
    tool_ok = code == 200 and "bash" in blob and "harness_ok" in blob
    score += 2.0 if tool_ok else 0.0
    row["tool"] = {"ok": tool_ok, "s": round(elapsed, 2), "blob": blob[:80]}

    hist = [
        {
            "role": "user",
            "content": (
                "Do 2 steps with tools: 1) echo step1_harness "
                "2) echo step2_harness. Continue after each tool result."
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "s1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"echo step1_harness"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "s1", "content": "step1_harness"},
    ]
    code, data, elapsed = chat(
        base,
        hist,
        temperature=temp,
        frequency_penalty=fp,
        max_tokens=256,
        tools=True,
        enable_thinking=think,
    )
    blob = tool_blob(msg_of(data))
    step2_ok = code == 200 and "bash" in blob and "step2" in blob
    score += 3.0 if step2_ok else 0.0
    row["step2"] = {"ok": step2_ok, "s": round(elapsed, 2), "blob": blob[:80]}

    code, data, elapsed = chat(
        base,
        [
            {
                "role": "user",
                "content": (
                    "Write a complete Python function parse_ini(text: str) -> dict "
                    "that parses INI-style sections and key=value lines. Include a "
                    "docstring and 8-12 lines of body. Do not repeat yourself. "
                    "Output only the function."
                ),
            }
        ],
        temperature=temp,
        frequency_penalty=fp,
        max_tokens=400,
        tools=False,
        enable_thinking=think,
    )
    msg = msg_of(data)
    text = content_of(msg)
    finish = finish_of(data)
    rep = repeat_score(text)
    n = len(text)
    long_ok = (
        code == 200
        and finish in ("stop", "length")
        and n >= 180
        and rep < 0.35
        and "def parse_ini" in text
    )
    score += 3.0 if long_ok else (1.0 if n >= 180 and rep < 0.35 else 0.0)
    if rep >= 0.35:
        notes.append(f"loopish rep={rep:.2f}")
    if n < 180:
        notes.append(f"short n={n}")
    row["long"] = {
        "ok": long_ok,
        "s": round(elapsed, 2),
        "n": n,
        "rep": round(rep, 3),
        "finish": finish,
    }

    row["score"] = round(score, 2)
    row["notes"] = "; ".join(notes)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8767")
    ap.add_argument("--quick", action="store_true", help="Card defaults only + neighbors")
    args = ap.parse_args()

    temps = (0.0, 0.3) if args.quick else (0.0, 0.3, 0.6)
    fps = (0.0, 0.2) if args.quick else (0.0, 0.15, 0.2, 0.3)
    thinks = (False,)

    rows: list[dict[str, Any]] = []
    print(f"sweep base={args.base} temps={temps} fps={fps}")
    for temp, fp, think in product(temps, fps, thinks):
        print(f"\n== temp={temp} fp={fp} think={think} ==")
        row = run_case(args.base, temp, fp, think)
        rows.append(row)
        print(
            f"  score={row['score']} ping={row['ping']['ok']} "
            f"tool={row['tool']['ok']} step2={row['step2']['ok']} "
            f"long={row['long']['ok']} n={row['long']['n']} "
            f"rep={row['long']['rep']} {row['notes']}"
        )

    rows.sort(key=lambda r: (-r["score"], r["long"]["rep"], r["temperature"]))
    print("\n== ranking ==")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:2d}. score={r['score']:4.1f}  temp={r['temperature']}  "
            f"fp={r['frequency_penalty']}  long_n={r['long']['n']}  "
            f"rep={r['long']['rep']}  {r['notes']}"
        )
    best = rows[0]
    print(
        "\nBEST: temperature={temperature} frequency_penalty={frequency_penalty} "
        "enable_thinking={enable_thinking} score={score}".format(**best)
    )
    out = "analysis-sweep-longrun.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"base": args.base, "rows": rows}, f, indent=2)
        print(f"wrote {out}")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
