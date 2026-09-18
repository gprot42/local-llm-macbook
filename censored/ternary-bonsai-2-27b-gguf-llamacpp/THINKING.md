# Thinking mode — Ternary Bonsai 2 27B

Bonsai 2 27B is a **reasoning model**: it can "think" (emit a hidden chain of thought as `reasoning_content`) before it answers or calls a tool. This stack serves it with thinking **off by default** and lets you turn it on per session, per request, or server‑wide. This page explains how, when it's worth it, and what it costs.

## TL;DR

| You want | Do this |
|---|---|
| Think for this OpenCode session | `/models` → **Ternary Bonsai 2 27B — thinking (1024‑token budget)** (`bonsai/ternary-bonsai-2-27b-think`) |
| Back to fast / no thinking | `/models` → `bonsai/ternary-bonsai-2-27b` (the default) |
| Think for everything, server‑wide | `./2_start_llama.sh --think` (2048 budget) or `--think-budget N` |
| Think from your own code | add `chat_template_kwargs: {"enable_thinking": true}` and `thinking_budget_tokens: N` to the request |

No server restart is needed for the OpenCode switch; the two entries point at the same server.

## How to enable it

### In OpenCode (per session)

1. Restart OpenCode once after installing the config (`./install-opencode-json.sh`) so it sees both entries.
2. Type `/models` and pick **Ternary Bonsai 2 27B — thinking (1024‑token budget)**.
3. The model's thinking now renders in the UI as reasoning blocks, then the answer or tool call follows.

Switch back with `/models` → `bonsai/ternary-bonsai-2-27b`.

Under the hood the `-think` entry in [`opencode.json`](opencode.json) is the same model with these `options`, which OpenCode passes verbatim into every request:

```json
"chat_template_kwargs": { "enable_thinking": true },
"thinking_budget_tokens": 1024,
"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0
```

`enable_thinking` switches the model's thinking on; `thinking_budget_tokens` is PrismML's per‑request cap (their llama.cpp fork honors it — upstream `reasoning_budget` is server‑wide only); the sampling values are the model card's thinking‑mode preset. `"reasoning": true` on the entry tells OpenCode to render the reasoning. Change the budget by editing that number and re‑running `./install-opencode-json.sh`.

### Server‑wide

```bash
./2_start_llama.sh --think               # thinking on, 2048-token budget (PrismML "Medium")
./2_start_llama.sh --think-budget 8192   # bigger budget ("High"); -1 = unlimited
./2_start_llama.sh --no-think            # default: thinking off
```

Or `BONSAI_THINK=1` / `BONSAI_THINK_BUDGET=N` in the environment. This switches the whole server to `--reasoning on --reasoning-preserve --reasoning-budget N` with the thinking sampling preset, so every client gets thinking — including OpenCode's default entry.

### Per request (API)

```bash
curl -s http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ternary-bonsai-2-27b",
  "messages": [{"role":"user","content":"Why does this deadlock? ..."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_budget_tokens": 1024,
  "max_tokens": 4000
}' | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('THINKING:', m.get('reasoning_content','')[:300]); print('ANSWER:', m['content'][:300])"
```

The thinking comes back in `reasoning_content`, separate from `content`; in streaming mode it arrives as `reasoning_content` deltas.

## Why you'd want thinking mode

The model was trained and benchmarked **in thinking mode** — that's where PrismML's headline numbers come from (98.2% of the FP16 baseline; math 96.6, LiveCodeBench 90.1, BFCL tool calling 74.9). Thinking lets it plan, check its own work, and back out of a wrong first idea before committing to an answer or a tool call. It pays off when the task is *hard to get right the first time*:

- **Debugging with non‑obvious causes** — race conditions, off‑by‑one/boundary bugs, "works locally, fails in CI", anything where the fix requires forming and testing a hypothesis.
- **Algorithmic or mathematical work** — complexity trade‑offs, invariants, numeric edge cases, anything you'd want a human to reason through on paper.
- **Design decisions** — choosing between architectures, planning a multi‑file refactor, deciding an API shape before writing code.
- **Ambiguous multi‑step tasks** where acting immediately tends to produce a wrong first attempt that then needs several tool‑loop iterations to repair.
- **Reviewing** — finding subtle problems in code or a plan, where breadth of consideration matters more than speed.

In non‑thinking mode the model still uses tools well and writes good code (Qwen's own coder models run this way), but it "acts first" — cheap mistakes get fixed by another tool call rather than avoided.

## Why it's *off* by default — the costs

Measured on this machine (M5 Max) during a one‑hour OpenCode game‑building session with thinking on:

1. **Time.** Roughly half of the 56k tokens generated were thinking. At 13–15 tok/s decode (20–45k context), a 1024‑token think adds up to ~75 s before the first visible token of every turn; 2048 adds up to ~2.5 min. The session took ~2× longer than it would have without thinking.
2. **Context.** OpenCode sends each step's `reasoning_content` back and the chat template renders it into the prompt for the whole tool loop (verified: 581 vs 40 prompt tokens with/without prior reasoning — regardless of `--reasoning-preserve`). So the context grows by the *budget* on every step. OpenCode's usable input window is `limit.context − limit.output = 49152 − 16384 = 32768`, and after a compaction the floor is already ~25–28k (14k baseline prompt + ~10k summary). With a 2048 budget, 2–3 steps refilled it and the session died with `Compaction exhausted: context still exceeds model limits after 3 attempts`. The 1024 budget halves that pressure; it does not remove it.
3. **Exhaustion if unbounded.** The model defaults to `xhigh` reasoning effort (PrismML: `low` is not supported). Without a budget it can spend the entire `max_tokens` thinking and produce no answer — OpenCode shows *"hit its output limit while reasoning and produced no actionable output"* and the file you asked for never gets written. Every thinking path in this stack is budgeted for that reason; never enable `enable_thinking` without `thinking_budget_tokens` (or the server's `--reasoning-budget`).

## Choosing a budget

| Budget | PrismML UI label | Use |
|---|---|---|
| 512 | Low | a quick sanity check before acting; minimal context cost |
| **1024** | — | the OpenCode `-think` entry: enough for real debugging/design reasoning, half the context cost of Medium |
| 2048 | Medium | server‑wide `--think` default; PrismML's recommended balance for chat |
| 8192 | High | single hard problems via the API; not for agent loops |
| −1 | Max | never in an agent loop — see cost 3 |

Rule of thumb: **think for the hard turn, then switch back.** Use the `-think` entry to diagnose or design, and the default entry to do the long build and the big file writes.

## Verifying it's on

- OpenCode: reasoning blocks appear before the answer.
- API: `reasoning_content` is non‑empty; with a budget, a forced long think ends at roughly the budget and `content` still follows. (Verified on this stack: cut at ~1000 tokens with `thinking_budget_tokens: 1024`; `finish=tool_calls` with tools.)
- Server log: `./2_start_llama.sh status` shows the mode; the startup line reads `reasoning on (budget N tokens)` or `reasoning off`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "produced no actionable output" / empty `content`, `finish=length` | thinking with no budget | add `thinking_budget_tokens` (OpenCode `-think` entry has it) or start the server with `--think-budget N` |
| `Compaction exhausted: context still exceeds model limits` | reasoning retained in context on a long session | switch to the default entry, lower the budget, or raise `limit.context` together with `BONSAI_CTX` |
| Thinking never appears in OpenCode | wrong entry selected, or the entry lacks `"reasoning": true` | `/models` → the `-think` entry; re‑run `./install-opencode-json.sh` |
| `<think>` text leaks into the answer | server started without `--jinja` / reasoning format | use `2_start_llama.sh` (it passes `--jinja`; the server extracts thinking into `reasoning_content`) |
| Turns feel slow | that's the think budget being spent first | lower the budget or use the default entry for routine work |
