# Findings — Gemma 4 26B-A4B (abliterated) on TensorFold

**Date:** 2026-09-30 · **Machine:** Apple M5 Max, 128 GB · **Engine:** TensorFold 0.5.0 + our two `gemma4` fixes (PR [ashhart/TensorFold#157](https://github.com/ashhart/TensorFold/pull/157))
**Model:** local self-quantised `gemma-4-26b-a4b-abliterated-4bit` (26B-total / 4B-active MoE, 4-bit, 4.502 bits/weight)
**Drafter:** `z-lab/gemma-4-26B-A4B-it-DFlash` (external DFlash) · **Conditions:** only this model loaded (uncontended GPU)

## Status

Uncensored **and** agentic-capable: chat is clean/uncensored, and tool calling now works in OpenCode after our two lane fixes. Fast — ~170–190 tok/s on code generation.

## What we changed in *our* model / stack

The 4-bit weights are **local** (not on Hugging Face). They were produced from an abliterated bf16 checkpoint, and the stack installs a patched TensorFold:

1. **Quantised** `SevenOfNine/Gemma-4-26B-A4B-It-Abliterated` (bf16, Heretic-abliterated) → 4-bit/group-64 with `mlx_lm.convert` (mlx-lm has `gemma4`). Result: 4.502 bpw, ~13 GB, at `~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit`.
2. **Swapped in the base pack's *gated* chat template** (`mlx-community/gemma-4-26b-a4b-it-4bit`). The abliterated pack (and supergemma packs) ship a template that force-opens the reasoning channel, leaking `<|channel>thought`; the base template gates it. Weights are abliterated (uncensored); template is the clean base one.
3. **Stack installs TensorFold from our fork branch** `gprot42/TensorFold@fix-gemma-tool-call-colon-prefix` (`TF_REPO`/`TF_VERSION` in `1_setup_download.sh`) so the two tool/agentic fixes below are present. Revert to the official release once PR #157 is merged.

## Two upstream fixes we made (PR #157)

Both were `gemma4`-lane bugs reproduced on the **stock** `mlx-community/gemma-4-26b-a4b-it-4bit` too (not our abliteration):

- **Tool calls leaked as text** (`<|tool_call>:NAME{...}` → content, `tool_calls: null`). The parser only handled the `call:NAME` form; the model also emits `:NAME` (no `call`). Made the `call` keyword optional in the gate + regex. Fixes upstream #121.
- **Spontaneous thought channel leaked when thinking off** — on a continuation after a tool result the model opens `<|channel>thought…<channel|>`, which fell to the gpt-oss `parse_harmony_output` path and leaked. Now Gemma replies route through `split_thinking` (channel-aware) whether or not thinking is on.

Both verified live (stock + abliterated, streaming + non-streaming); regression tests added; the lane's tool/parse/stream test suite stays green.

## Timings (TensorFold 0.5.0 + fixes, uncontended)

### Code generation (drafted)

| Task | Tokens | tok/s |
|---|---|---|
| Thread-safe LRU cache class | 250 | **172.9** |
| nth-prime function | 200 | **186.3** |
| flatten nested list (greedy) | 180 | 183.3 |

**Drafting vs serial** (same greedy prompt): 183.3 tok/s drafted vs 141.5 serial → **~1.30× from DFlash**. (MoE with ~4B active → fast even serial.)

### Agentic tool calls (system prompt + 6 tools)

| Request | Output tok | tok/s | result |
|---|---|---|---|
| list directory | 12 | 85.7 | `list` ✅ |
| write file (FastAPI hello) | 18 | 108.2 | `write` ✅ |
| edit a route | 13 | 111.5 | `edit`→call ✅ |
| continue after tool result | 19 | 104.7 | `read` ✅ (no channel leak) |

Tool-call replies are short (just the call), so per-call latency is **<0.2 s** and the tok/s is overhead-dominated (~85–111) rather than the sustained code-gen rate. The point: **every call now parses into structured `tool_calls`** — including the multi-turn continuation — so agentic coding works end-to-end.

### Prefill

Cold prefill ~3,000 tok/s (a ~21.7k-token prompt in ~7 s); warm/short prompts TTFT ~0.05–0.3 s (prefix cache).

## Caveats

- Single-run, indicative numbers; expect run-to-run variance (code ~165–190, prose ~140). Draft acceptance is higher on code than prose.
- Weights are local; the *recipe* above is the source of truth (reproducible on another machine). The TensorFold fixes are pinned via the fork until #157 merges.

## Narration & build prompt (2026-09-30)

The abliterated weights are chatty on **open-ended / research** prompts — the model narrates ("let's do it", "let's go", "the first action is bash") before acting. On **concrete coding tasks** it already goes straight to tool calls (0 preamble), and temperature (1.0 / 0.6 / 0.3) doesn't change it — it's the ablated style, not a bug (the `<|channel>thought` leaks seen in older sessions are fixed, see above).

**Mitigation:** tightened the OpenCode **build-agent prompt** to hard-suppress narration — a "No narration" paragraph that bans plan/preamble, tool-call announcements ("the first action is bash", "I'll now edit…", "let me…") and filler openers ("let's do it", "let's go", "sure", "great", "okay"); the only prose allowed is one final result line. Applied to the live `~/.config/opencode/opencode.json` and to the repo source `censored/ternary-bonsai-2-27b-tensorfold/opencode.json` (the generic build prompt the live config is installed from). Takes effect on **OpenCode restart** — it's a client-side prompt, so no TensorFold server restart is needed.

**Caveat:** this reduces narration, it does not eliminate it on open-ended asks — the tendency is in the weights, not the prompt.
