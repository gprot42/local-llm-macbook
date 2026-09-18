# Modifications — adding Ternary Bonsai 2 27B

What was changed to add and deploy **Ternary Bonsai 2 27B** (PrismML), announced by [Emad Mostaque](https://x.com/EMostaque/status/2100717168648184002) quoting PrismML. Deployed on port **8089** as `bonsai/ternary-bonsai-2-27b`.

## Key decision: why GGUF + a llama.cpp fork (not MLX)

Bonsai 2 packs are stored in a **rotated basis** (blockwise Hadamard rotation folded into the weights); the runtime must apply the matching transform to activations. The MLX pack declares `model_type: prism_hadamard_qwen35` / `requires_runtime: true`.

- **The MLX pack cannot be served as an OpenAI API.** Stock `mlx_vlm.server` / `mlx_lm.server` don't use the bundled loader — verified live: `Model type prism_hadamard_qwen35 not supported`. PrismML's own `start_mlx_server.sh` (in [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo), their stated source of truth) **refuses Bonsai 2**, warning that serving it through stock MLX "would return wrong output with no error." The MLX build is one-shot generation only.
- **Working path:** the **PQ2_0 (group-128) GGUF** served by **PrismML's llama.cpp fork** built from source with Metal. Stock llama.cpp / Ollama lack the PQ2_0 kernels. `llama-server --jinja` gives OpenAI tool calling; `--mmproj` gives vision.

An MLX stack was scaffolded first and abandoned once this was confirmed.

## New stack — `censored/ternary-bonsai-2-27b-gguf-llamacpp/`

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Clone PrismML's llama.cpp fork (branch `prism`, pinned tag `prism-b10683-d8f26ee`), build `llama-server` with Metal, download `PQ2_0` GGUF (7.21 GB) + `mmproj-Q8_0` (0.63 GB). |
| `2_start_llama.sh` | Serve on `:8089` — `llama-server -m <PQ2_0> --mmproj <mmproj> --alias ternary-bonsai-2-27b --jinja -ngl 999 -fa on -c 81920 -np 1 --cache-ram 24576 --reasoning off --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5`. `status` / `stop` subcommands; `--think` / `--think-budget N` switch to `--reasoning on --reasoning-preserve --reasoning-budget N` with the thinking preset (`1.0/0.95/20`). |
| `README.md` | Stack docs, incl. the stability-tuning table. |
| `kilo.json` | Per-stack Kilo config (provider `bonsai`, default model `bonsai/ternary-bonsai-2-27b`). |
| `opencode.json` + `install-opencode-json.sh` | OpenCode provider fragment + installer (merges into `~/.config/opencode/opencode.json`, sets `model`/`small_model`). Two model entries: `ternary-bonsai-2-27b` (default) and `ternary-bonsai-2-27b-think` (per-request thinking, 1024 budget). |

Git-ignored (built/downloaded locally): `engine/` (fork checkout + build), `models/` (~7.9 GB weights), `venv/`, `setup.out`, `.bonsai_llama.log`.

## Repo edits

- **`kilo.json`** (root, the live-config source): added the `bonsai` provider → `baseURL http://127.0.0.1:8089/v1`, model `ternary-bonsai-2-27b`, `tool_call: true`, `text + image`, `limit.context 49152 / output 8192`. Installed live via `install_kilo.sh`.
- **`sync_agent_prompts.py`**: registered the per-stack `kilo.json` so the shared prompt block stays in sync (`--check` green).
- **`README.md`**: models table + endpoints table rows; added `Ternary Bonsai 2 (8089)` to the ports line.
- **`README-models.md`**: catalog row.
- **`.gitignore`**: `**/ternary-bonsai-2-27b-gguf-llamacpp/{engine,models}/` + `setup.out`.

## Model facts

- Hadamard-rotated **ternary** (~1.72 bit/weight) quant of **Qwen3.8 27B**; ~9× smaller than FP16, ~98% of aggregate benchmark performance.
- **Text + image**, native tool calling (BFCL v3 ~74), **thinking model** (`xhigh` effort by default; served with reasoning off for agentic use, see below).
- Context 262K native. Served window `-c 81920`; Kilo/OpenCode `limit.context 49152` / `output 16384` (peak 65536 < 81920) — a coherent cap that compacts before overflow. Raise `BONSAI_CTX` + `limit.context` together for more.
- Throughput on M5 Max (this machine): decode ~46 tok/s short-context (PrismML spec 47.0 tg128), 36–40 at 5k depth; prefill ~690 tok/s cold at 5k (spec pp512 765). Prefill, not decode, is what OpenCode waits on: its baseline prompt is ~5.3k tokens bare (system + 9 tool schemas) and 14–16k with project instructions.

## Verification (live, on `:8089`)

- Text: `finish=stop`, `"bonsai ok"`.
- Tool calling (`--jinja`): `finish=tool_calls`, `get_weather({"city":"Paris"})`.
- Thinking: response returned `reasoning_content` separate from the answer.

## Stability pass from the PrismML docs (OpenCode)

Sources read: [prismml.com/news/bonsai-2-27b](https://prismml.com/news/bonsai-2-27b), [docs.prismml.com](https://docs.prismml.com/) (`run/server`, `run/llamacpp`, `resources/troubleshooting`, `bonsai-2-27b`), the [HF model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and the Bonsai-demo `scripts/common.sh`, `start_llama_server.sh`, `SPECULATIVE.md`. What they changed:

1. **Single slot (`-np 1`, `BONSAI_SLOTS`).** The fork's `llama-server` auto-selected `n_slots = 4, kv_unified = true`. The server log showed one OpenCode conversation hopping slots 1 → 3 → 2 with **full re-prefills of 13883 and 16170 tokens (42 s and 86 s at 187–330 tok/s)** — the "hang" at session start and after compaction. With one slot the window belongs to one conversation and cold prefill measures ~690 tok/s (the other slots' resident caches were also being scanned by attention).
2. **RAM prompt cache `--cache-ram 24576` (`BONSAI_CACHE_RAM`).** Log lines `making room for prompt cache entry, removing oldest entry (size = 2733 MiB)` showed the 8 GiB default evicting conversations: a hybrid-attention state is ~187 KiB/token including checkpoints, so a 49k conversation is ~9 GiB and could never be kept. Now OpenCode's concurrent title request (it goes to `small_model` on the same server) and subagents no longer cost a re-prefill — verified: conversation → title request → conversation re-processed **4 tokens**.
3. **Sampling presets per mode, server-side.** The model card lists two presets (also stored in the GGUF `general.sampling.*`): thinking `temp 1.0 / top-p 0.95 / top-k 20 / min-p 0 / presence 0`; non-thinking `temp 0.7 / top-p 0.8 / top-k 20 / min-p 0 / presence 1.5`. We were serving reasoning-off with the thinking preset. `2_start_llama.sh` now picks the preset from the mode (`BONSAI_PRESENCE` tunes the penalty).
4. **OpenCode `options` are passed verbatim.** Captured against a mock endpoint: `options: {temperature, topP, topK}` arrived as `temperature`, `topP`, `topK` — the camelCase keys are silently ignored by llama-server, so the old client overrides never applied; with no `options` OpenCode sends only `max_tokens`. The fragment now declares the model `"temperature": false` and no sampling options, so the server preset always wins (and the global `agent.build.temperature` set for GLM no longer leaks into Bonsai).
5. **Reasoning: tried on-with-budget, settled on off.** The card says the 27B defaults to `xhigh` effort (`low` is unsupported); PrismML's own answer to slow/exhausted replies is `--reasoning-budget N`. A budgeted default (`--reasoning on --reasoning-budget 2048 --reasoning-preserve`, ≈ their "Medium") was verified to work mechanically — a prompt engineered to think indefinitely was cut at the budget and still answered, tool calls emit after reasoning, and a tool-loop turn returning `reasoning_content` re-processed only 32 tokens — but a one-hour OpenCode game-building session under it showed the real cost: the context grew by the full generated count on every step (reasoning is sent back and kept for the whole tool loop), the session hit OpenCode's usable window (`49152 − 16384 = 32768`) with a post-compaction floor of ~25–28k (14.4k baseline prompt + ~10k summary), and ended in `Compaction exhausted: context still exceeds model limits after 3 attempts`; ~half of the 56k generated tokens were thinking, so it also ran ~2× slower. Default is therefore `--reasoning off` (the deprecated `--chat-template-kwargs enable_thinking` form dropped); `--think` / `--think-budget N` keep the budgeted mode available server-wide.
6. **Thinking on demand in OpenCode (`/models`).** OpenCode has no thinking toggle for a llama-server backend, but the reasoning-off server honors two per-request fields — `chat_template_kwargs: {"enable_thinking": true}` and PrismML's fork param `thinking_budget_tokens` (upstream `reasoning_budget` is server-wide only; without a budget the request ends `finish=length` with empty content). Since OpenCode passes model `options` verbatim, a second entry `bonsai/ternary-bonsai-2-27b-think` (`reasoning: true`, `thinking_budget_tokens: 1024`, thinking sampling preset in snake_case) makes thinking a `/models` choice. Verified: cut at ~1000 tokens with the answer following, `finish=tool_calls` with tools, default entry unaffected. The context-growth cost remains (sent-back reasoning is rendered even without `--reasoning-preserve`: 581 vs 40 prompt tokens), hence the smaller budget.

Evaluated and **not** adopted: speculative decoding (`SPECULATIVE.md`: on M5 Max only code/math gains ~1.2×, chat/reasoning is slower, "not recommended on Apple Silicon"; Bonsai 2's GGUF repo ships no drafter anyway); `BONSAI_KV4` / `q8_0` KV cache (memory-pressure tools — 81920 ctx is ~5 GiB here); larger micro-batches (benchmarked on a side port: default `-b 2048 -ub 512` 486–556 tok/s vs `-ub 1024` 461–471 vs `-ub 2048` 316–399).

## Fixes

- **Output exhausted while reasoning** ("produced no actionable output", e.g. the mario.html write never landed): the thinking model spent its whole budget reasoning. Fix: `--reasoning off` by default (originally paired with the since-deprecated `--chat-template-kwargs '{"enable_thinking": false}'`). A budgeted thinking mode (`--think`, `--reasoning-budget 2048`) exists and guarantees room for the tool call, but see item 5 above for why it is not the default.
- **Truncated `write` tool call** (large file, unterminated JSON): the client aborted the slow ~5-min generation at `chunkTimeout` (300s, ~6926 tokens). Raised `chunkTimeout 300000 -> 900000`, `timeout 900000 -> 1800000`, `limit.output 8192 -> 16384`, and the served window `-c 65536 -> 81920` (peak 49152+16384=65536 < 81920). Note: decode is ~36–46 tok/s (at PrismML's M5 Max spec), so a 10k-token one-shot file still takes ~4–5 min — GLM is faster for those.
- **Context overflow** (request 41206 > 32768): raised served window to `-c 65536` and set Kilo `limit.context 49152`/`output 8192` (was 32768). Server default `BONSAI_CTX` bumped to 65536.

## Run

```bash
cd censored/ternary-bonsai-2-27b-gguf-llamacpp
./1_setup_download.sh     # once — build fork + download weights (~5 min build, ~7.9 GB)
./2_start_llama.sh        # serve on :8089
# Kilo/OpenCode: provider `bonsai`, model `bonsai/ternary-bonsai-2-27b`, API http://127.0.0.1:8089/v1
```

## Not done

- Kilo config (`kilo.json`) was left as-is after the switch to OpenCode; it still points at the same server and inherits the server-side presets.
- OpenCode's per-session prefill cost (its ~5k-token tool/system prompt, more with project instructions) is inherent to the client; the server now handles it at ~690 tok/s and keeps it cached.
