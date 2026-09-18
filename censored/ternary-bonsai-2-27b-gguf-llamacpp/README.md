# Ternary Bonsai 2 27B (GGUF · llama.cpp fork)

Local **Ternary Bonsai 2 27B** ([prism-ml](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)) — a Hadamard‑rotated **ternary** (~1.72 bit/weight) quant of **Qwen3.8 27B**, ~9× smaller than FP16 while retaining ~98% of aggregate benchmark performance. Text **+ image**, native tool calling, thinking model. Served as an OpenAI API for **Kilo / OpenCode**.

| | |
|--|--|
| **API** | `http://127.0.0.1:8089/v1` |
| **Model IDs** | `bonsai/ternary-bonsai-2-27b` (default, no thinking) · `bonsai/ternary-bonsai-2-27b-think` (OpenCode, 1024‑token thinking budget) |
| **Weights** | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · `PQ2_0` (7.21 GB) + `mmproj-Q8_0` (0.63 GB) |
| **Engine** | PrismML's [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (Metal), `--jinja` tool calling, single slot + 24 GiB RAM prompt cache |

## Why the fork (not stock MLX/Ollama)

Bonsai 2 packs are stored in a **rotated basis**: each weight matrix is transformed by a blockwise Hadamard rotation before ternary assignment, and the runtime must apply the matching transform to activations. The pack declares `model_type: prism_hadamard_qwen35` / `requires_runtime: true`.

- **MLX packs cannot be served.** `mlx_lm.server` / `mlx_vlm.server` don't use the bundled loader; PrismML's own `start_mlx_server.sh` **refuses Bonsai 2**, warning it "would return wrong output with no error." (The MLX build is one‑shot generation only.)
- **The `PQ2_0` (group‑128) GGUF needs the fork's kernels** — stock llama.cpp / Ollama lack them.

So this stack builds PrismML's llama.cpp fork from source and serves the GGUF. Source of truth: [PrismML‑Eng/Bonsai‑demo](https://github.com/PrismML-Eng/Bonsai-demo).

## Quick start

```bash
cd censored/ternary-bonsai-2-27b-gguf-llamacpp

# 1. Build the fork (Metal) + download PQ2_0 GGUF + mmproj  (~5 min build, ~7.9 GB)
./1_setup_download.sh

# 2. Serve on :8089  (OpenAI-compatible, --jinja tool calling, image input, reasoning off)
./2_start_llama.sh
#    status / stop:  ./2_start_llama.sh status | ./2_start_llama.sh stop
#    thinking:       ./2_start_llama.sh --think   (2048-token budget; --think-budget N to change)

# 3. Kilo / OpenCode — provider `bonsai`, model `bonsai/ternary-bonsai-2-27b`, API http://127.0.0.1:8089/v1
```

## Notes

- **Context** defaults to `-c 81920` (override `BONSAI_CTX`). The model supports 262K; OpenCode/Kilo `limit.context` is capped at 49152 with `output` 16384 (peak 65536 < 81920) so the client compacts before the server overflows. Raise both together for more.
- **Sampling is owned by the server**, per mode, straight from the [PrismML model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#generation-parameters): non‑thinking (default) `temp 0.7 / top‑p 0.8 / top‑k 20 / min‑p 0 / presence 1.5`; thinking `temp 1.0 / top‑p 0.95 / top‑k 20 / min‑p 0 / presence 0`. The OpenCode model is declared `"temperature": false` so the client never overrides the preset (Qwen warns greedy decoding makes this family loop).
- **Reasoning is off by default** (`--reasoning off`, OpenCode model `"reasoning": false`) — for agentic use, not because the model can't think. Two measured reasons: unbounded, the 27B (`xhigh` effort) exhausts its output budget before the tool call; and even under a budget, OpenCode sends every step's `reasoning_content` back and the template keeps it for the whole tool loop, so context grows ~2k tokens per step — a one‑hour game‑building session ended in `Compaction exhausted: context still exceeds model limits after 3 attempts` and ran ~2× slower. **To think in OpenCode, switch models with `/models`** → `bonsai/ternary-bonsai-2-27b-think` (full guide: [THINKING.md](THINKING.md)): a second entry on the same server whose `options` turn thinking on per request (`chat_template_kwargs.enable_thinking`, PrismML's per‑request `thinking_budget_tokens: 1024`, thinking sampling preset) with `"reasoning": true` so the thinking renders. No restart needed. Verified: cut at ~1000 tokens with the answer/tool call following. Context still grows by the budget per step (sent‑back reasoning is rendered regardless of `--reasoning-preserve`), so use it for hard turns, not hour‑long builds. Server‑wide alternative: `./2_start_llama.sh --think` (2048 budget; `--think-budget N`, `-1` unlimited).
- Port **8089** so it runs beside the other stacks (see the root README ports table).
- `engine/`, `models/`, and `venv/` are git‑ignored (built/downloaded locally).

## Stability tuning (what the PrismML docs changed)

Measured on an M5 Max / 128 GB after reading [prismml.com/news/bonsai-2-27b](https://prismml.com/news/bonsai-2-27b), [docs.prismml.com](https://docs.prismml.com/run/server), the model card and the [Bonsai‑demo](https://github.com/PrismML-Eng/Bonsai-demo) scripts:

| Symptom in OpenCode | Cause | Fix in `2_start_llama.sh` |
|---|---|---|
| 40–90 s "hang" before the first token on a new session, then again after compaction | `llama-server` auto‑picked **4 slots sharing one unified KV**; the conversation hopped slots and re‑prefilled OpenCode's 14–16k‑token prompt at 187–330 tok/s | **`-np 1`** (`BONSAI_SLOTS`). One slot owns the window; cold prefill now runs at ~690 tok/s (PrismML's pp512 spec is 765) |
| Same re‑prefill after every title / subagent request | The 8 GiB default RAM prompt cache can't hold one full conversation (~187 KiB/token incl. hybrid‑attention checkpoints ⇒ ~9 GiB at 49k) | **`--cache-ram 24576`** (`BONSAI_CACHE_RAM`). Verified: title request in between ⇒ conversation restored, 4 tokens re‑processed |
| Output exhausted while reasoning | Thinking on at `xhigh` effort, unbounded | `--reasoning off` default; `--think` applies a 2048 budget |
| "Files aren't updated", an hour passes, the model keeps "outputting information" | **Doom loop**: the model started a dev server in the foreground with bash (10 s tool timeout kills it), `curl`ed it (dead), started it again — 148 server starts + 285 curls in ~90 min, no file written after the first 20 min. Alternating commands defeat OpenCode's repeat detector | Build‑agent prompt rules installed by `install-opencode-json.sh`: no foreground servers (`nohup … &` or don't serve static files), never a third retry of a failed command, stop when the files are written |
| `Compaction exhausted: context still exceeds model limits after 3 attempts` (thinking on) | Reasoning is retained in context for the whole tool loop (~2k/step); OpenCode's usable window is `49152 − 16384 = 32768` and the post‑compaction floor is ~25–28k (14k baseline prompt + ~10k summary), so 2–3 steps refill it | Reasoning off (default). Also helps: raise `limit.context`/`-c`, trim the project's instruction prompt |
| Client sampling never applied | OpenCode passes model `options` **verbatim** — `topP`/`topK` reached the server as unknown keys | Presets moved server‑side; OpenCode model `temperature: false` |

Decode speed is already at PrismML's spec (**~46 tok/s** short‑context, 36–40 at 5k depth, vs their 47.0 tg128 on M5 Max) — the slowness people feel is prefill. Not adopted, with reasons: **speculative decoding** (PrismML: net loss for chat/agent workloads on Apple Silicon, and no drafter ships for Bonsai 2); **KV4 / q8 KV** (only needed under memory pressure); **larger `-ub`** (benchmarked slower on Metal: 486–556 tok/s at the default vs 316–399 at `-ub 2048`).
