# Local LLM on MacBook

Apple Silicon stacks for **coding agents** (Kilo Code), chat, and experiments. OpenAI-compatible APIs; mostly MLX-native (`mlx_lm`, `vllm-mlx`, `mtplx`). Two stacks use Ollama + GGUF (Ornith, GLM).

Status: 🟢 working · 🟡 partial / flaky (e.g. Kilo filters some prompts)

### Censored (aligned)

| Role | Model | Directory | API |
|------|--------|-----------|-----|
| **Default coding** | 🟢 Qwen 3.6 27B (mtplx MTP) | [`qwen3-6-27b-coder-mtplx/`](qwen3-6-27b-coder-mtplx/) | `:8765/v1` |
| Stock Gemma | 🟢 Gemma 4 31B IT | [`gemma4-server-mlx-31b/`](gemma4-server-mlx-31b/) | `:8080/v1` |
| Multimodal / research | 🟢 DiffusionGemma 26B | [`diffusiongemma4-26b-a4b-mlx/`](diffusiongemma4-26b-a4b-mlx/) | `:8080/v1` |
| Guided agent trials | 🟢 Ornith 1.0 35B Q8 | [`ornith-1.0-35b-q8-gguf-ollama/`](ornith-1.0-35b-q8-gguf-ollama/) | `:18082/v1` |

### Uncensored

| Role | Model | Directory | API |
|------|--------|-----------|-----|
| Uncensored chat (JANG) | 🟡 Gemma 4 31B JANG_4M CRACK | [`uncensored/gemma4-jang-crack-31b-mlx/`](uncensored/gemma4-jang-crack-31b-mlx/) | `:8080/v1` |
| Uncensored chat (Heretic) | 🟢 Gemma 4 31B Heretic | [`uncensored/gemma4-server-heretic-31b-mlx/`](uncensored/gemma4-server-heretic-31b-mlx/) | `:8080/v1` |
| Uncensored MoE (Ollama) | 🟢 GLM-4.7-Flash Heretic | [`uncensored/glm-4.7-flash-heretic-gguf-ollama/`](uncensored/glm-4.7-flash-heretic-gguf-ollama/) | `:18083/v1` |

**Ports:** `8080` is shared (Gemma / Diffusion) — one of those at a time. Qwen, Ornith, and GLM can run together.

---

## Quick start

```bash
mkdir -p ~/.config/kilo
cp kilo.json ~/.config/kilo/kilo.jsonc   # once; default model = Qwen 3.6

cd qwen3-6-27b-coder-mtplx
./1_setup_download.sh && ./2_start_mtplx.sh     # once ~18 GB; then leave server running
kilo                                    # model: mtplx/qwen3.6-27b-mtplx
```

Other stacks: `1_*` setup/download → `2_*` start → pick that stack’s model in Kilo. Details in each directory’s README.

---

## When to use which

### Censored (aligned)

| Goal | Use | Avoid when |
|------|-----|------------|
| Snappy Kilo tool loops | 🟢 **Qwen 3.6** | Uncensored needs; frontier-level greenfield apps |
| Aligned Gemma 31B | 🟢 **Gemma stock IT** | Same limits as Heretic for heavy agents |
| Diffusion / vision experiments | 🟢 **DiffusionGemma** | Coding or reliable tool use |
| Guided Ollama agent trials | 🟢 **Ornith** | Fast iteration; unattended large tasks |

### Uncensored

| Goal | Use | Avoid when |
|------|-----|------------|
| Uncensored / low-refusal chat (Gemma) | 🟡 **Gemma JANG_4M CRACK** | **Kilo Code may filter** — some questions still get blocked; not of much use as a Kilo agent |
| Uncensored / uniform 4-bit Gemma | 🟢 **Gemma Heretic** | Want native multimodal without vision graft |
| Uncensored MoE coding (Ollama) | 🟢 **GLM-4.7 Flash Heretic** | Need vision; prefer MLX Gemma for chat UI polish |

**RAM:** ~128 GB → all stacks. ≤64 GB → Qwen and smaller only. Don’t load two huge models at once.

**Latency:** speculative decode helps generation, not prefill. Long Kilo histories still cost a large first token — compact or restart when context balloons ([Qwen README](qwen3-6-27b-coder-mtplx/README.md)).

---

## Kilo Code

Config order: `.kilo/kilo.jsonc` → project `kilo.json` → `~/.config/kilo/kilo.jsonc`. Root [`kilo.json`](kilo.json) defaults to **`mtplx/qwen3.6-27b-mtplx`**.

**Censored (aligned)**

| | Provider | Base URL | Model ID |
|---|----------|----------|----------|
| 🟢 | `mtplx` | `http://localhost:8765/v1` | `mtplx/qwen3.6-27b-mtplx` |
| 🟢 | `diffusiongemma` | `http://localhost:8080/v1` | `diffusiongemma/diffusiongemma-26b-a4b-it-bf16` |
| 🟢 | `ornith` | `http://127.0.0.1:18082/v1` | `ornith/ornith-1.0-35b-q8` |

**Uncensored**

| | Provider | Base URL | Model ID |
|---|----------|----------|----------|
| 🟡 | `openai-compatible` | `http://localhost:8080/v1` | `openai-compatible/gemma-4-31b-jang-crack-mlx` |
| 🟢 | `openai-compatible` | `http://localhost:8080/v1` | `openai-compatible/gemma-4-31b-heretic-mlx-4bit` |
| 🟢 | `glm` | `http://127.0.0.1:18083/v1` | `glm/glm-4.7-flash-heretic-q8` |

**Image attach (Kilo 7.3.x):** paperclip often missing — use Cmd+V, Shift+drag, or `@` → Attach. Need a vision model + its server. See [README-diffusiongemma4.md](README-diffusiongemma4.md).

---

## More docs

**Censored (aligned)**

- [qwen3-6-27b-coder-mtplx/README.md](qwen3-6-27b-coder-mtplx/README.md) — mtplx MTP, latency vs context  
- [README-diffusiongemma4.md](README-diffusiongemma4.md) · [ornith README](ornith-1.0-35b-q8-gguf-ollama/README.md)

**Uncensored**

- [gemma4-jang-crack-31b-mlx/README.md](uncensored/gemma4-jang-crack-31b-mlx/README.md) — JANG_4M CRACK (why over Heretic, no re-quantize needed)  
- [gemma4-server-heretic-31b-mlx/README.md](uncensored/gemma4-server-heretic-31b-mlx/README.md) — Heretic + proxy · [Continue.dev](uncensored/gemma4-server-heretic-31b-mlx/README.md#continuedev)  
- [GLM Heretic README](uncensored/glm-4.7-flash-heretic-gguf-ollama/README.md)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to API` | Start that provider’s server; check with `curl` |
| Wrong model / blank Base URL | Match `kilo.json` model ID + `baseURL` to the running server |
| OOM | Prefer Qwen 27B or smaller quantizations |
| Port in use | Stop the other stack or change `--port` |
| Kilo hangs / bad tool markup (Gemma) | Use JANG/Heretic start **with** proxy + gemma4 parsers; don’t enable Kilo “reasoning” unless the server is started for thinking |
| Ornith empty replies | Use the tool proxy (default); thinking can burn `max_tokens` |
| Vision only in CLI, not Kilo | Use the **server** path + vision model ID |
| Qwen fast in curl, slow in Kilo | Prefill/context — compact the session ([Qwen README](qwen3-6-27b-coder-mtplx/README.md)) |
