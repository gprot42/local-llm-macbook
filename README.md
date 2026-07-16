# Local LLM on MacBook

Apple Silicon stacks for **coding agents** (Kilo Code), chat, and experiments. OpenAI-compatible APIs; mostly MLX-native (`mlx_lm`, `vllm-mlx`, `mtplx`). Native Metal **ds4** for DeepSeek V4 Flash GGUF. Two stacks use Ollama + GGUF (Ornith, GLM).

Status: 🟢 working · 🟡 partial / flaky (e.g. Kilo filters some prompts)

## Model ↔ folder (select this in Kilo)

After `./1_setup_download.sh` + `./2_start_*.sh` in a folder, pick the matching **Kilo model ID** below (must match the running server).

### Censored (aligned)

| Folder | Select this model (Kilo) | Role | Modalities | API |
|--------|--------------------------|------|------------|-----|
| [`censored/qwen3-6-27b-coder-mtplx/`](censored/qwen3-6-27b-coder-mtplx/) | `mtplx/qwen3.6-27b-mtplx` | 🟢 **Default coding** — Qwen 3.6 27B (mtplx MTP) | Text | `:8765/v1` |
| [`censored/deepseek-v4-flash-ds4/`](censored/deepseek-v4-flash-ds4/) | `ds4/deepseek-v4-flash` | 🟢 **Great for coding** (128 GB, native Metal) | Text | `:8083/v1` |
| [`censored/deepseek-v4-flash-2bit-dq-mlx/`](censored/deepseek-v4-flash-2bit-dq-mlx/) | `deepseek-mlx/deepseek-v4-flash-2bit-dq` | 🟡 **Heavy coding** (128 GB, MLX) | Text | `:8082/v1` |
| [`censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/`](censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/) | `openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit` | 🟢 Stock Gemma 4 31B IT (AtomicChat 2026-07-15) | Text | `:8080/v1` |
| [`censored/diffusiongemma4-26b-a4b-mlx/`](censored/diffusiongemma4-26b-a4b-mlx/) | `diffusiongemma/diffusiongemma-26b-a4b-it-bf16` | 🟢 Multimodal / research | **Text + image** | `:8080/v1` |
| [`censored/ornith-1.0-35b-q8-gguf-ollama/`](censored/ornith-1.0-35b-q8-gguf-ollama/) | `ornith/ornith-1.0-35b-q8` | 🟢 Guided agent trials | Text | `:18082/v1` |

### Uncensored

| Folder | Select this model (Kilo) | Role | Modalities | API |
|--------|--------------------------|------|------------|-----|
| [`uncensored/gemma4-jang-crack-31b-mlx/`](uncensored/gemma4-jang-crack-31b-mlx/) | `openai-compatible/gemma-4-31b-jang-crack-mlx` | 🟡 Uncensored chat (JANG) | **Text + image** | `:8080/v1` |
| [`uncensored/gemma4-server-heretic-31b-mlx/`](uncensored/gemma4-server-heretic-31b-mlx/) | `openai-compatible/gemma-4-31b-heretic-mlx-4bit` | 🟢 Uncensored chat (Heretic) | **Text + image** | `:8080/v1` |
| [`uncensored/qwen3-32b-heretic-mlx/`](uncensored/qwen3-32b-heretic-mlx/) | `qwen3-heretic/qwen3-32b-heretic-mlx-5bit` | 🟢 Uncensored dense 32B (MLX; not 3.6/3.7) | Text | `:8084/v1` |
| [`uncensored/qwen3.5-122b-a10b-abliterated-mlx/`](uncensored/qwen3.5-122b-a10b-abliterated-mlx/) | `qwen35-122b-abliterated/qwen3.5-122b-a10b-abliterated-mlx-4bit` | 🟢 Uncensored MoE 122B (MLX) | Text | `:8085/v1` |
| [`uncensored/qwen3.5-122b-a10b-dflash-mlx/`](uncensored/qwen3.5-122b-a10b-dflash-mlx/) | `qwen35-122b-dflash/qwen3.5-122b-a10b-dflash` | 🟡 Uncensored 122B + **DFlash** (fast decode; Kilo agents flaky) | Text | `:8086/v1` |
| [`uncensored/glm-4.7-flash-heretic-gguf-ollama/`](uncensored/glm-4.7-flash-heretic-gguf-ollama/) | `glm/glm-4.7-flash-heretic-q8` | 🟢 Uncensored MoE (Ollama) | Text | `:18083/v1` |

**Ports:** `8080` is shared (Gemma / Diffusion) — one of those at a time. DeepSeek ds4 (`8083`), DeepSeek MLX (`8082`), Qwen3-32B Heretic (`8084`), Qwen3.5-122B Abliterated (`8085`), Qwen3.5-122B DFlash (`8086`), Qwen 3.6 mtplx (`8765`), Ornith (`18082`), and GLM (`18083`) can run together — but do **not** load multiple huge models at once on 128 GB.

---

## Quick start

```bash
mkdir -p ~/.config/kilo
cp kilo.json ~/.config/kilo/kilo.jsonc   # once; providers for all stacks

# Snappy coding (default recommendation)
cd censored/qwen3-6-27b-coder-mtplx
./1_setup_download.sh && ./2_start_mtplx.sh     # once ~18 GB; then leave server running
# In Kilo: model mtplx/qwen3.6-27b-mtplx

# Great for coding (native Metal DeepSeek — ~81 GB q2)
# cd ../deepseek-v4-flash-ds4 && ./1_setup_download.sh && ./2_start_ds4.sh
# In Kilo: model ds4/deepseek-v4-flash  (temp=1.0 top_p=1.0)
```

Other stacks: `1_*` setup/download → `2_*` start → select the model ID for that folder (table above). Details in each directory’s README.

Root [`kilo.json`](kilo.json) currently defaults to **`openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit`** (aligned stock Gemma text). Install globally with `cp kilo.json ~/.config/kilo/kilo.jsonc`. Switch `"model"` to any ID from the table above, or copy a stack’s local `kilo.json`.

---

## When to use which

### Censored (aligned)

| Goal | Use | Avoid when |
|------|-----|------------|
| Snappy Kilo tool loops | 🟢 **Qwen 3.6** | Uncensored needs; frontier-level greenfield apps |
| Great for coding — hard multi-file / SWE-style (128 GB, native Metal) | 🟢 **DeepSeek V4 Flash (ds4)** | You need snappy loops; RAM ≪ 128 GB |
| Hard multi-file via MLX | 🟡 **DeepSeek V4 Flash MLX** | Prefer ds4 for official GGUF path; RAM ≪ 128 GB |
| Aligned Gemma 31B | 🟢 **Gemma stock IT** | Same limits as Heretic for heavy agents |
| Diffusion / vision experiments | 🟢 **DiffusionGemma** | Coding or reliable tool use |
| Guided Ollama agent trials | 🟢 **Ornith** | Fast iteration; unattended large tasks |
| Uncensored large MoE (Qwen3.5 122B abliterated) | 🟢 **Qwen3.5-122B-A10B Abliterated** | RAM ≪ 128 GB; prefer Qwen3.6 27B mtplx for snappy coding |

### Uncensored

| Goal | Use | Avoid when |
|------|-----|------------|
| Uncensored / low-refusal chat (Gemma) | 🟡 **Gemma JANG_4M CRACK** | **Kilo Code may filter** — some questions still get blocked; not of much use as a Kilo agent |
| Uncensored / uniform 4-bit Gemma | 🟢 **Gemma Heretic** | Want native multimodal without vision graft |
| Uncensored dense 32B (MLX, **Qwen3** not 3.6) | 🟢 **Qwen3-32B Heretic** | Need Qwen3.6; prefer snappy `mtplx/qwen3.6-27b-mtplx` for coding |
| Uncensored 122B with **fast decode** (DFlash) | 🟡 **Qwen3.5-122B DFlash** | Long Kilo agent loops (context/prefill thrash); co-load another 70+ GB model; need vision; prefer Qwen3.6 mtplx for snappy tools |
| Uncensored MoE coding (Ollama) | 🟢 **GLM-4.7 Flash Heretic** | Need vision; prefer MLX Gemma for chat UI polish |

**RAM:** ~128 GB → all stacks. ~80 GB → skip full DeepSeek 2bit-DQ / prefer ds4 q2. ≤64 GB → Qwen and smaller only. Don’t load two huge models at once.

**Latency:** speculative decode helps generation, not prefill. Long Kilo histories still cost a large first token — compact or restart when context balloons ([Qwen README](censored/qwen3-6-27b-coder-mtplx/README.md)).

**DeepSeek sampling:** official defaults are `temperature=1.0`, `top_p=1.0` (see stack `kilo.json`). Qwen coding uses `0.6 / 0.95 / top_k 20`.

---

## Kilo Code

Config order: `.kilo/kilo.jsonc` → project `kilo.json` → `~/.config/kilo/kilo.jsonc`.

**Censored (aligned)** — folder → model to select

| | Folder | Base URL | Select model ID |
|---|--------|----------|-----------------|
| 🟢 | [`censored/qwen3-6-27b-coder-mtplx/`](censored/qwen3-6-27b-coder-mtplx/) | `http://localhost:8765/v1` | `mtplx/qwen3.6-27b-mtplx` |
| 🟢 | [`censored/deepseek-v4-flash-ds4/`](censored/deepseek-v4-flash-ds4/) | `http://127.0.0.1:8083/v1` | `ds4/deepseek-v4-flash` |
| 🟡 | [`censored/deepseek-v4-flash-2bit-dq-mlx/`](censored/deepseek-v4-flash-2bit-dq-mlx/) | `http://127.0.0.1:8082/v1` | `deepseek-mlx/deepseek-v4-flash-2bit-dq` |
| 🟢 | [`censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/`](censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/) | `http://localhost:8080/v1` | `openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit` |
| 🟢 | [`censored/diffusiongemma4-26b-a4b-mlx/`](censored/diffusiongemma4-26b-a4b-mlx/) | `http://localhost:8080/v1` | `diffusiongemma/diffusiongemma-26b-a4b-it-bf16` |
| 🟢 | [`censored/ornith-1.0-35b-q8-gguf-ollama/`](censored/ornith-1.0-35b-q8-gguf-ollama/) | `http://127.0.0.1:18082/v1` | `ornith/ornith-1.0-35b-q8` |

**Uncensored** — folder → model to select

| | Folder | Base URL | Select model ID |
|---|--------|----------|-----------------|
| 🟡 | [`uncensored/gemma4-jang-crack-31b-mlx/`](uncensored/gemma4-jang-crack-31b-mlx/) | `http://localhost:8080/v1` | `openai-compatible/gemma-4-31b-jang-crack-mlx` |
| 🟢 | [`uncensored/gemma4-server-heretic-31b-mlx/`](uncensored/gemma4-server-heretic-31b-mlx/) | `http://localhost:8080/v1` | `openai-compatible/gemma-4-31b-heretic-mlx-4bit` |
| 🟢 | [`uncensored/qwen3-32b-heretic-mlx/`](uncensored/qwen3-32b-heretic-mlx/) | `http://127.0.0.1:8084/v1` | `qwen3-heretic/qwen3-32b-heretic-mlx-5bit` |
| 🟢 | [`uncensored/qwen3.5-122b-a10b-abliterated-mlx/`](uncensored/qwen3.5-122b-a10b-abliterated-mlx/) | `http://127.0.0.1:8085/v1` | `qwen35-122b-abliterated/qwen3.5-122b-a10b-abliterated-mlx-4bit` |
| 🟡 | [`uncensored/qwen3.5-122b-a10b-dflash-mlx/`](uncensored/qwen3.5-122b-a10b-dflash-mlx/) | `http://127.0.0.1:8086/v1` | `qwen35-122b-dflash/qwen3.5-122b-a10b-dflash` |
| 🟢 | [`uncensored/glm-4.7-flash-heretic-gguf-ollama/`](uncensored/glm-4.7-flash-heretic-gguf-ollama/) | `http://127.0.0.1:18083/v1` | `glm/glm-4.7-flash-heretic-q8` |

Use **`127.0.0.1`** (not `localhost`) for ds4 / deepseek-mlx / Ollama stacks — macOS may resolve `localhost` to `::1`.

**Image attach (Kilo 7.3.x):** paperclip often missing — use Cmd+V, Shift+drag, or `@` → Attach. Need a vision model + its server. See [diffusion README](censored/diffusiongemma4-26b-a4b-mlx/README-diffusiongemma4.md).

---

## More docs

- **[README-models.md](README-models.md)** — **text-only vs multimodal** matrix, good / not-good use cases, DFlash notes

**Censored (aligned)**

- [qwen3-6-27b-coder-mtplx/README.md](censored/qwen3-6-27b-coder-mtplx/README.md) — mtplx MTP, latency vs context  
- [deepseek-v4-flash-ds4/README.md](censored/deepseek-v4-flash-ds4/README.md) — V4 Flash via antirez/ds4 (native Metal, resumable ~81 GB download)  
- [deepseek-v4-flash-2bit-dq-mlx/README.md](censored/deepseek-v4-flash-2bit-dq-mlx/README.md) — V4 Flash / community mlx-lm  
- [diffusiongemma4 README](censored/diffusiongemma4-26b-a4b-mlx/README-diffusiongemma4.md) · [ornith README](censored/ornith-1.0-35b-q8-gguf-ollama/README.md)  
- [gemma4 stock IT AtomicChat 2026-07-15](censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/README.md)  


**Uncensored**

- [gemma4-jang-crack-31b-mlx/README.md](uncensored/gemma4-jang-crack-31b-mlx/README.md) — JANG_4M CRACK (why over Heretic, no re-quantize needed)  
- [gemma4-server-heretic-31b-mlx/README.md](uncensored/gemma4-server-heretic-31b-mlx/README.md) — Heretic + proxy · [Continue.dev](uncensored/gemma4-server-heretic-31b-mlx/README.md#continuedev)  
- [qwen3-32b-heretic-mlx/README.md](uncensored/qwen3-32b-heretic-mlx/README.md) — Qwen3-32B Heretic (original Qwen3 dense, not 3.6/3.7)  
- [qwen3.5-122b-a10b-abliterated-mlx/README.md](uncensored/qwen3.5-122b-a10b-abliterated-mlx/README.md) — Qwen3.5-122B-A10B Abliterated MLX 4-bit (~70 GB; 128 GB recommended)  
- [qwen3.5-122b-a10b-dflash-mlx/README.md](uncensored/qwen3.5-122b-a10b-dflash-mlx/README.md) — same target + z-lab DFlash draft (fast decode; exactness verified)  
- [GLM Heretic README](uncensored/glm-4.7-flash-heretic-gguf-ollama/README.md)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to API` | Start that provider’s server; check with `curl` on the stack’s port |
| Wrong model / blank Base URL | Match `kilo.json` model ID + `baseURL` to the running server |
| OOM | Prefer Qwen 27B; DeepSeek wants ~128 GB — don’t run ds4 + MLX DeepSeek together |
| Port in use | `./2_start_*.sh status` / `restart` / `stop`, or change `--port` |
| Kilo hangs / bad tool markup (Gemma) | Use JANG/Heretic start **with** proxy + gemma4 parsers; don’t enable Kilo “reasoning” unless the server is started for thinking |
| DeepSeek slow first token | Expected (load/prefill of ~81–97 GB) — not a snappy tool-loop model |
| ds4 download `Connection reset by peer` | Normal on long HF transfers — re-run `./1_setup_download.sh`; `download_gguf.py` resumes `.part` automatically |
| ds4 “incomplete download” / refuses to start | Finish weights first: `python3 validate_model.py ds4/gguf/*.gguf` then re-run setup |
| Ornith empty replies | Use the tool proxy (default); thinking can burn `max_tokens` |
| Vision only in CLI, not Kilo | Use the **server** path + vision model ID |
| Qwen fast in curl, slow in Kilo | Prefill/context — compact the session ([Qwen README](censored/qwen3-6-27b-coder-mtplx/README.md)) |
