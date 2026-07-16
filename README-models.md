# Model guide — good vs not-good use cases

When to pick which local stack on Apple Silicon. Pair with the port/Kilo table in [README.md](README.md).

**Legend:** 🟢 working · 🟡 partial / flaky · **RAM** assumes unified memory on M-series.

---

## Modalities (text-only vs multimodal)

**What the stack accepts as input**, not the engine name. `mlx_vlm` / MTP does **not** mean vision: AtomicChat can use `mlx_vlm.server` only for optional **text MTP** (speculative decode).

| Stack | Folder | Input | Notes |
|-------|--------|-------|-------|
| **Qwen 3.6 27B mtplx** | `censored/qwen3-6-27b-coder-mtplx/` | **Text only** | Coding default; native MTP heads |
| **DeepSeek V4 Flash ds4** | `censored/deepseek-v4-flash-ds4/` | **Text only** | Native Metal GGUF |
| **DeepSeek V4 Flash MLX** | `censored/deepseek-v4-flash-2bit-dq-mlx/` | **Text only** | mlx-lm community path |
| **Gemma 4 31B AtomicChat** | `censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/` | **Text only** | Language quant + chat template; not a vision package |
| **DiffusionGemma 26B** | `censored/diffusiongemma4-26b-a4b-mlx/` | **Text + image** | Discrete diffusion VLM; research / image Q&A |
| **Ornith 1.0 35B** | `censored/ornith-1.0-35b-q8-gguf-ollama/` | **Text only** | Ollama GGUF |
| **Gemma 4 31B Heretic** | `uncensored/gemma4-server-heretic-31b-mlx/` | **Text + image** | Language is Heretic; vision **grafted** from stock IT (or text-only with `--skip-vision`) |
| **Gemma 4 31B JANG CRACK** | `uncensored/gemma4-jang-crack-31b-mlx/` | **Text + image** | Vision **native** in checkpoint (no graft step) |
| **Qwen3-32B Heretic** | `uncensored/qwen3-32b-heretic-mlx/` | **Text only** | Dense Qwen3, not 3.6 |
| **Qwen3.5-122B Abliterated** | `uncensored/qwen3.5-122b-a10b-abliterated-mlx/` | **Text only** | Large MoE AR path |
| **Qwen3.5-122B DFlash** | `uncensored/qwen3.5-122b-a10b-dflash-mlx/` | **Text only** | Same target + draft; text OpenAI server |
| **GLM-4.7 Flash Heretic** | `uncensored/glm-4.7-flash-heretic-gguf-ollama/` | **Text only** | Ollama GGUF |

### Related weights (not always a full stack)

| Weight / role | Modality | Purpose |
|---------------|----------|---------|
| `AtomicChat/gemma-4-31B-it-MLX-4bit` | **Text only** | Aligned Gemma target for this AtomicChat folder |
| `mlx-community/gemma-4-31b-it-4bit` | **Text + image** | Stock multimodal quant; **vision graft source** for Heretic (not the AtomicChat text stack) |
| `mlx-community/gemma-4-31B-it-assistant-bf16` | **Text draft only** | MTP speculative drafter (~1 GB); not a chat model; optional / skip for reliability |
| `z-lab/Qwen3.5-122B-A10B-DFlash` | **Text draft only** | DFlash block-diffusion draft; not standalone chat |

### Pick by modality

| You need… | Use |
|-----------|-----|
| Text coding / agents | Qwen 3.6, DeepSeek, AtomicChat, Ornith, Qwen3/3.5/GLM stacks |
| Images in Kilo (attach / paste) | **Heretic** (grafted), **JANG CRACK** (native), or **DiffusionGemma** (vision-first research) |
| Aligned Gemma text only | **AtomicChat** — do not expect image understanding |
| Vision weights for grafting | `mlx-community/gemma-4-31b-it-4bit`, not AtomicChat |

Kilo image attach needs a **vision** stack + its server running. See [README.md](README.md) (Image attach) and [DiffusionGemma README](censored/diffusiongemma4-26b-a4b-mlx/README-diffusiongemma4.md).

---

## Quick chooser

| You want… | Prefer | Avoid |
|-----------|--------|-------|
| Snappy Kilo tool loops | **Qwen 3.6 27B mtplx** (`:8765`) | DeepSeek / Qwen3.5-122B for “feel fast” |
| Hard multi-file coding quality | **DeepSeek V4 Flash ds4** (`:8083`) | ≤64 GB machines; co-loading another 70+ GB model |
| Uncensored chat (Gemma + vision) | **Gemma Heretic** or **JANG CRACK** (`:8080`) | Unattended huge agent refactors |
| Uncensored dense Qwen | **Qwen3-32B Heretic** (`:8084`) | Expecting Qwen3.6 / MTP speed |
| Uncensored big MoE (quality) | **Qwen3.5-122B Abliterated** (`:8085`) | RAM ≪ 128 GB; snappy loops |
| Uncensored 122B **fast decode** (🟡 flaky agents) | **Qwen3.5-122B DFlash** (`:8086`) | Long Kilo agent loops; prefill-heavy short turns; vision |
| Uncensored MoE coding (lighter) | **GLM-4.7 Flash Heretic** (`:18083`) | Vision; max coding quality vs DeepSeek |
| Multimodal / diffusion research | **DiffusionGemma** (`:8080`) | Agentic coding as primary job |
| Guided agent experiments | **Ornith** (`:18082`) | Fast iteration; unattended large tasks |
| Aligned stock Gemma | **Gemma 4 31B IT** (`:8080`) | Heavy uncensored needs |

**Don’t load two huge models at once on 128 GB** (DeepSeek + Qwen3.5-122B, two DeepSeeks, etc.). Port `8080` is shared among Gemma/Diffusion stacks — one at a time.

---

## Censored (aligned)

### Qwen 3.6 27B — mtplx (`censored/qwen3-6-27b-coder-mtplx/`)

| | |
|--|--|
| **Role** | 🟢 **Default coding** — snappy agent loops |
| **Modality** | **Text only** |
| **Engine / size** | mtplx + built-in MTP heads · ~18 GB MLX 4-bit |
| **API** | `:8765/v1` · Kilo: `mtplx/qwen3.6-27b-mtplx` |

**Good for**

- Day-to-day Kilo tool loops (`read`, edit, short turns)
- Fast decode via native MTP (no second draft model)
- Machines that can’t spare 70–100 GB for frontier MoE

**Not good for**

- Uncensored / low-refusal chat
- Frontier-level greenfield apps where DeepSeek quality wins
- Very long histories without compacting (prefill dominates; MTP helps decode, not prefill)

---

### DeepSeek V4 Flash — ds4 (`censored/deepseek-v4-flash-ds4/`)

| | |
|--|--|
| **Role** | 🟢 **Great for coding** — quality over latency |
| **Modality** | **Text only** |
| **Engine / size** | Native Metal `ds4` · ~81 GB q2-imatrix GGUF |
| **API** | `:8083/v1` (proxy; server on `:18083`) · Kilo: `ds4/deepseek-v4-flash` |

**Good for**

- Hard multi-file / SWE-style agent work
- Long context, real tool calling (use **`./2_start_ds4.sh`** so the kilo proxy is up — thinking OFF by default)
- Best DeepSeek path on Mac in this repo

**Not good for**

- Ultra-snappy tool loops (prefer Qwen 3.6 mtplx)
- RAM well under ~128 GB
- Co-loading with MLX DeepSeek or Qwen3.5-122B

**Sampling:** `temperature=1.0`, `top_p=1.0` (official defaults).

---

### DeepSeek V4 Flash — MLX 2bit-DQ (`censored/deepseek-v4-flash-2bit-dq-mlx/`)

| | |
|--|--|
| **Role** | 🟡 Heavy coding via MLX |
| **Modality** | **Text only** |
| **Engine / size** | `mlx_lm` (community V4 fork) · ~97 GB |
| **API** | `:8082/v1` · Kilo: `deepseek-mlx/deepseek-v4-flash-2bit-dq` |

**Good for**

- Same class of hard coding/reasoning as ds4 when you want the MLX stack
- Experimenting with mlx-lm harness / patches

**Not good for**

- Prefer **ds4** for the more polished Metal + proxy path
- Snappy iteration; machines under ~128 GB
- Treating 2-bit MoE as cloud-frontier on huge greenfield apps

---

### Gemma 4 31B IT stock AtomicChat (`censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/`)

| | |
|--|--|
| **Role** | 🟢 Aligned Gemma 31B (2026-07-15 chat-template / tool-calling fix) |
| **Modality** | **Text only** (language quant; not multimodal) |
| **Engine / size** | mlx-lm (default) / optional mlx-vlm + text MTP · AtomicChat 4-bit (~17 GB) |
| **HF** | `AtomicChat/gemma-4-31B-it-MLX-4bit` |
| **API** | `:8080/v1` · Kilo: `openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit` |

**Good for**

- Aligned chat and light agents with updated tool-call / chat templates
- Optional **text** MTP experiments on Gemma (drafter is not vision)

**Not good for**

- Image / multimodal input (use Heretic, JANG, or DiffusionGemma)
- Uncensored needs (use Heretic / JANG)
- Primary coding agent vs Qwen 3.6 / DeepSeek
- Vision graft source (use `mlx-community/gemma-4-31b-it-4bit` for multimodal shards)

---

### DiffusionGemma 26B A4B (`censored/diffusiongemma4-26b-a4b-mlx/`)

| | |
|--|--|
| **Role** | 🟢 Multimodal / research |
| **Modality** | **Text + image** (diffusion VLM) |
| **Engine / size** | mlx-vlm diffusion · ~52 GB bf16 |
| **API** | `:8080/v1` · Kilo: `diffusiongemma/diffusiongemma-26b-a4b-it-bf16` |

**Good for**

- Image + text Q&A, description, diffusion-LM experiments
- Researching non-autoregressive generation

**Not good for**

- Reliable agentic coding (often narrates instead of tool-calling)
- Production Kilo multi-file refactors

---

### Ornith 1.0 35B (`censored/ornith-1.0-35b-q8-gguf-ollama/`)

| | |
|--|--|
| **Role** | 🟢 Guided agent trials |
| **Modality** | **Text only** |
| **Engine / size** | Ollama GGUF Q8 · ~37 GB |
| **API** | `:18082/v1` · Kilo: `ornith/ornith-1.0-35b-q8` |

**Good for**

- Trying guided / thinking-style agent behavior on Ollama
- Side-by-side comparison with MLX stacks

**Not good for**

- Fast iteration (slow first load; thinking can burn `max_tokens` without the tool proxy)
- Unattended large coding tasks vs Qwen 3.6 / DeepSeek

---

## Uncensored

### Gemma 4 31B Heretic (`uncensored/gemma4-server-heretic-31b-mlx/`)

| | |
|--|--|
| **Role** | 🟢 Uncensored chat (uniform 4-bit + vision graft) |
| **Modality** | **Text + image** (after graft; text-only if `--skip-vision`) |
| **Engine / size** | vllm-mlx + Kilo proxy · ~20 GB |
| **API** | `:8080/v1` · Kilo: `openai-compatible/gemma-4-31b-heretic-mlx-4bit` |

**Good for**

- Low-refusal chat with a battle-tested proxy + gemma4 parsers
- Text + image (vision grafted from stock IT)

**Not good for**

- Max refusal-bench compliance (JANG may score higher)
- Unattended huge multi-file agent builds vs Qwen/DeepSeek/GLM

---

### Gemma 4 31B JANG CRACK (`uncensored/gemma4-jang-crack-31b-mlx/`)

| | |
|--|--|
| **Role** | 🟡 Strongest low-refusal Gemma path |
| **Modality** | **Text + image** (native in checkpoint) |
| **Engine / size** | vllm-mlx · mixed quant, native multimodal |
| **API** | `:8080/v1` · Kilo: `openai-compatible/gemma-4-31b-jang-crack-mlx` |

**Good for**

- Maximum uncensored compliance among Gemma stacks here
- Multimodal out of the box without a separate vision graft step

**Not good for**

- **Kilo may still filter** some prompts client-side — limited as a pure Kilo agent
- Primary coding default (prefer Qwen 3.6 / GLM / DeepSeek)

---

### Qwen3-32B Heretic (`uncensored/qwen3-32b-heretic-mlx/`)

| | |
|--|--|
| **Role** | 🟢 Uncensored dense **Qwen3** (not 3.6/3.7) |
| **Modality** | **Text only** |
| **Engine / size** | mlx_lm · ~22.5 GB 5-bit |
| **API** | `:8084/v1` · Kilo: `qwen3-heretic/qwen3-32b-heretic-mlx-5bit` |

**Good for**

- Uncensored general chat / coding on original Qwen3 dense 32B
- Fits 64 GB+; comfortable on 80–128 GB with long sessions

**Not good for**

- Expecting Qwen3.6 features / MTP snappy coding default
- Vision (text-only stack)

---

### Qwen3.5-122B-A10B Abliterated (`uncensored/qwen3.5-122b-a10b-abliterated-mlx/`)

| | |
|--|--|
| **Role** | 🟢 Uncensored large MoE (~10B active) |
| **Modality** | **Text only** |
| **Engine / size** | mlx_lm · ~70 GB 4-bit |
| **API** | `:8085/v1` · Kilo: `qwen35-122b-abliterated/qwen3.5-122b-a10b-abliterated-mlx-4bit` |

**Good for**

- High-capacity uncensored MoE reasoning / coding when you have **128 GB**
- Stronger “big model” feel than 27–35B stacks while staying local

**Not good for**

- Snappy tool loops (prefer Qwen 3.6 mtplx)
- RAM ≪ 128 GB; co-loading DeepSeek or another 70+ GB model
- Peak decode speed (use the **DFlash** stack below when you want that)

---

### Qwen3.5-122B-A10B + DFlash (`uncensored/qwen3.5-122b-a10b-dflash-mlx/`)

| | |
|--|--|
| **Role** | 🟡 Uncensored 122B with **block-diffusion speculative decode** (Kilo agents flaky) |
| **Modality** | **Text only** (draft is text speculative, not vision) |
| **Engine / size** | dflash-mlx · same ~65–70 GB target + ~1.5 GB draft |
| **API** | `:8086/v1` · Kilo: `qwen35-122b-dflash/qwen3.5-122b-a10b-dflash` |

**Good for**

- Same quality as the abliterated target with **much faster decode** when accept length is high
- Verified exact greedy match vs plain target (lossless speculative decoding)
- Long generations where decode dominates wall time

**Not good for**

- Long unattended **Kilo agent** loops (context bloat / prefill thrash; often feels stuck between tool turns)
- Prefill-heavy / very short turns (DFlash does not speed prefill)
- Vision / rich multimodal (text-only OpenAI server)
- Co-loading another huge model; RAM ≪ 128 GB
- Expecting NVIDIA SGLang 3–4× marketing numbers on every workload

**Status:** 🟡 Stack works (API, tool-call conversion, decode speed) but agent reliability is partial — see [`AGENT_OPS.md`](uncensored/qwen3.5-122b-a10b-dflash-mlx/AGENT_OPS.md).

---

### GLM-4.7 Flash Heretic (`uncensored/glm-4.7-flash-heretic-gguf-ollama/`)

| | |
|--|--|
| **Role** | 🟢 Uncensored MoE coding (Ollama) |
| **Modality** | **Text only** |
| **Engine / size** | Ollama GGUF · ~18–32 GB (q4–q8) |
| **API** | `:18083/v1` · Kilo: `glm/glm-4.7-flash-heretic-q8` |

**Good for**

- Uncensored coding agents with a smaller footprint than Qwen3.5-122B
- Easy Ollama lifecycle; can sit beside Ornith on a different port

**Not good for**

- Vision / multimodal
- Absolute max coding quality vs DeepSeek V4 on hard SWE tasks

---

## DFlash (`z-lab/Qwen3.5-122B-A10B-DFlash`)

[HF: z-lab/Qwen3.5-122B-A10B-DFlash](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) is a **~0.8B / ~1.5 GB draft** for **block-diffusion speculative decoding** — not a standalone chat model.

### What it is

- Proposes multi-token blocks; the **target** verifies them (lossless if exact mode passes)
- Official demos: SGLang on multi-GPU NVIDIA
- **This repo:** working MLX path at [`uncensored/qwen3.5-122b-a10b-dflash-mlx/`](uncensored/qwen3.5-122b-a10b-dflash-mlx/) (patched dflash-mlx + abliterated MLX target)

### Paths

| Path | Status |
|------|--------|
| `mlx_lm.server` alone | **No** DFlash loop |
| SGLang + NVIDIA | Upstream supported |
| **This Mac stack (`:8086`)** | **Yes** — exact greedy smoke passed; high accept length on coding prompt |

### Bottom line

Use **`:8085`** for plain AR abliterated 122B; use **`:8086`** when you want DFlash decode acceleration with the same target quality.

---

## RAM cheat sheet

| Machine | Reasonable picks |
|---------|------------------|
| **~128 GB** | All stacks; never two huge models resident |
| **~80 GB** | Skip full DeepSeek 2bit-DQ; prefer ds4 q2 carefully or Qwen/Gemma/GLM |
| **≤64 GB** | Qwen 3.6, Qwen3-32B, Gemma, GLM, Ornith — not 122B / full DeepSeek |

---

## Latency notes (all stacks)

- Speculative decode (MTP, DFlash, draft models) speeds **generation**, not **prefill**.
- Long Kilo histories → large first-token latency; compact or restart when context balloons.
- DeepSeek first token is slow after load (~81–97 GB) — expected, not a misconfiguration.
