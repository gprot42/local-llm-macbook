# Gemma 4 31B JANG_4M CRACK — MLX Server

Uncensored **abliterated** Gemma 4 31B (dealignai CRACK v2) via **vllm-mlx** on Apple Silicon.

**Status: 🟡 partial / flaky for Kilo Code** — Kilo may still filter prompts; some questions get blocked, so this model is **not of much use** as a Kilo agent. Prefer 🟢 stacks (Qwen / GLM) for agent work. Fine for local API experiments where the filter is not in the path.

| | |
|---|---|
| **Status** | 🟡 (Kilo filtering) |
| **Project dir** | `gemma4-jang-crack-31b-mlx` |
| **Model dir** | `gemma-4-31b-jang-crack-mlx` |
| **HF repo** | [`dealignai/Gemma-4-31B-JANG_4M-CRACK`](https://huggingface.co/dealignai/Gemma-4-31B-JANG_4M-CRACK) |
| **Size** | ~22 GB (JANG mixed 4/8-bit, ~5.1 avg bits) |
| **RAM** | **64–80 GB+** unified memory recommended for agent sessions |
| **API** | `http://localhost:8080/v1` |
| **Kilo model ID** | `gemma-4-31b-jang-crack-mlx` |
| **Modalities** | Text + image (vision shipped in checkpoint; no stock graft) |

> Shares `:8080` with other Gemma stacks — only one at a time.

---

## Why this model over Heretic uncensored Gemma?

Sibling stack: [`../gemma4-server-heretic-31b-mlx/`](../gemma4-server-heretic-31b-mlx/)  
(`mlx-community/gemma-4-31B-it-uncensored-heretic-4bit`)

| | **JANG_4M CRACK (this)** | **Heretic 4-bit** |
|---|---|---|
| **Publisher** | [dealignai](https://huggingface.co/dealignai) | mlx-community / Heretic abliteration |
| **Abliteration** | CRACK v2 — architecture-aware, per-layer surgery on refusal-critical layers (o_proj / down_proj mid-stack), improved vectors + thinking stability | Classic Heretic / abliteration pass on Gemma 4 IT |
| **Reported refusal bench** | **93.7% HarmBench** compliance (300 prompts); 8/8 security prompts on card | Strong uncensored chat; less formal category breakdown published with the MLX pack |
| **Capability retention** | Card: MMLU ~71.5% vs base ~76.5% (−5 pts) with coherence checks passing | Good general quality; depends on upstream Heretic conversion |
| **Quantization** | **JANG importance quant** — sensitive tensors keep more bits (mixed 4/8, ~5.1 avg); vision float16 | Uniform MLX **4-bit** affine (group 64) |
| **Vision** | **Included** in the Hub package | Upstream language-only; this repo **grafts** stock IT vision |
| **Thinking mode** | Card claims **v2 thinking-ON stability** (fewer degenerate loops) | Works with gemma4 reasoning parser; less dealign-specific tuning |
| **Setup complexity** | Download → run | Download + optional vision graft from stock 31B IT |
| **Best for** | Strongest low-refusal Gemma 4 31B on MLX with native multimodal + smarter quant | Smaller/simpler uniform 4-bit path; already battle-tested with this repo’s proxy |

**Choose JANG_4M CRACK when** you want maximum uncensored compliance, multimodal out of the box, and a quality-preserving mixed quant rather than flat 4-bit.

**Choose Heretic when** you already run that stack, prefer the mlx-community uniform 4-bit layout, or need the vision-graft workflow you already know.

Both use the same Kilo proxy + gemma4 tool/reasoning parsers. Neither is ideal for unattended huge multi-file agent builds vs Qwen 3.6 / GLM for coding loops — see root README “When to use which.”

---

## Quantization / macOS — do we need a 4-bit script?

**No — not for the published Hub weights.**

`dealignai/Gemma-4-31B-JANG_4M-CRACK` is already:

1. **MLX-native** safetensors (library: `mlx`)
2. **JANG v2** quantized with `mx.quantize` (profile `JANG_4M`, target ~4 bits, **actual ~5.1** avg bits)
3. Sized for Apple Silicon (~18–22 GB weights + vision), not a multi-hundred-GB bf16 dump

So the Mac “optimize” step is **download**, not re-quantize. A second pass of `mlx_vlm.convert -q` on these weights would **double-quantize** and wreck quality.

Optional script (only for **full-precision** sources):

```bash
./quantize_to_mlx_4bit.sh /path/to/bf16-model [output-dir]
```

It **refuses** to re-quantize a tree that already has `jang_config.json`. Use it only if you start from bf16/fp16 Gemma 4 and want a uniform 4-bit MLX export.

---

## Quick start

```bash
cd uncensored/gemma4-jang-crack-31b-mlx

# 1. Download ~22 GB JANG weights + create venv
./1_setup_download.sh

# 2. Start OpenAI-compatible server on :8080 (Kilo proxy on by default)
./2_start_mlx.sh

# 3. Kilo Code — launch from this directory so kilo.json is picked up
kilo
```

Raw vllm-mlx only:

```bash
./2_start_mlx.sh --no-proxy
```

Clear a stuck port:

```bash
./2_start_mlx.sh restart
```

---

## Inference tips (from model card)

| Setting | Thinking OFF | Thinking ON |
|---------|--------------|-------------|
| Temperature | 0.0 – 1.0 | **0.3 – 0.7** (avoid pure greedy) |
| Repetition penalty | 1.00 | **1.15 – 1.25** |
| Top P | 0.95 | 0.95 |

Local `kilo.json` uses temp **0.35** (build) / **0.15** (plan) for agent stability.

The Kilo proxy **defaults thinking OFF** (stable for agents). Opt in only with
`enable_thinking: true` **and** `repetition_penalty` ≈ 1.2 — otherwise this
checkpoint tends to plan-loop. With the gemma4 reasoning parser on, short
`max_tokens` can fill only `reasoning_content` and leave `content` null — use
≥128 completion tokens for plain chat smoke tests.

### Smoke test

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gemma-4-31b-jang-crack-mlx",
    "messages":[{"role":"user","content":"Reply with exactly: PONG"}],
    "max_tokens":64,
    "temperature":0.35,
    "enable_thinking": false
  }'
```

Expect `choices[0].message.content` containing `PONG`.

---

## Stack

Same Python pin family as the Heretic Gemma stack, but **this directory is standalone**
(no symlinks to `../gemma4-server-heretic-31b-mlx/`). Shared scripts were copied in so
JANG-specific fixes can diverge safely.

| Package | Notes |
|---------|--------|
| **vllm-mlx** ≥ 0.4 | OpenAI-compatible server |
| **mlx-vlm** 0.5–0.6.x | Multimodal load path |
| **mlx-lm** ≥ 0.31 | Chat CLI / text path |

Local copies (edit freely here): `gemma4_mlx_kilo_proxy.py`, `patches/`, `apply_local_patches.sh`, `check_upstream_patches.sh`, `validate_model.py`, `requirements.txt`.

---

## Scripts

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Venv + deps + download JANG weights |
| `2_start_mlx.sh` | vllm-mlx ± Kilo proxy on :8080 |
| `3_chat.sh` | Terminal chat (`mlx_lm chat`) |
| `quantize_to_mlx_4bit.sh` | **Optional** — only for full-precision → uniform 4-bit |
| `kilo.json` | Provider config for this model |

---

## Architecture

**Default (Kilo proxy on):**

```
Kilo Code / Continue.dev  ──→  :8080  gemma4_mlx_kilo_proxy.py  ──→  :8090  vllm-mlx
```

**With `--no-proxy`:**

```
Kilo Code / Continue.dev  ──→  :8080  vllm-mlx  ──→  gemma-4-31b-jang-crack-mlx
```

---

## Related

- [Heretic uncensored 31B](../gemma4-server-heretic-31b-mlx/README.md) — uniform 4-bit + vision graft  
- [Stock Gemma 4 31B IT AtomicChat 2026-07-15](../../censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/) — aligned  
- [Root README](../../README.md) — all models + Kilo matrix  
