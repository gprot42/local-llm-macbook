# Muse Glimmer 30B — mlx-vlm + DFlash

Run **Muse Glimmer 30B** locally on Apple Silicon with [mlx-vlm](https://github.com/blaizzy/mlx-vlm)
(OpenAI-compatible API for Kilo Code). Default weights are the **mlx-community 4-bit**
quant (~19.4 GB) plus Meta’s official **DFlash** drafter.

Announcement: [Alexandr Wang, 2026-08-10](https://x.com/alexandr_wang/status/2086756152034066792) ·
[Meta research post](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) ·
[HF base](https://huggingface.co/meta-models/Muse-Glimmer-30B) ·
[developer docs](https://dev.meta.ai/docs/muse-glimmer)

**Port `:8087`** so it can sit beside Gemma/Diffusion (`:8080`), DeepSeek MLX (`:8082`),
ds4 (`:8083`), and Qwen mtplx (`:8765` / `:8766`).

| | |
|--|--|
| **HF target** | `mlx-community/Muse-Glimmer-30B-4bit` |
| **DFlash drafter** | `meta-models/Muse-Glimmer-30B-assistant` (~5 GB BF16) |
| **Modalities** | **Text + image** in, text out |
| **License** | Apache 2.0 |
| **Kilo model ID** | `muse-glimmer/muse-glimmer-30b-mlx` |
| **API** | `http://127.0.0.1:8087/v1` |
| **Sampling** | `temperature=1.0`, `top_p=0.95`, `top_k=64` (official) |

Muse Glimmer is a ~29.6B dense transformer with a Perception Encoder (ViT-G/14)
and a DFlash block-diffusion drafter (16-token blocks). It was distilled from
Muse Spark for **local agents**: tool calling, failure recovery, and long-horizon
planning on a 24–32 GB envelope when quantized.

## Quick start

```bash
# 1. Install mlx-vlm and download 4-bit + DFlash assistant
./1_setup_download.sh
# deps only:
#   ./1_setup_download.sh --deps-only
# other quants:
#   ./1_setup_download.sh 8bit
#   ./1_setup_download.sh mxfp4
#   GLIMMER_HF_MODEL=mlx-community/Muse-Glimmer-30B-6bit ./1_setup_download.sh

# 2. Start server (DFlash on if the assistant is present; harness gate ON)
./2_start_mlx.sh
# If port 8087 is stuck:  ./2_start_mlx.sh restart

# 3. Harness (also run automatically after start unless --no-harness-gate)
python3 test_harness.py --gate
python3 test_harness.py              # fuller live suite
python3 test_harness.py --quick

# 4. Kilo Code — launch from this directory
kilo
# Model: muse-glimmer/muse-glimmer-30b-mlx
```

## Architecture

```
Kilo Code (TUI)
      │
      ▼  http://127.0.0.1:8087/v1   (OpenAI-compatible)
  mlx_vlm.server
      │  --draft-kind dflash  (optional, default when assistant is present)
      ├── target:  mlx-community/Muse-Glimmer-30B-4bit   (~19.4 GB)
      └── drafter: meta-models/Muse-Glimmer-30B-assistant  (~5 GB)
```

Do **not** use `mlx_lm.server` — Muse Glimmer is multimodal and mlx-lm cannot
load it. mlx-vlm ≥ **0.6.12** auto-detects the official assistant as DFlash.

DFlash speeds **decode**, not prefill. Long Kilo histories still cost a large
first-token wait.

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv + mlx-vlm + model / drafter pull; writes `.mlx_config` |
| `2_start_mlx.sh` | Serve on `:8087`; DFlash on by default; post-start harness gate |
| `test_harness.py` | Live API / tool-call smoke tests (accounts for always-on thinking) |
| `validate_model.py` | Check local weight shards before start |
| `kilo.json` | Kilo provider: `muse-glimmer/muse-glimmer-30b-mlx` → `:8087/v1` |

## Model resolution

| Arg / env | Effect |
|-----------|--------|
| `./1_setup_download.sh` / `4bit` / `30b` | **4-bit** `mlx-community/Muse-Glimmer-30B-4bit` |
| `5bit` / `6bit` / `8bit` / `mxfp4` | matching mlx-community quant |
| `official` / `bf16` | `mlx-community/Muse-Glimmer-30B-bf16` (large) |
| `./1_setup_download.sh org/repo` | explicit HF repo |
| `--skip-dflash-download` | target only |
| `GLIMMER_HF_MODEL=...` | force target repo |
| `GLIMMER_ALIAS=...` | OpenAI model id (default `muse-glimmer-30b-mlx`) |

## Reasoning

Thinking **cannot be switched off**. Control *how much* with a system line:

```
Reasoning strength: low | medium | high | xhigh
```

Use **high** / **xhigh** for coding and agent loops (this folder’s `kilo.json`
sets `high` on build/debug). Use **low** for short chat / harness.

Give the model room: `max_tokens` under a few hundred often blanks both
`content` and `reasoning_content` mid-thought. The start script default is
`--max-tokens 8192 --thinking-budget 2048`.

Do **not** stop on `<|eom|>` — that is end-of-message, not end-of-turn, and
collapses parallel tool calling.

## Harness

`test_harness.py` hits the public OpenAI API only (no Kilo session DB).

| Mode | What it covers |
|------|----------------|
| `--gate` | Reachable, `/v1/models`, short chat, bash tool call, SSE stream, multi-turn tool result |
| default | Gate-level + health, unicode, empty tool result, multi-step continue, concurrent chats |
| `--quick` | Skip multi-step + concurrent |
| `--strict` | Soft (model-behavior) failures become hard |

```bash
python3 test_harness.py --base http://127.0.0.1:8087 --model muse-glimmer-30b-mlx
```

Exit codes: `0` pass · `1` hard fail · `2` unreachable.

`./2_start_mlx.sh` runs `--gate` after ready. Disable with `--no-harness-gate`.

## Options

```bash
./2_start_mlx.sh --port 8088
./2_start_mlx.sh --no-dflash
./2_start_mlx.sh --with-dflash --draft-block-size 16
./2_start_mlx.sh --low-memory          # DFlash off, KV 8192
./2_start_mlx.sh --thinking-budget 512
./2_start_mlx.sh restart
./2_start_mlx.sh stop
./2_start_mlx.sh status
```

Update `kilo.json` `baseURL` if you change `--port`.

## Official GGUF / llama.cpp (not this folder)

Meta’s first-class GGUF path is
[`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
(`KQuant-17GB-Q4_K_M` + `mmproj` + `dflash`). Needs **llama.cpp ≥ b10353**.
Ollama also ships `muse-glimmer:30b-mlx`. This stack stays on **mlx-vlm** so it
matches the other Apple Silicon folders here.

## Troubleshooting

**Setup exit 2 / “No pullable weights”**

HF id typo or mid-upload. Search
[Muse-Glimmer-30B](https://huggingface.co/models?search=Muse-Glimmer-30B)
and pass `org/repo` / `GLIMMER_HF_MODEL`.

**Harness gate soft-fails tool call or empty content**

Thinking ate `max_tokens`, or the model chatted instead of calling tools. Re-run
the gate; use `Reasoning strength: low` for short prompts. Soft fails do not
stop the server.

**Kilo slow, curl fast**

Prefill / context size — compact the session. DFlash does not help the first
token of a long history.

**Don’t load two 20+ GB models at once** on ≤128 GB unified memory.
