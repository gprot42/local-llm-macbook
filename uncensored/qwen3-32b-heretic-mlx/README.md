# Qwen3-32B Heretic — MLX Server

Uncensored / abliterated **Qwen3-32B** (original **Qwen3** dense line, **not** Qwen3.6 or Qwen3.7) on Apple Silicon via **mlx_lm.server**.

| | |
|---|---|
| **Project dir** | `qwen3-32b-heretic-mlx` |
| **Model dir** | `qwen3-32b-heretic-mlx-5bit` |
| **HF repo** | [`Wwayu/Qwen3-32B-heretic-mlx-5Bit`](https://huggingface.co/Wwayu/Qwen3-32B-heretic-mlx-5Bit) |
| **Source** | [`igriv/Qwen3-32B-heretic`](https://huggingface.co/igriv/Qwen3-32B-heretic) (Heretic of [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B)) |
| **Series** | **Qwen3** dense 32B — **not** Qwen3.6 (27B / 35B-A3B) or Qwen3.7 (API) |
| **Size** | ~22.5 GB (MLX 5-bit) |
| **RAM** | **64 GB+** unified; **80–128 GB** recommended for long agent sessions |
| **API** | `http://127.0.0.1:8084/v1` |
| **Kilo model ID** | `qwen3-heretic/qwen3-32b-heretic-mlx-5bit` |
| **Modalities** | Text only |

> There is no official `Qwen/Qwen3-32B-heretic` repo. This stack uses the
> community Heretic abliteration + MLX 5-bit conversion above.
>
> **Not the coding default.** For aligned Qwen3.6 + MTP, use
> [`../../censored/qwen3-6-27b-coder-mtplx/`](../../censored/qwen3-6-27b-coder-mtplx/) (`:8765`).

---

## Quick start

```bash
cd uncensored/qwen3-32b-heretic-mlx

# 1. Create venv + download ~22.5 GB weights
./1_setup_download.sh

# 2. Start OpenAI-compatible server on :8084
./2_start_mlx.sh

# 3. Kilo Code — launch from this directory so kilo.json is picked up
kilo
# Model: qwen3-heretic/qwen3-32b-heretic-mlx-5bit
```

Or merge the provider from this directory’s `kilo.json` (or root `kilo.json`) into
`~/.config/kilo/kilo.jsonc` and set:

```text
model: qwen3-heretic/qwen3-32b-heretic-mlx-5bit
```

Clear a stuck port:

```bash
./2_start_mlx.sh restart
```

Terminal chat without the HTTP server:

```bash
./3_chat.sh
```

### Health check

```bash
./2_start_mlx.sh status
curl -s http://127.0.0.1:8084/v1/models | python3 -m json.tool
```

Smoke completion:

```bash
curl -s http://127.0.0.1:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-32b-heretic-mlx-5bit",
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 32,
    "temperature": 0.6
  }' | python3 -m json.tool
```

### Stop

```bash
./2_start_mlx.sh stop
```

---

## Architecture

```
Kilo Code / curl / any OpenAI client
      │
      ▼  http://127.0.0.1:8084/v1
  mlx_lm.server
      │
      ▼
  qwen3-32b-heretic-mlx-5bit  (local MLX weights)
```

Port **8084** so this can run alongside:

| Stack | Port |
|-------|------|
| Gemma / Diffusion | `:8080` |
| DeepSeek MLX | `:8082` |
| DeepSeek ds4 | `:8083` |
| **This stack (Qwen3-32B Heretic)** | **`:8084`** |
| Qwen3.6 mtplx (aligned coding) | `:8765` |
| Ornith Ollama | `:18082` |
| GLM Heretic Ollama | `:18083` |

Still avoid loading multiple huge models at once on ≤128 GB.

---

## Qwen series map (do not confuse)

| Name | Open local? | This stack? |
|------|-------------|-------------|
| **Qwen3-32B** | Yes — dense ~33B | **Yes** (Heretic MLX 5-bit) |
| **Qwen3.6** | Yes — 27B dense, 35B-A3B MoE | No — see `censored/qwen3-6-27b-coder-mtplx/` |
| **Qwen3.7** | No — cloud/API (e.g. Max) | No |

---

## Scripts

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Create/repair venv, install deps, download MLX 5-bit weights |
| `2_start_mlx.sh` | Start / stop / status for `mlx_lm.server` on `:8084` |
| `3_chat.sh` | Terminal chat without the HTTP server |
| `validate_model.py` | Refuse to start if weight shards are incomplete |
| `kilo.json` | Kilo provider: `qwen3-heretic` → `http://127.0.0.1:8084/v1` |
| `requirements.txt` | Python deps for the venv |

### Server options (`2_start_mlx.sh`)

```bash
./2_start_mlx.sh                  # start on :8084
./2_start_mlx.sh restart          # free port, then start
./2_start_mlx.sh stop
./2_start_mlx.sh status
./2_start_mlx.sh --port 8085
./2_start_mlx.sh --temp 0.7 --top-p 0.9
```

### Sampling

Qwen3 thinking-mode defaults used in `kilo.json`:

| Param | Value |
|-------|-------|
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |

Qwen3 may emit `<think>…</think>` reasoning blocks depending on the chat
template / client settings. That is normal model behavior, not a stack bug.

---

## Notes

- **MLX-native** safetensors — download only; do **not** re-quantize with
  `mlx_lm.convert -q` (would double-quantize).
- Downloads are **resumable**. If interrupted: re-run `./1_setup_download.sh`.
- After moving this folder, re-run `./1_setup_download.sh --skip-download` so
  venv shebangs are repaired.
- For a lighter uncensored MoE path, see
  [`../glm-4.7-flash-heretic-gguf-ollama/`](../glm-4.7-flash-heretic-gguf-ollama/).
- For uncensored Gemma chat, see
  [`../gemma4-server-heretic-31b-mlx/`](../gemma4-server-heretic-31b-mlx/).
