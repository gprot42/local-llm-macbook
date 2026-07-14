# Qwen3.5-122B-A10B Abliterated — MLX Server

Uncensored **Qwen3.5-122B-A10B** (MoE, ~10B active) abliterated on Apple Silicon via **mlx_lm.server**.

| | |
|---|---|
| **Project dir** | `qwen3.5-122b-a10b-abliterated-mlx` |
| **Model dir** | `qwen3.5-122b-a10b-abliterated-mlx-4bit` |
| **HF repo (default)** | [`vanch007/Qwen3.5-122B-A10B-abliterated-4bit-vlm-mlx-cs2764-final`](https://huggingface.co/vanch007/Qwen3.5-122B-A10B-abliterated-4bit-vlm-mlx-cs2764-final) |
| **Source lineage** | Abliterated quant of [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) (Feb 2026) |
| **Series** | **Qwen3.5** MoE 122B total / ~10B active — uncensored |
| **Local format** | MLX 4-bit (~70 GB; pre-quantized on Hub) |
| **API** | `http://127.0.0.1:8085/v1` |
| **Kilo model ID** | `qwen35-122b-abliterated/qwen3.5-122b-a10b-abliterated-mlx-4bit` |
| **RAM** | **128 GB** unified recommended |

> Full BF16 abliterated (`huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated`, ~250 GB) is too large to download + convert on a typical 128 GB Mac. Prefer a prebuilt MLX quant.

---

## Quick start

```bash
cd uncensored/qwen3.5-122b-a10b-abliterated-mlx

# 1. Download prebuilt MLX 4-bit (~70 GB)
./1_setup_download.sh

# 2. Serve OpenAI API on :8085
./2_start_mlx.sh

# 3. Kilo — uses repo-root ../../kilo.json (no local kilo.json)
# Switch model to: qwen35-122b-abliterated/qwen3.5-122b-a10b-abliterated-mlx-4bit
kilo
```

```bash
./2_start_mlx.sh restart
./2_start_mlx.sh status
./2_start_mlx.sh stop
./3_chat.sh
```

---

## Architecture

```
Kilo / curl
    │  http://127.0.0.1:8085/v1
    ▼
mlx_lm.server
    ▼
qwen3.5-122b-a10b-abliterated-mlx-4bit
```

Port **8085** (beside Qwen3-32B Heretic on `:8084`, Qwen3.6 mtplx on `:8765`).
Do **not** co-load with DeepSeek V4 or another 70+ GB model on 128 GB.

---

## Scripts

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv + resumable HF download (+ optional `mlx_lm convert -q` for BF16 overrides) |
| `download_resumable.py` | HTTP Range resume for partial shards (stock `hf download` restarts them) |
| `2_start_mlx.sh` | start / stop / status on `:8085` |
| `3_chat.sh` | terminal chat |
| `validate_model.py` | refuse start if shards incomplete |

Kilo provider lives in the **universal** repo-root config: [`../../kilo.json`](../../kilo.json)  
(`qwen35-122b-abliterated` → `http://127.0.0.1:8085/v1`). There is no local `kilo.json` in this stack.

### Override model id

```bash
# Heretic-method mixed 3.8-bit MLX (~63 GB)
HF_REPO=TheCluster/Qwen3.5-122B-A10B-Heretic-v2-MLX-mixed-3.8bit ./1_setup_download.sh

# 3-bit abliterated MLX (~62 GB)
HF_REPO=osmapi/Qwen3.5-122B-A10B-Abliterated-MLX-3 ./1_setup_download.sh

# Mixed 3.6-bit abliterated MLX (~61 GB)
HF_REPO=Jcoa/Qwen3.5-122B-A10B-Abliterated-MLX-mixed3_6 ./1_setup_download.sh

# Full BF16 abliterated (huge; convert may OOM on 128 GB)
HF_REPO=huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated ./1_setup_download.sh
```

### Sampling (root `kilo.json` agent defaults)

| Param | Value |
|-------|-------|
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |

---

## Hardware notes (M5 Max 128 GB)

| | |
|---|---|
| Weights (4-bit) | ~70 GB |
| Active params | ~10B / token (MoE) |
| Expected | Fits with room for moderate context |
| Long context | Lower `--max-tokens` / client context if memory pressure rises |

---

## Related

- **DFlash (faster decode, same target):** [`../qwen3.5-122b-a10b-dflash-mlx/`](../qwen3.5-122b-a10b-dflash-mlx/) on `:8086`
- Qwen3-32B Heretic (true Qwen3 dense): [`../qwen3-32b-heretic-mlx/`](../qwen3-32b-heretic-mlx/)
- Aligned Qwen3.6 27B coding: [`../../censored/qwen3-6-27b-coder-mtplx/`](../../censored/qwen3-6-27b-coder-mtplx/)
