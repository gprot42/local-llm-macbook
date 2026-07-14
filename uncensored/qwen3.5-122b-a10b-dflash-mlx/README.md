# Qwen3.5-122B-A10B + DFlash (MLX)

Speculative decoding for the **122B-A10B** MoE target using
[`z-lab/Qwen3.5-122B-A10B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash)
via a patched [dflash-mlx](https://github.com/Aryagm/dflash-mlx) runtime.

| | |
|--|--|
| **Project dir** | `qwen3.5-122b-a10b-dflash-mlx` |
| **Target (default)** | Sibling abliterated MLX 4-bit (~65–70 GB) |
| **Draft** | `z-lab/Qwen3.5-122B-A10B-DFlash` (~1.5 GB BF16) |
| **Runtime** | Vendored `dflash-mlx/` (+ `qwen3_5_moe` adapter) |
| **API** | `http://127.0.0.1:8086/v1` |
| **Model ID** | `qwen3.5-122b-a10b-dflash` |
| **RAM** | **128 GB** unified recommended |
| **Kilo / agent ops** | [`AGENT_OPS.md`](AGENT_OPS.md) · [`kilo_system_prompt.txt`](kilo_system_prompt.txt) · `./status_dflash.sh` |

This is **not** a drop-in for `mlx_lm.server`. DFlash needs the draft/verify loop
in dflash-mlx (hidden-state taps + KV/linear-state rollback).

---

## Prerequisites

1. Sibling target already downloaded:

```bash
cd ../qwen3.5-122b-a10b-abliterated-mlx
./1_setup_download.sh
```

Or set `TARGET_MODEL` to any local MLX `qwen3_5_moe` / `qwen3_5` pack with **48** layers
compatible with the draft’s `target_layer_ids`.

2. **Do not** co-load DeepSeek V4 or another 70+ GB model on 128 GB.

---

## Quick start

```bash
cd uncensored/qwen3.5-122b-a10b-dflash-mlx

# venv + install dflash-mlx + download draft (~1.5 GB)
./1_setup_download.sh

# Optional but recommended: greedy exactness vs plain target
./3_smoke_exactness.sh
# or faster: ./3_smoke_exactness.sh --quick

# OpenAI-compatible server (text-only)
./2_start_dflash.sh
```

```bash
curl -s http://127.0.0.1:8086/health
curl -s http://127.0.0.1:8086/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-122b-a10b-dflash",
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

```bash
./2_start_dflash.sh status
./2_start_dflash.sh restart
./2_start_dflash.sh stop
```

---

## Architecture

```
Kilo / curl
    │  http://127.0.0.1:8086/v1
    ▼
dflash-mlx-openai-server
    │  draft block proposal (DFlash ~0.8B)
    │  target verify + accept prefix (exact)
    ▼
target: qwen3.5-122b MLX 4-bit (abliterated)  +  draft: z-lab DFlash
```

Port **8086** (plain abliterated AR server stays on **:8085**).

---

## What we patched

Upstream dflash-mlx only registered `qwen3` / `qwen3_5`. This stack’s
`dflash-mlx/` vendor also:

- Registers **`qwen3_5_moe`** → same Qwen3.5 adapter
- Prepares the custom `forward_dflash` model file for MoE packs
- Sanitizes MoE expert weight layouts + drops vision tower keys

See `dflash-mlx/dflash_mlx/adapters.py` and `custom_qwen35_model.py`.

---

## Expectations

| | |
|--|--|
| **Correctness** | Greedy DFlash should match plain target (`./3_smoke_exactness.sh`) |
| **Speed** | Decode speedup only (not prefill). Abliterated **4-bit** + Mac may land **below** NVIDIA SGLang 3–4× claims; any solid accept length is a win |
| **Server** | Text-only OpenAI chat completions (no vision, minimal tooling vs full Kilo proxies) |
| **First load** | Minutes for ~65 GB target residency |

If exactness fails, open an issue with smoke log output — do not treat the server as production-correct until PASS.

---

## Override paths

```bash
TARGET_MODEL=/path/to/other-mlx-122b \
DRAFT_REPO=z-lab/Qwen3.5-122B-A10B-DFlash \
  ./1_setup_download.sh
```

---

## Related

- Plain AR abliterated server: [`../qwen3.5-122b-a10b-abliterated-mlx/`](../qwen3.5-122b-a10b-abliterated-mlx/)
- Model guide: [`../../README-models.md`](../../README-models.md)
- Upstream DFlash (CUDA/SGLang): [HF model card](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash)
