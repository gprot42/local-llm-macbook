# Gemma 4 (MLLM / `gemma4_text`): fast TextModel path, stalls, and API gaps

## Environment

- **vllm-mlx**: 0.3.0 (PyPI latest as of 2026-06-02)
- **mlx-vlm**: ≥ 0.5.0 recommended (RoPE offset + thread-local MLX streams)
- **mlx-lm**: ≥ 0.31.0 (`mlx_lm.models.gemma4_text` exists)
- **Model**: `mlx-community/gemma-4-26B-A4B-it-heretic-4bit` (and other Gemma 4 MLLM checkpoints with `text_config.model_type == "gemma4_text"`)
- **Client**: Kilo Code / OpenAI-compatible agents

Reproducer project (local patches + analysis): community usage of Gemma 4 Heretic on Apple Silicon via vllm-mlx.

---

## Summary

Serving Gemma 4 multimodal weights through vllm-mlx 0.3.0 has three interacting problems:

1. **`text_model_from_vlm.py` always imports Qwen 3.5** → Gemma 4 fast path fails → silent **MLLM fallback** → empty-delta loops and broken tool-call streaming after turn 2.
2. **No OpenAI `logit_bias` passthrough** → cannot ban leaked Harmony control tokens (`<|channel>`, etc.) at sampling time.
3. **Gemma 4 + continuous batching** needs a small attention mask trim when batched masks are longer than sliding-window keys (vllm-mlx `main` already has this in `patches/gemma4_mllm.py`; mlx-vlm ≥ 0.5.0 fixed the older BatchKVCache RoPE bug).

Items (1) and (2) are **not** fixed in the PyPI 0.3.0 wheel. Item (3) is partially addressed on vllm-mlx GitHub `main` and mlx-vlm ≥ 0.5.0.

---

## Bug 1: `text_model_from_vlm` hardcodes `qwen3_5` (primary root cause)

### Current upstream (vllm-mlx 0.3.0)

`vllm_mlx/text_model_from_vlm.py` always does:

```python
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
args = TextModelArgs.from_dict(text_config)
```

For Gemma 4, `text_config` uses `model_type: "gemma4_text"` with different field shapes. This raises **`ZeroDivisionError`** in `TextModelArgs.__post_init__` (head_dim derivation).

### Observable startup

```
ERROR:vllm_mlx.text_model_from_vlm:Failed to build TextModel from vlm: division by zero
INFO:vllm_mlx.engine.simple:SimpleEngine loaded: gemma-4-26b-heretic-mlx-4bit (MLLM=True)
```

`MLLM=True` means the **slow mlx_vlm MLLM path** is used even for **text-only** chat. That path mishandles thinking-channel exit → **empty `delta` SSE chunks** at high rate until `max_tokens`, hanging clients for minutes.

### Proposed fix

Dispatch on `text_config["model_type"]` (or `config["model_type"]`):

```python
def _import_text_model_classes(model_type: str):
    if model_type == "gemma4_text":
        from mlx_lm.models.gemma4_text import Model as TextModel, ModelArgs as TextModelArgs
        return TextModel, TextModelArgs
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    return TextModel, TextModelArgs
```

Then in `build_text_model()`:

```python
model_type = text_config.get("model_type") or config.get("model_type", "")
imported = _import_text_model_classes(model_type)
if imported is None:
    return None
TextModel, TextModelArgs = imported
```

**Success criterion:** startup log shows fast TextModel path and **`MLLM=False`** (or equivalent) for pure-text Gemma 4 serves; turn 2+ agent sessions emit real tool-call deltas instead of empty-delta loops.

A full working patch is available in the community reproducer's `patches/text_model_from_vlm.py` (tested with mlx-lm 0.31.3).

---

## Bug 2: Missing `logit_bias` on chat completions API

### Symptom

Heretic / Harmony-tuned checkpoints sometimes emit raw control tokens (`<|channel>`, `<|think|>`, …) as **text** before tool calls. Without sampling-time bias, vllm-mlx's tool parser never recovers; agent UIs see narration instead of `write` / `edit` tool calls.

OpenAI API supports `logit_bias: {"token_id": bias}`. vllm-mlx 0.3.0 does not accept or forward this field to `mlx_lm.sample_utils.make_logits_processors(logit_bias=...)`.

### Proposed fix (3 files)

1. **`api/models.py`** — add to `ChatCompletionRequest`:
   ```python
   logit_bias: dict[str, float] | None = None
   ```
2. **`server.py`** — pass `request.logit_bias` into engine `chat_kwargs`.
3. **`engine/simple.py`** — `kwargs.pop("logit_bias")`, coerce keys to `int`, pass to `make_logits_processors(logit_bias=...)`.

Reference implementation: community `patches/api/models.py`, `patches/server.py`, `patches/engine/simple.py`.

---

## Bug 3: Gemma 4 attention under continuous batching (partially fixed upstream)

### Historical issue (mlx-vlm < 0.5.0)

`Attention.__call__` held a reference to `cache.offset`; `BatchKVCache` mutates offset in-place → wrong RoPE on queries → garbled tokens when batching enabled.

**Fixed in mlx-vlm ≥ 0.5.0:**

```python
offset = mx.array(cache.offset) if cache is not None else 0
```

### Remaining vllm-mlx issue (mask shape)

With **continuous batching**, masks are sometimes sized for the full prompt while sliding-window layers cap `keys` shorter → `scaled_dot_product_attention` shape mismatch.

vllm-mlx **GitHub `main`** already ships `patches/gemma4_mllm.py` that trims the mask to `keys.shape[-2]` (see [main branch file](https://github.com/waybarrios/vllm-mlx/blob/main/vllm_mlx/patches/gemma4_mllm.py)). Please include this in the next PyPI release alongside mlx-vlm ≥ 0.5.0.

---

## Bug 4: mlx-vlm worker-thread MLX stream (fixed in mlx-vlm ≥ 0.5.0)

mlx-vlm < 0.5.0 used a process-global `generation_stream`; media requests on worker threads failed with `There is no Stream(gpu, N) in current thread`.

mlx-vlm 0.5.0+ uses `mx.new_thread_local_stream()`. vllm-mlx should document **`mlx-vlm>=0.5.0`** for Gemma 4 MLLM serves.

---

## Workarounds today (outside vllm-mlx)

| Workaround | Trade-off |
|------------|-----------|
| Local patches to `text_model_from_vlm.py` + `logit_bias` | Must re-apply after every `pip install` |
| 8k-line HTTP proxy in front of vllm-mlx | Fragile; masks engine bugs |
| `mlx_lm.server` for text-only (no vllm-mlx) | Loses vllm-mlx batching / multimodal API |

**Recommended product direction:** merge Bug 1 + Bug 2 into vllm-mlx; depend on mlx-vlm ≥ 0.5.0; ship Bug 3 mask patch in the wheel.

---

## Verification checklist

After fixes, serving `gemma-4-26b-heretic-mlx-4bit` should satisfy:

- [ ] No `division by zero` from `text_model_from_vlm` at startup
- [ ] Engine not stuck in `MLLM=True` for text-only chat
- [ ] Multi-turn Kilo prompt completes with tool calls (not 90s empty-delta stall)
- [ ] Optional: `logit_bias` in request JSON reaches sampler
- [ ] With `--continuous-batching`, no mask/attention shape errors on Gemma 4

---

## References

- mlx-lm Gemma 4 text model: `mlx_lm.models.gemma4_text`
- mlx-vlm RoPE fix: ≥ 0.5.0 `language.py` defensive `mx.array(cache.offset)`
- Related mlx-vlm: PR #564 (BatchKVCache / RoPE, per vllm-mlx main patch comment)

Happy to open a PR with the `text_model_from_vlm` + `logit_bias` changes if maintainers want a starting point.