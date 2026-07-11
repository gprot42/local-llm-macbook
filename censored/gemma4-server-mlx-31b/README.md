# Gemma 4 31B IT — MLX + Kilo Code

Run **Gemma 4 31B IT (4-bit)** locally via **mlx-lm** by default, or opt into **mlx-vlm** with **MTP speculative decoding**.

## Quick start

```bash
# 1. Download target (~20 GB) + MTP assistant (~1 GB) and install deps
./1_setup_download.sh

# 2. Start the OpenAI-compatible server on :8080 (no MTP by default)
./2_start_mlx.sh

# 3. Open your project and launch Kilo Code
#    kilo.json in the workspace root is auto-loaded
kilo
```

## Architecture

**Default (MTP off):**

```
Kilo Code ──→ mlx_lm.server ──→ gemma-4-31b-it-mlx-4bit
```

**With MTP (`./2_start_mlx.sh --with-mtp`):**

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8080/v1   (OpenAI-compatible)
  mlx_vlm.server
      │  speculative decode: --draft-kind mtp
      ├── target:  gemma-4-31b-it-mlx-4bit        (~20 GB, 4-bit)
      └── drafter: gemma-4-31b-it-assistant-mlx-bf16  (~1 GB, shares K/V)
```

MTP uses Google's [Gemma 4 assistant drafter](https://ai.google.dev/gemma/docs/mtp/mtp) built into mlx-vlm. Measured greedy speedups on Apple Silicon are up to **~2.3×** on the 31B model at block size 4 (decode only; long agent prefill is unchanged). It is opt-in because the non-MTP server is simpler and more stable for Kilo agent work.

### MTP pairing (target + assistant)

`gemma-4-31b-it-assistant-mlx-bf16` is the official MTP drafter for **`google/gemma-4-31b-it`**. It is compatible with the 4-bit MLX target in this repo:

| | Target `gemma-4-31b-it-mlx-4bit` | Assistant `gemma-4-31b-it-assistant-mlx-bf16` |
|---|---|---|
| Role | Full model (quality) | Small drafter (~4 layers, ~1 GB) |
| Quantization | 4-bit (memory) | bf16 (size is small either way) |
| Hidden size | 5376 | `backbone_hidden_size` 5376 (must match) |
| Vocabulary | 262,144 | 262,144 (same tokenizer) |

The assistant does not replace the target — it only proposes tokens; the 31B model verifies them and shares K/V. Use the **27B assistant** only if you switch the target to `gemma-4-27b-it-4bit`. At `temperature=0`, output should match non-MTP greedy generation.

## Files

| File | Purpose |
|---|---|
| `1_setup_download.sh` | Create venv, install mlx/mlx-lm/mlx-vlm, download target + MTP assistant |
| `2_start_mlx.sh` | Start server on port 8080 (no MTP by default; pass `--with-mtp` to enable) |
| `apply_local_patches.sh` | Copy `patches/` into the venv on each start (survives `pip install -U`) |
| `kilo.json` | Kilo Code config for this project |
| `patches/sample_utils.py` | top-p fix for 3-D MTP verify logits |
| `patches/app.py` | Map API model id `default_model` → CLI `--model` (mlx_lm convention) |
| `patches/models/gemma4/language.py` | MTP rollback when `accepted` is a Python list |
| `patches/patch_generation.py` | Close `BatchGenerator` after MTP errors (avoids hung server) |

## Kilo Code

Launch Kilo from **this directory** so `kilo.json` is picked up:

```bash
cd gemma4-server-mlx-31b
./2_start_mlx.sh    # terminal 1
kilo                # terminal 2 — same directory
```

| Setting | Value | Why |
|---------|-------|-----|
| Model ID | `gemma-4-31b-it-mlx-4bit` | Recommended; matches the local server model |
| Model ID (alt.) | `default_model` | Also works — mapped to the same local weights at startup |
| Context limit | 32k | Faster prefill; fits agent sessions without OOM |
| Output limit | 8k | Enough for edits; reduces truncated tool JSON |
| Agent temp | 0.35 | Gemma at `temperature=0` stalls after tool calls |
| Reasoning | **Off** in UI | Checked → infinite `<thinking>` loops |

> For heavier agent tuning (proxy, fuzzy edits, Heretic model), use
> `gemma4-server-heretic-31b-mlx`. For coding speed on Qwen, use `qwen3-6-27b-coder-mtplx`.

## Options

### Enable MTP

```bash
./2_start_mlx.sh --with-mtp
```

Uses `mlx_vlm.server` plus the Gemma 4 assistant drafter. This can improve decode throughput, but Kilo wall-clock speed also includes prompt prefill, tool calls, edits, and file IO.

### MTP block size

```bash
./2_start_mlx.sh --draft-block-size 4   # default; try 2–4
```

Only applies with `--with-mtp`. Larger blocks can increase speed when acceptance stays high; if the server logs low acceptance, reduce the block size.

### Skip assistant download

```bash
./1_setup_download.sh --skip-mtp-download
./2_start_mlx.sh
```

### Change port

```bash
./2_start_mlx.sh --port 8081
```

Then update `kilo.json`:

```json
"baseURL": "http://localhost:8081/v1"
```

### Quick CLI test (no server)

```bash
source venv/bin/activate
mlx_lm.generate --model gemma-4-31b-it-mlx-4bit --prompt "Hello, who are you?"
```

MTP CLI test (loads both models):

```bash
mlx_vlm.generate --model gemma-4-31b-it-mlx-4bit \
  --draft-model gemma-4-31b-it-assistant-mlx-bf16 \
  --draft-kind mtp --draft-block-size 4 \
  --prompt "Say hi in one sentence." --max-tokens 32 --temperature 0.35
```

### API smoke test (server running)

```bash
# List models
curl http://localhost:8080/v1/models

# Chat — explicit model id
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31b-it-mlx-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'

# Chat — default_model alias (Kilo / Cursor style)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

## Troubleshooting

**`mlx_vlm.server` / assistant not found**

```bash
./1_setup_download.sh
```

**MTP falls back to mlx_lm.server at startup**

Assistant weights are missing or incomplete. Re-download:

```bash
hf download mlx-community/gemma-4-31B-it-assistant-bf16 \
  --local-dir gemma-4-31b-it-assistant-mlx-bf16
```

**`default_model` / HuggingFace 404 after first request**

Some clients send `"model": "default_model"` (mlx_lm convention). Upstream `mlx_vlm.server` treated that as a HuggingFace repo id, unloaded the local weights, and failed with `Repository Not Found`.

Fix: restart so patches apply (`./2_start_mlx.sh restart`). You should see `Using cached model: gemma-4-31b-it-mlx-4bit` on chat requests, not `Loading model from: default_model`.

**`model not found` in Kilo**

Use `openai-compatible/gemma-4-31b-it-mlx-4bit` as in `kilo.json`, or ensure the OpenAI provider `baseURL` is `http://localhost:8080/v1` while `./2_start_mlx.sh` is running.

**MTP crash / hung server after first error**

Two upstream bugs, fixed by local patches (re-applied on every `./2_start_mlx.sh`):

1. `'list' object has no attribute 'max'` — `patches/models/gemma4/language.py` (MTP KV rollback).
2. Server stops responding after an MTP error — `patches/patch_generation.py` (must `batch_gen.close()` before clearing).

```bash
./apply_local_patches.sh
./2_start_mlx.sh restart
```

When started with `--with-mtp`, startup runs an MTP smoke test; if it fails, the script exits instead of leaving a wedged server.

**MTP works but Kilo hangs**

Restart the server after any crash. Only run one server on port 8080. By default, `./2_start_mlx.sh` uses `mlx_lm.server` (no assistant, slower decode, more stable). Use `./2_start_mlx.sh restart --with-mtp` only when you specifically want MTP.

**Port 8080 already in use**

```bash
./2_start_mlx.sh restart
# or:
./2_start_mlx.sh --port 8081
```

**MTP errors / garbled output at high temperature**

MTP works best with moderate sampling (`temperature` 0.2–0.6, `top_p` 0.9–0.95). Try `--draft-block-size 2` or restart without `--with-mtp`.

**Out of memory / exit code 134 / Metal crash (macOS Report)**

This is **not** a Python traceback you can catch — MLX reports a failed Metal command buffer, throws a C++ exception on `com.Metal.CompletionQueueDispatch`, and the process calls `abort()` (**SIGABRT**, exit **134**).

Typical signature in Console or terminal:

```text
[METAL] Command buffer execution failed: Insufficient Memory
libc++abi: terminating due to uncaught exception of type std::runtime_error
```

What the crash report means:

| Signal | Meaning |
|--------|---------|
| Thread 31 `com.Metal.CompletionQueueDispatch` | GPU driver finished a command buffer with an OOM error |
| `mlx::core::gpu::check_error` | MLX detected the Metal failure and aborted |
| `asyncio_0` / `asyncio_1` / AnyIO workers | Often overlapping Kilo chat streams hitting the same server |
| ~5 min uptime then crash | Weights fit (~20 GB), but **KV cache + MTP + concurrent requests** pushed unified memory over the limit |

The 31B 4-bit target needs ~20 GB for weights; the MTP assistant adds ~1 GB; **long or concurrent agent sessions** add much more for KV cache. On **24 GB** Macs (common for `Mac17,7`-class hardware), 32k context + MTP + parallel tool rounds is often too much.

Mitigations (try in order):

```bash
./2_start_mlx.sh restart
./2_start_mlx.sh --low-memory          # disables MTP, caps KV at 8k tokens
./2_start_mlx.sh                       # default MTP-off path
./2_start_mlx.sh --with-mtp --draft-block-size 2  # if you opt into MTP
./2_start_mlx.sh --with-mtp --max-kv-size 8192    # cap KV while using MTP
```

In Kilo: lower `limit.context` in `kilo.json` (e.g. **16384**), avoid parallel agents, close other GPU-heavy apps (browser, second MLX server).

On machines with **32 GB+** unified memory, opt-in MTP + 32k context is often fine. On **24 GB**, prefer the default non-MTP mode, `--low-memory`, or the 27B model + matching assistant.