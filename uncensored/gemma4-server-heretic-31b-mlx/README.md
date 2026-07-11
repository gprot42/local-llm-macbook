# Gemma 4 31B Heretic Uncensored — MLX Server

Dedicated **31B Heretic** (uncensored / abliterated) setup via **vllm-mlx** on Apple Silicon.

| | |
|---|---|
| **Project dir** | `gemma4-server-heretic-31b-mlx` |
| **Model dir** | `gemma-4-31b-heretic-mlx-4bit` |
| **HF repo** | [`mlx-community/gemma-4-31B-it-uncensored-heretic-4bit`](https://huggingface.co/mlx-community/gemma-4-31B-it-uncensored-heretic-4bit) |
| **Size** | ~20 GB (4-bit) |
| **RAM** | **80 GB+** unified memory recommended for agent sessions |
| **API** | `http://localhost:8080/v1` |
| **Kilo model ID** | `gemma-4-31b-heretic-mlx-4bit` |
| **Modalities** | Text + image (vision tower grafted from stock `gemma-4-31b-it-mlx-4bit`; language remains Heretic) |

> Not the stock Google IT weights. For **aligned** 31B IT + optional MTP, use
> [`../gemma4-server-mlx-31b/`](../gemma4-server-mlx-31b/).

---

## Quick start

```bash
cd gemma4-server-heretic-31b-mlx

# 1. Download Heretic language weights + stock vision (~2.3 GB) + graft + venv
./1_setup_download.sh

# 2. Start OpenAI-compatible server on :8080 (Kilo proxy on by default)
./2_start_mlx.sh

# 3. Kilo Code — launch from this directory so kilo.json is picked up
kilo
```

`1_setup_download.sh` pulls `mlx-community/gemma-4-31B-it-uncensored-heretic-4bit` (language-only upstream), then auto-downloads stock vision from `mlx-community/gemma-4-31b-it-4bit` (shard 4 only, or reuses `../gemma4-server-mlx-31b/` if present) and runs `graft_vision_from_stock.sh`. Use `--skip-vision` for a text-only install.

The Kilo steering proxy is **on by default** (Harmony bias, temp floor, tool repair, stall guards). Public API stays `http://localhost:8080/v1`; vllm-mlx listens on `:8090`. For raw vllm-mlx only:

```bash
./2_start_mlx.sh --no-proxy
```

vllm-mlx also enables **Gemma 4 native parsers by default**:

| Flag | Default | Purpose |
|------|---------|---------|
| `--tool-call-parser gemma4` | on | Turns `<\|tool_call>call:write{…}` into real OpenAI `tool_calls` |
| `--reasoning-parser gemma4` | on | Splits `<\|channel>thought…` out of visible content |

Without those, Kilo shows raw channel/tool markup as chat text (the “strange output” failure mode). Disable only for debugging:

```bash
./2_start_mlx.sh --no-auto-tool-choice --no-reasoning-parser
```

Clear a stuck port:

```bash
./2_start_mlx.sh restart
```

---

## Stack (current)

| Package | Version / range | Notes |
|---------|-----------------|--------|
| **vllm-mlx** | **≥ 0.4.0** (tested 0.4.0) | OpenAI-compatible server; includes `gemma4_text` dispatch + `logit_bias` |
| **mlx-vlm** | ≥ 0.5, &lt; 0.7 (tested 0.6.4) | Multimodal path; RoPE offset + thread-local streams |
| **mlx-lm** | ≥ 0.31 (tested 0.31.3) | Fast text path / chat CLI |
| **mlx** | (via deps, tested 0.32.0) | Apple Silicon runtime |
| **transformers** | ≥ 5.5, **&lt; 5.13** | Cap required by mlx-vlm 0.6.x |

Install / refresh deps:

```bash
./1_setup_download.sh --skip-download   # repair venv if moved + ensure deps
# or:
./venv/bin/python -m pip install -r requirements.txt
./apply_local_patches.sh
./check_upstream_patches.sh
```

If you **rename or move** this folder, `1_setup_download.sh` / `2_start_mlx.sh` detect a relocated venv and rewrite shebangs / `activate` paths automatically.

---

## Architecture

**Default (Kilo proxy on):**

```
Kilo Code / Continue.dev  ──→  :8080  gemma4_mlx_kilo_proxy.py  ──→  :8090  vllm-mlx
```

**With `--no-proxy`:**

```
Kilo Code / Continue.dev  ──→  :8080  vllm-mlx  ──→  gemma-4-31b-heretic-mlx-4bit
```

vllm-mlx is preferred over raw `mlx_vlm` for continuous batching, paged KV, and dual OpenAI/Anthropic APIs. MTP speculative decoding is **not** used here (Heretic 4-bit has no matching assistant MTP heads). For MTP experiments on **stock** 31B IT, see `../gemma4-server-mlx-31b/`.

---

## Scripts

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Create/repair venv, install deps, download 31B Heretic weights |
| `2_start_mlx.sh` | Start vllm-mlx (± Kilo proxy) on :8080; applies patches every start |
| `3_chat.sh` | Terminal chat without the HTTP server (`python -m mlx_lm chat`) |
| `4_configure_continue_dev.sh` | Write `~/.continue/config.json` for this model |
| `apply_local_patches.sh` | Version-aware patch apply into the venv |
| `check_upstream_patches.sh` | Verify upstream fixes + remaining local patches |
| `validate_model.py` | Refuse to start if weight shards are incomplete |

### Server options (`2_start_mlx.sh`)

```bash
./2_start_mlx.sh                    # proxy :8080 → vllm-mlx :8090 (default)
./2_start_mlx.sh --no-proxy         # raw vllm-mlx on :8080
./2_start_mlx.sh restart            # kill :8080/:8090, then start
./2_start_mlx.sh --batching         # multi-user (needs lots of RAM)
./2_start_mlx.sh --debug            # verbose proxy logs
./2_start_mlx.sh --enable-metrics
./2_start_mlx.sh --enable-auto-tool-choice
./2_start_mlx.sh --help
```

---

## Files

| File | Purpose |
|------|---------|
| `gemma4_mlx_kilo_proxy.py` | Default Kilo proxy: tool repair, fuzzy edit, Harmony bias, stall/empty-delta guards |
| `kilo.json` | Kilo Code provider config (31B Heretic, 32k context) |
| `requirements.txt` | Python dependency ranges |
| `DEVELOPER-bugs.md` | Known vllm-mlx / Gemma 4 issues and history |
| `patches/` | Local fixes (mostly legacy for vllm-mlx 0.3.x; see below) |
| `tests/` | Lean proxy pure-function suite (`./tests/run_tests.sh`) |

### Patches (vllm-mlx 0.4.0+)

On **vllm-mlx ≥ 0.4.0**, these fixes are **already upstream** and are **not** overwritten:

- `gemma4_text` fast TextModel dispatch (`text_model_from_vlm.py`)
- `logit_bias` on chat requests + server wiring

Still applied every start:

- `patches/gemma4_mllm.py` — Attention mask trim for BatchedEngine

Older full-file copies under `patches/` (`text_model_from_vlm.py`, `api/models.py`, `engine/simple.py`, `server.py`, legacy mlx-vlm `sample_utils` / `generate`) are kept for **vllm-mlx 0.3.x / mlx-vlm &lt; 0.5** only. `apply_local_patches.sh` skips them on the current stack so they cannot downgrade 0.4.0.

Verify:

```bash
./check_upstream_patches.sh          # expect PASS on gemma4_text + logit_bias + gemma4_mllm
./check_upstream_patches.sh --fetch  # also diff against pristine PyPI wheels
```

---

## Kilo Code

| Field | Value |
|-------|-------|
| Base URL | `http://localhost:8080/v1` |
| Model ID | `gemma-4-31b-heretic-mlx-4bit` |
| Reasoning | **Unchecked** (checked → infinite `<thinking>` loops) |
| API key | blank |
| Context | 32 768 (safer than 128k for 31B KV pressure) |
| Output | 8 192 |
| Agent temp | 0.35 (Gemma at `temperature=0` can stall after tool calls) |

```bash
# Per-project (this folder)
cd gemma4-server-heretic-31b-mlx
./2_start_mlx.sh   # terminal 1
kilo                     # terminal 2

# Or install globally
mkdir -p ~/.config/kilo
cp kilo.json ~/.config/kilo/kilo.jsonc
```

### Manual UI provider

1. Settings → Providers → Add custom provider  
2. Display name: `Gemma 4 31B Heretic`  
3. Base URL: `http://localhost:8080/v1`  
4. Model ID: **`gemma-4-31b-heretic-mlx-4bit`** (exact)  
5. Reasoning: **off**

Wrong model IDs (e.g. `gemma-4-31b-it-mlx-4bit` from the stock folder) will not match this server’s local weights.

---

## Continue.dev

[Continue.dev](https://continue.dev) is an open-source AI assistant for VS Code / VSCodium.
Unlike Kilo Code, it is **interactive**: it reads, explains, and suggests, and only edits
when you explicitly apply a change. Better fit for “review and suggest improvements”
where you want proposals, not autonomous implementation.

### Continue does not write files

Continue is a **chat interface**. It returns suggestions as text; it does not modify disk
even if you say “implement this.” To apply changes, use **Kilo Code** (or apply edits by hand).

```
Continue.dev  →  “review index.html and suggest improvements”
                 → numbered list of suggestions

Kilo Code     →  “implement suggestion #3 from Continue”
                 → files written
```

| Task | Continue.dev | Kilo Code |
|------|--------------|-----------|
| Review / suggest improvements | ✅ List of proposals | ⚠ Often auto-implements |
| Inline explanations / Q&A | ✅ Excellent | ✅ Good |
| Autocomplete | ✅ Built-in | ✅ Built-in |
| Long autonomous coding | ⚠ Weaker | ✅ Better (when not looping) |
| File edits without confirmation | ❌ Always asks | ⚠ May edit silently |

### One-time setup

```bash
cd gemma4-server-heretic-31b-mlx

# Model must already be downloaded (./1_setup_download.sh)
./4_configure_continue_dev.sh            # write ~/.continue/config.json (proxy :8080)
./4_configure_continue_dev.sh --install  # also install the VS Code / VSCodium extension
./4_configure_continue_dev.sh --direct   # only if you start the server with --no-proxy
./4_configure_continue_dev.sh --port 8081  # custom public port
```

`--install` uses `codium` or `code` if available. If neither CLI is on PATH:
`Cmd+Shift+P` → **Shell Command: Install 'code' command in PATH** (or `codium`).

### Daily workflow

```bash
./2_start_mlx.sh    # leave running
# Open VS Code / VSCodium → Cmd+L (Continue panel)
# Model: “Gemma 4 31B Heretic Uncensored (MLX)”
```

Reload if the editor was already open: `Cmd+Shift+P` → **Developer: Reload Window**.

### First-run wizard (VSCodium / older Continue)

Open VSX builds may show **Select LLM Provider** instead of reading `config.json`:

1. Choose **Other OpenAI-compatible API** (Local + Open-Source)
2. Expand **▶ Advanced (optional)**
3. Server URL: `http://localhost:8080/v1`
4. Model preset: **Llama2 or CodeLlama** is fine (chat template only; works with Gemma)
5. Ensure `./2_start_mlx.sh` is running before chatting

Older Continue used `config.py`; newer versions use `config.json` (what the script writes).

### Manual `~/.continue/config.json`

```json
{
  "models": [
    {
      "title": "Gemma 4 31B Heretic Uncensored (MLX)",
      "provider": "openai",
      "model": "gemma-4-31b-heretic-mlx-4bit",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "local"
    }
  ]
}
```

`apiBase` should match the public server port (default proxy on `:8080`). With
`./2_start_mlx.sh --no-proxy`, the same URL is raw vllm-mlx.

### Slash commands & autocomplete

If the configure script added them, use `/review`, `/improve`, `/explain` on selected code
(suggestions only — no auto-apply). Edit `~/.continue/config.json` → `slashCommands` to customize.

Autocomplete hits the same model at lower temperature. If 31B is too slow on keystrokes,
disable tab autocomplete (`continue.enableTabAutocomplete`) or point it at a smaller local
model (e.g. Qwen 3.6 via mtplx on `:8765`).

### When to use which

- **Continue** — review lists, inline explain, you control every edit
- **Kilo** — multi-step implementation, autonomous file edits, agent tool loops

---

## Smoke tests

```bash
# List models
curl -s http://localhost:8080/v1/models | head

# Chat
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-heretic-mlx-4bit",
    "messages": [{"role": "user", "content": "Say hi in one sentence."}],
    "temperature": 0.35,
    "max_tokens": 64
  }'

# Interactive (no server)
./3_chat.sh

# Proxy pure-function tests (no GPU)
./tests/run_tests.sh
```

---

## Memory & stability (31B-specific)

| Symptom | Cause / fix |
|---------|-------------|
| Exit **134** / Metal “Insufficient Memory” | Uncatchable C++ abort on GPU OOM. Weights ~20 GB fit; **KV + concurrent Kilo streams** push over the limit. Restart; lower context in `kilo.json` (e.g. 16384); close other MLX servers. |
| Port 8080 “dead” / wrong model answers | Stale sibling server still bound. **Only one** of: this folder, `gemma4-server-mlx-31b`, diffusion, etc. on `:8080`. Use `./2_start_mlx.sh restart`. |
| OOM on 64 GB Macs | 31B is aimed at **80 GB+** unified memory. Close other MLX servers and lower context in `kilo.json`. |
| Continuous batching | Default **off** for single-user throughput. Enable `--batching` only if you need multi-client and have headroom (128 GB+). |
| Reasoning checked in Kilo | Can encourage long CoT loops — leave Kilo’s Reasoning UI **off**. Server-side `--reasoning-parser gemma4` (default) is different: it *strips* channel thought into `reasoning_content` so tools parse cleanly. |
| Raw `<\|channel>thought` / `<\|tool_call>` in the chat | vllm-mlx started without gemma4 parsers (old default). Restart with current `./2_start_mlx.sh` (parsers on by default). |

```bash
# Free :8080 cleanly
./2_start_mlx.sh restart

# See who owns the port
lsof -i :8080
```

---

## What “Heretic / uncensored” means

The Heretic line is an abliterated / decensored fine-tune (refusal directions reduced), not Google’s stock safety-aligned IT weights. Tags on the card typically include `heretic`, `uncensored`, `decensored`, `abliterated`. Use responsibly — there is little built-in safety net.

Stack: **vllm-mlx 0.4+** (Gemma 4 text path + logit_bias upstream) plus a small local **gemma4_mllm** mask-trim patch. The Kilo proxy (default on) adds agent-side guards. See **`DEVELOPER-bugs.md`** and upstream [vllm-mlx#590](https://github.com/waybarrios/vllm-mlx/issues/590).

---

## Related projects

| Folder | Model | Stack |
|--------|-------|-------|
| **This** | 31B Heretic 4-bit | vllm-mlx 0.4+ + Kilo proxy (default on) |
| [`../gemma4-server-mlx-31b/`](../gemma4-server-mlx-31b/) | 31B **IT** 4-bit (aligned) | mlx-lm / mlx-vlm + optional MTP |
| [`../qwen3-6-27b-coder-mtplx/`](../qwen3-6-27b-coder-mtplx/) | Qwen3.6 27B | mtplx MTP coding |

Broader guides: [README.md](../README.md), [README-diffusiongemma4.md](../README-diffusiongemma4.md).
