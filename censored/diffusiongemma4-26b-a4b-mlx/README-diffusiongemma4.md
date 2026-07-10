# DiffusionGemma 26B A4B IT — MLX Server (`diffusiongemma4-26b-a4b-mlx/`)

Run Google's **DiffusionGemma** (discrete diffusion VLM) locally on Apple Silicon via **mlx-vlm**.
Text + image in, text out. OpenAI-compatible API on port **8080**.

Upstream model: [`mlx-community/diffusiongemma-26B-A4B-it-bf16`](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-bf16) (~52 GB bf16).

---

## What is DiffusionGemma?

DiffusionGemma is a **diffusion-based** language model (not autoregressive). It denoises a token
canvas block-by-block using Google's Entropy Bound (EB) sampler. The MLX port runs through
`mlx_vlm` with multimodal support (Gemma 4 vision encoder).

| | DiffusionGemma (this project) | Gemma 4 Heretic (vllm-mlx) |
|--|--------------------------------|----------------------------|
| Architecture | Discrete diffusion VLM | Autoregressive MoE VLM |
| Engine | `mlx_vlm.server` | `vllm-mlx` |
| Size (bf16) | ~52 GB | ~18 GB (4-bit 26B) |
| Image input | Yes | Yes |
| Agentic coding (Kilo) | Weak — narrates instead of tool-calling | **Recommended** |
| Best use | Multimodal Q&A, image description, experimentation | Daily coding agent, tool use |

> **For Kilo Code / agentic coding**, use
> [gemma4-server-uncensored-31b-mlx/README.md](gemma4-server-uncensored-31b-mlx/README.md) or
> [qwen3-6-27b-coder-mtplx/README.md](qwen3-6-27b-coder-mtplx/README.md) instead.
> DiffusionGemma works for simple tool calls (`ls`, `cat`) but often explains instead of acting
> on complex agent prompts (todo lists, multi-file refactors).

### Preventing Kilo agent stalls

If Kilo shows raw `call:todowrite{...}` text and the turn ends with todos remaining:

1. **Restart the server** so streaming patches apply: `./2_run_mlx.sh restart server`
2. **Reload VS Code** after any `kilo.json` change
3. **Use Gemma 4 for coding** — put a project `kilo.json` pointing at
   `openai-compatible/gemma-4-26b-heretic-mlx-4bit` (see `web-airports/kilo.json` example)
4. **Use DiffusionGemma only for vision** — screenshots, UI debugging with image attach
5. Patches in `patches/responses_state.py` now:
   - Hide `call:fn{...}` from streamed chat text (clients get `tool_calls` at end)
   - Parse multiple tool calls per turn (`todowrite` then `read`)

---

## Requirements

- Apple Silicon Mac with **64 GB+** unified memory (model is ~52 GB bf16)
- Python 3.12+ (venv uses system `python3`)
- `hf` CLI (installed into project venv by setup script)
- ~55 GB free disk for weights

---

## Quick Start

```bash
cd diffusiongemma4-26b-a4b-mlx

# 1. Download weights + install deps (~52 GB, one-time)
./1_setup_download.sh

# 2. Smoke test (one-shot text generation)
./2_run_mlx.sh

# 3. Start OpenAI-compatible API for Kilo / curl
./2_run_mlx.sh server
```

Server endpoint: **http://127.0.0.1:8080/v1**

Verify:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `1_setup_download.sh` | Create venv, install deps, download model shards |
| `2_run_mlx.sh` | Generate, chat, or server mode |
| `apply_local_patches.sh` | Apply mlx-vlm patches (called automatically by `2_run_mlx.sh`) |
| `validate_model.py` | Verify all weight shards are present |
| `kilo.json` | Per-project Kilo Code config |

### `1_setup_download.sh` options

```bash
./1_setup_download.sh                 # download + install (default)
./1_setup_download.sh --skip-download # deps only
./1_setup_download.sh --force         # re-download all shards
```

### `2_run_mlx.sh` modes

```bash
./2_run_mlx.sh                              # one-shot text (default prompt)
./2_run_mlx.sh chat                         # interactive terminal chat
./2_run_mlx.sh server                       # OpenAI API on :8080
./2_run_mlx.sh restart server               # kill :8080 occupant, start fresh

# Image + text
./2_run_mlx.sh --image photo.jpg --prompt "Describe this image."

# Thinking / reasoning mode
./2_run_mlx.sh --enable-thinking --prompt "Solve step by step."

# Diffusion sampler tuning (Google EB defaults)
./2_run_mlx.sh --max-denoising-steps 48 --block-length 256 --threshold 0.1
```

Server defaults:

- **Port:** 8080 (`--port` to override)
- **max-tokens:** 4096 for server mode (Kilo/agent use; generate/chat use 512)
- **Patches:** applied on every start

---

## Kilo Code Setup

### Global config (all projects)

```bash
mkdir -p ~/.config/kilo
cp kilo.json ~/.config/kilo/kilo.jsonc
# or from repo root:
cp ../kilo.json ~/.config/kilo/kilo.jsonc
```

### Per-project override

Place `kilo.json` in the project root, or use `diffusiongemma4-26b-a4b-mlx/kilo.json` when
working from that directory.

| Field | Value |
|-------|-------|
| Model | `diffusiongemma/diffusiongemma-26b-a4b-it-bf16` |
| Base URL | `http://localhost:8080/v1` |
| API key | `local` (any string; server does not validate) |
| Image support | **On** — `attachment: true` + `modalities.input: ["text", "image"]` |
| Reasoning | **Unchecked** in Kilo UI |
| Context limit | 32768 (configured in `kilo.json`) |

### Image upload in Kilo Code (7.3.x)

#### `kilo.json` is already correct — no fix needed for vision

The provider entry sets everything Kilo needs:

```json
"attachment": true,
"modalities": { "input": ["text", "image"], "output": ["text"] }
```

If `./2_run_mlx.sh --image …` describes your screenshot correctly but Kilo shows **no paperclip**,
the backend and config are fine. The gap is **Kilo's chat UI**, not `kilo.json`.

#### CLI vs Kilo — two different paths

| | CLI (`./2_run_mlx.sh --image`) | Kilo Code |
|--|---------------------------|-----------|
| Engine | `mlx_vlm.generate` (direct) | `mlx_vlm.server` on `:8080` |
| Uses `kilo.json` | No | Yes |
| Image input | `--image` flag | Paste / Shift+drag / `@` Attach file |
| Server required | No | **Yes** — run `./2_run_mlx.sh server` first |

Verified locally (Jun 2026): a Desktop screenshot via CLI returns an accurate description
(NeoPad tabs, 2026 notes, status bar). Same model + image config works; Kilo just needs the
server running and a non-obvious attach method.

#### No paperclip in 7.3.40 — regression, not missing config

**Kilo Code 7.3.40 has no toolbar paperclip / attach button.** This is a known regression from
Kilo 5.x and [legacy kilocode](https://github.com/Kilo-Org/kilocode-legacy), which had a visible
image picker in the chat input. Kilo 4.24 also documented a paperclip / "Add Image" flow
([blog post](https://blog.kilo.ai/p/kilo-code-4240-image-attachments)).

The 7.3.40 extension's own docs (`docs/features/file-attachments.md`) still list a toolbar
attach button as **remaining work** (P2). GitHub issue
[#6078](https://github.com/Kilo-Org/kilocode/issues/6078) was closed "Done" (Apr 2026), but the
toolbar file-picker is **still absent** — partial work (paste, Shift+drag) may have been mistaken
for completion.

**Upstream tracking** (local repro + priority rationale posted Jun 2026):

| Issue | Status | Topic |
|-------|--------|-------|
| [#8641](https://github.com/Kilo-Org/kilocode/issues/8641) | Open | Attach button missing — regression from 5.x (**primary thread**) |
| [#8277](https://github.com/Kilo-Org/kilocode/issues/8277) | Open | Attach file button missing from chat interface |
| [#8451](https://github.com/Kilo-Org/kilocode/issues/8451) | Open | Cannot drag and drop images into chat input |
| [#9384](https://github.com/Kilo-Org/kilocode/issues/9384) | Open | Prompt attachments and file-part parity |
| [#6078](https://github.com/Kilo-Org/kilocode/issues/6078) | Closed | Toolbar picker still missing despite "Done" |

#### How to attach images today (7.3.40)

After updating `kilo.json` and starting the server:

1. **Reload VS Code** — Cmd+Shift+P → **Developer: Reload Window**
2. Select model **`diffusiongemma/diffusiongemma-26b-a4b-it-bf16`** in the Kilo picker
   (not `mtplx/qwen3.6-27b-mtplx` — text-only in our `kilo.json`)
3. Attach an image using one of these:
   - **Paste** — copy a screenshot to the clipboard, focus the chat input, **Cmd+V**
   - **Drag and drop** — hold **Shift** while dragging an image onto the chat input
     (VS Code disables webview drops unless Shift is held; see
     [vscode#182449](https://github.com/microsoft/vscode/issues/182449))
   - Type **`@`** → **Attach file** and pick a PNG/JPEG/GIF/WebP image

Supported formats: PNG, JPEG, GIF, WebP (and PDF). You should see a preview chip above the
input before sending. If paste fails, confirm the active model supports image input.

#### CLI alternative (bypasses Kilo UI entirely)

```bash
./2_run_mlx.sh --image test-images/red-square.png --prompt "What color is this image?"
./2_run_mlx.sh --image ~/Desktop/"Screenshot 2026-06-10 at 23.21.50.png" --prompt "What do you see?"
```

#### Server must be running for Kilo

```bash
cd diffusiongemma4-26b-a4b-mlx
./2_run_mlx.sh server          # or ./2_run_mlx.sh restart server
curl http://127.0.0.1:8080/v1/models   # should list diffusiongemma-26b-a4b-it-bf16
```

Reload VS Code after config changes (Cmd+Shift+P → **Developer: Reload Window**).

---

## API Examples

### Text chat

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma-26b-a4b-it-bf16",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 256
  }'
```

### Tool calling (Kilo format)

The model emits `call:bash{command:ls -F}` text. A local patch in `patches/responses_state.py`
converts this to structured `tool_calls` for Kilo.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma-26b-a4b-it-bf16",
    "messages": [{"role": "user", "content": "List files"}],
    "tools": [{"type": "function", "function": {
      "name": "bash",
      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
    }}],
    "max_tokens": 128
  }'
```

Expect `finish_reason: "tool_calls"` with `{"command": "ls -F"}` in the response.

---

## Local Patches

`mlx_vlm` only parsed tool calls wrapped in `<|tool_call>…` markers. DiffusionGemma emits bare
`call:fn{…}` syntax. This project patches `mlx_vlm/server/responses_state.py` on every
`2_run_mlx.sh` invocation.

```
patches/responses_state.py  →  venv/.../mlx_vlm/server/responses_state.py
```

Re-applied automatically; survives `pip install --upgrade` only until the next `./2_run_mlx.sh`.

---

## Architecture

```
Kilo Code / curl  →  http://127.0.0.1:8080/v1  →  mlx_vlm.server  →  DiffusionGemma 26B bf16
```

Model ID exposed by the API: `diffusiongemma-26b-a4b-it-bf16` (local directory name).

Only **one** server should own port 8080. If you also run Gemma 4 or another MLX server,
stop the other process first or use a different port.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `address already in use` on :8080 | `./2_run_mlx.sh restart server` — or another MLX server is running; stop it first |
| `./2_run_mlx.sh server` exits immediately with "already running" | Server is fine — check `curl http://127.0.0.1:8080/health` |
| Kilo shows `call:bash{…}` as text, no tool run | Restart server (`./2_run_mlx.sh restart server`) so patches apply |
| Kilo hangs or truncates mid-response | Server now uses `--max-tokens 4096`; restart if an old 512-token instance is running |
| Kilo narrates instead of running commands | Model limitation — switch to Gemma 4 or Qwen3.6 for agent work |
| `Cannot connect to API` | Start server: `./2_run_mlx.sh server` |
| `model not found` in Kilo | Model ID must be exactly `diffusiongemma-26b-a4b-it-bf16` |
| Incomplete weights | `./1_setup_download.sh` or `python3 validate_model.py diffusiongemma-26b-a4b-it-bf16` |
| GPU memory error on long prompts | `./2_run_mlx.sh server --prefill-step-size 256` |
| Port conflict with Gemma 4 server | Stop the other server, or `./2_run_mlx.sh server --port 8081` and update `kilo.json` `baseURL` |
| No paperclip in Kilo chat | Expected in 7.3.x — not a `kilo.json` bug; use paste, Shift+drag, or `@` Attach file |
| CLI vision works, Kilo does not | Kilo needs `./2_run_mlx.sh server` running; CLI uses direct `generate` mode |
| Picked wrong model in Kilo | Use `diffusiongemma/diffusiongemma-26b-a4b-it-bf16`, not mtplx/Qwen (no image in config) |

### Check what's on port 8080

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl http://127.0.0.1:8080/health
```

---

## Project Layout

```
diffusiongemma4-26b-a4b-mlx/
├── 1_setup_download.sh      # download + venv setup
├── 2_run_mlx.sh                 # generate / chat / server
├── apply_local_patches.sh   # mlx-vlm tool-call patch
├── validate_model.py        # shard integrity check
├── kilo.json                # Kilo Code provider config
├── requirements.txt
├── patches/
│   └── responses_state.py   # bare call:fn{…} tool parser fix
├── diffusiongemma-26b-a4b-it-bf16/   # weights (gitignored, ~52 GB)
└── venv/
```

Weights are gitignored. Re-download with `./1_setup_download.sh` on a fresh clone.

---

## See Also

- [README.md](README.md) — overview of all local LLM servers
- [gemma4-server-uncensored-31b-mlx/README.md](gemma4-server-uncensored-31b-mlx/README.md) — Gemma 4 Heretic via vllm-mlx (preferred for agents)
- [qwen3-6-27b-coder-mtplx/README.md](qwen3-6-27b-coder-mtplx/README.md) — Qwen3.6 coding with MTP
- [Hugging Face model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)