# Gemma 4 31B IT AtomicChat — MLX + Kilo Code (2026-07-15)

Run **Gemma 4 31B IT (4-bit)** from **[AtomicChat/gemma-4-31B-it-MLX-4bit](https://huggingface.co/AtomicChat/gemma-4-31B-it-MLX-4bit)** locally via **mlx-lm** by default, or opt into **mlx-vlm** with **MTP speculative decoding**.

This folder is the **2026-07-15** stock stack update: Google’s Gemma 4 chat-template / tool-calling refresh, re-quantized by AtomicChat with **mlx-lm 0.31.3**. It replaces the older `mlx-community/gemma-4-31b-it-4bit` snapshot (last rebuilt ~2026-07-05 with the pre-fix template).

| | |
|--|--|
| **HF target** | `AtomicChat/gemma-4-31B-it-MLX-4bit` |
| **Local dir** | `gemma-4-31b-it-atomicchat-mlx-4bit` (~17 GB) |
| **Modalities** | **Text only** (not multimodal) |
| **Kilo model ID** | `openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit` |
| **API** | `http://localhost:8080/v1` |
| **Why this quant** | Ships Google’s updated `chat_template.jinja` (null handling, reasoning preservation, turn-tag balance, tool-call fixes) |

## Models & modalities

| Weight / role | Local dir / HF | Modalities | Role |
|---------------|----------------|------------|------|
| **Target (this stack)** | `gemma-4-31b-it-atomicchat-mlx-4bit` · `AtomicChat/gemma-4-31B-it-MLX-4bit` | **Text only** | Full chat model (language + updated chat template). Default server: `mlx_lm.server`. |
| **MTP assistant (optional)** | `gemma-4-31b-it-assistant-mlx-bf16` · `mlx-community/gemma-4-31B-it-assistant-bf16` | **Text draft only** | Speculative drafter (~1 GB). Not a chat model; not vision. Only used with `./2_start_mlx.sh --with-mtp`. Skip for reliability (`--skip-mtp-download`). |
| **Stock multimodal (elsewhere)** | `mlx-community/gemma-4-31b-it-4bit` | **Text + image** | Multimodal quant; vision graft source for Heretic. **Not** this AtomicChat text package. |
| **Uncensored + vision** | Heretic / JANG stacks under `uncensored/` | **Text + image** | See monorepo [README-models.md](../../README-models.md) modality matrix. |
| **Diffusion VLM** | `censored/diffusiongemma4-26b-a4b-mlx/` | **Text + image** | Image Q&A / research — not this folder. |

**Do not confuse engines with modalities:** optional `--with-mtp` runs `mlx_vlm.server` for **text** speculative decode only. AtomicChat still has **no image understanding**. Prefer default `mlx_lm.server` for agents.

Full repo matrix (every stack): [README-models.md — Modalities](../../README-models.md#modalities-text-only-vs-multimodal).

## Quick start

```bash
# 1. Download target (~17 GB) + MTP assistant (~1 GB) and install deps
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
Kilo Code ──→ mlx_lm.server ──→ gemma-4-31b-it-atomicchat-mlx-4bit
```

**With MTP (`./2_start_mlx.sh --with-mtp`):**

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8080/v1   (OpenAI-compatible)
  mlx_vlm.server
      │  speculative decode: --draft-kind mtp
      ├── target:  gemma-4-31b-it-atomicchat-mlx-4bit  (~17 GB, 4-bit)
      └── drafter: gemma-4-31b-it-assistant-mlx-bf16     (~1 GB, shares K/V)
```

MTP uses Google's [Gemma 4 assistant drafter](https://ai.google.dev/gemma/docs/mtp/mtp) built into mlx-vlm. Measured greedy speedups on Apple Silicon are up to **~2.3×** on the 31B model at block size 4 (decode only; long agent prefill is unchanged). It is opt-in because the non-MTP server is simpler and more stable for Kilo agent work.

### MTP pairing (target + assistant)

`gemma-4-31b-it-assistant-mlx-bf16` is the official MTP drafter for **`google/gemma-4-31B-it`**. It is intended to pair with this AtomicChat 4-bit target:

| | Target `gemma-4-31b-it-atomicchat-mlx-4bit` | Assistant `gemma-4-31b-it-assistant-mlx-bf16` |
|---|---|---|
| Role | Full model (quality) | Small drafter (~4 layers, ~1 GB) |
| Quantization | 4-bit (memory) | bf16 (size is small either way) |
| Hidden size | 5376 | `backbone_hidden_size` 5376 (must match) |
| Vocabulary | 262,144 | 262,144 (same tokenizer) |

The assistant does not replace the target — it only proposes tokens; the 31B model verifies them and shares K/V. At `temperature=0`, output should match non-MTP greedy generation.

## Files

| File | Purpose |
|---|---|
| `1_setup_download.sh` | Create venv, install mlx/mlx-lm/mlx-vlm, download AtomicChat target + MTP assistant |
| `2_start_mlx.sh` | Start Kilo proxy on `:8080` → MLX engine on `:8090`; runs `test_harness.py --gate` by default |
| `gemma4_kilo_proxy.py` | Harness proxy: thinking off, compaction tool-strip, empty-tool recovery |
| `research-tool.py` | Agent research: check/grep/repos/**paths** (dwc3 trees) — not raw curl |
| `test_harness.py` | Outside-Kilo contract tests + `--gate` post-start check (not a Kilo emulator) |
| `kilo_lite_loop.py` | Multi-step tool-loop shape tests only (not full Kilo) |
| `apply_local_patches.sh` | Copy `patches/` into the venv on each start (survives `pip install -U`) |
| `kilo.json` | Kilo Code config for this project (model + Gemma temps + harness) |
| `../../kilo.json` | **Monorepo global** Kilo (all providers + default model) — install with `../../install_kilo.sh` |
| `patches/sample_utils.py` | top-p fix for 3-D MTP verify logits |
| `patches/app.py` | Map API model id `default_model` → CLI `--model` (mlx_lm convention) |
| `patches/models/gemma4/language.py` | MTP rollback when `accepted` is a Python list |
| `patches/patch_generation.py` | Close `BatchGenerator` after MTP errors (avoids hung server) |

## Kilo Code

Launch Kilo from **this directory** so the **project** `kilo.json` is picked up:

```bash
cd gemma4-server-atomicchat-mlx-31b-2026-07-15
./2_start_mlx.sh    # terminal 1
kilo                # terminal 2 — same directory
```

### Global vs project `kilo.json`

| Config | Path | Role |
|--------|------|------|
| **Project** | `./kilo.json` (this folder) | AtomicChat model ID, Gemma agent temps (0.35), answer-then-halt harness |
| **Monorepo global** | [`../../kilo.json`](../../kilo.json) | All stack providers + default model + shared harness |
| **Installed global** | `~/.config/kilo/kilo.jsonc` | What Kilo uses when no project file wins |

When you change harness prompts or the default model for **everyone**, update **`../../kilo.json`** (monorepo root) **and** install it:

```bash
# from this folder:
../../install_kilo.sh
# or from monorepo root:
# ./install_kilo.sh
```

Keep this folder’s `kilo.json` in sync for Gemma-specific temps/prompts when editing agent rules for AtomicChat sessions.

| Setting | Value | Why |
|---------|-------|-----|
| Model ID | `gemma-4-31b-it-atomicchat-mlx-4bit` | Recommended; matches the local server model |
| Model ID (alt.) | `default_model` | Also works — mapped to the same local weights at startup |
| Context limit | 32k | Faster prefill; fits agent sessions without OOM |
| Output limit | 4k | Room for edits; leaves headroom for compaction |
| `compaction.reserved` | 12288 | Start auto-compact earlier so summaries fit |
| `agent.compaction` | short plain-text prompt | Stop Goal/Progress template bloat on summarize turns |
| `tool_output` | 200 lines / 16 KiB | Cap tool dumps before they fill the window |
| Agent temp | 0.35 | Gemma at `temperature=0` stalls after tool calls |
| Reasoning | **Off** in UI | Checked → infinite `<thinking>` loops |
| Public API | `:8080` via `gemma4_kilo_proxy` | Strips tools on compaction (default) |

### Harness reliability (ContextOverflow / compaction)

Long Kilo sessions used to hit:

`ContextOverflowError: Compaction exhausted: context still exceeds model limits after 3 attempts`

Cause: auto-compaction still sent `tools`, so Gemma kept exploring instead of writing a short summary (`pruned=0` in Kilo logs). Fix in this stack:

| Layer | What it does |
|-------|----------------|
| `gemma4_kilo_proxy.py` (default) | **Thinking OFF** so Kilo gets `content`; **never strip tools** when `tools` + `tool_choice≠none` (agent turns); compaction only for `tool_choice=none` / real summary user turns; remap reasoning→content; truncate huge tool results |
| `kilo.json` `compaction.reserved` | Leaves ~12k tokens free so compact runs before the window is full |
| `agent.compaction` prompt | Forces ≤~40 line plain text — no Goal/Progress spam |
| Tighter `tool_output` | Smaller dumps so prune/truncation has less to keep |

```bash
./2_start_mlx.sh                 # proxy :8080 → engine :8090 (default)
./2_start_mlx.sh --no-proxy      # raw mlx on :8080 (curl smoke only)
./2_start_mlx.sh restart         # clear both ports, then start
./2_start_mlx.sh --debug         # proxy DEBUG + harness traces
./2_start_mlx.sh --no-harness-log
curl -s http://127.0.0.1:8080/healthz
# Standalone harness tests (outside Kilo — contract gate, not a full Kilo emulator):
python3 test_harness.py --gate       # post-start gate (also run by ./2_start_mlx.sh by default)
python3 test_harness.py              # full unit+live+fringe
python3 test_harness.py --strict
python3 test_harness.py --unit-only
python3 test_harness.py --quick
python3 test_harness.py --fringe-only
# Multi-step *loop shape* only (still not full Kilo — no session/compaction UI):
python3 kilo_lite_loop.py
python3 kilo_lite_loop.py --strict
# Start options:
./2_start_mlx.sh --no-harness-gate   # skip gate
./2_start_mlx.sh --harness-lite      # also run kilo_lite_loop after gate
# tail harness traces (no message bodies):
tail -f /tmp/gemma4_kilo_proxy.log
```

| Tool | Emulates Kilo? | Purpose |
|------|----------------|---------|
| `test_harness.py` | **No** | Proxy + wire contract regressions |
| `kilo_lite_loop.py` | **Loop shape only** | Multi-step tool rounds + empty-tool recovery |
| Full Kilo TUI/session | Real product | Permissions, compaction UI, prune — not reimplemented |

**Empty tool recovery:** if the latest tool result is empty, the proxy injects a system nudge: do not write a revised plan — next action must be a local `ls`/`glob`/`grep`/`read`.

Reload Kilo after changing `kilo.json`. After a failed overflow session, **start a new chat** (the dead session cannot recover).

**Continue must act:** prompts now require that “continue” / “continue if you have next steps” runs tools for the next unfinished step (list/read the named path) instead of rewriting Goal/Progress templates. If an old session only re-summarized, start a **new chat** with a short handoff: goal + next path to open.

> For heavier agent tuning (fuzzy edits, Heretic model, Harmony bias), use
> `gemma4-server-heretic-31b-mlx`. For coding speed on Qwen, use `qwen3-6-27b-coder-mtplx`.

## Options

### Enable MTP

```bash
./2_start_mlx.sh --with-mtp
```

Uses `mlx_vlm.server` plus the Gemma 4 assistant drafter. This can improve decode throughput, but Kilo wall-clock speed also includes prompt prefill, tool calls, edits, and file IO. If MTP fails to load this AtomicChat quant under mlx-vlm, stay on the default `mlx_lm.server` path.

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
mlx_lm.generate --model gemma-4-31b-it-atomicchat-mlx-4bit --prompt "Hello, who are you?"
```

MTP CLI test (loads both models):

```bash
mlx_vlm.generate --model gemma-4-31b-it-atomicchat-mlx-4bit \
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
  -d '{"model":"gemma-4-31b-it-atomicchat-mlx-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'

# Chat — default_model alias (Kilo / Cursor style)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

## Changelog (this folder)

| Date | Change |
|------|--------|
| **2026-07-15** | Switched target to `AtomicChat/gemma-4-31B-it-MLX-4bit` (Google chat-template / tool-calling fixes). Folder: `gemma4-server-atomicchat-mlx-31b-2026-07-15` (was `gemma4-server-mlx-31b`). Kilo model id: `gemma-4-31b-it-atomicchat-mlx-4bit`. |
| earlier | Stock stack on `mlx-community/gemma-4-31b-it-4bit` + optional MTP |

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

Fix: restart so patches apply (`./2_start_mlx.sh restart`). You should see `Using cached model: gemma-4-31b-it-atomicchat-mlx-4bit` on chat requests, not `Loading model from: default_model`.

**`model not found` in Kilo**

Use `openai-compatible/gemma-4-31b-it-atomicchat-mlx-4bit` as in `kilo.json`, or ensure the OpenAI provider `baseURL` is `http://localhost:8080/v1` while `./2_start_mlx.sh` is running.

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
| ~5 min uptime then crash | Weights fit (~17 GB), but **KV cache + MTP + concurrent requests** pushed unified memory over the limit |

The 31B 4-bit target needs ~17 GB for weights; the MTP assistant adds ~1 GB; **long or concurrent agent sessions** add much more for KV cache. On **24 GB** Macs (common for `Mac17,7`-class hardware), 32k context + MTP + parallel tool rounds is often too much.

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
