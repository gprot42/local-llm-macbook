# DeepSeek V4 — DwarfStar (ds4)

**Great for coding.** Run **DeepSeek V4 Flash** with [antirez/ds4](https://github.com/antirez/ds4) — a **native Metal** inference engine (not MLX, not llama.cpp). Strong multi-file agent work, real tool calling (DSML), long context, and solid tokens/s on M-series unified memory.

Tuned for **M5 Max with 128 GB** unified memory (this machine). Status: 🟢 working.

**Harness (required for Kilo agents):** `./2_start_ds4.sh` starts **ds4-server on `:18083`** plus **`ds4_kilo_proxy` on `:8083`**. The proxy defaults **thinking OFF**, floors `max_tokens`, and soft-repairs truncated tool JSON — without it, Kilo often hits broken tool args and mid-fix aborts.

### When to use this model

**DeepSeek V4 Flash via ds4** (native GGUF, ~81 GB q2-imatrix) — **great for coding agents**

- **Best for coding:** hard multi-file / SWE-style work in Kilo; deep code review; agent tool loops (`read`, edits, skills) when quality beats snappy iteration; long context (tens of k tokens); OpenAI **or** Anthropic APIs
- **Also good:** interactive CLI, native `ds4-agent`, multi-turn chat with thinking
- **Not ideal:** ultra-fast tool loops (prefer [Qwen 3.6 mtplx](../qwen3-6-27b-coder-mtplx/)); arbitrary third-party GGUFs (this engine only loads antirez’s published layout)

### Observed coding performance (M5 Max 128 GB, q2-imatrix)

From a live Kilo agent session (explore + many parallel `read` tools):

| Metric | Typical | Notes |
|--------|---------|--------|
| Prefill | ~200–415 t/s | Slows as prompt grows |
| Decode | ~25–30 t/s early; ~16 t/s near ~50k ctx | Expected long-context drop |
| Tools | Tool calling works | Uncapped parallel `read` storms balloon context — **mitigated in `kilo.json`** |
| Load | ~12 s Metal residency for ~83 GiB mapped | One-time per process start |

### ds4 vs MLX DeepSeek in this repo

| | **This stack (ds4)** | [deepseek-v4-flash-2bit-dq-mlx](../deepseek-v4-flash-2bit-dq-mlx/) |
|--|----------------------|---------------------------------------------------------------------|
| Engine | Native C + Metal (`ds4-server`) | Python `mlx-lm` + custom OpenAI wrapper |
| Weights | `antirez/deepseek-v4-gguf` (~81 GB q2) | `mlx-community/DeepSeek-V4-Flash-2bit-DQ` (~97 GB) |
| Port | **8083** | 8082 |
| APIs | OpenAI chat/completions + Responses + Anthropic messages | OpenAI chat/completions |
| Extra | SSD streaming, KV-on-disk, `ds4-agent`, distributed | MLX harness / prefill tricks |

Do **not** load both huge DeepSeek stacks at once on 128 GB.

---

## Recommended model

| Target | Disk | Fits 128 GB? | Notes |
|--------|------|--------------|-------|
| **q2-imatrix** (default) | ~81 GB | Yes | Best default for 96/128 GB |
| q2-q4-imatrix | ~98 GB | Tight | Last 6 layers q4 — higher quality |
| q4-imatrix | ~153 GB | No (full resident) | ≥256 GB, or try `--ssd-streaming` |
| pro-q2-imatrix | ~430 GB | Streaming only | Experimental on 128 GB |

Only GGUFs from [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) work.

---

## Quick start

```bash
cd censored/deepseek-v4-flash-ds4

# Clone engine + Metal build + download ~81 GB q2-imatrix (one-time)
./1_setup_download.sh

# Higher-quality mix (~98 GB) instead:
# ./1_setup_download.sh q2-q4-imatrix

# Start OpenAI/Anthropic-compatible API on port 8083
./2_start_ds4.sh
```

Build only (weights already present):

```bash
./1_setup_download.sh --skip-download
```

Force rebuild binaries:

```bash
./1_setup_download.sh --rebuild --skip-download
```

Re-download from scratch (deletes local GGUF + `.part`):

```bash
./1_setup_download.sh --force-download
```

Downloads are **resumable** and auto-retried. Flash GGUFs use `download_gguf.py` (HTTP
Range resume, up to 200 attempts with backoff) so `Connection reset by peer` mid-transfer
no longer aborts the whole setup. If a transfer stops you will see a `gguf/*.gguf.part`
file — re-run `./1_setup_download.sh` (same target). Setup also:

- Preflights git / curl / make / Xcode CLT
- Checks free disk before continuing
- Retries clone + download on network errors
- Rebuilds once after `make clean` if the Metal build fails
- Validates finished GGUFs (exact HF size + magic) via `validate_model.py`

`./2_start_ds4.sh` refuses to start on incomplete weights, is idempotent when the
server is already healthy, and waits longer for graceful stop on large loads.

---

## Kilo Code

```bash
# This stack only (includes agent context discipline)
cp kilo.json /path/to/your/project/kilo.json
cp AGENTS.md /path/to/your/project/AGENTS.md   # optional but recommended

# Global install for this machine:
cp kilo.json ~/.config/kilo/kilo.jsonc

# Or use the monorepo root (all providers, including ds4):
#   cp ../../kilo.json ~/.config/kilo/kilo.jsonc
# then set "model": "ds4/deepseek-v4-flash"
```

| Field | Value |
|-------|-------|
| Base URL | `http://127.0.0.1:8083/v1` (use `127.0.0.1`, not `localhost`) |
| Provider / Model | `ds4/deepseek-v4-flash` |
| Timeouts | 900s / header 180s / chunk 300s (large prefill) |

DeepSeek V4 sampling defaults: `temperature=1.0`, `top_p=1.0` (set in this stack’s `kilo.json`). In thinking mode the server may ignore client sampling knobs (matches DeepSeek fixed-thinking behavior).

### Harness reliability (important)

Two failure modes showed up in real Kilo sessions:

| Symptom | Cause | Mitigation in this stack |
|---------|--------|---------------------------|
| `JSON Parse error: Expected '}'` on tool args | Thinking + short budget truncates DSML/tool JSON | **`ds4_kilo_proxy`**: thinking OFF, `max_tokens` floor, JSON close repair |
| “Response ended without a finish reason” / plan-only, no edit | CoT burns tokens; agent never applies the fix | Proxy + agent prompts: **finish the job**, prefer `bash`, no invented tools |
| Context 10k → 70k in a few turns | Parallel `read` storms | Prompts + `tool_output` caps + `compaction.prune` |

| Knob | Setting | Why |
|------|---------|-----|
| **Public API** | `:8083` → `ds4_kilo_proxy` → `:18083` ds4-server | Agent-safe defaults without changing Kilo baseURL |
| Thinking | default **disabled** | Tool calls land in content, not truncated CoT |
| `max_tokens` floor | 8192 (proxy + server `-n`) | Enough room for complete tool argument JSON |
| `agent.*` prompts | Finish job; ≤3–4 reads; only schema tools | Stops diagnose-and-bail + explore storms |
| `agent.*.steps` | 12–40 | Bounds runaway loops |
| `instructions` | `AGENTS.md` | Durable project rules |
| `compaction.prune` | `true` | Drops old tool outputs |
| `tool_output` | 400 lines / 32 KiB | Truncates huge file dumps |

```bash
./2_start_ds4.sh status          # public :8083 + upstream :18083
./2_start_ds4.sh restart         # reload engine + proxy
./2_start_ds4.sh --no-proxy      # raw ds4 on :8083 (not agent-safe)
curl -s http://127.0.0.1:8083/healthz
```

Reload Kilo / VS Code after changing `kilo.json`. Start a **new chat** after a large review (~50–60% of the context bar).

---

## API test

```bash
curl http://127.0.0.1:8083/v1/models

curl http://127.0.0.1:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"Say hi in 5 words."}],
    "max_tokens":32,
    "stream":false
  }'
```

Supported by `ds4-server` (upstream):

- `GET /v1/models`
- `POST /v1/chat/completions` (tools + streaming)
- `POST /v1/responses` (Codex-style)
- `POST /v1/messages` (Anthropic / Claude Code style)
- `POST /v1/completions`

---

## CLI / agent (no HTTP)

```bash
cd ds4

# Interactive chat
./ds4 --nothink

# One-shot
./ds4 -p "Explain Redis streams in one paragraph." --nothink

# Native coding agent (KV sessions under ~/.ds4/kvcache)
./ds4-agent --chdir "$(pwd)"
```

---

## Memory tips (128 GB)

- Default server: `--ctx 100000` — full 1M context is ~26 GB of KV alone; too much with an 81 GB model.
- Larger context if you free RAM: `./2_start_ds4.sh --ctx 200000`
- Cooler / quieter: `./2_start_ds4.sh --power 50`
- Model larger than RAM: `./2_start_ds4.sh --ssd-streaming` (and optionally `--ssd-streaming-cache-experts 32GB`)
- Optional speculative MTP (experimental):  
  `(cd ds4 && ./download_model.sh mtp)` then `./2_start_ds4.sh --mtp`
- On-disk KV cache for prefix reuse: default under `$TMPDIR/ds4-kv`

---

## Server lifecycle

```bash
./2_start_ds4.sh status
./2_start_ds4.sh stop
./2_start_ds4.sh restart
./2_start_ds4.sh --port 8084 --ctx 131072
```

---

## Ports

| Server | Port |
|--------|------|
| Gemma / Diffusion MLX | 8080 |
| DeepSeek MLX | 8082 |
| **DeepSeek ds4 (this)** | **8083** |
| Qwen3.6 mtplx | 8765 |

---

## Layout

```text
deepseek-v4-flash-ds4/
  1_setup_download.sh   # clone + Metal build + resilient GGUF download
  2_start_ds4.sh        # ds4-server wrapper (validates weights, status/stop/restart)
  download_gguf.py      # Range-resume HF downloader (survives connection resets)
  validate_model.py     # exact size + GGUF magic checks
  kilo.json             # Kilo provider ds4 + agent harness prompts
  AGENTS.md             # durable agent rules (tools, finish job, context)
  ds4_kilo_proxy.py     # thinking-off + tool JSON soft-repair proxy
  README.md
  bin/curl-resilient    # curl retry shim (PRO path / upstream download_model.sh)
  ds4/                  # git clone of antirez/ds4 (not committed)
    ds4, ds4-server, …  # built binaries
    gguf/               # downloaded weights (+ *.gguf.part while resuming)
    ds4flash.gguf       # symlink to selected main GGUF
```

Root repo Kilo config: [`../../kilo.json`](../../kilo.json) also registers the `ds4` provider (same Base URL / model IDs).

Upstream docs: [antirez/ds4 README](https://github.com/antirez/ds4).
