# Qwen3.8-27B — mtplx MTP Server + harness

Run **Qwen3.8-27B** locally on Apple Silicon with [mtplx](https://github.com/youssofal/MTPLX)
(native MTP speculative decoding when the checkpoint includes MTP heads), served as an
OpenAI-compatible API for Kilo Code.

**Port `:8766`** so it can sit beside the Qwen3.6 stack on `:8765`.

> **Weights status (Aug 2026):** Alibaba announced open weights for Qwen3.8-27B alongside
> Qwen3.8-Max (~2026-08-12 UTC+8 per community reports). Official / mlx-community /
> mtplx-optimized repos may not exist until then. `./1_setup_download.sh` installs the
> venv + mtplx and **preflights HF** so a missing repo does not look like a token error
> (exit **2** = deps OK, weights not published). Pass `QWEN38_HF_MODEL` or a raw `org/repo`
> once a quant is live. Until then, keep using
> [`../qwen3-6-27b-coder-mtplx/`](../qwen3-6-27b-coder-mtplx/).

## Quick start

```bash
# 1. Install mtplx and download weights (when published)
./1_setup_download.sh
# deps only (no pull):
#   ./1_setup_download.sh --deps-only
# or pin a quant as soon as it exists:
#   ./1_setup_download.sh mlx-community/Qwen3.8-27B-4bit
#   QWEN38_HF_MODEL=Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed ./1_setup_download.sh

# 2. Start server (keeps running until Ctrl+C; post-start harness gate ON by default)
./2_start_mtplx.sh
# If port 8766 is stuck:  ./2_start_mtplx.sh restart

# 3. Harness (also run automatically after start unless --no-harness-gate)
python3 test_harness.py --gate
python3 test_harness.py              # fuller live suite
python3 test_harness.py --quick

# 4. Kilo Code
#    Model: mtplx-qwen38/qwen3.8-27b-mtplx  (kilo.json here)
cd /your/project
kilo
```

## Architecture

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8766/v1   (OpenAI-compatible)
  mtplx serve
      │  ↑ MTP speculative decoding when checkpoint has MTP heads
      │  └── draft: model's own built-in MTP heads (no second model)
      ▼
  Qwen3.8-27B (MLX / mtplx quant, Apple Silicon)
```

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv + mtplx + model pull; writes `.mtplx_config` |
| `2_start_mtplx.sh` | Serve on `:8766`; optional `--harness-gate` |
| `test_harness.py` | Live API resilience / tool-call smoke tests |
| `kilo.json` | Kilo provider: `mtplx-qwen38/qwen3.8-27b-mtplx` → `:8766/v1` |

## Model resolution (4-bit by default)

**Default quant is 4-bit:** `mlx-community/Qwen3.8-27B-4bit`.

Bare `./1_setup_download.sh` never pulls bf16 or full official weights. If a 4-bit-class
checkpoint is already in the HF cache (mlx 4-bit or Youssofal MTPLX-optimized), that
cached repo is reused; otherwise setup pulls the mlx-community 4-bit id.

| Arg / env | Effect |
|-----------|--------|
| `./1_setup_download.sh` / `4bit` / `27b` | **4-bit** `mlx-community/Qwen3.8-27B-4bit` |
| `./1_setup_download.sh mtplx` | Youssofal MTPLX-optimized (4-bit class, best MTP when present) |
| `./1_setup_download.sh official` | full `Qwen/Qwen3.8-27B` (not default) |
| `./1_setup_download.sh bf16` | `mlx-community/Qwen3.8-27B-bf16` (not default) |
| `./1_setup_download.sh org/repo` | explicit HF repo |
| `QWEN38_HF_MODEL=...` | force repo |
| `QWEN38_ALIAS=...` | OpenAI model id (default `qwen3.8-27b-mtplx`) |

## Harness

`test_harness.py` hits the public OpenAI API only (no Kilo session DB).

| Mode | What it covers |
|------|----------------|
| `--gate` | Reachable, `/v1/models`, short chat, bash tool call, SSE stream, multi-turn tool result |
| default | Gate-level + health, unicode, empty tool result, multi-step continue, concurrent chats |
| `--quick` | Skip multi-step + concurrent |
| `--strict` | Soft (model-behavior) failures become hard |

```bash
python3 test_harness.py --base http://127.0.0.1:8766 --model qwen3.8-27b-mtplx
```

Exit codes: `0` pass · `1` hard fail · `2` unreachable.

`./2_start_mtplx.sh` runs `--gate` after ready. Disable with `--no-harness-gate`.

## Options

### Port

```bash
./2_start_mtplx.sh --port 8767
```

Update `kilo.json` `baseURL` to match.

### MTP depth / profile

```bash
./2_start_mtplx.sh --depth 2
./2_start_mtplx.sh --profile performance-cold --max
./2_start_mtplx.sh --profile burst   # alias for performance-cold --max
```

### Sampling (Kilo)

Default coding settings match the 3.6 stack: `temperature=0.6`, `top_p=0.95`, `top_k=20`.
Server is started with `--reasoning off` so tool loops are not wrapped in thinking.

## Latency notes (same as Qwen3.6)

- **MTP speeds decode**, not prefill. Long Kilo histories still cost a large first-token wait.
- Tool turns often prevent KV postcommit reuse → each step can re-prefill history.
- Compact / restart the Kilo session past ~15–20k tokens for usable latency.

Quick sanity check:

```bash
curl -s http://localhost:8766/v1/models
curl -s http://localhost:8766/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mtplx","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20}'
```

## Comparison

| | `qwen3-6-27b-coder-mtplx` | `qwen3-8-27b-coder-mtplx` (this) |
|---|---|---|
| Model family | Qwen3.6-27B | Qwen3.8-27B |
| Default port | **8765** | **8766** |
| Kilo model id | `mtplx/qwen3.6-27b-mtplx` | `mtplx-qwen38/qwen3.8-27b-mtplx` |
| Harness | (curl / manual) | `test_harness.py` + post-start gate |

Both can run side-by-side on different ports — do **not** load two large models at once on
≤128 GB unified memory.

## Troubleshooting

**Setup exit 2 / “No pullable weights”**

Weights not public yet (or only a README placeholder exists). HF often returns 401
“denied access” for missing ids — that is **not** a login problem. Re-run
`./1_setup_download.sh` after [search](https://huggingface.co/models?search=Qwen3.8-27B)
shows a real quant, or pass `org/repo` / `QWEN38_HF_MODEL` explicitly.

**Harness gate soft-fails tool call**

Model may chat instead of calling tools on a cold prompt. Re-run `test_harness.py --gate`;
if still soft-only, try a mtplx-optimized coding quant when available. Soft fails do not
stop the server.

**Kilo slow, curl fast**

Prefill / context size — compact the session. See the Qwen3.6 README latency section for detail.
