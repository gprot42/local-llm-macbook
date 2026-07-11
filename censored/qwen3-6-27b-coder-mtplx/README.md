# Qwen3.6 — mtplx MTP Server

Run Qwen3.6 locally on Apple Silicon with **native MTP speculative decoding** via
[mtplx](https://github.com/youssofal/MTPLX), served as an OpenAI-compatible API for Kilo Code.

## Quick start

```bash
# 1. Install mtplx and download model (~18 GB for 27B)
./1_setup_download.sh

# 2. Start the mtplx server (keeps running until Ctrl+C)
./2_start_mtplx.sh
# If port 8765 is stuck / wrong process:  ./2_start_mtplx.sh restart

# 3. Open a project and launch Kilo Code
#    Model: mtplx/qwen3.6-27b-mtplx  (kilo.json here or ../kilo.json)
cd /your/project
kilo
```

## Architecture

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8765/v1   (OpenAI-compatible)
  mtplx serve
      │  ↑ MTP speculative decoding (D3)
      │  └── draft: model's own built-in MTP heads (no second model)
      ▼
  Qwen3.6-27B or 35B-A3B (MLX 4-bit, Apple Silicon)
```

No Ollama. No external drafter model. mtplx uses the MTP heads that ship
**inside** the Qwen3.6 checkpoint — zero extra RAM overhead.

## Performance expectations

The numbers in mtplx marketing and benchmarks measure **decode speed** on relatively short
prompts. Day-to-day use with Kilo Code (long context + tool calls) feels very different.

| Scenario | What you typically see (27B, M5 Max class) | Notes |
|----------|--------------------------------------------|-------|
| Server warmup | ~20–25 tok/s | One-time after load |
| Short chat (<1k prompt tokens) | ~20–40 tok/s end-to-end | MTP helps decode; prefill is small |
| Decode only (`tok_s` in server logs) | ~20–40 tok/s (up to ~60+ on ideal short runs) | MTP ~2× vs no-MTP at temp=0.6 on **decode** |
| Agent session (20k–35k prompt tokens) | **30–90+ s before first token**, then short replies | Dominated by **prefill**, not MTP |
| Agent end-to-end (`end_to_end_tok_s` in logs) | Often **under 5 tok/s** | Prefill + tiny tool-call outputs |

**MTP speeds up token generation; it does not speed up prefill.** A 30k-token prompt still has to
be processed through the model before the first streamed token — expect ~30s of `mtplx_stream_silence`
in logs on large contexts.

**Tool-calling agents are slower than chat.** When the assistant emits tool calls, mtplx marks
session postcommit as unsafe (`tool_call_history_rewrite`) and cannot reuse KV cache between
turns. Each Kilo step tends to re-prefill the full growing history instead of a ~1s suffix update.

**Keep context small for usable latency.** Restart or compact the Kilo session once history grows
past ~15–20k tokens. Avoid re-injecting large file dumps every turn.

Quick sanity check (server idle, small prompt):

```bash
curl -s http://localhost:8765/health
curl -s http://localhost:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-mtplx","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20}'
```

If that is fast but Kilo is slow, the server is fine — the bottleneck is context size and agent traffic.

## Models

| Model | Size | MTP (decode, short context) | SWE-bench | Best For |
|-------|------|-----------------------------|-----------|----------|
| `Qwen3.6-27B-MTPLX-Optimized-Speed` | ~18 GB | ~2× decode speedup; ~20–60 tok/s decode | — | Default — best MTP acceptance |
| `Qwen3.6-35B-A3B-4bit` | ~22 GB | ~1× today (quantized MTP weights) | **73.4%** | Benchmark score; not faster MTP yet |

> The 27B checkpoint keeps MTP weights in BF16 (Youssofal optimized build) — use this for local speed.
> The 35B-A3B MoE scores higher on coding benchmarks but needs a BF16-MTP build before MTP pays off;
> see [Low MTP acceptance rate on 35B-A3B](#low-mtp-acceptance-rate-on-35b-a3b) below.

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Install mtplx venv + download model weights |
| `2_start_mtplx.sh` | Start mtplx OpenAI-compatible server on :8765 |
| `kilo.json` | Kilo Code config — points to mtplx API |

## Options

### Use the 35B MoE model

```bash
./1_setup_download.sh 35b
./2_start_mtplx.sh --model 35b
```

Then update `kilo.json` default model to `mlx-community/Qwen3.6-35B-A3B-4bit`.

### Override port

```bash
./2_start_mtplx.sh --port 8766
```

Then update `kilo.json`:
```json
"baseURL": "http://localhost:8766/v1"
```

### MTP depth tuning

```bash
./2_start_mtplx.sh --depth 2   # D2: safest, highest acceptance (~1.8× decode on short context)
./2_start_mtplx.sh --depth 3   # D3: default (~2× decode on short context)
./2_start_mtplx.sh --depth 4   # D4: more aggressive; lower acceptance on long outputs
```

### Performance profiles

```bash
./2_start_mtplx.sh --profile sustained           # default — stable thermals for long sessions
./2_start_mtplx.sh --profile performance-cold --max  # faster decode/prefill, louder fans; still slow at 30k+ context
./2_start_mtplx.sh --profile burst               # alias for performance-cold --max
./2_start_mtplx.sh --profile stable              # conservative, for compatibility
```

Valid profile names: `sustained` · `performance-cold` · `stable` · `exact` · `max-diagnostic`

## Comparison with Gemma 4 MLX setup

| | `gemma4-server-heretic-31b-mlx` | `qwen3-6-27b-coder-mtplx` |
|---|---|---|
| Model | Gemma 4 26B Heretic (MoE) | Qwen3.6 27B or 35B-A3B |
| MTP | No (Gemma 4 lacks MTP heads) | Yes — helps decode on short context; less impact on long agent turns |
| Coding (SWE-bench) | competitive | 73.4% (35B) |
| Agentic (Terminal-Bench) | — | 51.5% (35B) |
| Port | 8080 | **8765** |

Both servers can run simultaneously on different ports.

## Troubleshooting

**Kilo / agent feels very slow (30s+ between steps, `mtplx_stream_silence` in logs)**

The server is usually working; the workload is the issue.

1. Check server logs for `prompt_tokens` — if often **20k+**, trim history or start a new session.
2. Look for `unsafe_reason: tool_call_history_rewrite` — normal with tool calls; caching between turns is disabled.
3. Short outputs (dozens of tokens) after huge prompts still cost a full prefill each time.
4. Only one generation runs at a time — other clients queue behind an active Kilo session.
5. For snappier **short** interactions (not long agent marathons): `./2_start_mtplx.sh --profile performance-cold --max`

**`ERROR: .mtplx_config not found`**
```bash
./1_setup_download.sh    # re-run setup
```

**`ERROR: venv not found`**
```bash
./1_setup_download.sh    # creates the venv
```

**Low MTP acceptance rate on 35B-A3B**

The root cause is MTP weight quantization. Two things must hold for high acceptance:

1. **MTP weights must stay in BF16.** The `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint
   quantizes the entire model including the MTP transformer layer and LM head. Quantization
   error in the MTP head compounds through the expert routing prediction — acceptance drops
   from ~79-85% to ~20-46%, which yields only ~1.03× speedup instead of the expected ~1.18×.

2. **The MoE expert routing prediction is sensitive.** Dense models (27B) tolerate quantized
   MTP weights better than MoE models because there is no expert-gating step in the draft path.
   The Qwen3.6-35B-A3B has 256 experts with 8 routed per token — a small MTP weight error can
   pick the wrong expert set, breaking the draft token entirely.

**What to expect today:**

| Model | MTP weights | Acceptance | Speedup |
|-------|-------------|------------|---------|
| `Qwen3.6-27B-MTPLX-Optimized-Speed` | BF16 (kept by Youssofal) | ~95-98% (D3) | **~2.24×** |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | 4-bit quantized | ~20-46% | ~1.03× |
| `Qwen3.6-35B-A3B` with BF16 MTP weights *(not yet published)* | BF16 | ~79-85% | ~1.18× |

**Recommendation:** Use the 27B model until a `Qwen3.6-35B-A3B-MTPLX-Optimized` build is
published (same treatment Youssofal applied to the 27B — keeps MTP weights in BF16 while
quantizing only the backbone). Watch https://huggingface.co/Youssofal for updates.

If raw coding benchmark score (73.4% SWE-bench) matters more than MTP speed, run the 35B-A3B
without MTP by passing `--depth 0` to `./2_start_mtplx.sh` — you still get full MLX throughput
(~85 tok/s), just without the speculative multiplier.

**Kilo Code doesn't connect**
```bash
curl http://localhost:8765/v1/models    # check if server is running
./2_start_mtplx.sh                    # restart if needed
```

**Port 8765 already in use**

Often a leftover from another project (e.g. `python -m http.server 8765`), not mtplx.
```bash
./2_start_mtplx.sh status          # show what holds the port
./2_start_mtplx.sh restart         # free the port and start mtplx (same as sibling gemma/ornith scripts)
./2_start_mtplx.sh stop            # free the port only
# or pick a different port:
./2_start_mtplx.sh --port 8766
```
If you change the port, update `baseURL` in `kilo.json` (and parent `../kilo.json`) to match.
Also set the Kilo model to `mtplx/qwen3.6-27b-mtplx` (not `openai-compatible/...`).
**mtplx version check**
```bash
source venv/bin/activate
mtplx --version
pip install --upgrade mtplx   # update if needed
```
