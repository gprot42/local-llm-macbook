# Qwen3.8-27B OBLITERATED — mtplx MTP Server + harness

Uncensored **Qwen3.8-27B** via [OBLITERATUS](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
(weight-space refusal removal of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)),
served locally on Apple Silicon with [mtplx](https://github.com/youssofal/MTPLX)
(native MTP speculative decoding — this MLX quant still has MTP heads).

**Port `:8767`** so it can sit beside aligned Qwen3.8 (`:8766`) and Qwen3.6 (`:8765`).
Do **not** load two large models at once on ≤128 GB unified memory.

Copied from [`../../censored/qwen3-8-27b-coder-mtplx/`](../../censored/qwen3-8-27b-coder-mtplx/)
and retargeted at the OBLITERATUS Hub pack.

> **Do not `mtplx pull OBLITERATUS/Qwen3.8-27B-OBLITERATED`.** That repo is a
> kitchen-sink (~270 GB): bf16 shards, five GGUFs, plus MLX. Setup snapshots
> only `mlx-4bit/` (~14 GB, default) or `mlx-8bit/` (~27 GB).

## Quick start

```bash
cd uncensored/qwen3-8-27b-obliterated-mtplx

# 1. Install mtplx and snapshot mlx-4bit/ (~14 GB)
./1_setup_download.sh
# deps only (no pull):
#   ./1_setup_download.sh --deps-only
# 8-bit (~27 GB):
#   ./1_setup_download.sh 8bit

# 2. Start server (keeps running until Ctrl+C; post-start harness gate ON by default)
./2_start_mtplx.sh
# If port 8767 is stuck:  ./2_start_mtplx.sh restart

# 3. Harness (also run automatically after start unless --no-harness-gate)
python3 test_harness.py --gate
python3 test_harness.py              # fuller live suite
python3 test_harness.py --quick

# 4. Kilo Code
#    Model: mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx  (kilo.json here)
cd /your/project
kilo
```

## Architecture

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8767/v1   (OpenAI-compatible)
  mtplx serve
      │  ↑ MTP speculative decoding (checkpoint has mtp_num_hidden_layers=1)
      │  └── draft: model's own built-in MTP heads (no second model)
      ▼
  OBLITERATUS/Qwen3.8-27B-OBLITERATED  mlx-4bit/  (local ./models/mlx-4bit)
```

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv + mtplx + **subfolder** snapshot; writes `.mtplx_config` |
| `2_start_mtplx.sh` | Serve on `:8767`; optional `--harness-gate` |
| `test_harness.py` | Live API resilience / tool-call smoke tests |
| `kilo.json` | Kilo provider: `mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx` → `:8767/v1` |

## Model resolution (4-bit by default)

**Default quant is 4-bit:** Hub path `mlx-4bit/` inside
[`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED).

| Arg / env | Effect |
|-----------|--------|
| `./1_setup_download.sh` / `4bit` / `27b` | **4-bit** `./models/mlx-4bit` (~14 GB) |
| `./1_setup_download.sh 8bit` | **8-bit** `./models/mlx-8bit` (~27 GB) |
| `QWEN38_OBL_HF_REPO=...` | Hub repo (still only the mlx subfolder is pulled) |
| `QWEN38_OBL_ALIAS=...` | OpenAI model id (default `qwen3.8-27b-obliterated-mtplx`) |

## Sampling (model card)

The OBLITERATUS card says these settings matter:

| setting | this stack | why |
|---------|------------|-----|
| **temperature** | **0** | greedy; temps above 0.5 degrade quality |
| **repetition_penalty** | mtplx **`frequency_penalty=0.2`** | card 1.15 is essential vs import loops; mtplx has no `repetition_penalty` flag |
| **top_p / top_k** | unused (`top_p=1.0`) | greedy; sampling not needed |
| **enable_thinking** | **off** (`--reasoning off`) | thinking burns token budget |
| **system prompt** | Kilo still sends its agent prompt | card prefers empty for refusal-sensitive chat; Kilo tools need a harness |
| **max_new_tokens** | card **≥ 2048** for long answers | harness smoke tests stay short |

`2_start_mtplx.sh` and `test_harness.py` send `frequency_penalty=0.2` as the
mtplx mapping of HF `repetition_penalty=1.15`.

## Harness

`test_harness.py` hits the public OpenAI API only (no Kilo session DB).

| Mode | What it covers |
|------|----------------|
| `--gate` | Reachable, `/v1/models`, short chat, bash tool call, SSE stream, multi-turn tool result |
| default | Gate-level + health, unicode, empty tool result, multi-step continue, concurrent chats |
| `--quick` | Skip multi-step + concurrent |
| `--strict` | Soft (model-behavior) failures become hard |

```bash
python3 test_harness.py --base http://127.0.0.1:8767 --model qwen3.8-27b-obliterated-mtplx
```

Exit codes: `0` pass · `1` hard fail · `2` unreachable.

`./2_start_mtplx.sh` runs `--gate` after ready. Disable with `--no-harness-gate`.

## Options

### Port

```bash
./2_start_mtplx.sh --port 8768
```

Update `kilo.json` `baseURL` to match.

### MTP depth / profile

```bash
./2_start_mtplx.sh --depth 2
./2_start_mtplx.sh --profile performance-cold --max
./2_start_mtplx.sh --profile burst   # alias for performance-cold --max
```

## Latency notes (same as Qwen3.6 / 3.8)

- **MTP speeds decode**, not prefill. Long Kilo histories still cost a large first-token wait.
- Tool turns often prevent KV postcommit reuse → each step can re-prefill history.
- Compact / restart the Kilo session past ~15–20k tokens for usable latency.

Quick sanity check:

```bash
curl -s http://localhost:8767/v1/models
curl -s http://localhost:8767/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-obliterated-mtplx","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20}'
```

## Comparison

| | `qwen3-8-27b-coder-mtplx` (aligned) | `qwen3-8-27b-obliterated-mtplx` (this) |
|---|---|---|
| Weights | Qwen / mlx-community / Youssofal | [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) |
| Default port | **8766** | **8767** |
| Kilo model id | `mtplx-qwen38/qwen3.8-27b-mtplx` | `mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx` |
| Default sampling | temp 0.6 / top_p 0.95 / top_k 20 | **temp 0** / top_p 1.0 |
| Download | `mtplx pull` of a 4-bit repo | **subfolder snapshot** of `mlx-4bit/` only |

## Research context

This checkpoint had safety guardrails removed in the weights. Use it on your
own hardware, for your own research / local-first work. You are responsible
for what you generate.

Capability: the card reports MMLU 81.4% vs 87.4% stock (−6.0pp) after
multi-direction abliteration.

## Troubleshooting

**Setup tried to pull hundreds of GB**

You ran `mtplx pull OBLITERATUS/Qwen3.8-27B-OBLITERATED`. Cancel it. Use
`./1_setup_download.sh` (mlx-4bit only).

**Harness gate soft-fails tool call**

Model may chat instead of calling tools on a cold prompt. Re-run `test_harness.py --gate`.
Soft fails do not stop the server.

**Kilo slow, curl fast**

Prefill / context size — compact the session. See the Qwen3.6 README latency section.
