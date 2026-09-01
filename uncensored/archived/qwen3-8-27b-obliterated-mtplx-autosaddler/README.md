# Qwen3.8-27B OBLITERATED V3 — mtplx MTP Server + harness

Uncensored **Qwen3.8-27B V3** via [OBLITERATUS](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
(weight-space refusal removal of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)),
served locally on Apple Silicon with [mtplx](https://github.com/youssofal/MTPLX)
(native MTP speculative decoding — convert keeps MTP heads from the V3 extra shard).

**Engine `:8767`**, **Kilo/harness proxy `:8768`** (beside aligned Qwen3.8 `:8766` and Qwen3.6 `:8765`).
Do **not** load two large models at once on ≤128 GB unified memory.

Copied from [`../../censored/qwen3-8-27b-coder-mtplx/`](../../censored/qwen3-8-27b-coder-mtplx/)
and retargeted at the OBLITERATUS Hub pack.

> **Do not `mtplx pull OBLITERATUS/Qwen3.8-27B-OBLITERATED`.** That repo is a
> kitchen-sink (GGUF + leftover 18-shard files + V3 bf16). Hub **removed**
> `mlx-4bit/` / `mlx-8bit/` with V3 (those were V1/V2). Setup snapshots **V3
> bf16** (~56 GB) and converts locally to `mlx-4bit/` (~14 GB, default) or
> `mlx-8bit/` (~27 GB).
>
> GGUF users: Pliny re-uploaded V3 GGUFs on 2026-08-23 after a broken
> conversion ([tweet](https://x.com/elder_plinius/status/2091339694303068467)).
> bf16 was always fine. **This stack does not use GGUF.**

## Quick start

```bash
cd uncensored/qwen3-8-27b-obliterated-mtplx

# once
./1_setup_download.sh          # V3 bf16 → mlx-4bit (~14 GB)
# ./1_setup_download.sh 8bit   # or 8-bit (~27 GB)

# every session
./2_start_mtplx.sh             # engine + proxy + smoke-check. Leave it running.

# then in your project — use this folder's kilo.json (already points at :8768)
kilo                           # model: mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx
```

You do **not** edit `kilo.json` before `./2_start_mtplx.sh`. The start script does not
read it. Sampling and loop middleware are forced by the proxy on `:8768`. `kilo.json`
is only for **Kilo** (after the server is up): model id, `baseURL`, steps, and the
short agent prompt. It already matches this stack — use that file (or copy it into
the project) instead of changing ports or temperature.

If the engine is already running, `./2_start_mtplx.sh` attaches, smoke-checks, and
leaves it. `restart` if you pulled new proxy code.

```bash
./2_start_mtplx.sh restart       # stuck ports / full reload
./2_start_mtplx.sh check         # smoke-check only (server already up)
./2_start_mtplx.sh check-agent   # longer train/dev tool-loop eval
./2_start_mtplx.sh stop
```

## Architecture

```
Kilo Code (TUI)
      │
      ▼  http://localhost:8768/v1   (kilo proxy — card sampling forced)
  qwen38_obl_kilo_proxy.py
      │  temp=0  frequency_penalty=0.3  thinking off  finish-the-job loop
      ▼  http://127.0.0.1:8767/v1
  mtplx serve
      │  ↑ MTP speculative decoding (checkpoint has mtp_num_hidden_layers=1)
      │  └── draft: model's own built-in MTP heads (no second model)
      ▼
  OBLITERATUS/Qwen3.8-27B-OBLITERATED  V3 bf16 → local mlx-4bit/
```

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv + mtplx + **V3 bf16 snapshot + local MLX convert**; writes `.mtplx_config` |
| `2_start_mtplx.sh` | mtplx on `:8767` + kilo proxy on `:8768`; optional `--harness-gate` |
| `qwen38_obl_kilo_proxy.py` | Card sampling + scoped loop middleware (empty/fake-action/prose-loop/JIT continue); reloads `.autosaddler/active.json` |
| `autosaddler.py` | Persistent Diagnosis–Patch–EvoDAG optimizer |
| `test_harness.py` | Live API smoke + real tool-loop mini-batch / diagnosis / `--optimize` (hits `:8768`) |
| `AUTOSADDLER.md` | Optimizer loop, persistence, patch taxonomy |
| `kilo.json` | Kilo provider: `mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx` → `:8768/v1` |

## Model resolution (4-bit by default)

**Default quant is 4-bit**, converted locally from V3 bf16 (Hub no longer
ships `mlx-4bit/`).

| Arg / env | Effect |
|-----------|--------|
| `./1_setup_download.sh` / `4bit` / `27b` | snapshot V3 bf16 → **4-bit** `./models/mlx-4bit` (~14 GB) |
| `./1_setup_download.sh 8bit` | same bf16 → **8-bit** `./models/mlx-8bit` (~27 GB) |
| `./1_setup_download.sh --force` | re-snapshot + reconvert even if mlx looks complete |
| `QWEN38_OBL_HF_REPO=...` | Hub repo (still never a full `mtplx pull`) |
| `QWEN38_OBL_ALIAS=...` | OpenAI model id (default `qwen3.8-27b-obliterated-mtplx`) |

## Sampling (model card)

The OBLITERATUS card says these settings matter:

| setting | this stack | why |
|---------|------------|-----|
| **temperature** | **0** | greedy; temps above 0.5 degrade quality |
| **repetition_penalty** | mtplx **`frequency_penalty=0.3`** | card 1.15; sweep on :8767: greedy + 0.3 gave the longest non-loop answers |
| **top_p / top_k** | unused (`top_p=1.0`) | greedy; sampling not needed |
| **enable_thinking** | **off** (`--reasoning off`) | V3 thinking ON works; OFF is still the card default for direct answers |
| **system prompt** | short finish-the-job agent prompt in stack `kilo.json` | card prefers empty for chat; without a loop prompt Kilo stops after 1 of N |
| **max_new_tokens** | card **≥ 2048** | tool-loop harness tests use 2048; PING stays tiny |
| **agent steps** | **50** (build/code) | stack `kilo.json` had no `steps`, so the agent gave up early |

`2_start_mtplx.sh`, the kilo proxy, and `test_harness.py` send
`frequency_penalty=0.3` as the mtplx mapping of HF `repetition_penalty=1.15`
(picked from `sweep_longrun.py` for longest greedy answers without loops).

## Harness

`test_harness.py` hits the public OpenAI API only (no Kilo session DB).

The **agent harness** is `kilo.json` + `qwen38_obl_kilo_proxy.py` — everything around
the weights that shapes long-horizon tool use. It follows
[AutoSaddler](https://arxiv.org/abs/2608.23041) (Park et al., 2026): automatic
harness optimization from agent execution traces.

### AutoSaddler principles (what we copy)

1. **Harness ≠ prompt.** Search over prompts, tools, *and* middleware (hooks,
   agent-loop logic, infra). Prompt-only tuning (GEPA-style) saturates earlier.
2. **Diagnose traces, not symptoms.** Read the full rollout (every tool call,
   empty result, plan-without-tools) and name the root cause
   (`no_tool_call`, `stopped_after_1`, `empty_result_idle`, …) before changing
   anything. Shallow “it failed, add a rule” patches miss the real bug.
3. **Capability before steering.** First change what the agent *can* do (loop
   logic, tool implementations, output caps). Then, if needed, change how it
   *chooses* (prompt rules, tool descriptions, just-in-time reminders).
   Capability patches fix at a similar rate with fewer regressions (~8% vs ~17%).
4. **Structured, scoped patches.** One layer at a time. Hooks fire only on
   matching state (empty tool, fake action, prose loop). Always-on PreToolUse
   text on high-frequency tools overfits and regresses other tasks.
5. **Replace, don’t stack, prompt rules.** Finite attention: a new sentence
   should replace or merge an old one, not append forever.
6. **Generalize, don’t hotfix.** Keep a change only if it helps the mini-batch
   *and* a held-out split. Trajectory-specific repairs look good on one task
   and fail the next. Traces live in `logs/harness-traces/`.

Paper results (for context, not this 27B): +9.0 / +9.6 / +10.0 pp on GAIA2,
SWE-Bench Pro, Terminal-Bench 2.0 vs the base harness.

The optimizer **is shipped** — see [`AUTOSADDLER.md`](AUTOSADDLER.md). Each
`optimize` run resumes the EvoDAG under `.autosaddler/`. The proxy reloads
`active.json` by mtime so accepted patches apply to Kilo without a restart.

```bash
./2_start_mtplx.sh                  # engine + proxy; loads last active harness
./2_start_mtplx.sh optimize         # mini-batch → diagnose → patch → train/dev → EvoDAG
python3 autosaddler.py --status
python3 test_harness.py --optimize --iters 3
```

Acceptance: keep a patch only if the **train mini-batch improves** and the
**held-out dev split does not regress**. Live recovery firings go to
`.autosaddler/live-events.jsonl` for the next diagnose. This is offline
mini-batch learning — re-run `optimize` to self-improve.

### What this stack does

| Patch kind (AutoSaddler) | Here |
|--------------------------|------|
| Agent loop logic (capability) | Empty-tool recovery, fake-action recovery, prose-loop recovery, just-in-time continue after a tool result |
| Infra (capability) | Cap tool outputs at 30k chars; repair truncated tool-call JSON |
| Tool descriptions (steering) | Harness schemas say *when* to glob/grep/read vs bash |
| Prompt (steering) | Short finish-the-job + verify-before-done in `kilo.json` (replacement, not stacked rules) |
| Generalization gate | `--agent` mini-batch has a **train** split and a held-out **dev** split |

| Mode | What it covers |
|------|----------------|
| `--gate` | Reachable, `/v1/models`, short chat, bash tool call, SSE stream, multi-turn tool result |
| `--unit` | Offline proxy middleware + sandbox + failure-taxonomy tests (no server) |
| default | Unit + live contract, including a **real 2-round executed tool loop** (not pre-stuffed history) |
| `--quick` | Skip multi-step, real loop, concurrent |
| `--agent` | Train/dev mini-batch of real loops with trace diagnosis (`no_tool_call`, `stopped_after_1`, `empty_result_idle`, …) |
| `--optimize` | Persistent Diagnosis–Patch–EvoDAG loop (writes `.autosaddler/`) |
| `--strict` | Soft (model-behavior) failures become hard |

You do not need these for day-to-day Kilo use (`./2_start_mtplx.sh` already smoke-checks).
Direct `python3 test_harness.py …` is only if you want a specific slice:

```bash
./2_start_mtplx.sh check
./2_start_mtplx.sh check-agent
./2_start_mtplx.sh optimize
python3 test_harness.py --unit          # offline only
python3 test_harness.py --agent --strict
```

Exit codes: `0` pass · `1` hard fail · `2` unreachable.

Disable the post-start smoke-check with `--no-harness-gate`.

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
curl -s http://localhost:8768/v1/models
curl -s http://localhost:8768/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-obliterated-mtplx","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20}'
```

## Comparison

| | `qwen3-8-27b-coder-mtplx` (aligned) | `qwen3-8-27b-obliterated-mtplx` (this) |
|---|---|---|
| Weights | Qwen / mlx-community / Youssofal | [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) |
| Default port | **8766** | **8768** (proxy) / **8767** (engine) |
| Kilo model id | `mtplx-qwen38/qwen3.8-27b-mtplx` | `mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx` |
| Default sampling | temp 0.6 / top_p 0.95 / top_k 20 | **temp 0** / top_p 1.0 |
| Download | `mtplx pull` of a 4-bit repo | **V3 bf16 snapshot + local `mlx_lm convert`** (Hub mlx folders gone) |

## Research context

This checkpoint had safety guardrails removed in the weights. Use it on your
own hardware, for your own research / local-first work. You are responsible
for what you generate.

Capability: V3 card reports MMLU **82.3%** vs **84.5%** stock (**−2.1pp**).
V1 was −6.0pp; V2 was near-stock but still deflected. V3 is the “genuinely
answers” revision.

## Troubleshooting

**Setup tried to pull hundreds of GB**

You ran `mtplx pull OBLITERATUS/Qwen3.8-27B-OBLITERATED`. Cancel it. Use
`./1_setup_download.sh` (V3 bf16 → mlx-4bit only).

**Still running the Aug-20 mlx-4bit snapshot (V1)**

Hub deleted those folders. Rebuild V3:

```bash
./1_setup_download.sh --force
./2_start_mtplx.sh restart
```

**Harness gate soft-fails tool call**

Model may chat instead of calling tools on a cold prompt. Re-run `test_harness.py --gate`.
Soft fails do not stop the server.

**Kilo slow, curl fast**

Prefill / context size — compact the session. See the Qwen3.6 README latency section.
