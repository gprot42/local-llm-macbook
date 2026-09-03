# Qwen3.8-27B OBLITERATED V3 — mtplx MTP Server + harness

**Status: 📦 archived / 🟡 unstable.** Kilo agent loops hang (session-in-flight, 10+ min Thinking spinner, Next-steps recap loops). Not recommended as a daily driver. 2026-09-02: thinking is now ON for tool turns (see [Thinking](#thinking-2026-09-02)); re-evaluate after a few sessions.

Uncensored **Qwen3.8-27B V3** via [OBLITERATUS](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
(weight-space refusal removal of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)),
served locally on Apple Silicon with [mtplx](https://github.com/youssofal/MTPLX)
(native MTP speculative decoding — convert keeps MTP heads from the V3 extra shard).

**Engine `:8767`**, **Kilo/harness proxy `:8768`** (beside aligned Qwen3.8 `:8766` and Qwen3.6 `:8765`).
Do **not** load two large models at once on ≤128 GB unified memory.
Do **not** start this stack and the archived AutoSaddler copy together — they share `:8767`/`:8768`.

Production stack (no AutoSaddler daemon). Copied from
[`qwen3-8-27b-obliterated-mtplx-autosaddler/`](../qwen3-8-27b-obliterated-mtplx-autosaddler/)
and stripped of the EvoDAG optimizer. Setup will **symlink** a complete sibling
`models/mlx-4bit` (or `bf16-v3`) instead of re-downloading ~56 GB.

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
cd uncensored/archived/qwen3-8-27b-obliterated-mtplx

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
      │  temp=0  frequency_penalty=0.3  thinking ON for tool turns (medium, 4096-tok budget)
      │  finish-the-job loop middleware
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
| `2_start_mtplx.sh` | mtplx on `:8767` + kilo proxy on `:8768`; optional `--harness-gate`; exports the thinking env (effort/budget) to engine + proxy |
| `qwen38_obl_kilo_proxy.py` | Card sampling + scoped loop middleware (empty/fake-action/prose-loop/JIT continue) + **first-turn guard** (a plan dump instead of a tool call right after the prompt is retried, not handed back) + **English-only enforcement** (CJK replies are detected and regenerated once; `QWEN38_OBL_ENGLISH=0` disables) |
| `test_harness.py` | Live API smoke + real tool-loop mini-batch (hits `:8768`) |
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
| **enable_thinking** | **ON for tool turns**, off for chat/compaction (proxy-owned) | Card default is off for *direct answers*; with thinking off the agent loop was a chain of greedy no-deliberation steps (re-read the same file, recap, hand back). Tool turns now think at `reasoning_effort=medium`, bounded by the mtplx thinking guard (`MTPLX_THINKING_BUDGET=4096`). Engine `:8767` default stays off for raw curls. |
| **system prompt** | short finish-the-job agent prompt in stack `kilo.json` | card prefers empty for chat; without a loop prompt Kilo stops after 1 of N |
| **max_new_tokens** | card **≥ 2048** (+ thinking budget on tool turns → floor 6144, cap 8192) | the `<think>` segment comes out of the same max_tokens; `kilo.json` `limit.output` is 8192 |
| **agent steps** | **50** (build/code) | stack `kilo.json` had no `steps`, so the agent gave up early |

**Kilo context limits (measured 2026-09-02).** Decode is ~25 tok/s at ≤30k
prompt tokens and 2–5 tok/s past ~50k, so the global root
[`kilo.json`](../../../kilo.json) caps this model at `limit.context` **49152** /
`limit.output` **4096**: with `compaction.reserved` 16384 that makes compaction
fire at `context − reserved − output` ≈ **28k**, before the collapse. The stack
`kilo.json` keeps `49152 / 8192` for single-stack use. Either way the proxy
floors agent turns at `AGENT_MAX_TOKENS_FLOOR` (2048 + thinking budget = 6144),
so a 4096 output cap never truncates a tool turn. Strict JSON cannot carry a
comment — this paragraph is where that measurement lives.

`2_start_mtplx.sh`, the kilo proxy, and `test_harness.py` send
`frequency_penalty=0.3` as the mtplx mapping of HF `repetition_penalty=1.15`
(picked from `sweep_longrun.py` for longest greedy answers without loops).

### Thinking (2026-09-02)

Traced root cause of "stops after light thinking, needs another prompt": the
stack ran the model with thinking fully off (`--reasoning off` + proxy
`enable_thinking=false`), so every agent step was a greedy reply with no
deliberation. Live sessions showed the pattern — read a file, re-read the
same file until the repeat guard removed the tools, or a recap after 3–5
rounds — and a human had to type "continue". Capability before steering:
the proxy now turns thinking **on** for requests that carry tools and leaves
it off for compaction, title generation, plain chat and the repeat-guard
"answer in text" turn (no budget guard applies there and a tiny `max_tokens`
would be eaten by `<think>`).

| Env (read by `2_start_mtplx.sh`, engine and proxy) | Default | Effect |
|---|---|---|
| `QWEN38_OBL_THINKING` | `1` | `0` restores thinking-off everywhere |
| `QWEN38_OBL_REASONING_EFFORT` | `medium` | `low` / `medium` / `xhigh` — the Qwen3.8 template accepts exactly these |
| `QWEN38_OBL_THINKING_BUDGET` | `4096` | reasoning tokens per agent turn; engine thinking guard (`MTPLX_THINKING_BUDGET`, exported by the start script) force-closes `<think>` at the budget so a self-doubt loop cannot hold the engine |
| `QWEN38_OBL_CYCLE_WINDOW` / `QWEN38_OBL_CYCLE_MIN` | `8` / `3` | window of assistant tool turns / recurrences of one command that trip the loop nudge + xhigh escalation |
| `QWEN38_OBL_CYCLE_BREAK` | `6` | recurrences that ban the command and force the break (retry, else visible stop) |

The agent `max_tokens` floor is `2048 + budget`. Reasoning streams as
`reasoning_content`; every proxy check (early-stop, CJK, repeat guard) reads
`content` only. `restart` after changing any of these — the engine flag and
the proxy both need a reload.

```bash
QWEN38_OBL_REASONING_EFFORT=xhigh QWEN38_OBL_THINKING_BUDGET=6144 ./2_start_mtplx.sh restart
./2_start_mtplx.sh reload-proxy   # proxy-only reload after middleware changes (engine stays warm)
```

### Follow-through (2026-09-02, second pass)

First live session with thinking on: 9 tool rounds in 90 s, then the turn
ended — not by the model, by the harness. Two proxy limits were doing the
quitting: the repeat guard removed the tools after the third identical
`cat …` call (a forced text answer ends Kilo's turn), and the after-tool
nudge cap (8 rounds) stripped the continue nudge mid-task, after which the
stream is passthrough and no response-side check runs at all. Changes:

| Mechanism | Before | Now |
|---|---|---|
| Duplicate tool call (response side) | forwarded; counted toward the repeat guard | caught **before** it is forwarded: one retry with a `[Harness] REPEATED ACTION` user turn that quotes the head of the result the model skipped, then a different step |
| Repeat guard: tools removed / hard stop | 3 / 4 identical turns | **5 / 6** — last resort, not the second step |
| After-tool nudge cap | 8 rounds | **40** (`QWEN38_OBL_AFTER_TOOL_MAX`); Kilo `steps=50` is the hard cap |
| Rollout trace | none (only request flags) | `logs/live-trace.jsonl`: per response — last tool result head, reasoning size, content head, tool calls, finish reason, middleware flags; plus an `[out]` log line |

Third pass (same day), from the first trace: the turn ended because the model
wrote its tool call as **prose** — `<bash>\n\nls -la …; ls … | head -60` as
plain content, no `tool_calls`. mtplx's parser only knows the Qwen
`<tool_call>{json}</tool_call>` form, the early-stop regex saw no plan
words, so the text passed as a final answer and Kilo ended the turn. The
proxy now converts prose tool calls into real `tool_calls` before the
client sees them (`[agent] prose tool call -> tool_calls`): Kilo/Cline XML
`<tool>…</tool>` (closed or not, raw or `<key>value</key>` args mapped by
the tool's schema), an unclosed `<tool_call>{json}`, and a reply that is
only a shell code fence. No retry, no extra engine round.

Fourth pass (from the 19:16–19:18 trace): the model no longer quit early —
it ran 12 rounds — but it **cycled** `ls recent` → `read SUMMARY.md` →
`tail JOURNAL.md` → back again, re-reading the same status files without
doing the research. The repeat guard never fired because it only sees
CONSECUTIVE identical calls, and this was a period-3 cycle (`repeat=1`
throughout). Two additions:

- **Cycle guard** (`mw_cycle_guard`): over the last `QWEN38_OBL_CYCLE_WINDOW`
  (8) assistant tool turns, if any one tool signature recurs
  `QWEN38_OBL_CYCLE_MIN` (3) times it fires `[Harness] LOOP DETECTED`,
  naming the repeated commands and telling the model to take a *different*
  forward action (or conclude) — tools are KEPT (redirect, not stop),
  unlike the consecutive repeat guard. `cycle=N` now shows in `[steer]`.
- **`[Tool calls: …]` / `[Calling tool: …]`** added to the prose-tool-call
  converter — round 12 ended the turn by writing its bash call in that
  bracketed prose form (finish=stop).

Fifth pass (the cycle guard's nudge was ignored): trace 19:24-19:34 showed
the model run `tail -c 2000 .../JOURNAL.md` eight times across 16 rounds. The
cycle guard fired `[Harness] LOOP DETECTED` four times; the model ignored it
and kept re-reading -- reasoning was only ~49 chars/turn (greedy, no real
deliberation). So the harness stops asking and forces the break, on a ladder
that keeps "keep going" options open before giving up:

1. **Nudge** (`cycle_max >= QWEN38_OBL_CYCLE_MIN`, 3) -- `[Harness] LOOP
   DETECTED`, tools kept.
2. **Think harder** -- that turn's `reasoning_effort` is escalated to `xhigh`
   automatically (a looping turn gets maximum deliberation to reason its way
   out), tools kept. Shows as `xhigh` in the `[out]` line.
3. **Ban + break** (`cycle_max >= QWEN38_OBL_CYCLE_BREAK`, 6) -- the repeated
   command is banned response-side: one hard retry quoting its stale result;
   if the model takes a different action the loop is broken and forwarded.
4. **Visible stop** -- only if it repeats the banned command even after the
   ban: the turn ends with `[Harness] Stopped: stuck in a read-only loop ...`
   (a real message the user sees) instead of spinning forever.

Regression tests: `test_cycle_break.py` (stubborn model -> visible stop) plus
the recover branch (breaks out -> new action forwarded). The consecutive
repeat guard's own hard stop (`REPEAT_HARD_STOP`, 6) covers the perfectly
adjacent case with an equivalent message.

**This is a capability ceiling, not just a harness bug.** At `temperature=0`
with ~49 reasoning chars/turn the model deterministically re-derives the same
action; the ladder breaks the loop but cannot make the model do real
research. The two real levers remain: run at `xhigh` globally
(`QWEN38_OBL_REASONING_EFFORT=xhigh QWEN38_OBL_THINKING_BUDGET=6144`) and give
Kilo a concrete task ("produce an exploitability finding for path X"), not
"continue the research".

Diagnose the next early stop from the trace, not the symptom:


```bash
tail -n 5 logs/live-trace.jsonl | python3 -c 'import sys,json; [print(json.dumps(json.loads(l), indent=1)[:1500]) for l in sys.stdin]'
```

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

The AutoSaddler EvoDAG optimizer is **not** in this folder. Use
[`../qwen3-8-27b-obliterated-mtplx-autosaddler/`](../qwen3-8-27b-obliterated-mtplx-autosaddler/)
if you want `./2_start_mtplx.sh optimize`.

### What this stack does

| Patch kind (AutoSaddler) | Here |
|--------------------------|------|
| Agent loop logic (capability) | Empty-tool recovery, fake-action recovery, prose-loop recovery; JIT continue for the first 40 tool rounds after the user (`QWEN38_OBL_AFTER_TOOL_MAX`), then the nudge is stripped; a Next-steps/plan dump with no tool_calls is retried once per turn on the **first reply after the prompt** and on every nudged tool round, never re-running a synthetic command already in history. Plain answers pass through with one upstream call. Harness flags: `mw_first_turn`, `mw_early_stop`, `mw_after_tool` |
| Infra (capability) | Cap tool outputs at 30k chars; repair truncated tool-call JSON; bash-sanitize only *broken* commands (keep `cd && ./script`, `which`, pipes) |
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
| `--strict` | Soft (model-behavior) failures become hard |

You do not need these for day-to-day Kilo use (`./2_start_mtplx.sh` already smoke-checks).
Direct `python3 test_harness.py …` is only if you want a specific slice:

```bash
./2_start_mtplx.sh check
./2_start_mtplx.sh check-agent
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
