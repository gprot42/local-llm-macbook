# AutoSaddler on this stack

Persistent harness optimizer from [Park et al., 2026](https://arxiv.org/abs/2608.23041)
(*Automatic Harness Optimization with Durable Updates from Agent Execution Traces*).

The **task agent** is Qwen3.8-27B OBLITERATED on mtplx (`:8767`) behind
`qwen38_obl_kilo_proxy.py` (`:8768`). The **optimizer** is `autosaddler.py`: it
treats the harness as a named component map (prompts, tool descriptions,
middleware flags), not unconstrained repo edits.

Each stack keeps its own `.autosaddler/` EvoDAG.

## What it is (and is not)

It **is** the paper loop: mini-batch → diagnose traces → one structured patch →
verify on the same batch → evaluate a held-out split → reflect → EvoDAG → evolve.

It is **not** per-Kilo-turn learning. Weights do not change. A Kilo chat does not
rewrite `kilo.json`. Improvement happens when you run `optimize`; accepted patches
then persist and the proxy reloads them without a restart.

Kilo’s fixed `kilo.json` options are **steering / infra**. The proxy still **forces
card sampling** (temp 0, `frequency_penalty=0.3`, thinking off) so a global Muse
config cannot override the model card. AutoSaddler overlays loop prefix, tool
descriptions, and middleware flags on top of that.

## Loop (one iteration)

| Step | Paper | Here |
|------|--------|------|
| 1 | Evaluate H_n on mini-batch B_n ⊂ D_train | `test_harness.py` train tasks (`two_step_echo`, `empty_recovery`) |
| 2 | Diagnose failed traces + harness code | Labels `no_tool_call` / `stopped_after_1` / `empty_result_idle` / `fake_action` / `prose_loop` plus last-round evidence; optional local-model JSON patch |
| 3 | Structured patch Δθ (one layer) | Named component: `loop_prefix`, nudges, `tool_desc.*`, `mw_*`, `empty_streak_min` |
| 4 | Verify J(B_n, H′) > J(B_n, H_n) | Re-run the same train tasks; reject train regressions |
| 5 | If improved, evaluate D_dev | Hold-out `glob_then_read`, `three_step_holdout`; reject if dev score drops |
| 6 | Reflect | Classify **fixed / regressed / still-failing / still-passing**; write a lesson |
| 7 | EvoDAG + evolve H_{n+1} | Node + edge in `.autosaddler/evodag.json`; next parent is accepted child or best-dev; evolution may merge accepted components |

**Phased patch scheduling:** first half of `--iters` prefers **capability**
(middleware flags, infra). Second half prefers **steering** (prompt / tool
descriptions / PreToolUse hook text). Replace, do not stack.

**Acceptance:** keep as `active.json` only if train improves (or is already
perfect) **and** dev does not regress. Rejected patches stay in the DAG as
lessons. Final active harness is argmax J_dev among recorded candidates.

## Persistence

All under `.autosaddler/` (gitignored):

| File | Role |
|------|------|
| `active.json` | Winner the **proxy reloads by mtime** (no restart) |
| `evodag.json` | Nodes (harness + scores + lessons) and edges (Δθ) |
| `events.jsonl` | Append-only optimizer events |
| `live-events.jsonl` | Proxy recovery firings for the next diagnose |
| `candidates/h_*.json` | Content-addressed snapshots |

Re-running `optimize` **resumes** the DAG (`tried` patch keys are not repeated).

## Hands-off cycle (default)

`./2_start_mtplx.sh` starts a **background daemon**. Use Kilo as usual.

1. Recoveries append to `live-events.jsonl`.
2. After **90s** of quiet, one optimize iteration runs.
3. An accepted saddle is `active.json`; the proxy reloads it on the next turn.

`--no-autosaddler-daemon` disables this.

## Commands (this Qwen3.8 mtplx stack)

Needs engine + proxy up (`./2_start_mtplx.sh`). Public API is `:8768`.

```bash
./2_start_mtplx.sh                  # engine + proxy + hands-off daemon
./2_start_mtplx.sh status
./2_start_mtplx.sh optimize         # optional one-shot
python3 autosaddler.py --status
python3 autosaddler.py --daemon --iters 1 --idle 90
python3 test_autosaddler.py
```

`--no-llm` uses only the template taxonomy (no extra model call for patch text).

## Live overlay

`qwen38_obl_kilo_proxy.py` reads `load_active()` on each tool turn:

- finish-the-job prefix (`[AS_LOOP]…[/AS_LOOP]`, replaced not stacked)
- empty / fake-action / prose-loop nudge text
- middleware enable flags and `empty_streak_min`
- card sampling remains forced regardless of `kilo.json` temperature

`test_harness.py` uses the same overlay for agent-loop prompts and tool
descriptions, so optimize and Kilo share one saddle.

## Patch taxonomy (Table 1 of the paper)

| Kind | Subtype | Component keys |
|------|---------|----------------|
| Capability | Agent loop logic | `mw_empty_tool`, `mw_fake_action`, `mw_prose_loop`, `mw_after_tool`, `mw_force_tools_on_fake_prose` |
| Capability | Infra | `empty_streak_min` (tool-output cap stays in the proxy) |
| Steering | Prompt rule modify/add | `loop_prefix` (must keep `Do not stop after 1 of N.`) |
| Steering | PreToolUse hook | `empty_tool_nudge`, `fake_action_nudge`, `prose_loop_nudge` |
| Steering | Tool description | `tool_desc.bash` / `read` / `glob` / `grep` |

Unknown keys are rejected. That is structured intervention, not Meta-Harness-style
unconstrained edits.
