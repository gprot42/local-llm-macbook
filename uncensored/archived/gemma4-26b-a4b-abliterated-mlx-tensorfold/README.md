# Gemma 4 26B-A4B (abliterated, uncensored) — TensorFold

Run an **uncensored** Gemma 4 26B-A4B MoE on Apple Silicon with
[TensorFold](https://github.com/ashhart/TensorFold) 0.5.0 (family `gemma4`).
Text. The weights are a **locally self-quantised** 4-bit pack (not on Hugging
Face) — see *Reproduce* below. Censored sibling: [`../../censored/gemma4-26b-a4b-mlx-tensorfold/`](../../censored/gemma4-26b-a4b-mlx-tensorfold/).

**Port `:8094`** (reliability proxy; engine `:8104`). Do not bind `:8080` or the other model ports.

| | Source | Size |
|--|--|--|
| Model | **local** `~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit` | ~13 GB |
| Drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` | ~0.8 GB |
| Engine | TensorFold **v0.6.0 + 4 patch branches** (thought-channel fix, repetition penalty, tool-call repair, prompt-cache fix for agentic steps), branch `combined-v0.6.0` in `./.tensorfold-src`, installed by `1_setup_download.sh` | — |

The pack was quantised from `SevenOfNine/Gemma-4-26B-A4B-It-Abliterated` (a
Heretic-abliterated Gemma 4, bf16) and given the **base pack's standard Gemma
template** so the reasoning channel stays gated — verified clean (no `<|channel>`
leak) **and** uncensored on the `gemma4` lane. Fits a 128 GB M5 Max easily.

## Reproduce the local pack

The 4-bit weights aren't published; recreate them with:

```bash
# 1) download the abliterated bf16 source (~52 GB, transient)
python -c "from huggingface_hub import snapshot_download; snapshot_download('SevenOfNine/Gemma-4-26B-A4B-It-Abliterated')"

# 2) quantise to 4-bit / group 64 (mlx_lm has gemma4 support)
mlx_lm.convert --hf-path SevenOfNine/Gemma-4-26B-A4B-It-Abliterated \
  -q --q-bits 4 --q-group-size 64 \
  --mlx-path ~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit

# 3) swap in the base pack's gated template (SevenOfNine's forces the channel)
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('mlx-community/gemma-4-26b-a4b-it-4bit'))")
cp "$BASE"/chat_template.jinja "$BASE"/tokenizer_config.json \
   ~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit/
```

The bf16 source (~52 GB) can be deleted after step 2. To make this shareable,
upload the 4-bit dir to an HF repo and point `HF_MODEL` (in `.tensorfold_config`)
at it instead of the local path.

## Quick start

```bash
./1_setup_download.sh          # venv only; model is already local
./2_start_tensorfold.sh        # engine :8104 + reliability proxy :8094, supervised, detached
python3 test_harness.py --gate # 8 quick checks (also run automatically after start)
python3 test_e2e_opencode.py   # full OpenCode task through the stack, ~5-15 min
```

The start command returns once the stack answers. The stack runs detached, in
its own session with no terminal, so closing a terminal, an editor or the Claude
app no longer stops it (on 2026-10-01 a closed terminal tab took the whole stack
down with it). Manage it with `./2_start_tensorfold.sh status`, `logs` (follows
`.tensorfold_stack.log`), `stop` and `restart`; `--foreground` runs it attached
as before. `stop` and `restart` end the supervisor first, so it does not relaunch
the engine, and only ever signal the processes *listening* on `:8094`/`:8104`
(the old `lsof -ti :PORT` also matched connected clients such as OpenCode).

**Thinking mode.** Off by default. The model entry in `opencode.json` declares
`"reasoning": true`, so OpenCode offers the variants low, medium and high; picking
one sends `reasoning_effort` and the engine thinks for that request. The proxy gives
it a thinking budget of 1,024 / 2,048 / 4,096 tokens (`HARNESS_THINKING_BUDGET`
for other requests) and raises that step's token cap by the same amount, so the
answer or tool call still fits. Reasoning comes back separately and never in the
answer. Expect a step to take roughly two to eight times longer.

OpenCode: `./install-opencode-json.sh --force` merges the
`gemma4-abliterated-tensorfold` provider and sets it default. The model entry
carries `"temperature": true` on purpose: OpenCode drops the agent's
temperature for any custom-provider model that does not declare it, and this
pack degenerates at the engine default of 1.0 (see `findings.md`). Kilo id:
`tensorfold-gemma4-abliterated/gemma-4-26b-a4b-abliterated-tensorfold`.

## Reliability layer (`loop_proxy.py` + `harness_judge.py`)

Long agentic runs on this pack used to die in three ways: the model narrated
tool calls instead of making them, slid into LaTeX / punctuation soup, or
looped one unit for six minutes to the 16384-token cap. `:8094` is a proxy in
front of the engine that makes a step fail *cleanly and invisibly* instead:

- **Judge + retry.** A request that offers tools is an agentic step. Its reply
  is buffered (TensorFold emits tool calls only at the end anyway), the text is
  judged as it streams (loop, symbol soup, narration, leaked `<|tool_call>` /
  `<|channel>` envelopes) and the whole reply at the end (truncation at the
  token cap, unusable tool arguments). A failure cancels the generation at the
  engine, saves the discarded text under `.harness_failures/`, and retries the
  step with a fresh seed, temperature 0.3 → 0.2, `min_p` and a corrective user
  message (the last attempt adds a repetition penalty for loops). After three
  attempts the turn ends with a `[Harness] Stopped: …` message that says why.
  Plain chat (no tools) streams live; only a loop or a leak cuts it, with a note.
- **Repeat guard.** A tool call that returns the same result again with no
  write in between is refused in place (`[Harness] REFUSED`), then stopped.
  The same file rewritten 4 times in one turn gets a directive to verify or
  edit instead (live, `z/tetris.js` was rewritten 13 times in 214 s); at 6 the
  tools are withheld for one reply so the model sums up and the turn ends with
  its own report; 10 ends the turn. Doomed calls are caught before they run and
  the step is retried: a no-op `edit` (old and new text identical), an `edit`
  identical to one already applied this turn (a repeat corrupts the file), a
  call missing an argument its tool's schema requires, a `bash` command that is
  only comments, a path with backticks or quotes in it, an argument carrying a
  raw template token such as `<|"|>`, and a `write` that would replace a file
  written earlier in the turn with a stub (under 35% of its size). A final
  report that cites a check (`node --check`, `pytest`, …) that never ran in the
  turn is retried once so the model actually runs it; it never stops a turn.
- **Check nudges.** After a code file (`.js`, `.py`, `.sh`, `.json`) changes, the
  next request tells the model to run that file's syntax check before changing
  it again. Retries of behavioural mistakes sample wider (0.6 → 0.8) instead of
  tighter, and a no-op or repeated edit is retried without the edit tool.
- **Repairs and soft checks.** Backticks or quotes around a path are stripped
  and the call relayed, rather than retried. A rewrite at 35–60% of the file's
  largest version this turn, a final report while a code file sits below 60%
  of its peak or was not checked after its last change, a report that pastes
  a file's code instead of writing it, and a write to a versioned copy
  (`x_v2.js`, `x_final.js`) of a file already written, are each retried once
  (never a stop).
- **Model limit.** On a single-file game the model still fails about one run in
  three by looping inside long writes; the harness makes that fail cleanly and
  honestly. For large single-file writes use a stronger model or the base pack.
- **Sampling defaults.** Fields a client omits get temperature 0.45, top-p 0.9,
  top-k 40, min-p 0.05 (the engine runs the same defaults). Client values win.
- **Per-step token cap** of 6144 for agentic steps (a legit large file fits; a
  runaway at 45 tok/s no longer runs six minutes), stall timeout 240 s, and the
  proxy waits up to 90 s for an engine that is restarting instead of failing.
- **Supervisor** in `2_start_tensorfold.sh`: the engine is probed every 20 s and
  restarted after a crash or three failed probes; the proxy is restarted if it
  dies. `./2_start_tensorfold.sh status` shows both.
- **Health:** `curl -s localhost:8094/harness/health` — upstream state, counters
  (steps, retries, stops, cuts, stalls, repeat nudges), failures by kind, config.

Knobs (environment, read at start): `HARNESS_STEP_MAX_TOKENS=6144`,
`HARNESS_MAX_ATTEMPTS=3`, `HARNESS_STALL_SECONDS=240`, `HARNESS_STREAM_PROSE=1`
(stream agentic prose live; failures are then cut, not retried),
`LOOP_MAX_ROUNDS=200`, `SAMPLING_TEMPERATURE=0.45` (and `_TOP_P`, `_TOP_K`,
`_MIN_P`). Tests: `python3 loop_proxy.py --self-test` (judge calibration,
repeat guard, and 12 scripted engine scenarios: soup mid-stream, truncation,
three failures → stop, live chat cut, defaults, 503 retry, 400 passthrough,
stall, engine down, health), `python3 test_harness.py --gate`, and
`python3 test_e2e_opencode.py` for the real thing (it fails on a missing expected
file, code without a game loop and key handler for the default prompt, or a turn
the harness had to stop).

## Notes

- **Why local:** every ready-made uncensored 4-bit Gemma-4-26B-A4B pack was
  unusable on the lane — `…-heretic-4bit`/`supergemma…-multimodal` use 8-bit MLP
  (refused), and `supergemma…-uncensored-v2` / `SevenOfNine` ship a reasoning
  **channel** template that leaks `<|channel>` markers. Self-quantising +
  swapping the gated template is the clean path.
- **Drafter:** the base-model DFlash drafter works against the abliterated
  target (drafts are exact-verified). If acceptance is poor, `--no-drafts`.
- Model dir is git-ignored (`model/` and the cache path); the *recipe* above is
  the source of truth.
- **Engine build:** upstream v0.6.0 plus three patch branches that are not upstream
  yet (see `findings.md`, "Rebase onto TensorFold v0.6.0", and `tensorfold-pr-drafts.md`).
  Do **not** run `tensorfold update` in this venv: it installs the stock release over the
  patches (the engine starts with `--no-update-check` so it does not suggest it).
  Rebuild: `./1_setup_download.sh` (uses `./.tensorfold-src` when present). Roll back to the
  previous 0.5.0-based build: `venv/bin/pip install --no-deps --force-reinstall
  .tensorfold-rollback/tensorfold-0.5.0-py3-none-any.whl`, then `./2_start_tensorfold.sh restart`.
- `.harness_failures/` (git-ignored) keeps the last 60 discarded replies: each file
  has the reason, the request summary and the text, for diagnosing a new failure shape.
