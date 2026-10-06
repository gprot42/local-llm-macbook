# Gemma 4 26B-A4B (base) — TensorFold

Run the **base (censored) Gemma 4 26B-A4B** MoE on Apple Silicon with
[TensorFold](https://github.com/ashhart/TensorFold) 0.6.6 (family `gemma4`).
Text output. This is the **MoE** checkpoint (26B total / 4B active); TensorFold's
Gemma 4 kernels cover the MoE layout only and **refuse the 31B dense** packs, so
the dense `gemma-4-31B-it` MLX stacks stay on their own servers.

**Port `:8092`** (reliability proxy; engine `:8102`). Do not bind `:8080` (Gemma / Diffusion) or the
other model ports in this repo.

| | Hugging Face | Size |
|--|--|--|
| Model | `mlx-community/gemma-4-26b-a4b-it-4bit` (base, censored) | ~15 GB |
| Drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` | ~0.8 GB |

The pack is `model_type: gemma4`, `enable_moe_block: true` (128 experts, top-8),
MLX 4-bit in groups of 64 with an 8-bit router — exactly what the `gemma4` lane
requires. Gemma 4 has **no built-in MTP head**, so fast-tier speed comes from the
external DFlash drafter (`z-lab/gemma-4-26B-A4B-it-DFlash`), verified exact
against the engine's own sample. Without a drafter the lane decodes one token per
round (`--no-drafts`).

Fits a 128 GB M5 Max with room to spare (~15 GB weights + a 0.4B drafter). Do not
run it next to another 27B/large stack — one large model at a time.

## Quick start

```bash
./1_setup_download.sh
./2_start_tensorfold.sh          # engine :8102 + reliability proxy :8092, supervised
# stuck port:  ./2_start_tensorfold.sh restart

python3 test_harness.py --gate   # 8 quick checks (also run automatically after start)
python3 test_e2e_opencode.py     # full OpenCode task through the stack, ~5-15 min
```

OpenCode, after the server is up:

```bash
./install-opencode-json.sh --force
```

That merges the `gemma4-tensorfold` provider into
`~/.config/opencode/opencode.json` and sets the default model to
`gemma4-tensorfold/gemma-4-26b-a4b-tensorfold`. Other providers stay.
Restart OpenCode if it is already open. The model entry carries
`"temperature": true` on purpose: OpenCode drops the agent's temperature for
any custom-provider model that does not declare it (see the abliterated
sibling's `findings.md`), and the proxy fills in the engine defaults for any
field a client omits.

Kilo id: `tensorfold-gemma4/gemma-4-26b-a4b-tensorfold`.

## Architecture

```
Kilo / OpenCode
    │  http://127.0.0.1:8092/v1
    ▼
loop_proxy.py            (reliability proxy: repeat guard, step judge + retry, sampling defaults, :8092)
    │  http://127.0.0.1:8102/v1
    ▼
tensorfold serve         (gemma4 lane, DFlash drafter then exact verify, :8102)
    ▼
mlx-community/gemma-4-26b-a4b-it-4bit
```

`./1_setup_download.sh --no-drafter` skips the DFlash drafter (one token per round).

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv, TensorFold `v0.6.6`, pull weights, write `.tensorfold_config` |
| `2_start_tensorfold.sh` | Engine on `:8102`, reliability proxy on `:8092`, supervisor (engine probed every 20 s, restarted on crash). Proxy self-test before start, harness gate after |
| `loop_proxy.py` | Reliability proxy: repeat guard, step judge + retry, sampling defaults, `/harness/health` (same logic as the abliterated stack; ports and defaults differ) |
| `harness_judge.py` | Pure judge rules for one reply (loops, symbol soup, narration, leaks, truncation, bad arguments); the abliterated stack's rules plus the two adjustments under *Calibration* |
| `test_harness.py` | Live API checks (tools, stream, multi-turn); `--gate` also checks the proxy's health and sampling defaults |
| `test_e2e_opencode.py` | Headless `opencode run` of a real coding task through the whole stack |
| `kilo.json` | Provider `tensorfold-gemma4` |
| `opencode.json` | OpenCode provider fragment `gemma4-tensorfold` |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and context

Server defaults are `temperature=0.45`, `top_p=0.9`, `top_k=40`
(`2_start_tensorfold.sh`; `SAMPLING_TEMPERATURE`, `_TOP_P`, `_TOP_K` override
them) — the coding-agent values calibrated on the abliterated sibling, whose
long agentic runs degenerate at Gemma's 1.0 / 0.95 / 64 chat preset; this pack
has the same family and template. `min-p 0.05` is added when the installed
TensorFold has `--min-p` (0.5.0+; the pinned v0.6.6 does). The proxy
fills the same values into any request that omits them; a request that sets
its own values wins. The engine starts with thinking **off** (`--no-thinking`).
OpenCode's `…-thinking` model turns it on per request (`reasoning_effort`, or
`enable_thinking`); the proxy budgets 1,024 / 2,048 / 4,096 tokens for low /
medium / high and raises that step's token cap by the same amount. Pass
`--thinking` on the start script to think on every request.

Context cap defaults to 131072 (server `-c`). The client `limit` in
`kilo.json` / `opencode.json` is 49152 / output 16384 (usable ~33k) — the same
conservative window used across these stacks; raise it if you need more, keeping
`context + output ≤` the server `-c`.

Pinned to **TensorFold 0.6.6** (`TF_VERSION` in `1_setup_download.sh`), upstream
`ashhart/TensorFold`. The gemma4 MoE lane and the tool-call / thought-channel
fixes are in this release. Bump only after re-checking the model loads and
drafted replies stay byte-identical to `"draft": false`.

## Harness (`loop_proxy.py` + `harness_judge.py`)

`loop_proxy.py` owns `:8092` and forwards to the engine on `:8102`. It is the
reliability layer built for the abliterated Gemma 4 stack ([`../../uncensored/archived/gemma4-26b-a4b-abliterated-mlx-tensorfold/`](../../uncensored/archived/gemma4-26b-a4b-abliterated-mlx-tensorfold/), see its
`findings.md` for the failure shapes and the calibration method), ported here on
2026-09-30 with this stack's ports and sampling:

- **Judge + retry.** A request that offers tools is an agentic step. Its reply
  is buffered (TensorFold emits tool calls only at the end, and in-progress
  tool-call text never streams as content), the text is judged as it streams
  (loop, symbol soup, narration instead of tool calls, leaked tool-call / think
  envelopes) and the whole reply at the end (truncation at the token cap,
  unusable tool arguments, a no-op `edit`, a comment-only `bash` command, a
  `write` that would replace a file written earlier in the turn with a stub).
  A failure cancels the generation at the engine, saves the discarded text
  under `.harness_failures/`, and retries the step with a fresh seed,
  temperature 0.3 → 0.2 and a corrective user message (the last attempt adds a
  repetition penalty for loops; `min_p` and the penalty reach the pinned
  TensorFold 0.6.6). After
  three attempts the turn ends with a `[Harness] Stopped: …` message that says
  why. Plain chat (no tools) streams live; only a loop or a leak cuts it.
- **Repeat guard.** A tool call that returns the same result again with no
  write in between is refused in place (`[Harness] REFUSED`), then stopped.
  The same file rewritten 4 times in one turn gets a directive to verify or
  edit instead; at 6 the tools are withheld for one reply so the model sums
  up; 10 ends the turn.
- **Sampling defaults.** Fields a client omits get temperature 0.45, top-p
  0.9, top-k 40 (the engine runs the same defaults). Client values win.
- **Per-step token cap** of 6144 for agentic steps, stall timeout 240 s, and
  the proxy waits up to 90 s for an engine that is restarting.
- **Supervisor** in `2_start_tensorfold.sh`: the engine is probed every 20 s
  and restarted after a crash or three failed probes; the proxy is restarted
  if it dies. `./2_start_tensorfold.sh status` shows both; `stop` and `restart`
  end a running supervisor first (otherwise it would relaunch the engine).
- **Health:** `curl -s localhost:8092/harness/health` — upstream state, counters
  (steps, retries, stops, cuts, stalls, repeat and churn nudges), failures by kind.

**Calibration.** The judge was replayed over the 13 assistant messages the `gemma4-tensorfold` provider produced in OpenCode (11 text parts, 4 tool calls, 2026-09-29, the heretic pack on this stack's ports): the only trips are three genuine `<|channel>thought` leaks from before TensorFold's channel fix, which is what the judge should catch; healthy replies max out at symbol share 0.12 and no repeats. The thresholds come from the abliterated sibling's 90-message calibration (same family, same template). Two adjustments were made in this copy of `harness_judge.py`, both from replaying the judge over the other local TensorFold models' transcripts as extra evidence (Ternary Bonsai 2: 263 messages, zero trips; Qwen3.8-27B: 1063 messages, one false positive): markdown table rows are exempt from the duplicated-line rule (a healthy 11k-char sectioned report repeated the table header `| File | Lines |` once per section, six times; the loop rule still catches contiguous repeats), and the leak rule also matches the Hermes `<tool_call>{…}` envelope and a `<think>` opener (neither occurs in any healthy transcript). Thresholds are otherwise the abliterated Gemma calibration.

Knobs (environment, read at start): `HARNESS_STEP_MAX_TOKENS=6144`,
`HARNESS_MAX_ATTEMPTS=3`, `HARNESS_STALL_SECONDS=240`, `HARNESS_STREAM_PROSE=1`
(stream agentic prose live; failures are then cut, not retried),
`LOOP_MAX_ROUNDS=200`, `SAMPLING_TEMPERATURE=0.45` (and `_TOP_P`, `_TOP_K`,
`_MIN_P`). Tests: `python3 loop_proxy.py --self-test` (judge calibration cases,
repeat and churn guards, scripted engine scenarios), `python3 test_harness.py
--gate` (adds `harness proxy health` and `harness sampling defaults`), and
`python3 test_e2e_opencode.py` for a real OpenCode task through the stack.
`.harness_failures/` (git-ignored) keeps the last 60 discarded replies.

## Troubleshooting

**Harness gate fails `harness proxy health`** — `:8092` is answered by something
that is not this stack's `loop_proxy.py` (an old proxy or a bare engine). Run
`./2_start_tensorfold.sh restart`.

**`tensorfold info` rejects the repo** — the installed CLI is older than the
pinned release, or the pack is not the MoE checkpoint (dense Gemma 4 is refused).
Re-run `./1_setup_download.sh`.

**Lane refuses the checkpoint** — the `gemma4` kernels need `enable_moe_block`
in every layer and **4-bit MLP** projections with only the **router at 8-bit**.
Many uncensored packs (`mlx-community/…-heretic-4bit`, `supergemma…-multimodal`)
quantise the MLP at 8-bit and are refused — verify a pack's `config.json`
quantization block before use. `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
has the correct layout; the lane's reference `mlx-community/gemma-4-26b-a4b-it-4bit`
is the censored fallback.

**Drafter does not load** — the DFlash drafter is optional; start with
`--no-drafts` to serve without it (slower, one token per round) and report the
loader error.
