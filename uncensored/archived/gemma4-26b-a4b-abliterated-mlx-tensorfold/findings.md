# Findings — Gemma 4 26B-A4B (abliterated) on TensorFold

**Date:** 2026-09-30 · **Machine:** Apple M5 Max, 128 GB · **Engine:** TensorFold 0.5.0 + our two `gemma4` fixes (PR [ashhart/TensorFold#157](https://github.com/ashhart/TensorFold/pull/157))
**Model:** local self-quantised `gemma-4-26b-a4b-abliterated-4bit` (26B-total / 4B-active MoE, 4-bit, 4.502 bits/weight)
**Drafter:** `z-lab/gemma-4-26B-A4B-it-DFlash` (external DFlash) · **Conditions:** only this model loaded (uncontended GPU)

## Status

Uncensored **and** agentic-capable: chat is clean/uncensored, and tool calling now works in OpenCode after our two lane fixes. Fast — ~170–190 tok/s on code generation.

## What we changed in *our* model / stack

The 4-bit weights are **local** (not on Hugging Face). They were produced from an abliterated bf16 checkpoint, and the stack installs a patched TensorFold:

1. **Quantised** `SevenOfNine/Gemma-4-26B-A4B-It-Abliterated` (bf16, Heretic-abliterated) → 4-bit/group-64 with `mlx_lm.convert` (mlx-lm has `gemma4`). Result: 4.502 bpw, ~13 GB, at `~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit`.
2. **Swapped in the base pack's *gated* chat template** (`mlx-community/gemma-4-26b-a4b-it-4bit`). The abliterated pack (and supergemma packs) ship a template that force-opens the reasoning channel, leaking `<|channel>thought`; the base template gates it. Weights are abliterated (uncensored); template is the clean base one.
3. **Stack installs TensorFold from our fork branch** `gprot42/TensorFold@fix-gemma-tool-call-colon-prefix` (`TF_REPO`/`TF_VERSION` in `1_setup_download.sh`) so the two tool/agentic fixes below are present. Revert to the official release once PR #157 is merged.

## Two upstream fixes we made (PR #157)

Both were `gemma4`-lane bugs reproduced on the **stock** `mlx-community/gemma-4-26b-a4b-it-4bit` too (not our abliteration):

- **Tool calls leaked as text** (`<|tool_call>:NAME{...}` → content, `tool_calls: null`). The parser only handled the `call:NAME` form; the model also emits `:NAME` (no `call`). Made the `call` keyword optional in the gate + regex. Fixes upstream #121.
- **Spontaneous thought channel leaked when thinking off** — on a continuation after a tool result the model opens `<|channel>thought…<channel|>`, which fell to the gpt-oss `parse_harmony_output` path and leaked. Now Gemma replies route through `split_thinking` (channel-aware) whether or not thinking is on.
- **Thought channel that opens *anywhere*, not just at the start (2026-09-30, commit `c3f3f1b`)** — the fix above only stripped a channel anchored at position 0. Live, the packs also open it *after* visible text, behind a leading newline, or behind a stray/doubled `<|channel>` opener (seen in OpenCode sessions, e.g. `<|channel><|channel>thought\n<channel|>The findings…`), and those leaked the markers into the answer. `split_thinking` now finds the opener wherever it is (text before = answer; stray/doubled partial opener dropped; block stripped, recursively for the remainder), and streaming still holds a partial opening tag so the visible answer only grows.

All three verified live (stock + abliterated, streaming + non-streaming); regression tests added; **full suite green (3463 passed, 440 skipped)**. The local venv is reinstalled from the fork commit and the `:8104` engine restarted, so the running server carries all three fixes.

## Timings (TensorFold 0.5.0 + fixes, uncontended)

### Code generation (drafted)

| Task | Tokens | tok/s |
|---|---|---|
| Thread-safe LRU cache class | 250 | **172.9** |
| nth-prime function | 200 | **186.3** |
| flatten nested list (greedy) | 180 | 183.3 |

**Drafting vs serial** (same greedy prompt): 183.3 tok/s drafted vs 141.5 serial → **~1.30× from DFlash**. (MoE with ~4B active → fast even serial.)

### Agentic tool calls (system prompt + 6 tools)

| Request | Output tok | tok/s | result |
|---|---|---|---|
| list directory | 12 | 85.7 | `list` ✅ |
| write file (FastAPI hello) | 18 | 108.2 | `write` ✅ |
| edit a route | 13 | 111.5 | `edit`→call ✅ |
| continue after tool result | 19 | 104.7 | `read` ✅ (no channel leak) |

Tool-call replies are short (just the call), so per-call latency is **<0.2 s** and the tok/s is overhead-dominated (~85–111) rather than the sustained code-gen rate. The point: **every call now parses into structured `tool_calls`** — including the multi-turn continuation — so agentic coding works end-to-end.

### Prefill

Cold prefill ~3,000 tok/s (a ~21.7k-token prompt in ~7 s); warm/short prompts TTFT ~0.05–0.3 s (prefix cache).

## Caveats

- Single-run, indicative numbers; expect run-to-run variance (code ~165–190, prose ~140). Draft acceptance is higher on code than prose.
- Weights are local; the *recipe* above is the source of truth (reproducible on another machine). The TensorFold fixes are pinned via the fork until #157 merges.

## Narration & build prompt (2026-09-30)

The abliterated weights are chatty on **open-ended / research** prompts — the model narrates ("let's do it", "let's go", "the first action is bash") before acting. On **concrete coding tasks** it already goes straight to tool calls (0 preamble), and temperature (1.0 / 0.6 / 0.3) doesn't change it — it's the ablated style, not a bug (the `<|channel>thought` leaks seen in older sessions are fixed, see above).

**Mitigation:** tightened the OpenCode **build-agent prompt** to hard-suppress narration — a "No narration" paragraph that bans plan/preamble, tool-call announcements ("the first action is bash", "I'll now edit…", "let me…") and filler openers ("let's do it", "let's go", "sure", "great", "okay"); the only prose allowed is one final result line. Applied to the live `~/.config/opencode/opencode.json` and to the repo source `censored/ternary-bonsai-2-27b-tensorfold/opencode.json` (the generic build prompt the live config is installed from). Takes effect on **OpenCode restart** — it's a client-side prompt, so no TensorFold server restart is needed.

**Caveat:** this reduces narration, it does not eliminate it on open-ended asks — the tendency is in the weights, not the prompt.

## Large-file generation slop (abliteration) (2026-09-30)

Asking this pack to write a whole file (e.g. a single-file HTML game) can produce a wall of repeated markup, e.g. `write:value:<|"|>Wrote file successfully.</td></tr></tbody></table>` × N. Root-caused: the model emits a `write` tool call whose `content` string is a long HTML file, degenerates into a repetition loop (`</td></tr></tbody></table>`), fills the token budget so `finish_reason=length` before the closing `<tool_call|>`, and the **unterminated** tool-call block can't be parsed — so the raw `<|tool_call>call:write{content:<|"|>…` leaks into the chat as text.

This is **not** the channel bug and **not** a lane bug. Evidence: on the *same* prompt the **censored base** pack (`:8102`) returns a clean, complete `write` call (`finish_reason=tool_calls`, 6096-char HTML arg, no repetition, no leak); the **abliterated** pack hit `length` with an unterminated, leaked call. So it's **abliteration-induced degeneration** on long generations. TensorFold exposes only temperature/top-p/top-k — **no repetition penalty** — so there is no engine knob to damp the loop.

Mitigations (none free): (1) use the **base pack** or a stronger model for large single-file writes (best reliability; loses "uncensored"); (2) raise the per-request output cap so *legit* large files finish before truncation (does nothing for a true loop); (3) upstream: add a repetition penalty to TensorFold (real fix, larger change); (4) engine-cosmetic: hide an unterminated `<|tool_call>…` block at `finish` so it doesn't slop into chat (stops the wall of text, but the file still isn't written). Bottom line: the abliterated pack is chat-strong but degrades on long agentic file writes.

## Repetition / frequency / presence penalty — TensorFold feature (2026-09-30)

Mitigation (3) above, built as its own feature on a **separate branch and PR** (kept out of the
#157 leak-fix branch on purpose). TensorFold previously exposed only temperature / top-p / top-k,
so there was no way to damp the loop; this adds the standard penalties.

**Branch:** `gprot42/TensorFold@feat-repetition-penalty` (cut from upstream `main`).

**Changes, in the order made — follow these to reproduce the feature:**

1. `src/tensorfold/engine/exact_sampling.py` — `Sampling` gains `repetition_penalty`,
   `frequency_penalty`, `presence_penalty`, `penalty_last_n` (every one a no-op at its default)
   plus a `has_penalty` flag. New `penalized()` applies the HF repetition rule (divide a positive
   logit / multiply a negative one) and the OpenAI-style frequency/presence subtraction to the
   candidate logits, *before* top-k/top-p/min-p/Gumbel, so a much-repeated token drops among the
   candidates. `choose()`, `choose_rows()` and `sample_rows()` take an optional per-row `recent`
   history (None = no penalty); the MLX nucleus fast path is skipped when a penalty is on.
2. `src/tensorfold/server/request_options.py` — `parse_numbers` validates the four fields
   (`repetition_penalty` > 0; frequency/presence may be negative like OpenAI; `penalty_last_n` an
   integer), and `_resolve_sampling` puts them on the `Sampling`.
3. `src/tensorfold/cli.py` — `--repetition-penalty`, `--frequency-penalty`, `--presence-penalty`,
   `--penalty-last-n`, and reads the same keys from the model's `generation_config.json`.
4. `src/tensorfold/server/http.py` — forward those four keys from the request body (the HTTP layer
   whitelists body keys, so without this the fields were silently dropped — the bug that made an
   early live test look like a no-op).
5. **The effecting change — decode reroute (`feat` commit `412f613`):** a request that sets a
   penalty runs its stream with **drafts off** (`LaneStream`'s built-in serial reference mode) and
   draws through the extended **CPU exact path** with its reply-so-far history, so the penalty is
   applied exactly per token. The Metal decode kernel keeps its batched fast path untouched whenever
   no stream in the round has a penalty.
   - `server/scheduler.py`: drafts off when `sampling.has_penalty`.
   - `engine/lane_family.py`: `_draw()` routes a penalised draw to `exact_sampling.sample_rows`
     with `recent=history`; new `_recent(stream, n)` helper.
   - `engine/family_shared.py`: `_draw_streams` carries the stream and does a per-stream CPU draw
     when any stream in the round is penalised.
   - `engine/family_prefill.py`: pass each row's `recent` on the first / pipelined tokens.

**Design note.** The penalty deliberately reroutes to CPU (drafts off) instead of modifying the
hand-written Metal sampling kernel — correct and self-contained, at the cost of the speculative
speedup *while a penalty is active*. Teaching the kernel (and the CUDA rank headers / `pack_sampling`)
to carry per-row history for full-speed penalised drafting is a later step.

**How to enable (client side).** Set `repetition_penalty` (try 1.2–1.5) in the request body, or in
OpenCode's model `options`, or start the server with `--repetition-penalty`. It only applies when
`temperature > 0` (a greedy request has no `Sampling`).

**Tests.** `tests/test_repetition_penalty.py` covers the penalty math, the request validation and
resolution, the selection flips, and the per-row history; the decode-loop / lane / stream suites and
the full suite stay green.

**Live build.** The abliterated venv (`:8104`) runs a *local combined build*: the `feat` branch plus
the #157 channel/tool fixes overlaid, so the running server has both. Reproduce with
`pip install --force-reinstall --no-deps <fork checkout>` from a tree that has both, then restart
`./2_start_tensorfold.sh`. (The stack's `TF_VERSION` pin still points at the #157 branch; switch it
once the two PRs merge upstream.)

**Result (live A/B on the game prompt).** The penalty *engages* — verified two ways. Speed: with no
penalty the draw runs the drafted fast path at ~180 tok/s; with a penalty it drops to ~123 tok/s,
exactly the loss of speculative decoding as the stream falls back to the serial CPU path (so the
reroute is confirmed active). Behaviour: no penalty → `finish=length`, 1600 tokens, the
`</td></tr></tbody></table>` loop; `repetition_penalty=1.3` → the loop is gone (1.15 is too weak and
still loops). **But it does not cleanly fix the large-file write.** On
this abliterated model the penalty just moves the failure: the model either stops early with a
half-written file, or degenerates into punctuation soup (`}|}{|}:{}…`) inside the content string, so
the `write` call — even when it is terminated with `<tool_call|>` — has malformed arguments and still
leaks as text (`tool_calls: []`). The reason is fundamental: repetition penalty punishes tokens that
*legitimately* repeat in structured output (HTML tags, JSON braces / quotes / `<|"|>`), so it fights
the loop and the file's own syntax at once. It is a good general anti-repetition knob for prose/chat;
it is the wrong single tool for long agentic file writes.

**So the game-slop verdict stands:** for reliable large single-file writes use the base pack or a
stronger model. The repetition penalty is a useful, now-available knob (best on prose), and a fuller
agentic fix would pair a gentler penalty with mitigation (4) above — repair or hide an unterminated /
malformed `<|tool_call>` block at `finish` instead of leaking it.

## Unterminated tool-call repair — mitigation (4) prototype (2026-09-30)

The other half of the slop: even with the loop broken, a large `write` can run out of tokens after it
opens `<|tool_call>call:write{…}` but before `<tool_call|>`. With no complete block to match, the raw
markup falls through into the reply text (`tool_calls: []`, content full of `<|tool_call>…`). This
prototype makes that fail cleanly instead.

**Branch:** `gprot42/TensorFold@feat-toolcall-repair` (off upstream `main`). **No PR yet** — held until
#157 is approved and after live testing.

**Change (one file): `src/tensorfold/server/tools.py`.**

- `_repair_gemma_call(fragment, known)` — best-effort `(name, arguments)` from an unterminated
  `call:NAME{…}`: keeps the complete `key:value` pairs and salvages a truncated final string value to
  the end of the reply.
- `_repair_leaked_tool_calls(content, known)` — for each leaked `<|tool_call>` opener: a **terminated**
  block (has `<tool_call|>`) is left exactly as normal parsing left it (a small terminated malformed
  call staying as text is deliberate upstream behaviour); an **unterminated** block is repaired into a
  structured call when it names an offered tool, hidden when it names an unoffered tool, and plain text
  that merely mentions the marker is left untouched.
- `parse_tool_calls_from_content` runs the repair on both the no-envelope early return and the normal
  end — only on the lenient reply path (`max_calls is None`) and only when `<|tool_call>` is still in
  the text.

**Scope.** Every block reaching the repair already failed the strict parse (unterminated at the token
limit; an unbalanced `<|"|>` string; or a value the model emitted as a bare degenerate token instead of
a `<|"|>` string). The rule is by *salvageability*, not by a single syntactic signature: if the args
can be salvaged for an offered tool (at least one field) the block becomes a structured call; an
*unterminated* call that can't be salvaged is hidden (it was cut off, its text is useless); a
*terminated* block that can't be salvaged (a bare `{location}` with no `key:value`) is left as text,
matching upstream's deliberate behaviour; text after a terminated block is preserved. Live-observed
shapes this covers: the truncated big write, the `glob{pattern:<|"|>*/}` unclosed string, and
`write{content:<|"|>…<|"|>,filePath:mapsto_path_…_now}` (balanced delimiters, degenerate bare value).

**Tests.** `tests/test_toolcall_repair.py` (repair, truncated-string salvage, hide an unoffered call,
keep plain text, terminated-left-as-text, complete-call regression); existing tool-call suites green.

**Live build.** Overlaid into the abliterated venv (`:8104`), which now runs channel fix + penalty +
the #157 tool-call fix + this repair together (a *local* build; each of the three is a distinct branch,
not one PR).

**Result (live).** Forcing the cut-off (`max_tokens=300`) on the game prompt: without the repair this
leaks `<|tool_call>call:write{content:<|"|><!DOCTYPE…` as raw text; with it the reply is a structured
`write` call carrying the salvaged partial content (~946 chars of valid HTML) with **zero markup in the
content**. At a normal token cap the same prompt completes as an ordinary `write` call (no leak either
way). Caveat: the salvaged call only has the fields the model emitted before the cut — here `content`
but no `path` (the model wrote content first) — so it is a *clean, retryable* partial, not a
guaranteed-valid write. That is the intended win: no wall of markup in the chat, a structured call
instead. Pair it with a stronger model (or the base pack) when the write must actually succeed.

All three live-observed leak shapes are verified fixed on the running server: the truncated big write,
the `glob{pattern:<|"|>*/}` unclosed string, and the `write{…,filePath:mapsto_path_…_now}` degenerate
bare value each come back as a structured call with **no markup in the content**, while a bare
`{location}` still stays as text (upstream). A real agentic "find all Python files" request returns a
clean `glob` call. Full suite on the repair branch: 3469 passed, 440 skipped.

## OpenCode drops the agent temperature unless the model entry says `temperature: true` (2026-09-30)

**What we saw.** Every agentic step in the sessions that degenerated ran at the engine default
temperature 1.0, although the live build agent is configured with `temperature: 0.35`. Captured on
the wire (a logging proxy in front of the engine, `opencode run` headless):

| model entry | request body OpenCode sent (build agent) |
|---|---|
| `"temperature": false` (what this stack shipped) | `max_tokens=16384, top_p=0.9, tools=9, stream=true` — **no `temperature`** |
| `"temperature": true` | `temperature=0.35, top_p=0.9, max_tokens=16384, …` |

TensorFold treats an absent temperature as "use the server default" (`--temperature 1.0` at the
time), so the abliterated pack sampled at 1.0 / top-p 0.95 / top-k 64 for the whole run.

**Why (OpenCode 1.18.33 source, `packages/opencode/src/session/llm/request.ts`):**

```ts
temperature: input.model.capabilities.temperature
    ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
    : undefined,
topP: input.agent.topP ?? ProviderTransform.topP(input.model),
topK: ProviderTransform.topK(input.model),
```

and `packages/opencode/src/provider/provider.ts`, for a model defined in `opencode.json`:

```ts
temperature: model.temperature ?? existingModel?.capabilities.temperature ?? false,
```

So the model-level `temperature` is a **capability flag** ("this model accepts the temperature
parameter" — it exists because OpenAI's o-series reject it), it **defaults to `false` for a
custom provider's models**, and when it is false the agent's temperature is dropped silently.

**Bug or not?** Not a bug in the strict sense: the behaviour is deliberate and consistent with the
flag's meaning, and our `opencode.json` set the flag to `false` (copied from the Qwen TensorFold
stack). Certainty is high — two captured request bodies both ways plus the source. What *is* worth
reporting upstream is the footgun around it: (1) the default is `false` for custom providers, so a
local model silently ignores `agent.temperature` unless you know to add `"temperature": true`;
(2) `top_p` is still sent when the flag is false, so the flag does not really mean "no sampling
parameters"; (3) `agent.top_k` is never sent (only `ProviderTransform.topK`); (4) no warning at
config load when an agent sets a temperature the model entry will drop. A GitHub issue would be a
low-priority UX/consistency report ("warn or document when agent.temperature is dropped; send
top_p under the same gate"), not a blocker: the fix on our side is the one-line `"temperature":
true` in `opencode.json` (applied to the repo file and the live `~/.config/opencode/opencode.json`
via `./install-opencode-json.sh --force`), and loop_proxy.py now fills in sane sampling for any
client that omits it, so the stack no longer depends on the client getting this right.

**Related property we rely on (TensorFold, verified live):** with tools offered, TensorFold streams
*prose* as content deltas but emits tool calls only at the end, as structured `tool_calls` deltas
(`stream_tool_call_deltas` after `app.chat` returns; in-progress `<|tool_call>` text is hidden by
`hide_tool_calls`). A probe that asked for a `write` produced 0 content deltas and 2 `tool_calls`
deltas. So a proxy judging the streamed text of an agentic step sees prose only, never code — which
is why loop_proxy.py can apply prose-shaped rules (symbol share, LaTeX density, narration) without
false positives on the file being written. The flip side: a tool call whose *content* loops is
invisible until it finishes, which is what the per-step token cap is for.

## Reliability layer: step judge + retry proxy, supervisor, sampling (2026-09-30)

**Symptom (OpenCode session `ses_f0bfa4ce…`, prompt "create a tetris game in folder named z").**
The transcript shows the failure recur three times in one session, all in the *build* agent with
tools offered:

| step | what the model did | outcome |
|---|---|---|
| 21:34 | after 3 writes, a 3,606-token *text* reply: a leaked `<|tool_call>call:write{…}` whose args were malformed | shown as prose, file not written |
| 21:36 | "is game complete?" → 2 writes, then **16,384 tokens** of prose (`finish=length`): a plan, then "Wait / Actually / Let's" rambling, then LaTeX soup (629 `$`) | **6 minutes** at 44.8 tok/s (draft acceptance collapsed), nothing written |
| 21:48 | "hi" → "The Tetris game is fully prepared … the write calls will run …" then `$-$-$(-x[m|size(-$` soup | aborted by hand after 70 s |

Three separate weaknesses lined up:

1. **Sampling.** Every step ran at temperature 1.0 / top-p 0.95 / top-k 64 — OpenCode never sent the
   agent's 0.35 because the model entry said `"temperature": false` (previous section). The
   abliterated weights degenerate at 1.0 on long generations; at 0.3–0.5 they mostly do not.
2. **No judge.** The proxy relayed bytes. Nothing distinguished a tool call from narration about a
   tool call, or prose from soup, so a bad step ran to the 16384-token cap, and its text entered the
   transcript, where it primes the next step to do the same.
3. **Deterministic failure.** TensorFold keys an omitted `seed` to the prompt, so re-sending the same
   step reproduces the same degenerate output byte for byte (two 1,627-token replies in the log share
   `sha=13421e8be708`). A retry must change the seed or the sampling.

**Fix (all in this stack; `loop_proxy.py`, new `harness_judge.py`, `2_start_tensorfold.sh`,
`opencode.json`, `kilo.json`, `test_harness.py`, new `test_e2e_opencode.py`).**

*Judge (`harness_judge.py`).* Pure functions over the reply text and the finished reply. Rules and
thresholds were calibrated on all 90 assistant messages this model produced in OpenCode (37 clean
final replies, 49 tool-call steps, 4 failures); every failure is far past every threshold and no
healthy reply crosses one:

| rule | threshold | healthy max (corpus) | failures |
|---|---|---|---|
| loop: one unit repeated back to back (prose) | 12 repeats, unit 3–160 chars, or tail compresses <8% | 5 (in code) | 700 (`</td></tr></tbody></table>`) |
| loop: same non-trivial line | 6 times | 1 | 98 ("Let's go.") |
| gibberish: non-alphanumeric share of a 600-char window (fences, inline code, table rules stripped) | ≥0.40 once or ≥0.30 twice in a row | 0.17 prose, 0.22 leaked code | 0.36–0.57 |
| gibberish: `$` per 600-char window | 12 | 2 in a whole reply | 40+ |
| narration: "let's go / the first action is / wait, actually" phrases, no tool call | ≥8 hits, ≥1.5 per kchar, ≥3,000 chars | 6 hits in 2.2k chars (below the size floor) | 134 and 414 hits |
| leak | raw `<|tool_call>…{`, `<|channel>thought`, reply opening `thought\n` (unfenced) | quoted in backticks: ignored | 2 |
| truncated | `finish_reason=length` or `completion_tokens == max_tokens` (agentic only); when the capped content repeats a unit 12+ times or a 16+-char line 10+ times it is reported as **loop** instead, so the retry names the real cause and the last attempt gets the repetition penalty | a large distinct file stays `truncated` | 1; later **3 of 3** capped writes in a live session were runaways (a comment pair ×180, `colors_final_final…`, a paragraph ×20) |
| bad-args | non-JSON arguments; a path >400 chars / multi-line / a bare token >40 chars with no `/` or `.`; a string argument looping ≥40 repeats; an `edit` with identical old/new strings; a `bash` command that is only comments; a path containing `` ` " ' < > | ``; any argument containing a raw template token (`<|"|>`, `<|tool_call>`, `<|channel>`…) | 20-row matrix literal passes; paths with spaces/parens pass | `filePath: mapsto_path_…_now`; 5× no-op edit; 2× comment-only bash; `write z/tetris.js```` then 12 failing `rm` steps; bash `부터: rm …<|"|>` (live e2e runs) |
| shrinking-rewrite (proxy, uses the turn's history) | a `write` under 35% of the largest earlier write to the same path this turn, when that was ≥1,500 chars | a modest trim passes | the 10th rewrite of `z/tetris.js`: 542 bytes over a complete game |
| churn (proxy repeat guard) | same path written 4× → directive; 6× → tools withheld for one reply (model sums up); 10× → stop | 3 rewrites are iteration | 13 rewrites in 214 s, 10 in 234 s (live e2e runs) |

Streaming replays: on the Tetris soup the judge fires at 2,700 chars (≈700 tokens, ~4 s in) instead of
7,700+; on the "Let's go" narration at 3,000 chars instead of 45,000; on the table loop at the first
window. Gibberish and narration rules apply only when tools were offered — a plain chat may be math
or ASCII art — and narration/runaway are skipped when the reply also carries a real tool call.

*Proxy (`loop_proxy.py`).* An agentic request (tools offered) is buffered — TensorFold emits tool
calls only after the reply is complete, so what streams is prose, and a healthy step's prose is a
short result line or nothing. The text is judged every 200 chars and the whole reply at the end. A
failure closes the upstream socket (the engine's `socket_cancellation` cancels the job within one
decode round), saves the discarded text to `.harness_failures/`, and retries the step: fresh `seed`,
temperature 0.3 then 0.2, top-p 0.9, top-k 40, min-p 0.05, plus a user-role `[Harness] Your previous
reply to this step was discarded: <why>. Redo the step now …` message; the last attempt adds
`repetition_penalty 1.15` for loop-shaped failures (it drops the draft fast path, 180 → ~125 tok/s, so
it is used last). After 3 attempts the turn ends with a synthetic `[Harness] Stopped: the model failed
this step 3 times (<reasons>) …` assistant message instead of garbage. Plain chat streams live and is
only cut (loop / leak) with a note. SSE keepalive comments hold the client connection while a reply is
buffered or retried. Also: sampling defaults for omitted fields, a 6144-token cap per agentic step
(OpenCode asks for 16384), `include_usage` upstream so truncation is visible, a 240 s stall timeout,
transient 5xx and stalls retried without a directive, and the proxy waits up to 90 s for an engine that
is restarting. `GET /harness/health` reports upstream state, counters and config.

*Engine and configs.* `2_start_tensorfold.sh` now serves with `--temperature 0.45 --top-p 0.9 --top-k
40 --min-p 0.05 --max-tokens 8192` (min-p and seed keep the drafted fast path: 189–215 tok/s measured;
the repetition penalty does not: 124 tok/s), runs the proxy self-test before starting, and supervises:
engine probed every 20 s, restarted after a crash or 3 failed probes, proxy restarted if it dies.
`opencode.json`: `"temperature": true`. `kilo.json`: agents 1.0/0.95/64 → 0.35/0.9/40.

**Verification.**
- `python3 loop_proxy.py --self-test`: judge calibration cases, the repeat guard, and 12 scenarios
  against a scripted fake engine (soup mid-stream → retry with directive/seed/temperature, truncated
  call → retry, three loops → `[Harness] Stopped` and 3 dump files, healthy call/report pass through
  unchanged with one upstream request, live chat cut, defaults and step cap, non-stream retry, 503
  retried / 400 passed through, stall → retry, engine down → 503 in <10 s, health counters). Green.
- `python3 test_harness.py --gate` after restart: 8/8 (adds `harness proxy health` and `harness
  sampling defaults`).
- Live forced failure and `python3 test_e2e_opencode.py` (OpenCode headless, the exact Tetris prompt):
  see the next section.

**Caveats.** Agentic prose is shown when the step finishes, not as it streams (set
`HARNESS_STREAM_PROSE=1` to stream it and give up retries). A `write` whose *content* loops is
invisible to the proxy until the step ends — the 6144-token cap bounds that at ~35 s healthy / ~2 min
at collapsed draft speed — and then it is judged as a loop and retried. Thresholds are calibrated on
this model's transcripts; a different model may need `harness_judge.py` re-checked against its own
healthy output (the dump directory makes that a five-minute job). The judge cannot tell a *wrong* final
report from a right one; it only guarantees the transcript stays clean and the step is retried when the
model fails to act.

## TensorFold 0.6.0 and PR #157 — what upstream now covers (2026-09-30, 21:31 UTC release)

**Is 0.6.0 out?** Yes: GitHub release `v0.6.0`, published 2026-09-30T21:31Z (tag `c812f8b`). It is a
GitHub release only — `pip index`/PyPI has no `tensorfold` package; this stack installs from git.

**What it fixes for us.** From the release notes and the maintainer's reply on #121 (comment
`5920104995`, 21:32Z): *"The parser takes that form now, beside `call:name`, and an unoffered name
stays text … This is in 0.6.0."* So the **bare `<|tool_call>:NAME{…}` leak (#121) is fixed upstream**,
with our transcript's seven tool names as the regression test. Also relevant to agentic use in 0.6.0:
a prompt past the context window now returns OpenAI's `context_length_exceeded` (so OpenCode compacts
instead of retrying), Python-spelled argument values (`False`, `None`) decode to their schema types,
and tool-call arguments stream as they are written — **on the CUDA server**; the Mac path still emits
tool calls at the end (the property loop_proxy.py relies on; nothing changes for us there).

**What it does not cover.** PR #157 was **closed unmerged** at 21:34Z — the maintainer implemented fix
1 independently. Nothing in the 0.6.0 notes mentions the **spontaneous / mid-text thought-channel leak
with thinking off** (fix 2 of #157 and commit `c3f3f1b`), the **repetition / frequency / presence
penalties** (`feat-repetition-penalty`), or the **unterminated / malformed tool-call repair**
(`feat-toolcall-repair`, never sent as a PR). And none of the *harness* failures in this file
(narration instead of tool calls, LaTeX / symbol soup, loops, temperature 1.0 via the OpenCode flag)
are engine parser issues; 0.6.0 does not touch them — that is what the proxy layer above is for.

**So: upgrade?** Not needed for stability today. The running build carries everything (upstream main
`9cd52ab` + the three fork branches). Moving to 0.6.0 means rebasing the fork's three branches onto
`v0.6.0`: the colon-prefix fix collides with upstream's own version of it (drop ours, keep the channel
part), and `tools.py` has changed under the repair branch. That is a contained, one-sitting job and
worth doing once, then pointing `TF_VERSION` at the rebased combined branch. Filed as a follow-up.

**Reproducibility fix.** The venv's TensorFold came from a scratchpad checkout that no longer exists
(`direct_url.json` pointed into another session's temp dir), and `1_setup_download.sh` would have
reinstalled only `fix-gemma-tool-call-colon-prefix`, silently losing the penalty and the repair. A
test merge shows the three branches (each 3 commits on upstream `9cd52ab`) **merge cleanly**, and the
result is code-identical to the running venv (the only diff is a two-line comment in `server/tools.py`).
`1_setup_download.sh` now clones the fork, merges the three branches into a local `combined` branch and
installs from that tree, so a fresh machine gets the same engine the tests were run against.

## Live results (2026-09-30, 22:35–23:02): forced failure, five end-to-end runs, crash recovery

All on the running stack (engine `:8104` + `loop_proxy.py` `:8094`), OpenCode 1.18.33 headless via
`test_e2e_opencode.py`, the exact prompt of the failed session: *"create a tetris game in folder named z"*.
Every rule below was added because a run showed the pattern; each later run shows the earlier fix holding.

| run | proxy state | what happened | outcome |
|---|---|---|---|
| forced truncation (direct API, `max_tokens=300`, whole-game write) | judge + retry | 3 attempts, each `truncated` (seed / temperature / directive recorded per dump) | clean `[Harness] Stopped … (truncated)` in **10.2 s**, no leaked markup |
| e2e 1 | judge + retry | `mkdir`, 2 writes (game complete, `node --check` ok, 16 functions), then the **same no-op `edit`** (old = new) 5× | repeat guard nudged twice, **stopped the turn at 37 s** with a message naming the file |
| e2e 2 | + nudge re-seeds / widens temperature | game written, then **13 rewrites of `z/tetris.js` in 214 s** (213–1902 tokens each); one write's content **looped** (`// The player reset should be done befor…` × 40+) → judged `loop`, retried | stopped by hand as the churn baseline; files valid |
| e2e 3 | + churn guard (4: directive, 10: stop) + no-op-edit rule | churn directive at the 4th write → the model **did run `node --check`**, then two comment-only `bash` "checks", then kept rewriting | stopped at the **10th write (234 s)**; the last write was a **542-byte stub** over a full game |
| e2e 4 | + churn-finish (6: tools withheld) + shrinking-rewrite + comment-only bash | wrote ``z/tetris.js` `` and ``z/tetris.js`` `` (backticks in the path), then **12 failing `rm` variants**, one with a leaked `<|"|>` token; a stub rewrite was caught (`shrinking-rewrite`); one write hit the 6144 cap (`truncated`, retried) | stopped by hand; path-junk + template-token rules added |
| **e2e 5** | **all of the above** | `mkdir`, `index.html`, `tetris.js`; **3 stub rewrites caught and retried** (1347 / 1053 / 1421 chars over 4617–6085) — each retry came back complete; churn directive at the 4th write → `node --check` → **the model's own final report** | **PASS in 129 s**: 2 files, `node --check` ok, 15 functions, correct run instructions; 0 stops, 0 garbage in the transcript |
| crash test | supervisor | `kill -9` the engine; a chat request sent 2 s later | supervisor restart #1, **engine ready after 11 s**; the request was held by the proxy and **answered `PONG` after 32.6 s**, 0 upstream errors |

**What the numbers say.** The original session lost six minutes to one runaway step and ended in soup;
the same prompt now finishes in about two minutes with a verified game and a report, and every failure
shape seen along the way (no-op edit, comment-only check, backtick path, leaked token, stub rewrite,
content loop, truncation, rewrite churn) is caught before it costs more than one step. Where the model
cannot be steered (identical edits at 0.35, ten rewrites), the turn ends in seconds with a message that
says exactly what to type next, and nothing unreadable reaches the transcript.

**Not fixed by the harness (by design).** The judge does not know whether a *plausible* final report is
true — it guarantees the transcript is clean and that the model acts instead of narrating. Rewrite churn
is bounded (4 / 6 / 10), not eliminated: the abliterated weights like to "improve" a finished file, and
only a stronger model changes that. Thresholds are calibrated on this model's own transcripts; re-check
`harness_judge.py` against another model's healthy output before reusing it (the dump directory makes
that quick). Discarded replies: `.harness_failures/` (kept to the newest 60).

## Rebase onto TensorFold v0.6.0 (2026-09-30, 23:00–23:40)

**Result.** Our three patch sets now sit on upstream **v0.6.0** (release commit `c464617`; the tag
object is `c812f8b`), and the stack runs that build. The bare-`:NAME` parser change is dropped
because 0.6.0 has its own equivalent. Everything else carried over. One conflict would have
**silently broken the repetition penalty** under a mechanical resolution; it was re-applied by hand
and verified live (below).

**What was kept, dropped, and where it went.**

| old commit (on 0.5.0 `9cd52ab`) | what | on v0.6.0 |
|---|---|---|
| `5bb1a85` | parse `<|tool_call>:NAME{…}` without `call` | **dropped**: 0.6.0's `_GEMMA_CALL_RE = ^(?:call)?:…` and `startswith(("call:", ":"))` do the same (#121) |
| `879227f` | route replies through `split_thinking` whenever the tokenizer uses channel markers | `fdf543c`; conflict in the `app.py` import list (upstream dropped `render_prompt_ids`), both logic hunks applied |
| `c3f3f1b` | strip a thought channel that opens anywhere | `88d0cf5`, clean |
| `28c63b4` | repetition / frequency / presence penalties (sampling, request options, CLI) | `bbf25fb`; upstream moved the whole argument parser from `cli.py` into a new `cli_args.py`, so the four `--*-penalty` / `--penalty-last-n` flags moved there; `_generation_config` merged with upstream's new `config.get(k) is not None` filter |
| `412f613` | penalised streams run with drafts off and draw through the CPU exact path | `b3ce386`; **see the trap below** |
| `05f9713` | forward the penalty keys from the HTTP body | `14839e2`, clean |
| `0275f8d`, `662fb36`, `cd54c65` | repair or hide unterminated / malformed tool-call blocks | `99f12f6`, `dc1eae5`, `d8f1591`, clean on top of upstream's parser change |

**The trap.** Upstream moved `Scheduler._start_job` / `_open_job` out of `server/scheduler.py` into a
new `server/prompt_fill.py`. Git reported a conflict in `scheduler.py` that looked like "our whole old
method versus nothing". Taking upstream's side there — the obvious resolution — applies cleanly and
passes every unit test, but drops the gate that turns drafts off for a penalised stream. The penalty
would then only reach the rows drawn on the CPU path while drafted rows went through the Metal kernel
without it: a silently weaker, inconsistent penalty. The gate (`use_drafts = drafts and not
sampling.has_penalty`) was re-applied in `prompt_fill.py`, where the stream is now built, and amended
into the same commit.

A regression test now guards the trap: `tests/test_repetition_penalty.py::
test_a_penalised_stream_is_opened_with_drafts_off_and_a_plain_one_keeps_them` opens a job through the
real scheduler and asserts a penalised stream has drafts off and no proposer, while a plain one keeps
both. Verified both ways: it **fails with the gate removed and passes with it** (commit `3d29610`).

**Branches** (local, in the stack's git-ignored `./.tensorfold-src`, remote `origin` =
`gprot42/TensorFold`; **not pushed yet**):

- `combined-v0.6.0` — 10 commits on `c464617` (8 carried over, the penalty regression test, and the
  thinking-off channel tests); what the stack installs.
- `gemma4-thought-channel-v0.6.0` (3, incl. the thinking-off tests), `feat-repetition-penalty-v0.6.0` (4, incl. the test),
  `feat-toolcall-repair-v0.6.0` (3) — one per upstream PR. They touch disjoint files, and merging the
  three reproduces `combined-v0.6.0` exactly (0 differing files). PR titles and bodies:
  `tensorfold-pr-drafts.md`.

**Tests (TensorFold's own suite, on the rebased tree, Python 3.12 + MLX 0.32.3, M5 Max).**

| run | result |
|---|---|
| feature tests (`test_lane_stream_text`, `test_repetition_penalty`, `test_toolcall_repair`, `test_tool_call_content`) | 51 passed, 1 skipped |
| full suite, `combined-v0.6.0` | **3,809 passed, 488 skipped, 1 failed** in 6 m 52 s |
| the one failure, `test_glm5_prompt_kernels.py::test_fused_index_scores_are_the_three_ops`, on **stock v0.6.0** | fails identically, 3 of 3 runs |

The failure is upstream's, not ours: `index_fits()` for the GLM-5.3 fused index-score prompt kernel
returns False on this M5 Max, and the test asserts it fits. None of our commits touches GLM or kernel
code. It is deterministic, not flaky, and GLM-only: the Gemma lane never calls it. (A first full run
was discarded: branches were switched in the same checkout while it ran, so its result proved nothing.
The numbers above come from a checkout nothing touched during the run.)

**Install and rollback.** Runtime dependencies are identical between 0.5.0 and 0.6.0 (`mlx
>=0.32.2,<0.32.4`, `mlx-lm >=0.31.3,<0.32`, …), so the engine was swapped with `pip install
--no-deps`, and `pip check` stays clean. `1_setup_download.sh` now installs from `./.tensorfold-src`
when it is a git checkout (`TF_LOCAL_SRC`); otherwise it clones the fork and merges the three
`-v0.6.0` branches, and says clearly if they are not pushed. The previous engine is kept as
`.tensorfold-rollback/tensorfold-0.5.0-py3-none-any.whl` (identical to what ran before, except one
comment line); rolling back is one `pip install --no-deps --force-reinstall` of it plus a restart.
The engine now starts with `--no-update-check`: 0.6.0 checks GitHub at start and suggests
`tensorfold update`, which would replace the patched build with the stock release.

**Proxy changes for 0.6.0.** 0.6.0 returns OpenAI's `context_length_exceeded` when a prompt plus its
reply limit does not fit, so clients compact instead of retrying. For a stream the engine raises it
inside `app.chat`, after its own stream headers, as an SSE `error` event carrying the `code`; the
proxy relays that event unchanged. Separately, the proxy used to open its client stream *before*
asking the engine, which turned a refusal the engine sends *before* its stream (a 400, a 503 capacity
refusal) into an error event inside a 200. It now opens the client stream only after the engine has
answered 200, so a 503 reaches OpenCode as a 503 and gets OpenCode's own retry with backoff. Two
self-tests cover it. The test fake engine also got the proxy's disconnect guard, so the self-test
prints no tracebacks and a real one would stand out.

**What 0.6.0 brings that matters on this Mac.** The #121 parse (redundant with ours),
`context_length_exceeded`, Python-literal tool arguments (`False`, `None`) decoding to their schema
types, several prompts filling side by side (fewest tokens left first), a prompt cache that keeps
each conversation's newest checkpoint and grows into memory the model leaves idle, and Prometheus
`/metrics`. The rest of the release (RTX 40, bf16 CUDA prompts, streamed tool-call arguments on CUDA,
GLM-5.3, Flash Next) does not touch this stack.

**The swap (23:31).** An OpenCode session was running through the stack, so the swap waited until
the proxy had seen no client connection for 60 s. The old supervisor was stopped with SIGTERM to its
own process, not Ctrl-C in its tab: Ctrl-C also kills the `tee` it logs through, and its cleanup can
then die on SIGPIPE and leave the engine running with nothing supervising it. It shut down in 3 s.
Then `pip install --no-deps --force-reinstall` of the rebased wheel (`tensorfold --version` 0.5.0 →
**0.6.0**, `pip check` clean, the repair and the drafts-off gate present), and a fresh
`./2_start_tensorfold.sh restart`: proxy self-test ok, engine loaded with `--no-update-check`, and a
new 0.6.0 line in the log — *prompt cache up to 63.7 GiB: the memory the weights, a 131,072-token
request and a shared round leave idle*. Gate: 8 of 8.

**Live checks on 0.6.0.**

| check | result |
|---|---|
| tool calls still arrive as structured deltas, prose as content (what the proxy relies on) | `write` in 2 `tool_calls` deltas, 0 content deltas, `usage` present |
| repetition penalty turns drafting off (the re-applied gate) | plain: 135 tokens at 229 tok/s, 101 of 137 drafted accepted. `repetition_penalty 1.15`: 76 tokens at 134 tok/s in 75 rounds, **0 drafted** |
| a whole-file `write` cut at 300 tokens (repair) | structured `write` call with 987 chars of salvaged arguments, **no markup** in content |
| `context_length_exceeded`, non-streaming, through the proxy | HTTP 400, `code: context_length_exceeded`, OpenAI's message |
| `context_length_exceeded`, streaming, through the proxy | one SSE `error` event with the code (the engine sends it inside its stream) |
| proxy health | ok; the two probes above count as `upstream_errors` |

**A live session on the harness (23:13–23:27, before the swap).** An OpenCode session in `~/z`
("create a tetris game in a new folder called z") went through the proxy with **no stops**. Six steps
failed and each recovered on retry: two stub rewrites of `~/z/tetris.js` (840 chars over 12,642;
1,347/1,053/1,421-char stubs had been caught the same way in the e2e run), and four steps that hit
the 6,144-token cap. Three of those were runaways inside the `write` content: a comment pair ×180,
`colors_final_final_final…`, a six-line paragraph ×20. The fourth produced 6,144 tokens with no
visible text and no call. The churn guard warned three times, and the turn ended with the model's
own 286-char report.

That session added three harness rules, all in the proxy and judge self-tests:

- **A capped reply that repeats is named `loop`, not `truncated`.** Before, the retry message said
  "do less per step" and the last attempt got no repetition penalty. Re-judging the three saved dumps
  names all three loops; a large file of distinct lines stays `truncated`.
- **A capped reply with no text and no call is `silent`.** It spent the step in hidden reasoning,
  most likely a thought channel opened with thinking off and never closed. The retry says to answer
  directly.
- **The proxy now keeps `reasoning_content`.** It is still relayed unchanged; it is also judged for
  loops when a reply is capped, and saved in the failure dump. Before, a silent runaway left an empty
  dump.

**End-to-end on 0.6.0, first run: FAIL — a harness gap, not the engine.** The Tetris task wrote both
files, then sent one `edit` four times in a row. OpenCode answered "Edit applied successfully" every
time: its edit tool matches loosely, so each identical edit rewrote a slightly different region, and
`tetris.js` stopped parsing (`SyntaxError: missing ) after argument list`). The repeat guard nudged
twice and stopped the turn at the fourth repeat, as designed, but it acts on the *next* request, after
the client has already run the call. For a read-only command that is harmless. For an edit, the file
has already changed. The same repeated-edit habit showed up on 0.5.0 (e2e run 1, where old = new), so
it is the model, not 0.6.0. The same run also produced an `edit` with no `oldString`, which OpenCode
rejected after a wasted step.

**Fix: two checks that run before the client executes anything** (the proxy holds every agentic reply
until it is judged, so this is the one place a call can still be stopped):

- **`repeat-edit`**: an `edit` / `patch` / `multiedit` / `apply_patch` identical to one already
  applied *successfully* in this turn is failed and the step retried with "applying the same edit
  twice changes the file twice and corrupts it — read the file or run a check, or make a different
  change". An identical edit whose earlier try *failed* is still allowed, since it may succeed after
  the file changed. Replaying the failed run's ten calls through the check blocks the second
  application (call 7 of 10), so only the first edit would have reached the file.
- **`missing-args`**: a call that lacks a key its tool's own JSON schema marks `required` (taken from
  the request's `tools`) is failed and retried.
- The after-the-fact message for a repeated edit no longer says "it was not run again", which was
  false for an edit. It now says the edit was applied again and the file may contain it twice.

**End-to-end on 0.6.0, second run (with the two checks): PASS in 54 s.** `mkdir`, `index.html`, then
`tetris.js`. One stub rewrite was caught and retried (1,459 chars over 4,698), and the churn directive
fired at the fourth write. The model wrote its own final report. `tetris.js` passes `node --check`
and has 14 functions.

**One more gap from that run: a report that claims checks it never ran.** The report ended with
"Final verification: `node --check z/tetris.js` (No syntax errors found)", yet the run's only tool
calls were `mkdir` and five writes. The file happened to be valid. A report like that is how a user
ends up trusting a broken file. New rule, **`unverified-claim`**: a final report (no tool calls) that
quotes a check command (`node --check`, `py_compile`, `pytest`, `npm test`, `tsc`, `cargo test`,
`go test`, `ruff`, `eslint`, `bash -n`, `shellcheck`, `mypy`, …) next to claim wording ("verified",
"no syntax errors", "passed", …), when no bash call in the turn ran that command, is retried once with
"run it now with the bash tool and report what it prints, or leave the claim out". If the model claims
it again, the report passes through and is counted in `unverified_claims`, so this rule never stops a
turn. Replayed over every final report this model has written in OpenCode (45 reports, 21 sessions),
it flags exactly one: this one. None of the reports whose check really ran are flagged.

**End-to-end on 0.6.0, third run: FAIL — blind edits, then a model set on one wrong action.** After
the two writes, the model made two different edits without looking at the file in between, and one
of them broke `tetris.js` (`SyntaxError: Unexpected token '}'`). It then tried a no-op edit (caught),
an edit it had already applied (blocked before it ran), and the same no-op edit again. The third
attempt's output was byte-identical to the first (same engine `sha`), despite a fresh seed and a
corrective message. The harness stopped the turn after three attempts, so nothing got worse, but the
file stayed broken.

Two lessons, four changes:

- **Lower temperature is the wrong retry for a behavioural mistake.** Cooling helps when the model
  degenerates (loops, soup, runaways). When it is set on one wrong action, cooling only makes it
  repeat that action more faithfully. Retries now pick their schedule by failure kind: degeneration
  0.3 → 0.2, behavioural (`noop-edit`, `repeat-edit`, `bad-args`, `missing-args`,
  `shrinking-rewrite`, `unverified-claim`, `unchecked-change`, `narration`) 0.6 → 0.8.
- **Take the tool away.** After a `noop-edit` (a new reason, split out of `bad-args`) or a
  `repeat-edit`, the retry is offered no edit tool, so the model has to check the file, write it
  whole, or report.
- **Check before changing the same file again.** Replaying all 95 write/edit steps this model made
  in OpenCode, **56** changed a code file that had changed earlier in the turn and had not been looked
  at since. As a post-generation rule alone (`unchecked-change`, one retry, never a stop), that would
  discard a whole generated write each time. So the main mechanism is steering *before* generation:
  when the latest round changed a `.js`/`.py`/`.sh`/`.json` file successfully, the next request gets
  "`<file>` just changed. Run `node --check <file>` … now" (counted as `check_nudges`). That costs
  nothing when the model complies, and `unchecked-change` is only the backstop.
- **A bug the self-test caught:** the proxy's own directives are user-role messages in the forwarded
  copy, and every turn check took "the last user message" as the start of the turn. After any nudge
  (churn, repeat, now check), the checks saw an empty turn, and a stub rewrite could slip through. The
  bug had been latent since the churn directive was added. `[Harness]` directives no longer count as
  turn boundaries, in the shared helper and in the stub check, which had its own copy of the logic.
  Covered by a regression test.
- **A conflict the new nudge created, caught live:** the repeat guard's cycle rule nudges when one call
  is issued three times in a turn, whatever its output, and stops the turn at six. That exists to catch
  alternating loops. Once the check nudge had the model running `node --check z/tetris.js` after every
  write, the cycle rule told it to stop checking, and at six checks it would have ended the turn. A
  call re-issued after a file changed is now verification, not a cycle, the same exemption the
  identical-output rule already had. A test covers eight write → check rounds. The end-to-end series
  that was running when this showed up was stopped and restarted on the fixed proxy.

## The maintainer's request on #157: the channel fix as its own PR on 0.6.0 (2026-10-01)

**What they asked** (ashhart, 2026-09-30 21:34 UTC, comment `5920133225`, as the PR was closed): the
bare `:NAME{...}` fix is in 0.6.0 (`4447ac30`). The thought-channel leak with thinking off "is a real
bug we don't fix yet … Could you send that part as its own PR on 0.6.0? One thing to cover in it:
gpt-oss replies with thinking off must come out exactly as they do today, so a harmony test beside
your Gemma one. We'll take it into 0.6.1."

**What was done.**

- Branch `gemma4-thought-channel-v0.6.0` on v0.6.0 (`c464617`): the two channel commits plus a new
  `tests/test_thinking_off_channels.py`. It runs scripted replies through the real `ChatApp` and HTTP
  handler with thinking off, streamed and not, using the harness `tests/test_think_call.py` already
  has:

  | test | stock v0.6.0 | branch |
  |---|---|---|
  | Gemma tokenizer (has `<channel|>`), empty spontaneous thought block | **fails**: `content` is `'<|channel>thought\n<channel|>The files are a.py and b.py.'` | passes |
  | gpt-oss tokenizer (no `<channel|>`), Harmony reply | passes: content `Hello there.`, reasoning `Think it over.`, streamed the same | passes, identical |
  | plain reply, both tokenizers | passes | passes |

  The Harmony expectations were measured on stock v0.6.0 first and pinned, so the test states "exactly
  as today" as a fact rather than an assumption. Why it holds: the change keys on
  `think_markers == CHANNEL_MARKERS`, which only a tokenizer with Gemma's `<channel|>` token satisfies.
  Full suite on the branch alone: **3,788 passed, 488 skipped, 1 failed**; the failure is the upstream
  GLM kernel test that fails the same way on stock v0.6.0 here.
- **Identity.** The rebase had stamped the new commits with a personal email. Every commit on all four
  `-v0.6.0` branches was rewritten to the public `gprot42 <gprot42@example.com>` (author dates kept),
  before anything was pushed. Four cherry-pick trailers pointing at local-only commits, and a stale
  0.5.0-era suite count in one commit message, were removed. The hashes in this file are the
  rewritten ones.
- **Pushed:** `gemma4-thought-channel-v0.6.0` → `gprot42/TensorFold` (head `946612e`).
- **Not done:** opening the PR on `ashhart/TensorFold`. The session's permission policy refused to
  create a public item, so it needs one command from the user (in `tensorfold-pr-drafts.md`; the body
  is `tensorfold-pr-channel-body.md`).

**Why the harness stays out of the way.** Four separations, each checked:

1. **The PR contains engine code only.** `git diff --stat v0.6.0..gemma4-thought-channel-v0.6.0` is
   `server/app.py`, `server/text.py` and two test files. The repetition penalty and tool-call repair
   are separate branches, and the reliability layer (`loop_proxy.py`, `harness_judge.py`) is not in the
   TensorFold repo at all.
2. **It is tested on its own.** The suite ran on a worktree of that branch alone, not on the combined
   engine the stack runs, and the new tests ran on stock v0.6.0 for the baseline.
3. **The evidence bypasses the proxy.** Through `:8094` this bug is invisible by design: the judge
   treats a raw `<|channel>thought` in `content` as a leak, cancels the reply and retries it, so a
   client never sees it. Any reproduction for upstream has to use the engine directly (`:8104`) or,
   as here, the deterministic `ChatApp` tests, which involve no proxy and no sampling.
4. **Neither depends on the other.** The installed engine already carries the fix, so on this stack
   the proxy's leak rule is a second line of defence, not a requirement. Once 0.6.1 ships the fix,
   the stack can drop this patch and keep the proxy unchanged.

## End-to-end series on 0.6.0, and what they changed (2026-10-01, 00:00–00:40)

A background `tar` backup of `~/src` ran at about 90% CPU through these runs, so engine speed was
roughly half normal (38–90 tok/s for writes instead of 150–220). The runs were slower, not different in
kind.

**Series 2 (three runs, after the check-nudge / cycle fixes):**

| run | verdict then | what it really was |
|---|---|---|
| 1 | PASS, 213 s | complete game (14 functions), `node --check` after each write, model's own report; 4 retries (3 stub rewrites, 1 plain-text "thought" opener caught as a leak) |
| 2 | "PASS" | **not a pass**: only `index.html`. The `tetris.js` step failed three times: two writes looping at the cap ("// We'll use a loop to try shifting." ×55), then a complete 2,065-token write rejected **only** because its path was ``z/tetris.js``` with a backtick |
| 3 | "PASS" | **not a pass**: a 120-line `tetris.js` of piece tables, no loop, no rendering, no input, ending "// This will be the final attempt for the file.", and a report claiming a finished, verified game |

Two of the "passes" were hollow, so the test was fixed as well as the harness:

- **Path repair instead of rejection.** When the only defect is backticks, quotes or blanks around a
  path value, the proxy strips them and relays the corrected call, rebuilding the stream or JSON body.
  Junk *inside* a path is still a retry. Run 2's third attempt would have succeeded.
- **Partial rewrites.** Write sizes per run, relative to that file's largest earlier version this turn:
  the failed run relayed rewrites at 41%, 48%, 42% and 39%; the passing runs ended at 80–100%. A rewrite
  at 35–60% of the peak is now retried once (`partial-rewrite`); under 35% stays a hard failure. It is
  soft so that a genuine simplification survives a second attempt.
- **A report over a regressed file.** A final report while a code file this turn wrote is below 60% of
  its own peak is retried once, with the numbers (`regressed-file`).
- **Planning notes in code are not a signal.** 51 of this model's 74 code writes contain comments like
  "// Let's …" or "// Wait, …", and 29 end with one, including the complete 12,642-char game from the
  live session (28 of them). A rule on them would reject working files, so there is none.
- **The e2e test is stricter.** It fails when an expected file is missing (`--expect`, default
  `z/*.html z/*.js`), when the code lacks what the task needs (`--expect-code`, default a game loop and
  a key handler), or when the turn ends with a `[Harness] Stopped` message. Under these rules runs 2 and
  3 above are failures, which is what they were.

**Series 3, strict criteria, run 1: FAIL — and the three rules it produced.** Every write of
`tetris.js` but the last was followed by `node --check`, as the check nudge asks. After the last write,
a 4,998-char file with `offset =--;` on line 195, the model skipped the check and wrote a 6,006-char
final report. It opened with `<thought>\n</thought><table><tr><td></td></tr></table>`, then pasted a
complete-looking game in a code block ("the final version … was consolidated into one clean file
below"), then "Open z/index.html to play!". The good code stayed in the chat and the broken code in
the file. The test's new checks caught it: syntax error, no game loop, no key handler. Three more soft
checks, each one retry and never a stop, ordered most severe first:

- **`code-in-report`**: a final report carrying 1,500+ chars of fenced code while the turn wrote code
  files. The retry says "write it there with the write tool and run its check, then report in a few
  lines without code". Showing code with no file written in the turn is still a normal answer.
- **`unchecked-final`**: a final report while code files changed after their last look. The retry names
  every such file, up to eight, in one command (`node --check a.js && python3 -m py_compile b.py`), so a
  big turn costs one retry, not one per file.
- **Pseudo thought tags**: a reply that opens with `<thought>` or `<think>` is a leak, like the opening
  "thought\n" the judge already caught. Only at the very start: `<think>` quoted mid-sentence is fine.

**Series 3, run 2: FAIL, and one interaction fixed.** The model rewrote `tetris.js` six times. The
checks caught six stub rewrites and a partial one, and asked for `node --check` five times. At the
sixth write the churn limit withheld every tool to force a report. The report said "verified for
syntax" over a `tetris.js` that fails `node --check` (line 236). With every tool withheld, the model
could not have run the check even if asked. Now, when the churn limit is reached and a changed file has
not been checked since, only `write` is withheld: `bash`, `read` and `edit` stay, and the directive
says "run `node --check …`; if it reports an error, fix it with a small edit; then report what works
and what does not." When everything is already checked, the tools are withheld as before.

**Series 4, every rule live (strict criteria).**

| run | result | what happened |
|---|---|---|
| 1 | **PASS**, 23 s | `mkdir`, `index.html`, a 5,384-char `tetris.js` (16 functions, game loop, key handler), `node --check`, report. No retries |
| 2 | **PASS**, 127 s | complete game (14 functions, loop, keys). 4 stub and 1 partial rewrite caught and retried; the model ran the `node --check` its report cites. A burst of `index.html` rewrites ended at the churn limit with a summary |
| 3 | **FAIL**, 863 s | the model looped inside long writes again and again ("// Let's just use a simple …" ×12+ at the cap, even on the repetition-penalty attempt), then wrote stubs to **new names** (`tetris_v2.js`, `tetris_final.js`) where no size check looks. The turn ended with `[Harness] Stopped … (loop, missing-args)`. No false claim, no garbage in the transcript, but no game |

**Last rule, `sibling-copy`** (soft, one retry): a write to a versioned copy of a file this turn already
wrote (`x_v2`, `x_final`, `x-new`, `x.2`, `_copy`, `_fixed` …, same extension) is retried with "put the
complete file in `x` itself". Replayed over every write this model made in OpenCode (137 writes, 32
sessions), it fires exactly twice: run 3's two copies. It went live after series 4, and the gate passes
8 of 8 with it.

**Where this leaves the stack.** Every failure mode from the original sessions is caught: LaTeX soup,
loops, narration instead of tool calls, leaked envelopes, runaways to the token cap, silent reasoning
and truncation. So are the ones found on the way: no-op and repeated edits, stub and partial rewrites,
backtick paths (now repaired), unchecked changes, reports that claim unrun checks, code pasted into
reports, and versioned copies. The harness catches each one before it costs more than a step or two,
and a failed turn now *says* it failed instead of claiming success. What it cannot do is make this
model write a 5 KB program reliably. On the Tetris prompt the model itself still fails about one run
in three, by looping inside long file writes. The findings above have said since the start that the
remedy for that is a stronger model or the censored base pack for large single-file writes, not more
rules.
