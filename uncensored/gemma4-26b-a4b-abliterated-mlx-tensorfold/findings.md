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

**Scope.** A block is treated as broken when it is *unterminated* (no `<tool_call|>`, cut off at the
token limit) **or** has an *unbalanced* `<|"|>` string delimiter — an odd count, i.e. the model opened
a string value and never closed it. That second case covers a live `glob` leak
(`<|tool_call>call:glob{pattern:<|"|>*/}<tool_call|>` — terminated but the pattern string never closed)
and the penalty-induced corrupted write. A terminated block with *balanced* delimiters that still fails
to parse (e.g. a bare `{location}`) is left as text, matching upstream's deliberate behaviour; text
after a terminated block is preserved.

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
