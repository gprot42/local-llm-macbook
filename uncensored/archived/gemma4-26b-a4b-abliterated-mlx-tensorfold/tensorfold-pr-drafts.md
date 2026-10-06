# TensorFold upstream PRs — drafts for the v0.6.0-based fork branches (2026-09-30)

Three fork branches, each rebased onto `ashhart/TensorFold` tag `v0.6.0` (`c812f8b`) and cut from the
combined branch that runs in this stack. They touch disjoint files, merge cleanly to the combined tree,
and each carries its own regression tests. Branch names end in `-v0.6.0`; the pre-rebase branches
(`fix-gemma-tool-call-colon-prefix`, `feat-repetition-penalty`, `feat-toolcall-repair`) stay as they are.

PR 1 is pushed. For PRs 2 and 3 (not requested upstream yet), push first:

```bash
cd .tensorfold-src && git push origin feat-repetition-penalty-v0.6.0 feat-toolcall-repair-v0.6.0 combined-v0.6.0
```

Then, for each PR, `gh pr create --repo ashhart/TensorFold --base main --head gprot42:<branch> --title "<title>" --body-file <(sed -n '/^## PR 1/,/^## PR 2/p' tensorfold-pr-drafts.md)` (adjust the range per PR).

---

## PR 1 — `gemma4-thought-channel-v0.6.0` (requested by the maintainer; branch pushed, PR not opened yet)

The maintainer asked for this on #157 (comment 5920133225, 2026-09-30 21:34 UTC): *"Could you send
that part as its own PR on 0.6.0? One thing to cover in it: gpt-oss replies with thinking off must
come out exactly as they do today, so a harmony test beside your Gemma one. We'll take it into
0.6.1."*

The branch `gemma4-thought-channel-v0.6.0` is **pushed** to `gprot42/TensorFold` (head `946612e`, base
v0.6.0 `c464617`, three commits: the two channel fixes plus `tests/test_thinking_off_channels.py`).
The body is in `tensorfold-pr-channel-body.md`. Open the PR with:

```bash
gh pr create --repo ashhart/TensorFold --base main --head gprot42:gemma4-thought-channel-v0.6.0 --title "gemma4: strip a spontaneous thought channel when thinking is off (Harmony unchanged)" --body-file tensorfold-pr-channel-body.md
```

or in the browser: https://github.com/ashhart/TensorFold/compare/main...gprot42:TensorFold:gemma4-thought-channel-v0.6.0?expand=1

---

## PR 2 — `feat-repetition-penalty-v0.6.0`

**Title:** sampling: repetition / frequency / presence penalties (request, CLI, generation_config)

**Body:**

TensorFold exposes temperature / top-p / top-k / min-p but no repetition penalty, so there is no knob against a reply that loops (`</td></tr></tbody></table>` × 700 on an abliterated Gemma 4 pack, at 45 tok/s to the token cap). This adds the standard OpenAI-style `frequency_penalty` / `presence_penalty` and the HF-style `repetition_penalty`, plus `penalty_last_n`, on the Mac lanes.

Three commits:

1. `engine/exact_sampling.py`: `Sampling` gains the four fields (every one a no-op at its default) and `has_penalty`; `penalized()` applies the HF rule (divide a positive logit / multiply a negative one) and the frequency/presence subtraction to the candidate logits before top-k/top-p/min-p/Gumbel; `choose()`, `choose_rows()`, `sample_rows()` take an optional per-row `recent` history. `server/request_options.py` validates the fields (`repetition_penalty` > 0; frequency/presence may be negative; `penalty_last_n` an integer) and resolves them onto `Sampling`. `cli_args.py` / `cli.py`: `--repetition-penalty`, `--frequency-penalty`, `--presence-penalty`, `--penalty-last-n`, and the keys are read from `generation_config.json`.
2. Decode path: a stream whose sampling has a penalty runs with drafts off (`server/prompt_fill.py`) and draws through the extended CPU exact path with its reply-so-far history (`engine/lane_family.py` `_draw`/`_recent`, `engine/family_shared.py` per-stream draw when any stream in the round is penalised, `engine/family_prefill.py` passes each row's `recent`). The Metal decode kernel keeps its batched fast path whenever no stream in the round is penalised.
3. `server/http.py`: forward the four keys from the request body (the HTTP layer whitelists body keys).

Design note: the penalty deliberately reroutes to CPU with drafts off instead of teaching the hand-written Metal sampling kernel (and the CUDA rank headers / `pack_sampling`) to carry per-row history; that costs the speculative speed-up while a penalty is active (measured: ~180 → ~125 tok/s on gemma-4-26b-a4b) and is correct and self-contained. Full-speed penalised drafting is a later step.

Tests: `tests/test_repetition_penalty.py` — penalty math, request validation and resolution, selection flips, per-row history; the decode-loop / lane / stream suites stay green.

---

## PR 3 — `feat-toolcall-repair-v0.6.0`

**Title:** tools: repair or hide an unterminated / malformed gemma tool-call block instead of leaking it

**Body:**

A large `write` can run out of tokens after it opens `<|tool_call>call:write{…}` but before `<tool_call|>`, or close the block with an unbalanced `<|"|>` string, or emit a value as a bare degenerate token (`filePath:mapsto_path_…_now`). With no complete block to match, the raw markup falls through into the reply text (`tool_calls: []`, `content` full of `<|tool_call>…`), the client shows a wall of markup and the call never runs.

Three commits in `server/tools.py` (+ `tests/test_toolcall_repair.py`):

1. `_repair_gemma_call(fragment, known)`: best-effort `(name, arguments)` from an unterminated `call:NAME{…}` / `:NAME{…}` — complete `key:value` pairs are kept and a truncated final string value is salvaged to the end of the reply. `_repair_leaked_tool_calls(content, known)`: for each leaked opener, an unterminated block is repaired into a structured call when it names an offered tool, hidden when it names an unoffered one, and plain text that merely mentions the marker is untouched. Runs only on the lenient reply path (`max_calls is None`) and only when `<|tool_call>` is still in the text.
2. A *terminated* block whose string has an unbalanced `<|"|>` is repaired the same way.
3. Scope by salvageability rather than one syntactic signature: if the args can be salvaged for an offered tool (at least one field) the block becomes a structured call; an unterminated call that cannot be salvaged is hidden; a terminated block that cannot be salvaged (a bare `{location}`) stays as text, matching the existing deliberate behaviour; text after a terminated block is preserved.

Verified live (gemma-4-26b-a4b, M5 Max): forcing the cut-off with `max_tokens=300` on a whole-file write, the reply is a structured `write` call carrying the salvaged partial content with zero markup in `content`; the `glob{pattern:<|"|>*/}` unclosed string and the degenerate bare `filePath` value each come back as structured calls; a bare `{location}` stays as text; a real "find all Python files" request returns a clean `glob` call. The salvaged call only has the fields the model emitted before the cut, so it is a clean, retryable partial rather than a guaranteed-valid write — the point is no markup in the chat and a structured call the client can act on.
