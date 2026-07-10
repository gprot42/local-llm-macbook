# Developer Notes — gemma4-server-uncensored-mlx-31b

Diagnosed bugs, root causes, and **lean** proxy workarounds in
`gemma4_mlx_kilo_proxy.py` (~1.6k lines; rewritten from a ~10k-line special-case
proxy). Task-specific hacks (hardcoded 1942/zzz-test games, deterministic HTML
fallbacks, multi-layer planning nudges) were removed.

**This folder:** Gemma 4 **31B** Heretic (`gemma-4-31b-heretic-mlx-4bit`).

Upstream tracking: [vllm-mlx#590](https://github.com/waybarrios/vllm-mlx/issues/590).
Patch status: `./check_upstream_patches.sh --fetch`

---

## Lean proxy — what it still does

| Feature | Purpose |
|---|---|
| Tool name/arg remap | Cline/Roo → Kilo schema; AskQuestion / TodoWrite repair |
| Fuzzy `old_string` | Indent drift repair for StrReplace edits |
| Harmony `logit_bias` | Ban tokens 98/100/101 on agentic turns |
| Temp floor 0.35 + thinking off | Avoid temp=0 agent stalls after tools |
| Empty-delta abort | vllm-mlx empty `delta:{}` spin → graceful `finish_reason=stop` |
| Stall abort | No token progress → graceful stop (not OpenAI error) |
| Args cap | Tool-arg repetition loops |
| Strip planning tools | After todowrite with no write yet, hide planning tools |
| Single-flight lock | One generation at a time (MLX safety) |
| Model rewrite | `--model` forces local weight id |

---

## TL;DR — where the bug actually lives

| Layer | Bug? | Notes |
|---|---|---|
| **Kilo Code** (IDE plugin) | No | Behaves correctly. Renders whatever the model emits. |
| **`gemma4_mlx_kilo_proxy.py`** | Workarounds only | Lean guards + tool repair — does not fix engine path |
| **Gemma 4 weights** | No | Valid tokens including tool-call structure |
| **`mlx_vlm` Gemma 4 attention** | Patched | `BatchKVCache.offset` — `patches/gemma4_mllm.py` |
| **`vllm_mlx.text_model_from_vlm`** | **YES — primary root cause** | Hard-coded Qwen3.5 TextModel for every MLLM → Gemma 4 `division by zero` → MLLM fallback. Local fix: `patches/text_model_from_vlm.py` |
| **`mlx_vlm.generate` stream ownership** | Patched | `Stream(gpu, N)` after text→media — `patches/engine/simple.py` + `patches/generate.py` |

---

## Bug 1: vllm-mlx falls back to MLLM path for Gemma 4 (root cause)

### Where

`venv/lib/python3.14/site-packages/vllm_mlx/text_model_from_vlm.py:49`

```python
# Always import from qwen3_5 — TextModel and TextModelArgs handle both
# dense and MoE natively …
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
```

The comment claims "TextModel handles both dense and MoE natively" but only
within the Qwen 3.5 family.  Gemma 4's `text_config` has different field names
and shapes (notably `head_dim`, `num_attention_heads`, `partial_rotary_factor`)
and triggers a `ZeroDivisionError` inside
`TextModelArgs.from_dict()` or its `__post_init__`.

### Observable symptom on startup

```
ERROR:vllm_mlx.text_model_from_vlm:Failed to build TextModel from vlm: division by zero
INFO:vllm_mlx.engine.simple:SimpleEngine loaded: gemma-4-26b-heretic-mlx-4bit (MLLM=True)
```

`MLLM=True` here is *not* a correct config — it means "fast TextModel path
failed; fall back to the mlx_vlm multimodal path".  Even for pure text
generation, every request now goes through code paths designed for
vision/audio tokens that don't correctly handle Gemma 4's `<|channel>thought…<channel|>`
thinking-block exit when the next expected token is a tool-call delta.

### Downstream consequence: empty-delta loop

After the model exits a thinking block, the MLLM-path generation loop hits a
state where it emits SSE chunks with completely empty deltas at full speed
(~40/sec) until `max_tokens` is reached.  Captured from `[stall-dump]` log:

```json
data: {"id":"chatcmpl-e7729881","object":"chat.completion.chunk",
       "choices":[{"index":0,"delta":{},"finish_reason":null}],"usage":null}
```

- `delta.content` absent
- `delta.tool_calls` absent
- `finish_reason` stays `null`
- `usage.completion_tokens` stops advancing

At 16 384 `max_tokens` × ~25 ms/empty-chunk this is **≈7 minutes** of CPU
spinning per stuck stream with no output.

### Pattern of failure

| Turn | Messages | What model produces | Result |
|---|---|---|---|
| 1 | `[system, user]` | Tool call (e.g. `todowrite`) | OK — clean stream end |
| 2+ | `[system, user, assistant, tool, …]` | Thinking → some text narration → empty-delta loop | Stuck.  No tool call emitted. |

Even when turn 2 *intends* to emit a tool call, it often emits 100–800 chars
of plain narration first (`"I'll help you create…"`, `"First, I'll list the
files…"`), then enters the empty-delta loop before the tool delta arrives.
In one observed case the tool delta did eventually arrive after ~12 s of
empty chunks (`[reclassify] text → tool (late tool-call delta arrived after
247 chunks; discarding 848 buffered text chars)`).  In most cases it never
arrives.

This is a symptom of the wrong code path being used for token routing —
nothing about Gemma 4 the model is producing invalid output.

### Possible upstream fixes (not yet applied)

1. **Patch `text_model_from_vlm.py` to dispatch by architecture.**
   Detect `config["model_type"]` and import `mlx_lm.models.gemma3` (or
   whatever the correct mlx_lm class is for Gemma 4 weights) instead of
   blindly using `qwen3_5`.  This is the clean fix — restores the fast text
   path and avoids the MLLM fallback entirely.

2. **Use `mlx_lm` directly** for text-only Gemma 4 inference, bypassing
   `vllm-mlx` for this model.  Simpler than patching but loses vllm-mlx
   features (disconnect guard, batching, etc.).

3. **Patch the MLLM path itself** to correctly emit tool-call deltas after a
   thinking-block exit.  Hardest, requires understanding mlx-vlm's generation
   loop in detail.

---

## Bug 2: Gemma 4 Attention vs BatchKVCache (already patched)

### Where

`mlx_vlm.models.gemma4.language.Attention.__call__`

### Symptom

When `BatchKVCache` is used, `cache.offset` is an `mx.array` whose `__iadd__`
mutates in place.  Gemma 4's `Attention` reads `cache.offset` into a local
variable *before* calling `update_and_fetch`, then re-uses the same local for
RoPE on queries.  Because the local is the same object as `cache.offset`, it
gets silently mutated by `update_and_fetch`, giving queries the wrong RoPE
position and producing garbage tokens.

### Fix

`patches/gemma4_mllm.py` monkey-patches `Attention.__call__` to snapshot the
offset *before* any cache mutation:

```python
off = cache.offset
offset = (off + 0) if isinstance(off, mx.array) else off
```

Status: applied at startup, idempotent (`_batch_patched` guard).

---

## Bug 3: Stream hangs forever, `UnknownError` in Kilo (workaround applied)

### Symptom

Kilo UI spinner runs for 90 s, then surfaces:

```
UnknownError: "Model stalled — no new content for 92s
              (emitted 0 tokens then froze).  Likely cause: model is
              looping on whitespace / end-of-turn tokens after a
              thinking block."
```

This is a downstream effect of **Bug 1**.  Kilo would then retry, which
re-queued behind the same stuck upstream — making things worse.

### Workaround in `gemma4_mlx_kilo_proxy.py`

Two guards added in `stream_gen`:

**`[empty-delta-abort]`** (fires in ~2–3 s)

Tracks `empty_delta_streak` — consecutive chunks where
`usage.completion_tokens` did not change *after* `response_type` was
classified.  Once 100 in a row contribute no new tokens, the proxy:

1. Closes upstream HTTP stream.
2. Injects synthetic chunk with `finish_reason: "stop"`.
3. Sends `data: [DONE]`.

Kilo sees a normal completion (possibly empty / text-only) and moves on.

**`[stall-abort]`** (90 s fallback, also graceful)

If `empty_delta_abort` doesn't fire (e.g. `response_type` never got
classified because the stream produced zero useful chunks), this time-based
fallback also sends `finish_reason: "stop"` — **not** an OpenAI error payload,
because errors trigger Kilo to retry and re-queue behind the same broken
stream.

### Verified result

Before fix:
```
01:00:36 ERROR  gemma4_mlx_kilo_proxy — [stall-abort] no token change for 92.5s — aborting upstream
                  ↑ 92 seconds wasted, UnknownError shown to user
```

After fix:
```
01:05:31 WARNING gemma4_mlx_kilo_proxy — [empty-delta-abort] 100 consecutive chunks with no
                  token change after classification (response_type=text,
                  token_count=353, chunks=686).  vllm-mlx is looping on empty
                  deltas — injecting finish_reason=stop and terminating gracefully.
                  ↑ 14 seconds, no error popup, Kilo moves on
```

### What the workaround does NOT fix

If turn 2 emits 353 tokens of plain narration text and *then* enters the
empty-delta loop without ever emitting a tool call, Kilo will render the
narration and then have nothing to do (because the model didn't call the
expected file-write tool).  The user sees a To-Dos checklist and "First,
I'll list the files…" with no files written.

This is **Bug 1** showing through.  The proper fix is to repair the
text-model fast path so vllm-mlx doesn't use the MLLM fallback for Gemma 4.

---

## Bug 5: Harmony token leak + planning-stall infinite loop (root-cause fix)

### Symptom

On any agentic Kilo Code prompt (e.g. *"create arkanoid game with
powerups and fast playing chiptune music"*) the model produces:

```
[first-delta] type=text content_preview='<|channel>' elapsed=3.27s
[harmony-leak-suspect] first content delta contains Harmony control marker
[reclassify] text → tool (late tool-call delta arrived ...)
[tool-stream] mode=stream first_name='todowrite' buffered_pre_decision=0
[args-cap] tool='todowrite' exceeded 8192B args cap (got 8700 chars,
           tokens=18, chunks=19) — model in repetition loop
```

Every retry produces the **same** `<|channel>...` prefix, so Kilo's UI
loops indefinitely on "0 / 6 todos done" with no files written.

### Root cause

The `gemma-4-26b-heretic-mlx-4bit` distillation data contains OpenAI
Harmony control tokens.  Two specific token IDs from
`gemma-4-26b-heretic-mlx-4bit/tokenizer.json` `added_tokens` derail the
upstream `gemma4_tool_parser`:

| ID | Token | Effect |
|---|---|---|
| 100 | `<\|channel>` | Harmony channel header — parser sees as plain text, can't recover |
| 98 | `<\|think\|>` | DeepSeek-style CoT header — same failure mode |

Once the prefix is emitted, the upstream parser is locked into a bad
state.  Prompt engineering can't override this — the model literally
has very high probability mass on `<\|channel>` as the first decode
token of any tool-call turn.

### Fix (3-layer)

**Layer 1 — sampler-level token ban** (eliminates the root cause):

Three coordinated patches add `logit_bias` end-to-end to the
`vllm_mlx` server:

```diff
patches/api/models.py        # +  logit_bias: Optional[Dict[str, float]] = None
patches/server.py            # +  forward logit_bias into chat_kwargs
patches/engine/simple.py     # +  coerce {str: float} → {int: float}, pass to
                             #    mlx_lm.sample_utils.make_logits_processors
```

These are mirrored from venv-site-packages into the in-tree `patches/`
directory and copied back at startup by `2_start_mlx.sh` so they
survive `pip install --upgrade vllm_mlx`.

The proxy then injects `_HARMONY_LOGIT_BIAS` for every agentic
request inside `_force_xhigh_settings`:

```python
_HARMONY_LOGIT_BIAS = {"100": -100.0, "98": -100.0}  # <\|channel>, <\|think\|>
```

Caller-supplied `logit_bias` entries win on key collision so a power
user can still positively boost these tokens for debugging without a
code change.

**Layer 2 — strip planning tools on retry** (defense in depth):

Even with the Harmony tokens banned, the model occasionally re-calls
`todowrite` on retry instead of `write`.  `_strip_planning_tools_if_stuck`
removes `todowrite` / `todoread` / `update_todo_list` from the tools
list on any turn where:

  * at least one prior assistant turn called a planning tool, **and**
  * no prior assistant turn has called a write tool yet

The first `todowrite` call is allowed (model gets to plan once); from
the second turn onward, the only tools the model can see are the
write tools, so it has no escape hatch.

**Layer 3 — harsher post-strip nudge**:

When Layer 2 strips tools, `_upgrade_stall_nudge_after_strip` swaps the
standard `_TEXT_STALL_HARD_CREATE` user message for the harsher
`_TEXT_STALL_HARD_CREATE_NO_PLANNING` variant that:

  * names the specific tools that were removed (so the model
    understands the rejection)
  * explicitly bans `<\|channel>`, `<\|think\|>`, `<\|message>`,
    `analysis`, `commentary` at the prompt level (belt-and-braces
    alongside `_HARMONY_LOGIT_BIAS`)
  * anchors decode to "start with a direct `tool_calls` entry for
    `write` — no preamble, no thinking block"

### Verification

```bash
venv/bin/python -c "from vllm_mlx.api.models import ChatCompletionRequest; \
    print('logit_bias' in ChatCompletionRequest.model_json_schema()['properties'])"
# → True

venv/bin/python tests/test_steer_planning_stall.py
# → 21 passed, 0 failed
```

### What this fix does NOT cover

Layer 1 only suppresses two specific token IDs (100, 98).  If the
heretic distillation introduces new Harmony / CoT variants in a
retrained build, the proxy will need updated IDs.  Run the
`scan_tokenizer.py` helper to enumerate `added_tokens` against the
banned set on each model bump.

---

## Bug 4: Ghost streams on restart (minor)

### Symptom

On `./2_start_mlx.sh` startup, 4–5 parallel requests arrive at the proxy
simultaneously, all from the same Kilo conversation but with different
embedded timestamps.

### Cause

Kilo Code queues unacknowledged requests across sessions and replays them
when the endpoint comes back up.  Not a proxy or model bug.

### Impact + mitigation

Each ghost stream occupies the `SimpleEngine` (single-user mode, no batching)
for up to 90 s.  With 5 ghosts that's ~7.5 minutes before fresh requests run.

The `[empty-delta-abort]` fix from Bug 3 reduces each ghost's hold time from
90 s to ~2–3 s, so the queue clears in under a minute rather than over an
hour.  No proxy-side fix for the ghost replay itself.

---

## Bug 6: Byte-cap-aborted text-collapse loop after Harmony fix (root-cause fix)

### Symptom

After Bug 5 banned `<|channel>` / `<|think|>` at the sampler, the model
no longer emits Harmony control markers — but on long create-tasks
(e.g. "create arkanoid game with powerups and fast playing chiptune
music") it instead falls into a **plain-text repetition loop**:

```
piece...
1. I'll start by exploring the directory structure.
2. I'll create a single HTML file.
3. I'll implement the game logic.
4. I'break into pieces of code and break it into pieces of code and
   break it into pieces of code and break it into pieces of code...
```

The model emits a planning checklist, items 3-4 dissolve into
token-level repetition, and the upstream stream guard catches it:

```
[text-mode-runaway] byte-cap exceeded (4098 > 4096B) at total=4098
chars, tokens=1258, chunks=1259 — model is stuck in text loop
instead of calling a tool; emitting graceful stop
```

The graceful stop hands the truncated ~4 KB assistant message back to
Kilo, which then **re-sends the same task** as a new user message:

```
[request] 4 messages [system, user, assistant, user]
```

`_break_text_stall` walks back from this conversation tail to detect
stalls, but its STOP condition is "break at the first real user
message".  The trailing `user_retry` triggers `break` on iteration 1,
so `consecutive_text` stays at 0 and the HARD `write` nudge never
fires.  The model loops forever, wasting ~16 s of decode per retry.

### Cause

Two collaborating issues:

1. **No Harmony exit hatch.**  With Bug 5's logit_bias active, the
   model can't ESCAPE planning text via the easy `<|channel>` route.
   For some create-tasks the next-best high-probability completion at
   `temperature=0.35` is a checklist that then collapses into a
   2-token cycle ("break it into pieces of code") that runs the
   text-cap.

2. **Walk-back blindness to retry pattern.**  `_break_text_stall`
   was designed for the shape `[..., a_text(1), a_text(2)]` where
   two consecutive text turns are visible in history.  Kilo's retry
   shape `[..., a_text(collapse), u_retry]` interposes a fresh user
   message that terminates the walk-back before the collapse is
   seen.

### Fix (defense-in-depth, Layers 1-6)

**Layer 1 — collapse threshold constant** (`gemma4_mlx_kilo_proxy.py`)

```
_TEXT_STALL_COLLAPSE_THRESHOLD = 2048
```

Half the agentic text byte-cap (4096).  A trailing assistant text
turn ≥ 2 KB on a create-task is conclusively a stall — legitimate
planning prose is typically 200-1000 chars.

**Layer 2 — trailing-collapse rule** (walk-back)

Accumulate `trailing_text_chars` across the consecutive-text run.
Fire if `consecutive_text >= 1 AND trailing_text_chars >= threshold`
even without a second text turn.  Catches the non-retry shape
`[s, u, a_text(4 KB)]`.

**Layer 3 — retry pre-scan**

Before the walk-back, do a bounded pre-scan that finds the most
recent assistant text-only turn, ignoring intervening user-retry
messages.  If its size ≥ threshold AND there's no intervening
write-tool call in the same task window, set `recent_collapse=True`.
Catches the Kilo retry shape `[s, u, a_text(4 KB), u_retry]`.

The `recent_collapse` flag is OR'd into `should_fire`.  The pre-scan
itself short-circuits when it sees a prior write call so a model
that has already written something can collapse without
re-triggering the create-tailored nudge (which would re-anchor it
to the original task and potentially overwrite progress).

**Layer 4 — short marker-collapse detection + expanded Harmony bias**

A second live failure (2026-05-13 15:26) showed a short collapse:

```
[text-mode-runaway] sentence '<channel|><thought>' repeats ≥5 times
consecutively in tail (519 chars) ...
```

This pattern aborts at ~500 chars (well below the 2 KB threshold), so
byte-size heuristics alone do not fire on the next turn.

Two changes close that gap:

1. **Sampler suppression expanded**: `_HARMONY_LOGIT_BIAS` now also bans
   token id `101` (`<channel|>`) in addition to `100` and `98`.
2. **Marker-collapse detector**: `_looks_like_control_marker_collapse()`
   flags short repetitive control-marker text (notably repeated
   `<channel|><thought>`), and `_break_text_stall` treats that signal
   as equivalent to a byte-cap collapse for both trailing and retry
   pre-scan paths.

### Verification

```
$ venv/bin/python tests/test_steer_planning_stall.py
...
PASS  test_break_text_stall_fires_on_kilo_retry_with_collapse
PASS  test_break_text_stall_fires_on_trailing_collapse_no_retry
PASS  test_break_text_stall_skips_short_planning_text
PASS  test_break_text_stall_skips_collapse_when_not_create_task
PASS  test_break_text_stall_skips_collapse_when_write_already_done
PASS  test_break_text_stall_legacy_two_short_turns_still_fires
PASS  test_collapse_threshold_constant_is_sensible

39 passed, 0 failed
```

In production logs the new path emits:

```
[text-stall] HARD break: ... recent_text_chars=4094 ...
  collapse=False recent_collapse=True — injecting user-turn write nudge
```

and for short marker loops:

```
[text-stall] HARD break: ... trailing_marker_collapse=True ...
```

**Layer 5 — delimiter-token loop detector + no-tool text deadline**

Another live failure showed long prose loops dominated by punctuation-
delimited repeated tokens (for example `the-the-the-...` and
`enough-enough-enough-...`).  These can evade the original whitespace-
only word-repeat regex and may continue streaming for >10 s before a
user interrupt.

Two additional guards were added in text mode:

1. `_TEXT_TOKEN_REPEAT_WITH_DELIMS_RE` detects repeated tokens joined by
   punctuation delimiters (`-`, `_`, `,`, `.`, `;`, `:` ...), and aborts
   as `[text-mode-runaway]`.
2. Agentic no-tool text deadline (`8 s`): if write-mode has already
   emitted substantial text (`>512` chars) without any tool stream
   starting, abort early with a graceful stop.

**Layer 6 — first-turn create tasks force the write tool**

The abort guards reduce the wait, but they still let the first turn begin
as planning prose.  For first-turn create/build/implement requests with a
detected write tool, the proxy now sets:

```
tool_choice = {"type": "function", "function": {"name": "<write-tool>"}}
```

and filters the tool list to that write tool only.  This triggers the
upstream forced-tool path, which injects a template-level instruction and
disables thinking for that turn.  The goal is to prevent first-turn prose
loops entirely rather than merely abort them quickly.

Forced-write requests also get a longer pre-first-delta
low-chunk-rate grace (`_LOW_CHUNK_RATE_FORCED_WRITE_PREFILL_S = 45`)
while `response_type is None` and `token_count == 0`.  Without this
grace the normal aggressive 10 s agentic low-chunk-rate guard can kill
the request during prompt/tool-template prefill before the model emits
the first real `write` delta.

Once a real `write` stream starts, it also gets a longer slow-token-rate
warmup (`_LOW_TOKEN_RATE_WRITE_TOOL_AFTER_S = 60`).  This prevents the
generic 6 s agentic slow-rate guard from killing a valid large write
after the first `write` delta while the model is paused before the next
argument chunk.

### Residual risk

The 2 KB threshold is a heuristic.  If a future Kilo system prompt
or tool description balloons legitimate planning prose past 2 KB, we
will start false-firing.  Mitigations:

- The nudge content is helpful even when "incorrectly" applied — it
  reinforces the write-tool requirement on a create task.
- Monitor `recent_text_chars` in `[text-stall]` log lines.  If legit
  planning ever exceeds 1.5 KB, raise the threshold to 3 KB (still
  under the 4 KB byte-cap).
- Adding a "is this text repetitive?" detector (sliding-window
  n-gram repeat count) would make the trigger precise but adds
  complexity; deferred until the heuristic actually false-fires.

---

## Architecture overview

```
Kilo Code (port 8080)
    │  OpenAI-compatible SSE
    ▼
gemma4_mlx_kilo_proxy.py  ← proxy / middleware
    │  • thinking-block filter (_ThinkingFilter)
    │  • write-mode injection
    │  • tool-call reclassify
    │  • empty-delta-abort / stall-abort   ← Bug 3 workaround
    │  OpenAI-compatible SSE (modified)
    ▼
vllm-mlx (port 8090)
    │  • text_model_from_vlm  ← Bug 1 root cause (hardcoded qwen3_5)
    │  • SimpleEngine (falls back to MLLM=True for Gemma 4)
    │  raw token stream
    ▼
mlx_vlm.models.gemma4
    │  • Attention (BatchKVCache offset bug — Bug 2, patched)
    ▼
Gemma 4 weights (MLX, 4-bit quantised)
```

`gemma4_mlx_kilo_proxy.py` is the only component that can be patched without
recompiling vllm-mlx.  The proper home for the Bug 1 fix is
`venv/lib/python3.14/site-packages/vllm_mlx/text_model_from_vlm.py` — a
mirrored copy in `patches/` and a setup-time install step would make this
reproducible.

---

## Log markers reference

| Token | Source | Meaning |
|---|---|---|
| `[stall-dump]` | proxy | 20 s of no token-count change — prints raw chunk bytes (diagnostic only) |
| `[empty-delta-abort]` | proxy | 100 consecutive empty `delta:{}` chunks post-classification → graceful stop |
| `[stall-abort]` | proxy | 90 s no-token-change fallback → graceful stop |
| `[reclassify] text → tool` | proxy | Tool-call delta arrived after stream was classified as text (Gemma 4's `<\|channel>thought…<channel\|>` looks like plain text initially) |
| `[stream-state]` | proxy | Periodic 30 s status report |
| `[harmony-leak-suspect]` | proxy | First content delta starts with `<\|channel>` / `<\|think\|>` — Bug 5 indicator; if no tool reclassify arrives within 8 s the stream is aborted |
| `[harmony-leak-abort]` | proxy | Bug 5 fast-abort fired (< 20 chunks, ≤ 4 tokens, Harmony marker, no reclassify) |
| `[args-cap]` | proxy | A streamed tool call's `arguments` field exceeded 8 KB — Bug 5 model-repetition indicator |
| `[slow-token-rate-abort]` | proxy | Tool-call stream < 2 tok/s for > 15 s — Bug 5 degraded-decode indicator |
| `[settings] agentic — logit_bias suppressing Harmony tokens` | proxy | Bug 5 Layer 1 active: `<\|channel>`/`<\|think\|>` banned at sampler |
| `[strip-planning] removed planning tools` | proxy | Bug 5 Layer 2 active: planning tools revoked because model hasn't written yet |
| `[strip-planning] upgraded trailing stall nudge to NO_PLANNING variant` | proxy | Bug 5 Layer 3 active: harsher post-strip user nudge installed |
| `[disconnect_guard]` | vllm-mlx | Upstream keep-alive — `poll #N elapsed=Ts` confirms the engine is alive but not producing |
| `ERROR:vllm_mlx.text_model_from_vlm: division by zero` | vllm-mlx | **Bug 1 indicator** — text fast path failed at startup, MLLM fallback engaged |
| `INFO:vllm_mlx.engine.simple: SimpleEngine loaded: … (MLLM=True)` | vllm-mlx | Confirms the broken fallback is in use |

---

## Bug 6: media request crashes after text route (`Stream(gpu, N)`)

### Symptom

After a successful text-only request, an image request can reach the media route
and immediately fail during prefill:

```text
INFO:vllm_mlx.engine.simple: Media request → MLLM path
ERROR:vllm_mlx.server:Streaming error, ensuring terminal frame:
There is no Stream(gpu, 2) in current thread.
```

### Root cause

`mlx_vlm.generate` keeps a module-level `generation_stream`. The server runs
blocking generation in worker threads, while the MLLM text fast path and media
path can enter different MLX stream contexts. A stream created or rebound for a
previous text request may not exist in the worker thread that later handles the
media request.

### Local patch

There are two coordinated local patches:

1. `patches/engine/simple.py` calls `_bind_worker_generation_streams(
thread_local=True, clear_cache=True)` immediately before media `chat()` and
`stream_chat()` dispatch, and runs media generation on the engine/event-loop
thread instead of `asyncio.to_thread()`. This is intentional: the model and
vision tensors are created on that thread, and moving Gemma4 media prefill to a
worker can still trip MLX stream ownership even after rebinding. The helper:

- synchronizes and clears MLX/Metal cache before rebinding
- creates a fresh `mx.new_stream(mx.default_device())` in the worker thread
- sets it as the current default stream
- updates `mlx_lm.generate.generation_stream` and
  `mlx_vlm.generate.generation_stream`
- clears cache again after media generation completes

2. `patches/generate.py` adds `_refresh_generation_stream()` at
`mlx_vlm.generate` request entry, creating a fresh worker-thread
`mx.new_stream(...)`, and uses that local `stream` for both
`mx.stream(...)` and `wired_limit(...)`. This fixes crashes that occur inside
`generate_step()` prefill before the engine-level rebinding can help.

`2_start_mlx.sh` copies this patched `simple.py` into the venv on every start,
and now also copies `patches/generate.py` into `mlx_vlm/generate.py`, so
restarting the server applies the fix.

---

## Bug 7: follow-up implementation prompt classified as read-only

### Symptom

After a user first asks for suggestions and then says something like:

```text
implement 3, 4, 7, 8, 2, 1
```

the proxy logged `[readonly] Review/suggest intent detected`, stripped write
tools, and eventually hit `[readonly-runaway]` instead of allowing the agent to
write files.

### Root cause

`_is_readonly_intent()` scanned every real user message in the conversation.
A stale earlier message such as "suggest improvements to 1942..." therefore
kept later implementation follow-ups stuck in read-only mode.

### Fix

`_is_readonly_intent()` now scans from the latest user turn backward, skips only
proxy-injected nudges, and classifies the first real user turn it sees. If that
latest turn contains create/update verbs such as `implement`, `create`,
`update`, `improve`, or `ensure`, the request is explicitly treated as writable
even when older turns asked for suggestions. Regression tests cover both the
follow-up implementation case and a latest-turn suggestions case.

---

## Startup UX: ready message

`2_start_mlx.sh` now waits for the public proxy port to bind after launching
`gemma4_mlx_kilo_proxy.py` and prints:

```text
→ Ready — open http://localhost:8080 in Kilo Code
```

This makes it clear when both vllm-mlx and the Kilo/Continue proxy are ready.

---

## Stream observability: tokens/sec

The proxy's periodic `[stream-state]` and final `[stream-end]` logs now include
`tps=...`. When TTFT is known, the value includes both total throughput and
post-TTFT decode throughput:

```text
tps=6.8 tok/s total, 9.4 tok/s decode
```

This separates prompt-prefill wait from actual token generation speed.

---

## Why this setup does not use MTP

This project runs `gemma-4-31b-heretic-mlx-4bit`. MTP is not enabled: Heretic 4-bit
does not ship matching MTP draft-head weights. Use stock IT + MTP in
`../gemma4-server-mlx-31b/` for speculative-decode experiments.

---

## Bug 8: write tool reported success but Kilo wrote nothing

### Symptom

For a follow-up prompt like:

```text
implement 3, 4, 7, 8, 2, 1
```

the model produced a `write` tool call and the proxy returned a deterministic
summary saying `Created /Users/aicoder/src/zzz-test/index.html`, but Kilo then
reported:

```text
no updates were written to disk
```

### Root cause

The proxy's post-write final shortcut assumed every latest `write` tool result
meant the client had successfully written the file. That was false: Kilo can
return a failed/no-op tool result such as "no updates were written to disk".
The proxy then hid the failure behind a success summary.

### Fix

`_latest_tool_result_is_for_write()` now rejects failed/no-op write results
before enabling the deterministic post-write summary. Suspicious write targets
are repaired to an existing game HTML file when possible, but the proxy no
longer preemptively bypasses the model/tool path for numbered 1942 UX
follow-ups.

### Historical follow-up fix

The first deterministic-write implementation attempted to call ASGI `send()`
from inside the `StreamingResponse` body generator, where `send` is not in
scope. That crashed with:

```text
NameError: name 'send' is not defined
```

That preemptive branch has since been removed.

### Follow-up fix 2

The deterministic fallback originally emitted a relative `index.html` path when
no explicit target file was present. Some clients can display the subsequent
summary while failing to materialize a relative tool path in the intended
workspace, which looks like "Created index.html" but no visible file change.

Fallback writes now resolve to an absolute target path:

1. explicit `.html` path in the prompt
2. existing workspace `1942.html` from `environment_details`
3. workspace `index.html`
4. `/Users/aicoder/src/zzz-test/1942.html` when that local test workspace exists
5. `/Users/aicoder/src/zzz-test/index.html`
4. absolute `./index.html` as a final fallback

The post-write deterministic summary also uses the latest real user request as
the `Task:` line, not stale earlier suggestion text.

### Follow-up fix 3

The deterministic preemptive write path was removed because it could choose the
wrong target file and made repeated prompts feel cached. Writes now flow through
the model/tool path again. The remaining repair/fallback helpers prefer an
existing `1942.html` in the active workspace before `index.html`, and HTML
fallback output gets a `gemma4-mlx-kilo-write-stamp` comment containing a
millisecond timestamp and task snippet so repaired writes are visibly fresh.

---

## Bug 9: readonly 1942 suggestions hang and show nothing

### Symptom

A prompt like:

```text
suggest improvements to 1942 for the user experience
```

was correctly classified as readonly, but the model emitted a small amount of
filtered/non-visible text and then only heartbeat chunks. Kilo showed a blank
assistant response until the proxy hit the generic 90 second stall abort.

### Root cause

The readonly fallback only fired for byte-cap or empty-content runaway cases.
This failure mode produced a small token count, no visible bytes, and then a
long no-token-change stall, so neither readonly fallback fired early.

### Fix

The deterministic readonly shortcut was removed because the user wants every
query to be fresh rather than cached. The proxy still keeps readonly runaway
guards, but it no longer bypasses the model for known 1942 UX suggestion
prompts.

---

## Privacy / freshness: query bodies are not logged by the proxy

`--debug` used to log a truncated request body, which included user prompts.
That made repeated local debugging feel like query caching and exposed prompt
text in terminal scrollback. The proxy now logs only request metadata in debug
mode: message count, role list, tool count, temperature, and stream flag. It no
longer logs full request bodies or prompt text. Upstream `vllm-mlx` may still
log its own short last-user preview; remove or reduce that in
`patches/server.py` if terminal logs must contain no prompt snippets at all.

---

