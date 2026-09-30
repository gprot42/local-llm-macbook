# Findings — Gemma 4 26B-A4B (abliterated) on TensorFold

**Date:** 2026-09-30 · **Machine:** Apple M5 Max, 128 GB · **Engine:** TensorFold 0.5.0 (`gemma4` lane, MLX)
**Model:** local self-quantised `gemma-4-26b-a4b-abliterated-4bit` (26B-total / 4B-active MoE, 4-bit, **4.502 bits/weight**)
**Drafter:** `z-lab/gemma-4-26B-A4B-it-DFlash` (external DFlash) · **Conditions:** only this model loaded (uncontended GPU), server-default sampling (temp 1.0 / top-p 0.95 / top-k 64)

## TL;DR

**It is very fast** — ~140–190 tok/s decode. It's a 4B-active MoE, so even serial decode is quick, and DFlash drafting adds ~1.3× on code. Prefill is ~3,000 tok/s. Caveat: fast for **chat**; agentic **tool use is currently broken** on the lane (see below).

## Decode speed (drafted)

| Workload | Tokens | Decode tok/s | Draft acceptance |
|---|---|---|---|
| Code (LRU cache class) | 250 | **182** | 152/220 (69%) |
| Code (ISO-8601 parser) | 250 | 164 | 147/274 (54%) |
| Prose (~150 words) | 150 | 139 | 70/172 (41%) |
| Code (nth-prime, greedy) | 200 | **190** | 132/193 (68%) |

Engine-reported `tok/s` (matches client wall-clock within noise). Code drafts better than prose — more predictable tokens → higher DFlash acceptance — the usual pattern.

## Drafting vs serial (same greedy prompt, nth-prime)

| Mode | Decode tok/s |
|---|---|
| Drafted (DFlash) | **190** |
| Serial (`"draft": false`) | 148 |

**≈1.28× speedup** from drafting on code. The base is already high because only ~4B params are active per token (MoE). The external base-model drafter accepts well against the abliterated target (drafts are exact-verified, so correctness is unaffected).

## Prefill

| Prompt | Tokens | Prefill time | Prefill tok/s |
|---|---|---|---|
| Large cold prompt | 21,682 | 7.0 s | **~3,090** |
| Short / cached | ≤9k | <0.3 s | (prefix-cache hit) |

Cold prefill ~3k tok/s; a warm/cached prefix returns in well under a second (TTFT ~0.05–0.3 s on repeat/short prompts).

## Notes & caveats

- **Numbers are single-run, indicative.** Expect some run-to-run variance (sampling + draft acceptance vary); code sits ~160–190 tok/s, prose ~140.
- **Uncensored + clean chat** verified (no `<|channel>` leak, refusals removed).
- **Tool calling is broken on the lane** — agentic requests leak `<|tool_call>:name{args}` as text instead of structured `tool_calls`. This is a TensorFold `gemma4`-lane parser bug affecting the stock pack too (not the abliteration), filed as **[ashhart/TensorFold#121](https://github.com/ashhart/TensorFold/issues/121)**, still unfixed on 0.5.0. So this model is great for chat but not yet usable for OpenCode agentic/tool workflows.

## Reproduce

Server on `:8104` (engine) / `:8094` (proxy):

```bash
./2_start_tensorfold.sh          # loads the local 4-bit pack + DFlash drafter
```

Then benchmark (decode / drafted-vs-serial / prefill):

```python
import json, time, http.client
def gen(p, mx, draft=True, temp=None):
    b={"model":"x","max_tokens":mx,"messages":[{"role":"user","content":p}]}
    if not draft: b["draft"]=False
    if temp is not None: b["temperature"]=temp
    t=time.time(); c=http.client.HTTPConnection("127.0.0.1",8104,timeout=600)
    c.request("POST","/v1/chat/completions",json.dumps(b),{"Content-Type":"application/json"})
    d=json.loads(c.getresponse().read()); el=time.time()-t; u=d["usage"]
    print(f"{u['completion_tokens']/el:.1f} tok/s  (prompt={u['prompt_tokens']})")
gen("Write a Python class for a thread-safe LRU cache.", 250)             # drafted
gen("Write a Python function returning the nth prime.", 200, draft=False) # serial
```

The engine also logs per-request `tok/s`, `accepted=<n>/<m>` (draft acceptance), `ttft`, and `prefill` to `.tensorfold_run.log`.
