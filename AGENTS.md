# Agent rules — local-llm-macbook

Local models (especially DeepSeek V4 Flash via **ds4**) are strong at coding but **expensive on context**.

For ds4, always start the stack with `./2_start_ds4.sh` so **`ds4_kilo_proxy`** is on `:8083` (thinking **OFF** by default). Raw ds4 thinking mode truncates tool JSON and aborts mid-fix. Prefill and multi-`read` rounds dominate wall time.

## Tool use

1. Cap parallel file reads at **3–4 per turn** (never 8–10).
2. Discover with **glob/grep** first; open only files needed for the next edit.
3. Do **not re-read** files already in the conversation.
4. Skip `node_modules/`, `dist/`, lockfiles, minified/generated assets.
5. Once you can edit or answer, **stop exploring**.

## Verify early

Run the project’s typecheck/build/tests and fix from those errors instead of another explore wave.

## Sessions

After a large review (~50–60% of the context bar), summarize and **start a new chat** for the next feature.

## Continue means act

When the user says **continue** / **keep going** / **continue if you have next steps**:

1. Run tools for the next unfinished step **immediately** (first response should prefer a tool call).
2. Take next steps from this chat, a summary the user pasted, or the last action you promised (e.g. “check that directory”).
3. If a path/directory was named → `list`/`read` it **now**. Do not only say you will.
4. Do **not** rewrite Goal/Progress/Next Steps templates. Do **not** invent a new feature.
5. Only ask what to do if there is truly no task in context.

## Multi-step completion

If the user asks for **N steps**, a checklist, or “do A then B then C”:

1. Complete **all** steps before stopping (not 1 of N).
2. After each tool result, call the **next** tool immediately.
3. Do not idle after a partial plan or a single grep/list.

## Empty tool results

If the latest tool output is **empty** / “(no output)” / useless remote HTML grep:

1. **Do not** write a revised plan or Goal/Progress template.
2. **Do** run a **local** tool next (`ls`/`glob`/`grep`/`read`) on a real workspace path.
3. If `FileNotFound`, list the **parent** directory — do not invent a new research plan.
4. Avoid another `curl | grep` of remote pages unless the user only asked for web docs.

## Conclude decisively

Brevity caps limit **padding and exploration**, not the substance of the answer.

1. Every final reply ends with a **definitive conclusion**: the direct answer, the root cause, or the verified result of the edit.
2. Never end with a light acknowledgement, a restatement of the request, or a partial observation. State what the tools proved and what it means.
3. Do not hedge (“it might be…”, “you could check…”) when you can check yourself — check, then state the result. If truly uncertain: single best answer + the one command that confirms it.
4. Coding/debugging: finish with **what changed**, **how it was verified** (exact command + result), and remaining caveats. Do not stop before the verification result exists.

### Canonical prompt block

The fenced block below is the **single source of truth** for the wording shipped
inside every `agent.*.prompt` in every `kilo.json`. Edit it here, then run
`./sync_agent_prompts.py` to push it into all of them; `--check` fails if any
copy has drifted. Do not hand-edit the block inside a `kilo.json`.

Kilo's `instructions: ["AGENTS.md"]` resolves against the *opened project*, not
this repo, and two stacks ship no `AGENTS.md` at all — so the text has to be
inline in each prompt. This block is how it stays identical.

<!-- BEGIN kilo:conclude-decisively -->
```text
Conclude decisively (critical):
- Brevity rules limit padding and exploration, NOT the substance of your answer. Every final reply must contain a definitive conclusion: the direct answer, the root cause, or the verified result of your edit.
- Never end with a light acknowledgement, a restatement of the request, or a partial observation. If you ran tools, state what they proved and what it means for the user's question.
- Commit to an answer. Do not hedge with "it might be" / "you could check" when you have the tools to check; check, then state the result. If genuinely uncertain, give your single best answer and the one command that confirms it.
- Q&A: answer completely, as long as the answer genuinely requires (usually short); the line cap is a target, not a reason to omit the conclusion.
- Coding/debugging: finish with what changed, how it was verified (exact command + result), and any remaining caveat. Do not stop before the verification result exists.
```
<!-- END kilo:conclude-decisively -->

Stack-specific detail: [`censored/deepseek-v4-flash-ds4/AGENTS.md`](censored/deepseek-v4-flash-ds4/AGENTS.md).
