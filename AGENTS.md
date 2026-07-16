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

Stack-specific detail: [`censored/deepseek-v4-flash-ds4/AGENTS.md`](censored/deepseek-v4-flash-ds4/AGENTS.md).
