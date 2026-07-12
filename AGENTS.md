# Agent rules — local-llm-macbook

Local models (especially DeepSeek V4 Flash via **ds4**) are strong at coding but **expensive on context**. Prefill and multi-`read` rounds dominate wall time.

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

Stack-specific detail: [`censored/deepseek-v4-flash-ds4/AGENTS.md`](censored/deepseek-v4-flash-ds4/AGENTS.md).
