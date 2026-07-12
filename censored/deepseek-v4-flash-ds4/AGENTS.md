# Agent rules — DeepSeek V4 Flash (ds4 / native Metal)

Local Flash is **great for coding** but **expensive on context**. Prefill and multi-`read` rounds dominate wall time. Follow this so agent sessions stay fast and under the context limit.

## Tool use

1. **Cap parallel reads** — at most **3–4** file opens per turn (never 8–10).
2. **Discover with glob/grep first** — then open only files needed for the next edit or answer.
3. **Do not re-read** files already in the conversation.
4. **Ignore noise** — `node_modules/`, `dist/`, `build/`, lockfiles, minified bundles, large generated assets.
5. **Act early** — once you can edit or answer, stop exploring.

## Verify instead of re-exploring

- Prefer the project’s typecheck/build/tests (`tsc`, `npm test`, `npm run build`) over another wave of blind reads.
- Fix from compiler/test output; only open files named by the failure.

## Session hygiene

- A deep review that hits **~50–60%** of the context window is done — summarize and **start a new chat** for the next feature.
- Do not paste huge file bodies into chat replies; short summaries and targeted diffs only.

## Sampling

Use DeepSeek defaults from `kilo.json`: `temperature=1.0`, `top_p=1.0`. Do not “optimize” sampling for coding quality.
