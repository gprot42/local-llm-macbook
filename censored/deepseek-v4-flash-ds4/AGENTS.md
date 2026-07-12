# Agent rules — DeepSeek V4 Flash (ds4 harness)

Local Flash is **great for coding** when the **harness** is correct. Raw ds4 defaults to **high-effort thinking**, which often:

- truncates tool-call JSON (`JSON Parse error: Expected '}'`)
- ends streams without a clean finish (`Response ended without a finish reason`)
- burns tokens on diagnosis and **never applies the fix**

`./2_start_ds4.sh` runs **`ds4_kilo_proxy`** on `:8083` (thinking **OFF** by default) in front of ds4-server on `:18083`.

## Finish the job

1. Diagnose **and implement**. Do not stop after explaining the bug.
2. Verify with the project’s typecheck/build/tests.
3. If a tool fails, retry with a simpler allowed tool — usually **`bash`**.

## Tool reliability

1. **Only** call tools present in the request schema. Never invent `background_process` or similar.
2. Prefer **`bash`** for shell: `cd /full/path && npm run dev -- --host 127.0.0.1 --port 5173`.
3. Keep arguments short. **Finish every JSON brace/quote** before ending the tool call.
4. Cap parallel file reads at **3–4** per turn; discover with glob/grep first.
5. Do not re-read files already in the conversation. Skip `node_modules/`, `dist/`, lockfiles.

## Session hygiene

- After a deep review (~50–60% of the context bar), summarize and **start a new chat**.
- Prefer short summaries over dumping large files into the reply.

## Sampling / thinking

- Defaults: `temperature=1.0`, `top_p=1.0`.
- Thinking is **disabled by the proxy** for agent reliability. To force thinking on a raw curl: `"think": true` or `"reasoning_effort": "high"`.
