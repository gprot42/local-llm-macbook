## Prompt cache: a history boundary for prompts that continue the model's own turn

### Problem

`PromptBlocks.render` finds the history boundary by rendering the conversation twice: once with
`add_generation_prompt=False` and once with the generation prompt, then taking the shared prefix.
When the template adds **no** generation suffix, both renders are identical and the
`len(history) < len(prompt)` test fails, so `history_len` is 0.

Gemma 4 does exactly that after a tool result: the reply continues the model's own turn, so
there is nothing to append. That is the most common prompt shape in an agentic loop (call a tool,
read the result, continue).

With `history_len == 0`, `choose_checkpoints` places no checkpoint near the end of the prompt. The
next step can only resume from the last turn start, so it re-reads everything after it. With
thinking on it's worse. The reply's thought block isn't in the next prompt, so the finished-reply
cache doesn't match either, and only the system block is reused.

### Fix

When the render found no boundary and the history render *is* the whole prompt, set the boundary one
token short of the end:

```python
if not history_len and len(prompt) > 1 and history == prompt:
    history_len = len(prompt) - 1
```

The prompt is all history, so any prefix of it is a valid boundary. Stopping one token short keeps
`history_len < len(prompt)` as before, so there is still a token left to prefill. This leaves
prompts with a generation suffix (every prompt ending in a user message) unchanged.

### Tests

`tests/test_history_after_tool.py` uses a stub tokenizer that, like Gemma 4, adds a generation
suffix only after a user message:

- A prompt ending in a user message keeps its old boundary (regression guard).
- A prompt ending in a tool result gets `history_len == len(prompt) - 1`, and `choose_checkpoints`
  returns a checkpoint there.

The second test fails on v0.6.0 and passes with the fix. The full suite passes on an M5 Max
(3,815 passed). The only failure is
`test_glm5_prompt_kernels::test_fused_index_scores_are_the_three_ops`, which also fails on
unmodified v0.6.0 on this machine and is unrelated.

### Measured effect

Six-step agentic session on the gemma4 lane (gemma-4-26b-a4b, DFlash drafter), OpenCode-style tool
loop. Prompt reuse on steps 2–6:

| | v0.6.0 | with fix |
|---|---|---|
| thinking off | 54% | 78% |
| thinking on  | 54% | 78% |

With thinking on, this was the difference between each step re-reading most of the conversation
and resuming near its end.

### Scope

- One file in the engine (`server/prompt_blocks.py`, +6 lines) and one new test file.
- It only applies when the two renders are identical, which means the template adds no
  generation suffix. Other templates take the existing path.
