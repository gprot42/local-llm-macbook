Follow-up to #157, as asked there: the thought-channel half on its own, on 0.6.0 (the bare `:NAME{...}` tool-call half is in 0.6.0 already, #121).

### The bug

With thinking off, Gemma 4 still opens a thought channel on some replies, usually the continuation after a tool result, and the block is often empty. That reply takes the `else` branch in `ChatApp.chat` and goes through `parse_harmony_output`, which only knows gpt-oss's `<|channel|>` markers, so Gemma's `<|channel>thought\n<channel|>` lands verbatim in `content`, streamed and not. On v0.6.0, the new end-to-end test below reproduces it through `ChatApp` and the HTTP handler: `content` comes back as `'<|channel>thought\n<channel|>The files are a.py and b.py.'`.

### The change (two commits)

1. `server/app.py`: route a reply through `split_thinking` with the channel markers whenever the tokenizer uses them (`self.think_markers == CHANNEL_MARKERS`), not only when thinking is on. Streaming and the finished reply both.
2. `server/text.py`: `split_thinking` finds the channel opener wherever it is. The block does not always sit at position 0: live, it also follows visible text, a leading newline, or a stray or doubled `<|channel>` opener. Text before it is answer, a stray partial opener is dropped, and the block is stripped (recursively for the remainder). Streaming still holds back a partial opening tag, so the visible answer only grows.

### gpt-oss / Harmony is unchanged

The new condition is true only for a tokenizer that has Gemma's `<channel|>` token (`think_markers()` returns `CHANNEL_MARKERS` only then). A gpt-oss tokenizer does not have it, so with thinking off its replies still take `parse_harmony_output`, exactly as before.

### Tests

New `tests/test_thinking_off_channels.py` runs scripted replies through `ChatApp` and the HTTP handler, thinking off, streamed and not:

| test | v0.6.0 | this branch |
|---|---|---|
| Gemma: a spontaneous empty thought block never reaches `content` | fails (the markup is in `content`) | passes |
| Harmony: `<|channel|>analysis…<|end|><|start|>assistant<|channel|>final<|message|>Hello there.` gives content `Hello there.`, reasoning `Think it over.`, streamed as `('Hello there.', 'Think it over.')` | passes | passes, same output |
| a plain reply is unchanged for both tokenizers | passes | passes |

The Harmony expectations were measured on v0.6.0 before the change and are pinned in the test. `tests/test_lane_stream_text.py` adds the text-level cases: channel after text, doubled opener, leading newline, and monotonic streaming.

Full suite on this branch (Apple M5 Max, Python 3.12, MLX 0.32.3): **3,788 passed, 488 skipped, 1 failed**. The one failure, `tests/test_glm5_prompt_kernels.py::test_fused_index_scores_are_the_three_ops`, fails identically on stock v0.6.0 on this machine (`index_fits()` returns False for the GLM-5.3 fused index-score kernel here). It is unrelated to this change.
