# Qwen3.8-27B — TensorFold server + harness

Run **Qwen3.8-27B** on Apple Silicon with [TensorFold](https://github.com/ashhart/TensorFold)
(Ash Hart, `@ashxhart`). Same coding job as
[`../archive/qwen3-8-27b-coder-mtplx/`](../archive/qwen3-8-27b-coder-mtplx/), different engine.

**Port `:8769`**. Qwen3.8 mtplx stays on `:8766`. Do not bind `:8080` (Gemma / Diffusion).

TensorFold's tested checkpoint for this family is MLX 4-bit, group size 64:

| | Hugging Face | Size |
|--|--|--|
| Model | `Vontra/Qwen3.8-27B-MLX-4bit` | 16.1 GB |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` | 3.8 GB |

`config.json` says `model_type: qwen3_5` and `Qwen3_5ForConditionalGeneration`. That is the Qwen3.8-27B dense checkpoint TensorFold ships kernels for, including a vision tower. The HTTP API does not accept images. This stack is text and tool calls.

On an M5 the lane kernels verify drafted tokens in one forward. On M1–M4, TensorFold uses its row-exact matvec instead. Drafted output is byte-identical to serial decoding either way. Upstream measurements on an M5 Max 128 GB (TensorFold README, not re-measured here): about 120–124 tok/s on a short answer with thinking, about 189 tok/s on code, about 27 tok/s with drafts off.

## Quick start

```bash
./1_setup_download.sh
./2_start_tensorfold.sh
# stuck port:  ./2_start_tensorfold.sh restart

python3 test_harness.py --gate
```

Kilo, from a project directory, after this server is up:

```text
tensorfold-qwen38/qwen3.8-27b-tensorfold
```

`kilo.json` in this folder points at `http://127.0.0.1:8769/v1`. This provider is not in the repo-root `kilo.json` catalog. Launch Kilo from this directory, or copy the provider block into your user config.

OpenCode, after the server is up:

```bash
./install-opencode-json.sh --force
```

That merges the `tensorfold` provider into `~/.config/opencode/opencode.json` and sets the default model to `tensorfold/qwen3.8-27b-tensorfold`. Other providers stay. Restart OpenCode if it is already open. The model id OpenCode sends is `qwen3.8-27b-tensorfold`, which is the id this server advertises. `temperature: false` leaves sampling at the server defaults (0.6 / 0.95 / top-k 20). Thinking stays off unless you restart the server with `--thinking`.

## Architecture

```
Kilo Code
    │  http://127.0.0.1:8769/v1
    ▼
tensorfold serve
    │  DFlash2 drafter (separate checkpoint), then exact verify
    ▼
Vontra/Qwen3.8-27B-MLX-4bit
```

The mtplx sibling drafts with MTP heads inside one checkpoint. This stack drafts with `z-lab/Qwen3.8-27B-DFlash2` once that repo is in the Hugging Face cache. `./1_setup_download.sh --no-drafter` skips it and the server decodes one token per round.

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv, `pip install` TensorFold from GitHub, pull weights, write `.tensorfold_config` |
| `2_start_tensorfold.sh` | Serve on `:8769`. Harness gate on by default |
| `test_harness.py` | Live API checks (tools, stream, multi-turn) |
| `kilo.json` | Provider `tensorfold-qwen38` |
| `opencode.json` | OpenCode provider fragment |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and thinking

Server defaults match the mtplx coder stack: `temperature=0.6`, `top_p=0.95`, `top_k=20`.

Thinking is **off** (`--no-thinking`) so tool loops are not wrapped in a think block. Turn it on with:

```bash
./2_start_tensorfold.sh --thinking
```

Context cap is 262144 tokens (prompt + reply), the checkpoint maximum. A 131072 cap rejected a 132953-token agent prompt. At a full 262144-token window the 16 attention layers need about 16 GiB of KV cache, which fits this 128 GB Mac on top of the ~20 GB of weights. Cap it lower with:

```bash
./2_start_tensorfold.sh --context 131072
```

## Harness

`loop_proxy.py` owns `:8769` and forwards to the engine on `:8779`, relaying streamed tokens as they arrive. After the latest user message it stops the turn, without calling the model, when the same tool call has returned the same result twice **with no `write`/`edit` in between** (an edit between two identical `node --check` runs is a normal verify loop and is forwarded), when the same call has been issued three times (so alternating commands still count), or when the turn has already run 200 tool calls (`--max-rounds` / `LOOP_MAX_ROUNDS`; a backstop only — a code-exploration turn of 48 distinct greps and reads was cut off when this was 48). A new user message starts that count over. OpenCode's own doom-loop check only sees repeated calls inside one assistant message, so one bash call per step never trips it — and a prompt rule alone did not hold: a session ran the same call 782 times before this proxy existed. `python3 loop_proxy.py --self-test` covers the repeat rules, the verify-loop exemption and streaming pass-through.

`test_harness.py` talks to the OpenAI API only.

| Mode | What it covers |
|------|----------------|
| `--gate` | Reachable, `/v1/models`, short chat, bash tool call, SSE stream, multi-turn tool result |
| default | Gate plus health, unicode, empty tool result, multi-step continue, concurrent chats |
| `--quick` | Skip multi-step and concurrent |
| `--strict` | Soft (model-behavior) failures become hard |

Exit codes: `0` pass, `1` hard fail, `2` unreachable.

`./2_start_tensorfold.sh` runs `--gate` after ready. Skip with `--no-harness-gate`. A soft tool-call miss does not stop the server.

## Options

```bash
./1_setup_download.sh --deps-only
./1_setup_download.sh --no-drafter
./2_start_tensorfold.sh --port 8770          # also edit kilo.json baseURL
./2_start_tensorfold.sh --no-drafts
./2_start_tensorfold.sh stop
./2_start_tensorfold.sh status
```

## Comparison

| | `qwen3-8-27b-coder-mtplx` | `qwen3-8-27b-coder-tensorfold` (this) |
|--|--|--|
| Engine | mtplx, MTP heads in the checkpoint | TensorFold, DFlash2 drafter |
| Default weights | `mlx-community/Qwen3.8-27B-4bit` when published | `Vontra/Qwen3.8-27B-MLX-4bit` |
| Port | 8766 | 8769 |
| Kilo id | `mtplx-qwen38/qwen3.8-27b-mtplx` | `tensorfold-qwen38/qwen3.8-27b-tensorfold` |

Do not load both at once on 128 GB.

## Troubleshooting

**`tensorfold info` rejects the repo**

The CLI only loads families it has kernels for, and the Qwen3.8 lane path wants 4-bit groups of 64. NVFP4, GPTQ, and AWQ are refused before download. Use the Vontra checkpoint above.

**Setup downloaded the CLI but not the weights**

Re-run `./1_setup_download.sh`. `tensorfold pull` resumes a partial Hugging Face cache (`~/.cache/huggingface`).

**Harness gate soft-fails the tool call**

The server is up. The model answered in prose instead of a tool call. Re-run `python3 test_harness.py --gate`. Soft fails leave the server running.

**Kilo is slow, curl is fast**

Prefill dominates on long histories. Drafts speed decode, not the first pass over a huge prompt. Compact the session.
