# Qwen3.8-27B OBLITERATED — TensorFold server + harness

Uncensored **Qwen3.8-27B V3** ([OBLITERATUS](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED), weight-space refusal removal of `Qwen/Qwen3.8-27B`) on the native [TensorFold](https://github.com/ashhart/TensorFold) engine. Text and tool calls. The MLX pack has a vision tower; this server does not load it.

TensorFold 1.0+ is a **native Zig binary** installed with Homebrew — it is no longer pip-installable, and there is no venv. The loop proxy and harness run on the system `python3` (standard library only). The OBLITERATED checkpoint's own chat template renders no tools, so the stack ships a Qwen3 tool-capable template (`chat_template_tools.jinja`, XML tool format) and passes it with `--chat-template`; without it, tool calls never reach the model.

The archived mtplx stack stays at [`../archived/qwen3-8-27b-obliterated-mtplx/`](../archived/qwen3-8-27b-obliterated-mtplx/) (`:8768`). This folder does not copy the ~14 GB shards. `./1_setup_download.sh` symlinks `models/mlx-4bit` at the convert that already lives under the autosaddler tree.

**Port `:8096`** (loop proxy) → engine `:8106`. Do not bind `:8080`. Do not co-load the aligned Qwen 3.8 TensorFold server on `:8769`.

| | Where | Size |
|--|--|--|
| Model | local `models/mlx-4bit` (V3, affine 4-bit, group 64) | ~14 GB |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` | ~3.8 GB |

`config.json` is `model_type: qwen3_5`, 64 layers, hidden 5120, untied embeddings. That is the dense shape the Qwen3.8 lane kernels read. TensorFold drops the MTP sidecar. DFlash2 was trained on the aligned 27B, so draft acceptance on these shifted weights will be lower than on `Vontra/Qwen3.8-27B-MLX-4bit`. Rejected drafts fall back to the target token.

Status: 🟢 migrated to the **native TensorFold engine (v1.0.5)** on 2026-10-10 and verified live on an M5 Max, macOS 27.0.1. The checkpoint loads (`tensorfold info` recognises qwen3_5 4-bit group 64), greedy decode drives the DFlash2 drafter (84 code tokens in 7 forward rounds vs 84 plain; 73% draft acceptance), drafted output is **bit-for-bit identical** to `"draft": false`, and the harness gate passes 6/6 (including the bash tool call and multi-turn tool result). v1.0.5 is the first 1.0.x whose Qwen3.8-27B Metal kernels compile on macOS 27 (earlier 1.0.x crash at load with `LibraryLoad`). This server is greedy and the proxy stops repeated tool calls; that does not by itself make long agent sessions stable.

## Quick start

```bash
./1_setup_download.sh
./2_start_tensorfold.sh
# stuck port:  ./2_start_tensorfold.sh restart

python3 test_harness.py --gate
python3 loop_proxy.py --self-test
```

Kilo, from a project directory, after this server is up. This provider is not in the repo-root `kilo.json`. Launch Kilo with this folder's `kilo.json`, or copy the provider block.

```text
tensorfold-qwen38-obl/qwen3.8-27b-obliterated-tensorfold
```

OpenCode, after the server is up:

```bash
./install-opencode-json.sh --force
```

That merges provider `tensorfold-qwen38-obl` and does not replace the aligned `tensorfold` provider. `temperature: false` leaves sampling at the server default, which is greedy. Restart OpenCode if it is already open.

## Architecture

```
Kilo / OpenCode
    │  http://127.0.0.1:8096/v1
    ▼
loop_proxy.py
    │  http://127.0.0.1:8106/v1
    ▼
tensorfold serve  (DFlash2, exact verify)
    ▼
models/mlx-4bit   →  archived autosaddler V3 convert
```

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | brew-install native TensorFold (≥1.0.5), symlink the 4-bit weights, ensure the tool template, pull DFlash2 |
| `2_start_tensorfold.sh` | Engine `:8106`, proxy `:8096`, `--chat-template` for tools, harness gate after ready |
| `chat_template_tools.jinja` | Qwen3 tool-capable chat template (the checkpoint's own has no tool support) |
| `loop_proxy.py` | Repeat-tool nudge and stop. `python3 loop_proxy.py --self-test` |
| `test_harness.py` | Live API checks. `--gate` is the post-start set |
| `kilo.json` | Provider `tensorfold-qwen38-obl`, context 49152, temperature 0 |
| `opencode.json` | OpenCode provider fragment |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and context

Server: `--temperature 0 --top-p 1 --no-thinking`. The OBLITERATUS card wants greedy sampling. TensorFold has no `frequency_penalty` or `repetition_penalty` flag. The mtplx stack used `frequency_penalty` 0.3 as its loop brake. Here the loop proxy is that brake.

Thinking stays off so tool calls are not wrapped in a think block. The mtplx notes found tool turns with thinking fully off became a chain of greedy steps. Turn thinking on for a session with:

```bash
./2_start_tensorfold.sh --thinking
```

The engine context cap is 262144. Kilo and OpenCode in this folder cap context at **49152** and output at **8192**. On the mtplx sibling, decode fell to 2–5 tok/s past about 50k prompt tokens, and the catalog was set so compaction fired before that. Raise the client cap only if you mean to.

Uses the **native TensorFold binary**, minimum **v1.0.5** (`TF_MIN_VERSION` in `1_setup_download.sh`), installed with `brew install ashhart/tensorfold/tensorfold`. v1.0.5 serves this affine 4-bit group-64 checkpoint with DFlash2, keeps per-conversation prompt states (`--prompt-cache-gib`, default 16), and is the first 1.0.x whose Qwen3.8-27B kernels compile on macOS 27. `2_start_tensorfold.sh` passes `--no-thinking` by default (add `--thinking` for reasoning-effort medium). Upgrade the engine with `brew upgrade tensorfold`.

## Harness

`loop_proxy.py` owns `:8096` and forwards to `:8106`. It relays streamed tokens as they arrive.

**Nudge** (the turn continues): the same call just returned the same result a 2nd or 3rd time with no `write`/`edit` in between, or the same call was issued a 3rd time. The proxy replaces that result with a `[Harness] REFUSED` error that quotes the original output, and from the 3rd repeat also adds a user-role directive.

**Stop** (the turn ends, the model is not called): the identical result comes back a 4th time, the same call is issued a 6th time, or the turn passes 200 tool calls (`--max-rounds` / `LOOP_MAX_ROUNDS`). A new user message resets the counts.

`test_harness.py` talks to the OpenAI API only. Requests use temperature 0.

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
./2_start_tensorfold.sh --port 8096
./2_start_tensorfold.sh --no-drafts
./2_start_tensorfold.sh --context 49152
./2_start_tensorfold.sh --prompt-cache-gib 24
./2_start_tensorfold.sh --thinking
./2_start_tensorfold.sh --no-tool-template     # serve without tool rendering
./2_start_tensorfold.sh stop
./2_start_tensorfold.sh status
```

Do not `mtplx pull OBLITERATUS/Qwen3.8-27B-OBLITERATED`. If the symlink target is missing, build the quant from the archived stack and re-run setup:

```bash
../archived/qwen3-8-27b-obliterated-mtplx/1_setup_download.sh
./1_setup_download.sh
```
