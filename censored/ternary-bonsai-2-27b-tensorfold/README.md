# Ternary Bonsai 2 27B — TensorFold

Run **Ternary Bonsai 2 27B** on Apple Silicon with [TensorFold](https://github.com/ashhart/TensorFold)
0.3.6.3 (family `bonsai`). Same model as
[`../ternary-bonsai-2-27b-gguf-llamacpp/`](../ternary-bonsai-2-27b-gguf-llamacpp/),
different weights and engine. This API is **text only**. Image input stays on
the GGUF sibling (`:8089`).

**Port `:8091`** (engine `:8101`). Do not bind `:8080` (Gemma / Diffusion).

| | Hugging Face | Size |
|--|--|--|
| Model | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | ~8.5 GB |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` | ~3.8 GB |

The pack is `model_type: prism_hadamard_qwen35`: ternary weights in a rotated basis, 2-bit groups of 128. TensorFold applies the version-1 Prism Hadamard contract and verifies drafts against its own serial sample. The vision tower in the pack is not loaded. Upstream, on an M5 Max, this path decoded 3.7–5.8× the pack's own mlx_lm runtime on code and 1.7–2.2× on chat (TensorFold 0.3.6.3 notes, not re-measured here).

## Quick start

```bash
./1_setup_download.sh
./2_start_tensorfold.sh
# stuck port:  ./2_start_tensorfold.sh restart

python3 test_harness.py --gate
```

Kilo, from this directory (or after the provider is in the root `kilo.json` you installed):

```text
tensorfold-bonsai/ternary-bonsai-2-27b-tensorfold
```

OpenCode, after the server is up:

```bash
./install-opencode-json.sh --force
```

That merges the `bonsai-tensorfold` provider into `~/.config/opencode/opencode.json`, sets the default model to `bonsai-tensorfold/ternary-bonsai-2-27b-tensorfold`, and copies the build-agent prompt. The GGUF `bonsai` provider and the Qwen `tensorfold` provider stay. Restart OpenCode if it is already open.

The id OpenCode sends is `ternary-bonsai-2-27b-tensorfold`, which is the id this server advertises. `temperature: false` leaves sampling at the server defaults (0.7 / 0.8 / top-k 20). Thinking stays off unless you pick the `-think` model or restart with `--thinking`.

## Architecture

```
Kilo / OpenCode
    │  http://127.0.0.1:8091/v1
    ▼
loop_proxy.py
    │  http://127.0.0.1:8101/v1
    ▼
tensorfold serve
    │  DFlash2 drafter, then exact verify
    ▼
prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
```

`./1_setup_download.sh --no-drafter` skips DFlash2 and the server decodes one token per round.

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv, TensorFold `v0.3.6.3`, pull weights, write `.tensorfold_config` |
| `2_start_tensorfold.sh` | Engine on `:8101`, loop proxy on `:8091`. Harness gate on by default |
| `loop_proxy.py` | Repeat guard in front of the engine |
| `test_harness.py` | Live API checks (tools, stream, multi-turn) |
| `kilo.json` | Provider `tensorfold-bonsai` |
| `opencode.json` | OpenCode provider fragment, including a per-request think model |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and thinking

Server defaults are the PrismML **non-thinking** preset: `temperature=0.7`, `top_p=0.8`, `top_k=20`. TensorFold's CLI has no presence-penalty flag, so the GGUF stack's `presence 1.5` is not applied here. Repetition in tool loops is handled by `loop_proxy.py`.

Thinking is **off** (`--no-thinking`). Two ways to turn it on:

```bash
./2_start_tensorfold.sh --thinking          # every request
```

Or, in OpenCode, `/models` → `bonsai-tensorfold/ternary-bonsai-2-27b-tensorfold-think`. That entry is an alias of the same server. Its options set `enable_thinking`, `reasoning_effort: medium`, `thinking_budget: 1024`, and the thinking preset `1.0 / 0.95 / top-k 20`. No restart. Sent-back reasoning still grows the tool loop, so use it for a hard turn, not an hour-long build.

Context cap is 262144 tokens (prompt + reply), the checkpoint maximum. Cap it lower with:

```bash
./2_start_tensorfold.sh --context 131072
```

Pinned to **TensorFold 0.3.6.3** (`TF_VERSION` in `1_setup_download.sh`). That is the release that added this pack. Bump the pin only after re-checking that drafted replies stay byte-identical to `"draft": false`.

## Harness

`loop_proxy.py` owns `:8091` and forwards to the engine on `:8101`. Same two tiers as the GGUF sibling and the Qwen TensorFold stack. **Nudge (turn continues):** a call that just returned the same result a second or third time, with no `write`/`edit` in between, or that was just issued a third time, is replaced with a `[Harness] REFUSED …` error that quotes the original output, and from the third repeat a user-role directive is added, then the request is forwarded. **Stop (turn ends, model not called):** the fourth identical result, the sixth issuance, or 200 tool calls. A new user message resets the counts. `python3 loop_proxy.py --self-test` covers the rules.

`./2_start_tensorfold.sh` runs `test_harness.py --gate` after ready. Skip with `--no-harness-gate`. A soft tool-call miss does not stop the server.

## Options

```bash
./1_setup_download.sh --deps-only
./1_setup_download.sh --no-drafter
./2_start_tensorfold.sh --port 8092          # engine follows at :8102; also edit kilo.json baseURL
./2_start_tensorfold.sh --no-drafts
./2_start_tensorfold.sh stop
./2_start_tensorfold.sh status
```

## Comparison

| | `ternary-bonsai-2-27b-gguf-llamacpp` | this folder |
|--|--|--|
| Engine | PrismML llama.cpp fork | TensorFold 0.3.6.3, DFlash2 |
| Weights | `PQ2_0` GGUF + mmproj | MLX 2-bit pack |
| Images | yes | no |
| Port | 8089 (engine 8099) | 8091 (engine 8101) |
| Kilo id | `bonsai/ternary-bonsai-2-27b` | `tensorfold-bonsai/ternary-bonsai-2-27b-tensorfold` |

Do not load both at once on 128 GB. Do not load this next to Qwen 3.8 TensorFold (`:8769`) either — same drafter, two full 27B weights.

## Troubleshooting

**`tensorfold info` rejects the repo**

The installed CLI is older than 0.3.6.3, or the pack is not the version-1 Prism contract (blocks of 1024, explicit signs, grouped DeltaNet, no MTP head). Re-run `./1_setup_download.sh`.

**Setup downloaded the CLI but not the weights**

Re-run `./1_setup_download.sh`. `tensorfold pull` resumes a partial Hugging Face cache (`~/.cache/huggingface`).

**Harness gate soft-fails the tool call**

The server is up. The model answered in prose instead of a tool call. Re-run `python3 test_harness.py --gate`. Soft fails leave the server running.
