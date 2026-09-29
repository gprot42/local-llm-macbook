# Gemma 4 26B-A4B (heretic) — TensorFold

Run the **uncensored/heretic Gemma 4 26B-A4B** MoE on Apple Silicon with
[TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.3 (family `gemma4`).
Text output. This is the **MoE** checkpoint (26B total / 4B active); TensorFold's
Gemma 4 kernels cover the MoE layout only and **refuse the 31B dense** packs, so
the dense `gemma-4-31B-it` MLX stacks stay on their own servers.

**Port `:8092`** (engine `:8102`). Do not bind `:8080` (Gemma / Diffusion) or the
other model ports in this repo.

| | Hugging Face | Size |
|--|--|--|
| Model | `mlx-community/gemma-4-26b-a4b-it-4bit` (base, censored) | ~15 GB |
| Drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` | ~0.8 GB |

The pack is `model_type: gemma4`, `enable_moe_block: true` (128 experts, top-8),
MLX 4-bit in groups of 64 with an 8-bit router — exactly what the `gemma4` lane
requires. Gemma 4 has **no built-in MTP head**, so fast-tier speed comes from the
external DFlash drafter (`z-lab/gemma-4-26B-A4B-it-DFlash`), verified exact
against the engine's own sample. Without a drafter the lane decodes one token per
round (`--no-drafts`).

Fits a 128 GB M5 Max with room to spare (~15 GB weights + a 0.4B drafter). Do not
run it next to another 27B/large stack — one large model at a time.

## Quick start

```bash
./1_setup_download.sh
./2_start_tensorfold.sh
# stuck port:  ./2_start_tensorfold.sh restart

python3 test_harness.py --gate
```

OpenCode, after the server is up:

```bash
./install-opencode-json.sh --force
```

That merges the `gemma4-tensorfold` provider into
`~/.config/opencode/opencode.json` and sets the default model to
`gemma4-tensorfold/gemma-4-26b-a4b-heretic-tensorfold`. Other providers stay.
Restart OpenCode if it is already open.

Kilo id: `tensorfold-gemma4/gemma-4-26b-a4b-heretic-tensorfold`.

## Architecture

```
Kilo / OpenCode
    │  http://127.0.0.1:8092/v1
    ▼
loop_proxy.py            (repeat guard, :8092)
    │  http://127.0.0.1:8102/v1
    ▼
tensorfold serve         (gemma4 lane, DFlash drafter then exact verify, :8102)
    ▼
mlx-community/gemma-4-26b-a4b-it-4bit
```

`./1_setup_download.sh --no-drafter` skips the DFlash drafter (one token per round).

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv, TensorFold `v0.3.6.3`, pull weights, write `.tensorfold_config` |
| `2_start_tensorfold.sh` | Engine on `:8102`, loop proxy on `:8092`. Harness gate on by default |
| `loop_proxy.py` | Repeat guard in front of the engine (identical to the sibling stacks) |
| `test_harness.py` | Live API checks (tools, stream, multi-turn) |
| `kilo.json` | Provider `tensorfold-gemma4` |
| `opencode.json` | OpenCode provider fragment `gemma4-tensorfold` |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and context

Server defaults are Gemma's: `temperature=1.0`, `top_p=0.95`, `top_k=64`
(`2_start_tensorfold.sh`). Thinking is **off** (`--no-thinking`); pass
`--thinking` to open a think block for every request.

Context cap defaults to 131072 (server `-c`). The client `limit` in
`kilo.json` / `opencode.json` is 49152 / output 16384 (usable ~33k) — the same
conservative window used across these stacks; raise it if you need more, keeping
`context + output ≤` the server `-c`.

Pinned to **TensorFold 0.3.6.3** (`TF_VERSION` in `1_setup_download.sh`) — the
release with the `gemma4` MoE lane. Bump only after re-checking the model loads
and drafted replies stay byte-identical to `"draft": false`.

## Harness

`loop_proxy.py` owns `:8092` and forwards to the engine on `:8102`. Same two
tiers as the sibling stacks (nudge on a repeat, stop on the 4th identical result
/ 6th issuance / 200 calls) and the same client-disconnect handling.
`python3 loop_proxy.py --self-test` covers the rules.

## Troubleshooting

**`tensorfold info` rejects the repo** — the installed CLI is older than 0.3.6.3,
or the pack is not the MoE checkpoint (dense Gemma 4 is refused). Re-run
`./1_setup_download.sh`.

**Lane refuses the checkpoint** — the `gemma4` kernels need `enable_moe_block`
in every layer and **4-bit MLP** projections with only the **router at 8-bit**.
Many uncensored packs (`mlx-community/…-heretic-4bit`, `supergemma…-multimodal`)
quantise the MLP at 8-bit and are refused — verify a pack's `config.json`
quantization block before use. `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
has the correct layout; the lane's reference `mlx-community/gemma-4-26b-a4b-it-4bit`
is the censored fallback.

**Drafter does not load** — the DFlash drafter is optional; start with
`--no-drafts` to serve without it (slower, one token per round) and report the
loader error.
