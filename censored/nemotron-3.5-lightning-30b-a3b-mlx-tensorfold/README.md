# Nemotron 3.5 Lightning 30B-A3B — TensorFold

Run **NVIDIA Nemotron 3.5 Lightning** (30B-total / 3B-active MoE) on Apple
Silicon with [TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.3
(family `nemotron_h`). Text. Hybrid Mamba/attention MoE with a **built-in MTP
head**, so it **self-drafts** — there is no external drafter to download.

**Port `:8093`** (engine `:8103`). Do not bind `:8080` (Gemma / Diffusion) or
the other model ports in this repo.

| | Hugging Face | Size |
|--|--|--|
| Model | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | ~17 GB |
| Drafter | none — self-drafts via the built-in MTP head (`mtp-4bit.safetensors`) | — |

Verified against the lane with `tensorfold info`: `model_type nemotron_h`,
52 layers, 128 routed experts (6 active/token), MLX 4-bit groups of 64, 256K
context. The `nemotron_h` family requires the MTP head file in the pack
(`mtp-4bit.safetensors`); with it, drafting is on by default and verified exact
against the engine's own sample. `--no-drafts` forces serial decode.

Fits a 128 GB M5 Max easily (~17 GB weights, no separate drafter). One large
model at a time.

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

That merges the `nemotron-tensorfold` provider into
`~/.config/opencode/opencode.json` and sets the default model to
`nemotron-tensorfold/nemotron-3.5-lightning-30b-tensorfold`. Other providers stay.
Restart OpenCode if it is already open.

Kilo id: `tensorfold-nemotron/nemotron-3.5-lightning-30b-tensorfold`.

## Architecture

```
Kilo / OpenCode
    │  http://127.0.0.1:8093/v1
    ▼
loop_proxy.py            (repeat guard, :8093)
    │  http://127.0.0.1:8103/v1
    ▼
tensorfold serve         (nemotron_h lane; MTP self-draft then exact verify, :8103)
    ▼
Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

`./2_start_tensorfold.sh --no-drafts` disables the MTP self-draft (serial decode).

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | venv, TensorFold `v0.3.6.3`, pull weights (incl. MTP head), write `.tensorfold_config` |
| `2_start_tensorfold.sh` | Engine on `:8103`, loop proxy on `:8093`. Harness gate on by default |
| `loop_proxy.py` | Repeat guard in front of the engine (identical to the sibling stacks) |
| `test_harness.py` | Live API checks (tools, stream, multi-turn) |
| `kilo.json` | Provider `tensorfold-nemotron` |
| `opencode.json` | OpenCode provider fragment `nemotron-tensorfold` |
| `install-opencode-json.sh` | Merge that fragment into `~/.config/opencode/opencode.json` |

## Sampling and context

Server defaults `temperature=0.6`, `top_p=0.95`, `top_k=20`
(`2_start_tensorfold.sh`). Thinking is **off** (`--no-thinking`); pass
`--thinking` to open a think block.

Context cap defaults to 262144 (the model's max). The client `limit` in
`kilo.json` / `opencode.json` is 49152 / output 16384 (usable ~33k) — the same
conservative window used across these stacks; raise it if you need more, keeping
`context + output ≤` the server `-c`.

Pinned to **TensorFold 0.3.6.3** (`TF_VERSION` in `1_setup_download.sh`) — the
release with the `nemotron_h` lane. Bump only after re-checking the model loads
(MTP head present) and drafted replies stay byte-identical to `"draft": false`.

## Harness

`loop_proxy.py` owns `:8093` and forwards to the engine on `:8103`. Same two
tiers as the sibling stacks (nudge on a repeat, stop on the 4th identical result
/ 6th issuance / 200 calls) and the same client-disconnect handling.
`python3 loop_proxy.py --self-test` covers the rules.

## Troubleshooting

**`tensorfold info` rejects the repo** — the installed CLI is older than 0.3.6.3,
or the pack is not `model_type nemotron_h`. Re-run `./1_setup_download.sh`.

**No speedup / decodes one token per round** — the pack is missing its MTP head
(`mtp-4bit.safetensors`); the lane needs it to self-draft. Use the tested
`Vontra/…-MLX-4bit` pack, which ships it, or accept serial decode.

**Censored** — this is the NVIDIA base model. An uncensored variant would need
the same checks: `nemotron_h` arch, MLX 4-bit the lane reads, and the MTP head
present for fast-tier.
