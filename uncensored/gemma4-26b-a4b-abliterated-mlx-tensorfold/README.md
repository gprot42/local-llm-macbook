# Gemma 4 26B-A4B (abliterated, uncensored) — TensorFold

Run an **uncensored** Gemma 4 26B-A4B MoE on Apple Silicon with
[TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.3 (family `gemma4`).
Text. The weights are a **locally self-quantised** 4-bit pack (not on Hugging
Face) — see *Reproduce* below. Censored sibling: [`../../censored/gemma4-26b-a4b-mlx-tensorfold/`](../../censored/gemma4-26b-a4b-mlx-tensorfold/).

**Port `:8094`** (engine `:8104`). Do not bind `:8080` or the other model ports.

| | Source | Size |
|--|--|--|
| Model | **local** `~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit` | ~13 GB |
| Drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` | ~0.8 GB |

The pack was quantised from `SevenOfNine/Gemma-4-26B-A4B-It-Abliterated` (a
Heretic-abliterated Gemma 4, bf16) and given the **base pack's standard Gemma
template** so the reasoning channel stays gated — verified clean (no `<|channel>`
leak) **and** uncensored on the `gemma4` lane. Fits a 128 GB M5 Max easily.

## Reproduce the local pack

The 4-bit weights aren't published; recreate them with:

```bash
# 1) download the abliterated bf16 source (~52 GB, transient)
python -c "from huggingface_hub import snapshot_download; snapshot_download('SevenOfNine/Gemma-4-26B-A4B-It-Abliterated')"

# 2) quantise to 4-bit / group 64 (mlx_lm has gemma4 support)
mlx_lm.convert --hf-path SevenOfNine/Gemma-4-26B-A4B-It-Abliterated \
  -q --q-bits 4 --q-group-size 64 \
  --mlx-path ~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit

# 3) swap in the base pack's gated template (SevenOfNine's forces the channel)
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('mlx-community/gemma-4-26b-a4b-it-4bit'))")
cp "$BASE"/chat_template.jinja "$BASE"/tokenizer_config.json \
   ~/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit/
```

The bf16 source (~52 GB) can be deleted after step 2. To make this shareable,
upload the 4-bit dir to an HF repo and point `HF_MODEL` (in `.tensorfold_config`)
at it instead of the local path.

## Quick start

```bash
./1_setup_download.sh          # venv only; model is already local
./2_start_tensorfold.sh
python3 test_harness.py --gate
```

OpenCode: `./install-opencode-json.sh --force` merges the
`gemma4-abliterated-tensorfold` provider and sets it default. Kilo id:
`tensorfold-gemma4-abliterated/gemma-4-26b-a4b-abliterated-tensorfold`.

## Notes

- **Why local:** every ready-made uncensored 4-bit Gemma-4-26B-A4B pack was
  unusable on the lane — `…-heretic-4bit`/`supergemma…-multimodal` use 8-bit MLP
  (refused), and `supergemma…-uncensored-v2` / `SevenOfNine` ship a reasoning
  **channel** template that leaks `<|channel>` markers. Self-quantising +
  swapping the gated template is the clean path.
- **Drafter:** the base-model DFlash drafter works against the abliterated
  target (drafts are exact-verified). If acceptance is poor, `--no-drafts`.
- Model dir is git-ignored (`model/` and the cache path); the *recipe* above is
  the source of truth. Pinned to TensorFold 0.3.6.3.
