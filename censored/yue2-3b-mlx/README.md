# YuE2-3B — MLX music generation

Run [YuE2](https://huggingface.co/m-a-p/YuE2-3B) locally on Apple Silicon.
Lyrics + style in, 48 kHz stereo song out, with an editable ABC score.

This is **not** a Kilo chat model. There is no `/v1/chat/completions` loop.
The stack wraps [daig/yue2-mlx](https://github.com/daig/yue2-mlx) (`lyra`) around
the official generator and listening decoder.

Upstream: [HF YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) ·
[demo](https://map-yue2.github.io/) ·
[YuE repo](https://github.com/multimodal-art-projection/YuE)

**Port `:8088`** so it can sit beside Muse Glimmer (`:8087`), Gemma/Diffusion
(`:8080`), DeepSeek MLX (`:8082`), ds4 (`:8083`), and mtplx (`:8765` / `:8766`).

| | |
|--|--|
| **HF generator** | `m-a-p/YuE2-3B` (~3.6B, CC BY-NC 4.0) |
| **HF decoder** | `m-a-p/YuE2-Vae` (listening VAE; not the legacy benchmark decoder) |
| **Engine** | [yue2-mlx](https://github.com/daig/yue2-mlx) / `lyra` (MLX AR+NAR, PyTorch MPS VAE) |
| **Modalities** | **Text (style + lyrics) in, audio out** |
| **License** | Weights **CC BY-NC 4.0**. Engine code Apache-2.0. |
| **API** | `http://127.0.0.1:8088` (`POST /generate`, not chat completions) |
| **Model id** | `yue2-3b-mlx` |
| **Harness** | `test_harness.py` (`--gate` on post-start; `--offline` without a server) |

yue2-mlx is an experimental BF16 MVP. Full-song listening acceptance and some
strict AR numerical checks are still open upstream. It is independent of the
YuE team. Official CUDA `yue2-infer` is a different (NVIDIA) path and must not
share this environment.

## Quick start

```bash
cd censored/yue2-3b-mlx

# 1. Clone yue2-mlx, uv sync, download + convert YuE2-3B and YuE2-Vae
./1_setup_download.sh
# deps only (no weight pull):
#   ./1_setup_download.sh --deps-only

# 2. Short installation clip (~16s piano-pop, supplied score)
./2_generate.sh
# full City Pop example (several minutes):
#   ./2_generate.sh examples/full-song.json --output outputs/full-song

# 3. Optional HTTP API
./2_start_server.sh
# If port 8088 is stuck:  ./2_start_server.sh restart

# 4. Harness
python3 test_harness.py --offline
python3 test_harness.py --gate          # after the server is up
```

Listen with `open outputs/quickstart/audio.flac`.

## Architecture

```
lyrics + style JSON
      │
      ├─ ./2_generate.sh
      └─ POST http://127.0.0.1:8088/generate
              │
              ▼
         lyra (yue2-mlx)
              ├── AR / NAR: MLX BF16   (converted m-a-p/YuE2-3B)
              └── VAE: PyTorch MPS FP32 (m-a-p/YuE2-Vae)
              │
              ▼
         outputs/<id>/audio.flac + score.abc + result.json
```

`cot=full` writes melody+chords then audio. `melody` plans melody only.
`off` skips the score. Supplying `abc` uses that composition; do not edit a
saved plan in place (copy `score.abc` and pass `--abc`).

One request at a time. Peak process footprint on a 32 GB M5 Air was ~12.6 GiB
for a ~3 minute song. The runtime enforces a sampled 16 GiB budget and prefers
AC power.

## Requirements

- Apple Silicon. Intel Macs are not supported by this runtime.
- **uv** and **Python 3.12** (`uv` can install 3.12).
- ~20 GB free disk for BF16 (download + converted copy + VAE).
- yue2-mlx documents **macOS 26.2+**. Older macOS may still clone; Metal
  behavior is unvalidated there.
- Do **not** set `PYTORCH_ENABLE_MPS_FALLBACK` or `PYTORCH_MPS_FAST_MATH`.
  Scripts export `MLX_ENABLE_TF32=0`.

## Files

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Clone pinned yue2-mlx, `uv sync --frozen`, `lyra prepare` |
| `2_generate.sh` | CLI generate / plan from JSON or `--style`/`--lyrics` |
| `2_start_server.sh` | HTTP API on `:8088`; post-start harness gate |
| `yue2_server.py` | stdlib HTTP wrapper around `YuE2Pipeline` |
| `test_harness.py` | Offline layout / CLI / live `/health` checks |
| `validate_model.py` | Generator + VAE shard check |
| `examples/quickstart.json` | ~16s English piano-pop (supplied ABC, 400 semantic tokens) |
| `examples/full-song.json` | Upstream City Pop demo (`今晚不眠`) |

Engine checkout lives in `engine/` (gitignored). Converted weights live in
`models/` (gitignored). `.yue2_config` is written by setup.

Pinned engine commit: `c0f0229df07daba14923627e9d78e084d82b7dc8`
(override with `YUE2_MLX_REF`).

## HTTP API

Not OpenAI chat. Useful endpoints:

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| GET | `/v1/models` | — |
| POST | `/generate` | song JSON (`style`, `lyrics`, `cot`, optional `seed`/`abc`) |
| POST | `/plan` | same; writes ABC only |

```bash
curl -s http://127.0.0.1:8088/health
curl -s http://127.0.0.1:8088/v1/models
curl -X POST http://127.0.0.1:8088/generate \
  -H 'Content-Type: application/json' \
  -d @examples/quickstart.json
```

A busy server returns **409**. Output dirs are timestamped so repeats do not
collide. Full songs can take 10–15 minutes; use a long client timeout.

## Generate options

```bash
./2_generate.sh examples/full-song.json --output outputs/city-pop
./2_generate.sh --plan examples/quickstart.json --output outputs/plan
./2_generate.sh request.json --abc edited.abc --mode full
./2_generate.sh --style "English jazz trio, 110 BPM" --lyrics $'[Verse]\nRain on brass'
./2_generate.sh --no-require-ac          # battery; still 16 GiB process cap
```

Copy `examples/full-song.json`, edit `style` and `lyrics`, pick a new `--output`.
Lyra refuses to write into a non-empty directory.

## Precision

| Arg | Meaning |
|-----|---------|
| `bf16` (default) | MVP path |
| `8bit` / `4bit` | Experimental AR quant; NAR/VAE stay BF16; extra files, not a 4× smaller install |

```bash
./1_setup_download.sh 8bit
./2_generate.sh --precision 8bit
```

## What this stack does not do

- Kilo / OpenAI tool loops. Point Kilo at Qwen, Muse Glimmer, or ds4 and call
  `./2_generate.sh` from bash if an agent needs a song.
- Official NVIDIA `yue2-infer` (do not mix into this venv).
- SheetSage2 / MERT2 transcription or source-audio covers (supply ABC yourself).
- Waveform inpainting. Editing style, lyrics, or ABC regenerates the recording.
- Best-of-N selection used in the WildSongBench tables.

## License

Weights (`YuE2-3B`, `YuE2-Vae`) are **CC BY-NC 4.0**. Do not assume commercial
rights. yue2-mlx code is Apache-2.0; vendored YuE code keeps its notices.
