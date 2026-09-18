# Ternary Bonsai 2 27B (GGUF · llama.cpp fork)

Local **Ternary Bonsai 2 27B** ([prism-ml](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)) — a Hadamard‑rotated **ternary** (~1.72 bit/weight) quant of **Qwen3.8 27B**, ~9× smaller than FP16 while retaining ~98% of aggregate benchmark performance. Text **+ image**, native tool calling, thinking model. Served as an OpenAI API for **Kilo / OpenCode**.

| | |
|--|--|
| **API** | `http://127.0.0.1:8089/v1` |
| **Kilo model ID** | `bonsai/ternary-bonsai-2-27b` |
| **Weights** | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · `PQ2_0` (7.21 GB) + `mmproj-Q8_0` (0.63 GB) |
| **Engine** | PrismML's [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (Metal), `--jinja` tool calling |

## Why the fork (not stock MLX/Ollama)

Bonsai 2 packs are stored in a **rotated basis**: each weight matrix is transformed by a blockwise Hadamard rotation before ternary assignment, and the runtime must apply the matching transform to activations. The pack declares `model_type: prism_hadamard_qwen35` / `requires_runtime: true`.

- **MLX packs cannot be served.** `mlx_lm.server` / `mlx_vlm.server` don't use the bundled loader; PrismML's own `start_mlx_server.sh` **refuses Bonsai 2**, warning it "would return wrong output with no error." (The MLX build is one‑shot generation only.)
- **The `PQ2_0` (group‑128) GGUF needs the fork's kernels** — stock llama.cpp / Ollama lack them.

So this stack builds PrismML's llama.cpp fork from source and serves the GGUF. Source of truth: [PrismML‑Eng/Bonsai‑demo](https://github.com/PrismML-Eng/Bonsai-demo).

## Quick start

```bash
cd censored/ternary-bonsai-2-27b-gguf-llamacpp

# 1. Build the fork (Metal) + download PQ2_0 GGUF + mmproj  (~5 min build, ~7.9 GB)
./1_setup_download.sh

# 2. Serve on :8089  (OpenAI-compatible, --jinja tool calling, image input)
./2_start_llama.sh
#    status / stop:  ./2_start_llama.sh status | ./2_start_llama.sh stop

# 3. Kilo / OpenCode — provider `bonsai`, model `bonsai/ternary-bonsai-2-27b`, API http://127.0.0.1:8089/v1
```

## Notes

- **Context** defaults to `-c 65536` (override `BONSAI_CTX`). The model supports far more; Kilo's `limit.context` is capped at 49152 (peak 57344 < 65536) to compact before overflow. Raise both together for more.
- **Sampling** starts at the Bonsai 2 base defaults (temp 1.0 / top‑p 0.95 / top‑k 20); Kilo overrides per agent in `kilo.json`.
- **Reasoning** is **off by default** (`--reasoning off`): the 27B is a thinking model, but under agentic use it exhausts its output budget thinking and never emits the tool call. Re-enable with `./2_start_llama.sh --think` (or `BONSAI_THINK=1`).
- Port **8089** so it runs beside the other stacks (see the root README ports table).
- `engine/`, `models/`, and `venv/` are git‑ignored (built/downloaded locally).
