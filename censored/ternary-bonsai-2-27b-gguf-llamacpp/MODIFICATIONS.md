# Modifications — adding Ternary Bonsai 2 27B

What was changed to add and deploy **Ternary Bonsai 2 27B** (PrismML), announced by [Emad Mostaque](https://x.com/EMostaque/status/2100717168648184002) quoting PrismML. Deployed on port **8089** as `bonsai/ternary-bonsai-2-27b`.

## Key decision: why GGUF + a llama.cpp fork (not MLX)

Bonsai 2 packs are stored in a **rotated basis** (blockwise Hadamard rotation folded into the weights); the runtime must apply the matching transform to activations. The MLX pack declares `model_type: prism_hadamard_qwen35` / `requires_runtime: true`.

- **The MLX pack cannot be served as an OpenAI API.** Stock `mlx_vlm.server` / `mlx_lm.server` don't use the bundled loader — verified live: `Model type prism_hadamard_qwen35 not supported`. PrismML's own `start_mlx_server.sh` (in [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo), their stated source of truth) **refuses Bonsai 2**, warning that serving it through stock MLX "would return wrong output with no error." The MLX build is one-shot generation only.
- **Working path:** the **PQ2_0 (group-128) GGUF** served by **PrismML's llama.cpp fork** built from source with Metal. Stock llama.cpp / Ollama lack the PQ2_0 kernels. `llama-server --jinja` gives OpenAI tool calling; `--mmproj` gives vision.

An MLX stack was scaffolded first and abandoned once this was confirmed.

## New stack — `censored/ternary-bonsai-2-27b-gguf-llamacpp/`

| File | Purpose |
|------|---------|
| `1_setup_download.sh` | Clone PrismML's llama.cpp fork (branch `prism`, pinned tag `prism-b10683-d8f26ee`), build `llama-server` with Metal, download `PQ2_0` GGUF (7.21 GB) + `mmproj-Q8_0` (0.63 GB). |
| `2_start_llama.sh` | Serve on `:8089` — `llama-server -m <PQ2_0> --mmproj <mmproj> --alias ternary-bonsai-2-27b --jinja -ngl 999 -fa on -c 65536 --temp 1.0 --top-p 0.95 --top-k 20`. `status` / `stop` subcommands. |
| `README.md` | Stack docs. |
| `kilo.json` | Per-stack Kilo config (provider `bonsai`, default model `bonsai/ternary-bonsai-2-27b`). |

Git-ignored (built/downloaded locally): `engine/` (fork checkout + build), `models/` (~7.9 GB weights), `venv/`, `setup.out`, `.bonsai_llama.log`.

## Repo edits

- **`kilo.json`** (root, the live-config source): added the `bonsai` provider → `baseURL http://127.0.0.1:8089/v1`, model `ternary-bonsai-2-27b`, `tool_call: true`, `text + image`, `limit.context 49152 / output 8192`. Installed live via `install_kilo.sh`.
- **`sync_agent_prompts.py`**: registered the per-stack `kilo.json` so the shared prompt block stays in sync (`--check` green).
- **`README.md`**: models table + endpoints table rows; added `Ternary Bonsai 2 (8089)` to the ports line.
- **`README-models.md`**: catalog row.
- **`.gitignore`**: `**/ternary-bonsai-2-27b-gguf-llamacpp/{engine,models}/` + `setup.out`.

## Model facts

- Hadamard-rotated **ternary** (~1.72 bit/weight) quant of **Qwen3.8 27B**; ~9× smaller than FP16, ~98% of aggregate benchmark performance.
- **Text + image**, native tool calling (BFCL v3 ~74), **thinking model** (reasoning stays on).
- Context 262K native. Served window `-c 65536`; Kilo `limit.context 49152` / `output 8192` (peak 57344 < 65536) — a coherent cap that compacts before overflow. Raise `BONSAI_CTX` + `limit.context` together for more.

## Verification (live, on `:8089`)

- Text: `finish=stop`, `"bonsai ok"`.
- Tool calling (`--jinja`): `finish=tool_calls`, `get_weather({"city":"Paris"})`.
- Thinking: response returned `reasoning_content` separate from the answer.

## Fixes

- **Context overflow** (request 41206 > 32768): raised served window to `-c 65536` and set Kilo `limit.context 49152`/`output 8192` (was 32768). Server default `BONSAI_CTX` bumped to 65536.

## Run

```bash
cd censored/ternary-bonsai-2-27b-gguf-llamacpp
./1_setup_download.sh     # once — build fork + download weights (~5 min build, ~7.9 GB)
./2_start_llama.sh        # serve on :8089
# Kilo/OpenCode: provider `bonsai`, model `bonsai/ternary-bonsai-2-27b`, API http://127.0.0.1:8089/v1
```

## Not done

- OpenCode's live config (`~/.config/opencode/opencode.json`) was **not** given the `bonsai` provider (only Kilo was configured).
- Optional `--no-think` toggle for `2_start_llama.sh` not added.
