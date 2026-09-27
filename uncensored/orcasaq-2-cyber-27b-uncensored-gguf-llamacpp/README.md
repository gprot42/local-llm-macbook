# OrcaSAQ-2 Cyber 27B Uncensored (GGUF · llama.cpp)

Local **OrcaSAQ-2 Cyber 27B Uncensored** ([orcarouter](https://huggingface.co/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF)) — a Qwen3.8‑27B fine‑tune (base `orcarouter/Qwen3.8-27B-Uncensored`, arch `qwen3_5`) for **defensive red teaming, vulnerability research, security coding, terminal workflows, and authorized security testing**. A single ~15.7 GB GGUF, 262K context. Served as an OpenAI API for **Kilo / OpenCode**, so your source, logs and findings stay on the machine.

> Use it only for authorized, defensive security work on systems and code you own or are permitted to test.

| | |
|--|--|
| **API** | `http://127.0.0.1:8090/v1` (loop proxy → engine `:8100`) |
| **Model IDs** | `orcasaq/orcasaq-2-cyber-27b` (default, no thinking) · `orcasaq/orcasaq-2-cyber-27b-think` (OpenCode, 1024‑token thinking budget) |
| **Weights** | [`orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF`](https://huggingface.co/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF) · `OrcaSAQ-2-27B-Uncensored.gguf` (15.7 GB) — **gated** |
| **Engine** | stock [llama.cpp](https://github.com/ggml-org/llama.cpp) (Metal) on `:8100`, `--jinja` tool calling, single slot + 24 GiB RAM prompt cache — behind `loop_proxy.py` on `:8090`, which ends a turn that repeats itself |

## Access (gated repo)

The GGUF is gated — you need a Hugging Face login **and** granted access:

1. `hf auth login` — stores a token in `~/.cache/huggingface/token` (outside this repo; nothing to commit).
2. Open the [model page](https://huggingface.co/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF) and click **“Agree and access repository”**; wait for approval.
3. `./1_setup_download.sh` then downloads it. (The script checks both and prints exactly what’s missing.)

## Quick start

```bash
cd uncensored/orcasaq-2-cyber-27b-uncensored-gguf-llamacpp

# 1. Build llama.cpp (Metal) + download the GGUF  (~5 min build, ~15.7 GB, needs HF access)
./1_setup_download.sh

# 2. Serve on :8090  (OpenAI-compatible, --jinja tool calling, reasoning off)
./2_start_llama.sh
#    status / stop:  ./2_start_llama.sh status | ./2_start_llama.sh stop
#    thinking:       ./2_start_llama.sh --think   (1024→2048-token budget; --think-budget N)

# 3. Point Kilo / OpenCode at it
./install-opencode-json.sh          # OpenCode: merges the orcasaq provider
#    Kilo: this stack's kilo.json is a project override; the root kilo.json / install_kilo.sh ships the shared config.
```

## Notes

- **Engine.** OrcaSAQ ships a standard `qwen3_5` GGUF, so **stock llama.cpp** serves it — no custom fork (unlike the Bonsai stack). `qwen3_5` needs a recent llama.cpp build; if the pinned `master` doesn’t know the architecture, point the setup at PrismML’s fork, which does: `ORCA_LLAMACPP_URL=https://github.com/PrismML-Eng/llama.cpp.git ORCA_LLAMACPP_REF=prism-b10743-adfffbe ./1_setup_download.sh`.
- **Sampling** is the Qwen3.8 base preset (non‑thinking `temp 0.7 / top‑p 0.8 / top‑k 20 / min‑p 0 / presence 1.5`; thinking `1.0 / 0.95 / 20 / presence 0`). These are inherited from the base model card — confirm against the gated OrcaSAQ card once you have access.
- **Reasoning off by default** for agentic use (same measured reasons as the Bonsai stack: unbounded thinking exhausts the output budget before the tool call, and replayed `reasoning_content` grows context until compaction fails). `--think` re‑enables it under a budget; the OpenCode `-think` model entry does the same per‑session via `/models`.
- **Loop guard.** `loop_proxy.py` on `:8090` fronts the engine on `:8100`. Agentic terminal/security workflows loop, and a prompt rule alone does not hold on a small local model. When the model has just repeated a call (same result a 2nd/3rd time with no `write`/`edit` between, or the same call issued a 3rd time) it replaces that result with a `[Harness] REFUSED …` error quoting the output and, from the 3rd repeat, adds a user‑role directive, then forwards — so the turn self‑corrects. It ends the turn only on the 4th identical result, the 6th issuance, or after 200 tool calls (`--max-rounds` / `LOOP_MAX_ROUNDS`). Streams pass through. `python3 loop_proxy.py --self-test`.
- **Why not TensorFold?** TensorFold serves MLX checkpoints, not GGUF, and no MLX build of OrcaSAQ is published. Its speed advantage comes from the DFlash2 drafter, which is paired to the *base* Qwen3.8‑27B — on this fine‑tune it would draft poorly and fall back toward serial speed. So llama.cpp on the shipped GGUF is the right path here.
- Port **8090** (proxy) / **8100** (engine) so it runs beside the other stacks (see the root README ports table). `engine/`, `models/`, `venv/` are git‑ignored.
