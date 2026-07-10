# GLM-4.7-Flash Heretic (Ollama GGUF, uncensored)

Local **uncensored** [GLM-4.7-Flash](https://huggingface.co/Olafangensan/GLM-4.7-Flash-heretic) via **Ollama** + a thin OpenAI proxy for **Kilo / OpenCode**.

API: `http://127.0.0.1:18083/v1`  
Weights: [DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF](https://huggingface.co/DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF)

~30B MoE / ~3B active — fits easily on M5 128 GB. Port **18083** so it can run beside Ornith (`:18082`).

## Run Q8_0 (~32 GB, default)

```bash
cd glm-4.7-flash-heretic-gguf-ollama

# once: install Ollama if needed
brew install ollama

# 1. Download Q8 weights (~32 GB)
./1_setup_download.sh

# 2. Start Ollama + proxy
./2_start_ollama.sh restart

# 3. In Kilo / OpenCode, pick:
#    glm/glm-4.7-flash-heretic-q8
```

### Smaller quant (Q6, ~25 GB)

```bash
./1_setup_download.sh q6
./2_start_ollama.sh --quant q6 restart
```

### Long context if needed

```bash
./2_start_ollama.sh --ctx-size 65536 restart
```

### Health check

```bash
./2_start_ollama.sh status
curl -s http://127.0.0.1:18083/healthz
ollama list | grep glm
```

Expect `ok: true` and `glm-4.7-flash-heretic-q8` listed.

### Stop

```bash
./2_start_ollama.sh stop
```

## Quants

| Quant | Size | Command |
|-------|------|---------|
| q4 | ~18.5 GB | `./1_setup_download.sh q4` |
| q5 | ~21.6 GB | `./1_setup_download.sh q5` |
| q6 | ~25 GB | `./1_setup_download.sh q6` |
| **q8** | **~32.1 GB** | **default** |

Model ids: `glm-4.7-flash-heretic-q4` · `…-q5` · `…-q6` · `…-q8`

## Clients

**Kilo** — sample provider in `./kilo.json` (`baseURL` `:18083`). Merge into `~/.config/kilo/kilo.jsonc` and reload.

**OpenCode** — point a provider at `http://127.0.0.1:18083/v1` with model id `glm-4.7-flash-heretic-q8`.

## Useful flags

```bash
./2_start_ollama.sh restart    # start / replace proxy
./2_start_ollama.sh --greedy   # temp=0
./2_start_ollama.sh --no-proxy # Ollama :11434 only
./2_start_ollama.sh status
./2_start_ollama.sh stop
```

Requires **Ollama** and enough RAM for weights + context (128 GB is more than enough for Q8 + long context).
