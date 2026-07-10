# Ornith-1.0-35B (Ollama GGUF)

Local [Ornith](https://deep-reinforce.com/ornith.html) via **Ollama** + a tool-call proxy for **Kilo / OpenCode**.

API: `http://127.0.0.1:18082/v1`  
Weights: [deepreinforce-ai/Ornith-1.0-35B-GGUF](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)

## Run Q8_0 (~37 GB, max quality)

```bash
cd ornith-1.0-35b-q8-gguf-ollama

# once: install Ollama if needed
brew install ollama

# 1. Download Q8 weights (~36.9 GB)
./1_setup_download.sh

# 2. Start Ollama + proxy
./2_start_ollama.sh restart

# 3. In Kilo / OpenCode, pick:
#    ornith/ornith-1.0-35b-q8
```

First reply is slow while Ollama loads the model. On 128 GB machines Q8 is fine; if RAM is tight:

```bash
./2_start_ollama.sh --ctx-size 65536 restart
```

### Health check

```bash
./2_start_ollama.sh status
curl -s http://127.0.0.1:18082/healthz
ollama list | grep ornith
```

Expect `ok: true` and `ornith-1.0-35b-q8` listed.

### Stop

```bash
./2_start_ollama.sh stop
```

## Clients

**Kilo** — provider is in `../kilo.json` and `~/.config/kilo/kilo.jsonc` (baseURL `:18082`). Reload the editor after starting the proxy.

**OpenCode** — once:

```bash
./install-opencode-json.sh
```

## Useful flags

```bash
./2_start_ollama.sh restart   # start / replace proxy
./2_start_ollama.sh --greedy  # temp=0
./2_start_ollama.sh status
./2_start_ollama.sh stop
```

Requires **Ollama** and enough RAM for weights + context (128 GB recommended for Q8 + long context).

### Abrupt / empty replies (thinking)

Ornith is a thinking model. Ollama’s OpenAI `/v1` endpoint **ignores** `think: false`, so without a workaround the model fills `reasoning` until `max_tokens` and returns **empty `content`** (`finish_reason: length`) — Kilo shows a long plan that ends abruptly.

The tool proxy defaults to `reasoning_effort: "none"` (and `think: false`) so answers/tool calls land in `content`. To re-enable thinking for a request, send `"think": true` or `"reasoning_effort": "low"|"medium"|"high"`.
