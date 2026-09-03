# GLM-4.7-Flash Heretic (Ollama GGUF, uncensored)

Local **uncensored** [GLM-4.7-Flash](https://huggingface.co/Olafangensan/GLM-4.7-Flash-heretic) via **Ollama** + a thin OpenAI proxy for **Kilo / OpenCode**.

API: `http://127.0.0.1:18083/v1` (proxy binds to 127.0.0.1 only; `--bind-all` for LAN)  
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

### Context window (default 64k, capped on purpose)

`num_ctx` defaults to **65536**, below the model's 131072 max. Decode speed on
this stack collapses at high context (~78 tok/s near-empty vs ~8 tok/s at ~73k,
and prompt prefill of an 80k input can take 15+ minutes), so sessions are capped
to stay responsive. Kilo's `limit.context` (65536) and `limit.output` (8192) in
`kilo.json` are set to match — **keep `limit.context` ≤ `num_ctx`**, or Ollama
silently truncates the prompt (dropping the oldest tokens, including the system
prompt). Raise all three together only if you accept the slowdown:

```bash
./2_start_ollama.sh --ctx-size 131072 restart   # + raise kilo.json limit.context
```

### Keep-alive / cold start

Ollama's default keep-alive is 5 min, after which the next request pays a
~10 s reload. The start script sets 30 min instead: `OLLAMA_KEEP_ALIVE` when it
launches Ollama itself, and a pre-load request with `keep_alive` when Ollama
was already running. Kilo's own `/v1` requests do not shorten it (verified on
Ollama 0.31.1).

```bash
./2_start_ollama.sh --keep-alive 2h restart   # or --no-warm to skip pre-load
```

### Streaming timeouts ("SSE read timed out")

Ollama's OpenAI endpoint streams reasoning tokens, but buffers a `tool_calls`
response and flushes the whole call as one chunk at the end. A large tool call
at high context can then go minutes with no bytes on the wire, and a client's
per-chunk watchdog (Kilo's `chunkTimeout`) aborts the stream with
`ProviderResponseStreamError: SSE read timed out`.

Three defences, all in this repo:

1. **Proxy heartbeat.** `openai_proxy.py` sends an SSE comment (`: keepalive`)
   whenever the upstream is silent for `--heartbeat` seconds (default 15). SSE
   parsers ignore the comment, but the bytes reset the client watchdog. Set
   `--heartbeat 0` to disable.
2. **`chunkTimeout` 900000** in `kilo.json`, matching the overall `timeout`, so
   a genuinely stalled generation still ends rather than hanging forever.
3. **`limit.output` 32768** in `kilo.json`, capping how large a single buffered
   tool call can grow.

### Reasoning-only turns ("...ended the response before returning usable output")

GLM-4.7-Flash is a thinking model, and it sometimes ends a turn inside the
reasoning channel: it either stops right after thinking or spends its whole
output budget on reasoning, leaving `content` empty with no tool call. Kilo
treats that as an incomplete response, retries twice, then fails with
`The provider repeatedly ended the response before returning usable output.`

`openai_proxy.py` recovers this: when a completion ends with reasoning but no
content and no tool call, the proxy promotes the reasoning into a `content`
delta (streaming) or field (non-streaming), so the client gets usable output
and the model's work is not discarded. Turns that already produce content or a
tool call are passed through untouched. Disable with `--no-reasoning-fallback`.

### Logs

| File | Contents |
|------|----------|
| `.glm_proxy.log` | Proxy requests + the heartbeat startup line |
| `.glm_ollama.log` | `ollama serve` stdout/stderr (per-boot timestamped) |

Both are git-ignored. The Ollama log is written only when this script starts
Ollama; if Ollama was already running, look at its own log instead.

### Moved / re-cloned the repo?

`weights/` is git-ignored, so it disappears when the checkout moves. If the
model is still registered in Ollama, the start script links `weights/…gguf`
back to Ollama's blob automatically (no 32 GB re-download). Only a changed
`--ctx-size` / sampling needs the real GGUF, so re-run `./1_setup_download.sh`
if `ollama create` complains.

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

**Kilo** — sample provider in `./kilo.json` (`baseURL` `:18083`) plus per-agent
prompts (build/plan/explore/debug/code, `steps` caps). The shared
"Conclude decisively" block is managed by `../../sync_agent_prompts.py`; do
not hand-edit it. Merge into `~/.config/kilo/kilo.jsonc` and reload.

**OpenCode** — point a provider at `http://127.0.0.1:18083/v1` with model id `glm-4.7-flash-heretic-q8`.

## Useful flags

```bash
./2_start_ollama.sh restart    # start / replace proxy
./2_start_ollama.sh --greedy   # temp=0 (Kilo's agent temperature still wins per request)
./2_start_ollama.sh --bind-all # expose proxy on 0.0.0.0 (LAN)
./2_start_ollama.sh --no-proxy # Ollama :11434 only
./2_start_ollama.sh status
./2_start_ollama.sh stop
```

Requires **Ollama** and enough RAM for weights + context (128 GB is more than enough for Q8 + long context).
