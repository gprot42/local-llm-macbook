# DeepSeek — MLX Server

Run DeepSeek locally on Apple Silicon via `mlx_lm.server`.

Tuned for **M5 Max with 128 GB** unified memory (this machine).

### When to use this model

**DeepSeek V4 Flash 2bit-DQ** (strong coding/reasoning, heavy — ~97 GB, 128 GB Mac)

- **Best:** hard coding/agent work in Kilo when quality beats speed; long-context reasoning; SWE-style multi-file tasks; local substitute for a strong cloud coder when you have the RAM
- **OK:** day-to-day coding, refactors, chat with `<think>`; agent loops if you can wait for load/prefill; general reasoning that is not ultra-latency-sensitive
- **Bad:** fast iteration / snappy tool loops (prefer Qwen 3.6); uncensored / low-refusal needs (use Heretic Gemma); machines under ~128 GB; treating 2-bit MoE as cloud-frontier on huge greenfield apps

---

## Recommended model

| Model | HF repo | Disk | Fits 128 GB? | Best for |
|-------|---------|------|--------------|----------|
| **DeepSeek-V4-Flash-2bit-DQ** (default) | `mlx-community/DeepSeek-V4-Flash-2bit-DQ` | ~97 GB | Yes (~30 GB headroom) | Latest DeepSeek — coding, reasoning, agents, 1M context |
| DeepSeek-R1-Distill-Qwen-32B-4bit | `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit` | ~18 GB | Yes (plenty of room) | Fast R1-style chain-of-thought, smaller footprint |

### Why V4 Flash 2bit-DQ?

On 128 GB RAM, larger DeepSeek checkpoints do **not** fit:

| Model | Size | Verdict on 128 GB |
|-------|------|-------------------|
| DeepSeek-V4-Flash-4bit | 151 GB | Too large (no KV cache headroom) |
| DeepSeek-V3 / V3.1 4bit | 378 GB | Won't load |
| DeepSeek-R1 4bit | 420 GB | Won't load |
| DeepSeek-V4-Pro 4bit | ~600+ GB | Won't load |

**DeepSeek-V4-Flash-2bit-DQ** uses dynamic mixed-precision (2-bit experts, higher-bit sensitive layers) and is the largest DeepSeek variant that runs comfortably on this Mac. It is a 284B MoE model with 13B active parameters — strong on SWE-bench (~79%), LiveCodeBench, and long-context tasks.

### DeepSeek-V4 runtime note

PyPI `mlx-lm` (0.31.x) does **not** include `mlx_lm.models.deepseek_v4` yet ([upstream PR #1192](https://github.com/ml-explore/mlx-lm/pull/1192) still open).

`./1_setup_download.sh` installs a community fork with V4 support:

```text
git+https://github.com/spicyneuron/mlx-lm.git@_ds4
```

| Package | Required | Why |
|---------|----------|-----|
| **mlx-lm** | git `spicyneuron/mlx-lm@_ds4` (not PyPI) | Provides `deepseek_v4` model class |
| **transformers** | latest (≥5.12; currently 5.13) | Tokenizer/config for V4; setup patches mlx-lm for 5.13+ `AutoTokenizer.register` API change |
| **mlx** | latest (0.32+) | Metal backend |
| **jinja2** | complete install | Broken/partial wheel breaks all `mlx_lm` imports |

`./1_setup_download.sh` always upgrades the stack and re-applies the mlx-lm patch. `./2_start_mlx.sh` re-applies the patch on start.

Override if needed:

```bash
MLX_LM_GIT_URL='git+https://github.com/Blaizzy/mlx-lm.git@pc/add-deepseekv4flash-model' \
  ./1_setup_download.sh --skip-download
```

---

## Quick start

```bash
cd censored/deepseek-v4-flash-2bit-dq-mlx

# Download ~97 GB model + install deps (one-time)
./1_setup_download.sh

# Start OpenAI-compatible API on port 8082
./2_start_mlx.sh
```

Smaller R1 distill alternative:

```bash
./1_setup_download.sh r1-32b
./2_start_mlx.sh --model r1-32b
```

Re-install V4 mlx-lm only (weights already present):

```bash
./1_setup_download.sh --skip-download
```

---

## Kilo Code

```bash
cp kilo.json /path/to/your/project/kilo.json
# or merge into ~/.config/kilo/kilo.jsonc
```

| Field | Value |
|-------|-------|
| Base URL | `http://127.0.0.1:8082/v1` (use `127.0.0.1`, not `localhost` — Node/Kilo may prefer IPv6 `::1` which this server does not bind) |
| Provider / Model | `deepseek-mlx/deepseek-v4-flash-2bit-dq` |

DeepSeek V4 recommends `temperature=1.0, top_p=1.0` (already set in `kilo.json`).

> **Reasoning:** V4 supports `<think>` reasoning modes. In Kilo, leave the provider **Reasoning** checkbox unchecked unless you want extended thinking output in the chat stream.

### Agent harness (anti-bomb)

`openai_server.py` + `harness.py` are tuned so greenfield prompts (e.g. “create a web adventure game”) do **not** burn the turn on “I’ll start by setting up…” monologues:

| Layer | What it does |
|-------|----------------|
| **Steer** | Appends implement-first rules to system + last user message (recency beats huge Kilo system prompts) |
| **Prefill** | For first-turn create/web tasks, forces the assistant to start inside an HTML/code fence |
| **Loop detector** | Stops rephrase restarts, outline spam, ellipsis thrash, fake tool transcripts, token-repeat garbage |
| **Stops** | EOS / next-`User:` markers end the turn |
| **Clamp** | Client `max_tokens` capped at 16k (Kilo often sends 32k) |

Opt out of prefill only: send `"no_prefill": true` (or `"harness_prefill": false`) in the chat-completions body.

Unit tests (no model load):

```bash
./venv/bin/python test_harness.py
```

---

## API test

```bash
curl http://localhost:8082/v1/models

curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash-2bit-dq","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}'
```

---

## Memory tips (128 GB)

- Default V4 Flash leaves ~30 GB for OS + KV cache — enough for 32k–128k context in most sessions.
- If you hit Metal OOM, limit the prompt KV cache:
  `./2_start_mlx.sh --prompt-cache-bytes 8589934592`  # 8 GiB
- Cap default completion length: `./2_start_mlx.sh --max-tokens 4096`
- For maximum speed with less quality, use the R1 distill: `./1_setup_download.sh r1-32b`

> Note: older docs mentioned `--max-kv-size`; that flag was removed from `mlx_lm.server`. Use `--prompt-cache-bytes` / `--max-tokens` instead.

---

## Ports

| Server | Port |
|--------|------|
| Gemma 4 MLX | 8080 |
| **DeepSeek MLX** | **8082** |
| Qwen3.6 mtplx | 8765 |
