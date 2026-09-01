# Agent ops — Qwen3.5-122B DFlash (Kilo)

Local OpenAI API for coding agents on this machine.

| | |
|--|--|
| **API** | `http://127.0.0.1:8086/v1` |
| **Model id** | `qwen3.5-122b-a10b-dflash` |
| **Start / stop** | `./2_start_dflash.sh` · `restart` · `stop` · `status` |
| **Health / busy** | `./status_dflash.sh` or `curl -s http://127.0.0.1:8086/health` |
| **Log** | `.dflash_server.log` |

Kilo system prompt: see [`kilo_system_prompt.txt`](kilo_system_prompt.txt) (paste into Kilo custom instructions).

---

## Hard limits (this stack)

| Constraint | Practical value |
|------------|-----------------|
| Comfortable prompt | ≲ 8–10k tokens |
| Auto-trim budget | 12 288 tokens (oldest history dropped) |
| Tools `max_tokens` ceiling | 2048 (even if client sends 8192) |
| Generation wall (tools / plain) | ~90s / ~120s |
| Kilo shell tool timeout | often 120s |
| Concurrent generations | **serialized** (one at a time) |
| Co-load another 60 GB+ model | avoid on 128 GB (swap kills speed) |

Prefill of large prompts dominates wall time. Early tool-call stop helps **decode**, not prefill.

---

## Prompt pattern (implement-first)

**Avoid**

```text
Implement everything and don't stop until it's all done.
```

**Prefer**

```text
Milestone only: <one concrete change> in <paths>.
Done when: <exact command> exits 0.
Constraints: only touch listed paths; no scope creep.
Use tools to edit; then run the command. Stop when done or blocked with exact error.
```

### Milestone template

```text
You are implementing software in a local repo. Prefer code over plans.

Repo: <absolute path>
Milestone: <one sentence>
Done when:
  1) <expected file change>
  2) command: `<exact command>` exits 0
Constraints:
  - Only touch: <paths>
  - No refactors outside scope
  - If blocked, stop with the exact error and next file to inspect
Start by reading the relevant files, then edit, then run the command.
```

Track multi-hour work in-repo:

```markdown
## Progress
- [x] M1: …
- [ ] M2: …
- [ ] M3: …
```

Each chat: *do the next unchecked milestone only; check it off when tests pass.*

---

## Chat hygiene

1. **New chat per major milestone** — do not run epics in one thread.
2. When the server log shows `auto-trimmed context`, start a **new chat** and paste only: goal, key paths, last error/test output.
3. Do not paste huge logs into chat; put them in files and `grep`/`read` subsets.
4. If a turn sits “thinking” with no tools for ~2+ minutes, cancel and run `./status_dflash.sh`.

---

## Kilo settings

| Setting | Value |
|---------|--------|
| Base URL | `http://127.0.0.1:8086/v1` |
| Model | `qwen3.5-122b-a10b-dflash` |
| Stream | On |
| Max tokens | 2k–4k is enough (server clamps tool turns) |
| Concurrent agent sessions on same port | avoid |
| Custom instructions | contents of `kilo_system_prompt.txt` |

---

## Ops loop

| Symptom | Action |
|---------|--------|
| Waiting 5+ min | `./status_dflash.sh` — high `prompt_tokens` → cancel, new chat |
| `auto-trimmed context` in log | new chat; carry only essentials |
| Prints `[tool_calls] [...]` as text | new chat (poisoned history); server converts dumps on fresh turns |
| Shell killed ~120s | split commands; no long interactive waits |
| Server dead / wrong code | `./2_start_dflash.sh restart` then wait for health |
| Everything slow | memory/swap; unload other large models |

### Status one-liner

```bash
./status_dflash.sh
# or:
curl -s http://127.0.0.1:8086/health && echo && tail -20 .dflash_server.log
```

---

## Server behaviors (for operators)

- **Tool XML / dumps** → converted to OpenAI `tool_calls` (`finish_reason: tool_calls`).
- **Oversized history** → auto-trim (system + newest turns; fat tool dumps capped).
- **Serialized MLX** → second request queues behind the lock.
- **Connection: close** → fewer stuck keep-alives after client cancel.

Details live in `dflash-mlx/dflash_mlx/openai_server.py`.
