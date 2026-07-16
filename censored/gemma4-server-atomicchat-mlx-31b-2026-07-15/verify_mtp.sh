#!/usr/bin/env bash
# verify_mtp.sh — Confirm the server on :8080 is mlx_vlm with MTP and measure decode speed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-gemma-4-31b-it-atomicchat-mlx-4bit}"

echo "=== MTP verification (http://${HOST}:${PORT}/v1) ==="
echo ""

pids="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
if [ -z "$pids" ]; then
    echo "ERROR: Nothing listening on port ${PORT}."
    echo "  Start: ./2_start_mlx.sh"
    exit 1
fi

found_mtp=false
for pid in $pids; do
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if echo "$args" | grep -q 'mlx_vlm.server'; then
        echo "→ Server PID:  $pid"
        echo "→ Command:     ${args:0:120}..."
        if echo "$args" | grep -qE 'draft-kind[[:space:]]+mtp'; then
            echo "→ MTP flags:   --draft-kind mtp present ✓"
            draft_model="$(echo "$args" | sed -n 's/.*--draft-model \([^ ]*\).*/\1/p')"
            block_size="$(echo "$args" | sed -n 's/.*--draft-block-size \([^ ]*\).*/\1/p')"
            [ -n "$draft_model" ] && echo "→ Drafter:     $draft_model"
            [ -n "$block_size" ] && echo "→ Block size:  $block_size"
            found_mtp=true
        else
            echo "WARNING: mlx_vlm.server running but --draft-kind mtp not found."
        fi
    elif echo "$args" | grep -q 'mlx_lm.server'; then
        echo "→ Server PID:  $pid (mlx_lm.server — MTP disabled)"
        echo "  Start with MTP: ./2_start_mlx.sh restart --with-mtp"
        exit 2
    fi
done

if [ "$found_mtp" = false ]; then
    echo "WARNING: No mlx_vlm.server with MTP on port ${PORT}."
fi
echo ""

if ! curl -sf --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: API not responding on http://${HOST}:${PORT}/v1"
    exit 1
fi

echo "→ MTP speeds decode/generation tokens only."
echo "  Kilo wall-clock time also includes prompt prefill, tool calls, edit application, and file IO."
echo ""
echo "→ Smoke test (greedy, max_tokens=16) ..."
resp="$(curl -sf --max-time 90 -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":16,\"stream\":false,\"temperature\":0}")" || {
    echo "ERROR: Chat request failed or timed out (server may be wedged)."
    echo "  Try: ./2_start_mlx.sh restart"
    exit 1
}

python3 -c "
import json, sys
r = json.loads(sys.argv[1])
t = r.get('timings') or {}
content = (r['choices'][0]['message'].get('content') or '')[:60]
tps = t.get('predicted_per_second')
print(f'→ Reply:       {content!r}')
print(f'→ Decode:      {t.get(\"predicted_n\", \"?\")} tokens in {t.get(\"predicted_ms\", 0):.0f} ms')
if tps is not None:
    print(f'→ Decode tok/s: {tps:.1f}  (MTP typically ~15–50+; without MTP often ~5–12)')
print(f'→ Peak memory: {t.get(\"peak_memory\", \"?\")} GB')
" "$resp"

echo ""
echo "→ Code-shaped benchmark (greedy, max_tokens=256) ..."
resp="$(curl -sf --max-time 180 -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a concise Python function that returns the Fibonacci number for n using iteration, then explain it briefly.\"}],\"max_tokens\":256,\"stream\":false,\"temperature\":0}")" || {
    echo "ERROR: Code benchmark failed or timed out (server may be wedged)."
    echo "  Try: ./2_start_mlx.sh restart"
    exit 1
}

python3 -c "
import json, sys
r = json.loads(sys.argv[1])
t = r.get('timings') or {}
choice = r['choices'][0]
msg = choice.get('message') or {}
content = (msg.get('content') or msg.get('reasoning') or '')[:90].replace('\n', ' ')
tps = t.get('predicted_per_second')
print(f'→ Finish:       {choice.get(\"finish_reason\")}')
print(f'→ Tokens:       prompt={t.get(\"prompt_n\", \"?\")}  generated={t.get(\"predicted_n\", \"?\")}')
print(f'→ Prompt time:  {t.get(\"prompt_ms\", 0):.0f} ms  (not helped by MTP)')
print(f'→ Decode time:  {t.get(\"predicted_ms\", 0):.0f} ms')
if tps is not None:
    print(f'→ Decode tok/s: {tps:.1f}  (healthy MTP on 31B is usually ~15–50+ warm)')
print(f'→ Peak memory:  {t.get(\"peak_memory\", \"?\")} GB')
print(f'→ Sample:       {content!r}')
" "$resp"

echo ""
if [ "$found_mtp" = true ]; then
    echo "MTP verification OK"
    exit 0
fi
exit 2