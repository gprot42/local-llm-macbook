#!/usr/bin/env bash
# =============================================================================
# 2_start_mlx.sh — Serve Gemma 4 31B IT (MLX 4-bit) with optional MTP
#
# Default: mlx_lm.server without MTP (simpler, more stable for Kilo)
#   Target:  gemma-4-31b-it-mlx-4bit
#   --with-mtp: mlx_vlm.server + Gemma 4 MTP speculative decoding (~2× decode speed)
#   Drafter: gemma-4-31b-it-assistant-mlx-bf16 (--draft-kind mtp)
#   API model "default_model" is aliased to the target via MLX_VLM_DEFAULT_MODEL
#
# OpenAI-compatible API at http://localhost:8080/v1
#
# Usage:
#   ./2_start_mlx.sh                  # standard mlx_lm.server (no MTP, default)
#   ./2_start_mlx.sh --with-mtp       # mlx_vlm.server + MTP drafter
#   ./2_start_mlx.sh restart
#   ./2_start_mlx.sh restart --with-mtp
#   ./2_start_mlx.sh --port 8081
#   ./2_start_mlx.sh --draft-block-size 4
#   ./2_start_mlx.sh --no-verify-mtp   # skip post-start MTP timing check
#   ./verify_mtp.sh                    # check a running server anytime
#   ./2_start_mlx.sh --help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="gemma-4-31b-it-mlx-4bit"
MODEL_ID="gemma-4-31b-it-mlx-4bit"
HF_REPO="mlx-community/gemma-4-31b-it-4bit"
DRAFT_DIR="gemma-4-31b-it-assistant-mlx-bf16"
DRAFT_HF_REPO="mlx-community/gemma-4-31B-it-assistant-bf16"
PORT=8080
HOST="127.0.0.1"
DO_RESTART=false
ENABLE_MTP=false
DRAFT_BLOCK_SIZE=4
LOW_MEMORY=false
MAX_KV_SIZE=""
VERIFY_MTP=true

stop_server_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 1
    fi
}

for arg in "$@"; do
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        echo ""
        echo "Usage: ./2_start_mlx.sh [options]"
        echo ""
        echo "  restart              Stop any server on the port, then start fresh"
        echo "  --with-mtp           Enable MTP (mlx_vlm.server + assistant drafter)"
        echo "  --mtp                Alias for --with-mtp"
        echo "  --no-mtp             Use mlx_lm.server without speculative decoding (default)"
        echo "  --draft-block-size N MTP block size (default: 4)"
        echo "  --low-memory         Cap KV cache, disable MTP (24 GB Macs / Metal OOM)"
        echo "  --max-kv-size N      Max KV tokens (mlx_vlm.server only)"
        echo "  --no-verify-mtp      Skip post-start MTP config + decode speed check"
        echo "  --verify-mtp         Run MTP verify after start (default when MTP on)"
        echo "  --port PORT          Port to listen on (default: 8080)"
        echo "  --host HOST          Host to bind (default: 127.0.0.1)"
        echo ""
        echo "  ./verify_mtp.sh       Check MTP on a running server (no restart)"
        echo ""
        echo "Default (no MTP):"
        echo "  Kilo Code ──→ http://localhost:$PORT/v1  (mlx_lm.server)"
        echo "  Model ID in kilo.json: $MODEL_ID"
        echo ""
        echo "With MTP (--with-mtp):"
        echo "  Kilo Code ──→ http://localhost:$PORT/v1  (mlx_vlm.server + assistant drafter)"
        echo ""
        exit 0
    }
done

# Two-pass: pick up value-bearing args
i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        restart) DO_RESTART=true; ((i+=1)) ;;
        --no-mtp) ENABLE_MTP=false; ((i+=1)) ;;
        --with-mtp|--mtp) ENABLE_MTP=true; ((i+=1)) ;;
        --port) PORT="${args[$((i+1))]:-$PORT}"; ((i+=2)) ;;
        --host) HOST="${args[$((i+1))]:-$HOST}"; ((i+=2)) ;;
        --draft-block-size) DRAFT_BLOCK_SIZE="${args[$((i+1))]:-$DRAFT_BLOCK_SIZE}"; ((i+=2)) ;;
        --low-memory) LOW_MEMORY=true; ((i+=1)) ;;
        --max-kv-size) MAX_KV_SIZE="${args[$((i+1))]:-$MAX_KV_SIZE}"; ((i+=2)) ;;
        --no-verify-mtp) VERIFY_MTP=false; ((i+=1)) ;;
        --verify-mtp) VERIFY_MTP=true; ((i+=1)) ;;
        *) ((i+=1)) ;;
    esac
done

# MTP verify only applies when speculative decoding is on
if [ "$ENABLE_MTP" = false ]; then
    VERIFY_MTP=false
fi

print_mtp_verification() {
    echo ""
    echo "── MTP verification ──────────────────────────────────────────"
    echo "→ Engine:     mlx_vlm.server"
    echo "→ Target:     $MODEL_PATH"
    echo "→ Drafter:    $DRAFT_PATH"
    echo "→ Draft kind: mtp"
    echo "→ Block size: $DRAFT_BLOCK_SIZE"
    echo "→ Scope:      speeds up decode/generation tokens, not prompt prefill or tools"
    if ps -p "$SERVER_PID" -o args= 2>/dev/null | grep -qE 'draft-kind[[:space:]]+mtp'; then
        echo "→ Process:    PID $SERVER_PID has --draft-kind mtp ✓"
    else
        echo "WARNING: PID $SERVER_PID missing --draft-kind mtp in command line"
    fi
    echo "→ Smoke test (greedy, max_tokens=16) ..."
    local resp
    resp="$(curl -sf --max-time 90 -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":16,\"stream\":false,\"temperature\":0}")" || {
        echo "ERROR: MTP verify request failed or timed out."
        return 1
    }
    python3 -c "
import json, sys
r = json.loads(sys.argv[1])
t = r.get('timings') or {}
content = (r['choices'][0]['message'].get('content') or '')[:60]
tps = t.get('predicted_per_second')
print(f'→ Reply:        {content!r}')
print(f'→ Decode:       {t.get(\"predicted_n\", \"?\")} tokens in {t.get(\"predicted_ms\", 0):.0f} ms')
if tps is not None:
    print(f'→ Decode tok/s: {tps:.1f}  (MTP ~15–50+; mlx_lm.server ~5–12)')
print(f'→ Peak memory:  {t.get(\"peak_memory\", \"?\")} GB')
" "$resp"
    echo "→ Code-shaped benchmark (greedy, max_tokens=256) ..."
    resp="$(curl -sf --max-time 180 -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a concise Python function that returns the Fibonacci number for n using iteration, then explain it briefly.\"}],\"max_tokens\":256,\"stream\":false,\"temperature\":0}")" || {
        echo "ERROR: MTP code benchmark failed or timed out."
        return 1
    }
    python3 -c "
import json, sys
r = json.loads(sys.argv[1])
t = r.get('timings') or {}
choice = r['choices'][0]
msg = choice.get('message') or {}
content = (msg.get('content') or msg.get('reasoning') or '')[:80].replace('\n', ' ')
tps = t.get('predicted_per_second')
print(f'→ Finish:       {choice.get(\"finish_reason\")}')
print(f'→ Tokens:       prompt={t.get(\"prompt_n\", \"?\")}  generated={t.get(\"predicted_n\", \"?\")}')
print(f'→ Prompt time:  {t.get(\"prompt_ms\", 0):.0f} ms  (not helped by MTP)')
print(f'→ Decode time:  {t.get(\"predicted_ms\", 0):.0f} ms')
if tps is not None:
    print(f'→ Decode tok/s: {tps:.1f}  (healthy MTP on 31B is usually ~15–50+ warm)')
print(f'→ Sample:       {content!r}')
" "$resp"
    echo "→ Note: request logs stay quiet by design; uvicorn 200 OK + response timings prove success."
    echo "── MTP verification passed ─────────────────────────────────────"
    echo ""
}

if [ "$DO_RESTART" = true ]; then
    stop_server_on_port "$PORT"
    echo ""
fi

if [ "$LOW_MEMORY" = true ]; then
    ENABLE_MTP=false
    DRAFT_BLOCK_SIZE=2
    if [ -z "$MAX_KV_SIZE" ]; then
        MAX_KV_SIZE=8192
    fi
    echo "→ Low-memory mode: MTP off, max-kv-size=${MAX_KV_SIZE}"
    echo "  Tip: set Kilo context to 16k or lower in kilo.json"
    echo ""
fi

echo "=== Gemma 4 31B IT (MLX 4-bit) ==="
echo "→ Model:    $MODEL_DIR"
echo "→ MTP:      $([ "$ENABLE_MTP" = true ] && echo "enabled (mlx_vlm.server)" || echo "disabled (mlx_lm.server)")"
echo "→ Endpoint: http://localhost:$PORT/v1"
if [ "$ENABLE_MTP" = true ]; then
    echo "→ Expect:   startup verifies --draft-kind mtp and prints decode tok/s"
    echo "→ Note:     Kilo wall-clock speed also includes prompt prefill, tool calls, edits, and file IO"
fi
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

if [ "$ENABLE_MTP" = true ] && [ -f "$SCRIPT_DIR/apply_local_patches.sh" ]; then
    "$SCRIPT_DIR/apply_local_patches.sh"
fi

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"
validate_local_model() {
    python3 "$VALIDATE_MODEL" "$1"
}

# ── Resolve target model path ─────────────────────────────────────────────────
if [ -d "$MODEL_DIR" ]; then
    echo "→ Checking target model files ..."
    if validate_local_model "$MODEL_DIR"; then
        MODEL_PATH="$MODEL_DIR"
        echo "→ Using local weights: $MODEL_DIR"
    else
        echo ""
        echo "Fix: run ./1_setup_download.sh"
        exit 1
    fi
else
    echo "→ Local target not found — will stream from HuggingFace: $HF_REPO"
    MODEL_PATH="$HF_REPO"
fi

# ── Resolve MTP assistant path ────────────────────────────────────────────────
DRAFT_PATH=""
if [ "$ENABLE_MTP" = true ]; then
    if [ -d "$DRAFT_DIR" ] && validate_local_model "$DRAFT_DIR" >/dev/null 2>&1; then
        DRAFT_PATH="$DRAFT_DIR"
        echo "→ MTP assistant: $DRAFT_DIR"
    elif [ -d "$DRAFT_DIR" ]; then
        echo "WARNING: MTP assistant weights incomplete."
        echo "  Fix: ./1_setup_download.sh   or: hf download $DRAFT_HF_REPO --local-dir $DRAFT_DIR"
        echo "  Falling back to mlx_lm.server (no MTP). Run without --with-mtp to silence this warning."
        ENABLE_MTP=false
    else
        echo "→ MTP assistant not downloaded — streaming from HuggingFace: $DRAFT_HF_REPO"
        DRAFT_PATH="$DRAFT_HF_REPO"
    fi
fi
echo ""

# ── Cleanup handler ───────────────────────────────────────────────────────────
SERVER_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down server ..."
    [ -n "$SERVER_PID" ] && kill -TERM "$SERVER_PID" 2>/dev/null
    sleep 1
    [ -n "$SERVER_PID" ] && kill -KILL "$SERVER_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM HUP

# ── Start server ──────────────────────────────────────────────────────────────
# Clients (Kilo, Cursor, etc.) often send model "default_model" (mlx_lm convention).
export MLX_VLM_DEFAULT_MODEL="$MODEL_PATH"

if [ "$ENABLE_MTP" = true ]; then
    VLM_EXTRA_ARGS=()
    if [ -n "$MAX_KV_SIZE" ]; then
        VLM_EXTRA_ARGS+=(--max-kv-size "$MAX_KV_SIZE")
    fi
    echo "→ Launching mlx_vlm.server with MTP:"
    echo "  mlx_vlm.server --model $MODEL_PATH --draft-model $DRAFT_PATH --draft-kind mtp --draft-block-size $DRAFT_BLOCK_SIZE --port $PORT"
    mlx_vlm.server \
        --model "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --draft-model "$DRAFT_PATH" \
        --draft-kind mtp \
        --draft-block-size "$DRAFT_BLOCK_SIZE" \
        --max-tokens 4096 \
        "${VLM_EXTRA_ARGS[@]}" \
        &
else
    # mlx_lm.server — sampling defaults for Kilo agent use (see kilo.json).
    echo "→ Launching mlx_lm.server without MTP:"
    echo "  mlx_lm.server --model $MODEL_PATH --port $PORT --temp 0.35 --top-p 0.95"
    mlx_lm.server \
        --model "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --temp 0.35 \
        --top-p 0.95 \
        --top-k 50 \
        --max-tokens 4096 \
        &
fi
SERVER_PID=$!

# ── Wait for server to be ready ───────────────────────────────────────────────
echo "→ Waiting for server to be ready ..."
for i in $(seq 1 180); do
    if curl -sf --max-time 1 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "→ Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited during startup."
        if lsof -ti ":${PORT}" >/dev/null 2>&1; then
            echo "       Port ${PORT} is in use. Run: ./2_start_mlx.sh restart"
        fi
        exit 1
    fi
    sleep 1
done

# ── MTP smoke / verify (rollback, wedged BatchGenerator, visible confirmation) ─
if [ "$ENABLE_MTP" = true ]; then
    if [ "$VERIFY_MTP" = true ]; then
        if ! print_mtp_verification; then
            echo "ERROR: MTP verification failed. Check server logs above."
            echo "       Try: ./apply_local_patches.sh && ./2_start_mlx.sh restart"
            echo "       Or: ./2_start_mlx.sh restart"
            kill -TERM "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    else
        echo "→ MTP smoke test ..."
        if ! curl -sf --max-time 45 -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0}" \
            | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]['message']['content']" >/dev/null 2>&1; then
            echo "ERROR: MTP smoke test failed. Check server logs above."
            echo "       Try: ./apply_local_patches.sh && ./2_start_mlx.sh restart"
            echo "       Or temporarily: ./2_start_mlx.sh restart"
            kill -TERM "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        echo "→ MTP smoke test passed"
    fi
fi

echo ""
echo "============================================================"
if [ "$ENABLE_MTP" = true ]; then
    echo "  READY — mlx_vlm.server + Gemma 4 MTP (block size ${DRAFT_BLOCK_SIZE})"
else
    echo "  READY — mlx_lm.server (no MTP)"
fi
echo "============================================================"
echo "  OpenAI API:   http://localhost:$PORT/v1"
echo "  Model ID:     $MODEL_ID"
echo ""
echo "  kilo.json model:  openai-compatible/$MODEL_ID"
echo ""
echo "  curl test:"
echo "    curl http://localhost:$PORT/v1/models"
if [ "$ENABLE_MTP" = true ]; then
    echo "  MTP verify (anytime):"
    echo "    ./verify_mtp.sh"
    echo ""
    echo "  Log note:"
    echo "    mlx_vlm.server normally prints only cache/model reuse + HTTP 200 lines per request."
    echo "    Use ./verify_mtp.sh or response.timings.predicted_per_second to confirm MTP speed."
fi
echo "============================================================"
echo ""

wait "$SERVER_PID" 2>/dev/null
cleanup