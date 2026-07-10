#!/usr/bin/env bash
# =============================================================================
# 2_start_mlx.sh — Serve DeepSeek via mlx_lm.server (OpenAI-compatible API)
#
# DeepSeek V4 Flash 2bit-DQ (strong coding/reasoning, heavy — ~97 GB, 128 GB Mac)
# Best: hard coding/agent work in Kilo when quality beats speed; long-context
#       reasoning; SWE-style multi-file tasks; local substitute for a strong
#       cloud coder when you have the RAM
# OK:   day-to-day coding, refactors, chat with <think>; agent loops if you can
#       wait for load/prefill; general reasoning that is not ultra-latency-sensitive
# Bad:  fast iteration / snappy tool loops (prefer Qwen 3.6); uncensored /
#       low-refusal needs (use Heretic Gemma); machines under ~128 GB; treating
#       2-bit MoE as cloud-frontier on huge greenfield apps
#
# Default model: deepseek-v4-flash-2bit-dq  (~97 GB, fits 128 GB M5 Max)
# API: http://localhost:8082/v1
#
# Usage:
#   ./2_start_mlx.sh
#   ./2_start_mlx.sh restart
#   ./2_start_mlx.sh stop
#   ./2_start_mlx.sh status
#   ./2_start_mlx.sh --model r1-32b
#   ./2_start_mlx.sh --port 8083
#   ./2_start_mlx.sh --prompt-cache-bytes 8589934592
#   ./2_start_mlx.sh --max-tokens 8192
#   ./2_start_mlx.sh --help
#
# Commands:
#   (default)   Start the OpenAI-compatible server
#   restart     Stop anything on the port, then start
#   stop        Stop process(es) on the port
#   status      Show whether the server is healthy on the port
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.deepseek_config"

PORT=8082
HOST="127.0.0.1"
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
MODEL_OVERRIDE=""
PROMPT_CACHE_BYTES=""
PROMPT_CACHE_SIZE=""
MAX_TOKENS=""
TEMP="1.0"
TOP_P="1.0"
LOG_LEVEL="INFO"

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

port_pids() {
    lsof -ti ":${PORT}" 2>/dev/null || true
}

describe_port_holder() {
    local pids
    pids="$(port_pids)"
    if [ -z "$pids" ]; then
        echo "(none)"
        return
    fi
    # shellcheck disable=SC2086
    ps -p $pids -o pid=,command= 2>/dev/null | sed 's/^/  /' || echo "  pid(s): ${pids//$'\n'/ }"
}

server_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
        || curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

for arg in "$@"; do
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
        exit 0
    }
done

i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        --port) PORT="${args[$((i+1))]:-$PORT}"; i=$((i + 2)) ;;
        --host) HOST="${args[$((i+1))]:-$HOST}"; i=$((i + 2)) ;;
        --model) MODEL_OVERRIDE="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --prompt-cache-bytes) PROMPT_CACHE_BYTES="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --prompt-cache-size) PROMPT_CACHE_SIZE="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --max-tokens) MAX_TOKENS="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --temp) TEMP="${args[$((i+1))]:-$TEMP}"; i=$((i + 2)) ;;
        --top-p) TOP_P="${args[$((i+1))]:-$TOP_P}"; i=$((i + 2)) ;;
        --log-level) LOG_LEVEL="${args[$((i+1))]:-$LOG_LEVEL}"; i=$((i + 2)) ;;
        # Removed from mlx_lm.server — map common old flag to a helpful error.
        --max-kv-size)
            echo "ERROR: --max-kv-size is no longer supported by mlx_lm.server."
            echo "       Use --prompt-cache-bytes BYTES to limit prompt KV cache memory,"
            echo "       or --max-tokens N for default completion length."
            echo "  e.g. ./2_start_mlx.sh --prompt-cache-bytes 8589934592"
            exit 1
            ;;
        *)
            echo "ERROR: Unknown argument: ${args[$i]}"
            echo "       Run ./2_start_mlx.sh --help"
            exit 1
            ;;
    esac
done

# ── status / stop (no model load required) ────────────────────────────────────
if [[ "${DO_STATUS}" == true ]]; then
    echo "=== DeepSeek MLX status (port ${PORT}) ==="
    pids="$(port_pids)"
    if [ -z "$pids" ]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if server_healthy; then
        echo "→ Health:    OK  http://127.0.0.1:${PORT}/v1/models"
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")"
        echo "→ Model ID:  ${model_id}"
        echo "→ API:       http://127.0.0.1:${PORT}/v1"
        exit 0
    fi
    echo "→ Health:    FAIL — port bound but /v1/models (and /health) failed"
    echo "             (still loading ~97 GB, or not this server)"
    echo "  Fix:       ./2_start_mlx.sh restart"
    exit 1
fi

if [[ "${DO_STOP}" == true ]]; then
    if [ -z "$(port_pids)" ]; then
        echo "→ No process on port ${PORT}"
        exit 0
    fi
    stop_server_on_port "${PORT}"
    echo "→ Stopped (port ${PORT})"
    exit 0
fi

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
else
    HF_REPO="mlx-community/DeepSeek-V4-Flash-2bit-DQ"
    MODEL_DIR="$SCRIPT_DIR/deepseek-v4-flash-2bit-dq"
    MODEL_ID="deepseek-v4-flash-2bit-dq"
    MODEL_CHOICE="v4-flash"
fi

if [[ -n "${MODEL_OVERRIDE}" ]]; then
    case "${MODEL_OVERRIDE}" in
        v4-flash)
            HF_REPO="mlx-community/DeepSeek-V4-Flash-2bit-DQ"
            MODEL_DIR="$SCRIPT_DIR/deepseek-v4-flash-2bit-dq"
            MODEL_ID="deepseek-v4-flash-2bit-dq"
            MODEL_CHOICE="v4-flash"
            ;;
        r1-32b)
            HF_REPO="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
            MODEL_DIR="$SCRIPT_DIR/deepseek-r1-distill-qwen-32b-4bit"
            MODEL_ID="deepseek-r1-distill-qwen-32b-4bit"
            MODEL_CHOICE="r1-32b"
            ;;
        *)
            if [[ -d "${MODEL_OVERRIDE}" ]]; then
                MODEL_DIR="${MODEL_OVERRIDE}"
                MODEL_ID="$(basename "${MODEL_OVERRIDE}")"
            else
                echo "ERROR: Unknown --model '${MODEL_OVERRIDE}'."
                echo "       Use v4-flash, r1-32b, or a local model directory path."
                exit 1
            fi
            ;;
    esac
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

# Re-apply local patches (2bit-DQ SwitchGLU + transformers 5.13 register).
if [[ -f "$SCRIPT_DIR/apply_local_patches.sh" ]]; then
    bash "$SCRIPT_DIR/apply_local_patches.sh" || true
fi

# DeepSeek-V4 needs a forked mlx-lm (PyPI 0.31.x has no deepseek_v4).
needs_v4=false
if [[ "${MODEL_CHOICE:-}" == "v4-flash" ]] || [[ -f "$MODEL_DIR/config.json" ]]; then
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        mt="$("$VENV_PY" -c "import json; print(json.load(open('$MODEL_DIR/config.json')).get('model_type',''))" 2>/dev/null || true)"
        [[ "$mt" == "deepseek_v4" ]] && needs_v4=true
    else
        needs_v4=true
    fi
fi

if [[ "$needs_v4" == true ]]; then
    if ! "$VENV_PY" -c "import importlib; importlib.import_module('mlx_lm.models.deepseek_v4')" 2>/dev/null; then
        echo "ERROR: mlx_lm.models.deepseek_v4 is not available in this venv."
        echo "       PyPI mlx-lm does not support DeepSeek-V4 yet."
        echo "       Fix:  ./1_setup_download.sh --skip-download"
        exit 1
    fi
fi

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Download weights first:"
    echo "  ./1_setup_download.sh"
    exit 1
fi

if ! "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"; then
    echo ""
    echo "ERROR: Local weights incomplete or corrupt — refusing to start."
    echo "       Resume download with: ./1_setup_download.sh"
    exit 1
fi

MODEL_PATH="$MODEL_DIR"
echo "→ Using local weights: $MODEL_DIR"

if [[ "$DO_RESTART" == true ]]; then
    stop_server_on_port "$PORT"
fi

if lsof -ti ":${PORT}" >/dev/null 2>&1; then
    echo "ERROR: Port ${PORT} already in use."
    echo "       ./2_start_mlx.sh status   # what's on the port"
    echo "       ./2_start_mlx.sh restart  # stop then start"
    echo "       ./2_start_mlx.sh stop     # free the port"
    echo "       ./2_start_mlx.sh --port N # use another port"
    exit 1
fi

# Use the simple single-process server. Stock mlx_lm.server crashes with
# Metal GPU timeouts on this 2bit-DQ checkpoint under its threaded design.
OPENAI_SERVER="$SCRIPT_DIR/openai_server.py"
if [[ ! -f "$OPENAI_SERVER" ]]; then
    echo "ERROR: $OPENAI_SERVER not found"
    exit 1
fi

SERVER_ARGS=(
    --model "$MODEL_PATH"
    --model-id "$MODEL_ID"
    --port "$PORT"
    --host "$HOST"
    --temp "$TEMP"
    --top-p "$TOP_P"
    --log-level "$LOG_LEVEL"
    # Agent harness: stop "I'm ready..." loops, no long <think>, larger defaults
    --thinking-mode chat
    --repetition-penalty 1.12
    --frequency-penalty 0.35
    --max-tokens "${MAX_TOKENS:-8192}"
)

if [[ -n "${PROMPT_CACHE_BYTES}" || -n "${PROMPT_CACHE_SIZE}" ]]; then
    echo "→ NOTE: --prompt-cache-* is not used by openai_server.py (ignored)."
fi

echo ""
echo "=== DeepSeek MLX Server (stable OpenAI wrapper) ==="
echo "→ Model:    $MODEL_PATH"
echo "→ Model ID: $MODEL_ID"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo "→ Sampling: temp=${TEMP}, top_p=${TOP_P} (DeepSeek V4 recommended)"
echo "→ Harness:  thinking=chat · rep=1.12 · freq=0.35 · steer+prefill · loop-detector · User/EOS stops"
echo "→ Backend:  openai_server.py (not mlx_lm.server — avoids Metal GPU abort)"
echo "→ max_tokens default: ${MAX_TOKENS:-8192}"
echo ""
echo "  $VENV_PY $OPENAI_SERVER ${SERVER_ARGS[*]}"
echo ""
echo "  Wait until you see 'OpenAI API ready' (~15–40s for ~97 GB load),"
echo "  then point Kilo at http://127.0.0.1:${PORT}/v1  (use 127.0.0.1, not localhost)"
echo ""

exec "$VENV_PY" "$OPENAI_SERVER" "${SERVER_ARGS[@]}"
