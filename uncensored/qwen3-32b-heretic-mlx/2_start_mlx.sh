#!/usr/bin/env bash
# =============================================================================
# 2_start_mlx.sh — Serve Qwen3-32B Heretic via mlx_lm.server (OpenAI API)
#
# Uncensored / abliterated Qwen3-32B (MLX 5-bit, ~22.5 GB).
# API: http://127.0.0.1:8084/v1
#
# Usage:
#   ./2_start_mlx.sh
#   ./2_start_mlx.sh restart
#   ./2_start_mlx.sh stop
#   ./2_start_mlx.sh status
#   ./2_start_mlx.sh --port 8085
#   ./2_start_mlx.sh --temp 0.7 --top-p 0.9
#   ./2_start_mlx.sh --help
#
# Commands:
#   (default)   Start the OpenAI-compatible server (foreground)
#   restart     Stop anything on the port, then start
#   stop        Stop process(es) on the port
#   status      Show whether the server is healthy on the port
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.qwen3_heretic_config"

PORT=8084
HOST="127.0.0.1"
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
TEMP="0.6"
TOP_P="0.95"
TOP_K="20"
MAX_TOKENS=""
CHAT_TEMPLATE_ARGS=""

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
        --temp) TEMP="${args[$((i+1))]:-$TEMP}"; i=$((i + 2)) ;;
        --top-p) TOP_P="${args[$((i+1))]:-$TOP_P}"; i=$((i + 2)) ;;
        --top-k) TOP_K="${args[$((i+1))]:-$TOP_K}"; i=$((i + 2)) ;;
        --max-tokens) MAX_TOKENS="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --chat-template-args) CHAT_TEMPLATE_ARGS="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        *)
            echo "ERROR: Unknown argument: ${args[$i]}"
            echo "       Run ./2_start_mlx.sh --help"
            exit 1
            ;;
    esac
done

# ── status / stop (no model load required) ────────────────────────────────────
if [[ "${DO_STATUS}" == true ]]; then
    echo "=== Qwen3-32B Heretic MLX status (port ${PORT}) ==="
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
    echo "             (still loading ~22 GB, or not this server)"
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
    HF_REPO="Wwayu/Qwen3-32B-heretic-mlx-5Bit"
    MODEL_DIR="$SCRIPT_DIR/qwen3-32b-heretic-mlx-5bit"
    MODEL_ID="qwen3-32b-heretic-mlx-5bit"
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

if ! "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
    echo "ERROR: mlx_lm not installed in venv."
    echo "       Fix:  ./1_setup_download.sh --skip-download"
    exit 1
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

# Prefer python -m for reliable venv resolution; fall back to console script.
if "$VENV_PY" -c "import mlx_lm.server" 2>/dev/null; then
    SERVER_CMD=("$VENV_PY" -m mlx_lm.server)
elif [[ -x "$SCRIPT_DIR/venv/bin/mlx_lm.server" ]]; then
    SERVER_CMD=("$SCRIPT_DIR/venv/bin/mlx_lm.server")
else
    echo "ERROR: mlx_lm.server not available. Run ./1_setup_download.sh --skip-download"
    exit 1
fi

SERVER_ARGS=(
    --model "$MODEL_PATH"
    --port "$PORT"
    --host "$HOST"
)

# Optional sampling defaults (clients may still override per-request).
# mlx_lm.server accepts --temp / --top-p on recent versions; probe carefully.
if "$VENV_PY" -c "import inspect, mlx_lm.server as s; print(inspect.signature(s.main) if hasattr(s,'main') else '')" 2>/dev/null | grep -q temp; then
    SERVER_ARGS+=(--temp "$TEMP" --top-p "$TOP_P")
fi

if [[ -n "${MAX_TOKENS}" ]]; then
    SERVER_ARGS+=(--max-tokens "$MAX_TOKENS")
fi

if [[ -n "${CHAT_TEMPLATE_ARGS}" ]]; then
    SERVER_ARGS+=(--chat-template-args "$CHAT_TEMPLATE_ARGS")
fi

echo ""
echo "=== Qwen3-32B Heretic — mlx_lm.server ==="
echo "→ Model:    $MODEL_PATH"
echo "→ Model ID: $MODEL_ID  (path basename; clients may send any id)"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo "→ Sampling: temp=${TEMP}, top_p=${TOP_P}, top_k=${TOP_K} (Qwen recommended; client can override)"
echo "→ HF source: ${HF_REPO:-Wwayu/Qwen3-32B-heretic-mlx-5Bit}"
echo "→ Base:      igriv/Qwen3-32B-heretic (Heretic abliteration of Qwen/Qwen3-32B)"
echo ""
echo "  ${SERVER_CMD[*]} ${SERVER_ARGS[*]}"
echo ""
echo "  Wait until the server prints that it is listening (~30–90s load),"
echo "  then point Kilo at http://127.0.0.1:${PORT}/v1  (use 127.0.0.1, not localhost)"
echo ""
echo "  Smoke test:"
echo "    curl -s http://127.0.0.1:${PORT}/v1/models | python3 -m json.tool"
echo ""

exec "${SERVER_CMD[@]}" "${SERVER_ARGS[@]}"
