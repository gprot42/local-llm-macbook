#!/usr/bin/env bash
# =============================================================================
# 2_start_dflash.sh — Serve Qwen3.5-122B + DFlash via dflash-mlx OpenAI server
#
# API: http://127.0.0.1:8086/v1
# Model ID: qwen3.5-122b-a10b-dflash
#
# Loads ~65 GB MLX target + ~1.5 GB draft. Do not co-load DeepSeek / another
# 70+ GB model on 128 GB machines.
#
# Usage:
#   ./2_start_dflash.sh
#   ./2_start_dflash.sh restart | stop | status
#   ./2_start_dflash.sh --port 8087
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.dflash_122b_config"
LOG_FILE="$SCRIPT_DIR/.dflash_server.log"
PID_FILE="$SCRIPT_DIR/.dflash_server.pid"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: missing $CONFIG_FILE — run ./1_setup_download.sh first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

PORT="${PORT:-8086}"
HOST="${HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-qwen3.5-122b-a10b-dflash}"
DO_RESTART=false
DO_STOP=false
DO_STATUS=false

while [ $# -gt 0 ]; do
    case "$1" in
        restart) DO_RESTART=true ;;
        stop) DO_STOP=true ;;
        status) DO_STATUS=true ;;
        --port) PORT="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --help|-h)
            awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
    shift
done

stop_server_on_port() {
    local port="$1"
    local pids
    # Only kill LISTENers — lsof -ti :port also matches clients (e.g. Kilo)
    # with ESTABLISHED connections and would murder the IDE agent.
    pids="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [ -z "$pids" ] && [ -f "$PID_FILE" ]; then
        local file_pid
        file_pid="$(tr -d '[:space:]' <"$PID_FILE" 2>/dev/null || true)"
        if [ -n "$file_pid" ] && kill -0 "$file_pid" 2>/dev/null; then
            pids="$file_pid"
        fi
    fi
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
}

if [ "$DO_STATUS" = true ]; then
    if lsof -ti ":${PORT}" >/dev/null 2>&1; then
        echo "DFlash server listening on ${HOST}:${PORT}"
        curl -s "http://${HOST}:${PORT}/health" || true
        echo
        curl -s "http://${HOST}:${PORT}/v1/models" || true
        echo
    else
        echo "Not running on port ${PORT}"
        exit 1
    fi
    exit 0
fi

if [ "$DO_STOP" = true ]; then
    stop_server_on_port "$PORT"
    echo "Stopped."
    exit 0
fi

if [ "$DO_RESTART" = true ]; then
    stop_server_on_port "$PORT"
fi

if lsof -ti ":${PORT}" >/dev/null 2>&1; then
    echo "Port ${PORT} already in use. Use: ./2_start_dflash.sh restart|stop|status" >&2
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "ERROR: venv missing — run ./1_setup_download.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/venv/bin/activate"

if [ ! -f "$TARGET_MODEL/config.json" ]; then
    echo "ERROR: TARGET_MODEL invalid: $TARGET_MODEL" >&2
    exit 1
fi
if [ ! -f "$DRAFT_MODEL/model.safetensors" ]; then
    echo "ERROR: DRAFT_MODEL invalid: $DRAFT_MODEL" >&2
    exit 1
fi

echo "→ Starting dflash-mlx OpenAI server"
echo "    target: $TARGET_MODEL"
echo "    draft:  $DRAFT_MODEL"
echo "    api:    http://${HOST}:${PORT}/v1  model_id=${MODEL_ID}"
echo "    log:    $LOG_FILE"
echo "    (first load of ~65 GB target can take several minutes)"

# Agent-friendly defaults:
#  - default max_tokens when Kilo omits it (tool JSON needs headroom)
#  - hard ceiling so a huge client budget cannot run for hours
#  - generation is serialized inside the server (MLX is not multi-request safe)
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-4096}"
MAX_TOKENS_CEILING="${MAX_TOKENS_CEILING:-8192}"

nohup dflash-mlx-openai-server \
    --host "$HOST" \
    --port "$PORT" \
    --model-id "$MODEL_ID" \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    --max-tokens-ceiling "$MAX_TOKENS_CEILING" \
    --log-level INFO \
    >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

echo "PID $(cat "$PID_FILE") — tail -f $LOG_FILE"
echo "    max_tokens default=${DEFAULT_MAX_TOKENS} ceiling=${MAX_TOKENS_CEILING}"
echo "Health (after load): curl -s http://${HOST}:${PORT}/health"
