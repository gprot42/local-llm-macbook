#!/usr/bin/env bash
# =============================================================================
# 2_start_ollama.sh — Serve Ornith-1.0-35B GGUF via Ollama
#
# OpenAI-compatible API: http://127.0.0.1:18082/v1  (OpenCode/Kilo → proxy → Ollama :11434)
# Use 127.0.0.1 — not localhost (macOS resolves localhost to ::1; proxy is IPv4).
#
# Registers the local GGUF as an Ollama model (if missing), then starts the
# ornith_tool_proxy.py — repairs tool calls for OpenCode / Kilo.
#
# Requires:
#   brew install ollama
#   ./1_setup_download.sh   (GGUF weights in weights/)
#
# Options:
#   --ollama-port N   Ollama HTTP port (default: 11434)
#   --proxy-port N    Harness proxy port (default: 18082)
#   --ctx-size N      Context window passed to Ollama (default: 131072)
#   --quant q4|q5|q6|q8   Override quant (default q8; reads .ornith35b_config)
#   --temp T          Sampling temperature (default: 0.6 per Ornith model card)
#   --greedy          Shorthand for --temp 0 (deterministic coding)
#   --no-proxy        Talk to Ollama directly (skip tool-call repair proxy)
#   start             Start proxy in background (default; survives terminal close)
#   foreground        Run proxy in foreground (logs to terminal; Ctrl+C to stop)
#   status            Show proxy + Ollama health
#   stop              Stop harness proxy (Ollama daemon may keep running)
#   restart           Stop proxy, then start again in background
#   --help, -h        Show this help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.ornith35b_config"
MODELFILE="$SCRIPT_DIR/.ollama.Modelfile"
PID_FILE="$SCRIPT_DIR/.ornith_proxy.pid"
LOG_FILE="$SCRIPT_DIR/.ornith_proxy.log"

OLLAMA_PORT=11434
PROXY_PORT=18082
HOST="0.0.0.0"
CTX_SIZE=131072
TEMP="0.6"
TOP_P="0.95"
TOP_K=20
QUANT_OVERRIDE=""
USE_PROXY=true
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
RUN_MODE="daemon"   # daemon | foreground

for arg in "$@"; do
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        sed -n '3,27p' "$0" | sed 's/^# \?//'
        exit 0
    }
done

i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        start) RUN_MODE="daemon"; ((i+=1)) ;;
        foreground) RUN_MODE="foreground"; ((i+=1)) ;;
        status) DO_STATUS=true; ((i+=1)) ;;
        stop) DO_STOP=true; ((i+=1)) ;;
        restart) DO_RESTART=true; RUN_MODE="daemon"; ((i+=1)) ;;
        --ollama-port) OLLAMA_PORT="${args[$((i+1))]:-$OLLAMA_PORT}"; ((i+=2)) ;;
        --proxy-port) PROXY_PORT="${args[$((i+1))]:-$PROXY_PORT}"; ((i+=2)) ;;
        --ctx-size) CTX_SIZE="${args[$((i+1))]:-$CTX_SIZE}"; ((i+=2)) ;;
        --quant) QUANT_OVERRIDE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --temp) TEMP="${args[$((i+1))]:-$TEMP}"; ((i+=2)) ;;
        --greedy) TEMP="0"; ((i+=1)) ;;
        --no-proxy) USE_PROXY=false; ((i+=1)) ;;
        *) ((i+=1)) ;;
    esac
done

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
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
    fi
}

proxy_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(tr -d '[:space:]' <"$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    lsof -ti ":${PROXY_PORT}" 2>/dev/null | head -1 || true
}

proxy_healthy() {
    curl -sf "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null 2>&1
}

clear_proxy_state() {
    rm -f "$PID_FILE"
}

ollama_api_ok() {
    curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null 2>&1
}

ensure_ollama_running() {
    if ollama_api_ok; then
        echo "→ Ollama already running on :${OLLAMA_PORT}"
        return 0
    fi
    if ! command -v ollama >/dev/null 2>&1; then
        echo "ERROR: ollama not found. Run: brew install ollama"
        exit 1
    fi
    echo "→ Starting Ollama on :${OLLAMA_PORT}..."
    OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" ollama serve >/dev/null 2>&1 &
    local tries=0
    while ! ollama_api_ok; do
        sleep 1
        tries=$((tries + 1))
        if [[ $tries -ge 15 ]]; then
            echo "ERROR: Ollama failed to start on :${OLLAMA_PORT}"
            exit 1
        fi
    done
    echo "→ Ollama ready"
}

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
else
    HF_REPO="deepreinforce-ai/Ornith-1.0-35B-GGUF"
    QUANT="q8"
    GGUF_FILE="ornith-1.0-35b-Q8_0.gguf"
    MODEL_PATH="$SCRIPT_DIR/weights/$GGUF_FILE"
    MODEL_ID="ornith-1.0-35b-q8"
fi

if [[ -n "${QUANT_OVERRIDE}" ]]; then
    case "${QUANT_OVERRIDE}" in
        q4) GGUF_FILE="ornith-1.0-35b-Q4_K_M.gguf"; MODEL_ID="ornith-1.0-35b-q4" ;;
        q5) GGUF_FILE="ornith-1.0-35b-Q5_K_M.gguf"; MODEL_ID="ornith-1.0-35b-q5" ;;
        q6) GGUF_FILE="ornith-1.0-35b-Q6_K.gguf";   MODEL_ID="ornith-1.0-35b-q6" ;;
        q8) GGUF_FILE="ornith-1.0-35b-Q8_0.gguf";   MODEL_ID="ornith-1.0-35b-q8" ;;
        *)  echo "ERROR: unknown quant '${QUANT_OVERRIDE}'. Use q4, q5, q6, or q8."; exit 1 ;;
    esac
    MODEL_PATH="$SCRIPT_DIR/weights/$GGUF_FILE"
fi

if [[ "$DO_STATUS" == true ]]; then
    echo "=== Ornith harness status ==="
    if ollama_api_ok; then
        echo "→ Ollama:    running on :${OLLAMA_PORT}"
    else
        echo "→ Ollama:    not responding on :${OLLAMA_PORT}"
    fi
    local_pid="$(proxy_pid)"
    if [[ -n "$local_pid" ]] && proxy_healthy; then
        echo "→ Proxy:     running (pid ${local_pid}) on :${PROXY_PORT}"
        echo "→ API:       http://127.0.0.1:${PROXY_PORT}/v1"
    elif [[ -n "$local_pid" ]]; then
        echo "→ Proxy:     pid ${local_pid} on :${PROXY_PORT} but /healthz failed"
    else
        echo "→ Proxy:     not running on :${PROXY_PORT}"
    fi
    exit 0
fi

if [[ "$DO_STOP" == true ]]; then
    if lsof -ti ":${PROXY_PORT}" >/dev/null 2>&1; then
        stop_server_on_port "$PROXY_PORT"
        echo "→ Harness proxy stopped (port ${PROXY_PORT})"
    else
        echo "→ No harness proxy running on port ${PROXY_PORT}"
    fi
    clear_proxy_state
    echo "→ Ollama daemon on :${OLLAMA_PORT} left running (use 'ollama stop' per model if needed)"
    exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama not found. Run: brew install ollama"
    exit 1
fi

VALIDATE="$SCRIPT_DIR/validate_model.py"
echo "→ Validating model weights..."
if ! python3 "$VALIDATE" "$MODEL_PATH"; then
    echo ""
    echo "ERROR: model weights failed validation: $MODEL_PATH"
    echo "       Run: ./1_setup_download.sh ${QUANT_OVERRIDE:-q8}"
    exit 1
fi
echo ""

if [[ "$USE_PROXY" == true ]]; then
    if [[ "$DO_RESTART" == true ]]; then
        if lsof -ti ":${PROXY_PORT}" >/dev/null 2>&1; then
            echo "→ Stopping existing harness proxy on :${PROXY_PORT}..."
            stop_server_on_port "$PROXY_PORT"
            clear_proxy_state
        fi
    elif proxy_healthy; then
        echo "→ Harness proxy already running on :${PROXY_PORT} (pid $(proxy_pid))"
        echo "→ API:       http://127.0.0.1:${PROXY_PORT}/v1"
        echo "→ Use './2_start_ollama.sh restart' to replace it"
        exit 0
    elif lsof -ti ":${PROXY_PORT}" >/dev/null 2>&1; then
        echo "→ Stale process on :${PROXY_PORT} (not healthy) — stopping..."
        stop_server_on_port "$PROXY_PORT"
        clear_proxy_state
    fi
fi

ensure_ollama_running

# Always materialize desired Modelfile so --temp / --ctx-size take effect on restart.
DESIRED_MODELFILE="$(cat <<EOF
FROM ${MODEL_PATH}
PARAMETER num_ctx ${CTX_SIZE}
PARAMETER temperature ${TEMP}
PARAMETER top_p ${TOP_P}
PARAMETER top_k ${TOP_K}
PARAMETER repeat_penalty 1.0
EOF
)"
NEED_CREATE=false
if ! ollama show "${MODEL_ID}" >/dev/null 2>&1; then
    NEED_CREATE=true
    echo "→ Registering Ollama model '${MODEL_ID}' from GGUF..."
elif [[ ! -f "$MODELFILE" ]] || [[ "$(cat "$MODELFILE")" != "$DESIRED_MODELFILE" ]]; then
    NEED_CREATE=true
    echo "→ Modelfile params changed (ctx/temp/sampling) — recreating '${MODEL_ID}'..."
else
    echo "→ Ollama model '${MODEL_ID}' already registered (params unchanged)"
fi
if [[ "$NEED_CREATE" == true ]]; then
    printf '%s\n' "$DESIRED_MODELFILE" >"$MODELFILE"
    ollama create "$MODEL_ID" -f "$MODELFILE"
fi

UPSTREAM="http://127.0.0.1:${OLLAMA_PORT}/v1"

if [[ "$USE_PROXY" == true ]]; then
    PROXY_SCRIPT="$SCRIPT_DIR/ornith_tool_proxy.py"
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "ERROR: proxy not found: $PROXY_SCRIPT"
        exit 1
    fi

    PYTHON="${SCRIPT_DIR}/venv/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        PYTHON="python3"
    fi
    if ! "$PYTHON" -c "import httpx, fastapi, uvicorn" 2>/dev/null; then
        echo "→ Installing proxy dependencies into venv (httpx, fastapi, uvicorn)..."
        if [[ -x "${SCRIPT_DIR}/venv/bin/pip" ]]; then
            "${SCRIPT_DIR}/venv/bin/pip" install -q httpx fastapi uvicorn
        else
            "$PYTHON" -m pip install -q httpx fastapi uvicorn
        fi
    fi

    echo ""
    echo "=== Ornith-1.0-35B — Ollama + agent proxy ==="
    echo "→ Model:     $MODEL_PATH"
    echo "→ Ollama:    ${MODEL_ID} @ http://127.0.0.1:${OLLAMA_PORT}"
    echo "→ API:       http://127.0.0.1:${PROXY_PORT}/v1  (OpenCode / Kilo — use 127.0.0.1)"
    echo "→ Context:   ${CTX_SIZE} tokens"
    echo "→ Sampling:  temp=${TEMP}, top_p=${TOP_P}, top_k=${TOP_K}"
    echo ""
    echo "  OpenCode:  ./install-opencode-json.sh  then open OpenCode Desktop"
    echo "  Kilo Code: cd /your/project && kilo"
    echo "  Alt:       ollama run hf.co/deepreinforce-ai/Ornith-1.0-35B-GGUF"
    echo ""

    if [[ "$RUN_MODE" == "foreground" ]]; then
        echo "→ Foreground mode — Ctrl+C stops the proxy"
        echo ""
        exec "$PYTHON" "$PROXY_SCRIPT" \
            --host "$HOST" \
            --port "$PROXY_PORT" \
            --upstream "$UPSTREAM"
    fi

    echo "→ Starting proxy in background (logs: ${LOG_FILE})"
    nohup "$PYTHON" "$PROXY_SCRIPT" \
        --host "$HOST" \
        --port "$PROXY_PORT" \
        --upstream "$UPSTREAM" \
        >>"$LOG_FILE" 2>&1 &
    PROXY_PID=$!
    echo "$PROXY_PID" >"$PID_FILE"
    disown "$PROXY_PID" 2>/dev/null || true

    echo "→ Waiting for proxy on :${PROXY_PORT} ..."
    ready=false
    for _ in $(seq 1 30); do
        if proxy_healthy; then
            ready=true
            break
        fi
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo "ERROR: proxy exited during startup. Last log lines:"
            tail -20 "$LOG_FILE" 2>/dev/null || true
            clear_proxy_state
            exit 1
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        echo "ERROR: proxy did not become healthy within 30s. Last log lines:"
        tail -20 "$LOG_FILE" 2>/dev/null || true
        stop_server_on_port "$PROXY_PORT"
        clear_proxy_state
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  READY — proxy pid ${PROXY_PID} on :${PROXY_PORT}"
    echo "  API:  http://127.0.0.1:${PROXY_PORT}/v1"
    echo "  Logs: ${LOG_FILE}"
    echo "  Stop: ./2_start_ollama.sh stop"
    echo "============================================================"
    exit 0
fi

echo ""
echo "=== Ornith-1.0-35B — Ollama direct (no proxy) ==="
echo "→ Model:     $MODEL_PATH"
echo "→ Ollama:    ${MODEL_ID} @ http://127.0.0.1:${OLLAMA_PORT}"
echo "→ API:       http://127.0.0.1:${OLLAMA_PORT}/v1"
echo "→ Context:   ${CTX_SIZE} tokens"
echo ""
echo "  Point OpenCode/Kilo baseURL at Ollama directly when using --no-proxy."
echo ""