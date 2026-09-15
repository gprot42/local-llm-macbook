#!/usr/bin/env bash
# =============================================================================
# 2_start_server.sh — Serve YuE2-3B over HTTP (not a Kilo chat model)
#
# API at http://127.0.0.1:8088
#   GET  /health
#   GET  /v1/models
#   POST /generate     JSON song request → artifacts + audio path
#   POST /plan         JSON song request → ABC plan only
#
# Port 8088 sits beside Muse Glimmer :8087, Gemma/Diffusion :8080,
# DeepSeek MLX :8082, ds4 :8083, mtplx :8765/:8766.
#
# Options:
#   --port PORT           Public API port (default: 8088)
#   --host HOST           Bind host (default: 127.0.0.1)
#   --precision P         bf16 | 8bit | 4bit
#   --lazy                Load weights on first generate (faster process start)
#   --no-require-ac       Allow battery
#   --harness-gate        Run test_harness.py --gate after ready (default: on)
#   --no-harness-gate     Skip post-start harness gate
#   restart               Stop anything on the port, then start fresh
#   stop                  Stop process(es) on the port
#   status                Show whether the API is healthy
#   --help, -h
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_yue2_common.sh"

PORT="${DEFAULT_PORT}"
HOST="${DEFAULT_HOST}"
PRECISION_OVERRIDE=""
LAZY=false
REQUIRE_AC=true
HARNESS_GATE=true
DO_RESTART=false
DO_STOP=false
DO_STATUS=false

i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,26p' "$0" | sed 's/^# //;s/^#//'
            exit 0
            ;;
        restart) DO_RESTART=true; ((i+=1)) ;;
        stop) DO_STOP=true; ((i+=1)) ;;
        status) DO_STATUS=true; ((i+=1)) ;;
        --port) PORT="${args[$((i+1))]:-$PORT}"; ((i+=2)) ;;
        --host) HOST="${args[$((i+1))]:-$HOST}"; ((i+=2)) ;;
        --precision) PRECISION_OVERRIDE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --lazy) LAZY=true; ((i+=1)) ;;
        --no-require-ac) REQUIRE_AC=false; ((i+=1)) ;;
        --harness-gate) HARNESS_GATE=true; ((i+=1)) ;;
        --no-harness-gate) HARNESS_GATE=false; ((i+=1)) ;;
        *)
            echo "ERROR: unknown argument '${args[$i]}'" >&2
            exit 1
            ;;
    esac
done

api_healthy() {
    curl -sf --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1
}

describe_port_holder() {
    local pids
    pids="$(port_pids "${PORT}")"
    if [[ -z "${pids}" ]]; then
        echo "(none)"
        return
    fi
    # shellcheck disable=SC2086
    ps -p $pids -o pid=,command= 2>/dev/null | sed 's/^/  /' || echo "  pid(s): ${pids//$'\n'/ }"
}

if [[ "${DO_STATUS}" == true ]]; then
    echo "=== YuE2 status (port ${PORT}) ==="
    pids="$(port_pids "${PORT}")"
    if [[ -z "${pids}" ]]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if api_healthy; then
        echo "→ Health:    OK  http://${HOST}:${PORT}/health"
        curl -sf --max-time 2 "http://${HOST}:${PORT}/health" || true
        echo
        echo "→ API:       http://${HOST}:${PORT}"
        exit 0
    fi
    echo "→ Health:    FAIL — something is bound but /health failed"
    echo "  Fix:       ./2_start_server.sh restart"
    exit 1
fi

if [[ "${DO_STOP}" == true ]]; then
    if [[ -z "$(port_pids "${PORT}")" ]]; then
        echo "→ No process on port ${PORT}"
        exit 0
    fi
    stop_server_on_port "${PORT}"
    echo "→ Stopped (port ${PORT})"
    exit 0
fi

if [[ "${DO_RESTART}" == true ]]; then
    stop_server_on_port "${PORT}"
fi

if [[ -n "$(port_pids "${PORT}")" ]]; then
    echo "ERROR: port ${PORT} is already in use:"
    describe_port_holder
    echo "  Fix: ./2_start_server.sh restart"
    exit 1
fi

load_yue2_config
if [[ -n "${PRECISION_OVERRIDE}" ]]; then
    PRECISION="${PRECISION_OVERRIDE}"
fi

python3 "${VALIDATE_MODEL}" "${MODEL_DIR}" --vae "${VAE_PATH}" --paths "${PATHS_FILE}"

SERVER_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down YuE2 server ..."
    [[ -n "${SERVER_PID}" ]] && kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${SERVER_PID}" ]] && kill -KILL "${SERVER_PID}" 2>/dev/null || true
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM HUP

SERVE_CMD=(
    "${SCRIPT_DIR}/yue2_server.py"
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL_DIR}"
    --vae "${VAE_PATH}"
    --precision "${PRECISION}"
    --alias "${MODEL_ALIAS}"
    --outputs "${OUTPUTS_DIR}"
)
if [[ "${LAZY}" == true ]]; then
    SERVE_CMD+=(--lazy)
fi
if [[ "${REQUIRE_AC}" == true ]]; then
    SERVE_CMD+=(--require-ac)
fi

echo "=== YuE2-3B — HTTP server ==="
echo "→ Model:      ${MODEL_DIR}"
echo "→ VAE:        ${VAE_PATH}"
echo "→ Precision:  ${PRECISION}"
echo "→ Port:       ${PORT}"
echo "→ API:        http://${HOST}:${PORT}"
echo "→ Load:       $([ "${LAZY}" == true ] && echo lazy || echo preload)"
echo ""

mkdir -p "${OUTPUTS_DIR}"
prepare_lyra_env
(
    cd "${ENGINE_DIR}"
    uv run python "${SERVE_CMD[@]}"
) &
SERVER_PID=$!

echo "→ Waiting for server to be ready ..."
READY=false
# Preload can take several minutes on first Metal compile.
for i in $(seq 1 600); do
    if api_healthy; then
        echo "→ Server ready after ${i}s"
        READY=true
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: yue2_server.py exited unexpectedly. Check the output above." >&2
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" != true ]]; then
    echo "ERROR: server did not become ready within 600s." >&2
    echo "       Re-run: ./2_start_server.sh restart" >&2
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

echo ""
echo "============================================================"
echo "  READY — YuE2-3B MLX"
echo "============================================================"
echo "  API:          http://${HOST}:${PORT}"
echo "  Model ID:     ${MODEL_ALIAS}"
echo "  Health:       curl http://${HOST}:${PORT}/health"
echo "  Generate:     curl -X POST http://${HOST}:${PORT}/generate \\"
echo "                  -H 'Content-Type: application/json' \\"
echo "                  -d @examples/quickstart.json"
echo "  CLI:          ./2_generate.sh"
echo "  harness:      python3 test_harness.py --gate"
echo "  Note:         not a Kilo chat-completions model"
echo "============================================================"

if [[ "${HARNESS_GATE}" == true ]]; then
    echo ""
    echo "→ Post-start harness gate ..."
    if python3 "${SCRIPT_DIR}/test_harness.py" --gate \
        --base "http://${HOST}:${PORT}" \
        --model "${MODEL_ALIAS}"; then
        echo "→ Harness gate: PASS"
    else
        echo "→ Harness gate: FAIL"
        echo "  Server stays up. Re-run: python3 test_harness.py --gate"
    fi
fi

echo ""
echo "→ Server running (Ctrl+C to stop)"
wait "${SERVER_PID}"
