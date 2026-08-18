#!/usr/bin/env bash
# =============================================================================
# 2_start_mlx.sh — Serve Muse Glimmer 30B via mlx_vlm.server
#
# OpenAI-compatible API at http://127.0.0.1:8087/v1
# Port 8087 avoids Gemma/Diffusion :8080, DeepSeek MLX :8082, ds4 :8083,
# Qwen3.5 DFlash :8086, and mtplx :8765/:8766.
#
# DFlash is ON by default when the official assistant is present
# (--draft-kind dflash, block size 16).
#
# Options:
#   --port PORT           Public API port (default: 8087)
#   --host HOST           Bind host (default: 127.0.0.1)
#   --model REF           HF repo, local dir, or size token (4bit/5bit/8bit/…)
#   --with-dflash         Force DFlash (error if drafter missing)
#   --no-dflash           Target only
#   --draft-block-size N  DFlash block size (default: 16)
#   --max-tokens N        Server generation cap (default: 8192)
#   --thinking-budget N   Thinking-block cap (default: 2048)
#   --max-kv-size N       KV cache token cap
#   --low-memory          Disable DFlash, cap KV at 8192
#   --harness-gate        Run test_harness.py --gate after ready (default: on)
#   --no-harness-gate     Skip post-start harness gate
#   restart               Stop anything on the port, then start fresh
#   stop                  Stop process(es) on the port
#   status                Show whether the API is healthy
#   --help, -h
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.mlx_config"

PORT=8087
HOST="127.0.0.1"
PROFILE_DFLASH="auto"
DRAFT_BLOCK_SIZE=16
MAX_TOKENS=8192
THINKING_BUDGET=2048
MAX_KV_SIZE=""
LOW_MEMORY=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
HARNESS_GATE=true
MODEL_OVERRIDE=""

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

api_healthy() {
    curl -sf --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1
}

i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,28p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart)   DO_RESTART=true;                          ((i+=1)) ;;
        stop)      DO_STOP=true;                             ((i+=1)) ;;
        status)    DO_STATUS=true;                           ((i+=1)) ;;
        --port)    PORT="${args[$((i+1))]:-$PORT}";          ((i+=2)) ;;
        --host)    HOST="${args[$((i+1))]:-$HOST}";          ((i+=2)) ;;
        --model)   MODEL_OVERRIDE="${args[$((i+1))]:-}";     ((i+=2)) ;;
        --with-dflash|--dflash) PROFILE_DFLASH="on";         ((i+=1)) ;;
        --no-dflash) PROFILE_DFLASH="off";                   ((i+=1)) ;;
        --draft-block-size) DRAFT_BLOCK_SIZE="${args[$((i+1))]:-$DRAFT_BLOCK_SIZE}"; ((i+=2)) ;;
        --max-tokens) MAX_TOKENS="${args[$((i+1))]:-$MAX_TOKENS}"; ((i+=2)) ;;
        --thinking-budget) THINKING_BUDGET="${args[$((i+1))]:-$THINKING_BUDGET}"; ((i+=2)) ;;
        --max-kv-size) MAX_KV_SIZE="${args[$((i+1))]:-$MAX_KV_SIZE}"; ((i+=2)) ;;
        --low-memory) LOW_MEMORY=true;                       ((i+=1)) ;;
        --harness-gate)    HARNESS_GATE=true;                ((i+=1)) ;;
        --no-harness-gate) HARNESS_GATE=false;               ((i+=1)) ;;
        *) ((i+=1)) ;;
    esac
done

if [[ "${DO_STATUS}" == true ]]; then
    echo "=== Muse Glimmer status (port ${PORT}) ==="
    pids="$(port_pids)"
    if [ -z "$pids" ]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if api_healthy; then
        echo "→ Health:    OK  http://${HOST}:${PORT}/v1/models"
        model_id="$(curl -sf --max-time 2 "http://${HOST}:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")"
        echo "→ Model ID:  ${model_id}"
        echo "→ API:       http://${HOST}:${PORT}/v1"
        exit 0
    fi
    echo "→ Health:    FAIL — something is bound but /v1/models failed"
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
    echo "ERROR: .mlx_config not found. Run ./1_setup_download.sh first."
    exit 1
fi

HF_MODEL="${HF_MODEL:-mlx-community/Muse-Glimmer-30B-4bit}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_ALIAS="${MODEL_ALIAS:-muse-glimmer-30b-mlx}"
DRAFT_HF="${DRAFT_HF:-meta-models/Muse-Glimmer-30B-assistant}"
DRAFT_DIR="${DRAFT_DIR:-${SCRIPT_DIR}/muse-glimmer-30b-assistant}"

if [[ -n "${MODEL_OVERRIDE}" ]]; then
    case "${MODEL_OVERRIDE}" in
        30b|4bit|auto|mlx) HF_MODEL="mlx-community/Muse-Glimmer-30B-4bit"; MODEL_DIR="" ;;
        5bit)  HF_MODEL="mlx-community/Muse-Glimmer-30B-5bit"; MODEL_DIR="" ;;
        6bit)  HF_MODEL="mlx-community/Muse-Glimmer-30B-6bit"; MODEL_DIR="" ;;
        8bit)  HF_MODEL="mlx-community/Muse-Glimmer-30B-8bit"; MODEL_DIR="" ;;
        mxfp4) HF_MODEL="mlx-community/Muse-Glimmer-30B-mxfp4"; MODEL_DIR="" ;;
        official|bf16) HF_MODEL="mlx-community/Muse-Glimmer-30B-bf16"; MODEL_DIR="" ;;
        *)
            HF_MODEL="${MODEL_OVERRIDE}"
            if [[ -d "${MODEL_OVERRIDE}" ]]; then
                MODEL_DIR="$(cd "${MODEL_OVERRIDE}" && pwd)"
            else
                MODEL_DIR=""
            fi
            ;;
    esac
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run ./1_setup_download.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PATH="${VENV_DIR}/bin:${PATH}"

if [[ "${LOW_MEMORY}" == true ]]; then
    PROFILE_DFLASH="off"
    if [[ -z "${MAX_KV_SIZE}" ]]; then
        MAX_KV_SIZE=8192
    fi
    echo "→ Low-memory mode: DFlash off, max-kv-size=${MAX_KV_SIZE}"
    echo ""
fi

if [[ "${DO_RESTART}" == true ]]; then
    if [ -n "$(port_pids)" ]; then
        echo "→ restart: clearing port ${PORT} ..."
        stop_server_on_port "${PORT}"
    fi
elif [ -n "$(port_pids)" ]; then
    if api_healthy; then
        echo "→ Muse Glimmer already healthy on :${PORT}"
        echo "→ API:      http://${HOST}:${PORT}/v1"
        model_id="$(curl -sf --max-time 2 "http://${HOST}:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${MODEL_ALIAS}")"
        echo "→ Model ID: ${model_id}"
        echo ""
        echo "  Use: ./2_start_mlx.sh restart   # to reload"
        echo "       ./2_start_mlx.sh stop      # to free the port"
        if [[ "${HARNESS_GATE}" == true ]]; then
            echo ""
            echo "→ Running harness gate against existing server ..."
            python3 "${SCRIPT_DIR}/test_harness.py" --gate --base "http://${HOST}:${PORT}" \
                --model "${model_id}" || true
        fi
        exit 0
    fi
    echo "ERROR: Port ${PORT} is already in use by an unhealthy process:"
    describe_port_holder
    echo ""
    echo "  Free it with:  ./2_start_mlx.sh restart"
    exit 1
fi

VALIDATE_MODEL="${SCRIPT_DIR}/validate_model.py"

resolve_model_path() {
    if [[ -n "${MODEL_DIR}" && -d "${MODEL_DIR}" ]]; then
        if python3 "${VALIDATE_MODEL}" "${MODEL_DIR}" >/dev/null 2>&1; then
            echo "${MODEL_DIR}"
            return 0
        fi
        echo "ERROR: local target incomplete: ${MODEL_DIR}" >&2
        python3 "${VALIDATE_MODEL}" "${MODEL_DIR}" >&2 || true
        echo "Fix: run ./1_setup_download.sh" >&2
        exit 1
    fi
    echo "${HF_MODEL}"
}

MODEL_PATH="$(resolve_model_path)"

ENABLE_DFLASH=false
DRAFT_PATH=""
if [[ -d "${DRAFT_DIR}" ]] && python3 "${VALIDATE_MODEL}" "${DRAFT_DIR}" >/dev/null 2>&1; then
    DRAFT_PATH="${DRAFT_DIR}"
elif [[ "${PROFILE_DFLASH}" == "on" ]]; then
    DRAFT_PATH="${DRAFT_HF}"
fi

case "${PROFILE_DFLASH}" in
    on)
        if [[ -z "${DRAFT_PATH}" ]]; then
            echo "ERROR: --with-dflash but no drafter. Run ./1_setup_download.sh"
            exit 1
        fi
        ENABLE_DFLASH=true
        ;;
    off)
        ENABLE_DFLASH=false
        ;;
    auto)
        if [[ -n "${DRAFT_PATH}" ]]; then
            ENABLE_DFLASH=true
        fi
        ;;
esac

export MLX_USE_DEFAULT_DEVICE=gpu
export MLX_VLM_DEFAULT_MODEL="${MODEL_PATH}"

SERVER_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down mlx_vlm.server ..."
    [[ -n "${SERVER_PID}" ]] && kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${SERVER_PID}" ]] && kill -KILL "${SERVER_PID}" 2>/dev/null || true
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM HUP

echo "=== Muse Glimmer 30B — mlx_vlm.server ==="
echo "→ Model:    ${MODEL_PATH}"
echo "→ Alias:    ${MODEL_ALIAS}"
echo "→ DFlash:   $([ "${ENABLE_DFLASH}" = true ] && echo "on (${DRAFT_PATH}, block ${DRAFT_BLOCK_SIZE})" || echo "off")"
echo "→ Port:     ${PORT}"
echo "→ API:      http://${HOST}:${PORT}/v1"
echo "→ Sampling: temp=1.0  top_p=0.95  top_k=64  (official; set in Kilo)"
echo "→ Reasoning: always on — put 'Reasoning strength: low|medium|high|xhigh' in the system prompt"
echo ""

SERVE_CMD=(
    python -m mlx_vlm.server
    --model "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --trust-remote-code
    --max-tokens "${MAX_TOKENS}"
    --enable-thinking
    --thinking-budget "${THINKING_BUDGET}"
)
if [[ "${ENABLE_DFLASH}" == true ]]; then
    SERVE_CMD+=(
        --draft-model "${DRAFT_PATH}"
        --draft-kind dflash
        --draft-block-size "${DRAFT_BLOCK_SIZE}"
    )
fi
if [[ -n "${MAX_KV_SIZE}" ]]; then
    SERVE_CMD+=(--max-kv-size "${MAX_KV_SIZE}")
fi

echo "→ Starting: ${SERVE_CMD[*]}"
echo ""
"${SERVE_CMD[@]}" &
SERVER_PID=$!

echo "→ Waiting for server to be ready ..."
READY=false
for i in $(seq 1 360); do
    if curl -sf --max-time 1 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "→ Server ready after ${i}s"
        READY=true
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: mlx_vlm.server exited unexpectedly. Check the output above."
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" != true ]]; then
    echo "ERROR: server did not become ready within 360s."
    echo "       Check unified memory and re-run: ./2_start_mlx.sh restart"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

LIVE_MODEL_ID="$(curl -sf "http://${HOST}:${PORT}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${MODEL_ALIAS}")"

echo ""
echo "============================================================"
echo "  READY — Muse Glimmer 30B"
echo "============================================================"
echo "  API:          http://${HOST}:${PORT}/v1"
echo "  Model ID:     ${LIVE_MODEL_ID}"
echo "  DFlash:       $([ "${ENABLE_DFLASH}" = true ] && echo on || echo off)"
echo ""
echo "  Kilo Code:  model muse-glimmer/muse-glimmer-30b-mlx  (see kilo.json)"
echo "  curl test:  curl http://${HOST}:${PORT}/v1/chat/completions \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"model\":\"${LIVE_MODEL_ID}\",\"messages\":[{\"role\":\"system\",\"content\":\"Reasoning strength: low.\"},{\"role\":\"user\",\"content\":\"What is 17*23? Reply with just the number.\"}],\"max_tokens\":512}'"
echo "  harness:    python3 test_harness.py --gate"
echo "============================================================"

if [[ "${HARNESS_GATE}" == true ]]; then
    echo ""
    echo "→ Post-start harness gate ..."
    if python3 "${SCRIPT_DIR}/test_harness.py" --gate \
        --base "http://${HOST}:${PORT}" \
        --model "${LIVE_MODEL_ID}"; then
        echo "→ Harness gate: PASS"
    else
        gate_rc=$?
        echo "→ Harness gate: FAIL (exit ${gate_rc}) — server still running"
        echo "  Re-run: python3 test_harness.py --base http://${HOST}:${PORT}"
    fi
    echo ""
fi

wait "${SERVER_PID}" 2>/dev/null
cleanup
