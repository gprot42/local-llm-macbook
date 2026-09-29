#!/usr/bin/env bash
# =============================================================================
# 2_start_tensorfold.sh — TensorFold OpenAI server for Gemma 4 26B-A4B (base, censored)
#
# Listens at http://127.0.0.1:8092/v1
# The engine itself is on :8102. loop_proxy.py owns :8092 and refuses a tool
# call that has already returned the same result twice.
# Gemma 4 runs on TensorFold's gemma4 lane (MoE-only kernels): this is the
# base (censored) MoE pack. Do not use :8080 (Gemma/Diffusion stacks).
#
# Thinking is off by default, so tool calls are not wrapped in a think block.
# Pass --thinking to turn it on.
#
# Options:
#   --port PORT       Listen port (default 8092; engine is this + 10)
#   --context N       Prompt + reply cap (default 131072; 0 = model max)
#   --model REPO      Hugging Face repo (default from .tensorfold_config)
#   --thinking        Open the think block
#   --no-thinking     Skip the think block (default)
#   --no-drafts       One token per round
#   --harness-gate    Run test_harness.py --gate after ready (default)
#   --no-harness-gate Skip the post-start gate
#   restart           Free the port, then start
#   stop              Stop whatever is bound to the port
#   status            Show whether /v1/models answers
#   --help, -h        Show this help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"

PORT=8092
ENGINE_PORT=8102
CLI_CONTEXT=""
MODEL_OVERRIDE=""
THINKING=false
NO_DRAFTS=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
HARNESS_GATE=true

stop_server_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL ${pids} 2>/dev/null || true
        sleep 1
    fi
}

port_pids() {
    lsof -ti ":${PORT}" 2>/dev/null || true
}

describe_port_holder() {
    local pids
    pids="$(port_pids)"
    if [[ -z "${pids}" ]]; then
        echo "(none)"
        return
    fi
    # shellcheck disable=SC2086
    ps -p ${pids} -o pid=,command= 2>/dev/null | sed 's/^/  /' || echo "  pid(s): ${pids//$'\n'/ }"
}

server_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

live_model_id() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
        || echo "${MODEL_ALIAS:-?}"
}

# ── Parse args ────────────────────────────────────────────────────────────────
args=("$@")
i=0
while [[ ${i} -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,24p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        --port)
            PORT="${args[$((i + 1))]:?--port needs a value}"
            i=$((i + 2))
            ;;
        --context)
            CLI_CONTEXT="${args[$((i + 1))]:?--context needs a value}"
            i=$((i + 2))
            ;;
        --model)
            MODEL_OVERRIDE="${args[$((i + 1))]:?--model needs a value}"
            i=$((i + 2))
            ;;
        --thinking) THINKING=true; i=$((i + 1)) ;;
        --no-thinking) THINKING=false; i=$((i + 1)) ;;
        --no-drafts) NO_DRAFTS=true; i=$((i + 1)) ;;
        --harness-gate) HARNESS_GATE=true; i=$((i + 1)) ;;
        --no-harness-gate) HARNESS_GATE=false; i=$((i + 1)) ;;
        *)
            echo "ERROR: unknown argument: ${args[$i]}"
            exit 2
            ;;
    esac
done

# Engine stays 10 ports above the public proxy so --port keeps the pair together.
ENGINE_PORT=$((PORT + 10))

if [[ "${DO_STATUS}" == true ]]; then
    echo "=== TensorFold status (port ${PORT}) ==="
    pids="$(port_pids)"
    if [[ -z "${pids}" ]]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if server_healthy; then
        echo "→ Health:   OK  http://127.0.0.1:${PORT}/v1/models"
        echo "→ Model ID: $(live_model_id)"
        echo "→ API:      http://127.0.0.1:${PORT}/v1"
        exit 0
    fi
    echo "→ Health:   FAIL — something is bound but /v1/models failed"
    echo "  Fix:      ./2_start_tensorfold.sh restart"
    exit 1
fi

if [[ "${DO_STOP}" == true ]]; then
    stop_server_on_port "${PORT}"
    stop_server_on_port "${ENGINE_PORT}"
    echo "→ Stopped (ports ${PORT} and ${ENGINE_PORT})"
    exit 0
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found. Run ./1_setup_download.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [[ -n "${MODEL_OVERRIDE}" ]]; then
    HF_MODEL="${MODEL_OVERRIDE}"
fi
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-26b-a4b-tensorfold}"
CONTEXT="${CONTEXT:-131072}"
if [[ -n "${CLI_CONTEXT}" ]]; then
    CONTEXT="${CLI_CONTEXT}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run ./1_setup_download.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if [[ "${DO_RESTART}" == true ]]; then
    echo "→ restart: clearing ports ${PORT} and ${ENGINE_PORT} ..."
    stop_server_on_port "${PORT}"
    stop_server_on_port "${ENGINE_PORT}"
elif [[ -n "$(port_pids)" ]]; then
    if server_healthy; then
        echo "→ TensorFold already healthy on :${PORT}"
        echo "→ API:      http://127.0.0.1:${PORT}/v1"
        echo "→ Model ID: $(live_model_id)"
        echo ""
        echo "  Use: ./2_start_tensorfold.sh restart"
        echo "       ./2_start_tensorfold.sh stop"
        if [[ "${HARNESS_GATE}" == true ]]; then
            echo ""
            echo "→ Running harness gate against existing server ..."
            python3 "${SCRIPT_DIR}/test_harness.py" --gate --base "http://127.0.0.1:${PORT}" \
                --model "$(live_model_id)" || true
        fi
        exit 0
    fi
    echo "ERROR: Port ${PORT} is already in use by something that is not a healthy TensorFold server:"
    describe_port_holder
    echo ""
    echo "  Free it with: ./2_start_tensorfold.sh restart"
    exit 1
fi

weights_cached() {
    python3 - "$1" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
    snapshot_download(sys.argv[1], local_files_only=True)
except Exception:
    sys.exit(1)
PY
}

ensure_weights() {
    local repo
    for repo in "$@"; do
        [[ -z "${repo}" ]] && continue
        if weights_cached "${repo}"; then
            echo "→ Cached: ${repo}"
            continue
        fi
        echo "→ Downloading / resuming: ${repo}"
        tensorfold pull "${repo}"
    done
}

echo "→ Checking weights ..."
ensure_weights "${HF_MODEL}" "${DRAFTER:-}"
echo ""

export MLX_USE_DEFAULT_DEVICE=gpu

TF_PID=""
PROXY_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down TensorFold ..."
    [[ -n "${PROXY_PID}" ]] && kill -TERM "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${TF_PID}" ]] && kill -TERM "${TF_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${PROXY_PID}" ]] && kill -KILL "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${TF_PID}" ]] && kill -KILL "${TF_PID}" 2>/dev/null || true
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    stop_server_on_port "${ENGINE_PORT}" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM HUP

echo "=== Gemma 4 26B-A4B (base) — TensorFold ==="
echo "→ Model:    ${HF_MODEL}"
echo "→ Drafter:  ${DRAFTER:-none}"
echo "→ Alias:    ${MODEL_ALIAS}"
echo "→ Context:  ${CONTEXT}"
echo "→ Thinking: $([[ "${THINKING}" == true ]] && echo on || echo off)"
echo "→ Drafts:   $([[ "${NO_DRAFTS}" == true ]] && echo off || echo on)"
echo "→ Port:     ${PORT}"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo ""

SERVE_CMD=(
    tensorfold serve "${HF_MODEL}"
    --host 127.0.0.1
    --port "${ENGINE_PORT}"
    --name "${MODEL_ALIAS}"
    --context "${CONTEXT}"
    --temperature 1.0
    --top-p 0.95
    --top-k 64
)
if [[ "${THINKING}" == true ]]; then
    SERVE_CMD+=(--thinking)
else
    SERVE_CMD+=(--no-thinking)
fi
if [[ "${NO_DRAFTS}" == true || -z "${DRAFTER:-}" ]]; then
    SERVE_CMD+=(--no-drafts)
else
    SERVE_CMD+=(--drafter "${DRAFTER}")
fi

echo "→ Starting: ${SERVE_CMD[*]}"
echo ""
"${SERVE_CMD[@]}" &
TF_PID=$!

echo "→ Waiting for engine on :${ENGINE_PORT} (first load can take a few minutes) ..."
READY=false
for i in $(seq 1 600); do
    if curl -sf --max-time 1 "http://127.0.0.1:${ENGINE_PORT}/v1/models" >/dev/null 2>&1; then
        echo "→ Engine ready after ${i}s"
        READY=true
        break
    fi
    if ! kill -0 "${TF_PID}" 2>/dev/null; then
        echo "ERROR: tensorfold exited before it was ready. See the log above."
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" != true ]]; then
    echo "ERROR: server did not become ready within 600s."
    kill -TERM "${TF_PID}" 2>/dev/null || true
    exit 1
fi

echo "→ Starting loop proxy on :${PORT} -> :${ENGINE_PORT}"
python3 "${SCRIPT_DIR}/loop_proxy.py" --listen "${PORT}" --upstream "127.0.0.1:${ENGINE_PORT}" &
PROXY_PID=$!
for i in $(seq 1 20); do
    if curl -sf --max-time 1 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo "ERROR: loop proxy exited. See the log above."
        exit 1
    fi
    sleep 1
    if [[ "${i}" == 20 ]]; then
        echo "ERROR: loop proxy did not answer on :${PORT}"
        exit 1
    fi
done

LIVE_MODEL_ID="$(live_model_id)"

echo ""
echo "============================================================"
echo "  READY — TensorFold serving ${HF_MODEL}"
echo "============================================================"
echo "  API:        http://127.0.0.1:${PORT}/v1"
echo "  Model ID:   ${LIVE_MODEL_ID}"
echo "  Kilo:       tensorfold-gemma4/gemma-4-26b-a4b-tensorfold  (kilo.json)"
echo "  curl:       curl http://127.0.0.1:${PORT}/v1/chat/completions \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32}'"
echo "  harness:    python3 test_harness.py --gate"
echo "============================================================"

if [[ "${HARNESS_GATE}" == true ]]; then
    echo ""
    echo "→ Post-start harness gate ..."
    if python3 "${SCRIPT_DIR}/test_harness.py" --gate \
        --base "http://127.0.0.1:${PORT}" \
        --model "${LIVE_MODEL_ID}"; then
        echo "→ Harness gate: PASS"
    else
        gate_rc=$?
        echo "→ Harness gate: FAIL (exit ${gate_rc}) — server still running"
        echo "  Re-run: python3 test_harness.py --base http://127.0.0.1:${PORT}"
    fi
    echo ""
fi

wait "${TF_PID}" 2>/dev/null
cleanup
