#!/usr/bin/env bash
# =============================================================================
# 2_start_tensorfold.sh — TensorFold (native v1.0.x) for Qwen3.8-27B OBLITERATED
#
# Listens at http://127.0.0.1:8096/v1
# The native engine is on :8106. loop_proxy.py owns :8096 and refuses a tool
# call that has already returned the same result twice.
# The archived mtplx sibling stays on :8768 (proxy) / :8767 (engine).
# Do not use :8080: Gemma and Diffusion stacks already share that port.
# Do not co-load the aligned Qwen 3.8 TensorFold server on :8769.
#
# TensorFold 1.0+ is a native binary (brew install ashhart/tensorfold/tensorfold);
# there is no venv. The loop proxy and harness run on the system python3.
#
# Sampling is greedy (temperature 0), the OBLITERATUS card setting, which is
# also what lets the DFlash2 drafter accelerate on the native engine — a
# sampled request would decode plain. Thinking is off unless you pass --thinking.
#
# The OBLITERATED checkpoint's own chat_template.jinja has no tool support, so
# the native engine would never put the tool list in the prompt. We pass a
# Qwen3 tool-capable template (chat_template_tools.jinja, XML tool format) with
# --chat-template so tool calls render and parse. Disable with --no-tool-template.
#
# Options:
#   --port PORT            Listen port (default 8096; engine is this plus 10)
#   --context N            Prompt + reply cap (default 262144, the checkpoint max; 0 = model max)
#   --model PATH           Checkpoint directory or repo (default from .tensorfold_config)
#   --prompt-cache-gib N   Prompt-state cache size, GiB (default 16; 0 turns it off)
#   --chat-template FILE   Override the tool-capable chat template
#   --no-tool-template     Serve with the checkpoint's own template (no tool rendering)
#   --thinking             Open the think block (Qwen3.8 reasoning-effort medium)
#   --no-thinking          Skip the think block (default)
#   --no-drafts            One token per round
#   --harness-gate         Run test_harness.py --gate after ready (default)
#   --no-harness-gate      Skip the post-start gate
#   restart                Free the port, then start
#   stop                   Stop whatever is bound to the port
#   status                 Show whether /v1/models answers
#   --help, -h             Show this help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"

PORT=8096
ENGINE_PORT=8106
CLI_CONTEXT=""
MODEL_OVERRIDE=""
PROMPT_CACHE_GIB="${QWEN38_OBL_PROMPT_CACHE_GIB:-16}"
TOOL_TEMPLATE="${SCRIPT_DIR}/chat_template_tools.jinja"
USE_TOOL_TEMPLATE=true
THINKING=false
NO_DRAFTS=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
HARNESS_GATE=true

# Native binary lives in Homebrew's bin; put it on PATH for this process.
for brew_bin in /opt/homebrew/bin /usr/local/bin; do
    [[ -d "${brew_bin}" ]] && case ":${PATH}:" in *":${brew_bin}:"*) ;; *) PATH="${brew_bin}:${PATH}" ;; esac
done
export PATH

stop_server_on_port() {
    local port="$1"
    local pids
    # Listeners only: never a client that merely connected to the port.
    pids="$(lsof -nP -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -nP -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL ${pids} 2>/dev/null || true
        sleep 1
    fi
}

port_pids() {
    lsof -nP -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null || true
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
            sed -n '3,33p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        --port)
            PORT="${args[$((i + 1))]:?--port needs a value}"
            ENGINE_PORT=$((PORT + 10))
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
        --prompt-cache-gib)
            PROMPT_CACHE_GIB="${args[$((i + 1))]:?--prompt-cache-gib needs a value}"
            i=$((i + 2))
            ;;
        --chat-template)
            TOOL_TEMPLATE="${args[$((i + 1))]:?--chat-template needs a value}"
            USE_TOOL_TEMPLATE=true
            i=$((i + 2))
            ;;
        --no-tool-template) USE_TOOL_TEMPLATE=false; i=$((i + 1)) ;;
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

if ! command -v tensorfold >/dev/null 2>&1; then
    echo "ERROR: tensorfold (native binary) not found on PATH."
    echo "       Run ./1_setup_download.sh  (it brew-installs ashhart/tensorfold/tensorfold)."
    exit 1
fi

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
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-27b-obliterated-tensorfold}"
CONTEXT="${CONTEXT:-262144}"
if [[ -n "${CLI_CONTEXT}" ]]; then
    CONTEXT="${CLI_CONTEXT}"
fi

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

# Weights: a local directory is validated by its files; a Hub repo is pulled
# (tensorfold pull is idempotent — it fetches only the pieces it lacks).
ensure_weights() {
    local repo
    for repo in "$@"; do
        [[ -z "${repo}" ]] && continue
        if [[ -d "${repo}" ]]; then
            if [[ -f "${repo}/config.json" && -f "${repo}/model.safetensors.index.json" ]]; then
                echo "→ Local weights: ${repo}"
                continue
            fi
            echo "ERROR: ${repo} is not a complete MLX checkpoint."
            echo "       Run ./1_setup_download.sh"
            exit 1
        fi
        echo "→ Ensuring cached: ${repo}"
        tensorfold pull "${repo}"
    done
}

echo "→ Checking weights ..."
ensure_weights "${HF_MODEL}" "${DRAFTER:-}"

if [[ "${USE_TOOL_TEMPLATE}" == true && ! -f "${TOOL_TEMPLATE}" ]]; then
    echo "ERROR: tool template not found: ${TOOL_TEMPLATE}"
    echo "       Run ./1_setup_download.sh (it installs it), or pass --no-tool-template"
    echo "       to serve with the checkpoint's own template (tool calls will not render)."
    exit 1
fi
echo ""

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

echo "=== Qwen3.8-27B OBLITERATED — TensorFold (native $(tensorfold --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)) ==="
echo "→ Model:      ${HF_MODEL}"
echo "→ Drafter:    ${DRAFTER:-none}"
echo "→ Alias:      ${MODEL_ALIAS}"
echo "→ Context:    ${CONTEXT}"
echo "→ Cache:      ${PROMPT_CACHE_GIB} GiB"
echo "→ Thinking:   $([[ "${THINKING}" == true ]] && echo on || echo off)"
echo "→ Drafts:     $([[ "${NO_DRAFTS}" == true ]] && echo off || echo on)"
echo "→ Tools:      $([[ "${USE_TOOL_TEMPLATE}" == true ]] && echo "${TOOL_TEMPLATE##*/}" || echo "checkpoint template (no tools)")"
echo "→ Port:       ${PORT}"
echo "→ API:        http://127.0.0.1:${PORT}/v1"
echo ""

SERVE_CMD=(
    tensorfold serve "${HF_MODEL}"
    --host 127.0.0.1
    --port "${ENGINE_PORT}"
    --name "${MODEL_ALIAS}"
    --context "${CONTEXT}"
    --temperature 0
    --top-p 1
    --prompt-cache-gib "${PROMPT_CACHE_GIB}"
    --backend auto
    --no-update-check
)
if [[ "${USE_TOOL_TEMPLATE}" == true ]]; then
    SERVE_CMD+=(--chat-template "${TOOL_TEMPLATE}")
fi
if [[ "${THINKING}" == true ]]; then
    SERVE_CMD+=(--thinking --reasoning-effort medium)
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
echo "  Kilo:       tensorfold-qwen38-obl/qwen3.8-27b-obliterated-tensorfold  (kilo.json)"
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
