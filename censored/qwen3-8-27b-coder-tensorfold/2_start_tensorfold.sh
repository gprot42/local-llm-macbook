#!/usr/bin/env bash
# =============================================================================
# 2_start_tensorfold.sh — TensorFold OpenAI server for Qwen3.8-27B
#
# Listens at http://127.0.0.1:8769/v1
# The engine itself is on :8779. loop_proxy.py owns :8769 and refuses a tool
# call that has already returned the same result twice.
# Port 8769 sits beside Qwen3.8 mtplx on :8766 and Qwen3.6 mtplx on :8765.
# Do not use :8080: Gemma and Diffusion stacks already share that port.
#
# TensorFold 1.0+ is a native binary (brew install ashhart/tensorfold/tensorfold);
# there is no venv. The loop proxy and harness run on the system python3.
#
# Sampling follows the Qwen3 coder preset (temperature 0.6, top-p 0.95, top-k 20).
# Since TensorFold 1.0.6 the DFlash2 drafter accelerates SAMPLED requests too
# (~125 tok/s code on an M-series Mac, vs ~20 on 1.0.5), so the coder preset gets
# the draft speed-up — no need to serve greedy for performance.
# Thinking is off by default. Pass --thinking to turn it on.
#
# The Vontra checkpoint's own chat template renders tools, so no --chat-template
# is needed here (unlike the obliterated sibling).
#
# Options:
#   --port PORT            Listen port (default 8769)
#   --context N            Prompt + reply cap (default 262144, the checkpoint max; 0 = model max)
#   --model REPO           Hugging Face repo (default from .tensorfold_config)
#   --prompt-cache-gib N   Prompt-state cache size, GiB (default 16; 0 turns it off)
#   --thinking             Open the think block (Qwen3.8 reasoning-effort medium)
#   --no-thinking          Skip the think block (default)
#   --no-drafts            One token per round
#   --harness-gate         Run test_harness.py --gate after ready (default)
#   --no-harness-gate      Skip the post-start gate
#   start                  Start in the background (same as the default)
#   --foreground, -f       Run attached to this terminal instead of detaching
#   restart                Free the ports, then start (in the background)
#   stop                   Stop the background server and free the ports
#   status                 Show whether /v1/models answers
#   logs                   Tail the background server's log
#   --help, -h             Show this help
#
# By default the server starts DETACHED: it keeps running after this terminal
# closes, and the command returns right away. Use --foreground to watch it live.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"
PID_FILE="${SCRIPT_DIR}/.tensorfold_supervisor.pid"   # the detached start script, so stop/restart can end it
STACK_LOG="${SCRIPT_DIR}/.tensorfold_stack.log"       # detached stdout/stderr
ORIG_ARGS=("$@")

PORT=8769
ENGINE_PORT=8779
CLI_CONTEXT=""
MODEL_OVERRIDE=""
PROMPT_CACHE_GIB="${QWEN38_PROMPT_CACHE_GIB:-16}"
PARALLEL="${QWEN38_PARALLEL:-1}"   # 1.0.6+ reserves stream memory per slot up front; single-user needs few
THINKING=false

# Native binary lives in Homebrew's bin; put it on PATH for this process.
for brew_bin in /opt/homebrew/bin /usr/local/bin; do
    [[ -d "${brew_bin}" ]] && case ":${PATH}:" in *":${brew_bin}:"*) ;; *) PATH="${brew_bin}:${PATH}" ;; esac
done
export PATH
NO_DRAFTS=false
FOREGROUND=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
DO_LOGS=false
HARNESS_GATE=true

stop_server_on_port() {
    local port="$1"
    local pids
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

stop_supervisor() {
    local pid
    [[ -f "${PID_FILE}" ]] || return 0
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "→ Stopping background server (pid ${pid}) ..."
        kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        sleep 2
        kill -KILL "-${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
}

# Relaunch this script in its own session (detached from the terminal), logging
# to STACK_LOG, and record the detached pid in PID_FILE. start_new_session runs
# os.setsid in the child, so macOS (no setsid binary) detaches cleanly too.
run_detached() {
    local relaunch=("$0") a
    # Drop action subcommands: the parent already handled them, and passing
    # 'restart' to the child would make its stop_supervisor kill itself (the pid
    # file now holds the child's own pid). The child only needs to start.
    if [[ ${#ORIG_ARGS[@]} -gt 0 ]]; then
        for a in "${ORIG_ARGS[@]}"; do
            case "${a}" in
                start|restart|stop|status|logs) ;;
                *) relaunch+=("${a}") ;;
            esac
        done
    fi
    relaunch+=(--foreground)
    : > "${STACK_LOG}" 2>/dev/null || true
    python3 - "${STACK_LOG}" "${PID_FILE}" "${relaunch[@]}" <<'PY'
import os, sys, subprocess
log, pidfile = sys.argv[1], sys.argv[2]
cmd = sys.argv[3:]
f = open(log, "ab", buffering=0)
p = subprocess.Popen(cmd, stdout=f, stderr=f, stdin=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)
with open(pidfile, "w") as pf:
    pf.write(str(p.pid))
print(p.pid)
PY
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
        start) i=$((i + 1)) ;;   # background is the default; accept it explicitly
        --foreground|-f) FOREGROUND=true; i=$((i + 1)) ;;
        --detach|-d) FOREGROUND=false; i=$((i + 1)) ;;
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        logs) DO_LOGS=true; i=$((i + 1)) ;;
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
        --prompt-cache-gib)
            PROMPT_CACHE_GIB="${args[$((i + 1))]:?--prompt-cache-gib needs a value}"
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

if [[ "${DO_LOGS}" == true ]]; then
    if [[ ! -f "${STACK_LOG}" ]]; then
        echo "→ No log yet at ${STACK_LOG}. Start the server first."
        exit 1
    fi
    exec tail -n 200 -f "${STACK_LOG}"
fi

if [[ "${DO_STOP}" == true ]]; then
    stop_supervisor
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
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-27b-tensorfold}"
CONTEXT="${CONTEXT:-262144}"
if [[ -n "${CLI_CONTEXT}" ]]; then
    CONTEXT="${CLI_CONTEXT}"
fi

if [[ "${DO_RESTART}" == true ]]; then
    echo "→ restart: clearing ports ${PORT} and ${ENGINE_PORT} ..."
    stop_supervisor
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

# Default: re-launch detached and return. The child re-runs with --foreground
# and falls through to the real serve loop below.
if [[ "${FOREGROUND}" != true ]]; then
    echo "→ Starting Qwen3.8-27B (aligned) in the background ..."
    pid="$(run_detached)"
    echo "→ Background server started (supervisor pid ${pid})"
    echo "  API:   http://127.0.0.1:${PORT}/v1   (ready in ~30s — watch with 'logs')"
    echo "  Logs:  ./2_start_tensorfold.sh logs"
    echo "  Stop:  ./2_start_tensorfold.sh stop"
    echo "  Check: ./2_start_tensorfold.sh status"
    exit 0
fi

# A local directory is validated by its files; a Hub repo is pulled (tensorfold
# pull is idempotent — it fetches only the pieces it lacks).
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

echo "=== Qwen3.8-27B — TensorFold (native $(tensorfold --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)) ==="
echo "→ Model:    ${HF_MODEL}"
echo "→ Drafter:  ${DRAFTER:-none}"
echo "→ Alias:    ${MODEL_ALIAS}"
echo "→ Context:  ${CONTEXT}"
echo "→ Cache:    ${PROMPT_CACHE_GIB} GiB"
echo "→ Thinking: $([[ "${THINKING}" == true ]] && echo on || echo off)"
echo "→ Drafts:   $([[ "${NO_DRAFTS}" == true ]] && echo off || echo on) (DFlash2 drafts sampled requests too since TF 1.0.6)"
echo "→ Port:     ${PORT}"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo ""

SERVE_CMD=(
    tensorfold serve "${HF_MODEL}"
    --host 127.0.0.1
    --port "${ENGINE_PORT}"
    --name "${MODEL_ALIAS}"
    --context "${CONTEXT}"
    --temperature 0.6
    --top-p 0.95
    --top-k 20
    --prompt-cache-gib "${PROMPT_CACHE_GIB}"
    --parallel "${PARALLEL}"
    --backend auto
    --no-update-check
)
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
echo "  Kilo:       tensorfold-qwen38/qwen3.8-27b-tensorfold  (kilo.json)"
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
