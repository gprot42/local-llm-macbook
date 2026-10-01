#!/usr/bin/env bash
# =============================================================================
# 2_start_tensorfold.sh — TensorFold OpenAI server for Gemma 4 26B-A4B (abliterated, uncensored)
#
# Listens at http://127.0.0.1:8094/v1
# The engine itself is on :8104. loop_proxy.py owns :8094: it refuses repeated
# tool calls, judges every agentic step (loops, symbol soup, narration instead
# of tool calls, leaked envelopes, truncation, bad arguments), cancels a failing
# generation at the engine and retries the step with fresh sampling, and fills
# in sane sampling for clients that omit it. GET /harness/health has counters.
# Gemma 4 runs on TensorFold's gemma4 lane (MoE-only kernels). Do not use :8080.
#
# Sampling: the engine defaults to temperature 0.45 / top-p 0.9 / top-k 40 /
# min-p 0.05 (Gemma's 1.0 / 0.95 / 64 degenerates on this abliterated pack in
# long agentic runs). A request that sets its own values wins.
#
# Detached (default): the stack runs in its own session with no terminal, so
# closing a terminal, an editor or the Claude app does not take it down (that
# is how it died on 2026-10-01: the tab it ran in was closed, and the hangup
# killed the supervisor, the engine and the proxy together). The command returns
# once the stack answers; output goes to .tensorfold_stack.log
# (./2_start_tensorfold.sh logs). --foreground keeps the old attached behaviour.
#
# Supervision (default on): after start, the engine is probed every 20 s and
# restarted if the process exits or fails three probes in a row; the proxy is
# restarted if it dies. Clients see a pause, not an error: the proxy waits out
# an engine restart for up to 90 s.
#
# Thinking is off by default, so tool calls are not wrapped in a think block.
#
# Options:
#   --port PORT       Listen port (default 8094; engine is this + 10)
#   --context N       Prompt + reply cap (default 131072; 0 = model max)
#   --model REPO      Hugging Face repo / local dir (default from .tensorfold_config)
#   --temperature T   Engine default temperature (default 0.45)
#   --thinking        Open the think block
#   --no-thinking     Skip the think block (default)
#   --no-drafts       One token per round
#   --no-supervise    Exit when the engine exits (no auto-restart)
#   --foreground      Run attached to this terminal (Ctrl-C stops; closing it stops)
#   --harness-gate    Run test_harness.py --gate after ready (default)
#   --no-harness-gate Skip the post-start gate
#   restart           Stop the supervisor and the servers, then start
#   stop              Stop the supervisor, then whatever listens on the two ports
#   status            Show the supervisor, /v1/models and /harness/health
#   logs              Follow .tensorfold_stack.log
#   --help, -h        Show this help
#
# Proxy knobs are environment variables read by loop_proxy.py:
#   HARNESS_STEP_MAX_TOKENS=6144  HARNESS_MAX_ATTEMPTS=3  HARNESS_STALL_SECONDS=240
#   HARNESS_STREAM_PROSE=1 (stream agentic prose live; failures are cut, not retried)
#   LOOP_MAX_ROUNDS=200
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"

PORT=8094
ENGINE_PORT=8104
CLI_CONTEXT=""
MODEL_OVERRIDE=""
THINKING=false
NO_DRAFTS=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
HARNESS_GATE=true
SUPERVISE=true
FOREGROUND=false
DO_LOGS=false
FORWARD=()                                              # arguments the detached run is started with
PID_FILE="${SCRIPT_DIR}/.tensorfold_supervisor.pid"     # the supervising script, so stop/restart can end it
STACK_LOG="${SCRIPT_DIR}/.tensorfold_stack.log"         # a detached run's output
STACK_LOG_MAX_BYTES=$((50 * 1024 * 1024))

# Sampling defaults: engine CLI and the proxy's fill-ins for omitted fields agree.
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.45}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-40}"
SAMPLING_MIN_P="${SAMPLING_MIN_P:-0.05}"
ENGINE_MAX_TOKENS="${ENGINE_MAX_TOKENS:-8192}"   # engine cap when a request sets none (OpenCode sets its own)

SUPERVISE_INTERVAL="${SUPERVISE_INTERVAL:-20}"   # seconds between health probes
SUPERVISE_PROBE_TIMEOUT="${SUPERVISE_PROBE_TIMEOUT:-10}"
SUPERVISE_FAILS="${SUPERVISE_FAILS:-3}"           # consecutive failed probes before a restart
SUPERVISE_MAX_RESTARTS="${SUPERVISE_MAX_RESTARTS:-20}"

log() { echo "$(date '+%H:%M:%S') $*"; }

# The process LISTENING on a port. `lsof -ti :PORT` also lists every client
# connected to it, which made stop/restart send SIGTERM to a connected OpenCode.
listen_pids() {
    lsof -nP -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true
}

# A supervised start relaunches the engine when it dies, so stop/restart end the
# supervising script first (its trap shuts the engine and proxy down), then clear
# the ports. Only this stack's script is touched: the pidfile is per directory and
# the process is checked to be a 2_start_tensorfold.sh.
stop_supervisor() {
    local pid i
    [[ -f "${PID_FILE}" ]] || return 0
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" && "${pid}" != "$$" ]] && kill -0 "${pid}" 2>/dev/null \
        && ps -p "${pid}" -o command= 2>/dev/null | grep -q 2_start_tensorfold; then
        echo "→ Stopping supervisor (pid ${pid}) ..."
        kill -TERM "${pid}" 2>/dev/null || true
        for i in $(seq 1 15); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
}

supervisor_pid() {
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null \
        && ps -p "${pid}" -o command= 2>/dev/null | grep -q 2_start_tensorfold; then
        echo "${pid}"
    fi
}

stop_server_on_port() {
    local port="$1"
    local pids
    pids="$(listen_pids "${port}")"
    if [[ -z "${pids}" ]]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(listen_pids "${port}")"
    if [[ -n "${pids}" ]]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL ${pids} 2>/dev/null || true
        sleep 1
    fi
}

port_pids() {
    listen_pids "${PORT}"
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

engine_healthy() {
    curl -sf --max-time "${1:-2}" "http://127.0.0.1:${ENGINE_PORT}/v1/models" >/dev/null 2>&1
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
            sed -n '3,54p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        logs) DO_LOGS=true; i=$((i + 1)) ;;
        --foreground) FOREGROUND=true; i=$((i + 1)) ;;
        --detach) FOREGROUND=false; i=$((i + 1)) ;;
        --port)
            PORT="${args[$((i + 1))]:?--port needs a value}"
            FORWARD+=(--port "${PORT}")
            i=$((i + 2))
            ;;
        --context)
            CLI_CONTEXT="${args[$((i + 1))]:?--context needs a value}"
            FORWARD+=(--context "${CLI_CONTEXT}")
            i=$((i + 2))
            ;;
        --model)
            MODEL_OVERRIDE="${args[$((i + 1))]:?--model needs a value}"
            FORWARD+=(--model "${MODEL_OVERRIDE}")
            i=$((i + 2))
            ;;
        --temperature)
            SAMPLING_TEMPERATURE="${args[$((i + 1))]:?--temperature needs a value}"
            FORWARD+=(--temperature "${SAMPLING_TEMPERATURE}")
            i=$((i + 2))
            ;;
        --thinking) THINKING=true; FORWARD+=(--thinking); i=$((i + 1)) ;;
        --no-thinking) THINKING=false; FORWARD+=(--no-thinking); i=$((i + 1)) ;;
        --no-drafts) NO_DRAFTS=true; FORWARD+=(--no-drafts); i=$((i + 1)) ;;
        --no-supervise) SUPERVISE=false; FORWARD+=(--no-supervise); i=$((i + 1)) ;;
        --supervise) SUPERVISE=true; FORWARD+=(--supervise); i=$((i + 1)) ;;
        --harness-gate) HARNESS_GATE=true; FORWARD+=(--harness-gate); i=$((i + 1)) ;;
        --no-harness-gate) HARNESS_GATE=false; FORWARD+=(--no-harness-gate); i=$((i + 1)) ;;
        *)
            echo "ERROR: unknown argument: ${args[$i]}"
            exit 2
            ;;
    esac
done

# Engine stays 10 ports above the public proxy so --port keeps the pair together.
ENGINE_PORT=$((PORT + 10))

if [[ "${DO_LOGS}" == true ]]; then
    [[ -f "${STACK_LOG}" ]] || { echo "No log yet: ${STACK_LOG}"; exit 1; }
    exec tail -n 100 -f "${STACK_LOG}"
fi

if [[ "${DO_STATUS}" == true ]]; then
    echo "=== TensorFold status (port ${PORT}) ==="
    sup="$(supervisor_pid)"
    echo "→ Supervisor: ${sup:-none running}${sup:+  (log: ${STACK_LOG})}"
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
        if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/harness/health" >/dev/null 2>&1; then
            echo "→ Harness:  $(curl -sf --max-time 3 "http://127.0.0.1:${PORT}/harness/health" \
                | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('ok') else 'UPSTREAM DOWN', d.get('counts'), 'failures', d.get('failures'))" 2>/dev/null)"
        else
            echo "→ Harness:  /harness/health not answering — :${PORT} is not loop_proxy.py (restart to fix)"
        fi
        exit 0
    fi
    echo "→ Health:   FAIL — something is bound but /v1/models failed"
    echo "  Fix:      ./2_start_tensorfold.sh restart"
    exit 1
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
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-26b-a4b-abliterated-tensorfold}"
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
    echo "→ restart: stopping the supervisor and clearing ports ${PORT} and ${ENGINE_PORT} ..."
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

# ── Detached start ───────────────────────────────────────────────────────────────
# Re-run this script with --foreground in a new session (setsid: no controlling
# terminal, so no hangup reaches it), output appended to the stack log. Return once
# the stack answers, reporting the post-start gate.
DETACH_LAUNCHER='import os, sys
log, pid_file, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
if os.fork():
    os._exit(0)
os.setsid()                                   # a new session: no controlling terminal, no hangup
with open(pid_file, "w") as handle:           # the same pid the supervising bash keeps after exec
    handle.write(f"{os.getpid()}\n")
out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
os.dup2(out, 1)
os.dup2(out, 2)
os.execvp(argv[0], argv)'

start_detached() {
    local size start_bytes i pid="" ready=false gate=""
    if [[ -f "${STACK_LOG}" ]]; then
        size="$(wc -c < "${STACK_LOG}" | tr -d ' ')"
        if [[ "${size}" -gt "${STACK_LOG_MAX_BYTES}" ]]; then
            mv -f "${STACK_LOG}" "${STACK_LOG}.1"
        fi
    fi
    { echo ""; echo "===== $(date '+%Y-%m-%d %H:%M:%S') detached start ====="; } >> "${STACK_LOG}"
    start_bytes="$(wc -c < "${STACK_LOG}" | tr -d ' ')"
    rm -f "${PID_FILE}"
    python3 -c "${DETACH_LAUNCHER}" "${STACK_LOG}" "${PID_FILE}" \
        bash "${SCRIPT_DIR}/2_start_tensorfold.sh" --foreground ${FORWARD[@]+"${FORWARD[@]}"}
    echo "→ Starting detached (log: ${STACK_LOG}) ..."
    for i in $(seq 1 700); do
        sleep 1
        [[ -z "${pid}" && -f "${PID_FILE}" ]] && pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/harness/health" >/dev/null 2>&1; then
            ready=true
            break
        fi
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        if [[ -z "${pid}" && "${i}" -ge 15 ]]; then
            break
        fi
        if (( i % 15 == 0 )); then
            echo "  ... ${i}s: $(tail -c +"$((start_bytes + 1))" "${STACK_LOG}" | grep -E '^→ ' | tail -1)"
        fi
    done
    if [[ "${ready}" != true ]]; then
        echo "ERROR: the stack did not come up. Last lines of ${STACK_LOG}:"
        tail -c +"$((start_bytes + 1))" "${STACK_LOG}" | tail -25 | sed 's/^/  /'
        exit 1
    fi
    if [[ "${HARNESS_GATE}" == true ]]; then
        for i in $(seq 1 120); do
            gate="$(tail -c +"$((start_bytes + 1))" "${STACK_LOG}" | grep -m1 -E 'Harness gate: (PASS|FAIL)' || true)"
            [[ -n "${gate}" ]] && break
            sleep 1
        done
    fi
    echo ""
    echo "============================================================"
    echo "  READY — detached, supervisor pid ${pid}"
    echo "============================================================"
    echo "  API:      http://127.0.0.1:${PORT}/v1"
    echo "  Health:   http://127.0.0.1:${PORT}/harness/health"
    if [[ -n "${gate}" ]]; then
        echo "  Gate:     ${gate#*Harness gate: }"
    fi
    echo "  Log:      ${STACK_LOG}   (./2_start_tensorfold.sh logs)"
    echo "  Stop:     ./2_start_tensorfold.sh stop"
    echo "============================================================"
    exit 0
}

if [[ "${FOREGROUND}" != true ]]; then
    start_detached
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
        if [[ -d "${repo}" ]]; then
            echo "→ Local: ${repo}"
            continue
        fi
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

echo "→ Proxy self-test ..."
if ! python3 "${SCRIPT_DIR}/loop_proxy.py" --self-test >/dev/null 2>&1; then
    echo "ERROR: loop_proxy.py --self-test failed. Run it directly to see why:"
    echo "  python3 ${SCRIPT_DIR}/loop_proxy.py --self-test"
    exit 1
fi
echo "→ Proxy self-test: ok"
echo ""

export MLX_USE_DEFAULT_DEVICE=gpu

TF_PID=""
PROXY_PID=""
SHUTTING_DOWN=false
cleanup() {
    SHUTTING_DOWN=true
    echo ""
    echo "→ Shutting down TensorFold ..."
    [[ -n "${PROXY_PID}" ]] && kill -TERM "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${TF_PID}" ]] && kill -TERM "${TF_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${PROXY_PID}" ]] && kill -KILL "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${TF_PID}" ]] && kill -KILL "${TF_PID}" 2>/dev/null || true
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    stop_server_on_port "${ENGINE_PORT}" >/dev/null 2>&1 || true
    rm -f "${PID_FILE}"
    exit 0
}
trap cleanup INT TERM HUP
echo $$ > "${PID_FILE}"

echo "=== Gemma 4 26B-A4B (abliterated, uncensored) — TensorFold ==="
echo "→ Model:    ${HF_MODEL}"
echo "→ Drafter:  ${DRAFTER:-none}"
echo "→ Alias:    ${MODEL_ALIAS}"
echo "→ Context:  ${CONTEXT}"
echo "→ Sampling: temperature ${SAMPLING_TEMPERATURE}, top-p ${SAMPLING_TOP_P}, top-k ${SAMPLING_TOP_K}, min-p ${SAMPLING_MIN_P}"
echo "→ Thinking: $([[ "${THINKING}" == true ]] && echo on || echo off)"
echo "→ Drafts:   $([[ "${NO_DRAFTS}" == true ]] && echo off || echo on)"
echo "→ Supervise: $([[ "${SUPERVISE}" == true ]] && echo "on (probe every ${SUPERVISE_INTERVAL}s, restart after ${SUPERVISE_FAILS} misses)" || echo off)"
echo "→ Port:     ${PORT}"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo ""

SERVE_CMD=(
    tensorfold serve "${HF_MODEL}"
    --host 127.0.0.1
    --port "${ENGINE_PORT}"
    --name "${MODEL_ALIAS}"
    --context "${CONTEXT}"
    --max-tokens "${ENGINE_MAX_TOKENS}"
    --temperature "${SAMPLING_TEMPERATURE}"
    --top-p "${SAMPLING_TOP_P}"
    --top-k "${SAMPLING_TOP_K}"
    --min-p "${SAMPLING_MIN_P}"
    # The engine is a patched build (1_setup_download.sh). 0.6.0 asks GitHub for a newer release at
    # start and suggests `tensorfold update`, which would install the stock release over the patches.
    --no-update-check
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

PROXY_CMD=(
    python3 "${SCRIPT_DIR}/loop_proxy.py"
    --listen "${PORT}"
    --upstream "127.0.0.1:${ENGINE_PORT}"
    --defaults "temperature=${SAMPLING_TEMPERATURE},top_p=${SAMPLING_TOP_P},top_k=${SAMPLING_TOP_K},min_p=${SAMPLING_MIN_P}"
    --failures-dir "${SCRIPT_DIR}/.harness_failures"
)

start_engine() {
    echo "→ Starting: ${SERVE_CMD[*]}"
    echo ""
    "${SERVE_CMD[@]}" &
    TF_PID=$!
}

# Waits up to $1 seconds for the engine to answer /v1/models. 0 = ready, 1 = exited, 2 = timeout.
wait_engine_ready() {
    local limit="$1" i
    for i in $(seq 1 "${limit}"); do
        if engine_healthy 1; then
            echo "→ Engine ready after ${i}s"
            return 0
        fi
        if ! kill -0 "${TF_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done
    return 2
}

start_proxy() {
    echo "→ Starting loop proxy on :${PORT} -> :${ENGINE_PORT}"
    "${PROXY_CMD[@]}" &
    PROXY_PID=$!
    local i
    for i in $(seq 1 20); do
        if server_healthy; then
            return 0
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            echo "ERROR: loop proxy exited. See the log above."
            return 1
        fi
        sleep 1
    done
    echo "ERROR: loop proxy did not answer on :${PORT}"
    return 1
}

run_gate() {
    local model_id
    model_id="$(live_model_id)"
    echo ""
    echo "→ Post-start harness gate ..."
    if python3 "${SCRIPT_DIR}/test_harness.py" --gate \
        --base "http://127.0.0.1:${PORT}" \
        --model "${model_id}"; then
        echo "→ Harness gate: PASS"
    else
        local gate_rc=$?
        echo "→ Harness gate: FAIL (exit ${gate_rc}) — server still running"
        echo "  Re-run: python3 test_harness.py --base http://127.0.0.1:${PORT}"
    fi
    echo ""
}

start_engine
echo "→ Waiting for engine on :${ENGINE_PORT} (first load can take a few minutes) ..."
if ! wait_engine_ready 600; then
    rc=$?
    if [[ "${rc}" == 1 ]]; then
        echo "ERROR: tensorfold exited before it was ready. See the log above."
    else
        echo "ERROR: server did not become ready within 600s."
        kill -TERM "${TF_PID}" 2>/dev/null || true
    fi
    exit 1
fi

if ! start_proxy; then
    exit 1
fi

LIVE_MODEL_ID="$(live_model_id)"

echo ""
echo "============================================================"
echo "  READY — TensorFold serving ${HF_MODEL}"
echo "============================================================"
echo "  API:        http://127.0.0.1:${PORT}/v1"
echo "  Model ID:   ${LIVE_MODEL_ID}"
echo "  Health:     http://127.0.0.1:${PORT}/harness/health"
echo "  Kilo:       tensorfold-gemma4-abliterated/gemma-4-26b-a4b-abliterated-tensorfold  (kilo.json)"
echo "  curl:       curl http://127.0.0.1:${PORT}/v1/chat/completions \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32}'"
echo "  harness:    python3 test_harness.py --gate      # quick"
echo "              python3 test_e2e_opencode.py        # full OpenCode task, ~5-15 min"
echo "============================================================"

if [[ "${HARNESS_GATE}" == true ]]; then
    run_gate
fi

if [[ "${SUPERVISE}" != true ]]; then
    wait "${TF_PID}" 2>/dev/null
    cleanup
fi

# ── Supervisor ──────────────────────────────────────────────────────────────────
# The proxy keeps answering (and holds client requests for up to 90 s) while
# the engine is restarted, so a crash or hang costs the agent a pause, not the turn.
misses=0
restarts=0
while true; do
    sleep "${SUPERVISE_INTERVAL}" &
    wait $! || true          # a plain sleep would delay the TERM trap by up to SUPERVISE_INTERVAL
    if [[ "${SHUTTING_DOWN}" == true ]]; then
        break
    fi
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        log "→ [supervisor] loop proxy exited — restarting it"
        stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
        start_proxy || log "→ [supervisor] proxy restart failed; will retry in ${SUPERVISE_INTERVAL}s"
        continue
    fi
    reason=""
    if ! kill -0 "${TF_PID}" 2>/dev/null; then
        reason="engine process exited"
    elif engine_healthy "${SUPERVISE_PROBE_TIMEOUT}"; then
        misses=0
        continue
    else
        misses=$((misses + 1))
        if [[ "${misses}" -lt "${SUPERVISE_FAILS}" ]]; then
            log "→ [supervisor] engine health probe failed (${misses}/${SUPERVISE_FAILS})"
            continue
        fi
        reason="engine failed ${misses} health probes in a row"
    fi
    restarts=$((restarts + 1))
    log "→ [supervisor] ${reason} — restart #${restarts}"
    if [[ "${restarts}" -gt "${SUPERVISE_MAX_RESTARTS}" ]]; then
        log "→ [supervisor] more than ${SUPERVISE_MAX_RESTARTS} restarts; giving up"
        cleanup
    fi
    kill -TERM "${TF_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${TF_PID}" 2>/dev/null || true
    stop_server_on_port "${ENGINE_PORT}" >/dev/null 2>&1 || true
    misses=0
    start_engine
    if wait_engine_ready 600; then
        log "→ [supervisor] engine back on :${ENGINE_PORT}"
        if [[ "${HARNESS_GATE}" == true ]]; then
            run_gate
        fi
    else
        log "→ [supervisor] engine did not come back; retrying in ${SUPERVISE_INTERVAL}s"
    fi
done
cleanup
