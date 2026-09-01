#!/usr/bin/env bash
# =============================================================================
# 2_start_mtplx.sh — Start mtplx OpenAI-compatible server for
#                    Qwen3.8-27B OBLITERATED
#
# Public API (Kilo / harness):  http://127.0.0.1:8768/v1  ← kilo proxy
# Upstream engine:              http://127.0.0.1:8767/v1  ← raw mtplx
# Proxy forces card sampling + scoped loop middleware. Point Kilo at kilo.json.
#
# Everyday:
#   ./2_start_mtplx.sh             start (or attach) + smoke-check, then kilo
#   ./2_start_mtplx.sh restart     reload engine + proxy
#   ./2_start_mtplx.sh check       smoke-check a server that is already up
#   ./2_start_mtplx.sh check-agent longer train/dev tool-loop eval
#   ./2_start_mtplx.sh stop | status
#
# Options:
#   --port  PORT    Raw mtplx port (default: 8767)
#   --proxy-port P  Public Kilo/harness port (default: 8768)
#   --model REF     4bit | 8bit | local path (reads .mtplx_config by default)
#   --profile NAME  mtplx profile: sustained (default) | performance-cold | stable | burst*
#   --depth N       MTP speculation depth: 2-4 (default: 3)
#   --max           Enable ThermalForge fan control (max sustained throughput)
#   --harness-gate  Run test_harness.py --gate after ready (default: on)
#   --no-harness-gate  Skip post-start harness gate
#   --no-proxy      Bind mtplx on --port only (not agent-safe)
#   --help, -h      Show this help
#   * burst is an alias for --profile performance-cold --max
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.mtplx_config"
MODELS_ROOT="${SCRIPT_DIR}/models"

# ── Defaults ──────────────────────────────────────────────────────────────────
PORT=8767
PROXY_PORT=8768
PROFILE="sustained"
DEPTH=3
MODEL_OVERRIDE=""
MAX_FANS=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
DO_CHECK=false
DO_CHECK_AGENT=false
HARNESS_GATE=true
USE_PROXY=true
PROXY_PY="${SCRIPT_DIR}/qwen38_obl_kilo_proxy.py"
PROXY_PID=""

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

public_port() {
    if [[ "${USE_PROXY}" == true ]]; then
        echo "${PROXY_PORT}"
    else
        echo "${PORT}"
    fi
}

port_pids() {
    local port="${1:-$PORT}"
    lsof -ti ":${port}" 2>/dev/null || true
}

proxy_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null 2>&1
}

describe_port_holder() {
    local port="${1:-$PORT}"
    local pids
    pids="$(port_pids "$port")"
    if [ -z "$pids" ]; then
        echo "(none)"
        return
    fi
    # shellcheck disable=SC2086
    ps -p $pids -o pid=,command= 2>/dev/null | sed 's/^/  /' || echo "  pid(s): ${pids//$'\n'/ }"
}

mtplx_healthy() {
    local port="${1:-$PORT}"
    curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

live_model_id() {
    local pub
    pub="$(public_port)"
    curl -sf --max-time 2 "http://127.0.0.1:${pub}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
        || echo "${MODEL_ALIAS:-qwen3.8-27b-obliterated-mtplx}"
}

start_kilo_proxy() {
    if [[ "${USE_PROXY}" != true ]]; then
        return 0
    fi
    if [[ ! -f "${PROXY_PY}" ]]; then
        echo "ERROR: proxy missing: ${PROXY_PY}"
        exit 1
    fi
    echo "→ Starting Kilo proxy on :${PROXY_PORT} (latest middleware) ..."
    python3 "${PROXY_PY}" -v --port "${PROXY_PORT}" --upstream "http://127.0.0.1:${PORT}" &
    PROXY_PID=$!
    local i
    for i in $(seq 1 30); do
        if proxy_healthy; then
            echo "→ Proxy ready after ${i}s"
            return 0
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            echo "ERROR: kilo proxy exited unexpectedly."
            exit 1
        fi
        sleep 0.2
    done
    echo "ERROR: kilo proxy did not become ready on :${PROXY_PORT}"
    exit 1
}

run_harness() {
    # $1 = extra flags, e.g. "--gate" or "--gate --agent"
    local extra="${1:---gate}"
    local pub model
    pub="$(public_port)"
    model="$(live_model_id)"
    echo ""
    echo "→ Harness ${extra}  (http://127.0.0.1:${pub}  model=${model})"
    # shellcheck disable=SC2086
    if python3 "${SCRIPT_DIR}/test_harness.py" ${extra} \
        --base "http://127.0.0.1:${pub}" \
        --model "${model}"; then
        echo "→ Harness: PASS"
        return 0
    fi
    local rc=$?
    echo "→ Harness: FAIL (exit ${rc}) — server still running"
    echo "  Re-run: ./2_start_mtplx.sh check"
    return "${rc}"
}

print_use_it() {
    local pub
    pub="$(public_port)"
    echo ""
    echo "============================================================"
    echo "  Using it"
    echo "============================================================"
    echo "  API:   http://127.0.0.1:${pub}/v1"
    echo "  Kilo:  model  mtplx-qwen38-obl/qwen3.8-27b-obliterated-mtplx"
    echo "         (kilo.json in this folder already points here)"
    echo ""
    echo "  That's it. Optional:"
    echo "    ./2_start_mtplx.sh check         # smoke"
    echo "    ./2_start_mtplx.sh check-agent   # longer loop eval"
    echo "============================================================"
}

# ── Parse args ────────────────────────────────────────────────────────────────
i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart)      DO_RESTART=true;                       ((i+=1)) ;;
        stop)         DO_STOP=true;                          ((i+=1)) ;;
        status)       DO_STATUS=true;                        ((i+=1)) ;;
        check)        DO_CHECK=true;                         ((i+=1)) ;;
        check-agent)  DO_CHECK_AGENT=true;                   ((i+=1)) ;;
        --port)    PORT="${args[$((i+1))]:-$PORT}";          ((i+=2)) ;;
        --proxy-port) PROXY_PORT="${args[$((i+1))]:-$PROXY_PORT}"; ((i+=2)) ;;
        --model)   MODEL_OVERRIDE="${args[$((i+1))]:-}";     ((i+=2)) ;;
        --profile) PROFILE="${args[$((i+1))]:-$PROFILE}";    ((i+=2)) ;;
        --depth)   DEPTH="${args[$((i+1))]:-$DEPTH}";        ((i+=2)) ;;
        --max)     MAX_FANS=true;                            ((i+=1)) ;;
        --harness-gate)    HARNESS_GATE=true;                ((i+=1)) ;;
        --no-harness-gate) HARNESS_GATE=false;               ((i+=1)) ;;
        --no-proxy) USE_PROXY=false;                         ((i+=1)) ;;
        *) ((i+=1)) ;;
    esac
done

# Map friendly alias "burst" → mtplx native profile name
if [[ "${PROFILE}" == "burst" ]]; then
    PROFILE="performance-cold"
    MAX_FANS=true
fi

# ── status / stop (no full config required) ───────────────────────────────────
if [[ "${DO_STATUS}" == true ]]; then
    pub="$(public_port)"
    echo "=== OBLITERATED mtplx status (public :${pub}) ==="
    echo "→ Engine :${PORT}:"
    describe_port_holder "${PORT}"
    if [[ "${USE_PROXY}" == true ]]; then
        echo "→ Proxy  :${PROXY_PORT}:"
        describe_port_holder "${PROXY_PORT}"
    fi
    if mtplx_healthy "${pub}"; then
        echo "→ Health:    OK  http://127.0.0.1:${pub}/v1/models"
        if [[ "${USE_PROXY}" == true ]] && proxy_healthy; then
            echo "→ Proxy:     OK  http://127.0.0.1:${PROXY_PORT}/healthz  (card sampling forced)"
        fi
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${pub}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")"
        echo "→ Model ID:  ${model_id}"
        echo "→ API:       http://127.0.0.1:${pub}/v1"
        exit 0
    fi
    echo "→ Health:    FAIL — public :${pub} /v1/models failed"
    echo "  Fix:       ./2_start_mtplx.sh restart"
    exit 1
fi

if [[ "${DO_STOP}" == true ]]; then
    stop_server_on_port "${PROXY_PORT}"
    stop_server_on_port "${PORT}"
    echo "→ Stopped (engine :${PORT}, proxy :${PROXY_PORT})"
    exit 0
fi

# ── Load config ───────────────────────────────────────────────────────────────
if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
else
    echo "ERROR: .mtplx_config not found. Run ./1_setup_download.sh first."
    exit 1
fi

# Model override from --model flag
if [[ -n "${MODEL_OVERRIDE}" ]]; then
    case "${MODEL_OVERRIDE}" in
        27b|4bit|auto|mlx)
            HF_MODEL="${MODELS_ROOT}/mlx-4bit"
            ;;
        8bit)
            HF_MODEL="${MODELS_ROOT}/mlx-8bit"
            ;;
        *)
            HF_MODEL="${MODEL_OVERRIDE}"
            ;;
    esac
fi

MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-27b-obliterated-mtplx}"
HF_REPO="${HF_REPO:-OBLITERATUS/Qwen3.8-27B-OBLITERATED}"
WEIGHT_VERSION="${WEIGHT_VERSION:-}"

# Refuse to treat the kitchen-sink Hub id as a pull target.
if [[ "${HF_MODEL}" == "${HF_REPO}" || "${HF_MODEL}" == "OBLITERATUS/Qwen3.8-27B-OBLITERATED" ]]; then
    echo "ERROR: HF_MODEL is the full Hub repo id (${HF_MODEL})."
    echo "  That pack includes GGUF + leftover shards + bf16 (hundreds of GB)."
    echo "  Snapshot V3 bf16 and convert locally:"
    echo "    ./1_setup_download.sh 4bit"
    echo "    ./2_start_mtplx.sh --model 4bit"
    exit 1
fi

if [[ "${WEIGHT_VERSION}" != "V3" ]]; then
    echo "WARN: local weights are not V3 (WEIGHT_VERSION=${WEIGHT_VERSION:-unset})."
    echo "  Hub removed V1/V2 mlx-4bit/; GGUFs were re-uploaded after a broken conversion."
    echo "  This stack uses V3 bf16 → local MLX. Rebuild with:"
    echo "    ./1_setup_download.sh --force"
    echo ""
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run ./1_setup_download.sh first."
    exit 1
fi
export PATH="${VENV_DIR}/bin:${PATH}"
hash -r 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

# ── check / check-agent (server must already be up) ───────────────────────────
if [[ "${DO_CHECK}" == true || "${DO_CHECK_AGENT}" == true ]]; then
    pub="$(public_port)"
    if ! mtplx_healthy "${pub}" && ! mtplx_healthy "${PORT}"; then
        echo "ERROR: nothing is listening. Start it first:"
        echo "  ./2_start_mtplx.sh"
        exit 2
    fi
    if [[ "${USE_PROXY}" == true ]] && ! proxy_healthy; then
        start_kilo_proxy
        disown "${PROXY_PID}" 2>/dev/null || true
    fi
    extra="--gate"
    [[ "${DO_CHECK_AGENT}" == true ]] && extra="--gate --agent"
    run_harness "${extra}"
    exit $?
fi

# ── Port conflict handling ────────────────────────────────────────────────────
if [[ "${DO_RESTART}" == true ]]; then
    echo "→ restart: clearing engine :${PORT} and proxy :${PROXY_PORT} ..."
    stop_server_on_port "${PROXY_PORT}"
    stop_server_on_port "${PORT}"
elif mtplx_healthy "${PORT}"; then
    echo "→ mtplx already healthy on :${PORT} (engine left running)"
    if [[ "${USE_PROXY}" == true ]] && ! proxy_healthy; then
        start_kilo_proxy
    elif [[ "${USE_PROXY}" == true ]]; then
        echo "→ Proxy already up on :${PROXY_PORT} (restart to reload middleware)"
    fi
    pub="$(public_port)"
    echo "→ API:      http://127.0.0.1:${pub}/v1"
    echo "→ Model ID: $(live_model_id)"
    if [[ "${HARNESS_GATE}" == true ]]; then
        run_harness "--gate" || true
    fi
    print_use_it
    if [[ -n "${PROXY_PID}" ]]; then
        disown "${PROXY_PID}" 2>/dev/null || true
    fi
    exit 0
elif [ -n "$(port_pids "${PORT}")" ]; then
    echo "ERROR: Port ${PORT} is already in use by a non-mtplx (or unhealthy) process:"
    describe_port_holder "${PORT}"
    echo ""
    echo "  Free it with:"
    echo "    ./2_start_mtplx.sh restart"
    echo ""
    exit 1
fi

# ── Verify model weights are fully downloaded ─────────────────────────────────
# Exit codes from verify_model_weights (Python):
#   0  = all weight shards present
#   10 = model path missing
#   11 = partial download (index/tokenizer without complete shards)
verify_model_weights() {
    python3 - "${1}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if not path.exists():
    sys.exit(10)
index = path / "model.safetensors.index.json"
if not index.is_file():
    sys.exit(11)
try:
    data = json.loads(index.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(11)
weight_map = data.get("weight_map") if isinstance(data, dict) else None
if not isinstance(weight_map, dict) or not weight_map:
    sys.exit(11)
for name in weight_map.values():
    if not isinstance(name, str) or not name.strip():
        sys.exit(11)
    shard = path / name
    try:
        if not shard.is_file() or shard.stat().st_size <= 0:
            sys.exit(11)
    except OSError:
        sys.exit(11)
PY
}

ensure_model_weights() {
    local status
    verify_model_weights "${HF_MODEL}" && return 0
    status=$?

    case "${status}" in
        10)
            echo "→ Model weights: not cached at ${HF_MODEL}"
            ;;
        11)
            echo "→ Model weights: incomplete (interrupted download?)"
            ;;
        *)
            echo "ERROR: could not verify model weights (exit ${status})"
            exit 1
            ;;
    esac

    echo "ERROR: local MLX folder incomplete. Do not mtplx-pull the Hub repo."
    echo "       Re-run: ./1_setup_download.sh"
    echo "       (snapshots mlx-4bit/ or mlx-8bit/ only, ~14–27 GB)"
    exit 1
}

echo "→ Checking model weights ..."
ensure_model_weights
echo "→ Model weights: complete"
echo ""

# ── Apple Silicon performance environment ─────────────────────────────────────
export MLX_USE_DEFAULT_DEVICE=gpu
export MLX_MEMORY_BUDGET_GB=0  # unlimited (uses unified memory pool)

# ── Cleanup handler ───────────────────────────────────────────────────────────
MTPLX_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down OBLITERATED stack ..."
    [[ -n "${PROXY_PID}" ]] && kill -TERM "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${MTPLX_PID}" ]] && kill -TERM "${MTPLX_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${PROXY_PID}" ]] && kill -KILL "${PROXY_PID}" 2>/dev/null || true
    [[ -n "${MTPLX_PID}" ]] && kill -KILL "${MTPLX_PID}" 2>/dev/null || true
    stop_server_on_port "${PROXY_PORT}" >/dev/null 2>&1 || true
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM HUP

echo "=== Qwen3.8-27B OBLITERATED — mtplx MTP Server ==="
echo "→ Model:    ${HF_MODEL}"
echo "→ Alias:    ${MODEL_ALIAS}"
echo "→ Profile:  ${PROFILE}"
echo "→ MTP depth: D${DEPTH}"
echo "→ Engine:   :${PORT}"
if [[ "${USE_PROXY}" == true ]]; then
    echo "→ Proxy:    :${PROXY_PORT}  (Kilo / harness)"
    echo "→ API:      http://localhost:${PROXY_PORT}/v1"
else
    echo "→ API:      http://localhost:${PORT}/v1  (--no-proxy)"
fi
echo "→ Sampling: temperature=0 frequency_penalty=0.3 reasoning=off (OBLITERATUS card)"
echo ""
echo "→ Starting mtplx serve ..."
echo ""

SERVE_CMD=(
    mtplx serve
    --model "${HF_MODEL}"
    --port "${PORT}"
    --profile "${PROFILE}"
    --depth "${DEPTH}"
    --default-temperature 0
    --default-top-p 1.0
    --default-frequency-penalty 0.3
    --reasoning off
    --model-id "${MODEL_ALIAS}"
)
[[ "${MAX_FANS}" == "true" ]] && SERVE_CMD+=(--max)

"${SERVE_CMD[@]}" &
MTPLX_PID=$!

echo "→ Waiting for server to be ready ..."
READY=false
for i in $(seq 1 300); do
    if curl -sf --max-time 1 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "→ Server ready after ${i}s"
        READY=true
        break
    fi
    if ! kill -0 "${MTPLX_PID}" 2>/dev/null; then
        echo "ERROR: mtplx exited unexpectedly. Check the output above."
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" != true ]]; then
    echo "ERROR: server did not become ready within 300s."
    echo "       Check Metal/GPU memory and re-run: ./2_start_mtplx.sh restart"
    kill -TERM "${MTPLX_PID}" 2>/dev/null || true
    exit 1
fi

start_kilo_proxy

echo ""
echo "============================================================"
echo "  READY — mtplx serving ${HF_MODEL}"
echo "============================================================"
echo "  API:          http://localhost:$(public_port)/v1"
if [[ "${USE_PROXY}" == true ]]; then
    echo "  Engine:       http://127.0.0.1:${PORT}/v1  (raw mtplx)"
    echo "  Proxy:        card sampling + loop middleware"
fi
echo "  Model ID:     $(live_model_id)"

if [[ "${HARNESS_GATE}" == true ]]; then
    run_harness "--gate" || true
fi
print_use_it
echo ""

if [[ -n "${PROXY_PID}" ]]; then
    wait "${MTPLX_PID}" "${PROXY_PID}" 2>/dev/null
else
    wait "${MTPLX_PID}" 2>/dev/null
fi
cleanup
