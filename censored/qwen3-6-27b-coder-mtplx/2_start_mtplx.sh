#!/usr/bin/env bash
# =============================================================================
# 2_start_mtplx.sh — Start mtplx OpenAI-compatible server for Qwen3.6
#
# mtplx serves at http://localhost:8765/v1  (OpenAI-compatible)
# Kilo Code picks this up via kilo.json in this directory (or parent kilo.json).
#
# MTP speedup at temp=0.6, top_p=0.95, top_k=20 (Qwen recommended):
#   Qwen3.6-27B:  ~28 tok/s → ~63 tok/s  (2.24× on M5 Max)
#   Qwen3.6-35B-A3B: ~85 tok/s → ~101 tok/s (+18-20% on M3/M5 Ultra)
#
# Options:
#   --port  PORT    Override port (default: 8765)
#   --model SIZE    Override model: 27b or 35b (reads .mtplx_config by default)
#   --profile NAME  mtplx profile: sustained (default) | performance-cold | stable | burst*
#   --depth N       MTP speculation depth: 2-4 (default: 3)
#   --max           Enable ThermalForge fan control (max sustained throughput)
#   restart         Stop anything on the port, then start fresh
#   stop            Stop process(es) on the port
#   status          Show whether mtplx is healthy on the port
#   --help, -h      Show this help
#   * burst is an alias for --profile performance-cold --max
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.mtplx_config"

# ── Defaults ──────────────────────────────────────────────────────────────────
PORT=8765
PROFILE="sustained"
DEPTH=3
MODEL_OVERRIDE=""
MAX_FANS=false
DO_RESTART=false
DO_STOP=false
DO_STATUS=false

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

mtplx_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

# ── Parse args ────────────────────────────────────────────────────────────────
i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,23p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        restart)   DO_RESTART=true;                          ((i+=1)) ;;
        stop)      DO_STOP=true;                             ((i+=1)) ;;
        status)    DO_STATUS=true;                           ((i+=1)) ;;
        --port)    PORT="${args[$((i+1))]:-$PORT}";          ((i+=2)) ;;
        --model)   MODEL_OVERRIDE="${args[$((i+1))]:-}";     ((i+=2)) ;;
        --profile) PROFILE="${args[$((i+1))]:-$PROFILE}";    ((i+=2)) ;;
        --depth)   DEPTH="${args[$((i+1))]:-$DEPTH}";        ((i+=2)) ;;
        --max)     MAX_FANS=true;                            ((i+=1)) ;;
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
    echo "=== mtplx status (port ${PORT}) ==="
    pids="$(port_pids)"
    if [ -z "$pids" ]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if mtplx_healthy; then
        echo "→ Health:    OK  http://127.0.0.1:${PORT}/v1/models"
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")"
        echo "→ Model ID:  ${model_id}"
        echo "→ API:       http://127.0.0.1:${PORT}/v1"
        exit 0
    fi
    echo "→ Health:    FAIL — something is bound but is not mtplx (/v1/models failed)"
    echo "  Fix:       ./2_start_mtplx.sh restart"
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
        27b) HF_MODEL="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed" ;;
        35b) HF_MODEL="mlx-community/Qwen3.6-35B-A3B-4bit" ;;
        *)   HF_MODEL="${MODEL_OVERRIDE}" ;;  # raw HF path
    esac
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run ./1_setup_download.sh first."
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# ── Port conflict handling (same pattern as sibling gemma/ornith scripts) ─────
if [[ "${DO_RESTART}" == true ]]; then
    if [ -n "$(port_pids)" ]; then
        echo "→ restart: clearing port ${PORT} ..."
        stop_server_on_port "${PORT}"
    fi
elif [ -n "$(port_pids)" ]; then
    if mtplx_healthy; then
        echo "→ mtplx already healthy on :${PORT}"
        echo "→ API:      http://127.0.0.1:${PORT}/v1"
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${MODEL_ALIAS:-?}")"
        echo "→ Model ID: ${model_id}"
        echo ""
        echo "  Use: ./2_start_mtplx.sh restart   # to reload"
        echo "       ./2_start_mtplx.sh stop      # to free the port"
        exit 0
    fi
    echo "ERROR: Port ${PORT} is already in use by a non-mtplx (or unhealthy) process:"
    describe_port_holder
    echo ""
    echo "  This often happens when another project left \`python -m http.server ${PORT}\`"
    echo "  (or a dead mtplx) bound. Free it with:"
    echo ""
    echo "    ./2_start_mtplx.sh restart"
    echo ""
    exit 1
fi

# ── Verify model weights are fully downloaded ─────────────────────────────────
# Exit codes from verify_model_weights (Python):
#   0  = all weight shards present (and MTPLX contract files for optimized 27B)
#   10 = model not in cache
#   11 = partial download (index/tokenizer without complete shards)
#   12 = MTPLX contract files missing (Youssofal optimized checkpoints only)
verify_model_weights() {
    python3 - "${1}" <<'PY'
import sys
from pathlib import Path

from mtplx.hf_loader import (
    cached_model_is_complete,
    cached_model_path,
    repo_id_from_model_ref,
    validate_mtplx_model_files,
)

model_ref = sys.argv[1]
repo_id = repo_id_from_model_ref(model_ref)
local = Path(model_ref).expanduser()
if local.exists():
    path = local
elif repo_id:
    path = cached_model_path(repo_id)
else:
    sys.exit(10)

if not path.exists():
    sys.exit(10)
if not cached_model_is_complete(path):
    sys.exit(11)
if repo_id and repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx"):
    validation = validate_mtplx_model_files(path)
    if not validation["ok"]:
        sys.exit(12)
PY
}

ensure_model_weights() {
    local status
    verify_model_weights "${HF_MODEL}" && return 0
    status=$?

    case "${status}" in
        10)
            echo "→ Model weights: not cached"
            ;;
        11)
            echo "→ Model weights: incomplete (interrupted download?)"
            ;;
        12)
            echo "→ Model weights: missing MTPLX sidecar files (mtp.safetensors / mtplx_runtime.json)"
            ;;
        *)
            echo "ERROR: could not verify model weights (exit ${status})"
            exit 1
            ;;
    esac

    echo "→ Downloading / resuming: ${HF_MODEL}"
    echo "  (run ./1_setup_download.sh to reinstall; mtplx pull resumes partial downloads)"
    echo ""
    if ! mtplx pull "${HF_MODEL}"; then
        echo "ERROR: model download failed."
        exit 1
    fi

    if ! verify_model_weights "${HF_MODEL}"; then
        echo "ERROR: model weights still incomplete after download."
        echo "       Check disk space and network, then run: ./1_setup_download.sh"
        exit 1
    fi
}

echo "→ Checking model weights ..."
ensure_model_weights
echo "→ Model weights: complete"
echo ""

# ── Apple Silicon performance environment ─────────────────────────────────────
# MLX uses Metal; these env vars tune the Metal / unified memory path.

# Prefer Metal GPU over CPU for all MLX ops.
export MLX_USE_DEFAULT_DEVICE=gpu

# Aggressive memory recycling — avoids fragmentation on long sessions.
export MLX_MEMORY_BUDGET_GB=0  # unlimited (uses unified memory pool)

# ── Cleanup handler ───────────────────────────────────────────────────────────
MTPLX_PID=""
cleanup() {
    echo ""
    echo "→ Shutting down mtplx ..."
    [[ -n "${MTPLX_PID}" ]] && kill -TERM "${MTPLX_PID}" 2>/dev/null || true
    sleep 1
    [[ -n "${MTPLX_PID}" ]] && kill -KILL "${MTPLX_PID}" 2>/dev/null || true
    # Also clear anything still holding the port (children / re-spawn)
    stop_server_on_port "${PORT}" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM HUP

echo "=== Qwen3.6 — mtplx MTP Server ==="
echo "→ Model:    ${HF_MODEL}"
echo "→ Profile:  ${PROFILE}"
echo "→ MTP depth: D${DEPTH}"
echo "→ Port:     ${PORT}"
echo "→ API:      http://localhost:${PORT}/v1"
echo ""
echo "→ Starting mtplx serve ..."
echo ""

# ── Build mtplx serve command ─────────────────────────────────────────────────
# --model              : HF repo id or local path (required named arg)
# --profile            : runtime profile (sustained = best for long coding sessions)
# --depth N            : MTP speculation depth (D3 = best balance of speed vs acceptance)
# --port               : HTTP port for OpenAI-compatible API
# --default-temperature / --default-top-p : Qwen3.6 recommended coding settings
# --reasoning off      : disable thinking-mode wrapping for Kilo Code tool calls
# --max                : opt into ThermalForge fan control for max sustained throughput
SERVE_CMD=(
    mtplx serve
    --model "${HF_MODEL}"
    --port "${PORT}"
    --profile "${PROFILE}"
    --depth "${DEPTH}"
    --default-temperature 0.6
    --default-top-p 0.95
    --reasoning off
    --model-id "${MODEL_ALIAS}"
)
[[ "${MAX_FANS}" == "true" ]] && SERVE_CMD+=(--max)

# ── Launch mtplx serve ────────────────────────────────────────────────────────
"${SERVE_CMD[@]}" &
MTPLX_PID=$!

# ── Wait for server to be ready ───────────────────────────────────────────────
echo "→ Waiting for server to be ready ..."
# Model load can take a few minutes on cold start; allow longer than weight-check
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

# ── Print model ID for kilo.json ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  READY — mtplx serving ${HF_MODEL}"
echo "============================================================"
echo "  API:          http://localhost:${PORT}/v1"
echo "  Model ID:     $(curl -sf http://localhost:${PORT}/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${HF_MODEL}")"
echo ""
echo "  Kilo Code:  model mtplx/qwen3.6-27b-mtplx  (see kilo.json)"
echo "  curl test:  curl http://localhost:${PORT}/v1/chat/completions \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}'"
echo "============================================================"
echo ""
echo "  MTP speedup:  ~2.24× vs no-MTP at temp=0.6 (D3)"
echo "  Profile:      ${PROFILE} (use --profile burst for bench, not prod)"
echo "============================================================"

# ── Keep alive ────────────────────────────────────────────────────────────────
wait "${MTPLX_PID}" 2>/dev/null
cleanup
