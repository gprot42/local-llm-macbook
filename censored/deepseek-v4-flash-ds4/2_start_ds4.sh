#!/usr/bin/env bash
# =============================================================================
# 2_start_ds4.sh — Serve DeepSeek V4 via DwarfStar ds4-server
#
# Native Metal engine (antirez/ds4). OpenAI + Anthropic compatible.
# Default: q2-imatrix Flash GGUF, context 100k (sensible on 128 GB with ~81 GB model).
#
# API: http://127.0.0.1:8083/v1
# Use 127.0.0.1 — not localhost (macOS may resolve localhost to ::1).
#
# Usage:
#   ./2_start_ds4.sh
#   ./2_start_ds4.sh restart
#   ./2_start_ds4.sh stop
#   ./2_start_ds4.sh status
#   ./2_start_ds4.sh --port 8084
#   ./2_start_ds4.sh --ctx 200000
#   ./2_start_ds4.sh --ssd-streaming
#   ./2_start_ds4.sh --power 50
#   ./2_start_ds4.sh --mtp
#   ./2_start_ds4.sh --help
#
# Thinking is controlled per-request (not a server flag):
#   thinking={type:disabled}, think=false, or model=deepseek-chat → non-thinking
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.ds4_config"
VALIDATE_PY="$SCRIPT_DIR/validate_model.py"

PORT=8083
HOST="127.0.0.1"
CTX=100000
KV_DISK_DIR="${TMPDIR:-/tmp}/ds4-kv"
KV_DISK_SPACE_MB=8192
POWER=""
SSD_STREAMING=false
SSD_CACHE_EXPERTS=""
USE_MTP=false
MODEL_PATH=""
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
SKIP_VALIDATE=false

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
    # Wait up to ~15s for graceful shutdown (large models can take a moment).
    local i
    for i in $(seq 1 15); do
        pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
        [ -z "$pids" ] && return 0
        sleep 1
    done
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

server_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

resolve_realpath() {
    local p="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$p" 2>/dev/null && return
    fi
    if command -v realpath >/dev/null 2>&1; then
        realpath "$p" 2>/dev/null && return
    fi
    readlink "$p" 2>/dev/null || echo "$p"
}

for arg in "$@"; do
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
        exit 0
    }
done

i=0; args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        restart) DO_RESTART=true; i=$((i + 1)) ;;
        stop) DO_STOP=true; i=$((i + 1)) ;;
        status) DO_STATUS=true; i=$((i + 1)) ;;
        --port) PORT="${args[$((i+1))]:-$PORT}"; i=$((i + 2)) ;;
        --host) HOST="${args[$((i+1))]:-$HOST}"; i=$((i + 2)) ;;
        --ctx) CTX="${args[$((i+1))]:-$CTX}"; i=$((i + 2)) ;;
        --kv-disk-dir) KV_DISK_DIR="${args[$((i+1))]:-$KV_DISK_DIR}"; i=$((i + 2)) ;;
        --kv-disk-space-mb) KV_DISK_SPACE_MB="${args[$((i+1))]:-$KV_DISK_SPACE_MB}"; i=$((i + 2)) ;;
        --power) POWER="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --ssd-streaming) SSD_STREAMING=true; i=$((i + 1)) ;;
        --ssd-streaming-cache-experts)
            SSD_STREAMING=true
            SSD_CACHE_EXPERTS="${args[$((i+1))]:-}"
            i=$((i + 2))
            ;;
        --mtp) USE_MTP=true; i=$((i + 1)) ;;
        --model|-m) MODEL_PATH="${args[$((i+1))]:-}"; i=$((i + 2)) ;;
        --skip-validate) SKIP_VALIDATE=true; i=$((i + 1)) ;;
        *)
            echo "ERROR: Unknown argument: ${args[$i]}"
            echo "       Run ./2_start_ds4.sh --help"
            exit 1
            ;;
    esac
done

# ── status / stop ─────────────────────────────────────────────────────────────
if [[ "${DO_STATUS}" == true ]]; then
    echo "=== ds4-server status (port ${PORT}) ==="
    pids="$(port_pids)"
    if [ -z "$pids" ]; then
        echo "→ Port ${PORT}: free (no server)"
        exit 1
    fi
    echo "→ Process(es) on :${PORT}:"
    describe_port_holder
    if server_healthy; then
        echo "→ Health:    OK  http://127.0.0.1:${PORT}/v1/models"
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")"
        echo "→ Model ID:  ${model_id}"
        echo "→ API:       http://127.0.0.1:${PORT}/v1"
        exit 0
    fi
    echo "→ Health:    FAIL — port bound but /v1/models failed"
    echo "             (still loading GGUF, or not this server)"
    echo "  Fix:       ./2_start_ds4.sh restart"
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

# ── resolve engine + model ────────────────────────────────────────────────────
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
fi
DS4_DIR="${DS4_DIR:-$SCRIPT_DIR/ds4}"
API_MODEL_ID="${API_MODEL_ID:-deepseek-v4-flash}"

if [[ ! -x "$DS4_DIR/ds4-server" ]]; then
    echo "ERROR: ds4-server not found at $DS4_DIR/ds4-server"
    echo "       Run ./1_setup_download.sh first."
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    if [[ -L "$DS4_DIR/ds4flash.gguf" || -f "$DS4_DIR/ds4flash.gguf" ]]; then
        MODEL_PATH="$DS4_DIR/ds4flash.gguf"
    else
        # Helpful hint when only a partial exists
        part="$(ls -1 "$DS4_DIR"/gguf/*.gguf.part 2>/dev/null | head -1 || true)"
        echo "ERROR: No model at $DS4_DIR/ds4flash.gguf"
        if [[ -n "$part" ]]; then
            echo "       Found incomplete download: $part"
            if [[ -f "$VALIDATE_PY" ]]; then
                # Validate the non-.part name for progress messaging
                base="${part%.part}"
                python3 "$VALIDATE_PY" "$base" 2>&1 || true
            fi
            echo "       Resume: ./1_setup_download.sh"
        else
            echo "       Run: ./1_setup_download.sh q2-imatrix"
        fi
        exit 1
    fi
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Broken symlink check
if [[ -L "$MODEL_PATH" && ! -e "$MODEL_PATH" ]]; then
    echo "ERROR: Model symlink is broken: $MODEL_PATH"
    echo "       Run: ./1_setup_download.sh"
    exit 1
fi

# Resolve symlink for size display (safe: no shell interpolation of path into python)
MODEL_REAL="$(resolve_realpath "$MODEL_PATH")"
MODEL_SIZE="$(du -h "$MODEL_REAL" 2>/dev/null | awk '{print $1}' || echo '?')"

# Refuse to start on incomplete / corrupt weights (common after interrupted HF curl).
if [[ "$SKIP_VALIDATE" != true && -f "$VALIDATE_PY" ]]; then
    if ! python3 "$VALIDATE_PY" "$MODEL_PATH"; then
        echo ""
        echo "ERROR: Model weights incomplete or invalid — refusing to start."
        part="$(ls -1 "$DS4_DIR"/gguf/*.gguf.part 2>/dev/null | head -1 || true)"
        [[ -n "$part" ]] && echo "       Partial file: $part"
        echo "       Resume download: ./1_setup_download.sh"
        echo "       (override only if you know what you're doing: --skip-validate)"
        exit 1
    fi
elif [[ "$SKIP_VALIDATE" == true ]]; then
    echo "→ WARNING: --skip-validate set; not checking GGUF completeness"
fi

MTP_PATH=""
if [[ "$USE_MTP" == true ]]; then
    # download_model.sh mtp writes under gguf/
    cand="$(ls -1 "$DS4_DIR"/gguf/*MTP*.gguf 2>/dev/null | head -1 || true)"
    if [[ -z "$cand" ]]; then
        echo "ERROR: --mtp requested but no MTP GGUF under $DS4_DIR/gguf/"
        echo "       Run: (cd ds4 && ./download_model.sh mtp)"
        echo "       or:  ./1_setup_download.sh mtp"
        exit 1
    fi
    if [[ "$SKIP_VALIDATE" != true && -f "$VALIDATE_PY" ]]; then
        if ! python3 "$VALIDATE_PY" "$cand"; then
            echo "ERROR: MTP GGUF incomplete or invalid: $cand"
            echo "       Re-run: ./1_setup_download.sh mtp"
            exit 1
        fi
    fi
    MTP_PATH="$cand"
fi

# ── Port conflict handling (idempotent when already healthy) ──────────────────
if [[ "$DO_RESTART" == true ]]; then
    if [ -n "$(port_pids)" ]; then
        echo "→ restart: clearing port ${PORT} ..."
        stop_server_on_port "$PORT"
    fi
elif [ -n "$(port_pids)" ]; then
    if server_healthy; then
        echo "→ ds4-server already healthy on :${PORT}"
        echo "→ API:      http://127.0.0.1:${PORT}/v1"
        model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${API_MODEL_ID}")"
        echo "→ Model ID: ${model_id}"
        echo ""
        echo "  Use: ./2_start_ds4.sh restart   # to reload"
        echo "       ./2_start_ds4.sh stop      # to free the port"
        exit 0
    fi
    echo "ERROR: Port ${PORT} is already in use by a non-ds4 (or still-loading) process:"
    describe_port_holder
    echo ""
    echo "  Free it with:"
    echo "    ./2_start_ds4.sh restart"
    echo "    ./2_start_ds4.sh stop"
    echo "    ./2_start_ds4.sh --port N"
    exit 1
fi

mkdir -p "$KV_DISK_DIR"

SERVER_ARGS=(
    -m "$MODEL_PATH"
    --chdir "$DS4_DIR"
    --host "$HOST"
    --port "$PORT"
    --ctx "$CTX"
    --kv-disk-dir "$KV_DISK_DIR"
    --kv-disk-space-mb "$KV_DISK_SPACE_MB"
)

if [[ -n "$POWER" ]]; then
    SERVER_ARGS+=(--power "$POWER")
fi
if [[ "$SSD_STREAMING" == true ]]; then
    SERVER_ARGS+=(--ssd-streaming)
    if [[ -n "$SSD_CACHE_EXPERTS" ]]; then
        SERVER_ARGS+=(--ssd-streaming-cache-experts "$SSD_CACHE_EXPERTS")
    fi
fi
if [[ -n "$MTP_PATH" ]]; then
    SERVER_ARGS+=(--mtp "$MTP_PATH" --mtp-draft 2)
fi

echo ""
echo "=== DwarfStar ds4-server (DeepSeek V4) ==="
echo "→ Engine:   $DS4_DIR"
echo "→ Model:    $MODEL_PATH  ($MODEL_SIZE)"
echo "→ Realpath: $MODEL_REAL"
echo "→ API:      http://127.0.0.1:${PORT}/v1"
echo "→ Model ID: $API_MODEL_ID  (also deepseek-v4-pro alias)"
echo "→ Context:  $CTX tokens"
echo "→ KV disk:  $KV_DISK_DIR (${KV_DISK_SPACE_MB} MB budget)"
if [[ "$SSD_STREAMING" == true ]]; then
    echo "→ SSD stream: ON${SSD_CACHE_EXPERTS:+ (experts cache $SSD_CACHE_EXPERTS)}"
fi
if [[ -n "$POWER" ]]; then
    echo "→ Power:    ${POWER}%"
fi
if [[ -n "$MTP_PATH" ]]; then
    echo "→ MTP:      $MTP_PATH (experimental)"
fi
echo ""
echo "  $DS4_DIR/ds4-server ${SERVER_ARGS[*]}"
echo ""
echo "  Wait until the model is loaded (can take a few minutes for ~81 GB),"
echo "  then: curl http://127.0.0.1:${PORT}/v1/models"
echo "  Kilo: provider ds4 / model ds4/deepseek-v4-flash"
echo ""

# Run from engine tree so relative metal/*.metal paths resolve even without --chdir
cd "$DS4_DIR"
exec ./ds4-server "${SERVER_ARGS[@]}"
