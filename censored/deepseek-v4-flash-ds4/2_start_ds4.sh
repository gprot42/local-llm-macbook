#!/usr/bin/env bash
# =============================================================================
# 2_start_ds4.sh — Serve DeepSeek V4 via DwarfStar ds4-server + Kilo proxy
#
# Native Metal engine (antirez/ds4). OpenAI + Anthropic compatible.
# Default: q2-imatrix Flash GGUF, context 100k (sensible on 128 GB with ~81 GB model).
#
# Public API (Kilo):  http://127.0.0.1:8083/v1   ← ds4_kilo_proxy (thinking OFF by default)
# Upstream engine:    http://127.0.0.1:18083/v1  ← raw ds4-server
# Use 127.0.0.1 — not localhost (macOS may resolve localhost to ::1).
#
# Why the proxy?
#   ds4 defaults chat requests to high-effort thinking. That burns tokens on CoT,
#   truncates tool-call JSON ("Expected '}'"), and aborts mid-fix. The proxy
#   forces thinking={type:disabled}, floors max_tokens, and soft-repairs tool args.
#
# Usage:
#   ./2_start_ds4.sh
#   ./2_start_ds4.sh restart
#   ./2_start_ds4.sh stop
#   ./2_start_ds4.sh status
#   ./2_start_ds4.sh --port 8084
#   ./2_start_ds4.sh --upstream-port 18083
#   ./2_start_ds4.sh --no-proxy          # raw ds4-server on --port (not agent-safe)
#   ./2_start_ds4.sh --ctx 200000
#   ./2_start_ds4.sh --ssd-streaming
#   ./2_start_ds4.sh --power 50
#   ./2_start_ds4.sh --mtp
#   ./2_start_ds4.sh --help
#
# Re-enable thinking per request: "think": true or "reasoning_effort": "high"
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.ds4_config"
VALIDATE_PY="$SCRIPT_DIR/validate_model.py"
PROXY_PY="$SCRIPT_DIR/ds4_kilo_proxy.py"
SERVER_PID_FILE="$SCRIPT_DIR/.ds4_server.pid"
PROXY_PID_FILE="$SCRIPT_DIR/.ds4_proxy.pid"
SERVER_LOG="$SCRIPT_DIR/.ds4_server.log"
PROXY_LOG="$SCRIPT_DIR/.ds4_proxy.log"

PORT=8083
UPSTREAM_PORT=18083
HOST="127.0.0.1"
CTX=100000
KV_DISK_DIR="${TMPDIR:-/tmp}/ds4-kv"
KV_DISK_SPACE_MB=8192
POWER=""
SSD_STREAMING=false
SSD_CACHE_EXPERTS=""
USE_MTP=false
USE_PROXY=true
MODEL_PATH=""
DO_RESTART=false
DO_STOP=false
DO_STATUS=false
SKIP_VALIDATE=false
DEFAULT_MAX_TOKENS=8192

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

stop_pidfile() {
    local pf="$1"
    if [[ ! -f "$pf" ]]; then
        return 0
    fi
    local pid
    pid="$(tr -d '[:space:]' <"$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
}

port_pids() {
    local port="${1:-$PORT}"
    lsof -ti ":${port}" 2>/dev/null || true
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

endpoint_healthy() {
    local port="$1"
    curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

proxy_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 \
        || endpoint_healthy "$PORT"
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

stop_all() {
    stop_pidfile "$PROXY_PID_FILE"
    stop_pidfile "$SERVER_PID_FILE"
    stop_server_on_port "$PORT"
    if [[ "$USE_PROXY" == true ]]; then
        stop_server_on_port "$UPSTREAM_PORT"
    fi
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
        --upstream-port) UPSTREAM_PORT="${args[$((i+1))]:-$UPSTREAM_PORT}"; i=$((i + 2)) ;;
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
        --no-proxy) USE_PROXY=false; i=$((i + 1)) ;;
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
    echo "=== ds4 harness status ==="
    echo "→ Public :${PORT}  (Kilo baseURL)"
    describe_port_holder "$PORT"
    if [[ "$USE_PROXY" == true ]]; then
        echo "→ Upstream :${UPSTREAM_PORT}  (raw ds4-server)"
        describe_port_holder "$UPSTREAM_PORT"
    fi
    if proxy_healthy; then
        echo "→ Health:    OK  http://127.0.0.1:${PORT}/v1/models"
        if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
            echo "→ Proxy:     OK  http://127.0.0.1:${PORT}/healthz  (thinking default OFF)"
        else
            echo "→ Proxy:     (direct ds4-server — no harness proxy)"
        fi
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
    stop_all
    echo "→ Stopped (public :${PORT}${USE_PROXY:+, upstream :${UPSTREAM_PORT}})"
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

if [[ "$USE_PROXY" == true && ! -f "$PROXY_PY" ]]; then
    echo "ERROR: proxy script missing: $PROXY_PY"
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    if [[ -L "$DS4_DIR/ds4flash.gguf" || -f "$DS4_DIR/ds4flash.gguf" ]]; then
        MODEL_PATH="$DS4_DIR/ds4flash.gguf"
    else
        part="$(ls -1 "$DS4_DIR"/gguf/*.gguf.part 2>/dev/null | head -1 || true)"
        echo "ERROR: No model at $DS4_DIR/ds4flash.gguf"
        if [[ -n "$part" ]]; then
            echo "       Found incomplete download: $part"
            if [[ -f "$VALIDATE_PY" ]]; then
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

if [[ -L "$MODEL_PATH" && ! -e "$MODEL_PATH" ]]; then
    echo "ERROR: Model symlink is broken: $MODEL_PATH"
    echo "       Run: ./1_setup_download.sh"
    exit 1
fi

MODEL_REAL="$(resolve_realpath "$MODEL_PATH")"
MODEL_SIZE="$(du -h "$MODEL_REAL" 2>/dev/null | awk '{print $1}' || echo '?')"

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
    echo "→ restart: clearing public :${PORT}${USE_PROXY:+ and upstream :${UPSTREAM_PORT}} ..."
    stop_all
elif proxy_healthy; then
    echo "→ ds4 harness already healthy on :${PORT}"
    echo "→ API:      http://127.0.0.1:${PORT}/v1"
    if curl -sf --max-time 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
        echo "→ Proxy:    thinking default OFF (agent-safe)"
    fi
    model_id="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${API_MODEL_ID}")"
    echo "→ Model ID: ${model_id}"
    echo ""
    echo "  Use: ./2_start_ds4.sh restart   # to reload"
    echo "       ./2_start_ds4.sh stop      # to free the ports"
    exit 0
elif [ -n "$(port_pids "$PORT")" ]; then
    echo "ERROR: Port ${PORT} is already in use by a non-ready process:"
    describe_port_holder "$PORT"
    echo ""
    echo "  Free it with:"
    echo "    ./2_start_ds4.sh restart"
    echo "    ./2_start_ds4.sh stop"
    echo "    ./2_start_ds4.sh --port N"
    exit 1
fi

if [[ "$USE_PROXY" == true && -n "$(port_pids "$UPSTREAM_PORT")" ]]; then
    if ! endpoint_healthy "$UPSTREAM_PORT"; then
        echo "→ Clearing stale upstream :${UPSTREAM_PORT} ..."
        stop_server_on_port "$UPSTREAM_PORT"
    fi
fi

mkdir -p "$KV_DISK_DIR"

ENGINE_PORT="$PORT"
if [[ "$USE_PROXY" == true ]]; then
    ENGINE_PORT="$UPSTREAM_PORT"
fi

SERVER_ARGS=(
    -m "$MODEL_PATH"
    --chdir "$DS4_DIR"
    --host "$HOST"
    --port "$ENGINE_PORT"
    --ctx "$CTX"
    -n "$DEFAULT_MAX_TOKENS"
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
echo "=== DwarfStar ds4 harness (DeepSeek V4) ==="
echo "→ Engine:   $DS4_DIR"
echo "→ Model:    $MODEL_PATH  ($MODEL_SIZE)"
echo "→ Realpath: $MODEL_REAL"
echo "→ Public:   http://127.0.0.1:${PORT}/v1   (Kilo baseURL)"
if [[ "$USE_PROXY" == true ]]; then
    echo "→ Upstream: http://127.0.0.1:${UPSTREAM_PORT}/v1  (raw ds4-server)"
    echo "→ Proxy:    ON — thinking default OFF, max_tokens floor ${DEFAULT_MAX_TOKENS}"
else
    echo "→ Proxy:    OFF (--no-proxy) — thinking may default ON (not agent-safe)"
fi
echo "→ Model ID: $API_MODEL_ID"
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
if [[ "$USE_PROXY" == true ]]; then
    echo "  python3 $PROXY_PY --host $HOST --port $PORT --upstream http://127.0.0.1:${UPSTREAM_PORT}"
fi
echo ""
echo "  Wait until the model is loaded (can take a few minutes for ~81 GB),"
echo "  then: curl http://127.0.0.1:${PORT}/v1/models"
echo "  Kilo: provider ds4 / model ds4/deepseek-v4-flash"
echo ""

# Run engine from ds4 tree so metal/*.metal paths resolve
cd "$DS4_DIR"

if [[ "$USE_PROXY" != true ]]; then
    exec ./ds4-server "${SERVER_ARGS[@]}"
fi

# ── Background engine + foreground proxy (proxy is the public process) ────────
: >"$SERVER_LOG"
nohup ./ds4-server "${SERVER_ARGS[@]}" >>"$SERVER_LOG" 2>&1 &
echo $! >"$SERVER_PID_FILE"
echo "→ ds4-server pid $(cat "$SERVER_PID_FILE")  log: $SERVER_LOG"

# Wait for engine to accept /v1/models (load can take minutes)
echo "→ Waiting for upstream model load on :${UPSTREAM_PORT} ..."
for i in $(seq 1 600); do
    if endpoint_healthy "$UPSTREAM_PORT"; then
        echo "→ Upstream ready after ~${i}s"
        break
    fi
    if ! kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
        echo "ERROR: ds4-server exited during load. Tail of $SERVER_LOG:"
        tail -40 "$SERVER_LOG" || true
        exit 1
    fi
    if [[ $((i % 15)) -eq 0 ]]; then
        echo "   … still loading (${i}s)"
    fi
    sleep 1
    if [[ "$i" -eq 600 ]]; then
        echo "ERROR: timed out waiting for ds4-server on :${UPSTREAM_PORT}"
        tail -40 "$SERVER_LOG" || true
        exit 1
    fi
done

cd "$SCRIPT_DIR"
: >"$PROXY_LOG"
# Proxy in foreground so Ctrl+C / terminal lifecycle is clear; also write pid
python3 "$PROXY_PY" --host "$HOST" --port "$PORT" --upstream "http://127.0.0.1:${UPSTREAM_PORT}" \
    >>"$PROXY_LOG" 2>&1 &
echo $! >"$PROXY_PID_FILE"
PROXY_PID="$(cat "$PROXY_PID_FILE")"
echo "→ proxy pid $PROXY_PID  log: $PROXY_LOG"

cleanup() {
    stop_pidfile "$PROXY_PID_FILE"
    stop_pidfile "$SERVER_PID_FILE"
    stop_server_on_port "$PORT"
    stop_server_on_port "$UPSTREAM_PORT"
}
trap cleanup EXIT INT TERM

for i in $(seq 1 30); do
    if curl -sf --max-time 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
        echo "→ Proxy ready  http://127.0.0.1:${PORT}/v1"
        break
    fi
    sleep 0.2
done

echo ""
echo "Harness up. Leave this running (or use nohup). Logs:"
echo "  tail -f $SERVER_LOG"
echo "  tail -f $PROXY_LOG"
echo ""

# Keep script alive while both children run
while kill -0 "$(cat "$SERVER_PID_FILE" 2>/dev/null)" 2>/dev/null \
   && kill -0 "$(cat "$PROXY_PID_FILE" 2>/dev/null)" 2>/dev/null; do
    sleep 2
done

echo "→ A harness process exited — shutting down remaining."
exit 1
