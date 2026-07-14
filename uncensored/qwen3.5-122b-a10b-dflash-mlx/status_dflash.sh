#!/usr/bin/env bash
# status_dflash.sh — quick reliability snapshot for the DFlash OpenAI server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="$SCRIPT_DIR/.dflash_122b_config"
LOG_FILE="$SCRIPT_DIR/.dflash_server.log"
PID_FILE="$SCRIPT_DIR/.dflash_server.pid"

if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi
PORT="${PORT:-8086}"
HOST="${HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-qwen3.5-122b-a10b-dflash}"

echo "=== DFlash status $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "api:    http://${HOST}:${PORT}/v1"
echo "model:  ${MODEL_ID}"
echo "log:    ${LOG_FILE}"
echo

# PID / process
if [ -f "$PID_FILE" ]; then
    PID="$(tr -d '[:space:]' <"$PID_FILE" || true)"
    echo "pid_file: ${PID:-empty}"
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
        ps -p "$PID" -o pid=,stat=,etime=,%cpu=,%mem=,rss= | awk '{
            printf "process:  pid=%s stat=%s etime=%s cpu=%s%% mem=%s%% rss_kb=%s\n", $1,$2,$3,$4,$5,$6
        }'
    else
        echo "process:  NOT RUNNING (stale pid file?)"
    fi
else
    echo "pid_file: missing"
fi

# Listener only (not ESTABLISHED clients)
LISTEN_PID="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
if [ -n "$LISTEN_PID" ]; then
    echo "listen:   :${PORT} pid=${LISTEN_PID}"
else
    echo "listen:   nothing on :${PORT}"
fi

CLIENTS="$(lsof -nP -iTCP:"${PORT}" -sTCP:ESTABLISHED 2>/dev/null | awk 'NR>1 {print $1" pid="$2}' | sort -u || true)"
if [ -n "$CLIENTS" ]; then
    echo "clients:"
    echo "$CLIENTS" | sed 's/^/  /'
else
    echo "clients:  none"
fi
echo

echo "=== health ==="
if HEALTH="$(curl -sS -m 3 "http://${HOST}:${PORT}/health" 2>/dev/null)"; then
    echo "$HEALTH"
    if command -v python3 >/dev/null 2>&1; then
        HEALTH_JSON="$HEALTH" python3 -c '
import json, os
h = json.loads(os.environ["HEALTH_JSON"])
busy = h.get("busy")
pt = h.get("prompt_tokens")
ct = h.get("completion_tokens")
bf = h.get("busy_for_s")
mode = h.get("mode")
print()
if busy:
    print(f"hint: BUSY mode={mode} prompt_tokens={pt} completion_tokens={ct} busy_for_s={bf}")
    if isinstance(pt, int) and pt > 10000:
        print("hint: large prompt — cancel Kilo turn / new chat if wait exceeds ~2 min")
    if isinstance(bf, (int, float)) and bf > 120:
        print("hint: busy >120s — prefer new chat or smaller history")
else:
    print("hint: idle — ready for requests")
' 2>/dev/null || true
    fi
else
    echo "UNREACHABLE (loading or stopped?)"
    echo "  ./2_start_dflash.sh status | restart"
fi
echo

echo "=== models ==="
curl -sS -m 3 "http://${HOST}:${PORT}/v1/models" 2>/dev/null || echo "(no response)"
echo
echo

echo "=== log tail (${LOG_FILE}) ==="
if [ -f "$LOG_FILE" ]; then
    tail -n 25 "$LOG_FILE"
else
    echo "(no log file)"
fi
