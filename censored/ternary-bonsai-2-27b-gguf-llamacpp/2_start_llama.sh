#!/usr/bin/env bash
# 2_start_llama.sh — Serve Ternary Bonsai 2 27B (PQ2_0 GGUF) as an OpenAI API
# via PrismML's llama.cpp fork. Public API on :8089/v1. Run ./1_setup_download.sh first.
#
#   --port PORT   Public API port (default: 8089)
#   --host HOST   Bind host (default: 127.0.0.1)
#   --ctx N       Context window (default: 65536; BONSAI_CTX env also works)
#   status | stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/engine/build/bin/llama-server"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL="${MODELS_DIR}/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MMPROJ="${MODELS_DIR}/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
ALIAS=ternary-bonsai-2-27b
HOST=127.0.0.1
PORT=8089
CTX="${BONSAI_CTX:-65536}"
CMD=start

args=("$@")
for ((i=0; i<${#args[@]}; )); do
  case "${args[$i]}" in
    --port) PORT="${args[$((i+1))]:-$PORT}"; ((i+=2)) ;;
    --host) HOST="${args[$((i+1))]:-$HOST}"; ((i+=2)) ;;
    --ctx)  CTX="${args[$((i+1))]:-$CTX}"; ((i+=2)) ;;
    status|stop|start) CMD="${args[$i]}"; ((i+=1)) ;;
    *) echo "unknown arg: ${args[$i]}" >&2; exit 2 ;;
  esac
done

port_pid() { lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | head -1; }

case "${CMD}" in
  status)
    pid="$(port_pid || true)"
    [[ -n "${pid}" ]] && { echo "→ serving on :${PORT} (pid ${pid})"; curl -s -m3 "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true; } || echo "→ not running on :${PORT}"
    exit 0 ;;
  stop)
    pid="$(port_pid || true)"
    [[ -n "${pid}" ]] && { echo "→ stopping pid ${pid}"; kill -TERM "${pid}" 2>/dev/null || true; } || echo "→ nothing on :${PORT}"
    exit 0 ;;
esac

[[ -x "${BIN}" ]] || { echo "ERROR: llama-server not built — run ./1_setup_download.sh first." >&2; exit 1; }
[[ -f "${MODEL}" ]] || { echo "ERROR: model missing: ${MODEL} — run ./1_setup_download.sh." >&2; exit 1; }
[[ -n "$(port_pid || true)" ]] && { echo "→ already serving on :${PORT} (pid $(port_pid))."; exit 0; }

MMPROJ_ARG=()
[[ -f "${MMPROJ}" ]] && MMPROJ_ARG=(--mmproj "${MMPROJ}") || echo "→ note: mmproj missing, image input disabled"

echo "=== Ternary Bonsai 2 27B — llama.cpp fork on http://${HOST}:${PORT}/v1 ==="
echo "→ model $(basename "${MODEL}") | ctx ${CTX} | --jinja tool calling"
LOG="${SCRIPT_DIR}/.bonsai_llama.log"
# --jinja: native OpenAI-style tool calling. Sampling = Bonsai 2 base defaults
# (temp 1.0 / top-p 0.95 / top-k 20); Kilo overrides per-agent. Thinking on.
nohup "${BIN}" -m "${MODEL}" "${MMPROJ_ARG[@]}" --alias "${ALIAS}" \
  --host "${HOST}" --port "${PORT}" -ngl 999 -fa on -c "${CTX}" \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 \
  >>"${LOG}" 2>&1 &
SRV=$!
echo "→ pid ${SRV}; log ${LOG}; waiting for readiness ..."
for _ in $(seq 1 180); do
  if curl -sf -m2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "✅ ready — OpenAI API http://${HOST}:${PORT}/v1 (model id: ${ALIAS})"
    exit 0
  fi
  kill -0 "${SRV}" 2>/dev/null || { echo "ERROR: server exited — see ${LOG}"; tail -20 "${LOG}"; exit 1; }
  sleep 1
done
echo "ERROR: not ready in 180s — see ${LOG}"; tail -20 "${LOG}"; exit 1
