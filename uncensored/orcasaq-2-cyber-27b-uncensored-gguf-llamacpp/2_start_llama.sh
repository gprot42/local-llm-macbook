#!/usr/bin/env bash
# 2_start_llama.sh — Serve OrcaSAQ-2 Cyber 27B Uncensored (GGUF) as an OpenAI API
# via llama.cpp. Public API on :8090/v1 (loop proxy) -> engine on :8100.
# Run ./1_setup_download.sh first.
#
#   --port PORT       Public API port — the loop proxy (default: 8090)
#   --engine-port N   llama-server port behind the proxy (default: 8100; ORCA_ENGINE_PORT)
#   --host HOST       Bind host for the public port (default: 127.0.0.1; the engine stays loopback)
#   --ctx N           Context window (default: 81920; ORCA_CTX env also works; card max 262144)
#   --no-think        Reasoning off (default; ORCA_THINK=0) — direct tool calls, fastest
#   --think           Reasoning on, capped at the default 2048-token budget (ORCA_THINK=1)
#   --think-budget N  Reasoning on, capped at N tokens (-1 = unlimited; ORCA_THINK_BUDGET=N)
#   status | stop
#
# Sampling is the Qwen3.8 base preset (this is a Qwen3.8-27B fine-tune). Confirm
# against the gated model card if OrcaSAQ recommends different values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/engine/build/bin/llama-server"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL="${MODELS_DIR}/OrcaSAQ-2-27B-Uncensored.gguf"
ALIAS=orcasaq-2-cyber-27b
HOST=127.0.0.1
PORT=8090
# loop_proxy.py owns the public port and forwards to llama-server on the engine
# port. When the model has just repeated a call (same result a 2nd/3rd time with
# no write/edit between, or the same call issued a 3rd time) it corrects in place:
# replaces that result with a "[Harness] REFUSED" error quoting the original
# output (plus a user-role directive from the 3rd repeat) and forwards, so the
# turn continues without the user. It only ends a turn, without calling the
# model, on the 4th identical result, the 6th issuance, or after 200 tool calls
# (LOOP_MAX_ROUNDS). Streamed tokens pass through. Agentic terminal / security
# workflows loop, and a prompt rule alone does not hold on a small local model.
ENGINE_PORT="${ORCA_ENGINE_PORT:-8100}"
PROXY="${SCRIPT_DIR}/loop_proxy.py"
CTX="${ORCA_CTX:-81920}"
SLOTS="${ORCA_SLOTS:-1}"
CACHE_RAM="${ORCA_CACHE_RAM:-24576}"
PRESENCE="${ORCA_PRESENCE:-1.5}"
# Reasoning OFF by default for agentic use (see the Bonsai stack's README for the
# measured reasons: unbounded thinking exhausts the output budget before the tool
# call, and replayed reasoning_content grows context until compaction fails).
THINK="${ORCA_THINK:-0}"
THINK_BUDGET="${ORCA_THINK_BUDGET:-2048}"
CMD=start

args=("$@")
for ((i=0; i<${#args[@]}; )); do
  case "${args[$i]}" in
    --port) PORT="${args[$((i+1))]:-$PORT}"; ((i+=2)) ;;
    --engine-port) ENGINE_PORT="${args[$((i+1))]:-$ENGINE_PORT}"; ((i+=2)) ;;
    --host) HOST="${args[$((i+1))]:-$HOST}"; ((i+=2)) ;;
    --ctx)  CTX="${args[$((i+1))]:-$CTX}"; ((i+=2)) ;;
    --think)    THINK=1; ((i+=1)) ;;
    --think-budget) THINK=1; THINK_BUDGET="${args[$((i+1))]:-2048}"; ((i+=2)) ;;
    --no-think) THINK=0; ((i+=1)) ;;
    status|stop|start) CMD="${args[$i]}"; ((i+=1)) ;;
    *) echo "unknown arg: ${args[$i]}" >&2; exit 2 ;;
  esac
done

pid_on() { lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -1; }
stop_port() { local pid; pid="$(pid_on "$1" || true)"; [[ -n "${pid}" ]] && { echo "→ stopping :$1 (pid ${pid})"; kill -TERM "${pid}" 2>/dev/null || true; } || echo "→ nothing on :$1"; }

case "${CMD}" in
  status)
    ppid="$(pid_on "${PORT}" || true)"; epid="$(pid_on "${ENGINE_PORT}" || true)"
    [[ -n "${ppid}" ]] && echo "→ loop proxy on :${PORT} (pid ${ppid})" || echo "→ loop proxy not running on :${PORT}"
    [[ -n "${epid}" ]] && echo "→ llama-server on :${ENGINE_PORT} (pid ${epid})" || echo "→ llama-server not running on :${ENGINE_PORT}"
    [[ -n "${ppid}" ]] && curl -s -m3 "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true
    exit 0 ;;
  stop)
    stop_port "${PORT}"; stop_port "${ENGINE_PORT}"
    exit 0 ;;
esac

[[ -x "${BIN}" ]] || { echo "ERROR: llama-server not built — run ./1_setup_download.sh first." >&2; exit 1; }
[[ -f "${MODEL}" ]] || { echo "ERROR: model missing: ${MODEL} — run ./1_setup_download.sh." >&2; exit 1; }
[[ -f "${PROXY}" ]] || { echo "ERROR: loop proxy missing: ${PROXY}" >&2; exit 1; }
[[ -n "$(pid_on "${PORT}" || true)" ]] && { echo "→ already serving on :${PORT} (pid $(pid_on "${PORT}")). Use: $0 stop"; exit 0; }
[[ -n "$(pid_on "${ENGINE_PORT}" || true)" ]] && { echo "→ engine port :${ENGINE_PORT} busy (pid $(pid_on "${ENGINE_PORT}")). Use: $0 stop"; exit 1; }

if [[ "${THINK}" == "1" ]]; then
  REASON_ARGS=(--reasoning on --reasoning-preserve --reasoning-budget "${THINK_BUDGET}")
  SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0)
  MODE="on (budget $([[ "${THINK_BUDGET}" == "-1" ]] && echo unlimited || echo "${THINK_BUDGET} tokens"))"
else
  REASON_ARGS=(--reasoning off)
  SAMPLING=(--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty "${PRESENCE}" --repeat-penalty 1.0)
  MODE=off
fi
echo "=== OrcaSAQ-2 Cyber 27B Uncensored — llama.cpp on http://${HOST}:${PORT}/v1 (engine :${ENGINE_PORT}) ==="
echo "→ model $(basename "${MODEL}") | ctx ${CTX} | slots ${SLOTS} | cache-ram ${CACHE_RAM} MiB | --jinja tool calling | reasoning ${MODE}"
LOG="${SCRIPT_DIR}/.orcasaq_llama.log"
PROXY_LOG="${SCRIPT_DIR}/.orcasaq_proxy.log"
# The engine binds loopback only; the loop proxy is what listens on ${HOST}:${PORT}.
nohup "${BIN}" -m "${MODEL}" --alias "${ALIAS}" \
  --host 127.0.0.1 --port "${ENGINE_PORT}" -ngl 999 -fa on -c "${CTX}" -np "${SLOTS}" \
  --cache-ram "${CACHE_RAM}" --jinja "${REASON_ARGS[@]}" "${SAMPLING[@]}" \
  >>"${LOG}" 2>&1 &
SRV=$!
echo "→ engine pid ${SRV}; log ${LOG}; waiting for readiness ..."
READY=false
for _ in $(seq 1 180); do
  if curl -sf -m2 "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then READY=true; break; fi
  kill -0 "${SRV}" 2>/dev/null || { echo "ERROR: engine exited — see ${LOG}"; tail -20 "${LOG}"; exit 1; }
  sleep 1
done
[[ "${READY}" == true ]] || { echo "ERROR: engine not ready in 180s — see ${LOG}"; tail -20 "${LOG}"; exit 1; }

nohup python3 "${PROXY}" --listen "${HOST}:${PORT}" --upstream "127.0.0.1:${ENGINE_PORT}" >>"${PROXY_LOG}" 2>&1 &
PXY=$!
for _ in $(seq 1 20); do
  if curl -sf -m2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "✅ ready — OpenAI API http://${HOST}:${PORT}/v1 (model id: ${ALIAS}) | loop proxy pid ${PXY} -> engine :${ENGINE_PORT}"
    exit 0
  fi
  kill -0 "${PXY}" 2>/dev/null || { echo "ERROR: loop proxy exited — see ${PROXY_LOG}"; tail -20 "${PROXY_LOG}"; stop_port "${ENGINE_PORT}"; exit 1; }
  sleep 1
done
echo "ERROR: loop proxy not answering on :${PORT} — see ${PROXY_LOG}"; tail -20 "${PROXY_LOG}"; stop_port "${ENGINE_PORT}"; exit 1
