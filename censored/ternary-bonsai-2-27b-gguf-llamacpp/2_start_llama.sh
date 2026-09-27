#!/usr/bin/env bash
# 2_start_llama.sh — Serve Ternary Bonsai 2 27B (PQ2_0 GGUF) as an OpenAI API
# via PrismML's llama.cpp fork. Public API on :8089/v1. Run ./1_setup_download.sh first.
#
#   --port PORT       Public API port — the loop proxy (default: 8089)
#   --engine-port N   llama-server port behind the proxy (default: 8099; BONSAI_ENGINE_PORT)
#   --host HOST       Bind host for the public port (default: 127.0.0.1; the engine stays on loopback)
#   --ctx N           Context window (default: 81920; BONSAI_CTX env also works)
#   --no-think        Reasoning off (default; BONSAI_THINK=0) — direct tool calls, fastest
#   --think           Reasoning on, capped at the default 2048-token budget (BONSAI_THINK=1)
#   --think-budget N  Reasoning on, capped at N tokens (2048 ≈ PrismML "Medium", 512 "Low";
#                     -1 = unlimited; BONSAI_THINK_BUDGET=N)
#   status | stop
#
# Tuning env (defaults chosen for one OpenCode/Kilo user on Apple Silicon — see README):
#   BONSAI_SLOTS=1          llama-server slots (-np). 1 = the whole window + KV cache belong
#                           to one conversation. The auto default (4, unified KV) split the
#                           window and re-prefilled OpenCode's 14k-token prompt on every slot hop.
#   BONSAI_CACHE_RAM=24576  RAM prompt cache (MiB). A full 49k-token conversation is ~9 GiB of
#                           state; the 8 GiB default couldn't hold one, so every title/subagent
#                           request forced a from-scratch re-prefill.
#   BONSAI_PRESENCE=1.5     presence penalty in non-thinking mode (PrismML/Qwen instruct preset).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/engine/build/bin/llama-server"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL="${MODELS_DIR}/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MMPROJ="${MODELS_DIR}/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
ALIAS=ternary-bonsai-2-27b
HOST=127.0.0.1
PORT=8089
# loop_proxy.py owns the public port and forwards to llama-server on the engine
# port. When the model has just repeated a call (same result a 2nd/3rd time with
# no write/edit between, or the same call issued a 3rd time) it nudges in place:
# appends a "[Harness] do not run it again" note to that tool result and forwards,
# so the turn continues without the user. It only ends a turn, without calling
# the model, on the 4th identical result, the 6th issuance, or after 200 tool
# calls (LOOP_MAX_ROUNDS) — a prompt rule alone did not hold (one session
# repeated a call 782 times). Streamed tokens pass through.
ENGINE_PORT="${BONSAI_ENGINE_PORT:-8099}"
PROXY="${SCRIPT_DIR}/loop_proxy.py"
CTX="${BONSAI_CTX:-81920}"
SLOTS="${BONSAI_SLOTS:-1}"
CACHE_RAM="${BONSAI_CACHE_RAM:-24576}"
PRESENCE="${BONSAI_PRESENCE:-1.5}"
# Reasoning OFF by default for agentic use. Two measured reasons (see README):
#  1. Unbounded, the model (xhigh effort) burns the whole output budget thinking
#     and never emits the tool call ("produced no actionable output").
#  2. Even budgeted, OpenCode sends every step's reasoning_content back and the
#     template keeps it for the whole tool loop, so the context grows ~2k per
#     step -> compaction can't get under the limit ("Compaction exhausted") and
#     turns take ~2x longer.
# --think / --think-budget N re-enable it under a budget (PrismML's UI calls
# 2048 "Medium", 8192 "High"); -1 lifts the cap.
THINK="${BONSAI_THINK:-0}"
THINK_BUDGET="${BONSAI_THINK_BUDGET:-2048}"
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

MMPROJ_ARG=()
[[ -f "${MMPROJ}" ]] && MMPROJ_ARG=(--mmproj "${MMPROJ}") || echo "→ note: mmproj missing, image input disabled"

# Sampling presets from the PrismML model card (= the GGUF's general.sampling.*
# metadata and Qwen3.8's generation_config). The client must NOT override these
# (OpenCode: model "temperature": false, no sampling in options) or the preset
# for the active mode is lost.
if [[ "${THINK}" == "1" ]]; then
  # Thinking mode: temp 1.0 / top-p 0.95 / top-k 20 / min-p 0 / presence 0.
  # --reasoning-preserve keeps prior-turn reasoning_content in the prompt (the
  # Qwen3.8 template supports it) so tool loops stay coherent and cache-friendly.
  REASON_ARGS=(--reasoning on --reasoning-preserve --reasoning-budget "${THINK_BUDGET}")
  SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0)
  MODE="on (budget $([[ "${THINK_BUDGET}" == "-1" ]] && echo unlimited || echo "${THINK_BUDGET} tokens"))"
else
  # Instruct / non-thinking mode (default): temp 0.7 / top-p 0.8 / top-k 20 /
  # min-p 0 / presence 1.5 (Qwen: never greedy-decode this family — it loops).
  REASON_ARGS=(--reasoning off)
  SAMPLING=(--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty "${PRESENCE}" --repeat-penalty 1.0)
  MODE=off
fi
echo "=== Ternary Bonsai 2 27B — llama.cpp fork on http://${HOST}:${PORT}/v1 (engine :${ENGINE_PORT}) ==="
echo "→ model $(basename "${MODEL}") | ctx ${CTX} | slots ${SLOTS} | cache-ram ${CACHE_RAM} MiB | --jinja tool calling | reasoning ${MODE}"
LOG="${SCRIPT_DIR}/.bonsai_llama.log"
PROXY_LOG="${SCRIPT_DIR}/.bonsai_proxy.log"
# -fa on + default batch (-b 2048 / -ub 512): benchmarked fastest prefill on
# Metal (larger -ub was slower). No speculative decoding: PrismML measures it as
# a net loss for chat/agent workloads on Apple Silicon. The engine binds
# loopback only; the proxy is what listens on ${HOST}:${PORT}.
nohup "${BIN}" -m "${MODEL}" "${MMPROJ_ARG[@]}" --alias "${ALIAS}" \
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
