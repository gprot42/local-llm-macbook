#!/usr/bin/env bash
# 2_start_llama.sh — Serve Ternary Bonsai 2 27B (PQ2_0 GGUF) as an OpenAI API
# via PrismML's llama.cpp fork. Public API on :8089/v1. Run ./1_setup_download.sh first.
#
#   --port PORT       Public API port (default: 8089)
#   --host HOST       Bind host (default: 127.0.0.1)
#   --ctx N           Context window (default: 81920; BONSAI_CTX env also works)
#   --think           Reasoning on with the default budget (default; BONSAI_THINK=1)
#   --think-budget N  Reasoning on, capped at N tokens (default 2048 ≈ PrismML "Medium";
#                     -1 = unlimited; BONSAI_THINK_BUDGET=N)
#   --no-think        Reasoning off (BONSAI_THINK=0) — direct tool calls, fastest
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
CTX="${BONSAI_CTX:-81920}"
SLOTS="${BONSAI_SLOTS:-1}"
CACHE_RAM="${BONSAI_CACHE_RAM:-24576}"
PRESENCE="${BONSAI_PRESENCE:-1.5}"
# Reasoning ON by default, but BUDGETED: this is a thinking model at "xhigh"
# effort, and unbounded it burns the whole output budget thinking and never emits
# the tool call ("hit its output limit while reasoning and produced no actionable
# output"). --reasoning-budget caps the thinking (PrismML's UI calls 2048
# "Medium", 8192 "High") so the tool call / file write always has room.
# --think-budget -1 lifts the cap; --no-think switches to direct answers.
THINK="${BONSAI_THINK:-1}"
THINK_BUDGET="${BONSAI_THINK_BUDGET:-2048}"
CMD=start

args=("$@")
for ((i=0; i<${#args[@]}; )); do
  case "${args[$i]}" in
    --port) PORT="${args[$((i+1))]:-$PORT}"; ((i+=2)) ;;
    --host) HOST="${args[$((i+1))]:-$HOST}"; ((i+=2)) ;;
    --ctx)  CTX="${args[$((i+1))]:-$CTX}"; ((i+=2)) ;;
    --think)    THINK=1; ((i+=1)) ;;
    --think-budget) THINK=1; THINK_BUDGET="${args[$((i+1))]:-2048}"; ((i+=2)) ;;
    --no-think) THINK=0; ((i+=1)) ;;
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
  # Instruct / non-thinking mode (--no-think): temp 0.7 / top-p 0.8 / top-k 20 /
  # min-p 0 / presence 1.5 (Qwen: never greedy-decode this family — it loops).
  REASON_ARGS=(--reasoning off)
  SAMPLING=(--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty "${PRESENCE}" --repeat-penalty 1.0)
  MODE=off
fi
echo "=== Ternary Bonsai 2 27B — llama.cpp fork on http://${HOST}:${PORT}/v1 ==="
echo "→ model $(basename "${MODEL}") | ctx ${CTX} | slots ${SLOTS} | cache-ram ${CACHE_RAM} MiB | --jinja tool calling | reasoning ${MODE}"
LOG="${SCRIPT_DIR}/.bonsai_llama.log"
# -fa on + default batch (-b 2048 / -ub 512): benchmarked fastest prefill on
# Metal (larger -ub was slower). No speculative decoding: PrismML measures it as
# a net loss for chat/agent workloads on Apple Silicon.
nohup "${BIN}" -m "${MODEL}" "${MMPROJ_ARG[@]}" --alias "${ALIAS}" \
  --host "${HOST}" --port "${PORT}" -ngl 999 -fa on -c "${CTX}" -np "${SLOTS}" \
  --cache-ram "${CACHE_RAM}" --jinja "${REASON_ARGS[@]}" "${SAMPLING[@]}" \
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
