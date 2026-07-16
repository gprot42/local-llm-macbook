#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Gemma 4 31B JANG_4M CRACK — vllm-mlx Server + Kilo Code Proxy ==="

DO_RESTART=false
# Default on: Kilo/Continue agent turns need Harmony bias, temp floor, tool repair.
USE_PROXY=true
PROXY_DEBUG=false
CONTINUOUS_BATCHING=false
ENABLE_METRICS=false
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER="gemma4"
REASONING_PARSER="gemma4"
API_KEY=""
RATE_LIMIT=""

for arg in "$@"; do
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        echo ""
        echo "Usage: ./2_start_mlx.sh [options]"
        echo ""
        echo "  Gemma 4 31B JANG_4M CRACK (dealignai/Gemma-4-31B-JANG_4M-CRACK)"
        echo "  Already MLX-native mixed quant (~22 GB). 64–80 GB+ unified RAM recommended."
        echo ""
        echo "  --proxy                    Enable Kilo steering proxy on :8080 (default: on)"
        echo "  --no-proxy                 Raw vllm-mlx on :8080 (no steering proxy)"
        echo "  --batching                 Enable continuous batching (multi-user; needs lots of RAM)"
        echo "  --no-batching              Disable continuous batching (default)"
        echo "  --debug                    Verbose DEBUG logging in the proxy"
        echo "  --enable-metrics           Expose /metrics on vllm-mlx and proxy"
        echo "  --enable-auto-tool-choice  Enable tool-call parsing (default: on, parser gemma4)"
        echo "  --no-auto-tool-choice      Disable tool-call parsing"
        echo "  --tool-call-parser PARSER  Override tool parser (default: gemma4)"
        echo "  --reasoning-parser PARSER  Reasoning splitter (default: gemma4)"
        echo "  --no-reasoning-parser      Disable reasoning parser"
        echo "  --api-key KEY              Require API key for all requests"
        echo "  --rate-limit N             Requests-per-minute limit"
        echo "  restart                    Stop :8080/:8090, then start"
        echo "  --help, -h                 Show this help"
        echo ""
        echo "Related:"
        echo "  ../gemma4-server-heretic-31b-mlx/  — Heretic 4-bit sibling"
        echo "  ../gemma4-server-atomicchat-mlx-31b-2026-07-15/  — stock aligned 31B IT (AtomicChat)"
        echo ""
        exit 0
    }
    [[ "$arg" == "--proxy" ]]                 && USE_PROXY=true
    [[ "$arg" == "--no-proxy" ]]              && USE_PROXY=false
    [[ "$arg" == "--batching" ]]              && CONTINUOUS_BATCHING=true
    [[ "$arg" == "--no-batching" ]]           && CONTINUOUS_BATCHING=false
    [[ "$arg" == "--debug" ]]                 && PROXY_DEBUG=true
    [[ "$arg" == "--enable-metrics" ]]        && ENABLE_METRICS=true
    [[ "$arg" == "--enable-auto-tool-choice" ]] && ENABLE_AUTO_TOOL_CHOICE=true
    [[ "$arg" == "--no-auto-tool-choice" ]]  && ENABLE_AUTO_TOOL_CHOICE=false
    [[ "$arg" == "--no-reasoning-parser" ]]  && REASONING_PARSER=""
    [[ "$arg" == "restart" ]] && DO_RESTART=true
done

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
        echo "→ Force-stopping stubborn process(es) on port ${port} ..."
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 1
    fi
}

i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --tool-call-parser) TOOL_CALL_PARSER="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --reasoning-parser) REASONING_PARSER="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --api-key)          API_KEY="${args[$((i+1))]:-}";          ((i+=2)) ;;
        --rate-limit)       RATE_LIMIT="${args[$((i+1))]:-}";       ((i+=2)) ;;
        *) ((i+=1)) ;;
    esac
done

if [ "$ENABLE_AUTO_TOOL_CHOICE" = true ] && [ -z "$TOOL_CALL_PARSER" ]; then
    TOOL_CALL_PARSER="gemma4"
fi
if [ "$ENABLE_AUTO_TOOL_CHOICE" = false ]; then
    TOOL_CALL_PARSER=""
fi

MODEL_DIR="gemma-4-31b-jang-crack-mlx"
HF_REPO="dealignai/Gemma-4-31B-JANG_4M-CRACK"
echo "→ Using 31B JANG_4M CRACK (uncensored, multimodal)"

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"

if [ ! -f "$VALIDATE_MODEL" ]; then
    echo "ERROR: $VALIDATE_MODEL not found."
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Download all weights first:"
    echo "  ./1_setup_download.sh"
    exit 1
fi
echo "→ Verifying all model weights are present ..."
if ! python3 "$VALIDATE_MODEL" "$MODEL_DIR"; then
    echo ""
    echo "Server will not start until every weight shard is on disk."
    echo "Fix: ./1_setup_download.sh"
    echo "  or: hf download $HF_REPO --local-dir $MODEL_DIR"
    exit 1
fi
MODEL_PATH="$MODEL_DIR"
echo "→ Using validated local weights: $MODEL_DIR"

PUBLIC_PORT=8080
MLX_PORT=8090
[ "$USE_PROXY" = false ] && MLX_PORT=$PUBLIC_PORT

if [ "$DO_RESTART" = true ]; then
    stop_server_on_port "$PUBLIC_PORT"
    if [ "$USE_PROXY" = true ] && [ "$MLX_PORT" != "$PUBLIC_PORT" ]; then
        stop_server_on_port "$MLX_PORT"
    fi
    echo ""
fi

repair_relocated_venv() {
    local venv="$SCRIPT_DIR/venv"
    [[ -d "$venv/bin" ]] || return 0

    local sample="" f shebang interp old_path=""
    for sample in "$venv/bin/vllm-mlx" "$venv/bin/pip" "$venv/bin/hf" "$venv/bin/mlx_lm.chat"; do
        [[ -f "$sample" ]] || continue
        shebang="$(head -1 "$sample" 2>/dev/null || true)"
        if [[ "$shebang" == "#!"* ]]; then
            interp="${shebang#\#!}"
        elif [[ "$shebang" == /*/python* ]]; then
            interp="$shebang"
        else
            continue
        fi
        if [[ -n "$interp" && ! -e "$interp" ]]; then
            old_path="$(dirname "$(dirname "$interp")")"
            break
        fi
    done

    if [[ -z "$old_path" && -f "$venv/bin/activate" ]]; then
        local old_ve
        old_ve="$(grep -E 'export VIRTUAL_ENV=' "$venv/bin/activate" | head -1 | sed -E 's/.*VIRTUAL_ENV=//' | tr -d "\"'")"
        if [[ -n "$old_ve" && "$old_ve" != "$venv" && ! -d "$old_ve" ]]; then
            old_path="$old_ve"
        fi
    fi

    if [[ -z "$old_path" || "$old_path" == "$venv" ]]; then
        return 0
    fi

    echo "→ Detected moved project directory; repairing venv paths"
    echo "   $old_path → $venv"
    while IFS= read -r -d '' f; do
        if grep -qF "$old_path" "$f" 2>/dev/null; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                sed -i '' "s|${old_path}|${venv}|g" "$f"
            else
                sed -i "s|${old_path}|${venv}|g" "$f"
            fi
        fi
        shebang="$(head -1 "$f" 2>/dev/null || true)"
        if [[ "$shebang" == "$venv/bin/python"* ]]; then
            local tmp
            tmp="$(mktemp)"
            { echo "#!$shebang"; tail -n +2 "$f"; } > "$tmp"
            mv "$tmp" "$f"
            chmod +x "$f" 2>/dev/null || true
        fi
    done < <(find "$venv/bin" -type f -print0 2>/dev/null)
    if [[ -f "$venv/pyvenv.cfg" ]] && grep -qF "$old_path" "$venv/pyvenv.cfg" 2>/dev/null; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            sed -i '' "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg"
        else
            sed -i "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg"
        fi
    fi
    echo "→ venv paths repaired."
}

repair_relocated_venv

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import vllm_mlx" 2>/dev/null; then
    echo "→ Creating virtualenv..."
    python3 -m venv venv
    VENV_PY="$SCRIPT_DIR/venv/bin/python"
    echo "→ Installing vllm-mlx and proxy deps (first run — takes a minute)..."
    "$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade pip
    "$VENV_PY" -m pip install --quiet --no-cache-dir -r requirements.txt \
        || { echo "ERROR: dependency install failed"; exit 1; }
    echo "→ Dependencies installed."
fi

export PATH="$SCRIPT_DIR/venv/bin:$PATH"

chmod +x apply_local_patches.sh check_upstream_patches.sh 2>/dev/null || true
./apply_local_patches.sh

echo "→ Continuous batching:       $([ "$CONTINUOUS_BATCHING" = true ] && echo "Enabled (multi-user)" || echo "Disabled (single-user max throughput)")"
echo "→ Kilo Code / Continue proxy:$([ "$USE_PROXY" = true ] && echo " Enabled (default; pass --no-proxy for raw vllm-mlx)" || echo " Disabled (--no-proxy)")"
echo "→ Proxy debug logging:       $([ "$PROXY_DEBUG" = true ] && echo "Enabled (--debug)" || echo "Disabled")"
echo "→ Metrics endpoint:          $([ "$ENABLE_METRICS" = true ] && echo "Enabled (--enable-metrics)" || echo "Disabled")"
echo "→ Auto tool choice:          $([ "$ENABLE_AUTO_TOOL_CHOICE" = true ] && echo "Enabled (parser: ${TOOL_CALL_PARSER})" || echo "Disabled")"
echo "→ Reasoning parser:          $([ -n "$REASONING_PARSER" ] && echo "$REASONING_PARSER" || echo "Disabled")"
[ -n "$API_KEY" ]           && echo "→ API key auth:              Enabled"
[ -n "$RATE_LIMIT" ]        && echo "→ Rate limit:                ${RATE_LIMIT} req/min"
echo "→ Public endpoint:           http://localhost:$PUBLIC_PORT/v1"
[ "$USE_PROXY" = true ] && echo "→ Internal vllm-mlx:         http://localhost:$MLX_PORT/v1"
echo "→ Model:                     $MODEL_PATH"
echo ""

cleanup() {
    echo ""
    echo "→ Shutting down..."
    [ -n "${PROXY_PID:-}" ] && kill -TERM "$PROXY_PID" 2>/dev/null
    [ -n "${MLX_PID:-}" ]   && kill -TERM "$MLX_PID"   2>/dev/null
    for _ in 1 2 3 4; do
        sleep 0.5
        [ -n "${MLX_PID:-}" ] && kill -0 "$MLX_PID" 2>/dev/null || break
    done
    [ -n "${PROXY_PID:-}" ] && kill -KILL "$PROXY_PID" 2>/dev/null
    [ -n "${MLX_PID:-}" ]   && kill -KILL "$MLX_PID"   2>/dev/null
    pkill -KILL -P $$ 2>/dev/null
    kill -KILL 0 2>/dev/null
    exit 0
}
trap cleanup INT TERM HUP

if ! "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" >/dev/null 2>&1; then
    echo "ERROR: model weights failed re-check immediately before server start."
    "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    exit 1
fi

VLLM_CMD=("$VENV_PY" -m vllm_mlx.cli serve "$MODEL_PATH" --port "$MLX_PORT" --host 127.0.0.1)
[ "$CONTINUOUS_BATCHING" = true ]     && VLLM_CMD+=(--continuous-batching)
[ "$ENABLE_METRICS" = true ]          && VLLM_CMD+=(--enable-metrics)
if [ "$ENABLE_AUTO_TOOL_CHOICE" = true ]; then
    VLLM_CMD+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
[ -n "$REASONING_PARSER" ]            && VLLM_CMD+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$API_KEY" ]                     && VLLM_CMD+=(--api-key "$API_KEY")
[ -n "$RATE_LIMIT" ]                  && VLLM_CMD+=(--rate-limit "$RATE_LIMIT")

"${VLLM_CMD[@]}" &
MLX_PID=$!

if [ "$USE_PROXY" = true ]; then
    echo "→ Waiting for vllm-mlx to bind 127.0.0.1:$MLX_PORT ..."
    for i in $(seq 1 180); do
        if curl -sf --max-time 1 "http://127.0.0.1:$MLX_PORT/v1/models" >/dev/null 2>&1; then
            echo "→ vllm-mlx is up after ${i}s — starting proxy."
            break
        fi
        if ! kill -0 "$MLX_PID" 2>/dev/null; then
            echo "ERROR: vllm-mlx process exited during startup."
            if lsof -ti ":$MLX_PORT" >/dev/null 2>&1; then
                echo "       Port $MLX_PORT is in use. Run: ./2_start_mlx.sh restart"
            fi
            echo "       On 31B, exit 134 / Metal OOM is common if another MLX server is running."
            cleanup
        fi
        sleep 1
    done
    PROXY_CMD=("$VENV_PY" "$SCRIPT_DIR/gemma4_mlx_kilo_proxy.py"
        --upstream "http://127.0.0.1:$MLX_PORT"
        --host 127.0.0.1
        --port "$PUBLIC_PORT"
        --model "$MODEL_DIR")
    [ "$PROXY_DEBUG" = true ] && PROXY_CMD+=(--debug)
    "${PROXY_CMD[@]}" &
    PROXY_PID=$!
    for i in $(seq 1 30); do
        if curl -sf --max-time 1 "http://127.0.0.1:$PUBLIC_PORT/healthz" >/dev/null 2>&1; then
            echo ""
            echo "============================================================"
            echo "  READY — Gemma 4 31B JANG_4M CRACK (vllm-mlx + Kilo proxy)"
            echo "============================================================"
            echo "  OpenAI API:  http://localhost:$PUBLIC_PORT/v1"
            echo "  Model ID:    $MODEL_DIR"
            echo "  Kilo Code:   open http://localhost:$PUBLIC_PORT"
            echo "============================================================"
            echo ""
            break
        fi
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo "ERROR: proxy process exited during startup."
            cleanup
        fi
        sleep 1
    done
else
    echo "→ Waiting for vllm-mlx to bind 127.0.0.1:$PUBLIC_PORT ..."
    for i in $(seq 1 180); do
        if curl -sf --max-time 1 "http://127.0.0.1:$PUBLIC_PORT/v1/models" >/dev/null 2>&1; then
            echo ""
            echo "============================================================"
            echo "  READY — Gemma 4 31B JANG_4M CRACK (vllm-mlx direct)"
            echo "============================================================"
            echo "  OpenAI API:  http://localhost:$PUBLIC_PORT/v1"
            echo "  Model ID:    $MODEL_DIR"
            echo "  Kilo Code:   point baseURL at http://localhost:$PUBLIC_PORT/v1"
            echo "============================================================"
            echo ""
            break
        fi
        if ! kill -0 "$MLX_PID" 2>/dev/null; then
            echo "ERROR: vllm-mlx process exited during startup."
            if lsof -ti ":$PUBLIC_PORT" >/dev/null 2>&1; then
                echo "       Port $PUBLIC_PORT is in use. Run: ./2_start_mlx.sh restart"
            fi
            echo "       Never run sibling Gemma servers on :8080 at the same time."
            cleanup
        fi
        sleep 1
    done
fi

while kill -0 "$MLX_PID" 2>/dev/null; do
    wait "$MLX_PID" 2>/dev/null
done
cleanup
