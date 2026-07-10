#!/usr/bin/env bash
# =============================================================================
# 2_run_mlx.sh — Run DiffusionGemma 26B A4B IT (MLX bf16)
#
# Model:  mlx-community/diffusiongemma-26B-A4B-it-bf16
# Engine: mlx-vlm (discrete diffusion VLM — text + image in, text out)
#
# Usage:
#   ./2_run_mlx.sh                                    # quick text smoke test
#   ./2_run_mlx.sh chat                               # interactive terminal chat
#   ./2_run_mlx.sh server                             # OpenAI API at :8080
#   ./2_run_mlx.sh stop                               # stop server on :8080
#   ./2_run_mlx.sh restart server                     # stop :8080, then start fresh
#   ./2_run_mlx.sh --prompt "Why is the sky blue?"    # custom text prompt
#   ./2_run_mlx.sh --image photo.jpg --prompt "Describe this image."
#   ./2_run_mlx.sh --enable-thinking --prompt "Solve 2+2 step by step."
#
# Diffusion sampling defaults follow Google's recommended EB sampler settings.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="diffusiongemma-26b-a4b-it-bf16"
HF_REPO="mlx-community/diffusiongemma-26B-A4B-it-bf16"
MODEL_PATH="$MODEL_DIR"
MODE="generate"
PORT=8080
HOST="127.0.0.1"

PROMPT="Why is the sky blue?"
IMAGE=""
MAX_TOKENS=512
SERVER_MAX_TOKENS=4096
TEMPERATURE=0.8
MAX_DENOISING_STEPS=48
BLOCK_LENGTH=256
THRESHOLD=0.1
STABILITY_STEPS=2
ENABLE_THINKING=false
PREFILL_STEP_SIZE=512
EXTRA_ARGS=()
DO_RESTART=false
STARTING_FILE="$SCRIPT_DIR/.server-starting"

stop_server_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 1
    fi
}

ensure_server_port() {
    local pids
    pids="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    if [[ "$DO_RESTART" == true ]]; then
        stop_server_on_port "$PORT"
        return 0
    fi

    if curl -sf --max-time 3 "http://${HOST}:${PORT}/health" 2>/dev/null \
        | grep -q "$MODEL_DIR"; then
        echo "→ Server already running on http://${HOST}:${PORT}/v1"
        echo "→ Model: $MODEL_DIR"
        echo "→ To stop:    ./2_run_mlx.sh stop"
        echo "→ To restart: ./2_run_mlx.sh restart server"
        exit 0
    fi

    echo "ERROR: Port ${PORT} is already in use by another process."
    echo "       Run: ./2_run_mlx.sh stop"
    echo "       Or:  ./2_run_mlx.sh restart server"
    echo "       Or:  ./2_run_mlx.sh server --port 8081  (then update kilo.json baseURL)"
    exit 1
}

ensure_server_start_slot() {
    if [[ -f "$STARTING_FILE" ]]; then
        local start_pid
        start_pid="$(<"$STARTING_FILE")"
        if [[ -n "$start_pid" ]] && kill -0 "$start_pid" 2>/dev/null; then
            echo "→ Server start already in progress (pid ${start_pid})."
            echo "→ Wait for model load to finish, then check: curl http://${HOST}:${PORT}/health"
            exit 0
        fi
        rm -f "$STARTING_FILE"
    fi
    echo $$ >"$STARTING_FILE"
    trap 'rm -f "$STARTING_FILE"' EXIT INT TERM
}

usage() {
    echo ""
    echo "Usage: ./2_run_mlx.sh [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  (default)   One-shot text generation (mlx_vlm.generate)"
    echo "  chat        Interactive multi-turn chat (mlx_vlm.chat)"
    echo "  server      OpenAI-compatible HTTP API (mlx_vlm.server)"
    echo "  stop        Stop any server on the port"
    echo "  restart     Stop any server on the port, then start (use with server)"
    echo ""
    echo "Options:"
    echo "  --prompt TEXT              User prompt (default: \"$PROMPT\")"
    echo "  --image PATH               Image path or URL (repeatable)"
    echo "  --max-tokens N             Max output tokens (default: $MAX_TOKENS)"
    echo "  --temperature T            Sampling temperature (default: $TEMPERATURE)"
    echo "  --max-denoising-steps N    Diffusion denoising steps (default: $MAX_DENOISING_STEPS)"
    echo "  --block-length N           Canvas block size (default: $BLOCK_LENGTH)"
    echo "  --threshold T              Entropy bound for EB sampler (default: $THRESHOLD)"
    echo "  --stability-steps N        Adaptive stop stability steps (default: $STABILITY_STEPS)"
    echo "  --enable-thinking          Enable reasoning / thinking mode"
    echo "  --prefill-step-size N      Lower if you hit GPU memory errors (default: $PREFILL_STEP_SIZE)"
    echo "  --port PORT                Server port (default: $PORT)"
    echo "  --host HOST                Server bind address (default: $HOST)"
    echo "  --help, -h                 Show this help"
    echo ""
    echo "Examples:"
    echo "  ./2_run_mlx.sh"
    echo "  ./2_run_mlx.sh chat --enable-thinking"
    echo "  ./2_run_mlx.sh --image ~/Pictures/test.jpg --prompt \"What is in this image?\""
    echo "  ./2_run_mlx.sh server"
    echo "  ./2_run_mlx.sh stop"
    echo "  ./2_run_mlx.sh restart server"
    echo ""
    exit 0
}

ensure_model_weights() {
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "ERROR: Model weights not found at: $MODEL_DIR"
        echo "       Run: ./1_setup_download.sh"
        exit 1
    fi

    echo "→ Verifying all weights are downloaded ..."
    if python3 "$SCRIPT_DIR/validate_model.py" "$MODEL_DIR"; then
        MODEL_PATH="$MODEL_DIR"
        return 0
    fi

    echo ""
    echo "ERROR: Local model weights are incomplete or still downloading."
    echo "       Run: ./1_setup_download.sh"
    MISSING=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && MISSING+=("$line")
    done < <(python3 "$SCRIPT_DIR/validate_model.py" --list-missing "$MODEL_DIR" 2>/dev/null || true)
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo ""
        echo "Missing files:"
        for f in "${MISSING[@]}"; do
            echo "  - $f"
        done
    fi
    exit 1
}

activate_venv() {
    if [[ ! -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
        echo "ERROR: venv not found. Run ./1_setup_download.sh first."
        exit 1
    fi
    source "$SCRIPT_DIR/venv/bin/activate"
    export PATH="$SCRIPT_DIR/venv/bin:$PATH"
}

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        chat) MODE="chat"; ((i+=1)) ;;
        server) MODE="server"; ((i+=1)) ;;
        stop) MODE="stop"; ((i+=1)) ;;
        restart) DO_RESTART=true; ((i+=1)) ;;
        --help|-h) usage ;;
        --prompt)
            PROMPT="${args[$((i+1))]:-}"
            ((i+=2))
            ;;
        --image)
            IMAGE="${args[$((i+1))]:-}"
            EXTRA_ARGS+=(--image "$IMAGE")
            ((i+=2))
            ;;
        --max-tokens)
            MAX_TOKENS="${args[$((i+1))]:-$MAX_TOKENS}"
            SERVER_MAX_TOKENS="$MAX_TOKENS"
            ((i+=2))
            ;;
        --temperature)
            TEMPERATURE="${args[$((i+1))]:-$TEMPERATURE}"
            ((i+=2))
            ;;
        --max-denoising-steps)
            MAX_DENOISING_STEPS="${args[$((i+1))]:-$MAX_DENOISING_STEPS}"
            ((i+=2))
            ;;
        --block-length)
            BLOCK_LENGTH="${args[$((i+1))]:-$BLOCK_LENGTH}"
            ((i+=2))
            ;;
        --threshold)
            THRESHOLD="${args[$((i+1))]:-$THRESHOLD}"
            ((i+=2))
            ;;
        --stability-steps)
            STABILITY_STEPS="${args[$((i+1))]:-$STABILITY_STEPS}"
            ((i+=2))
            ;;
        --enable-thinking)
            ENABLE_THINKING=true
            ((i+=1))
            ;;
        --prefill-step-size)
            PREFILL_STEP_SIZE="${args[$((i+1))]:-$PREFILL_STEP_SIZE}"
            ((i+=2))
            ;;
        --port)
            PORT="${args[$((i+1))]:-$PORT}"
            ((i+=2))
            ;;
        --host)
            HOST="${args[$((i+1))]:-$HOST}"
            ((i+=2))
            ;;
        *)
            echo "Unknown argument: ${args[$i]}"
            usage
            ;;
    esac
done

for ((j = 0; j < ${#EXTRA_ARGS[@]}; j += 2)); do
    if [[ "${EXTRA_ARGS[j]}" == "--image" ]]; then
        img="${EXTRA_ARGS[j + 1]}"
        if [[ "$img" != http://* && "$img" != https://* && ! -f "$img" ]]; then
            echo "ERROR: Image not found: $img"
            echo "       Pass a real file path, e.g. test-images/red-square.png or ~/Desktop/screenshot.png"
            exit 1
        fi
    fi
done

echo "=== DiffusionGemma 26B A4B IT (MLX bf16) ==="
echo "→ Mode: $MODE"
echo ""

if [[ "$MODE" == "stop" ]]; then
    pids="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        echo "→ No process listening on port ${PORT}."
    else
        stop_server_on_port "$PORT"
        echo "→ Server stopped."
    fi
    rm -f "$STARTING_FILE"
    exit 0
fi

activate_venv
chmod +x "$SCRIPT_DIR/apply_local_patches.sh" 2>/dev/null || true
"$SCRIPT_DIR/apply_local_patches.sh"
ensure_model_weights

COMMON_ARGS=(
    --model "$MODEL_PATH"
    --prefill-step-size "$PREFILL_STEP_SIZE"
)

THINKING_ARGS=()
if [[ "$ENABLE_THINKING" == true ]]; then
    THINKING_ARGS+=(--enable-thinking)
fi

case "$MODE" in
    generate)
        GEN_ARGS=(
            "${COMMON_ARGS[@]}"
            --trust-remote-code
            --prompt "$PROMPT"
            --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE"
            --max-denoising-steps "$MAX_DENOISING_STEPS"
            --block-length "$BLOCK_LENGTH"
            --threshold "$THRESHOLD"
            --stability-steps "$STABILITY_STEPS"
            --verbose
            "${THINKING_ARGS[@]}"
            "${EXTRA_ARGS[@]}"
        )
        echo "→ Running: python -m mlx_vlm.generate ${GEN_ARGS[*]}"
        echo ""
        exec python -m mlx_vlm.generate "${GEN_ARGS[@]}"
        ;;
    chat)
        CHAT_ARGS=(
            "${COMMON_ARGS[@]}"
            --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE"
            --verbose
            "${THINKING_ARGS[@]}"
        )
        echo "→ Interactive chat. Type messages and press Enter; Ctrl+C to exit."
        echo ""
        exec python -m mlx_vlm.chat "${CHAT_ARGS[@]}"
        ;;
    server)
        ensure_server_port
        ensure_server_start_slot
        SERVER_ARGS=(
            --model "$MODEL_PATH"
            --host "$HOST"
            --port "$PORT"
            --trust-remote-code
            --prefill-step-size "$PREFILL_STEP_SIZE"
            --max-tokens "$SERVER_MAX_TOKENS"
            "${THINKING_ARGS[@]}"
        )
        echo "→ Starting OpenAI-compatible API at http://${HOST}:${PORT}/v1"
        echo "→ Server max-tokens: $SERVER_MAX_TOKENS (Kilo/agent default)"
        echo "→ Test: curl http://${HOST}:${PORT}/v1/models"
        echo ""
        python -m mlx_vlm.server "${SERVER_ARGS[@]}"
        ;;
esac