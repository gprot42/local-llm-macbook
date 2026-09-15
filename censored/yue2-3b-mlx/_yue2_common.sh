#!/usr/bin/env bash
# Shared paths/helpers for the YuE2-3B MLX stack. Source from sibling scripts.

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${STACK_DIR}/.yue2_config"
ENGINE_DIR="${STACK_DIR}/engine"
MODELS_DIR="${STACK_DIR}/models"
CONVERTED_DIR="${MODELS_DIR}/converted"
HF_CACHE_DIR="${MODELS_DIR}/hf-cache"
PATHS_FILE="${MODELS_DIR}/paths.json"
OUTPUTS_DIR="${STACK_DIR}/outputs"
EXAMPLES_DIR="${STACK_DIR}/examples"
VALIDATE_MODEL="${STACK_DIR}/validate_model.py"

ENGINE_REPO="${YUE2_MLX_REPO:-https://github.com/daig/yue2-mlx.git}"
ENGINE_REF="${YUE2_MLX_REF:-c0f0229df07daba14923627e9d78e084d82b7dc8}"
DEFAULT_PRECISION="${YUE2_PRECISION:-bf16}"
DEFAULT_ALIAS="${YUE2_ALIAS:-yue2-3b-mlx}"
DEFAULT_HOST="${YUE2_HOST:-127.0.0.1}"
DEFAULT_PORT="${YUE2_PORT:-8088}"

load_yue2_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: missing ${CONFIG_FILE}. Run ./1_setup_download.sh first." >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    PRECISION="${PRECISION:-${DEFAULT_PRECISION}}"
    MODEL_ALIAS="${MODEL_ALIAS:-${DEFAULT_ALIAS}}"
    MODEL_DIR="${MODEL_DIR:-${CONVERTED_DIR}}"
    if [[ -z "${VAE_PATH:-}" && -f "${PATHS_FILE}" ]]; then
        VAE_PATH="$(python3 -c "import json; print(json.load(open('${PATHS_FILE}'))['vae'])")"
    fi
    if [[ -z "${VAE_PATH:-}" ]]; then
        echo "ERROR: VAE path missing. Re-run ./1_setup_download.sh" >&2
        return 1
    fi
    return 0
}

prepare_lyra_env() {
    export MLX_ENABLE_TF32=0
    unset PYTORCH_ENABLE_MPS_FALLBACK || true
    unset PYTORCH_MPS_FAST_MATH || true
}

run_lyra() {
    prepare_lyra_env
    if [[ ! -d "${ENGINE_DIR}" ]]; then
        echo "ERROR: engine checkout missing at ${ENGINE_DIR}. Run ./1_setup_download.sh" >&2
        return 1
    fi
    (
        cd "${ENGINE_DIR}"
        uv run lyra "$@"
    )
}

run_engine_python() {
    prepare_lyra_env
    (
        cd "${ENGINE_DIR}"
        uv run python "$@"
    )
}

stop_server_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi
    echo "→ Stopping process(es) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti ":${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "→ Force-stopping stubborn process(es) ..."
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 1
    fi
}

port_pids() {
    lsof -ti ":${1}" 2>/dev/null || true
}
