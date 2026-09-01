#!/usr/bin/env bash
# =============================================================================
# 3_chat.sh — Terminal chat with Qwen3.5-122B-A10B abliterated (no HTTP server)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.qwen35_122b_config"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
else
    MODEL_DIR="$SCRIPT_DIR/qwen3.5-122b-a10b-abliterated-mlx-4bit"
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model not found: $MODEL_DIR"
    echo "       Run ./1_setup_download.sh first."
    exit 1
fi

# shellcheck source=/dev/null
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

if "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
    exec "$VENV_PY" -m mlx_lm chat --model "$MODEL_DIR" "$@"
fi

if [[ -x "$SCRIPT_DIR/venv/bin/mlx_lm.chat" ]]; then
    exec "$SCRIPT_DIR/venv/bin/mlx_lm.chat" --model "$MODEL_DIR" "$@"
fi

echo "ERROR: mlx_lm.chat not available. Run ./1_setup_download.sh --skip-download"
exit 1
