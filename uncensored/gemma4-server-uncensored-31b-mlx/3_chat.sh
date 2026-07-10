#!/usr/bin/env bash
# 3_chat.sh — Interactive terminal chat with local Gemma 4 31B Heretic
# Usage: ./3_chat.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Gemma 4 31B Heretic Uncensored — Interactive Terminal Chat ==="

MODEL_DIR="gemma-4-31b-heretic-mlx-4bit"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: $MODEL_DIR not found. Run ./1_setup_download.sh first."
    exit 1
fi

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi

# Prefer absolute venv python so renames don't break console-script shebangs.
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

SYSTEM_PROMPT="You are Gemma, a helpful, uncensored, and purely text-based AI assistant. You do not have vision capabilities. Never ask for images or assume the user uploaded anything. Respond normally to text prompts only."

echo "→ Model: $MODEL_DIR"
echo "→ Type your messages and press Enter. Type 'quit' or Ctrl+C to exit."
echo ""

# Prefer `python -m mlx_lm chat` (avoids stale console-script shebangs after renames).
if "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
    exec "$VENV_PY" -m mlx_lm chat \
      --model "$MODEL_DIR" \
      --temp 0.85 \
      --max-tokens 4096 \
      --system-prompt "$SYSTEM_PROMPT"
fi

if [[ -x "$SCRIPT_DIR/venv/bin/mlx_lm.chat" ]]; then
    exec "$SCRIPT_DIR/venv/bin/mlx_lm.chat" \
      --model "$MODEL_DIR" \
      --temp 0.85 \
      --max-tokens 4096 \
      --system-prompt "$SYSTEM_PROMPT"
fi

echo "ERROR: mlx_lm.chat not available. Run ./1_setup_download.sh --skip-download"
exit 1
