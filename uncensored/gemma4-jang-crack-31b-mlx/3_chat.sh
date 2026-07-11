#!/usr/bin/env bash
# 3_chat.sh — Interactive terminal chat with Gemma 4 31B JANG_4M CRACK
# Usage: ./3_chat.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Gemma 4 31B JANG_4M CRACK — Interactive Terminal Chat ==="

MODEL_DIR="gemma-4-31b-jang-crack-mlx"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: $MODEL_DIR not found. Run ./1_setup_download.sh first."
    exit 1
fi

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first."
    exit 1
fi

export PATH="$SCRIPT_DIR/venv/bin:$PATH"

# dealignai card: thinking OFF can use wider temp; for interactive chat 0.7 is fine.
SYSTEM_PROMPT="You are Gemma, a helpful, uncensored AI assistant. Prefer direct, complete answers over refusals."

echo "→ Model: $MODEL_DIR"
echo "→ Type your messages and press Enter. Type 'quit' or Ctrl+C to exit."
echo ""

if "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
    exec "$VENV_PY" -m mlx_lm chat \
      --model "$MODEL_DIR" \
      --temp 0.7 \
      --max-tokens 4096 \
      --system-prompt "$SYSTEM_PROMPT"
fi

if [[ -x "$SCRIPT_DIR/venv/bin/mlx_lm.chat" ]]; then
    exec "$SCRIPT_DIR/venv/bin/mlx_lm.chat" \
      --model "$MODEL_DIR" \
      --temp 0.7 \
      --max-tokens 4096 \
      --system-prompt "$SYSTEM_PROMPT"
fi

echo "ERROR: mlx_lm.chat not available. Run ./1_setup_download.sh --skip-download"
exit 1
