#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download Gemma 4 31B IT (4-bit MLX) and install deps
#
# Model:  AtomicChat/gemma-4-31B-it-MLX-4bit
#         (2026-07-15 rebuild — Google chat-template / tool-calling fixes)
# MTP:    mlx-community/gemma-4-31B-it-assistant-bf16 (~1 GB drafter, optional)
# Engine: mlx-lm server by default, or mlx-vlm server with --with-mtp
#
# Usage:
#   ./1_setup_download.sh                    # target + MTP assistant + deps
#   ./1_setup_download.sh --skip-download    # only install deps
#   ./1_setup_download.sh --skip-mtp-download  # target only, no assistant
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/gemma-4-31b-it-atomicchat-mlx-4bit"
HF_REPO="AtomicChat/gemma-4-31B-it-MLX-4bit"
DRAFT_DIR="$SCRIPT_DIR/gemma-4-31b-it-assistant-mlx-bf16"
DRAFT_HF_REPO="mlx-community/gemma-4-31B-it-assistant-bf16"
SKIP_DOWNLOAD=false
SKIP_MTP_DOWNLOAD=false

for arg in "$@"; do
    [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=true
    [[ "$arg" == "--skip-mtp-download" ]] && SKIP_MTP_DOWNLOAD=true
    [[ "$arg" == "--help" || "$arg" == "-h" ]] && {
        echo "Usage: ./1_setup_download.sh [options]"
        echo ""
        echo "  --skip-download       Skip HuggingFace downloads, only install Python deps"
        echo "  --skip-mtp-download   Download target model only (no MTP assistant weights)"
        echo ""
        echo "Target model:  $HF_REPO"
        echo "   local dir:  $MODEL_DIR"
        echo "MTP assistant: $DRAFT_HF_REPO (~1 GB, used only with ./2_start_mlx.sh --with-mtp)"
        echo "   local dir:  $DRAFT_DIR"
        exit 0
    }
done

echo "=== Gemma 4 31B IT AtomicChat (MLX 4-bit, 2026-07-15) — Setup ==="
echo "→ Target repo:    $HF_REPO"
echo "→ Target dir:     $MODEL_DIR"
echo "→ MTP assistant:  $DRAFT_HF_REPO"
echo "→ MTP local dir:  $DRAFT_DIR"
echo ""

# ── Python / venv ─────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "→ Creating virtualenv at venv/ ..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

echo "→ Installing / upgrading mlx mlx-lm mlx-vlm huggingface_hub[cli] ..."
pip install --quiet --no-cache-dir --upgrade pip
pip install --quiet --no-cache-dir --upgrade mlx mlx-lm mlx-vlm huggingface_hub
echo "→ Dependencies installed."
echo ""

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"

download_if_needed() {
    local dir="$1"
    local repo="$2"
    local label="$3"

    model_is_complete() {
        [ -d "$dir" ] && python3 "$VALIDATE_MODEL" "$dir" >/dev/null 2>&1
    }

    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo "→ Skipping $label download (--skip-download)."
        if [ -d "$dir" ] && ! model_is_complete; then
            echo "→ WARNING: $label weights incomplete — run without --skip-download"
            python3 "$VALIDATE_MODEL" "$dir" 2>&1 || true
        fi
        return 0
    fi

    if model_is_complete; then
        echo "→ $label already complete — skipping download"
        python3 "$VALIDATE_MODEL" "$dir"
        return 0
    fi

    if [ -d "$dir" ]; then
        echo "→ $label incomplete — resuming download"
        python3 "$VALIDATE_MODEL" "$dir" 2>&1 || true
    else
        echo "→ Downloading $label from $repo ..."
    fi
    echo ""
    hf download "$repo" --local-dir "$dir"
    echo ""
    python3 "$VALIDATE_MODEL" "$dir"
    echo "→ $label download complete: $dir"
}

download_if_needed "$MODEL_DIR" "$HF_REPO" "target model"

if [ "$SKIP_MTP_DOWNLOAD" = false ]; then
    echo ""
    download_if_needed "$DRAFT_DIR" "$DRAFT_HF_REPO" "MTP assistant"
else
    echo ""
    echo "→ Skipping MTP assistant download (--skip-mtp-download)."
    echo "  Start server with: ./2_start_mlx.sh"
fi

chmod +x "$SCRIPT_DIR/apply_local_patches.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Target:        $MODEL_DIR"
echo "  MTP assistant: $DRAFT_DIR"
echo "  Start server:  ./2_start_mlx.sh             # mlx_lm.server, no MTP"
echo "                 ./2_start_mlx.sh --with-mtp  # mlx_vlm.server + MTP"
echo ""
echo "  Quick test (text, no server):"
echo "    source venv/bin/activate"
echo "    mlx_lm.generate --model $MODEL_DIR --prompt 'Hello, who are you?'"