#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download GLM-4.7-Flash Heretic (uncensored) GGUF
#
# HF repo: DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF
# Tuned for M5 MacBook with 128 GB unified memory.
#
# Quant options (first arg):
#   q8   Q8_0  (~32.1 GB) — default, max quality
#   q6   Q6_K  (~25.0 GB) — near-lossless + room for long context
#   q5   Q5_K_M (~21.6 GB)
#   q4   Q4_K_M (~18.5 GB)
#
# Downloads are resumable — re-run the same command after an interrupt or failure.
#
# Usage:
#   ./1_setup_download.sh              # Q8_0 (default)
#   ./1_setup_download.sh q6           # Q6_K smaller/faster
#   ./1_setup_download.sh --skip-download
#   ./1_setup_download.sh --force        # Re-download from scratch
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

# DavidAU naming prefix for this quant pack
PREFIX="GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU"

QUANT="q8"
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false

for arg in "$@"; do
    case "$arg" in
        q4|q5|q6|q8) QUANT="$arg" ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --force) FORCE_DOWNLOAD=true ;;
        --help|-h)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

case "${QUANT}" in
    q4) GGUF_FILE="${PREFIX}-Q4_K_M.gguf"; GGUF_DESC="~18.5 GB Q4_K_M" ;;
    q5) GGUF_FILE="${PREFIX}-Q5_K_M.gguf"; GGUF_DESC="~21.6 GB Q5_K_M" ;;
    q6) GGUF_FILE="${PREFIX}-Q6_K.gguf";   GGUF_DESC="~25.0 GB Q6_K (near-lossless)" ;;
    q8) GGUF_FILE="${PREFIX}-Q8_0.gguf";   GGUF_DESC="~32.1 GB Q8_0 (default, max quality)" ;;
esac

MODEL_PATH="$WEIGHTS_DIR/$GGUF_FILE"
MODEL_ID="glm-4.7-flash-heretic-${QUANT}"

echo "=== GLM-4.7-Flash Heretic (uncensored) — Setup (Ollama / M5 128 GB) ==="
echo "→ Repo:   $HF_REPO"
echo "→ Quant:  $GGUF_DESC"
echo "→ File:   $GGUF_FILE"
echo ""

# ── Ollama ────────────────────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    echo "→ ollama not found — install before serving:"
    echo "    brew install ollama"
    echo ""
else
    echo "→ ollama: $(command -v ollama)"
    echo ""
fi

# ── Python venv (hf download only) ────────────────────────────────────────────
_venv_ok=false
if [ -x "$SCRIPT_DIR/venv/bin/python3" ] && [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    if "$SCRIPT_DIR/venv/bin/python3" -c "import sys" 2>/dev/null; then
        if head -1 "$SCRIPT_DIR/venv/bin/pip" 2>/dev/null | grep -q "$SCRIPT_DIR/venv"; then
            _venv_ok=true
        fi
    fi
fi
if [ "$_venv_ok" != true ]; then
    echo "→ Creating virtualenv at venv/ ..."
    rm -rf "$SCRIPT_DIR/venv"
    python3 -m venv "$SCRIPT_DIR/venv"
fi
VENV_PY="$SCRIPT_DIR/venv/bin/python3"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
"$VENV_PIP" install --quiet --no-cache-dir --upgrade pip huggingface_hub httpx
echo "→ huggingface_hub installed."
echo ""

VALIDATE="$SCRIPT_DIR/validate_model.py"
DOWNLOAD="$SCRIPT_DIR/download_gguf.py"
mkdir -p "$WEIGHTS_DIR"

download_gguf() {
    local dest="$1"
    local remote="$2"
    local label="$3"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo "→ Skipping $label (--skip-download)."
        return 0
    fi

    echo "→ $label"
    local -a dl_args=(
        "$VENV_PY" "$DOWNLOAD"
        --repo "$HF_REPO"
        --remote "$remote"
        --dest "$dest"
        --validate-script "$VALIDATE"
    )
    if [ "$FORCE_DOWNLOAD" = true ]; then
        dl_args+=(--force)
    fi
    "${dl_args[@]}"
    echo "→ $label complete: $dest"
}

download_gguf "$MODEL_PATH" "$GGUF_FILE" "main model ($GGUF_FILE)"

cat > "$SCRIPT_DIR/.glm47_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_REPO="${HF_REPO}"
QUANT="${QUANT}"
GGUF_FILE="${GGUF_FILE}"
MODEL_PATH="${MODEL_PATH}"
MODEL_ID="${MODEL_ID}"
EOF

chmod +x "$SCRIPT_DIR/validate_model.py" "$SCRIPT_DIR/download_gguf.py" \
    "$SCRIPT_DIR/2_start_ollama.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        $MODEL_PATH"
echo "  Model ID:     $MODEL_ID"
echo "  Start Ollama: ./2_start_ollama.sh"
echo ""
echo "  Quick test:"
echo "    ./2_start_ollama.sh"
echo "    curl http://127.0.0.1:18083/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words.\"}],\"max_tokens\":32}'"
