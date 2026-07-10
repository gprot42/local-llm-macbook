#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download DiffusionGemma 26B A4B IT (MLX bf16) + deps
#
# Model:  mlx-community/diffusiongemma-26B-A4B-it-bf16  (~52 GB)
# Engine: mlx-vlm (discrete diffusion VLM)
#
# Usage:
#   ./1_setup_download.sh                    # download weights + install deps
#   ./1_setup_download.sh --skip-download    # only install Python deps
#   ./1_setup_download.sh --force            # re-download all weight shards
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/diffusiongemma-26b-a4b-it-bf16"
HF_REPO="mlx-community/diffusiongemma-26B-A4B-it-bf16"
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --force|--force-download) FORCE_DOWNLOAD=true ;;
        --help|-h)
            echo "Usage: ./1_setup_download.sh [options]"
            echo ""
            echo "  --skip-download   Install Python deps only"
            echo "  --force           Re-download all files (passes --force-download to hf)"
            echo ""
            echo "Model:     $HF_REPO"
            echo "Local dir: $MODEL_DIR"
            echo "Size:      ~52 GB bf16 — recommend 64 GB+ unified memory"
            exit 0
            ;;
    esac
done

echo "=== DiffusionGemma 26B A4B IT (MLX bf16) — Setup ==="
echo "→ Model repo: $HF_REPO"
echo "→ Local dir:  $MODEL_DIR"
echo ""

# ── Python / venv ─────────────────────────────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    echo "→ Creating virtualenv at venv/ ..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

echo "→ Installing / upgrading mlx mlx-vlm huggingface_hub ..."
pip install --quiet --no-cache-dir --upgrade pip
pip install --quiet --no-cache-dir --upgrade -r "$SCRIPT_DIR/requirements.txt"
echo "→ Dependencies installed."
echo ""

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"
chmod +x "$VALIDATE_MODEL" 2>/dev/null || true

model_is_complete() {
    [[ -d "$MODEL_DIR" ]] && python3 "$VALIDATE_MODEL" "$MODEL_DIR" >/dev/null 2>&1
}

list_missing_files() {
    python3 "$VALIDATE_MODEL" --list-missing "$MODEL_DIR" 2>/dev/null || true
}

download_missing() {
    local hf_flags=()
    export HF_HUB_ENABLE_HF_TRANSFER=1

    if [[ "$FORCE_DOWNLOAD" == true ]]; then
        hf_flags+=(--force-download)
        echo "→ Force download: fetching full repo (existing files may be replaced)"
        hf download "$HF_REPO" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
        return
    fi

    hf_flags+=(--no-force-download)

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "→ First download of $HF_REPO (~52 GB, may take a while) ..."
        hf download "$HF_REPO" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
        return
    fi

    MISSING=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && MISSING+=("$line")
    done < <(list_missing_files)
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        echo "→ All weight files already on disk — nothing to download"
        return
    fi

    echo "→ Fetching ${#MISSING[@]} missing file(s) (skipping files already present):"
    for f in "${MISSING[@]}"; do
        echo "     - $f"
    done
    hf download "$HF_REPO" "${MISSING[@]}" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
}

if [[ "$SKIP_DOWNLOAD" == true ]]; then
    echo "→ Skipping download (--skip-download passed)."
    if [[ -d "$MODEL_DIR" ]] && ! model_is_complete; then
        echo "→ WARNING: local model is incomplete — run without --skip-download"
        python3 "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    fi
elif model_is_complete && [[ "$FORCE_DOWNLOAD" != true ]]; then
    echo "→ Model already complete — skipping download (use --force to re-fetch)"
    python3 "$VALIDATE_MODEL" "$MODEL_DIR"
else
    if [[ -d "$MODEL_DIR" ]]; then
        echo "→ Checking which files are still needed ..."
        python3 "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    fi
    echo ""
    download_missing
    echo ""
    if ! python3 "$VALIDATE_MODEL" "$MODEL_DIR"; then
        echo ""
        echo "ERROR: model is still incomplete after download."
        exit 1
    fi
fi

chmod +x "$SCRIPT_DIR/2_run_mlx.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:  $MODEL_DIR"
echo "  Run:    ./2_run_mlx.sh                  # quick text test"
echo "          ./2_run_mlx.sh chat              # interactive chat"
echo "          ./2_run_mlx.sh --image photo.jpg # vision prompt"
echo "          ./2_run_mlx.sh server            # OpenAI-compatible API"
echo ""