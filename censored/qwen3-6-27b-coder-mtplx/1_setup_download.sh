#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install mtplx and download Qwen3.6 model weights
#
# mtplx is a native-MTP speculative decoding engine for Apple Silicon.
# It runs the model's own built-in MTP heads as a speculative drafter,
# delivering ~2.24× tok/s at temp=0.6 with no extra RAM cost.
#
# Model options (pass as first arg):
#   27b     Qwen3.6-27B  (dense, ~18 GB,  PROVEN: 28 → 63 tok/s on M5 Max)  ← default
#   35b     Qwen3.6-35B-A3B MoE (3B active, ~22 GB, best coding benchmarks)
#
# Usage:
#   ./1_setup_download.sh        # 27B (default)
#   ./1_setup_download.sh 35b    # 35B MoE
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SIZE="${1:-27b}"

# ── Model selection ───────────────────────────────────────────────────────────
case "${MODEL_SIZE}" in
    27b)
        # Optimized mtplx checkpoint — MTP weights kept in BF16 for max acceptance
        HF_MODEL="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
        MODEL_ALIAS="qwen3.6-27b-mtplx"
        MODEL_SIZE_DESC="~18 GB"
        ;;
    35b)
        # Standard mlx-community 4-bit — MTP heads present; acceptance ~79-85%
        # NOTE: mtplx will use MTP heads automatically if found in the checkpoint
        HF_MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
        MODEL_ALIAS="qwen3.6-35b-a3b"
        MODEL_SIZE_DESC="~22 GB"
        ;;
    *)
        echo "ERROR: Unknown model size '${MODEL_SIZE}'. Use 27b or 35b."
        exit 1
        ;;
esac

echo "=== Qwen3.6 + mtplx Setup (${MODEL_SIZE^^}) ==="
echo "→ Model:   ${HF_MODEL}"
echo "→ Alias:   ${MODEL_ALIAS}"
echo "→ Size:    ${MODEL_SIZE_DESC}"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3 || true)
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: Python 3.10+ required. Install via: brew install python@3.11"
    exit 1
fi

PY_VER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "${PY_VER}" | cut -d. -f1)
PY_MINOR=$(echo "${PY_VER}" | cut -d. -f2)
if [[ "${PY_MAJOR}" -lt 3 ]] || [[ "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 10 ]]; then
    echo "ERROR: Python 3.10+ required (found ${PY_VER}). Install via: brew install python@3.11"
    exit 1
fi
echo "→ Python: ${PY_VER} (${PYTHON})"

# ── Create / reuse virtualenv ─────────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/venv"
if [[ -d "${VENV_DIR}" ]]; then
    echo "→ Using existing venv: ${VENV_DIR}"
else
    echo "→ Creating venv: ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "→ venv active: $(python --version)"

# ── Install / upgrade mtplx ───────────────────────────────────────────────────
echo ""
echo "→ Installing / upgrading mtplx ..."
pip install --upgrade pip --quiet
pip install --upgrade mtplx

echo ""
echo "→ mtplx version: $(mtplx --version 2>/dev/null || python -c 'import mtplx; print(mtplx.__version__)' 2>/dev/null || echo 'installed')"

# ── Download model ────────────────────────────────────────────────────────────
echo ""
echo "→ Downloading model: ${HF_MODEL} (${MODEL_SIZE_DESC}) ..."
echo "  This is downloaded once and cached by mtplx (~/.cache/huggingface)"
echo ""

mtplx pull "${HF_MODEL}"
echo "→ Model download complete."

# ── Write config ──────────────────────────────────────────────────────────────
cat > "${SCRIPT_DIR}/.mtplx_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_MODEL="${HF_MODEL}"
MODEL_ALIAS="${MODEL_ALIAS}"
MODEL_SIZE="${MODEL_SIZE}"
EOF

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        ${HF_MODEL}"
echo "  Config:       ${SCRIPT_DIR}/.mtplx_config"
echo ""
echo "  Start server: ./2_start_mtplx.sh"
