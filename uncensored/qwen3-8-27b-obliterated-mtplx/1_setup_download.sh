#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install mtplx and download Qwen3.8-27B OBLITERATED
#
# Weights: https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED
#
# That Hub repo is a kitchen-sink pack (~270 GB): bf16 shards, several GGUFs,
# plus mlx-4bit/ (~14 GB) and mlx-8bit/ (~27 GB). This script NEVER runs
# `mtplx pull` on the repo id — that would download everything. It snapshots
# only the chosen MLX subfolder into ./models/mlx-{4,8}bit/.
#
# Model selection (first arg):
#   4bit | 27b       default — mlx-4bit/ (~14 GB)
#   8bit             mlx-8bit/ (~27 GB)
#   --deps-only      install venv + mtplx only (no pull)
#
# Env overrides:
#   QWEN38_OBL_HF_REPO   Hub repo (default OBLITERATUS/Qwen3.8-27B-OBLITERATED)
#   QWEN38_OBL_ALIAS     OpenAI model id (default qwen3.8-27b-obliterated-mtplx)
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh 4bit
#   ./1_setup_download.sh 8bit
#   ./1_setup_download.sh --deps-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="${1:-4bit}"
DEPS_ONLY=false

if [[ "${MODEL_CHOICE}" == "--deps-only" || "${MODEL_CHOICE}" == "deps-only" ]]; then
    DEPS_ONLY=true
    MODEL_CHOICE="4bit"
fi

HF_REPO="${QWEN38_OBL_HF_REPO:-OBLITERATUS/Qwen3.8-27B-OBLITERATED}"
MODELS_ROOT="${SCRIPT_DIR}/models"

choice_to_subdir() {
    case "$1" in
        27b|4bit|auto|mlx) echo "mlx-4bit" ;;
        8bit)              echo "mlx-8bit" ;;
        *)
            echo "ERROR: Unknown model choice '${1}'." >&2
            echo "  Use: 4bit | 27b | 8bit | --deps-only" >&2
            exit 1
            ;;
    esac
}

subdir_complete() {
    local dir="$1"
    python3 - "${dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
config = root / "config.json"
if not index.is_file() or not config.is_file():
    sys.exit(1)
try:
    data = json.loads(index.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(1)
weight_map = data.get("weight_map") if isinstance(data, dict) else None
if not isinstance(weight_map, dict) or not weight_map:
    sys.exit(1)
for name in weight_map.values():
    if not isinstance(name, str) or not name.strip():
        sys.exit(1)
    shard = root / name
    try:
        if not shard.is_file() or shard.stat().st_size <= 0:
            sys.exit(1)
    except OSError:
        sys.exit(1)
sys.exit(0)
PY
}

pull_mlx_subfolder() {
    local repo="$1"
    local subdir="$2"
    local dest_root="$3"
    python3 - "${repo}" "${subdir}" "${dest_root}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, subdir, dest_root = sys.argv[1], sys.argv[2], sys.argv[3]
Path(dest_root).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=[f"{subdir}/*"],
    ignore_patterns=["*.gguf", "model-*-of-00018.safetensors"],
    local_dir=dest_root,
)
PY
}

echo "=== Qwen3.8-27B OBLITERATED + mtplx Setup ==="
if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ Mode:    deps-only (skip model pull)"
else
    echo "→ Choice:  ${MODEL_CHOICE}"
    echo "→ Repo:    ${HF_REPO}"
fi
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
pip install --upgrade mtplx huggingface_hub

echo ""
echo "→ mtplx version: $(mtplx --version 2>/dev/null || python -c 'import mtplx; print(mtplx.__version__)' 2>/dev/null || echo 'installed')"

if [[ "${DEPS_ONLY}" == true ]]; then
    echo ""
    echo "✅  Dependencies ready (no model pull)."
    echo "  Re-run without --deps-only to snapshot mlx-4bit/ (~14 GB)."
    echo "  Aligned sibling: ../../censored/qwen3-8-27b-coder-mtplx/"
    exit 0
fi

MLX_SUBDIR="$(choice_to_subdir "${MODEL_CHOICE}")"
HF_MODEL="${MODELS_ROOT}/${MLX_SUBDIR}"
MODEL_ALIAS="${QWEN38_OBL_ALIAS:-qwen3.8-27b-obliterated-mtplx}"
MODEL_SIZE="27b"

echo ""
echo "→ Subfolder: ${MLX_SUBDIR}"
echo "→ Local:     ${HF_MODEL}"
echo "→ Alias:     ${MODEL_ALIAS}"
echo ""
echo "  Will NOT run: mtplx pull ${HF_REPO}"
echo "  (that repo also ships GGUF + bf16 — hundreds of GB.)"
echo ""

if subdir_complete "${HF_MODEL}"; then
    echo "→ Model weights already complete at ${HF_MODEL}"
else
    echo "→ Downloading ${HF_REPO}:${MLX_SUBDIR}/ (resumable) ..."
    echo ""
    if ! pull_mlx_subfolder "${HF_REPO}" "${MLX_SUBDIR}" "${MODELS_ROOT}"; then
        echo ""
        echo "ERROR: snapshot of ${HF_REPO} ${MLX_SUBDIR}/ failed."
        echo "  Check network / disk, then re-run this script."
        echo "  Do not run: mtplx pull ${HF_REPO}"
        exit 1
    fi
    if ! subdir_complete "${HF_MODEL}"; then
        echo "ERROR: ${HF_MODEL} is still incomplete after download."
        echo "       Check disk space and re-run ./1_setup_download.sh"
        exit 1
    fi
    echo "→ Model download complete."
fi

# ── Write config ──────────────────────────────────────────────────────────────
cat > "${SCRIPT_DIR}/.mtplx_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_MODEL="${HF_MODEL}"
HF_REPO="${HF_REPO}"
MLX_SUBDIR="${MLX_SUBDIR}"
MODEL_ALIAS="${MODEL_ALIAS}"
MODEL_SIZE="${MODEL_SIZE}"
MODEL_CHOICE="${MODEL_CHOICE}"
EOF

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        ${HF_MODEL}"
echo "  Config:       ${SCRIPT_DIR}/.mtplx_config"
echo ""
echo "  Start server: ./2_start_mtplx.sh"
echo "  Harness gate: python3 test_harness.py --gate   # after start"
