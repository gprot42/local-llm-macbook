#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install mtplx and download Qwen3.8-27B OBLITERATED V3
#
# Weights: https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED
#
# V3 (2026-08-23): Hub no longer ships mlx-4bit/ / mlx-8bit/ (those were V1/V2;
# commit "Remove stale MLX files"). GGUFs were re-uploaded after a broken
# conversion (Pliny: bf16 was always fine; GGUFs were not). This stack does
# not use GGUF. It snapshots V3 bf16 (~56 GB, 28-shard + extra/MTP) and
# converts locally with mlx_lm to ./models/mlx-{4,8}bit/.
#
# NEVER `mtplx pull` the repo id — that kitchen-sink is GGUF + leftover
# 18-shard files + bf16 (~270 GB+).
#
# Model selection (first non-flag arg):
#   4bit | 27b       default — local MLX 4-bit (~14 GB after convert)
#   8bit             local MLX 8-bit (~27 GB after convert)
#   --deps-only      install venv + mtplx only (no pull)
#   --force          re-snapshot V3 bf16 and reconvert even if mlx looks complete
#
# Env overrides:
#   QWEN38_OBL_HF_REPO   Hub repo (default OBLITERATUS/Qwen3.8-27B-OBLITERATED)
#   QWEN38_OBL_ALIAS     OpenAI model id (default qwen3.8-27b-obliterated-mtplx)
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh 4bit
#   ./1_setup_download.sh 8bit
#   ./1_setup_download.sh --force
#   ./1_setup_download.sh --deps-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="4bit"
DEPS_ONLY=false
FORCE=false

for arg in "$@"; do
    case "${arg}" in
        --help|-h)
            sed -n '3,32p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --deps-only|deps-only)
            DEPS_ONLY=true
            ;;
        --force|--refresh)
            FORCE=true
            ;;
        4bit|27b|8bit|auto|mlx)
            MODEL_CHOICE="${arg}"
            ;;
        *)
            echo "ERROR: Unknown argument '${arg}'."
            echo "  Use: 4bit | 27b | 8bit | --deps-only | --force"
            exit 1
            ;;
    esac
done

HF_REPO="${QWEN38_OBL_HF_REPO:-OBLITERATUS/Qwen3.8-27B-OBLITERATED}"
MODELS_ROOT="${SCRIPT_DIR}/models"
BF16_DIR="${MODELS_ROOT}/bf16-v3"
WEIGHT_VERSION="V3"

choice_to_subdir() {
    case "$1" in
        27b|4bit|auto|mlx) echo "mlx-4bit" ;;
        8bit)              echo "mlx-8bit" ;;
        *)
            echo "ERROR: Unknown model choice '${1}'." >&2
            echo "  Use: 4bit | 27b | 8bit | --deps-only | --force" >&2
            exit 1
            ;;
    esac
}

choice_to_qbits() {
    case "$1" in
        8bit) echo 8 ;;
        *)    echo 4 ;;
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

mlx_has_mtp() {
    local dir="$1"
    python3 - "${dir}" <<'PY'
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1]) / "config.json"
try:
    data = json.loads(cfg.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(1)
text = data.get("text_config") if isinstance(data.get("text_config"), dict) else data
n = text.get("mtp_num_hidden_layers") if isinstance(text, dict) else None
sys.exit(0 if isinstance(n, int) and n >= 1 else 1)
PY
}

hub_has_mlx_subdir() {
    local repo="$1"
    local subdir="$2"
    python3 - "${repo}" "${subdir}" <<'PY'
import sys
from huggingface_hub import HfApi

repo, subdir = sys.argv[1], sys.argv[2].rstrip("/")
prefix = subdir + "/"
api = HfApi()
try:
    for item in api.list_repo_tree(repo, repo_type="model", recursive=True):
        path = getattr(item, "path", "") or ""
        if path.startswith(prefix) and path.endswith(".safetensors"):
            sys.exit(0)
except Exception:
    sys.exit(1)
sys.exit(1)
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

pull_v3_bf16() {
    local repo="$1"
    local dest="$2"
    python3 - "${repo}" "${dest}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
Path(dest).mkdir(parents=True, exist_ok=True)
# V3 live weights are the 28-shard bf16 pack + extra (MTP + vision).
# Ignore GGUF (re-uploaded 2026-08-23 after a broken conversion), leftover
# 18-shard files, and mmproj. Do not pull the kitchen-sink repo as a whole.
snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-*-of-00028.safetensors",
        "model-extra-*.safetensors",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "abliteration_metadata.json",
    ],
    ignore_patterns=[
        "*.gguf",
        "mmproj*",
        "model-*-of-00018.safetensors",
        "mlx-4bit/*",
        "mlx-8bit/*",
    ],
    local_dir=dest,
)
PY
}

echo "=== Qwen3.8-27B OBLITERATED ${WEIGHT_VERSION} + mtplx Setup ==="
if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ Mode:    deps-only (skip model pull)"
else
    echo "→ Choice:  ${MODEL_CHOICE}"
    echo "→ Repo:    ${HF_REPO}"
    echo "→ Version: ${WEIGHT_VERSION} (bf16 snapshot → local MLX convert)"
    if [[ "${FORCE}" == true ]]; then
        echo "→ Force:   re-snapshot + reconvert"
    fi
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
    echo "  Re-run without --deps-only to snapshot V3 bf16 and convert mlx-4bit/."
    echo "  Aligned sibling: ../../censored/qwen3-8-27b-coder-mtplx/"
    exit 0
fi

MLX_SUBDIR="$(choice_to_subdir "${MODEL_CHOICE}")"
Q_BITS="$(choice_to_qbits "${MODEL_CHOICE}")"
HF_MODEL="${MODELS_ROOT}/${MLX_SUBDIR}"
MODEL_ALIAS="${QWEN38_OBL_ALIAS:-qwen3.8-27b-obliterated-mtplx}"
MODEL_SIZE="27b"
PREV_VERSION=""
if [[ -f "${SCRIPT_DIR}/.mtplx_config" ]]; then
    PREV_VERSION="$(grep -E '^WEIGHT_VERSION=' "${SCRIPT_DIR}/.mtplx_config" 2>/dev/null \
        | tail -1 | cut -d= -f2- | tr -d '"' || true)"
fi

echo ""
echo "→ MLX dest:  ${HF_MODEL}"
echo "→ Alias:     ${MODEL_ALIAS}"
echo "→ Quant:     ${Q_BITS}-bit (group 64, affine)"
echo ""
echo "  Will NOT run: mtplx pull ${HF_REPO}"
echo "  (GGUF + leftover 18-shard + bf16 — hundreds of GB; GGUF V3 was re-uploaded"
echo "   after a broken conversion. This stack converts V3 bf16 locally.)"
echo ""

NEED_BUILD=false
if [[ "${FORCE}" == true ]]; then
    NEED_BUILD=true
    echo "→ --force: ignoring existing ${HF_MODEL}"
elif ! subdir_complete "${HF_MODEL}"; then
    NEED_BUILD=true
elif [[ "${PREV_VERSION}" != "${WEIGHT_VERSION}" ]]; then
    NEED_BUILD=true
    echo "→ Local ${MLX_SUBDIR} is complete but not ${WEIGHT_VERSION}"
    echo "  (Hub dropped V1/V2 mlx folders; Aug-20 snapshot is stale.)"
    echo "  Rebuilding from V3 bf16. Ctrl+C to keep the old quant."
else
    echo "→ Model weights already complete at ${HF_MODEL} (${WEIGHT_VERSION})"
fi

if [[ "${NEED_BUILD}" == true ]]; then
    if hub_has_mlx_subdir "${HF_REPO}" "${MLX_SUBDIR}"; then
        echo "→ Hub still has ${MLX_SUBDIR}/ — snapshotting that (no local convert) ..."
        echo ""
        if ! pull_mlx_subfolder "${HF_REPO}" "${MLX_SUBDIR}" "${MODELS_ROOT}"; then
            echo ""
            echo "ERROR: snapshot of ${HF_REPO} ${MLX_SUBDIR}/ failed."
            echo "  Check network / disk, then re-run this script."
            echo "  Do not run: mtplx pull ${HF_REPO}"
            exit 1
        fi
    else
        echo "→ Hub has no ${MLX_SUBDIR}/ (removed with V1/V2 MLX)."
        echo "→ Snapshotting V3 bf16 into ${BF16_DIR} (~56 GB, resumable) ..."
        echo ""
        if ! pull_v3_bf16 "${HF_REPO}" "${BF16_DIR}"; then
            echo ""
            echo "ERROR: snapshot of ${HF_REPO} V3 bf16 failed."
            echo "  Check network / disk, then re-run this script."
            echo "  Do not run: mtplx pull ${HF_REPO}"
            exit 1
        fi
        if ! subdir_complete "${BF16_DIR}"; then
            echo "ERROR: ${BF16_DIR} is still incomplete after download."
            echo "       Check disk space and re-run ./1_setup_download.sh"
            exit 1
        fi
        echo "→ V3 bf16 complete."

        if [[ -d "${HF_MODEL}" ]]; then
            STALE_BAK="${HF_MODEL}.pre-${WEIGHT_VERSION}.bak"
            if [[ -d "${STALE_BAK}" ]]; then
                echo "→ Removing previous backup ${STALE_BAK}"
                rm -rf "${STALE_BAK}"
            fi
            echo "→ Moving stale ${HF_MODEL} → ${STALE_BAK}"
            mv "${HF_MODEL}" "${STALE_BAK}"
        fi

        echo "→ Converting V3 bf16 → MLX ${Q_BITS}-bit (slow; uses unified memory) ..."
        echo ""
        if ! python -m mlx_lm convert \
            --hf-path "${BF16_DIR}" \
            --mlx-path "${HF_MODEL}" \
            -q --q-bits "${Q_BITS}" --q-group-size 64; then
            echo ""
            echo "ERROR: mlx_lm convert failed."
            echo "  bf16 is still at ${BF16_DIR}"
            if [[ -n "${STALE_BAK:-}" && -d "${STALE_BAK}" ]]; then
                echo "  Previous quant: ${STALE_BAK}"
            fi
            exit 1
        fi
        echo "→ Convert complete. You can delete ${BF16_DIR} later to free ~56 GB."
    fi

    if ! subdir_complete "${HF_MODEL}"; then
        echo "ERROR: ${HF_MODEL} is still incomplete after download/convert."
        echo "       Check disk space and re-run ./1_setup_download.sh --force"
        exit 1
    fi
    if ! mlx_has_mtp "${HF_MODEL}"; then
        echo "ERROR: converted ${HF_MODEL} has no MTP heads (mtp_num_hidden_layers)."
        echo "       V3 extra shard (model-extra-*.safetensors) must be in the bf16 snapshot."
        exit 1
    fi
    echo "→ Model download/convert complete (MTP heads present)."
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
WEIGHT_VERSION="${WEIGHT_VERSION}"
BF16_DIR="${BF16_DIR}"
EOF

echo ""
echo "✅  Setup complete (${WEIGHT_VERSION})!"
echo ""
echo "  Model:        ${HF_MODEL}"
echo "  Config:       ${SCRIPT_DIR}/.mtplx_config"
echo ""
echo "  Start server: ./2_start_mtplx.sh"
echo "  Harness gate: python3 test_harness.py --gate   # after start"
echo "  Rebuild V3:   ./1_setup_download.sh --force"
