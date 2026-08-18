#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install mlx-vlm and download Muse Glimmer 30B
#
# Announced 2026-08-10 by Alexandr Wang / Meta Superintelligence Labs:
#   https://x.com/alexandr_wang/status/2086756152034066792
#   https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model
#
# Default: mlx-community 4-bit (~19.4 GB) + official DFlash assistant (~5 GB).
# Engine is mlx-vlm (multimodal + native --draft-kind dflash). mlx_lm.server
# cannot load this architecture.
#
# Model selection (first arg):
#   4bit | 30b | auto     default — mlx-community/Muse-Glimmer-30B-4bit
#   5bit | 6bit | 8bit    higher-quality mlx-community quants
#   mxfp4                 mlx-community/Muse-Glimmer-30B-mxfp4
#   official | bf16       mlx-community/Muse-Glimmer-30B-bf16 (large)
#   <org/repo>            raw Hugging Face repo id
#   --deps-only           venv + mlx-vlm only (no pull)
#   --skip-dflash-download  skip DFlash assistant (~5 GB)
#
# Env:
#   GLIMMER_HF_MODEL   force target HF repo
#   GLIMMER_ALIAS      OpenAI model id (default muse-glimmer-30b-mlx)
#   GLIMMER_DRAFT_HF   DFlash drafter (default meta-models/Muse-Glimmer-30B-assistant)
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh 4bit
#   ./1_setup_download.sh --deps-only
#   GLIMMER_HF_MODEL=mlx-community/Muse-Glimmer-30B-8bit ./1_setup_download.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="4bit"
DEPS_ONLY=false
SKIP_DFLASH=false

for arg in "$@"; do
    case "${arg}" in
        --help|-h)
            sed -n '3,32p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --deps-only|deps-only) DEPS_ONLY=true ;;
        --skip-dflash-download|--skip-dflash) SKIP_DFLASH=true ;;
        --*)
            echo "ERROR: unknown option '${arg}'" >&2
            exit 1
            ;;
        *) MODEL_CHOICE="${arg}" ;;
    esac
done

DEFAULT_4BIT="mlx-community/Muse-Glimmer-30B-4bit"
DRAFT_HF="${GLIMMER_DRAFT_HF:-meta-models/Muse-Glimmer-30B-assistant}"
DRAFT_DIR="${SCRIPT_DIR}/muse-glimmer-30b-assistant"

# ── HF helpers (run under venv python after activate) ─────────────────────────
hf_in_cache() {
    local repo="$1"
    python3 - "${repo}" <<'PY' 2>/dev/null
import sys
from pathlib import Path
try:
    from huggingface_hub import try_to_load_from_cache
except Exception:
    sys.exit(1)
repo = sys.argv[1]
for name in ("config.json", "model.safetensors.index.json", "model.safetensors", "tokenizer.json"):
    p = try_to_load_from_cache(repo, name)
    if p and Path(p).is_file():
        sys.exit(0)
sys.exit(1)
PY
}

hf_repo_has_weights() {
    local repo="$1"
    python3 - "${repo}" <<'PY' 2>/dev/null
import re
import sys
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError

repo = sys.argv[1]
weight_re = re.compile(r"(\.safetensors$|\.npz$|\.bin$|\.gguf$)", re.I)
try:
    info = model_info(repo)
except RepositoryNotFoundError:
    sys.exit(1)
except GatedRepoError:
    print(f"  (gated) {repo}", file=sys.stderr)
    sys.exit(0)
except HfHubHTTPError as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in (401, 403, 404):
        sys.exit(1)
    raise
except Exception:
    sys.exit(1)

names = [s.rfilename for s in (info.siblings or [])]
if any(weight_re.search(n or "") for n in names):
    sys.exit(0)
real = [n for n in names if n not in (".gitattributes", "README.md", ".gitignore")]
sys.exit(0 if len(real) >= 3 else 1)
PY
}

resolve_hf_model() {
    local choice="$1"
    if [[ -n "${GLIMMER_HF_MODEL:-}" ]]; then
        echo "${GLIMMER_HF_MODEL}"
        return 0
    fi
    case "${choice}" in
        30b|4bit|auto|mlx) echo "${DEFAULT_4BIT}" ;;
        5bit)     echo "mlx-community/Muse-Glimmer-30B-5bit" ;;
        6bit)     echo "mlx-community/Muse-Glimmer-30B-6bit" ;;
        8bit)     echo "mlx-community/Muse-Glimmer-30B-8bit" ;;
        mxfp4)    echo "mlx-community/Muse-Glimmer-30B-mxfp4" ;;
        official|bf16) echo "mlx-community/Muse-Glimmer-30B-bf16" ;;
        */*)      echo "${choice}" ;;
        *)
            echo "ERROR: Unknown model choice '${choice}'." >&2
            echo "  Use: 4bit | 5bit | 6bit | 8bit | mxfp4 | official | --deps-only | <org/repo>" >&2
            exit 1
            ;;
    esac
}

local_dir_for_repo() {
    local repo="$1"
    local slug
    slug="$(echo "${repo}" | tr '/:' '--' | tr '[:upper:]' '[:lower:]')"
    echo "${SCRIPT_DIR}/${slug}"
}

echo "=== Muse Glimmer 30B + mlx-vlm Setup ==="
if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ Mode:    deps-only (skip model pull)"
else
    echo "→ Choice:  ${MODEL_CHOICE}"
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

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "→ venv active: $(python --version)"

# ── Install / upgrade mlx-vlm (Muse Glimmer needs >= 0.6.12) ──────────────────
echo ""
echo "→ Installing / upgrading mlx mlx-lm mlx-vlm huggingface_hub[cli] ..."
pip install --upgrade pip --quiet
pip install --upgrade "mlx-vlm>=0.6.12" mlx mlx-lm "huggingface_hub[cli]"

echo ""
echo "→ mlx-vlm: $(python -c 'import mlx_vlm; print(getattr(mlx_vlm, "__version__", "installed"))' 2>/dev/null || echo installed)"

if [[ "${DEPS_ONLY}" == true ]]; then
    echo ""
    echo "✅  Dependencies ready (no model pull)."
    echo "  Re-run without --deps-only to download weights."
    exit 0
fi

HF_MODEL="$(resolve_hf_model "${MODEL_CHOICE}")"
MODEL_ALIAS="${GLIMMER_ALIAS:-muse-glimmer-30b-mlx}"
MODEL_DIR="$(local_dir_for_repo "${HF_MODEL}")"
VALIDATE_MODEL="${SCRIPT_DIR}/validate_model.py"

echo ""
echo "→ Target:  ${HF_MODEL}"
echo "→ Local:   ${MODEL_DIR}"
echo "→ Alias:   ${MODEL_ALIAS}"
echo "→ Drafter: ${DRAFT_HF}"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "→ Checking Hugging Face for pullable weights ..."
if hf_in_cache "${HF_MODEL}" || [[ -d "${MODEL_DIR}" && -f "${MODEL_DIR}/config.json" ]]; then
    echo "→ Found local / cached files — will resume or reuse."
elif hf_repo_has_weights "${HF_MODEL}"; then
    echo "→ Remote repo looks pullable."
else
    echo ""
    echo "ERROR: No pullable weights for ${HF_MODEL}."
    echo "  Search: https://huggingface.co/models?search=Muse-Glimmer-30B"
    echo "  Official: https://huggingface.co/meta-models/Muse-Glimmer-30B"
    echo "  MLX 4-bit: https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit"
    exit 2
fi

download_if_needed() {
    local dir="$1"
    local repo="$2"
    local label="$3"

    if [[ -d "${dir}" ]] && python3 "${VALIDATE_MODEL}" "${dir}" >/dev/null 2>&1; then
        echo "→ ${label} already complete — skipping download"
        python3 "${VALIDATE_MODEL}" "${dir}"
        return 0
    fi

    if [[ -d "${dir}" ]]; then
        echo "→ ${label} incomplete — resuming download"
        python3 "${VALIDATE_MODEL}" "${dir}" 2>&1 || true
    else
        echo "→ Downloading ${label} from ${repo} ..."
    fi
    echo ""
    hf download "${repo}" --local-dir "${dir}"
    echo ""
    python3 "${VALIDATE_MODEL}" "${dir}"
    echo "→ ${label} download complete: ${dir}"
}

download_if_needed "${MODEL_DIR}" "${HF_MODEL}" "target model"

if [[ "${SKIP_DFLASH}" == false ]]; then
    echo ""
    if hf_repo_has_weights "${DRAFT_HF}" || hf_in_cache "${DRAFT_HF}" || [[ -d "${DRAFT_DIR}" ]]; then
        download_if_needed "${DRAFT_DIR}" "${DRAFT_HF}" "DFlash assistant"
    else
        echo "WARNING: DFlash assistant ${DRAFT_HF} not pullable — skip, start without --with-dflash."
    fi
else
    echo ""
    echo "→ Skipping DFlash assistant (--skip-dflash-download)."
fi

# ── Write config ──────────────────────────────────────────────────────────────
cat > "${SCRIPT_DIR}/.mlx_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_MODEL="${HF_MODEL}"
MODEL_DIR="${MODEL_DIR}"
MODEL_ALIAS="${MODEL_ALIAS}"
MODEL_CHOICE="${MODEL_CHOICE}"
DRAFT_HF="${DRAFT_HF}"
DRAFT_DIR="${DRAFT_DIR}"
EOF

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        ${HF_MODEL}"
echo "  Config:       ${SCRIPT_DIR}/.mlx_config"
echo ""
echo "  Start server: ./2_start_mlx.sh              # DFlash on if drafter is present"
echo "                ./2_start_mlx.sh --no-dflash  # target only"
echo "  Harness gate: python3 test_harness.py --gate   # after start"
