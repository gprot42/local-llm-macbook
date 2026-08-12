#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install mtplx and download Qwen3.8-27B weights
#
# Qwen3.8-27B was announced open-weight (Aug 2026) as the practical companion
# to Qwen3.8-Max. Official HF IDs land first; mtplx-optimized / mlx-community
# quants usually follow within hours of the base release.
#
# Expected open-weight window (community reports): ~2026-08-12 10:00 UTC+8.
# Until then this script installs the stack and exits cleanly with code 2
# (weights not published) instead of a misleading HF "denied access" error.
#
# Model selection (first arg):
#   27b | 4bit       default — 4-bit MLX quant (mlx-community/Qwen3.8-27B-4bit)
#   mtplx            Youssofal MTPLX-optimized (4-bit class when published; best MTP)
#   official | bf16  full / bf16 weights (large; not the default)
#   <org/repo>       raw Hugging Face repo id
#   --deps-only      install venv + mtplx only (no pull)
#
# Default is always 4-bit. Env overrides:
#   QWEN38_HF_MODEL  force a specific HF repo (wins over choice)
#   QWEN38_ALIAS     OpenAI model id exposed by mtplx (default qwen3.8-27b-mtplx)
#
# Usage:
#   ./1_setup_download.sh                 # 4-bit (default)
#   ./1_setup_download.sh 4bit
#   ./1_setup_download.sh mtplx
#   ./1_setup_download.sh --deps-only
#   QWEN38_HF_MODEL=someone/Qwen3.8-27B-4bit ./1_setup_download.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="${1:-4bit}"
DEPS_ONLY=false

if [[ "${MODEL_CHOICE}" == "--deps-only" || "${MODEL_CHOICE}" == "deps-only" ]]; then
    DEPS_ONLY=true
    MODEL_CHOICE="4bit"
fi

# Default 4-bit target (same role as Qwen3.6's ~18 GB path).
DEFAULT_4BIT="mlx-community/Qwen3.8-27B-4bit"

# Optional 4-bit-class alternates. Prefer cache hits, then any published repo
# that actually has weight files (not README-only placeholders).
AUTO_4BIT_CANDIDATES=(
    "mlx-community/Qwen3.8-27B-4bit"
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-V2"
    "lmstudio-community/Qwen3.8-27B-MLX-4bit"
    "unsloth/Qwen3.8-27B-MLX-4bit"
)

# Weight-ish file patterns that mean a repo is actually pullable.
WEIGHT_NAME_RE='(\.safetensors$|\.npz$|\.bin$|\.gguf$|model.*\.json$|weights\.npz$)'

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
for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
    p = try_to_load_from_cache(repo, name)
    if p and Path(p).is_file():
        sys.exit(0)
sys.exit(1)
PY
}

# Exit 0 if remote repo exists AND has weight files (not README-only skeleton).
hf_repo_has_weights() {
    local repo="$1"
    python3 - "${repo}" <<'PY' 2>/dev/null
import re
import sys
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError

repo = sys.argv[1]
weight_re = re.compile(r"(\.safetensors$|\.npz$|\.bin$|\.gguf$|model.*\.json$)", re.I)
try:
    info = model_info(repo)
except RepositoryNotFoundError:
    sys.exit(1)
except GatedRepoError:
    # Gated but real — caller may need HF_TOKEN; treat as present.
    print(f"  (gated) {repo}", file=sys.stderr)
    sys.exit(0)
except HfHubHTTPError as e:
    # 401 on missing private-style ids is common on HF HTML; treat as missing.
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in (401, 403, 404):
        sys.exit(1)
    raise
except Exception:
    sys.exit(1)

siblings = info.siblings or []
names = [s.rfilename for s in siblings]
if any(weight_re.search(n or "") for n in names):
    sys.exit(0)
# Some MLX quants only list shards without config yet mid-upload — require >2 files
# beyond gitattributes/README to avoid README-only placeholders.
real = [n for n in names if n not in (".gitattributes", "README.md", ".gitignore")]
sys.exit(0 if len(real) >= 3 else 1)
PY
}

# Best-effort discover a published 4-bit-class Qwen3.8-27B MLX quant on HF.
hf_discover_4bit() {
    python3 <<'PY' 2>/dev/null
import re
import sys
from huggingface_hub import HfApi, model_info

api = HfApi()
prefer = re.compile(r"(4bit|4-bit|mlx-4|MLX-4bit|MTPLX)", re.I)
weight_re = re.compile(r"(\.safetensors$|\.npz$|\.bin$|\.gguf$)", re.I)
queries = ["Qwen3.8-27B", "Qwen3.8-27B-4bit", "Qwen3.8-27B MLX"]
seen = set()
candidates = []

for q in queries:
    try:
        models = list(api.list_models(search=q, limit=40, sort="lastModified"))
    except Exception:
        continue
    for m in models:
        rid = m.id
        if rid in seen:
            continue
        seen.add(rid)
        low = rid.lower()
        if "qwen3.8-27b" not in low and "qwen3.8_27b" not in low:
            continue
        # Skip known empty placeholders / non-MLX schemes when better options exist.
        candidates.append(rid)

# Prefer mlx / mtplx / 4bit naming, then anything with real weights.
def score(rid: str) -> tuple:
    s = 0
    low = rid.lower()
    if "mlx-community" in low:
        s += 50
    if "youssofal" in low or "mtplx" in low:
        s += 40
    if prefer.search(rid):
        s += 30
    if "lmstudio" in low or "unsloth" in low:
        s += 20
    if any(x in low for x in ("fp8", "nvfp4", "gguf", "ninfer")):
        s -= 10
    return (-s, rid)

for rid in sorted(candidates, key=score):
    try:
        info = model_info(rid)
    except Exception:
        continue
    names = [s.rfilename for s in (info.siblings or [])]
    if any(weight_re.search(n or "") for n in names):
        print(rid)
        sys.exit(0)
sys.exit(1)
PY
}

resolve_hf_model() {
    local choice="$1"
    if [[ -n "${QWEN38_HF_MODEL:-}" ]]; then
        echo "${QWEN38_HF_MODEL}"
        return 0
    fi
    case "${choice}" in
        27b|4bit|auto|mlx)
            local cand
            # 1) Prefer a 4-bit-class candidate already in the HF cache.
            for cand in "${AUTO_4BIT_CANDIDATES[@]}"; do
                if hf_in_cache "${cand}"; then
                    echo "${cand}"
                    return 0
                fi
            done
            # 2) Prefer a published candidate that has real weight files.
            for cand in "${AUTO_4BIT_CANDIDATES[@]}"; do
                if hf_repo_has_weights "${cand}"; then
                    echo "${cand}"
                    return 0
                fi
            done
            # 3) Discover any newly published Qwen3.8-27B MLX/4bit repo.
            if cand="$(hf_discover_4bit)"; then
                echo "${cand}"
                return 0
            fi
            # 4) Fall through to default id (pull will preflight-fail clearly).
            echo "${DEFAULT_4BIT}"
            ;;
        mtplx)    echo "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed" ;;
        official) echo "Qwen/Qwen3.8-27B" ;;
        bf16)     echo "mlx-community/Qwen3.8-27B-bf16" ;;
        */*)      echo "${choice}" ;;
        *)
            echo "ERROR: Unknown model choice '${choice}'." >&2
            echo "  Use: 4bit | 27b | mtplx | official | bf16 | --deps-only | <org/repo>" >&2
            exit 1
            ;;
    esac
}

echo "=== Qwen3.8-27B + mtplx Setup ==="
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
    echo "  Re-run without --deps-only once Qwen3.8-27B weights are on Hugging Face."
    echo "  Sibling stack while waiting: ../qwen3-6-27b-coder-mtplx/"
    exit 0
fi

# Resolve model only after huggingface_hub is available.
HF_MODEL="$(resolve_hf_model "${MODEL_CHOICE}")"
MODEL_ALIAS="${QWEN38_ALIAS:-qwen3.8-27b-mtplx}"
MODEL_SIZE="27b"

echo ""
echo "→ Model:   ${HF_MODEL}"
echo "→ Alias:   ${MODEL_ALIAS}"

# ── Preflight: distinguish missing/placeholder vs real gated/auth issues ─────
echo ""
echo "→ Checking Hugging Face for pullable weights ..."
if hf_in_cache "${HF_MODEL}"; then
    echo "→ Found in local HF cache — will use cached files."
elif hf_repo_has_weights "${HF_MODEL}"; then
    echo "→ Remote repo looks pullable."
else
    echo ""
    echo "ERROR: No pullable weights for ${HF_MODEL}."
    echo ""
    echo "  This is almost always \"not published yet\", not a broken install."
    echo "  Hugging Face often returns 401/\"denied access\" for missing repo ids;"
    echo "  that is NOT a token problem when the repo simply does not exist."
    echo ""
    echo "  Status as of this stack:"
    echo "    • Official Qwen3.8-27B open weights announced for ~2026-08-12 (UTC+8)"
    echo "    • mlx-community / Youssofal quants usually appear hours after base weights"
    echo "    • README-only placeholders on HF are ignored by this preflight"
    echo ""
    echo "  What worked just now:"
    echo "    • venv:  ${VENV_DIR}"
    echo "    • mtplx: installed — re-run this script once a repo is live"
    echo ""
    echo "  When a quant lands:"
    echo "    ./1_setup_download.sh"
    echo "    ./1_setup_download.sh org/actual-repo-id"
    echo "    QWEN38_HF_MODEL=org/repo ./1_setup_download.sh"
    echo ""
    echo "  Sibling stack while waiting: ../qwen3-6-27b-coder-mtplx/"
    echo "  Search: https://huggingface.co/models?search=Qwen3.8-27B"
    # Exit 2 = deps OK, weights unavailable (callers can treat separately from real failures).
    exit 2
fi

# ── Download model ────────────────────────────────────────────────────────────
echo ""
echo "→ Downloading model: ${HF_MODEL} ..."
echo "  Cached under ~/.cache/huggingface (resumable)."
echo ""

if ! mtplx pull "${HF_MODEL}"; then
    echo ""
    echo "ERROR: mtplx pull failed for ${HF_MODEL}."
    echo ""
    echo "  Likely causes:"
    echo "    • Weights not published yet / mid-upload"
    echo "    • Gated repo — run:  huggingface-cli login"
    echo "    • Wrong / typo'd repo id"
    echo "    • Network / disk space"
    echo ""
    echo "  When a quant lands, re-run with an explicit id:"
    echo "    ./1_setup_download.sh mlx-community/Qwen3.8-27B-4bit"
    echo "    QWEN38_HF_MODEL=org/repo ./1_setup_download.sh"
    echo ""
    echo "  Sibling stack while waiting: ../qwen3-6-27b-coder-mtplx/"
    exit 1
fi
echo "→ Model download complete."

# ── Write config ──────────────────────────────────────────────────────────────
cat > "${SCRIPT_DIR}/.mtplx_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_MODEL="${HF_MODEL}"
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
