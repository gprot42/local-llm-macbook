#!/usr/bin/env bash
# apply_local_patches.sh — Copy project patches/ into the active venv.
# Called by 2_start_mlx.sh on every start (pip upgrade cannot revert fixes):
#   sample_utils.py       — MTP top-p on 3-D verify logits
#   server/app.py         — default_model → MLX_VLM_DEFAULT_MODEL
#   models/gemma4/language.py — MTP rollback accepts Python lists
#   patch_generation.py — close BatchGenerator after MTP errors (no hung server)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f venv/bin/activate ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source venv/bin/activate

SITE="venv/lib/python3.14/site-packages"
if [[ ! -d "$SITE" ]]; then
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
fi

if [[ ! -d patches ]]; then
    echo "→ No patches/ directory — nothing to apply."
    exit 0
fi

echo "→ Applying local patches to $SITE ..."

if [[ -f patches/sample_utils.py && -d "$SITE/mlx_vlm" ]]; then
    cp -f patches/sample_utils.py "$SITE/mlx_vlm/sample_utils.py"
    echo "   mlx_vlm/sample_utils.py (MTP top-p fix)"
fi

if [[ -f patches/app.py && -d "$SITE/mlx_vlm/server" ]]; then
    cp -f patches/app.py "$SITE/mlx_vlm/server/app.py"
    echo "   mlx_vlm/server/app.py (default_model alias)"
fi

if [[ -f patches/models/gemma4/language.py && -d "$SITE/mlx_vlm/models/gemma4" ]]; then
    cp -f patches/models/gemma4/language.py "$SITE/mlx_vlm/models/gemma4/language.py"
    echo "   mlx_vlm/models/gemma4/language.py (MTP rollback list fix)"
fi

if [[ -f patches/patch_generation.py && -f "$SITE/mlx_vlm/server/generation.py" ]]; then
    python3 patches/patch_generation.py "$SITE/mlx_vlm/server/generation.py"
fi

echo "→ Patches applied."