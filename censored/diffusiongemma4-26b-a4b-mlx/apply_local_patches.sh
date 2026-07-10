#!/usr/bin/env bash
# apply_local_patches.sh — Copy project patches/ into the active venv.
# Called by 2_run_mlx.sh on every start (pip upgrade cannot revert fixes).
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

if [[ -f patches/responses_state.py && -d "$SITE/mlx_vlm/server" ]]; then
    cp -f patches/responses_state.py "$SITE/mlx_vlm/server/responses_state.py"
    echo "   mlx_vlm/server/responses_state.py (bare call:fn{...} tool parsing)"
fi

echo "→ Patches applied."