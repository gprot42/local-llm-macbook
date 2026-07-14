#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — DFlash draft + dflash-mlx env for Qwen3.5-122B-A10B
#
# Target (large): reuses the sibling abliterated MLX 4-bit pack by default
#   ../qwen3.5-122b-a10b-abliterated-mlx/qwen3.5-122b-a10b-abliterated-mlx-4bit
#
# Draft (~1.5 GB BF16): z-lab/Qwen3.5-122B-A10B-DFlash
#   Loaded natively by dflash-mlx (no mlx_lm convert required).
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh --skip-download
#   TARGET_MODEL=/path/to/mlx-target DRAFT_REPO=z-lab/... ./1_setup_download.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --force|--force-download) FORCE_DOWNLOAD=true ;;
        --help|-h)
            awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
    esac
done

DRAFT_REPO="${DRAFT_REPO:-z-lab/Qwen3.5-122B-A10B-DFlash}"
DRAFT_DIR="${DRAFT_DIR:-$SCRIPT_DIR/Qwen3.5-122B-A10B-DFlash}"

SIBLING_TARGET="$SCRIPT_DIR/../qwen3.5-122b-a10b-abliterated-mlx/qwen3.5-122b-a10b-abliterated-mlx-4bit"
TARGET_MODEL="${TARGET_MODEL:-}"
if [ -z "$TARGET_MODEL" ]; then
    if [ -d "$SIBLING_TARGET" ]; then
        TARGET_MODEL="$(cd "$SIBLING_TARGET" && pwd)"
    else
        TARGET_MODEL="$SIBLING_TARGET"
    fi
fi

CONFIG_FILE="$SCRIPT_DIR/.dflash_122b_config"
VENV_DIR="$SCRIPT_DIR/venv"
DFLASH_SRC="$SCRIPT_DIR/dflash-mlx"

echo "==> Qwen3.5-122B DFlash setup"
echo "    target: $TARGET_MODEL"
echo "    draft:  $DRAFT_REPO → $DRAFT_DIR"

if [ ! -d "$DFLASH_SRC/dflash_mlx" ]; then
    echo "ERROR: missing vendored dflash-mlx at $DFLASH_SRC" >&2
    echo "Clone it first: git clone https://github.com/Aryagm/dflash-mlx.git dflash-mlx" >&2
    exit 1
fi

# --- Python env ---
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating venv ..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools >/dev/null

echo "→ Installing dflash-mlx (editable) + deps ..."
python -m pip install -e "$DFLASH_SRC" >/dev/null
# Ensure recent mlx-lm with qwen3_5_moe
python -m pip install -U "mlx-lm>=0.31" "mlx>=0.32" huggingface-hub >/dev/null

# --- Draft download ---
if [ "$SKIP_DOWNLOAD" = false ]; then
    if [ -f "$DRAFT_DIR/model.safetensors" ] && [ -f "$DRAFT_DIR/config.json" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        echo "→ Draft already present: $DRAFT_DIR"
    else
        echo "→ Downloading draft (~1.5 GB) $DRAFT_REPO ..."
        mkdir -p "$DRAFT_DIR"
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${DRAFT_REPO}",
    local_dir="${DRAFT_DIR}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("draft ok")
PY
    fi
else
    echo "→ Skipping draft download (--skip-download)"
fi

if [ ! -f "$DRAFT_DIR/config.json" ] || [ ! -f "$DRAFT_DIR/model.safetensors" ]; then
    echo "ERROR: draft incomplete at $DRAFT_DIR (need config.json + model.safetensors)" >&2
    exit 1
fi

if [ ! -f "$TARGET_MODEL/config.json" ]; then
    echo "ERROR: target model not found at: $TARGET_MODEL" >&2
    echo "Run sibling stack first:" >&2
    echo "  cd ../qwen3.5-122b-a10b-abliterated-mlx && ./1_setup_download.sh" >&2
    echo "Or set TARGET_MODEL=/path/to/mlx-qwen3.5-122b" >&2
    exit 1
fi

# Sanity: MoE + layer count match draft taps
python - <<PY
import json
from pathlib import Path

target = Path(r"""${TARGET_MODEL}""")
draft = Path(r"""${DRAFT_DIR}""")
tc = json.loads((target / "config.json").read_text())
dc = json.loads((draft / "config.json").read_text())
text = tc.get("text_config") or tc
n_layers = text.get("num_hidden_layers")
layer_ids = (dc.get("dflash_config") or {}).get("target_layer_ids") or []
print(f"target model_type={tc.get('model_type')} layers={n_layers} hidden={text.get('hidden_size')}")
print(f"draft target_layer_ids={layer_ids} num_target_layers={dc.get('num_target_layers')}")
if tc.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
    raise SystemExit(f"unsupported target model_type={tc.get('model_type')}")
if n_layers is None:
    raise SystemExit("target missing num_hidden_layers")
bad = [i for i in layer_ids if i < 0 or i >= n_layers]
if bad:
    raise SystemExit(f"draft layer ids out of range for target: {bad}")
print("config cross-check OK")
PY

# Persist paths for 2_start / smoke
cat > "$CONFIG_FILE" <<EOF
TARGET_MODEL=$(cd "$TARGET_MODEL" && pwd)
DRAFT_MODEL=$(cd "$DRAFT_DIR" && pwd)
DRAFT_REPO=${DRAFT_REPO}
PORT=8086
HOST=127.0.0.1
MODEL_ID=qwen3.5-122b-a10b-dflash
EOF

echo ""
echo "Setup complete."
echo "  Config: $CONFIG_FILE"
echo "  Next:   ./2_start_dflash.sh          # OpenAI API on :8086"
echo "          ./3_smoke_exactness.sh       # greedy exactness vs plain target"
echo "          ./3_smoke_exactness.sh --quick  # short generate only"
