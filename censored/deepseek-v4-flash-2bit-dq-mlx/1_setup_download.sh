#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download DeepSeek MLX model and install deps
#
# DeepSeek V4 Flash 2bit-DQ (strong coding/reasoning, heavy — ~97 GB, 128 GB Mac)
# Best: hard coding/agent work in Kilo when quality beats speed; long-context
#       reasoning; SWE-style multi-file tasks; local substitute for a strong
#       cloud coder when you have the RAM
# OK:   day-to-day coding, refactors, chat with <think>; agent loops if you can
#       wait for load/prefill; general reasoning that is not ultra-latency-sensitive
# Bad:  fast iteration / snappy tool loops (prefer Qwen 3.6); uncensored /
#       low-refusal needs (use Heretic Gemma); machines under ~128 GB; treating
#       2-bit MoE as cloud-frontier on huge greenfield apps
#
# Model options (pass as first arg):
#   v4-flash   DeepSeek-V4-Flash-2bit-DQ  (~97 GB, 284B MoE / 13B active)  ← default
#   r1-32b     DeepSeek-R1-Distill-Qwen-32B-4bit  (~18 GB, fast reasoning)
#
# Usage:
#   ./1_setup_download.sh              # V4 Flash (best for 128 GB M5 Max)
#   ./1_setup_download.sh r1-32b       # smaller R1 distill
#   ./1_setup_download.sh --skip-download
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="${1:-v4-flash}"
SKIP_DOWNLOAD=false

# PyPI mlx-lm (≤0.31.3) has no deepseek_v4 model class. Install the community
# fork that carries Flash/Pro support + long-context / tool-call fixes.
# Track: https://github.com/ml-explore/mlx-lm/pull/1192
MLX_LM_GIT_URL="${MLX_LM_GIT_URL:-git+https://github.com/spicyneuron/mlx-lm.git@_ds4}"

# Latest transformers is fine; setup patches mlx-lm for the 5.13+ register API change.
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.12}"

for arg in "$@"; do
    [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=true
done

case "${MODEL_CHOICE}" in
    v4-flash|--skip-download)
        if [[ "${MODEL_CHOICE}" == "--skip-download" ]]; then
            MODEL_CHOICE="v4-flash"
            SKIP_DOWNLOAD=true
        fi
        HF_REPO="mlx-community/DeepSeek-V4-Flash-2bit-DQ"
        MODEL_DIR="$SCRIPT_DIR/deepseek-v4-flash-2bit-dq"
        MODEL_DESC="~97 GB — DeepSeek-V4-Flash (284B MoE, 13B active, 1M context)"
        ;;
    r1-32b)
        HF_REPO="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
        MODEL_DIR="$SCRIPT_DIR/deepseek-r1-distill-qwen-32b-4bit"
        MODEL_DESC="~18 GB — DeepSeek-R1 distilled into Qwen 32B (chain-of-thought)"
        ;;
    --help|-h)
        awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL_CHOICE}'. Use v4-flash or r1-32b."
        exit 1
        ;;
esac

MODEL_ID="$(basename "$MODEL_DIR")"

echo "=== DeepSeek MLX Setup (${MODEL_CHOICE}) ==="
echo "→ Repo:  $HF_REPO"
echo "→ Dir:   $MODEL_DIR"
echo "→ Size:  $MODEL_DESC"
echo ""

if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "→ Creating virtualenv at venv/ ..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Prefer the venv interpreter explicitly (works even if activate is skipped later).
VENV_PY="$SCRIPT_DIR/venv/bin/python"
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

echo "→ Installing / upgrading pip + latest stack (${TRANSFORMERS_SPEC}) ..."
"$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade pip
"$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade \
    mlx \
    huggingface_hub \
    jinja2 \
    numpy \
    protobuf \
    pyyaml \
    sentencepiece \
    safetensors \
    "${TRANSFORMERS_SPEC}"

echo "→ Installing mlx-lm with DeepSeek-V4 support from:"
echo "   $MLX_LM_GIT_URL"
echo "   (PyPI mlx-lm has no deepseek_v4 module yet)"
"$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade --force-reinstall --no-deps "$MLX_LM_GIT_URL"
# Ensure runtime deps stay latest after the git install.
"$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade \
    mlx \
    numpy \
    protobuf \
    pyyaml \
    sentencepiece \
    jinja2 \
    huggingface_hub \
    safetensors \
    "${TRANSFORMERS_SPEC}"

# 2bit-DQ SwitchGLU + transformers 5.13 tokenizer register fix
if [[ -x "$SCRIPT_DIR/apply_local_patches.sh" ]]; then
    "$SCRIPT_DIR/apply_local_patches.sh"
else
    bash "$SCRIPT_DIR/apply_local_patches.sh"
fi
echo "→ Dependencies installed."
echo ""

# Fail fast with the real import error (not a silent bool).
if [[ "$MODEL_CHOICE" == "v4-flash" ]]; then
    if ! "$VENV_PY" - <<'PY'
import importlib
import traceback
import sys
try:
    importlib.import_module("mlx_lm.models.deepseek_v4")
except Exception:
    traceback.print_exc()
    sys.exit(1)
print("OK")
PY
    then
        echo ""
        echo "ERROR: mlx_lm.models.deepseek_v4 is not importable after install."
        echo "       Common causes:"
        echo "         - wrong mlx-lm fork / branch (need deepseek_v4)"
        echo "         - transformers 5.13+ without the setup script patch"
        echo "       Override fork:  MLX_LM_GIT_URL=git+https://github.com/... ./1_setup_download.sh"
        exit 1
    fi
    echo "→ Verified: mlx_lm.models.deepseek_v4 is available."
    "$VENV_PY" - <<'PY'
from importlib.metadata import version
for pkg in ("mlx", "mlx-lm", "transformers", "huggingface_hub", "tokenizers"):
    try:
        print(f"   {pkg:16s} {version(pkg)}")
    except Exception:
        print(f"   {pkg:16s} (not installed)")
PY
    echo ""
fi

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"

download_if_needed() {
    local dir="$1"
    local repo="$2"
    local label="$3"

    model_is_complete() {
        [ -d "$dir" ] && "$VENV_PY" "$VALIDATE_MODEL" "$dir" >/dev/null 2>&1
    }

    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo "→ Skipping $label download (--skip-download)."
        if [ -d "$dir" ] && ! model_is_complete; then
            echo "→ WARNING: $label weights incomplete — run without --skip-download"
            "$VENV_PY" "$VALIDATE_MODEL" "$dir" 2>&1 || true
        fi
        return 0
    fi

    if model_is_complete; then
        echo "→ $label already complete — skipping download"
        "$VENV_PY" "$VALIDATE_MODEL" "$dir"
        return 0
    fi

    if [ -d "$dir" ]; then
        echo "→ $label incomplete — resuming download"
        "$VENV_PY" "$VALIDATE_MODEL" "$dir" 2>&1 || true
    else
        echo "→ Downloading $label from $repo ..."
    fi
    echo ""
    hf download "$repo" --local-dir "$dir"
    echo ""
    "$VENV_PY" "$VALIDATE_MODEL" "$dir"
    echo "→ $label download complete: $dir"
}

download_if_needed "$MODEL_DIR" "$HF_REPO" "model"

cat > "$SCRIPT_DIR/.deepseek_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_REPO="${HF_REPO}"
MODEL_DIR="${MODEL_DIR}"
MODEL_ID="${MODEL_ID}"
MODEL_CHOICE="${MODEL_CHOICE}"
MLX_LM_GIT_URL="${MLX_LM_GIT_URL}"
EOF

chmod +x "$SCRIPT_DIR/validate_model.py" "$SCRIPT_DIR/2_start_mlx.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        $MODEL_DIR"
echo "  Model ID:     $MODEL_ID"
echo "  mlx-lm:       $MLX_LM_GIT_URL"
echo "  Start server: ./2_start_mlx.sh"
echo ""
echo "  Quick test (no server):"
echo "    source venv/bin/activate"
echo "    mlx_lm.generate --model $MODEL_DIR --prompt 'Hello, who are you?' --max-tokens 32"
