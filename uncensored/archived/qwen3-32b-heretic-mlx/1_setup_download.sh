#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download Qwen3-32B Heretic (MLX 5-bit) + install deps
#
# HF repo: Wwayu/Qwen3-32B-heretic-mlx-5Bit
#   (MLX conversion of igriv/Qwen3-32B-heretic — Heretic abliteration of Qwen3-32B)
#
# ~22.5 GB weights. 64 GB+ unified memory recommended; 80–128 GB for long context.
#
# Usage:
#   ./1_setup_download.sh              # download + venv
#   ./1_setup_download.sh --skip-download
#   ./1_setup_download.sh --force      # re-download even if complete
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

HF_REPO="Wwayu/Qwen3-32B-heretic-mlx-5Bit"
MODEL_DIR="$SCRIPT_DIR/qwen3-32b-heretic-mlx-5bit"
MODEL_ID="qwen3-32b-heretic-mlx-5bit"
MODEL_DESC="~22.5 GB — Qwen3-32B Heretic (MLX 5-bit, uncensored/abliterated)"

echo "=== Qwen3-32B Heretic — Setup (MLX 5-bit) ==="
echo "→ Repo:  $HF_REPO"
echo "→ Dir:   $MODEL_DIR"
echo "→ Size:  $MODEL_DESC"
echo "→ RAM:   64 GB+ unified (80–128 GB for long agent context)"
echo ""

# ── Repair venv after project directory rename/move ──────────────────────────
repair_relocated_venv() {
    local venv="$SCRIPT_DIR/venv"
    [[ -d "$venv/bin" ]] || return 0

    local sample="" shebang interp old_path=""
    for sample in "$venv/bin/mlx_lm.server" "$venv/bin/pip" "$venv/bin/hf" "$venv/bin/mlx_lm.chat"; do
        [[ -f "$sample" ]] || continue
        shebang="$(head -1 "$sample" 2>/dev/null || true)"
        if [[ "$shebang" == "#!"* ]]; then
            interp="${shebang#\#!}"
        elif [[ "$shebang" == /*/python* ]]; then
            interp="$shebang"
        else
            continue
        fi
        if [[ -n "$interp" && ! -e "$interp" ]]; then
            old_path="$(dirname "$(dirname "$interp")")"
            break
        fi
    done

    if [[ -z "$old_path" && -f "$venv/bin/activate" ]]; then
        local old_ve
        old_ve="$(grep -E 'export VIRTUAL_ENV=' "$venv/bin/activate" | head -1 | sed -E 's/.*VIRTUAL_ENV=//' | tr -d "\"'")"
        if [[ -n "$old_ve" && "$old_ve" != "$venv" && ! -d "$old_ve" ]]; then
            old_path="$old_ve"
        fi
    fi

    [[ -n "$old_path" ]] || return 0
    echo "→ Detected relocated venv (old path: $old_path)"
    echo "→ Rewriting shebangs and activate scripts → $venv"

    local f
    for f in "$venv/bin"/*; do
        [[ -f "$f" && -r "$f" ]] || continue
        if head -1 "$f" 2>/dev/null | grep -qF "$old_path"; then
            # Rewrite only the shebang line's absolute path prefix.
            sed -i '' "1s|${old_path}|${venv}|g" "$f" 2>/dev/null || \
                sed -i "1s|${old_path}|${venv}|g" "$f" 2>/dev/null || true
        fi
    done
    for f in "$venv/bin/activate" "$venv/bin/activate.csh" "$venv/bin/activate.fish"; do
        [[ -f "$f" ]] || continue
        if grep -qF "$old_path" "$f" 2>/dev/null; then
            sed -i '' "s|${old_path}|${venv}|g" "$f" 2>/dev/null || \
                sed -i "s|${old_path}|${venv}|g" "$f" 2>/dev/null || true
        fi
    done
    if [[ -f "$venv/pyvenv.cfg" ]] && grep -qF "$old_path" "$venv/pyvenv.cfg" 2>/dev/null; then
        sed -i '' "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg" 2>/dev/null || \
            sed -i "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg" 2>/dev/null || true
    fi
    echo "→ Venv repair done."
}

repair_relocated_venv

# ── Python venv ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: Python 3.10+ required. Install via: brew install python@3.12"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "→ Creating virtualenv at venv/ (${PYTHON}) ..."
    "$PYTHON" -m venv "$SCRIPT_DIR/venv"
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

echo "→ Installing / upgrading pip + MLX stack ..."
"$VENV_PY" -m pip install --quiet --upgrade pip
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    "$VENV_PY" -m pip install --quiet --upgrade -r "$SCRIPT_DIR/requirements.txt"
else
    "$VENV_PY" -m pip install --quiet --upgrade \
        "mlx>=0.26" \
        "mlx-lm>=0.28" \
        "huggingface_hub>=0.26" \
        "transformers>=4.51" \
        jinja2 \
        numpy \
        protobuf \
        pyyaml \
        sentencepiece \
        safetensors \
        fastapi \
        uvicorn \
        httpx
fi
echo "→ Dependencies installed."
"$VENV_PY" - <<'PY'
from importlib.metadata import version
for pkg in ("mlx", "mlx-lm", "transformers", "huggingface_hub"):
    try:
        print(f"   {pkg:16s} {version(pkg)}")
    except Exception:
        print(f"   {pkg:16s} (not installed)")
PY
echo ""

VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"

model_is_complete() {
    [ -d "$MODEL_DIR" ] && "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" >/dev/null 2>&1
}

if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "→ Skipping download (--skip-download)."
    if [ -d "$MODEL_DIR" ] && ! model_is_complete; then
        echo "→ WARNING: weights incomplete — run without --skip-download"
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    elif model_is_complete; then
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
    fi
elif model_is_complete && [ "$FORCE_DOWNLOAD" = false ]; then
    echo "→ Model already complete — skipping download"
    "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
else
    if [ -d "$MODEL_DIR" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        echo "→ Model incomplete — resuming download"
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    else
        echo "→ Downloading $HF_REPO → $MODEL_DIR ..."
        if [ "$FORCE_DOWNLOAD" = true ]; then
            echo "   (--force: re-fetching)"
        fi
    fi
    echo ""
    FORCE_FLAG=()
    [[ "$FORCE_DOWNLOAD" == true ]] && FORCE_FLAG=(--force-download)
    hf download "$HF_REPO" --local-dir "$MODEL_DIR" "${FORCE_FLAG[@]}"
    echo ""
    "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
    echo "→ Download complete: $MODEL_DIR"
fi

cat > "$SCRIPT_DIR/.qwen3_heretic_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_REPO="${HF_REPO}"
MODEL_DIR="${MODEL_DIR}"
MODEL_ID="${MODEL_ID}"
EOF

chmod +x "$SCRIPT_DIR/validate_model.py" \
    "$SCRIPT_DIR/2_start_mlx.sh" \
    "$SCRIPT_DIR/3_chat.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:        $MODEL_DIR"
echo "  Model ID:     $MODEL_ID"
echo "  Start server: ./2_start_mlx.sh"
echo "  Terminal chat: ./3_chat.sh"
echo ""
echo "  Quick test (no server):"
echo "    source venv/bin/activate"
echo "    mlx_lm.generate --model $MODEL_DIR --prompt 'Hello, who are you?' --max-tokens 64"
echo ""
