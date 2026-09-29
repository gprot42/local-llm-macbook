#!/usr/bin/env bash
# =============================================================================
# quantize_to_mlx_4bit.sh — OPTIONAL re-quantize to uniform MLX 4-bit
#
# DO YOU NEED THIS?
#   Almost never for dealignai/Gemma-4-31B-JANG_4M-CRACK.
#
#   The Hub package is ALREADY Mac-optimized:
#     - MLX-native safetensors (not PyTorch bf16)
#     - JANG importance quant (mixed 4/8-bit, ~5.1 avg bits)
#     - ~18–22 GB on disk; vision in float16
#     - Ready for vllm-mlx / mlx-vlm after ./1_setup_download.sh
#
#   Re-running a naive 4-bit convert on the published JANG weights is wrong:
#   they are already quantized. Double-quantizing destroys quality.
#
# WHEN TO USE THIS SCRIPT
#   Only if you have a *full-precision* (bf16/fp16) Gemma 4 31B checkpoint
#   (stock IT or a full-precision abliterated export) and want a standard
#   mlx-community-style uniform 4-bit tree for smaller disk / simpler loaders.
#
# Usage:
#   ./quantize_to_mlx_4bit.sh /path/to/bf16-or-fp16-model [output-dir]
#
# Example (if you ever obtain full-precision CRACK/IT weights):
#   ./quantize_to_mlx_4bit.sh ./gemma-4-31b-it-bf16 ./gemma-4-31b-custom-mlx-4bit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC="${1:-}"
OUT="${2:-$SCRIPT_DIR/gemma-4-31b-custom-mlx-4bit}"

if [[ -z "$SRC" || "$SRC" == "-h" || "$SRC" == "--help" ]]; then
    sed -n '2,35p' "$SCRIPT_DIR/quantize_to_mlx_4bit.sh" | sed -E 's/^# ?//'
    exit 0
fi

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source model dir not found: $SRC" >&2
    exit 1
fi

# Refuse to "quantize" the published JANG tree (already quantized).
if [[ -f "$SRC/jang_config.json" ]]; then
    echo "ERROR: $SRC looks like a JANG quant (jang_config.json present)." >&2
    echo "       Do not re-quantize it. Use ./1_setup_download.sh and load as-is." >&2
    exit 1
fi
if grep -q '"bits"' "$SRC/config.json" 2>/dev/null && \
   grep -qE '"bits"[[:space:]]*:[[:space:]]*[0-9]+' "$SRC/config.json" 2>/dev/null; then
    # Heuristic: already has MLX quantization block
    if grep -q '"quantization"' "$SRC/config.json" 2>/dev/null; then
        echo "WARNING: $SRC/config.json already has a quantization block." >&2
        echo "         Re-quantizing an already-quantized MLX model will hurt quality." >&2
        read -r -p "Continue anyway? [y/N] " ans
        [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
    fi
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv missing. Run ./1_setup_download.sh --skip-download first." >&2
    exit 1
fi
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

echo "=== Optional: convert full-precision → MLX uniform 4-bit ==="
echo "→ Source: $SRC"
echo "→ Output: $OUT"
echo "→ This needs a lot of RAM/disk and can take a long time."
echo ""

# Prefer mlx_vlm.convert for multimodal Gemma 4; fall back to mlx_lm.convert.
if "$VENV_PY" -c "import mlx_vlm" 2>/dev/null; then
    echo "→ Using python -m mlx_vlm.convert -q --q-bits 4"
    "$VENV_PY" -m mlx_vlm.convert \
        --hf-path "$SRC" \
        --mlx-path "$OUT" \
        -q --q-bits 4 --q-group-size 64
elif "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
    echo "→ mlx_vlm missing; using python -m mlx_lm.convert -q --q-bits 4 (text path)"
    "$VENV_PY" -m mlx_lm.convert \
        --hf-path "$SRC" \
        --mlx-path "$OUT" \
        -q --q-bits 4 --q-group-size 64
else
    echo "ERROR: neither mlx_vlm nor mlx_lm importable. Install: ./1_setup_download.sh --skip-download" >&2
    exit 1
fi

echo ""
echo "✅  Wrote uniform 4-bit MLX model to: $OUT"
echo "   Point 2_start_mlx.sh MODEL_DIR at this path only if you intentionally"
echo "   switched away from the published JANG_4M CRACK weights."
echo ""
