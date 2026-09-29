#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download Gemma 4 31B JANG_4M CRACK (MLX) + install deps
#
# dealignai/Gemma-4-31B-JANG_4M-CRACK is already MLX-native and pre-quantized
# (JANG mixed 4/8-bit, ~5.1 avg bits, ~22 GB). No local re-quantize step needed.
#
# Usage:
#   ./1_setup_download.sh              # download weights + venv/deps
#   ./1_setup_download.sh --skip-download
#   ./1_setup_download.sh --force      # re-download even if files exist
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
            echo "Usage: ./1_setup_download.sh [options]"
            echo ""
            echo "  (default)         gemma-4-31b-jang-crack-mlx  (~22 GB, JANG mixed quant)"
            echo "  --skip-download   Install Python deps only"
            echo "  --force           Re-download all files (passes --force-download to hf)"
            echo ""
            echo "Model is already Mac-optimized (MLX safetensors + JANG quant)."
            echo "You do NOT need to run quantize_to_mlx_4bit.sh for the Hub weights."
            echo ""
            echo "Sibling uncensored Gemma:"
            echo "  ../gemma4-server-heretic-31b-mlx/  — Heretic 4-bit (+ vision graft)"
            exit 0
            ;;
    esac
done

MODEL_DIR="$SCRIPT_DIR/gemma-4-31b-jang-crack-mlx"
HF_REPO="dealignai/Gemma-4-31B-JANG_4M-CRACK"

echo "=== Gemma 4 31B JANG_4M CRACK — Setup ==="
echo "→ Model repo: $HF_REPO"
echo "→ Local dir:  $MODEL_DIR"
echo "→ Format:     MLX-native JANG v2 (~5.1 avg bits, vision included)"
echo "→ RAM tip:    64–80 GB+ unified memory for long agent sessions"
echo ""

# ── Repair venv after project directory rename/move ──────────────────────────
repair_relocated_venv() {
    local venv="$SCRIPT_DIR/venv"
    [[ -d "$venv/bin" ]] || return 0

    local sample="" f shebang interp old_path=""
    for sample in "$venv/bin/vllm-mlx" "$venv/bin/pip" "$venv/bin/hf" "$venv/bin/mlx_lm.chat"; do
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

    if [[ -z "$old_path" || "$old_path" == "$venv" ]]; then
        return 0
    fi

    echo "→ Detected moved project directory; repairing venv paths"
    echo "   $old_path → $venv"
    while IFS= read -r -d '' f; do
        if grep -qF "$old_path" "$f" 2>/dev/null; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                sed -i '' "s|${old_path}|${venv}|g" "$f"
            else
                sed -i "s|${old_path}|${venv}|g" "$f"
            fi
        fi
        shebang="$(head -1 "$f" 2>/dev/null || true)"
        if [[ "$shebang" == "$venv/bin/python"* ]]; then
            local tmp
            tmp="$(mktemp)"
            { echo "#!$shebang"; tail -n +2 "$f"; } > "$tmp"
            mv "$tmp" "$f"
            chmod +x "$f" 2>/dev/null || true
        fi
    done < <(find "$venv/bin" -type f -print0 2>/dev/null)
    if [[ -f "$venv/pyvenv.cfg" ]] && grep -qF "$old_path" "$venv/pyvenv.cfg" 2>/dev/null; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            sed -i '' "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg"
        else
            sed -i "s|${old_path}|${venv}|g" "$venv/pyvenv.cfg"
        fi
    fi
    echo "→ venv paths repaired."
    echo ""
}

repair_relocated_venv

# ── Python / venv ─────────────────────────────────────────────────────────────
if [[ ! -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    echo "→ Creating virtualenv at venv/ ..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

VENV_PY="$SCRIPT_DIR/venv/bin/python"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

if [[ ! -x "$SCRIPT_DIR/venv/bin/vllm-mlx" ]] && ! "$VENV_PY" -c "import vllm_mlx" 2>/dev/null; then
    echo "→ Installing vllm-mlx and proxy deps (first run may take a minute) ..."
    "$VENV_PY" -m pip install --quiet --no-cache-dir --upgrade pip
    "$VENV_PY" -m pip install --quiet --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"
    echo "→ Dependencies installed."
else
    echo "→ Python deps already installed — skipping pip"
fi
echo ""

# ── Download weights ──────────────────────────────────────────────────────────
VALIDATE_MODEL="$SCRIPT_DIR/validate_model.py"
chmod +x "$VALIDATE_MODEL" 2>/dev/null || true

model_is_complete() {
    [[ -d "$MODEL_DIR" ]] && "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" >/dev/null 2>&1
}

list_missing_files() {
    "$VENV_PY" "$VALIDATE_MODEL" --list-missing "$MODEL_DIR" 2>/dev/null || true
}

resolve_hf_bin() {
    if [[ -x "$SCRIPT_DIR/venv/bin/hf" ]]; then
        echo "$SCRIPT_DIR/venv/bin/hf"
    elif command -v hf >/dev/null 2>&1; then
        echo "hf"
    else
        echo "ERROR: 'hf' CLI not found. Install deps: ./1_setup_download.sh --skip-download" >&2
        return 1
    fi
}

download_missing() {
    local hf_flags=()
    local hf_bin
    hf_bin="$(resolve_hf_bin)" || exit 1

    if [[ "$FORCE_DOWNLOAD" == true ]]; then
        hf_flags+=(--force-download)
        echo "→ Force download: fetching full repo (existing files may be replaced)"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        "$hf_bin" download "$HF_REPO" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
        return
    fi

    hf_flags+=(--no-force-download)
    export HF_HUB_ENABLE_HF_TRANSFER=1

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "→ First download of $HF_REPO ..."
        "$hf_bin" download "$HF_REPO" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
        return
    fi

    MISSING=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && MISSING+=("$line")
    done < <(list_missing_files)
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        echo "→ All weight files already on disk — nothing to download"
        return
    fi

    echo "→ Fetching ${#MISSING[@]} missing file(s) (skipping files already present):"
    for f in "${MISSING[@]}"; do
        echo "     - $f"
    done
    "$hf_bin" download "$HF_REPO" "${MISSING[@]}" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
}

if [[ "$SKIP_DOWNLOAD" == true ]]; then
    echo "→ Skipping download (--skip-download passed)."
    if [[ -d "$MODEL_DIR" ]] && ! model_is_complete; then
        echo "→ WARNING: local model is incomplete — run without --skip-download"
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    fi
elif model_is_complete && [[ "$FORCE_DOWNLOAD" != true ]]; then
    echo "→ Model already complete — skipping download (use --force to re-fetch)"
    "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
else
    if [[ -d "$MODEL_DIR" ]]; then
        echo "→ Checking which files are still needed ..."
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    fi
    echo ""
    download_missing
    echo ""
    if ! "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"; then
        echo ""
        echo "ERROR: model is still incomplete after download."
        exit 1
    fi
fi

# Optional: confirm vision tensors exist (JANG ships multimodal; no stock graft needed).
if [[ -d "$MODEL_DIR" ]] && model_is_complete; then
    if [[ -f "$MODEL_DIR/jang_config.json" ]]; then
        echo "→ jang_config.json present (JANG v2 metadata)"
    fi
    if "$VENV_PY" - "$MODEL_DIR" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
idx = root / "model.safetensors.index.json"
if not idx.is_file():
    raise SystemExit(1)
wm = json.loads(idx.read_text()).get("weight_map") or {}
ok = any(isinstance(k, str) and (k.startswith("vision_tower") or k.startswith("embed_vision") or "vision" in k) for k in wm)
raise SystemExit(0 if ok else 1)
PY
    then
        echo "→ Vision weights present in index (multimodal, no graft step)"
    else
        echo "→ NOTE: vision keys not obvious in weight_map — text path still works"
    fi
fi

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:          $MODEL_DIR"
echo "  Quantization:   JANG_4M (mixed 4/8-bit, ~5.1 avg) — already Mac-optimized"
echo "  Modalities:     text + image (shipped with checkpoint)"
echo "  Kilo model ID:  gemma-4-31b-jang-crack-mlx"
echo "  Start server:   ./2_start_mlx.sh"
echo ""
echo "  Why not re-quantize? See README.md § Quantization / macOS"
echo ""
