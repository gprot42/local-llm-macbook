#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download Gemma 4 31B Heretic (MLX 4-bit) + install deps
#
# Also fetches stock vision (shard 4 + config) and grafts it into Heretic so the
# checkpoint is multimodal. Upstream Heretic 4-bit is language_model-only.
#
# Usage:
#   ./1_setup_download.sh              # download 31B heretic + stock vision + graft
#   ./1_setup_download.sh --skip-download
#   ./1_setup_download.sh --force      # re-download even if files exist
#   ./1_setup_download.sh --skip-vision  # language-only Heretic (no stock vision)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false
SKIP_VISION=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --force|--force-download) FORCE_DOWNLOAD=true ;;
        --skip-vision|--no-vision) SKIP_VISION=true ;;
        --help|-h)
            echo "Usage: ./1_setup_download.sh [options]"
            echo ""
            echo "  (default)         gemma-4-31b-heretic-mlx-4bit  (~18–20 GB after vision graft)"
            echo "  --skip-download   Install Python deps only (still runs vision graft if possible)"
            echo "  --force           Re-download all files (passes --force-download to hf)"
            echo "  --skip-vision     Do not download/graft stock vision (text-only server path)"
            echo ""
            echo "By default, existing files are kept and only missing shards are fetched."
            echo ""
            echo "Multimodal note:"
            echo "  mlx-community Heretic 4-bit ships language weights only. This script also"
            echo "  downloads stock vision (~2.3 GB shard 4) from mlx-community/gemma-4-31b-it-4bit"
            echo "  (or reuses a local stock-vision cache if present) and grafts vision_tower +"
            echo "  embed_vision into Heretic. Language stays uncensored Heretic."
            echo ""
            echo "For full stock 31B IT (aligned, not Heretic):"
            echo "  ../gemma4-server-atomicchat-mlx-31b-2026-07-15/  (stock IT chat; vision still from mlx-community)"
            exit 0
            ;;
        26b|26B)
            echo "ERROR: This project is 31B-only."
            echo "  Stock 31B IT: ../gemma4-server-atomicchat-mlx-31b-2026-07-15/"
            exit 1
            ;;
    esac
done

MODEL_DIR="$SCRIPT_DIR/gemma-4-31b-heretic-mlx-4bit"
HF_REPO="mlx-community/gemma-4-31B-it-uncensored-heretic-4bit"
# Stock IT 4-bit — source of vision_tower + embed_vision (Heretic HF package omits them)
# Vision still comes from mlx-community multimodal 4-bit (AtomicChat stock is text-focused).
STOCK_VISION_REPO="mlx-community/gemma-4-31b-it-4bit"
STOCK_SIBLING_DIR="$SCRIPT_DIR/stock-vision-source"
STOCK_CACHE_DIR="$SCRIPT_DIR/stock-vision-source"

echo "=== Gemma 4 31B Heretic Uncensored — Setup ==="
echo "→ Model repo: $HF_REPO"
echo "→ Local dir:  $MODEL_DIR"
echo "→ RAM tip:    80 GB+ unified memory recommended for agent sessions"
echo ""

# ── Repair venv after project directory rename/move ──────────────────────────
# Console-script shebangs and activate embed absolute paths. If this folder was
# renamed (e.g. uncensored-31b-mlx → heretic-31b-mlx), rewrite them in place.
repair_relocated_venv() {
    local venv="$SCRIPT_DIR/venv"
    [[ -d "$venv/bin" ]] || return 0

    local sample="" f shebang interp old_path=""
    for sample in "$venv/bin/vllm-mlx" "$venv/bin/pip" "$venv/bin/hf" "$venv/bin/mlx_lm.chat"; do
        [[ -f "$sample" ]] || continue
        shebang="$(head -1 "$sample" 2>/dev/null || true)"
        # Strip exact "#!" prefix only (never use ##! — that can leave a leading '#').
        if [[ "$shebang" == "#!"* ]]; then
            interp="${shebang#\#!}"
        elif [[ "$shebang" == /*/python* ]]; then
            # Already-mangled shebang (path only) from an earlier broken repair.
            interp="$shebang"
        else
            continue
        fi
        if [[ -n "$interp" && ! -e "$interp" ]]; then
            # shebang is .../venv/bin/pythonX.Y → old venv root is dirname dirname
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
        # Restore missing #! on console scripts whose first line is a bare interpreter path
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
    # hf download accepts explicit filenames — only these are requested from the Hub.
    "$hf_bin" download "$HF_REPO" "${MISSING[@]}" --local-dir "$MODEL_DIR" "${hf_flags[@]}"
}

# True when Heretic already has vision tensors (post-graft).
heretic_has_vision() {
    [[ -d "$MODEL_DIR" ]] || return 1
    "$VENV_PY" - "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
idx = root / "model.safetensors.index.json"
shard = root / "model-00004-of-00004.safetensors"
if not idx.is_file() or not shard.is_file():
    raise SystemExit(1)
wm = json.loads(idx.read_text()).get("weight_map") or {}
if not any(isinstance(k, str) and (k.startswith("vision_tower") or k.startswith("embed_vision")) for k in wm):
    raise SystemExit(1)
# Index alone is not enough — force re-download of Heretic can leave stale index + text-only shard.
import mlx.core as mx
keys = mx.load(str(shard)).keys()
raise SystemExit(0 if any(k.startswith("vision_tower") or k.startswith("embed_vision") for k in keys) else 1)
PY
}

# Prefer sibling full stock tree; else local cache with only vision-bearing files.
stock_vision_source_ready() {
    local dir="$1"
    [[ -f "$dir/model-00004-of-00004.safetensors" && -f "$dir/config.json" ]] || return 1
    # Shard must actually contain vision (not an empty/wrong file).
    "$VENV_PY" - "$dir/model-00004-of-00004.safetensors" <<'PY'
import sys
import mlx.core as mx
keys = mx.load(sys.argv[1]).keys()
raise SystemExit(0 if any(k.startswith("vision_tower") or k.startswith("embed_vision") for k in keys) else 1)
PY
}

resolve_stock_vision_dir() {
    if stock_vision_source_ready "$STOCK_SIBLING_DIR"; then
        echo "$STOCK_SIBLING_DIR"
        return 0
    fi
    if stock_vision_source_ready "$STOCK_CACHE_DIR"; then
        echo "$STOCK_CACHE_DIR"
        return 0
    fi
    return 1
}

# Download only what the graft needs (~2.3 GB shard 4 + tiny configs), not full stock 18 GB.
download_stock_vision_source() {
    local hf_bin
    local hf_flags=(--no-force-download)
    hf_bin="$(resolve_hf_bin)" || exit 1
    if [[ "$FORCE_DOWNLOAD" == true ]]; then
        hf_flags=(--force-download)
    fi

    if stock_dir="$(resolve_stock_vision_dir 2>/dev/null)"; then
        echo "→ Stock vision source ready: $stock_dir"
        return 0
    fi

    echo "→ Downloading stock vision source from $STOCK_VISION_REPO"
    echo "   (shard 4 + config only — ~2.3 GB; not the full 18 GB stock model)"
    mkdir -p "$STOCK_CACHE_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    "$hf_bin" download "$STOCK_VISION_REPO" \
        model-00004-of-00004.safetensors \
        config.json \
        processor_config.json \
        --local-dir "$STOCK_CACHE_DIR" \
        "${hf_flags[@]}"

    if ! stock_vision_source_ready "$STOCK_CACHE_DIR"; then
        echo "ERROR: stock vision source incomplete after download: $STOCK_CACHE_DIR" >&2
        exit 1
    fi
    echo "→ Stock vision cached at $STOCK_CACHE_DIR"
}

ensure_vision_grafted() {
    if [[ "$SKIP_VISION" == true ]]; then
        echo "→ Skipping stock vision graft (--skip-vision)"
        return 0
    fi
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "→ No Heretic model dir yet — skip vision graft"
        return 0
    fi
    if ! model_is_complete; then
        echo "→ Heretic incomplete — skip vision graft until language weights are complete"
        return 0
    fi

    chmod +x "$SCRIPT_DIR/graft_vision_from_stock.sh" 2>/dev/null || true

    if heretic_has_vision && [[ "$FORCE_DOWNLOAD" != true ]]; then
        echo "→ Heretic already multimodal (vision grafted) — OK"
        return 0
    fi

    # After --force, HF may have restored text-only shard4; re-graft.
    if heretic_has_vision && [[ "$FORCE_DOWNLOAD" == true ]]; then
        echo "→ Force mode: re-checking vision graft after Heretic re-download"
    fi

    download_stock_vision_source
    local stock_dir
    stock_dir="$(resolve_stock_vision_dir)" || {
        echo "ERROR: no stock vision source available for graft" >&2
        exit 1
    }

    echo "→ Grafting stock vision into Heretic (language stays uncensored) ..."
    "$SCRIPT_DIR/graft_vision_from_stock.sh" "$MODEL_DIR" "$stock_dir"

    if ! heretic_has_vision; then
        echo "ERROR: vision graft did not produce vision tensors in $MODEL_DIR" >&2
        exit 1
    fi
    echo "→ Multimodal Heretic ready (language=Heretic, vision=stock IT)"
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

# Always attempt vision graft after language weights are in place (unless --skip-vision).
echo ""
ensure_vision_grafted

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Model:          $MODEL_DIR"
if heretic_has_vision 2>/dev/null; then
    echo "  Modalities:     text + image (stock vision grafted)"
else
    echo "  Modalities:     text only (run without --skip-vision to graft stock vision)"
fi
echo "  Kilo model ID:  gemma-4-31b-heretic-mlx-4bit"
echo "  Verify patches: ./check_upstream_patches.sh --fetch"
echo "  Start server:   ./2_start_mlx.sh"
echo ""
echo "  Stock vision cache / sibling: $STOCK_CACHE_DIR or $STOCK_SIBLING_DIR"
echo ""
