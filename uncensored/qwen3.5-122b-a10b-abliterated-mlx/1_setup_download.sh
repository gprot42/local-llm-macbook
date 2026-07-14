#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Download Qwen3.5-122B-A10B abliterated (MLX 4-bit) + deps
#
# Default: prebuilt MLX 4-bit abliterated pack (~70 GB) that fits 128 GB unified:
#
#   vanch007/Qwen3.5-122B-A10B-abliterated-4bit-vlm-mlx-cs2764-final
#     ← abliterated quant of Qwen/Qwen3.5-122B-A10B
#
# Full BF16 abliterated (huihui-ai/… ~250 GB) does NOT fit convert-on-device on
# 128 GB / ~250 GB free disk — use a prebuilt MLX quant instead.
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh --skip-download
#   ./1_setup_download.sh --force
#   HF_REPO=other/org-model ./1_setup_download.sh
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

# Prebuilt MLX 4-bit abliterated (default). Override with any HF id.
HF_REPO="${HF_REPO:-vanch007/Qwen3.5-122B-A10B-abliterated-4bit-vlm-mlx-cs2764-final}"
SRC_DIR="$SCRIPT_DIR/hf-source"
MODEL_DIR="$SCRIPT_DIR/qwen3.5-122b-a10b-abliterated-mlx-4bit"
MODEL_ID="qwen3.5-122b-a10b-abliterated-mlx-4bit"
MODEL_DESC="Qwen3.5-122B-A10B abliterated → MLX 4-bit (~70 GB)"

echo "=== Qwen3.5-122B-A10B Abliterated — Setup (MLX 4-bit) ==="
echo "→ HF repo:  $HF_REPO"
echo "→ Source:   $SRC_DIR  (only used if convert needed)"
echo "→ MLX dir:  $MODEL_DIR"
echo "→ Size:     $MODEL_DESC"
echo "→ RAM:      128 GB unified recommended (MoE ~10B active / ~70 GB weights)"
echo ""
echo "  Alternatives:"
echo "    HF_REPO=TheCluster/Qwen3.5-122B-A10B-Heretic-v2-MLX-mixed-3.8bit"
echo "    HF_REPO=osmapi/Qwen3.5-122B-A10B-Abliterated-MLX-3"
echo "    HF_REPO=Jcoa/Qwen3.5-122B-A10B-Abliterated-MLX-mixed3_6"
echo "    HF_REPO=huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated  # BF16 ~250 GB — convert needs more RAM/disk"
echo ""

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
        jinja2 numpy protobuf pyyaml sentencepiece safetensors fastapi uvicorn httpx
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
DOWNLOAD_RESUMABLE="$SCRIPT_DIR/download_resumable.py"

model_is_complete() {
    local dir="$1"
    [ -d "$dir" ] && "$VENV_PY" "$VALIDATE_MODEL" "$dir" >/dev/null 2>&1
}

looks_quantized() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] && grep -q 'quantization' "$dir/config.json" 2>/dev/null
}

has_weight_shards() {
    local dir="$1"
    [[ -f "$dir/model.safetensors.index.json" ]] || \
        compgen -G "$dir/model-*-of-*.safetensors" >/dev/null 2>&1 || \
        [[ -f "$dir/model.safetensors" ]] || \
        [[ -f "$dir/weights.safetensors" ]]
}

# Resolve HF tree root (some repos nest BF16 under bf16/)
resolve_weight_root() {
    local dir="$1"
    if has_weight_shards "$dir"; then
        echo "$dir"
        return 0
    fi
    if has_weight_shards "$dir/bf16"; then
        echo "$dir/bf16"
        return 0
    fi
    return 1
}

if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "→ Skipping download (--skip-download)."
    if model_is_complete "$MODEL_DIR"; then
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
    else
        echo "→ WARNING: MLX weights incomplete at $MODEL_DIR"
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR" 2>&1 || true
    fi
else
    if model_is_complete "$MODEL_DIR" && [ "$FORCE_DOWNLOAD" = false ]; then
        echo "→ MLX model already complete — skipping download/convert"
        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
    else
        echo "→ Fetching $HF_REPO ..."
        echo "   If you get 401/403:  hf auth login"
        echo "   Resume: completed shards skip; partial shards continue via HTTP Range"
        echo "   (stock hf download does NOT resume multi-GB partials — we use download_resumable.py)"
        echo ""
        FORCE_FLAG=()
        [[ "$FORCE_DOWNLOAD" == true ]] && FORCE_FLAG=(--force-download)

        # Prefer download straight into MODEL_DIR when the hub pack is already MLX-quantized.
        # Fall back to SRC_DIR + convert for raw BF16 overrides (needs huge RAM/disk).
        mkdir -p "$MODEL_DIR"

        # download_resumable.py wraps snapshot_download with stable .incomplete files +
        # HTTP Range resume. Own the PID so Ctrl+C can TERM cleanly (flush partials)
        # before escalating to KILL (which would strand buffers).
        DOWNLOAD_PID=""
        stop_download() {
            echo ""
            echo "→ Download interrupted (Ctrl+C)."
            if [[ -n "${DOWNLOAD_PID:-}" ]] && kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
                # TERM first so open files flush and stable .incomplete is kept.
                kill -TERM "$DOWNLOAD_PID" 2>/dev/null || true
                local _i
                for _i in $(seq 1 50); do
                    kill -0 "$DOWNLOAD_PID" 2>/dev/null || break
                    sleep 0.1
                done
                if kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
                    # Hard-kill avoids Python threading._shutdown KeyboardInterrupt spam.
                    kill -KILL "$DOWNLOAD_PID" 2>/dev/null || true
                fi
                # Reap without re-raising set -e on non-zero wait status.
                wait "$DOWNLOAD_PID" 2>/dev/null || true
            fi
            # Promote any UUID orphans left by a hard-kill into stable resume paths.
            "$VENV_PY" "$DOWNLOAD_RESUMABLE" "$HF_REPO" --local-dir "$MODEL_DIR" --cleanup-only 2>/dev/null || true
            echo "  Partial files kept under: $MODEL_DIR"
            echo "  Re-run ./1_setup_download.sh — completed shards skip, partials Range-resume."
            exit 130
        }
        trap stop_download INT TERM

        set +e
        # Unbuffered so "resuming …" lines show up during long transfers.
        PYTHONUNBUFFERED=1 "$VENV_PY" "$DOWNLOAD_RESUMABLE" "$HF_REPO" \
            --local-dir "$MODEL_DIR" "${FORCE_FLAG[@]}" &
        DOWNLOAD_PID=$!
        wait "$DOWNLOAD_PID"
        download_rc=$?
        set -e
        DOWNLOAD_PID=""
        trap - INT TERM

        # 130=SIGINT, 143=SIGTERM, 137=SIGKILL — user/system abort, not a repo error.
        if [[ "$download_rc" -eq 130 || "$download_rc" -eq 143 || "$download_rc" -eq 137 ]]; then
            echo ""
            echo "→ Download interrupted."
            "$VENV_PY" "$DOWNLOAD_RESUMABLE" "$HF_REPO" --local-dir "$MODEL_DIR" --cleanup-only 2>/dev/null || true
            echo "  Partial files kept under: $MODEL_DIR"
            echo "  Re-run ./1_setup_download.sh — completed shards skip, partials Range-resume."
            exit 130
        elif [[ "$download_rc" -ne 0 ]]; then
            echo ""
            echo "ERROR: Failed to download $HF_REPO (exit $download_rc)"
            echo "  • 401/403 → hf auth login  (accept license on the model page if gated)"
            echo "  • 404     → confirm the repo id / visibility"
            echo "  • Default: vanch007/Qwen3.5-122B-A10B-abliterated-4bit-vlm-mlx-cs2764-final"
            echo "  • Override: HF_REPO=org/name ./1_setup_download.sh"
            echo "  • BF16 abliterated (huge): HF_REPO=huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated"
            echo "  • Ctrl+C is safe — re-run continues partial shards (HTTP Range resume)"
            exit 1
        fi

        WEIGHT_ROOT=""
        if WEIGHT_ROOT="$(resolve_weight_root "$MODEL_DIR")"; then
            if looks_quantized "$WEIGHT_ROOT"; then
                if [[ "$WEIGHT_ROOT" != "$MODEL_DIR" ]]; then
                    echo "→ Quantized weights under $WEIGHT_ROOT — promoting to $MODEL_DIR"
                    shopt -s dotglob
                    tmp_promote="${MODEL_DIR}.promote.$$"
                    mv "$WEIGHT_ROOT" "$tmp_promote"
                    rm -rf "$MODEL_DIR"
                    mv "$tmp_promote" "$MODEL_DIR"
                    shopt -u dotglob
                fi
                echo "→ Hub pack is already MLX-quantized at $MODEL_DIR"
            else
                echo "→ Converting HF weights → MLX 4-bit (slow; needs lots of RAM + disk) ..."
                echo "   WARN: full 122B BF16 is ~250 GB — convert may OOM on 128 GB machines."
                rm -rf "$SRC_DIR"
                mv "$MODEL_DIR" "$SRC_DIR"
                WEIGHT_ROOT="$(resolve_weight_root "$SRC_DIR")" || {
                    echo "ERROR: Downloaded tree has no weight shards: $SRC_DIR"
                    exit 1
                }
                mkdir -p "$(dirname "$MODEL_DIR")"
                "$VENV_PY" -m mlx_lm convert \
                    --hf-path "$WEIGHT_ROOT" \
                    --mlx-path "$MODEL_DIR" \
                    -q
                echo "→ Convert complete. You can delete $SRC_DIR to free disk if desired."
            fi
        else
            echo "ERROR: Downloaded tree has no weight shards: $MODEL_DIR"
            exit 1
        fi

        "$VENV_PY" "$VALIDATE_MODEL" "$MODEL_DIR"
        echo "→ Ready: $MODEL_DIR"
    fi
fi

cat > "$SCRIPT_DIR/.qwen35_122b_config" << EOF
# Written by 1_setup_download.sh — do not edit manually
HF_REPO="${HF_REPO}"
MODEL_DIR="${MODEL_DIR}"
MODEL_ID="${MODEL_ID}"
EOF

chmod +x "$SCRIPT_DIR/validate_model.py" \
    "$SCRIPT_DIR/download_resumable.py" \
    "$SCRIPT_DIR/2_start_mlx.sh" \
    "$SCRIPT_DIR/3_chat.sh" 2>/dev/null || true

echo ""
echo "✅  Setup complete (or deps-only)!"
echo ""
echo "  Model:        $MODEL_DIR"
echo "  Model ID:     $MODEL_ID"
echo "  Start server: ./2_start_mlx.sh"
echo "  Terminal chat: ./3_chat.sh"
echo ""
