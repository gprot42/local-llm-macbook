#!/usr/bin/env bash
# apply_local_patches.sh — Copy project patches/ into the active venv.
# Called by 2_start_mlx.sh on every start (pip upgrade cannot revert fixes).
#
# Uses the venv python by absolute path (does not rely on `source activate`),
# so it still works after the project directory is renamed/moved.
#
# vllm-mlx 0.4.0+ already includes:
#   - gemma4_text dispatch in text_model_from_vlm.py
#   - logit_bias on ChatCompletionRequest + server wiring
# Overwriting those with the older patches/ copies would *downgrade* 0.4.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY=""
for cand in "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/venv/bin/python3"; do
    if [[ -x "$cand" ]]; then
        VENV_PY="$cand"
        break
    fi
done
if [[ -z "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first." >&2
    exit 1
fi

SITE="$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')"

if [[ ! -d patches ]]; then
    echo "→ No patches/ directory — nothing to apply."
    exit 0
fi

pkg_ver() {
    local pkg="$1"
    "$VENV_PY" -c "from importlib.metadata import version; print(version('$pkg'))" 2>/dev/null || echo "0"
}

# Compare dotted versions: prints "yes" if $1 >= $2
ver_ge() {
    local have="$1" need="$2"
    "$VENV_PY" - "$have" "$need" <<'PY'
import sys
def parts(v: str):
    out = []
    for p in v.split("."):
        digits = "".join(c for c in p if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)
print("yes" if parts(sys.argv[1]) >= parts(sys.argv[2]) else "no")
PY
}

file_contains() {
    [[ -f "$1" ]] && grep -q "$2" "$1"
}

VLLM_VER="$(pkg_ver vllm-mlx)"
MLX_VLM_VER="$(pkg_ver mlx-vlm)"
VLLM_GE_040="$(ver_ge "$VLLM_VER" "0.4.0")"
MLX_VLM_GE_050="$(ver_ge "$MLX_VLM_VER" "0.5.0")"

echo "→ Applying local patches to $SITE ..."
echo "   (vllm-mlx $VLLM_VER, mlx-vlm $MLX_VLM_VER)"

# ── mlx-vlm legacy patches (only < 0.5) ──────────────────────────────────────
if [[ "$MLX_VLM_GE_050" == "no" && -f patches/sample_utils.py && -d "$SITE/mlx_vlm" ]]; then
    cp -f patches/sample_utils.py "$SITE/mlx_vlm/sample_utils.py"
    echo "   mlx_vlm/sample_utils.py (legacy mlx-vlm < 0.5 MTP top-p fix)"
elif [[ "$MLX_VLM_GE_050" == "yes" ]]; then
    echo "   mlx_vlm/sample_utils.py skipped (mlx-vlm ≥ 0.5 already has full sample_utils)"
fi

if [[ "$MLX_VLM_GE_050" == "no" && -f patches/generate.py && -d "$SITE/mlx_vlm" ]]; then
    cp -f patches/generate.py "$SITE/mlx_vlm/generate.py"
    echo "   mlx_vlm/generate.py (mlx-vlm < 0.5.0)"
elif [[ "$MLX_VLM_GE_050" == "yes" ]]; then
    echo "   mlx_vlm/generate.py skipped (thread-local stream in mlx-vlm ≥ 0.5.0 generate/)"
fi

# ── vllm-mlx: gemma4 mask-trim (still useful on 0.4.0) ───────────────────────
if [[ -f patches/gemma4_mllm.py && -d "$SITE/vllm_mlx/patches" ]]; then
    cp -f patches/gemma4_mllm.py "$SITE/vllm_mlx/patches/gemma4_mllm.py"
    echo "   vllm_mlx/patches/gemma4_mllm.py (mask trim for BatchedEngine)"
fi

# ── vllm-mlx: text-only VLM checkpoints (Heretic 4-bit has no vision shards) ─
# config.json still says Gemma4ForConditionalGeneration + image_token_id, but
# weight_map is language_model.* only. Without this, mlx_vlm strict load dies
# with "Missing 211 parameters: vision_tower...".
UTILS_PY="$SITE/vllm_mlx/api/utils.py"
if [[ -f "$UTILS_PY" ]]; then
    if file_contains "$UTILS_PY" '_weights_are_text_only_vlm'; then
        echo "   vllm_mlx/api/utils.py already has text-only VLM guard — skip"
    else
        "$VENV_PY" - "$UTILS_PY" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = '''def is_mllm_model(model_name: str) -> bool:
    """Check if a model name or path indicates a multimodal language model.

    Two complementary validations are run:

    1. config.json inspection: when ``model_name`` resolves to a local
       directory containing a readable config.json, inspect the model's
       own metadata (``architectures`` field, ``vision_config``,
       ``audio_config``, etc.). Authoritative when available because it
       reflects what the model actually is, not how it is named on disk.

    2. Legacy substring match against ``MLLM_PATTERNS``: applied when no
       config.json is reachable (e.g., a HuggingFace repo ID before the
       weights are downloaded). Preserves the historical behaviour.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the model is detected as multimodal (MLLM/VLM).
    """
    config = _try_read_config_json(model_name)
    if config is not None:
        return _config_indicates_vlm(config)
    return _check_legacy_string_patterns(model_name)'''
        new = '''def _weights_are_text_only_vlm(model_name: str) -> bool:
    """True when config looks multimodal but weight shards have no vision/audio.

    Some abliterated / text-only conversions (e.g. Gemma 4 Heretic 4-bit)
    keep ``Gemma4ForConditionalGeneration`` + image_token_id in config.json
    while shipping only ``language_model.*`` tensors. Loading those through
    mlx-vlm with strict=True fails with hundreds of missing vision_tower
    parameters. Treat them as LLM so vllm-mlx uses mlx_lm instead.
    """
    try:
        candidate = Path(model_name)
    except (TypeError, ValueError):
        return False
    if not candidate.is_dir():
        return False
    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        if index_path.stat().st_size > _MAX_CONFIG_JSON_BYTES * 16:
            return False
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    multimodal_markers = (
        "vision_tower",
        "embed_vision",
        "multi_modal_projector",
        "audio_tower",
        "embed_audio",
        "sound_encoder",
    )
    for key in weight_map:
        if not isinstance(key, str):
            continue
        for marker in multimodal_markers:
            if marker in key:
                return False
    return any(
        isinstance(k, str)
        and (k.startswith("language_model") or ".layers." in k or k.startswith("model."))
        for k in weight_map
    )


def is_mllm_model(model_name: str) -> bool:
    """Check if a model name or path indicates a multimodal language model.

    Two complementary validations are run:

    1. config.json inspection: when ``model_name`` resolves to a local
       directory containing a readable config.json, inspect the model's
       own metadata (``architectures`` field, ``vision_config``,
       ``audio_config``, etc.). Authoritative when available because it
       reflects what the model actually is, not how it is named on disk.

    2. Legacy substring match against ``MLLM_PATTERNS``: applied when no
       config.json is reachable (e.g., a HuggingFace repo ID before the
       weights are downloaded). Preserves the historical behaviour.

    Local text-only VLM checkpoints (config multimodal, weights language-only)
    are treated as LLM so mlx_lm can load them.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the model is detected as multimodal (MLLM/VLM).
    """
    config = _try_read_config_json(model_name)
    if config is not None:
        if not _config_indicates_vlm(config):
            return False
        if _weights_are_text_only_vlm(model_name):
            return False
        return True
    return _check_legacy_string_patterns(model_name)'''
        if old not in text:
            raise SystemExit(f"is_mllm_model block not found in {path}")
        path.write_text(text.replace(old, new, 1))
        print(f"patched {path}", flush=True)
PY
        echo "   vllm_mlx/api/utils.py (text-only VLM → LLM load path)"
    fi
fi

# ── vllm-mlx 0.3.x-only full-file patches (merged upstream in 0.4.0) ─────────
if [[ "$VLLM_GE_040" == "yes" ]]; then
    echo "   vllm_mlx/text_model_from_vlm.py skipped (gemma4_text dispatch in 0.4.0+)"
    echo "   vllm_mlx/api/models.py skipped (logit_bias in 0.4.0+)"
    echo "   vllm_mlx/server.py skipped (logit_bias wiring in 0.4.0+)"
    echo "   vllm_mlx/engine/simple.py skipped (0.4.0 engine rewrite; do not downgrade)"
else
    # 0.3.x: apply full-file patches only when the fix is still missing
    if [[ -f patches/text_model_from_vlm.py && -d "$SITE/vllm_mlx" ]]; then
        if file_contains "$SITE/vllm_mlx/text_model_from_vlm.py" 'gemma4_text'; then
            echo "   vllm_mlx/text_model_from_vlm.py already has gemma4_text — skip"
        else
            cp -f patches/text_model_from_vlm.py "$SITE/vllm_mlx/text_model_from_vlm.py"
            echo "   vllm_mlx/text_model_from_vlm.py (gemma4_text dispatch)"
        fi
    fi
    if [[ -f patches/api/models.py && -d "$SITE/vllm_mlx/api" ]]; then
        if file_contains "$SITE/vllm_mlx/api/models.py" 'logit_bias'; then
            echo "   vllm_mlx/api/models.py already has logit_bias — skip"
        else
            cp -f patches/api/models.py "$SITE/vllm_mlx/api/models.py"
            echo "   vllm_mlx/api/models.py (logit_bias field)"
        fi
    fi
    if [[ -f patches/server.py && -d "$SITE/vllm_mlx" ]]; then
        if file_contains "$SITE/vllm_mlx/server.py" 'logit_bias'; then
            echo "   vllm_mlx/server.py already has logit_bias wiring — skip"
        else
            cp -f patches/server.py "$SITE/vllm_mlx/server.py"
            echo "   vllm_mlx/server.py"
        fi
    fi
    if [[ -f patches/engine/simple.py && -d "$SITE/vllm_mlx/engine" ]]; then
        if file_contains "$SITE/vllm_mlx/engine/simple.py" 'logit_bias'; then
            echo "   vllm_mlx/engine/simple.py already has logit_bias — skip"
        else
            cp -f patches/engine/simple.py "$SITE/vllm_mlx/engine/simple.py"
            echo "   vllm_mlx/engine/simple.py (logit_bias for 0.3.x)"
        fi
    fi
fi

echo "→ Patches applied."
