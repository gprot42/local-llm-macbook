#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — TensorFold (native v1.0.x) for Qwen3.8-27B OBLITERATED
#
# TensorFold 1.0+ is a native Zig binary; it is no longer pip-installable
# (`pip install git+...` stops with a message pointing here). This script
# installs the native binary with Homebrew and links the local MLX 4-bit
# weights — no venv, no pip. The loop proxy and harness run on the system
# python3 (standard library only).
#
# Reuses the V3 MLX 4-bit convert already on disk (affine 4-bit, group 64,
# ~14 GB). Does not download the Hub repo: that tree is GGUF plus leftover
# shards plus bf16. TensorFold serves the qwen3_5 checkpoint and drafts with
# z-lab/Qwen3.8-27B-DFlash2.
#
#   ./1_setup_download.sh                 # brew install + link weights + drafter
#   ./1_setup_download.sh --deps-only     # brew install + link, no drafter download
#   ./1_setup_download.sh --no-drafter    # serial decode, no DFlash2
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"
MODELS_DIR="${SCRIPT_DIR}/models"
MLX_DIR="${MODELS_DIR}/mlx-4bit"

DEFAULT_DRAFTER="z-lab/Qwen3.8-27B-DFlash2"
MODEL_ALIAS="${QWEN38_OBL_ALIAS:-qwen3.8-27b-obliterated-tensorfold}"
CONTEXT="${QWEN38_OBL_CONTEXT:-262144}"

# TensorFold native binary. 1.0.5 is the first release whose Qwen3.8-27B
# Metal kernels compile on macOS 26.3 / 27 (Metal 4.1); earlier 1.0.x crash at
# load with LibraryLoad. Bump only after re-checking that the checkpoint loads
# (`tensorfold info`) and a drafted reply equals `"draft": false`.
TF_FORMULA="${TF_FORMULA:-ashhart/tensorfold/tensorfold}"
TF_MIN_VERSION="${TF_MIN_VERSION:-1.0.5}"

# The autosaddler tree holds the real files. The mtplx folder symlinks them.
SIBLING_AUTOSADDLER="${SCRIPT_DIR}/../archived/qwen3-8-27b-obliterated-mtplx-autosaddler/models/mlx-4bit"
SIBLING_MTPLX="${SCRIPT_DIR}/../archived/qwen3-8-27b-obliterated-mtplx/models/mlx-4bit"

DEPS_ONLY=false
WITH_DRAFTER=true

for arg in "$@"; do
    case "${arg}" in
        --deps-only|deps-only) DEPS_ONLY=true ;;
        --no-drafter) WITH_DRAFTER=false ;;
        --help|-h)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: ${arg}"
            exit 2
            ;;
    esac
done

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: TensorFold on macOS needs Apple Silicon (arm64). Got $(uname -m)."
    exit 1
fi

mlx_complete() {
    local dir="$1"
    [[ -f "${dir}/config.json" && -f "${dir}/model.safetensors.index.json" && -f "${dir}/tokenizer.json" ]]
}

# Put Homebrew's bin on PATH for this process so a fresh install is found.
for brew_bin in /opt/homebrew/bin /usr/local/bin; do
    [[ -d "${brew_bin}" ]] && case ":${PATH}:" in *":${brew_bin}:"*) ;; *) PATH="${brew_bin}:${PATH}" ;; esac
done
export PATH

version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

ensure_tensorfold() {
    local have ver
    if command -v tensorfold >/dev/null 2>&1; then
        ver="$(tensorfold --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
        if [[ -n "${ver}" ]] && version_ge "${ver}" "${TF_MIN_VERSION}"; then
            echo "→ TensorFold: $(tensorfold --version 2>&1 | head -1) (${TF_MIN_VERSION}+ OK)"
            return 0
        fi
        echo "→ TensorFold ${ver:-unknown} is older than ${TF_MIN_VERSION}; upgrading ..."
    fi
    if ! command -v brew >/dev/null 2>&1; then
        echo "ERROR: Homebrew not found. Install the native binary one of these ways:"
        echo "         brew install ${TF_FORMULA}"
        echo "       or download the macOS arm64 archive from:"
        echo "         https://github.com/ashhart/TensorFold/releases"
        exit 1
    fi
    echo "→ Installing the native TensorFold binary: brew install ${TF_FORMULA}"
    if command -v tensorfold >/dev/null 2>&1; then
        brew upgrade "${TF_FORMULA##*/}" 2>/dev/null || brew install "${TF_FORMULA}"
    else
        brew install "${TF_FORMULA}"
    fi
    have="$(command -v tensorfold || true)"
    if [[ -z "${have}" ]]; then
        echo "ERROR: tensorfold is still not on PATH after install."
        exit 1
    fi
    ver="$(tensorfold --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if ! version_ge "${ver:-0}" "${TF_MIN_VERSION}"; then
        echo "ERROR: installed TensorFold ${ver:-unknown} is below ${TF_MIN_VERSION}."
        exit 1
    fi
    echo "→ TensorFold: $(tensorfold --version 2>&1 | head -1)"
}

link_weights() {
    local src
    if mlx_complete "${MLX_DIR}"; then
        echo "→ Weights already at ${MLX_DIR}"
        return 0
    fi
    for src in "${SIBLING_AUTOSADDLER}" "${SIBLING_MTPLX}"; do
        if mlx_complete "${src}"; then
            mkdir -p "${MODELS_DIR}"
            # Resolve so the link target is the directory that holds the shards.
            src="$(cd "${src}" && pwd -P)"
            ln -sfn "${src}" "${MLX_DIR}"
            echo "→ Linked ${MLX_DIR} -> ${src}"
            return 0
        fi
    done
    echo "ERROR: no complete mlx-4bit checkpoint found."
    echo "       Expected one of:"
    echo "         ${SIBLING_AUTOSADDLER}"
    echo "         ${SIBLING_MTPLX}"
    echo "       Build it with the archived stack, then re-run this script:"
    echo "         ../archived/qwen3-8-27b-obliterated-mtplx/1_setup_download.sh"
    exit 1
}

echo "→ Linking OBLITERATED V3 mlx-4bit (no second copy of the shards) ..."
link_weights

ensure_tensorfold

echo "→ Model:  ${MLX_DIR}"
if [[ "${WITH_DRAFTER}" == true ]]; then
    echo "→ Draft:  ${DEFAULT_DRAFTER}"
else
    echo "→ Draft:  off"
fi

echo ""
echo "→ Checking local checkpoint ..."
if ! tensorfold info "${MLX_DIR}"; then
    echo "ERROR: tensorfold info failed for ${MLX_DIR}."
    echo "       The native engine must recognise it as qwen3_5, 4-bit group 64."
    exit 1
fi

# The OBLITERATED checkpoint's chat_template.jinja renders no tools, so the
# native engine would never place the tool list in the prompt. Ship a Qwen3
# tool-capable template (XML tool format) for 2_start_tensorfold.sh to pass with
# --chat-template. It is committed in the stack; if it is missing, copy one from
# a base Qwen3.8-27B MLX checkpoint in the Hugging Face cache.
TOOL_TEMPLATE="${SCRIPT_DIR}/chat_template_tools.jinja"
if [[ -f "${TOOL_TEMPLATE}" ]] && grep -q tool_call "${TOOL_TEMPLATE}" 2>/dev/null; then
    echo "→ Tool template: ${TOOL_TEMPLATE} (present)"
else
    found=""
    for cand in "${HOME}"/.cache/huggingface/hub/*Qwen3.8-27B*/snapshots/*/chat_template.jinja; do
        [[ -f "${cand}" ]] && grep -q tool_call "${cand}" 2>/dev/null || continue
        found="${cand}"; break
    done
    if [[ -n "${found}" ]]; then
        cp "${found}" "${TOOL_TEMPLATE}"
        echo "→ Tool template: copied from ${found}"
    else
        echo "WARNING: no tool-capable chat template found. Tool calls will not render."
        echo "         Pull a base Qwen3.8-27B MLX checkpoint, or add chat_template_tools.jinja,"
        echo "         or start with --no-tool-template."
    fi
fi

DRAFTER=""
if [[ "${WITH_DRAFTER}" == true ]]; then
    DRAFTER="${DEFAULT_DRAFTER}"
fi

cat > "${CONFIG_FILE}" <<EOF
# Generated by 1_setup_download.sh. Re-run that script to change it.
HF_MODEL=${MLX_DIR}
DRAFTER=${DRAFTER}
MODEL_ALIAS=${MODEL_ALIAS}
CONTEXT=${CONTEXT}
EOF
echo "→ Wrote ${CONFIG_FILE}"

if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ --deps-only: skipping drafter download."
    echo "  Next: ./2_start_tensorfold.sh   (it pulls the drafter if DRAFTER is set)"
    exit 0
fi

if [[ -n "${DRAFTER}" ]]; then
    echo ""
    echo "→ Downloading drafter into the Hugging Face cache ..."
    tensorfold pull "${DRAFTER}"
fi

echo ""
echo "============================================================"
echo "  SETUP DONE"
echo "============================================================"
echo "  Engine:  $(tensorfold --version 2>&1 | head -1) (native)"
echo "  Model:   ${MLX_DIR}"
echo "  Drafter: ${DRAFTER:-none}"
echo "  Alias:   ${MODEL_ALIAS}"
echo "  Next:    ./2_start_tensorfold.sh"
echo "============================================================"
