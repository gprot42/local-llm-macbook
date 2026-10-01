#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install TensorFold for Gemma 4 26B-A4B (abliterated, uncensored, LOCAL pack)
#
# TensorFold (https://github.com/ashhart/TensorFold) serves Gemma 4 on its
# gemma4 lane. The kernels cover the MoE checkpoints only (enable_moe_block in
# every layer) and read MLX 4-bit weights in groups of 32/64 with an 8-bit
# router. This base pack is that MoE checkpoint; the 31B dense
# Gemma 4 packs are a different architecture the lane refuses.
#
#   HF model   mlx-community/gemma-4-26b-a4b-it-4bit  (~15 GB, MoE 26B-A4B; base, censored)
#   Drafter    z-lab/gemma-4-26B-A4B-it-DFlash               (~0.8 GB, optional, DFlash)
#
# Defaults to the base (censored) pack — the lane's reference model, standard
# Gemma template, clean output. The uncensored 4-bit packs don't work here:
#   - …-heretic-4bit / supergemma …-multimodal: 8-bit MLP, the lane refuses.
#   - supergemma …-uncensored-v2: right quant but a reasoning-CHANNEL template
#     (<|channel>thought…) the lane doesn't parse, so channel markers leak.
# For clean uncensored, self-quantise an abliterated bf16 (e.g. SevenOfNine/
# Gemma-4-26B-A4B-It-Abliterated) to 4-bit with the standard Gemma template.
#
# Usage:
#   ./1_setup_download.sh                 # venv + CLI + model + drafter
#   ./1_setup_download.sh --deps-only     # venv + CLI only
#   ./1_setup_download.sh --no-drafter    # model only (serial / no DFlash2)
#   ./1_setup_download.sh org/repo        # different main checkpoint
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_FILE="${SCRIPT_DIR}/.tensorfold_config"

DEFAULT_MODEL="${HOME}/.cache/mlx-converts/gemma-4-26b-a4b-abliterated-4bit"
DEFAULT_DRAFTER="z-lab/gemma-4-26B-A4B-it-DFlash"
MODEL_ALIAS="${GEMMA4_ALIAS:-gemma-4-26b-a4b-abliterated-tensorfold}"
CONTEXT="${GEMMA4_CONTEXT:-131072}"

DEPS_ONLY=false
WITH_DRAFTER=true
MODEL="${GEMMA4_HF_MODEL:-${DEFAULT_MODEL}}"

for arg in "$@"; do
    case "${arg}" in
        --deps-only|deps-only) DEPS_ONLY=true ;;
        --no-drafter) WITH_DRAFTER=false ;;
        --help|-h)
            sed -n '3,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            MODEL="${arg}"
            ;;
    esac
done

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: TensorFold on macOS needs Apple Silicon (arm64). Got $(uname -m)."
    exit 1
fi

pick_python() {
    local candidate ver major minor
    for candidate in /opt/homebrew/bin/python3.12 python3.12 python3; do
        if ! command -v "${candidate}" >/dev/null 2>&1; then
            continue
        fi
        ver="$("${candidate}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        major="${ver%%.*}"
        minor="${ver#*.}"
        if [[ "${major}" -ge 3 && "${minor}" -ge 11 ]]; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

PY="$(pick_python)" || {
    echo "ERROR: TensorFold needs Python 3.11+. Install it with: brew install python@3.12"
    exit 1
}

echo "→ Python: ${PY} ($("${PY}" --version))"
echo "→ Model:  ${MODEL}"
if [[ "${WITH_DRAFTER}" == true ]]; then
    echo "→ Draft:  ${DEFAULT_DRAFTER}"
else
    echo "→ Draft:  off"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "→ Creating venv at ${VENV_DIR}"
    "${PY}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
# TensorFold: upstream v0.6.0 plus three fork branches that are not upstream yet
# (see findings.md, "Rebase onto v0.6.0", and tensorfold-pr-drafts.md):
#   gemma4-thought-channel-v0.6.0     thought-channel leak fixes with thinking off
#   feat-repetition-penalty-v0.6.0    repetition / frequency / presence penalties
#   feat-toolcall-repair-v0.6.0       repair or hide an unterminated / malformed tool call
#   prompt-cache-tool-turn-v0.6.0     prompt-cache checkpoint for agentic steps after a tool result
# (0.6.0 itself parses the bare <|tool_call>:NAME form, so the old colon-prefix branch is gone.)
# The combined branch is checked out in ./.tensorfold-src and installed from there; on a
# machine without it, the fork branches are merged on top of TF_BASE as before.
#   TF_REPO=owner/repo      fork to clone (default gprot42/TensorFold)
#   TF_BASE=branch          branch to start from (default gemma4-thought-channel-v0.6.0)
#   TF_BRANCHES="a b"       branches merged on top (default the two feat-*-v0.6.0 branches)
#   TF_VERSION=ref          set to a tag/branch/commit to skip the merge and install that ref
#                           straight from TF_REPO (e.g. TF_REPO=ashhart/TensorFold TF_VERSION=v0.6.0)
#   TF_LOCAL_SRC=dir        install from a local TensorFold checkout instead; wins over everything
#                           above. Default: ./.tensorfold-src when it is a git checkout (the rebased
#                           v0.6.0 combined branch lives there; see tensorfold-pr-drafts.md).
TF_LOCAL_SRC="${TF_LOCAL_SRC:-}"
if [[ -z "${TF_LOCAL_SRC}" && -d "${SCRIPT_DIR}/.tensorfold-src/.git" ]]; then
    TF_LOCAL_SRC="${SCRIPT_DIR}/.tensorfold-src"
fi
TF_REPO="${TF_REPO:-gprot42/TensorFold}"
TF_BASE="${TF_BASE:-gemma4-thought-channel-v0.6.0}"
TF_BRANCHES="${TF_BRANCHES:-feat-repetition-penalty-v0.6.0 feat-toolcall-repair-v0.6.0 prompt-cache-tool-turn-v0.6.0}"
TF_VERSION="${TF_VERSION:-}"
if [[ -n "${TF_LOCAL_SRC}" ]]; then
    echo "→ TensorFold: local checkout ${TF_LOCAL_SRC} ($(git -C "${TF_LOCAL_SRC}" log --oneline -1 2>/dev/null || echo 'not a git tree'))"
    python -m pip install --upgrade "${TF_LOCAL_SRC}"
elif [[ -n "${TF_VERSION}" ]]; then
    echo "→ TensorFold: ${TF_REPO}@${TF_VERSION} (single ref, no fork merge)"
    python -m pip install --upgrade "git+https://github.com/${TF_REPO}.git@${TF_VERSION}"
else
    TF_SRC="${SCRIPT_DIR}/.tensorfold-src"
    echo "→ TensorFold: ${TF_REPO} ${TF_BASE} + ${TF_BRANCHES} → ${TF_SRC}"
    rm -rf "${TF_SRC}"
    git clone -q --filter=blob:none "https://github.com/${TF_REPO}.git" "${TF_SRC}"
    for branch in ${TF_BASE} ${TF_BRANCHES}; do
        if ! git -C "${TF_SRC}" rev-parse -q --verify "origin/${branch}" >/dev/null; then
            echo "ERROR: branch ${branch} is not on ${TF_REPO}. The v0.6.0 branches are pushed from a machine"
            echo "       that has them in .tensorfold-src (see tensorfold-pr-drafts.md), or set TF_LOCAL_SRC."
            exit 1
        fi
    done
    git -C "${TF_SRC}" checkout -q -b combined "origin/${TF_BASE}"
    for branch in ${TF_BRANCHES}; do
        if ! git -C "${TF_SRC}" -c user.name=setup -c user.email=setup@local merge -q --no-edit "origin/${branch}"; then
            echo "ERROR: merging ${branch} onto ${TF_BASE} conflicts; resolve in ${TF_SRC} or set TF_VERSION to a single ref."
            exit 1
        fi
    done
    echo "→ Combined tree: $(git -C "${TF_SRC}" log --oneline -1)"
    python -m pip install --upgrade "${TF_SRC}"
fi

echo "→ $(tensorfold --version)"
echo ""
tensorfold models
echo ""
echo "→ Checking ${MODEL} (config.json only, no weights) ..."
if ! tensorfold info "${MODEL}"; then
    echo "ERROR: tensorfold info failed for ${MODEL}."
    echo "       This CLI only serves families it has kernels for."
    exit 1
fi

DRAFTER=""
if [[ "${WITH_DRAFTER}" == true ]]; then
    DRAFTER="${DEFAULT_DRAFTER}"
fi

cat > "${CONFIG_FILE}" <<EOF
# Generated by 1_setup_download.sh. Re-run that script to change it.
HF_MODEL=${MODEL}
DRAFTER=${DRAFTER}
MODEL_ALIAS=${MODEL_ALIAS}
CONTEXT=${CONTEXT}
EOF
echo "→ Wrote ${CONFIG_FILE}"

if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ --deps-only: skipping weight download."
    echo "  Pull later with: ./2_start_tensorfold.sh   (it resumes the download)"
    exit 0
fi

echo ""
echo "→ Downloading weights into the Hugging Face cache ..."
if [[ -d "${MODEL}" ]]; then
    echo "→ Local model dir: ${MODEL} (self-quantised; skip pull)"
    [[ -n "${DRAFTER}" ]] && tensorfold pull "${DRAFTER}"
elif [[ -n "${DRAFTER}" ]]; then
    tensorfold pull "${MODEL}" "${DRAFTER}"
else
    tensorfold pull "${MODEL}"
fi

echo ""
echo "============================================================"
echo "  SETUP DONE"
echo "============================================================"
echo "  Model:   ${MODEL}"
echo "  Drafter: ${DRAFTER:-none}"
echo "  Alias:   ${MODEL_ALIAS}"
echo "  Next:    ./2_start_tensorfold.sh"
echo "============================================================"
