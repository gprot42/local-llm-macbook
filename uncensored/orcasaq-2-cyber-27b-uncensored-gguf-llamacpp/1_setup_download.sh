#!/usr/bin/env bash
# 1_setup_download.sh — Build llama.cpp (Metal) + fetch the OrcaSAQ-2 Cyber 27B
# Uncensored GGUF, so it can be served as an OpenAI API for Kilo / OpenCode.
#
# OrcaSAQ-2 Cyber 27B Uncensored is a Qwen3.8-27B fine-tune (base:
# orcarouter/Qwen3.8-27B-Uncensored, arch qwen3_5) for defensive red teaming,
# vulnerability research, security coding and terminal workflows. It ships as a
# single ~15.7 GB GGUF, so a recent stock llama.cpp serves it — no custom fork.
#   Model card: https://huggingface.co/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF
#
# The HF repo is GATED: you must (1) be logged in (`hf auth login`) and (2) have
# been granted access on the model page before the download works.
#
# Usage:
#   ./1_setup_download.sh              # build llama.cpp + download the GGUF
#   ./1_setup_download.sh --build-only # skip the model download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${SCRIPT_DIR}/engine"          # llama.cpp checkout + build
MODELS_DIR="${SCRIPT_DIR}/models"
VENV_DIR="${SCRIPT_DIR}/venv"              # just for huggingface_hub (downloads)
# Stock llama.cpp. qwen3_5 (Qwen3.8) needs a recent build; if this ref does not
# know the architecture, point these at PrismML's fork (which does):
#   ORCA_LLAMACPP_URL=https://github.com/PrismML-Eng/llama.cpp.git ORCA_LLAMACPP_REF=prism-b10743-adfffbe
LLAMACPP_URL="${ORCA_LLAMACPP_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMACPP_REF="${ORCA_LLAMACPP_REF:-master}"
HF_REPO="orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF"
GGUF_GLOB="OrcaSAQ-2-27B-Uncensored.gguf"   # single ~15.7 GB file, text-only

BUILD_ONLY=false
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=true

echo "=== OrcaSAQ-2 Cyber 27B Uncensored (GGUF / llama.cpp) setup ==="
for tool in git cmake clang; do command -v "$tool" >/dev/null || { echo "ERROR: $tool missing" >&2; exit 1; }; done

# ── 1. Clone llama.cpp ────────────────────────────────────────────────────────
if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
  echo "→ Cloning ${LLAMACPP_URL} @ ${LLAMACPP_REF}"
  git clone --filter=blob:none "${LLAMACPP_URL}" "${ENGINE_DIR}"
  git -C "${ENGINE_DIR}" checkout "${LLAMACPP_REF}"
else
  echo "→ llama.cpp already cloned at ${ENGINE_DIR}"
fi

# ── 2. Build llama-server with Metal ──────────────────────────────────────────
BIN="${ENGINE_DIR}/build/bin/llama-server"
if [[ ! -x "${BIN}" ]]; then
  echo "→ Building llama-server (Metal) — this takes a few minutes"
  cmake -B "${ENGINE_DIR}/build" -S "${ENGINE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
  cmake --build "${ENGINE_DIR}/build" --config Release -j "$(sysctl -n hw.ncpu)" --target llama-server
else
  echo "→ llama-server already built: ${BIN}"
fi
"${BIN}" --version 2>&1 | head -2 || true

if [[ "${BUILD_ONLY}" == true ]]; then
  echo "✅ llama.cpp built (no model pull). Re-run without --build-only to fetch weights."
  exit 0
fi

# ── 3. Download the GGUF (gated) ──────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then python3 -m venv "${VENV_DIR}"; fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python3 -m pip install -q --upgrade "huggingface_hub[cli]" >/dev/null
mkdir -p "${MODELS_DIR}"

# Preflight: the repo is gated, so a missing login or ungranted access fails the
# download with a 401/403 buried in a stack trace. Catch it early with a clear
# message instead.
if ! hf auth whoami >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: not logged in to Hugging Face, and ${HF_REPO} is a gated repo.
  1. Create a token:   https://huggingface.co/settings/tokens
  2. Log in:           hf auth login
  3. Request access:   https://huggingface.co/${HF_REPO}  (click "Agree and access")
Then re-run ./1_setup_download.sh
EOF
  exit 1
fi

echo "→ Downloading ${GGUF_GLOB} (~15.7 GB) from ${HF_REPO}"
if ! hf download "${HF_REPO}" "${GGUF_GLOB}" --local-dir "${MODELS_DIR}" >/dev/null; then
  cat >&2 <<EOF
ERROR: download failed. If this is a 403/gated error, you are logged in but have
not been granted access yet: open https://huggingface.co/${HF_REPO} and click
"Agree and access repository", wait for approval, then re-run this script.
EOF
  exit 1
fi

echo "✅ Ready. Start the server with: ./2_start_llama.sh"
