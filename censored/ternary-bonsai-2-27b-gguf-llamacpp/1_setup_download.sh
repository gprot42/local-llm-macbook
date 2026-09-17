#!/usr/bin/env bash
# 1_setup_download.sh — Build PrismML's llama.cpp fork (Metal) + fetch the
# Ternary Bonsai 2 27B GGUF, so it can be served as an OpenAI API for Kilo/OpenCode.
#
# Why the fork: Bonsai 2 is a Hadamard-rotated ternary pack. Its PQ2_0 (group-128)
# GGUF needs the fork's custom kernels; stock llama.cpp / Ollama and the MLX
# servers cannot load it correctly (PrismML's own tooling refuses those paths).
# Source of truth: https://github.com/PrismML-Eng/Bonsai-demo
#
# Usage:
#   ./1_setup_download.sh              # clone+build fork, download PQ2_0 + mmproj
#   ./1_setup_download.sh --build-only # skip the model download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${SCRIPT_DIR}/engine"          # PrismML llama.cpp fork checkout + build
MODELS_DIR="${SCRIPT_DIR}/models"
VENV_DIR="${SCRIPT_DIR}/venv"              # just for huggingface_hub (downloads)
FORK_URL="https://github.com/PrismML-Eng/llama.cpp.git"
FORK_REF="${BONSAI_FORK_REF:-prism-b10683-d8f26ee}"   # matches the PQ2_0 kernels
HF_REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
# PQ2_0 (7.21 GB, fork group-128) + Q8_0 vision projector (0.63 GB).
GGUF_GLOB="Ternary-Bonsai-2-27B-PQ2_0.gguf"
MMPROJ_GLOB="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"

BUILD_ONLY=false
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=true

echo "=== Ternary Bonsai 2 27B (GGUF / llama.cpp fork) setup ==="
for tool in git cmake clang; do command -v "$tool" >/dev/null || { echo "ERROR: $tool missing" >&2; exit 1; }; done

# ── 1. Clone the fork at the pinned ref ───────────────────────────────────────
if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
  echo "→ Cloning ${FORK_URL} @ ${FORK_REF}"
  git clone --filter=blob:none "${FORK_URL}" "${ENGINE_DIR}"
  git -C "${ENGINE_DIR}" checkout "${FORK_REF}"
else
  echo "→ Fork already cloned at ${ENGINE_DIR}"
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
  echo "✅ Fork built (no model pull). Re-run without --build-only to fetch weights."
  exit 0
fi

# ── 3. Download the GGUF + mmproj ─────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then python3 -m venv "${VENV_DIR}"; fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python3 -m pip install -q --upgrade "huggingface_hub[cli]" >/dev/null
mkdir -p "${MODELS_DIR}"
echo "→ Downloading ${GGUF_GLOB} (~7.2 GB) + ${MMPROJ_GLOB} (~0.63 GB)"
hf download "${HF_REPO}" "${GGUF_GLOB}" "${MMPROJ_GLOB}" --local-dir "${MODELS_DIR}" >/dev/null

echo "✅ Ready. Start the server with: ./2_start_llama.sh"
