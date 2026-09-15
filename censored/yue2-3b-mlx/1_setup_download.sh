#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Install yue2-mlx (lyra) and prepare YuE2-3B + YuE2-Vae
#
# Apple Silicon music generation. Official CUDA YuE2 is not used here; the
# engine is daig/yue2-mlx (Python package lyra-yue2, CLI `lyra`).
#
# Default: BF16 conversion of m-a-p/YuE2-3B plus the listening decoder
# m-a-p/YuE2-Vae (~7.8 GB download, ~15 GB on disk after conversion).
#
# Precision (first arg):
#   bf16 | 3b | auto     default — MVP path
#   8bit | 4bit          experimental AR quant (still keeps BF16 NAR/VAE)
#   --deps-only          clone engine + uv sync (no weight pull / convert)
#
# Env:
#   YUE2_MLX_REPO   engine git URL (default https://github.com/daig/yue2-mlx.git)
#   YUE2_MLX_REF    pinned commit (default c0f0229df07daba14923627e9d78e084d82b7dc8)
#   YUE2_PRECISION  bf16|8bit|4bit
#   YUE2_ALIAS      API model id (default yue2-3b-mlx)
#
# Usage:
#   ./1_setup_download.sh
#   ./1_setup_download.sh bf16
#   ./1_setup_download.sh --deps-only
#   ./1_setup_download.sh 8bit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_yue2_common.sh"

PRECISION_CHOICE="${DEFAULT_PRECISION}"
DEPS_ONLY=false

for arg in "$@"; do
    case "${arg}" in
        --help|-h)
            sed -n '3,27p' "$0" | sed 's/^# //;s/^#//'
            exit 0
            ;;
        --deps-only|deps-only|--skip-download) DEPS_ONLY=true ;;
        --force|--force-download) ;;
        --*)
            echo "ERROR: unknown option '${arg}'" >&2
            exit 1
            ;;
        bf16|3b|auto|mlx) PRECISION_CHOICE="bf16" ;;
        8bit|4bit) PRECISION_CHOICE="${arg}" ;;
        *)
            echo "ERROR: unknown precision '${arg}'. Use bf16 | 8bit | 4bit | --deps-only" >&2
            exit 1
            ;;
    esac
done

echo "=== YuE2-3B MLX (lyra) — Setup ==="
echo "→ Engine:     ${ENGINE_REPO}"
echo "→ Pin:        ${ENGINE_REF}"
echo "→ Precision:  ${PRECISION_CHOICE}"
if [[ "${DEPS_ONLY}" == true ]]; then
    echo "→ Mode:       deps-only (skip prepare)"
fi
echo ""

ARCH="$(uname -m)"
if [[ "${ARCH}" != "arm64" ]]; then
    echo "ERROR: YuE2 MLX is Apple Silicon only (found ${ARCH})." >&2
    exit 1
fi
echo "→ Arch:       ${ARCH}"

SW_VERS="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "→ macOS:      ${SW_VERS}"
# yue2-mlx documents macOS 26.2+; warn but do not hard-fail on older builds.
if [[ "${SW_VERS}" =~ ^([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
    if [[ "${major}" -lt 26 ]]; then
        echo "WARNING: yue2-mlx is tested on macOS 26.2+. This is ${SW_VERS}."
    fi
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required (yue2-mlx pins Python 3.12 via uv)." >&2
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  Or:      brew install uv" >&2
    exit 1
fi
echo "→ uv:         $(uv --version 2>/dev/null || echo installed)"

echo ""
echo "→ Ensuring Python 3.12 for uv ..."
uv python install 3.12 >/dev/null

if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
    echo "→ Cloning yue2-mlx into ${ENGINE_DIR} ..."
    git clone "${ENGINE_REPO}" "${ENGINE_DIR}"
fi
echo "→ Fetching pin ${ENGINE_REF} ..."
git -C "${ENGINE_DIR}" fetch --force origin "${ENGINE_REF}" 2>/dev/null \
    || git -C "${ENGINE_DIR}" fetch --tags --force origin 2>/dev/null \
    || true
git -C "${ENGINE_DIR}" checkout --detach "${ENGINE_REF}"
ENGINE_SHA="$(git -C "${ENGINE_DIR}" rev-parse HEAD)"
echo "→ Engine SHA: ${ENGINE_SHA}"

echo ""
echo "→ uv sync --frozen (locked lyra-yue2 environment) ..."
(
    cd "${ENGINE_DIR}"
    uv sync --frozen
)

if [[ "${DEPS_ONLY}" == true ]]; then
    echo ""
    echo "✅  Engine ready (no weight prepare)."
    echo "  Re-run without --deps-only to download and convert YuE2-3B + YuE2-Vae."
    exit 0
fi

mkdir -p "${HF_CACHE_DIR}" "${CONVERTED_DIR}" "${OUTPUTS_DIR}"

AVAILABLE_GB="$(df -g "${STACK_DIR}" | awk 'NR==2 {print $4}')"
if [[ -n "${AVAILABLE_GB}" && "${AVAILABLE_GB}" -lt 20 ]]; then
    echo "WARNING: only ${AVAILABLE_GB} GB free. BF16 setup wants ~20 GB (weights + conversion)."
fi

echo ""
echo "→ Preparing generator + VAE (lyra prepare, precision=${PRECISION_CHOICE}) ..."
echo "  This downloads pinned HF snapshots and converts the generator. First run is slow."
prepare_lyra_env
PREPARE_LOG="$(mktemp -t yue2-prepare.XXXXXX)"
set +e
(
    cd "${ENGINE_DIR}"
    uv run lyra prepare \
        --cache-dir "${HF_CACHE_DIR}" \
        --output "${CONVERTED_DIR}" \
        --precision "${PRECISION_CHOICE}"
) | tee "${PREPARE_LOG}"
PREPARE_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREPARE_RC}" -ne 0 ]]; then
    echo "ERROR: lyra prepare failed (exit ${PREPARE_RC}). See ${PREPARE_LOG}" >&2
    exit "${PREPARE_RC}"
fi

python3 - "${PREPARE_LOG}" "${PATHS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8")
payload = None
try:
    payload = json.loads(text)
except json.JSONDecodeError:
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and "vae" in obj:
            payload = obj
        idx = end
if not isinstance(payload, dict) or "vae" not in payload:
    sys.stderr.write("ERROR: lyra prepare did not print a JSON object with a 'vae' path\n")
    sys.exit(1)
if "model" not in payload:
    payload["model"] = str(out_path.parent / "converted")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"→ Wrote {out_path}")
PY

VAE_PATH="$(python3 -c "import json; print(json.load(open('${PATHS_FILE}'))['vae'])")"
MODEL_DIR="$(python3 -c "import json, pathlib; p=json.load(open('${PATHS_FILE}')); print(p.get('model') or '${CONVERTED_DIR}')")"
MODEL_ALIAS="${YUE2_ALIAS:-${DEFAULT_ALIAS}}"

echo ""
python3 "${VALIDATE_MODEL}" "${MODEL_DIR}" --vae "${VAE_PATH}" --paths "${PATHS_FILE}"

cat > "${CONFIG_FILE}" << EOF
# Written by 1_setup_download.sh — do not edit manually
ENGINE_DIR="${ENGINE_DIR}"
ENGINE_REPO="${ENGINE_REPO}"
ENGINE_REF="${ENGINE_SHA}"
PRECISION="${PRECISION_CHOICE}"
MODEL_DIR="${MODEL_DIR}"
VAE_PATH="${VAE_PATH}"
HF_CACHE_DIR="${HF_CACHE_DIR}"
PATHS_FILE="${PATHS_FILE}"
MODEL_ALIAS="${MODEL_ALIAS}"
EOF

echo ""
echo "✅  YuE2-3B MLX ready"
echo "  Model:     ${MODEL_DIR}"
echo "  VAE:       ${VAE_PATH}"
echo "  Precision: ${PRECISION_CHOICE}"
echo "  Alias:     ${MODEL_ALIAS}"
echo ""
echo "  Next:"
echo "    ./2_generate.sh                      # ~16s piano-pop clip (quickstart)"
echo "    ./2_generate.sh examples/full-song.json"
echo "    ./2_start_server.sh                  # HTTP API on :${DEFAULT_PORT}"
echo "    python3 test_harness.py --offline"
