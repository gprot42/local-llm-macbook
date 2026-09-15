#!/usr/bin/env bash
# =============================================================================
# 2_generate.sh — Generate or plan a YuE2 song via lyra
#
# Usage:
#   ./2_generate.sh
#   ./2_generate.sh examples/full-song.json
#   ./2_generate.sh request.json --output outputs/my-song
#   ./2_generate.sh --plan examples/quickstart.json --output outputs/plan
#   ./2_generate.sh --style "English jazz trio, 110 BPM" --lyrics $'[Verse]\nRain on brass'
#   ./2_generate.sh --abc edited.abc --mode full request.json
#
# Options:
#   --output DIR       Artifact directory (must be absent or empty)
#   --plan             Stop after ABC / symbolic plan (no audio)
#   --abc FILE         Supply a score (overrides request abc)
#   --mode MODE        full | melody | off
#   --precision P      bf16 | 8bit | 4bit (default from setup)
#   --no-require-ac    Allow battery (runtime still has a 16 GiB budget)
#   --quiet            Hide lyra progress
#   --help, -h
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_yue2_common.sh"

REQUEST=""
OUTPUT=""
DO_PLAN=false
ABC_FILE=""
MODE=""
PRECISION_OVERRIDE=""
REQUIRE_AC=true
QUIET=false
STYLE=""
LYRICS=""

i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
        --help|-h)
            sed -n '3,22p' "$0" | sed 's/^# //;s/^#//'
            exit 0
            ;;
        --output) OUTPUT="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --plan) DO_PLAN=true; ((i+=1)) ;;
        --abc) ABC_FILE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --mode|--cot) MODE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --precision) PRECISION_OVERRIDE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --no-require-ac) REQUIRE_AC=false; ((i+=1)) ;;
        --quiet) QUIET=true; ((i+=1)) ;;
        --style) STYLE="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --lyrics) LYRICS="${args[$((i+1))]:-}"; ((i+=2)) ;;
        --*)
            echo "ERROR: unknown option '${args[$i]}'" >&2
            exit 1
            ;;
        *)
            if [[ -n "${REQUEST}" ]]; then
                echo "ERROR: extra argument '${args[$i]}'" >&2
                exit 1
            fi
            REQUEST="${args[$i]}"
            ((i+=1))
            ;;
    esac
done

load_yue2_config

if [[ -n "${PRECISION_OVERRIDE}" ]]; then
    PRECISION="${PRECISION_OVERRIDE}"
fi

TMP_REQUEST=""
cleanup_tmp() {
    [[ -n "${TMP_REQUEST}" && -f "${TMP_REQUEST}" ]] && rm -f "${TMP_REQUEST}"
}
trap cleanup_tmp EXIT

if [[ -n "${STYLE}" || -n "${LYRICS}" ]]; then
    if [[ -z "${STYLE}" || -z "${LYRICS}" ]]; then
        echo "ERROR: --style and --lyrics must be used together" >&2
        exit 1
    fi
    TMP_REQUEST="$(mktemp -t yue2-request.XXXXXX.json)"
    python3 - "${TMP_REQUEST}" "${STYLE}" "${LYRICS}" "${MODE:-full}" <<'PY'
import json, sys, time
path, style, lyrics, cot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
payload = {
    "id": f"cli-{int(time.time())}",
    "style": style,
    "lyrics": lyrics,
    "cot": cot,
    "seed": 831001,
}
json.dump(payload, open(path, "w"), ensure_ascii=False, indent=2)
PY
    REQUEST="${TMP_REQUEST}"
elif [[ -z "${REQUEST}" ]]; then
    REQUEST="${EXAMPLES_DIR}/quickstart.json"
fi

if [[ ! -f "${REQUEST}" ]]; then
    echo "ERROR: request file not found: ${REQUEST}" >&2
    exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
    REQ_ID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('id') or 'song')" "${REQUEST}")"
    if [[ "${DO_PLAN}" == true ]]; then
        OUTPUT="${OUTPUTS_DIR}/${REQ_ID}-plan"
    else
        OUTPUT="${OUTPUTS_DIR}/${REQ_ID}"
    fi
fi

if [[ "${OUTPUT}" != /* ]]; then
    OUTPUT="${STACK_DIR}/${OUTPUT}"
fi

if [[ -e "${OUTPUT}" ]]; then
    echo "ERROR: output '${OUTPUT}' already exists. Lyra will not overwrite. Pick a new --output." >&2
    exit 1
fi

python3 "${VALIDATE_MODEL}" "${MODEL_DIR}" --vae "${VAE_PATH}" --paths "${PATHS_FILE}"

LYRA_ARGS=()
if [[ "${DO_PLAN}" == true ]]; then
    LYRA_ARGS+=(plan "${REQUEST}")
else
    LYRA_ARGS+=(generate "${REQUEST}")
fi
LYRA_ARGS+=(
    --model "${MODEL_DIR}"
    --vae "${VAE_PATH}"
    --precision "${PRECISION}"
    --offline
    --output "${OUTPUT}"
)
if [[ "${REQUIRE_AC}" == true ]]; then
    LYRA_ARGS+=(--require-ac)
fi
if [[ "${QUIET}" == true ]]; then
    LYRA_ARGS+=(--quiet)
fi
if [[ -n "${ABC_FILE}" ]]; then
    LYRA_ARGS+=(--abc "${ABC_FILE}")
fi
if [[ -n "${MODE}" ]]; then
    LYRA_ARGS+=(--mode "${MODE}")
fi

echo "=== YuE2-3B generate ==="
echo "→ Request:    ${REQUEST}"
echo "→ Output:     ${OUTPUT}"
echo "→ Precision:  ${PRECISION}"
echo "→ Mode:       $([ "${DO_PLAN}" == true ] && echo plan || echo generate)"
echo ""

run_lyra "${LYRA_ARGS[@]}"

echo ""
if [[ "${DO_PLAN}" == true ]]; then
    echo "✅  Plan written to ${OUTPUT}"
    [[ -f "${OUTPUT}/score.abc" ]] && echo "  Score: ${OUTPUT}/score.abc"
else
    echo "✅  Song written to ${OUTPUT}"
    [[ -f "${OUTPUT}/audio.flac" ]] && echo "  Audio: ${OUTPUT}/audio.flac"
    [[ -f "${OUTPUT}/score.abc" ]] && echo "  Score: ${OUTPUT}/score.abc"
    [[ -f "${OUTPUT}/result.json" ]] && echo "  Result: ${OUTPUT}/result.json"
    if [[ -f "${OUTPUT}/audio.flac" ]]; then
        echo "  Listen: open ${OUTPUT}/audio.flac"
    fi
fi
