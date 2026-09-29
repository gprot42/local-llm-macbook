#!/usr/bin/env bash
# =============================================================================
# install-opencode-json.sh — Point OpenCode at the local TensorFold Gemma 4 server
#
# Merges the gemma4-tensorfold provider from ./opencode.json into
# ~/.config/opencode/opencode.json. Other providers stay. Sets model and
# small_model to gemma4-tensorfold/gemma-4-26b-a4b-heretic-tensorfold. Does not touch agent
# prompts or permissions.
#
# Usage:
#   ./install-opencode-json.sh              # merge (asks first if a config exists)
#   ./install-opencode-json.sh --force      # merge without a prompt
#   ./install-opencode-json.sh --launch     # merge, then open OpenCode
#   ./install-opencode-json.sh --help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/opencode.json"
DEST_DIR="${HOME}/.config/opencode"
DEST="${DEST_DIR}/opencode.json"
FORCE=false
LAUNCH=false
PROVIDER_ID="gemma4-tensorfold"

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --launch) LAUNCH=true ;;
        --help|-h)
            sed -n '3,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '${arg}'. Use --help."
            exit 1
            ;;
    esac
done

if [[ ! -f "${SOURCE}" ]]; then
    echo "ERROR: source config not found: ${SOURCE}"
    exit 1
fi

echo "→ Validating ${SOURCE}..."
if ! python3 -m json.tool "${SOURCE}" >/dev/null; then
    echo "ERROR: ${SOURCE} is not valid JSON"
    exit 1
fi

mkdir -p "${DEST_DIR}"

backup_config() {
    local backup="${DEST}.bak.$(date +%Y%m%d-%H%M%S)"
    if [[ -f "${DEST}" ]]; then
        cp "${DEST}" "${backup}"
        echo "→ Backed up existing config to ${backup}"
    fi
}

if [[ -f "${DEST}" ]]; then
    if [[ "${FORCE}" == true ]]; then
        backup_config
    else
        echo "→ Existing OpenCode config: ${DEST}"
        read -r -p "  Back up and merge ${PROVIDER_ID} provider? [y/N] " reply
        case "${reply}" in
            [yY]|[yY][eE][sS])
                backup_config
                ;;
            *)
                echo "Aborted."
                exit 0
                ;;
        esac
    fi
fi

python3 - "${SOURCE}" "${DEST}" "${PROVIDER_ID}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
dest_path = Path(sys.argv[2])
provider_id = sys.argv[3]
source = json.loads(source_path.read_text(encoding="utf-8"))

if dest_path.exists():
    dest = json.loads(dest_path.read_text(encoding="utf-8"))
else:
    dest = {}

dest.setdefault("provider", {})
dest["provider"][provider_id] = source["provider"][provider_id]
dest["model"] = source["model"]
if "small_model" in source:
    dest["small_model"] = source["small_model"]
if "$schema" in source and "$schema" not in dest:
    dest["$schema"] = source["$schema"]

dest_path.write_text(json.dumps(dest, indent=2) + "\n", encoding="utf-8")
dest_path.chmod(0o600)
PY

default_model="$(python3 -c "import json; print(json.load(open('${SOURCE}'))['model'])")"
base_url="$(python3 -c "import json; print(json.load(open('${SOURCE}'))['provider']['${PROVIDER_ID}']['options']['baseURL'])")"

echo ""
echo "=== OpenCode config updated ==="
echo "→ Source:  ${SOURCE}"
echo "→ Dest:    ${DEST}"
echo "→ Model:   ${default_model}"
echo "→ API:     ${base_url}"
echo ""
echo "  The server must be running:"
echo "    ./2_start_tensorfold.sh"
echo "  Restart OpenCode if it is already open so it reloads this file."
echo ""

if [[ "${LAUNCH}" == true ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        for app_name in "OpenCode" "OpenCode Desktop"; do
            if open -a "${app_name}" 2>/dev/null; then
                echo "→ Launched ${app_name}"
                exit 0
            fi
        done
        echo "→ OpenCode Desktop not found. Install from https://opencode.ai"
    elif command -v opencode >/dev/null 2>&1; then
        opencode &
        echo "→ Launched opencode CLI"
    else
        echo "→ OpenCode CLI not found in PATH"
    fi
fi
