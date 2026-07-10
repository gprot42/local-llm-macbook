#!/usr/bin/env bash
# =============================================================================
# install-opencode-json.sh — Sync Ornith-1.0-35B Ollama settings into OpenCode config
#
# Merges the ornith provider from ./opencode.json into
# ~/.config/opencode/opencode.json (preserves your other providers).
#
# Usage:
#   ./install-opencode-json.sh              # merge provider (backs up existing)
#   ./install-opencode-json.sh --force      # merge without prompt
#   ./install-opencode-json.sh --launch     # merge, then open OpenCode Desktop
#   ./install-opencode-json.sh --help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/opencode.json"
DEST_DIR="${HOME}/.config/opencode"
DEST="${DEST_DIR}/opencode.json"
FORCE=false
LAUNCH=false
PROVIDER_ID="ornith"

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --launch) LAUNCH=true ;;
        --help|-h)
            sed -n '3,12p' "$0" | sed 's/^# \?//'
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

dest = {}
if dest_path.exists():
    dest = json.loads(dest_path.read_text(encoding="utf-8"))

dest.setdefault("provider", {})
dest["provider"][provider_id] = source["provider"][provider_id]

merge_keys = (
    "$schema",
    "model",
    "small_model",
    "default_agent",
    "shell",
    "permission",
    "lsp",
    "formatter",
    "compaction",
    "skills",
    "agent",
)
for key in merge_keys:
    if key in source:
        dest[key] = source[key]

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
echo "  Start the server before using OpenCode:"
echo "    ./2_start_ollama.sh"
echo ""

if [[ "${LAUNCH}" == true ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        for app_name in "OpenCode" "OpenCode Desktop"; do
            if open -a "${app_name}" 2>/dev/null; then
                echo "→ Launched ${app_name}"
                exit 0
            fi
        done
        echo "→ OpenCode Desktop not found — install from https://opencode.ai"
    else
        if command -v opencode >/dev/null 2>&1; then
            opencode &
            echo "→ Launched opencode CLI"
        else
            echo "→ OpenCode CLI not found in PATH"
        fi
    fi
fi