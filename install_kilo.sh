#!/usr/bin/env bash
# Install monorepo root kilo.json → ~/.config/kilo/kilo.jsonc
#
# Source of truth for *global* Kilo (all providers + default model + agent harness):
#   local-llm-macbook/kilo.json
#
# Stack folders (e.g. censored/gemma4-server-atomicchat-.../kilo.json) are
# project overrides when you launch Kilo from that directory. When you change
# harness prompts or the default model for everyone, edit the *root* file and
# run this script (or copy manually).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/kilo.json"
DEST_DIR="${HOME}/.config/kilo"
DEST="$DEST_DIR/kilo.jsonc"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 1
fi

python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$SRC"

mkdir -p "$DEST_DIR"
if [[ -f "$DEST" ]]; then
  bak="$DEST.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$DEST" "$bak"
  echo "→ Backed up previous global → $bak"
fi

cp "$SRC" "$DEST"
echo "→ Installed global Kilo config:"
echo "   $SRC"
echo "   → $DEST"
python3 -c "
import json
d=json.load(open('$DEST'))
print('   default model:', d.get('model'))
print('   providers:', ', '.join(sorted(d.get('provider', {}))))
"
