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

# Last matching rule wins. A later "deny" (or a scalar bash policy without
# the allow-list) would overwrite the global permission we ship here.
python3 - "$SRC" <<'PY'
import json, sys

path = sys.argv[1]
data = json.load(open(path))
perm = data.get("permission")
if not isinstance(perm, dict):
    raise SystemExit(f"ERROR: {path} is missing permission object")

def reject_deny(value, loc="permission"):
    if value == "deny":
        raise SystemExit(f"ERROR: {path} has deny at {loc} (not allowed)")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_deny(child, f"{loc}.{key}")

reject_deny(perm)

bash = perm.get("bash")
if not isinstance(bash, dict):
    raise SystemExit(
        f"ERROR: {path} permission.bash must be a pattern map "
        "with '*': 'ask' and explicit allow rules (not a scalar)"
    )
required = {
    "*": "ask",
    "ls *": "allow",
    "pwd *": "allow",
    "cat *": "allow",
    "head *": "allow",
    "tail *": "allow",
    "adb *": "allow",
}
for pattern, action in required.items():
    got = bash.get(pattern)
    if got != action:
        raise SystemExit(
            f"ERROR: {path} permission.bash[{pattern!r}] must be {action!r}, got {got!r}"
        )

if data.get("default_agent") != "code":
    raise SystemExit(f"ERROR: {path} default_agent must be 'code' (Ask/Plan deny adb)")

for name in ("build", "code", "debug"):
    agent = (data.get("agent") or {}).get(name) or {}
    agent_bash = ((agent.get("permission") or {}).get("bash"))
    if not isinstance(agent_bash, dict) or agent_bash.get("adb *") != "allow":
        raise SystemExit(
            f"ERROR: {path} agent.{name}.permission.bash['adb *'] must be 'allow'"
        )

# Every stack folder that ships its own kilo.json must stay reachable from the
# global config too, otherwise Kilo shows "cannot connect to api server" when
# launched outside that folder. provider id -> (baseURL, model id).
required_providers = {
    "ds4": ("http://127.0.0.1:8083/v1", "deepseek-v4-flash"),
    "mtplx": ("http://localhost:8765/v1", "qwen3.6-27b-mtplx"),
    "mtplx-qwen38-obl": ("http://127.0.0.1:8768/v1", "qwen3.8-27b-obliterated-mtplx"),
    "muse-glimmer": ("http://127.0.0.1:8087/v1", "muse-glimmer-30b-mlx"),
}
providers = data.get("provider") or {}
for pid, (base_url, model_id) in required_providers.items():
    prov = providers.get(pid)
    if not isinstance(prov, dict):
        raise SystemExit(f"ERROR: {path} provider.{pid} is missing")
    got_url = (prov.get("options") or {}).get("baseURL")
    if got_url != base_url:
        raise SystemExit(
            f"ERROR: {path} provider.{pid}.options.baseURL must be {base_url!r}, got {got_url!r}"
        )
    if model_id not in (prov.get("models") or {}):
        raise SystemExit(f"ERROR: {path} provider.{pid}.models is missing {model_id!r}")

model = data.get("model") or ""
if "/" not in model or model.split("/", 1)[0] not in providers:
    raise SystemExit(f"ERROR: {path} model {model!r} does not reference a defined provider")
PY

# The shared `Conclude decisively` block is inline in 18 agent prompts across
# four kilo.json files (Kilo's `instructions` cannot reach them -- see
# AGENTS.md). Refuse to install a root config whose copy has drifted.
if [[ -x "$ROOT/sync_agent_prompts.py" ]]; then
  "$ROOT/sync_agent_prompts.py" --check
fi

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
