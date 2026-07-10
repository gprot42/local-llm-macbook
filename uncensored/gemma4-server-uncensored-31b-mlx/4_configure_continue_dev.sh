#!/usr/bin/env bash
# 4_configure_continue_dev.sh — Configure Continue.dev for Gemma 4 31B Heretic.
#
# Continue.dev does not have Kilo Code's "accomplish the task" system prompt,
# so "review and suggest" tasks work correctly — the model reads and writes a
# suggestions list without auto-implementing anything.
#
# Usage:
#   ./4_configure_continue_dev.sh             # public API on :8080 (matches default proxy mode)
#   ./4_configure_continue_dev.sh --proxy     # same as default (explicit)
#   ./4_configure_continue_dev.sh --direct    # label as direct (same URL if server used --no-proxy)
#   ./4_configure_continue_dev.sh --port 8080 # custom public port (default 8080)
#   ./4_configure_continue_dev.sh --install   # also install the VS Code extension

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROXY_PORT=8080
# Matches 2_start_mlx.sh default (proxy on public :8080).
USE_PROXY=true
INSTALL_EXT=false
MODEL_NAME="gemma-4-31b-heretic-mlx-4bit"

for arg in "$@"; do
    [[ "$arg" == "--proxy" ]]   && USE_PROXY=true
    [[ "$arg" == "--direct" ]]  && USE_PROXY=false
    [[ "$arg" == "--install" ]] && INSTALL_EXT=true
    if [[ "$arg" =~ ^--port=(.+) ]]; then PROXY_PORT="${BASH_REMATCH[1]}"; fi
done
if [[ "$*" =~ --port[[:space:]]+([0-9]+) ]]; then PROXY_PORT="${BASH_REMATCH[1]}"; fi

# Public base URL is always the client-facing port (proxy or raw vllm-mlx).
API_BASE="http://localhost:${PROXY_PORT}/v1"
if [[ "$USE_PROXY" == "true" ]]; then
    echo "→ Using proxy endpoint: ${API_BASE} (default server mode)"
else
    echo "→ Direct MLX endpoint:  ${API_BASE} (server must be started with --no-proxy)"
fi

# ── Install VS Code / VSCodium extension ───────────────────────────────────
if [[ "$INSTALL_EXT" == "true" ]]; then
    if command -v codium &>/dev/null; then
        echo "→ Installing Continue.dev VSCodium extension..."
        codium --install-extension Continue.continue
    elif command -v code &>/dev/null; then
        echo "→ Installing Continue.dev VS Code extension..."
        code --install-extension Continue.continue
    else
        echo "⚠  Neither 'codium' nor 'code' CLI found. Install manually:"
        echo "   VSCodium → Extensions → search 'Continue' → Install"
        echo "   VS Code  → Extensions → search 'Continue' → Install"
    fi
fi

# ── Write ~/.continue/config.json ─────────────────────────────────────────
CONTINUE_DIR="$HOME/.continue"
CONFIG_FILE="$CONTINUE_DIR/config.json"
mkdir -p "$CONTINUE_DIR"

# Back up any existing config
if [[ -f "$CONFIG_FILE" ]]; then
    BACKUP="${CONFIG_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$CONFIG_FILE" "$BACKUP"
    echo "→ Existing config backed up to: $BACKUP"
fi

cat > "$CONFIG_FILE" <<JSONEOF
{
  "models": [
    {
      "title": "Gemma 4 31B Heretic Uncensored (MLX)",
      "provider": "openai",
      "model": "${MODEL_NAME}",
      "apiBase": "${API_BASE}",
      "apiKey": "none",
      "contextLength": 32768,
      "completionOptions": {
        "temperature": 0.35,
        "topP": 0.95,
        "maxTokens": 8192
      }
    }
  ],
  "tabAutocompleteModel": {
    "title": "Gemma 4 31B Heretic (MLX) — Autocomplete",
    "provider": "openai",
    "model": "${MODEL_NAME}",
    "apiBase": "${API_BASE}",
    "apiKey": "none",
    "completionOptions": {
      "temperature": 0.1,
      "topP": 0.9,
      "maxTokens": 512,
      "stop": ["\n\n", "\`\`\`"]
    }
  },
  "contextProviders": [
    { "name": "code" },
    { "name": "docs" },
    { "name": "diff" },
    { "name": "terminal" },
    { "name": "problems" },
    { "name": "folder" },
    { "name": "codebase" }
  ],
  "slashCommands": [
    {
      "name": "review",
      "description": "Review this code for reliability issues, silent failures, and UX problems",
      "prompt": "Review this code and suggest improvements for reliability, error handling, and user experience. Focus on things that could fail silently. List your suggestions as numbered items. Do not make any changes."
    },
    {
      "name": "explain",
      "description": "Explain what this code does",
      "prompt": "Explain what this code does in plain language."
    },
    {
      "name": "improve",
      "description": "Suggest improvements (does NOT auto-apply)",
      "prompt": "List specific improvements for this code. Explain why each change would help. Do not apply any changes — just provide the list."
    }
  ]
}
JSONEOF

echo ""
echo "=== Continue.dev configured ==="
echo ""
echo "  Config:  ${CONFIG_FILE}"
echo "  Model:   ${MODEL_NAME}"
echo "  API:     ${API_BASE}"
echo ""
echo "Next steps:"
echo "  1. Start the MLX server:  ./2_start_mlx.sh"
echo "  2. Open VS Code / VSCodium and reload (Cmd+Shift+P → Developer: Reload Window)"
echo "  3. The Continue panel should show 'Gemma 4 31B Heretic Uncensored (MLX)'"
echo ""
echo "Useful commands in Continue:"
echo "  /review   — suggest improvements without making changes"
echo "  /improve  — propose refactors without auto-applying"
echo "  /explain  — explain selected code"
echo ""
echo "For agent mode (auto-apply changes), use @codebase and ask Continue"
echo "to 'implement' — it will ask for confirmation before editing files."
