#!/usr/bin/env bash
# Quick pure-function suite (no server, no network).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ"

if [[ -x "$PROJ/venv/bin/python" ]]; then
  PY="$PROJ/venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if "$PY" -c "import pytest" 2>/dev/null; then
  exec "$PY" -m pytest tests/test_pure_functions.py -q "$@"
fi
# Fallback: self-runner baked into the test module
exec "$PY" tests/test_pure_functions.py
