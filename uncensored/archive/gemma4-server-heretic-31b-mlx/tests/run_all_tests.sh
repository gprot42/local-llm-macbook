#!/usr/bin/env bash
# Full lean-proxy suite. Legacy multi-file suite (test_extended, etc.) targeted
# the 10k-line proxy and is retired.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORTS_DIR="$SCRIPT_DIR/reports"
mkdir -p "$REPORTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.txt"

{
  echo "=== gemma4_mlx_kilo_proxy lean suite — $TIMESTAMP ==="
  echo ""
  "$SCRIPT_DIR/run_tests.sh" 2>&1
  echo ""
  echo "=== test_harness.py --unit-only ==="
  echo ""
  if [[ -x "$PROJ/venv/bin/python" ]]; then
    PY="$PROJ/venv/bin/python"
  else
    PY="${PYTHON:-python3}"
  fi
  "$PY" "$PROJ/test_harness.py" --unit-only 2>&1
  echo ""
  echo "=== done ==="
} | tee "$REPORT"

ln -sfn "$(basename "$REPORT")" "$REPORTS_DIR/latest.txt"
echo "Report: $REPORT"
