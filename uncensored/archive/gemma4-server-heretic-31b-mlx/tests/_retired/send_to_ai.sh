#!/usr/bin/env bash
# send_to_ai.sh
# ─────────────────────────────────────────────────────────────────────────────
# Prepare a ready-to-paste message for Claude Sonnet from the latest test
# report.  Copies to the macOS clipboard (pbcopy) and also prints a preview.
#
# Usage:
#   ./tests/send_to_ai.sh                   # use latest report
#   ./tests/send_to_ai.sh --report FILE     # use a specific report
#   ./tests/send_to_ai.sh --mode triage     # short triage prompt (default)
#   ./tests/send_to_ai.sh --mode fix        # ask for concrete code patches
#   ./tests/send_to_ai.sh --mode diff       # ask for git-diff patches only
#   ./tests/send_to_ai.sh --no-copy         # print only, don't copy
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="$SCRIPT_DIR/reports"
REPORT=""
MODE="triage"
NO_COPY=0

for arg in "$@"; do
  case "$arg" in
    --report) shift; REPORT="$1" ;;
    --mode)   shift; MODE="$1"   ;;
    --no-copy) NO_COPY=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
  esac
done

# ── Find report ───────────────────────────────────────────────────────────────
if [[ -z "$REPORT" ]]; then
  REPORT="$REPORTS_DIR/latest.txt"
fi

if [[ ! -f "$REPORT" ]]; then
  echo "ERROR: no report found at $REPORT" >&2
  echo "Run ./tests/run_50_tests.sh first to generate one." >&2
  exit 1
fi

REPORT_SIZE=$(wc -c < "$REPORT")
REPORT_LINES=$(wc -l < "$REPORT")
echo "Report : $REPORT"
echo "Size   : $REPORT_SIZE bytes / $REPORT_LINES lines"

# ── Extract only the failures section if the report is large ─────────────────
MAX_BYTES=180000   # stay comfortably under Claude's context window
CONTENT=""

if [[ $REPORT_SIZE -gt $MAX_BYTES ]]; then
  echo "Report > ${MAX_BYTES}B — extracting FAILURES section only..."
  # Extract from '=== FAILURES ===' to '=== short test summary ===' or end
  CONTENT=$(awk '/^=+ FAILURES =+/{found=1} found{print} /^=+ short test summary =+/{exit}' "$REPORT")
  if [[ -z "$CONTENT" ]]; then
    # Fallback: take the last 1000 lines
    CONTENT=$(tail -1000 "$REPORT")
  fi
else
  CONTENT=$(cat "$REPORT")
fi

# ── Choose prompt ─────────────────────────────────────────────────────────────
case "$MODE" in
  triage)
    PROMPT="Summarise the test results above. Group the failures by likely root cause and rank them by severity. List each failing test, what it tests, and the one-line fix you'd recommend. Which failure should I address first?"
    ;;
  fix)
    PROMPT="For each FAILED test above: (a) identify the root cause in gemma4_mlx_kilo_proxy.py, (b) quote the relevant existing code block, (c) show the exact replacement code. Be specific — include file line numbers where possible."
    ;;
  diff)
    PROMPT="For each FAILED test above, output a minimal unified diff (git diff format) that fixes gemma4_mlx_kilo_proxy.py. Output diffs only — no explanations. If multiple tests share one root cause, produce a single diff that fixes all of them."
    ;;
  *)
    PROMPT="$MODE"   # treat --mode as a raw prompt string
    ;;
esac

# ── Assemble final message ────────────────────────────────────────────────────
CONTEXT_HEADER="You are analysing test results for gemma4_mlx_kilo_proxy.py, a local-LLM steering proxy for the Kilo Code agentic coding assistant. The proxy intercepts OpenAI-compatible streaming responses from a vllm-mlx Gemma 4 26B model and applies tool-call repair, stall detection, fuzzy string matching, and guard logic.

The test suite covers pure functions (_fuzzy_find, _repair_tool_call_args, _break_text_stall, _classify_pseudo_tool_text, etc.), ProxyConfig/GuardConfig, _ProxyMetrics, stream helper classes (StallDetector, TextModeGuard, ToolStreamState), and replay-based integration tests using _ReplayTransport + httpx.ASGITransport.

─────────────────── TEST REPORT ───────────────────
$CONTENT
────────────────────────────────────────────────────

$PROMPT"

CHAR_COUNT=${#CONTEXT_HEADER}
echo "Message: $CHAR_COUNT chars (~$((CHAR_COUNT / 4)) tokens)"

# ── Copy / print ──────────────────────────────────────────────────────────────
if [[ $NO_COPY -eq 0 ]]; then
  if command -v pbcopy &>/dev/null; then
    echo "$CONTEXT_HEADER" | pbcopy
    echo "✓ Copied to macOS clipboard (pbcopy)."
    echo ""
    echo "Next steps:"
    echo "  1. Open Claude Sonnet at https://claude.ai  (or Kilo Code chat)"
    echo "  2. Paste with Cmd+V"
    echo "  3. Press Enter"
  elif command -v xclip &>/dev/null; then
    echo "$CONTEXT_HEADER" | xclip -selection clipboard
    echo "✓ Copied to clipboard (xclip)."
  elif command -v xsel &>/dev/null; then
    echo "$CONTEXT_HEADER" | xsel --clipboard --input
    echo "✓ Copied to clipboard (xsel)."
  else
    echo "No clipboard command found (pbcopy/xclip/xsel). Printing instead:"
    echo ""
    echo "$CONTEXT_HEADER"
  fi
else
  echo "$CONTEXT_HEADER"
fi
