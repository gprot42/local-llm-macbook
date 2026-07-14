#!/usr/bin/env bash
# =============================================================================
# 3_smoke_exactness.sh — Load target+draft and validate DFlash
#
# Default: greedy exactness (DFlash text == plain target text).
# --quick: short DFlash generate only (no plain baseline; faster debug).
#
# Usage:
#   ./3_smoke_exactness.sh
#   ./3_smoke_exactness.sh --quick
#   MAX_NEW_TOKENS=32 ./3_smoke_exactness.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="$SCRIPT_DIR/.dflash_122b_config"

QUICK=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --help|-h)
            awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: run ./1_setup_download.sh first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/venv/bin/activate"

export TARGET_MODEL DRAFT_MODEL
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
export QUICK

python - <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

from dflash_mlx import DFlashGenerator
from dflash_mlx.benchmark_cli import run_one_prompt

target = os.environ["TARGET_MODEL"]
draft = os.environ["DRAFT_MODEL"]
max_new = int(os.environ.get("MAX_NEW_TOKENS", "48"))
quick = os.environ.get("QUICK", "false").lower() == "true"
prompt = "Write a quicksort in Python. Keep it short."

print(f"Loading target={target}", flush=True)
print(f"       draft={draft}", flush=True)
t0 = time.time()
runner = DFlashGenerator(target_model=target, draft_model=draft)
print(f"Loaded in {time.time() - t0:.1f}s", flush=True)
print(f"resolved target={runner.target_model_path}", flush=True)
print(f"draft layers={runner.draft.target_layer_ids} block={runner.draft.block_size}", flush=True)

prompt_tokens = runner.encode_prompt(prompt)
print(f"prompt tokens={len(prompt_tokens.tolist())}", flush=True)

t1 = time.time()
dflash = runner.generate_from_tokens(
    prompt_tokens,
    max_new_tokens=max_new,
    temperature=0.0,
    verify_mode="parallel-replay",
)
dt = time.time() - t1
metrics = dflash.metrics
print("--- DFlash ---", flush=True)
print(dflash.text[:500], flush=True)
print(json.dumps({k: metrics[k] for k in metrics if k != "profile"}, indent=2, default=str), flush=True)
print(f"wall_s={dt:.2f} gen_tokens={len(dflash.generated_tokens)}", flush=True)

if quick:
    accept = float(metrics.get("avg_acceptance_length") or 0)
    if accept <= 0 and len(dflash.generated_tokens) == 0:
        raise SystemExit("FAIL: no tokens generated")
    print("PASS: quick generate completed", flush=True)
    sys.exit(0)

plain_args = Namespace(
    max_new_tokens=max_new,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    min_tokens_to_keep=1,
)
print("--- plain target (greedy) ---", flush=True)
t2 = time.time()
plain = run_one_prompt(
    runner.target.model,
    runner.target.tokenizer,
    prompt_tokens.tolist(),
    plain_args,
)
print(plain.output_text[:500], flush=True)
print(f"plain wall_s={time.time() - t2:.2f}", flush=True)

if dflash.text != plain.output_text:
    print("FAIL: DFlash output differs from plain greedy target output.", flush=True)
    print(f"dflash_len={len(dflash.text)} plain_len={len(plain.output_text)}", flush=True)
    # show first mismatch
    for i, (a, b) in enumerate(zip(dflash.text, plain.output_text)):
        if a != b:
            print(f"first char mismatch at {i}: {a!r} vs {b!r}", flush=True)
            print(f"ctx dflash: {dflash.text[max(0,i-40):i+40]!r}", flush=True)
            print(f"ctx plain:  {plain.output_text[max(0,i-40):i+40]!r}", flush=True)
            break
    else:
        print("one is a prefix of the other", flush=True)
    raise SystemExit(1)

accept = float(metrics.get("avg_acceptance_length") or 0)
if accept <= 0:
    raise SystemExit("FAIL: DFlash did not accept any draft tokens.")

print(f"PASS: exact greedy match; avg_acceptance_length={accept:.3f}", flush=True)
PY
