#!/usr/bin/env bash
# apply_local_patches.sh — Copy project patches into the active venv.
# Safe to re-run; called by 1_setup_download.sh and 2_start_mlx.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY=""
for cand in "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/venv/bin/python3"; do
    if [[ -x "$cand" ]]; then
        VENV_PY="$cand"
        break
    fi
done
if [[ -z "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first." >&2
    exit 1
fi

SITE="$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "→ Applying local patches to $SITE ..."

# ── deepseek_v4: unfused SwitchGLU for 2bit-DQ (mixed group_size gate/up) ──
if [[ -f patches/deepseek_v4.py && -d "$SITE/mlx_lm/models" ]]; then
    cp -f patches/deepseek_v4.py "$SITE/mlx_lm/models/deepseek_v4.py"
    echo "   mlx_lm/models/deepseek_v4.py (SwitchGLU / no gate-up fuse for 2bit-DQ)"
fi

# ── server: wait for model load before opening HTTP port ──────────────────
if [[ -f patches/server.py && -d "$SITE/mlx_lm" ]]; then
    cp -f patches/server.py "$SITE/mlx_lm/server.py"
    echo "   mlx_lm/server.py (open :port only after model load — avoids Metal GPU timeouts)"
fi

# ── tokenizer_utils: transformers 5.13+ AutoTokenizer.register compat ──────
"$VENV_PY" - <<'PY'
from pathlib import Path
import re
import site

MARKER = "GROK_PATCH: transformers 5.13+ NewlineTokenizer register compat"
CLEAN = f"""# {MARKER}
try:
    AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)
except (AttributeError, TypeError, ValueError):
    pass
"""
BLOCK_RE = re.compile(
    r"(?:# [^\n]*NewlineTokenizer[^\n]*\n)?"
    r"(?:try:\n(?:[ \t]*# [^\n]*\n)?)*"
    r"[ \t]*AutoTokenizer\.register\(\"NewlineTokenizer\", fast_tokenizer_class=NewlineTokenizer\)\n"
    r"(?:except \(AttributeError, TypeError, ValueError\):\n[ \t]+pass\n)*",
    re.M,
)
for sp in site.getsitepackages():
    path = Path(sp) / "mlx_lm" / "tokenizer_utils.py"
    if not path.is_file():
        continue
    text = path.read_text()
    if MARKER in text:
        print(f"   mlx_lm/tokenizer_utils.py (already patched)")
        break
    if "AutoTokenizer.register(\"NewlineTokenizer\"" not in text:
        break
    new_text, n = BLOCK_RE.subn(CLEAN + "\n", text, count=1)
    if n:
        path.write_text(new_text)
        print(f"   mlx_lm/tokenizer_utils.py (transformers 5.13+ register compat)")
    break
PY

echo "→ Patches applied."
