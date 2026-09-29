#!/usr/bin/env python3
"""Idempotent patch: close BatchGenerator on MTP errors (prevents hung server)."""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# gemma4-server: close batch_gen on MTP error"
OLD = """                active.clear()
                batch_gen = None
                mx.clear_cache()
                gc.collect()"""
NEW = f"""                active.clear()
                if batch_gen is not None:
                    try:
                        batch_gen.close()
                    except Exception:
                        pass
                    batch_gen = None
                {MARKER}
                mx.clear_cache()
                gc.collect()"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_generation.py <site-packages/mlx_vlm/server/generation.py>")
        return 2
    path = Path(sys.argv[1])
    text = path.read_text()
    if MARKER in text:
        print("   mlx_vlm/server/generation.py (batch_gen recovery — already applied)")
        return 0
    if OLD not in text:
        print(
            "ERROR: generation.py changed upstream; update patches/patch_generation.py",
            file=sys.stderr,
        )
        return 1
    path.write_text(text.replace(OLD, NEW, 1))
    print("   mlx_vlm/server/generation.py (batch_gen recovery on MTP error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())