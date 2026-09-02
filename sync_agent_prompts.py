#!/usr/bin/env python3
"""Keep the shared `Conclude decisively` block identical in every agent prompt.

The block is inline in 18 `agent.*.prompt` strings across four `kilo.json`
files. It has to be: Kilo resolves `instructions: ["AGENTS.md"]` against the
*opened project*, not this repo, root `kilo.json` ships to
`~/.config/kilo/kilo.jsonc` where that path means someone else's project, and
two of the four stacks have no `AGENTS.md` at all. So the text cannot be
hoisted out -- but it can be generated, which is what this does.

AGENTS.md holds the canonical copy between the `kilo:conclude-decisively`
markers. This script pushes it into every prompt (default) or verifies that
none has drifted (`--check`, non-zero exit on drift -- use it in CI).

Rewrites happen on the raw file text, replacing only the JSON-escaped block, so
formatting, key order and comments elsewhere in the file are untouched.

    ./sync_agent_prompts.py           # write
    ./sync_agent_prompts.py --check   # verify only
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENTS = ROOT / "AGENTS.md"
BEGIN = "<!-- BEGIN kilo:conclude-decisively -->"
END = "<!-- END kilo:conclude-decisively -->"
HEAD = "Conclude decisively (critical):"

TARGETS = [
    Path("kilo.json"),
    Path("censored/deepseek-v4-flash-ds4/kilo.json"),
    Path("censored/gemma4-server-atomicchat-mlx-31b-2026-07-15/kilo.json"),
    Path("censored/muse-glimmer-30b-mlx/kilo.json"),
]


def canonical() -> str:
    """The block as it must appear inside a prompt string."""
    text = AGENTS.read_text(encoding="utf-8")
    try:
        body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    except IndexError:
        sys.exit(f"ERROR: {AGENTS} is missing the kilo:conclude-decisively markers")
    m = re.search(r"```[a-z]*\n(.*?)\n```", body, re.S)
    if not m:
        sys.exit(f"ERROR: no fenced block between the markers in {AGENTS}")
    block = m.group(1).strip()
    if not block.startswith(HEAD):
        sys.exit(f"ERROR: canonical block must start with {HEAD!r}")
    return block


def blocks_in(prompt: str) -> list[str]:
    """Every `Conclude decisively` block in one prompt (normally 0 or 1)."""
    out = []
    for m in re.finditer(re.escape(HEAD), prompt):
        rest = prompt[m.start():]
        end = re.search(r"\n\n(?!-)", rest)
        out.append(rest[: end.start()] if end else rest.rstrip())
    return out


def main(argv: list[str]) -> int:
    check = "--check" in argv[1:]
    want = canonical()
    esc_want = json.dumps(want)[1:-1]
    drifted: list[str] = []
    changed: list[str] = []
    total = 0

    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            sys.exit(f"ERROR: {rel} not found")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        stale: list[str] = []
        for name, agent in sorted((data.get("agent") or {}).items()):
            for block in blocks_in(agent.get("prompt") or ""):
                total += 1
                if block != want:
                    stale.append(f"{rel}::{name}")
        if not stale:
            continue
        drifted.extend(stale)
        if check:
            continue
        new = raw
        for block in {b for n, a in (data.get("agent") or {}).items()
                      for b in blocks_in(a.get("prompt") or "") if b != want}:
            new = new.replace(json.dumps(block)[1:-1], esc_want)
        json.loads(new)  # never write a file we just broke
        if new != raw:
            path.write_text(new, encoding="utf-8")
            changed.append(str(rel))

    if check:
        if drifted:
            print(f"DRIFT: {len(drifted)} of {total} copies differ from AGENTS.md")
            for w in drifted:
                print("  -", w)
            print("Run ./sync_agent_prompts.py to fix.")
            return 1
        print(f"ok: {total} copies match the canonical block in AGENTS.md")
        return 0

    if changed:
        print(f"synced {len(drifted)} of {total} copies in {len(changed)} file(s):")
        for f in changed:
            print("  -", f)
    else:
        print(f"ok: {total} copies already match the canonical block in AGENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
