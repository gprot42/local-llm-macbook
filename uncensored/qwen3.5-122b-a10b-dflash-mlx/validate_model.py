#!/usr/bin/env python3
"""Refuse start if draft/target paths are incomplete."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    config_path = Path(__file__).resolve().parent / ".dflash_122b_config"
    if not config_path.exists():
        print("missing .dflash_122b_config — run 1_setup_download.sh", file=sys.stderr)
        return 1

    env: dict[str, str] = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

    target = Path(env.get("TARGET_MODEL", ""))
    draft = Path(env.get("DRAFT_MODEL", ""))
    ok = True

    if not (target / "config.json").is_file():
        print(f"target missing config: {target}", file=sys.stderr)
        ok = False
    else:
        tc = json.loads((target / "config.json").read_text())
        if tc.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
            print(f"unexpected target model_type={tc.get('model_type')}", file=sys.stderr)
            ok = False
        idx = target / "model.safetensors.index.json"
        single = target / "model.safetensors"
        if idx.is_file():
            weight_map = json.loads(idx.read_text()).get("weight_map") or {}
            missing = sorted(
                {target / fname for fname in set(weight_map.values()) if not (target / fname).is_file()}
            )
            if missing:
                print(f"target missing {len(missing)} shard(s), e.g. {missing[0]}", file=sys.stderr)
                ok = False
        elif not single.is_file():
            print(f"target has no weights under {target}", file=sys.stderr)
            ok = False

    if not (draft / "config.json").is_file() or not (draft / "model.safetensors").is_file():
        print(f"draft incomplete: {draft}", file=sys.stderr)
        ok = False
    else:
        dc = json.loads((draft / "config.json").read_text())
        if "dflash_config" not in dc:
            print("draft config missing dflash_config", file=sys.stderr)
            ok = False

    if ok:
        print("OK: target + draft look complete")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
