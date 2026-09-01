#!/usr/bin/env python3
"""Verify a local MLX model directory has all required weights.

Exit codes:
  0 — model complete
  1 — directory exists but weights are incomplete or corrupt
  2 — directory does not exist
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def validate(model_dir: Path) -> tuple[list[str], list[str], int]:
    errors: list[str] = []

    if not model_dir.is_dir():
        return (["model directory does not exist"], [], 0)

    for name in REQUIRED:
        path = model_dir / name
        if not path.is_file():
            errors.append(f"missing required file: {name}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required file: {name}")

    index_path = model_dir / "model.safetensors.index.json"
    single_shard = model_dir / "model.safetensors"
    shards: list[str] = []
    expected_bytes = 0
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid model.safetensors.index.json: {exc}")
            index = {}
        expected_bytes = int(index.get("metadata", {}).get("total_size", 0) or 0)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            errors.append("model.safetensors.index.json has no weight shards listed")
    elif single_shard.is_file():
        if single_shard.stat().st_size == 0:
            errors.append("empty weight shard: model.safetensors")
        else:
            shards = ["model.safetensors"]
            expected_bytes = single_shard.stat().st_size
    else:
        # Some MLX packs use model-0000N-of-0000M.safetensors without an index.
        numbered = sorted(model_dir.glob("model-*-of-*.safetensors"))
        if numbered:
            shards = [p.name for p in numbered]
            expected_bytes = sum(p.stat().st_size for p in numbered)
        else:
            errors.append(
                "missing weights: need model.safetensors.index.json, "
                "model.safetensors, or model-*-of-*.safetensors"
            )

    for shard in shards:
        path = model_dir / shard
        if not path.is_file():
            errors.append(f"missing weight shard: {shard}")
        elif path.stat().st_size == 0:
            errors.append(f"empty weight shard: {shard}")

    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    if cache_dir.is_dir():
        incomplete = sorted(cache_dir.glob("*.incomplete"))
        if incomplete:
            errors.append(
                f"incomplete download(s) in cache ({len(incomplete)} file(s))"
            )

    if shards and not errors:
        actual_bytes = sum((model_dir / s).stat().st_size for s in shards)
        if expected_bytes and actual_bytes < expected_bytes * 0.99:
            errors.append(
                f"weight shards too small: {actual_bytes} bytes "
                f"(expected ~{expected_bytes} from index metadata)"
            )
        # Heuristic: Qwen3-32B 5-bit should be well over 15 GB.
        if actual_bytes < 15_000_000_000:
            errors.append(
                f"weight shards suspiciously small for 32B 5-bit "
                f"({actual_bytes / 1e9:.2f} GB); re-run download"
            )

    return (errors, shards, expected_bytes)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_model.py MODEL_DIR [MODEL_DIR ...]", file=sys.stderr)
        return 2

    exit_code = 0
    for arg in sys.argv[1:]:
        model_dir = Path(arg)
        if not model_dir.is_dir():
            print(f"ERROR: not a directory: {model_dir}", file=sys.stderr)
            exit_code = 2
            continue

        errors, shards, expected_bytes = validate(model_dir)
        if errors:
            print(f"ERROR: {model_dir} — incomplete or corrupt:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"OK: {model_dir} — {len(shards)} weight shard(s)")
        if shards:
            actual_bytes = sum((model_dir / s).stat().st_size for s in shards)
            print(f"OK: total weight size {actual_bytes / 1e9:.2f} GB")
            if expected_bytes:
                print(f"OK: index metadata total_size ~{expected_bytes / 1e9:.2f} GB")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
