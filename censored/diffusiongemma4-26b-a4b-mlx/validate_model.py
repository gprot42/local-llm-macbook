#!/usr/bin/env python3
"""Verify a local MLX model directory has all required weights.

Exit codes:
  0 — model complete
  1 — directory exists but weights are incomplete or corrupt
  2 — directory does not exist

Flags:
  --list-missing   Print missing filenames (one per line) to stdout; exit 0 if none.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

SHARD_NAME_RE = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")


def validate(model_dir: Path) -> tuple[list[str], list[str], int, list[str]]:
    """Return (errors, shard_names, expected_bytes, missing_relative_paths)."""
    errors: list[str] = []
    missing: list[str] = []

    def _need(rel: str, *, empty_is_missing: bool = True) -> None:
        path = model_dir / rel
        if not path.is_file():
            if rel not in missing:
                missing.append(rel)
            errors.append(f"missing required file: {rel}" if rel in REQUIRED else f"missing weight shard: {rel}")
        elif empty_is_missing and path.stat().st_size == 0:
            if rel not in missing:
                missing.append(rel)
            errors.append(f"empty weight shard: {rel}" if rel.endswith(".safetensors") else f"empty required file: {rel}")

    if not model_dir.is_dir():
        return (["model directory does not exist"], [], 0, [])

    for name in REQUIRED:
        _need(name)

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    shards: list[str] = []
    expected_bytes = 0

    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid model.safetensors.index.json: {exc}")
            index = {}
        expected_bytes = int(index.get("metadata", {}).get("total_size", 0) or 0)
        weight_map = index.get("weight_map") or {}
        if not weight_map:
            errors.append("model.safetensors.index.json has empty weight_map")
        else:
            shards = sorted(set(weight_map.values()))
            if not shards:
                errors.append("model.safetensors.index.json has no weight shards listed")
            for shard in shards:
                _need(shard)

        _check_contiguous_sharded_files(model_dir, shards, errors, missing)
    elif single_path.is_file() and single_path.stat().st_size > 0:
        shards = ["model.safetensors"]
        expected_bytes = single_path.stat().st_size
    else:
        errors.append(
            "missing weights: need model.safetensors.index.json + shards "
            "or model.safetensors"
        )
        if not index_path.is_file():
            missing.append("model.safetensors.index.json")
        if not single_path.is_file():
            missing.append("model.safetensors")

    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    if cache_dir.is_dir():
        incomplete = sorted(cache_dir.glob("*.incomplete"))
        if incomplete:
            errors.append(
                f"incomplete download(s) in cache ({len(incomplete)} file(s))"
            )

    if shards and not any("missing" in e or "empty" in e for e in errors):
        actual_bytes = sum((model_dir / s).stat().st_size for s in shards)
        if expected_bytes and actual_bytes < expected_bytes * 0.99:
            errors.append(
                f"weight shards too small: {actual_bytes} bytes "
                f"(expected ~{expected_bytes} from index metadata)"
            )
            for shard in shards:
                if shard not in missing:
                    missing.append(shard)

    deduped_errors: list[str] = []
    seen_err: set[str] = set()
    for err in errors:
        if err not in seen_err:
            seen_err.add(err)
            deduped_errors.append(err)

    deduped_missing = sorted(set(missing))
    return (deduped_errors, shards, expected_bytes, deduped_missing)


def _check_contiguous_sharded_files(
    model_dir: Path,
    shards: list[str],
    errors: list[str],
    missing: list[str],
) -> None:
    """Ensure model-00001-of-00003 … model-000NN-of-000NN all exist on disk."""
    by_total: dict[int, set[int]] = {}
    for name in shards:
        m = SHARD_NAME_RE.match(name)
        if not m:
            continue
        num, total = int(m.group(1)), int(m.group(2))
        by_total.setdefault(total, set()).add(num)

    for total in sorted(by_total):
        for num in range(1, total + 1):
            expected = f"model-{num:05d}-of-{total:05d}.safetensors"
            path = model_dir / expected
            if not path.is_file():
                if expected not in missing:
                    missing.append(expected)
                msg = f"missing weight shard: {expected}"
                if msg not in errors:
                    errors.append(msg)
            elif path.stat().st_size == 0:
                if expected not in missing:
                    missing.append(expected)
                msg = f"empty weight shard: {expected}"
                if msg not in errors:
                    errors.append(msg)


def main() -> int:
    args = sys.argv[1:]
    list_missing = False
    if args and args[0] == "--list-missing":
        list_missing = True
        args = args[1:]

    if len(args) != 1:
        print(
            "usage: validate_model.py [--list-missing] MODEL_DIR",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args[0]).resolve()
    if not model_dir.is_dir():
        if list_missing:
            return 2
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return 2

    errors, shards, expected_bytes, missing_files = validate(model_dir)

    if list_missing:
        for name in missing_files:
            print(name)
        return 0 if not missing_files else 1

    if errors:
        print("ERROR: Local model weights are incomplete or corrupt:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(shards)} weight shard(s), all tensors mapped, config/tokenizer present")
    if expected_bytes:
        actual_bytes = sum((model_dir / s).stat().st_size for s in shards)
        print(f"OK: total weight size {actual_bytes / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())