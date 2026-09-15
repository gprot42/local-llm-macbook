#!/usr/bin/env python3
"""Verify YuE2 converted generator + VAE directories.

Exit codes:
  0 — complete
  1 — incomplete or corrupt
  2 — path does not exist / usage error
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_dir(model_dir: Path, label: str) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    if not model_dir.is_dir():
        return ([f"{label} directory does not exist: {model_dir}"], [], 0)

    config = model_dir / "config.json"
    if not _nonempty(config):
        errors.append(f"{label}: missing or empty config.json")

    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    shards: list[str] = []
    expected_bytes = 0
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: invalid model.safetensors.index.json: {exc}")
            index = {}
        expected_bytes = int(index.get("metadata", {}).get("total_size", 0) or 0)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            errors.append(f"{label}: model.safetensors.index.json has no shards")
    elif single.is_file():
        shards = ["model.safetensors"]
        expected_bytes = single.stat().st_size
    else:
        extras = sorted(
            p.name
            for p in model_dir.iterdir()
            if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
        )
        if extras:
            shards = extras
            expected_bytes = sum((model_dir / s).stat().st_size for s in shards)
        else:
            errors.append(
                f"{label}: missing weights (need model.safetensors, an index, or *.npz)"
            )

    for shard in shards:
        path = model_dir / shard
        if not path.is_file():
            errors.append(f"{label}: missing weight shard: {shard}")
        elif path.stat().st_size == 0:
            errors.append(f"{label}: empty weight shard: {shard}")

    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    if cache_dir.is_dir():
        incomplete = sorted(cache_dir.glob("*.incomplete"))
        if incomplete:
            errors.append(
                f"{label}: incomplete download(s) in cache ({len(incomplete)} file(s))"
            )

    if shards and not errors and expected_bytes:
        actual = sum((model_dir / s).stat().st_size for s in shards)
        if actual < expected_bytes * 0.99:
            errors.append(
                f"{label}: weight shards too small: {actual} bytes "
                f"(expected ~{expected_bytes})"
            )

    return (errors, shards, expected_bytes)


def validate_paths_file(paths_file: Path, model_dir: Path, vae_dir: Path | None) -> list[str]:
    errors: list[str] = []
    if not paths_file.is_file():
        return [f"missing paths file: {paths_file}"]
    try:
        payload = json.loads(paths_file.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid paths JSON: {exc}"]
    if not isinstance(payload, dict):
        return ["paths JSON is not an object"]
    if "vae" not in payload:
        errors.append("paths JSON missing 'vae'")
    else:
        vae = Path(payload["vae"])
        if not vae.exists():
            errors.append(f"paths.vae does not exist: {vae}")
        if vae_dir is not None and vae.resolve() != vae_dir.resolve():
            errors.append(f"paths.vae mismatch: {vae} vs --vae {vae_dir}")
    if "model" in payload:
        listed = Path(payload["model"])
        if listed.exists() and listed.resolve() != model_dir.resolve():
            # Conversion output dir can be a parent of listed files; warn only if missing.
            if not model_dir.exists():
                errors.append(f"paths.model does not match MODEL_DIR: {listed}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YuE2 converted weights")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None)
    args = parser.parse_args()

    exit_code = 0
    errors, shards, expected = validate_dir(args.model_dir, "generator")
    if errors:
        print(f"ERROR: {args.model_dir} — incomplete or corrupt:", flush=True)
        for err in errors:
            print(f"  - {err}")
        exit_code = 1 if args.model_dir.is_dir() else 2
    else:
        print(f"OK: generator {args.model_dir} — {len(shards)} weight file(s)")
        if expected:
            actual = sum((args.model_dir / s).stat().st_size for s in shards)
            print(f"OK: generator weights {actual / 1e9:.2f} GB")

    if args.vae is not None:
        verrs, vshards, vexpected = validate_dir(args.vae, "vae")
        if verrs:
            print(f"ERROR: {args.vae} — incomplete or corrupt:")
            for err in verrs:
                print(f"  - {err}")
            exit_code = 1 if exit_code == 0 and args.vae.is_dir() else (exit_code or 2)
        else:
            print(f"OK: vae {args.vae} — {len(vshards)} weight file(s)")
            if vexpected:
                actual = sum((args.vae / s).stat().st_size for s in vshards)
                print(f"OK: vae weights {actual / 1e9:.2f} GB")

    if args.paths is not None:
        perrs = validate_paths_file(args.paths, args.model_dir, args.vae)
        if perrs:
            print(f"ERROR: {args.paths}:")
            for err in perrs:
                print(f"  - {err}")
            exit_code = exit_code or 1
        else:
            print(f"OK: paths {args.paths}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
