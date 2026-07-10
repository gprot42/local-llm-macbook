#!/usr/bin/env python3
"""Verify local GGUF weights for Ornith-1.0-35B MoE model.

Checks:
  - file exists
  - exact byte size matches the published Hugging Face artifact
  - GGUF magic bytes and supported version
  - general.architecture metadata references a Qwen MoE model

Exit codes:
  0 — all checks passed
  1 — file exists but failed validation
  2 — file missing
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = 0x46554747  # "GGUF"
SUPPORTED_VERSIONS = {2, 3}
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# Exact sizes from Hugging Face Hub (deepreinforce-ai/Ornith-1.0-35B-GGUF).
EXPECTED_BYTES: dict[str, int] = {
    "ornith-1.0-35b-Q4_K_M.gguf": 21_166_757_760,
    "ornith-1.0-35b-Q5_K_M.gguf": 24_729_130_848,
    "ornith-1.0-35b-Q6_K.gguf": 28_514_152_288,
    "ornith-1.0-35b-Q8_0.gguf": 36_903_138_880,
    "ornith-1.0-35b-bf16.gguf": 69_376_636_800,
}


def _read_u32(handle: BinaryIO) -> int:
    data = handle.read(4)
    if len(data) != 4:
        raise ValueError("unexpected end of file")
    return struct.unpack("<I", data)[0]


def _read_u64(handle: BinaryIO) -> int:
    data = handle.read(8)
    if len(data) != 8:
        raise ValueError("unexpected end of file")
    return struct.unpack("<Q", data)[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("unexpected end of file")
    return data.decode("utf-8", errors="replace")


def _skip_value(handle: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        handle.seek(_read_u64(handle), 1)
        return
    if value_type == GGUF_TYPE_ARRAY:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        for _ in range(count):
            _skip_value(handle, element_type)
        return
    scalar_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    size = scalar_sizes.get(value_type)
    if size is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    handle.seek(size, 1)


def _read_gguf_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) < 8:
            raise ValueError("file too short for GGUF header")
        magic, version = struct.unpack("<II", header)
        if magic != GGUF_MAGIC:
            raise ValueError("invalid GGUF magic bytes")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version}")
        return magic, version


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("rb") as handle:
        magic, version = struct.unpack("<II", handle.read(8))
        if magic != GGUF_MAGIC:
            raise ValueError("invalid GGUF magic bytes")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version}")

        _read_u64(handle)  # tensor_count
        kv_count = _read_u64(handle)
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            if value_type == GGUF_TYPE_STRING:
                metadata[key] = _read_string(handle)
            else:
                _skip_value(handle, value_type)
    return metadata


def validate_gguf(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file not found"

    size = path.stat().st_size
    if size == 0:
        return False, "file is empty"

    expected_size = EXPECTED_BYTES.get(path.name)
    if expected_size is not None:
        if size != expected_size:
            if size < expected_size:
                pct = 100 * size / expected_size
                return False, (
                    f"incomplete download ({size} / {expected_size} bytes, {pct:.1f}%); "
                    "re-run ./1_setup_download.sh to resume"
                )
            return False, f"unexpected file size ({size} bytes, expected {expected_size})"
    elif size < 500_000_000:
        return False, f"file too small ({size} bytes)"

    try:
        _, version = _read_gguf_header(path)
    except ValueError as exc:
        return False, str(exc)

    try:
        metadata = _read_metadata(path)
    except ValueError as exc:
        return False, f"failed to parse GGUF metadata: {exc}"

    architecture = metadata.get("general.architecture", "")
    if "qwen" not in architecture.lower():
        return False, (
            f"unexpected architecture '{architecture or '<missing>'}' "
            "(expected a Qwen MoE GGUF — Ornith-1.0-35B is qwen35moe)"
        )

    return True, f"GGUF v{version}, {architecture}, {size / 1e9:.2f} GB"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_model.py GGUF_PATH [GGUF_PATH ...]", file=sys.stderr)
        return 2

    exit_code = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        ok, detail = validate_gguf(path)
        if ok:
            print(f"OK: {path} — {detail}")
        else:
            print(f"ERROR: {path} — {detail}", file=sys.stderr)
            exit_code = 2 if "not found" in detail else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())