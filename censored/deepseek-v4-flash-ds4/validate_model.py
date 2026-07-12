#!/usr/bin/env python3
"""Validate DeepSeek V4 GGUF weights for the ds4 stack.

Checks:
  - file exists (and is not only a partial .part sibling)
  - exact byte size when the artifact is known (from Hugging Face x-linked-size)
  - GGUF magic bytes and supported version

Exit codes:
  0 — all checks passed
  1 — file exists but failed validation (incomplete / corrupt / wrong size)
  2 — file missing
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF"
SUPPORTED_VERSIONS = {2, 3}

# Exact sizes from Hugging Face Hub (antirez/deepseek-v4-gguf), via x-linked-size.
# PRO multi-hundred-GB files are omitted — use the HF CLI path and size check loosely.
EXPECTED_BYTES: dict[str, int] = {
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf": 86_720_111_488,
    "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf": 97_591_747_456,
    "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf": 164_633_502_592,
    "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf": 3_807_602_400,
}

# Minimum floor when size is unknown (reject empty / tiny stubs).
MIN_UNKNOWN_BYTES = 1_000_000


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _has_gguf_header(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
    except OSError as exc:
        return False, f"cannot read file: {exc}"
    if len(header) < 8:
        return False, "file too short for GGUF header"
    magic, version = struct.unpack("<II", header)
    if magic != GGUF_MAGIC:
        return False, "invalid GGUF magic bytes (not a GGUF file)"
    if version not in SUPPORTED_VERSIONS:
        return False, f"unsupported GGUF version {version}"
    return True, f"GGUF v{version}"


def validate_gguf(path: Path) -> tuple[bool, str]:
    """Return (ok, detail). ok is True only for a complete, valid GGUF."""
    if path.is_symlink():
        try:
            path = path.resolve(strict=True)
        except OSError:
            return False, f"broken symlink: {path}"

    if not path.is_file():
        # download_model.sh writes to "$out.part" until complete
        part = Path(str(path) + ".part")
        if part.is_file():
            size = part.stat().st_size
            expected = EXPECTED_BYTES.get(path.name)
            if expected:
                pct = 100.0 * size / expected
                return (
                    False,
                    f"incomplete download ({_format_gb(size)} / {_format_gb(expected)}, "
                    f"{pct:.1f}%): {part.name}",
                )
            return False, f"incomplete download (partial only): {part.name} ({_format_gb(size)})"
        return False, f"missing: {path}"

    size = path.stat().st_size
    name = path.name
    # If validating a .part path directly, strip for expected lookup
    if name.endswith(".part"):
        lookup_name = name[: -len(".part")]
    else:
        lookup_name = name

    expected = EXPECTED_BYTES.get(lookup_name)

    ok_header, header_detail = _has_gguf_header(path)
    if not ok_header:
        return False, header_detail

    if expected is not None:
        if size < expected:
            pct = 100.0 * size / expected
            return (
                False,
                f"incomplete download ({size} / {expected} bytes, {pct:.1f}% = "
                f"{_format_gb(size)} / {_format_gb(expected)}); re-run ./1_setup_download.sh",
            )
        if size > expected:
            return (
                False,
                f"unexpected file size ({size} bytes, expected {expected}); "
                f"delete and re-download",
            )
    elif size < MIN_UNKNOWN_BYTES:
        return False, f"file too small ({size} bytes) for a model weight"

    # Still named .part even if size matches — treat as incomplete until renamed.
    if name.endswith(".part"):
        return False, f"partial download still named {name}; re-run ./1_setup_download.sh"

    return True, f"{header_detail}, {_format_gb(size)}, {lookup_name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DeepSeek V4 GGUF for ds4")
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF path(s) to validate")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )
    args = parser.parse_args()

    worst = 0
    for raw in args.paths:
        path = raw.expanduser()
        ok, detail = validate_gguf(path)
        if ok:
            if not args.quiet:
                print(f"OK: {path} — {detail}")
            continue
        if "missing:" in detail:
            print(f"MISSING: {path} — {detail}", file=sys.stderr)
            worst = max(worst, 2)
        else:
            print(f"ERROR: {path} — {detail}", file=sys.stderr)
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    sys.exit(main())
