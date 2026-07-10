#!/usr/bin/env python3
"""Resumable GGUF download from Hugging Face Hub.

Downloads directly to the destination path and resumes interrupted transfers
on the next run. Re-running is safe: completed files are skipped via validate_model.py.
"""
from __future__ import annotations

import argparse
import os

# Default hub read timeout is 10s — too short for multi-GB downloads over slow links.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import struct

import httpx
from huggingface_hub import get_hf_file_metadata, hf_hub_url, try_to_load_from_cache
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import build_hf_headers, get_token

MAX_DOWNLOAD_ATTEMPTS = 100
RETRY_BASE_SECONDS = 5
RETRY_MAX_SECONDS = 120

_TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    OSError,
)

GGUF_MAGIC = 0x46554747  # "GGUF"


def _load_validate_gguf(validate_script: Path):
    """Load validate_gguf from validate_model.py in-process.

    Avoids re-execing sys.executable (breaks if the project dir is renamed
    mid-download, or if the venv python path is otherwise unavailable).
    """
    script = validate_script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"validate script not found: {script}")
    spec = importlib.util.spec_from_file_location("_ornith_validate_model", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validate script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_gguf


def _validate(path: Path, validate_script: Path) -> bool:
    try:
        validate_gguf = _load_validate_gguf(validate_script)
    except Exception as exc:
        print(f"ERROR: failed to load validator ({exc})", file=sys.stderr)
        return False

    ok, detail = validate_gguf(path)
    if ok:
        print(f"OK: {path} — {detail}")
        return True
    print(f"ERROR: {path} — {detail}", file=sys.stderr)
    return False


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _retry_delay(attempt: int) -> float:
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** min(attempt - 1, 5)))


def _resolve_resume_size(dest: Path, expected_size: int | None) -> int:
    resume_size = dest.stat().st_size if dest.is_file() else 0
    if resume_size > 0 and not _has_valid_gguf_header(dest):
        print(
            f"→ Partial file at {dest} is not a valid GGUF header; restarting download",
            file=sys.stderr,
        )
        dest.unlink()
        return 0

    if expected_size is not None:
        if resume_size > expected_size:
            print(
                f"→ Local file larger than remote ({_format_gb(resume_size)} > {_format_gb(expected_size)}); restarting",
                file=sys.stderr,
            )
            dest.unlink()
            return 0
        if resume_size == expected_size:
            return resume_size
    return resume_size


def _has_valid_gguf_header(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        magic = struct.unpack("<I", handle.read(4))[0]
    return magic == GGUF_MAGIC


def download_resumable(
    *,
    repo_id: str,
    remote_path: str,
    dest: Path,
    validate_script: Path,
    force: bool = False,
) -> int:
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and _validate(dest, validate_script):
        print(f"→ Already complete — skipping download")
        return 0

    if force and dest.is_file():
        print(f"→ Removing existing file (--force): {dest}")
        dest.unlink()

    token = get_token()

    if not force:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=remote_path)
        if isinstance(cached, str) and Path(cached).is_file():
            print(f"→ Copying from Hugging Face cache: {cached}")
            shutil.copyfile(cached, dest)
            if _validate(dest, validate_script):
                return 0
            print("→ Cached copy failed validation; falling back to HTTP download", file=sys.stderr)
            dest.unlink(missing_ok=True)

    hub_url = hf_hub_url(repo_id=repo_id, filename=remote_path, repo_type="model")
    headers = build_hf_headers(token=token)
    probe = get_hf_file_metadata(hub_url, token=token, headers=headers, retry_on_errors=True)
    expected_size = probe.size

    resume_size = _resolve_resume_size(dest, expected_size)
    if expected_size is not None and resume_size == expected_size:
        if _validate(dest, validate_script):
            return 0
        print("→ File size matches remote but validation failed; restarting download", file=sys.stderr)
        dest.unlink()
        resume_size = 0

    if resume_size > 0:
        if expected_size is not None:
            pct = 100 * resume_size / expected_size
            print(
                f"→ Resuming download at {_format_gb(resume_size)} / {_format_gb(expected_size)} ({pct:.1f}%)"
            )
        else:
            print(f"→ Resuming download at {_format_gb(resume_size)}")
    elif expected_size is not None:
        print(f"→ Downloading {_format_gb(expected_size)} to {dest}")
    else:
        print(f"→ Downloading to {dest}")

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        resume_size = _resolve_resume_size(dest, expected_size)
        if expected_size is not None and resume_size == expected_size:
            break

        metadata = get_hf_file_metadata(hub_url, token=token, headers=headers, retry_on_errors=True)
        download_url = metadata.location
        if metadata.size is not None:
            expected_size = metadata.size

        try:
            mode = "ab" if resume_size > 0 else "wb"
            with dest.open(mode) as handle:
                http_get(
                    download_url,
                    handle,
                    headers=headers,
                    expected_size=expected_size,
                    resume_size=resume_size,
                    displayed_filename=dest.name,
                )
            break
        except _TRANSIENT_ERRORS as exc:
            if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                print(f"→ Download failed after {attempt} attempts: {exc}", file=sys.stderr)
                raise
            delay = _retry_delay(attempt)
            saved = dest.stat().st_size if dest.is_file() else 0
            print(
                f"→ Download interrupted ({exc}); saved {_format_gb(saved)}; "
                f"retrying in {delay:.0f}s (attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS})...",
                file=sys.stderr,
            )
            time.sleep(delay)

    if not _validate(dest, validate_script):
        return 1

    print(f"→ Download complete: {dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable GGUF download from Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Hugging Face repo id")
    parser.add_argument("--remote", required=True, help="Path of the file inside the repo")
    parser.add_argument("--dest", required=True, type=Path, help="Local destination path")
    parser.add_argument(
        "--validate-script",
        type=Path,
        default=Path(__file__).with_name("validate_model.py"),
        help="Path to validate_model.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing destination file and download from scratch",
    )
    args = parser.parse_args()

    return download_resumable(
        repo_id=args.repo,
        remote_path=args.remote,
        dest=args.dest,
        validate_script=args.validate_script.resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())