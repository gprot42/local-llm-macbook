#!/usr/bin/env python3
"""Resumable DeepSeek V4 GGUF download from Hugging Face (stdlib only).

Designed for multi-hour ~80 GB transfers that often die with:
  curl: (56) Recv failure: Connection reset by peer

Strategy:
  - Write to <dest>.part (same layout as ds4/download_model.sh)
  - HTTP Range resume on every attempt
  - Exponential backoff, many attempts
  - Re-resolve the CDN URL each attempt (HF signed URLs expire)
  - Validate exact size + GGUF magic via validate_model.py
  - Atomic rename .part → final when complete
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HF_REPO_DEFAULT = "antirez/deepseek-v4-gguf"
MAX_DOWNLOAD_ATTEMPTS = int(os.environ.get("DS4_DOWNLOAD_ATTEMPTS", "200"))
RETRY_BASE_SECONDS = float(os.environ.get("DS4_RETRY_BASE", "5"))
RETRY_MAX_SECONDS = float(os.environ.get("DS4_RETRY_MAX", "120"))
CONNECT_TIMEOUT = float(os.environ.get("DS4_CONNECT_TIMEOUT", "30"))
READ_TIMEOUT = float(os.environ.get("DS4_READ_TIMEOUT", "120"))
PROGRESS_EVERY_BYTES = 256 * 1024 * 1024  # ~256 MiB

GGUF_MAGIC = 0x46554747  # "GGUF"

# Known sizes (HF x-linked-size) — keep in sync with validate_model.py
EXPECTED_BYTES: dict[str, int] = {
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf": 86_720_111_488,
    "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf": 97_591_747_456,
    "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf": 164_633_502_592,
    "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf": 3_807_602_400,
}

_USER_AGENT = "deepseek-v4-flash-ds4-downloader/1.0 (+https://github.com/antirez/ds4)"


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _retry_delay(attempt: int) -> float:
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** min(attempt - 1, 5)))


def _load_validate_gguf(validate_script: Path):
    script = validate_script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"validate script not found: {script}")
    spec = importlib.util.spec_from_file_location("_ds4_validate_model", script)
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


def _has_valid_gguf_header(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        magic = struct.unpack("<I", handle.read(4))[0]
    return magic == GGUF_MAGIC


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _hf_token() -> str | None:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        try:
            return token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


def _build_headers(token: str | None, *, range_start: int | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "*/*",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if range_start is not None and range_start > 0:
        headers["Range"] = f"bytes={range_start}-"
    return headers


def _open_url(url: str, headers: dict[str, str], *, method: str = "GET"):
    req = urllib.request.Request(url, headers=headers, method=method)
    # Single timeout covers connect+read socket ops; short idle stalls raise and we resume.
    timeout = max(CONNECT_TIMEOUT, READ_TIMEOUT)
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def _probe_size(url: str, token: str | None) -> int | None:
    """Return Content-Length / x-linked-size when available."""
    headers = _build_headers(token)
    # Prefer HEAD; some CDNs only send size on GET — fall back carefully.
    for method in ("HEAD", "GET"):
        try:
            # For GET probe, only read headers via range 0-0 if possible.
            probe_headers = dict(headers)
            if method == "GET":
                probe_headers["Range"] = "bytes=0-0"
            with _open_url(url, probe_headers, method=method) as resp:
                linked = resp.headers.get("X-Linked-Size") or resp.headers.get("x-linked-size")
                if linked and linked.isdigit():
                    return int(linked)
                cr = resp.headers.get("Content-Range") or ""
                # bytes 0-0/TOTAL
                if "/" in cr:
                    total = cr.rsplit("/", 1)[-1]
                    if total.isdigit():
                        return int(total)
                cl = resp.headers.get("Content-Length")
                if method == "HEAD" and cl and cl.isdigit():
                    return int(cl)
        except Exception as exc:
            print(f"→ size probe {method} failed ({exc}); continuing", file=sys.stderr)
            continue
    return None


def _resolve_resume_size(part: Path, expected_size: int | None) -> int:
    if not part.is_file():
        return 0
    size = part.stat().st_size
    if size > 0 and not _has_valid_gguf_header(part):
        print(
            f"→ Partial {part.name} has invalid GGUF header; restarting",
            file=sys.stderr,
        )
        part.unlink()
        return 0
    if expected_size is not None:
        if size > expected_size:
            print(
                f"→ Partial larger than remote ({_format_gb(size)} > {_format_gb(expected_size)}); restarting",
                file=sys.stderr,
            )
            part.unlink()
            return 0
        if size == expected_size:
            return size
    return size


def _copy_stream(resp, handle, *, start_size: int, expected_size: int | None) -> int:
    """Copy response body to handle; return total bytes on disk after write."""
    written_this_session = 0
    last_report = start_size
    chunk = 8 * 1024 * 1024  # 8 MiB
    while True:
        data = resp.read(chunk)
        if not data:
            break
        handle.write(data)
        written_this_session += len(data)
        now = start_size + written_this_session
        if now - last_report >= PROGRESS_EVERY_BYTES or (
            expected_size and now >= expected_size
        ):
            if expected_size:
                pct = 100.0 * now / expected_size
                print(
                    f"→ progress {_format_gb(now)} / {_format_gb(expected_size)} ({pct:.1f}%)",
                    flush=True,
                )
            else:
                print(f"→ progress {_format_gb(now)}", flush=True)
            last_report = now
    handle.flush()
    return start_size + written_this_session


def download_resumable(
    *,
    repo_id: str,
    remote_path: str,
    dest: Path,
    validate_script: Path,
    force: bool = False,
    link_as: Path | None = None,
) -> int:
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dest) + ".part")
    expected_size = EXPECTED_BYTES.get(dest.name) or EXPECTED_BYTES.get(remote_path)

    if force:
        for p in (dest, part):
            if p.is_file() or p.is_symlink():
                print(f"→ Removing (--force): {p}")
                p.unlink()
    elif dest.is_file() and _validate(dest, validate_script):
        print("→ Already complete — skipping download")
        if link_as is not None:
            _link(dest, link_as)
        return 0

    # If a previous run left a complete-looking dest that fails validation, re-part it.
    if dest.is_file() and not force:
        if expected_size is not None and dest.stat().st_size < expected_size:
            print(f"→ Incomplete final file; moving to {part.name} for resume")
            if part.exists():
                part.unlink()
            dest.rename(part)
        elif not _validate(dest, validate_script):
            print("→ Existing file failed validation; restarting via .part", file=sys.stderr)
            if part.exists():
                part.unlink()
            dest.rename(part)

    token = _hf_token()
    url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_path}"

    if expected_size is None:
        probed = _probe_size(url, token)
        if probed:
            expected_size = probed
            print(f"→ Remote size: {_format_gb(expected_size)}")

    resume_size = _resolve_resume_size(part, expected_size)
    if expected_size is not None and resume_size == expected_size:
        print("→ Partial already full size; finalizing")
        part.replace(dest)
        if _validate(dest, validate_script):
            if link_as is not None:
                _link(dest, link_as)
            return 0
        print("→ Size match but validation failed; restarting", file=sys.stderr)
        dest.unlink(missing_ok=True)
        resume_size = 0

    if resume_size > 0:
        if expected_size:
            pct = 100.0 * resume_size / expected_size
            print(
                f"→ Resuming {remote_path} at {_format_gb(resume_size)} / "
                f"{_format_gb(expected_size)} ({pct:.1f}%)"
            )
        else:
            print(f"→ Resuming {remote_path} at {_format_gb(resume_size)}")
    else:
        if expected_size:
            print(f"→ Downloading {remote_path} ({_format_gb(expected_size)})")
        else:
            print(f"→ Downloading {remote_path}")
    print(f"→ Destination: {dest}")
    print(f"→ Attempts: up to {MAX_DOWNLOAD_ATTEMPTS} (connection resets resume automatically)")
    print()

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        resume_size = _resolve_resume_size(part, expected_size)
        if expected_size is not None and resume_size == expected_size:
            break

        headers = _build_headers(token, range_start=resume_size if resume_size > 0 else None)
        try:
            resp = _open_url(url, headers)
            try:
                status = getattr(resp, "status", None) or resp.getcode()
                # 200 on a resume request means server ignored Range — restart from 0
                # only if we already had data and got a full-body 200 with matching total.
                if resume_size > 0 and status == 200:
                    cl = resp.headers.get("Content-Length")
                    if cl and expected_size and int(cl) == expected_size:
                        print(
                            "→ Server ignored Range (HTTP 200 full body); restarting from 0",
                            file=sys.stderr,
                        )
                        resp.close()
                        part.unlink(missing_ok=True)
                        resume_size = 0
                        headers = _build_headers(token)
                        resp = _open_url(url, headers)
                        status = getattr(resp, "status", None) or resp.getcode()
                    elif cl and expected_size and int(cl) == expected_size - resume_size:
                        # Some servers send 200 with the remaining length only — OK.
                        pass
                    else:
                        # Ambiguous 200 with partial on disk: prefer Content-Range if present.
                        cr = resp.headers.get("Content-Range") or ""
                        if not cr.startswith("bytes "):
                            print(
                                f"→ Unexpected HTTP 200 while resuming from {resume_size}; "
                                f"retrying with fresh URL",
                                file=sys.stderr,
                            )
                            raise ConnectionError(f"unexpected HTTP {status} on resume")

                if status not in (200, 206):
                    raise ConnectionError(f"unexpected HTTP status {status}")

                mode = "ab" if resume_size > 0 else "wb"
                with part.open(mode) as handle:
                    _copy_stream(
                        resp,
                        handle,
                        start_size=resume_size,
                        expected_size=expected_size,
                    )
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError) as exc:
            # HTTP 416: range past EOF — treat as complete if sizes match
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                have = part.stat().st_size if part.is_file() else 0
                if expected_size and have >= expected_size:
                    print("→ HTTP 416 (already complete)")
                    break
            saved = part.stat().st_size if part.is_file() else 0
            if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                print(
                    f"ERROR: download failed after {attempt} attempts: {exc}",
                    file=sys.stderr,
                )
                print(f"       Partial kept at {part} ({_format_gb(saved)})", file=sys.stderr)
                return 1
            delay = _retry_delay(attempt)
            if expected_size:
                pct = 100.0 * saved / expected_size
                print(
                    f"→ Interrupted ({exc}); saved {_format_gb(saved)} / "
                    f"{_format_gb(expected_size)} ({pct:.1f}%); "
                    f"retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} in {delay:.0f}s ...",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"→ Interrupted ({exc}); saved {_format_gb(saved)}; "
                    f"retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} in {delay:.0f}s ...",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(delay)
            continue

        # Successful stream end — check completeness
        have = part.stat().st_size if part.is_file() else 0
        if expected_size is None or have >= expected_size:
            break
        # Stream closed early without exception (CDN idle disconnect)
        delay = _retry_delay(attempt)
        pct = 100.0 * have / expected_size
        print(
            f"→ Connection closed early at {_format_gb(have)} / {_format_gb(expected_size)} "
            f"({pct:.1f}%); retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} in {delay:.0f}s ...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)

    have = part.stat().st_size if part.is_file() else 0
    if expected_size is not None and have != expected_size:
        print(
            f"ERROR: final size mismatch ({have} != {expected_size})",
            file=sys.stderr,
        )
        return 1

    part.replace(dest)
    if not _validate(dest, validate_script):
        return 1

    if link_as is not None:
        _link(dest, link_as)

    print(f"→ Download complete: {dest}")
    return 0


def _link(dest: Path, link_as: Path) -> None:
    link_as = link_as.expanduser().resolve()
    link_as.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prefer relative symlink when both under same parent tree
        rel = os.path.relpath(dest, start=link_as.parent)
        if link_as.is_symlink() or link_as.exists():
            link_as.unlink()
        link_as.symlink_to(rel)
        print(f"→ Linked {link_as} -> {rel}")
    except OSError:
        if link_as.is_symlink() or link_as.exists():
            link_as.unlink()
        link_as.symlink_to(dest)
        print(f"→ Linked {link_as} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable DeepSeek V4 GGUF download")
    parser.add_argument("--repo", default=HF_REPO_DEFAULT, help="Hugging Face repo id")
    parser.add_argument("--remote", required=True, help="Filename inside the repo")
    parser.add_argument("--dest", required=True, type=Path, help="Local destination .gguf path")
    parser.add_argument(
        "--validate-script",
        type=Path,
        default=Path(__file__).with_name("validate_model.py"),
    )
    parser.add_argument(
        "--link",
        type=Path,
        default=None,
        help="Optional symlink to create (e.g. ds4/ds4flash.gguf)",
    )
    parser.add_argument("--force", action="store_true", help="Delete local files and re-download")
    args = parser.parse_args()

    return download_resumable(
        repo_id=args.repo,
        remote_path=args.remote,
        dest=args.dest,
        validate_script=args.validate_script.resolve(),
        force=args.force,
        link_as=args.link,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\n→ Interrupted by user. Partial progress is kept — re-run the same command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
