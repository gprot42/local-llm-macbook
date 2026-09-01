#!/usr/bin/env python3
"""Resumable Hugging Face snapshot download for local-dir models.

huggingface_hub ≥1.x (PR #4228) writes each attempt to a process-unique
``*.{uuid}.incomplete`` file and deletes it on failure — so re-running
``hf download`` restarts multi-GB shards from zero and can leave orphan
partials after SIGKILL.

This wrapper:

1. Promotes the largest UUID orphan per etag to a stable ``.incomplete`` path
2. Patches ``_download_to_tmp_and_move`` to Range-resume via HTTP
3. Leaves stable partials on interrupt so the next run continues
4. Cleans UUID orphans and empty lock files

Usage:
  python download_resumable.py REPO_ID --local-dir DIR
  python download_resumable.py REPO_ID --local-dir DIR --force-download
  python download_resumable.py REPO_ID --local-dir DIR --cleanup-only
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Prefer HTTP Range resume over xet for files under the hub's 50 GB HTTP cap.
# Must be set before huggingface_hub.constants is first imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import constants, snapshot_download  # noqa: E402
from huggingface_hub import file_download as fd  # noqa: E402
from huggingface_hub.file_download import (  # noqa: E402
    _check_disk_space,
    _chmod_and_move,
    http_get,
    xet_get,
)
from huggingface_hub.utils._runtime import is_xet_available  # noqa: E402

# UUID incomplete:  <short_hash>.<etag>.<8hex>.incomplete
# Stable incomplete: <short_hash>.<etag>.incomplete
_UUID_INCOMPLETE_RE = re.compile(
    r"^(?P<stem>.+)\.(?P<uuid>[0-9a-f]{8})\.incomplete$",
    re.IGNORECASE,
)


def _cache_download_dir(local_dir: Path) -> Path:
    return local_dir / ".cache" / "huggingface" / "download"


def _format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def promote_and_cleanup_incompletes(local_dir: Path) -> tuple[int, int, int]:
    """Promote largest UUID orphan per stem → stable path; delete the rest.

    Returns (promoted_count, removed_count, bytes_freed_after_promote).
    """
    cache = _cache_download_dir(local_dir)
    if not cache.is_dir():
        return (0, 0, 0)

    # Group UUID incompletes by stable target name (stem without uuid).
    uuid_groups: dict[str, list[Path]] = defaultdict(list)
    stable_files: dict[str, Path] = {}

    for path in cache.glob("*.incomplete"):
        m = _UUID_INCOMPLETE_RE.match(path.name)
        if m:
            stable_name = f"{m.group('stem')}.incomplete"
            uuid_groups[stable_name].append(path)
        else:
            stable_files[path.name] = path

    promoted = 0
    removed = 0
    freed = 0

    for stable_name, variants in uuid_groups.items():
        stable_path = cache / stable_name
        candidates = list(variants)
        if stable_path.is_file():
            candidates.append(stable_path)

        best = max(candidates, key=lambda p: p.stat().st_size if p.is_file() else 0)
        best_size = best.stat().st_size if best.is_file() else 0

        if best_size <= 0:
            for p in variants:
                try:
                    sz = p.stat().st_size
                    p.unlink(missing_ok=True)
                    removed += 1
                    freed += sz
                except OSError:
                    pass
            continue

        # Ensure stable path holds the best partial.
        if best.resolve() != stable_path.resolve():
            if stable_path.exists():
                try:
                    freed += stable_path.stat().st_size
                except OSError:
                    pass
                stable_path.unlink(missing_ok=True)
            best.replace(stable_path)
            promoted += 1
            print(
                f"   resume: keeping {_format_bytes(best_size)} partial → {stable_name}",
                flush=True,
            )
        elif stable_path.is_file() and best.resolve() == stable_path.resolve():
            if best_size > 0:
                print(
                    f"   resume: existing {_format_bytes(best_size)} partial → {stable_name}",
                    flush=True,
                )

        for p in variants:
            if p.exists() and p.resolve() != stable_path.resolve():
                try:
                    sz = p.stat().st_size
                    p.unlink(missing_ok=True)
                    removed += 1
                    freed += sz
                except OSError:
                    pass

    # Drop empty lock files left after hard-kills.
    for lock in cache.glob("*.lock"):
        try:
            if lock.is_file() and lock.stat().st_size == 0:
                lock.unlink(missing_ok=True)
        except OSError:
            pass

    return (promoted, removed, freed)


def cleanup_all_incompletes(local_dir: Path) -> int:
    """Remove every .incomplete under the local-dir download cache."""
    cache = _cache_download_dir(local_dir)
    if not cache.is_dir():
        return 0
    n = 0
    freed = 0
    for path in cache.glob("*.incomplete"):
        try:
            freed += path.stat().st_size
            path.unlink(missing_ok=True)
            n += 1
        except OSError:
            pass
    for lock in cache.glob("*.lock"):
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass
    if n:
        print(f"→ Cleaned {n} incomplete file(s) ({_format_bytes(freed)})", flush=True)
    return n


def _install_resumable_download_patch() -> None:
    """Replace hub's non-resumable UUID temp download with stable Range resume."""

    def _download_to_tmp_and_move(
        incomplete_path: Path,
        destination_path: Path,
        url_to_download: str,
        headers: dict[str, str],
        expected_size: int | None,
        filename: str,
        force_download: bool,
        etag: str | None,
        xet_file_data,
        tqdm_class=None,
    ) -> None:
        if destination_path.exists() and not force_download:
            return

        # Stable name (no per-process UUID) so re-runs can continue.
        tmp_path = incomplete_path
        if force_download and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

        resume_size = 0
        if tmp_path.exists() and not force_download:
            resume_size = tmp_path.stat().st_size
            if expected_size is not None and resume_size > expected_size:
                tmp_path.unlink(missing_ok=True)
                resume_size = 0
            elif expected_size is not None and resume_size == expected_size:
                _chmod_and_move(tmp_path, destination_path)
                return

        if expected_size is not None:
            remaining = max(expected_size - resume_size, 0)
            if remaining > 0:
                _check_disk_space(remaining, tmp_path.parent)
                _check_disk_space(remaining, destination_path.parent)

        # Files >50 GB must use xet; partial xet files are not HTTP-Range-resumable.
        use_xet = (
            xet_file_data is not None
            and is_xet_available()
            and not constants.HF_HUB_DISABLE_XET
            and expected_size is not None
            and expected_size > constants.MAX_HTTP_DOWNLOAD_SIZE
        )

        try:
            if use_xet:
                # Restart xet reconstruct into stable path (no cross-run chunk assemble).
                if resume_size > 0:
                    tmp_path.unlink(missing_ok=True)
                    resume_size = 0
                xet_get(
                    incomplete_path=tmp_path,
                    xet_file_data=xet_file_data,
                    headers=headers,
                    expected_size=expected_size,
                    displayed_filename=filename,
                    tqdm_class=tqdm_class,
                )
            else:
                mode = "ab" if resume_size > 0 else "wb"
                if resume_size > 0:
                    print(
                        f"   resuming {filename} from {_format_bytes(resume_size)}"
                        + (
                            f" / {_format_bytes(expected_size)}"
                            if expected_size
                            else ""
                        ),
                        flush=True,
                    )
                with tmp_path.open(mode) as f:
                    http_get(
                        url_to_download,
                        f,
                        headers=headers,
                        expected_size=expected_size,
                        resume_size=resume_size,
                        displayed_filename=filename,
                        tqdm_class=tqdm_class,
                    )

            _chmod_and_move(tmp_path, destination_path)
        except BaseException:
            # Keep stable partial for the next run (unlike stock hub finally:unlink).
            raise

    fd._download_to_tmp_and_move = _download_to_tmp_and_move  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face repo id (org/name)")
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Directory to place the snapshot (MODEL_DIR)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only promote/cleanup incomplete files; do not download",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel download workers (default 4; lower is kinder on flaky links)",
    )
    args = parser.parse_args(argv)

    local_dir: Path = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Preparing resume cache under {local_dir}", flush=True)
    promoted, removed, freed = promote_and_cleanup_incompletes(local_dir)
    if promoted or removed:
        print(
            f"→ Resume prep: promoted {promoted}, removed {removed} orphan incomplete(s)"
            + (f", freed {_format_bytes(freed)}" if freed else ""),
            flush=True,
        )
    else:
        print("→ Resume prep: no orphan incomplete files", flush=True)

    if args.cleanup_only:
        return 0

    # Ensure constants match env even if hub was imported elsewhere first.
    constants.HF_HUB_DISABLE_XET = True

    _install_resumable_download_patch()

    print(f"→ Downloading {args.repo_id} → {local_dir}", flush=True)
    print("   (HTTP Range resume enabled; completed files are skipped)", flush=True)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_dir),
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
    except KeyboardInterrupt:
        print(
            "\n→ Interrupted — partial files kept for resume.",
            file=sys.stderr,
            flush=True,
        )
        promote_and_cleanup_incompletes(local_dir)
        return 130

    # Full success: drop any leftover temps/locks so validate_model stays green.
    cleanup_all_incompletes(local_dir)
    print(f"→ Download finished: {local_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
