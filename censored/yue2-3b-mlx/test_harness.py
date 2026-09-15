#!/usr/bin/env python3
"""Harness checks for YuE2-3B MLX (lyra).

Offline checks never generate audio. --gate hits the live HTTP API if the
server is up (health + /v1/models). --live runs the ~16s quickstart clip.

Usage:
  python3 test_harness.py --offline
  python3 test_harness.py --gate
  python3 test_harness.py --live
  python3 test_harness.py --base http://127.0.0.1:8088 --gate

Exit codes:
  0  all required checks passed
  1  one or more required checks failed
  2  connectivity failure (--gate with no healthy endpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

STACK = Path(__file__).resolve().parent
DEFAULT_BASE = "http://127.0.0.1:8088"
DEFAULT_MODEL = "yue2-3b-mlx"
ENGINE = STACK / "engine"
PATHS = STACK / "models" / "paths.json"
CONFIG = STACK / ".yue2_config"
VALIDATE = STACK / "validate_model.py"
QUICKSTART = STACK / "examples" / "quickstart.json"
FULL_SONG = STACK / "examples" / "full-song.json"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


@dataclass
class Report:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        mark = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
        extra = f" — {check.detail}" if check.detail else ""
        print(f"  [{mark}] {check.name}{extra}")

    def failed_required(self) -> list[CheckResult]:
        return [c for c in self.checks if c.required and not c.ok]


def http_json(url: str, method: str = "GET", body: dict | None = None, timeout: float = 10) -> tuple[int, dict | None, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None
            return resp.status, parsed, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, raw
    except urllib.error.URLError as exc:
        return 0, None, str(exc.reason if hasattr(exc, "reason") else exc)


def load_shell_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if not CONFIG.is_file():
        return values
    for line in CONFIG.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key] = val.strip().strip('"')
    return values


def check_examples(report: Report) -> None:
    for path in (QUICKSTART, FULL_SONG):
        if not path.is_file():
            report.add(CheckResult(f"example {path.name}", False, "missing"))
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            report.add(CheckResult(f"example {path.name}", False, f"invalid JSON: {exc}"))
            continue
        missing = [k for k in ("style", "lyrics", "cot") if k not in payload]
        if missing:
            report.add(CheckResult(f"example {path.name}", False, f"missing {missing}"))
        elif payload["cot"] not in ("full", "melody", "off"):
            report.add(CheckResult(f"example {path.name}", False, f"bad cot={payload['cot']}"))
        else:
            report.add(CheckResult(f"example {path.name}", True, f"cot={payload['cot']}"))


def check_engine(report: Report) -> None:
    pyproject = ENGINE / "pyproject.toml"
    if not pyproject.is_file():
        report.add(
            CheckResult(
                "engine checkout",
                False,
                "missing engine/ — run ./1_setup_download.sh",
            )
        )
        return
    text = pyproject.read_text()
    ok = "lyra-yue2" in text or 'name = "lyra' in text
    report.add(CheckResult("engine checkout", ok, str(ENGINE)))
    help_cmd = ["uv", "run", "lyra", "--help"]
    try:
        proc = subprocess.run(
            help_cmd,
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "MLX_ENABLE_TF32": "0"},
        )
    except FileNotFoundError:
        report.add(CheckResult("lyra --help", False, "uv not on PATH"))
        return
    except subprocess.TimeoutExpired:
        report.add(CheckResult("lyra --help", False, "timed out"))
        return
    ok_help = proc.returncode == 0 and (
        "generate" in proc.stdout.lower() or "generate" in proc.stderr.lower()
    )
    detail = "ok" if ok_help else (proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")
    report.add(CheckResult("lyra --help", ok_help, detail[:200]))


def check_weights(report: Report) -> None:
    cfg = load_shell_config()
    model_dir = Path(cfg.get("MODEL_DIR") or (STACK / "models" / "converted"))
    vae = Path(cfg["VAE_PATH"]) if "VAE_PATH" in cfg else None
    if not model_dir.exists() and not PATHS.is_file():
        report.add(
            CheckResult(
                "converted weights",
                False,
                "run ./1_setup_download.sh (not --deps-only)",
            )
        )
        return
    cmd = [sys.executable, str(VALIDATE), str(model_dir)]
    if vae is not None:
        cmd += ["--vae", str(vae)]
    if PATHS.is_file():
        cmd += ["--paths", str(PATHS)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    detail = (proc.stdout.strip() or proc.stderr.strip() or f"exit {proc.returncode}").splitlines()[-1]
    report.add(CheckResult("converted weights", ok, detail[:240]))


def check_live(report: Report, base: str, model: str, require: bool) -> int | None:
    """Return 2 if required live endpoint is down."""
    health_url = base.rstrip("/") + "/health"
    status, payload, raw = http_json(health_url, timeout=5)
    if status == 0:
        report.add(
            CheckResult(
                "GET /health",
                not require,
                raw or "unreachable",
                required=require,
            )
        )
        return 2 if require else None
    ok = status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
    report.add(CheckResult("GET /health", ok, f"HTTP {status}"))
    models_url = base.rstrip("/") + "/v1/models"
    status, payload, raw = http_json(models_url, timeout=5)
    ids: list[str] = []
    if isinstance(payload, dict):
        ids = [m.get("id", "") for m in payload.get("data") or [] if isinstance(m, dict)]
    if status == 200 and ids and model not in ids:
        report.add(CheckResult("GET /v1/models", False, f"expected {model}, got {ids}"))
    else:
        report.add(
            CheckResult(
                "GET /v1/models",
                status == 200,
                f"HTTP {status} ids={ids or ['?']}",
            )
        )
    return None


def run_live_generate(report: Report, base: str) -> None:
    request = json.loads(QUICKSTART.read_text())
    request["id"] = f"harness-{int(time.time())}"
    print("  (this generates the ~16s quickstart clip; can take several minutes)")
    status, payload, raw = http_json(
        base.rstrip("/") + "/generate",
        method="POST",
        body=request,
        timeout=1800,
    )
    if status != 200 or not isinstance(payload, dict):
        report.add(CheckResult("POST /generate quickstart", False, f"HTTP {status} {raw[:200]}"))
        return
    audio = payload.get("audio")
    ok = bool(audio) and Path(str(audio)).is_file()
    report.add(
        CheckResult(
            "POST /generate quickstart",
            ok,
            f"audio={audio} truncated={payload.get('truncated')}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="YuE2-3B harness")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--quick", action="store_true", help="alias for --offline")
    args = parser.parse_args()
    if args.quick:
        args.offline = True

    parsed = urlparse(args.base)
    if not parsed.scheme:
        args.base = "http://" + args.base

    report = Report()
    print("=== YuE2-3B harness ===")
    check_examples(report)
    check_engine(report)
    check_weights(report)

    if args.offline and not args.gate and not args.live:
        failed = report.failed_required()
        print()
        if failed:
            print(f"FAIL: {len(failed)} required check(s)")
            return 1
        print("PASS: offline checks")
        return 0

    require_live = args.gate or args.live
    live_rc = check_live(report, args.base, args.model, require=require_live)
    if live_rc == 2 and require_live:
        print()
        print(f"FAIL: no healthy endpoint at {args.base}")
        print("  Start with ./2_start_server.sh, or use --offline")
        return 2

    if args.live:
        run_live_generate(report, args.base)

    failed = report.failed_required()
    print()
    if failed:
        print(f"FAIL: {len(failed)} required check(s)")
        return 1
    print("PASS: harness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
