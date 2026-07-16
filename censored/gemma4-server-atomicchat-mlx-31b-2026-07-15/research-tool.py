#!/usr/bin/env python3
"""
research-tool — comprehensive doc / URL research helper for local LLMs

Designed for Pixel / GrapheneOS / AOSP research under tight context budgets.
Uses curl with a simple User-Agent (avoids Kilo/Chrome DevSite OAuth loops).

Commands (see --help and `research-tool.py help`):

  check       HTTP status + final URL (one or many)
  fetch       Download body (capped / cached / optional save)
  grep        Fetch URL(s) and search; agent-friendly snippets (not raw HTML)
  repos       Search android.googlesource.com repo index by name (use instead of grepping /)
  paths       Probe known kernel trees for a path/driver (dwc3, drivers/usb/…)
  resolve     Suggest corrected URL variants (no network, or --probe)
  probe       Try variants until one returns 200; stop spraying after hits
  bulletin    Build + optionally check AOSP / Pixel security bulletin URLs
  links       Extract hrefs from a page (optional filter)
  cves        Extract CVE-YYYY-NNNNN IDs from a page (never invent)
  title       HTTP meta: status, final URL, content-type, rough title
  local       Search project files first (rg; context-cheap)
  find        Compact local + optional remote (+ optional googlesource repos)
  known       Print curated known-good URLs for this project
  cache       list | clear | path — on-disk fetch cache
  multi       Batch: check several URLs → compact table
  text        Fetch and strip HTML → plain text (capped)
  suggest     Given a bad URL or topic keyword, suggest where to look

Prefer this tool over raw curl for all HTTP research.

Env:
  RESEARCH_UA          User-Agent (default: Mozilla/5.0)
  RESEARCH_TIMEOUT     curl max-time seconds (default: 30)
  RESEARCH_CACHE_DIR   cache directory (default: .research-cache/)
  RESEARCH_MAX_BYTES   default body cap for stdout (default: 120000)
  RESEARCH_PROJECT     project root for `local` (default: script dir)
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

VERSION = "1.3.0"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_UA = os.environ.get("RESEARCH_UA", "Mozilla/5.0")
DEFAULT_TIMEOUT = int(os.environ.get("RESEARCH_TIMEOUT", "30"))
DEFAULT_MAX_BYTES = int(os.environ.get("RESEARCH_MAX_BYTES", "120000"))
CACHE_DIR = Path(
    os.environ.get("RESEARCH_CACHE_DIR", str(SCRIPT_DIR / ".research-cache"))
)
PROJECT_ROOT = Path(os.environ.get("RESEARCH_PROJECT", str(SCRIPT_DIR)))
MAX_REDIRS = 10
MAX_PROBE_ATTEMPTS = 8  # hard stop: do not spray sibling URLs
DEFAULT_SNIPPET = int(os.environ.get("RESEARCH_SNIPPET", "160"))
DEFAULT_GREP_MATCHES = 20
GOOGLESURCE_ROOT = "https://android.googlesource.com/"

# Trees to probe for in-tree driver paths (not repo-name search).
DEFAULT_KERNEL_TREES: list[tuple[str, list[str]]] = [
    ("kernel/common", ["HEAD", "android15-6.6", "android14-6.1", "android-mainline"]),
    ("device/google/shusky-kernels/6.1", ["HEAD"]),
    ("device/google/shusky-kernels/5.15", ["HEAD"]),
    ("device/google/shusky-kernel", ["HEAD"]),
    ("kernel/devices/google/shusky", ["HEAD"]),
    ("kernel/gs", ["HEAD"]),
]

# Short subsystem names → likely in-tree paths
PATH_ALIASES: dict[str, list[str]] = {
    "dwc3": ["drivers/usb/dwc3", "drivers/usb/dwc3/core.c", "drivers/usb/dwc3/gadget.c"],
    "xhci": ["drivers/usb/host/xhci-hcd.c", "drivers/usb/host/xhci.c", "drivers/usb/host"],
    "gadget": ["drivers/usb/gadget", "drivers/usb/gadget/udc"],
    "typec": ["drivers/usb/typec", "drivers/usb/typec/tcpm"],
    "usb": ["drivers/usb", "drivers/usb/core", "drivers/usb/host", "drivers/usb/gadget"],
    "fbe": ["fs/crypto", "fs/f2fs"],
    "f2fs": ["fs/f2fs"],
    "ufshcd": ["drivers/ufs", "drivers/ufs/core"],
    "ufs": ["drivers/ufs", "drivers/scsi/ufs"],
    "keymint": ["trusty", "drivers/trusty"],
}

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
HREF_RE = re.compile(
    r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE
)
TITLE_RE = re.compile(
    r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL
)

# Curated known-good URLs for husky / GrapheneOS research (prefer local docs too)
KNOWN_URLS: dict[str, list[tuple[str, str]]] = {
    "bulletins_pixel": [
        (
            "Pixel Feb 2026 bulletin",
            "https://source.android.com/docs/security/bulletin/pixel/2026/2026-02-01",
        ),
        (
            "Pixel bulletin index",
            "https://source.android.com/docs/security/bulletin/pixel",
        ),
    ],
    "bulletins_aosp": [
        (
            "AOSP ASB index",
            "https://source.android.com/docs/security/bulletin",
        ),
        (
            "AOSP ASB 2023-10-01 (history)",
            "https://source.android.com/docs/security/bulletin/2023-10-01",
        ),
        (
            "AOSP ASB 2023-11-01 (history)",
            "https://source.android.com/docs/security/bulletin/2023-11-01",
        ),
    ],
    "security": [
        (
            "Verified Boot / AVB",
            "https://source.android.com/docs/security/features/verifiedboot/avb",
        ),
        (
            "Encryption overview",
            "https://source.android.com/docs/security/features/encryption",
        ),
        (
            "GrapheneOS security model",
            "https://grapheneos.org/features#exploit-protection",
        ),
        (
            "GrapheneOS releases",
            "https://grapheneos.org/releases",
        ),
        (
            "GrapheneOS FAQ",
            "https://grapheneos.org/faq",
        ),
    ],
    "devices": [
        (
            "Device codenames / build numbers",
            "https://source.android.com/docs/setup/reference/build-numbers",
        ),
        (
            "AOSP building (mentions aosp_husky)",
            "https://source.android.com/docs/setup/build/building",
        ),
        (
            "Factory images (Pixel)",
            "https://developers.google.com/android/images",
        ),
        (
            "GrapheneOS install (Pixel 8 Pro)",
            "https://grapheneos.org/install/cli",
        ),
        (
            "GrapheneOS releases",
            "https://grapheneos.org/releases",
        ),
    ],
    "kernel": [
        (
            "Android common kernel",
            "https://android.googlesource.com/kernel/common/",
        ),
        (
            "Pixel kernel (gs-google)",
            "https://android.googlesource.com/kernel/gs/",
        ),
    ],
    "nvd": [
        (
            "NVD search",
            "https://nvd.nist.gov/vuln/search",
        ),
    ],
}

# Topic → candidate URLs for `find` / `suggest`
TOPIC_CANDIDATES: dict[str, list[str]] = {
    "husky": [
        "https://source.android.com/docs/setup/reference/build-numbers",
        "https://source.android.com/docs/setup/build/building",
        "https://grapheneos.org/releases",
        "https://developers.google.com/android/images",
        "https://grapheneos.org/install/cli",
        "https://android.googlesource.com/device/google/shusky/",
        "https://android.googlesource.com/device/google/shusky-kernels/6.1/",
        "https://android.googlesource.com/kernel/devices/google/shusky/",
        "https://android.googlesource.com/kernel/gs/",
        "https://android.googlesource.com/kernel/common/",
    ],
    "shusky": [
        "https://android.googlesource.com/device/google/shusky/",
        "https://android.googlesource.com/device/google/shusky-kernel/",
        "https://android.googlesource.com/device/google/shusky-kernels/6.1/",
        "https://android.googlesource.com/kernel/devices/google/shusky/",
        "https://source.android.com/docs/setup/reference/build-numbers",
    ],
    "pixel8": [
        "https://source.android.com/docs/setup/reference/build-numbers",
        "https://source.android.com/docs/setup/build/building",
        "https://grapheneos.org/releases",
        "https://developers.google.com/android/images",
    ],
    "pixel 8 pro": [
        "https://source.android.com/docs/setup/reference/build-numbers",
        "https://source.android.com/docs/setup/build/building",
        "https://grapheneos.org/releases",
        "https://developers.google.com/android/images",
    ],
    "bulletin": [
        "https://source.android.com/docs/security/bulletin",
        "https://source.android.com/docs/security/bulletin/pixel",
        "https://source.android.com/docs/security/bulletin/pixel/2026/2026-02-01",
    ],
    "fbe": [
        "https://source.android.com/docs/security/features/encryption",
        "https://source.android.com/docs/security/features/encryption/file-based",
    ],
    "avb": [
        "https://source.android.com/docs/security/features/verifiedboot/avb",
    ],
    "fastboot": [
        "https://source.android.com/docs/core/architecture/bootloader/fastboot",
        "https://android.googlesource.com/platform/system/core/+/main/fastboot/",
    ],
}

# Local project files agents should read first
LOCAL_PRIORITY = [
    "STATUS.md",
    "FINAL-SUMMARY.md",
    "PIN-RECOVERY-STRATEGY.md",
    "RESEARCH-FETCH-AND-CVE-NOTES.md",
    "EXPLOIT-PATH-VS-PHOTOS.md",
    "QUICK-REFERENCE.md",
    "AGENTS.md",
    "README.md",
]


# ---------------------------------------------------------------------------
# HTTP via curl
# ---------------------------------------------------------------------------

def _curl_base(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    return [
        "curl",
        "-sS",
        "-A",
        DEFAULT_UA,
        "--max-time",
        str(timeout),
        "--compressed",
        "-L",
        "--max-redirs",
        str(MAX_REDIRS),
    ]


def http_check(
    url: str, timeout: int = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """Return http_code, url_effective, content_type, size (from headers only)."""
    fmt = (
        "http_code=%{http_code}\\n"
        "url_effective=%{url_effective}\\n"
        "content_type=%{content_type}\\n"
        "size_download=%{size_download}\\n"
        "time_total=%{time_total}\\n"
        "redirect_url=%{redirect_url}\\n"
    )
    cmd = _curl_base(timeout) + [
        "-o",
        "/dev/null",
        "-w",
        fmt,
        # Prefer HEAD; some hosts mishandle HEAD → fall back handled by caller
        "-I",
        url,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return {"ok": False, "error": "curl not found", "url": url}

    data: dict[str, Any] = {"url": url, "ok": False}
    for line in (out.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    code = str(data.get("http_code", "0"))
    data["http_code"] = code
    data["ok"] = code == "200"
    if out.returncode != 0 and not data.get("http_code"):
        data["error"] = (out.stderr or "").strip() or f"curl exit {out.returncode}"
    return data


def http_fetch(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    use_cache: bool = True,
    max_age: int = 86400,
) -> dict[str, Any]:
    """
    Fetch body via curl. Returns dict with ok, http_code, body (bytes),
    url_effective, from_cache, path.
    """
    cache_path = _cache_path(url)
    if use_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        meta_path = cache_path.with_suffix(".meta.json")
        if age <= max_age and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                body = cache_path.read_bytes()
                return {
                    "ok": meta.get("http_code") == "200",
                    "http_code": str(meta.get("http_code", "0")),
                    "url_effective": meta.get("url_effective", url),
                    "body": body,
                    "from_cache": True,
                    "cache_path": str(cache_path),
                    "url": url,
                }
            except (OSError, json.JSONDecodeError):
                pass

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()
    fmt = "http_code=%{http_code}\\nurl_effective=%{url_effective}\\n"
    cmd = _curl_base(timeout) + ["-o", tmp_path, "-w", fmt, url]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        os.unlink(tmp_path)
        return {"ok": False, "error": "curl not found", "url": url, "body": b""}

    meta: dict[str, str] = {}
    for line in (out.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

    try:
        body = Path(tmp_path).read_bytes()
    except OSError:
        body = b""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    code = meta.get("http_code", "0")
    result = {
        "ok": code == "200",
        "http_code": code,
        "url_effective": meta.get("url_effective", url),
        "body": body,
        "from_cache": False,
        "url": url,
        "curl_stderr": (out.stderr or "").strip(),
    }

    if use_cache and code == "200" and body:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(body)
            cache_path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "url": url,
                        "http_code": code,
                        "url_effective": result["url_effective"],
                        "fetched_at": time.time(),
                        "bytes": len(body),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            result["cache_path"] = str(cache_path)
        except OSError:
            pass

    return result


def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    # short readable suffix
    host = urlparse(url).netloc.replace(".", "_")[:40]
    return CACHE_DIR / f"{host}_{h}.body"


# ---------------------------------------------------------------------------
# URL resolution / bulletin schema
# ---------------------------------------------------------------------------

def warn_url_hygiene(url: str) -> list[str]:
    warns: list[str] = []
    if "/docs/security/bulletins/" in url:
        warns.append(
            "path uses 'bulletins' (plural) — usually 404; prefer singular 'bulletin'"
        )
    if re.search(r"/docs/security/bulletin/\d{4}-\d{2}$", url):
        warns.append(
            "missing day in bulletin date — prefer YYYY-MM-01 (e.g. 2023-10-01)"
        )
    if "source.android.com" in url or "developer.android.com" in url:
        warns.append(
            "DevSite: use this tool or curl -A 'Mozilla/5.0' — not Kilo webfetch"
        )
    return warns


def bulletin_urls(kind: str, year_month: str) -> list[str]:
    """
    kind: aosp | pixel | both
    year_month: YYYY-MM or YYYY-MM-DD
    """
    m = re.match(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$", year_month.strip())
    if not m:
        raise ValueError(f"bad date '{year_month}'; want YYYY-MM or YYYY-MM-01")
    y, mo, day = m.group(1), m.group(2), m.group(3) or "01"
    full = f"{y}-{mo}-{day}"
    aosp = f"https://source.android.com/docs/security/bulletin/{full}"
    pixel = (
        f"https://source.android.com/docs/security/bulletin/pixel/{y}/{full}"
    )
    kind = kind.lower()
    if kind == "aosp":
        return [aosp]
    if kind == "pixel":
        return [pixel]
    return [pixel, aosp]


def resolve_variants(url: str) -> list[dict[str, str]]:
    """
    Generate likely corrected URL variants from a wrong/guessed URL.
    Does not network. Ordered: most likely first. Deduped.

    Applies transform rules iteratively so chained fixes work, e.g.
    …/bulletins/2023-10 → …/bulletin/2023-10 → …/bulletin/2023-10-01
    """
    variants: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(u: str, reason: str) -> bool:
        u = u.strip()
        if not u or u in seen:
            return False
        seen.add(u)
        variants.append({"url": u, "reason": reason})
        return True

    add(url.strip(), "original")

    # BFS-style: expand each known URL with pure transforms (bounded)
    i = 0
    while i < len(variants) and len(variants) < 24:
        u = variants[i]["url"]
        i += 1

        # bulletins → bulletin
        if "/bulletins/" in u:
            add(u.replace("/bulletins/", "/bulletin/"), "singular bulletin")

        # /bulletin/YYYY-MM → /bulletin/YYYY-MM-01
        m = re.search(
            r"(https?://[^/]+/docs/security/bulletin)/(\d{4}-\d{2})/?$", u
        )
        if m:
            add(f"{m.group(1)}/{m.group(2)}-01", "add day -01")

        # /bulletin/YYYY-MM-DD → Pixel path + indexes
        m = re.search(
            r"https?://source\.android\.com/docs/security/bulletin/"
            r"(\d{4})-(\d{2})-(\d{2})/?$",
            u,
        )
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            add(
                f"https://source.android.com/docs/security/bulletin/pixel/{y}/{y}-{mo}-{d}",
                "Pixel bulletin path",
            )
            add(
                "https://source.android.com/docs/security/bulletin/pixel",
                "Pixel bulletin index",
            )
            add(
                "https://source.android.com/docs/security/bulletin",
                "AOSP bulletin index",
            )

        # /bulletin/pixel/YYYY-MM-01 missing year folder
        m = re.search(
            r"https?://source\.android\.com/docs/security/bulletin/pixel/"
            r"(\d{4}-\d{2}-\d{2})/?$",
            u,
        )
        if m:
            full = m.group(1)
            y = full[:4]
            add(
                f"https://source.android.com/docs/security/bulletin/pixel/{y}/{full}",
                "Pixel path needs /pixel/YYYY/YYYY-MM-01",
            )

        # building-devices vs building
        if "building-devices" in u:
            add(
                u.replace("building-devices", "building"),
                "building-devices often wrong; try building",
            )

        # common codename / build-numbers landing
        if "build/building" in u or "building-devices" in u:
            add(
                "https://source.android.com/docs/setup/reference/build-numbers",
                "device codenames live under build-numbers",
            )

        # googlesource slash variants
        if "googlesource.com" in u and not u.endswith("/"):
            add(u + "/", "googlesource trailing slash")
        if "googlesource.com" in u and u.endswith("/") and u.rstrip("/").count("/") >= 3:
            add(u.rstrip("/"), "googlesource no trailing slash")
        # googlesource often needs /+/HEAD or /+/refs/heads/main for browse
        if "googlesource.com" in u and "/+/" not in u:
            base = u.rstrip("/")
            add(base + "/+/HEAD", "googlesource /+/HEAD")
            add(base + "/+/refs/heads/main", "googlesource main ref")

        if u.startswith("http://"):
            add("https://" + u[len("http://") :], "force https")

        if "developer.android.com/docs" in u:
            add(
                u.replace("developer.android.com", "source.android.com"),
                "try source.android.com host",
            )

        # strip accidental trailing punctuation from LLM copy-paste
        if u[-1:] in ".,;)":
            add(u.rstrip(".,;)"), "strip trailing punctuation")

    return variants


def probe_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_attempts: int = MAX_PROBE_ATTEMPTS,
) -> dict[str, Any]:
    """Try original + variants until 200 or max_attempts."""
    variants = resolve_variants(url)
    attempts: list[dict[str, Any]] = []
    for i, v in enumerate(variants):
        if i >= max_attempts:
            break
        for w in warn_url_hygiene(v["url"]):
            pass  # collected in output via variants
        r = http_check(v["url"], timeout=timeout)
        entry = {
            "url": v["url"],
            "reason": v["reason"],
            "http_code": r.get("http_code"),
            "url_effective": r.get("url_effective"),
            "ok": r.get("ok", False),
        }
        attempts.append(entry)
        if r.get("ok"):
            return {
                "found": True,
                "winner": entry,
                "attempts": attempts,
                "stopped_early": True,
            }
    return {
        "found": False,
        "winner": None,
        "attempts": attempts,
        "stopped_early": False,
        "hint": (
            "No variant returned 200. Stop inventing sibling paths; "
            "use `known`, `bulletin`, or project RESEARCH-FETCH-AND-CVE-NOTES.md."
        ),
    }


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        ad = dict(attrs)
        if tag == "a" and ad.get("href"):
            self.hrefs.append(ad["href"] or "")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)


def strip_html(raw: str) -> str:
    # remove scripts/styles
    raw = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def extract_readable_page(body: bytes, max_chars: int = 40000) -> tuple[str, str]:
    """
    Agent-friendly page text: prefer <article>/<main>/devsite body, strip chrome.
    Returns (title, plain_text).
    """
    try:
        raw = body.decode("utf-8", errors="replace")
    except Exception:
        return "", ""
    title = ""
    tm = TITLE_RE.search(raw)
    if tm:
        title = re.sub(r"\s+", " ", html.unescape(tm.group(1))).strip()

    chunk = raw
    for pat in (
        r'(?is)<div[^>]+class="[^"]*devsite-article-body[^"]*"[^>]*>(.*)</div>\s*(?:<footer|</article|$)',
        r'(?is)<article[^>]*>(.*?)</article>',
        r'(?is)<main[^>]*>(.*?)</main>',
        r'(?is)id=["\']main-content["\'][^>]*>(.*)',
        r'(?is)<div[^>]+itemprop=["\']articleBody["\'][^>]*>(.*?)</div>',
    ):
        mm = re.search(pat, raw)
        if mm and len(mm.group(1)) > 400:
            chunk = mm.group(1)
            break

    plain = strip_html(chunk)
    # drop ultra-common chrome leftovers
    drop_lines = re.compile(
        r"(?i)^(skip to main content|android open source project|search|"
        r"appearance|sign in|contents|on this page|was this helpful)\s*$"
    )
    lines = [ln for ln in plain.splitlines() if not drop_lines.match(ln.strip())]
    plain = "\n".join(lines)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
    if len(plain) > max_chars:
        plain = plain[:max_chars] + f"\n\n… truncated readable text at {max_chars} chars. Prefer: research-tool grep PATTERN URL"
    return title, plain



def extract_title(body: bytes) -> str:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return ""
    m = TITLE_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip()
    return ""


def extract_links(body: bytes, base_url: str, pattern: Optional[str] = None) -> list[str]:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return []
    hrefs = HREF_RE.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    cre = re.compile(pattern, re.IGNORECASE) if pattern else None
    for h in hrefs:
        if h.startswith("#") or h.startswith("javascript:"):
            continue
        abs_u = urljoin(base_url, h)
        if cre and not cre.search(abs_u) and not cre.search(h):
            continue
        if abs_u not in seen:
            seen.add(abs_u)
            out.append(abs_u)
    return out


def extract_cves(body: bytes) -> list[str]:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return []
    found = CVE_RE.findall(text)
    # normalize case, preserve order unique
    seen: set[str] = set()
    out: list[str] = []
    for c in found:
        c2 = c.upper()
        if c2 not in seen:
            seen.add(c2)
            out.append(c2)
    return out


def looks_like_html(body: bytes) -> bool:
    head = body[:800].lower()
    return b"<!doctype html" in head or b"<html" in head or b"<head" in head


def soft_break_minified(text: str) -> str:
    """Turn minified HTML into greppable lines without full strip."""
    if text.count("\n") >= 20:
        return text
    # break after tags / between repo list items
    text = re.sub(r">\s*<", ">\n<", text)
    text = re.sub(r"</a>", "</a>\n", text)
    return text


def snippet_around(
    line: str, cre: re.Pattern[str], width: int = DEFAULT_SNIPPET
) -> list[str]:
    out: list[str] = []
    for m in cre.finditer(line):
        half = max(24, width // 2)
        start = max(0, m.start() - half)
        end = min(len(line), m.end() + half)
        snip = line[start:end]
        # collapse whitespace for readability
        snip = re.sub(r"\s+", " ", snip).strip()
        if start > 0:
            snip = "…" + snip
        if end < len(line):
            snip = snip + "…"
        if snip not in out:
            out.append(snip)
        if len(out) >= 8:
            break
    return out


def grep_bytes(
    body: bytes,
    pattern: str,
    context: int = 2,
    max_matches: int = DEFAULT_GREP_MATCHES,
    ignore_case: bool = True,
    plain: bool = False,
    snippet_width: int = DEFAULT_SNIPPET,
    auto_html: bool = True,
) -> list[dict[str, Any]]:
    """
    Grep body for pattern. Agent-friendly defaults:
    - HTML auto-detected → strip tags (plain) + soft-break minified pages
    - Long lines → short snippets around the match (not 50KB lines)
    """
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return []

    htmlish = looks_like_html(body)
    if plain or (auto_html and htmlish):
        # keep structure-ish breaks then strip tags
        text = soft_break_minified(text)
        text = strip_html(text)
    elif htmlish:
        text = soft_break_minified(text)

    flags = re.IGNORECASE if ignore_case else 0
    try:
        cre = re.compile(pattern, flags)
    except re.error as e:
        return [{"error": f"bad regex: {e}"}]

    lines = text.splitlines()
    # If still a tiny number of huge lines, force character-window search
    if len(lines) <= 3 and any(len(L) > 2000 for L in lines):
        blob = "\n".join(lines)
        matches: list[dict[str, Any]] = []
        for sn in snippet_around(blob, cre, snippet_width):
            matches.append(
                {
                    "line": 1,
                    "match": sn,
                    "snippet": sn,
                    "context": [{"line": 1, "text": sn}],
                }
            )
            if len(matches) >= max_matches:
                break
        return matches

    matches = []
    for i, line in enumerate(lines):
        if not cre.search(line):
            continue
        if len(line) > snippet_width * 2:
            snips = snippet_around(line, cre, snippet_width)
            for sn in snips:
                matches.append(
                    {
                        "line": i + 1,
                        "match": sn,
                        "snippet": sn,
                        "context": [{"line": i + 1, "text": sn}],
                    }
                )
                if len(matches) >= max_matches:
                    return matches
        else:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            ctx = []
            for j in range(start, end):
                t = lines[j]
                if len(t) > snippet_width * 2:
                    t = t[: snippet_width] + "…"
                ctx.append({"line": j + 1, "text": t})
            matches.append(
                {
                    "line": i + 1,
                    "match": line[:snippet_width],
                    "snippet": line[:snippet_width],
                    "context": ctx,
                }
            )
            if len(matches) >= max_matches:
                break
    return matches


REPO_ITEM_RE = re.compile(
    r'class="RepoList-item"\s+href="([^"]+)"[^>]*>\s*'
    r'<span class="RepoList-itemName">([^<]*)</span>\s*'
    r'(?:<span class="RepoList-itemDescription">([^<]*)</span>)?',
    re.IGNORECASE,
)


def extract_googlesource_repos(
    body: bytes, base_url: str = GOOGLESURCE_ROOT
) -> list[dict[str, str]]:
    """Parse android.googlesource.com repo list page into name/url/description."""
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in REPO_ITEM_RE.finditer(text):
        href, name, desc = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
        name = html.unescape(name)
        desc = html.unescape(desc)
        url = urljoin(base_url, href)
        if name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "url": url, "description": desc})
    # Fallback: any path-like href under root that looks like a repo
    if not out:
        for href in HREF_RE.findall(text):
            if not href.startswith("/") or href.startswith("/+/") or href.startswith("/#"):
                continue
            # /kernel/common/ style
            if href.count("/") < 2:
                continue
            name = href.strip("/")
            if not name or name in seen:
                continue
            if any(x in name for x in ("static", "accounts", "+")):
                continue
            seen.add(name)
            out.append(
                {
                    "name": name,
                    "url": urljoin(base_url, href if href.endswith("/") else href + "/"),
                    "description": "",
                }
            )
    return out


def filter_repos(
    repos: list[dict[str, str]], pattern: str, ignore_case: bool = True
) -> list[dict[str, str]]:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        cre = re.compile(pattern, flags)
    except re.error:
        cre = re.compile(re.escape(pattern), flags)
    hits = []
    for r in repos:
        blob = f"{r.get('name','')} {r.get('description','')} {r.get('url','')}"
        if cre.search(blob):
            hits.append(r)
    return hits


def is_googlesource_index(url: str) -> bool:
    p = urlparse(url)
    if "googlesource.com" not in p.netloc:
        return False
    path = (p.path or "/").rstrip("/")
    return path == "" or path == "/"



def googlesource_tree_url(repo: str, ref: str, rel_path: str = "") -> str:
    """Build a Gitiles tree/blob URL for android.googlesource.com."""
    repo = repo.strip("/")
    ref = (ref or "HEAD").strip("/")
    if ref != "HEAD" and not ref.startswith("refs/"):
        ref = f"refs/heads/{ref}"
    base = f"https://android.googlesource.com/{repo}/+/{ref}"
    rel_path = (rel_path or "").strip("/")
    if not rel_path:
        return base + "/"
    # files keep extension; directories get trailing slash for Gitiles
    if re.search(r"\.[a-zA-Z0-9]{1,8}$", rel_path):
        return f"{base}/{rel_path}"
    return f"{base}/{rel_path}/"


def expand_path_queries(query: str) -> list[str]:
    """Map dwc3 / drivers/usb/dwc3 → concrete relative paths to probe."""
    q = query.strip().strip("/")
    if not q:
        return []
    out: list[str] = []
    low = q.lower()
    if low in PATH_ALIASES:
        out.extend(PATH_ALIASES[low])
    # also treat query itself as a path
    if "/" in q or q.endswith((".c", ".h", ".S")):
        out.insert(0, q)
    elif low not in PATH_ALIASES:
        # bare token: try as leaf under common roots later + as path
        out.append(q)
        out.append(f"drivers/usb/{q}")
        out.append(f"drivers/{q}")
    # unique preserve order
    seen: set[str] = set()
    uniq = []
    for p in out:
        p2 = p.strip("/")
        if p2 and p2 not in seen:
            seen.add(p2)
            uniq.append(p2)
    return uniq


def parse_gitiles_json(body: bytes) -> dict[str, Any]:
    """Gitiles JSON is prefixed with )]}' anti-XSSI line."""
    try:
        text = body.decode("utf-8", errors="replace").lstrip()
    except Exception:
        return {}
    if text.startswith(")]}'"):
        text = text[4:].lstrip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def list_gitiles_dir(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    use_cache: bool = True,
    max_age: int = 86400,
) -> dict[str, Any]:
    """GET ?format=JSON for a tree URL; return names/types."""
    sep = "&" if "?" in url else "?"
    jurl = f"{url.rstrip('/')}{sep}format=JSON"
    r = http_fetch(jurl, timeout=timeout, use_cache=use_cache, max_age=max_age)
    if not r.get("ok"):
        return {"ok": False, "http_code": r.get("http_code"), "entries": [], "url": jurl}
    data = parse_gitiles_json(r["body"])
    entries = []
    for e in data.get("entries") or []:
        if isinstance(e, dict) and e.get("name"):
            entries.append(
                {
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "mode": e.get("mode"),
                }
            )
    return {
        "ok": True,
        "http_code": r.get("http_code"),
        "entries": entries,
        "url": jurl,
        "id": data.get("id"),
    }


def probe_kernel_paths(
    query: str,
    trees: Optional[list[tuple[str, list[str]]]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    list_dir: bool = True,
    max_hits: int = 40,
) -> dict[str, Any]:
    """
    Probe DEFAULT_KERNEL_TREES for paths matching query (e.g. dwc3).
    Returns hits with http_code and optional directory listing.
    """
    paths = expand_path_queries(query)
    trees = trees or DEFAULT_KERNEL_TREES
    attempts: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for repo, refs in trees:
        for ref in refs:
            for rel in paths:
                url = googlesource_tree_url(repo, ref, rel)
                r = http_check(url, timeout=timeout)
                entry = {
                    "repo": repo,
                    "ref": ref,
                    "path": rel,
                    "url": url,
                    "http_code": r.get("http_code"),
                    "ok": r.get("ok", False),
                }
                attempts.append(entry)
                if r.get("ok"):
                    if list_dir and not re.search(r"\.[a-zA-Z0-9]{1,8}$", rel):
                        listing = list_gitiles_dir(url, timeout=timeout)
                        if listing.get("ok"):
                            entry["entries"] = [
                                e["name"] for e in (listing.get("entries") or [])[:40]
                            ]
                            entry["entry_count"] = len(listing.get("entries") or [])
                    hits.append(entry)
                    if len(hits) >= max_hits:
                        return {
                            "query": query,
                            "paths": paths,
                            "hits": hits,
                            "attempts": attempts,
                            "truncated": True,
                        }
    return {
        "query": query,
        "paths": paths,
        "hits": hits,
        "attempts": attempts,
        "truncated": False,
    }


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def local_search(
    pattern: str,
    path: Optional[Path] = None,
    glob: Optional[str] = None,
    max_matches: int = 40,
    context: int = 1,
) -> dict[str, Any]:
    root = path or PROJECT_ROOT
    if shutil.which("rg"):
        cmd = [
            "rg",
            "--color",
            "never",
            "-n",
            "-i",
            f"-C{context}",
            "--max-count",
            str(max_matches),
            "-e",
            pattern,
        ]
        if glob:
            cmd.extend(["--glob", glob])
        # skip heavy / irrelevant trees
        for g in (
            "!.research-cache",
            "!archive/vendor-src-snapshots/**",
            "!**/.git/**",
            "!**/__pycache__/**",
            "!*.img",
            "!*.apk",
        ):
            cmd.extend(["--glob", g])
        cmd.append(str(root))
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "tool": "rg",
            "pattern": pattern,
            "exit": p.returncode,
            "stdout": (p.stdout or "")[:80000],
            "stderr": (p.stderr or "")[:2000],
            "root": str(root),
        }

    # fallback: pure python walk of markdown/text
    hits: list[str] = []
    cre = re.compile(pattern, re.IGNORECASE)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", ".research-cache", "__pycache__", "node_modules"}
            and "vendor-src-snapshots" not in dirpath
        ]
        for fn in filenames:
            if not fn.endswith((".md", ".txt", ".sh", ".py", ".json")):
                continue
            fp = Path(dirpath) / fn
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if cre.search(line):
                    rel = fp.relative_to(root)
                    hits.append(f"{rel}:{i+1}:{line[:200]}")
                    if len(hits) >= max_matches:
                        return {
                            "tool": "python",
                            "pattern": pattern,
                            "stdout": "\n".join(hits),
                            "root": str(root),
                        }
    return {
        "tool": "python",
        "pattern": pattern,
        "stdout": "\n".join(hits) if hits else "(no matches)",
        "root": str(root),
    }


def local_priority_hint(pattern: str) -> list[str]:
    """Which priority docs mention the pattern (names only)."""
    cre = re.compile(pattern, re.IGNORECASE)
    hits: list[str] = []
    for name in LOCAL_PRIORITY:
        fp = PROJECT_ROOT / name
        if not fp.is_file():
            # also check archive/
            fp2 = PROJECT_ROOT / "archive" / name
            if fp2.is_file():
                fp = fp2
            else:
                continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if cre.search(text):
            hits.append(str(fp.relative_to(PROJECT_ROOT)))
    return hits


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def emit(data: Any, as_json: bool = False, plain_text: Optional[str] = None) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif plain_text is not None:
        print(plain_text)
    else:
        # human compact
        if isinstance(data, dict):
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        else:
            print(data)


def print_check_table(rows: list[dict[str, Any]]) -> str:
    lines = ["CODE  OK   URL", "-" * 72]
    for r in rows:
        code = str(r.get("http_code", "?"))
        ok = "yes" if r.get("ok") else "no"
        url = r.get("url_effective") or r.get("url") or ""
        lines.append(f"{code:4}  {ok:3}  {url}")
        if r.get("url") and r.get("url_effective") and r["url"] != r.get("url_effective"):
            lines.append(f"           requested: {r['url']}")
    return "\n".join(lines)


def truncate_body(body: bytes, max_bytes: int) -> tuple[bytes, bool]:
    if len(body) <= max_bytes:
        return body, False
    return body[:max_bytes], True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    rows = []
    rc = 0
    for url in args.urls:
        for w in warn_url_hygiene(url):
            print(f"WARN: {w}", file=sys.stderr)
        r = http_check(url, timeout=args.timeout)
        rows.append(r)
        if not r.get("ok"):
            rc = 1
    if args.json:
        emit(rows, as_json=True)
    else:
        print(print_check_table(rows))
        for r in rows:
            if r.get("content_type"):
                print(f"# content_type={r.get('content_type')} time={r.get('time_total')}s")
    return rc


def cmd_multi(args: argparse.Namespace) -> int:
    return cmd_check(args)  # same; alias


def cmd_fetch(args: argparse.Namespace) -> int:
    """
    Fetch a URL for agents.

    Default for HTML: readable plain text (main content), NOT raw DevSite chrome.
    Use --raw for original HTML bytes; --out FILE always writes full body.
    Prefer grep/repos when searching for a token.
    """
    url = args.url
    for w in warn_url_hygiene(url):
        print(f"WARN: {w}", file=sys.stderr)
    r = http_fetch(
        url,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        max_age=args.cache_max_age,
    )
    if not r.get("ok"):
        print(
            f"ERROR: http_code={r.get('http_code')} url={url} "
            f"effective={r.get('url_effective')} {r.get('error','')}",
            file=sys.stderr,
        )
        if args.json:
            emit({k: v for k, v in r.items() if k != "body"}, as_json=True)
        return 1

    body: bytes = r["body"]
    if args.out:
        Path(args.out).write_bytes(body)
        print(
            f"wrote {args.out} ({len(body)} bytes) http={r['http_code']} "
            f"cache={r.get('from_cache', False)}",
            file=sys.stderr,
        )
        if args.json:
            emit(
                {
                    "ok": True,
                    "http_code": r["http_code"],
                    "bytes": len(body),
                    "out": args.out,
                    "url_effective": r.get("url_effective"),
                    "from_cache": r.get("from_cache"),
                },
                as_json=True,
            )
        return 0

    want_raw = getattr(args, "raw", False)
    htmlish = looks_like_html(body)
    # Agent default: never dump raw HTML chrome to stdout
    if htmlish and not want_raw:
        max_chars = args.max_bytes if args.max_bytes != DEFAULT_MAX_BYTES else 40000
        if getattr(args, "plain", False) or True:
            title, plain = extract_readable_page(body, max_chars=max_chars)
        meta = (
            f"# http={r.get('http_code')} title={title!r} bytes={len(body)} "
            f"mode=readable\n"
            f"# Prefer: {sys.argv[0]} grep PATTERN '{url}'\n"
            f"# Full HTML: {sys.argv[0]} fetch --raw --out /tmp/page.html '{url}'\n"
        )
        if args.json:
            emit(
                {
                    "ok": True,
                    "http_code": r["http_code"],
                    "title": title,
                    "text": plain,
                    "mode": "readable",
                    "bytes": len(body),
                    "from_cache": r.get("from_cache"),
                },
                as_json=True,
            )
        else:
            sys.stdout.write(meta)
            sys.stdout.write(plain)
            if not plain.endswith("\n"):
                sys.stdout.write("\n")
        if r.get("from_cache"):
            print(f"# from_cache={r.get('cache_path')}", file=sys.stderr)
        return 0

    chunk, truncated = truncate_body(body, args.max_bytes)
    if args.plain:
        out_text = strip_html(chunk.decode("utf-8", errors="replace"))
        if args.json:
            emit(
                {
                    "ok": True,
                    "http_code": r["http_code"],
                    "text": out_text,
                    "truncated": truncated,
                    "from_cache": r.get("from_cache"),
                },
                as_json=True,
            )
        else:
            sys.stdout.write(out_text)
            if not out_text.endswith("\n"):
                sys.stdout.write("\n")
    else:
        sys.stdout.buffer.write(chunk)
    if truncated:
        print(
            f"\n… truncated at {args.max_bytes} bytes "
            f"(full={len(body)}). Use --out FILE for full body.",
            file=sys.stderr,
        )
    if r.get("from_cache"):
        print(f"# from_cache={r.get('cache_path')}", file=sys.stderr)
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    pattern = args.pattern
    urls = args.urls
    any_hit = False
    report: list[dict[str, Any]] = []
    snippet_w = getattr(args, "snippet", DEFAULT_SNIPPET)
    # default agent mode: plain HTML + snippets (disable with --raw-lines)
    use_plain = args.plain or not getattr(args, "raw_lines", False)
    raw_lines = getattr(args, "raw_lines", False)
    if raw_lines:
        use_plain = args.plain

    for url in urls:
        for w in warn_url_hygiene(url):
            print(f"WARN: {w}", file=sys.stderr)

        # Googlesource root is a repo index — prefer repos filter, not HTML dump
        if is_googlesource_index(url) and not getattr(args, "force_html", False):
            r = http_fetch(
                url if url.endswith("/") else url + "/",
                timeout=args.timeout,
                use_cache=not args.no_cache,
                max_age=args.cache_max_age,
            )
            entry: dict[str, Any] = {
                "url": url,
                "http_code": r.get("http_code"),
                "url_effective": r.get("url_effective"),
                "ok": r.get("ok"),
                "mode": "googlesource_repos",
                "repos": [],
                "matches": [],
            }
            if not r.get("ok"):
                entry["error"] = r.get("error") or f"HTTP {r.get('http_code')}"
                report.append(entry)
                if not args.json:
                    print(f"\n=== {url} ===")
                    print(f"FAIL http_code={r.get('http_code')}")
                continue
            repos = extract_googlesource_repos(r["body"], r.get("url_effective") or url)
            hits = filter_repos(repos, pattern, ignore_case=not args.case_sensitive)
            entry["repos"] = hits
            entry["repo_total"] = len(repos)
            entry["match_count"] = len(hits)
            if hits:
                any_hit = True
            report.append(entry)
            if not args.json:
                print(
                    f"\n=== googlesource repos matching {pattern!r}  "
                    f"http={r.get('http_code')}  hits={len(hits)}/{len(repos)} ==="
                )
                print(
                    f"# Tip: root index is not a code search. "
                    f"Use: {sys.argv[0]} repos '{pattern}'"
                )
                if not hits:
                    print("(no repo name/description matches)")
                    # still show a few high-signal related guesses
                    print("# try: kernel/gs, kernel/common, device/google/*")
                for h in hits[: args.max_matches]:
                    print(f"  {h['name']}")
                    print(f"    {h['url']}")
                    if h.get("description"):
                        print(f"    {h['description'][:120]}")
            continue

        r = http_fetch(
            url,
            timeout=args.timeout,
            use_cache=not args.no_cache,
            max_age=args.cache_max_age,
        )
        entry = {
            "url": url,
            "http_code": r.get("http_code"),
            "url_effective": r.get("url_effective"),
            "ok": r.get("ok"),
            "from_cache": r.get("from_cache"),
            "matches": [],
        }
        if not r.get("ok"):
            entry["error"] = r.get("error") or f"HTTP {r.get('http_code')}"
            report.append(entry)
            if not args.json:
                print(f"\n=== {url} ===")
                print(f"FAIL http_code={r.get('http_code')}")
                if is_googlesource_index(url) or "googlesource.com" in url:
                    print(
                        f"# hint: {sys.argv[0]} repos '{pattern}'  "
                        f"or probe URL variants",
                        file=sys.stderr,
                    )
            continue

        matches = grep_bytes(
            r["body"],
            pattern,
            context=args.context,
            max_matches=args.max_matches,
            ignore_case=not args.case_sensitive,
            plain=use_plain and not raw_lines,
            snippet_width=snippet_w,
            auto_html=not raw_lines,
        )
        entry["matches"] = matches
        entry["match_count"] = len(matches)
        if matches:
            any_hit = True
        report.append(entry)

        if not args.json:
            print(
                f"\n=== {url}  http={r.get('http_code')}  "
                f"matches={len(matches)}  mode={'raw' if raw_lines else 'snippet'} ==="
            )
            if not matches:
                print("(no matches)")
            for m in matches:
                if "error" in m:
                    print(f"ERROR: {m['error']}")
                    continue
                sn = m.get("snippet") or m.get("match") or ""
                print(f"L{m['line']}: {sn}")
                # only show multi-line context when short and useful
                if (
                    not raw_lines
                    and len(m.get("context") or []) > 1
                    and all(len(c.get("text", "")) < 120 for c in m["context"])
                ):
                    for c in m["context"]:
                        mark = ">" if c["line"] == m["line"] else " "
                        print(f"  {mark}{c['line']}: {c['text']}")

    if args.json:
        emit({"pattern": pattern, "results": report}, as_json=True)

    if not any_hit:
        if not args.json:
            print(
                "\n# No matches. Prefer:\n"
                f"#   {sys.argv[0]} repos '{pattern}'     # googlesource repo names\n"
                f"#   {sys.argv[0]} local '{pattern}'\n"
                f"#   {sys.argv[0]} find '{pattern}' --remote --compact\n"
                f"#   {sys.argv[0]} suggest '{pattern}'\n"
                "# Do not raw-curl DevSite / dump HTML into context.",
                file=sys.stderr,
            )
        return 1
    return 0



def cmd_paths(args: argparse.Namespace) -> int:
    """Probe kernel trees for a driver/path (dwc3, drivers/usb/dwc3, …)."""
    query = args.query
    trees = DEFAULT_KERNEL_TREES
    if args.repo:
        refs = args.ref or ["HEAD"]
        trees = [(args.repo.strip("/"), refs)]
    result = probe_kernel_paths(
        query,
        trees=trees,
        timeout=args.timeout,
        list_dir=not args.no_list,
        max_hits=args.max_hits,
    )
    hits = result["hits"]
    if args.json:
        emit(result, as_json=True)
    else:
        print(f"# paths: {query!r}  expanded={result['paths']}")
        print(f"# hits={len(hits)}  attempts={len(result['attempts'])}")
        if not hits:
            print("(no path found in default trees)")
            print("# tried repos: " + ", ".join(t[0] for t in trees))
            print(
                f"# tip: {sys.argv[0]} paths drivers/usb/dwc3 "
                f"--repo kernel/common --ref HEAD"
            )
        for h in hits:
            print(f"[{h['http_code']}] {h['repo']} @ {h['ref']}  {h['path']}")
            print(f"     {h['url']}")
            if h.get("entries") is not None:
                names = h["entries"]
                print(
                    f"     entries({h.get('entry_count', len(names))}): "
                    + ", ".join(names[:25])
                    + ("…" if len(names) > 25 else "")
                )
        if hits:
            print(
                f"# next: {sys.argv[0]} grep PATTERN URL  "
                f"or fetch --raw --out FILE for a blob"
            )
    return 0 if hits else 1

def cmd_repos(args: argparse.Namespace) -> int:
    """Search android.googlesource.com repository index by name/description."""
    pattern = args.pattern
    base = args.base.rstrip("/") + "/"
    r = http_fetch(
        base,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        max_age=args.cache_max_age,
    )
    if not r.get("ok"):
        print(f"ERROR: http_code={r.get('http_code')} url={base}", file=sys.stderr)
        return 1
    repos = extract_googlesource_repos(r["body"], base)
    hits = filter_repos(repos, pattern, ignore_case=not args.case_sensitive)
    # optional live check first N
    if args.check:
        for h in hits[: args.check_max]:
            cr = http_check(h["url"], timeout=args.timeout)
            h["http_code"] = cr.get("http_code")
            h["ok"] = cr.get("ok")

    if args.json:
        emit(
            {
                "base": base,
                "pattern": pattern,
                "repo_total": len(repos),
                "hits": hits,
                "count": len(hits),
            },
            as_json=True,
        )
    else:
        print(
            f"# googlesource repos matching {pattern!r}: "
            f"{len(hits)} / {len(repos)} indexed"
        )
        if not hits:
            print("(none)")
            # helpful pixel kernel guesses
            print("# related probes (not necessarily matching pattern):")
            for guess in (
                "kernel/gs",
                "kernel/common",
                "kernel/google-modules",
                "device/google/gs-common",
                "device/google/tangorpro",
            ):
                print(f"  https://android.googlesource.com/{guess}/")
        for h in hits[: args.max]:
            code = h.get("http_code")
            prefix = f"[{code}] " if code else ""
            print(f"{prefix}{h['name']}")
            print(f"  {h['url']}")
            if h.get("description"):
                print(f"  {h['description'][:160]}")
        if len(hits) > args.max:
            print(f"# … {len(hits) - args.max} more (raise --max)")
    return 0 if hits else 1


def cmd_resolve(args: argparse.Namespace) -> int:
    variants = resolve_variants(args.url)
    warns = warn_url_hygiene(args.url)
    if args.probe:
        result = probe_url(args.url, timeout=args.timeout, max_attempts=args.max_attempts)
        if args.json:
            emit({"warnings": warns, "variants": variants, "probe": result}, as_json=True)
        else:
            for w in warns:
                print(f"WARN: {w}")
            print(print_check_table(
                [
                    {
                        "http_code": a["http_code"],
                        "ok": a["ok"],
                        "url": a["url"],
                        "url_effective": a.get("url_effective"),
                    }
                    for a in result["attempts"]
                ]
            ))
            for a in result["attempts"]:
                print(f"  reason: {a['reason']}")
            if result["found"]:
                print(f"\nWINNER: {result['winner']['url']}")
            else:
                print(f"\n{result.get('hint')}")
        return 0 if result.get("found") else 1

    if args.json:
        emit({"warnings": warns, "variants": variants}, as_json=True)
    else:
        for w in warns:
            print(f"WARN: {w}")
        print("Suggested URL variants (most likely first):")
        for i, v in enumerate(variants, 1):
            print(f"  {i}. {v['url']}")
            print(f"      ({v['reason']})")
        print(
            f"\nProbe live: {sys.argv[0]} resolve --probe '{args.url}'"
        )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    args.probe = True
    return cmd_resolve(args)


def cmd_bulletin(args: argparse.Namespace) -> int:
    try:
        urls = bulletin_urls(args.kind, args.date)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    rows = []
    if args.check or args.probe:
        for u in urls:
            r = http_check(u, timeout=args.timeout)
            rows.append(r)
        if args.json:
            emit({"urls": urls, "checks": rows}, as_json=True)
        else:
            for u in urls:
                print(u)
            print()
            print(print_check_table(rows))
        return 0 if any(r.get("ok") for r in rows) else 1

    if args.json:
        emit({"urls": urls}, as_json=True)
    else:
        for u in urls:
            print(u)
        print(
            f"\n# check: {sys.argv[0]} bulletin --check {args.kind} {args.date}",
            file=sys.stderr,
        )
    return 0


def cmd_links(args: argparse.Namespace) -> int:
    r = http_fetch(
        args.url,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        max_age=args.cache_max_age,
    )
    if not r.get("ok"):
        print(f"ERROR: http_code={r.get('http_code')}", file=sys.stderr)
        return 1
    links = extract_links(r["body"], r.get("url_effective") or args.url, args.pattern)
    if args.max:
        links = links[: args.max]
    if args.json:
        emit(
            {
                "url": args.url,
                "count": len(links),
                "links": links,
            },
            as_json=True,
        )
    else:
        print(f"# {len(links)} links from {args.url}")
        for L in links:
            print(L)
    return 0 if links else 1


def cmd_cves(args: argparse.Namespace) -> int:
    r = http_fetch(
        args.url,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        max_age=args.cache_max_age,
    )
    if not r.get("ok"):
        print(f"ERROR: http_code={r.get('http_code')}", file=sys.stderr)
        return 1
    cves = extract_cves(r["body"])
    title = extract_title(r["body"])
    if args.json:
        emit(
            {
                "url": args.url,
                "title": title,
                "http_code": r.get("http_code"),
                "cves": cves,
                "count": len(cves),
                "note": "IDs extracted from page only — never invent CVEs",
            },
            as_json=True,
        )
    else:
        print(f"# title: {title}")
        print(f"# {len(cves)} CVEs from {args.url}")
        for c in cves:
            print(c)
        if not cves:
            print("(none found on page)", file=sys.stderr)
    return 0 if cves else 1


def cmd_title(args: argparse.Namespace) -> int:
    # check + light fetch for title
    r = http_fetch(
        args.url,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        max_age=args.cache_max_age,
    )
    title = extract_title(r.get("body") or b"") if r.get("body") else ""
    data = {
        "url": args.url,
        "http_code": r.get("http_code"),
        "url_effective": r.get("url_effective"),
        "ok": r.get("ok"),
        "title": title,
        "bytes": len(r.get("body") or b""),
        "from_cache": r.get("from_cache"),
    }
    if args.json:
        emit(data, as_json=True)
    else:
        print(f"http_code={data['http_code']}")
        print(f"ok={data['ok']}")
        print(f"url_effective={data['url_effective']}")
        print(f"title={title}")
        print(f"bytes={data['bytes']}")
    return 0 if r.get("ok") else 1


def cmd_local(args: argparse.Namespace) -> int:
    # priority files first
    pri = local_priority_hint(args.pattern)
    result = local_search(
        args.pattern,
        path=Path(args.path) if args.path else None,
        glob=args.glob,
        max_matches=args.max_matches,
        context=args.context,
    )
    if args.json:
        emit({"priority_hits": pri, "search": result}, as_json=True)
    else:
        if pri:
            print("# Priority project docs mentioning pattern:")
            for p in pri:
                print(f"  - {p}")
            print()
        print(f"# local search via {result.get('tool')} in {result.get('root')}")
        print(result.get("stdout") or "(no matches)")
        if result.get("stderr"):
            print(result["stderr"], file=sys.stderr)
    out = (result.get("stdout") or "").strip()
    if not pri and (not out or out == "(no matches)"):
        return 1
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """Local-first discovery, then optional remote candidates (compact by default)."""
    topic = args.topic.strip().lower()
    pattern = args.pattern or args.topic
    compact = not getattr(args, "verbose", False)
    max_local = min(args.max_matches, 12 if compact else args.max_matches)
    pri = local_priority_hint(pattern)
    local = local_search(
        pattern,
        max_matches=max_local,
        context=0 if compact else args.context,
    )

    candidates = list(TOPIC_CANDIDATES.get(topic, []))
    if not candidates:
        for k, urls in TOPIC_CANDIDATES.items():
            if topic in k or k in topic:
                candidates.extend(urls)
    seen: set[str] = set()
    cand_u: list[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            cand_u.append(u)

    remote_results: list[dict[str, Any]] = []
    if args.remote and cand_u:
        for u in cand_u[: args.max_remote]:
            r = http_fetch(
                u,
                timeout=args.timeout,
                use_cache=not args.no_cache,
                max_age=args.cache_max_age,
            )
            entry: dict[str, Any] = {
                "url": u,
                "http_code": r.get("http_code"),
                "ok": r.get("ok"),
            }
            if r.get("ok"):
                matches = grep_bytes(
                    r["body"],
                    pattern,
                    context=0,
                    max_matches=3,
                    plain=True,
                    snippet_width=120,
                )
                entry["match_count"] = len(matches)
                entry["matches"] = matches
            remote_results.append(entry)

    repo_hits: list[dict[str, str]] = []
    if getattr(args, "repos", False) or (
        args.remote and topic in {"husky", "shiba", "pixel8", "pixel 8 pro", "kernel"}
    ):
        rr = http_fetch(
            GOOGLESURCE_ROOT,
            timeout=args.timeout,
            use_cache=not args.no_cache,
            max_age=args.cache_max_age,
        )
        if rr.get("ok"):
            repo_hits = filter_repos(
                extract_googlesource_repos(rr["body"]), pattern
            )

    # In-tree driver/path probe (dwc3 is not a repo name — repos won't match)
    path_result: Optional[dict[str, Any]] = None
    want_paths = (
        getattr(args, "paths", False)
        or args.remote
        or (
            getattr(args, "repos", False)
            and pattern.lower() in PATH_ALIASES
        )
    )
    path_like = bool(
        "/" in pattern
        or pattern.lower() in PATH_ALIASES
        or re.match(r"^[a-zA-Z][a-zA-Z0-9_-]{1,40}$", pattern or "")
    )
    skip_path_topics = {
        "husky",
        "shiba",
        "pixel8",
        "pixel 8 pro",
        "bulletin",
        "avb",
        "fastboot",
    }
    if want_paths and path_like and topic not in skip_path_topics:
        path_result = probe_kernel_paths(
            pattern,
            timeout=args.timeout,
            list_dir=True,
            max_hits=getattr(args, "max_path_hits", 15),
        )

    # Cap local stdout for agents
    local_out = local.get("stdout") or ""
    if compact and local_out:
        lines = local_out.splitlines()
        # keep only match lines (rg uses ':' for matches)
        keep = [ln for ln in lines if ":" in ln and not ln.startswith("--")]
        if len(keep) > max_local:
            keep = keep[:max_local] + [f"… ({len(keep) - max_local} more local hits)"]
        local_out = "\n".join(keep) if keep else local_out[:2000]

    payload = {
        "topic": args.topic,
        "pattern": pattern,
        "priority_docs": pri,
        "local": {"tool": local.get("tool"), "stdout": local_out},
        "remote_candidates": cand_u,
        "remote_results": remote_results,
        "googlesource_repos": repo_hits,
        "kernel_paths": path_result,
        "advice": (
            "Use research-tool (not curl). "
            "repos = repo NAMES only; paths = in-tree drivers (dwc3). "
            "bulletin pixel YYYY-MM for CVEs. Never invent CVE IDs."
        ),
    }

    if args.json:
        emit(payload, as_json=True)
    else:
        print(f"# find: {args.topic!r}  (compact={compact})")
        if pri:
            print("## Priority local docs")
            for p in pri:
                print(f"  - {p}")
        print("## Local hits (capped)")
        print(local_out or "(no matches)")
        if cand_u:
            print("## Remote candidates")
            for u in cand_u:
                print(f"  - {u}")
        if remote_results:
            print("## Remote probe")
            for e in remote_results:
                mc = e.get("match_count", "-")
                print(f"  [{e.get('http_code')}] matches={mc}  {e['url']}")
                for m in e.get("matches") or []:
                    sn = m.get("snippet") or m.get("match") or ""
                    print(f"      L{m['line']}: {sn[:120]}")
        if repo_hits:
            print(f"## Googlesource repos matching {pattern!r}")
            for h in repo_hits[:20]:
                print(f"  - {h['name']}  {h['url']}")
        elif args.remote or getattr(args, "repos", False):
            print(
                "## Googlesource repos: (no name match — "
                "names only; for drivers use paths)"
            )
        if path_result is not None:
            hits = path_result.get("hits") or []
            print(
                f"## Kernel path probe for {pattern!r}  "
                f"hits={len(hits)} expanded={path_result.get('paths')}"
            )
            if not hits:
                print("  (no path in default trees)")
                print(f"  try: {sys.argv[0]} paths {pattern}")
            for h in hits[:20]:
                print(f"  [{h['http_code']}] {h['repo']}@{h['ref']}  {h['path']}")
                print(f"       {h['url']}")
                if h.get("entries"):
                    print(
                        "       files: "
                        + ", ".join(h["entries"][:20])
                        + ("…" if len(h["entries"]) > 20 else "")
                    )
        print(f"# {payload['advice']}")
    return 0


def cmd_known(args: argparse.Namespace) -> int:
    cats = args.category
    data = KNOWN_URLS
    if cats:
        data = {c: KNOWN_URLS[c] for c in cats if c in KNOWN_URLS}
        missing = [c for c in cats if c not in KNOWN_URLS]
        if missing:
            print(
                f"WARN: unknown categories {missing}; "
                f"have: {', '.join(KNOWN_URLS)}",
                file=sys.stderr,
            )
    if args.check:
        rows = []
        for cat, items in data.items():
            for title, url in items:
                r = http_check(url, timeout=args.timeout)
                r["title"] = title
                r["category"] = cat
                rows.append(r)
        if args.json:
            emit(rows, as_json=True)
        else:
            for r in rows:
                mark = "OK " if r.get("ok") else "BAD"
                print(f"{mark} {r.get('http_code')}  [{r.get('category')}] {r.get('title')}")
                print(f"     {r.get('url')}")
        return 0 if all(r.get("ok") for r in rows) else 1

    if args.json:
        emit(
            {
                c: [{"title": t, "url": u} for t, u in items]
                for c, items in data.items()
            },
            as_json=True,
        )
    else:
        print("# Known-good research URLs (project-curated)")
        print(f"# categories: {', '.join(KNOWN_URLS.keys())}")
        for cat, items in data.items():
            print(f"\n## {cat}")
            for title, url in items:
                print(f"  - {title}")
                print(f"    {url}")
        print(
            f"\n# Also read local: {', '.join(LOCAL_PRIORITY[:5])} …",
            file=sys.stderr,
        )
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    action = args.action
    if action == "path":
        print(CACHE_DIR)
        return 0
    if action == "clear":
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"cleared {CACHE_DIR}")
        else:
            print("cache already empty")
        return 0
    if action == "list":
        if not CACHE_DIR.exists():
            print("(empty)")
            return 0
        rows = []
        for p in sorted(CACHE_DIR.glob("*.meta.json")):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {"url": p.name}
            rows.append(meta)
        if args.json:
            emit(rows, as_json=True)
        else:
            for m in rows:
                print(
                    f"{m.get('bytes', '?'):>8}  {m.get('http_code', '?')}  "
                    f"{m.get('url', '')}"
                )
        return 0
    print(f"unknown cache action: {action}", file=sys.stderr)
    return 2


def cmd_text(args: argparse.Namespace) -> int:
    args.plain = True
    return cmd_fetch(args)


def cmd_suggest(args: argparse.Namespace) -> int:
    q = args.query.strip()
    ql = q.lower()
    suggestions: list[dict[str, str]] = []

    # If it looks like a URL, resolve variants
    if ql.startswith("http://") or ql.startswith("https://"):
        for v in resolve_variants(q):
            suggestions.append({"type": "url_variant", **v})
        suggestions.append(
            {
                "type": "action",
                "url": "",
                "reason": f"probe: {sys.argv[0]} probe '{q}'",
            }
        )
    else:
        # topic keywords
        for k, urls in TOPIC_CANDIDATES.items():
            if k in ql or ql in k or any(w in k for w in ql.split() if len(w) > 3):
                for u in urls:
                    suggestions.append(
                        {"type": "topic_url", "url": u, "reason": f"topic:{k}"}
                    )
        # bulletin date?
        m = re.search(r"(20\d{2})-(\d{2})", q)
        if m or "bulletin" in ql or "cve" in ql:
            ym = m.group(0) if m else "2026-02"
            for u in bulletin_urls("both", ym):
                suggestions.append(
                    {"type": "bulletin", "url": u, "reason": f"bulletin {ym}"}
                )
        # husky / device
        if "husky" in ql or "pixel 8" in ql or "shiba" in ql:
            for title, url in KNOWN_URLS.get("devices", []):
                suggestions.append(
                    {"type": "device", "url": url, "reason": title}
                )
        # always suggest local first
        suggestions.insert(
            0,
            {
                "type": "action",
                "url": "",
                "reason": f"local first: {sys.argv[0]} local '{q}'",
            },
        )
        suggestions.insert(
            1,
            {
                "type": "action",
                "url": "",
                "reason": f"find: {sys.argv[0]} find '{q}' --remote",
            },
        )

    # de-dupe by url+reason
    seen: set[str] = set()
    uniq: list[dict[str, str]] = []
    for s in suggestions:
        key = s.get("url", "") + "|" + s.get("reason", "")
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    if args.json:
        emit({"query": q, "suggestions": uniq}, as_json=True)
    else:
        print(f"# suggestions for: {q}")
        for s in uniq:
            if s.get("url"):
                print(f"  [{s['type']}] {s['url']}")
                print(f"           {s['reason']}")
            else:
                print(f"  [{s['type']}] {s['reason']}")
    return 0


def cmd_help(_: argparse.Namespace) -> int:
    print(__doc__)
    print(
        """
Quick recipes (local LLM / agent):

  # 1. Prefer local project docs
  ./research-tool.py local husky
  ./research-tool.py find husky
  ./research-tool.py find husky --remote

  # 2. Check whether a URL exists before dumping body
  ./research-tool.py check 'https://source.android.com/docs/setup/build/building-devices'
  ./research-tool.py multi URL1 URL2 URL3

  # 3. Fix wrong bulletin / doc paths
  ./research-tool.py resolve 'https://source.android.com/docs/security/bulletins/2023-10'
  ./research-tool.py probe  'https://source.android.com/docs/security/bulletin/2026-02-01'
  ./research-tool.py bulletin pixel 2026-02 --check

  # 4. Grep remote (snippets — never raw curl | grep HTML)
  ./research-tool.py grep husky \\
      'https://source.android.com/docs/setup/build/building' \\
      'https://source.android.com/docs/setup/reference/build-numbers'

  # 4b. Googlesource repo NAMES (Pixel 8 Pro trees use shusky, not husky alone)
  ./research-tool.py repos husky
  ./research-tool.py repos 'kernel/(gs|common)|shusky'
  ./research-tool.py paths dwc3
  ./research-tool.py paths drivers/usb/dwc3 --repo kernel/common
  ./research-tool.py find dwc3 --remote --paths
  ./research-tool.py grep husky 'https://android.googlesource.com/'   # auto → repos mode

  # 5. Extract structure without flooding context
  ./research-tool.py title URL
  ./research-tool.py links URL --pattern bulletin
  ./research-tool.py cves URL
  ./research-tool.py text URL --max-bytes 40000

  # 6. Known-good catalog + cache
  ./research-tool.py known
  ./research-tool.py known --check
  ./research-tool.py cache list

  # 7. Machine-readable for agents
  ./research-tool.py check URL --json
  ./research-tool.py grep CVE-2026 URL --json

Do NOT use Kilo webfetch on source.android.com / developer.android.com.
Do NOT use raw curl|grep for research — use research-tool (snippets / repos).
After one failed host pattern, stop spraying URLs — use resolve/probe/known/repos.
""".strip()
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research-tool",
        description="LLM-friendly Android/GrapheneOS doc research tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `research-tool help` for recipes.",
    )
    p.add_argument("--version", action="version", version=f"research-tool {VERSION}")

    # Shared flags on each subparser so `cmd --json` works
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"curl timeout seconds (default {DEFAULT_TIMEOUT})",
    )
    common.add_argument(
        "--json", action="store_true", help="JSON output (agent-friendly)"
    )

    sub = p.add_subparsers(dest="command", required=True)

    def add_sub(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        parents = list(kwargs.pop("parents", []))
        parents.insert(0, common)
        return sub.add_parser(name, parents=parents, **kwargs)

    def add_cache_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--no-cache", action="store_true", help="bypass on-disk cache")
        sp.add_argument(
            "--cache-max-age",
            type=int,
            default=86400,
            help="cache freshness seconds (default 86400)",
        )

    sp = add_sub("check", help="HTTP status + final URL")
    sp.add_argument("urls", nargs="+", help="URL(s) to check")
    sp.set_defaults(func=cmd_check)

    sp = add_sub("multi", help="Alias for check on many URLs")
    sp.add_argument("urls", nargs="+")
    sp.set_defaults(func=cmd_multi)

    sp = add_sub(
        "fetch",
        help="Download page (HTML→readable text by default; --raw for HTML)",
    )
    sp.add_argument("url")
    sp.add_argument("--out", "-o", help="write full raw body to file")
    sp.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"stdout cap (default {DEFAULT_MAX_BYTES}; readable HTML uses 40k)",
    )
    sp.add_argument(
        "--plain",
        action="store_true",
        help="force strip-all HTML (default already readable for HTML pages)",
    )
    sp.add_argument(
        "--raw",
        action="store_true",
        help="emit raw HTML/bytes to stdout (floods context — avoid in agents)",
    )
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_fetch)

    sp = add_sub("text", help="Fetch + strip HTML (capped plain text)")
    sp.add_argument("url")
    sp.add_argument("--out", "-o")
    sp.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_text, plain=True)

    sp = add_sub("grep", help="Fetch URL(s) and search (snippets; no HTML dumps)")
    sp.add_argument("pattern", help="regex pattern")
    sp.add_argument("urls", nargs="+", help="one or more URLs")
    sp.add_argument("--case-sensitive", action="store_true")
    sp.add_argument("-C", "--context", type=int, default=1)
    sp.add_argument("--max-matches", type=int, default=DEFAULT_GREP_MATCHES)
    sp.add_argument(
        "--snippet",
        type=int,
        default=DEFAULT_SNIPPET,
        help=f"chars around match (default {DEFAULT_SNIPPET})",
    )
    sp.add_argument(
        "--plain",
        action="store_true",
        help="force strip HTML (default already auto-strips HTML)",
    )
    sp.add_argument(
        "--raw-lines",
        action="store_true",
        help="old behaviour: dump long HTML lines (bad for agents)",
    )
    sp.add_argument(
        "--force-html",
        action="store_true",
        help="do not switch googlesource / to repos mode",
    )
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_grep)

    sp = add_sub(
        "paths",
        help="Probe kernel trees for driver/path (dwc3, drivers/usb/dwc3)",
    )
    sp.add_argument("query", help="subsystem or path (dwc3, drivers/usb/dwc3)")
    sp.add_argument("--repo", help="limit to one repo (e.g. kernel/common)")
    sp.add_argument(
        "--ref",
        action="append",
        help="git ref (repeatable); default HEAD + common android branches",
    )
    sp.add_argument("--no-list", action="store_true", help="skip JSON directory listing")
    sp.add_argument("--max-hits", type=int, default=40)
    sp.set_defaults(func=cmd_paths)

    sp = add_sub(
        "repos",
        help="Search android.googlesource.com repo names (prefer over grepping /)",
    )
    sp.add_argument("pattern", help="regex against repo name/description")
    sp.add_argument(
        "--base",
        default=GOOGLESURCE_ROOT,
        help="index URL (default android.googlesource.com/)",
    )
    sp.add_argument("--case-sensitive", action="store_true")
    sp.add_argument("--max", type=int, default=50)
    sp.add_argument(
        "--check",
        action="store_true",
        help="HTTP-check first hits (slow)",
    )
    sp.add_argument("--check-max", type=int, default=10)
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_repos)

    sp = add_sub("resolve", help="Suggest corrected URL variants")
    sp.add_argument("url")
    sp.add_argument(
        "--probe",
        action="store_true",
        help="actually check variants until 200 (max attempts)",
    )
    sp.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_PROBE_ATTEMPTS,
        help=f"probe cap (default {MAX_PROBE_ATTEMPTS})",
    )
    sp.set_defaults(func=cmd_resolve)

    sp = add_sub("probe", help="Try URL variants until HTTP 200")
    sp.add_argument("url")
    sp.add_argument("--max-attempts", type=int, default=MAX_PROBE_ATTEMPTS)
    sp.set_defaults(func=cmd_probe)

    sp = add_sub("bulletin", help="Canonical AOSP/Pixel bulletin URLs")
    sp.add_argument(
        "kind",
        choices=["aosp", "pixel", "both"],
        help="bulletin tree",
    )
    sp.add_argument("date", help="YYYY-MM or YYYY-MM-01")
    sp.add_argument("--check", action="store_true", help="HEAD/check constructed URLs")
    sp.add_argument("--probe", action="store_true", help="same as --check")
    sp.set_defaults(func=cmd_bulletin)

    sp = add_sub("links", help="Extract hrefs from page")
    sp.add_argument("url")
    sp.add_argument("--pattern", "-e", help="filter regex on href")
    sp.add_argument("--max", type=int, default=200)
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_links)

    sp = add_sub("cves", help="Extract CVE IDs from page (no invention)")
    sp.add_argument("url")
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_cves)

    sp = add_sub("title", help="Status + HTML title")
    sp.add_argument("url")
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_title)

    sp = add_sub("local", help="Search project files first (rg)")
    sp.add_argument("pattern")
    sp.add_argument("--path", help="override project root")
    sp.add_argument("--glob", help="rg --glob")
    sp.add_argument("--max-matches", type=int, default=40)
    sp.add_argument("-C", "--context", type=int, default=1)
    sp.set_defaults(func=cmd_local)

    sp = add_sub(
        "find",
        help="Compact local-first discovery + optional remote/repos",
    )
    sp.add_argument("topic", help="topic or keyword (e.g. husky, bulletin, fbe)")
    sp.add_argument("--pattern", help="override grep pattern (default: topic)")
    sp.add_argument(
        "--remote",
        action="store_true",
        help="also fetch/grep remote candidate URLs",
    )
    sp.add_argument(
        "--repos",
        action="store_true",
        help="also search googlesource repo index (names only)",
    )
    sp.add_argument(
        "--paths",
        action="store_true",
        help="also probe in-tree kernel paths (dwc3, drivers/usb/…)",
    )
    sp.add_argument("--max-path-hits", type=int, default=15)
    sp.add_argument(
        "--verbose",
        action="store_true",
        help="full local rg dump (default is capped/compact)",
    )
    sp.add_argument(
        "--compact",
        action="store_true",
        help="explicit compact mode (default)",
    )
    sp.add_argument("--max-remote", type=int, default=5)
    sp.add_argument("--max-matches", type=int, default=30)
    sp.add_argument("-C", "--context", type=int, default=1)
    add_cache_flags(sp)
    sp.set_defaults(func=cmd_find)

    sp = add_sub("known", help="Curated known-good URLs for this project")
    sp.add_argument(
        "category",
        nargs="*",
        help=f"optional filter: {', '.join(KNOWN_URLS)}",
    )
    sp.add_argument("--check", action="store_true", help="live-check all listed URLs")
    sp.set_defaults(func=cmd_known)

    sp = add_sub("cache", help="Manage fetch cache")
    sp.add_argument("action", choices=["list", "clear", "path"])
    sp.set_defaults(func=cmd_cache)

    sp = add_sub("suggest", help="Suggest URLs/actions for a query or bad URL")
    sp.add_argument("query")
    sp.set_defaults(func=cmd_suggest)

    sp = add_sub("help", help="Extended help + recipes")
    sp.set_defaults(func=cmd_help)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    if not hasattr(args, "json"):
        args.json = False
    if not hasattr(args, "timeout"):
        args.timeout = DEFAULT_TIMEOUT
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
