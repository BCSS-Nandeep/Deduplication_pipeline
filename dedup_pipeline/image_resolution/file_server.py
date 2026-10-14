"""Locate and fetch files on the media file server.

Behaviour mirrors DOPAMS-BE ``src/routes/fileProxy.ts``:

* the URL is ``FILE_SERVER_BASE_URL`` + the relative path from the database;
* each path segment is decoded then re-encoded (``encodeFileUrlForFetch``);
* a path with no extension is tried as-is, then with each of
  ``EXTENSION_FALLBACKS`` in order (``fetchUpstreamWithExtensionFallback``).

Differences, all deliberate: requests have timeouts and bounded retries, only
the bytes needed are read (HTTP Range), redirects may not leave the file
server, and the proxy's login requirement does not apply to this job.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol
from urllib.parse import quote, unquote, urlsplit

DEFAULT_FILE_SERVER_BASE_URL = "http://192.168.103.106:8080/files/"
# Same list and order as EXTENSION_FALLBACKS in fileProxy.ts.
EXTENSION_FALLBACKS = (".jpg", ".jpeg", ".png", ".webp", ".pdf", ".doc", ".docx")

_HAS_EXTENSION = re.compile(r"\.[a-z0-9]{2,5}$", re.IGNORECASE)
_SCHEME = re.compile(r"^https?://", re.IGNORECASE)


class InvalidPath(ValueError):
    """The database path cannot be turned into a safe file-server URL."""


@dataclass(frozen=True)
class Candidate:
    url: str
    suffix: str  # "" when the path is used as stored


def build_candidates(path_ref: str, base_url: str, extensions: tuple[str, ...] = EXTENSION_FALLBACKS) -> list[Candidate]:
    base = base_url if base_url.endswith("/") else base_url + "/"
    rel = path_ref.strip()
    if _SCHEME.match(rel):
        if not rel.startswith(base):
            raise InvalidPath("absolute URL is outside the configured file server")
        rel = rel[len(base):]
    rel = rel.lstrip("/")
    if not rel:
        raise InvalidPath("empty path")
    segments = rel.split("/")
    for segment in segments:
        decoded = unquote(segment)
        if decoded in (".", "..") or "\\" in decoded or any(ord(ch) < 32 for ch in decoded):
            raise InvalidPath("path contains traversal or control characters")
    # encodeURIComponent leaves !*'() unescaped; match it.
    encoded = "/".join(quote(unquote(segment), safe="!*'()") for segment in segments)
    url = base + encoded
    if _HAS_EXTENSION.search(encoded):
        return [Candidate(url, "")]
    return [Candidate(url, "")] + [Candidate(url + ext, ext) for ext in extensions]


# ------------------------------------------------------------------- fetching

@dataclass
class FetchResult:
    status: Optional[int]  # HTTP status, or None for a transport failure
    body: bytes = b""
    truncated: bool = False  # more bytes exist than ``max_bytes``
    error: Optional[str] = None


class Fetcher(Protocol):
    def fetch(self, url: str, max_bytes: int) -> FetchResult: ...


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            raise urllib.error.HTTPError(req.full_url, 403, "redirect to another host refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpFetcher:
    """GET up to ``max_bytes`` of a URL using a Range request, with bounded retries."""

    def __init__(self, timeout: float = 15.0, retries: int = 2, backoff: float = 0.5):
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._opener = urllib.request.build_opener(_SameHostRedirects)

    def fetch(self, url: str, max_bytes: int) -> FetchResult:
        result = FetchResult(status=None, error="not attempted")
        for attempt in range(self._retries + 1):
            result = self._once(url, max_bytes, use_range=True)
            if result.status == 416:  # empty file: nothing satisfies a byte range
                result = self._once(url, max_bytes, use_range=False)
            transient = result.status is None or result.status >= 500 or result.status == 429
            if not transient:
                return result
            if attempt < self._retries:
                time.sleep(self._backoff * (2 ** attempt))
        return result

    def _once(self, url: str, max_bytes: int, use_range: bool) -> FetchResult:
        headers = {"User-Agent": "dedup-pipeline-image-resolution"}
        if use_range:
            headers["Range"] = f"bytes=0-{max_bytes}"  # one extra byte reveals truncation
        request = urllib.request.Request(url, headers=headers)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                body = response.read(max_bytes + 1)
                return FetchResult(response.status, body[:max_bytes], truncated=len(body) > max_bytes)
        except urllib.error.HTTPError as exc:
            return FetchResult(exc.code, error=f"HTTP {exc.code}")
        except Exception as exc:  # timeouts, resets, DNS, refused connections
            return FetchResult(None, error=type(exc).__name__)
