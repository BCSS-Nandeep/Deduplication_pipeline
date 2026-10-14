"""HTTP client for the existing FR-DOPAMS service (``C:\\Users\\Dell\\Desktop\\FR-DOPAMS``).

Only endpoints that exist in its source are called, with the contracts its code
implements (see ``tests/test_fr_contract.py``, which checks them against that source):

``GET /health``    -> ``{"status": "ok"}``
``POST /compare``  -> multipart fields ``file_a`` and ``file_b`` (one image each), header ``X-API-Key``.
                      Stateless: enrolls nothing, searches no gallery, writes no log, touches no database.
                      200: ``{"success": true, "data": {"status": "compared", "similarity": <raw cosine, may be
                      negative>, "metric": "cosine", "same_person": <similarity >= threshold>, "threshold": ...,
                      "threshold_name", "model", ...}, "errors": []}``.
                      422: ``{"success": false, "errors": [{"side": "a"|"b", "code": no_face|multiple_faces|
                      low_quality|invalid_image, "reason", "message"}, ...]}``: facts about the images. 401/503:
                      key refused / endpoint disabled. 413: request too large. 500: service fault.
``POST /search``   -> multipart field ``file`` (one image). Success body:
                      ``{"success": true, "results": [{"rank", "person_id", "similarity", ...}], "threshold": ...}``.
                      ``similarity`` is cosine similarity of L2-normalised ArcFace embeddings, so it
                      lies in [-1, 1]; it is the best-pose score of an *enrolled gallery person* against the
                      uploaded image, not a score between two uploaded images.
Errors             -> ``{"detail": "<message>"}`` with HTTP 400 (image or precondition problem),
                      422 (request schema), 500 (unexpected exception). There is no authentication.

``/search`` appends a row to FR-DOPAMS ``match_logs`` for every successful call, so it is refused
unless ``FrConfig.allow_audit_log_writes`` is set. Nothing in this client logs image bytes, response
bodies (they contain names of gallery persons) or credentials.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

from ..image_resolution.formats import sniff_format
from .config import FrConfig
from .errors import (
    IMAGE_KINDS,
    FrAuthError,
    FrContractError,
    FrImageRejected,
    FrInvalidResponse,
    FrPairRejected,
    FrServerError,
    FrServiceNotReady,
    FrUnavailable,
    FrWritePolicyError,
    SideRejection,
)

log = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 1024 * 1024
EXPECTED_SERVICE_TITLE = "Face Enrollment and Matching API"  # FastAPI title in FR-DOPAMS app.py
_RETRYABLE_STATUS = frozenset({502, 503, 504})

# Messages raised in FR-DOPAMS src/face_service.py (``ValueError`` -> HTTP 400 detail).
# tests/test_fr_contract.py checks these still appear in that source.
_REJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^No face detected"), "no_face"),
    (re.compile(r"faces? of similar size detected"), "multiple_faces"),
    (re.compile(r"^Face is too small|^Face is too blurred"), "low_quality"),
    (re.compile(r"^Empty image data|^Unable to decode image|^Image is larger than"), "invalid_image"),
)
_NOT_READY = re.compile(r"^No faces are enrolled yet")


def classify_bad_request(detail: str) -> str:
    """Map an HTTP 400 ``detail`` to an image kind, ``service_not_ready`` or ``unrecognised``."""
    if _NOT_READY.search(detail):
        return "service_not_ready"
    for pattern, kind in _REJECTION_PATTERNS:
        if pattern.search(detail):
            return kind
    return "unrecognised"


@dataclass(frozen=True)
class SearchHit:
    person_id: str
    similarity: float  # cosine, [-1, 1]; gallery person vs the uploaded image


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    threshold: Optional[float]


def encode_multipart_files(parts: list[tuple[str, bytes, str, str]]) -> tuple[bytes, str]:
    """Encode ``(field, content, content_type, filename)`` file parts as multipart/form-data."""
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for field, content, content_type, filename in parts:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("ascii")
        )
        chunks.append(content + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def encode_multipart(field: str, content: bytes, content_type: str, filename: str) -> tuple[bytes, str]:
    return encode_multipart_files([(field, content, content_type, filename)])


_CONTENT_TYPES = {"jpeg": ("image/jpeg", "jpg"), "png": ("image/png", "png"), "webp": ("image/webp", "webp")}


def _image_part(field: str, image_bytes: bytes) -> tuple[str, bytes, str, str]:
    content_type, ext = _CONTENT_TYPES.get(sniff_format(image_bytes[:64]), ("application/octet-stream", "bin"))
    return field, image_bytes, content_type, f"image.{ext}"  # a neutral file name: nothing identifying leaves the pipeline


@dataclass(frozen=True)
class CompareResult:
    """``data`` of a successful ``POST /compare``."""

    similarity: float  # raw cosine, [-1, 1]; may be negative
    metric: str
    same_person: bool  # the service's decision, using ``threshold``
    threshold: float
    threshold_name: str
    model: str


class FrDopamsClient:
    def __init__(self, config: FrConfig):
        self._config = config
        parts = urlsplit(config.base_url)
        self._https = parts.scheme == "https"
        self._host = parts.hostname or ""
        self._port = parts.port
        self._prefix = parts.path.rstrip("/")

    # ---------------------------------------------------------- endpoints

    def health(self) -> dict[str, Any]:
        """``GET /health``. Read-only; the only endpoint with no side effects."""
        body = self._json(*self._request("GET", "/health")[:2])
        # FR-DOPAMS answers exactly {"status": "ok"} (app.py). Extra keys mean some other service
        # is listening on this address, whose "ok" says nothing about face recognition.
        if body != {"status": "ok"}:
            raise FrInvalidResponse("/health did not return exactly {'status': 'ok'}; this may not be FR-DOPAMS")
        return body

    def verify_service(self, endpoints: Optional[dict[str, tuple[str, ...]]] = None) -> None:
        """Read-only preflight: confirm this really is FR-DOPAMS and exposes the endpoints we call.

        ``endpoints`` maps a POST path to the multipart fields it must require (default: ``/compare``
        with ``file_a`` and ``file_b``). Uses ``GET /health`` and FastAPI's ``GET /openapi.json`` only. Run
        it once before processing, so a wrong address, a different application on the same port, an
        FR-DOPAMS without ``/compare``, or API drift fails fast and clearly.
        """
        required = endpoints if endpoints is not None else {"/compare": ("file_a", "file_b")}
        self.health()
        spec = self._json(*self._request("GET", "/openapi.json")[:2])
        try:
            title = spec["info"]["title"]
        except (KeyError, TypeError):
            raise FrContractError("/openapi.json has no title") from None
        if title != EXPECTED_SERVICE_TITLE:
            raise FrContractError(f"unexpected service {title!r} (expected {EXPECTED_SERVICE_TITLE!r})")
        for path, fields in required.items():
            try:
                schema = spec["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
                if "$ref" in schema:
                    schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
                present = set(schema["properties"]) >= set(fields) and set(fields) <= set(schema.get("required", []))
            except (KeyError, TypeError):
                present = False
            if not present:
                raise FrContractError(f"service does not expose POST {path} requiring multipart fields {list(fields)}")

    def compare(self, image_a: bytes, image_b: bytes) -> CompareResult:
        """``POST /compare``: stateless image-to-image comparison (nothing enrolled, logged or stored).

        Raises ``FrPairRejected`` when FR-DOPAMS found an image unusable (every unusable side is
        listed), and ``FrOperationalError`` subclasses for everything else. Safe to retry.
        """
        body, content_type = encode_multipart_files([_image_part("file_a", image_a), _image_part("file_b", image_b)])
        headers = {"Content-Type": content_type}
        if self._config.api_key:
            headers["X-API-Key"] = self._config.api_key
        status, data = self._request("POST", "/compare", body, headers)[:2]
        if status == 200:
            return self._parse_compare(self._json(status, data))
        parsed = self._error_body(data)
        if status == 422:
            rejections = self._rejections(parsed)
            if rejections:
                raise FrPairRejected(rejections)
            raise FrContractError("/compare returned 422 without side-specific errors (request shape is wrong)")
        codes = [e.get("code") for e in parsed.get("errors", []) if isinstance(e, dict)]
        if status == 401 or (status == 503 and "compare_disabled" in codes):
            raise FrAuthError("POST /compare refused the API key" if status == 401 else "POST /compare is disabled on the service")
        if status == 500:
            raise FrServerError("/compare returned HTTP 500 (internal_error)")
        raise FrContractError(f"/compare returned unexpected HTTP {status}")

    @staticmethod
    def _error_body(data: bytes) -> dict[str, Any]:
        try:
            body = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _rejections(body: dict[str, Any]) -> tuple[SideRejection, ...]:
        found = []
        errors = body.get("errors")
        if not isinstance(errors, list):
            return ()
        for item in errors:
            if (
                isinstance(item, dict)
                and item.get("side") in ("a", "b")
                and item.get("code") in IMAGE_KINDS
            ):
                found.append(SideRejection(item["side"], item["code"], item.get("reason"), str(item.get("message", ""))[:200]))
        return tuple(sorted(found, key=lambda r: r.side))

    @staticmethod
    def _parse_compare(body: Any) -> CompareResult:
        data = body.get("data") if isinstance(body, dict) else None
        try:
            similarity, threshold = data["similarity"], data["threshold"]
            valid = (
                body["success"] is True
                and data["status"] == "compared"
                and data["metric"] == "cosine"
                and isinstance(data["same_person"], bool)
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (similarity, threshold))
                and -1.0 <= similarity <= 1.0
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise FrInvalidResponse("/compare body does not match the documented contract")
        return CompareResult(
            float(similarity), "cosine", data["same_person"], float(threshold),
            str(data.get("threshold_name", "")), str(data.get("model", "")),
        )

    def search(self, image_bytes: bytes) -> SearchResult:
        """``POST /search`` with one image.

        Succeeding proves FR-DOPAMS found exactly one dominant, adequately sized and sharp face.
        Raises ``FrImageRejected`` when it refused the image, ``FrServiceNotReady`` when its gallery
        is empty (it then never looked at the image), and ``FrOperationalError`` subclasses otherwise.
        """
        if not self._config.allow_audit_log_writes:
            raise FrWritePolicyError(
                "POST /search appends a row to FR-DOPAMS match_logs on every success. "
                "Set allow_audit_log_writes only if that has been approved."
            )
        kind, ext = _CONTENT_TYPES.get(sniff_format(image_bytes[:64]), ("application/octet-stream", "bin"))
        body, content_type = encode_multipart("file", image_bytes, kind, f"image.{ext}")
        status, data = self._request("POST", "/search", body, {"Content-Type": content_type})[:2]
        if status == 200:
            return self._parse_search(self._json(status, data))
        self._raise_for_error(status, data, "/search")
        raise AssertionError("unreachable")

    # ------------------------------------------------------------- parsing

    @staticmethod
    def _parse_search(body: Any) -> SearchResult:
        if not isinstance(body, dict) or body.get("success") is not True or not isinstance(body.get("results"), list):
            raise FrInvalidResponse("/search body lacks success=true and a results list")
        hits = []
        for item in body["results"]:
            similarity = item.get("similarity") if isinstance(item, dict) else None
            person_id = item.get("person_id") if isinstance(item, dict) else None
            if not isinstance(person_id, str) or isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
                raise FrInvalidResponse("/search result lacks person_id/similarity")
            hits.append(SearchHit(person_id, float(similarity)))  # names, crime fields etc. are dropped here
        threshold = body.get("threshold")
        return SearchResult(tuple(hits), float(threshold) if isinstance(threshold, (int, float)) else None)

    @staticmethod
    def _json(status: int, data: bytes) -> Any:
        try:
            return json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise FrInvalidResponse(f"HTTP {status} body is not JSON") from None

    def _raise_for_error(self, status: int, data: bytes, endpoint: str) -> None:
        detail = self._error_detail(data)
        if status == 400:
            kind = classify_bad_request(detail)
            if kind == "service_not_ready":
                raise FrServiceNotReady("FR-DOPAMS gallery is empty")
            if kind == "unrecognised":
                raise FrContractError(f"{endpoint} returned an unrecognised 400: {detail[:120]}")
            raise FrImageRejected(kind, detail[:200])
        if status == 500:
            raise FrServerError(f"{endpoint} returned HTTP 500")
        raise FrContractError(f"{endpoint} returned unexpected HTTP {status}")

    @staticmethod
    def _error_detail(data: bytes) -> str:
        try:
            detail = json.loads(data).get("detail")
        except (ValueError, UnicodeDecodeError, AttributeError):
            return ""
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):  # FastAPI 422: keep field locations and messages only
            return "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg', '')}" for d in detail if isinstance(d, dict))
        return ""

    # ----------------------------------------------------------- transport

    def _request(self, method: str, path: str, body: Optional[bytes] = None, headers: Optional[dict[str, str]] = None):
        cfg = self._config
        last = "no attempt"
        for attempt in range(cfg.retries + 1):
            started = time.monotonic()
            try:
                status, data = self._once(method, path, body, headers or {})
            except (OSError, http.client.HTTPException) as exc:  # refused, reset, timed out, malformed reply
                last = type(exc).__name__
                log.warning("FR-DOPAMS %s %s failed (%s), attempt %d/%d", method, path, last, attempt + 1, cfg.retries + 1)
            else:
                log.debug("FR-DOPAMS %s %s -> %d in %.2fs", method, path, status, time.monotonic() - started)
                # A deliberate 503 from /compare ("compare_disabled") is configuration, not an outage.
                if status not in _RETRYABLE_STATUS or b"compare_disabled" in data:
                    return status, data
                last = f"HTTP {status}"
            if attempt < cfg.retries:
                time.sleep(cfg.backoff * (2 ** attempt))
        raise FrUnavailable(f"{method} {path} failed after {cfg.retries + 1} attempts ({last})")

    def _once(self, method: str, path: str, body: Optional[bytes], headers: dict[str, str]) -> tuple[int, bytes]:
        cfg = self._config
        factory = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        conn = factory(self._host, self._port, timeout=cfg.connect_timeout)
        try:
            conn.connect()
            conn.sock.settimeout(cfg.read_timeout)  # connect and read have separate limits
            conn.request(method, self._prefix + path, body=body, headers={"Accept": "application/json", **headers})
            response = conn.getresponse()
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise FrInvalidResponse("response larger than the allowed size")
            return response.status, data
        finally:
            conn.close()
