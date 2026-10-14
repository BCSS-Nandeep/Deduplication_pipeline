"""Resolve one media record to a validated, usable image (or a recorded reason why not)."""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .file_server import (
    DEFAULT_FILE_SERVER_BASE_URL,
    EXTENSION_FALLBACKS,
    Fetcher,
    InvalidPath,
    build_candidates,
)
from .formats import SUPPORTED_IMAGE_TYPES, sniff_format, validate_image
from .models import TRANSIENT_STATUSES, VERIFIED_ACCESSIBLE, MediaResult, MediaRow, Status

log = logging.getLogger(__name__)

PROBE_BYTES = 64


@dataclass(frozen=True)
class ResolverConfig:
    base_url: str = DEFAULT_FILE_SERVER_BASE_URL
    extensions: tuple[str, ...] = EXTENSION_FALLBACKS
    max_image_bytes: int = 10 * 1024 * 1024  # FR-DOPAMS rejects uploads above MAX_UPLOAD_MB=10
    min_dimension: int = 32
    probe_unverified: bool = False  # also try rows whose availability_status is not VERIFIED_ACCESSIBLE
    category_priority: tuple[str, ...] = ("person/MEDIA", "person/IDENTITY_DETAILS")

    def signature(self, face_check: bool) -> dict[str, Any]:
        """Everything that changes a result; a checkpoint from other settings is discarded."""
        return {
            "base_url": self.base_url,
            "extensions": list(self.extensions),
            "max_image_bytes": self.max_image_bytes,
            "min_dimension": self.min_dimension,
            "probe_unverified": self.probe_unverified,
            "face_check": face_check,
        }


@dataclass(frozen=True)
class FaceCheckResult:
    """Outcome of an optional face-detectability check on decoded image bytes."""

    status: str  # "ok" | "no_face" | "multiple_faces" | "low_quality"
    detail: Optional[str] = None


FaceChecker = Callable[[bytes], FaceCheckResult]
_FACE_STATUS = {
    "no_face": Status.NO_FACE,
    "multiple_faces": Status.MULTIPLE_FACES,
    "low_quality": Status.LOW_QUALITY,
}


def resolve_media(
    row: MediaRow, config: ResolverConfig, fetcher: Fetcher, face_check: Optional[FaceChecker] = None
) -> MediaResult:
    path_ref, path_source = row.chosen_path()
    result = MediaResult(
        media_id=row.media_id,
        status=Status.INTERNAL_ERROR,
        availability_status=row.availability_status,
        path_ref=path_ref,
        path_source=path_source,
    )

    def done(status: Status, reason: Optional[str] = None) -> MediaResult:
        result.status = status
        result.skip_reason = reason
        return result

    if row.availability_status != VERIFIED_ACCESSIBLE and not config.probe_unverified:
        return done(Status.AVAILABILITY_NOT_VERIFIED, f"availability_status is {row.availability_status}")
    if path_ref is None:
        return done(Status.NO_FILE_PATH, "no usable path column on the media record")
    try:
        candidates = build_candidates(path_ref, config.base_url, config.extensions)
    except InvalidPath as exc:
        return done(Status.INVALID_PATH, str(exc))

    transient = denied = False
    empty_hit = unknown_hit = None
    found = None
    for candidate in candidates:
        probe = fetcher.fetch(candidate.url, PROBE_BYTES)
        if probe.status in (200, 206):
            if not probe.body:
                empty_hit = empty_hit or candidate
                continue
            detected = sniff_format(probe.body)
            if detected in ("unknown", "html"):
                unknown_hit = unknown_hit or (candidate, detected)
                continue
            found = (candidate, detected)
            break
        if probe.status is None or probe.status >= 500 or probe.status == 429:
            transient = True
        elif probe.status in (401, 403):
            denied = True
        # any other 4xx (normally 404) just means "not at this name"

    if found is None:
        if transient:
            return done(Status.FETCH_ERROR, "file server error or timeout while locating the file")
        if empty_hit is not None:
            result.resolved_suffix = empty_hit.suffix
            result.byte_size = 0
            return done(Status.EMPTY_FILE, "file exists but has no content")
        if unknown_hit is not None:
            result.resolved_suffix, result.detected_type = unknown_hit[0].suffix, unknown_hit[1]
            return done(Status.UNKNOWN_FORMAT, f"content is not a recognised file type ({unknown_hit[1]})")
        if denied:
            return done(Status.ACCESS_DENIED, "file server refused access (HTTP 401/403)")
        return done(Status.FILE_NOT_FOUND, "no file at the path or any supported extension")

    candidate, detected = found
    result.resolved_suffix, result.detected_type = candidate.suffix, detected
    if detected == "pdf":
        return done(Status.PDF_SKIPPED, "PDF documents are not used for face recognition")
    if detected not in SUPPORTED_IMAGE_TYPES:
        return done(Status.UNSUPPORTED_FORMAT, f"{detected} is not a supported image type (JPEG, PNG, WebP)")

    full = fetcher.fetch(candidate.url, config.max_image_bytes)
    if full.status not in (200, 206):
        if full.status in (404, 410):
            return done(Status.FILE_NOT_FOUND, "file disappeared between probe and download")
        return done(Status.FETCH_ERROR, f"download failed ({full.error or full.status})")
    if full.truncated:
        return done(Status.FILE_TOO_LARGE, f"file is larger than {config.max_image_bytes} bytes")
    result.byte_size = len(full.body)

    check = validate_image(full.body, detected, config.min_dimension)
    result.width, result.height = check.width, check.height
    if check.status is not Status.USABLE:
        return done(check.status, check.reason)

    if face_check is not None:
        try:
            outcome = face_check(full.body)
        except Exception as exc:
            result.face_check_status = "error"
            return done(Status.FACE_CHECK_ERROR, f"face check failed: {type(exc).__name__}")
        result.face_check_status = outcome.status
        if outcome.status != "ok":
            return done(_FACE_STATUS.get(outcome.status, Status.FACE_CHECK_ERROR), outcome.detail)
    return done(Status.USABLE)


# ----------------------------------------------------------------- checkpoint

class Checkpoint:
    """Append-only JSONL log of finished media results, so a rerun skips finished work.

    A stored result is reused only if the media record still has the same path
    and availability, the run settings are unchanged, and the outcome was not
    transient. A half-written last line (killed run) is ignored.
    """

    def __init__(self, path: Path, signature: dict[str, Any], fresh: bool = False):
        self._path = path
        self._signature = signature
        self._lock = threading.Lock()
        self._results: dict[str, MediaResult] = {}
        if not fresh:
            self._load()
        if not self._results:
            self._rewrite()
        self._handle = open(self._path, "a", encoding="utf-8")

    def _load(self) -> None:
        if not self._path.is_file():
            return
        with open(self._path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records or records[0].get("type") != "header" or records[0].get("signature") != self._signature:
            log.warning("Checkpoint settings differ or file is unreadable; starting from scratch.")
            return
        for record in records[1:]:
            if record.get("type") == "result":
                result = MediaResult.from_dict(record["result"])
                self._results[result.media_id] = result

    def _rewrite(self) -> None:
        self._write_atomic(self._results.values())

    def _write_atomic(self, results: Iterable[MediaResult]) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"type": "header", "signature": self._signature}) + "\n")
            for result in results:
                fh.write(json.dumps({"type": "result", "result": result.to_dict()}) + "\n")
        os.replace(tmp, self._path)

    def get(self, row: MediaRow) -> Optional[MediaResult]:
        stored = self._results.get(row.media_id)
        if stored is None or stored.status in TRANSIENT_STATUSES:
            return None
        path_ref, _ = row.chosen_path()
        if stored.path_ref != path_ref or stored.availability_status != row.availability_status:
            return None
        return stored

    def put(self, result: MediaResult) -> None:
        with self._lock:
            self._results[result.media_id] = result
            self._handle.write(json.dumps({"type": "result", "result": result.to_dict()}) + "\n")
            self._handle.flush()

    def close(self, keep: Iterable[str]) -> None:
        """Compact to the media ids of this run (drops stale rows) and close."""
        self._handle.close()
        wanted = set(keep)
        self._write_atomic(r for m, r in sorted(self._results.items()) if m in wanted)


def resolve_all(
    rows: list[MediaRow],
    config: ResolverConfig,
    fetcher: Fetcher,
    checkpoint: Checkpoint,
    workers: int = 8,
    face_check: Optional[FaceChecker] = None,
    progress_every: int = 500,
) -> dict[str, MediaResult]:
    """Resolve every row; returns results keyed by media_id."""
    results: dict[str, MediaResult] = {}
    pending: list[MediaRow] = []
    for row in rows:
        cached = checkpoint.get(row)
        if cached is not None:
            results[row.media_id] = cached
        else:
            pending.append(row)
    log.info("Media rows: %d total, %d reused from checkpoint, %d to resolve", len(rows), len(results), len(pending))

    def work(row: MediaRow) -> MediaResult:
        try:
            return resolve_media(row, config, fetcher, face_check)
        except Exception as exc:  # one bad record must not stop the run
            log.exception("Unexpected failure resolving media %s", row.media_id)
            path_ref, path_source = row.chosen_path()
            return MediaResult(
                row.media_id, Status.INTERNAL_ERROR, row.availability_status, path_ref, path_source,
                skip_reason=f"unexpected error: {type(exc).__name__}",
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result.media_id] = result
            checkpoint.put(result)
            if index % progress_every == 0:
                log.info("Resolved %d / %d", index, len(pending))
    return results
