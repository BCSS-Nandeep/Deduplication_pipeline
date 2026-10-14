"""Load the selected image of a person from the Phase 2 outputs.

Reuses Phase 2's URL building, file-server fetcher and signature detection; it only re-reads a file
that Phase 2 already validated, using the extension that Phase 2 recorded as working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..image_resolution.file_server import Fetcher, InvalidPath, build_candidates
from ..image_resolution.formats import SUPPORTED_IMAGE_TYPES, sniff_format
from ..image_resolution.manifest import MANIFEST_FILE, PERSON_SUMMARY_FILE
from ..image_resolution.resolver import ResolverConfig


class ImageBytes(bytes):
    """Image data labelled with the manifest identity of the file it came from (``cache_key``: media id, path, recorded extension).

    It is an ordinary ``bytes`` everywhere. The label lets an in-process face engine remember its answer for a file within one run; it is an identity
    taken from the manifest, never a hash or fingerprint of the content."""

    cache_key: Optional[str] = None


class ImageLoadError(Exception):
    """The image Phase 2 selected can no longer be loaded. ``status`` uses Phase 2 status names."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class PersonImage:
    """What Phase 2 recorded for one person."""

    person_id: str
    person_status: str  # usable_image_available | no_person_media_records | no_accessible_media | no_usable_image
    # Every manifest record Phase 2 judged usable, in its ``person_rank`` order (the first is the selected one).
    usable_images: tuple[dict[str, Any], ...] = ()
    unusable_reasons: Optional[dict[str, int]] = None

    @property
    def selected(self) -> Optional[dict[str, Any]]:
        return self.usable_images[0] if self.usable_images else None

    @property
    def has_usable_image(self) -> bool:
        return bool(self.usable_images)


class ImageIndex:
    """Person -> usable images, read from ``person_summary.jsonl`` and ``image_manifest.jsonl``."""

    def __init__(self, people: dict[str, PersonImage]):
        self._people = people

    @classmethod
    def from_output_dir(cls, directory: Path) -> "ImageIndex":
        usable: dict[str, list[dict[str, Any]]] = {}
        with open(directory / MANIFEST_FILE, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record["usable_for_face_recognition"]:
                    usable.setdefault(record["person_id"], []).append(record)
        rows = []
        with open(directory / PERSON_SUMMARY_FILE, encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return cls.from_parsed(usable, rows)

    @classmethod
    def from_parsed(cls, usable: dict[str, list[dict[str, Any]]], person_rows: list[dict[str, Any]]) -> "ImageIndex":
        """Build the index from already parsed records (``usable``: person -> usable manifest records; ``person_rows``: the person summaries).
        Lets the input validation, which parses both files anyway, hand its records over instead of the files being parsed a second time."""
        people: dict[str, PersonImage] = {}
        for row in person_rows:
            ordered = sorted(usable.get(row["person_id"], []), key=lambda r: (r["person_rank"], r["media_id"]))
            people[row["person_id"]] = PersonImage(
                row["person_id"], row["person_status"], tuple(ordered), row.get("unusable_reasons")
            )
        return cls(people)

    def get(self, person_id: str) -> PersonImage:
        # A person absent from the manifest was never resolved; say so rather than guessing.
        return self._people.get(person_id) or PersonImage(person_id, "not_in_manifest")


class ManifestImageLoader:
    def __init__(self, config: ResolverConfig, fetcher: Fetcher):
        self._config, self._fetcher = config, fetcher

    def load(self, record: dict[str, Any]) -> bytes:
        try:
            candidates = build_candidates(record["path_ref"], self._config.base_url, self._config.extensions)
        except InvalidPath as exc:
            raise ImageLoadError("invalid_path", str(exc)) from None
        wanted = record.get("resolved_suffix") or ""
        url = next((c.url for c in candidates if c.suffix == wanted), None)
        if url is None:
            raise ImageLoadError("file_not_found", "recorded extension is no longer a candidate")
        result = self._fetcher.fetch(url, self._config.max_image_bytes)
        if result.status in (404, 410):
            raise ImageLoadError("file_not_found", "file no longer on the file server")
        if result.status in (401, 403):
            raise ImageLoadError("access_denied", f"HTTP {result.status}")
        if result.status not in (200, 206):
            raise ImageLoadError("fetch_error", result.error or f"HTTP {result.status}")
        if result.truncated:
            raise ImageLoadError("file_too_large", f"larger than {self._config.max_image_bytes} bytes")
        if sniff_format(result.body[:64]) not in SUPPORTED_IMAGE_TYPES:
            raise ImageLoadError("unknown_format", "content is no longer a supported image")
        data = ImageBytes(result.body)
        data.cache_key = f"{record.get('media_id')}@{record['path_ref']}{wanted}"
        return data
