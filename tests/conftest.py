"""Shared fixtures: generated image bytes, an in-memory file server, a fake DB connection."""

from __future__ import annotations

import io
from typing import Any, Optional, Union

import pytest
from PIL import Image

from dedup_pipeline.image_resolution.file_server import FetchResult, build_candidates
from dedup_pipeline.image_resolution.models import MediaRow
from dedup_pipeline.image_resolution.resolver import ResolverConfig

BASE = "http://files.test/files/"


def make_image(fmt: str, size=(120, 120)) -> bytes:
    # Noise-free gradient keeps the files small but fully decodable.
    image = Image.new("RGB", size)
    image.putdata([(x % 256, (x // size[0]) % 256, 90) for x in range(size[0] * size[1])])
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def make_portrait_image(fmt: str = "JPEG", size=(180, 220)) -> bytes:
    """Synthetic head-and-shoulders layout for Phase 1 media classification tests.

    Not a real photograph and not derived from production media.
    """
    width, height = size
    image = Image.new("RGB", size, (40, 55, 70))
    pixels = image.load()
    cx, cy = width // 2, int(height * 0.38)
    rx, ry = int(width * 0.28), int(height * 0.22)
    for y in range(height):
        for x in range(width):
            dx = (x - cx) / max(1, rx)
            dy = (y - cy) / max(1, ry)
            if dx * dx + dy * dy <= 1.0:
                pixels[x, y] = (210, 160, 130)  # skin-like oval
            elif abs(x - cx) < width * 0.22 and cy + ry < y < int(height * 0.92):
                pixels[x, y] = (55, 70, 110)  # shoulders
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def make_document_image(fmt: str = "JPEG", size=(420, 260)) -> bytes:
    """Synthetic identity-document layout (not real Aadhaar and not production data)."""
    width, height = size
    image = Image.new("RGB", size, (245, 245, 248))
    pixels = image.load()
    header = int(height * 0.22)
    footer = int(height * 0.82)
    for y in range(height):
        for x in range(width):
            if y < header:
                pixels[x, y] = (25, 70, 160)
            elif y >= footer:
                pixels[x, y] = (20, 55, 130)
            elif y % 7 == 0:
                # Dense horizontal "text" lines without OCR or real personal data.
                pixels[x, y] = (30, 30, 30) if (x // 3 + y) % 5 else (220, 220, 225)
    # Small corner photo patch (ID cards place a photo aside, not as a full-frame portrait).
    for y in range(int(height * 0.30), int(height * 0.70)):
        for x in range(int(width * 0.72), int(width * 0.92)):
            pixels[x, y] = (200, 155, 125)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def make_group_image(fmt: str = "JPEG", size=(240, 180)) -> bytes:
    """Several skin-like blobs; classifier should leave this as UNKNOWN."""
    image = Image.new("RGB", size, (90, 100, 110))
    pixels = image.load()
    centres = [(50, 70), (120, 75), (190, 70)]
    for cx, cy in centres:
        for y in range(cy - 28, cy + 28):
            for x in range(cx - 22, cx + 22):
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    if ((x - cx) / 22) ** 2 + ((y - cy) / 28) ** 2 <= 1:
                        pixels[x, y] = (205, 155, 125)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def make_blank_image(fmt: str = "JPEG", size=(160, 160)) -> bytes:
    image = Image.new("RGB", size, (128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def make_ambiguous_header_image(fmt: str = "JPEG", size=(200, 200)) -> bytes:
    """Coloured header band plus a central skin-like block (D1 regression fixture).

    Must not be accepted as CANDIDATE_PORTRAIT; not real ID or production data.
    """
    width, height = size
    image = Image.new("RGB", size, (180, 180, 190))
    pixels = image.load()
    for y in range(0, int(height * 0.15)):
        for x in range(width):
            pixels[x, y] = (40, 80, 140)
    for y in range(int(height * 0.20), int(height * 0.80)):
        for x in range(int(width * 0.30), int(width * 0.70)):
            pixels[x, y] = (160, 140, 120)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n" + b"x" * 200


class FakeFetcher:
    """Serves ``files`` keyed by URL. A value is bytes (200), an int (HTTP status) or an Exception."""

    def __init__(self, files: dict[str, Union[bytes, int, Exception]]):
        self.files = files
        self.calls: list[tuple[str, int]] = []

    def fetch(self, url: str, max_bytes: int) -> FetchResult:
        self.calls.append((url, max_bytes))
        entry = self.files.get(url, 404)
        if isinstance(entry, Exception):
            return FetchResult(None, error=type(entry).__name__)
        if isinstance(entry, int):
            return FetchResult(entry, error=f"HTTP {entry}")
        return FetchResult(206, entry[:max_bytes], truncated=len(entry) > max_bytes)

    @property
    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


def url_for(path: str, suffix: str = "") -> str:
    return build_candidates(path, BASE)[0].url + suffix


def media_row(
    media_id: str = "M1",
    person_id: str = "P1",
    path: Optional[str] = "person/media/abc",
    category: str = "person/MEDIA",
    availability: str = "VERIFIED_ACCESSIBLE",
) -> MediaRow:
    return MediaRow(media_id, person_id, category, availability, resolved_relative_path=path)


@pytest.fixture
def config() -> ResolverConfig:
    return ResolverConfig(base_url=BASE)


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    # Portrait-like so Phase 1 media classification keeps the image usable for face matching.
    return make_portrait_image("JPEG")


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    return make_portrait_image("PNG")


@pytest.fixture(scope="session")
def webp_bytes() -> bytes:
    return make_portrait_image("WEBP")


# -------------------------------------------------------------- fake database

class FakeCursor:
    def __init__(self, db: "FakeDb"):
        self._db = db
        self._result: list[tuple] = []

    def execute(self, sql: str, params: Any = None) -> "FakeCursor":
        from dedup_pipeline.db import check_statement_is_read_only

        check_statement_is_read_only(sql)  # the fake enforces the same guard as production
        self._db.statements.append(sql)
        if sql.startswith("SHOW transaction_read_only") or sql.startswith("SHOW default_transaction_read_only"):
            self._result = [(self._db.read_only,)]
        elif "inet_server_addr()" in sql:
            self._result = [(getattr(self._db, "server_address", "local"),)]
        elif "current_database()" in sql:
            self._result = [(self._db.database, "PostgreSQL 16.14 test")]
        elif "FROM public.persons_unified" in sql:
            self._result = list(self._db.persons)
        elif "FROM public.media_unified" in sql:
            wanted = set(params[1])
            self._result = [r for r in self._db.media if r[1] in wanted]
        else:
            raise AssertionError(f"unexpected SQL: {sql}")
        return self

    def fetchone(self):
        return self._result[0]

    def fetchall(self):
        return self._result

    def close(self) -> None:
        pass


class FakeDb:
    def __init__(self, persons, media, read_only="on", database="dopams_cctns_v2", server_address="local"):
        self.persons, self.media = persons, media
        self.read_only, self.database = read_only, database
        self.server_address = server_address
        self.statements: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        pass
