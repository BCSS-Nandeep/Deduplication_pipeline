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
    return make_image("JPEG")


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    return make_image("PNG")


@pytest.fixture(scope="session")
def webp_bytes() -> bytes:
    return make_image("WEBP")


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
    def __init__(self, persons, media, read_only="on", database="dopams_cctns_v2"):
        self.persons, self.media = persons, media
        self.read_only, self.database = read_only, database
        self.statements: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        pass
