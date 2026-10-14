"""File-type detection by signature and image decoding checks.

Neither the file name, the database ``mime_type`` (NULL for every row) nor the
file server's Content-Type (always ``application/octet-stream``) is trusted.
"""

from __future__ import annotations

import io
import warnings
import zlib
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .models import Status

SUPPORTED_IMAGE_TYPES = frozenset({"jpeg", "png", "webp"})
_PIL_FORMAT = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


def sniff_format(head: bytes) -> str:
    """Name the file type from its leading bytes, or ``"unknown"``."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head.startswith(b"BM") and len(head) >= 14:
        return "bmp"
    if head.startswith(b"PK\x03\x04"):
        return "zip"  # also .docx / .xlsx
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"  # legacy .doc / .xls
    if head.lstrip()[:5].lower() in (b"<html", b"<!doc"):
        return "html"  # e.g. a server error page returned with HTTP 200
    return "unknown"


@dataclass(frozen=True)
class ImageCheck:
    status: Status
    width: Optional[int] = None
    height: Optional[int] = None
    reason: Optional[str] = None


def validate_image(data: bytes, detected_type: str, min_dimension: int = 32) -> ImageCheck:
    """Fully decode ``data``; only a clean decode of the expected format is usable."""
    expected = _PIL_FORMAT.get(detected_type)
    if expected is None:
        return ImageCheck(Status.UNSUPPORTED_FORMAT, reason=f"{detected_type} is not a supported image type")
    try:
        with warnings.catch_warnings():
            # Decompression-bomb warnings become errors so huge images are rejected, not decoded.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format != expected:
                    return ImageCheck(Status.CORRUPT_IMAGE, reason=f"signature says {expected} but decoder found {probe.format}")
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return ImageCheck(Status.IMAGE_TOO_LARGE, reason="image pixel count exceeds the safe decoding limit")
    except (OSError, SyntaxError, ValueError, EOFError, zlib.error, Image.UnidentifiedImageError) as exc:
        return ImageCheck(Status.CORRUPT_IMAGE, reason=f"decode failed: {type(exc).__name__}")
    if min(width, height) < min_dimension:
        return ImageCheck(
            Status.IMAGE_TOO_SMALL, width, height, f"image is {width}x{height}; minimum side is {min_dimension}px"
        )
    return ImageCheck(Status.USABLE, width, height)
