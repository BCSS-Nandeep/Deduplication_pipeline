"""Data shapes shared by the image-resolution modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

VERIFIED_ACCESSIBLE = "VERIFIED_ACCESSIBLE"
PATH_FIELDS = ("resolved_relative_path", "source_local_path", "original_attach_path")


class Status(str, Enum):
    """Why a media record is, or is not, usable for face recognition."""

    USABLE = "usable"
    # Skipped before any network access.
    AVAILABILITY_NOT_VERIFIED = "availability_not_verified"
    NO_FILE_PATH = "no_file_path"
    INVALID_PATH = "invalid_path"
    # File-server outcomes.
    FILE_NOT_FOUND = "file_not_found"
    ACCESS_DENIED = "access_denied"
    FETCH_ERROR = "fetch_error"
    # Content outcomes.
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    PDF_SKIPPED = "pdf_skipped"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNKNOWN_FORMAT = "unknown_format"
    CORRUPT_IMAGE = "corrupt_image"
    IMAGE_TOO_SMALL = "image_too_small"
    IMAGE_TOO_LARGE = "image_too_large"
    # Optional face-detectability check (supplied later by the FR-DOPAMS stage).
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOW_QUALITY = "low_quality"
    FACE_CHECK_ERROR = "face_check_error"
    # Phase 1 media classification (after decode, before face matching).
    IDENTITY_DOCUMENT_EXCLUDED = "identity_document_excluded"
    MEDIA_CLASS_UNCERTAIN = "media_class_uncertain"
    # A bug rather than a data problem.
    INTERNAL_ERROR = "internal_error"


# Outcomes that may succeed on a later run, so a checkpoint never trusts them.
TRANSIENT_STATUSES = frozenset({Status.FETCH_ERROR, Status.INTERNAL_ERROR, Status.FACE_CHECK_ERROR})


def normalize_path(path: str) -> str:
    """Canonical form used for lookups and for shared-path detection."""
    return path.strip().lstrip("/")


@dataclass(frozen=True)
class Person:
    person_id: str
    source_system: str


@dataclass(frozen=True)
class MediaRow:
    media_id: str
    person_id: str
    attachment_category: Optional[str]
    availability_status: str
    resolved_relative_path: Optional[str] = None
    source_local_path: Optional[str] = None
    original_attach_path: Optional[str] = None

    def chosen_path(self) -> tuple[Optional[str], Optional[str]]:
        """Same precedence as DOPAMS-BE ``unified-profile.ts`` ``loadFiles``."""
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if value and value.strip():
                return normalize_path(value), name
        return None, None


@dataclass
class MediaResult:
    media_id: str
    status: Status
    availability_status: str
    path_ref: Optional[str] = None
    path_source: Optional[str] = None
    resolved_suffix: Optional[str] = None
    detected_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    byte_size: Optional[int] = None
    skip_reason: Optional[str] = None
    face_check_status: str = "not_checked"
    media_class: Optional[str] = None
    media_class_confidence: Optional[float] = None
    media_class_reason: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.status is Status.USABLE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaResult":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields["status"] = Status(fields["status"])
        return cls(**fields)
