"""Error types. The key distinction: a face problem is data; an operational error is not.

``FrImageRejected`` means FR-DOPAMS looked at the image and refused it (no usable
face). ``FrOperationalError`` means the service itself could not do its job. The
two must never be conflated: an outage is not a face mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class FrError(Exception):
    """Base class for every FR-DOPAMS client error."""


class FrConfigError(FrError):
    """Missing or invalid client configuration."""


class FrWritePolicyError(FrError):
    """The call would append rows to the FR-DOPAMS database and was not explicitly allowed."""


class FrImageRejected(FrError):
    """FR-DOPAMS refused the image itself. ``kind`` is one of IMAGE_KINDS."""

    def __init__(self, kind: str, detail: str, side: Optional[str] = None):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.side = kind, detail, side


IMAGE_KINDS = ("no_face", "multiple_faces", "low_quality", "invalid_image")


@dataclass(frozen=True)
class SideRejection:
    """FR-DOPAMS refused the image on one side ("a" or "b") of a comparison."""

    side: str
    kind: str  # one of IMAGE_KINDS
    reason: Optional[str]  # e.g. too_small, too_blurred, undecodable
    detail: str


class FrPairRejected(FrError):
    """``POST /compare`` found one or both images unusable. Every unusable side is listed."""

    def __init__(self, rejections: tuple[SideRejection, ...]):
        super().__init__(", ".join(f"{r.side}: {r.kind}" for r in rejections))
        self.rejections = rejections

    def for_side(self, side: str) -> Optional[SideRejection]:
        return next((r for r in self.rejections if r.side == side), None)


class FrOperationalError(FrError):
    kind = "operational"

    def __init__(self, detail: str = ""):
        super().__init__(f"{self.kind}: {detail}" if detail else self.kind)
        self.detail = detail


class FrUnavailable(FrOperationalError):
    """Connection failure, timeout, or 502/503/504 that persisted through all retries."""

    kind = "unavailable"


class FrServerError(FrOperationalError):
    """HTTP 500 from the service (an unexpected exception inside FR-DOPAMS)."""

    kind = "server_error"


class FrServiceNotReady(FrOperationalError):
    """The service is up but has nothing to search (empty gallery), so it never examined the image."""

    kind = "service_not_ready"


class FrInvalidResponse(FrOperationalError):
    """HTTP success, but the body does not match the contract this client was written against."""

    kind = "invalid_response"


class FrAuthError(FrOperationalError):
    """The service refused our API key (HTTP 401), or has /compare disabled (HTTP 503)."""

    kind = "auth_error"


class FrContractError(FrOperationalError):
    """A status or message this client does not recognise (API drift, or a bug in our request)."""

    kind = "contract_error"
