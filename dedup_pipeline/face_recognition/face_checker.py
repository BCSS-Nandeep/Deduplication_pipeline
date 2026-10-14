"""Adapter from the FR-DOPAMS client to the Phase 2 ``FaceChecker`` hook.

``resolve_media(..., face_check=FrDopamsFaceChecker(client))`` runs after an image has already been
decoded successfully. The only existing endpoint that examines a single image is ``POST /search``,
whose success implies FR-DOPAMS found one dominant, large-enough, sharp-enough face. Because every
successful call writes a ``match_logs`` row in the FR-DOPAMS database, the client refuses it unless
``allow_audit_log_writes`` is set; this checker is therefore *off by default* and is not wired into the
CLI. Operational failures are raised, never turned into a face verdict: the resolver records them as
``face_check_error`` (retried on the next run), not as ``no_face``.
"""

from __future__ import annotations

from ..image_resolution.resolver import FaceCheckResult
from .client import FrDopamsClient
from .errors import FrImageRejected


class FrDopamsFaceChecker:
    def __init__(self, client: FrDopamsClient):
        self._client = client

    def __call__(self, image_bytes: bytes) -> FaceCheckResult:
        try:
            self._client.search(image_bytes)
        except FrImageRejected as rejected:
            # FR-DOPAMS could not decode or accept the image although Pillow could: report it as
            # unusable quality with the reason, rather than inventing a face verdict.
            kind = "low_quality" if rejected.kind == "invalid_image" else rejected.kind
            return FaceCheckResult(kind, f"FR-DOPAMS: {rejected.detail}")
        return FaceCheckResult("ok")
