"""Face evidence for one candidate pair, with honest statuses and deterministic fallback.

Rules:

* both persons have a usable image -> ask the comparer (``POST /compare`` via ``FrDopamsPairComparer``);
* either has none -> ``skipped_no_image``; no comparison, ``face_similarity`` stays ``None``;
* if FR-DOPAMS refuses an image (``no_face``, ``multiple_faces``, ``low_quality``, ``invalid_image``), the
  person's *next* usable image, in Phase 2 ``person_rank`` order, is tried; only the failing side moves,
  at most ``max_images_per_person`` images per person. Every attempt is recorded in ``attempts``. When a
  side runs out of images, the real failure status is kept (never turned into a similarity of 0);
* a service failure (unreachable, auth, 5xx, unreadable image file) is ``error`` with an ``error_kind``
  and is never retried with another image: it says nothing about the faces and must not look like a
  mismatch;
* no comparer -> ``not_checked``.

``face_similarity`` is what the service returned, raw (cosine, range [-1, 1], may be negative), with its
``similarity_metric``. ``matched`` / ``not_matched`` is the *service's* ``same_person`` decision and
``decision_threshold`` the threshold behind it; this module applies no threshold of its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from .client import FrDopamsClient
from .errors import FrOperationalError, FrPairRejected
from .image_source import ImageIndex, ImageLoadError, ManifestImageLoader, PersonImage


class PairStatus(str, Enum):
    # Names follow the face_match_status vocabulary in the (unapplied) dedup migration.
    NOT_CHECKED = "not_checked"
    SKIPPED_NO_IMAGE = "skipped_no_image"
    SKIPPED_SHARED_IMAGE = "skipped_shared_image"  # the same file is attached to both persons: not independent evidence
    NO_FACE = "no_face"
    LOW_QUALITY = "low_quality"
    INVALID_IMAGE = "invalid_image"
    MULTIPLE_FACES = "multiple_faces"
    COMPARED = "compared"  # similarity measured, service gave no same-person decision
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    ERROR = "error"


_REJECTION_STATUS = {
    "no_face": PairStatus.NO_FACE,
    "multiple_faces": PairStatus.MULTIPLE_FACES,
    "low_quality": PairStatus.LOW_QUALITY,
    "invalid_image": PairStatus.INVALID_IMAGE,
}


@dataclass(frozen=True)
class FallbackPolicy:
    # Failures that belong to one particular image, so another image of the same person may succeed.
    fallback_on: frozenset[str] = frozenset({"no_face", "multiple_faces", "low_quality", "invalid_image"})
    max_images_per_person: int = 3


@dataclass(frozen=True)
class ComparisonOutcome:
    similarity: float
    metric: str  # as the service documents it, e.g. "cosine"
    decision: Optional[bool] = None  # the service's same-person decision, if it makes one
    threshold: Optional[float] = None  # the service's threshold behind that decision
    model: Optional[str] = None  # the face model that produced the embeddings, as reported by the service


class PairComparer(Protocol):
    def compare(self, image_a: bytes, image_b: bytes) -> ComparisonOutcome:
        """May raise ``FrPairRejected`` (side-specific) or ``FrOperationalError``."""


class FrDopamsPairComparer:
    """``PairComparer`` backed by the FR-DOPAMS ``POST /compare`` endpoint."""

    def __init__(self, client: FrDopamsClient):
        self._client = client

    def compare(self, image_a: bytes, image_b: bytes) -> ComparisonOutcome:
        result = self._client.compare(image_a, image_b)
        return ComparisonOutcome(result.similarity, result.metric, result.same_person, result.threshold, result.model or None)


@dataclass(frozen=True)
class PairFaceEvidence:
    person_a_id: str
    person_b_id: str
    status: PairStatus
    face_similarity: Optional[float] = None
    similarity_metric: Optional[str] = None
    decision_threshold: Optional[float] = None
    model: Optional[str] = None
    reason: Optional[str] = None
    failed_side: Optional[str] = None  # "a", "b" or "both"
    side_failures: Optional[dict[str, str]] = None  # {"a": "no_face", ...} for the sides that ended failed
    error_kind: Optional[str] = None
    media_id_a: Optional[str] = None  # the images behind this result (the last pair tried)
    media_id_b: Optional[str] = None
    fallback_used: bool = False
    attempts: tuple[dict[str, Any], ...] = ()
    person_status_a: Optional[str] = None
    person_status_b: Optional[str] = None
    unusable_reasons_a: Optional[dict[str, int]] = None
    unusable_reasons_b: Optional[dict[str, int]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["attempts"] = list(self.attempts)
        return data


def compare_pair(
    person_a_id: str,
    person_b_id: str,
    index: ImageIndex,
    loader: ManifestImageLoader,
    comparer: Optional[PairComparer],
    policy: FallbackPolicy = FallbackPolicy(),
    known_rejection: Optional[Callable[[str, str], Optional[FrPairRejected]]] = None,
) -> PairFaceEvidence:
    """``known_rejection(media_a, media_b)`` (opt-in, Phase 9) may return the ``FrPairRejected`` the service would raise for this exact
    pair of images when stored earlier answers already determine it completely; the attempt then goes through the very same rejection
    handling below, without downloading the images or calling the service. ``None`` (the default, and the answer whenever anything is
    not fully known) means the attempt is made as before."""
    people: dict[str, PersonImage] = {"a": index.get(person_a_id), "b": index.get(person_b_id)}
    identity = {
        "person_a_id": person_a_id,
        "person_b_id": person_b_id,
        "person_status_a": people["a"].person_status,
        "person_status_b": people["b"].person_status,
        "unusable_reasons_a": people["a"].unusable_reasons or None,
        "unusable_reasons_b": people["b"].unusable_reasons or None,
    }

    if not (people["a"].has_usable_image and people["b"].has_usable_image):
        missing = [side for side in "ab" if not people[side].has_usable_image]
        return PairFaceEvidence(
            status=PairStatus.SKIPPED_NO_IMAGE,
            reason="no usable image for " + " and ".join(f"person {s}" for s in missing),
            failed_side=missing[0] if len(missing) == 1 else None,
            media_id_a=people["a"].selected["media_id"] if people["a"].selected else None,
            media_id_b=people["b"].selected["media_id"] if people["b"].selected else None,
            **identity,
        )
    if comparer is None:
        return PairFaceEvidence(
            status=PairStatus.NOT_CHECKED,
            reason="no pairwise face comparison capability configured",
            media_id_a=people["a"].selected["media_id"],  # type: ignore[index]
            media_id_b=people["b"].selected["media_id"],  # type: ignore[index]
            **identity,
        )

    candidates = {s: people[s].usable_images[: max(1, policy.max_images_per_person)] for s in "ab"}
    position = {"a": 0, "b": 0}
    attempts: list[dict[str, Any]] = []

    while True:
        records = {s: candidates[s][position[s]] for s in "ab"}
        media = {"media_id_a": records["a"]["media_id"], "media_id_b": records["b"]["media_id"]}
        used_fallback = position["a"] > 0 or position["b"] > 0

        def result(status: PairStatus, **fields: Any) -> PairFaceEvidence:
            return PairFaceEvidence(
                status=status, fallback_used=used_fallback, attempts=tuple(attempts), **media, **identity, **fields
            )

        known = known_rejection(media["media_id_a"], media["media_id_b"]) if known_rejection is not None else None
        images: dict[str, bytes] = {}
        for side in "ab" if known is None else "":
            try:
                images[side] = loader.load(records[side])
            except ImageLoadError as exc:
                attempts.append({**media, "outcome": "error", "error_kind": f"image_load:{exc.status}"})
                return result(
                    PairStatus.ERROR, reason=exc.detail, failed_side=side, error_kind=f"image_load:{exc.status}"
                )

        try:
            if known is not None:
                raise known
            outcome = comparer.compare(images["a"], images["b"])
        except FrPairRejected as rejected:
            failed = {r.side: r for r in rejected.rejections if r.side in ("a", "b")}
            attempts.append({**media, "outcome": "rejected", "rejections": {s: r.kind for s, r in sorted(failed.items())}})
            if not failed:
                return result(PairStatus.ERROR, reason="rejection without a side", error_kind="contract_error")
            stuck = [
                s for s, r in failed.items()
                if r.kind not in policy.fallback_on or position[s] + 1 >= len(candidates[s])
            ]
            if stuck:
                first = failed["a"] if "a" in failed else failed["b"]
                return result(
                    _REJECTION_STATUS[first.kind],
                    reason=first.detail or None,
                    failed_side="both" if len(failed) == 2 else first.side,
                    side_failures={s: r.kind for s, r in sorted(failed.items())},
                )
            for side in failed:  # only the failing side(s) move to their next image
                position[side] += 1
            continue
        except FrOperationalError as exc:
            attempts.append({**media, "outcome": "error", "error_kind": exc.kind})
            return result(PairStatus.ERROR, reason=exc.detail or None, error_kind=exc.kind)

        attempts.append({**media, "outcome": "compared"})
        if outcome.decision is None:
            status = PairStatus.COMPARED
        else:
            status = PairStatus.MATCHED if outcome.decision else PairStatus.NOT_MATCHED
        return result(
            status,
            face_similarity=outcome.similarity,
            similarity_metric=outcome.metric,
            decision_threshold=outcome.threshold,
            model=outcome.model,
        )
