"""Face evidence for candidate pairs: reuse the Phase 3/4 integration, add caching, shared-image handling and limits.

* Pairs where both persons have usable images go to ``compare_pair`` (FR-DOPAMS ``POST /compare``, with the
  deterministic fallback). Everything else is skipped with its real reason; no similarity is ever invented.
* A file attached to *both* persons is not independent evidence (it would compare an image with itself). Such
  images are excluded from the comparison for that pair and the pair is flagged.
* Results are appended to a cache, so reruns - and rescoring with different weights - do not call the service again.
  Operational errors are never reused. If the service keeps failing, the stage stops and the run is marked incomplete.
* Raw cosine similarity is kept as returned, in [-1, 1]. ``face_field_score`` is a documented heuristic transformation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from ..face_recognition.errors import FrPairRejected, SideRejection
from ..face_recognition.image_source import ImageIndex, ManifestImageLoader, PersonImage
from ..face_recognition.pair_evidence import (
    FallbackPolicy, PairComparer, PairFaceEvidence, PairStatus, compare_pair,
)
from .config import ScoringConfig

log = logging.getLogger(__name__)

# Operational problems that a later run may fix. They keep a run "incomplete" and are never taken from the cache.
RETRYABLE_ERROR_KINDS = frozenset(
    {"unavailable", "server_error", "auth_error", "invalid_response", "contract_error", "service_not_ready",
     "model_mismatch", "image_load:fetch_error", "operational"}
)
SERVICE_DOWN_KINDS = frozenset({"unavailable", "server_error", "auth_error", "invalid_response", "contract_error", "service_not_ready"})
SCORED_STATUSES = frozenset({PairStatus.COMPARED, PairStatus.MATCHED, PairStatus.NOT_MATCHED})
FACE_STATUS_REASONS = {
    PairStatus.SKIPPED_NO_IMAGE: "no usable image on at least one record",
    PairStatus.SKIPPED_SHARED_IMAGE: "the only usable image(s) are the same file on both records",
    PairStatus.NOT_CHECKED: "face comparison was not run",
    PairStatus.NO_FACE: "no detectable face in the image(s) tried",
    PairStatus.MULTIPLE_FACES: "several faces of similar size in the image(s) tried",
    PairStatus.LOW_QUALITY: "face too small or too blurred in the image(s) tried",
    PairStatus.INVALID_IMAGE: "the service could not use the image file(s)",
    PairStatus.ERROR: "face comparison failed for an operational reason",
}


class FaceStageAborted(RuntimeError):
    """The face service kept failing; completed results are saved and the run is incomplete."""


def face_field_score(cosine: float, config: ScoringConfig) -> float:
    """Heuristic map of raw cosine to [0, 1] for the weighted score: 0 at/below the 'different people' anchor, 1 at/above
    the 'same person' anchor, linear between. NOT a probability and not calibrated; see the Phase 6 report."""
    low, high = config.face_different_anchor, config.face_same_anchor
    return max(0.0, min(1.0, (cosine - low) / (high - low)))


def is_retryable(evidence: PairFaceEvidence) -> bool:
    return evidence.status is PairStatus.ERROR and (evidence.error_kind or "operational") in RETRYABLE_ERROR_KINDS


def needs_retry(evidence: PairFaceEvidence) -> bool:
    """Required face processing is still outstanding: a retryable error, or no comparison capability was configured."""
    return is_retryable(evidence) or evidence.status is PairStatus.NOT_CHECKED


# ------------------------------------------------------------------- image outcome memory

IMAGE_FAILURE_KINDS = frozenset({"no_face", "multiple_faces", "low_quality", "invalid_image"})


class ImageOutcomeMemory:
    """What the face service already said about individual images, recovered from stored comparisons.

    A service answer about an image is a property of its bytes and the model, so a stored ``no_face`` is reliable.
    The memory is read once, at the start of a run, and never updated during it (so ranking is a pure function of
    the stored results and reruns are deterministic). Operational errors are never remembered.
    """

    def __init__(self, outcomes: Optional[dict[str, str]] = None):
        self._outcomes = outcomes or {}
        self.no_face_text: Optional[str] = None  # the service's wording of a no_face rejection, when stored results show exactly one

    def __len__(self) -> int:
        return len(self._outcomes)

    def known(self, media_id: str) -> Optional[str]:
        return self._outcomes.get(media_id)

    def failed(self, media_id: str) -> bool:
        return self._outcomes.get(media_id) in IMAGE_FAILURE_KINDS

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for outcome in self._outcomes.values():
            tally[outcome] = tally.get(outcome, 0) + 1
        return dict(sorted(tally.items()))

    @classmethod
    def from_cache_file(cls, path: Path, model: str, engine_fingerprint: Optional[str] = None) -> "ImageOutcomeMemory":
        """``engine_fingerprint`` is ``None`` for the HTTP API (its stored results carry none); an in-process engine passes its own, and
        only results made by that exact engine configuration are remembered."""
        outcomes: dict[str, str] = {}
        no_face_texts: set[str] = set()
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if rec["signature"].get("model") != model or rec["signature"].get("fingerprint") != engine_fingerprint:
                            continue
                        attempts = rec["evidence"].get("attempts") or []
                        if rec["evidence"].get("status") == "no_face":
                            no_face_texts.add(rec["evidence"].get("reason"))  # the service's own text for a no_face rejection
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
                    for att in attempts:
                        ma, mb = att.get("media_id_a"), att.get("media_id_b")
                        if att.get("outcome") == "compared" and ma and mb:
                            outcomes[ma] = outcomes[mb] = "ok"  # a comparison needs a usable face on both sides
                        elif att.get("outcome") == "rejected" and ma and mb:
                            rejections = att.get("rejections") or {}  # the service reports every unusable side
                            for side, media in (("a", ma), ("b", mb)):
                                outcomes[media] = rejections.get(side, "ok")
        memory = cls(outcomes)
        # Only usable if the stored evidence shows ONE text for it: then it is a constant of the service, not per-image detail.
        memory.no_face_text = next(iter(no_face_texts)) if len(no_face_texts) == 1 and isinstance(next(iter(no_face_texts)), str) else None
        return memory


class KnownRejections:
    """Opt-in reuse of stored face-service answers for a whole attempt (Phase 9; off unless the backend asks for it).

    A comparison needs a usable face on both sides, and the service reports *every* unusable side of a rejected comparison, so once
    stored results say what the service answered for each of the two images, a rejection of this exact pair of images is already
    determined. Reproducing it is exact only when the rejection record can be rebuilt completely: both images must have a known outcome,
    at least one must be a failure, and the first failing side (the one whose text becomes the evidence ``reason``) must be a ``no_face``,
    whose wording is a constant of the service learned from the stored evidence (other kinds carry per-image detail such as pixel sizes
    and face counts that is not stored). Anything else returns ``None`` and the attempt is made as before.

    The memory is frozen at the start of the run, so what is reused is a pure function of the stored results (deterministic reruns).
    Like the stored pair results themselves, this assumes the image files have not changed since they were first judged."""

    def __init__(self, memory: ImageOutcomeMemory):
        self._memory = memory
        self._lock = threading.Lock()
        self.reused_attempts = 0

    @property
    def usable(self) -> bool:
        return self._memory.no_face_text is not None

    def __call__(self, media_a: str, media_b: str) -> Optional[FrPairRejected]:
        text = self._memory.no_face_text
        known = {"a": self._memory.known(media_a), "b": self._memory.known(media_b)}
        if text is None or known["a"] is None or known["b"] is None:
            return None
        failed = {side: kind for side, kind in known.items() if kind in IMAGE_FAILURE_KINDS}
        if not failed or failed["a" if "a" in failed else "b"] != "no_face":
            return None
        first = "a" if "a" in failed else "b"
        with self._lock:
            self.reused_attempts += 1
        return FrPairRejected(tuple(SideRejection(side, kind, None, text if side == first else "") for side, kind in sorted(failed.items())))


def measured_detection_rates(memory: ImageOutcomeMemory, index: ImageIndex, person_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Face-detection success of remembered images by Phase 2 attachment category (a measurement, not a classifier)."""
    categories: dict[str, str] = {}
    for pid in person_ids:
        for record in index.get(pid).usable_images:
            categories[record["media_id"]] = record.get("attachment_category") or "unknown"
    stats: dict[str, list[int]] = {}
    for media, category in categories.items():
        outcome = memory.known(media)
        if outcome is not None:
            row = stats.setdefault(category, [0, 0])
            row[0] += 1
            row[1] += outcome == "ok"
    return {c: {"images_with_known_outcome": n, "face_usable": ok, "rate": round(ok / n, 3)} for c, (n, ok) in sorted(stats.items())}


# ------------------------------------------------------------------------ shared images

class PairIndexView:
    """An ImageIndex view for one pair: files attached to both persons removed (see module docstring) and, under the
    ``outcome_memory`` policy, images the service already refused moved behind the others.

    Ranking is stable: ``(known failure, Phase 2 person_rank, media_id)``. Phase 2's order already puts the
    ``person/MEDIA`` category first, which measured far better than ``person/IDENTITY_DETAILS``. No other image
    attribute (orientation, resolution, aspect, size) predicted success on held-out data, so none is used."""

    def __init__(self, index: ImageIndex, person_a: str, person_b: str, memory: Optional[ImageOutcomeMemory] = None,
                 ranking: str = "phase2_rank", attempt_limit: int = 3):
        a, b = index.get(person_a), index.get(person_b)
        self.shared_paths = {r["path_ref"] for r in a.usable_images if r.get("path_ref")} & \
                            {r["path_ref"] for r in b.usable_images if r.get("path_ref")}
        self._memory, self._ranking, self._limit = memory, ranking, attempt_limit
        self.selection: dict[str, list[dict[str, Any]]] = {}
        self._views = {person_a: self._filtered(a), person_b: self._filtered(b)}
        self.usable_before = (a.has_usable_image, b.has_usable_image)

    def _filtered(self, person: PersonImage) -> PersonImage:
        kept = [r for r in person.usable_images if r.get("path_ref") not in self.shared_paths]
        if self._ranking == "outcome_memory" and self._memory is not None:
            kept = sorted(kept, key=lambda r: (self._memory.failed(r["media_id"]), r.get("person_rank", 0), r["media_id"]))
        notes = []
        for position, r in enumerate(kept, 1):
            known = self._memory.known(r["media_id"]) if self._memory is not None else None
            if self._ranking == "outcome_memory" and known in IMAGE_FAILURE_KINDS:
                reason = f"moved behind other images: the face service already refused this image ({known})"
            elif r.get("path_ref") in self.shared_paths:
                reason = "same file attached to both persons"
            else:
                reason = f"Phase 2 order (category {r.get('attachment_category')})"
            notes.append({"media_id": r["media_id"], "position": position, "category": r.get("attachment_category"),
                          "within_attempt_limit": position <= self._limit, "reason": reason})
        self.selection[person.person_id] = notes
        return PersonImage(person.person_id, person.person_status, tuple(kept), person.unusable_reasons)

    def get(self, person_id: str) -> PersonImage:
        return self._views[person_id]


def inputs_key(index: ImageIndex, person_a: str, person_b: str, config: ScoringConfig) -> str:
    """Identifies the images a comparison would use, so stored results are reused only when they would be identical."""
    parts = []
    for pid in (person_a, person_b):
        images = index.get(pid).usable_images[: config.face_max_images_per_person]
        parts.append(",".join(f"{r['media_id']}@{r.get('path_ref')}{r.get('resolved_suffix') or ''}" for r in images))
    return "|".join(parts)


# ---------------------------------------------------------------------------- one pair

def evaluate_pair(
    candidate_id: str, person_a: str, person_b: str, index: ImageIndex, loader: ManifestImageLoader,
    comparer: Optional[PairComparer], config: ScoringConfig, memory: Optional[ImageOutcomeMemory] = None,
    known_rejections: Optional[KnownRejections] = None,
) -> tuple[PairFaceEvidence, dict[str, Any]]:
    """Evidence plus the image-selection record (policy and the ranked images considered for each person)."""
    view = PairIndexView(index, person_a, person_b, memory, config.image_ranking, config.face_max_images_per_person)
    selection = {"policy": config.image_ranking, "attempt_limit": config.face_max_images_per_person,
                 "a": view.selection[person_a], "b": view.selection[person_b]}
    if all(view.usable_before) and not (view.get(person_a).has_usable_image and view.get(person_b).has_usable_image):
        a, b = index.get(person_a), index.get(person_b)
        return PairFaceEvidence(
            person_a, person_b, PairStatus.SKIPPED_SHARED_IMAGE,
            reason="every usable image of at least one person is also attached to the other person",
            media_id_a=a.selected["media_id"], media_id_b=b.selected["media_id"],
            person_status_a=a.person_status, person_status_b=b.person_status,
        ), selection
    evidence = compare_pair(person_a, person_b, view, loader, comparer, FallbackPolicy(max_images_per_person=config.face_max_images_per_person),  # type: ignore[arg-type]
                            known_rejection=known_rejections if known_rejections is not None and known_rejections.usable else None)
    if evidence.status in SCORED_STATUSES and evidence.model and evidence.model != config.face_model:
        return PairFaceEvidence(
            person_a, person_b, PairStatus.ERROR, reason=f"service model {evidence.model!r} is not the configured {config.face_model!r}",
            error_kind="model_mismatch", media_id_a=evidence.media_id_a, media_id_b=evidence.media_id_b, attempts=evidence.attempts,
        ), selection
    return evidence, selection


# ------------------------------------------------------------------------------- cache

def evidence_from_dict(data: dict[str, Any]) -> PairFaceEvidence:
    fields = {k: v for k, v in data.items() if k in PairFaceEvidence.__dataclass_fields__}
    fields["status"] = PairStatus(fields["status"])
    fields["attempts"] = tuple(fields.get("attempts") or ())
    return PairFaceEvidence(**fields)


class FaceCache:
    """Append-only JSONL of face evidence. Later entries win; unreadable or retryable entries are ignored."""

    def __init__(self, path: Path, signature: dict[str, Any]):
        self.path, self.signature = path, signature
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, PairFaceEvidence, Optional[dict[str, Any]]]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if rec.get("signature") != signature:
                            continue
                        evidence = evidence_from_dict(rec["evidence"])
                    except (ValueError, KeyError, TypeError):
                        continue  # half-written line or an older format
                    self._entries[rec["candidate_id"]] = (rec["inputs"], evidence, rec.get("selection"))  # older entries have no selection
        self._handle = open(path, "a", encoding="utf-8", newline="\n")

    def get(self, candidate_id: str, inputs: str) -> Optional[PairFaceEvidence]:
        entry = self._entries.get(candidate_id)
        if entry is None or entry[0] != inputs or needs_retry(entry[1]):
            return None
        return entry[1]

    def selection(self, candidate_id: str) -> Optional[dict[str, Any]]:
        entry = self._entries.get(candidate_id)
        return entry[2] if entry else None

    def put(self, candidate_id: str, inputs: str, evidence: PairFaceEvidence, selection: Optional[dict[str, Any]] = None) -> None:
        if evidence.status is PairStatus.NOT_CHECKED:
            return  # nothing was compared; there is nothing to remember
        record = {"candidate_id": candidate_id, "inputs": inputs, "signature": self.signature, "evidence": evidence.to_dict(),
                  "selection": selection}
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._entries[candidate_id] = (inputs, evidence, selection)

    def close(self) -> None:
        self._handle.close()


# ------------------------------------------------------------------------------- stage

@dataclass
class FaceStageStats:
    attempted: int = 0  # comparisons sent to the service (including fallback retries inside one pair)
    pairs_evaluated: int = 0
    reused_from_cache: int = 0
    service_calls: int = 0
    seconds: float = 0.0
    retryable_errors: int = 0
    aborted: Optional[str] = None
    attempts_reused_from_stored_answers: Optional[int] = None  # only set when the opt-in reuse is on (None keeps the default summary unchanged)
    selections: dict = field(default_factory=dict)  # candidate_id -> stored image-selection record (None for older results)


@dataclass(frozen=True)
class FaceTask:
    candidate_id: str
    person_a: str
    person_b: str
    inputs: str


def run_face_stage(
    tasks: Iterable[FaceTask], index: ImageIndex, loader: ManifestImageLoader, comparer: Optional[PairComparer],
    cache: FaceCache, config: ScoringConfig, call_counter: Optional[Callable[[], int]] = None,
    memory: Optional[ImageOutcomeMemory] = None, known_rejections: Optional[KnownRejections] = None,
) -> tuple[dict[str, PairFaceEvidence], FaceStageStats]:
    """Evaluate every task not already in the cache, in bounded batches with bounded concurrency."""
    results: dict[str, PairFaceEvidence] = {}
    stats = FaceStageStats()
    started = time.perf_counter()
    pending: list[FaceTask] = []
    for task in tasks:
        cached = cache.get(task.candidate_id, task.inputs)
        if cached is not None:
            results[task.candidate_id] = cached
            stats.selections[task.candidate_id] = cache.selection(task.candidate_id)
            stats.reused_from_cache += 1
        else:
            pending.append(task)
    log.info("Face stage: %d to evaluate, %d reused from cache", len(pending), stats.reused_from_cache)

    consecutive_down = 0
    with ThreadPoolExecutor(max_workers=config.face_concurrency) as pool:
        for start in range(0, len(pending), config.face_batch_size):
            batch = pending[start:start + config.face_batch_size]
            futures: dict[Future, FaceTask] = {}
            queue = iter(batch)
            done_in_batch = 0
            # keep at most `face_concurrency` requests in flight, so the limit also holds while a batch drains
            while done_in_batch < len(batch):
                while len(futures) < config.face_concurrency and not stats.aborted:
                    task = next(queue, None)
                    if task is None:
                        break
                    futures[pool.submit(evaluate_pair, task.candidate_id, task.person_a, task.person_b, index, loader, comparer, config, memory, known_rejections)] = task
                if not futures:
                    break
                finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in finished:
                    task = futures.pop(future)
                    done_in_batch += 1
                    selection = None
                    try:
                        evidence, selection = future.result()
                    except Exception as exc:  # a bug in one pair must not discard the others
                        log.exception("Unexpected failure evaluating %s", task.candidate_id)
                        evidence = PairFaceEvidence(task.person_a, task.person_b, PairStatus.ERROR,
                                                    reason=f"unexpected error: {type(exc).__name__}", error_kind="operational")
                    results[task.candidate_id] = evidence
                    stats.selections[task.candidate_id] = selection
                    cache.put(task.candidate_id, task.inputs, evidence, selection)  # retryable results are stored but never reused
                    stats.pairs_evaluated += 1
                    stats.attempted += max(1, sum(1 for a in evidence.attempts if a.get("outcome") in ("compared", "rejected")))
                    if needs_retry(evidence):
                        stats.retryable_errors += 1
                    consecutive_down = consecutive_down + 1 if (evidence.error_kind in SERVICE_DOWN_KINDS) else 0
                    if consecutive_down >= config.face_circuit_breaker and not stats.aborted:
                        stats.aborted = f"{consecutive_down} consecutive face-service failures (last: {evidence.error_kind})"
                if stats.aborted and not futures:
                    break
            if stats.aborted:
                break
        if stats.aborted:
            log.error("Face stage stopped: %s", stats.aborted)
    stats.seconds = round(time.perf_counter() - started, 2)
    if known_rejections is not None:
        stats.attempts_reused_from_stored_answers = known_rejections.reused_attempts
    if call_counter:
        stats.service_calls = call_counter()
    return results, stats
