"""score-final criterion rules. Historical score-v1/v2/v3 scorers do not call this module.

Face similarity from the validated engine is cosine in [-1, 1]. The 85% threshold is applied to the existing
piecewise map (0.20 -> 0, 0.58 -> 1), not to the raw cosine. A mapped score must be strictly greater than 0.85
to keep that score. A comparison that was actually run and falls at or below 0.85 scores 0 and keeps its weight.
A comparison that was not run is unavailable and drops out of the denominator.
"""

from __future__ import annotations

from typing import Optional

from ..face_recognition.pair_evidence import PairFaceEvidence, PairStatus
from ..identity.names import NormalizedName, compare_identity_names, normalize_identity_name
from .attributes import Address, PersonAttributes
from .config import ADDRESS_COMPONENT_WEIGHTS, ScoringConfig
from .face import FACE_STATUS_REASONS, SCORED_STATUSES, face_field_score
from .fields import FieldResult

_NOT_IN_FINAL = "not_in_score_final"
_STATE_ALIASES = {
    "ts": "telangana", "tg": "telangana", "telangana": "telangana",
    "ap": "andhra pradesh", "andhra": "andhra pradesh", "andhra pradesh": "andhra pradesh",
}
_COUNTRY_ALIASES = {"india": "india", "bharat": "india", "in": "india"}
_SPELLING = 0.92
_SPELLING_SCORE = 0.85


def excluded_signal(name: str, config: ScoringConfig) -> FieldResult:
    return FieldResult(
        name, config.effective_weight(name), False, None,
        "This signal is not used by score-final.", _NOT_IN_FINAL,
    )


def _stored_name(person: PersonAttributes, kind: str) -> NormalizedName:
    stored = person.name_normalization if kind == "name" else person.relative_normalization
    if isinstance(stored, NormalizedName):
        return stored
    raw = person.name if kind == "name" else person.relative_name
    return normalize_identity_name(raw)


def _name_result(field: str, left: PersonAttributes, right: PersonAttributes, config: ScoringConfig) -> FieldResult:
    weight = config.effective_weight(field)
    a = _stored_name(left, field)
    b = _stored_name(right, field)
    compared = compare_identity_names(a, b)
    detail = {
        "status": compared.status,
        "method": compared.method,
        "rule_version": a.rule_version,
        "used_initial": compared.used_initial,
        "normalization_a": a.to_dict(),
        "normalization_b": b.to_dict(),
    }
    values = {"a": a.display or a.original, "b": b.display or b.original}
    if not compared.available or compared.score is None:
        return FieldResult(field, weight, False, None, compared.reason, "name_missing", None, values, detail)
    conflict = None
    if compared.score < config.name_conflict_below and field == "name" and not compared.used_initial:
        conflict = "hard"
    elif field == "relative_name" and compared.score < config.relative_conflict_below and not compared.used_initial:
        conflict = "hard"
    return FieldResult(field, weight, True, compared.score, compared.reason, None, conflict, values, detail)


def score_name_final(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    return _name_result("name", a, b, config)


def score_relative_final(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    return _name_result("relative_name", a, b, config)


def _canon(value: str, aliases: dict[str, str]) -> str:
    key = " ".join(value.lower().split())
    return aliases.get(key, key)


def _text_score(left: str, right: str, aliases: Optional[dict[str, str]] = None) -> Optional[float]:
    if not left or not right:
        return None
    a = _canon(left, aliases or {})
    b = _canon(right, aliases or {})
    if a == b:
        return 1.0
    from difflib import SequenceMatcher
    if SequenceMatcher(None, a, b).ratio() >= _SPELLING:
        return _SPELLING_SCORE
    return 0.0


def _pin_score(left: str, right: str) -> Optional[float]:
    if not left or not right:
        return None
    return 1.0 if left == right else 0.0


def _components(left: Address, right: Address) -> tuple[float, float, dict[str, Optional[float]]]:
    """Return ``(weighted points, available weight, per-component scores)``. A missing component is None."""
    scores = {
        "village": _text_score(left.village, right.village),
        "mandal": _text_score(left.mandal, right.mandal),
        "district": _text_score(left.district, right.district),
        "state": _text_score(left.state, right.state, _STATE_ALIASES),
        "pin": _pin_score(left.pin, right.pin),
        "country": _text_score(left.country, right.country, _COUNTRY_ALIASES),
    }
    points = 0.0
    available = 0.0
    for name, score in scores.items():
        if score is None:
            continue
        weight = float(ADDRESS_COMPONENT_WEIGHTS[name])
        available += weight
        points += weight * score
    return points, available, scores


def score_address_final(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    weight = config.effective_weight("address")
    if not a.addresses or not b.addresses:
        return FieldResult("address", weight, False, None, "No structured address on at least one record.", "no_comparable_address")
    same = [(x, y) for x in a.addresses for y in b.addresses if x.kind and x.kind == y.kind]
    cross = [(x, y) for x in a.addresses for y in b.addresses if not (x.kind and x.kind == y.kind)]
    pool = same or cross
    ranked = []
    for left, right in pool:
        points, available, scores = _components(left, right)
        if available <= 0:
            continue
        ranked.append((available, 1 if left.kind == "present" else 0, points / available, scores, left.kind or "unspecified"))
    if not ranked:
        return FieldResult("address", weight, False, None,
                           "Address text is present but none of village, mandal, district, state, PIN or country can be compared.",
                           "no_comparable_address_component")
    # The kind with the most comparable weight wins. Score is not used to pick the kind, so a thin matching
    # address cannot hide a richer conflicting one. Present wins a tie.
    available, _, score, scores, kind = max(ranked, key=lambda item: (item[0], item[1]))
    conflict = "soft" if any(value == 0.0 for value in scores.values() if value is not None) else None
    compared = [name for name, value in scores.items() if value is not None]
    unavailable = [name for name, value in scores.items() if value is None]
    # Renormalising a district-only match to 1 still uses only that component. It is not complete address evidence.
    complete = len(unavailable) == 0
    across = not same
    if conflict:
        status = "conflicting"
    elif complete:
        status = "matching"
    else:
        status = "partial"
    text = f"Address score {score:.2f} from {kind} components" + (" (present compared with permanent)." if across else ".")
    if not complete:
        text += (
            f" Compared only {', '.join(compared)} ({available:.0%} of the address-component weight). "
            "This is not complete address evidence. Absent components are excluded, not treated as matches."
        )
    if conflict:
        text += " Differing components score 0 and stay in the address denominator. This does not reject identity."
    detail = {
        "status": status,
        "component_scores": scores,
        "component_weights": dict(ADDRESS_COMPONENT_WEIGHTS),
        "components_compared": compared,
        "components_unavailable": unavailable,
        "available_component_weight": round(available, 4),
        "address_evidence_complete": complete,
        "address_kind": kind,
        "across_address_types": across,
        "ignored": ["house_number", "street", "locality", "path"],
    }
    values = {
        "a_present_text": a.present_address_text, "b_present_text": b.present_address_text,
        "a_permanent_text": a.permanent_address_text, "b_permanent_text": b.permanent_address_text,
    }
    return FieldResult("address", weight, True, round(score, 4), text, None, conflict, values, detail)


def face_field_final(evidence: Optional[PairFaceEvidence], config: ScoringConfig) -> FieldResult:
    """85% applies to the mapped 0-1 score. Unavailable comparisons contribute no score and no weight."""
    weight = config.effective_weight("face")
    if evidence is None or evidence.status in (PairStatus.SKIPPED_NO_IMAGE, PairStatus.SKIPPED_SHARED_IMAGE, PairStatus.NOT_CHECKED):
        reason = "no_face_evidence" if evidence is None else evidence.status.value
        text = "Face comparison was not performed, so the face weight is excluded."
        if evidence is not None and evidence.reason:
            text = evidence.reason + " The face weight is excluded."
        return FieldResult("face", weight, False, None, text, reason, None, {}, {"status": "unavailable"})
    if evidence.status is PairStatus.ERROR or evidence.status not in SCORED_STATUSES:
        # no_face, low_quality, invalid_image, multiple_faces, and operational errors are not a mismatch.
        why = FACE_STATUS_REASONS.get(evidence.status, evidence.status.value)
        kind = "technical_error" if evidence.status is PairStatus.ERROR else "unavailable"
        return FieldResult(
            "face", weight, False, None,
            f"Face evidence excluded ({why}). This is not a similarity of zero.",
            evidence.status.value, None, {},
            {"status": kind, "error_kind": evidence.error_kind},
        )
    cosine = evidence.face_similarity
    if cosine is None or isinstance(cosine, bool) or not isinstance(cosine, (int, float)) or cosine != cosine or not -1.0 <= float(cosine) <= 1.0:
        return FieldResult(
            "face", weight, False, None,
            "The face model returned no usable similarity in [-1, 1]. The face weight is excluded until a valid comparison exists.",
            "invalid_face_score", None, {}, {"status": "technical_error", "raw_cosine": cosine},
        )
    mapped = round(face_field_score(float(cosine), config), 4)
    passed = mapped > config.face_pass_threshold
    score = mapped if passed else 0.0
    outcome = "passed" if passed else "failed"
    text = (
        f"Raw cosine {float(cosine):.3f} maps to {mapped:.4f} on the validated 0-1 scale "
        f"(0.20 -> 0, 0.58 -> 1). The face criterion passes only when that score is strictly above "
        f"{config.face_pass_threshold:.2f}. Threshold {outcome}; criterion score {score:.4f}."
    )
    conflict = None if passed else "soft"
    return FieldResult(
        "face", weight, True, score, text, None, conflict,
        {},
        {
            "status": "matching" if passed else "conflicting",
            "raw_cosine": round(float(cosine), 4),
            "mapped_score": mapped,
            "pass_threshold": config.face_pass_threshold,
            "threshold_outcome": outcome,
            "scale": "mapped_cosine_0_1",
            "metric": evidence.similarity_metric,
            "model": evidence.model,
        },
    )
