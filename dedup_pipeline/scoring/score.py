"""Combine field results into an explainable match score and the reviewer-facing record.

    match_score       = sum(weight x field_score) / sum(applicable weights)
    evidence_coverage = applicable configured weight / total configured weight

Only applicable evidence enters either number. Skipped or failed face comparison is excluded, with the reason
recorded, and is never scored as zero. The score orders work for human reviewers; it is not a probability of identity
and nothing here decides anything: every record leaves with ``review.status == "pending"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

from ..face_recognition.image_source import PersonImage
from ..face_recognition.pair_evidence import PairFaceEvidence, PairStatus
from .attributes import PersonAttributes
from .config import FIELD_NAMES, RULE_VERSION, SCHEMA_VERSION, ScoringConfig
from .face import FACE_STATUS_REASONS, SCORED_STATUSES, face_field_score, needs_retry
from .fields import SCORERS, FieldResult

SCORE_LABEL = "Match score (0-100). A prioritisation aid for human review; NOT a probability that the records are the same person."
BAND_TEXT = {"A": "highest score band", "B": "high", "C": "moderate", "D": "low", "E": "insufficient evidence"}


def face_field(evidence: Optional[PairFaceEvidence], config: ScoringConfig) -> FieldResult:
    weight = config.effective_weight("face")
    if evidence is None:
        return FieldResult("face", weight, False, None, "No face comparison: not both records have a usable image.",
                           "no_comparison_attempted")
    if evidence.status in SCORED_STATUSES and evidence.face_similarity is not None:
        cosine = evidence.face_similarity
        score = face_field_score(cosine, config)
        in_score = config.face_in_score
        text = (f"Raw cosine similarity {cosine:.3f} (range -1 to 1) mapped to {score:.2f} by a documented heuristic "
                f"({config.face_different_anchor:.2f} -> 0, {config.face_same_anchor:.2f} -> 1)."
                + ("" if in_score else " Shown only; face evidence is switched off in this configuration."))
        conflict = "soft" if cosine < config.face_different_anchor else None
        if conflict:
            text += " Similarity is in the range seen between different people (poor or old photos can also cause this)."
        return FieldResult("face", weight, in_score, score, text, None if in_score else "face_in_score_disabled", conflict,
                           {"raw_cosine": round(cosine, 4)})
    reason = FACE_STATUS_REASONS.get(evidence.status, evidence.status.value)
    detail = f" ({evidence.reason})" if evidence.reason and evidence.status is PairStatus.ERROR else ""
    return FieldResult("face", weight, False, None, f"Face evidence excluded from the score: {reason}{detail}.", evidence.status.value)


@dataclass
class ScoreOutcome:
    match_score: Optional[float]
    identity_score: Optional[float]
    coverage: float
    applicable_fields: int
    insufficient: bool
    band: str
    fields: dict[str, FieldResult]


def combine(fields: dict[str, FieldResult], config: ScoringConfig) -> ScoreOutcome:
    total = sum(config.effective_weight(n) for n in FIELD_NAMES)
    used = [f for f in fields.values() if f.applicable and f.weight > 0 and f.score is not None]
    applicable_weight = sum(f.weight for f in used)
    score = sum(f.weight * f.score for f in used) / applicable_weight if applicable_weight > 0 else None  # type: ignore[operator]
    identity = [f for f in used if f.field != "face"]
    identity_weight = sum(f.weight for f in identity)
    identity_score = sum(f.weight * f.score for f in identity) / identity_weight if identity_weight > 0 else None  # type: ignore[operator]
    coverage = applicable_weight / total if total > 0 else 0.0
    insufficient = score is None or coverage < config.min_coverage or len(used) < config.min_applicable_fields
    band = "E"
    if not insufficient:
        band = next(name for name, floor in config.bands if score >= floor)  # type: ignore[operator]
    return ScoreOutcome(score, identity_score, coverage, len(used), insufficient, band, fields)


def score_pair(a: PersonAttributes, b: PersonAttributes, evidence: Optional[PairFaceEvidence], config: ScoringConfig) -> ScoreOutcome:
    fields = {r.field: r for r in (scorer(a, b, config) for scorer in SCORERS)}
    fields["face"] = face_field(evidence, config)
    return combine({n: fields[n] for n in FIELD_NAMES}, config)


# --------------------------------------------------------------------- reviewer record

def pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 100, 1)


def image_link(base_url: str, origin: str, path_ref: str, suffix: str) -> str:
    target = f"{base_url.rstrip('/')}/{path_ref.lstrip('/')}{suffix or ''}"
    return f"{origin.rstrip('/')}/api/file-proxy?url={quote(target, safe='')}"


def image_entries(person: PersonImage, used: set[str], config: ScoringConfig, base_url: str) -> list[dict[str, Any]]:
    entries = []
    for rank, record in enumerate(person.usable_images[: config.image_links_per_person], 1):
        entries.append({
            "media_id": record["media_id"], "person_rank": record.get("person_rank", rank), "selected": rank == 1,
            "used_in_face_comparison": record["media_id"] in used,
            "file_proxy_link": image_link(base_url, config.file_proxy_origin, record["path_ref"], record.get("resolved_suffix") or ""),
            "detected_type": record.get("detected_type"),
        })
    return entries


def display_status(evidence: PairFaceEvidence) -> str:
    """Status shown to reviewers. The service's matched / not_matched wording is a threshold result, not an identity
    verdict, so both are shown as ``compared`` and the relation to the threshold is a separate field."""
    return "compared" if evidence.status in SCORED_STATUSES else evidence.status.value


def face_block(evidence: Optional[PairFaceEvidence], outcome: ScoreOutcome, config: ScoringConfig) -> dict[str, Any]:
    if evidence is None:
        return {"status": "skipped_no_image", "comparison_attempted": False, "raw_cosine": None, "in_match_score": False,
                "note": "No face comparison because at least one record has no usable image. No similarity is recorded."}
    face = outcome.fields["face"]
    block: dict[str, Any] = {
        "status": display_status(evidence),
        "comparison_attempted": evidence.status not in (PairStatus.SKIPPED_NO_IMAGE, PairStatus.SKIPPED_SHARED_IMAGE, PairStatus.NOT_CHECKED),
        "raw_cosine": evidence.face_similarity, "metric": evidence.similarity_metric, "cosine_range": [-1.0, 1.0],
        "model": evidence.model, "field_score": None if face.score is None or not face.applicable else round(face.score, 4),
        "in_match_score": face.applicable, "exclusion_reason": face.reason_not_applicable if not face.applicable else None,
        "failed_side": evidence.failed_side, "side_failures": evidence.side_failures, "error_kind": evidence.error_kind,
        "retry_needed": needs_retry(evidence), "reason": evidence.reason,
        "fallback_used": evidence.fallback_used, "attempts": list(evidence.attempts),
        "media_id_a": evidence.media_id_a, "media_id_b": evidence.media_id_b,
    }
    if evidence.status in (PairStatus.MATCHED, PairStatus.NOT_MATCHED):
        block["service_threshold"] = evidence.decision_threshold
        block["relative_to_service_threshold"] = "above" if evidence.status is PairStatus.MATCHED else "below"
        block["service_threshold_note"] = "FR-DOPAMS 1:1 verification threshold (calibrated on 5 people). Not an identity decision."
    return block


REVIEW_GROUPS = {
    0: ("0_all_by_score", "All candidates, highest match score first"),
    1: ("1_no_hard_conflict", "No hard conflict - highest match score first"),
    2: ("2_hard_conflict", "Hard conflict present - read the conflicting evidence first; highest match score first"),
    3: ("3_insufficient_evidence", "Insufficient evidence - a separate review group"),
    4: ("4_technical_failure", "Unresolved technical failure - retry or investigate before relying on this record"),
}


def review_group_number(config: ScoringConfig, insufficient: bool, hard_conflict: bool, technical_failure: bool) -> int:
    """Where a candidate sits in the review queue. A group is an ordering aid only; it decides nothing."""
    if config.review_order == "score":
        return 0
    if technical_failure:
        return 4
    if insufficient:
        return 3
    return 2 if hard_conflict else 1


def dob_comparison_status(field: FieldResult) -> Optional[str]:
    detail = field.detail
    if not detail:
        return None
    if detail.get("method") == "dob_exact":
        return detail["status"]
    a, b = detail["a"]["dob_status"], detail["b"]["dob_status"]
    if (detail.get("method") or "").startswith("dob_vs_recorded_age"):
        return "not_compared_dob_on_one_record_only (its implied age was compared with the other record's recorded age)"
    return f"not_compared_no_valid_dob_on_either_record (A: {a}, B: {b})"


def age_summary(field: FieldResult, include_values: bool) -> Optional[dict[str, Any]]:
    """The age/DOB evidence a reviewer needs in one place (score-v2 only)."""
    detail = field.detail
    if not detail:
        return None
    summary: dict[str, Any] = {
        "rule_version": detail["rule_version"], "reference_date": detail["reference_date"], "method": detail.get("method"),
        "comparison_status": detail.get("status"), "dob_comparison_status": dob_comparison_status(field),
        "age_tolerance_years": detail["age_tolerance_years"], "gap_years": detail.get("gap_years"),
        "estimate_cap": detail.get("estimate_cap"), "applicable": field.applicable,
        "score": None if field.score is None else round(field.score, 4),
        "record_a": detail["a"], "record_b": detail["b"], "inconsistencies": detail["data_quality"],
    }
    if detail["rule_version"] == "score-v3":  # FIR-anchored evidence: how each side's age was dated, and what needs a human
        summary["review_flags"] = detail.get("review_flags", [])
        summary["evidence"] = {k: v for k, v in detail.items() if k in (
            "birth_year_range_distance_years", "birth_year_range_shared_years", "birth_year_centre_gap_years", "fir_years_a", "fir_years_b",
            "same_fir_year", "per_observation", "birth_year_tolerance_years", "birth_year_conflict_gap_years", "reference_date_use")}
    if include_values and field.values:
        for side in ("a", "b"):
            v = field.values[side]
            summary[f"recorded_age_{side}"], summary[f"dob_{side}"] = v["recorded_age"], v["dob"]
            summary[f"age_implied_by_dob_{side}"], summary[f"birth_year_range_{side}"] = v["age_implied_by_dob"], v["birth_year_range"]
    return summary


def selection_block(selection: Optional[dict[str, Any]], evidence: Optional[PairFaceEvidence], images_a: PersonImage,
                    images_b: PersonImage, config: ScoringConfig) -> dict[str, Any]:
    """Why these images were tried, in what order, and what the service said about each (image-selection rationale)."""
    if evidence is None or evidence.status in (PairStatus.SKIPPED_NO_IMAGE, PairStatus.NOT_CHECKED):
        return {"applicable": False, "note": "No image selection: no face comparison was possible or attempted."}
    results: dict[str, str] = {}
    for att in evidence.attempts:
        ma, mb = att.get("media_id_a"), att.get("media_id_b")
        if att.get("outcome") == "compared":
            results[ma] = results[mb] = "face usable"
        elif att.get("outcome") == "rejected":
            rejections = att.get("rejections") or {}
            results[ma], results[mb] = rejections.get("a", "face usable"), rejections.get("b", "face usable")
        elif att.get("outcome") == "error":
            results.setdefault(ma, "service error: not an image result")
            results.setdefault(mb, "service error: not an image result")
    stored = selection is not None
    limit = config.face_max_images_per_person

    def side(key: str, person: PersonImage) -> list[dict[str, Any]]:
        if stored:
            notes = [dict(n) for n in selection[key]]
        else:
            notes = [{"media_id": r["media_id"], "position": i, "category": r.get("attachment_category"), "within_attempt_limit": i <= limit,
                      "reason": f"Phase 2 order (category {r.get('attachment_category')}); compared before image-outcome ranking existed"}
                     for i, r in enumerate(person.usable_images, 1)]
        for n in notes:
            n["attempted"] = n["media_id"] in results
            n["service_result"] = results.get(n["media_id"])
        return notes

    return {"applicable": True, "policy": selection["policy"] if stored else "phase2_rank", "recorded_with_comparison": stored,
            "attempt_limit_per_person": limit, "a": side("a", images_a), "b": side("b", images_b),
            "final_status": display_status(evidence), "fallback_used": evidence.fallback_used,
            "note": ("Category person/MEDIA is ranked first because it measured far better at yielding a usable face; no reliable "
                     "portrait-vs-document classifier is available, so no image is excluded on a guess.")}


def build_record(
    candidate: dict[str, Any], a: PersonAttributes, b: PersonAttributes, outcome: ScoreOutcome,
    evidence: Optional[PairFaceEvidence], images_a: PersonImage, images_b: PersonImage, shared_paths: set[str],
    config: ScoringConfig, base_url: str, image_limitations: list[str], selection: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    include = config.include_values
    matching = [{"field": n, "score": round(f.score, 4), "explanation": f.explanation}
                for n, f in outcome.fields.items() if f.applicable and f.score is not None and f.score >= config.match_threshold]
    conflicts = [{"field": n, "severity": f.conflict, "explanation": f.explanation}
                 for n, f in outcome.fields.items() if f.conflict]
    quality = outcome.fields["age_dob"].detail.get("data_quality", [])  # a record whose own age and DOB disagree
    conflicts += [{"field": "age_dob", "severity": "data_quality",
                   "explanation": (f"Record {q['record'].upper()}: recorded age {q['recorded_age']} contradicts the age its own DOB implies at its FIR "
                                   f"date {q['fir_date']} ({q['age_implied_by_dob']}). The DOB was used; check the record.") if "fir_date" in q else
                                  (f"Record {q['record'].upper()}: recorded age {q['recorded_age']} contradicts the age its own DOB implies "
                                   f"({q['age_implied_by_dob']}). The DOB was used; check the record.")} for q in quality]
    review_flags = outcome.fields["age_dob"].detail.get("review_flags", [])  # score-v3: contradicting / adjusted age observations
    conflicts += [{"field": "age_dob", "severity": "data_quality", "explanation": f"Record {f['record'].upper()}: {f['explanation']}"} for f in review_flags]
    missing = [{"field": n, "reason": f.reason_not_applicable, "explanation": f.explanation}
               for n, f in outcome.fields.items() if not f.applicable]
    used = {m for m in (evidence.media_id_a, evidence.media_id_b) if m} if evidence else set()
    if evidence:
        used |= {a_[k] for a_ in evidence.attempts for k in ("media_id_a", "media_id_b") if a_.get(k)}
    indicators = []
    if evidence is not None and needs_retry(evidence):
        indicators.append("face_comparison_retry_needed")
    if shared_paths:
        indicators.append("same_image_file_attached_to_both_records")
    indicators += image_limitations
    if outcome.insufficient:
        indicators.append("insufficient_evidence")
    hard_conflict = any(c["severity"] == "hard" for c in conflicts)
    if hard_conflict:
        indicators.append("hard_conflict_present")
    indicators += [f"age_dob_inconsistent_record_{q['record']}" for q in quality]
    indicators += [f"{f['issue']}_record_{f['record']}" for f in review_flags]
    technical_failure = (evidence is not None and needs_retry(evidence)) or any(i.startswith("image_resolution_") for i in image_limitations)
    group = REVIEW_GROUPS[review_group_number(config, outcome.insufficient, hard_conflict, technical_failure)]
    band_note = f"{BAND_TEXT[outcome.band]} - ordering aid only, not a decision"
    if hard_conflict:
        band_note += "; has a hard conflict - read the conflicting attributes first (a weighted average can hide one strong contradiction)"
    return {
        "schema_version": SCHEMA_VERSION, "rule_version": config.rule_version,
        "candidate_id": candidate["candidate_id"],
        "person_a_id": candidate["person_a_id"], "person_b_id": candidate["person_b_id"],
        "source_a": candidate["source_a"], "source_b": candidate["source_b"], "source_pair": candidate["source_pair"],
        "source_record_a": candidate["source_record_a"], "source_record_b": candidate["source_record_b"],
        "match_score_label": SCORE_LABEL, "match_score_percent": pct(outcome.match_score),
        "identity_fields_score_percent": pct(outcome.identity_score), "evidence_coverage_percent": pct(outcome.coverage),
        "applicable_fields": outcome.applicable_fields, "score_band": outcome.band,
        "score_band_note": band_note, "has_hard_conflict": hard_conflict,
        "insufficient_evidence": outcome.insufficient,
        "review_group": group[0], "review_group_label": group[1],
        "fields": {n: f.to_dict(include) for n, f in outcome.fields.items()},
        "age_dob_summary": age_summary(outcome.fields["age_dob"], include),
        "matching_attributes": matching, "conflicting_attributes": conflicts,
        "hard_conflicts": [c for c in conflicts if c["severity"] == "hard"], "missing_evidence": missing,
        "face": face_block(evidence, outcome, config),
        "image_selection": selection_block(selection, evidence, images_a, images_b, config),
        "images": {
            "a": image_entries(images_a, used, config, base_url), "b": image_entries(images_b, used, config, base_url),
            "shared_image_file": bool(shared_paths),
            "access_note": "Open links in the authorised DOPAMS session (/api/file-proxy requires login).",
        },
        "candidate_generation": {"rule_names": candidate["rule_names"], "rules": candidate["rules"],
                                 "note": "Name-based rules guarantee a similar name by construction; weigh the other fields."},
        "processing": {"indicators": indicators, "complete": not (evidence is not None and needs_retry(evidence))},
        "review": {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None},
    }
