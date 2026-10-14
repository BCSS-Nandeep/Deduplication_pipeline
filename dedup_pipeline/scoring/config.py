"""Scoring configuration: versioned, serialisable, validated. Nothing here is hard-wired into the scorers.

Weights are *relative importance* of each kind of evidence; they are heuristics chosen by inspecting the data
(see the Phase 6 report), not fitted to labels. The resulting score is a prioritisation aid for human reviewers.
It is not a probability and has not been calibrated against a labelled dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from typing import Any, Mapping, Optional

RULE_VERSION_V1 = "score-v1"  # kept selectable and unchanged: age = integer difference of recorded ages, Phase 2 image order, score ordering
RULE_VERSION_V2 = "score-v2"  # reference-date age logic, DOB policy, image-outcome memory, conflict-aware review order
RULE_VERSION_V3 = "score-v3"  # FIR-date-anchored age evidence: birth-year ranges from (recorded age, FIR date), observations with provenance
RULE_VERSION = RULE_VERSION_V2  # the default for new runs (score-v3 is selected explicitly until its results are validated)
RULE_VERSIONS = (RULE_VERSION_V1, RULE_VERSION_V2, RULE_VERSION_V3)
OBSERVATION_CONFLICT_POLICIES = ("unavailable", "hull")
# Fields that only score-v3 reads. They are left out of the serialised config of v1/v2 runs so those configurations
# (and their signatures) stay exactly what they were before score-v3 existed.
V3_ONLY_FIELDS = ("fir_timezone", "fir_date_min_year", "fir_date_placeholder_dates", "birth_year_tolerance_years",
                  "birth_year_conflict_gap_years", "observation_conflict_policy")
SCHEMA_VERSION = "review-v2"
SCHEMA_VERSION_V1 = "review-v1"
IMAGE_RANKINGS = ("phase2_rank", "outcome_memory")
REVIEW_ORDERS = ("score", "conflict_free_first")
LEAP_POLICIES = ("mar1", "feb28")

FIELD_NAMES = ("name", "alias", "relative_name", "phone", "age_dob", "gender", "address", "face")
DECISIONS = ("confirm_same_person", "reject_match", "needs_further_review", "unable_to_verify")

# Equals the lower bound of the cosine range the service reports for the *same* person
# (FR-DOPAMS README "Choosing thresholds": same person 0.58-0.81, different people <= 0.21; measured on 5 people).
FACE_ANCHOR_DIFFERENT = 0.20
FACE_ANCHOR_SAME = 0.58

DEFAULT_WEIGHTS = {
    "name": 0.25, "alias": 0.05, "relative_name": 0.15, "phone": 0.15,
    "age_dob": 0.10, "gender": 0.05, "address": 0.05, "face": 0.20,
}


@dataclass(frozen=True)
class ScoringConfig:
    rule_version: str = RULE_VERSION
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # --- face evidence. cosine -> [0, 1] by a piecewise-linear map: <= different_anchor -> 0, >= same_anchor -> 1.
    face_in_score: bool = True
    face_different_anchor: float = FACE_ANCHOR_DIFFERENT
    face_same_anchor: float = FACE_ANCHOR_SAME
    face_model: str = "buffalo_l"
    face_max_images_per_person: int = 3  # the existing deterministic fallback limit
    face_concurrency: int = 8  # Phase 11: pairs in flight at once. Measured on the CPU engine (200 cases): 4 -> 48 s, 8 -> 35 s, 12 -> 30 s, 16 -> 29 s for +300 MB per step
    face_batch_size: int = 100  # comparisons submitted, completed and checkpointed per batch
    face_circuit_breaker: int = 8  # consecutive service failures before the face stage stops (run -> incomplete)

    # --- name-like fields
    match_threshold: float = 0.85  # a field score at/above this is listed as a matching attribute
    name_conflict_below: float = 0.50
    relative_conflict_below: float = 0.40
    alias_positive_only_threshold: float = 0.85  # one-sided alias counts only when it clearly matches

    # --- age / DOB
    age_min: int = 5
    age_max: int = 100
    age_scores: tuple[float, ...] = (1.0, 0.9, 0.75, 0.55, 0.35, 0.2)  # index = |age difference| in years; beyond -> 0
    age_conflict_over: int = 5  # |difference| greater than this is a hard conflict
    dob_swapped_day_month_score: float = 0.8
    dob_same_year_score: float = 0.4  # score-v1 only

    # --- score-v2 age / DOB: every age is evaluated at ONE reference date for the whole run (never implicit)
    reference_date: Optional[str] = None  # ISO date, required for score-v2; recorded in the config and the run summary
    leap_day_birthday: str = "mar1"  # a 29 Feb birthday is reached on 1 Mar ("mar1") or 28 Feb ("feb28") in non-leap years
    dob_min_year: int = 1900
    dob_placeholder_dates: tuple[str, ...] = ("1900-01-01", "1899-12-31", "1800-01-01", "1970-01-01", "0001-01-01", "1111-11-11", "9999-12-31")
    dob_day_scores: tuple[tuple[int, float], ...] = ((0, 1.0), (1, 0.9), (31, 0.5), (366, 0.25))  # |difference| in days <= N -> score
    dob_conflict_over_days: int = 31  # a DOB difference above this is a hard conflict
    age_tolerance_years: int = 2  # recorded ages may be this many years apart (stale or approximate) without being flagged
    age_gap_factors: tuple[float, ...] = (1.0, 0.85, 0.6, 0.35, 0.15)  # index = whole years between the two age estimates; beyond -> 0
    age_conflict_gap_years: int = 5  # a gap above this is a hard conflict
    age_estimate_max_score: float = 0.7  # age vs age is an estimate: it can never score like an exact DOB match
    dob_vs_age_max_score: float = 0.8  # a DOB against a recorded age
    age_dob_internal_tolerance_years: int = 1  # recorded age vs the age its own DOB implies, inside one record

    # --- score-v3 FIR-date-anchored age evidence (reference_date is then only the validation cut-off, never an age anchor)
    fir_timezone: str = "Asia/Kolkata"  # a FIR timestamp is turned into the calendar date it has in this zone (the database stores UTC)
    fir_date_min_year: int = 1980  # a FIR dated before this is treated as invalid, not as evidence
    fir_date_placeholder_dates: tuple[str, ...] = ("1900-01-01", "1899-12-31", "1800-01-01", "1970-01-01", "0001-01-01", "1111-11-11", "9999-12-31")
    birth_year_tolerance_years: int = 1  # possible birth-year ranges at most this many years apart are accepted as approximate
    birth_year_conflict_gap_years: int = 4  # ranges further apart than this are a hard conflict (4 here == age gap > 5 at one date)
    observation_conflict_policy: str = "unavailable"  # one record's own age observations contradict: "unavailable" (not compared) | "hull" (compare the widest range)

    # --- score-v2 image selection and review order. None -> the default of the rule version.
    image_ranking: Optional[str] = None  # "phase2_rank" (v1) | "outcome_memory" (v2)
    review_order: Optional[str] = None  # "score" (v1) | "conflict_free_first" (v2)

    # --- address (structured state / district / mandal from person_enrichment.address_resolution)
    address_scores: Mapping[str, float] = field(default_factory=lambda: {"mandal": 1.0, "district": 0.7, "state": 0.3, "none": 0.0})

    # --- evidence sufficiency and prioritisation bands (labels for ordering, never decisions)
    min_coverage: float = 0.40
    min_applicable_fields: int = 3
    bands: tuple[tuple[str, float], ...] = (("A", 0.85), ("B", 0.70), ("C", 0.50), ("D", 0.0))

    # --- processing
    batch_size: int = 500
    attribute_cache_max: int = 60_000
    include_values: bool = True  # show compared values (names, ages) to the reviewer; phones masked, addresses never shown
    image_links_per_person: int = 5
    file_server_base_url: str = ""  # empty -> FILE_SERVER_BASE_URL env or the image-resolution default
    file_proxy_origin: str = ""  # e.g. "https://dopams.example"; empty -> relative /api/file-proxy links
    allow_stale_candidates: bool = False

    def __post_init__(self) -> None:
        if self.rule_version not in RULE_VERSIONS:
            raise ValueError(f"rule_version must be one of {RULE_VERSIONS}")
        v2 = self.rule_version != RULE_VERSION_V1  # score-v2 and score-v3 share the reference-date, image-ranking and review-order machinery
        object.__setattr__(self, "image_ranking", self.image_ranking or ("outcome_memory" if v2 else "phase2_rank"))
        object.__setattr__(self, "review_order", self.review_order or ("conflict_free_first" if v2 else "score"))
        if self.image_ranking not in IMAGE_RANKINGS or self.review_order not in REVIEW_ORDERS or self.leap_day_birthday not in LEAP_POLICIES:
            raise ValueError(f"image_ranking in {IMAGE_RANKINGS}, review_order in {REVIEW_ORDERS}, leap_day_birthday in {LEAP_POLICIES}")
        if v2:
            try:
                date.fromisoformat(self.reference_date or "")
            except ValueError:
                raise ValueError(f"{self.rule_version} needs reference_date as an ISO date (YYYY-MM-DD); it is never taken implicitly from the clock") from None
        if self.rule_version == RULE_VERSION_V3:
            if (self.observation_conflict_policy not in OBSERVATION_CONFLICT_POLICIES or not self.fir_timezone
                    or self.birth_year_tolerance_years < 0 or self.birth_year_conflict_gap_years < self.birth_year_tolerance_years):
                raise ValueError(f"score-v3: observation_conflict_policy in {OBSERVATION_CONFLICT_POLICIES}, a fir_timezone, and "
                                 "0 <= birth_year_tolerance_years <= birth_year_conflict_gap_years")
        if self.age_tolerance_years < 0 or self.age_conflict_gap_years < self.age_tolerance_years or not self.age_gap_factors:
            raise ValueError("age tolerance must be >= 0, below the conflict gap, and age_gap_factors must not be empty")
        if any(not 0.0 <= f <= 1.0 for f in self.age_gap_factors) or not (0 < self.age_estimate_max_score <= 1 and 0 < self.dob_vs_age_max_score <= 1):
            raise ValueError("age factors and estimate caps must lie in [0, 1]")
        days = [d for d, _ in self.dob_day_scores]
        if days != sorted(days) or not days or any(not 0.0 <= s <= 1.0 for _, s in self.dob_day_scores):
            raise ValueError("dob_day_scores must be ordered by day limit with scores in [0, 1]")
        unknown = set(self.weights) - set(FIELD_NAMES)
        missing = set(FIELD_NAMES) - set(self.weights)
        if unknown or missing:
            raise ValueError(f"weights must name exactly {FIELD_NAMES} (unknown={sorted(unknown)}, missing={sorted(missing)})")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("weights must be non-negative with a positive total")
        if not (0.0 <= self.face_different_anchor < self.face_same_anchor <= 1.0):
            raise ValueError("face anchors must satisfy 0 <= different < same <= 1")
        if self.face_max_images_per_person < 1 or self.face_concurrency < 1 or self.face_batch_size < 1 or self.batch_size < 1:
            raise ValueError("concurrency, batch sizes and image limits must be positive")
        if not (0.0 <= self.min_coverage <= 1.0) or self.min_applicable_fields < 1:
            raise ValueError("min_coverage must be in [0, 1] and min_applicable_fields at least 1")
        thresholds = [t for _, t in self.bands]
        if thresholds != sorted(thresholds, reverse=True) or thresholds[-1] != 0.0:
            raise ValueError("bands must be ordered from high to low and end at 0.0")

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))

    def effective_weight(self, name: str) -> float:
        return 0.0 if name == "face" and not self.face_in_score else float(self.weights[name])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["weights"] = dict(self.weights)
        data["address_scores"] = dict(self.address_scores)
        data["age_scores"] = list(self.age_scores)
        data["age_gap_factors"] = list(self.age_gap_factors)
        data["dob_placeholder_dates"] = list(self.dob_placeholder_dates)
        data["dob_day_scores"] = [list(p) for p in self.dob_day_scores]
        data["bands"] = [list(b) for b in self.bands]
        data["fir_date_placeholder_dates"] = list(self.fir_date_placeholder_dates)
        if self.rule_version != RULE_VERSION_V3:
            for key in V3_ONLY_FIELDS:
                data.pop(key, None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoringConfig":
        known = {f.name for f in fields(cls)}
        if set(data) - known:
            raise ValueError(f"unknown scoring configuration keys: {sorted(set(data) - known)}")
        values = dict(data)
        if "weights" in values:
            values["weights"] = {**DEFAULT_WEIGHTS, **values["weights"]}
        for key in ("age_scores", "age_gap_factors", "dob_placeholder_dates", "fir_date_placeholder_dates"):
            if key in values:
                values[key] = tuple(values[key])
        if "dob_day_scores" in values:
            values["dob_day_scores"] = tuple((int(d), float(s)) for d, s in values["dob_day_scores"])
        if "bands" in values:
            values["bands"] = tuple((str(n), float(t)) for n, t in values["bands"])
        return cls(**values)

    def signature(self) -> str:
        """Everything that changes a score or its presentation; stored with every run."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def face_signature(self) -> dict[str, Any]:
        """What makes a stored face comparison reusable: model and the fallback limit. Weights and the image-ranking
        policy do not matter: a comparison of two given images is valid evidence whichever order they were tried in,
        so stored results are reused and a new ranking only affects comparisons that are computed afresh."""
        return {"model": self.face_model, "max_images_per_person": self.face_max_images_per_person}
