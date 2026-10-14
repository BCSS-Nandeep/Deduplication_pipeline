"""score-v3 age evidence: a recorded age is a statement made *at a FIR date*, so it is anchored there, not at today.

What the schema supports (verified read-only, see the Phase 8 report): ``persons_unified.age`` is a single value per person;
``accused_unified(person_id, crime_id)`` links a person to the FIR(s) they are accused in; ``crimes_unified.fir_date`` is the
FIR registration timestamp. Recorded ages match the age implied by the DOB *at the FIR date* far better than at today's date.

Method, for one observation (recorded age A, FIR date F):

* possible birth years = every calendar year y with ``age_at(31 Dec y, F) <= A <= age_at(1 Jan y, F)`` (``age.birth_year_range``);
  this is normally two years, because the birthday relative to F is unknown. Age 26 at a 2022 FIR and age 30 at a 2026 FIR
  both allow birth year 1996 and therefore are consistent.
* the *current date is never used*. No FIR date -> no birth-year estimate (the observation is "unavailable", with the reason).

Several observations of one record (several FIR links) are kept individually with their provenance. They are combined only if
they agree: the common birth years (consistent), or the widest range if they are within the tolerance of each other
(within_tolerance); if they contradict, the record is flagged for human review and (policy ``unavailable``) not compared.
Nothing is ever resolved by taking the newest or oldest FIR.

Comparison, in this order (DOB comparisons and age-estimate comparisons are never mixed up):

1. both records have a valid DOB: compare the dates (as in score-v2);
2. a valid DOB on one record and age observations on the other: compare the age the DOB implies *at each FIR date* with the age recorded then;
3. age observations only: compare the two sets of possible birth years. The conflict class uses the *distance between the ranges*
   (0 when they share a year), so birthday uncertainty alone can never create a conflict; the score uses the distance between their centres.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .age import _factor, _gap_status, age_at, birth_year_range, parse_age, parse_dob
from .config import ScoringConfig
from .fields import FieldResult

if TYPE_CHECKING:
    from .attributes import PersonAttributes

_YMD = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
AGE_SOURCE = "persons_unified.age"
LINK_SOURCE = "accused_unified.person_id -> crimes_unified.fir_date"


def parse_fir_date(raw: Any, reference: date, config: ScoringConfig) -> tuple[Optional[date], str]:
    """``(date, "valid")`` or ``(None, "missing" | "invalid:<reason>")``. The reference date is only a cut-off: a FIR cannot be later."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, "missing"
    if isinstance(raw, datetime):
        parsed: Optional[date] = raw.date()
    elif isinstance(raw, date):
        parsed = raw
    elif isinstance(raw, str):
        m = _YMD.match(raw.strip())
        if not m:
            return None, "invalid:unparseable"
        try:
            parsed = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None, "invalid:impossible_date"
    else:
        return None, "invalid:unparseable"
    if parsed.isoformat() in config.fir_date_placeholder_dates:
        return None, "invalid:placeholder"
    if parsed.year < config.fir_date_min_year:
        return None, "invalid:before_minimum_year"
    if parsed > date.fromisoformat(config.reference_date or ""):
        return None, "invalid:after_reference_date"
    return parsed, "valid"


@dataclass(frozen=True)
class AgeObservation:
    """One recorded age with the FIR it is anchored to, and everything needed to audit it."""

    age: Optional[int]
    age_status: str
    crime_id: Optional[str]
    accused_id: Optional[str]
    fir_date: Optional[date]
    fir_date_status: str
    usable: bool
    unavailable_reason: Optional[str]
    year_range: Optional[tuple[int, int]]  # possible birth years at the FIR date; None when unavailable
    implied_age_by_own_dob: Optional[int]  # age the record's own DOB implies at this FIR date

    def to_dict(self, include_values: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "age_source": AGE_SOURCE, "link_source": LINK_SOURCE, "method": "recorded_age_at_fir_date",
            "age_status": self.age_status, "fir_date_status": self.fir_date_status, "fir_year": self.fir_date.year if self.fir_date else None,
            "usable": self.usable, "unavailable_reason": self.unavailable_reason,
        }
        if include_values:
            out.update({"crime_id": self.crime_id, "accused_id": self.accused_id, "fir_date": self.fir_date.isoformat() if self.fir_date else None,
                        "recorded_age": self.age, "birth_year_range": list(self.year_range) if self.year_range else None,
                        "age_implied_by_own_dob_at_fir_date": self.implied_age_by_own_dob})
        return out


@dataclass(frozen=True)
class AgeEvidence:
    dob: Optional[date]
    dob_status: str
    observations: tuple[AgeObservation, ...]
    observation_status: str  # none_usable | single | consistent | within_tolerance | conflicting
    merged_range: Optional[tuple[int, int]]  # None when no usable observation, or they conflict and the policy is "unavailable"
    quality: tuple[dict[str, Any], ...]  # recorded age contradicting the record's own DOB at the FIR date
    flags: tuple[dict[str, Any], ...]  # things a human must look at (contradicting observations, ...)

    @property
    def usable_observations(self) -> tuple[AgeObservation, ...]:
        return tuple(o for o in self.observations if o.usable)

    @property
    def inconsistent(self) -> bool:
        return bool(self.quality)

    @property
    def conflicting(self) -> bool:
        return self.observation_status == "conflicting"

    @property
    def age_status(self) -> str:
        valid = [o for o in self.observations if o.age_status == "valid"]
        if valid:
            return "valid"
        statuses = sorted({o.age_status for o in self.observations})
        return statuses[0] if len(statuses) == 1 else "+".join(statuses)

    def recorded_ages(self) -> list[int]:
        return sorted({o.age for o in self.observations if o.age is not None})

    def unavailable_reasons(self) -> list[str]:
        return sorted({o.unavailable_reason for o in self.observations if o.unavailable_reason})

    def side_state(self) -> str:
        """``dob`` | ``age_at_fir_date`` | the reason there is nothing to compare. Used for the not-applicable explanation."""
        if self.dob is not None:
            return "dob"
        if self.merged_range is not None:
            return "age_at_fir_date"
        if self.conflicting:
            return "age_observations_conflict"
        return "+".join(self.unavailable_reasons()) or "no_age_or_dob"


def _merge(ranges: Sequence[tuple[int, int]], config: ScoringConfig) -> tuple[str, Optional[tuple[int, int]]]:
    if not ranges:
        return "none_usable", None
    if len(ranges) == 1:
        return "single", ranges[0]
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if lo <= hi:
        return "consistent", (lo, hi)
    hull = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    if lo - hi <= config.birth_year_tolerance_years:
        return "within_tolerance", hull
    return "conflicting", hull if config.observation_conflict_policy == "hull" else None


def build_age_evidence(raw_dob: Any, raw_age: Any, fir_rows: Optional[Sequence[tuple[Any, Any, Any]]], config: ScoringConfig) -> AgeEvidence:
    """``fir_rows`` = (accused_id, crime_id, fir_date) for every accused row of the person; empty/None = no FIR link found."""
    reference = date.fromisoformat(config.reference_date or "")
    dob, dob_status = parse_dob(raw_dob, reference, config)
    age, age_status = parse_age(raw_age, config)
    rows = list(fir_rows or [])
    if not rows:
        rows = [(None, None, None)]
    observations, quality = [], []
    for accused_id, crime_id, raw_fir in rows:
        fir, fir_status = parse_fir_date(raw_fir, reference, config)
        reason: Optional[str] = None
        if age is None:
            reason = f"age_{age_status}"
        elif crime_id is None and accused_id is None:
            reason = "no_fir_link"
        elif crime_id is None:
            reason = "accused_row_without_fir"
        elif fir is None:
            reason = f"fir_date_{fir_status.replace('invalid:', 'invalid_')}"
        implied = None
        if reason is None and dob is not None:
            if fir < dob:  # type: ignore[operator]
                reason = "fir_date_before_dob"
            else:
                implied = age_at(dob, fir, config.leap_day_birthday)  # type: ignore[arg-type]
        usable = reason is None
        year_range = birth_year_range(age, fir, config.leap_day_birthday) if usable else None  # type: ignore[arg-type]
        observation = AgeObservation(age, age_status, crime_id, accused_id, fir, fir_status, usable, reason, year_range, implied)
        observations.append(observation)
        if usable and implied is not None and abs(implied - age) > config.age_dob_internal_tolerance_years:  # type: ignore[operator]
            quality.append({"record": "", "issue": "recorded_age_contradicts_own_dob_at_fir_date", "recorded_age": age, "age_implied_by_dob": implied,
                            "fir_date": fir.isoformat(), "crime_id": crime_id})  # type: ignore[union-attr]
    status, merged = _merge([o.year_range for o in observations if o.year_range], config)
    flags: list[dict[str, Any]] = []
    if status == "conflicting":
        flags.append({"record": "", "issue": "conflicting_age_observations",
                      "explanation": "The ages recorded against this person's different FIRs cannot be reconciled (their possible birth years are "
                                     f"{'; '.join(str(list(o.year_range)) for o in observations if o.year_range)}). No FIR was preferred; check the record."})
    elif status == "within_tolerance":
        flags.append({"record": "", "issue": "age_observations_differ_within_tolerance",
                      "explanation": "The ages recorded against this person's different FIRs differ slightly; the widest range of possible birth years was used."})
    if dob is not None and any(o.unavailable_reason == "fir_date_before_dob" for o in observations):
        flags.append({"record": "", "issue": "fir_date_before_own_dob", "explanation": "A FIR is dated before this record's own date of birth; that FIR was not used."})
    return AgeEvidence(dob, dob_status, tuple(observations), status, merged, tuple(quality), tuple(flags))


def evidence_from_attributes(a: "PersonAttributes") -> AgeEvidence:
    if a.age_info is None or not isinstance(a.age_info, AgeEvidence):
        raise ValueError("score-v3 age comparison needs attributes built with FIR links and a reference date")
    return a.age_info


# --------------------------------------------------------------------------------- comparison

def _side_detail(e: AgeEvidence) -> dict[str, Any]:
    return {"dob_status": e.dob_status, "age_status": e.age_status, "inconsistent_age_and_dob": e.inconsistent,
            "observation_status": e.observation_status, "state": e.side_state(),
            "observations": [o.to_dict(include_values=False) for o in e.observations]}


def _side_values(e: AgeEvidence) -> dict[str, Any]:
    ages = e.recorded_ages()
    implied = sorted({o.implied_age_by_own_dob for o in e.observations if o.implied_age_by_own_dob is not None})
    return {"recorded_age": ages[0] if len(ages) == 1 else (ages or None), "dob": e.dob.isoformat() if e.dob else None,
            "age_implied_by_dob": implied[0] if len(implied) == 1 else (implied or None),
            "birth_year_range": list(e.merged_range) if e.merged_range else None,
            "observations": [o.to_dict(include_values=True) for o in e.observations]}


def range_relation(ra: tuple[int, int], rb: tuple[int, int]) -> dict[str, int]:
    """Distance between two ranges of possible birth years (0 = they share a year), shared years, and the centre-to-centre gap (floored)."""
    g = max(ra[0], rb[0]) - min(ra[1], rb[1])
    return {"distance_years": max(0, g), "shared_years": 0 if g > 0 else 1 - g,
            "centre_gap_years": abs((ra[0] + ra[1]) - (rb[0] + rb[1])) // 2}


def _range_status(distance: int, config: ScoringConfig) -> tuple[str, Optional[str]]:
    if distance == 0:
        return "consistent", None
    if distance <= config.birth_year_tolerance_years:
        return "within_tolerance", None
    if distance <= config.birth_year_conflict_gap_years:
        return "outside_tolerance", "soft"
    return "conflict", "hard"


def compare_age_dob_v3(a: "PersonAttributes", b: "PersonAttributes", config: ScoringConfig) -> FieldResult:
    weight = config.effective_weight("age_dob")
    ea, eb = evidence_from_attributes(a), evidence_from_attributes(b)
    quality = [{**q, "record": side} for side, e in (("a", ea), ("b", eb)) for q in e.quality]
    flags = [{**f, "record": side} for side, e in (("a", ea), ("b", eb)) for f in e.flags]
    detail = {"rule_version": config.rule_version, "reference_date": config.reference_date, "reference_date_use": "validation cut-off only; never an age anchor",
              "age_tolerance_years": config.age_tolerance_years, "birth_year_tolerance_years": config.birth_year_tolerance_years,
              "birth_year_conflict_gap_years": config.birth_year_conflict_gap_years, "leap_day_birthday": config.leap_day_birthday,
              "a": _side_detail(ea), "b": _side_detail(eb), "data_quality": quality, "review_flags": flags}
    values = {"a": _side_values(ea), "b": _side_values(eb)}

    def result(score, text, conflict, method, status, extra) -> FieldResult:
        return FieldResult("age_dob", weight, True, score, text, None, conflict, values, {**detail, "method": method, "status": status, **extra})

    # 1. exact dates (identical to score-v2) -------------------------------------------------
    if ea.dob and eb.dob:
        diff = abs((ea.dob - eb.dob).days)
        if diff == 0:
            return result(1.0, "Identical dates of birth.", None, "dob_exact", "identical", {"difference_days": 0})
        if ea.dob.year == eb.dob.year and ea.dob.month == eb.dob.day and ea.dob.day == eb.dob.month:
            return result(config.dob_swapped_day_month_score, "Dates of birth differ only by swapped day and month.", None,
                          "dob_exact", "swapped_day_month", {"difference_days": diff})
        score = next((s for limit, s in config.dob_day_scores if diff <= limit), 0.0)
        conflict = "hard" if diff > config.dob_conflict_over_days else None
        return result(score, f"Dates of birth differ by {diff} day(s).", conflict, "dob_exact",
                      "near_match" if score > 0 and not conflict else "different", {"difference_days": diff})

    # 2. a DOB against age observations: the DOB's age is taken at each observation's own FIR date ----------
    for dob_side, age_side, dob_e, age_e in (("a", "b", ea, eb), ("b", "a", eb, ea)):
        if dob_e.dob and not age_e.dob and age_e.merged_range is not None:
            per_obs = []
            for o in age_e.usable_observations:
                implied = age_at(dob_e.dob, o.fir_date, config.leap_day_birthday)  # type: ignore[arg-type]
                per_obs.append({"fir_year": o.fir_date.year, "recorded_age": o.age, "age_implied_by_dob_at_fir_date": implied,  # type: ignore[union-attr]
                                "gap_years": abs(implied - o.age)})  # type: ignore[operator]
            gap = max(p["gap_years"] for p in per_obs)  # the DOB must fit every observation; the worst one decides (nothing is picked as 'best')
            status, conflict = _gap_status(gap, config)
            score = _factor(gap, config) * config.dob_vs_age_max_score
            text = (f"Record {dob_side.upper()} has a date of birth, record {age_side.upper()} only recorded age(s) tied to FIR date(s). The DOB implies "
                    f"{'/'.join(str(p['age_implied_by_dob_at_fir_date']) for p in per_obs)} at "
                    f"{'FIR year ' + '/'.join(str(p['fir_year']) for p in per_obs)}; recorded {'/'.join(str(p['recorded_age']) for p in per_obs)} "
                    f"({gap} year(s) apart, {status.replace('_', ' ')}).")
            return result(score, text, conflict, "dob_vs_recorded_age_at_fir_date", status,
                          {"gap_years": gap, "gap_factor": _factor(gap, config), "estimate_cap": config.dob_vs_age_max_score,
                           "dob_side": dob_side, "per_observation": per_obs})

    # 3. age observations only: compare possible birth years ------------------------------------------------
    if not ea.dob and not eb.dob and ea.merged_range is not None and eb.merged_range is not None:
        rel = range_relation(ea.merged_range, eb.merged_range)
        status, conflict = _range_status(rel["distance_years"], config)
        score = _factor(rel["centre_gap_years"], config) * config.age_estimate_max_score
        fa = sorted({o.fir_date.year for o in ea.usable_observations})  # type: ignore[union-attr]
        fb = sorted({o.fir_date.year for o in eb.usable_observations})  # type: ignore[union-attr]
        text = (f"Only recorded ages are available, each tied to its own FIR date (FIR year {'/'.join(map(str, fa))} vs {'/'.join(map(str, fb))}). "
                f"Possible birth years {list(ea.merged_range)} vs {list(eb.merged_range)}: {rel['distance_years']} year(s) apart, "
                f"{rel['shared_years']} shared ({status.replace('_', ' ')}). This is an estimate: equal birth-year ranges do not prove equal birth dates.")
        return result(score, text, conflict, "estimated_birth_year_ranges_at_fir_date", status,
                      {"birth_year_range_distance_years": rel["distance_years"], "birth_year_range_shared_years": rel["shared_years"],
                       "birth_year_centre_gap_years": rel["centre_gap_years"], "gap_factor": _factor(rel["centre_gap_years"], config),
                       "estimate_cap": config.age_estimate_max_score, "fir_years_a": fa, "fir_years_b": fb,
                       "same_fir_year": fa == fb and len(fa) == 1})

    # nothing comparable: say why, per side --------------------------------------------------------------
    sa, sb = ea.side_state(), eb.side_state()
    kind = "age_evidence_unavailable"
    if "age_observations_conflict" in (sa, sb):
        kind = "age_observations_conflict"
    reason = f"{kind}:{sa}/{sb}"
    return FieldResult("age_dob", weight, False, None,
                       f"No comparable age or date of birth (record A: {sa}; record B: {sb}). An age without a usable FIR date is not compared.",
                       reason, None, values, {**detail, "method": None, "status": kind})
