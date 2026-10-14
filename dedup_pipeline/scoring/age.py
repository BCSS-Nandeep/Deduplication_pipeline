"""score-v2 age / date-of-birth evidence, evaluated at one explicit reference date.

Formulae (R = reference date, A = recorded integer age):

* ``age_at(dob, R)`` = R.year - dob.year, minus 1 if the birthday has not yet happened in R's year
  (a 29 Feb birthday is reached on 1 Mar or 28 Feb in non-leap years, per ``leap_day_birthday``).
* Possible birth years for a recorded age A: every calendar year y in which some date has ``age_at(date, R) == A``,
  i.e. ``age_at(31 Dec y) <= A <= age_at(1 Jan y)``. This is normally ``[R.year - A - 1, R.year - A]``
  and shrinks to one year when R is 1 Jan or 31 Dec. It is an estimate, not a date of birth.

Comparison, in this order (never choosing whichever scores higher):

1. both records have a valid DOB: compare the dates (``dob_exact``);
2. a valid DOB on one side and a valid recorded age on the other: compare the age the DOB implies at R with the
   recorded age (``dob_vs_recorded_age``);
3. valid recorded ages only: compare them and their birth-year ranges (``estimated_birth_year_ranges``).

For 2 and 3 the *gap* is the whole-year difference of the two age estimates. Within ``age_tolerance_years`` it is
accepted as stale or approximate data; the score is ``factor[gap] x cap`` where the cap (< 1) keeps an estimate from
ever scoring like an exact DOB match. Missing, invalid (placeholder, impossible, future, implausible) and
contradictory evidence are kept apart in the reasons and statuses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

from .config import ScoringConfig
from .fields import FieldResult

if TYPE_CHECKING:
    from .attributes import PersonAttributes

_DMY = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$")
_YMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")


def _birthday_in(year: int, dob: date, leap_policy: str) -> date:
    if dob.month == 2 and dob.day == 29:
        try:
            return date(year, 2, 29)
        except ValueError:
            return date(year, 3, 1) if leap_policy == "mar1" else date(year, 2, 28)
    return date(year, dob.month, dob.day)


def age_at(dob: date, reference: date, leap_policy: str = "mar1") -> int:
    return reference.year - dob.year - (1 if reference < _birthday_in(reference.year, dob, leap_policy) else 0)


def birth_year_range(age: int, reference: date, leap_policy: str = "mar1") -> tuple[int, int]:
    years = [y for y in range(reference.year - age - 2, reference.year - age + 2)
             if age_at(date(y, 12, 31), reference, leap_policy) <= age <= age_at(date(y, 1, 1), reference, leap_policy)]
    return min(years), max(years)


def parse_dob(raw: Any, reference: date, config: ScoringConfig) -> tuple[Optional[date], str]:
    """``(date, "valid")`` or ``(None, "missing" | "invalid:<reason>")``."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, "missing"
    parsed: Optional[date] = None
    if isinstance(raw, datetime):
        parsed = raw.date()
    elif isinstance(raw, date):
        parsed = raw
    elif isinstance(raw, str):
        text = raw.strip()
        m = _YMD.match(text)
        parts = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
        if not m and (m := _DMY.match(text)):
            parts = (int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if parts is None:
            return None, "invalid:unparseable"
        try:
            parsed = date(*parts)
        except ValueError:
            return None, "invalid:impossible_date"
    else:
        return None, "invalid:unparseable"
    if parsed.isoformat() in config.dob_placeholder_dates:
        return None, "invalid:placeholder"
    if parsed.year < config.dob_min_year:
        return None, "invalid:before_minimum_year"
    if parsed > reference:
        return None, "invalid:in_the_future"
    if not config.age_min <= age_at(parsed, reference, config.leap_day_birthday) <= config.age_max:
        return None, "invalid:implausible_age_at_reference_date"
    return parsed, "valid"


def parse_age(raw: Any, config: ScoringConfig) -> tuple[Optional[int], str]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, "missing"
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None, "invalid:not_a_number"
    try:
        number = float(raw)
    except ValueError:
        return None, "invalid:not_a_number"
    if number != int(number):
        return None, "invalid:not_a_whole_number"
    if not config.age_min <= int(number) <= config.age_max:
        return None, "invalid:out_of_range"
    return int(number), "valid"


@dataclass(frozen=True)
class AgeInfo:
    dob: Optional[date]
    dob_status: str
    age: Optional[int]
    age_status: str
    derived_age: Optional[int]  # the age the valid DOB implies at the reference date
    year_range: Optional[tuple[int, int]]  # possible birth years: the DOB year, or the estimate from the recorded age
    inconsistent: bool  # DOB and recorded age of this record disagree by more than the internal tolerance

    @property
    def state(self) -> str:
        if self.dob is not None or self.age is not None:
            return "usable"
        return "invalid" if "invalid" in self.dob_status + self.age_status else "missing"


def build_age_info(raw_dob: Any, raw_age: Any, config: ScoringConfig) -> Optional[AgeInfo]:
    """``None`` for score-v1 (no reference date)."""
    if not config.reference_date:
        return None
    reference = date.fromisoformat(config.reference_date)
    dob, dob_status = parse_dob(raw_dob, reference, config)
    age, age_status = parse_age(raw_age, config)
    derived = age_at(dob, reference, config.leap_day_birthday) if dob else None
    if dob:
        year_range = (dob.year, dob.year)
    elif age is not None:
        year_range = birth_year_range(age, reference, config.leap_day_birthday)
    else:
        year_range = None
    inconsistent = dob is not None and age is not None and abs(derived - age) > config.age_dob_internal_tolerance_years  # type: ignore[operator]
    return AgeInfo(dob, dob_status, age, age_status, derived, year_range, inconsistent)


# --------------------------------------------------------------------------------- comparison

def _factor(gap: int, config: ScoringConfig) -> float:
    return config.age_gap_factors[gap] if gap < len(config.age_gap_factors) else 0.0


def _gap_status(gap: int, config: ScoringConfig) -> tuple[str, Optional[str]]:
    if gap == 0:
        return "consistent", None
    if gap <= config.age_tolerance_years:
        return "within_tolerance", None
    if gap <= config.age_conflict_gap_years:
        return "outside_tolerance", "soft"
    return "conflict", "hard"


def _side(info: AgeInfo) -> dict[str, Any]:
    return {"dob_status": info.dob_status, "age_status": info.age_status, "inconsistent_age_and_dob": info.inconsistent}


def _quality(info_a: AgeInfo, info_b: AgeInfo) -> list[dict[str, Any]]:
    return [{"record": side, "issue": "recorded_age_contradicts_own_dob", "recorded_age": i.age, "age_implied_by_dob": i.derived_age}
            for side, i in (("a", info_a), ("b", info_b)) if i.inconsistent]


def _values(info_a: AgeInfo, info_b: AgeInfo) -> dict[str, Any]:
    side = lambda i: {"recorded_age": i.age, "dob": i.dob.isoformat() if i.dob else None, "age_implied_by_dob": i.derived_age,
                      "birth_year_range": list(i.year_range) if i.year_range else None}
    return {"a": side(info_a), "b": side(info_b)}


def compare_age_dob(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    weight = config.effective_weight("age_dob")
    ia, ib = a.age_info, b.age_info
    base = {"reference_date": config.reference_date, "age_tolerance_years": config.age_tolerance_years,
            "leap_day_birthday": config.leap_day_birthday, "rule_version": config.rule_version}
    if ia is None or ib is None:
        raise ValueError("score-v2 age comparison needs attributes built with a reference date")
    quality = _quality(ia, ib)
    detail = {**base, "a": _side(ia), "b": _side(ib), "data_quality": quality}

    def result(score, text, conflict, method, status, extra) -> FieldResult:
        return FieldResult("age_dob", weight, True, score, text, None, conflict, _values(ia, ib), {**detail, "method": method, "status": status, **extra})

    # 1. exact dates ------------------------------------------------------------------
    if ia.dob and ib.dob:
        diff = abs((ia.dob - ib.dob).days)
        if diff == 0:
            return result(1.0, "Identical dates of birth.", None,
                          "dob_exact", "identical", {"difference_days": 0})
        swapped = ia.dob.year == ib.dob.year and ia.dob.month == ib.dob.day and ia.dob.day == ib.dob.month
        if swapped:
            return result(config.dob_swapped_day_month_score, "Dates of birth differ only by swapped day and month.", None,
                          "dob_exact", "swapped_day_month", {"difference_days": diff})
        score = next((s for limit, s in config.dob_day_scores if diff <= limit), 0.0)
        conflict = "hard" if diff > config.dob_conflict_over_days else None
        status = "near_match" if score > 0 and not conflict else "different"
        return result(score, f"Dates of birth differ by {diff} day(s).", conflict, "dob_exact", status, {"difference_days": diff})

    # 2. a DOB against a recorded age -------------------------------------------------------
    for dob_side, age_side, dob_info, age_info in (("a", "b", ia, ib), ("b", "a", ib, ia)):
        if dob_info.dob and not age_info.dob and age_info.age is not None:
            gap = abs(dob_info.derived_age - age_info.age)  # type: ignore[operator]
            status, conflict = _gap_status(gap, config)
            score = _factor(gap, config) * config.dob_vs_age_max_score
            in_range = age_info.year_range[0] <= dob_info.dob.year <= age_info.year_range[1]  # type: ignore[index]
            text = (f"Record {dob_side.upper()} has a date of birth, record {age_side.upper()} only a recorded age. The DOB implies age "
                    f"{dob_info.derived_age} at {config.reference_date}; the recorded age is {age_info.age} ({gap} year(s) apart, {status.replace('_', ' ')}).")
            return result(score, text, conflict, "dob_vs_recorded_age", status,
                          {"gap_years": gap, "gap_factor": _factor(gap, config), "estimate_cap": config.dob_vs_age_max_score, "dob_side": dob_side,
                           "dob_year_inside_birth_year_range_of_other": in_range})

    # 3. recorded ages only (estimates) -----------------------------------------------------
    if ia.age is not None and ib.age is not None and not ia.dob and not ib.dob:
        gap = abs(ia.age - ib.age)
        status, conflict = _gap_status(gap, config)
        lo = max(ia.year_range[0], ib.year_range[0])  # type: ignore[index]
        hi = min(ia.year_range[1], ib.year_range[1])  # type: ignore[index]
        overlap = max(0, hi - lo + 1)
        score = _factor(gap, config) * config.age_estimate_max_score
        text = (f"Only recorded ages are available ({gap} year(s) apart, {status.replace('_', ' ')}); possible birth years overlap in {overlap} year(s). "
                "This is an estimate: equal ages do not prove equal birth years.")
        return result(score, text, conflict, "estimated_birth_year_ranges", status,
                      {"gap_years": gap, "gap_factor": _factor(gap, config), "estimate_cap": config.age_estimate_max_score,
                       "birth_year_range_overlap_years": overlap, "birth_year_range_gap_years": 0 if overlap else lo - hi})

    # nothing comparable: say whether it is missing or invalid ----------------------------
    states = f"{ia.state}/{ib.state}"
    kind = "invalid_age_or_dob" if "invalid" in states else ("age_or_dob_on_one_side_only" if "usable" in states else "age_and_dob_missing")
    missing = FieldResult("age_dob", weight, False, None,
                          f"No comparable age or date of birth (record A: {ia.state}, record B: {ib.state}).",
                          f"{kind}:{states}", None, _values(ia, ib) if "usable" in states else {}, {**detail, "method": None, "status": kind})
    return missing
