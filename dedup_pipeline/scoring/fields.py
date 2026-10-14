"""Identity-field scorers. Each returns a ``FieldResult`` that says what was compared, the score, and why.

``applicable=False`` means the field carries no information for this pair (data missing, or a one-sided value that
cannot support or contradict anything) and is excluded from both the score and the coverage numerator. A field is
only ``applicable`` with a score when both sides can actually be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .attributes import Address, PersonAttributes
from .config import ScoringConfig
from .similarity import name_similarity


@dataclass
class FieldResult:
    field: str
    weight: float
    applicable: bool
    score: Optional[float] = None
    explanation: str = ""
    reason_not_applicable: Optional[str] = None
    conflict: Optional[str] = None  # None | "hard" | "soft"
    values: dict[str, Any] = field(default_factory=dict)  # shown to the reviewer only when include_values is set
    detail: dict[str, Any] = field(default_factory=dict)  # method, reference date, tolerance, statuses: no personal values

    def to_dict(self, include_values: bool) -> dict[str, Any]:
        data: dict[str, Any] = {
            "weight": round(self.weight, 4), "applicable": self.applicable,
            "score": None if self.score is None else round(self.score, 4), "explanation": self.explanation,
        }
        if self.reason_not_applicable:
            data["not_applicable_reason"] = self.reason_not_applicable
        if self.conflict:
            data["conflict"] = self.conflict
        if self.detail:
            data["detail"] = self.detail
        if include_values and self.values:
            data["values"] = self.values
        return data


def _na(name: str, config: ScoringConfig, reason: str, text: str) -> FieldResult:
    return FieldResult(name, config.effective_weight(name), False, None, text, reason)


def _mask(phone: Optional[str]) -> str:
    return f"******{phone[-4:]}" if phone else ""


# ------------------------------------------------------------------------------ names

def score_name(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    result = name_similarity(a.name, b.name)
    if result is None:
        return _na("name", config, "name_missing", "A name is missing on at least one record.")
    score, method = result
    conflict = "hard" if score < config.name_conflict_below else None
    text = f"Name similarity {score:.2f} ({method})."
    return FieldResult("name", config.effective_weight("name"), True, score, text, None, conflict,
                       {"a": a.name, "b": b.name})


def score_relative_name(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    result = name_similarity(a.relative_name, b.relative_name)
    if result is None:
        return _na("relative_name", config, "relative_name_missing", "Relative's name missing on at least one record.")
    score, method = result
    conflict = "hard" if score < config.relative_conflict_below else None
    return FieldResult("relative_name", config.effective_weight("relative_name"), True, score,
                       f"Relative-name similarity {score:.2f} ({method}).", None, conflict,
                       {"a": a.relative_name, "b": b.relative_name})


def score_alias(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    """Best similarity between either alias and the other person's name or alias.

    With an alias on both sides the comparison is informative either way. With an alias on one side only, an alias
    that does not match the other name says nothing (aliases are optional), so only a clear match counts.
    """
    pairs = [(x, y) for x, y in ((a.alias, b.name), (b.alias, a.name), (a.alias, b.alias)) if x and y]
    if not pairs:
        return _na("alias", config, "no_alias", "Neither record has an alias to compare.")
    best = max((name_similarity(x, y) for x, y in pairs), key=lambda r: r[0])
    both = bool(a.alias and b.alias)
    if not both and best[0] < config.alias_positive_only_threshold:
        return _na("alias", config, "one_sided_alias_without_match",
                   "Only one record has an alias and it does not match the other record's name.")
    return FieldResult("alias", config.effective_weight("alias"), True, best[0],
                       f"Alias similarity {best[0]:.2f} ({best[1]}).", None, None,
                       {"a": a.alias, "b": b.alias})


# ----------------------------------------------------------------------------- phone

def score_phone(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    if a.phone is None or b.phone is None:
        side = "either record" if a.phone is None and b.phone is None else ("record A" if a.phone is None else "record B")
        why = a.phone_reason if a.phone is None else b.phone_reason
        return _na("phone", config, f"no_usable_phone:{why}", f"No usable phone number on {side} ({why}).")
    values = {"a": _mask(a.phone), "b": _mask(b.phone)}
    if a.phone == b.phone:
        return FieldResult("phone", config.effective_weight("phone"), True, 1.0, "Same phone number.", None, None, values)
    return FieldResult("phone", config.effective_weight("phone"), True, 0.0,
                       "Different phone numbers (one person can have several, so this is only weak evidence).",
                       None, "soft", values)


# ------------------------------------------------------------------------- age / DOB

def score_age_dob(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    if config.rule_version == "score-v3":  # FIR-date-anchored age evidence in age_v3.py
        from .age_v3 import compare_age_dob_v3

        return compare_age_dob_v3(a, b, config)
    if config.rule_version != "score-v1":  # score-v2: reference-date logic in age.py
        from .age import compare_age_dob

        return compare_age_dob(a, b, config)
    return score_age_dob_v1(a, b, config)


def score_age_dob_v1(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    """score-v1, kept byte-for-byte in behaviour: integer difference of recorded ages, DOB year logic."""
    weight = config.effective_weight("age_dob")
    if a.dob and b.dob:
        if a.dob == b.dob:
            return FieldResult("age_dob", weight, True, 1.0, "Same date of birth.", None, None, {"a": a.dob.isoformat(), "b": b.dob.isoformat()})
        swapped = a.dob.year == b.dob.year and a.dob.month == b.dob.day and a.dob.day == b.dob.month
        if swapped:
            score, text, conflict = config.dob_swapped_day_month_score, "Dates of birth differ only by swapped day and month.", None
        elif a.dob.year == b.dob.year:
            score, text, conflict = config.dob_same_year_score, "Dates of birth are in the same year but differ.", "hard"
        else:
            score, text, conflict = 0.0, "Dates of birth differ.", "hard"
        return FieldResult("age_dob", weight, True, score, text, None, conflict, {"a": a.dob.isoformat(), "b": b.dob.isoformat()})
    if a.age is None or b.age is None:
        why = a.age_reason if a.age is None else b.age_reason
        return _na("age_dob", config, f"no_comparable_age:{why}", f"No comparable date of birth or plausible age ({why}).")
    diff = abs(a.age - b.age)
    score = config.age_scores[diff] if diff < len(config.age_scores) else 0.0
    conflict = "hard" if diff > config.age_conflict_over else None
    return FieldResult("age_dob", weight, True, score,
                       f"Recorded ages differ by {diff} year(s) (records are made at different times, so small gaps are normal).",
                       None, conflict, {"a": a.age, "b": b.age})


# ---------------------------------------------------------------------------- gender

def score_gender(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    if a.gender is None or b.gender is None:
        return _na("gender", config, "gender_not_definite", "Gender missing or not Male/Female on at least one record.")
    if a.gender == b.gender:
        return FieldResult("gender", config.effective_weight("gender"), True, 1.0, f"Same gender ({a.gender}).", None, None,
                           {"a": a.gender, "b": b.gender})
    return FieldResult("gender", config.effective_weight("gender"), True, 0.0, "Recorded genders differ.", None, "hard",
                       {"a": a.gender, "b": b.gender})


# --------------------------------------------------------------------------- address

def _level(x: Address, y: Address) -> str:
    if x.district and x.district == y.district:
        return "mandal" if x.mandal and x.mandal == y.mandal else "district"
    return "state" if x.state and x.state == y.state else "none"


_ORDER = ("none", "state", "district", "mandal")


def score_address(a: PersonAttributes, b: PersonAttributes, config: ScoringConfig) -> FieldResult:
    """Best agreement over present/permanent address combinations, at state / district / mandal level."""
    combos = [(x, y) for x in a.addresses for y in b.addresses]
    if not combos:
        return _na("address", config, "no_comparable_address", "No structured address on at least one record.")
    best = max((_level(x, y) for x, y in combos), key=_ORDER.index)
    score = config.address_scores[best]
    text = {"mandal": "Same district and mandal in at least one address.", "district": "Same district in at least one address.",
            "state": "Same state only; districts are missing or differ.", "none": "Addresses do not agree at state or district level."}[best]
    conflict = "soft" if best in ("none", "state") and all(x.district and y.district for x, y in combos) else None
    if conflict:
        text += " (People move, so this is only weak evidence.)"
    return FieldResult("address", config.effective_weight("address"), True, score, text, None, conflict, {"best_level": best})


SCORERS = (score_name, score_alias, score_relative_name, score_phone, score_age_dob, score_gender, score_address)
