"""score-v2 age / DOB logic: reference date, DOB comparison, birth-year ranges, tolerance, inconsistencies, validity."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from dedup_pipeline.candidates.config import CandidateConfig
from dedup_pipeline.scoring.age import age_at, birth_year_range, build_age_info, parse_age, parse_dob
from dedup_pipeline.scoring.attributes import build_attributes
from dedup_pipeline.scoring.config import ScoringConfig
from dedup_pipeline.scoring.fields import score_age_dob
from dedup_pipeline.scoring.score import score_pair

from .scoring_fakes import V1CONFIG, V2CONFIG

REF = date(2026, 10, 12)
CFG = V2CONFIG(reference_date="2026-10-12")


def person(pid, age=None, dob=None, config=CFG, name="Ramesh Kumar"):
    row = (pid, "V2", f"r-{pid}", name, None, "Suresh Kumar", None, age, dob, "Male", None)
    return build_attributes(row, config, CandidateConfig())


def compare(a_kw, b_kw, config=CFG):
    return score_age_dob(person("a", config=config, **a_kw), person("b", config=config, **b_kw), config)


# --------------------------------------------------------------------------- reference date

def test_reference_date_is_mandatory_valid_and_recorded():
    for missing in (None, "", "not-a-date", "2026-13-40"):
        with pytest.raises(ValueError, match="reference_date"):
            ScoringConfig(rule_version="score-v2", reference_date=missing)
    ScoringConfig(rule_version="score-v1")  # score-v1 never needed one and still does not
    cfg = V2CONFIG(reference_date="2025-01-01")
    assert cfg.to_dict()["reference_date"] == "2025-01-01" and ScoringConfig.from_dict(cfg.to_dict()).reference_date == "2025-01-01"
    assert "2025-01-01" in cfg.signature()  # a different reference date is a different configuration, never a silent change


def test_the_reference_date_changes_the_estimates_and_is_stamped_on_every_comparison():
    early, late = V2CONFIG(reference_date="2026-01-01"), V2CONFIG(reference_date="2026-12-31")
    assert birth_year_range(30, date(2026, 1, 1)) == (1995, 1996) and birth_year_range(30, date(2026, 12, 31)) == (1996, 1996)
    for cfg in (early, late):
        result = compare({"age": 30}, {"age": 31}, cfg)
        assert result.detail["reference_date"] == cfg.reference_date and result.detail["age_tolerance_years"] == 2


def test_every_candidate_in_a_run_uses_the_same_reference_date():
    results = [compare({"age": a}, {"age": b}) for a, b in ((30, 30), (25, 26), (40, 44))]
    assert {r.detail["reference_date"] for r in results} == {"2026-10-12"}
    assert person("x").age_info is not None and person("y", config=V1CONFIG()).age_info is None  # v1 has no reference date at all


# -------------------------------------------------------------------------- the age maths

def test_birth_year_range_example_from_the_specification():
    assert birth_year_range(30, date(2026, 10, 12)) == (1995, 1996)  # reference year 2026, age 30 -> born 1995 or 1996


@pytest.mark.parametrize("dob,expected", [(date(1996, 10, 12), 30), (date(1996, 10, 13), 29), (date(1996, 10, 11), 30), (date(1995, 10, 13), 30),
                                          (date(1995, 10, 12), 31), (date(1996, 1, 1), 30), (date(1996, 12, 31), 29)])
def test_birthday_boundaries_on_the_reference_date(dob, expected):
    assert age_at(dob, REF) == expected  # a birthday ON the reference date has already been reached


@pytest.mark.parametrize(
    "policy,reference,expected",
    [("mar1", date(2026, 2, 28), 25), ("mar1", date(2026, 3, 1), 26), ("feb28", date(2026, 2, 28), 26), ("feb28", date(2026, 2, 27), 25),
     ("mar1", date(2028, 2, 29), 28), ("feb28", date(2028, 2, 29), 28), ("mar1", date(2028, 2, 28), 27)],
)
def test_leap_day_birthdays(policy, reference, expected):
    assert age_at(date(2000, 2, 29), reference, policy) == expected


def test_leap_day_policy_changes_which_year_ranges_are_possible():
    # a person born 29 Feb 2000 is 25 on 2026-02-28 under mar1 but 26 under feb28
    assert birth_year_range(25, date(2026, 2, 28), "mar1")[0] <= 2000 <= birth_year_range(25, date(2026, 2, 28), "mar1")[1]
    assert birth_year_range(26, date(2026, 2, 28), "feb28")[0] <= 2000 <= birth_year_range(26, date(2026, 2, 28), "feb28")[1]


@pytest.mark.parametrize("age", [5, 18, 30, 45, 100])
@pytest.mark.parametrize("reference", [date(2026, 1, 1), date(2026, 2, 28), date(2026, 3, 1), date(2026, 10, 12), date(2026, 12, 31), date(2028, 2, 29)])
def test_birth_year_range_matches_brute_force(age, reference):
    possible = set()
    day = date(reference.year - age - 2, 1, 1)
    while day <= date(reference.year - age + 1, 12, 31):
        if age_at(day, reference) == age:
            possible.add(day.year)
        day = date.fromordinal(day.toordinal() + 1)
    assert set(range(birth_year_range(age, reference)[0], birth_year_range(age, reference)[1] + 1)) == possible


# --------------------------------------------------------------------------- DOB against DOB

def test_equal_dob_values_score_the_strongest_agreement():
    result = compare({"dob": date(1990, 3, 4)}, {"dob": date(1990, 3, 4)})
    assert result.score == 1.0 and result.detail["method"] == "dob_exact" and result.detail["status"] == "identical" and result.conflict is None


@pytest.mark.parametrize(
    "other,score,conflict,status",
    [(date(1990, 3, 5), 0.9, None, "near_match"), (date(1990, 4, 1), 0.5, None, "near_match"), (date(1990, 9, 20), 0.25, "hard", "different"),
     (date(1985, 3, 4), 0.0, "hard", "different"), (date(1990, 4, 3), 0.8, None, "swapped_day_month")],
)
def test_different_dob_values_follow_the_configured_policy(other, score, conflict, status):
    result = compare({"dob": date(1990, 3, 4)}, {"dob": other})
    assert (result.score, result.conflict, result.detail["status"]) == (score, conflict, status)


def test_dob_policy_is_configurable():
    strict = V2CONFIG(dob_day_scores=((0, 1.0),), dob_conflict_over_days=0)
    result = compare({"dob": date(1990, 3, 4)}, {"dob": date(1990, 3, 5)}, strict)
    assert result.score == 0.0 and result.conflict == "hard"
    with pytest.raises(ValueError):
        V2CONFIG(dob_day_scores=((31, 0.5), (0, 1.0)))


# ---------------------------------------------------------------- estimated birth-year ranges

def test_equal_ages_never_score_like_an_exact_dob_match():
    result = compare({"age": 30}, {"age": 30})
    assert result.score == pytest.approx(0.7) and result.score < 1.0 and result.detail["method"] == "estimated_birth_year_ranges"
    assert result.values["a"]["birth_year_range"] == [1995, 1996] and result.values["b"]["birth_year_range"] == [1995, 1996]
    assert result.detail["birth_year_range_overlap_years"] == 2 and "do not prove equal birth years" in result.explanation


def test_same_recorded_age_can_hide_different_birth_years():
    # age 30 at the reference date: born 1995 or 1996 - two people with this "same age" may differ by a year
    same = compare({"age": 30}, {"age": 30})
    assert same.values["a"]["birth_year_range"] == [1995, 1996] and same.score < compare({"dob": date(1996, 3, 3)}, {"dob": date(1996, 3, 3)}).score
    # ages differing by one: their ranges still overlap in exactly one year but they are not the same estimate
    neighbour = compare({"age": 30}, {"age": 31})
    assert neighbour.detail["birth_year_range_overlap_years"] == 1 and neighbour.score < same.score
    separate = compare({"age": 30}, {"age": 33})
    assert separate.detail["birth_year_range_overlap_years"] == 0 and separate.detail["birth_year_range_gap_years"] == 2


@pytest.mark.parametrize(
    "a,b,status,conflict,factor",
    [(30, 30, "consistent", None, 1.0), (30, 31, "within_tolerance", None, 0.85), (30, 32, "within_tolerance", None, 0.6),
     (30, 33, "outside_tolerance", "soft", 0.35), (30, 35, "outside_tolerance", "soft", 0.0), (30, 36, "conflict", "hard", 0.0)],
)
def test_age_tolerance_and_conflict_thresholds(a, b, status, conflict, factor):
    result = compare({"age": a}, {"age": b})
    assert (result.detail["status"], result.conflict) == (status, conflict)
    expected = {0: 1.0, 1: 0.85, 2: 0.6, 3: 0.35, 4: 0.15}.get(abs(a - b), 0.0)
    assert result.score == pytest.approx(expected * 0.7) and result.detail["gap_years"] == abs(a - b)


def test_age_tolerance_and_curve_are_configurable():
    tight = V2CONFIG(age_tolerance_years=0, age_gap_factors=(1.0, 0.5), age_conflict_gap_years=1)
    one = compare({"age": 30}, {"age": 31}, tight)
    assert one.detail["status"] == "outside_tolerance" and one.score == pytest.approx(0.35) and one.conflict == "soft"
    assert compare({"age": 30}, {"age": 32}, tight).conflict == "hard"
    loose = V2CONFIG(age_tolerance_years=4, age_conflict_gap_years=6)
    assert compare({"age": 30}, {"age": 34}, loose).detail["status"] == "within_tolerance"
    for bad in ({"age_tolerance_years": 5, "age_conflict_gap_years": 3}, {"age_gap_factors": ()}, {"age_estimate_max_score": 1.5}):
        with pytest.raises(ValueError):
            V2CONFIG(**bad)


def test_score_v1_still_gives_equal_ages_a_perfect_score():
    result = compare({"age": 30}, {"age": 30}, V1CONFIG())
    assert result.score == 1.0 and not result.detail  # the old behaviour, untouched


# ----------------------------------------------------------------- DOB against recorded age

def test_dob_in_one_record_and_age_in_the_other():
    consistent = compare({"dob": date(1995, 12, 1)}, {"age": 30})  # DOB implies 30 at the reference date
    assert consistent.detail["method"] == "dob_vs_recorded_age" and consistent.detail["status"] == "consistent"
    assert consistent.score == pytest.approx(0.8) and consistent.score < 1.0 and consistent.detail["dob_year_inside_birth_year_range_of_other"] is True
    assert compare({"age": 30}, {"dob": date(1995, 12, 1)}).detail["dob_side"] == "b"  # symmetric
    near = compare({"dob": date(1995, 12, 1)}, {"age": 32})
    assert near.detail["status"] == "within_tolerance" and near.score == pytest.approx(0.8 * 0.6)
    far = compare({"dob": date(1995, 12, 1)}, {"age": 45})
    assert far.conflict == "hard" and far.score == 0.0 and far.detail["gap_years"] == 15


def test_dob_birthday_boundary_against_a_recorded_age():
    just_turned = compare({"dob": date(1996, 10, 12)}, {"age": 30})
    not_yet = compare({"dob": date(1996, 10, 13)}, {"age": 30})
    assert just_turned.detail["status"] == "consistent" and not_yet.detail["status"] == "within_tolerance"  # 29 vs 30


# ----------------------------------------------------------------------- inconsistency

def test_a_record_whose_age_contradicts_its_own_dob_is_flagged_not_silently_resolved():
    result = compare({"dob": date(1996, 3, 3), "age": 45}, {"age": 30})
    assert result.detail["method"] == "dob_vs_recorded_age"  # the DOB is the documented precedence...
    assert result.detail["data_quality"] == [{"record": "a", "issue": "recorded_age_contradicts_own_dob", "recorded_age": 45, "age_implied_by_dob": 30}]
    assert result.detail["a"]["inconsistent_age_and_dob"] is True and result.detail["b"]["inconsistent_age_and_dob"] is False
    both = compare({"dob": date(1996, 3, 3), "age": 45}, {"dob": date(1996, 3, 3), "age": 20})
    assert {q["record"] for q in both.detail["data_quality"]} == {"a", "b"} and both.score == 1.0  # DOBs agree; the problems are still reported


def test_internal_consistency_tolerates_a_year_of_staleness_only():
    assert not build_age_info(date(1996, 3, 3), 29, CFG).inconsistent and not build_age_info(date(1996, 3, 3), 31, CFG).inconsistent
    assert build_age_info(date(1996, 3, 3), 33, CFG).inconsistent


# ------------------------------------------------------------- invalid, placeholder, missing

@pytest.mark.parametrize(
    "raw,status",
    [("1900-01-01", "invalid:placeholder"), (date(1970, 1, 1), "invalid:placeholder"), ("31-02-1990", "invalid:impossible_date"), ("1990-13-01", "invalid:impossible_date"),
     ("not a date", "invalid:unparseable"), (12345, "invalid:unparseable"), (date(1850, 6, 1), "invalid:before_minimum_year"),
     (date(2030, 1, 1), "invalid:in_the_future"), (date(2024, 1, 1), "invalid:implausible_age_at_reference_date"), (None, "missing"), ("  ", "missing")],
)
def test_invalid_placeholder_and_missing_dob_are_never_valid_evidence(raw, status):
    assert parse_dob(raw, REF, CFG) == (None, status)


@pytest.mark.parametrize("raw,expected", [("1990-03-04", date(1990, 3, 4)), ("04-03-1990", date(1990, 3, 4)), ("04/03/1990", date(1990, 3, 4)),
                                          (datetime(1990, 3, 4, 10, 0), date(1990, 3, 4))])
def test_valid_dob_formats_normalise_to_one_date_type(raw, expected):
    assert parse_dob(raw, REF, CFG) == (expected, "valid")


@pytest.mark.parametrize("raw,status", [(0, "invalid:out_of_range"), (3, "invalid:out_of_range"), (150, "invalid:out_of_range"), (671, "invalid:out_of_range"),
                                        (30.5, "invalid:not_a_whole_number"), ("abc", "invalid:not_a_number"), (True, "invalid:not_a_number"), (None, "missing")])
def test_invalid_and_missing_ages(raw, status):
    assert parse_age(raw, CFG) == (None, status)
    assert parse_age("30", CFG) == (30, "valid") and parse_age(30.0, CFG) == (30, "valid")


def test_an_invalid_dob_falls_back_to_a_valid_age_and_says_so():
    result = compare({"dob": "1900-01-01", "age": 30}, {"age": 30})
    assert result.detail["method"] == "estimated_birth_year_ranges" and result.detail["a"]["dob_status"] == "invalid:placeholder"


def test_missing_invalid_and_one_sided_evidence_are_distinguished_and_never_scored_as_zero():
    both_missing = compare({}, {})
    invalid = compare({"dob": "1900-01-01", "age": 671}, {"age": 30, "dob": None})
    one_side = compare({"age": 30}, {})
    assert not both_missing.applicable and both_missing.score is None and both_missing.reason_not_applicable == "age_and_dob_missing:missing/missing"
    assert not invalid.applicable and invalid.reason_not_applicable == "invalid_age_or_dob:invalid/usable"
    assert one_side.reason_not_applicable.startswith("age_or_dob_on_one_side_only") and one_side.score is None
    assert invalid.detail["status"] in ("invalid_age_or_dob", "age_or_dob_on_one_side_only")
    both_invalid = compare({"age": 671}, {"dob": date(2030, 1, 1)})
    assert both_invalid.reason_not_applicable.startswith("invalid_age_or_dob") and both_invalid.score is None


def test_missing_age_and_dob_on_both_sides_is_excluded_from_the_denominator_not_scored_zero():
    a, b = person("a", age=None), person("b", age=None)
    outcome = score_pair(a, b, None, CFG)
    assert not outcome.fields["age_dob"].applicable and outcome.match_score == 1.0  # name and relative name agree; age adds nothing either way
    with_age = score_pair(person("a", age=30), person("b", age=30), None, CFG)
    assert with_age.coverage > outcome.coverage and with_age.match_score < 1.0  # an estimate (0.7) now participates, but never as 1.0
