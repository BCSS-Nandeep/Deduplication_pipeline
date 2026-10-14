"""score-v3 age evidence: ages anchored to FIR dates, birth-year ranges, observations, DOB kept separate, unavailable evidence."""

from __future__ import annotations

import itertools
from datetime import date, datetime, timezone

import pytest

from dedup_pipeline.scoring import ScoringConfig
from dedup_pipeline.scoring.age import birth_year_range, compare_age_dob
from dedup_pipeline.scoring.age_v3 import build_age_evidence, compare_age_dob_v3, parse_fir_date, range_relation
from dedup_pipeline.scoring.attributes import PersonAttributes, build_attributes
from dedup_pipeline.scoring.config import V3_ONLY_FIELDS
from dedup_pipeline.scoring.fields import score_age_dob
from dedup_pipeline.candidates.config import CandidateConfig

from .scoring_fakes import V1CONFIG, V2CONFIG, V3CONFIG

D = date


def person(config, age, fir=None, dob=None, pid="p"):
    """fir: a date, a list of dates, or a list of full (accused_id, crime_id, date) rows."""
    rows = None
    if fir is not None:
        items = fir if isinstance(fir, list) else [fir]
        rows = [it if isinstance(it, tuple) else (f"acc-{pid}-{i}", f"crime-{pid}-{i}", it) for i, it in enumerate(items)]
    return PersonAttributes(person_id=pid, source_system="V2", source_record_id=f"r-{pid}", age_info=build_age_evidence(dob, age, rows, config))


def cmp(config, a, b):
    return compare_age_dob_v3(a, b, config)


# ------------------------------------------------------------------ ranges from age + FIR date

def test_different_fir_years_with_consistent_ages_are_consistent_not_a_conflict():
    c = V3CONFIG()
    a, b = person(c, 26, D(2022, 3, 1), pid="a"), person(c, 30, D(2026, 7, 1), pid="b")
    assert a.age_info.merged_range == (1995, 1996) and b.age_info.merged_range == (1995, 1996)
    r = cmp(c, a, b)
    assert (r.applicable, r.conflict, r.detail["status"], r.detail["method"]) == (True, None, "consistent", "estimated_birth_year_ranges_at_fir_date")
    assert r.score == pytest.approx(1.0 * c.age_estimate_max_score)  # an estimate never scores like an exact DOB
    # the same two records read at ONE date (score-v2) look 4 years apart:
    v2 = V2CONFIG()
    r2 = compare_age_dob(build_attributes_v2(v2, 26), build_attributes_v2(v2, 30), v2)
    assert r2.conflict == "soft" and r2.detail["gap_years"] == 4


def build_attributes_v2(config, age, dob=None):
    row = ("p", "V2", "r", "n", None, None, None, age, dob, None, None)
    return build_attributes(row, config, CandidateConfig())


def test_ages_in_the_same_fir_year_match_the_score_v2_classification():
    c, v2 = V3CONFIG(), V2CONFIG(reference_date="2024-06-15")
    for gap in range(0, 10):
        a, b = person(c, 30, D(2024, 6, 15), pid="a"), person(c, 30 + gap, D(2024, 6, 15), pid="b")
        new = cmp(c, a, b)
        old = compare_age_dob(build_attributes_v2(v2, 30), build_attributes_v2(v2, 30 + gap), v2)
        assert new.conflict == old.conflict, gap  # same FIR date: v3 reduces to the same soft/hard decision
        assert new.score == pytest.approx(old.score), gap


def test_two_ages_one_year_apart_in_one_year_score_below_equal_ages():
    c = V3CONFIG()
    same = cmp(c, person(c, 30, D(2024, 6, 1)), person(c, 30, D(2024, 6, 1)))
    one = cmp(c, person(c, 30, D(2024, 6, 1)), person(c, 31, D(2024, 6, 1)))
    assert one.score < same.score and one.conflict is None


@pytest.mark.parametrize("fir,age,expected", [
    (D(2024, 1, 1), 30, (1993, 1994)),    # on 1 Jan the birthday may or may not have passed: two years
    (D(2024, 12, 31), 30, (1994, 1994)),  # on 31 Dec every birthday of the year has passed: one year
    (D(2024, 6, 30), 30, (1993, 1994)),
    (D(2024, 2, 29), 30, (1993, 1994)),
])
def test_birthday_uncertainty_in_the_birth_year_range(fir, age, expected):
    assert birth_year_range(age, fir) == expected
    assert person(V3CONFIG(), age, fir).age_info.merged_range == expected


def test_birthday_uncertainty_alone_never_creates_a_conflict():
    c = V3CONFIG()
    # same age, one FIR on 1 Jan and one on 31 Dec of the same year: ranges (1993,1994) and (1994,1994) overlap
    r = cmp(c, person(c, 30, D(2024, 1, 1)), person(c, 30, D(2024, 12, 31)))
    assert r.conflict is None and r.detail["status"] == "consistent" and r.detail["birth_year_range_shared_years"] == 1
    # ages 31 vs 30 on the two dates: ranges (1992,1993) and (1994,1994) are 1 year apart -> within tolerance, not a conflict
    r = cmp(c, person(c, 31, D(2024, 1, 1)), person(c, 30, D(2024, 12, 31)))
    assert r.conflict is None and r.detail["status"] in ("consistent", "within_tolerance")


def test_range_relation_distance_and_centres():
    assert range_relation((1995, 1996), (1996, 1997)) == {"distance_years": 0, "shared_years": 1, "centre_gap_years": 1}
    assert range_relation((1995, 1996), (1995, 1996)) == {"distance_years": 0, "shared_years": 2, "centre_gap_years": 0}
    assert range_relation((1995, 1996), (1998, 1999)) == {"distance_years": 2, "shared_years": 0, "centre_gap_years": 3}


def test_far_apart_birth_years_are_a_hard_conflict_and_thresholds_are_configurable():
    c = V3CONFIG()
    a, b = person(c, 24, D(2018, 6, 1)), person(c, 24, D(2026, 6, 1))  # born ~1993/94 vs ~2001/02: 8 apart
    r = cmp(c, a, b)
    assert r.conflict == "hard" and r.detail["status"] == "conflict" and r.detail["birth_year_range_distance_years"] > c.birth_year_conflict_gap_years
    lenient = V3CONFIG(birth_year_conflict_gap_years=9, birth_year_tolerance_years=2)
    r = cmp(lenient, person(lenient, 24, D(2018, 6, 1)), person(lenient, 24, D(2026, 6, 1)))
    assert r.conflict == "soft"


# ----------------------------------------------------------------- unavailable evidence (never today's date)

@pytest.mark.parametrize("fir_rows,reason", [
    (None, "no_fir_link"),
    ([], "no_fir_link"),
    ([("acc", "crime", None)], "fir_date_missing"),
    ([("acc", "crime", D(1900, 1, 1))], "fir_date_invalid_placeholder"),
    ([("acc", "crime", D(1970, 1, 1))], "fir_date_invalid_placeholder"),
    ([("acc", "crime", D(2026, 10, 13))], "fir_date_invalid_after_reference_date"),
    ([("acc", "crime", D(1975, 5, 5))], "fir_date_invalid_before_minimum_year"),
    ([("acc", "crime", "not-a-date")], "fir_date_invalid_unparseable"),
    ([("acc", "crime", "2024-13-45")], "fir_date_invalid_impossible_date"),
    ([("acc", None, None)], "accused_row_without_fir"),
])
def test_a_missing_or_invalid_fir_date_makes_the_age_unavailable(fir_rows, reason):
    c = V3CONFIG()
    a = PersonAttributes("a", "V2", "ra", age_info=build_age_evidence(None, 30, fir_rows, c))
    assert a.age_info.merged_range is None and a.age_info.observation_status == "none_usable" and a.age_info.unavailable_reasons() == [reason]
    b = person(c, 30, D(2024, 6, 1), pid="b")
    r = cmp(c, a, b)
    assert r.applicable is False and r.score is None and r.conflict is None
    assert r.reason_not_applicable.startswith("age_evidence_unavailable:") and reason in r.reason_not_applicable
    assert "not compared" in r.explanation


def test_the_current_date_is_never_a_substitute_for_a_missing_fir_date():
    # whatever the reference date, a person with an age and no FIR date gets no birth-year estimate
    for reference in ("2026-10-12", "2019-01-01"):
        c = V3CONFIG(reference_date=reference)
        e = build_age_evidence(None, 30, None, c)
        assert e.merged_range is None and e.observations[0].year_range is None and e.observations[0].fir_date is None


def test_an_invalid_age_is_unavailable_with_its_own_reason():
    c = V3CONFIG()
    for raw, why in ((None, "age_missing"), (3, "age_invalid:out_of_range"), (150, "age_invalid:out_of_range"), ("abc", "age_invalid:not_a_number")):
        e = build_age_evidence(None, raw, [("a", "c", D(2024, 1, 1))], c)
        assert e.merged_range is None and e.unavailable_reasons() == [why]


def test_the_reference_date_only_bounds_the_fir_date():
    c = V3CONFIG(reference_date="2026-10-12")
    assert parse_fir_date(D(2026, 10, 12), D(2026, 10, 12), c) == (D(2026, 10, 12), "valid")
    assert parse_fir_date(D(2026, 10, 13), D(2026, 10, 12), c)[1] == "invalid:after_reference_date"
    stamp = datetime(2024, 3, 1, 22, 0, tzinfo=timezone.utc)
    assert parse_fir_date(stamp, D(2026, 10, 12), c) == (D(2024, 3, 1), "valid")  # the calendar date as given; zone conversion happens in the query


# ------------------------------------------------------------ several FIRs for one record: observations, never newest/oldest

def test_observations_keep_individual_provenance():
    c = V3CONFIG()
    rows = [("acc-1", "crime-1", D(2020, 5, 1)), ("acc-2", "crime-2", D(2021, 5, 1))]
    e = build_age_evidence(None, 30, rows, c)
    assert [(o.accused_id, o.crime_id, o.fir_date, o.age) for o in e.observations] == [("acc-1", "crime-1", D(2020, 5, 1), 30), ("acc-2", "crime-2", D(2021, 5, 1), 30)]
    assert all(o.year_range for o in e.observations) and e.observation_status in ("consistent", "within_tolerance")
    dicts = [o.to_dict() for o in e.observations]
    assert all(d["age_source"] == "persons_unified.age" and d["link_source"].startswith("accused_unified") and d["fir_date"] for d in dicts)


def test_ranges_that_agree_are_intersected_and_close_ones_use_the_widest_range():
    c = V3CONFIG()
    close = build_age_evidence(None, 30, [("a1", "c1", D(2020, 6, 1)), ("a2", "c2", D(2021, 6, 1))], c)
    assert close.observation_status == "consistent" and close.merged_range == (1990, 1990) and not close.flags
    near = build_age_evidence(None, 30, [("a1", "c1", D(2020, 6, 1)), ("a2", "c2", D(2022, 6, 1))], c)
    assert near.observation_status == "within_tolerance" and near.merged_range == (1989, 1992)
    assert [f["issue"] for f in near.flags] == ["age_observations_differ_within_tolerance"]


def test_contradicting_observations_are_flagged_and_not_resolved_by_picking_one():
    c = V3CONFIG()
    rows = [("a1", "c1", D(2016, 6, 1)), ("a2", "c2", D(2026, 6, 1))]  # one age 30 recorded against FIRs ten years apart
    e = build_age_evidence(None, 30, rows, c)
    assert e.observation_status == "conflicting" and e.conflicting and e.merged_range is None  # neither the newest nor the oldest was chosen
    assert [f["issue"] for f in e.flags] == ["conflicting_age_observations"]
    assert all(o.usable and o.year_range for o in e.observations)  # both observations are kept with their ranges
    # the outcome does not depend on the order the database returned the rows in
    for order in itertools.permutations(rows):
        again = build_age_evidence(None, 30, list(order), c)
        assert (again.observation_status, again.merged_range, [f["issue"] for f in again.flags]) == (e.observation_status, None, ["conflicting_age_observations"])
    a, b = PersonAttributes("a", "V2", "ra", age_info=e), person(c, 30, D(2024, 6, 1), pid="b")
    r = cmp(c, a, b)
    assert r.applicable is False and r.reason_not_applicable.startswith("age_observations_conflict:")
    assert [f["issue"] for f in r.detail["review_flags"]] == ["conflicting_age_observations"] and r.detail["review_flags"][0]["record"] == "a"


def test_hull_policy_compares_the_widest_range_but_still_flags_the_record():
    c = V3CONFIG(observation_conflict_policy="hull")
    rows = [("a1", "c1", D(2016, 6, 1)), ("a2", "c2", D(2026, 6, 1))]
    a = PersonAttributes("a", "V2", "ra", age_info=build_age_evidence(None, 30, rows, c))
    assert a.age_info.merged_range == (1985, 1996)
    r = cmp(c, a, person(c, 30, D(2024, 6, 1), pid="b"))
    assert r.applicable and r.conflict is None and r.detail["review_flags"][0]["issue"] == "conflicting_age_observations"


def test_one_usable_and_one_unusable_observation_use_the_usable_one_and_say_so():
    c = V3CONFIG()
    e = build_age_evidence(None, 30, [("a1", "c1", D(2024, 6, 1)), ("a2", "c2", None)], c)
    assert e.observation_status == "single" and e.merged_range == (1993, 1994) and e.unavailable_reasons() == ["fir_date_missing"]


# ------------------------------------------------ DOB comparisons are kept apart from age estimates

def test_dob_against_dob_is_exactly_the_score_v2_comparison():
    v2, c = V2CONFIG(), V3CONFIG()
    pairs = [(D(1996, 3, 3), D(1996, 3, 3)), (D(1996, 3, 3), D(1996, 3, 4)), (D(1996, 3, 3), D(1996, 5, 3)), (D(1996, 3, 3), D(1990, 1, 1)), (D(1996, 3, 4), D(1996, 4, 3))]
    for da, db in pairs:
        new = cmp(c, person(c, None, None, dob=da, pid="a"), person(c, None, None, dob=db, pid="b"))
        old = compare_age_dob(build_attributes_v2(v2, None, da), build_attributes_v2(v2, None, db), v2)
        assert (new.score, new.conflict, new.detail["method"], new.detail["status"]) == (old.score, old.conflict, old.detail["method"], old.detail["status"])


def test_dob_against_age_uses_the_age_the_dob_implies_at_the_fir_date():
    c = V3CONFIG()
    dob = person(c, None, None, dob=D(1996, 3, 3), pid="a")
    aged = person(c, 26, D(2022, 3, 1), pid="b")  # the DOB implies 25 on 1 Mar 2022 (birthday is 3 Mar)
    r = cmp(c, dob, aged)
    assert r.detail["method"] == "dob_vs_recorded_age_at_fir_date" and r.detail["gap_years"] == 1 and r.conflict is None
    assert r.detail["per_observation"] == [{"fir_year": 2022, "recorded_age": 26, "age_implied_by_dob_at_fir_date": 25, "gap_years": 1}]
    assert r.score == pytest.approx(0.85 * c.dob_vs_age_max_score)
    assert cmp(c, aged, dob).detail["dob_side"] == "b"
    # score-v2 reads the same recorded age at today's date and reports a 4-year gap
    v2 = V2CONFIG()
    old = compare_age_dob(build_attributes_v2(v2, None, D(1996, 3, 3)), build_attributes_v2(v2, 26), v2)
    assert old.detail["gap_years"] == 4 and old.conflict == "soft"


def test_dob_against_age_with_no_fir_date_is_unavailable_not_estimated_at_today():
    c = V3CONFIG()
    r = cmp(c, person(c, None, None, dob=D(1996, 3, 3), pid="a"), person(c, 26, None, pid="b"))
    assert r.applicable is False and "no_fir_link" in r.reason_not_applicable


def test_dob_against_several_consistent_observations_uses_the_worst_fit_and_lists_all():
    c = V3CONFIG()
    other = PersonAttributes("b", "V2", "rb", age_info=build_age_evidence(None, 30, [("a1", "c1", D(2020, 6, 1)), ("a2", "c2", D(2022, 6, 1))], c))
    r = cmp(c, person(c, None, None, dob=D(1990, 3, 3), pid="a"), other)
    assert len(r.detail["per_observation"]) == 2 and r.detail["gap_years"] == max(p["gap_years"] for p in r.detail["per_observation"])


def test_a_records_own_age_and_dob_are_checked_at_the_fir_date_not_today():
    c = V3CONFIG()
    ok = build_age_evidence(D(1996, 3, 3), 26, [("a", "c", D(2022, 3, 1))], c)  # DOB implies 25 at the FIR; recorded 26: within 1
    assert ok.quality == () and not ok.inconsistent
    bad = build_age_evidence(D(1996, 3, 3), 30, [("a", "c", D(2022, 3, 1))], c)  # recorded 30 vs 25 at the FIR date
    assert bad.inconsistent and bad.quality[0]["age_implied_by_dob"] == 25 and bad.quality[0]["fir_date"] == "2022-03-01"
    r = cmp(c, PersonAttributes("a", "V2", "ra", age_info=bad), person(c, None, None, dob=D(1996, 3, 3), pid="b"))
    assert r.detail["method"] == "dob_exact" and r.conflict is None  # the DOB is what is compared; the contradiction is only reported
    assert r.detail["data_quality"][0]["record"] == "a"


def test_a_fir_dated_before_the_records_own_dob_is_not_used():
    c = V3CONFIG()
    e = build_age_evidence(D(2000, 1, 1), 22, [("a", "c", D(1995, 1, 1))], c)
    assert e.merged_range is None and e.unavailable_reasons() == ["fir_date_before_dob"] and e.flags[0]["issue"] == "fir_date_before_own_dob"


def test_a_dob_alone_needs_no_fir_date():
    c = V3CONFIG()
    r = cmp(c, person(c, 30, None, dob=D(1996, 3, 3), pid="a"), person(c, 30, None, dob=D(1996, 3, 3), pid="b"))
    assert r.applicable and r.score == 1.0 and r.detail["method"] == "dob_exact"


# --------------------------------------------------------------------------------- configuration and versioning

def test_score_v3_configuration_is_validated_and_documented_in_the_serialised_form():
    c = V3CONFIG()
    d = c.to_dict()
    assert d["rule_version"] == "score-v3" and all(k in d for k in V3_ONLY_FIELDS) and d["fir_timezone"] == "Asia/Kolkata"
    assert ScoringConfig.from_dict(d) == c and ScoringConfig.from_dict(d).signature() == c.signature()
    with pytest.raises(ValueError):
        ScoringConfig(rule_version="score-v3")  # a reference date is required (validation cut-off), never read from the clock
    with pytest.raises(ValueError):
        V3CONFIG(observation_conflict_policy="newest")
    with pytest.raises(ValueError):
        V3CONFIG(birth_year_tolerance_years=5, birth_year_conflict_gap_years=2)


def test_v1_and_v2_configurations_do_not_change():
    for config in (V1CONFIG(), V2CONFIG()):
        assert not any(k in config.to_dict() for k in V3_ONLY_FIELDS)
        assert ScoringConfig.from_dict(config.to_dict()) == config


def test_the_dispatch_keeps_each_version_on_its_own_logic():
    v1, v2, v3 = V1CONFIG(), V2CONFIG(), V3CONFIG()
    a1 = PersonAttributes("a", "V2", "ra", age=30)
    assert score_age_dob(a1, a1, v1).detail == {}  # v1: no detail block
    a2 = build_attributes_v2(v2, 30)
    assert score_age_dob(a2, a2, v2).detail["method"] == "estimated_birth_year_ranges"
    a3 = person(v3, 30, D(2024, 6, 1))
    assert score_age_dob(a3, a3, v3).detail["method"] == "estimated_birth_year_ranges_at_fir_date"
    with pytest.raises(ValueError):
        score_age_dob(a2, a2, v3)  # score-v2 attributes cannot be scored as score-v3 (no FIR anchoring) - never silently mixed
