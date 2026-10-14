"""Scoring maths: field scorers, normalisation over applicable evidence, coverage, face transformation, configuration."""

from __future__ import annotations

from datetime import date

import pytest

from dedup_pipeline.candidates.config import CandidateConfig
from dedup_pipeline.face_recognition.pair_evidence import PairFaceEvidence, PairStatus
from dedup_pipeline.scoring.attributes import Address, PersonAttributes, build_attributes
from dedup_pipeline.scoring.config import DEFAULT_WEIGHTS, FIELD_NAMES, ScoringConfig
from dedup_pipeline.scoring.face import face_field_score
from dedup_pipeline.scoring.fields import (
    score_address, score_age_dob, score_alias, score_gender, score_name, score_phone, score_relative_name,
)
from dedup_pipeline.scoring.score import combine, face_field, score_pair
from dedup_pipeline.scoring.similarity import name_similarity

from .scoring_fakes import V1CONFIG

CFG = V1CONFIG()


def attrs(pid="a", **kw) -> PersonAttributes:
    return PersonAttributes(person_id=pid, source_system="V2", source_record_id=f"r-{pid}", **kw)


def row(pid="p", name="Ramesh Kumar", alias=None, relative="Suresh Kumar", phone=None, age=30, dob=None, gender="Male", address=None):
    return (pid, "V2", f"r-{pid}", name, alias, relative, phone, age, dob, gender, address)


def evidence(status=PairStatus.NOT_MATCHED, similarity=None, **kw):
    return PairFaceEvidence("a", "b", status, face_similarity=similarity, similarity_metric="cosine" if similarity is not None else None, **kw)


# ----------------------------------------------------------------------------- names

@pytest.mark.parametrize(
    "a,b,low,high",
    [("ramesh kumar", "ramesh kumar", 1.0, 1.0), ("sai shiva alle", "alle sai shiva", 1.0, 1.0), ("ramesh kumar", "ramesh kumaar", 0.9, 1.0),
     ("r kumar", "ramesh kumar", 0.85, 0.99), ("mohd ali", "mohammed ali", 0.75, 0.99), ("ramesh kumar", "suresh babu", 0.0, 0.5),
     ("ramesh", "ramesh kumar", 0.6, 0.9)],
)
def test_name_similarity_ranges(a, b, low, high):
    score, method = name_similarity(a, b)
    assert low <= score <= high and method


def test_name_similarity_is_symmetric_and_blank_safe():
    assert name_similarity("ramesh kumar", "kumar r") == name_similarity("kumar r", "ramesh kumar")
    assert name_similarity("", "ramesh") is None and name_similarity("ramesh", "") is None


def test_name_and_relative_conflicts_are_hard_only_below_their_thresholds():
    assert score_name(attrs(name="ramesh kumar"), attrs("b", name="suresh babu"), CFG).conflict == "hard"
    assert score_name(attrs(name="ramesh kumar"), attrs("b", name="ramesh kumaar"), CFG).conflict is None
    assert score_relative_name(attrs(relative_name="venkat rao"), attrs("b", relative_name="lakshmi devi"), CFG).conflict == "hard"
    assert score_relative_name(attrs(relative_name="venkat rao"), attrs("b", relative_name="venkat rao"), CFG).score == 1.0


def test_missing_names_make_the_field_not_applicable_with_a_reason():
    result = score_relative_name(attrs(relative_name=""), attrs("b", relative_name="x y"), CFG)
    assert not result.applicable and result.score is None and result.reason_not_applicable == "relative_name_missing"


# ---------------------------------------------------------------------------- alias

def test_alias_is_informative_when_both_have_one_or_when_a_one_sided_alias_clearly_matches():
    both = score_alias(attrs(alias="raju bhai"), attrs("b", alias="raju bhai"), CFG)
    assert both.applicable and both.score == 1.0
    cross = score_alias(attrs(alias="raju bhai", name="ravi teja"), attrs("b", name="raju bhai"), CFG)
    assert cross.applicable and cross.score == 1.0
    silent = score_alias(attrs(alias="raju bhai", name="ravi teja"), attrs("b", name="somebody else"), CFG)
    assert not silent.applicable and silent.reason_not_applicable == "one_sided_alias_without_match"  # an unmatched optional alias proves nothing
    assert not score_alias(attrs(), attrs("b"), CFG).applicable


# ---------------------------------------------------------------------------- phone

def test_phone_same_different_and_unusable():
    cfg = CandidateConfig()
    same = score_phone(build_attributes(row("a", phone="9876543211"), CFG, cfg), build_attributes(row("b", phone="+91 98765 43211"), CFG, cfg), CFG)
    assert same.score == 1.0 and same.conflict is None and same.values == {"a": "******3211", "b": "******3211"}  # masked
    different = score_phone(build_attributes(row("a", phone="9876543211"), CFG, cfg), build_attributes(row("b", phone="9123456780"), CFG, cfg), CFG)
    assert different.applicable and different.score == 0.0 and different.conflict == "soft"
    junk = score_phone(build_attributes(row("a", phone="9999999999"), CFG, cfg), build_attributes(row("b", phone="9999999999"), CFG, cfg), CFG)
    assert not junk.applicable and "repetitive_digits" in junk.reason_not_applicable  # the same junk number is not evidence
    assert not score_phone(build_attributes(row("a", phone=None), CFG, cfg), build_attributes(row("b", phone="9876543211"), CFG, cfg), CFG).applicable


# ------------------------------------------------------------------------ age / DOB

@pytest.mark.parametrize("a,b,score,conflict", [(30, 30, 1.0, None), (30, 31, 0.9, None), (30, 33, 0.55, None), (30, 35, 0.2, None),
                                                 (30, 36, 0.0, "hard"), (20, 45, 0.0, "hard")])
def test_age_difference_scores_and_conflict(a, b, score, conflict):
    result = score_age_dob(attrs(age=a), attrs("b", age=b), CFG)
    assert (result.score, result.conflict) == (score, conflict)


def test_implausible_ages_are_not_compared():
    cfg = CandidateConfig()
    for bad in (0, 3, 150, 671, None):
        a = build_attributes(row("a", age=bad), CFG, cfg)
        assert a.age is None
        assert not score_age_dob(a, build_attributes(row("b", age=30), CFG, cfg), CFG).applicable


def test_dob_takes_precedence_and_handles_swapped_day_month():
    same = score_age_dob(attrs(dob=date(1990, 3, 4), age=30), attrs("b", dob=date(1990, 3, 4), age=60), CFG)
    assert same.score == 1.0 and same.conflict is None  # DOB agreement wins over a (junk) age gap
    swapped = score_age_dob(attrs(dob=date(1990, 3, 4)), attrs("b", dob=date(1990, 4, 3)), CFG)
    assert swapped.score == CFG.dob_swapped_day_month_score and swapped.conflict is None
    assert score_age_dob(attrs(dob=date(1990, 3, 4)), attrs("b", dob=date(1990, 7, 9)), CFG).conflict == "hard"
    assert score_age_dob(attrs(dob=date(1990, 3, 4)), attrs("b", dob=date(1985, 3, 4)), CFG).score == 0.0


# ------------------------------------------------------------------------ gender

def test_gender_only_compares_definite_values():
    assert score_gender(attrs(gender="Male"), attrs("b", gender="Male"), CFG).score == 1.0
    clash = score_gender(attrs(gender="Male"), attrs("b", gender="Female"), CFG)
    assert clash.score == 0.0 and clash.conflict == "hard"
    cfg = CandidateConfig()
    other = build_attributes(row("b", gender="Others"), CFG, cfg)
    assert other.gender is None and not score_gender(attrs(gender="Male"), other, CFG).applicable
    assert build_attributes(row("c", gender="  FEMALE "), CFG, cfg).gender == "Female"


# ----------------------------------------------------------------------- address

def addr(state="", district="", mandal=""):
    return (Address(state, district, mandal),)


@pytest.mark.parametrize(
    "a,b,score,conflict",
    [(addr("ts", "hyderabad", "x"), addr("ts", "hyderabad", "x"), 1.0, None), (addr("ts", "hyderabad", "x"), addr("ts", "hyderabad", "y"), 0.7, None),
     (addr("ts", "hyderabad"), addr("ts", "warangal"), 0.3, "soft"), (addr("ts", "hyderabad"), addr("ap", "guntur"), 0.0, "soft"),
     (addr("ts"), addr("ts", "warangal"), 0.3, None)],
)
def test_address_levels(a, b, score, conflict):
    result = score_address(attrs(addresses=a), attrs("b", addresses=b), CFG)
    assert (result.score, result.conflict) == (score, conflict)


def test_address_uses_the_best_present_or_permanent_combination_and_needs_data():
    a = (Address("ts", "hyderabad"), Address("ap", "guntur"))
    b = (Address("ap", "guntur"),)
    assert score_address(attrs(addresses=a), attrs("b", addresses=b), CFG).score == 0.7  # permanent matches permanent
    assert not score_address(attrs(addresses=()), attrs("b", addresses=b), CFG).applicable


# ------------------------------------------------------------- normalisation and coverage

def test_score_is_the_weighted_mean_over_applicable_fields_only():
    a = attrs(name="ramesh kumar", relative_name="suresh kumar", gender="Male", age=30)
    b = attrs("b", name="ramesh kumar", relative_name="suresh kumar", gender="Male", age=33)
    outcome = score_pair(a, b, None, CFG)
    # applicable: name .25 x 1, relative .15 x 1, age_dob .10 x .55, gender .05 x 1; alias/phone/address/face are not applicable
    expected = (0.25 * 1 + 0.15 * 1 + 0.10 * 0.55 + 0.05 * 1) / (0.25 + 0.15 + 0.10 + 0.05)
    assert outcome.match_score == pytest.approx(expected)
    assert outcome.coverage == pytest.approx((0.25 + 0.15 + 0.10 + 0.05) / 1.0)  # applicable weight / total configured weight
    assert outcome.applicable_fields == 4 and not outcome.insufficient


def test_every_field_applicable_gives_full_coverage_and_a_plain_weighted_mean():
    fields = {n: type("F", (), {"field": n, "applicable": True, "weight": DEFAULT_WEIGHTS[n], "score": 0.5, "conflict": None})() for n in FIELD_NAMES}
    outcome = combine(fields, CFG)
    assert outcome.match_score == pytest.approx(0.5) and outcome.coverage == pytest.approx(1.0)


def test_skipped_or_failed_face_is_excluded_from_both_numerator_and_denominator_never_zero():
    a = attrs(name="ramesh kumar", relative_name="suresh kumar")
    b = attrs("b", name="ramesh kumar", relative_name="suresh kumar")
    base = score_pair(a, b, None, CFG)
    for status in (PairStatus.SKIPPED_NO_IMAGE, PairStatus.SKIPPED_SHARED_IMAGE, PairStatus.NO_FACE, PairStatus.MULTIPLE_FACES,
                   PairStatus.LOW_QUALITY, PairStatus.INVALID_IMAGE, PairStatus.ERROR, PairStatus.NOT_CHECKED):
        outcome = score_pair(a, b, evidence(status, reason="why"), CFG)
        assert outcome.match_score == pytest.approx(base.match_score) and outcome.coverage == pytest.approx(base.coverage), status
        face = outcome.fields["face"]
        assert not face.applicable and face.score is None and face.reason_not_applicable == status.value and "excluded" in face.explanation


def test_face_evidence_enters_through_the_documented_transformation_and_changes_the_score():
    a = attrs(name="ramesh kumar", relative_name="suresh kumar")
    b = attrs("b", name="ramesh kumar", relative_name="suresh kumar")
    base = score_pair(a, b, None, CFG)
    high = score_pair(a, b, evidence(PairStatus.MATCHED, 0.70), CFG)
    low = score_pair(a, b, evidence(PairStatus.NOT_MATCHED, 0.05), CFG)
    assert high.match_score == pytest.approx(base.match_score) and high.coverage > base.coverage  # face agrees: score 1.0, more coverage
    assert low.match_score == pytest.approx((0.25 + 0.15 + 0.0) / (0.25 + 0.15 + 0.20))  # face disagrees: counted as 0 with its weight
    assert low.match_score < base.match_score
    assert low.fields["face"].conflict == "soft"
    middle = score_pair(a, b, evidence(PairStatus.NOT_MATCHED, 0.39), CFG).fields["face"]
    assert middle.score == pytest.approx((0.39 - 0.20) / (0.58 - 0.20))


@pytest.mark.parametrize("cosine,expected", [(-1.0, 0.0), (-0.3, 0.0), (0.0, 0.0), (0.20, 0.0), (0.39, 0.5), (0.58, 1.0), (0.9, 1.0), (1.0, 1.0)])
def test_face_transformation_is_piecewise_linear_and_bounded(cosine, expected):
    assert face_field_score(cosine, CFG) == pytest.approx(expected)


def test_raw_negative_cosine_is_kept_in_the_field_values_not_clamped_or_zeroed():
    field = face_field(evidence(PairStatus.NOT_MATCHED, -0.12), CFG)
    assert field.values["raw_cosine"] == -0.12 and field.score == 0.0 and "-0.120" in field.explanation


def test_switching_face_off_keeps_it_visible_but_out_of_the_score():
    cfg = V1CONFIG(face_in_score=False)
    a, b = attrs(name="ramesh kumar"), attrs("b", name="ramesh kumar")
    with_face = score_pair(a, b, evidence(PairStatus.MATCHED, 0.9), cfg)
    assert with_face.match_score == score_pair(a, b, None, cfg).match_score and not with_face.fields["face"].applicable
    assert with_face.fields["face"].values["raw_cosine"] == 0.9


def test_insufficient_evidence_by_coverage_or_field_count_and_band_E():
    only_name = score_pair(attrs(name="ramesh kumar"), attrs("b", name="ramesh kumar"), None, CFG)
    assert only_name.insufficient and only_name.band == "E" and only_name.match_score == 1.0  # a perfect score on thin evidence is flagged
    enough = score_pair(attrs(name="ramesh kumar", relative_name="a b", gender="Male"), attrs("b", name="ramesh kumar", relative_name="a b", gender="Male"), None, CFG)
    assert not enough.insufficient and enough.band == "A"
    none = score_pair(attrs(), attrs("b"), None, CFG)
    assert none.match_score is None and none.insufficient and none.band == "E"


@pytest.mark.parametrize("score,band", [(0.99, "A"), (0.85, "A"), (0.8499, "B"), (0.70, "B"), (0.55, "C"), (0.50, "C"), (0.2, "D")])
def test_bands_are_threshold_labels_only(score, band):
    fields = {n: type("F", (), {"field": n, "applicable": True, "weight": 1.0, "score": score, "conflict": None})() for n in FIELD_NAMES}
    cfg = V1CONFIG(weights={n: 1.0 for n in FIELD_NAMES})
    assert combine(fields, cfg).band == band


# ------------------------------------------------------------------------- configuration

def test_configuration_is_validated_versioned_and_round_trips():
    cfg = ScoringConfig.from_dict({"rule_version": "score-v1", "weights": {"name": 0.5}, "face_concurrency": 2})
    assert cfg.weights["name"] == 0.5 and cfg.weights["phone"] == DEFAULT_WEIGHTS["phone"] and cfg.rule_version == "score-v1"
    assert ScoringConfig.from_dict(cfg.to_dict()).signature() == cfg.signature()
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0) and set(DEFAULT_WEIGHTS) == set(FIELD_NAMES)
    for bad in ({"weights": {"name": -1}}, {"face_different_anchor": 0.7}, {"face_concurrency": 0}, {"min_coverage": 2}, {"bands": [["A", 0.5]]},
                {"nonsense": 1}, {"weights": {"unknown_field": 1}}):
        with pytest.raises((ValueError, TypeError)):
            ScoringConfig.from_dict({"rule_version": "score-v1", **bad})
    assert V1CONFIG(face_concurrency=8).face_signature() == V1CONFIG().face_signature()  # concurrency never invalidates stored faces
    assert V1CONFIG(weights={**DEFAULT_WEIGHTS, "name": 0.9}).face_signature() == V1CONFIG().face_signature()  # nor do weights
