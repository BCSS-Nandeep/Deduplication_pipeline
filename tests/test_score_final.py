"""score-final: authoritative weights, shared name rules, face threshold, address components, candidate signals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dedup_pipeline.candidates import CandidateConfig, RunOptions, run_candidate_generation
from dedup_pipeline.candidates.config import ALL_STRATEGIES, final_candidate_config
from dedup_pipeline.face_recognition.pair_evidence import PairFaceEvidence, PairStatus
from dedup_pipeline.identity.names import compare_identity_names, normalize_identity_name
from dedup_pipeline.scoring.attributes import Address, PersonAttributes, build_attributes
from dedup_pipeline.scoring.config import (
    ADDRESS_COMPONENT_WEIGHTS, DEFAULT_WEIGHTS, FINAL_WEIGHTS, RULE_VERSION, ScoringConfig,
)
from dedup_pipeline.scoring.face import face_field_score
from dedup_pipeline.scoring.fields import FieldResult, score_address, score_age_dob, score_alias, score_gender, score_phone
from dedup_pipeline.scoring.score import combine, score_pair
from tests.candidates_fakes import FakePersonDb, person

FINAL = ScoringConfig(rule_version="score-final")


def attrs(pid="a", **kw) -> PersonAttributes:
    return PersonAttributes(person_id=pid, source_system="V2", source_record_id=f"r-{pid}", **kw)


def named(pid, name, relative="", phone=None, gender="Male", addresses=()):
    row = (pid, "V2", f"r-{pid}", name, None, relative, phone, 30, None, gender, None)
    built = build_attributes(row, FINAL, CandidateConfig())
    if addresses:
        return PersonAttributes(**{**built.__dict__, "addresses": tuple(addresses)})
    return built


def face(status, similarity=None, **kw):
    return PairFaceEvidence("a", "b", status, face_similarity=similarity,
                            similarity_metric="cosine" if similarity is not None else None, model="buffalo_l", **kw)


def cosine_for(mapped: float) -> float:
    return 0.20 + mapped * (0.58 - 0.20)


# ------------------------------------------------------------------- weights

def test_authoritative_weights_sum_to_100_and_historical_default_is_unchanged():
    assert RULE_VERSION == "score-v2"
    active = {name: weight for name, weight in FINAL_WEIGHTS.items() if weight}
    assert active == {"name": 0.25, "face": 0.25, "relative_name": 0.20, "phone": 0.15, "address": 0.10, "gender": 0.05}
    assert sum(active.values()) == pytest.approx(1.0)
    assert sum(FINAL.weights.values()) == pytest.approx(1.0)
    assert FINAL.weights["alias"] == 0 and FINAL.weights["age_dob"] == 0
    assert sum(ADDRESS_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert ADDRESS_COMPONENT_WEIGHTS["village"] == 0.35
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    assert "face_pass_threshold" not in ScoringConfig(rule_version="score-v1").to_dict()
    assert ScoringConfig(rule_version="score-final").face_pass_threshold == 0.85


def test_formula_with_face_available_and_with_face_unavailable():
    weights = FINAL.weights

    def field(name, score, applicable=True):
        return FieldResult(name, weights[name], applicable, score if applicable else None, name)

    both = {name: field(name, score) for name, score in {
        "name": 0.90, "face": 0.92, "relative_name": 1.0, "phone": 1.0, "address": 0.80, "gender": 1.0,
    }.items()}
    both["alias"] = field("alias", None, False)
    both["age_dob"] = field("age_dob", None, False)
    full = combine(both, FINAL)
    assert full.match_score == pytest.approx(0.935)
    assert full.coverage == pytest.approx(1.0)
    assert round(full.match_score * 100, 2) == 93.5

    missing_face = dict(both)
    missing_face["face"] = field("face", None, False)
    bare = combine(missing_face, FINAL)
    assert bare.match_score == pytest.approx(0.94)
    assert bare.coverage == pytest.approx(0.75)
    assert round(bare.match_score * 100, 2) == 94.0

    none = {name: field(name, None, False) for name in weights}
    empty = combine(none, FINAL)
    assert empty.match_score is None and empty.insufficient and empty.coverage == 0


def test_invalid_score_is_rejected_and_zero_division_is_not_attempted():
    bad = {name: FieldResult(name, FINAL.weights[name], True, 1.2, name) for name in FINAL.weights}
    with pytest.raises(ValueError):
        combine(bad, FINAL)


# ------------------------------------------------------------------- names

@pytest.mark.parametrize("raw,cleaned", [
    ("Darla Ramesh Uganda", "darla ramesh"),
    ("Darla Ramesh Peddler", "darla ramesh"),
    ("Late Darla Ramesh", "darla ramesh"),
    ("Late Suresh Kumar", "suresh kumar"),
    ("Suresh Kumar Uganda", "suresh kumar"),
    ("  Darla,   Ramesh ", "darla ramesh"),
])
def test_metadata_cleaning(raw, cleaned):
    result = normalize_identity_name(raw)
    assert result.comparison == cleaned
    if "late" in raw.lower() or "uganda" in raw.lower() or "peddler" in raw.lower():
        assert result.removed
    again = normalize_identity_name(result.display)
    assert again.comparison == cleaned and again.removed == ()


def test_country_or_role_inside_a_name_is_kept_and_blank_input_is_blank():
    kept = normalize_identity_name("Uganda Kumar Reddy")
    assert kept.comparison == "uganda kumar reddy" and kept.removed == ()
    middle = normalize_identity_name("Darla Peddler Ramesh")
    assert middle.comparison == "darla peddler ramesh"
    assert normalize_identity_name(None).status == "blank"
    assert normalize_identity_name("   ").status == "blank"
    assert normalize_identity_name("Late").comparison == "late"


@pytest.mark.parametrize("left,right,method,score", [
    ("Darla Ramesh", "Darla Ramesh", "exact", 1.0),
    ("Darla Ramesh", "Ramesh Darla", "token_order", 1.0),
    ("Late Suresh Kumar", "Suresh Kumar", "exact", 1.0),
    ("Suresh Kumar", "Kumar Suresh", "token_order", 1.0),
])
def test_exact_and_token_order_names(left, right, method, score):
    compared = compare_identity_names(normalize_identity_name(left), normalize_identity_name(right))
    assert compared.method == method and compared.score == score and compared.status == "matching"


def test_initials_are_compatible_but_never_a_full_score():
    initial = compare_identity_names(normalize_identity_name("Darla Ramesh"), normalize_identity_name("D. Ramesh"))
    father = compare_identity_names(normalize_identity_name("Suresh Kumar"), normalize_identity_name("S. Kumar"))
    assert initial.used_initial and initial.score == pytest.approx(0.75) and initial.score < 1
    assert father.used_initial and father.score <= 0.75
    lone = compare_identity_names(normalize_identity_name("S"), normalize_identity_name("Suresh Kumar"))
    assert lone.score < 0.5
    partial = compare_identity_names(normalize_identity_name("Ramesh Kumar"), normalize_identity_name("Suresh Kumar"))
    assert partial.status == "partial" and partial.score < 1 and not partial.used_initial


# ------------------------------------------------------------------- face, phone, gender, address

def test_face_threshold_is_strictly_above_85_percent_of_the_mapped_score():
    passed = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, cosine_for(0.92)), FINAL)
    exact = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, cosine_for(0.85)), FINAL)
    below = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, 0.30), FINAL)
    assert passed.fields["face"].applicable and passed.fields["face"].score == pytest.approx(0.92)
    assert passed.fields["face"].detail["threshold_outcome"] == "passed"
    assert exact.fields["face"].applicable and exact.fields["face"].score == 0.0
    assert exact.fields["face"].detail["threshold_outcome"] == "failed"
    assert below.fields["face"].score == 0.0 and below.fields["face"].applicable
    assert below.coverage == pytest.approx(0.25)  # the failed comparison keeps its weight
    assert round(face_field_score(cosine_for(0.85), FINAL), 4) == pytest.approx(0.85)


@pytest.mark.parametrize("status", [
    PairStatus.SKIPPED_NO_IMAGE, PairStatus.NO_FACE, PairStatus.LOW_QUALITY, PairStatus.INVALID_IMAGE, PairStatus.MULTIPLE_FACES,
])
def test_missing_or_unusable_face_is_excluded_not_zero(status):
    outcome = score_pair(attrs("a"), attrs("b"), face(status), FINAL)
    assert outcome.fields["face"].score is None and not outcome.fields["face"].applicable
    assert outcome.match_score is None and outcome.insufficient


def test_face_service_failure_and_invalid_score_are_technical_and_excluded():
    failed = score_pair(attrs("a"), attrs("b"), face(PairStatus.ERROR, error_kind="unavailable"), FINAL)
    invalid = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, 1.5), FINAL)
    assert failed.fields["face"].detail["status"] == "technical_error"
    assert invalid.fields["face"].reason_not_applicable == "invalid_face_score"
    assert failed.fields["face"].score is None and invalid.fields["face"].score is None


def test_one_sided_and_absent_images_do_not_penalise_the_score():
    for evidence in (None, face(PairStatus.SKIPPED_NO_IMAGE)):
        outcome = score_pair(attrs("a"), attrs("b"), evidence, FINAL)
        assert not outcome.fields["face"].applicable and outcome.fields["face"].score is None


def test_phone_gender_and_unused_signals():
    same = named("a", "Darla Ramesh", phone="9876543211", gender="Male")
    other = named("b", "Darla Ramesh", phone="+91 98765 43211", gender="Male")
    conflict = named("c", "Darla Ramesh", phone="9123456780", gender="Female")
    junk = named("d", "Darla Ramesh", phone="9999999999", gender=None)
    assert score_phone(same, other, FINAL).score == 1.0
    assert score_phone(same, conflict, FINAL).score == 0.0 and score_phone(same, conflict, FINAL).applicable
    assert not score_phone(same, junk, FINAL).applicable
    assert score_gender(same, other, FINAL).score == 1.0
    assert score_gender(same, conflict, FINAL).score == 0.0
    assert not score_gender(same, junk, FINAL).applicable
    assert not score_alias(same, other, FINAL).applicable
    assert not score_age_dob(same, other, FINAL).applicable
    pair = score_pair(same, other, None, FINAL)
    assert pair.fields["alias"].reason_not_applicable == "not_in_score_final"
    assert pair.fields["age_dob"].reason_not_applicable == "not_in_score_final"
    assert pair.fields["face"].score is None and pair.fields["phone"].score == 1.0


def test_address_components_normalize_and_ignore_street():
    present = Address("telangana", "hyderabad", "shaikpet", "shaikpet", "500008", "india", "present")
    alias = Address("ts", "hyderabad", "shaikpet", "shaikpet", "500008", "bharat", "present")
    conflict = Address("andhra pradesh", "guntur", "tenali", "tenali", "522201", "india", "present")
    village_only = Address(village="shaikpet", kind="present")
    other_village = Address(village="shaikpet", kind="present")
    matched = score_address(attrs(addresses=(present,)), attrs(addresses=(alias,)), FINAL)
    assert matched.score == pytest.approx(1.0)
    assert matched.detail["component_scores"]["state"] == 1.0
    assert matched.detail["component_scores"]["country"] == 1.0
    differed = score_address(attrs(addresses=(present,)), attrs(addresses=(conflict,)), FINAL)
    assert differed.score < 1 and differed.detail["component_scores"]["district"] == 0.0
    partial = score_address(attrs(addresses=(village_only,)), attrs(addresses=(other_village,)), FINAL)
    assert partial.score == pytest.approx(1.0)
    assert partial.detail["available_component_weight"] == pytest.approx(0.35)
    assert partial.detail["address_evidence_complete"] is False
    assert partial.detail["status"] == "partial"
    assert partial.detail["components_compared"] == ["village"]
    district_only = Address(district="hyderabad", kind="present")
    other_district = Address(district="hyderabad", kind="present")
    thin = score_address(attrs(addresses=(district_only,)), attrs(addresses=(other_district,)), FINAL)
    assert thin.applicable and thin.score == pytest.approx(1.0)
    assert thin.detail["address_evidence_complete"] is False
    assert thin.detail["components_compared"] == ["district"]
    assert "village" in thin.detail["components_unavailable"]
    assert "not complete address evidence" in thin.explanation
    mismatch = score_address(
        attrs(addresses=(district_only,)),
        attrs(addresses=(Address(district="guntur", kind="present"),)),
        FINAL,
    )
    assert mismatch.applicable and mismatch.score == 0.0
    assert mismatch.detail["status"] == "conflicting"
    assert mismatch.reason_not_applicable is None
    row = (
        "p", "V2", "r", "Darla Ramesh", None, "Suresh Kumar", None, 30, None, "Male",
        {"present": {"state": "Telangana", "district": "Hyderabad", "mandal": "Shaikpet", "country": "India",
                     "house_number": "12", "street": "Main Road", "locality": "Bazaar", "path": "XX"},
         "permanent": {"state": "Andhra Pradesh", "district": "Guntur", "country": "India"}},
        "house 12, hyderabad 500008", None,
    )
    parsed = build_attributes(row, FINAL, CandidateConfig())
    assert parsed.addresses[0].pin == "500008"
    assert parsed.addresses[0].kind == "present"
    same_geo = build_attributes(
        ("q", "V2", "r2", "Darla Ramesh", None, "Suresh Kumar", None, 30, None, "Male",
         {"present": {"state": "Telangana", "district": "Hyderabad", "mandal": "Shaikpet", "country": "India",
                      "house_number": "99", "street": "Other Street"}},
         "somewhere else 500008", None),
        FINAL, CandidateConfig(),
    )
    agreement = score_address(parsed, same_geo, FINAL)
    assert agreement.score == pytest.approx(1.0)
    assert "house_number" in agreement.detail["ignored"]
    across = score_address(
        attrs(addresses=(Address("telangana", "hyderabad", "", "", "", "india", "present"),)),
        attrs(addresses=(Address("telangana", "hyderabad", "", "", "", "india", "permanent"),)),
        FINAL,
    )
    assert across.detail["across_address_types"] is True and across.applicable


def test_face_map_anchors_and_out_of_range_cosine_cannot_pass_by_clamping():
    assert face_field_score(0.20, FINAL) == 0.0
    assert face_field_score(0.58, FINAL) == 1.0
    assert face_field_score(1.5, FINAL) == 1.0  # the raw mapper clamps; the final rule must not use that as a match
    invalid = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, 1.5), FINAL)
    assert invalid.fields["face"].applicable is False
    assert invalid.fields["face"].detail["status"] == "technical_error"
    assert invalid.fields["face"].score is None
    low = score_pair(attrs("a"), attrs("b"), face(PairStatus.COMPARED, -1.0), FINAL)
    assert low.fields["face"].applicable and low.fields["face"].score == 0.0
    assert low.fields["face"].detail["mapped_score"] == 0.0


def test_full_evidence_mismatches_and_insufficient_evidence_stay_separate():
    present = Address("telangana", "hyderabad", "shaikpet", "shaikpet", "500008", "india", "present")
    other = Address("telangana", "hyderabad", "shaikpet", "shaikpet", "500008", "india", "present")
    left = named("a", "Darla Ramesh", "Suresh Kumar", "9876543211", addresses=(present,))
    right = named("b", "Darla Ramesh", "Suresh Kumar", "9876543211", addresses=(other,))
    full = score_pair(left, right, face(PairStatus.COMPARED, cosine_for(0.92)), FINAL)
    assert full.match_score == pytest.approx(0.98)  # name 1, face 0.92, relative 1, phone 1, address 1, gender 1
    assert full.coverage == pytest.approx(1.0)
    assert full.insufficient is False
    assert full.fields["address"].detail["address_evidence_complete"] is True

    no_face = score_pair(left, right, face(PairStatus.SKIPPED_NO_IMAGE), FINAL)
    assert no_face.fields["face"].applicable is False
    assert no_face.coverage == pytest.approx(0.75)
    assert no_face.fields["face"].reason_not_applicable == "skipped_no_image"

    phone_conflict = named("c", "Darla Ramesh", "Suresh Kumar", "9123456780", addresses=(present,))
    conflicted = score_pair(left, phone_conflict, None, FINAL)
    assert conflicted.fields["phone"].applicable and conflicted.fields["phone"].score == 0.0
    assert conflicted.fields["address"].applicable and conflicted.fields["address"].score == 1.0

    blank = score_pair(attrs("a"), attrs("b"), None, FINAL)
    assert blank.match_score is None and blank.insufficient and blank.coverage == 0
    assert blank.fields["alias"].reason_not_applicable == "not_in_score_final"
    assert blank.fields["age_dob"].reason_not_applicable == "not_in_score_final"


def test_district_only_address_is_not_listed_as_a_complete_match():
    from dedup_pipeline.scoring.score import build_record
    left = attrs("a", addresses=(Address(district="hyderabad", kind="present"),), name="Darla Ramesh",
                 relative_name="Suresh Kumar", phone="9876543211", gender="Male")
    right = attrs("b", addresses=(Address(district="hyderabad", kind="present"),), name="Darla Ramesh",
                  relative_name="Suresh Kumar", phone="9876543211", gender="Male")
    outcome = score_pair(left, right, None, FINAL)
    record = build_record(
        {"candidate_id": "a:b", "person_a_id": "a", "person_b_id": "b", "source_a": "V2", "source_b": "V2",
         "source_pair": "V2-V2", "source_record_a": "ra", "source_record_b": "rb",
         "rule_names": ["name_exact"], "rules": [{"rule": "name_exact"}]},
        left, right, outcome, None, _image("a"), _image("b"), set(), FINAL, "http://files", [],
    )
    assert record["score_final"]["address_evidence"]["complete"] is False
    assert record["score_final"]["address_evidence"]["components_compared"] == ["district"]
    assert "address" not in {item["field"] for item in record["matching_attributes"]}
    assert record["review"]["status"] == "pending" and record["review"]["decision"] is None
    assert record["score_final"]["automatic_merge"] is False
    assert "address" not in {item["field"] for item in record["missing_evidence"]}


def test_retryable_face_failure_is_stored_but_not_reused_as_a_result(tmp_path):
    from dedup_pipeline.scoring.face import FaceCache, needs_retry
    cache = FaceCache(tmp_path / "face_evidence_cache.jsonl", {"engine": "parity-test"})
    failed = face(PairStatus.ERROR, error_kind="unavailable")
    compared = face(PairStatus.COMPARED, cosine_for(0.92))
    cache.put("c-error", "inputs-1", failed)
    cache.put("c-ok", "inputs-2", compared)
    assert needs_retry(failed)
    assert cache.get("c-error", "inputs-1") is None
    reused = cache.get("c-ok", "inputs-2")
    assert reused is not None and reused.status is PairStatus.COMPARED
    assert reused.face_similarity == pytest.approx(cosine_for(0.92))
    cache.close()
    reopened = FaceCache(tmp_path / "face_evidence_cache.jsonl", {"engine": "parity-test"})
    assert reopened.get("c-error", "inputs-1") is None
    assert reopened.get("c-ok", "inputs-2") is not None
    reopened.close()


def test_review_record_explains_the_score_and_does_not_merge(tmp_path):
    from dedup_pipeline.scoring.score import build_record
    left = named("a1", "Late Darla Ramesh Uganda", "Late Suresh Kumar", "9876543211")
    right = named("b1", "Ramesh Darla", "S. Kumar", "9876543211")
    outcome = score_pair(left, right, None, FINAL)
    record = build_record(
        {"candidate_id": "a1:b1", "person_a_id": "a1", "person_b_id": "b1", "source_a": "V2", "source_b": "V2",
         "source_pair": "V2-V2", "source_record_a": "r-a1", "source_record_b": "r-b1",
         "rule_names": ["relative_exact"], "rules": [{"rule": "relative_exact"}]},
        left, right, outcome, None, _image("a1"), _image("b1"), set(), FINAL, "http://files", [],
    )
    assert record["review"]["status"] == "pending" and record["review"]["decision"] is None
    assert record["score_final"]["automatic_merge"] is False
    assert record["score_final"]["name_normalization_a"]["comparison"] == "darla ramesh"
    assert record["evidence_coverage_percent"] < 100
    assert record["face"]["in_match_score"] is False
    assert not (tmp_path / "should-not-write-db").exists()


def _image(pid):
    from dedup_pipeline.face_recognition.image_source import PersonImage
    return PersonImage(pid, "no_usable_image", (), {})


# ------------------------------------------------------------------- candidates

def _run(people, tmp_path: Path, config, name="out"):
    out = tmp_path / name
    summary = run_candidate_generation(FakePersonDb(people, config), config, RunOptions(output_dir=out))
    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    deferred = [json.loads(line) for line in (out / "deferred_blocks.jsonl").read_text(encoding="utf-8").splitlines()]
    return summary, {(row["person_a_id"], row["person_b_id"]): row["rule_names"] for row in rows}, deferred


def test_final_profile_generates_the_required_signals_and_not_alias(tmp_path):
    config = final_candidate_config()
    assert "alias_name" not in config.enabled_strategies()
    assert not any(name.startswith("age") or "dob" in name or name == "alias_name" for name in config.enabled_strategies())
    people = [
        person("a1", name="Late Darla Ramesh Uganda", relative="Late Suresh Kumar", phone="9876543211"),
        person("a2", name="Ramesh Darla", relative="Suresh Kumar", phone="+91 98765 43211"),
        person("a3", name="D. Ramesh", relative="Other Father"),
        person("a4", name="Dinesh Ramesh", relative="Suresh Kumar"),
        person("a5", name="Ravi Teja", relative="Kumar Suresh", phone="9123456780"),
        person("a6", name=None, relative=None, phone="9876543211"),
    ]
    _, pairs, _ = _run(people, tmp_path, config)
    assert "name_exact" in pairs[("a1", "a2")] or "token_sorted_name" in pairs[("a1", "a2")]
    assert "token_sorted_name" in pairs[("a1", "a2")]
    assert "relative_exact" in pairs[("a1", "a2")]
    assert "phone_exact" in pairs[("a1", "a2")] and "phone_exact" in pairs[("a1", "a6")]
    assert "name_initial" in pairs[("a1", "a3")]
    assert ("a1", "a4") not in pairs or "name_initial" not in pairs[("a1", "a4")]
    assert "relative_token_sorted" in pairs[("a2", "a5")] or "relative_exact" in pairs[("a1", "a5")]
    assert all("alias_name" not in rules for rules in pairs.values())


def test_common_initials_are_deferred_instead_of_expanded(tmp_path):
    people = [person("i0", name="S Kumar")]
    people += [person(f"i{n}", name=f"Suresh{n} Kumar") for n in range(1, 20)]
    config = final_candidate_config(name_initial_max_block_size=15, name_max_block_size=30, token_sorted_max_block_size=30,
                                    relative_exact_enabled=False, relative_token_sorted_enabled=False)
    _, pairs, deferred = _run(people, tmp_path, config)
    assert any(block["strategy"] == "name_initial" and block["reason"] == "block_too_large" for block in deferred)
    assert not any("name_initial" in rules for rules in pairs.values())


def test_final_name_sql_strips_metadata_in_linear_steps():
    from dedup_pipeline.candidates.sql import block_query
    sql = block_query("name_exact", final_candidate_config()).sql
    legacy = block_query("name_exact", CandidateConfig()).sql
    assert "c6 AS MATERIALIZED" in sql and sql.count("AS MATERIALIZED") == 8
    # The raw step splits the name once. The second call is the existing token-count filter.
    assert sql.count("regexp_split_to_array") == 2
    assert "c6 AS" not in legacy and "AS MATERIALIZED" not in legacy


def test_legacy_profile_does_not_silently_clean_metadata_or_add_strategies(tmp_path):
    people = [person("a1", name="Late Darla Ramesh"), person("a2", name="Darla Ramesh")]
    _, pairs, _ = _run(people, tmp_path, CandidateConfig())
    assert pairs == {}
    assert "name_initial" not in ALL_STRATEGIES or "name_initial" not in CandidateConfig().enabled_strategies()
