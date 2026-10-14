"""score-v2 end to end: review groups, age/DOB in the report, image selection and fallback, versioning and compatibility."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from dedup_pipeline.face_recognition import FrUnavailable
from dedup_pipeline.image_resolution.models import Status
from dedup_pipeline.scoring import ScoringOptions, run_scoring
from dedup_pipeline.scoring.face import ImageOutcomeMemory, measured_detection_rates
from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V1, WORKSHEET_COLUMNS_V2
from dedup_pipeline.scoring.worksheet import validate_worksheet

from .conftest import url_for
from .scoring_fakes import (
    FakeComparer, V1CONFIG, V2CONFIG, backend, build_world, outcome, pdict, rejection, seed_known_image_failure,
)

DOC, PHOTO = "person/IDENTITY_DETAILS", "person/MEDIA"


def go(world, tmp_path, comparer=None, config=None, label=None, root=None, verify=None, db=None):
    root = root or tmp_path / "scoring"
    summary = run_scoring(db or world.db(), config or V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, label, True),
                          backend(world, comparer, verify))
    return summary, root


def records(root: Path, summary: dict) -> dict[str, dict]:
    path = root / "runs" / summary["run_id"] / "scored_candidates.jsonl"
    return {r["candidate_id"]: r for r in map(json.loads, path.read_text(encoding="ascii").splitlines())}


def rows_of(root: Path, summary: dict) -> list[dict]:
    with open(root / "runs" / summary["run_id"] / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run_dir(root: Path, summary: dict) -> Path:
    return root / "runs" / summary["run_id"]


# ----------------------------------------------------------------------------- review order

def ordering_people():
    same = dict(relative="Raju Rao", phone=None, gender="Male")
    return [
        pdict("c1", name="Anil Kumar", age=30, phone="9876500001", **{k: v for k, v in same.items() if k != "phone"}),
        pdict("c2", name="Anil Kumar", age=34, phone="9876500001", **{k: v for k, v in same.items() if k != "phone"}),  # age gap 4: soft only
        pdict("h1", name="Bala Reddy", age=30, phone="9876500002", relative="Raju Rao", gender="Male"),
        pdict("h2", name="Bala Reddy", age=30, phone="9876500002", relative="Raju Rao", gender="Female"),  # perfect except gender
        pdict("i1", name="Chandu Rao", age=None, relative=None, gender=None),
        pdict("i2", name="Chandu Rao", age=None, relative=None, gender=None),
        pdict("t1", name="Dinesh Babu", age=30, relative="Raju Rao"),
        pdict("t2", name="Dinesh Babu", age=30, relative="Raju Rao"),
    ]


@pytest.fixture
def ordering_world(tmp_path):
    return build_world(tmp_path / "w", ordering_people(), {"t1": [("tx1", 1)], "t2": [("tx2", 2)]})


def test_conflict_free_candidates_come_first_then_hard_conflicts_then_insufficient_then_technical_failures(ordering_world, tmp_path):
    comparer = FakeComparer(ordering_world, lambda a, b: FrUnavailable("down"))
    summary, root = go(ordering_world, tmp_path, comparer)
    rows = rows_of(root, summary)
    assert [r["candidate_id"] for r in rows] == ["c1:c2", "h1:h2", "i1:i2", "t1:t2"]
    assert [r["review_group"] for r in rows] == ["1_no_hard_conflict", "2_hard_conflict", "3_insufficient_evidence", "4_technical_failure"]
    recs = records(root, summary)
    assert recs["h1:h2"]["match_score_percent"] > recs["c1:c2"]["match_score_percent"]  # the conflicting pair scores HIGHER...
    assert [r["review_order"] for r in rows] == ["1", "2", "3", "4"]  # ...but is reviewed after the conflict-free one
    assert summary["review_groups"]["candidates_per_group"] == {"1_no_hard_conflict": 1, "2_hard_conflict": 1, "3_insufficient_evidence": 1, "4_technical_failure": 1}


def test_within_a_group_candidates_are_ordered_by_score_highest_first(tmp_path):
    # two clean pairs with different strength: same-age pair outranks the age-gap pair
    people = [pdict("a1", name="Anil Kumar", age=30, relative="Raju Rao"), pdict("a2", name="Anil Kumar", age=30, relative="Raju Rao"),
              pdict("b1", name="Bala Reddy", age=30, relative="Raju Rao"), pdict("b2", name="Bala Reddy", age=34, relative="Raju Rao")]
    w = build_world(tmp_path / "w", people)
    summary, root = go(w, tmp_path)
    scores = [float(r["match_score_percent_NOT_A_PROBABILITY"]) for r in rows_of(root, summary)]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 2


def test_nothing_is_removed_and_nothing_is_decided_by_the_ordering(ordering_world, tmp_path):
    summary, root = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: outcome(0.9)))
    recs = records(root, summary)
    assert len(recs) == summary["candidates"]["generated"] == summary["candidates"]["scored"] == 4
    assert all(r["review"] == {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None} for r in recs.values())
    assert all(all(r[c] == "" for c in DECISION_COLUMNS) for r in rows_of(root, summary))
    assert summary["human_review"]["decisions_recorded"] == 0 and summary["review_groups"]["note"].endswith("no group is a decision.")


def test_a_high_score_never_hides_a_hard_conflict(ordering_world, tmp_path):
    summary, root = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: outcome(0.9)))
    rec = records(root, summary)["h1:h2"]
    assert rec["match_score_percent"] >= 85 and rec["score_band"] == "A" and rec["has_hard_conflict"] is True
    assert [c["field"] for c in rec["hard_conflicts"]] == ["gender"] and "hard_conflict_present" in rec["processing"]["indicators"]
    page = (run_dir(root, summary) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert page.count("HARD CONFLICT") == 1 and "2_hard_conflict" in page and "1_no_hard_conflict" in page
    assert page.index("1_no_hard_conflict") < page.index("2_hard_conflict") < page.index("3_insufficient_evidence")
    by_id = {r["candidate_id"]: r for r in rows_of(root, summary)}
    assert "gender" in by_id["h1:h2"]["hard_conflicts"] and by_id["c1:c2"]["hard_conflicts"] == ""


def test_the_old_score_order_is_available_by_configuration(ordering_world, tmp_path):
    summary, root = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: outcome(0.9)), config=V2CONFIG(review_order="score"))
    rows = rows_of(root, summary)
    ids = [r["candidate_id"] for r in rows]
    scores = [float(r["match_score_percent_NOT_A_PROBABILITY"]) for r in rows]
    assert {r["review_group"] for r in rows} == {"0_all_by_score"} and scores[:3] == sorted(scores[:3], reverse=True)
    assert ids.index("h1:h2") < ids.index("c1:c2")  # pure score order: the conflicting pair (higher score) comes first
    assert rows[-1]["candidate_id"] == "i1:i2" and rows[-1]["insufficient_evidence"] == "True"


def test_technical_failures_are_flagged_for_retry_not_hidden(ordering_world, tmp_path):
    summary, root = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: FrUnavailable("down")))
    row = rows_of(root, summary)[3]
    assert row["review_group"] == "4_technical_failure" and "face_comparison_retry_needed" in row["processing_flags"]
    assert summary["status"] == "incomplete" and summary["review_ready"] is False


# ----------------------------------------------------------------------------- age in the report

def age_world(tmp_path):
    people = [pdict("a1", name="Anil Kumar", age=30, dob=None), pdict("a2", name="Anil Kumar", age=31),
              pdict("d1", name="Bala Reddy", age=45, dob=date(1996, 3, 3)),  # recorded age contradicts its own DOB
              pdict("d2", name="Bala Reddy", age=30)]
    return build_world(tmp_path / "w", people)


def test_the_report_shows_reference_date_ages_ranges_tolerance_and_method(tmp_path):
    w = age_world(tmp_path)
    summary, root = go(w, tmp_path, config=V2CONFIG(reference_date="2026-10-12"))
    rec = records(root, summary)["a1:a2"]
    age = rec["age_dob_summary"]
    assert (age["reference_date"], age["age_tolerance_years"], age["method"], age["comparison_status"]) == ("2026-10-12", 2, "estimated_birth_year_ranges", "within_tolerance")
    assert (age["recorded_age_a"], age["recorded_age_b"]) == (30, 31) and age["birth_year_range_a"] == [1995, 1996] and age["birth_year_range_b"] == [1994, 1995]
    assert age["dob_comparison_status"].startswith("not_compared_no_valid_dob_on_either_record")
    row = next(r for r in rows_of(root, summary) if r["candidate_id"] == "a1:a2")
    assert (row["age_a"], row["age_b"], row["reference_date"], row["age_tolerance_years"]) == ("30", "31", "2026-10-12", "2")
    assert row["birth_year_range_a"] == "1995-1996" and row["birth_year_range_b"] == "1994-1995" and row["rule_version"] == "score-v2"
    assert row["age_comparison_method"] == "estimated_birth_year_ranges / within_tolerance" and row["dob_comparison_status"].startswith("not_compared")
    page = (run_dir(root, summary) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "2026-10-12" in page and "estimated_birth_year_ranges" in page and "estimates, not dates of birth" in page
    assert summary["age_dob"]["reference_date"] == "2026-10-12" and summary["reference_date"] == "2026-10-12"
    assert json.loads((run_dir(root, summary) / "scoring_config.json").read_text())["reference_date"] == "2026-10-12"


def test_one_reference_date_is_used_for_every_candidate_in_the_run(tmp_path):
    w = age_world(tmp_path)
    summary, root = go(w, tmp_path, config=V2CONFIG(reference_date="2027-02-03"))
    dates = {r["fields"]["age_dob"]["detail"]["reference_date"] for r in records(root, summary).values()}
    assert dates == {"2027-02-03"}


def test_a_record_with_age_and_dob_that_disagree_is_flagged_for_the_reviewer(tmp_path):
    w = age_world(tmp_path)
    summary, root = go(w, tmp_path)
    rec = records(root, summary)["d1:d2"]
    flags = [c for c in rec["conflicting_attributes"] if c["severity"] == "data_quality"]
    assert len(flags) == 1 and "contradicts the age its own DOB implies" in flags[0]["explanation"] and "age_dob_inconsistent_record_a" in rec["processing"]["indicators"]
    assert rec["has_hard_conflict"] is False and rec["age_dob_summary"]["inconsistencies"][0]["recorded_age"] == 45
    row = next(r for r in rows_of(root, summary) if r["candidate_id"] == "d1:d2")
    assert "recorded age 45" in row["age_dob_inconsistency"] and summary["age_dob"]["age_dob_inconsistencies_within_a_record"] == 1
    page = (run_dir(root, summary) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "Data quality:" in page


def test_include_values_false_withholds_ages_and_dobs_but_keeps_the_method_and_dates(tmp_path):
    w = age_world(tmp_path)
    summary, root = go(w, tmp_path, config=V2CONFIG(include_values=False))
    age = records(root, summary)["a1:a2"]["age_dob_summary"]
    assert "recorded_age_a" not in age and age["method"] == "estimated_birth_year_ranges" and age["reference_date"] == "2026-10-12"
    assert rows_of(root, summary)[0]["age_a"] == ""


# ----------------------------------------------------------------------- versions and compatibility

def test_versions_never_mix_and_earlier_runs_are_preserved(tmp_path):
    w = age_world(tmp_path)
    s1, root = go(w, tmp_path, config=V1CONFIG(), label="old-v1")
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in run_dir(root, s1).iterdir() if f.is_file()}
    s2, _ = go(w, tmp_path, config=V2CONFIG(), label="new-v2")
    assert (s1["rule_version"], s2["rule_version"]) == ("score-v1", "score-v2") and s2["resumed"] is False
    assert {r["rule_version"] for r in records(root, s1).values()} == {"score-v1"} and {r["rule_version"] for r in records(root, s2).values()} == {"score-v2"}
    assert {r["rule_version"] for r in rows_of(root, s2)} == {"score-v2"}
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in run_dir(root, s1).iterdir() if f.is_file()}
    assert after == before  # the score-v1 run was not touched by the later run
    assert json.loads((run_dir(root, s1) / "scoring_config.json").read_text())["config"]["rule_version"] == "score-v1"
    assert summary_has_no_cross_version_claim(s1, s2)


def summary_has_no_cross_version_claim(s1, s2) -> bool:
    return s1["rule_version"] != s2["rule_version"] and "reference_date" in s2 and s1["age_dob"] is None


def test_score_v1_age_scoring_is_unchanged_inside_the_new_code(tmp_path):
    w = age_world(tmp_path)
    s1, root = go(w, tmp_path, config=V1CONFIG())
    detail = records(root, s1)["a1:a2"]["fields"]["age_dob"]
    assert detail["score"] == 0.9 and "differ by 1 year" in detail["explanation"] and "detail" not in detail  # exactly the earlier explanation and score
    assert records(root, s1)["a1:a2"]["age_dob_summary"] is None and s1["reference_date"] is None


def test_worksheets_of_both_layouts_validate(tmp_path):
    w = age_world(tmp_path)
    for config, layout in ((V1CONFIG(), WORKSHEET_COLUMNS_V1), (V2CONFIG(), WORKSHEET_COLUMNS_V2)):
        summary, root = go(w, tmp_path, config=config, label=config.rule_version)
        path = run_dir(root, summary) / "review_worksheet.csv"
        header = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert header.split(",") == list(layout)
        result = validate_worksheet(path)
        assert result["pending"] == 2 and result["decisions"] == [] and result["problems"] == []
    path = run_dir(root, summary) / "review_worksheet.csv"
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    col = {n: rows[0].index(n) for n in DECISION_COLUMNS}
    rows[1][col["reviewer_decision"]], rows[1][col["reviewer_id"]], rows[1][col["reviewed_at"]] = "reject_match", "alice", "2026-10-12T09:00:00"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    result = validate_worksheet(path)
    assert [d["decision"] for d in result["decisions"]] == ["reject_match"] and result["pending"] == 1


def test_v2_reruns_are_deterministic_and_resume_after_an_outage(ordering_world, tmp_path):
    down = FakeComparer(ordering_world, lambda a, b: FrUnavailable("down"))
    s1, root = go(ordering_world, tmp_path, down)
    assert s1["status"] == "incomplete"
    up = FakeComparer(ordering_world, lambda a, b: outcome(0.7))
    s2, _ = go(ordering_world, tmp_path, up)
    assert s2["resumed"] is True and s2["run_id"] == s1["run_id"] and s2["status"] == "completed"
    again, _ = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: outcome(0.7)), label="again")
    for name in ("scored_candidates.jsonl", "review_worksheet.csv"):
        assert (run_dir(root, s2) / name).read_bytes() == (run_dir(root, again) / name).read_bytes(), name


# ---------------------------------------------------------------------------- image selection

def selection_world(tmp_path, images, unusable=None):
    people = [pdict("x1", name="Alle Sai Shiva", relative="Rao"), pdict("x2", name="Alle Sai Shiva", relative="Rao")]
    return build_world(tmp_path / "w", people, images, unusable=unusable)


def single_thread(**kw):
    return V2CONFIG(face_concurrency=1, **kw)


def test_a_portrait_photo_is_ranked_ahead_of_an_identity_document_image(tmp_path):
    w = selection_world(tmp_path, {"x1": [("doc", 1, DOC), ("photo", 2, PHOTO)], "x2": [("other", 3, PHOTO)]})
    comparer = FakeComparer(w, lambda a, b: outcome(0.7))
    summary, root = go(w, tmp_path, comparer, single_thread())
    assert comparer.calls == [("photo", "other")]  # the document scan was never needed
    sel = records(root, summary)["x1:x2"]["image_selection"]
    assert [n["media_id"] for n in sel["a"]] == ["photo", "doc"] and sel["a"][0]["service_result"] == "face usable" and sel["a"][1]["attempted"] is False
    assert sel["a"][0]["category"] == PHOTO and "Phase 2 order" in sel["a"][0]["reason"] and sel["policy"] == "outcome_memory"


def test_an_unknown_image_type_is_ranked_last_not_excluded(tmp_path):
    w = selection_world(tmp_path, {"x1": [("mystery", 1, "something/UNKNOWN"), ("doc", 2, DOC)], "x2": [("p", 3, PHOTO)]})
    comparer = FakeComparer(w, lambda a, b: rejection("a", "no_face") if a == "doc" else outcome(0.7))
    summary, root = go(w, tmp_path, comparer, single_thread())
    sel = records(root, summary)["x1:x2"]["image_selection"]["a"]
    assert [n["media_id"] for n in sel] == ["doc", "mystery"]  # the unclassified image is still available as a later fallback
    assert comparer.calls == [("doc", "p"), ("mystery", "p")] and records(root, summary)["x1:x2"]["face"]["fallback_used"] is True


def test_a_corrupt_image_rejected_by_the_image_run_is_never_considered(tmp_path):
    w = selection_world(tmp_path, {"x1": [("good", 1, PHOTO)], "x2": [("p", 2, PHOTO)]}, unusable={"x1": [("broken", Status.CORRUPT_IMAGE)]})
    comparer = FakeComparer(w, lambda a, b: outcome(0.7))
    summary, root = go(w, tmp_path, comparer, single_thread())
    sel = records(root, summary)["x1:x2"]["image_selection"]["a"]
    assert [n["media_id"] for n in sel] == ["good"] and comparer.calls == [("good", "p")]


@pytest.mark.parametrize("kind", ["no_face", "multiple_faces", "low_quality", "invalid_image"])
def test_every_image_rejection_falls_back_to_the_next_eligible_image_and_records_it(tmp_path, kind):
    w = selection_world(tmp_path, {"x1": [("a1", 1), ("a2", 2)], "x2": [("b1", 3)]})
    comparer = FakeComparer(w, lambda a, b: rejection("a", kind) if a == "a1" else outcome(0.7))
    summary, root = go(w, tmp_path, comparer, single_thread())
    rec = records(root, summary)["x1:x2"]
    sel, face = rec["image_selection"], rec["face"]
    assert comparer.calls == [("a1", "b1"), ("a2", "b1")] and face["fallback_used"] and face["status"] == "compared"
    assert [(n["media_id"], n["service_result"]) for n in sel["a"]] == [("a1", kind), ("a2", "face usable")]
    assert [a["outcome"] for a in face["attempts"]] == ["rejected", "compared"] and sel["final_status"] == "compared" and sel["fallback_used"]


def test_when_every_image_fails_the_attempt_limit_holds_and_the_real_status_is_kept(tmp_path):
    w = selection_world(tmp_path, {"x1": [(f"a{i}", i) for i in range(1, 6)], "x2": [("b1", 9)]})
    comparer = FakeComparer(w, lambda a, b: rejection("a", "no_face"))
    summary, root = go(w, tmp_path, comparer, single_thread())
    rec = records(root, summary)["x1:x2"]
    assert len(comparer.calls) == 3 and rec["face"]["status"] == "no_face" and rec["face"]["raw_cosine"] is None  # 3 attempts, then it stops
    sel = rec["image_selection"]["a"]
    assert [n["attempted"] for n in sel] == [True, True, True, False, False] and [n["within_attempt_limit"] for n in sel] == [True, True, True, False, False]
    assert summary["complete"] is True and rec["face"]["retry_needed"] is False  # a genuine face problem is final, not retried


def test_images_the_service_already_refused_are_tried_last_and_reach_deeper_images(tmp_path):
    w = selection_world(tmp_path, {"x1": [(f"a{i}", i) for i in range(1, 5)], "x2": [("b1", 9)]})
    root = tmp_path / "scoring"
    seed_known_image_failure(root, "a1")
    seed_known_image_failure(root, "a2")
    comparer = FakeComparer(w, lambda a, b: rejection("a", "no_face") if a in ("a1", "a2") else outcome(0.7))
    summary, _ = go(w, tmp_path, comparer, single_thread(), root=root)
    rec = records(root, summary)["x1:x2"]
    assert comparer.calls[0] == ("a3", "b1") and rec["face"]["status"] == "compared"  # the first call is not wasted on a known failure
    sel = rec["image_selection"]["a"]
    assert [n["media_id"] for n in sel] == ["a3", "a4", "a1", "a2"] and "already refused this image (no_face)" in sel[2]["reason"]
    # the earlier policy would have spent its first attempts on a1 and a2
    old = FakeComparer(w, lambda a, b: rejection("a", "no_face") if a in ("a1", "a2") else outcome(0.7))
    go(w, tmp_path, old, V2CONFIG(face_concurrency=1, image_ranking="phase2_rank"), root=tmp_path / "other-root")
    assert old.calls[0] == ("a1", "b1") and len(old.calls) == 3


def test_deeper_images_become_reachable_when_earlier_ones_are_known_failures(tmp_path):
    w = selection_world(tmp_path, {"x1": [(f"a{i}", i) for i in range(1, 5)], "x2": [("b1", 9)]})
    root = tmp_path / "scoring"
    for m in ("a1", "a2", "a3"):
        seed_known_image_failure(root, m)
    comparer = FakeComparer(w, lambda a, b: outcome(0.7) if a == "a4" else rejection("a", "no_face"))
    summary, _ = go(w, tmp_path, comparer, single_thread(), root=root)
    assert records(root, summary)["x1:x2"]["face"]["status"] == "compared" and comparer.calls == [("a4", "b1")]  # old order would exhaust 3 attempts on a1..a3
    other = FakeComparer(w, lambda a, b: outcome(0.7) if a == "a4" else rejection("a", "no_face"))
    s_old, r_old = go(w, tmp_path, other, V2CONFIG(face_concurrency=1, image_ranking="phase2_rank"), root=tmp_path / "other")
    assert records(r_old, s_old)["x1:x2"]["face"]["status"] == "no_face"


def test_a_file_server_failure_is_an_operational_error_not_an_image_rejection(tmp_path):
    w = selection_world(tmp_path, {"x1": [("a1", 1), ("a2", 2)], "x2": [("b1", 3)]})
    del w.files[url_for("person/media/a1", ".png")]  # the file server no longer has the first image
    comparer = FakeComparer(w, lambda a, b: outcome(0.7))
    summary, root = go(w, tmp_path, comparer, single_thread())
    face = records(root, summary)["x1:x2"]["face"]
    assert comparer.calls == [] and face["status"] == "error" and face["error_kind"] == "image_load:file_not_found" and face["fallback_used"] is False
    assert face["raw_cosine"] is None and face["failed_side"] == "a"  # no fallback: it is not a statement about the face


def test_shared_image_files_are_flagged_and_never_compared_with_themselves(tmp_path):
    people = [pdict("x1", name="Alle Sai Shiva", relative="Rao"), pdict("x2", name="Alle Sai Shiva", relative="Rao")]
    w = build_world(tmp_path / "w", people, {"x1": [("s1", 5)], "x2": [("s2", 5)]}, shared_paths={"s2": "person/media/s1"})
    comparer = FakeComparer(w, lambda a, b: outcome(1.0))
    summary, root = go(w, tmp_path, comparer)
    rec = records(root, summary)["x1:x2"]
    assert comparer.calls == [] and rec["face"]["status"] == "skipped_shared_image" and "same_image_file_attached_to_both_records" in rec["processing"]["indicators"]
    assert rec["image_selection"]["applicable"] is True and rec["images"]["shared_image_file"] is True


def test_operational_service_errors_are_not_retried_without_limit(tmp_path):
    w = selection_world(tmp_path, {"x1": [("a1", 1), ("a2", 2), ("a3", 3)], "x2": [("b1", 9)]})
    comparer = FakeComparer(w, lambda a, b: FrUnavailable("down"))
    summary, root = go(w, tmp_path, comparer, single_thread())
    assert len(comparer.calls) == 1 and records(root, summary)["x1:x2"]["face"]["fallback_used"] is False  # an outage never walks through the images


def test_the_measurement_used_to_justify_the_ranking_is_reported_not_assumed(tmp_path):
    w = selection_world(tmp_path, {"x1": [("doc1", 1, DOC), ("photo1", 2, PHOTO)], "x2": [("doc2", 3, DOC)]})
    root = tmp_path / "scoring"
    seed_known_image_failure(root, "doc1")
    comparer = FakeComparer(w, lambda a, b: outcome(0.7))
    summary, _ = go(w, tmp_path, comparer, single_thread(), root=root)
    info = summary["image_selection"]
    assert info["policy"] == "outcome_memory" and info["outcome_memory_images"] >= 1 and "No reliable portrait-vs-document signal" in info["limitation"]
    assert info["measured_face_usable_rate_by_category"][DOC] == {"images_with_known_outcome": 1, "face_usable": 0, "rate": 0.0}


# ----------------------------------------------------------------------------- memory unit tests

def test_outcome_memory_reads_stored_comparisons_conservatively(tmp_path):
    import json as _json

    path = tmp_path / "cache.jsonl"
    good = {"signature": {"model": "buffalo_l"}, "evidence": {"attempts": [
        {"media_id_a": "m1", "media_id_b": "m2", "outcome": "rejected", "rejections": {"a": "no_face"}},  # b was analysed fine
        {"media_id_a": "m3", "media_id_b": "m4", "outcome": "compared"},
        {"media_id_a": "m5", "media_id_b": "m6", "outcome": "error", "error_kind": "unavailable"},  # operational: remembered as nothing
        {"media_id_a": "m7", "media_id_b": "m8", "outcome": "rejected", "rejections": {"a": "low_quality", "b": "multiple_faces"}}]}}
    other_model = {"signature": {"model": "different"}, "evidence": {"attempts": [{"media_id_a": "z1", "media_id_b": "z2", "outcome": "compared"}]}}
    path.write_text("\n".join([_json.dumps(good), _json.dumps(other_model), '{"half written', "not json"]) + "\n")
    memory = ImageOutcomeMemory.from_cache_file(path, "buffalo_l")
    assert {m: memory.known(m) for m in ("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "z1")} == {
        "m1": "no_face", "m2": "ok", "m3": "ok", "m4": "ok", "m5": None, "m6": None, "m7": "low_quality", "m8": "multiple_faces", "z1": None}
    assert memory.failed("m1") and not memory.failed("m2") and not memory.failed("unknown") and len(memory) == 6
    assert ImageOutcomeMemory.from_cache_file(tmp_path / "missing.jsonl", "buffalo_l").counts() == {}


# ------------------------------------------------------------------------------------ safety

def test_v2_issues_only_read_only_statements_and_keeps_the_database_untouched(ordering_world, tmp_path):
    db = ordering_world.db()
    summary, _ = go(ordering_world, tmp_path, FakeComparer(ordering_world, lambda a, b: outcome(0.7)), db=db)
    assert db.statements and all(s.lstrip().lstrip("/*").split("*/", 1)[-1].strip().split(None, 1)[0].upper() in ("SELECT", "SHOW", "WITH") for s in db.statements)
    assert summary["database_writes"] == 0 and summary["database"]["transaction_read_only"] == "on"
