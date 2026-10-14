"""End-to-end Phase 6: real Phase 5 and Phase 2 outputs from a small world, scripted face service, fake read-only database."""

from __future__ import annotations

import csv
import hashlib  # tests may hash files to compare them; the package itself must not (checked below)
import json
from pathlib import Path

import pytest

from dedup_pipeline.candidates import CandidateConfig
from dedup_pipeline.db import ReadOnlyViolation
from dedup_pipeline.face_recognition import FrUnavailable
from dedup_pipeline.scoring import InputError, ScoringConfig, ScoringOptions, run_scoring
from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS, WORKSHEET_COLUMNS_V1, safe_cell
from dedup_pipeline.scoring.worksheet import validate_worksheet

from .scoring_fakes import FakeComparer, FakeScoringDb, V1CONFIG, backend, build_world, outcome, pdict, rejection

PACKAGE = Path(__file__).resolve().parent.parent / "dedup_pipeline" / "scoring"


def main_people():
    return [
        pdict("p01", name="Ramesh Kumar", relative="Suresh Kumar", age=30, phone="9876543211"),
        pdict("p02", name="Ramesh Kumar", relative="Suresh Kumar", age=31, phone="9876543211"),
        pdict("p03", name="Ramesh Kumar", relative="Suresh Kumar", age=30),
        pdict("p04", "V1", name="Ramesh Kumar", relative="Other Person", age=60, gender="Female"),
    ]


MAIN_IMAGES = {"p01": [("m01a", 1), ("m01b", 2)], "p02": [("m02", 3)]}


def go(world, tmp_path, comparer=None, config=None, label=None, verify=None, db=None, root=None, resume=True):
    root = root or tmp_path / "scoring"
    be = backend(world, comparer, verify)
    return run_scoring(db or world.db(), config or V1CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, label, resume), be), root


def records(root: Path, summary: dict) -> dict[str, dict]:
    path = root / "runs" / summary["run_id"] / "scored_candidates.jsonl"
    return {r["candidate_id"]: r for r in map(json.loads, path.read_text(encoding="ascii").splitlines())}


def run_dir(root: Path, summary: dict) -> Path:
    return root / "runs" / summary["run_id"]


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path / "w", main_people(), MAIN_IMAGES)


# --------------------------------------------------------------------- face coverage cases

def test_both_images_usable_calls_the_service_and_keeps_the_raw_cosine(world, tmp_path):
    comparer = FakeComparer(world, lambda a, b: outcome(0.63))
    summary, root = go(world, tmp_path, comparer)
    rec = records(root, summary)["p01:p02"]
    assert comparer.calls == [("m01a", "m02")]  # one call; the first-ranked image of each person
    face = rec["face"]
    assert (face["status"], face["raw_cosine"], face["model"], face["metric"]) == ("compared", 0.63, "buffalo_l", "cosine")
    assert (face["media_id_a"], face["media_id_b"], face["fallback_used"], face["in_match_score"]) == ("m01a", "m02", False, True)
    assert face["relative_to_service_threshold"] == "above" and "not an identity decision" in face["service_threshold_note"].lower()
    assert rec["fields"]["face"]["applicable"] and rec["fields"]["face"]["score"] == 1.0  # 0.63 is above the 0.58 anchor
    assert summary["face"]["eligible_pairs_both_usable_images"] == 1 and summary["face"]["status_counts"]["compared"] == 1 and summary["face"]["compared_relative_to_service_threshold_not_a_decision"] == {"above_service_threshold": 1}


def test_no_image_on_either_side_or_only_one_side_skips_the_comparison_and_keeps_other_evidence(world, tmp_path):
    comparer = FakeComparer(world, lambda a, b: outcome(0.9))
    summary, root = go(world, tmp_path, comparer)
    recs = records(root, summary)
    assert len(comparer.calls) == 1  # only p01-p02; nothing sent for the other five pairs
    one_side, neither = recs["p01:p03"], recs["p03:p04"]
    for rec in (one_side, neither):
        assert rec["face"]["status"] == "skipped_no_image" and rec["face"]["raw_cosine"] is None and rec["face"]["comparison_attempted"] is False
        face_field = rec["fields"]["face"]
        assert not face_field["applicable"] and face_field["score"] is None and face_field["not_applicable_reason"] == "skipped_no_image"
        assert rec["match_score_percent"] is not None and rec["evidence_coverage_percent"] < 100  # scored on the other evidence
    assert summary["face"]["skipped_no_image_detail"] == {"no_usable_image_on_both": 1, "no_usable_image_on_one_side": 4}
    assert one_side["fields"]["name"]["applicable"] and one_side["fields"]["phone"]["applicable"] is False


def test_a_face_mismatch_lowers_the_score_but_keeps_negative_cosine_and_is_not_a_decision(world, tmp_path):
    # separate output roots: within one root a stored result for the same images and model is (correctly) reused
    summary_match, root_a = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.70)), label="match", root=tmp_path / "root-a")
    summary_miss, root_b = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(-0.12)), label="miss", root=tmp_path / "root-b")
    hit, miss = records(root_a, summary_match)["p01:p02"], records(root_b, summary_miss)["p01:p02"]
    assert miss["face"]["raw_cosine"] == -0.12 and miss["face"]["status"] == "compared" and miss["face"]["relative_to_service_threshold"] == "below"  # -0.12 stays -0.12
    assert miss["fields"]["face"]["score"] == 0.0 and miss["match_score_percent"] < hit["match_score_percent"]
    assert any(c["field"] == "face" and c["severity"] == "soft" for c in miss["conflicting_attributes"])
    assert miss["review"]["status"] == "pending" and miss["review"]["decision"] is None  # even a clear mismatch is not rejected by software


def test_conflicting_attributes_are_listed_with_severity_and_matching_ones_separately(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)))
    rec = records(root, summary)["p01:p04"]
    conflicts = {c["field"]: c["severity"] for c in rec["conflicting_attributes"]}
    assert conflicts["gender"] == "hard" and conflicts["age_dob"] == "hard" and conflicts["relative_name"] == "hard"
    assert [m["field"] for m in rec["matching_attributes"]] == ["name"] and rec["has_hard_conflict"] is True
    assert "hard_conflict_present" in rec["processing"]["indicators"] and "read the conflicting attributes first" in rec["score_band_note"]
    clean = records(root, summary)["p01:p02"]
    assert not clean["has_hard_conflict"] and {"name", "relative_name", "phone"} <= {m["field"] for m in clean["matching_attributes"]}


def test_missing_evidence_is_listed_with_reasons(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)))
    missing = {m["field"]: m["reason"] for m in records(root, summary)["p01:p03"]["missing_evidence"]}
    assert missing["alias"] == "no_alias" and missing["address"] == "no_comparable_address"
    assert missing["face"] == "skipped_no_image" and missing["phone"].startswith("no_usable_phone")


def test_insufficient_evidence_is_flagged_and_sorted_last(tmp_path):
    people = [pdict("a1", name="Ramesh Kumar", relative=None, age=None, gender=None), pdict("a2", name="Ramesh Kumar", relative=None, age=None, gender=None),
              pdict("b1", name="Sita Devi", relative="Raju", age=30, phone="9876543299"), pdict("b2", name="Sita Devi", relative="Raju", age=30, phone="9876543299")]
    w = build_world(tmp_path / "w", people)
    summary, root = go(w, tmp_path)
    recs = records(root, summary)
    thin = recs["a1:a2"]
    assert thin["insufficient_evidence"] and thin["score_band"] == "E" and thin["match_score_percent"] == 100.0  # perfect but thin
    assert "insufficient_evidence" in thin["processing"]["indicators"] and summary["candidates"]["insufficient_evidence"] == 1
    rows = list(csv.DictReader(open(run_dir(root, summary) / "review_worksheet.csv", encoding="utf-8-sig")))
    assert [r["candidate_id"] for r in rows] == ["b1:b2", "a1:a2"]  # the well-supported pair first, thin evidence last


# -------------------------------------------------------------------------- fallback

def fallback_world(tmp_path):
    people = [pdict("q01", name="Alle Sai Shiva", relative="Rao"), pdict("q02", name="Alle Sai Shiva", relative="Rao")]
    return build_world(tmp_path / "w", people, {"q01": [("q1a", 10), ("q1b", 11)], "q02": [("q2", 12)]})


def test_failed_face_detection_falls_back_to_the_next_image_deterministically(tmp_path):
    w = fallback_world(tmp_path)
    comparer = FakeComparer(w, lambda a, b: rejection("a", "no_face") if a == "q1a" else outcome(0.72))
    summary, root = go(w, tmp_path, comparer)
    face = records(root, summary)["q01:q02"]["face"]
    assert comparer.calls == [("q1a", "q2"), ("q1b", "q2")]
    assert face["status"] == "compared" and face["fallback_used"] and (face["media_id_a"], face["media_id_b"]) == ("q1b", "q2")
    assert [(a["media_id_a"], a["outcome"]) for a in face["attempts"]] == [("q1a", "rejected"), ("q1b", "compared")]
    assert summary["face"]["fallback_used"] == 1
    used = {i["media_id"] for i in records(root, summary)["q01:q02"]["images"]["a"] if i["used_in_face_comparison"]}
    assert used == {"q1a", "q1b"}


@pytest.mark.parametrize("kind,status", [("no_face", "no_face"), ("multiple_faces", "multiple_faces"), ("low_quality", "low_quality"),
                                         ("invalid_image", "invalid_image")])
def test_each_failure_kind_keeps_its_own_status_and_is_not_a_similarity(tmp_path, kind, status):
    w = fallback_world(tmp_path)
    summary, root = go(w, tmp_path, FakeComparer(w, lambda a, b: rejection("b", kind)))
    rec = records(root, summary)["q01:q02"]
    assert rec["face"]["status"] == status and rec["face"]["raw_cosine"] is None and rec["face"]["failed_side"] == "b"
    assert rec["face"]["retry_needed"] is False and not rec["fields"]["face"]["applicable"]  # a genuine face problem is final, not retried
    assert summary["complete"] is True and summary["face"]["status_counts"] == {status: 1}


def test_the_same_file_on_both_records_is_not_compared_and_is_flagged(tmp_path):
    people = [pdict("s01", name="Gita Devi", relative="Rao"), pdict("s02", name="Gita Devi", relative="Rao")]
    w = build_world(tmp_path / "w", people, {"s01": [("ms1", 20)], "s02": [("ms2", 20)]}, shared_paths={"ms2": "person/media/ms1"})
    comparer = FakeComparer(w, lambda a, b: outcome(1.0))
    summary, root = go(w, tmp_path, comparer)
    rec = records(root, summary)["s01:s02"]
    assert comparer.calls == [] and rec["face"]["status"] == "skipped_shared_image" and rec["face"]["raw_cosine"] is None
    assert rec["images"]["shared_image_file"] and "same_image_file_attached_to_both_records" in rec["processing"]["indicators"]
    assert summary["face"]["same_image_file_on_both_records"] == 1 and summary["complete"] is True


# ----------------------------------------------- service failures, retries and completeness

def test_a_service_outage_is_retryable_incomplete_and_never_a_mismatch(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: FrUnavailable("connection refused")))
    rec = records(root, summary)["p01:p02"]
    assert rec["face"]["status"] == "error" and rec["face"]["error_kind"] == "unavailable" and rec["face"]["retry_needed"] is True
    assert rec["face"]["raw_cosine"] is None and not rec["fields"]["face"]["applicable"] and rec["processing"]["complete"] is False
    assert "face_comparison_retry_needed" in rec["processing"]["indicators"]
    assert summary["status"] == "incomplete" and summary["complete"] is False and summary["review_ready"] is False
    assert summary["face"]["retryable_errors"] == 1 and summary["face"]["error_kinds"] == {"unavailable": 1}
    assert json.loads((run_dir(root, summary) / "run_state.json").read_text())["status"] == "incomplete"
    assert json.loads((root / "latest.json").read_text())["status"] == "incomplete"


def test_rerun_resumes_the_incomplete_run_and_only_retries_what_failed(tmp_path):
    people = [pdict(f"r0{i}", name="Ramesh Kumar", relative="Suresh Kumar", age=30) for i in range(1, 4)]
    w = build_world(tmp_path / "w", people, {"r01": [("x1", 1)], "r02": [("x2", 2)], "r03": [("x3", 3)]})
    first = FakeComparer(w, lambda a, b: FrUnavailable("down") if "x3" in (a, b) else outcome(0.6))
    s1, root = go(w, tmp_path, first, config=V1CONFIG(face_concurrency=1))
    assert s1["status"] == "incomplete" and s1["face"]["retryable_errors"] == 2 and len(first.calls) == 3
    second = FakeComparer(w, lambda a, b: outcome(0.6))
    s2, _ = go(w, tmp_path, second, config=V1CONFIG(face_concurrency=1))
    assert s2["resumed"] is True and s2["run_id"] == s1["run_id"]  # the unfinished run is continued
    assert sorted(second.calls) == [("x1", "x3"), ("x2", "x3")]  # the stored x1-x2 result was not recomputed
    assert s2["status"] == "completed" and s2["face"]["pairs_reused_from_cache"] == 1 and s2["face"]["retryable_errors"] == 0
    assert (root / "latest.json").exists() and json.loads((root / "latest.json").read_text())["status"] == "completed"


def test_circuit_breaker_stops_the_face_stage_after_repeated_service_failures(tmp_path):
    people = [pdict(f"r0{i}", name="Ramesh Kumar", relative="Suresh Kumar") for i in range(1, 6)]
    w = build_world(tmp_path / "w", people, {f"r0{i}": [(f"x{i}", i)] for i in range(1, 6)})
    comparer = FakeComparer(w, lambda a, b: FrUnavailable("down"))
    summary, root = go(w, tmp_path, comparer, config=V1CONFIG(face_concurrency=1, face_circuit_breaker=2))
    assert len(comparer.calls) == 2  # it stopped hammering a dead service
    assert summary["face"]["stage_aborted"] and summary["face"]["retryable_errors"] == 10 and summary["status"] == "incomplete"
    assert all(r["face"]["retry_needed"] for r in records(root, summary).values() if r["face"]["status"] != "skipped_no_image")
    assert any(item.startswith("face_stage_stopped") for item in summary["limitations"])


def test_a_failed_service_check_marks_everything_retryable_without_sending_requests(world, tmp_path):
    comparer = FakeComparer(world, lambda a, b: outcome(0.9))

    def down():
        raise RuntimeError("not FR-DOPAMS")

    summary, root = go(world, tmp_path, comparer, verify=down)
    assert comparer.calls == [] and summary["face"]["service_check_error"].startswith("RuntimeError")
    assert records(root, summary)["p01:p02"]["face"]["status"] == "error" and summary["complete"] is False


def test_no_face_service_configured_means_pending_not_zero_and_never_cached(world, tmp_path):
    summary, root = go(world, tmp_path, None)
    rec = records(root, summary)["p01:p02"]
    assert rec["face"]["status"] == "not_checked" and rec["face"]["raw_cosine"] is None and rec["face"]["retry_needed"] is True
    assert summary["complete"] is False and summary["face"]["retryable_errors"] == 1
    cache = root / "face_evidence_cache.jsonl"
    assert not cache.exists() or cache.read_text().strip() == ""


def test_a_model_other_than_the_configured_one_is_an_error_not_evidence(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.9, model="some_other_model")))
    face = records(root, summary)["p01:p02"]["face"]
    assert face["status"] == "error" and face["error_kind"] == "model_mismatch" and face["raw_cosine"] is None and face["retry_needed"]


def test_stored_face_results_are_reused_even_when_weights_change(world, tmp_path):
    first = FakeComparer(world, lambda a, b: outcome(0.7))
    go(world, tmp_path, first, label="one")
    second = FakeComparer(world, lambda a, b: outcome(0.7))
    reweighted = V1CONFIG(weights={"name": 0.4, "alias": 0.05, "relative_name": 0.1, "phone": 0.1, "age_dob": 0.1, "gender": 0.05, "address": 0.05, "face": 0.15})
    summary, root = go(world, tmp_path, second, config=reweighted, label="two")
    assert second.calls == [] and summary["face"]["pairs_reused_from_cache"] == 1  # same images and model: no new service call
    assert summary["score"]["weights"]["name"] == 0.4


def test_changed_images_invalidate_the_stored_comparison(tmp_path):
    w1 = build_world(tmp_path / "a", main_people(), MAIN_IMAGES)
    go(w1, tmp_path, FakeComparer(w1, lambda a, b: outcome(0.7)), root=tmp_path / "shared", label="one")
    w2 = build_world(tmp_path / "b", main_people(), {"p01": [("m01a-new", 4)], "p02": [("m02", 3)]})
    second = FakeComparer(w2, lambda a, b: outcome(0.2))
    go(w2, tmp_path, second, root=tmp_path / "shared", label="two")
    assert second.calls == [("m01a-new", "m02")]


# --------------------------------------------------- deferred, limits, input validation

def test_deferred_pairs_are_reported_as_never_evaluated_and_are_not_scored(tmp_path):
    people = [pdict(f"d0{i}", name="Common Name", relative="Raj") for i in range(1, 5)] + \
             [pdict("e01", name="Rare Person", relative="Raj"), pdict("e02", name="Rare Person", relative="Raj")]
    w = build_world(tmp_path / "w", people, candidate_config=CandidateConfig(name_max_block_size=3, token_sorted_max_block_size=3))
    summary, root = go(w, tmp_path)
    recs = records(root, summary)
    assert set(recs) == {"e01:e02"}  # none of the six deferred d-pairs appear as scored or as non-matches
    assert summary["deferred"]["blocks"] == 2 and summary["deferred"]["pairs_never_evaluated_by_any_rule"] == 12
    assert summary["deferred"]["interpretation"] == "never evaluated; not non-matches"
    assert any(item.startswith("candidate_generation_deferred_blocks") for item in summary["limitations"])
    lines = [json.loads(x) for x in (run_dir(root, summary) / "deferred_unevaluated.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(x["evaluation_status"] == "never_evaluated" and "NOT a non-match" in x["interpretation"] for x in lines)
    assert summary["candidates"]["generated"] == summary["candidates"]["scored"] == 1


def test_partial_image_resolution_is_reported_as_a_limitation_and_nobody_is_assumed_checked(tmp_path):
    w = build_world(tmp_path / "w", main_people(), MAIN_IMAGES, partial_images=True)
    summary, _ = go(w, tmp_path, FakeComparer(w, lambda a, b: outcome(0.7)))
    assert summary["inputs"]["image_resolution"]["partial_run"] is True
    assert any(item.startswith("image_resolution_partial_run") for item in summary["limitations"])


def test_persons_missing_from_the_image_run_are_marked_not_checked_not_no_image(world, tmp_path):
    summary_path = world.image_dir / "person_summary.jsonl"
    kept = [line for line in summary_path.read_text().splitlines() if '"person_id":"p03"' not in line]
    summary_path.write_text("\n".join(kept) + "\n")
    meta = json.loads((world.image_dir / "run_summary.json").read_text())
    meta["persons"] -= 1
    (world.image_dir / "run_summary.json").write_text(json.dumps(meta))
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)))
    flags = records(root, summary)["p01:p03"]["processing"]["indicators"]
    assert "image_resolution_not_run_for_person_b" in flags
    assert any(item.startswith("image_resolution_person_count_mismatch") for item in summary["limitations"])


def test_inputs_must_be_complete_consistent_and_current(world, tmp_path):
    cand = world.candidate_dir
    summary_path = cand / "run_summary.json"
    good = summary_path.read_text()

    def expect(message, fix=None, **kw):
        with pytest.raises(InputError, match=message):
            go(world, tmp_path, None, root=tmp_path / f"r-{message[:6]}", **kw)
        if fix:
            fix()

    data = json.loads(good)
    summary_path.write_text(json.dumps({**data, "status": "failed"}))
    expect("not complete", lambda: summary_path.write_text(good))
    summary_path.write_text(json.dumps({**data, "generator_version": "candidates-v0"}))
    expect("unsupported candidate generator version", lambda: summary_path.write_text(good))
    state_path = cand / "run_state.json"
    state_good = state_path.read_text()
    state_path.write_text(json.dumps({**json.loads(state_good), "run_id": "someone-elses-run"}))
    expect("does not confirm the run", lambda: state_path.write_text(state_good))
    cand_lines = (cand / "candidates.jsonl").read_text().splitlines()
    (cand / "candidates.jsonl").write_text("\n".join(cand_lines[:-1]) + "\n")
    expect("candidates but run_summary.json says", lambda: (cand / "candidates.jsonl").write_text("\n".join(cand_lines) + "\n"))
    summary_path.unlink()
    expect("not found", lambda: summary_path.write_text(good))
    (world.image_dir / "run_summary.json").rename(world.image_dir / "gone.json")
    expect("Phase 2 image resolution: run_summary.json not found")
    (world.image_dir / "gone.json").rename(world.image_dir / "run_summary.json")
    expect("database changed since candidate generation", db=world.db(stamp=(999, "x", 999, "x")))


def test_stale_candidates_can_be_accepted_explicitly_and_are_reported(world, tmp_path):
    summary, _ = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)), db=world.db(stamp=(999, "x", 999, "x")),
                    config=V1CONFIG(allow_stale_candidates=True))
    assert "database_changed_since_candidate_generation" in summary["limitations"]


def test_candidates_that_disagree_with_the_database_fail_the_run_without_a_completion_marker(world, tmp_path):
    changed = [dict(p, source_system="V1") if p["person_id"] == "p01" else p for p in world.people]
    with pytest.raises(InputError, match="source systems differ"):
        go(world, tmp_path, None, db=FakeScoringDb(changed))
    root = tmp_path / "scoring"
    only = next((root / "runs").iterdir())
    assert not (only / "run_summary.json").exists() and json.loads((only / "run_state.json").read_text())["status"] == "failed"
    gone = [p for p in world.people if p["person_id"] != "p02"]
    with pytest.raises(InputError, match="not in the database"):
        go(world, tmp_path, None, db=FakeScoringDb(gone, stamp=world.db().stamp), root=tmp_path / "other")


# ----------------------------------------------- determinism, preservation, human review

def _stable(path: Path) -> bytes:
    return path.read_bytes()


def test_reruns_are_deterministic_and_previous_runs_are_preserved(world, tmp_path):
    comparer = lambda: FakeComparer(world, lambda a, b: outcome(0.61))
    s1, root = go(world, tmp_path, comparer(), label="first")
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in run_dir(root, s1).iterdir() if f.is_file()}
    s2, _ = go(world, tmp_path, comparer(), label="second")
    d1, d2 = run_dir(root, s1), run_dir(root, s2)
    for name in ("scored_candidates.jsonl", "review_worksheet.csv", "deferred_unevaluated.jsonl", "scoring_config.json"):
        assert _stable(d1 / name) == _stable(d2 / name), name
    assert _stable(d1 / "review_report" / "page_0001.html") == _stable(d2 / "review_report" / "page_0001.html")
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in d1.iterdir() if f.is_file()}
    assert after == before  # the earlier run's files are untouched
    assert json.loads((d1 / "scoring_config.json").read_text())["config"]["weights"]["name"] == 0.25
    with pytest.raises(InputError, match="already completed"):
        go(world, tmp_path, comparer(), label="first")


def test_completed_runs_are_never_resumed_a_new_directory_is_made(world, tmp_path):
    comparer = lambda: FakeComparer(world, lambda a, b: outcome(0.61))
    s1, root = go(world, tmp_path, comparer())
    s2, _ = go(world, tmp_path, comparer())
    assert s1["run_id"] != s2["run_id"] and s2["resumed"] is False and len(list((root / "runs").iterdir())) == 2


def test_every_candidate_stays_pending_whatever_its_score(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.9)))
    recs = records(root, summary)
    assert max(r["match_score_percent"] for r in recs.values() if r["match_score_percent"] is not None) >= 85  # a top-band candidate exists
    assert all(r["review"] == {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None} for r in recs.values())
    assert summary["human_review"]["status"] == "pending" and summary["human_review"]["decisions_recorded"] == 0
    assert summary["candidates"]["awaiting_human_review"] == summary["candidates"]["scored"] == len(recs)
    assert set(summary["human_review"]["allowed_decisions"]) == {"confirm_same_person", "reject_match", "needs_further_review", "unable_to_verify"}


def test_the_score_is_labelled_as_not_a_probability_everywhere(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.9)))
    rec = records(root, summary)["p01:p02"]
    assert "NOT a probability" in rec["match_score_label"] and summary["score"]["calibrated_probability"] is False
    assert "NOT_A_PROBABILITY" in " ".join(WORKSHEET_COLUMNS)
    page = (run_dir(root, summary) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "NOT a probability" in page and "Match score" in page and "pending" in page


def test_worksheet_has_empty_decision_columns_and_a_clear_decision_workflow(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.9)))
    d = run_dir(root, summary)
    with open(d / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == WORKSHEET_COLUMNS_V1 and len(rows) == 6  # a score-v1 run keeps writing the original layout
    assert all(all(r[c] == "" for c in DECISION_COLUMNS) and r["review_status"] == "pending" for r in rows)
    first = rows[0]
    for column in ("candidate_id", "person_a_id", "person_b_id", "source_a", "source_record_a", "source_b", "source_record_b",
                   "evidence_coverage_percent", "face_status", "face_raw_cosine", "field_scores", "score_explanation", "matching_attributes",
                   "conflicting_attributes", "missing_evidence", "image_links_a", "image_links_b", "candidate_rules", "processing_flags"):
        assert column in first
    assert "/api/file-proxy?url=" in first["image_links_a"] or first["image_links_a"] == ""
    assert "confirm_same_person" in (d / "REVIEW_INSTRUCTIONS.md").read_text() and "pending until an authorised human decides" in (d / "REVIEW_INSTRUCTIONS.md").read_text()
    html = (d / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "<form" not in html and "<button" not in html and "<input" not in html  # a read-only view: no control can record a decision


def test_worksheet_validation_accepts_only_explicit_complete_decisions(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.9)))
    path = run_dir(root, summary) / "review_worksheet.csv"
    assert validate_worksheet(path)["pending"] == 6 and validate_worksheet(path)["decisions"] == []  # untouched worksheet: nothing decided
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    col = {name: header.index(name) for name in DECISION_COLUMNS}
    rows[1][col["reviewer_decision"]], rows[1][col["reviewer_id"]], rows[1][col["reviewed_at"]] = "confirm_same_person", "alice", "2026-10-12T10:00:00"
    rows[2][col["reviewer_decision"]] = "reject_match"  # no reviewer id or time
    rows[3][col["reviewer_decision"]], rows[3][col["reviewer_id"]], rows[3][col["reviewed_at"]] = "auto_confirm", "bob", "2026-10-12"
    rows[4][col["reviewer_decision"]], rows[4][col["reviewer_id"]], rows[4][col["reviewed_at"]] = "needs_further_review", "carol", "not a date"
    rows[5][col["reviewer_decision"]], rows[5][col["reviewer_id"]], rows[5][col["reviewed_at"]] = "unable_to_verify", "dave", "2026-10-12T11:00:00Z"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    result = validate_worksheet(path, known_candidates=set(records(root, summary)))
    assert [d["decision"] for d in result["decisions"]] == ["confirm_same_person", "unable_to_verify"] and result["pending"] == 1
    assert len(result["problems"]) == 3 and any("reviewer_id is required" in p for p in result["problems"])
    assert any("must be one of" in p for p in result["problems"]) and any("ISO" in p for p in result["problems"])
    rows[0][0] = "renamed"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with pytest.raises(Exception, match="columns do not match"):
        validate_worksheet(path)


def test_spreadsheet_formula_injection_and_html_injection_are_neutralised(tmp_path):
    for dangerous in ("=HYPERLINK(\"http://x\")", "+1", "-1", "@cmd", "\tTAB"):
        assert safe_cell(dangerous).startswith("'")
    assert safe_cell("normal") == "normal" and safe_cell(5) == 5
    people = [pdict("a1", name="Ramesh Kumar", record="<img src=x onerror=alert(1)>"), pdict("a2", name="Ramesh Kumar")]
    w = build_world(tmp_path / "w", people)
    summary, root = go(w, tmp_path)
    page = (run_dir(root, summary) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "<img src=x" not in page and "&lt;img src=x" in page


def test_html_report_is_paginated_in_review_order_with_working_navigation(world, tmp_path):
    from dedup_pipeline.scoring.review import write_report

    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)))
    template = records(root, summary)["p01:p02"]
    many = ({**template, "candidate_id": f"x{i:04d}:y{i:04d}"} for i in range(250))
    out = tmp_path / "pages"
    assert write_report(out, many, {"banner": "b", "headline": "h"}) == 250
    pages = sorted(out.glob("page_*.html"))
    assert [p.name for p in pages] == ["page_0001.html", "page_0002.html", "page_0003.html"]
    texts = [p.read_text(encoding="utf-8") for p in pages]
    assert [t.count('<section class="card"') for t in texts] == [100, 100, 50]
    assert "page_0002.html" in texts[0] and "page_0001.html" not in texts[0]  # first page: no Previous
    assert "page_0001.html" in texts[1] and "page_0003.html" in texts[1]
    assert "page_0002.html" in texts[2] and "page_0004.html" not in texts[2] and 'href="#"' not in texts[2]  # last page: no dead Next link
    assert (out / "index.html").read_text(encoding="utf-8").count("page_000") == 3
    from dedup_pipeline.scoring.review import _link

    assert 'href="#"' in _link("javascript:alert(1)", "x") and 'href="#"' in _link("data:text/html,x", "x")  # unsafe schemes are inert
    assert 'href="/api/file-proxy?url=a"' in _link("/api/file-proxy?url=a", "x") and 'href="page_0002.html"' in _link("page_0002.html", "x")
    exact = tmp_path / "exact"
    assert write_report(exact, ({**template, "candidate_id": f"e{i:03d}:f{i:03d}"} for i in range(100)), {"banner": "b", "headline": "h"}) == 100
    assert [p.name for p in exact.glob("page_*.html")] == ["page_0001.html"]


def test_values_can_be_withheld_from_the_review_files(world, tmp_path):
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)), config=V1CONFIG(include_values=False))
    text = (run_dir(root, summary) / "scored_candidates.jsonl").read_text()
    assert "ramesh kumar" not in text.lower() and '"values"' not in text
    default, root2 = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)), root=tmp_path / "other")
    shown = (run_dir(root2, default) / "scored_candidates.jsonl").read_text()
    assert "ramesh kumar" in shown and "******3211" in shown and "9876543211" not in shown  # phones masked; addresses never shown


def test_image_links_use_the_authorised_file_proxy_and_never_embed_image_data(world, tmp_path):
    cfg = V1CONFIG(file_proxy_origin="https://dopams.example", file_server_base_url="http://files.test/files/")
    summary, root = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)), config=cfg)
    images = records(root, summary)["p01:p02"]["images"]
    link = images["a"][0]["file_proxy_link"]
    assert link == "https://dopams.example/api/file-proxy?url=http%3A%2F%2Ffiles.test%2Ffiles%2Fperson%2Fmedia%2Fm01a.png"
    assert images["a"][0]["used_in_face_comparison"] and not images["a"][1]["used_in_face_comparison"] and "base64" not in json.dumps(images)


# --------------------------------------------------------------------- summary content

def test_summary_reports_the_deliverable_numbers(world, tmp_path):
    summary, _ = go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.63)))
    c = summary["candidates"]
    assert c["generated"] == c["scored"] == 6 and sum(c["score_bands_prioritisation_only"].values()) == 6
    assert sum(c["evidence_coverage_percent_distribution"].values()) == 6 and set(c["evidence_coverage_percent_distribution"]) == {"0-20", "20-40", "40-60", "60-80", "80-100"}
    assert summary["score"]["formula"].startswith("match_score = sum(weight x field_score)") and summary["rule_version"] == "score-v1"
    assert summary["score"]["face_transformation"]["cosine_at_or_below"] == 0.2 and summary["score"]["calibrated_probability"] is False
    assert summary["workload"]["face_service_concurrency"] == 8 and summary["workload"]["attribute_queries"] >= 1
    assert summary["elapsed_seconds"] >= 0 and summary["peak_python_heap_mb"] >= 0 and summary["database_writes"] == 0
    assert summary["database"]["transaction_read_only"] == "on"
    check = summary["face"]["transformation_check_pseudo_labels"]
    assert "sanity check" in check["note"]


# ----------------------------------------------------- read-only enforcement and no writes

def test_a_connection_that_is_not_read_only_fails_closed_before_any_output(world, tmp_path):
    root = tmp_path / "scoring"
    with pytest.raises(ReadOnlyViolation):
        go(world, tmp_path, None, db=world.db(read_only="off"), root=root)
    assert not (root / "runs").exists()  # nothing was created, not even a run directory


def test_an_unapproved_database_fails_closed(world, tmp_path):
    with pytest.raises(ReadOnlyViolation, match="unapproved"):
        go(world, tmp_path, None, db=world.db(database="dopams_cctns"), root=tmp_path / "x")


def test_only_select_statements_reach_the_database_and_only_the_two_person_tables(world, tmp_path):
    db = world.db()
    go(world, tmp_path, FakeComparer(world, lambda a, b: outcome(0.7)), db=db)
    assert db.statements and all(s.lstrip().lstrip("/*").split("*/", 1)[-1].strip().split(None, 1)[0].upper() in ("SELECT", "SHOW", "WITH") for s in db.statements)
    attribute_sql = next(s for s in db.statements if "scoring:attributes" in s)
    assert "persons_unified" in attribute_sql and "person_enrichment" in attribute_sql
    assert not any("identity_links" in s or "media_unified" in s for s in db.statements)


def test_the_scoring_package_has_no_write_paths_gallery_calls_or_fingerprinting():
    source = "\n".join(p.read_text(encoding="utf-8-sig") for p in PACKAGE.rglob("*.py")).lower()
    for forbidden in ("insert into", "update public", "delete from", "executemany", "copy_from", ".commit(", "execute_values", "create table",
                      "identity_links", "person_canonical_mapping", "hashlib", "sha256", "imagehash", "phash", "md5", "perceptual",
                      "/search", "/match", "/enroll", "frdopamsfacechecker"):
        assert forbidden not in source, forbidden
    # the only SQL in the package selects from the two person tables
    sql_files = [p.name for p in PACKAGE.rglob("*.py") if "select " in p.read_text(encoding="utf-8-sig").lower() and "scoring:" in p.read_text(encoding="utf-8-sig")]
    assert sql_files == ["attributes.py"]
