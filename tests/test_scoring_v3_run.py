"""score-v3 end to end: same candidates and pending status, FIR-anchored age evidence in every output, v1/v2 preserved, read-only."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from dedup_pipeline.db import ReadOnlyViolation
from dedup_pipeline.scoring import ScoringOptions, run_scoring
from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V2, WORKSHEET_COLUMNS_V3
from dedup_pipeline.scoring.worksheet import validate_worksheet

from .scoring_fakes import V1CONFIG, V2CONFIG, V3CONFIG, backend, build_world, pdict

D = date


def fir(person, when):
    return [(f"acc-{person}", f"crime-{person}", when)]


def people():
    base = dict(relative="Raju Rao", gender="Male")
    return [
        # x: ages that look 8 years apart at one date, but were recorded at FIRs 8 years apart -> consistent in v3
        pdict("x1", name="Anil Kumar", age=24, phone="9876500001", fir=fir("x1", D(2018, 6, 1)), **base),
        pdict("x2", name="Anil Kumar", age=32, phone="9876500001", fir=fir("x2", D(2026, 6, 1)), **base),
        # y: equal ages recorded at FIRs 8 years apart -> 8 years apart in birth year (v2 sees no difference)
        pdict("y1", name="Bala Reddy", age=24, phone="9876500002", fir=fir("y1", D(2018, 6, 1)), **base),
        pdict("y2", name="Bala Reddy", age=24, phone="9876500002", fir=fir("y2", D(2026, 6, 1)), **base),
        # z: one record has no FIR link -> its age is unavailable in v3
        pdict("z1", name="Chandu Rao", age=30, phone="9876500003", fir=fir("z1", D(2024, 6, 1)), **base),
        pdict("z2", name="Chandu Rao", age=30, phone="9876500003", fir=[], **base),
        # m: one record linked to two FIRs ten years apart, one age
        pdict("m1", name="Dinesh Babu", age=30, phone="9876500004", fir=[("acc-m1a", "crime-m1a", D(2016, 6, 1)), ("acc-m1b", "crime-m1b", D(2026, 6, 1))], **base),
        pdict("m2", name="Dinesh Babu", age=30, phone="9876500004", fir=fir("m2", D(2024, 6, 1)), **base),
        # d: a DOB against an age recorded at a FIR
        pdict("d1", name="Eswar Naidu", age=None, dob=D(1996, 3, 3), phone="9876500005", fir=fir("d1", D(2024, 6, 1)), **base),
        pdict("d2", name="Eswar Naidu", age=26, phone="9876500005", fir=fir("d2", D(2022, 3, 1)), **base),
    ]


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path / "w", people())


def go(world, tmp_path, config, label, root=None, db=None):
    root = root or tmp_path / "scoring"
    summary = run_scoring(db or world.db(), config, ScoringOptions(world.candidate_dir, world.image_dir, root, label, True), backend(world, None, None))
    return summary, root


def run_dir(root, summary) -> Path:
    return root / "runs" / summary["run_id"]


def records(root, summary):
    path = run_dir(root, summary) / "scored_candidates.jsonl"
    return {r["candidate_id"]: r for r in map(json.loads, path.read_text(encoding="ascii").splitlines())}


def rows_of(root, summary):
    with open(run_dir(root, summary) / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def both(world, tmp_path):
    s2, root = go(world, tmp_path, V2CONFIG(), "run-v2")
    s3, _ = go(world, tmp_path, V3CONFIG(), "run-v3", root)
    return s2, s3, root


# ------------------------------------------------------------------ same candidates, still pending

def test_the_same_candidates_are_rescored_and_all_stay_pending(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2), records(root, s3)
    assert set(r2) == set(r3) and len(r3) == s3["candidates"]["generated"] == s3["candidates"]["scored"] == s2["candidates"]["scored"] == 5
    assert all(r["review"] == {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None} for r in r3.values())
    assert all(r["rule_version"] == "score-v3" for r in r3.values()) and all(r["rule_version"] == "score-v2" for r in r2.values())
    assert s3["human_review"]["decisions_recorded"] == 0 and s3["candidates"]["reviewed"] == 0
    assert all(all(r[c] == "" for c in DECISION_COLUMNS) for r in rows_of(root, s3))


def test_only_the_age_field_changes_between_v2_and_v3(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2), records(root, s3)
    for cid in r2:
        for name in ("name", "alias", "relative_name", "phone", "gender", "address", "face"):
            assert r2[cid]["fields"][name] == r3[cid]["fields"][name], (cid, name)


# ------------------------------------------------------------------------- the age changes, explained

def test_ages_recorded_at_different_fir_years_stop_being_a_conflict(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2)["x1:x2"], records(root, s3)["x1:x2"]
    assert r2["fields"]["age_dob"]["conflict"] == "hard" and r2["review_group"] == "2_hard_conflict"
    age = r3["age_dob_summary"]
    assert r3["fields"]["age_dob"].get("conflict") is None and r3["has_hard_conflict"] is False and r3["review_group"] == "1_no_hard_conflict"
    assert (age["method"], age["comparison_status"]) == ("estimated_birth_year_ranges_at_fir_date", "consistent")
    assert age["birth_year_range_a"] == [1993, 1994] and age["birth_year_range_b"] == [1993, 1994] and age["evidence"]["fir_years_a"] == [2018] and age["evidence"]["fir_years_b"] == [2026]
    assert r3["match_score_percent"] > r2["match_score_percent"]


def test_equal_ages_at_fir_years_far_apart_become_a_conflict(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2)["y1:y2"], records(root, s3)["y1:y2"]
    assert r2["fields"]["age_dob"].get("conflict") is None and r3["fields"]["age_dob"]["conflict"] == "hard"
    assert r3["review_group"] == "2_hard_conflict" and r3["match_score_percent"] < r2["match_score_percent"]
    assert [c["field"] for c in r3["hard_conflicts"]] == ["age_dob"]


def test_an_age_without_a_fir_date_is_unavailable_evidence_and_lowers_coverage(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2)["z1:z2"], records(root, s3)["z1:z2"]
    assert r2["fields"]["age_dob"]["applicable"] is True and r3["fields"]["age_dob"]["applicable"] is False
    assert r3["fields"]["age_dob"]["not_applicable_reason"].startswith("age_evidence_unavailable:") and "no_fir_link" in r3["fields"]["age_dob"]["not_applicable_reason"]
    assert r3["evidence_coverage_percent"] < r2["evidence_coverage_percent"] and r3["age_dob_summary"]["applicable"] is False
    assert {"field": "age_dob", "reason": r3["fields"]["age_dob"]["not_applicable_reason"], "explanation": r3["fields"]["age_dob"]["explanation"]} in r3["missing_evidence"]


def test_contradicting_fir_observations_are_flagged_for_a_human(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r3 = records(root, s3)["m1:m2"]
    assert r3["fields"]["age_dob"]["applicable"] is False and "age_observations_conflict" in r3["fields"]["age_dob"]["not_applicable_reason"]
    assert "conflicting_age_observations_record_a" in r3["processing"]["indicators"]
    assert any(c["severity"] == "data_quality" and "cannot be reconciled" in c["explanation"] for c in r3["conflicting_attributes"])
    assert r3["age_dob_summary"]["review_flags"][0]["record"] == "a"
    row = next(r for r in rows_of(root, s3) if r["candidate_id"] == "m1:m2")
    assert "conflicting_age_observations" in row["age_review_flags"] and "2016" in row["age_evidence_a"] and "2026" in row["age_evidence_a"]


def test_a_dob_is_compared_with_the_age_at_the_fir_date_and_kept_separate(world, tmp_path):
    s2, s3, root = both(world, tmp_path)
    r2, r3 = records(root, s2)["d1:d2"], records(root, s3)["d1:d2"]
    age = r3["age_dob_summary"]
    assert age["method"] == "dob_vs_recorded_age_at_fir_date" and age["gap_years"] == 1 and r3["fields"]["age_dob"].get("conflict") is None
    assert age["dob_comparison_status"].startswith("not_compared_dob_on_one_record_only")
    assert r2["age_dob_summary"]["method"] == "dob_vs_recorded_age" and r2["age_dob_summary"]["gap_years"] == 4  # v2 reads 26 at today's date
    assert r2["fields"]["age_dob"]["conflict"] == "soft" and r3["fields"]["age_dob"].get("conflict") is None


def test_the_summary_records_the_method_policy_and_provenance(world, tmp_path):
    _, s3, root = both(world, tmp_path)
    block = s3["age_dob"]
    assert block["anchoring"].startswith("each recorded age is anchored to the date of the FIR") and block["fir_timezone"] == "Asia/Kolkata"
    assert block["birth_year_tolerance_years"] == 1 and block["observation_conflict_policy"] == "unavailable"
    ev = s3["age_evidence_fir_anchored"]
    assert ev["persons"] == 10 and ev["persons_with_multiple_fir_links"] == 1 and ev["fir_link_rows_read"] == 10  # m1 has two links, z2 none
    assert ev["observation_status_per_person"]["conflicting"] == 1
    assert ev["age_unavailable_reasons_per_observation"] == {"age_missing": 1, "no_fir_link": 1}  # d1 has a DOB and no age; z2 has no FIR
    assert ev["review_flags"] == {"conflicting_age_observations": 1} and s3["database_writes"] == 0
    assert s3["score"]["calibrated_probability"] is False


# ------------------------------------------------------------------------------- provenance file

def test_the_age_evidence_provenance_has_one_line_per_person_and_no_names(world, tmp_path):
    _, s3, root = both(world, tmp_path)
    path = run_dir(root, s3) / "age_evidence_provenance.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="ascii").splitlines()]
    assert sorted(x["person_id"] for x in lines) == sorted(p["person_id"] for p in people()) and len({x["person_id"] for x in lines}) == 10
    m1 = next(x for x in lines if x["person_id"] == "m1")
    assert m1["observation_status"] == "conflicting" and m1["merged_birth_year_range"] is None and len(m1["observations"]) == 2
    assert {o["crime_id"] for o in m1["observations"]} == {"crime-m1a", "crime-m1b"} and all(o["usable"] and o["birth_year_range"] for o in m1["observations"])
    assert m1["review_flags"] == ["conflicting_age_observations"]
    z2 = next(x for x in lines if x["person_id"] == "z2")
    assert z2["age_evidence_state"] == "no_fir_link" and z2["observations"][0]["unavailable_reason"] == "no_fir_link" and z2["observations"][0]["usable"] is False
    x2 = next(x for x in lines if x["person_id"] == "x2")
    assert x2["observations"][0]["fir_date"] == "2026-06-01" and x2["observations"][0]["birth_year_range"] == [1993, 1994]
    text = path.read_text(encoding="ascii")
    assert "Anil Kumar" not in text and "Raju Rao" not in text and "9876500001" not in text
    summary = json.loads((run_dir(root, s3) / "age_evidence_summary.json").read_text())
    assert summary["persons"] == 10 and summary["age_source"] == "persons_unified.age" and "never used as the date an age was recorded" in summary["reference_date_use"]


# ------------------------------------------------------------------------------------ outputs

def test_the_worksheet_and_report_show_the_fir_anchoring_and_pass_validation(world, tmp_path):
    _, s3, root = both(world, tmp_path)
    rows = rows_of(root, s3)
    assert list(rows[0].keys()) == list(WORKSHEET_COLUMNS_V3) and WORKSHEET_COLUMNS_V3 != WORKSHEET_COLUMNS_V2
    assert set(WORKSHEET_COLUMNS_V2) <= set(WORKSHEET_COLUMNS_V3)
    row = next(r for r in rows if r["candidate_id"] == "x1:x2")
    assert (row["fir_year_a"], row["fir_year_b"], row["rule_version"]) == ("2018", "2026", "score-v3")
    assert "FIR 2018-06-01" in row["age_evidence_a"] and "birth years 1993-1994" in row["age_evidence_a"]
    assert row["age_comparison_method"] == "estimated_birth_year_ranges_at_fir_date / consistent"
    assert validate_worksheet(run_dir(root, s3) / "review_worksheet.csv")["pending"] == 5
    page = (run_dir(root, s3) / "review_report" / "page_0001.html").read_text(encoding="utf-8")
    assert "Age evidence anchored to FIR dates" in page and "conflicting_age_observations" in page and "no usable FIR date" in page
    assert "FIR the person is linked to" in (run_dir(root, s3) / "REVIEW_INSTRUCTIONS.md").read_text(encoding="utf-8")
    cfg = json.loads((run_dir(root, s3) / "scoring_config.json").read_text())
    assert cfg["rule_version"] == "score-v3" and cfg["config"]["birth_year_conflict_gap_years"] == 4 and cfg["config"]["fir_timezone"] == "Asia/Kolkata"


# --------------------------------------------------------------------------------- preservation

def digest(directory: Path) -> dict[str, str]:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.rglob("*")) if f.is_file()}


def test_a_v3_run_leaves_v1_and_v2_runs_byte_identical_and_uses_its_own_directory(world, tmp_path):
    s1, root = go(world, tmp_path, V1CONFIG(), "run-v1")
    s2, _ = go(world, tmp_path, V2CONFIG(), "run-v2", root)
    before = {s["run_id"]: digest(run_dir(root, s)) for s in (s1, s2)}
    s3, _ = go(world, tmp_path, V3CONFIG(), "run-v3", root)
    assert run_dir(root, s3) not in (run_dir(root, s1), run_dir(root, s2)) and s3["resumed"] is False
    assert {s["run_id"]: digest(run_dir(root, s)) for s in (s1, s2)} == before
    assert not (run_dir(root, s1) / "age_evidence_provenance.jsonl").exists() and not (run_dir(root, s2) / "age_evidence_provenance.jsonl").exists()
    assert "age_evidence_fir_anchored" not in s1 and "age_evidence_fir_anchored" not in s2
    assert not any(k in json.loads((run_dir(root, s2) / "scoring_config.json").read_text())["config"] for k in ("fir_timezone", "birth_year_tolerance_years"))


def test_v1_and_v2_do_not_issue_the_fir_query_and_v3_does(world, tmp_path):
    for config, expected in ((V1CONFIG(), 0), (V2CONFIG(), 0), (V3CONFIG(), 1)):
        db = world.db()
        go(world, tmp_path, config, f"q-{config.rule_version}", tmp_path / f"root-{config.rule_version}", db=db)
        assert sum("scoring:fir_links" in s for s in db.statements) == expected


def test_v3_reruns_are_deterministic(world, tmp_path):
    s_a, root = go(world, tmp_path, V3CONFIG(), "first")
    s_b, _ = go(world, tmp_path, V3CONFIG(), "second", root)
    a, b = digest(run_dir(root, s_a)), digest(run_dir(root, s_b))
    for name in ("scored_candidates.jsonl", "review_worksheet.csv", "age_evidence_provenance.jsonl", "age_evidence_summary.json", "scoring_config.json"):
        assert a[name] == b[name], name
    assert {k: v for k, v in a.items() if k.startswith("review_report")} == {k: v for k, v in b.items() if k.startswith("review_report")}


# --------------------------------------------------------------------------- comparison report

def test_the_v2_vs_v3_comparison_explains_every_age_change_and_writes_only_into_the_v3_run(world, tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_v2_v3", Path(__file__).resolve().parents[1] / "tools" / "compare_v2_v3.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    s2, s3, root = both(world, tmp_path)
    before = digest(run_dir(root, s2))
    monkeypatch.setattr("sys.argv", ["compare", "--v2", str(run_dir(root, s2)), "--v3", str(run_dir(root, s3))])
    assert tool.main() == 0
    out = json.loads((run_dir(root, s3) / "comparison_v2_vs_v3.json").read_text())
    assert out["candidates"] == 5 and out["same_candidate_set"] is True and out["v3_review_status"] == {"pending": 5} and out["non_age_fields_identical"] is True
    t = out["age_state_transitions_v2_to_v3"]
    # x pair cleared; y pair newly conflicting; z pair (no FIR link) and m pair (contradicting FIR observations) lost; d pair soft -> no conflict
    assert t == {"hard->no_conflict": 1, "no_conflict->hard": 1, "no_conflict->unavailable": 2, "soft->no_conflict": 1}
    assert out["age_only_hard_conflicts"]["v2_candidates"] == 1 and out["age_only_hard_conflicts"]["v3_candidates"] == 1
    assert out["age_only_hard_conflicts"]["v3_outcome_of_the_v2_set"] == {"no hard conflict in v3 (age no longer a conflict)": 1}
    assert out["age_only_hard_conflicts"]["v3_new_age_only_conflicts_by_v2_state"] == {"no_conflict": 1}
    explained = {k: v for k, v in out["age_conflict_changes_explained"].items()}
    assert "hard -> no_conflict" in explained and "no_conflict -> unavailable" in explained
    assert (run_dir(root, s3) / "COMPARISON_V2_VS_V3.md").read_text(encoding="utf-8").startswith("# score-v2 vs score-v3")
    with open(run_dir(root, s3) / "age_changes_v2_vs_v3.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["candidate_id"] for r in rows} >= {"x1:x2", "y1:y2", "z1:z2"} and all(r["explanation"] for r in rows)
    assert "Anil Kumar" not in (run_dir(root, s3) / "age_changes_v2_vs_v3.csv").read_text(encoding="utf-8")
    assert digest(run_dir(root, s2)) == before  # the v2 run was only read


# ------------------------------------------------------------------------------------ safety

def test_v3_issues_only_read_only_statements(world, tmp_path):
    db = world.db()
    s3, _ = go(world, tmp_path, V3CONFIG(), "ro", db=db)
    first_words = [s.lstrip().lstrip("/*").split("*/", 1)[-1].strip().split(None, 1)[0].upper() for s in db.statements]
    assert db.statements and all(w in ("SELECT", "SHOW", "WITH") for w in first_words)
    assert s3["database_writes"] == 0 and s3["database"]["transaction_read_only"] == "on"


def test_v3_refuses_a_connection_that_is_not_read_only(world, tmp_path):
    root = tmp_path / "scoring"
    with pytest.raises(ReadOnlyViolation):
        go(world, tmp_path, V3CONFIG(), "bad", root, db=world.db(read_only="off"))
    assert not (root / "runs").exists()
