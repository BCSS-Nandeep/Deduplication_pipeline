"""Phase 12 pilot tooling on synthetic files: blank reviewer copies, no overwrite, validation before agreement, kappa arithmetic.

No real candidate data and no real reviewer identity is used: the labels and ids below are synthetic."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase12_pilot", ROOT / "tools" / "phase12_pilot.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def write_csv(path: Path, columns, rows) -> Path:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def world(tmp_path):
    ws_rows = [{c: "" for c in WORKSHEET_COLUMNS_V2} | {"review_order": str(i), "candidate_id": f"a{i}:b{i}", "review_status": "pending", "match_score_percent_NOT_A_PROBABILITY": "50"}
               for i in range(1, 11)]
    worksheet = write_csv(tmp_path / "worksheet.csv", WORKSHEET_COLUMNS_V2, ws_rows)
    pilot_cols = ("stratum", "candidate_id", "review_order_in_worksheet", "review_group", "score_band", "match_score_percent", "has_hard_conflict",
                  "hard_conflict_fields", "face_status", "candidate_rules", "source_pair", "note")
    pilot_rows = [{c: "" for c in pilot_cols} | {"stratum": "s1" if i < 6 else "s2", "candidate_id": f"a{i}:b{i}", "review_order_in_worksheet": str(i), "source_pair": "V2-V2"}
                  for i in (2, 4, 6, 8)]
    pilot_rows += [{c: "" for c in pilot_cols} | {"stratum": "never", "candidate_id": "x1:y1"}, {c: "" for c in pilot_cols} | {"stratum": "never", "candidate_id": "x2:y2"}]
    pilot = write_csv(tmp_path / "pilot.csv", pilot_cols, pilot_rows)
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps({"pair": p, "audit_only_band": "C", "audit_only_score": 0.6, "hard_conflict": False, "insufficient": False, "status": "x"})
                                for p in ("x1:y1", "x2:y2", "x3:y3")) + "\n", encoding="utf-8")
    return tmp_path, pilot, worksheet, scores


def test_dry_run_finds_everything_and_writes_no_worksheet(world):
    tmp, pilot, ws, scores = world
    before = sorted(p.name for p in tmp.iterdir())
    r = tool.dry_run(pilot, ws, scores)
    assert (r["found_in_worksheet"], r["found_in_audit_scores"], r["missing"], r["pilot_candidates"]) == (4, 2, [], 6)
    assert r["worksheet_layout_known"] and r["decision_columns_blank_in_source_rows"] and sorted(p.name for p in tmp.iterdir()) == before


def test_extract_writes_blank_copies_in_the_validator_layout_and_never_overwrites(world):
    tmp, pilot, ws, scores = world
    out = tmp / "copies"
    r = tool.extract(pilot, ws, scores, "A", out)
    assert r["rows"] == 4 and r["recall_rows"] == 2
    cols, rows = tool.read_csv(out / "pilot_worksheet_A.csv")
    assert tuple(cols) == WORKSHEET_COLUMNS_V2 and [x["candidate_id"] for x in rows] == ["a2:b2", "a4:b4", "a6:b6", "a8:b8"]
    assert all(not x[k] for x in rows for k in DECISION_COLUMNS)
    v = tool.validate_worksheet(out / "pilot_worksheet_A.csv", {"a2:b2", "a4:b4", "a6:b6", "a8:b8"})
    assert (v["pending"], v["decisions"], v["problems"]) == (4, [], [])
    _, recall = tool.read_csv(out / "pilot_recall_check_A.csv")
    assert [x["candidate_id"] for x in recall] == ["x1:y1", "x2:y2"] and all(not x[k] for x in recall for k in DECISION_COLUMNS) and recall[0]["person_a_id"] == "x1"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        tool.extract(pilot, ws, scores, "A", out)
    with pytest.raises(SystemExit, match="label"):
        tool.extract(pilot, ws, scores, "Jane Doe", out)  # a person's name is refused as a label


def fill(path: Path, decisions: dict[str, str], reviewer: str) -> None:
    cols, rows = tool.read_csv(path)
    for r in rows:
        if r["candidate_id"] in decisions:
            r.update({"reviewer_decision": decisions[r["candidate_id"]], "reviewer_id": reviewer, "reviewed_at": "2026-10-14T10:00:00+05:30"})
    write_csv(path, cols, rows)


def test_agreement_is_measured_only_on_valid_independent_copies(world):
    tmp, pilot, ws, scores = world
    for label in ("A", "B"):
        tool.extract(pilot, ws, scores, label, tmp / "copies")
    a, b = tmp / "copies" / "pilot_worksheet_A.csv", tmp / "copies" / "pilot_worksheet_B.csv"
    fill(a, {"a2:b2": "confirm_same_person", "a4:b4": "reject_match", "a6:b6": "confirm_same_person", "a8:b8": "needs_further_review"}, "synthetic-reviewer-1")
    fill(b, {"a2:b2": "confirm_same_person", "a4:b4": "reject_match", "a6:b6": "reject_match", "a8:b8": "needs_further_review"}, "synthetic-reviewer-2")
    ra, rb = tmp / "copies" / "pilot_recall_check_A.csv", tmp / "copies" / "pilot_recall_check_B.csv"
    fill(ra, {"x1:y1": "reject_match", "x2:y2": "unable_to_verify"}, "synthetic-reviewer-1")
    fill(rb, {"x1:y1": "reject_match", "x2:y2": "unable_to_verify"}, "synthetic-reviewer-2")
    result = tool.measure(pilot, a, b, ra, rb)
    assert result["status"] == "computed"
    ag = result["agreement"]
    assert ag["decided_by_both"] == 6 and ag["raw_agreement"] == round(5 / 6, 4)
    assert [d["candidate_id"] for d in ag["disagreements_for_adjudication"]] == ["a6:b6"] and ag["disagreements_for_adjudication"][0]["stratum"] == "s2"
    assert ag["confusion_matrix_a_rows_b_columns"]["confirm_same_person"]["reject_match"] == 1
    # the same reviewer in both copies is not an independent review
    fill(b, {"a2:b2": "confirm_same_person"}, "synthetic-reviewer-1")
    assert "not independent" in tool.measure(pilot, a, b, None, None)["status"]


def test_invalid_entries_block_the_agreement(world):
    tmp, pilot, ws, scores = world
    for label in ("A", "B"):
        tool.extract(pilot, ws, scores, label, tmp / "copies")
    a, b = tmp / "copies" / "pilot_worksheet_A.csv", tmp / "copies" / "pilot_worksheet_B.csv"
    cols, rows = tool.read_csv(a)
    rows[0].update({"reviewer_decision": "probably_same", "reviewer_id": "", "reviewed_at": "yesterday"})
    write_csv(a, cols, rows)
    result = tool.measure(pilot, a, b, None, None)
    assert "fix the validation problems" in result["status"] and "agreement" not in result and result["validation"]["a"]["problems"] == 1


def test_cohen_kappa_matches_hand_calculation():
    # 10 items: A says yes 6 / no 4, B says yes 5 / no 5; they agree on 7 -> po 0.7, pe 0.6*0.5+0.4*0.5 = 0.5, kappa 0.4
    pairs = [("y", "y")] * 4 + [("y", "n")] * 2 + [("n", "y")] * 1 + [("n", "n")] * 3
    assert tool.cohen_kappa(pairs, ("y", "n")) == 0.4
    assert tool.cohen_kappa([], ("y", "n")) is None and tool.cohen_kappa([("y", "y")] * 3, ("y", "n")) == 1.0
