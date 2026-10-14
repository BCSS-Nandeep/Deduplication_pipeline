"""Phase 12: controlled manual-review pilot - preparation and agreement measurement. It never records, invents or changes a decision.

The pilot set is the existing Phase 7 pilot sample (output/audit/phase7/pilot_sample.csv, 120 candidates): 108 scored candidates that are in the
preserved score-v2 worksheet, and 12 "never evaluated" pairs from oversized blocks (recall check) that exist only as Phase 7 audit-only scores.
The sample file and the worksheet are only read.

    dry-run    check that every pilot candidate can be found; writes a JSON summary (ids and counts only), no worksheet
    extract    AFTER AUTHORISATION ONLY: one blank reviewer copy per reviewer label (worksheet layout, empty decision columns) plus the recall-check list
    agreement  after both reviewers finished: validate both copies and measure agreement (raw agreement, Cohen's kappa, confusion matrix, disagreements)

Decisions stay in the reviewers' local files; nothing is written to any database and nothing is merged.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dedup_pipeline.scoring.config import DECISIONS  # noqa: E402
from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V2  # noqa: E402
from dedup_pipeline.scoring.worksheet import validate_worksheet  # noqa: E402

PILOT = ROOT / "output" / "audit" / "phase7" / "pilot_sample.csv"
WORKSHEET = ROOT / "output" / "scoring" / "runs" / "20261012T125026Z" / "review_worksheet.csv"
AUDIT_SCORES = ROOT / "output" / "audit" / "phase7" / "deferred_pairs_audit_scores.jsonl"
RECALL_COLUMNS = ("recall_order", "candidate_id", "person_a_id", "person_b_id", "audit_only_score_NOT_A_PROBABILITY", "audit_only_band", "audit_only_hard_conflict",
                  "audit_only_insufficient", "note", *DECISION_COLUMNS)
RECALL_NOTE = ("Never evaluated by the pipeline (oversized block); no evidence card exists. Open both person records in the authorised DOPAMS session and decide "
               "from the records themselves. The audit-only score was computed by the Phase 7 audit, not by a pipeline run.")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def load_pilot(pilot: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _, rows = read_csv(pilot)
    return [r for r in rows if r["review_order_in_worksheet"]], [r for r in rows if not r["review_order_in_worksheet"]]


def audit_scores(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["pair"] in wanted:
                found[rec["pair"]] = rec
    return found


# --------------------------------------------------------------------------------------------------------- dry run

def dry_run(pilot: Path, worksheet: Path, scores: Path) -> dict[str, Any]:
    in_ws, never = load_pilot(pilot)
    cols, rows = read_csv(worksheet)
    by_id = {r["candidate_id"]: r for r in rows}
    audit = audit_scores(scores, {r["candidate_id"] for r in never})
    missing = sorted([r["candidate_id"] for r in in_ws if r["candidate_id"] not in by_id] + [r["candidate_id"] for r in never if r["candidate_id"] not in audit])
    blank = all(not (by_id[r["candidate_id"]].get(k) or "").strip() for r in in_ws if r["candidate_id"] in by_id for k in DECISION_COLUMNS)
    return {"pilot_candidates": len(in_ws) + len(never), "found_in_worksheet": sum(r["candidate_id"] in by_id for r in in_ws),
            "found_in_audit_scores": sum(r["candidate_id"] in audit for r in never), "missing": missing, "worksheet_layout_known": tuple(cols) == WORKSHEET_COLUMNS_V2,
            "decision_columns_blank_in_source_rows": blank, "strata": dict(sorted(collections.Counter(r["stratum"] for r in in_ws + never).items())),
            "source_pairs": dict(sorted(collections.Counter(r["source_pair"] or "never_evaluated" for r in in_ws + never).items())),
            "pilot_candidate_ids": sorted(r["candidate_id"] for r in in_ws + never), "files_written": "none (dry run)"}


# --------------------------------------------------------------------------------------------------------- extract

def extract(pilot: Path, worksheet: Path, scores: Path, label: str, out_dir: Path) -> dict[str, Any]:
    """Blank reviewer copies. Refuses to overwrite an existing copy (a reviewer's decisions must never be lost)."""
    if not label or not label.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--reviewer-label must be a short alphanumeric label such as A or B (not a person's name)")
    in_ws, never = load_pilot(pilot)
    cols, rows = read_csv(worksheet)
    if tuple(cols) != WORKSHEET_COLUMNS_V2:
        raise SystemExit("the worksheet does not have the expected score-v2 layout")
    by_id = {r["candidate_id"]: r for r in rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    ws_path, recall_path = out_dir / f"pilot_worksheet_{label}.csv", out_dir / f"pilot_recall_check_{label}.csv"
    for p in (ws_path, recall_path):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite a reviewer copy")
    chosen = sorted((by_id[r["candidate_id"]] for r in in_ws), key=lambda r: int(r["review_order"]))
    with open(ws_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in chosen:
            writer.writerow({**r, **{k: "" for k in DECISION_COLUMNS}})
    audit = audit_scores(scores, {r["candidate_id"] for r in never})
    with open(recall_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECALL_COLUMNS)
        writer.writeheader()
        for i, r in enumerate(sorted(never, key=lambda x: x["candidate_id"]), 1):
            a, b = r["candidate_id"].split(":")
            s = audit[r["candidate_id"]]
            writer.writerow({"recall_order": i, "candidate_id": r["candidate_id"], "person_a_id": a, "person_b_id": b, "audit_only_score_NOT_A_PROBABILITY": s["audit_only_score"],
                             "audit_only_band": s["audit_only_band"], "audit_only_hard_conflict": s["hard_conflict"], "audit_only_insufficient": s["insufficient"],
                             "note": RECALL_NOTE, **{k: "" for k in DECISION_COLUMNS}})
    return {"worksheet": str(ws_path), "rows": len(chosen), "recall_check": str(recall_path), "recall_rows": len(never)}


# ------------------------------------------------------------------------------------------------------- agreement

def _iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def recall_decisions(path: Path, known: set[str]) -> dict[str, Any]:
    """The recall-check list uses the same decision rules as the worksheet validator."""
    _, rows = read_csv(path)
    decisions, problems, pending = [], [], 0
    for line, r in enumerate(rows, 2):
        values = {k: (r.get(k) or "").strip() for k in DECISION_COLUMNS}
        if r["candidate_id"] not in known:
            problems.append(f"line {line}: unknown candidate {r['candidate_id']}")
            continue
        if not any(values.values()):
            pending += 1
            continue
        issues = []
        if values["reviewer_decision"] not in DECISIONS:
            issues.append(f"reviewer_decision must be one of {', '.join(DECISIONS)}")
        if not values["reviewer_id"]:
            issues.append("reviewer_id is required")
        if _iso(values["reviewed_at"]) is None:
            issues.append("reviewed_at must be an ISO date/time")
        if issues:
            problems.append(f"line {line}: " + "; ".join(issues))
        else:
            decisions.append({"candidate_id": r["candidate_id"], "decision": values["reviewer_decision"], "reviewer_id": values["reviewer_id"], "reviewed_at": values["reviewed_at"]})
    return {"decisions": decisions, "pending": pending, "problems": problems, "rows": len(rows)}


def cohen_kappa(pairs: list[tuple[str, str]], categories: Iterable[str]) -> Optional[float]:
    cats = list(categories)
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return 1.0 if expected == 1.0 else round((observed - expected) / (1 - expected), 4)


def agreement(a: dict[str, str], b: dict[str, str], strata: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """``a``/``b``: candidate id -> decision of reviewer A / B. Only candidates both decided are compared."""
    both = sorted(set(a) & set(b))
    pairs = [(a[c], b[c]) for c in both]
    binary = [("confirm" if x == "confirm_same_person" else "other", "confirm" if y == "confirm_same_person" else "other") for x, y in pairs]
    out: dict[str, Any] = {
        "decided_by_both": len(both), "decided_by_only_a": len(set(a) - set(b)), "decided_by_only_b": len(set(b) - set(a)),
        "raw_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
        "cohen_kappa_4_categories": cohen_kappa(pairs, DECISIONS),
        "cohen_kappa_confirm_vs_rest": cohen_kappa(binary, ("confirm", "other")),
        "confusion_matrix_a_rows_b_columns": {x: {y: sum(1 for p in pairs if p == (x, y)) for y in DECISIONS} for x in DECISIONS},
        "disagreements_for_adjudication": [{"candidate_id": c, "a": a[c], "b": b[c], **({"stratum": strata.get(c)} if strata else {})} for c in both if a[c] != b[c]],
    }
    if strata:
        per = collections.defaultdict(list)
        for c in both:
            per[strata.get(c, "?")].append(a[c] == b[c])
        out["agreement_by_stratum"] = {s: {"n": len(v), "agree": sum(v)} for s, v in sorted(per.items())}
    return out


def measure(pilot: Path, ws_a: Path, ws_b: Path, recall_a: Optional[Path], recall_b: Optional[Path]) -> dict[str, Any]:
    in_ws, never = load_pilot(pilot)
    known_ws, known_recall = {r["candidate_id"] for r in in_ws}, {r["candidate_id"] for r in never}
    strata = {r["candidate_id"]: r["stratum"] for r in in_ws + never}
    va, vb = validate_worksheet(ws_a, known_ws), validate_worksheet(ws_b, known_ws)
    result: dict[str, Any] = {"validation": {"a": {k: (len(v) if isinstance(v, list) else v) for k, v in va.items() if k != "decisions"} | {"problem_list": va["problems"][:50]},
                                              "b": {k: (len(v) if isinstance(v, list) else v) for k, v in vb.items() if k != "decisions"} | {"problem_list": vb["problems"][:50]}}}
    if va["problems"] or vb["problems"]:
        result["status"] = "fix the validation problems first; agreement is not computed on invalid worksheets"
        return result
    da = {d["candidate_id"]: d["decision"] for d in va["decisions"]}
    db = {d["candidate_id"]: d["decision"] for d in vb["decisions"]}
    reviewers = {"a": sorted({d["reviewer_id"] for d in va["decisions"]}), "b": sorted({d["reviewer_id"] for d in vb["decisions"]})}
    result["reviewer_ids"] = reviewers
    if set(reviewers["a"]) & set(reviewers["b"]):
        result["status"] = "the same reviewer_id appears in both copies: the reviews are not independent"
        return result
    if recall_a and recall_b:
        ra, rb = recall_decisions(recall_a, known_recall), recall_decisions(recall_b, known_recall)
        result["recall_validation"] = {"a": {k: (len(v) if isinstance(v, list) else v) for k, v in ra.items()}, "b": {k: (len(v) if isinstance(v, list) else v) for k, v in rb.items()}}
        if ra["problems"] or rb["problems"]:
            result["status"] = "fix the recall-check validation problems first"
            return result
        da |= {d["candidate_id"]: d["decision"] for d in ra["decisions"]}
        db |= {d["candidate_id"]: d["decision"] for d in rb["decisions"]}
    result["agreement"] = agreement(da, db, strata)
    result["status"] = "computed"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("extract", help="AFTER AUTHORISATION ONLY")
    e.add_argument("--reviewer-label", required=True, help="a neutral label such as A or B; never a person's name")
    e.add_argument("--out-dir", type=Path, required=True)
    m = sub.add_parser("agreement")
    m.add_argument("--a", type=Path, required=True)
    m.add_argument("--b", type=Path, required=True)
    m.add_argument("--recall-a", type=Path)
    m.add_argument("--recall-b", type=Path)
    m.add_argument("--out", type=Path, required=True)
    for p in (d, e, m):
        p.add_argument("--pilot", type=Path, default=PILOT)
    for p in (d, e):
        p.add_argument("--worksheet", type=Path, default=WORKSHEET)
        p.add_argument("--audit-scores", type=Path, default=AUDIT_SCORES)
    args = ap.parse_args()
    if args.command == "dry-run":
        result = dry_run(args.pilot, args.worksheet, args.audit_scores)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "pilot_candidate_ids"}, indent=1))
        return 0 if not result["missing"] else 1
    if args.command == "extract":
        print(json.dumps(extract(args.pilot, args.worksheet, args.audit_scores, args.reviewer_label, args.out_dir), indent=1))
        return 0
    result = measure(args.pilot, args.a, args.b, args.recall_a, args.recall_b)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "agreement"} | ({"raw_agreement": result["agreement"]["raw_agreement"], "kappa": result["agreement"]["cohen_kappa_4_categories"]} if "agreement" in result else {}), indent=1))
    return 0 if result.get("status") == "computed" else 1


if __name__ == "__main__":
    sys.exit(main())
