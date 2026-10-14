"""Compare a scoring run made with the local face engine against a run made with the HTTP API (same rule version, same inputs).

Reports, per record: candidate id/order, scores, bands, coverage, conflicts, review groups, review status and every field result; the face block
is compared on everything the evidence consists of. Differences that come from *when* the run happened (image-selection notes record what the
stored results already knew at the start of that run) are counted separately and listed, never hidden.

python tools/phase10_compare_runs.py --reference output/scoring/runs/20261012T125026Z --candidate output/scoring_local/runs/score-v2-local --out <file.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

VOLATILE_SUMMARY = {"run_id", "started_at", "finished_at", "elapsed_seconds", "peak_python_heap_mb", "peak_process_memory_mb", "workload", "database", "resumed"}


def load(run: Path):
    return [json.loads(line) for line in open(run / "scored_candidates.jsonl", encoding="ascii")]


def strip_selection(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "image_selection"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref, cand = load(args.reference), load(args.candidate)
    out: dict = {"records": [len(ref), len(cand)], "same_ids_in_same_order": [r["candidate_id"] for r in ref] == [r["candidate_id"] for r in cand]}
    exact = sel_only = other = 0
    other_keys: collections.Counter = collections.Counter()
    examples = []
    for r, c in zip(ref, cand):
        if r == c:
            exact += 1
        elif strip_selection(r) == strip_selection(c):
            sel_only += 1
        else:
            other += 1
            keys = [k for k in r if r.get(k) != c.get(k) and k != "image_selection"]
            other_keys.update(keys)
            if len(examples) < 5:
                examples.append({"candidate_id": r["candidate_id"], "keys": keys})
    out["records_byte_equal"] = exact
    out["records_equal_except_the_image_selection_notes"] = sel_only
    out["records_with_other_differences"] = other
    out["other_difference_keys"] = dict(other_keys)
    out["examples"] = examples
    key = lambda r: (r["candidate_id"], r["match_score_percent"], r["score_band"], r["evidence_coverage_percent"], r["has_hard_conflict"], r["review_group"],
                     json.dumps(r["conflicting_attributes"], sort_keys=True), json.dumps(r["fields"], sort_keys=True), r["review"]["status"], r["review"]["decision"])
    out["scores_bands_coverage_conflicts_groups_fields_status_identical"] = sum(key(r) == key(c) for r, c in zip(ref, cand))
    face_keys = ("status", "raw_cosine", "model", "metric", "field_score", "in_match_score", "failed_side", "side_failures", "error_kind", "fallback_used",
                 "attempts", "media_id_a", "media_id_b", "service_threshold", "relative_to_service_threshold", "reason", "retry_needed")
    out["face_blocks_identical"] = sum(all(r["face"].get(k) == c["face"].get(k) for k in face_keys) for r, c in zip(ref, cand))
    out["all_pending"] = all(c["review"] == {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None} for c in cand)
    # ordered worksheets
    def order(run: Path):
        import csv
        with open(run / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
            return [row["candidate_id"] for row in csv.DictReader(fh)]
    out["review_order_identical"] = order(args.reference) == order(args.candidate)
    sr = json.loads((args.reference / "run_summary.json").read_text(encoding="utf-8"))
    sc = json.loads((args.candidate / "run_summary.json").read_text(encoding="utf-8"))
    for s in (sr, sc):
        for k in VOLATILE_SUMMARY:
            s.pop(k, None)
        s["face"] = {k: v for k, v in s["face"].items() if k not in ("engine", "service_comparison_calls", "service_calls_counted_by_client", "service_seconds",
                                                                    "pairs_evaluated_this_run", "pairs_reused_from_cache")}
    diff = sorted(k for k in set(sr) | set(sc) if sr.get(k) != sc.get(k))
    out["run_summary_keys_that_differ_apart_from_timings_and_engine"] = diff
    out["run_summary_status"] = [sr["status"], sc["status"]]
    out["engine_recorded_in_candidate_run"] = json.loads((args.candidate / "run_summary.json").read_text(encoding="utf-8"))["face"]["engine"]
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "engine_recorded_in_candidate_run"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
