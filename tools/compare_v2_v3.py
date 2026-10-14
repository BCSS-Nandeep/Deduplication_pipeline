"""Compare a score-v2 run with a score-v3 run of the same candidates. Read-only on both runs; writes only into the v3 run directory.

python tools/compare_v2_v3.py --v2 output/scoring/runs/<v2> --v3 output/scoring_v3/runs/<v3>

Outputs (in the v3 run directory): comparison_v2_vs_v3.json, COMPARISON_V2_VS_V3.md, age_changes_v2_vs_v3.csv (one row per candidate whose age
evidence changed; identifiers and statuses only - no names, ages or dates of birth), age_conflict_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

BANDS = ("A", "B", "C", "D", "E")


def stream(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="ascii") as fh:
        for line in fh:
            yield json.loads(line)


def age_state(rec: dict[str, Any]) -> str:
    """none | soft | hard | unavailable - the state of the age/DOB field in one record."""
    f = rec["fields"]["age_dob"]
    if not f["applicable"]:
        return "unavailable"
    return f.get("conflict") or "no_conflict"


def slim(rec: dict[str, Any]) -> dict[str, Any]:
    f = rec["fields"]
    age = rec.get("age_dob_summary") or {}
    hard_fields = sorted(c["field"] for c in rec["hard_conflicts"])
    soft_fields = sorted(n for n, x in f.items() if x.get("conflict") == "soft")
    return {
        "score": rec["match_score_percent"], "band": rec["score_band"], "coverage": rec["evidence_coverage_percent"], "group": rec["review_group"],
        "hard": hard_fields, "soft": soft_fields, "age_state": age_state(rec), "age_score": f["age_dob"]["score"], "age_applicable": f["age_dob"]["applicable"],
        "method": age.get("method"), "status": age.get("comparison_status"), "gap": age.get("gap_years"),
        "reason": f["age_dob"].get("not_applicable_reason"), "insufficient": rec["insufficient_evidence"], "source_pair": rec["source_pair"],
        "evidence": age.get("evidence") or {}, "flags": [x["issue"] for x in age.get("review_flags", [])],
        "dob_a": age.get("record_a", {}).get("dob_status"), "dob_b": age.get("record_b", {}).get("dob_status"),
        "strong": {"name_exact", "phone_exact"} <= set(rec["candidate_generation"]["rule_names"]),
    }


def ranks(run: Path) -> dict[str, int]:
    with open(run / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        return {r["candidate_id"]: int(r["review_order"]) for r in csv.DictReader(fh)}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def fir_gap_bucket(ev: dict[str, Any]) -> str:
    a, b = ev.get("fir_years_a"), ev.get("fir_years_b")
    if not a or not b:
        return "n/a (a DOB was compared)"
    gap = min(abs(x - y) for x in a for y in b)
    return "0 (same year)" if gap == 0 else ("1" if gap == 1 else ("2-3" if gap <= 3 else "4+"))


def explain(v2: dict[str, Any], v3: dict[str, Any], detailed: bool = True) -> str:
    """One sentence per kind of change. ``detailed`` adds the numbers (CSV, per candidate); the summary tables group on the text without them."""
    s2, s3 = v2["age_state"], v3["age_state"]
    if s3 == "unavailable":
        r = (v3["reason"] or "").split(":", 1)[-1]
        why = ("an age was recorded but the record has no FIR link" if "no_fir_link" in r else
               "the age is unusable or missing on one record" if "age_missing" in r or "age_invalid" in r else
               "a record's age observations contradict each other" if "age_observations_conflict" in v3["reason"] else
               "a FIR date is missing or invalid" if "fir_date" in r else "no age or DOB on one record")
        return f"v3 cannot anchor the age ({why}); age is unavailable evidence, not compared"
    if v2["age_state"] == "unavailable":
        return "age comparable in v3 but not in v2"
    if s2 == s3:
        return "same conflict class"
    gap = fir_gap_bucket(v3["evidence"])
    if v3["method"] == "dob_vs_recorded_age_at_fir_date":
        return (f"a DOB was compared with the age recorded at the other record's FIR date instead of at the reference date: {s2} -> {s3}"
                + (f" (v3 gap {v3['gap']} year(s), v2 gap {v2['gap']} year(s))" if detailed else ""))
    if s2 in ("hard", "soft") and s3 in ("hard", "soft"):
        return (f"read at each record's FIR date (FIR years apart: {gap}) the possible birth years are "
                f"{'further apart' if s3 == 'hard' else 'closer'} than the single-date age gap: {s2} -> {s3}"
                + (f" (v2 age gap {v2['gap']} year(s); v3 birth-year range distance {v3['evidence'].get('birth_year_range_distance_years')} year(s))" if detailed else ""))
    if s3 == "no_conflict" and s2 in ("hard", "soft"):
        return f"v2 read both ages at one date; at each record's FIR date (FIR years apart: {gap}) the possible birth years are consistent"
    if s3 in ("hard", "soft") and s2 in ("no_conflict",):
        return f"similar ages, but recorded at FIR dates {gap} year(s) apart, so the possible birth years differ"
    if s3 == "no_conflict":
        return "conflict cleared"
    return f"conflict class changed ({s2} -> {s3}); FIR years apart: {gap}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--v3", type=Path, required=True)
    args = ap.parse_args()
    v2 = {r["candidate_id"]: slim(r) for r in stream(args.v2 / "scored_candidates.jsonl")}
    pend = Counter()
    v3 = {}
    for r in stream(args.v3 / "scored_candidates.jsonl"):
        v3[r["candidate_id"]] = slim(r)
        pend[r["review"]["status"]] += 1
    assert set(v2) == set(v3), "the two runs do not contain the same candidates"
    n = len(v3)
    rk2, rk3 = ranks(args.v2), ranks(args.v3)
    out: dict[str, Any] = {"candidates": n, "same_candidate_set": True, "v3_review_status": dict(pend)}

    # ---- scores
    deltas = [v3[c]["score"] - v2[c]["score"] for c in v2 if v3[c]["score"] is not None and v2[c]["score"] is not None]
    nz = [d for d in deltas if abs(d) > 1e-9]
    out["score"] = {
        "v2_mean": round(statistics.mean(v2[c]["score"] or 0 for c in v2), 2), "v3_mean": round(statistics.mean(v3[c]["score"] or 0 for c in v3), 2),
        "unchanged": n - len(nz), "increased": sum(d > 0 for d in nz), "decreased": sum(d < 0 for d in nz),
        "mean_abs_change_points": round(statistics.mean(abs(d) for d in deltas), 3), "max_increase": round(max(deltas), 2), "max_decrease": round(min(deltas), 2),
        "change_distribution": dict(sorted(Counter(("<= -10" if d <= -10 else "-10..-5" if d <= -5 else "-5..-1" if d <= -1 else "-1..0" if d < -1e-9 else "0" if abs(d) <= 1e-9
                                                    else "0..1" if d < 1 else "1..5" if d < 5 else "5..10" if d < 10 else ">= 10") for d in deltas).items())),
    }
    band_t = Counter((v2[c]["band"], v3[c]["band"]) for c in v2)
    out["bands"] = {"v2": {b: sum(v2[c]["band"] == b for c in v2) for b in BANDS}, "v3": {b: sum(v3[c]["band"] == b for c in v3) for b in BANDS},
                    "changed": sum(x for (a, b), x in band_t.items() if a != b), "transitions": {f"{a}->{b}": x for (a, b), x in sorted(band_t.items()) if a != b}}
    out["coverage"] = {"v2_mean_percent": round(statistics.mean(v2[c]["coverage"] for c in v2), 2), "v3_mean_percent": round(statistics.mean(v3[c]["coverage"] for c in v3), 2),
                       "age_field_applicable_v2": sum(v2[c]["age_applicable"] for c in v2), "age_field_applicable_v3": sum(v3[c]["age_applicable"] for c in v3),
                       "coverage_lower_in_v3": sum(v3[c]["coverage"] < v2[c]["coverage"] - 1e-9 for c in v2), "coverage_higher_in_v3": sum(v3[c]["coverage"] > v2[c]["coverage"] + 1e-9 for c in v2),
                       "insufficient_evidence_v2": sum(v2[c]["insufficient"] for c in v2), "insufficient_evidence_v3": sum(v3[c]["insufficient"] for c in v3),
                       "became_insufficient_in_v3": sum(v3[c]["insufficient"] and not v2[c]["insufficient"] for c in v2),
                       "no_longer_insufficient_in_v3": sum(v2[c]["insufficient"] and not v3[c]["insufficient"] for c in v2)}

    # ---- conflicts
    def conflict_counts(d: dict[str, dict]) -> dict[str, Any]:
        hard, soft = Counter(), Counter()
        for x in d.values():
            hard.update(x["hard"])
            soft.update(x["soft"])
        return {"candidates_with_a_hard_conflict": sum(bool(x["hard"]) for x in d.values()), "hard_by_field": dict(sorted(hard.items())),
                "soft_by_field": dict(sorted(soft.items())), "candidates_with_a_soft_conflict": sum(bool(x["soft"]) for x in d.values()),
                "age_field_state": dict(sorted(Counter(x["age_state"] for x in d.values()).items()))}
    out["conflicts"] = {"v2": conflict_counts(v2), "v3": conflict_counts(v3)}
    out["age_state_transitions_v2_to_v3"] = {f"{a}->{b}": x for (a, b), x in sorted(Counter((v2[c]["age_state"], v3[c]["age_state"]) for c in v2).items())}
    out["non_age_fields_identical"] = all([x for x in v2[c]["hard"] if x != "age_dob"] == [x for x in v3[c]["hard"] if x != "age_dob"] for c in v2)

    # ---- review groups and order
    grp_t = Counter((v2[c]["group"], v3[c]["group"]) for c in v2)
    out["review_groups"] = {"v2": dict(sorted(Counter(v2[c]["group"] for c in v2).items())), "v3": dict(sorted(Counter(v3[c]["group"] for c in v3).items())),
                            "moved": sum(x for (a, b), x in grp_t.items() if a != b), "transitions": {f"{a} -> {b}": x for (a, b), x in sorted(grp_t.items()) if a != b}}
    moves = [abs(rk3[c] - rk2[c]) for c in v2]
    top = lambda rk, k: {c for c, r in rk.items() if r <= k}
    out["review_order"] = {"mean_abs_rank_change": round(statistics.mean(moves), 1), "median_abs_rank_change": statistics.median(moves),
                           "top_500_overlap": len(top(rk2, 500) & top(rk3, 500)), "top_1000_overlap": len(top(rk2, 1000) & top(rk3, 1000)),
                           "top_2000_overlap": len(top(rk2, 2000) & top(rk3, 2000)),
                           "entered_top_1000": len(top(rk3, 1000) - top(rk2, 1000)), "left_top_1000": len(top(rk2, 1000) - top(rk3, 1000))}
    # why did the top of the queue change: for those that entered the top 1000, the v2 age state
    out["review_order"]["entered_top_1000_by_v2_age_state"] = dict(sorted(Counter(v2[c]["age_state"] for c in top(rk3, 1000) - top(rk2, 1000)).items()))
    out["review_order"]["left_top_1000_by_v3_age_state"] = dict(sorted(Counter(v3[c]["age_state"] for c in top(rk2, 1000) - top(rk3, 1000)).items()))

    # ---- age-related changes, explained
    changed = [c for c in v2 if (v2[c]["age_state"], v3[c]["age_state"]) != ("no_conflict", "no_conflict") or v2[c]["method"] != v3[c]["method"]]
    reasons = defaultdict(Counter)
    for c in v2:
        if v2[c]["age_state"] != v3[c]["age_state"]:
            reasons[f"{v2[c]['age_state']} -> {v3[c]['age_state']}"][explain(v2[c], v3[c], detailed=False)] += 1
    out["age_conflict_changes_explained"] = {k: dict(v.most_common()) for k, v in sorted(reasons.items())}
    by_gap = defaultdict(Counter)
    for c in v2:
        if v3[c]["age_applicable"] and v2[c]["age_applicable"]:
            by_gap[fir_gap_bucket(v3[c]["evidence"])][f"{v2[c]['age_state']} -> {v3[c]['age_state']}"] += 1
    out["by_fir_year_gap_when_both_comparable"] = {k: dict(sorted(v.items())) for k, v in sorted(by_gap.items())}
    out["age_unavailable_in_v3"] = {"candidates": sum(v3[c]["age_state"] == "unavailable" for c in v3), "of_which_comparable_in_v2": sum(v3[c]["age_state"] == "unavailable" and v2[c]["age_applicable"] for c in v2),
                                    "by_reason": dict(sorted(Counter((v3[c]["reason"] or "").split(":", 1)[-1] for c in v3 if v3[c]["age_state"] == "unavailable" and v2[c]["age_applicable"]).items())),
                                    "by_source_pair": dict(sorted(Counter(v3[c]["source_pair"] for c in v3 if v3[c]["age_state"] == "unavailable" and v2[c]["age_applicable"]).items()))}

    # ---- candidates whose only hard conflict is age
    only2 = [c for c in v2 if v2[c]["hard"] == ["age_dob"]]
    only3 = [c for c in v3 if v3[c]["hard"] == ["age_dob"]]
    def outcome(c: str) -> str:
        if not v3[c]["hard"]:
            return "no hard conflict in v3 (age no longer a conflict)" if v3[c]["age_applicable"] else "no hard conflict in v3 (age became unavailable evidence)"
        return "still a hard age conflict" if v3[c]["hard"] == ["age_dob"] else "hard conflict on other field(s) only" if "age_dob" not in v3[c]["hard"] else "age plus other hard conflicts"
    res = Counter(outcome(c) for c in only2)
    out["age_only_hard_conflicts"] = {
        "v2_candidates": len(only2), "v2_share_of_candidates_with_a_hard_conflict": pct(len(only2), out["conflicts"]["v2"]["candidates_with_a_hard_conflict"]),
        "v3_candidates": len(only3), "v3_not_in_v2": len(set(only3) - set(only2)),
        "v3_outcome_of_the_v2_set": dict(res.most_common()),
        "v2_set_by_fir_year_gap": dict(sorted(Counter(fir_gap_bucket(v3[c]["evidence"]) for c in only2).items())),
        "v2_set_still_conflicting_by_fir_year_gap": dict(sorted(Counter(fir_gap_bucket(v3[c]["evidence"]) for c in only2 if v3[c]["hard"] == ["age_dob"]).items())),
        "v2_set_by_source_pair": dict(sorted(Counter(v2[c]["source_pair"] for c in only2).items())),
        "v2_set_cleared_by_source_pair": dict(sorted(Counter(v3[c]["source_pair"] for c in only2 if "age_dob" not in v3[c]["hard"] and v3[c]["age_applicable"]).items())),
        "v2_set_group_v3": dict(sorted(Counter(v3[c]["group"] for c in only2).items())),
        "v2_set_band_v2": dict(sorted(Counter(v2[c]["band"] for c in only2).items())), "v2_set_band_v3": dict(sorted(Counter(v3[c]["band"] for c in only2).items())),
        "v2_set_in_band_A_or_B": sum(v2[c]["band"] in ("A", "B") for c in only2), "v2_set_in_band_A_or_B_and_hard_free_in_v3": sum(v2[c]["band"] in ("A", "B") and not v3[c]["hard"] for c in only2),
        "v3_new_age_only_conflicts_by_v2_state": dict(sorted(Counter(v2[c]["age_state"] for c in set(only3) - set(only2)).items())),
        "v3_new_age_only_conflicts_by_fir_year_gap": dict(sorted(Counter(fir_gap_bucket(v3[c]["evidence"]) for c in set(only3) - set(only2)).items())),
        "v3_new_age_only_conflicts_by_source_pair": dict(sorted(Counter(v3[c]["source_pair"] for c in set(only3) - set(only2)).items())),
    }
    # how much of every age-conflict change rests on V1 records (whose age semantics are only weakly supported)
    ch = [c for c in v2 if v2[c]["age_state"] != v3[c]["age_state"] and v2[c]["age_applicable"] and v3[c]["age_applicable"]]
    out["changes_by_source_pair_when_both_comparable"] = {sp: dict(sorted(Counter(f"{v2[c]['age_state']} -> {v3[c]['age_state']}" for c in ch if v3[c]["source_pair"] == sp).items()))
                                                          for sp in sorted({v3[c]["source_pair"] for c in ch})}
    out["age_only_hard_conflicts"]["interpretation"] = (
        "A candidate leaves this set because the two ages are consistent once each is read at its own FIR date (evidence-based), or because the age "
        "became unavailable (no usable FIR date): that is lost evidence, not resolved evidence, and is counted separately.")

    # ---- proxy check: pairs sharing an exact name AND an exact phone are very likely the same person. A better age rule should flag FEWER of them.
    strong = [c for c in v2 if v2[c]["strong"]]
    def proxy(d: dict[str, dict]) -> dict[str, Any]:
        res: dict[str, Any] = {"pairs": len(strong), "age_state": dict(sorted(Counter(d[c]["age_state"] for c in strong).items()))}
        for sp in sorted({d[c]["source_pair"] for c in strong}):
            sub = [c for c in strong if d[c]["source_pair"] == sp]
            res[sp] = {"pairs": len(sub), **dict(sorted(Counter(d[c]["age_state"] for c in sub).items()))}
        return res
    out["proxy_check_same_name_and_phone_pairs"] = {
        "note": "Not ground truth: exact name+phone pairs are only very likely the same person. A better age rule should call fewer of them conflicting, "
                "and 'unavailable' must not be read as agreement.", "v2": proxy(v2), "v3": proxy(v3),
        "conflict_rate_among_comparable_v2": round(sum(v2[c]["age_state"] in ("hard", "soft") for c in strong) / max(1, sum(v2[c]["age_applicable"] for c in strong)), 4),
        "conflict_rate_among_comparable_v3": round(sum(v3[c]["age_state"] in ("hard", "soft") for c in strong) / max(1, sum(v3[c]["age_applicable"] for c in strong)), 4),
        "hard_rate_among_comparable_v2": round(sum(v2[c]["age_state"] == "hard" for c in strong) / max(1, sum(v2[c]["age_applicable"] for c in strong)), 4),
        "hard_rate_among_comparable_v3": round(sum(v3[c]["age_state"] == "hard" for c in strong) / max(1, sum(v3[c]["age_applicable"] for c in strong)), 4),
        "fir_year_gap_ge_2": {"pairs": sum(fir_gap_bucket(v3[c]["evidence"]) in ("2-3", "4+") for c in strong),
                              "v2_conflicting": sum(v2[c]["age_state"] in ("hard", "soft") and fir_gap_bucket(v3[c]["evidence"]) in ("2-3", "4+") for c in strong),
                              "v3_conflicting": sum(v3[c]["age_state"] in ("hard", "soft") and fir_gap_bucket(v3[c]["evidence"]) in ("2-3", "4+") for c in strong)},
    }

    # ---- observations / provenance summary
    prov = json.loads((args.v3 / "age_evidence_summary.json").read_text())
    out["age_conflict_summary"] = {"persons": prov["persons"], "observation_status_per_person": prov["observation_status_per_person"], "review_flags": prov["review_flags"],
                                   "persons_with_multiple_fir_links": prov["persons_with_multiple_fir_links"], "unavailable_reasons": prov["age_unavailable_reasons_per_observation"],
                                   "own_dob_vs_age_at_fir_date_inconsistencies": prov["own_dob_vs_age_at_fir_date_inconsistencies"],
                                   "age_evidence_state_by_source": prov["age_evidence_state_by_source"], "fir_date_status": prov["fir_date_status_per_observation"],
                                   "candidates_with_a_review_flag_on_age": sum(bool(v3[c]["flags"]) for c in v3), "flags_on_candidates": dict(sorted(Counter(f for c in v3 for f in v3[c]["flags"]).items()))}
    s2 = json.loads((args.v2 / "run_summary.json").read_text())
    out["age_dob_data_quality"] = {"v2_inconsistencies_within_a_record": s2["age_dob"]["age_dob_inconsistencies_within_a_record"],
                                   "v3_inconsistencies_within_a_record_at_fir_date": json.loads((args.v3 / "run_summary.json").read_text())["age_dob"]["age_dob_inconsistencies_within_a_record"]}

    # ---- files
    rows = []
    for c in sorted(v2):
        if v2[c]["age_state"] != v3[c]["age_state"] or v2[c]["method"] != v3[c]["method"] or v2[c]["hard"] != v3[c]["hard"]:
            rows.append([c, v3[c]["source_pair"], v2[c]["age_state"], v3[c]["age_state"], v2[c]["method"], v3[c]["method"], v2[c]["gap"], fir_gap_bucket(v3[c]["evidence"]),
                         v2[c]["score"], v3[c]["score"], v2[c]["band"], v3[c]["band"], v2[c]["group"], v3[c]["group"], rk2[c], rk3[c],
                         ",".join(v2[c]["hard"]), ",".join(v3[c]["hard"]), ",".join(v3[c]["flags"]), explain(v2[c], v3[c])])
    with open(args.v3 / "age_changes_v2_vs_v3.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "source_pair", "age_state_v2", "age_state_v3", "age_method_v2", "age_method_v3", "age_gap_years_v2", "fir_year_gap_v3", "score_v2", "score_v3",
                    "band_v2", "band_v3", "group_v2", "group_v3", "review_order_v2", "review_order_v3", "hard_conflicts_v2", "hard_conflicts_v3", "age_review_flags_v3", "explanation"])
        w.writerows(rows)
    out["age_changes_rows"] = len(rows)
    (args.v3 / "age_conflict_summary.json").write_text(json.dumps(out["age_conflict_summary"], indent=2), encoding="utf-8")
    (args.v3 / "comparison_v2_vs_v3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(args.v3 / "COMPARISON_V2_VS_V3.md", out, args)
    print(json.dumps({k: out[k] for k in ("score", "bands", "coverage", "review_groups", "review_order")}, indent=1))
    print(json.dumps(out["age_only_hard_conflicts"], indent=1))
    return 0


def table(rows: list[list[Any]], header: list[str]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(str(x) for x in r) + " |" for r in rows)]) + "\n"


def write_markdown(path: Path, o: dict[str, Any], args) -> None:
    c, s, b, cov, g = o["conflicts"], o["score"], o["bands"], o["coverage"], o["review_groups"]
    a = o["age_only_hard_conflicts"]
    md = [f"# score-v2 vs score-v3 - same {o['candidates']:,} candidates\n",
          f"v2 run `{args.v2.name}`, v3 run `{args.v3.name}`. Candidate generation is unchanged; every candidate is still `pending` human review "
          f"({o['v3_review_status']}). Scores are prioritisation aids, not probabilities. Only the age/DOB field differs between the runs "
          f"(other fields identical: {o['non_age_fields_identical']}).\n",
          "## Scores and bands\n",
          f"Mean match score {s['v2_mean']} (v2) -> {s['v3_mean']} (v3). Unchanged {s['unchanged']:,}; increased {s['increased']:,}; decreased {s['decreased']:,}; "
          f"mean absolute change {s['mean_abs_change_points']} points (max +{s['max_increase']}, {s['max_decrease']}).\n",
          table([[k, v] for k, v in s["change_distribution"].items()], ["change (points)", "candidates"]),
          table([[x, b["v2"][x], b["v3"][x]] for x in BANDS], ["band", "v2", "v3"]) + f"\n{b['changed']:,} candidates changed band: {b['transitions']}\n",
          "## Coverage\n",
          f"Mean evidence coverage {cov['v2_mean_percent']}% -> {cov['v3_mean_percent']}%. The age field is comparable for {cov['age_field_applicable_v2']:,} candidates in v2 and "
          f"{cov['age_field_applicable_v3']:,} in v3. Coverage lower in v3 for {cov['coverage_lower_in_v3']:,}, higher for {cov['coverage_higher_in_v3']:,}. "
          f"Insufficient evidence: {cov['insufficient_evidence_v2']:,} -> {cov['insufficient_evidence_v3']:,} ({cov['became_insufficient_in_v3']:,} newly insufficient).\n",
          "## Conflicts\n",
          table([["candidates with a hard conflict", c["v2"]["candidates_with_a_hard_conflict"], c["v3"]["candidates_with_a_hard_conflict"]],
                 ["candidates with a soft conflict", c["v2"]["candidates_with_a_soft_conflict"], c["v3"]["candidates_with_a_soft_conflict"]],
                 *[[f"hard: {k}", c["v2"]["hard_by_field"].get(k, 0), c["v3"]["hard_by_field"].get(k, 0)] for k in sorted(set(c["v2"]["hard_by_field"]) | set(c["v3"]["hard_by_field"]))],
                 *[[f"soft: {k}", c["v2"]["soft_by_field"].get(k, 0), c["v3"]["soft_by_field"].get(k, 0)] for k in sorted(set(c["v2"]["soft_by_field"]) | set(c["v3"]["soft_by_field"]))]],
                ["", "v2", "v3"]),
          "Age field state, v2 -> v3:\n\n" + table([[k, v] for k, v in o["age_state_transitions_v2_to_v3"].items()], ["v2 -> v3", "candidates"]),
          "### Why the age conflicts changed\n",
          *[f"**{k}**\n\n" + table([[t, n] for t, n in v.items()], ["reason", "candidates"]) for k, v in o["age_conflict_changes_explained"].items()],
          "Where both versions could compare the ages, by how many years apart the two FIRs are:\n\n",
          *[f"* FIR years apart **{k}**: {v}\n" for k, v in o["by_fir_year_gap_when_both_comparable"].items()],
          f"\nAge became unavailable in v3 for {o['age_unavailable_in_v3']['candidates']:,} candidates ({o['age_unavailable_in_v3']['of_which_comparable_in_v2']:,} were comparable in v2): "
          f"{o['age_unavailable_in_v3']['by_reason']}; by source pair {o['age_unavailable_in_v3']['by_source_pair']}. This is lost evidence (no usable FIR date), not resolved evidence.\n",
          "## Candidates whose only hard conflict is age\n",
          f"v2: {a['v2_candidates']:,} candidates ({a['v2_share_of_candidates_with_a_hard_conflict']} of all hard-conflict candidates). v3: {a['v3_candidates']:,} "
          f"({a['v3_not_in_v2']:,} of them not age-only in v2).\n\n" + table([[k, v] for k, v in a["v3_outcome_of_the_v2_set"].items()], ["v3 outcome of the v2 set", "candidates"]),
          f"By FIR-year gap: {a['v2_set_by_fir_year_gap']}; still conflicting: {a['v2_set_still_conflicting_by_fir_year_gap']}. Band A/B in v2: {a['v2_set_in_band_A_or_B']:,} "
          f"({a['v2_set_in_band_A_or_B_and_hard_free_in_v3']:,} hard-conflict-free in v3). v3 group of the v2 set: {a['v2_set_group_v3']}.\n\n{a['interpretation']}\n",
          "## Review groups and ordering\n",
          table([[k, g["v2"].get(k, 0), g["v3"].get(k, 0)] for k in sorted(set(g["v2"]) | set(g["v3"]))], ["group", "v2", "v3"]) + f"\n{g['moved']:,} candidates moved group: {g['transitions']}\n\n"
          f"Order: mean absolute rank change {o['review_order']['mean_abs_rank_change']}; overlap of the first 500 / 1,000 / 2,000 = {o['review_order']['top_500_overlap']} / "
          f"{o['review_order']['top_1000_overlap']} / {o['review_order']['top_2000_overlap']}; entered the top 1,000: {o['review_order']['entered_top_1000']} "
          f"(by v2 age state {o['review_order']['entered_top_1000_by_v2_age_state']}); left it: {o['review_order']['left_top_1000']} (by v3 age state {o['review_order']['left_top_1000_by_v3_age_state']}).\n",
          "## Proxy check: pairs with the same exact name and phone\n",
          f"{o['proxy_check_same_name_and_phone_pairs']['note']}\n\n```\n{json.dumps(o['proxy_check_same_name_and_phone_pairs'], indent=2)}\n```\n",
          "## By source pair (where both versions could compare the ages)\n", f"```\n{json.dumps(o['changes_by_source_pair_when_both_comparable'], indent=2)}\n```\n",
          "## Age-evidence conflicts and provenance (v3)\n", f"```\n{json.dumps(o['age_conflict_summary'], indent=2)}\n```\n",
          f"Data quality (a record whose own age contradicts its own DOB): {o['age_dob_data_quality']}.\n",
          f"`age_changes_v2_vs_v3.csv` lists the {o['age_changes_rows']:,} candidates whose age state, method or hard-conflict set changed, with the explanation of each.\n"]
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
