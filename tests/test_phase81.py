"""Phase 8.1 investigation tool: the pure analysis functions, on synthetic score-v2 / score-v3 records."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from dedup_pipeline.scoring.age import birth_year_range

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase81_investigate", ROOT / "tools" / "phase81_investigate.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_the_independent_birth_year_derivation_matches_the_pipeline_everywhere():
    rng = random.Random(1)
    dates = [date(2024, 1, 1), date(2024, 12, 31), date(2024, 2, 29), date(2023, 3, 1)] + [date(2000, 1, 1) + timedelta(days=rng.randrange(9000)) for _ in range(300)]
    for fir, age in itertools.product(dates, (5, 17, 26, 30, 44, 99)):
        assert tool.independent_range(age, fir) == birth_year_range(age, fir), (fir, age)


def rec(pair="V1-V1", hard=(), age_state="no_conflict", applicable=True, score=60.0, band="C", group="1_no_hard_conflict", insufficient=False,
        method="estimated_birth_year_ranges", gap=1, reason=None, ages=(30, 31), firs=("2020-06-01", "2021-06-01"), evidence=None, i=0):
    ranges = [list(tool.independent_range(a, date.fromisoformat(f))) for a, f in zip(ages, firs)]
    values = {s: {"recorded_age": a, "birth_year_range": r, "observations": [{"usable": True, "fir_date": f, "crime_id": f"c{s}{i}", "fir_year": int(f[:4])}]}
              for s, a, r, f in zip("ab", ages, ranges, firs)}
    return {"a": f"pa{i}", "b": f"pb{i}", "source_pair": pair, "pair_class": tool.PAIR_CLASS[pair], "hard": list(hard), "soft": [], "age_state": age_state, "age_applicable": applicable,
            "score": score, "band": band, "group": group, "insufficient": insufficient, "method": method, "gap": gap, "reason": reason, "coverage": 60,
            "values": values, "evidence": {"birth_year_range_distance_years": max(0, max(ranges[0][0], ranges[1][0]) - min(ranges[0][1], ranges[1][1])), "fir_years_a": [int(firs[0][:4])], "fir_years_b": [int(firs[1][:4])]},
            "phone": 1.0, "name": 1.0, "rel": 1.0, "gender_conflict": None, "flags": [], "detail": {}, "strong": False}


def world():
    v2, v3 = {}, {}
    n = 0

    def add(cid, a, b):
        nonlocal n
        v2[cid], v3[cid] = a, b
        n += 1

    for k in range(8):  # cleared: age-only hard in v2, consistent in v3
        add(f"cleared{k}", rec(hard=["age_dob"], age_state="hard", gap=7, group="2_hard_conflict", i=n), rec(age_state="no_conflict", ages=(24, 32), firs=("2018-06-01", "2026-06-01"), i=n, score=70))
    for k in range(6):  # lost: age-only hard in v2, unavailable in v3
        add(f"lost{k}", rec(pair="V1-V2", hard=["age_dob"], age_state="hard", gap=10, group="2_hard_conflict", i=n),
            rec(pair="V1-V2", age_state="unavailable", applicable=False, reason="age_evidence_unavailable:no_fir_link/age_at_fir_date", method=None, i=n, score=75, band="B"))
    for k in range(7):  # new: no hard conflict in v2, age-only hard in v3
        add(f"new{k}", rec(pair="V2-V2", age_state="no_conflict", gap=0, ages=(24, 24), firs=("2018-06-01", "2018-06-01"), i=n),
            rec(pair="V2-V2", hard=["age_dob"], age_state="hard", ages=(24, 24), firs=("2018-06-01", "2026-06-01"), i=n, group="2_hard_conflict"))
    for k in range(5):  # still hard in both
        add(f"still{k}", rec(pair="V2-V2", hard=["age_dob"], age_state="hard", gap=9, group="2_hard_conflict", i=n), rec(pair="V2-V2", hard=["age_dob"], age_state="hard", ages=(20, 40), firs=("2020-06-01", "2020-06-01"), i=n, group="2_hard_conflict"))
    for k in range(5):  # became insufficient
        add(f"insuf{k}", rec(i=n, band="B"), rec(age_state="unavailable", applicable=False, insufficient=True, band="E", group="3_insufficient_evidence", reason="x:no_fir_link/a", method=None, i=n))
    for k in range(10):  # controls
        add(f"ctrl{k}", rec(pair="V2-V2", i=n), rec(pair="V2-V2", i=n))
    return v2, v3


def ranks(d, reverse=False):
    ids = sorted(d, reverse=reverse)
    return {c: i + 1 for i, c in enumerate(ids)}


def test_the_lost_evidence_cases_are_kept_apart_from_the_genuinely_consistent_ones():
    v2, v3 = world()
    res, rows = tool.task4(v2, v3, {f"lost{k}": ("hard", "conflict") for k in range(6)}, ranks(v2), ranks(v3))
    assert res["v2_age_only_hard_conflicts"] == 19
    assert res["outcome_in_v3_kept_separate"] == {"evidence_now_no_conflict": 8, "lost_age_evidence_unavailable_in_v3": 6, "still_a_hard_age_conflict": 5}
    assert sum(res["outcome_in_v3_kept_separate"].values()) == res["v2_age_only_hard_conflicts"]  # a partition: nothing counted twice or dropped
    assert res["genuinely_consistent_after_FIR_anchoring"] == 8 and res["lost_evidence_cases"] == 6 == len(rows)
    assert res["hypothetical_age_state_if_the_prefix_FIR_were_used"] == {"hard": 6} and "must not be read as evidence of agreement" in res["note"]
    assert all(r["v3_a_state"] == "no_fir_link" or r["v3_b_state"] == "no_fir_link" for r in rows)


def test_new_age_only_conflicts_are_recomputed_independently_and_birthday_uncertainty_cannot_explain_them():
    v2, v3 = world()
    res, rows = tool.task3(v2, v3, ranks(v3))
    assert res["count"] == 7 == len(rows) and res["independent_recomputation_disagreements"] == 0
    assert all(r["range_distance_years_independent"] > 4 for r in rows) and res["by_v2_age_state"] == {"no_conflict": 7}
    assert res["sensitivity_to_the_conflict_threshold_report_only"]["hard_if_conflict_gap_were_4"] == 7
    assert all(r["fir_year_gap"] == 8 for r in rows) and res["involving_at_least_one_V1_record"] == 0


def test_comparison_by_source_pair_adds_up():
    v2, v3 = world()
    out = tool.task5(v2, v3, ranks(v2), ranks(v3, reverse=True), [])
    allc = out["ALL"]
    assert allc["candidates"] == 41 == sum(out[k]["candidates"] for k in ("V1-only", "V2-only", "cross-source"))
    assert allc["became_insufficient"] == 5 == out["became_insufficient_detail"]["count"] and allc["age_only_hard_v2_v3"] == [19, 5 + 7]
    assert allc["age_only_hard_v2_outcome"] == {"evidence_now_no_conflict": 8, "lost_evidence": 6, "still_hard": 5}
    assert allc["new_age_only_hard_in_v3"] == 7 and out["V2-only"]["new_age_only_hard_in_v3"] == 7


def test_the_comparison_sample_is_stratified_deterministic_and_carries_no_names_or_phones():
    v2, v3 = world()
    a, census = tool.task6(v2, v3, ranks(v2), ranks(v3, reverse=True), {"ctrl0"}, per_cell=2)
    b, _ = tool.task6(v2, v3, ranks(v2), ranks(v3, reverse=True), {"ctrl0"}, per_cell=2)
    assert a == b  # deterministic
    assert len({r["candidate_id"] for r in a}) == len(a) and all(r["sample_label"].startswith("PHASE 8.1 COMPARISON SAMPLE") for r in a)
    by = {}
    for r in a:
        by.setdefault(r["stratum"], []).append(r)
    assert {r["candidate_id"][:5] for r in by["S2_age_only_hard_in_v2_lost_because_age_unavailable"]} == {"lost0", "lost1", "lost2", "lost3", "lost4", "lost5"} or all(
        r["candidate_id"].startswith("lost") for r in by["S2_age_only_hard_in_v2_lost_because_age_unavailable"])
    assert all(r["candidate_id"].startswith("new") for r in by["S3_new_age_only_hard_in_v3"]) and all(r["candidate_id"].startswith("insuf") for r in by["S5_became_insufficient_evidence"])
    ctrl = by["S9_control_age_unchanged_no_conflict"]
    assert ctrl and all(r["v2_score"] == r["v3_score"] and r["v2_age_state"] == r["v3_age_state"] == "no_conflict" for r in ctrl)
    assert all(len(v) <= 2 * 3 for v in by.values())  # at most per_cell per pair class
    assert census["_overlap_with_phase7_pilot"] == sum(r["in_original_phase7_pilot"] for r in a)
    forbidden = {"name", "full_name", "phone", "relative_name_text", "address", "age_a", "age_b", "recorded_age_a", "recorded_age_b", "dob"}
    assert not forbidden & set(a[0])  # scores only; no names, phones, addresses, ages or dates of birth


def test_the_original_phase7_pilot_file_is_never_written_by_the_tool():
    source = (ROOT / "tools" / "phase81_investigate.py").read_text(encoding="utf-8")
    assert "pilot_sample.csv" in source and 'open(args.pilot, encoding="utf-8-sig"' in source  # opened for reading only
    assert "write_csv(args.out / \"pilot" not in source and "audit\" / \"phase7\"" in source
    pilot = ROOT / "output" / "audit" / "phase7" / "pilot_sample.csv"
    if pilot.exists():
        assert hashlib.sha256(pilot.read_bytes()).hexdigest()  # present and readable; the byte-identity check is made in the phase report against the pre-run hash
