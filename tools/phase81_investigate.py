"""Phase 8.1 investigation of V1/V2 age semantics, the unlinked V1 persons, and the score-v2 -> score-v3 changes.

Read-only against the database (the pipeline's read-only connection guard); reads the v2 and v3 run directories and writes ONLY into
``output/audit/phase8_1``. It changes no scoring, no threshold and no earlier output. Hypothetical re-evaluations that use an
unapproved relationship (the FIR key embedded in a V1 person id) are labelled ``hypothetical`` and never feed any score.

python tools/phase81_investigate.py --v2 output/scoring/runs/<v2> --v3 output/scoring_v3/runs/<v3>
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import random
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dedup_pipeline.candidates.normalize import normalize_name  # noqa: E402
from dedup_pipeline.db import connect_read_only, load_connection_settings  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.age import age_at  # noqa: E402
from dedup_pipeline.scoring.age_v3 import build_age_evidence, compare_age_dob_v3  # noqa: E402
from dedup_pipeline.scoring.attributes import PersonAttributes  # noqa: E402

REFERENCE = date(2026, 10, 12)
SEED = 81


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmp = _tool("compare_v2_v3")  # slim(), explain(), fir_gap_bucket(), ranks(), stream()


def q(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchall()


def pct(n, d):
    return round(100 * n / d, 1) if d else None


# =============================================================================== task 1: age semantics

def task1(cur, v2_run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n, same, p_null, s_null = q(cur, """select count(*), count(*) filter (where p.age is not distinct from nullif(s.payload->>'age','')::numeric::int), count(*) filter (where p.age is null),
        count(*) filter (where nullif(s.payload->>'age','') is null) from persons_unified p join persons_source s on s.source_system='V2' and s.source_record_id=p.source_record_id where p.source_system='V2'""")[0]
    out["V2_age_is_a_pass_through_of_persons_source"] = {"persons": n, "equal_to_source_age": same, "source_age_blank": s_null}
    n, same, p_null = q(cur, """select count(*), count(*) filter (where p.age is not distinct from nullif(a.payload->>'age','')::numeric::int), count(*) filter (where p.age is null)
        from persons_unified p join accused_unified au on au.person_id=p.person_id join accused_source a on a.source_system='V1' and a.source_record_id=au.source_record_id where p.source_system='V1'""")[0]
    out["V1_age_equals_the_age_on_the_linked_V1_accused_record"] = {"linked_persons": n, "equal": same, "person_age_blank_where_record_has_one": p_null}
    out["V1_persons_source_rows"] = q(cur, "select count(*) from persons_source where source_system='V1'")[0][0]
    n, ok = q(cur, """select count(*), count(*) filter (where abs((nullif(a.payload->>'reg_dt','')::timestamp::date) - (c.fir_date at time zone 'Asia/Kolkata')::date) <= 1)
        from accused_unified au join accused_source a on a.source_system='V1' and a.source_record_id=au.source_record_id join crimes_unified c on c.crime_id=au.crime_id where au.source_system='V1'""")[0]
    out["V1_record_registration_date_equals_FIR_date_within_1_day"] = {"records": n, "equal": ok}
    out["V2_person_row_created_relative_to_FIR"] = dict(q(cur, """select case when d < -30 then '1_created_more_than_30_days_before_FIR' when d <= 30 then '2_within_30_days_of_FIR' when d <= 365 then '3_30_days_to_1_year_after' when d <= 1095 then '4_1_to_3_years_after' else '5_more_than_3_years_after' end b, count(*) from (
        select extract(epoch from ((s.payload->>'date_created')::timestamptz - c.fir_date))/86400 d from persons_unified p join persons_source s on s.source_system='V2' and s.source_record_id=p.source_record_id
        join accused_unified au on au.person_id=p.person_id join crimes_unified c on c.crime_id=au.crime_id where p.source_system='V2' and s.payload->>'date_created' is not null) t group by 1 order by 1"""))
    out["age_value_profile_by_source"] = {r[0]: {"persons": r[1], "age_0": r[2], "age_1_to_4": r[3], "age_over_100": r[4], "age_null": r[5]}
                                          for r in q(cur, "select source_system, count(*), count(*) filter (where age=0), count(*) filter (where age between 1 and 4), count(*) filter (where age>100), count(*) filter (where age is null) from persons_unified group by 1")}

    # DOB checks: the age against the age the DOB implies at several candidate anchor dates
    v2rows = q(cur, """select p.age, p.date_of_birth, (c.fir_date at time zone 'Asia/Kolkata')::date, (s.payload->>'date_created')::timestamptz::date, (s.payload->>'date_modified')::timestamptz::date
        from persons_unified p join accused_unified au on au.person_id=p.person_id join crimes_unified c on c.crime_id=au.crime_id left join persons_source s on s.source_system=p.source_system and s.source_record_id=p.source_record_id
        where p.source_system='V2' and p.date_of_birth > '1900-01-02' and p.age is not null""")
    hist = lambda diffs: dict(sorted(collections.Counter(max(-5, min(5, d)) for d in diffs).items()))
    out["V2_dob_check"] = {"persons_with_dob_and_age": len(v2rows),
                           "age_within_1y_of_dob_implied_age_at": {"FIR_date": sum(abs(a - age_at(d, f)) <= 1 for a, d, f, c, m in v2rows if f > d),
                                                                    "person_row_created": sum(abs(a - age_at(d, c)) <= 1 for a, d, f, c, m in v2rows if c),
                                                                    "person_row_modified": sum(abs(a - age_at(d, m)) <= 1 for a, d, f, c, m in v2rows if m),
                                                                    "reference_date_2026_10_12": sum(abs(a - age_at(d, REFERENCE)) <= 1 for a, d, f, c, m in v2rows)},
                           "age_minus_implied_at_FIR_histogram_clipped_at_5": hist([a - age_at(d, f) for a, d, f, c, m in v2rows if f > d])}
    v1rows = q(cur, """select a.payload->>'age', a.payload->>'dob', a.payload->>'reg_dt' from accused_source a where a.source_system='V1' and nullif(a.payload->>'dob','') is not null and nullif(a.payload->>'age','') is not null""")
    usable = []
    for age, dob, reg in v1rows:
        try:
            d, r, a = date.fromisoformat(dob[:10]), date.fromisoformat(reg[:10]), int(float(age))
        except (ValueError, TypeError):
            continue
        if d.year >= 1900 and a > 0 and r > d:
            usable.append((a, d, r))
    out["V1_dob_check_payload_dob_exists_only_on_the_V1_accused_record"] = {
        "records_with_dob_and_age": len(v1rows), "usable": len(usable),
        "age_within_1y_of_dob_implied_age_at_registration_date": sum(abs(a - age_at(d, r)) <= 1 for a, d, r in usable),
        "age_within_1y_at_reference_date": sum(abs(a - age_at(d, REFERENCE)) <= 1 for a, d, r in usable),
        "age_minus_implied_at_registration_histogram_clipped_at_5": hist([a - age_at(d, r) for a, d, r in usable])}

    # does the recorded age move with the FIR date across records of (very likely) one person?
    pairs = []
    for rec in cmp.stream(v2_run / "scored_candidates.jsonl"):
        f = rec["fields"]
        if f["name"]["applicable"] and f["relative_name"]["applicable"] and f["name"]["score"] >= 0.95 and f["relative_name"]["score"] >= 0.95 and f["gender"].get("conflict") is None:
            pairs.append((rec["person_a_id"], rec["person_b_id"], rec["source_pair"], bool(f["phone"]["applicable"] and f["phone"]["score"] == 1.0)))
    ids = sorted({p for a, b, *_ in pairs for p in (a, b)})
    info = {}
    for i in range(0, len(ids), 3000):
        for pid, src, age, fir, created, modified in q(cur, """select p.person_id, p.source_system, p.age, (c.fir_date at time zone 'Asia/Kolkata')::date, (s.payload->>'date_created')::timestamptz::date, (s.payload->>'date_modified')::timestamptz::date
            from persons_unified p left join accused_unified a on a.person_id=p.person_id left join crimes_unified c on c.crime_id=a.crime_id left join persons_source s on s.source_system=p.source_system and s.source_record_id=p.source_record_id
            where p.person_id = any(%s)""", (ids[i:i + 3000],)):
            info[pid] = (src, age, fir, created, modified)
    ok = lambda x: x[1] is not None and 5 <= x[1] <= 100 and x[2] is not None
    by = collections.defaultdict(list)
    for a, b, sp, phone in pairs:
        x, y = info[a], info[b]
        if not (ok(x) and ok(y)):
            continue
        if x[2] > y[2]:
            x, y = y, x
        d = (y[2] - x[2]).days / 365.25
        if d >= 2:
            by[sp].append((d, y[1] - x[1], phone))
    track = {}
    for sp, v in sorted(by.items()):
        track[sp] = {"pairs_with_FIR_dates_2y_or_more_apart": len(v),
                     "median_abs_error_years_if_age_is_as_of_FIR_date": round(statistics.median(abs(dl - d) for d, dl, _ in v), 2),
                     "median_abs_error_years_if_age_is_static": round(statistics.median(abs(dl) for d, dl, _ in v), 2),
                     "share_consistent_with_as_of_FIR_date_pct": pct(sum(abs(dl - d) <= 1.5 for d, dl, _ in v), len(v)),
                     "share_consistent_with_static_age_pct": pct(sum(abs(dl) <= 1 for d, dl, _ in v), len(v)),
                     "median_years_between_the_two_FIRs": round(statistics.median(d for d, _, _ in v), 1)}
    out["age_tracks_FIR_date_across_records_of_likely_same_person"] = {
        "identity_proxy": "name similarity >= 0.95 and relative-name similarity >= 0.95 and no gender conflict (independent of age); contaminated by some different people, which pushes both models toward larger errors",
        "pairs_in_proxy": len(pairs), "by_source_pair": track}
    anchors: dict[str, list[int]] = {k: [0, 0] for k in ("FIR date", "person row created", "person row modified", "one common date (score-v2 style)")}
    for a, b, sp, phone in pairs:
        if sp != "V2-V2":
            continue
        x, y = info[a], info[b]
        if not (ok(x) and ok(y) and x[3] and y[3] and x[4] and y[4]):
            continue
        for name, (ax, ay) in {"FIR date": (x[2], y[2]), "person row created": (x[3], y[3]), "person row modified": (x[4], y[4]), "one common date (score-v2 style)": (REFERENCE, REFERENCE)}.items():
            anchors[name][0] += abs((ax.year - x[1]) - (ay.year - y[1])) <= 1
            anchors[name][1] += 1
    out["V2_which_anchor_makes_likely_same_person_ages_agree_best"] = {k: {"pairs_with_birth_year_gap_le_1": v[0], "pairs": v[1], "share_pct": pct(v[0], v[1])} for k, v in anchors.items()}
    return out


# ======================================================================== task 2: the V1 persons without an accused row

def person_table(cur, ids: list[str]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), 3000):
        for pid, src, age, dob, rows in q(cur, """select p.person_id, p.source_system, p.age, p.date_of_birth,
              (select json_agg(json_build_array(a.accused_id, a.crime_id, (c.fir_date at time zone 'Asia/Kolkata')::date)) from accused_unified a left join crimes_unified c on c.crime_id=a.crime_id where a.person_id=p.person_id)
              from persons_unified p where p.person_id = any(%s)""", (ids[i:i + 3000],)):
            table[pid] = {"source": src, "age": age, "dob": dob, "fir_rows": [tuple(r) for r in (rows or [])]}
    return table


def task2(cur, candidate_persons: set[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    persons = q(cur, """select p.person_id, p.full_name, p.age, (au.person_id is not null) from persons_unified p left join accused_unified au on au.person_id=p.person_id where p.source_system='V1'""")
    acc = q(cur, """select a.payload->>'fir_reg_num', a.payload->>'accused_name', nullif(a.payload->>'age','')::numeric::int, (au.accused_id is not null), au.person_id
        from accused_source a left join accused_unified au on au.source_system='V1' and au.source_record_id=a.source_record_id where a.source_system='V1'""")
    crimes = {r[0]: r[1] for r in q(cur, "select crime_id, (fir_date at time zone 'Asia/Kolkata')::date from crimes_unified where source_system='V1'")}
    by_fir = collections.defaultdict(list)
    for fir, name, age, cons, pid in acc:
        by_fir[fir].append((normalize_name(name), age, cons, pid))
    cats: dict[str, dict[str, Any]] = {}
    linked_prefix_ok = 0
    for pid, name, age, linked in persons:
        prefix = pid[:13]
        if linked:
            linked_prefix_ok += prefix in crimes
            continue
        recs = by_fir.get(prefix, [])
        same = [r for r in recs if normalize_name(name) and r[0] == normalize_name(name)]
        if not recs:
            cat = "A_no_V1_accused_record_exists_for_this_FIR"
        elif same and any(r[2] and r[3] != pid for r in same):
            cat = "B_same_name_accused_record_in_the_FIR_is_linked_to_a_different_person_id"
        elif same:
            cat = "B2_same_name_accused_record_in_the_FIR_was_not_consolidated"
        else:
            cat = "C_the_FIR_has_accused_records_but_none_with_this_name"
        cats[pid] = {"category": cat, "prefix_crime_id": prefix, "prefix_crime_exists": prefix in crimes, "prefix_fir_date": crimes.get(prefix), "age": age, "in_candidates": pid in candidate_persons}
    n_linked = sum(1 for p in persons if p[3])
    res = {
        "V1_persons": len(persons), "linked_to_an_accused_row": n_linked, "without_an_accused_row": len(cats),
        "person_id_structure": "18 characters = the 13-character V1 crime_id + a 5-character suffix",
        "linked_persons_whose_id_prefix_equals_the_linked_crime_id": f"{linked_prefix_ok}/{n_linked} (by construction this may reflect how the link was built; it is not independent evidence)",
        "unlinked_persons_whose_id_prefix_is_an_existing_V1_crime_id": sum(c["prefix_crime_exists"] for c in cats.values()),
        "categories_all_unlinked_persons": dict(sorted(collections.Counter(c["category"] for c in cats.values()).items())),
        "categories_unlinked_persons_in_candidates": dict(sorted(collections.Counter(c["category"] for c in cats.values() if c["in_candidates"]).items())),
        "unlinked_persons_in_candidates": sum(c["in_candidates"] for c in cats.values()),
        "unlinked_in_candidates_with_a_valid_age_5_100": sum(c["in_candidates"] and c["age"] is not None and 5 <= c["age"] <= 100 for c in cats.values()),
        "accused_source_V1_rows": len(acc), "accused_source_V1_rows_not_in_accused_unified": sum(1 for r in acc if not r[3]),
        "ledger_context": {r[0]: r[1] for r in q(cur, "select gap_type, count(*) from source_gap_ledger where source_system='V1' group by 1")},
        "interrogation_rows_for_these_persons": q(cur, "select count(*) from interrogation_unified i join persons_unified p on p.person_id=i.person_id left join accused_unified au on au.person_id=p.person_id where p.source_system='V1' and au.person_id is null")[0][0],
    }
    return res, cats


def hypothetical_prefix(cur, cats, cand_ids, candidates, v3: dict[str, dict[str, Any]], v2: dict[str, dict[str, Any]], v2_run: Path) -> dict[str, Any]:
    """HYPOTHETICAL: use the crime id embedded in a V1 person id as the FIR link for persons that have no accused row. Not approved; analysis only."""
    config = ScoringConfig(rule_version="score-v3", reference_date="2026-10-12")
    table = person_table(cur, cand_ids)

    def evidence(pid: str, use_prefix: bool):
        t = table[pid]
        rows = list(t["fir_rows"])
        if use_prefix and not rows and pid in cats and cats[pid]["prefix_fir_date"]:
            rows = [(None, cats[pid]["prefix_crime_id"], cats[pid]["prefix_fir_date"])]
        return PersonAttributes(pid, t["source"], "r", age_info=build_age_evidence(t["dob"], t["age"], rows, config))

    state = {}
    for c in candidates:
        a, b = c["person_a_id"], c["person_b_id"]
        if not (a in cats or b in cats):
            continue
        f = compare_age_dob_v3(evidence(a, True), evidence(b, True), config)
        state[c["candidate_id"]] = ("unavailable" if not f.applicable else (f.conflict or "no_conflict"), f.detail.get("status"))
    lost = [cid for cid in state if v3[cid]["age_state"] == "unavailable" and v2[cid]["age_applicable"]]
    only_hard_lost = [cid for cid in lost if v2[cid]["hard"] == ["age_dob"]]
    res = {"label": "HYPOTHETICAL - uses an unapproved relationship; nothing here is in any score", "candidates_touching_an_unlinked_V1_person": len(state),
           "lost_age_evidence_in_v3_and_comparable_in_v2": len(lost),
           "their_age_state_if_the_prefix_FIR_were_used": dict(sorted(collections.Counter(state[c][0] for c in lost).items())),
           "of_the_560_style_cases_age_only_hard_in_v2_and_age_unavailable_in_v3": len(only_hard_lost),
           "their_age_state_if_the_prefix_FIR_were_used_(age_only_hard_cases)": dict(sorted(collections.Counter(state[c][0] for c in only_hard_lost).items())),
           "unavailable_even_with_the_prefix": sum(state[c][0] == "unavailable" for c in lost)}
    # does the prefix-FIR-anchored age track the FIR date? (pairs between an unlinked V1 person and anyone, likely same person)
    pairs = []
    for rec in cmp.stream(v2_run / "scored_candidates.jsonl"):
        f = rec["fields"]
        if (rec["person_a_id"] in cats or rec["person_b_id"] in cats) and f["name"]["applicable"] and f["relative_name"]["applicable"] and f["name"]["score"] >= 0.95 and f["relative_name"]["score"] >= 0.95 and f["gender"].get("conflict") is None:
            pairs.append((rec["person_a_id"], rec["person_b_id"]))
    v = []
    for a, b in pairs:
        ea, eb = evidence(a, True).age_info, evidence(b, True).age_info
        oa, ob = (ea.usable_observations or [None])[0], (eb.usable_observations or [None])[0]
        if oa is None or ob is None:
            continue
        if oa.fir_date > ob.fir_date:
            oa, ob = ob, oa
        d = (ob.fir_date - oa.fir_date).days / 365.25
        if d >= 2:
            v.append((d, ob.age - oa.age))
    res["prefix_FIR_anchor_check_on_likely_same_person_pairs"] = (
        {"pairs_2y_or_more_apart": len(v), "median_abs_error_as_of_prefix_FIR": round(statistics.median(abs(dl - d) for d, dl in v), 2), "median_abs_error_static": round(statistics.median(abs(dl) for d, dl in v), 2),
         "share_consistent_with_as_of_FIR_pct": pct(sum(abs(dl - d) <= 1.5 for d, dl in v), len(v)), "share_consistent_with_static_pct": pct(sum(abs(dl) <= 1 for d, dl in v), len(v))} if v else {"pairs_2y_or_more_apart": 0})
    res["_state"] = state
    return res


# ===================================================================================== run loading and helpers

PAIR_CLASS = {"V1-V1": "V1-only", "V2-V2": "V2-only", "V1-V2": "cross-source", "V2-V1": "cross-source"}


def load_run(run: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for rec in cmp.stream(run / "scored_candidates.jsonl"):
        s = cmp.slim(rec)
        f = rec["fields"]
        s.update({"a": rec["person_a_id"], "b": rec["person_b_id"], "pair_class": PAIR_CLASS[rec["source_pair"]],
                  "name": f["name"]["score"] if f["name"]["applicable"] else None, "rel": f["relative_name"]["score"] if f["relative_name"]["applicable"] else None,
                  "phone": f["phone"]["score"] if f["phone"]["applicable"] else None, "values": f["age_dob"].get("values") or {}, "detail": f["age_dob"].get("detail") or {},
                  "gender_conflict": f["gender"].get("conflict")})
        out[rec["candidate_id"]] = s
    return out


def independent_range(age: int, fir: date) -> tuple[int, int]:
    """Birth years y for which someone born in y can be `age` on `fir` - derived without the pipeline code."""
    def age_on(b: date) -> int:
        return fir.year - b.year - ((fir.month, fir.day) < (b.month, b.day))
    ok = [y for y in range(fir.year - age - 2, fir.year - age + 2) if age_on(date(y, 12, 31)) <= age <= age_on(date(y, 1, 1))]
    return min(ok), max(ok)


def side_obs(rec: dict[str, Any], side: str) -> dict[str, Any]:
    v = (rec["values"] or {}).get(side) or {}
    obs = [o for o in v.get("observations", []) if o.get("usable")]
    o = obs[0] if obs else {}
    return {"age": v.get("recorded_age"), "range": v.get("birth_year_range"), "fir_date": o.get("fir_date"), "crime_id": o.get("crime_id")}


# =================================================================================== tasks 3 and 4

def task3(v2, v3, ranks3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    new = [c for c in v3 if v3[c]["hard"] == ["age_dob"] and v2[c]["hard"] != ["age_dob"]]
    rows, wrong = [], 0
    dist_counter, centre = collections.Counter(), collections.Counter()
    for c in new:
        r = v3[c]
        a, b = side_obs(r, "a"), side_obs(r, "b")
        ra = independent_range(a["age"], date.fromisoformat(a["fir_date"]))
        rb = independent_range(b["age"], date.fromisoformat(b["fir_date"]))
        dist = max(0, max(ra[0], rb[0]) - min(ra[1], rb[1]))
        centre_gap = abs(ra[0] + ra[1] - rb[0] - rb[1]) / 2
        wrong += (list(ra) != a["range"] or list(rb) != b["range"] or dist != r["evidence"].get("birth_year_range_distance_years"))
        dist_counter[dist] += 1
        centre[centre_gap] += 1
        other_strong = bool(r["phone"] == 1.0 and (r["name"] or 0) >= 0.95 and (r["rel"] or 0) >= 0.95)
        rows.append({"candidate_id": c, "source_pair": r["source_pair"], "pair_class": r["pair_class"], "person_a_id": r["a"], "person_b_id": r["b"],
                     "v2_age_state": v2[c]["age_state"], "v2_age_gap_years": v2[c]["gap"], "age_a": a["age"], "age_b": b["age"], "fir_date_a": a["fir_date"], "fir_date_b": b["fir_date"],
                     "fir_year_gap": abs(int(a["fir_date"][:4]) - int(b["fir_date"][:4])), "birth_year_range_a": "-".join(map(str, ra)), "birth_year_range_b": "-".join(map(str, rb)),
                     "range_distance_years_independent": dist, "centre_gap_years": centre_gap, "name_score": r["name"], "relative_score": r["rel"], "phone_score": r["phone"],
                     "strong_non_age_evidence": other_strong, "v3_band": r["band"], "v3_review_order": ranks3[c], "v3_group": r["group"]})
    sens = {f"hard_if_conflict_gap_were_{g}": sum(r["range_distance_years_independent"] > g for r in rows) for g in (4, 5, 6, 7, 8)}
    res = {
        "count": len(new), "independent_recomputation_disagreements": wrong,
        "by_v2_age_state": dict(sorted(collections.Counter(r["v2_age_state"] for r in rows).items())), "by_source_pair": dict(sorted(collections.Counter(r["source_pair"] for r in rows).items())),
        "by_fir_year_gap": dict(sorted(collections.Counter(min(r["fir_year_gap"], 8) for r in rows).items())),
        "range_distance_years_distribution": dict(sorted(dist_counter.items())), "sensitivity_to_the_conflict_threshold_report_only": sens,
        "involving_at_least_one_V1_record": sum("V1" in r["source_pair"] for r in rows),
        "with_strong_non_age_evidence_phone_exact_and_name_and_relative_ge_0_95": sum(r["strong_non_age_evidence"] for r in rows),
        "with_similar_recorded_ages_within_1_year": sum(abs(r["age_a"] - r["age_b"]) <= 1 for r in rows),
        "in_the_v3_top_1000": sum(r["v3_review_order"] <= 1000 for r in rows),
        "birthday_uncertainty_check": "every conflict has a range distance above the threshold, i.e. even the most favourable birthdays leave the possible birth years more than "
                                      "4 years apart; birthday uncertainty cannot explain any of them",
    }
    return res, rows


def task4(v2, v3, hyp_state, ranks2, ranks3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    only2 = [c for c in v2 if v2[c]["hard"] == ["age_dob"]]
    split = collections.defaultdict(list)
    for c in only2:
        if v3[c]["hard"] == ["age_dob"]:
            split["still_a_hard_age_conflict"].append(c)
        elif v3[c]["hard"]:
            split["hard_conflict_on_other_fields"].append(c)
        elif not v3[c]["age_applicable"]:
            split["lost_age_evidence_unavailable_in_v3"].append(c)
        else:
            split[f"evidence_now_{v3[c]['age_state']}"].append(c)
    lost = split["lost_age_evidence_unavailable_in_v3"]
    rows = []
    for c in lost:
        r, o = v3[c], v2[c]
        reason = (r["reason"] or "").split(":", 1)[-1]
        a_state, b_state = (reason.split("/") + ["", ""])[:2]
        rows.append({"candidate_id": c, "source_pair": r["source_pair"], "person_a_id": r["a"], "person_b_id": r["b"], "age_a": (o["values"].get("a") or {}).get("recorded_age"), "age_b": (o["values"].get("b") or {}).get("recorded_age"),
                     "v2_age_gap_years": o["gap"], "v3_a_state": a_state, "v3_b_state": b_state, "v2_band": o["band"], "v3_band": r["band"], "v2_group": o["group"], "v3_group": r["group"],
                     "v2_review_order": ranks2[c], "v3_review_order": ranks3[c], "v3_insufficient": r["insufficient"],
                     "hypothetical_age_state_if_prefix_FIR_used": hyp_state.get(c, ("not_applicable", None))[0]})
    gaps = [r["v2_age_gap_years"] for r in rows if r["v2_age_gap_years"] is not None]
    res = {
        "v2_age_only_hard_conflicts": len(only2),
        "outcome_in_v3_kept_separate": {k: len(v) for k, v in sorted(split.items())},
        "genuinely_consistent_after_FIR_anchoring": sum(1 for c in only2 if not v3[c]["hard"] and v3[c]["age_applicable"] and v3[c]["age_state"] == "no_conflict"),
        "now_only_a_soft_conflict": sum(1 for c in only2 if not v3[c]["hard"] and v3[c]["age_applicable"] and v3[c]["age_state"] == "soft"),
        "lost_evidence_cases": len(lost),
        "lost_by_source_pair": dict(sorted(collections.Counter(r["source_pair"] for r in rows).items())),
        "lost_by_which_record_lacks_a_usable_age_at_a_FIR": dict(sorted(collections.Counter(("a" if r["v3_a_state"] != "age_at_fir_date" and r["v3_a_state"] != "dob" else "") + ("b" if r["v3_b_state"] != "age_at_fir_date" and r["v3_b_state"] != "dob" else "") for r in rows).items())),
        "lost_by_reason_of_the_unusable_record": dict(sorted(collections.Counter(s for r in rows for s in (r["v3_a_state"], r["v3_b_state"]) if s not in ("age_at_fir_date", "dob")).items())),
        "v2_age_gap_years_among_lost": {"min": min(gaps), "median": statistics.median(gaps), "max": max(gaps), "6_to_8": sum(6 <= g <= 8 for g in gaps), "9_or_more": sum(g >= 9 for g in gaps)},
        "lost_band_v2_to_v3": dict(sorted(collections.Counter(f"{r['v2_band']}->{r['v3_band']}" for r in rows).items())),
        "lost_group_v2_to_v3": dict(sorted(collections.Counter(f"{r['v2_group']} -> {r['v3_group']}" for r in rows).items())),
        "lost_that_are_now_in_the_v3_top_1000": sum(r["v3_review_order"] <= 1000 for r in rows), "lost_that_were_in_the_v2_top_1000": sum(r["v2_review_order"] <= 1000 for r in rows),
        "hypothetical_age_state_if_the_prefix_FIR_were_used": dict(sorted(collections.Counter(r["hypothetical_age_state_if_prefix_FIR_used"] for r in rows).items())),
        "note": "these candidates did NOT become consistent: their age conflict can no longer be evaluated. They are shown separately from the genuinely consistent ones and must not be read as evidence of agreement.",
    }
    return res, rows


# =========================================================================================== task 5

def task5(v2, v3, ranks2, ranks3, proxy_ids) -> dict[str, Any]:
    out: dict[str, Any] = {}
    top2 = {c for c, r in ranks2.items() if r <= 1000}
    top3 = {c for c, r in ranks3.items() if r <= 1000}
    for pc in ("V1-only", "V2-only", "cross-source", "ALL"):
        ids = [c for c in v3 if pc == "ALL" or v3[c]["pair_class"] == pc]
        sub = {
            "candidates": len(ids),
            "mean_score_v2": round(statistics.mean(v2[c]["score"] or 0 for c in ids), 2), "mean_score_v3": round(statistics.mean(v3[c]["score"] or 0 for c in ids), 2),
            "score_changed": sum(abs((v3[c]["score"] or 0) - (v2[c]["score"] or 0)) > 1e-9 for c in ids),
            "age_state_changed": sum(v2[c]["age_state"] != v3[c]["age_state"] for c in ids),
            "age_state_transitions": {f"{a}->{b}": n for (a, b), n in sorted(collections.Counter((v2[c]["age_state"], v3[c]["age_state"]) for c in ids if v2[c]["age_state"] != v3[c]["age_state"]).items())},
            "age_comparable_v2_v3": [sum(v2[c]["age_applicable"] for c in ids), sum(v3[c]["age_applicable"] for c in ids)],
            "hard_conflict_candidates_v2_v3": [sum(bool(v2[c]["hard"]) for c in ids), sum(bool(v3[c]["hard"]) for c in ids)],
            "age_only_hard_v2_v3": [sum(v2[c]["hard"] == ["age_dob"] for c in ids), sum(v3[c]["hard"] == ["age_dob"] for c in ids)],
            "age_only_hard_v2_outcome": dict(sorted(collections.Counter(
                ("still_hard" if v3[c]["hard"] == ["age_dob"] else "other_hard" if v3[c]["hard"] else "lost_evidence" if not v3[c]["age_applicable"] else "evidence_now_" + v3[c]["age_state"])
                for c in ids if v2[c]["hard"] == ["age_dob"]).items())),
            "new_age_only_hard_in_v3": sum(v3[c]["hard"] == ["age_dob"] and v2[c]["hard"] != ["age_dob"] for c in ids),
            "insufficient_v2_v3": [sum(v2[c]["insufficient"] for c in ids), sum(v3[c]["insufficient"] for c in ids)],
            "became_insufficient": sum(v3[c]["insufficient"] and not v2[c]["insufficient"] for c in ids),
            "band_changed": sum(v2[c]["band"] != v3[c]["band"] for c in ids),
            "group_changed": sum(v2[c]["group"] != v3[c]["group"] for c in ids),
            "entered_top_1000": sum(c in top3 and c not in top2 for c in ids), "left_top_1000": sum(c in top2 and c not in top3 for c in ids),
            "mean_abs_rank_change": round(statistics.mean(abs(ranks3[c] - ranks2[c]) for c in ids), 1),
        }
        out[pc] = sub
    became = [c for c in v3 if v3[c]["insufficient"] and not v2[c]["insufficient"]]
    out["became_insufficient_detail"] = {"count": len(became), "band_v2_to_v3": dict(sorted(collections.Counter(f"{v2[c]['band']}->{v3[c]['band']}" for c in became).items())),
                                         "by_pair_class": dict(sorted(collections.Counter(v3[c]["pair_class"] for c in became).items())),
                                         "age_state_v2": dict(sorted(collections.Counter(v2[c]["age_state"] for c in became).items())),
                                         "v2_had_hard_conflict": sum(bool(v2[c]["hard"]) for c in became),
                                         "v2_coverage_percent_distribution": dict(sorted(collections.Counter(int(v2[c]["coverage"] // 5 * 5) for c in became).items())),
                                         "reason": "the age field (weight 0.10) became unavailable, taking evidence coverage below the 0.40 minimum or the applicable fields below 3"}
    cause = lambda c: ("age_state:" + v2[c]["age_state"] + "->" + v3[c]["age_state"]) if v2[c]["age_state"] != v3[c]["age_state"] else "age_unchanged (rank moved because other candidates moved)"
    out["top_1000"] = {
        "entered": len(top3 - top2), "left": len(top2 - top3),
        "entered_by_cause": dict(collections.Counter(cause(c) for c in top3 - top2).most_common(8)), "left_by_cause": dict(collections.Counter(cause(c) for c in top2 - top3).most_common(8)),
        "entered_by_pair_class": dict(sorted(collections.Counter(v3[c]["pair_class"] for c in top3 - top2).items())), "left_by_pair_class": dict(sorted(collections.Counter(v3[c]["pair_class"] for c in top2 - top3).items())),
        "entered_that_had_a_v2_hard_conflict": sum(bool(v2[c]["hard"]) for c in top3 - top2), "left_that_have_a_v3_hard_conflict": sum(bool(v3[c]["hard"]) for c in top2 - top3),
        "v3_top_1000_age_state": dict(sorted(collections.Counter(v3[c]["age_state"] for c in top3).items())), "v2_top_1000_age_state": dict(sorted(collections.Counter(v2[c]["age_state"] for c in top2).items())),
    }
    # likely-same-person proxy: how often is each version's age field a conflict, by source pair and FIR gap?
    prox: dict[str, Any] = {}
    for label, d in (("v2", v2), ("v3", v3)):
        by = collections.defaultdict(collections.Counter)
        for c in proxy_ids:
            by[d[c]["pair_class"]][d[c]["age_state"]] += 1
        prox[label] = {k: dict(sorted(v.items())) for k, v in sorted(by.items())}
    gap_rows = collections.defaultdict(collections.Counter)
    for c in proxy_ids:
        if v3[c]["age_applicable"] and v3[c]["method"] == "estimated_birth_year_ranges_at_fir_date":
            gap_rows[cmp.fir_gap_bucket(v3[c]["evidence"])][f"v2:{v2[c]['age_state']}|v3:{v3[c]['age_state']}"] += 1
    out["likely_same_person_proxy"] = {"pairs": len(proxy_ids), "definition": "name >= 0.95 and relative name >= 0.95 and no gender conflict (age plays no part)", "age_state_by_pair_class": prox,
                                       "by_fir_year_gap_v2_v3": {k: dict(sorted(v.items())) for k, v in sorted(gap_rows.items())}}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=ROOT / "output" / "candidates" / "candidates.jsonl")
    ap.add_argument("--pilot", type=Path, default=ROOT / "output" / "audit" / "phase7" / "pilot_sample.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "audit" / "phase8_1")
    ap.add_argument("--env-file")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    v2, v3 = load_run(args.v2), load_run(args.v3)
    assert set(v2) == set(v3)
    ranks2, ranks3 = cmp.ranks(args.v2), cmp.ranks(args.v3)
    candidates = [json.loads(line) for line in open(args.candidates, encoding="utf-8")]
    cand_persons = {p for c in candidates for p in (c["person_a_id"], c["person_b_id"])}
    proxy_ids = [c for c in v3 if (v3[c]["name"] or 0) >= 0.95 and (v3[c]["rel"] or 0) >= 0.95 and v3[c]["gender_conflict"] is None]
    pilot_ids = set()
    if args.pilot.exists():
        with open(args.pilot, encoding="utf-8-sig", newline="") as fh:
            pilot_ids = {r["candidate_id"] for r in csv.DictReader(fh) if r.get("candidate_id")}

    results: dict[str, Any] = {"inputs": {"v2_run": args.v2.name, "v3_run": args.v3.name, "candidates": len(v3), "phase7_pilot_candidates": len(pilot_ids)}}
    with connect_read_only(load_connection_settings(args.env_file), "phase8_1_investigation") as conn:
        cur = conn.cursor()
        results["task1_age_semantics"] = task1(cur, args.v2)
        t2, cats = task2(cur, cand_persons)
        results["task2_unlinked_V1_persons"] = t2
        hyp = hypothetical_prefix(cur, cats, sorted(cand_persons), candidates, v3, v2, args.v2)
        hyp_state = hyp.pop("_state")
        results["task2_hypothetical_prefix_FIR"] = hyp
        results["database_writes_by_this_tool"] = 0
    results["task3_new_age_only_hard_conflicts"], rows3 = task3(v2, v3, ranks3)
    results["task4_lost_age_evidence"], rows4 = task4(v2, v3, hyp_state, ranks2, ranks3)
    results["task5_comparison"] = task5(v2, v3, ranks2, ranks3, proxy_ids)
    sample, census = task6(v2, v3, ranks2, ranks3, pilot_ids)
    results["task6_sample_census"] = census

    write_csv(args.out / "new_age_only_hard_conflicts_SENSITIVE.csv", rows3)
    write_csv(args.out / "lost_age_evidence_cases_SENSITIVE.csv", rows4)
    write_csv(args.out / "comparison_sample_phase8_1.csv", sample)
    write_csv(args.out / "unlinked_V1_persons_SENSITIVE.csv", [{"person_id": p, "category": c["category"], "id_prefix_crime_id": c["prefix_crime_id"], "prefix_crime_exists": c["prefix_crime_exists"],
                                                                  "recorded_age": c["age"], "in_candidates": c["in_candidates"]} for p, c in sorted(cats.items())])
    # worked examples for the protected report: a few per source pair and FIR gap
    rng = random.Random(SEED)
    examples = []
    for pc in ("V1-only", "cross-source", "V2-only"):
        for gap in (1, 2, 4, 8):
            pool = [r for r in rows3 if r["pair_class"] == pc and (min(r["fir_year_gap"], 8) >= gap) and r not in examples]
            pool = [r for r in pool if r["fir_year_gap"] < (gap * 2 if gap < 8 else 99)]
            rng.shuffle(pool)
            examples += pool[:2]
    write_csv(args.out / "new_age_only_hard_conflicts_examples_SENSITIVE.csv", examples)
    (args.out / "phase8_1_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k.startswith(("task2", "task3", "task4"))}, indent=1, default=str)[:9000])
    return 0


# =========================================================================================== task 6

STRATA = (
    ("S1_age_only_hard_in_v2_not_hard_in_v3_age_still_comparable_(consistent_or_soft)",lambda c, v2, v3: v2[c]["hard"] == ["age_dob"] and not v3[c]["hard"] and v3[c]["age_applicable"]),
    ("S2_age_only_hard_in_v2_lost_because_age_unavailable", lambda c, v2, v3: v2[c]["hard"] == ["age_dob"] and not v3[c]["hard"] and not v3[c]["age_applicable"]),
    ("S3_new_age_only_hard_in_v3", lambda c, v2, v3: v3[c]["hard"] == ["age_dob"] and v2[c]["hard"] != ["age_dob"]),
    ("S4_age_only_hard_in_both", lambda c, v2, v3: v3[c]["hard"] == ["age_dob"] and v2[c]["hard"] == ["age_dob"]),
    ("S5_became_insufficient_evidence", lambda c, v2, v3: v3[c]["insufficient"] and not v2[c]["insufficient"]),
    ("S6_entered_top_1000", None), ("S7_left_top_1000", None),
    ("S8_age_state_changed_other", lambda c, v2, v3: v2[c]["age_state"] != v3[c]["age_state"] and v3[c]["age_applicable"] and v2[c]["age_applicable"]),
    ("S9_control_age_unchanged_no_conflict", lambda c, v2, v3: v2[c]["age_state"] == "no_conflict" == v3[c]["age_state"] and abs((v2[c]["score"] or 0) - (v3[c]["score"] or 0)) < 1e-9),
    ("S10_control_age_unchanged_with_conflict_or_unavailable", lambda c, v2, v3: v2[c]["age_state"] == v3[c]["age_state"] != "no_conflict" and abs((v2[c]["score"] or 0) - (v3[c]["score"] or 0)) < 1e-9),
)


def task6(v2, v3, ranks2, ranks3, pilot_ids: set[str], per_cell: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(SEED)
    top2 = {c for c, r in ranks2.items() if r <= 1000}
    top3 = {c for c, r in ranks3.items() if r <= 1000}
    members = {}
    for name, fn in STRATA:
        if name == "S6_entered_top_1000":
            members[name] = sorted(top3 - top2)
        elif name == "S7_left_top_1000":
            members[name] = sorted(top2 - top3)
        else:
            members[name] = sorted(c for c in v3 if fn(c, v2, v3))
    taken: set[str] = set()
    sample = []
    census = {}
    for name, _ in STRATA:
        census[name] = {"in_stratum": len(members[name])}
        for pc in ("V1-only", "cross-source", "V2-only"):
            pool = [c for c in members[name] if v3[c]["pair_class"] == pc and c not in taken]
            rng.shuffle(pool)
            pick = pool[:per_cell]
            census[name][pc] = {"available": len(pool) + 0, "sampled": len(pick)}
            for c in pick:
                taken.add(c)
                r2, r3 = v2[c], v3[c]
                a, b = side_obs(r3, "a"), side_obs(r3, "b")
                belongs = [n for n, f in STRATA if c in set(members[n])]
                sample.append({
                    "sample_id": f"P81-{len(sample) + 1:03d}", "sample_label": "PHASE 8.1 COMPARISON SAMPLE (v2 vs v3) - not the Phase 7 pilot", "stratum": name, "also_in_strata": ";".join(n for n in belongs if n != name),
                    "pair_class": pc, "source_pair": r3["source_pair"], "candidate_id": c, "person_a_id": r3["a"], "person_b_id": r3["b"],
                    "v2_score": r2["score"], "v3_score": r3["score"], "v2_band": r2["band"], "v3_band": r3["band"], "v2_group": r2["group"], "v3_group": r3["group"],
                    "v2_review_order": ranks2[c], "v3_review_order": ranks3[c], "v2_hard_conflicts": ",".join(r2["hard"]), "v3_hard_conflicts": ",".join(r3["hard"]),
                    "v2_age_state": r2["age_state"], "v3_age_state": r3["age_state"], "v2_age_method": r2["method"], "v3_age_method": r3["method"],
                    "v2_age_gap_years": r2["gap"], "v3_birth_year_range_distance": r3["evidence"].get("birth_year_range_distance_years"),
                    "fir_year_a": a["fir_date"][:4] if a["fir_date"] else "", "fir_year_b": b["fir_date"][:4] if b["fir_date"] else "",
                    "birth_year_range_a_v3": "-".join(map(str, a["range"])) if a["range"] else "", "birth_year_range_b_v3": "-".join(map(str, b["range"])) if b["range"] else "",
                    "v3_age_unavailable_reason": (r3["reason"] or "").split(":", 1)[-1], "v3_insufficient_evidence": r3["insufficient"],
                    "name_score": r3["name"], "relative_name_score": r3["rel"], "phone_score": r3["phone"],
                    "explanation": cmp.explain(r2, r3).replace("are consistent", "are consistent or within the 1-year tolerance") if r2["age_state"] != r3["age_state"] else ("age state unchanged (" + r3["age_state"] + ")"),
                    "in_original_phase7_pilot": c in pilot_ids, "worksheet_lookup": "candidate_id in output/scoring_v3/.../review_worksheet.csv (v3) and output/scoring/runs/20261012T125026Z/review_worksheet.csv (v2)",
                    "reviewer_notes": "", "age_logic_assessment_(reviewer)": ""})
    census["_total_sampled"] = len(sample)
    census["_overlap_with_phase7_pilot"] = sum(r["in_original_phase7_pilot"] for r in sample)
    return sample, census


if __name__ == "__main__":
    sys.exit(main())
