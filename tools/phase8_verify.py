"""Phase 8 verification of a score-v3 run, with code that does not share the scoring implementation.

* recomputes the age field of every candidate compared by FIR-anchored birth-year ranges from the provenance file alone;
* re-reads 300 FIR dates straight from the database (read-only) and compares them with the provenance;
* checks the candidate set, pending status, and that no usable observation lacks a FIR date;
* compares output hashes and database counters with a snapshot taken before the run (--before / --after JSON files).

python tools/phase8_verify.py --v3 <run dir> --before snap_before.json --after snap_after.json [--env-file ...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def years_for(age: int, fir: date) -> tuple[int, int]:
    """Independent derivation: birth year y is possible iff someone born on 1 Jan y .. 31 Dec y can be `age` on the FIR date."""
    ok = []
    for y in range(fir.year - age - 2, fir.year - age + 2):
        def age_on(birth: date) -> int:
            return fir.year - birth.year - ((fir.month, fir.day) < (birth.month, birth.day))
        if age_on(date(y, 12, 31)) <= age <= age_on(date(y, 1, 1)):
            ok.append(y)
    return min(ok), max(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--before", type=Path)
    ap.add_argument("--after", type=Path)
    ap.add_argument("--env-file")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    checks: dict[str, object] = {}
    prov = {}
    for line in open(args.v3 / "age_evidence_provenance.jsonl", encoding="ascii"):
        r = json.loads(line)
        prov[r["person_id"]] = r

    # 1. every usable observation has a FIR date and a range that an independent derivation reproduces
    bad = 0
    for r in prov.values():
        for o in r["observations"]:
            if o["usable"]:
                if not o["fir_date"] or tuple(o["birth_year_range"]) != years_for(o["recorded_age"], date.fromisoformat(o["fir_date"])):
                    bad += 1
            elif o["birth_year_range"] is not None:
                bad += 1
    checks["observations_with_wrong_or_missing_range"] = bad
    checks["usable_observations"] = sum(o["usable"] for r in prov.values() for o in r["observations"])

    # 2. recompute the age-only comparison for every candidate that used it
    n = mism = 0
    tol, conflict_gap, factors, cap = 1, 4, (1.0, 0.85, 0.6, 0.35, 0.15), 0.7
    for line in open(args.v3 / "scored_candidates.jsonl", encoding="ascii"):
        rec = json.loads(line)
        f = rec["fields"]["age_dob"]
        if rec["review"]["status"] != "pending":
            mism += 1
        if (f.get("detail") or {}).get("method") != "estimated_birth_year_ranges_at_fir_date":
            continue
        ra, rb = prov[rec["person_a_id"]]["merged_birth_year_range"], prov[rec["person_b_id"]]["merged_birth_year_range"]
        g = max(ra[0], rb[0]) - min(ra[1], rb[1])
        dist = max(0, g)
        gap = abs(ra[0] + ra[1] - rb[0] - rb[1]) // 2
        want_conflict = None if dist <= tol else ("soft" if dist <= conflict_gap else "hard")
        want_score = round((factors[gap] if gap < len(factors) else 0.0) * cap, 4)
        n += 1
        if want_conflict != f.get("conflict") or abs(want_score - f["score"]) > 1e-4:
            mism += 1
    checks["age_range_comparisons_recomputed"], checks["age_range_comparisons_mismatching"] = n, mism

    # 3. FIR dates against the database (read-only)
    from dedup_pipeline.db import connect_read_only, load_connection_settings

    ids = sorted(p for p, r in prov.items() if any(o["fir_date"] for o in r["observations"]))
    random.Random(8).shuffle(ids)
    sample = ids[: args.sample]
    wrong = 0
    with connect_read_only(load_connection_settings(args.env_file), "phase8_verify") as conn:
        cur = conn.cursor()
        cur.execute("SELECT a.person_id, (c.fir_date AT TIME ZONE 'Asia/Kolkata')::date, c.fir_date FROM public.accused_unified a "
                    "JOIN public.crimes_unified c ON c.crime_id = a.crime_id WHERE a.person_id = ANY(%(ids)s)", {"ids": sample})
        rows = cur.fetchall()
        got = {}
        for pid, local, raw in rows:
            got.setdefault(pid, []).append((local.isoformat(), raw))
        for pid in sample:
            want = sorted(o["fir_date"] for o in prov[pid]["observations"] if o["fir_date"])
            if want != sorted(x[0] for x in got.get(pid, [])):
                wrong += 1
        checks["fir_dates_sampled_from_db"], checks["fir_dates_differing_from_db"] = len(sample), wrong
        checks["fir_dates_where_ist_date_differs_from_utc_date"] = sum(1 for v in got.values() for local, raw in v if local != raw.date().isoformat())
        cur.close()

    # 4. preservation and database
    if args.before and args.after:
        b, a = json.loads(args.before.read_text()), json.loads(args.after.read_text())
        changed = [k for k in b["files"] if a["files"].get(k) != b["files"][k]]
        checks["preserved_files_checked"] = len(b["files"])
        checks["preserved_files_changed_or_removed"] = changed
        checks["preserved_files_added"] = sorted(set(a["files"]) - set(b["files"]))
        for key in ("tuple_writes", "row_counts", "objects", "schema_hash", "indexes_hash"):
            checks[f"database_{key}_unchanged"] = b[key] == a[key]
    out = args.v3 / "VERIFICATION.json"
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    ok = (checks["observations_with_wrong_or_missing_range"] == 0 and checks["age_range_comparisons_mismatching"] == 0
          and checks["fir_dates_differing_from_db"] == 0)
    if args.before and args.after:
        ok = ok and not checks["preserved_files_changed_or_removed"] and all(v for k, v in checks.items() if k.startswith("database_"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
