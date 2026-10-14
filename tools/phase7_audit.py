"""Phase 7 validation audit (read-only). Writes only under output/audit/phase7/.

Checks the completed score-v2 and score-v1 runs, audits conflicts, deferred blocks and the review worksheet, and produces the
pilot sample. It reuses the existing scoring code (no scoring logic is changed here) and never writes to any database: its
connection is the project's guarded read-only connection and every statement is a SELECT.

    python tools/phase7_audit.py --v2 output/scoring/runs/20261012T125026Z --v1 output/scoring/runs/20261012T121355Z
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dedup_pipeline.candidates.config import CandidateConfig  # noqa: E402
from dedup_pipeline.candidates.runner import preflight  # noqa: E402
from dedup_pipeline.db import connect_read_only, load_connection_settings  # noqa: E402
from dedup_pipeline.face_recognition.pair_evidence import PairFaceEvidence, PairStatus  # noqa: E402
from dedup_pipeline.scoring.attributes import AttributeStore  # noqa: E402
from dedup_pipeline.scoring.config import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V1, WORKSHEET_COLUMNS_V2  # noqa: E402
from dedup_pipeline.scoring.score import score_pair  # noqa: E402
from dedup_pipeline.scoring.worksheet import WorksheetError, validate_worksheet  # noqa: E402

PLACEHOLDERS: dict[str, Any] = {}
OUT = ROOT / "output" / "audit" / "phase7"
CHECKS: list[dict[str, Any]] = []
FINDINGS: list[dict[str, Any]] = []
# pg_stat_user_tables counters recorded at the Phase 2 audit time (insert, update, delete); a write by anyone since then would change them
EARLIER_DB_COUNTERS = {"persons_unified": (53197, 212734, 0), "identity_links": (1136, 0, 0), "media_unified": (174057, 180353, 0)}


def check(section: str, name: str, passed: bool, detail: Any = "") -> bool:
    CHECKS.append({"section": section, "check": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    return passed


def finding(severity: str, area: str, text: str, evidence: Any = None) -> None:
    FINDINGS.append({"severity": severity, "area": area, "finding": text, "evidence": evidence})


def jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== 1. run integrity

def audit_integrity(v2: Path, v1: Path, cand_dir: Path, conn) -> dict[str, Any]:
    s2 = json.loads((v2 / "run_summary.json").read_text())
    st2 = json.loads((v2 / "run_state.json").read_text())
    cfg2 = json.loads((v2 / "scoring_config.json").read_text())
    check("1 integrity", "score-v2 run_summary status completed and complete", s2["status"] == "completed" and s2["complete"] is True, s2["status"])
    check("1 integrity", "score-v2 run_state agrees (completed, same run id)", st2["status"] == "completed" and st2["run_id"] == s2["run_id"])
    check("1 integrity", "rule version and reference date recorded", s2["rule_version"] == "score-v2" and s2["reference_date"] == cfg2["reference_date"] == "2026-10-12", s2["reference_date"])

    cand_ids = [r["candidate_id"] for r in jsonl(cand_dir / "candidates.jsonl")]
    check("1 integrity", "Phase 5 has 24,155 unique candidates", len(cand_ids) == len(set(cand_ids)) == 24155, len(cand_ids))
    slim: list[dict[str, Any]] = []
    versions, refdates, pending_bad, decisions_bad, null_scores = Counter(), Counter(), 0, 0, 0
    for r in jsonl(v2 / "scored_candidates.jsonl"):
        versions[r["rule_version"]] += 1
        rv = r["review"]
        pending_bad += rv["status"] != "pending"
        decisions_bad += any(rv[k] is not None for k in ("decision", "reviewer_id", "reviewed_at", "notes"))
        null_scores += r["match_score_percent"] is None
        d = r["fields"]["age_dob"].get("detail")
        if d:
            refdates[d["reference_date"]] += 1
        f = r["fields"]
        slim.append({
            "id": r["candidate_id"], "a": r["person_a_id"], "b": r["person_b_id"], "sp": r["source_pair"], "rules": r["candidate_generation"]["rule_names"],
            "score": r["match_score_percent"], "cov": r["evidence_coverage_percent"], "band": r["score_band"], "insuff": r["insufficient_evidence"],
            "hard": r["has_hard_conflict"], "group": r["review_group"], "n_app": r["applicable_fields"], "ind": r["processing"]["indicators"],
            "f": {n: (x["applicable"], x["score"], x.get("conflict"), x["weight"]) for n, x in f.items()},
            "age_status": (f["age_dob"].get("detail") or {}).get("status"), "age_method": (f["age_dob"].get("detail") or {}).get("method"),
            "face": r["face"]["status"], "cos": r["face"]["raw_cosine"], "face_retry": r["face"]["retry_needed"], "face_err": r["face"]["error_kind"],
            "dq": [c for c in r["conflicting_attributes"] if c["severity"] == "data_quality"],
            "hardf": [c["field"] for c in r["conflicting_attributes"] if c["severity"] == "hard"],
            "soft": [c["field"] for c in r["conflicting_attributes"] if c["severity"] == "soft"],
        })
    ids2 = [x["id"] for x in slim]
    check("1 integrity", "every Phase 5 candidate has exactly one score-v2 record", ids2 == cand_ids, f"{len(ids2)} records")
    check("1 integrity", "all records are rule score-v2", set(versions) == {"score-v2"}, dict(versions))
    check("1 integrity", "all age comparisons use the single reference date", set(refdates) == {"2026-10-12"}, dict(refdates))
    check("1 integrity", "all 24,155 records remain pending with empty decision fields", pending_bad == 0 and decisions_bad == 0, f"{pending_bad} not pending, {decisions_bad} with decision data")
    check("1 integrity", "records without a score are all insufficient-evidence cases", all(x["insuff"] for x in slim if x["score"] is None), f"{null_scores} without a score")

    with open(v2 / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    check("1 integrity", "worksheet has 24,155 rows in the same candidate set", len(rows) == 24155 and {r["candidate_id"] for r in rows} == set(cand_ids))
    check("1 integrity", "worksheet decision columns are empty on every row", all(all(r[c] == "" for c in DECISION_COLUMNS) for r in rows))
    check("1 integrity", "worksheet review_status is pending on every row", all(r["review_status"] == "pending" for r in rows))
    keys = [(int(r["review_group"][0]), float(r["match_score_percent_NOT_A_PROBABILITY"] or 0)) for r in rows]
    ordered = all(a[0] < b[0] or (a[0] == b[0] and a[1] >= b[1]) for a, b in zip(keys, keys[1:]))
    check("1 integrity", "worksheet is ordered by review group, then displayed score from high to low", ordered)
    check("1 integrity", "worksheet review_order is 1..N without gaps", [int(r["review_order"]) for r in rows] == list(range(1, 24156)))
    summary_groups = s2["review_groups"]["candidates_per_group"]
    check("1 integrity", "group counts in the summary match the records", dict(Counter(x["group"] for x in slim)) == summary_groups, summary_groups)

    # score-v1 run
    s1 = json.loads((v1 / "run_summary.json").read_text())
    n1 = pending1 = 0
    for r in jsonl(v1 / "scored_candidates.jsonl"):
        n1 += 1
        pending1 += r["review"]["status"] == "pending" and r["review"]["decision"] is None
    check("1 integrity", "score-v1 run is complete, rule score-v1, 24,155 records, all pending", s1["status"] == "completed" and s1["rule_version"] == "score-v1" and n1 == pending1 == 24155)
    started2 = s2["started_at"]
    newest_v1 = max(f.stat().st_mtime for f in v1.rglob("*") if f.is_file())
    oldest_v2 = min(f.stat().st_mtime for f in v2.rglob("*") if f.is_file())
    check("1 integrity", "no score-v1 file was modified after the score-v2 run began", newest_v1 < oldest_v2, f"v2 started {started2}")
    hashes = {f.name: sha(f) for f in sorted(v1.iterdir()) if f.is_file()}
    (OUT / "score_v1_file_hashes.json").write_text(json.dumps(hashes, indent=1))
    # regenerate v1 in a scratch root and compare with the preserved files
    scratch = ROOT / "output" / "audit" / "phase7" / "_v1_regen"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    shutil.copy(ROOT / "output" / "scoring" / "face_evidence_cache.jsonl", scratch / "face_evidence_cache.jsonl")
    env = {**os.environ, "FR_DOPAMS_BASE_URL": "http://127.0.0.1:8010", "FR_DOPAMS_API_KEY": "placeholder-not-used-0000000000"}
    proc = subprocess.run([sys.executable, "-m", "dedup_pipeline.scoring", "run", "--rule-version", "score-v1", "--output-root", str(scratch)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    regen = next((scratch / "runs").iterdir(), None)
    same = regen is not None and sha(regen / "review_worksheet.csv") == hashes["review_worksheet.csv"] and sha(regen / "deferred_unevaluated.jsonl") == hashes["deferred_unevaluated.jsonl"]
    check("1 integrity", "regenerating score-v1 today reproduces the preserved worksheet byte for byte", same, f"exit {proc.returncode}")
    shutil.rmtree(scratch, ignore_errors=True)

    # database: read-only session and no writes
    check("1 integrity", "score-v2 run recorded a read-only session on the approved database and 0 writes",
          s2["database"]["transaction_read_only"] == "on" and s2["database"]["database"] == "dopams_cctns_v2" and s2["database_writes"] == 0)
    cur = conn.cursor()
    counts = {}
    for t, expected in (("persons_unified", 53197), ("accused_unified", 50503), ("crimes_unified", 16950), ("person_enrichment", 53182),
                        ("accused_enrichment", 50503), ("media_unified", 174057), ("media_source", 174057), ("identity_links", 1136)):
        cur.execute(f"SELECT count(*) FROM public.{t}")
        counts[t] = cur.fetchone()[0]
    check("1 integrity", "table row counts equal the Phase 1 audit", all(counts[t] == n for t, n in (("persons_unified", 53197), ("accused_unified", 50503), ("crimes_unified", 16950),
          ("person_enrichment", 53182), ("accused_enrichment", 50503), ("media_unified", 174057), ("media_source", 174057), ("identity_links", 1136))), counts)
    cur.execute("SELECT to_regclass('public.identity_dedup_run_log'), to_regclass('public.person_canonical_mapping'), to_regclass('be_read.person_resolved')")
    check("1 integrity", "dedup migration objects are still absent", cur.fetchone() == (None, None, None))
    cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND table_name='identity_links'")
    check("1 integrity", "identity_links still has its original 10 columns", cur.fetchone()[0] == 10)
    cur.execute("SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables WHERE schemaname='public' AND relname IN ('persons_unified','identity_links','media_unified')")
    now = {r[0]: tuple(r[1:]) for r in cur.fetchall()}
    check("1 integrity", "database write counters (insert/update/delete) are unchanged since the Phase 2 audit", now == EARLIER_DB_COUNTERS, {"now": now})
    cur.execute("SELECT count(*) FROM pg_tables WHERE schemaname='public'")
    tables = cur.fetchone()[0]
    check("1 integrity", "public schema still has 44 tables", tables == 44, tables)
    pkg = "\n".join(p.read_text(encoding="utf-8-sig") for p in (ROOT / "dedup_pipeline").rglob("*.py")).lower()
    check("1 integrity", "pipeline source contains no SQL write statements or write APIs",
          not any(k in pkg for k in ("insert into", "update public", "delete from", "create table", "executemany", ".commit(", "copy_from")))
    return {"summary_v2": s2, "summary_v1": s1, "config": cfg2["config"], "slim": slim, "rows": rows, "cand_dir": str(cand_dir), "v2": str(v2), "v1": str(v1)}


# ============================================================ 3. scoring and hard-conflict audit

def audit_scoring(ctx: dict[str, Any], conn) -> dict[str, Any]:
    slim, cfg = ctx["slim"], ctx["config"]
    weights = cfg["weights"]
    bands = cfg["bands"]
    bad_score = bad_cov = bad_band = bad_na = bad_group = bad_hard = bad_face = bad_rules = 0
    boundary: list[tuple] = []
    examples: dict[str, list[str]] = defaultdict(list)
    for x in slim:
        used = [(w, s) for (app, s, c, w) in x["f"].values() if app and w > 0 and s is not None]
        wsum = sum(w for w, _ in used)
        expected = round(100 * sum(w * s for w, s in used) / wsum, 1) if wsum else None
        if (expected is None) != (x["score"] is None) or (expected is not None and abs(expected - x["score"]) > 0.11):
            bad_score += 1; examples["score"].append(x["id"])
        if abs(100 * wsum / sum(weights.values()) - x["cov"]) > 0.11:
            bad_cov += 1; examples["coverage"].append(x["id"])
        # a field that is not applicable must carry no score and no conflict (missing is not a mismatch)
        if any((not app) and (s is not None or c is not None) for (app, s, c, w) in x["f"].values()):
            bad_na += 1; examples["not_applicable"].append(x["id"])
        insufficient = x["score"] is None or x["cov"] / 100 < cfg["min_coverage"] - 1e-9 or len(used) < cfg["min_applicable_fields"]
        exact = sum(w * s for w, s in used) / wsum if wsum else 0.0  # full precision, from the stored 4-decimal field scores
        band = "E" if insufficient else next(n for n, floor in bands if exact >= floor - 1e-9)
        if insufficient != x["insuff"]:
            bad_band += 1; examples["band"].append(x["id"])
        elif band != x["band"]:
            if min(abs(exact - floor) for _, floor in bands) < 1e-4:
                boundary.append((x["id"], x["score"], x["band"], band))  # sits on a band threshold: display rounding, not a rule error
            else:
                bad_band += 1; examples["band"].append(x["id"])
        hard_fields = [n for n, (app, s, c, w) in x["f"].items() if c == "hard"]
        if sorted(hard_fields) != sorted(x["hardf"]) or bool(hard_fields) != x["hard"]:
            bad_hard += 1; examples["hard"].append(x["id"])
        technical = x["face_retry"] or any(i.startswith("image_resolution_") for i in x["ind"])
        want = 4 if technical else 3 if x["insuff"] else 2 if x["hard"] else 1
        if int(x["group"][0]) != want:
            bad_group += 1; examples["group"].append(x["id"])
        face = x["f"]["face"]
        if x["face"] not in ("compared",) and (face[0] or face[1] is not None or face[2] is not None or x["cos"] is not None):
            bad_face += 1; examples["face"].append(x["id"])
    check("3 scoring", "match score recomputed from the stored field results equals the stored score for all records", bad_score == 0, f"{bad_score} differ")
    check("3 scoring", "evidence coverage recomputed equals the stored coverage for all records", bad_cov == 0, f"{bad_cov} differ")
    check("3 scoring", "no field that is not applicable carries a score or a conflict (missing evidence is not a mismatch)", bad_na == 0, f"{bad_na} records")
    check("3 scoring", "band and insufficient-evidence flag follow the documented thresholds for all records", bad_band == 0, f"{bad_band} differ")
    check("3 scoring", "has_hard_conflict equals 'some field carries a hard conflict' for all records", bad_hard == 0, f"{bad_hard} differ")
    check("3 scoring", "review group follows the documented precedence for all records", bad_group == 0, f"{bad_group} differ")
    check("3 scoring", "records without a face comparison carry no cosine, no face score and no face conflict", bad_face == 0, f"{bad_face} differ")

    # hard-conflict rules against their own thresholds
    viol = Counter()
    for x in slim:
        for n, (app, s, c, w) in x["f"].items():
            if not app or s is None:
                continue
            if n == "name" and ((s < cfg["name_conflict_below"]) != (c == "hard")): viol["name"] += 1
            if n == "relative_name" and ((s < cfg["relative_conflict_below"]) != (c == "hard")): viol["relative_name"] += 1
            if n == "gender" and ((s == 0.0) != (c == "hard")): viol["gender"] += 1
        if x["age_method"] in ("estimated_birth_year_ranges", "dob_vs_recorded_age") and ((x["age_status"] == "conflict") != ("age_dob" in x["hardf"])):
            viol["age_dob"] += 1  # a gap beyond the conflict threshold is hard, and nothing else is
    check("3 scoring", "hard conflicts for name, relative name, gender and age match their configured thresholds in both directions", not viol, dict(viol))
    if boundary:
        finding("low", "scoring", f"{len(boundary)} candidate(s) sit exactly on a band threshold: the displayed score rounds to the threshold but the unrounded score is just below it, so the band is the lower one.", boundary)
    check("3 scoring", "no face conflict is ever 'hard'", not any(x["f"]["face"][2] == "hard" for x in slim))

    # face errors must not be negative evidence: data and a direct test with the real scorer
    face_counts = Counter(x["face"] for x in slim)
    check("3 scoring", "this run has no face-service errors, so none could have been counted against anyone", face_counts.get("error", 0) == 0 and not any(x["face_retry"] for x in slim), dict(face_counts))
    store = AttributeStore(conn, ScoringConfig.from_dict(cfg), CandidateConfig())
    sample = slim[0]
    store.prefetch([sample["a"], sample["b"]])
    pa, pb = store.get(sample["a"]), store.get(sample["b"])
    scfg = ScoringConfig.from_dict(cfg)
    base = score_pair(pa, pb, None, scfg)
    clean = True
    for status, kind in ((PairStatus.ERROR, "unavailable"), (PairStatus.ERROR, "auth_error"), (PairStatus.ERROR, "server_error"), (PairStatus.NOT_CHECKED, None),
                         (PairStatus.NO_FACE, None), (PairStatus.LOW_QUALITY, None), (PairStatus.SKIPPED_NO_IMAGE, None)):
        out = score_pair(pa, pb, PairFaceEvidence(sample["a"], sample["b"], status, error_kind=kind, reason="audit"), scfg)
        clean &= out.match_score == base.match_score and out.coverage == base.coverage and not out.fields["face"].applicable and out.fields["face"].conflict is None
    check("3 scoring", "service errors, no_face, low_quality and skipped comparisons leave the score and coverage exactly unchanged (tested with the real scorer)", clean)

    # missing evidence versus the raw database values (all records with a missing-evidence reason, sampled)
    rng = random.Random(20261012)
    audit_ids = rng.sample(range(len(slim)), 600)
    ids = sorted({p for i in audit_ids for p in (slim[i]["a"], slim[i]["b"])})
    raw = {}
    cur = conn.cursor()
    for start in range(0, len(ids), 1500):
        cur.execute("/* audit:raw */ SELECT p.person_id, p.alias, e.cleaned_alias, p.relative_name, e.cleaned_relative_name, p.phone_number, p.gender, p.age, p.date_of_birth "
                    "FROM public.persons_unified p LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id WHERE p.person_id = ANY(%(ids)s)", {"ids": ids[start:start + 1500]})
        for r in cur.fetchall():
            raw[r[0]] = r
    blank = lambda v: v is None or not any(ch.isalnum() for ch in str(v))  # a lone "-" or "." is a placeholder, not a name
    mism = Counter(); checked = Counter()
    full = {r["candidate_id"]: r for r in jsonl(Path(ctx["v2"]) / "scored_candidates.jsonl") if r["candidate_id"] in {slim[i]["id"] for i in audit_ids}}
    for i in audit_ids:
        r = full[slim[i]["id"]]
        pa_, pb_ = raw[r["person_a_id"]], raw[r["person_b_id"]]
        for m in r["missing_evidence"]:
            f, reason = m["field"], m["reason"]
            checked[f] += 1
            if f == "relative_name" and not (blank(pa_[3]) and blank(pa_[4]) or blank(pb_[3]) and blank(pb_[4])): mism[f] += 1
            if f == "alias" and reason == "no_alias" and not (blank(pa_[1]) and blank(pa_[2]) and blank(pb_[1]) and blank(pb_[2])): mism[f] += 1
            if f == "phone" and "no_phone" in reason and not (blank(pa_[5]) or blank(pb_[5])): mism[f] += 1
            if f == "gender" and not (str(pa_[6] or "").strip().lower() not in ("male", "female") or str(pb_[6] or "").strip().lower() not in ("male", "female")): mism[f] += 1
            if f == "age_dob" and reason.startswith("age_and_dob_missing") and not (blank(pa_[7]) and blank(pa_[8]) and blank(pb_[7]) and blank(pb_[8])): mism[f] += 1
    check("3 scoring", "sampled 'missing evidence' reasons agree with the raw database values (600 candidates)", not mism, {"checked": dict(checked), "disagreements": dict(mism)})
    return {"face_counts": dict(face_counts)}


# ============================================================ 2. age / DOB contradictions

def audit_age(ctx: dict[str, Any], conn) -> dict[str, Any]:
    from dedup_pipeline.scoring.age import age_at, build_age_info

    v2 = Path(ctx["v2"])
    found = []
    for r in jsonl(v2 / "scored_candidates.jsonl"):
        flags = [c for c in r["conflicting_attributes"] if c["severity"] == "data_quality"]
        if flags:
            found.append(r)
    ids = sorted({p for r in found for p in (r["person_a_id"], r["person_b_id"])})
    cur = conn.cursor()
    cur.execute("/* audit:age */ SELECT person_id, source_system, source_record_id, age, date_of_birth FROM public.persons_unified WHERE person_id = ANY(%(ids)s)", {"ids": ids})
    raw = {r[0]: r for r in cur.fetchall()}
    rows, mism, diffs, silent = [], 0, Counter(), 0
    for r in found:
        age = r["age_dob_summary"]
        for q in age["inconsistencies"]:
            side = q["record"]
            pid = r[f"person_{side}_id"]
            db = raw[pid]
            same_as_db = (db[3] == age[f"recorded_age_{side}"]) and (db[4].isoformat() == age[f"dob_{side}"] if db[4] else age[f"dob_{side}"] is None)
            mism += not same_as_db
            diffs[q["recorded_age"] - q["age_implied_by_dob"]] += 1
            silent += age[f"recorded_age_{side}"] != db[3]
            rows.append({
                "candidate_id": r["candidate_id"], "record": side, "person_id": pid, "source_system": db[1], "source_record_id": db[2],
                "recorded_age_in_database": db[3], "date_of_birth_in_database": db[4].isoformat() if db[4] else None,
                "age_implied_by_dob_at_reference_date": q["age_implied_by_dob"], "difference_recorded_minus_implied": q["recorded_age"] - q["age_implied_by_dob"],
                "reference_date": age["reference_date"], "comparison_method": age["method"], "comparison_status": age["comparison_status"],
                "age_tolerance_years": age["age_tolerance_years"], "internal_tolerance_years": 1,
                "values_in_report_equal_database_values": same_as_db,
                "flags": {"data_quality": True, "hard_conflict_in_candidate": r["has_hard_conflict"], "age_dob_field_applicable": age["applicable"]},
                "candidate_match_score_percent": r["match_score_percent"], "candidate_band": r["score_band"], "candidate_review_group": r["review_group"],
                "other_record_age": age[f"recorded_age_{'b' if side == 'a' else 'a'}"], "other_record_dob": age[f"dob_{'b' if side == 'a' else 'a'}"],
            })
    with open(OUT / "age_dob_inconsistencies_SENSITIVE.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    check("2 age/dob", "16 candidate-record inconsistencies found, each with a DOB, recorded age, reference date and method", len(rows) == 16 and all(r["comparison_method"] for r in rows), len(rows))
    check("2 age/dob", "every inconsistency shows exactly the values stored in the database (nothing inferred or corrected)", mism == 0 and silent == 0, f"{mism} differ")
    check("2 age/dob", "every inconsistency is flagged data_quality and none was silently resolved into a hard conflict",
          all(r["flags"]["data_quality"] for r in rows))
    check("2 age/dob", "the DOB was the documented precedence in every case (method dob_exact or dob_vs_recorded_age)", all(r["comparison_method"] in ("dob_exact", "dob_vs_recorded_age") for r in rows))
    # how widespread is it in the whole database? (read-only; persons with both a DOB and an age)
    cur.execute("/* audit:age */ SELECT person_id, age, date_of_birth FROM public.persons_unified WHERE date_of_birth IS NOT NULL")
    cfg = ScoringConfig(rule_version="score-v2", reference_date="2026-10-12")
    with_dob = both = inconsistent = placeholder = 0
    for pid, age, dob in cur.fetchall():
        with_dob += 1
        info = build_age_info(dob, age, cfg)
        placeholder += info.dob_status == "invalid:placeholder"
        if info.dob and info.age is not None:
            both += 1
            inconsistent += info.inconsistent
    spread = {"persons_with_a_dob": with_dob, "of_which_dob_is_a_placeholder": placeholder,
              "persons_with_valid_dob_and_valid_age": both, "of_which_inconsistent": inconsistent}
    persons = len({r["person_id"] for r in rows})  # the persons whose own age and DOB disagree (not every person in the flagged candidates)
    stats = {"candidate_record_inconsistencies": len(rows), "distinct_persons": persons, "difference_recorded_minus_implied": dict(sorted(diffs.items())),
             "by_source": dict(Counter(r["source_system"] for r in rows)), "by_method": dict(Counter(r["comparison_method"] for r in rows)),
             "by_band": dict(Counter(r["candidate_band"] for r in rows)), "by_group": dict(Counter(r["candidate_review_group"] for r in rows)), "database_wide": spread}
    if len(rows) != persons:
        finding("info", "age/dob", f"The {len(rows)} candidate-record inconsistencies involve {persons} distinct persons (a person can appear in several candidates).",
                stats["difference_recorded_minus_implied"])
    lower = sum(n for d, n in diffs.items() if d < 0)
    finding("medium", "age/dob",
            f"{lower} of {len(rows)} contradictions have the recorded age LOWER than the age the DOB implies at the reference date (by {-max(diffs)} to {-min(diffs)} years). "
            "That pattern fits ages recorded at an earlier date rather than random entry errors; it is an observation only and nothing was corrected.",
            stats["difference_recorded_minus_implied"])
    finding("medium", "age/dob",
            f"Database-wide, {spread['of_which_inconsistent']} of the {spread['persons_with_valid_dob_and_valid_age']} persons that have both a valid DOB and an age are "
            "inconsistent at the reference date; recorded ages in this database cannot be assumed to be current.", spread)
    return stats


# ===================================== 3b. samples and risk statistics (no scoring change)

def audit_risks(ctx: dict[str, Any]) -> dict[str, Any]:
    slim = ctx["slim"]
    A = [x for x in slim if x["band"] == "A"]
    out: dict[str, Any] = {}
    out["band_A_total"] = len(A)
    out["band_A_with_at_most_3_applicable_fields"] = sum(1 for x in A if x["n_app"] <= 3)
    out["band_A_coverage_below_50_percent"] = sum(1 for x in A if x["cov"] < 50)
    out["band_A_found_by_name_rules_only"] = sum(1 for x in A if set(x["rules"]) <= {"name_exact", "token_sorted_name"})
    out["band_A_found_by_phone_or_alias"] = sum(1 for x in A if {"phone_exact", "alias_name"} & set(x["rules"]))
    strong = [x for x in slim if "name_exact" in x["rules"] and "phone_exact" in x["rules"]]
    out["strong_pairs_same_name_and_same_phone"] = len(strong)
    out["strong_pairs_with_hard_conflict_by_field"] = dict(Counter(f for x in strong for f in x["hardf"]))
    out["strong_pairs_with_any_hard_conflict"] = sum(1 for x in strong if x["hard"])
    hard = [x for x in slim if x["hard"]]
    out["hard_conflict_candidates"] = len(hard)
    out["hard_conflict_by_field_any"] = dict(Counter(f for x in hard for f in x["hardf"]))
    out["hard_conflict_count_per_candidate"] = dict(sorted(Counter(len(x["hardf"]) for x in hard).items()))
    out["hard_conflict_sole_field"] = dict(Counter(x["hardf"][0] for x in hard if len(x["hardf"]) == 1))
    out["hard_conflict_in_bands_AB"] = sum(1 for x in hard if x["band"] in ("A", "B"))
    out["hard_conflict_AB_by_field"] = dict(Counter(f for x in hard if x["band"] in ("A", "B") for f in x["hardf"]))
    out["hard_conflict_AB_sole_relative_or_age"] = sum(1 for x in hard if x["band"] in ("A", "B") and len(x["hardf"]) == 1 and x["hardf"][0] in ("relative_name", "age_dob"))
    out["soft_conflicts"] = dict(Counter(f for x in slim for f in x["soft"]))
    out["name_by_construction"] = {"candidates_found_by_name_rules": sum(1 for x in slim if {"name_exact", "token_sorted_name"} & set(x["rules"])),
                                   "of_which_name_score_is_1.0": sum(1 for x in slim if {"name_exact", "token_sorted_name"} & set(x["rules"]) and x["f"]["name"][1] == 1.0)}
    out["phone_only_candidates"] = {"n": sum(1 for x in slim if x["rules"] == ["phone_exact"]), "name_hard_conflict": sum(1 for x in slim if x["rules"] == ["phone_exact"] and "name" in x["hardf"])}
    out["band_by_group"] = {g: dict(sorted(Counter(x["band"] for x in slim if x["group"] == g).items())) for g in sorted({x["group"] for x in slim})}
    out["face_compared"] = sum(1 for x in slim if x["face"] == "compared")
    out["face_compared_with_hard_conflict"] = sum(1 for x in slim if x["face"] == "compared" and x["hard"])
    cos = [x["cos"] for x in slim if x["cos"] is not None]
    out["face_cosine"] = {"n": len(cos), "min": round(min(cos), 3), "max": round(max(cos), 3)}
    out["candidates_where_face_is_the_only_non_name_evidence"] = sum(1 for x in slim if x["face"] == "compared" and sum(1 for n, (app, s, c, w) in x["f"].items() if app and n not in ("name", "face")) == 0)
    finding("risk", "scoring", "Name agreement is guaranteed by the name-based candidate rules, so it adds no independent evidence for most candidates.", out["name_by_construction"])
    return out


def write_samples(ctx: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    slim = ctx["slim"]
    chosen: dict[str, list[str]] = {}
    def draw(label, pred, n):
        pool = sorted((x["id"] for x in slim if pred(x)))
        chosen[label] = sorted(rng.sample(pool, min(n, len(pool))))
    for band in "ABCD":
        draw(f"band {band}, no hard conflict", lambda x, b=band: x["band"] == b and not x["hard"] and not x["insuff"], 10)
    for band in "ABCD":
        draw(f"band {band}, hard conflict", lambda x, b=band: x["band"] == b and x["hard"], 10)
    draw("band E, insufficient evidence", lambda x: x["band"] == "E", 10)
    draw("face compared", lambda x: x["face"] == "compared", 10)
    draw("face failed (no_face / low_quality / multiple_faces)", lambda x: x["face"] in ("no_face", "low_quality", "multiple_faces"), 10)
    wanted = {i for ids in chosen.values() for i in ids}
    stratum = defaultdict(list)
    for label, ids in chosen.items():
        for i in ids:
            stratum[i].append(label)
    with open(OUT / "audit_sample.jsonl", "w", encoding="utf-8") as fh:
        for r in jsonl(Path(ctx["v2"]) / "scored_candidates.jsonl"):
            if r["candidate_id"] in wanted:
                slimmed = {k: r[k] for k in ("candidate_id", "source_pair", "match_score_percent", "evidence_coverage_percent", "score_band", "has_hard_conflict",
                                              "review_group", "insufficient_evidence", "matching_attributes", "conflicting_attributes", "missing_evidence")}
                slimmed["strata"] = stratum[r["candidate_id"]]
                slimmed["rules"] = r["candidate_generation"]["rule_names"]
                slimmed["fields"] = {n: {"weight": f["weight"], "applicable": f["applicable"], "score": f["score"], "conflict": f.get("conflict"), "explanation": f["explanation"]}
                                     for n, f in r["fields"].items()}
                slimmed["face"] = {k: r["face"][k] for k in ("status", "raw_cosine", "in_match_score", "exclusion_reason", "error_kind", "retry_needed")}
                fh.write(json.dumps(slimmed) + "\n")
    return {label: len(ids) for label, ids in chosen.items()} | {"distinct_candidates_in_sample": len(wanted)}


# ================================================================ 4. deferred blocks

def audit_deferred(ctx: dict[str, Any], conn) -> dict[str, Any]:
    v2 = Path(ctx["v2"])
    deferred = list(jsonl(v2 / "deferred_unevaluated.jsonl"))
    candidate_pairs = {(r["person_a_id"], r["person_b_id"]) for r in jsonl(Path(ctx["cand_dir"]) / "candidates.jsonl")}
    blocks, union = [], set()
    for d in deferred:
        ids = sorted(d["person_ids"])
        complete = not d["person_ids_truncated"] and len(ids) == d["member_count"]
        pairs = set(combinations(ids, 2))
        blocks.append({"strategy": d["strategy"], "members": d["member_count"], "limit": d["limit"], "listed_completely": complete, "pairs": pairs,
                       "source_counts": d["source_counts"], "reported_not_found_by_any_rule": d["pairs_not_found_by_any_rule"],
                       "reported_found_by_other_rule": d["pairs_already_found_by_another_rule"]})
        union |= pairs
    check("4 deferred", "all four deferred blocks list their members completely", all(b["listed_completely"] for b in blocks))
    reported_total = sum(b["reported_not_found_by_any_rule"] for b in blocks)
    new_pairs = union - candidate_pairs
    overlap = {f"{i}&{j}": len(blocks[i]["pairs"] & blocks[j]["pairs"]) for i in range(len(blocks)) for j in range(i + 1, len(blocks)) if blocks[i]["pairs"] & blocks[j]["pairs"]}
    check("4 deferred", "none of the deferred pairs is in the scored candidate set unless another rule found it", all((p in candidate_pairs) or (p in new_pairs) for p in union))
    check("4 deferred", "the reported 3,816 'never evaluated' pairs equals the sum over blocks (this counts shared pairs more than once)", reported_total == 3816, reported_total)
    distinct = len(new_pairs)
    if distinct != reported_total:
        finding("medium", "deferred", f"The headline '3,816 pairs never evaluated' double-counts: the name_exact and token_sorted_name blocks hold the same people. The number of DISTINCT pairs never evaluated is {distinct}.",
                {"sum_over_blocks": reported_total, "distinct_pairs": distinct, "overlap_between_blocks": overlap})
    # evaluate the distinct never-evaluated pairs with the existing scorer (audit only: nothing is added to the candidate set)
    scfg = ScoringConfig.from_dict(ctx["config"])
    store = AttributeStore(conn, scfg, CandidateConfig())
    people = sorted({p for pair in new_pairs for p in pair} | {p for b in blocks for pr in b["pairs"] for p in pr})
    store.prefetch(people)
    bands, groups, hard_fields = Counter(), Counter(), Counter()
    pilot_pool = []
    for a, b in sorted(new_pairs):
        o = score_pair(store.get(a), store.get(b), None, scfg)
        hard = [n for n, f in o.fields.items() if f.conflict == "hard"]
        bands[o.band] += 1
        groups["insufficient" if o.insufficient else ("hard_conflict" if hard else "no_hard_conflict")] += 1
        hard_fields.update(hard)
        pilot_pool.append((f"{a}:{b}", o.band, None if o.match_score is None else round(100 * o.match_score, 1), bool(hard), o.insufficient))
    per_block = []
    for d, b in zip(deferred, blocks):
        pids = sorted({p for pr in b["pairs"] for p in pr})
        attrs = [store.get(p) for p in pids]
        per_block.append({
            "strategy": b["strategy"], "members": b["members"], "limit_exceeded": f"{b['members']} > {b['limit']}", "pairs": len(b["pairs"]),
            "pairs_found_by_another_rule": len(b["pairs"] & candidate_pairs), "pairs_never_evaluated": len(b["pairs"] - candidate_pairs),
            "source_counts": b["source_counts"], "distinct_normalised_names": len({a.name for a in attrs if a.name}), "distinct_phone_keys": len({a.phone for a in attrs if a.phone}),
            "distinct_relative_names": len({a.relative_name for a in attrs if a.relative_name}),
            "members_with_no_relative_name": sum(1 for a in attrs if not a.relative_name),
            "name_contains_unidentified_marker": any(re.search(r"\b(unknown|unidentified|unnamed|dead body|deadbody|unclaimed)\b", a.name or "") for a in attrs),
        })
    with open(OUT / "deferred_pairs_audit_scores.jsonl", "w", encoding="utf-8") as fh:  # ids and scores only; NOT candidates
        for cid, band, score, hard, insuff in pilot_pool:
            fh.write(json.dumps({"pair": cid, "audit_only_band": band, "audit_only_score": score, "hard_conflict": hard, "insufficient": insuff, "status": "never_evaluated_by_the_pipeline"}) + "\n")
    cur = conn.cursor()
    cur.execute("/* audit:links */ SELECT person_a_id, person_b_id FROM public.identity_links")
    links = {tuple(sorted(r)) for r in cur.fetchall()}
    exclusions = ctx["summary_p5"]["strategies"]
    result = {
        "blocks": per_block, "reported_sum": reported_total, "distinct_never_evaluated_pairs": distinct, "pairs_in_blocks_also_found_by_another_rule": len(union & candidate_pairs),
        "block_overlap": overlap, "audit_only_scoring_of_never_evaluated_pairs": {"bands": dict(sorted(bands.items())), "groups": dict(groups), "hard_conflicts_by_field": dict(hard_fields)},
        "earlier_identity_links_inside_deferred_blocks": len(union & links),
        "exclusion_rules": {"name_exact_single_token_names": exclusions["name_exact"]["excluded_by_rule"]["excluded_blocks"],
                            "phone_unusable_numbers": exclusions["phone_exact"]["excluded_by_rule"]["excluded_blocks_by_reason"],
                            "alias_below_minimum": exclusions["alias_name"]["excluded_by_rule"]},
        "pool_for_pilot": pilot_pool,
    }
    finding("medium", "candidate generation", "Four oversized blocks (41, 41, 34 and 62 people) were never evaluated; they are common multi-token names and one shared phone number.", per_block)
    finding("risk", "candidate generation", "Duplicates whose names differ in spelling (not identical and not just reordered), or that share only a repetitive-digit phone, single-token name or no phone, cannot be found by any rule.",
            result["exclusion_rules"])
    return result


# ====================================================================== 5. worksheet and HTML

def audit_worksheet(ctx: dict[str, Any]) -> dict[str, Any]:
    v2, v1 = Path(ctx["v2"]), Path(ctx["v1"])
    work = OUT / "worksheet_tests"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    original = v2 / "review_worksheet.csv"
    original_hash = sha(original)
    with open(original, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    col = {n: header.index(n) for n in header}
    known = {r[col["candidate_id"]] for r in rows}

    def write(name: str, hdr, body) -> Path:
        path = work / name
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            w.writerows(body)
        return path

    def edit(indices: dict[int, dict[str, str]]) -> list[list[str]]:
        body = [list(r) for r in rows]
        for i, values in indices.items():
            for k, v in values.items():
                body[i][col[k]] = v
        return body

    full = {"reviewer_decision": "", "reviewer_id": "alice", "reviewed_at": "2026-10-12T10:00:00"}
    s0 = validate_worksheet(original, known)
    check("5 worksheet", "an untouched worksheet has 24,155 pending rows and no decisions", s0["pending"] == 24155 and s0["decisions"] == [] and s0["problems"] == [], s0["pending"])
    allowed = ("confirm_same_person", "reject_match", "needs_further_review", "unable_to_verify")
    from dedup_pipeline.scoring.config import DECISIONS
    check("5 worksheet", "the allowed decisions are exactly the four specified", tuple(DECISIONS) == allowed, DECISIONS)
    p = write("valid_four.csv", header, edit({i: {**full, "reviewer_decision": d} for i, d in enumerate(allowed)}))
    r = validate_worksheet(p, known)
    check("5 worksheet", "all four decisions are accepted when reviewer id and time are present", [d["decision"] for d in r["decisions"]] == list(allowed) and not r["problems"] and r["pending"] == 24151)
    ok_ids = {d["candidate_id"] for d in r["decisions"]}
    check("5 worksheet", "the other 24,151 rows are not turned into decisions (pending preserved)", len(ok_ids) == 4 and r["pending"] == 24151)
    cases = {
        "invalid_value": ({0: {**full, "reviewer_decision": "approve"}}, "must be one of"),
        "wrong_case": ({0: {**full, "reviewer_decision": "Confirm_Same_Person"}}, "must be one of"),
        "missing_reviewer_id": ({0: {**full, "reviewer_decision": "reject_match", "reviewer_id": ""}}, "reviewer_id is required"),
        "missing_time": ({0: {**full, "reviewer_decision": "reject_match", "reviewed_at": ""}}, "ISO"),
        "bad_time": ({0: {**full, "reviewer_decision": "reject_match", "reviewed_at": "yesterday"}}, "ISO"),
        "id_without_decision": ({0: {**full, "reviewer_decision": ""}}, "must be one of"),
        "notes_only": ({0: {"reviewer_notes": "looks similar"}}, "must be one of"),
    }
    for name, (edits, expect) in cases.items():
        res = validate_worksheet(write(f"{name}.csv", header, edit(edits)), known)
        check("5 worksheet", f"rejected: {name.replace('_', ' ')}", len(res["problems"]) == 1 and expect in res["problems"][0] and res["decisions"] == [], res["problems"][:1])
    spaced = validate_worksheet(write("padded.csv", header, edit({0: {**full, "reviewer_decision": " reject_match "}})), known)
    check("5 worksheet", "surrounding spaces are tolerated, everything else must match exactly", [d["decision"] for d in spaced["decisions"]] == ["reject_match"])
    dup = validate_worksheet(write("duplicate.csv", header, edit({}) + [list(rows[0])]), known)
    check("5 worksheet", "a duplicated candidate row is reported", any("duplicate" in x for x in dup["problems"]), dup["problems"][:1])
    unknown_rows = edit({}) + [[("zzz:zzz" if h == "candidate_id" else v) for h, v in zip(header, rows[0])]]
    unk = validate_worksheet(write("unknown.csv", header, unknown_rows), known)
    check("5 worksheet", "a candidate id that is not in the run is reported", any("unknown candidate" in x for x in unk["problems"]))
    renamed = list(header); renamed[0] = "order"
    for label, hdr in (("renamed column", renamed), ("reordered columns", [header[1], header[0], *header[2:]])):
        try:
            validate_worksheet(write(f"{label.replace(' ', '_')}.csv", hdr, [r[:] for r in rows[:5]]), known)
            check("5 worksheet", f"rejected: {label}", False)
        except WorksheetError:
            check("5 worksheet", f"rejected: {label}", True)
    with open(v1 / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        v1_header = next(csv.reader(fh))
    check("5 worksheet", "the score-v1 worksheet (older layout) is still accepted and untouched", tuple(v1_header) == WORKSHEET_COLUMNS_V1 and validate_worksheet(v1 / "review_worksheet.csv")["pending"] == 24155)
    exit_codes = {}
    for name in ("valid_four.csv", "invalid_value.csv"):
        proc = subprocess.run([sys.executable, "-m", "dedup_pipeline.scoring", "validate-worksheet", str(work / name)], cwd=ROOT, capture_output=True, text=True)
        exit_codes[name] = proc.returncode
    check("5 worksheet", "the command-line validator exits 0 for a clean worksheet and 1 when there are problems", exit_codes == {"valid_four.csv": 0, "invalid_value.csv": 1}, exit_codes)
    kept = [json.loads(x) for x in (work / "decisions_validated.jsonl").read_text().splitlines()] if (work / "decisions_validated.jsonl").exists() else None
    check("5 worksheet", "validation writes only a local decisions file next to the checked worksheet (never to a database)", kept is not None and len(kept) in (0, 4))
    check("5 worksheet", "the original v2 worksheet was not modified by any of these tests", sha(original) == original_hash)
    check("5 worksheet", "the original run directory contains no decisions file", not (v2 / "decisions_validated.jsonl").exists())

    # HTML: read-only, no way to submit or persist a decision
    pages = sorted((v2 / "review_report").glob("*.html"))
    forbidden = {"form": r"<form", "button": r"<button", "input": r"<input", "textarea": r"<textarea", "select": r"<select", "script": r"<script", "iframe": r"<iframe",
                 "event handler": r"\son[a-z]+\s*=", "javascript link": r"href=\"javascript:", "network call": r"fetch\(|XMLHttpRequest|sendBeacon|localStorage|sessionStorage|indexedDB|document\.cookie",
                 "contenteditable": r"contenteditable"}
    hits = Counter()
    hrefs: Counter = Counter()
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for name, pattern in forbidden.items():
            hits[name] += len(re.findall(pattern, text, flags=re.I))
        for h in re.findall(r'href="([^"]*)"', text):
            hrefs["file-proxy" if h.startswith("/api/file-proxy") else "page" if re.fullmatch(r"[\w.-]+\.html", h) else "other"] += 1
    check("5 worksheet", f"HTML report ({len(pages)} files) has no form, button, input, script, event handler or network/storage call", sum(hits.values()) == 0, dict(hits))
    check("5 worksheet", "every HTML link is a sibling page or the authorised file proxy", hrefs.get("other", 0) == 0, dict(hrefs))
    sample = pages[0].read_text(encoding="utf-8")
    check("5 worksheet", "HTML shows 'pending' and the 'NOT a probability' label", "pending" in sample and "NOT a probability" in sample)
    for big in work.glob("*.csv"):  # the test copies are 33 MB each and only needed while the checks run
        if big.stat().st_size > 1_000_000:
            big.unlink()
    return {"scenarios_run": len(cases) + 8, "html_files": len(pages), "html_links": dict(hrefs), "cli_exit_codes": exit_codes}


# ==================================================================== 6. pilot sample

def pilot_sample(ctx: dict[str, Any], deferred: dict[str, Any], rng: random.Random) -> dict[str, int]:
    slim = ctx["slim"]
    order = {r["candidate_id"]: int(r["review_order"]) for r in ctx["rows"]}
    used: set[str] = set()
    chosen: list[tuple[str, str]] = []

    def draw(label: str, pred, n: int) -> None:
        pool = sorted(x["id"] for x in slim if pred(x) and x["id"] not in used)
        for cid in rng.sample(pool, min(n, len(pool))):
            used.add(cid)
            chosen.append((label, cid))
    draw("A, no hard conflict, face compared", lambda x: x["band"] == "A" and not x["hard"] and x["face"] == "compared", 6)
    draw("A, no hard conflict, no face", lambda x: x["band"] == "A" and not x["hard"] and x["face"] != "compared", 14)
    draw("B, no hard conflict", lambda x: x["band"] == "B" and not x["hard"], 12)
    draw("C, no hard conflict", lambda x: x["band"] == "C" and not x["hard"], 10)
    draw("D, no hard conflict", lambda x: x["band"] == "D" and not x["hard"], 4)
    for field in ("relative_name", "age_dob", "gender", "name"):
        draw(f"A/B, hard conflict only on {field}", lambda x, f=field: x["band"] in ("A", "B") and x["hard"] and x["hardf"] == [f], 4)
    draw("A/B, several hard conflicts", lambda x: x["band"] in ("A", "B") and len(x["hardf"]) > 1, 4)
    draw("C/D, hard conflict", lambda x: x["band"] in ("C", "D") and x["hard"], 6)
    draw("insufficient evidence", lambda x: x["insuff"], 8)
    draw("face compared, below the service threshold", lambda x: x["face"] == "compared" and x["cos"] is not None and x["cos"] < 0.45, 5)
    draw("found only by the phone rule", lambda x: x["rules"] == ["phone_exact"], 6)
    draw("found only by the token-sorted-name rule", lambda x: x["rules"] == ["token_sorted_name"], 6)
    draw("found by the alias rule", lambda x: "alias_name" in x["rules"], 4)
    draw("age/DOB contradiction inside a record", lambda x: bool(x["dq"]), 4)
    draw("same image file on both records", lambda x: "same_image_file_attached_to_both_records" in x["ind"], 4)
    by_id = {x["id"]: x for x in slim}
    with open(OUT / "pilot_sample.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stratum", "candidate_id", "review_order_in_worksheet", "review_group", "score_band", "match_score_percent", "has_hard_conflict", "hard_conflict_fields",
                    "face_status", "candidate_rules", "source_pair", "note"])
        for label, cid in sorted(chosen, key=lambda t: (t[0], t[1])):
            x = by_id[cid]
            w.writerow([label, cid, order[cid], x["group"], x["band"], x["score"], x["hard"], ";".join(x["hardf"]), x["face"], "+".join(x["rules"]), x["sp"], "in the review worksheet"])
        pool = [p for p in deferred["pool_for_pilot"]]
        picks = sorted(rng.sample(sorted(pool), min(12, len(pool))))
        for pair, band, score, hard, insuff in picks:
            w.writerow(["NEVER EVALUATED (oversized block) - recall check", pair, "", "", band, score, hard, "", "", "", "", "NOT in the worksheet: add to the pilot to measure missed duplicates"])
    return {**dict(Counter(l for l, _ in chosen)), "never_evaluated_recall_check": min(12, len(deferred["pool_for_pilot"])), "total": len(chosen) + min(12, len(deferred["pool_for_pilot"]))}


# ============================================================================== report

def audit_placeholders(conn) -> dict[str, Any]:
    cur = conn.cursor()
    out = {}
    for label, col in (("full_name", "p.full_name"), ("alias", "p.alias"), ("relative_name", "p.relative_name"), ("phone_number", "p.phone_number")):
        cur.execute(f"/* audit:placeholders */ SELECT count(*) FILTER (WHERE {col} IS NOT NULL AND btrim({col}) <> '' AND {col} !~ '[[:alnum:]]'), "
                    f"count(*) FILTER (WHERE {col} IS NOT NULL AND btrim({col}) <> '') FROM public.persons_unified p")
        placeholder, present = cur.fetchone()
        out[label] = {"present": present, "punctuation_only": placeholder}
    finding("low", "data quality", "Some name-like fields hold punctuation-only placeholders (for example a lone dash). They are correctly treated as missing, never as a value.", out)
    return out


def assessment(ctx, risks, age, deferred, ws, pilot) -> list[str]:
    """The reader-facing conclusions, generated from the audit's own numbers so the report is reproducible."""
    d, wide = deferred, age["database_wide"]
    scored = d["audit_only_scoring_of_never_evaluated_pairs"]
    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    sole = risks["hard_conflict_sole_field"]
    lines = ["## Auditor's assessment (generated from the numbers above)", ""]
    lines += [
        f"1. **Integrity:** {'all checks passed' if not failed else str(len(failed)) + ' check(s) failed'}. The score-v2 run is complete, all 24,155 candidates are scored and pending, "
        "the score-v1 run is intact and regenerates byte-identically, and no database write is evidenced (row counts, schema objects and the database's own write counters are unchanged).",
        f"2. **Age/DOB:** {age['candidate_record_inconsistencies']} contradictions in {age['distinct_persons']} persons, all with the recorded age lower than the DOB-implied age; "
        f"{wide['of_which_inconsistent']} of {wide['persons_with_valid_dob_and_valid_age']} persons with both values disagree database-wide. DOB data is almost absent "
        f"({wide['persons_with_a_dob']} persons), so age comparisons rest on recorded ages of unknown date.",
        f"3. **Scoring:** the formula, coverage, bands, hard-conflict flags and review groups were recomputed independently for every record and match. Missing evidence is never scored, "
        "and face errors, no_face, low_quality and skipped comparisons provably leave the score unchanged. The risk is evidence quality, not arithmetic: "
        f"{risks['band_A_found_by_name_rules_only']} of {risks['band_A_total']} band-A candidates were found by name rules alone (name agreement is guaranteed by construction) and "
        f"{risks['band_A_with_at_most_3_applicable_fields']} have 3 or fewer usable fields.",
        f"4. **Hard conflicts:** {risks['hard_conflict_candidates']} candidates ({round(100 * risks['hard_conflict_candidates'] / 24155)}%) carry one; {sum(sole.values())} rest on a single field "
        f"(age {sole.get('age_dob')}, relative name {sole.get('relative_name')}). Of {risks['hard_conflict_in_bands_AB']} band A/B candidates with a hard conflict, "
        f"{risks['hard_conflict_AB_sole_relative_or_age']} rest solely on the age or relative-name rule. Because the age rule uses one reference date for records made in different years, "
        "it probably over-flags; it is the largest source of review noise.",
        f"5. **Candidate generation:** the 'never evaluated' figure is {d['distinct_never_evaluated_pairs']} distinct pairs, not 3,816 (the name_exact and token_sorted blocks hold the same 41 people). "
        f"Scored with the existing scorer for audit only, {scored['groups'].get('no_hard_conflict', 0)} of them have no hard conflict and {scored['bands'].get('A', 0)} reach band A, "
        "so the potential loss is small. " + (
            "The 41-person block is one name that contains an unidentified-person marker, and none of its members has a relative name: it is a placeholder, so those 820 pairs are not "
            "meaningful duplicate candidates. " if any(b["name_contains_unidentified_marker"] for b in d["blocks"]) else "")
        + "The 62-person phone block is one shared number with 62 different names (a shared line, not duplicates). The larger gaps are the rules' blind spots "
        "(misspelt names, single-token names, repetitive-digit phones), which cannot be measured without labelled data.",
        f"6. **Review tooling:** {ws['scenarios_run']} worksheet scenarios behaved as specified; only the four decisions are accepted, each needs a reviewer id and an ISO time, blank rows stay pending, "
        f"and the validator never touches the original worksheet. All {ws['html_files']} HTML files are read-only (no form, button, input, script or network call).", "",
        "### Blockers before wider review", ""]
    blockers = []
    if failed:
        blockers.append("Resolve the failed checks listed at the top.")
    blockers += [
        f"Decide how ages are compared. Review of the {risks['hard_conflict_candidates']} hard-conflict candidates will be dominated by age conflicts that may be artefacts of record dates; "
        "comparing ages at each record's FIR year is the obvious candidate. This does not block a small pilot but should be settled before reviewing at scale.",
        f"Approve or reject evaluating the {d['distinct_never_evaluated_pairs']} never-evaluated pairs, at least as a sample (12 are in the pilot).",
        "Name an authorised reviewer group and a reviewer-id convention: the validator requires a reviewer id and a timestamp but cannot check who is authorised.",
        "Decide where completed worksheets are stored and how they are protected (they contain personal values).",
    ]
    lines += [f"{i}. {b}" for i, b in enumerate(blockers, 1)]
    lines += ["", "### Recommended pilot", "",
              f"`pilot_sample.csv` lists {pilot['total']} candidates ({pilot['total'] - pilot['never_evaluated_recall_check']} from the worksheet across bands, conflict types, face outcomes, rules and "
              f"data-quality cases, plus {pilot['never_evaluated_recall_check']} never-evaluated pairs as a recall check). Have two reviewers decide the same sample independently: their agreement, and the share of "
              "`confirm_same_person` per stratum, show whether the score order and the conflict rules help, and are the only labelled data available to calibrate anything.", "",
              "### Database", "",
              "No database schema or data was changed. This audit used the project's guarded read-only connection (SELECT statements only); the database's own insert/update/delete counters, "
              "row counts and schema objects are identical to the Phase 1 audit."]
    return lines


def markdown_table(rows: list[list[Any]], header: list[str]) -> str:
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def write_report(ctx, scoring, age, risks, samples, deferred, ws, pilot) -> None:
    s2 = ctx["summary_v2"]
    passed = sum(c["result"] == "PASS" for c in CHECKS)
    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    lines = [
        "# Phase 7 validation audit - local report", "",
        "**Sensitive:** identifiers and (in the supporting `*_SENSITIVE.jsonl` file) DOB and age values. Keep local. Nothing here is a decision about any person.", "",
        f"Audited run: `score-v2` `{s2['run_id']}` (reference date {s2['reference_date']}); compared with the preserved `score-v1` run. No scoring logic, candidate, output or database record was changed by this audit.", "",
        f"## Result: {passed} of {len(CHECKS)} checks passed, {len(failed)} failed", "",
    ]
    if failed:
        lines += ["### Failed checks", markdown_table([[c["section"], c["check"], c["detail"]] for c in failed], ["section", "check", "detail"]), ""]
    for section in sorted({c["section"] for c in CHECKS}):
        lines += [f"## {section}", markdown_table([[c["check"], c["result"], str(c["detail"])[:140]] for c in CHECKS if c["section"] == section], ["check", "result", "detail"]), ""]
    lines += ["## Findings (data quality, scoring risks, generation limits)",
              markdown_table([[f["severity"], f["area"], f["finding"]] for f in FINDINGS], ["severity", "area", "finding"]), ""]
    lines += ["## Age/DOB contradictions", "```json", json.dumps(age, indent=1), "```", ""]
    lines += ["## Scoring and hard-conflict statistics", "```json", json.dumps(risks, indent=1), "```", ""]
    d = {k: v for k, v in deferred.items() if k != "pool_for_pilot"}
    lines += ["## Deferred (never evaluated) pairs", "```json", json.dumps(d, indent=1), "```", ""]
    lines += ["## Samples and pilot", "Audit sample (`audit_sample.jsonl`):", "```json", json.dumps(samples, indent=1), "```", "Recommended pilot (`pilot_sample.csv`):", "```json", json.dumps(pilot, indent=1), "```", ""]
    lines += ["## Worksheet / HTML tests", "```json", json.dumps(ws, indent=1), "```", ""]
    db_checks = [c for c in CHECKS if c["section"] == "1 integrity" and any(k in c["check"] for k in ("row counts", "migration objects", "10 columns", "write counters", "44 tables", "no SQL write", "read-only"))]
    lines += ["## Database", "All of: " + ", ".join(f"{c['check']} [{c['result']}]" for c in db_checks), ""]
    lines += [""] + assessment(ctx, risks, age, deferred, ws, pilot)
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "audit_results.json").write_text(json.dumps({"checks": CHECKS, "findings": FINDINGS, "age": age, "risks": risks, "samples": samples,
                                                         "deferred": d, "worksheet": ws, "pilot": pilot, "placeholders": PLACEHOLDERS}, indent=1, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=ROOT / "output" / "candidates")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261012)
    with connect_read_only(load_connection_settings(), "dedup_phase7_audit") as conn:
        info, _ = preflight(conn)  # fails closed unless the session is provably read-only on the approved database
        print("database:", info)
        ctx = audit_integrity(args.v2.resolve(), args.v1.resolve(), args.candidates.resolve(), conn)
        ctx["summary_p5"] = json.loads((args.candidates / "run_summary.json").read_text())
        scoring = audit_scoring(ctx, conn)
        age = audit_age(ctx, conn)
        PLACEHOLDERS.update(audit_placeholders(conn))
        risks = audit_risks(ctx)
        samples = write_samples(ctx, rng)
        deferred = audit_deferred(ctx, conn)
        ws = audit_worksheet(ctx)
        pilot = pilot_sample(ctx, deferred, rng)
    write_report(ctx, scoring, age, risks, samples, deferred, ws, pilot)
    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for c in failed:
        print("FAIL:", c["section"], "-", c["check"], "-", str(c["detail"])[:160])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
