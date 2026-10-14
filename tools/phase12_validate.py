"""Phase 12: final validation and manual-review readiness. Read-only against the database; writes only under output/audit/phase12/.

    snapshot  : database observations (read-only session, schema / index / object hashes, row counts, write counters, dataset stamp)
                and a content hash of every preserved output file.
    checks    : every Phase 12 check with PASS / FAIL / NOT VERIFIED, written as validation_results.json.

    .face-venv\\Scripts\\python tools\\phase12_validate.py snapshot --out output\\audit\\phase12\\snapshot_start.json
    .face-venv\\Scripts\\python tools\\phase12_validate.py checks --start ... --end ... --fresh-run ... --regenerated-candidates ... --out ...

No check writes to the database, to a preserved output, or to a review sample. Nothing here decides anything about an identity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"

PRESERVED_V2 = OUT / "scoring" / "runs" / "20261012T125026Z"
PRESERVED_V1 = OUT / "scoring" / "runs" / "20261012T121355Z"
PRESERVED_V3 = OUT / "scoring_v3" / "runs" / "score-v3-20261012"
CANDIDATES = OUT / "candidates"
EXPECTED_CANDIDATES = 24155
# everything a later phase must not have changed (scratch directories output/perf*, and Phase 12's own directory, are excluded)
PRESERVED_DIRS = ("scoring", "scoring_v3", "scoring_local", "scoring_local_default", "candidates", "image_resolution",
                  "audit/phase7", "audit/phase8_1", "audit/phase9", "audit/phase10", "audit/phase10_1", "audit/phase11")
ROW_COUNT_TABLES = ("persons_unified", "accused_unified", "crimes_unified", "person_enrichment", "accused_enrichment", "identity_links")
PENDING = {"status": "pending", "decision": None, "reviewer_id": None, "reviewed_at": None, "notes": None}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ===================================================================================================== snapshot

def db_snapshot() -> dict[str, Any]:
    from dedup_pipeline.candidates.runner import preflight
    from dedup_pipeline.db import connect_read_only, load_connection_settings

    with connect_read_only(load_connection_settings(None), "phase12_validation") as conn:
        cur = conn.cursor()

        def q(sql: str):
            cur.execute(sql)
            return cur.fetchall()

        info, stamp = preflight(conn)  # read-only session on the approved database, the candidates' dataset stamp
        snap = {
            "taken_at": utc(), "preflight": info,
            "transaction_read_only": q("SHOW transaction_read_only")[0][0],
            "default_transaction_read_only": q("SHOW default_transaction_read_only")[0][0],
            # the same queries and serialisation as the earlier phase snapshots, so the values are comparable
            "schema_hash": hashlib.sha256(json.dumps(q("select table_name, column_name, data_type from information_schema.columns where table_schema='public' order by 1,2"), default=str).encode()).hexdigest(),
            "indexes_hash": hashlib.sha256(json.dumps(q("select indexname, indexdef from pg_indexes where schemaname='public' order by 1"), default=str).encode()).hexdigest(),
            "objects": list(q("select count(*) filter (where relkind='r'), count(*) filter (where relkind='i'), count(*) filter (where relkind='v'), count(*) from pg_class c join pg_namespace n on n.oid=c.relnamespace where n.nspname='public'")[0]),
            "row_counts": {t: q(f"select count(*) from public.{t}")[0][0] for t in ROW_COUNT_TABLES},
            "tuple_writes_all_tables": [int(x) for x in q("select coalesce(sum(n_tup_ins),0), coalesce(sum(n_tup_upd),0), coalesce(sum(n_tup_del),0) from pg_stat_user_tables")[0]],
            "per_table_writes": {r[0]: [int(r[1]), int(r[2]), int(r[3])] for r in q("select relname, n_tup_ins, n_tup_upd, n_tup_del from pg_stat_user_tables order by 1")},
            "dataset_stamp": stamp,
        }
        cur.close()
    return snap


def file_snapshot() -> dict[str, dict[str, Any]]:
    files = {}
    for sub in PRESERVED_DIRS:
        base = OUT / sub
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file():
                st = f.stat()
                files[str(f.relative_to(OUT))] = {"sha256": sha256(f), "size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    return files


def cmd_snapshot(args) -> int:
    snap = {"taken_at": utc(), "database": db_snapshot(), "files": file_snapshot()}
    args.out.write_text(json.dumps(snap, indent=1, default=str), encoding="utf-8")
    d = snap["database"]
    print(f"snapshot: {len(snap['files'])} files hashed; read_only={d['transaction_read_only']}; rows={d['row_counts']}; objects={d['objects']}")
    return 0


# ======================================================================================================= checks

class Checks:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add(self, cid: str, area: str, title: str, result: str, evidence: Any, note: str = "") -> None:
        assert result in ("PASS", "FAIL", "NOT VERIFIED")
        self.items.append({"id": cid, "area": area, "check": title, "result": result, "evidence": evidence, "note": note})

    def ok(self, cid, area, title, condition: bool, evidence, note="") -> None:
        self.add(cid, area, title, "PASS" if condition else "FAIL", evidence, note)


def run_records(run: Path) -> list[dict[str, Any]]:
    return list(stream(run / "scored_candidates.jsonl"))


def tally(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"records": len(records),
            "bands": dict(sorted(collections.Counter(r["score_band"] for r in records).items())),
            "review_groups": dict(sorted(collections.Counter(r["review_group"] for r in records).items())),
            "candidates_with_a_hard_conflict": sum(bool(r["hard_conflicts"]) for r in records),
            "hard_conflicts_by_field": dict(sorted(collections.Counter(c["field"] for r in records for c in r["hard_conflicts"]).items())),
            "insufficient_evidence": sum(bool(r["insufficient_evidence"]) for r in records),
            "pending": sum(r["review"] == PENDING for r in records)}


def worksheet_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def cmd_checks(args) -> int:
    from dedup_pipeline.candidates.config import CandidateConfig
    from dedup_pipeline.scoring import ScoringConfig
    from dedup_pipeline.scoring.config import DECISIONS, RULE_VERSION
    from dedup_pipeline.scoring.review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V2

    c = Checks()
    start, end = json.loads(args.start.read_text(encoding="utf-8-sig")), json.loads(args.end.read_text(encoding="utf-8-sig"))
    prior = json.loads(args.prior_db.read_text(encoding="utf-8-sig")) if args.prior_db and args.prior_db.exists() else None
    baseline_files = json.loads(args.baseline_files.read_text(encoding="utf-8-sig"))["files"] if args.baseline_files and args.baseline_files.exists() else None

    # ------------------------------------------------------------------------------------- 1. final validation
    cand_summary = json.loads((CANDIDATES / "run_summary.json").read_text(encoding="utf-8-sig"))
    cands = list(stream(CANDIDATES / "candidates.jsonl"))
    ids = [r["candidate_id"] for r in cands]
    c.ok("V01", "final validation", f"candidates.jsonl holds exactly {EXPECTED_CANDIDATES} unique candidate pairs in canonical order, equal to its run summary",
         len(ids) == len(set(ids)) == EXPECTED_CANDIDATES == cand_summary["union"]["candidate_pairs"]
         and all(r["person_a_id"] < r["person_b_id"] and r["candidate_id"] == f"{r['person_a_id']}:{r['person_b_id']}" for r in cands)
         and ids == sorted(ids, key=lambda x: tuple(x.split(":"))),
         {"records": len(ids), "unique": len(set(ids)), "run_summary_union": cand_summary["union"]["candidate_pairs"], "status": cand_summary["status"]})

    v2 = run_records(PRESERVED_V2)
    v2_summary = json.loads((PRESERVED_V2 / "run_summary.json").read_text(encoding="utf-8-sig"))
    t2 = tally(v2)
    c.ok("V02", "final validation", "the preserved score-v2 run contains every candidate exactly once (same 24,155 ids as candidate generation)",
         len(v2) == EXPECTED_CANDIDATES and [r["candidate_id"] for r in v2] == ids and v2_summary["status"] == "completed" and v2_summary["rule_version"] == "score-v2",
         {"records": len(v2), "same_ids_same_order_as_candidates": [r["candidate_id"] for r in v2] == ids, "status": v2_summary["status"], "rule_version": v2_summary["rule_version"]})
    c.ok("V03", "final validation", "every score-v2 candidate is pending: no decision, reviewer, timestamp or note",
         t2["pending"] == EXPECTED_CANDIDATES and v2_summary["human_review"]["decisions_recorded"] == 0 and v2_summary["candidates"]["reviewed"] == 0,
         {"pending": t2["pending"], "decisions_recorded": v2_summary["human_review"]["decisions_recorded"], "reviewed": v2_summary["candidates"]["reviewed"]})
    s_bands = {k[0] if k.startswith("E") else k: v for k, v in v2_summary["candidates"]["score_bands_prioritisation_only"].items()}
    s_groups = v2_summary["review_groups"]["candidates_per_group"]
    s_hard = sum(v for k, v in v2_summary["candidates"]["conflicts"].items() if k.endswith(":hard"))
    c.ok("V04", "final validation", "score bands, review groups and hard-conflict counts recomputed from the 24,155 records equal the preserved run summary",
         t2["bands"] == {k: v for k, v in s_bands.items() if v} and t2["review_groups"] == dict(sorted(s_groups.items()))
         and sum(t2["hard_conflicts_by_field"].values()) == s_hard and t2["insufficient_evidence"] == v2_summary["candidates"]["insufficient_evidence"],
         {"recomputed": t2, "run_summary": {"bands": s_bands, "review_groups": s_groups, "hard_conflicts_by_field_total": s_hard,
                                            "insufficient_evidence": v2_summary["candidates"]["insufficient_evidence"]}})
    for label, run in (("score-v1", PRESERVED_V1), ("score-v3", PRESERVED_V3)):
        recs = run_records(run)
        c.ok("V05" if label == "score-v1" else "V06", "final validation", f"the preserved {label} run contains the same 24,155 candidates, all pending",
             [r["candidate_id"] for r in recs] == ids and all(r["review"] == PENDING for r in recs) and recs[0]["rule_version"] == label,
             {"records": len(recs), "pending": sum(r["review"] == PENDING for r in recs), "rule_version": recs[0]["rule_version"]})
        del recs

    cols, rows = worksheet_rows(PRESERVED_V2 / "review_worksheet.csv")
    c.ok("V07", "final validation", "the preserved score-v2 worksheet has the generated layout, 24,155 rows in review order, every row pending with empty decision columns",
         tuple(cols) == WORKSHEET_COLUMNS_V2 and len(rows) == EXPECTED_CANDIDATES and [int(r["review_order"]) for r in rows] == list(range(1, EXPECTED_CANDIDATES + 1))
         and {r["candidate_id"] for r in rows} == set(ids) and all(r["review_status"] == "pending" for r in rows)
         and all(all(not (r[k] or "").strip() for k in DECISION_COLUMNS) for r in rows),
         {"layout_known": tuple(cols) == WORKSHEET_COLUMNS_V2, "rows": len(rows), "rows_with_any_decision_field_filled": sum(any((r[k] or "").strip() for k in DECISION_COLUMNS) for r in rows),
          "review_status": dict(collections.Counter(r["review_status"] for r in rows))})

    # the final pipeline state, re-run now with the current code (default engine, default rule version)
    if args.fresh_run and (args.fresh_run / "run_summary.json").exists():
        fresh = run_records(args.fresh_run)
        fs = json.loads((args.fresh_run / "run_summary.json").read_text(encoding="utf-8-sig"))
        tf = tally(fresh)
        by_id = {r["candidate_id"]: r for r in v2}
        key = lambda r: (r["candidate_id"], r["match_score_percent"], r["score_band"], r["evidence_coverage_percent"], r["has_hard_conflict"], r["review_group"],
                         json.dumps(r["hard_conflicts"], sort_keys=True), json.dumps(r["conflicting_attributes"], sort_keys=True), json.dumps(r["fields"], sort_keys=True),
                         json.dumps(r["face"], sort_keys=True), json.dumps(r["review"], sort_keys=True))
        same_key = sum(key(r) == key(by_id[r["candidate_id"]]) for r in fresh if r["candidate_id"] in by_id)
        byte_equal = sum(r == by_id.get(r["candidate_id"]) for r in fresh)
        only_selection = [r["candidate_id"] for r in fresh if r != by_id.get(r["candidate_id"]) and {k: v for k, v in r.items() if k != "image_selection"} == {k: v for k, v in by_id.get(r["candidate_id"], {}).items() if k != "image_selection"}]
        other = [r["candidate_id"] for r in fresh if r != by_id.get(r["candidate_id"]) and r["candidate_id"] not in set(only_selection)]
        fcols, frows = worksheet_rows(args.fresh_run / "review_worksheet.csv")
        c.ok("V08", "final validation", "a fresh run of the current pipeline (default command) produces the same 24,155 candidates, all pending",
             [r["candidate_id"] for r in fresh] == ids and tf["pending"] == EXPECTED_CANDIDATES and fs["status"] == "completed" and fs["rule_version"] == "score-v2",
             {"records": len(fresh), "pending": tf["pending"], "status": fs["status"], "rule_version": fs["rule_version"], "face_engine": fs["face"]["engine"].get("engine"),
              "provider": fs["face"]["engine"].get("provider"), "retryable_errors": fs["face"]["retryable_errors"]})
        c.ok("V09", "final validation", "fresh run vs preserved baseline: identical scores, bands, coverage, hard conflicts, conflicts, field evidence, face evidence and review status for every candidate",
             same_key == EXPECTED_CANDIDATES and not other,
             {"identical_on_all_compared_keys": same_key, "byte_equal_records": byte_equal, "differ_only_in_image_selection_notes": len(only_selection), "other_differences": other[:20]},
             "expected difference: the image_selection notes of the face pairs record what the stored results knew when a run started; this run began with an empty cache "
             "(the preserved run began with a filled one). Documented since Phase 10.")
        c.ok("V10", "final validation", "fresh run vs preserved baseline: identical counts per score band, review group and hard-conflict field, and identical review order",
             tf["bands"] == t2["bands"] and tf["review_groups"] == t2["review_groups"] and tf["hard_conflicts_by_field"] == t2["hard_conflicts_by_field"]
             and tf["candidates_with_a_hard_conflict"] == t2["candidates_with_a_hard_conflict"] and [r["candidate_id"] for r in frows] == [r["candidate_id"] for r in rows],
             {"fresh": {k: tf[k] for k in ("bands", "review_groups", "candidates_with_a_hard_conflict", "hard_conflicts_by_field")}, "review_order_identical": [r["candidate_id"] for r in frows] == [r["candidate_id"] for r in rows]})
        cfg_new = json.loads((args.fresh_run / "scoring_config.json").read_text(encoding="utf-8-sig"))["config"]
        cfg_old = json.loads((PRESERVED_V2 / "scoring_config.json").read_text(encoding="utf-8-sig"))["config"]
        cfg_diff = {k: [cfg_old.get(k), cfg_new.get(k)] for k in sorted(set(cfg_old) | set(cfg_new)) if cfg_old.get(k) != cfg_new.get(k)}
        c.ok("V11", "final validation", "fresh run configuration equals the preserved one except the documented Phase 11 face_concurrency change (4 -> 8, not a scoring setting)",
             cfg_diff == {"face_concurrency": [4, 8]}, {"differences": cfg_diff})
        del fresh
    else:
        c.add("V08", "final validation", "fresh run of the current pipeline", "NOT VERIFIED", {"fresh_run": str(args.fresh_run)}, "no fresh run was available")

    if args.regenerated_candidates and (args.regenerated_candidates / "candidates.jsonl").exists():
        regen = list(stream(args.regenerated_candidates / "candidates.jsonl"))
        strip = lambda r: {k: v for k, v in r.items() if k != "images"}
        same = [r["candidate_id"] for r in regen] == ids and all(strip(a) == strip(b) for a, b in zip(regen, cands))
        c.ok("V12", "final validation", "candidate generation re-run now against the current database reproduces the same 24,155 pairs and rule evidence",
             same, {"records": len(regen), "same_ids_same_order": [r["candidate_id"] for r in regen] == ids,
                    "records_equal_apart_from_the_optional_images_annotation": sum(strip(a) == strip(b) for a, b in zip(regen, cands))},
             "the preserved candidate file was produced without the optional per-record image annotation; it is ignored in this comparison")
    else:
        c.add("V12", "final validation", "candidate generation re-run", "NOT VERIFIED", {}, "no regenerated candidate set was available")

    # ------------------------------------------------------------------------------------------- 2. database safety
    for label, snap in (("start", start["database"]), ("end", end["database"])):
        c.ok(f"D01-{label}", "database safety", f"the validation session was read-only on the approved database ({label} of Phase 12)",
             snap["transaction_read_only"] == "on" and snap["default_transaction_read_only"] == "on" and snap["preflight"].get("database") == "dopams_cctns_v2",
             {"transaction_read_only": snap["transaction_read_only"], "default_transaction_read_only": snap["default_transaction_read_only"], "database": snap["preflight"].get("database")})
    s, e = start["database"], end["database"]
    for key, title in (("schema_hash", "column schema"), ("indexes_hash", "index definitions"), ("objects", "object counts (tables, indexes, views)"), ("row_counts", "row counts of the six tables the pipeline reads or could affect")):
        c.ok(f"D02-{key}", "database safety", f"{title} unchanged during Phase 12", s[key] == e[key], {"start": s[key], "end": e[key]})
        if prior:
            c.ok(f"D03-{key}", "database safety", f"{title} equal to the last Phase 11 observation", s[key] == prior.get(key), {"phase11": prior.get(key), "phase12_start": s[key]})
    c.ok("D04", "database safety", "identity_links row count unchanged (no identity link created or removed)", s["row_counts"]["identity_links"] == e["row_counts"]["identity_links"]
         and (prior is None or prior["row_counts"]["identity_links"] == s["row_counts"]["identity_links"]),
         {"phase11": prior["row_counts"]["identity_links"] if prior else None, "start": s["row_counts"]["identity_links"], "end": e["row_counts"]["identity_links"]})
    c.ok("D05", "database safety", "dataset stamp of the database equals the stamp recorded when the candidates were generated (inputs not stale)",
         s["dataset_stamp"] == cand_summary["dataset_stamp"] == e["dataset_stamp"], {"candidates": cand_summary["dataset_stamp"], "start": s["dataset_stamp"], "end": e["dataset_stamp"]})
    moved = {t: [b - a for a, b in zip(s["per_table_writes"].get(t, [0, 0, 0]), v)] for t, v in e["per_table_writes"].items() if v != s["per_table_writes"].get(t)}
    c.add("D06", "database safety", "database-wide write counters (observation only)", "NOT VERIFIED",
          {"start": s["tuple_writes_all_tables"], "end": e["tuple_writes_all_tables"], "tables_whose_counters_moved": moved},
          "write counters are shared by every session on the server, so they cannot prove or disprove who wrote. Changes are recorded, not attributed. "
          "The pipeline's own evidence is the read-only session (D01) and its statements being SELECT/SHOW only (test suite).")

    # --------------------------------------------------------------------------------------- 3. output preservation
    sf, ef = start["files"], end["files"]
    changed = sorted(k for k in sf if ef.get(k, {}).get("sha256") != sf[k]["sha256"])
    added = sorted(set(ef) - set(sf))
    c.ok("P01", "output preservation", "no preserved output file changed, disappeared or appeared during Phase 12", not changed and not added,
         {"files": len(sf), "changed_or_missing": changed[:20], "added": added[:20]})
    if baseline_files:
        base_changed = sorted(k for k, h in baseline_files.items() if sf.get(k, {}).get("sha256") != h)
        c.ok("P02", "output preservation", f"the {len(baseline_files)} files hashed before Phase 8 (score-v1/v2 runs, candidates, image resolution, Phase 7 audit incl. the pilot sample) are byte-identical",
             not base_changed, {"files": len(baseline_files), "changed_or_missing": base_changed[:20]})
    else:
        c.add("P02", "output preservation", "files hashed before Phase 8", "NOT VERIFIED", {}, "the pre-Phase 8 hash list was not available")
    for cid, rel in (("P03", "audit\\phase7\\pilot_sample.csv"), ("P04", "audit\\phase8_1\\comparison_sample_phase8_1.csv")):
        entry = sf.get(rel)
        in_base = baseline_files is not None and rel in baseline_files
        if entry and in_base:
            c.ok(cid, "output preservation", f"{rel} unchanged since it was first hashed", baseline_files[rel] == entry["sha256"] == ef[rel]["sha256"], {"sha256": entry["sha256"], "mtime": entry["mtime"]})
        elif entry:
            c.add(cid, "output preservation", f"{rel} unchanged since it was written", "NOT VERIFIED", {"sha256_now": entry["sha256"], "mtime": entry["mtime"], "unchanged_during_phase12": ef[rel]["sha256"] == entry["sha256"]},
                  "no content hash was recorded when the sample was created; it was unchanged throughout Phase 12 and its modification time is the creation run, but earlier changes "
                  "cannot be excluded by hash. Its hash is recorded now as the baseline for later phases.")
    late = {}
    for rel, meta in sf.items():
        top = rel.split("\\")[0] if not rel.startswith("audit") else "\\".join(rel.split("\\")[:2])
        late[top] = max(late.get(top, ""), meta["mtime"])
    c.add("P05", "output preservation", "latest modification time per preserved directory (information)", "PASS" if all(v < start["taken_at"] for v in late.values()) else "FAIL", late,
          "every preserved directory was last modified before Phase 12 started")

    # ------------------------------------------------------------------------------- 4. constraints and configuration
    from dataclasses import asdict  # noqa: F401

    c.ok("C01", "constraints", "score-v2 is the default rule version", RULE_VERSION == "score-v2" and ScoringConfig(reference_date="2026-10-12").rule_version == "score-v2", {"RULE_VERSION": RULE_VERSION})
    cur = ScoringConfig(reference_date="2026-10-12").to_dict()
    old = json.loads((PRESERVED_V2 / "scoring_config.json").read_text(encoding="utf-8-sig"))["config"]
    scoring_keys = [k for k in old if k not in ("face_concurrency", "face_batch_size", "batch_size", "attribute_cache_max", "file_server_base_url", "file_proxy_origin", "allow_stale_candidates", "include_values", "image_links_per_person")]
    diff = {k: [old[k], cur.get(k)] for k in scoring_keys if old[k] != cur.get(k)}
    c.ok("C02", "constraints", "weights, thresholds, bands, conflict limits, age rules and review order of the current default equal the preserved score-v2 configuration", not diff,
         {"settings_compared": len(scoring_keys), "differences": diff})
    cc_now = CandidateConfig().to_dict() if hasattr(CandidateConfig(), "to_dict") else None
    cc_then = cand_summary.get("config")
    c.ok("C03", "constraints", "candidate-generation configuration of the current code equals the one that produced the 24,155 candidates", cc_now == cc_then,
         {"differences": {k: [cc_then.get(k), cc_now.get(k)] for k in sorted(set(cc_then or {}) | set(cc_now or {})) if (cc_then or {}).get(k) != (cc_now or {}).get(k)} if cc_now is not None else "CandidateConfig.to_dict unavailable"})
    cli = (ROOT / "dedup_pipeline" / "scoring" / "cli.py").read_text(encoding="utf-8")
    c.ok("C04", "constraints", "no API / HTTP face fallback in the scoring command", not any(s in cli for s in ("FrDopamsClient", "FrDopamsPairComparer", "FrConfig", "--face-engine", "FR_DOPAMS")),
         {"forbidden_names_found": [s for s in ("FrDopamsClient", "FrDopamsPairComparer", "FrConfig", "--face-engine", "FR_DOPAMS") if s in cli]})
    c.ok("C05", "constraints", "scores are labelled heuristics, not calibrated probabilities", v2_summary["score"]["calibrated_probability"] is False and "NOT" in v2_summary["score"]["label"].upper(),
         {"calibrated_probability": v2_summary["score"]["calibrated_probability"], "label": v2_summary["score"]["label"]})
    img_cli = (ROOT / "dedup_pipeline" / "image_resolution" / "cli.py").read_text(encoding="utf-8")
    img_runner = (ROOT / "dedup_pipeline" / "image_resolution" / "runner.py").read_text(encoding="utf-8")
    c.ok("C06", "performance decision", "image-resolution concurrency stays at 8 workers", 'add_argument("--workers", type=int, default=8)' in img_cli and "workers: int = 8" in img_runner, {"cli_default": 8, "runner_default": 8})
    from dedup_pipeline.face_recognition.local_config import LocalFaceConfig, VALIDATED_PROVIDERS

    c.ok("C07", "performance decision", "GPU stays disabled: default provider cpu, only the CPU provider is validated, others refused without explicit acceptance",
         LocalFaceConfig().provider == "cpu" and VALIDATED_PROVIDERS == ("CPUExecutionProvider",) and LocalFaceConfig().accept_unvalidated_provider is False,
         {"provider": LocalFaceConfig().provider, "validated_providers": list(VALIDATED_PROVIDERS)})
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "dedup_pipeline").rglob("*.py"))
    c.ok("C08", "performance decision", "no process-based scoring and no image-download cache in the pipeline code", "ProcessPoolExecutor" not in src and "multiprocessing" not in src
         and "lru_cache" not in (ROOT / "dedup_pipeline" / "face_recognition" / "image_source.py").read_text(encoding="utf-8"),
         {"ProcessPoolExecutor": "ProcessPoolExecutor" in src, "multiprocessing": "multiprocessing" in src})

    # --------------------------------------------------------------------------------------- 5. review readiness
    instr = (PRESERVED_V2 / "REVIEW_INSTRUCTIONS.md").read_text(encoding="utf-8")
    c.ok("R01", "review readiness", "reviewer instructions define all four decisions and require reviewer_id and an ISO date/time",
         all(d in instr for d in DECISIONS) and "reviewer_id" in instr and "reviewed_at" in instr and "ISO" in instr and tuple(DECISIONS) == ("confirm_same_person", "reject_match", "needs_further_review", "unable_to_verify"),
         {"decisions": list(DECISIONS)})
    if args.validator_output and args.validator_output.exists():
        v = json.loads(args.validator_output.read_text(encoding="utf-8-sig"))
        c.ok("R02", "review readiness", "the worksheet validator accepts an unmodified copy of the preserved worksheet: 24,155 pending, 0 decisions, 0 problems",
             v["rows"] == EXPECTED_CANDIDATES and v["pending"] == EXPECTED_CANDIDATES and v["decisions"] == 0 and v["problems"] == 0, v)
    else:
        c.add("R02", "review readiness", "worksheet validator on a copy", "NOT VERIFIED", {}, "validator output missing")
    pilot = list(csv.DictReader(open(OUT / "audit" / "phase7" / "pilot_sample.csv", encoding="utf-8-sig", newline="")))
    order = {r["candidate_id"]: r["review_order"] for r in rows}
    in_ws = [p for p in pilot if p["review_order_in_worksheet"]]
    never = [p for p in pilot if not p["review_order_in_worksheet"]]
    deferred_scores = {r["pair"] for r in stream(OUT / "audit" / "phase7" / "deferred_pairs_audit_scores.jsonl")} if (OUT / "audit" / "phase7" / "deferred_pairs_audit_scores.jsonl").exists() else set()
    c.ok("R03", "review readiness", "the Phase 7 pilot sample is consistent with the preserved worksheet (108 rows at the recorded review order; 12 never-evaluated pairs present in the Phase 7 audit-only scores)",
         len(in_ws) == 108 and all(order.get(p["candidate_id"]) == p["review_order_in_worksheet"] for p in in_ws) and len(never) == 12 and all(p["candidate_id"] in deferred_scores for p in never)
         and len({p["candidate_id"] for p in pilot}) == 120,
         {"pilot_rows": len(pilot), "in_worksheet_matching_order": sum(order.get(p["candidate_id"]) == p["review_order_in_worksheet"] for p in in_ws), "never_evaluated": len(never),
          "never_evaluated_with_audit_scores": sum(p["candidate_id"] in deferred_scores for p in never)})
    comp = list(csv.DictReader(open(OUT / "audit" / "phase8_1" / "comparison_sample_phase8_1.csv", encoding="utf-8-sig", newline="")))
    c.ok("R04", "review readiness", "the Phase 8.1 comparison sample (135) references existing candidates and does not overlap the Phase 7 pilot",
         len(comp) == 135 and all(r["candidate_id"] in order for r in comp) and not ({r["candidate_id"] for r in comp} & {p["candidate_id"] for p in pilot}),
         {"rows": len(comp), "all_in_worksheet": all(r["candidate_id"] in order for r in comp), "overlap_with_phase7_pilot": len({r["candidate_id"] for r in comp} & {p["candidate_id"] for p in pilot})})
    if args.pilot_dry_run and args.pilot_dry_run.exists():
        d = json.loads(args.pilot_dry_run.read_text(encoding="utf-8-sig"))
        c.ok("R05", "review readiness", "the pilot extraction (dry run, nothing written) finds every pilot candidate and produces blank decision columns", d["found_in_worksheet"] == 108 and d["found_in_audit_scores"] == 12 and d["missing"] == [], d)
    else:
        c.add("R05", "review readiness", "pilot extraction dry run", "NOT VERIFIED", {}, "dry-run output missing")
    c.add("R06", "review readiness", "authorised reviewers named and briefed; reviewer ID convention agreed", "NOT VERIFIED", {},
          "requires the data owner: no reviewer identity has been invented or recorded")

    # ------------------------------------------------------------------------------------------------- 6. tests
    if args.test_results:
        for path in args.test_results:
            text = path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, \d+ warnings?)?", text)
            failed = re.search(r"(\d+) failed", text)
            c.ok(f"T-{path.stem}", "tests", f"test suite: {path.stem}", bool(m) and not failed, {"summary": m.group(0) if m else None, "failed": failed.group(0) if failed else None})

    results = {"generated_at": utc(), "start_snapshot": str(args.start), "end_snapshot": str(args.end), "checks": c.items,
               "totals": dict(collections.Counter(i["result"] for i in c.items))}
    args.out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    for i in c.items:
        print(f"{i['result']:13s} {i['id']:22s} {i['check'][:110]}")
    print(results["totals"])
    return 0 if not any(i["result"] == "FAIL" for i in c.items) else 1


SENSITIVE_MARK = "_SENSITIVE"
DESCRIPTIONS = {
    "PHASE_12_VALIDATION_REPORT.md": "the Phase 12 report (no personal data)",
    "PILOT_REVIEWER_BRIEFING_DRAFT.md": "draft briefing for the authorised pilot reviewers (no personal data); needs the data owner's approval",
    "validation_results.json": "every check with PASS / FAIL / NOT VERIFIED and its evidence (ids and counts only)",
    "phase12_manifest.json": "this manifest",
    "snapshot_start.json": "database observations and content hashes of every preserved output file at the start of Phase 12",
    "snapshot_end.json": "the same at the end of Phase 12",
    "baseline_files_pre_phase8.json": "content hashes of 526 output files recorded before Phase 8 (copied from the Phase 8 working snapshot)",
    "prior_db_end_of_phase11.json": "database observations recorded at the end of Phase 11 (copied from the Phase 11 working snapshot)",
    "validator_on_copy.json": "result of the worksheet validator run on a temporary copy of the preserved worksheet",
    "pilot_dry_run.json": "pilot extraction dry run: candidate ids, strata and counts; no worksheet written",
    "cmd_candidates.log": "exact command and output of the read-only candidate regeneration",
    "cmd_scoring.log": "exact command and summary output of the default score-v2 run",
    "cmd_checks.log": "exact command and output of the check run",
    "tests_face_venv.log": "full test suite, .face-venv",
    "tests_venv.log": "full test suite, original .venv",
}


def cmd_manifest(args) -> int:
    base = args.dir
    entries, sensitive_dirs = [], {}
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(base)
        top = rel.parts[0]
        if SENSITIVE_MARK in top and len(rel.parts) > 1:
            d = sensitive_dirs.setdefault(top, {"files": 0, "bytes": 0})
            d["files"] += 1
            d["bytes"] += f.stat().st_size
            continue
        entries.append({"path": str(rel), "bytes": f.stat().st_size, "sha256": sha256(f), "sensitive": SENSITIVE_MARK in str(rel),
                        "description": DESCRIPTIONS.get(rel.name, "")})
    key_files = {}
    for top in sensitive_dirs:
        for f in sorted((base / top).rglob("*")):
            if f.is_file() and f.name in ("scored_candidates.jsonl", "review_worksheet.csv", "candidates.jsonl", "run_summary.json", "scoring_config.json"):
                key_files[str(f.relative_to(base))] = sha256(f)
    manifest = {"generated_at": utc(), "directory": str(base.relative_to(ROOT)), "files": entries,
                "sensitive_directories": {k: {**v, "content": "personal data (names, ages, phone fragments) for authorised verification only; local storage only"} for k, v in sensitive_dirs.items()},
                "sensitive_key_file_sha256": key_files,
                "preserved_outputs": {"hash_list": "snapshot_end.json -> files", "files": len(json.loads((base / "snapshot_end.json").read_text(encoding="utf-8-sig"))["files"])}}
    (base / "phase12_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {len(entries)} files, sensitive directories {list(sensitive_dirs)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--dir", type=Path, default=OUT / "audit" / "phase12")
    s = sub.add_parser("snapshot")
    s.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("checks")
    k.add_argument("--start", type=Path, required=True)
    k.add_argument("--end", type=Path, required=True)
    k.add_argument("--prior-db", type=Path)
    k.add_argument("--baseline-files", type=Path)
    k.add_argument("--fresh-run", type=Path)
    k.add_argument("--regenerated-candidates", type=Path)
    k.add_argument("--validator-output", type=Path)
    k.add_argument("--pilot-dry-run", type=Path)
    k.add_argument("--test-results", type=Path, nargs="*")
    k.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "manifest":
        return cmd_manifest(args)
    return cmd_snapshot(args) if args.command == "snapshot" else cmd_checks(args)


if __name__ == "__main__":
    sys.exit(main())
