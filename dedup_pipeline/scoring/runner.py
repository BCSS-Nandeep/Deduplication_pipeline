"""Phase 6 orchestration: validate inputs -> face evidence (cached, bounded) -> score -> reviewer outputs, all local.

Read-only against the database; never writes a decision, merges, or calls a gallery endpoint. A run is complete only
if ``run_summary.json`` says ``status: completed``; it is written last. A run with retryable failures is ``incomplete``
and its outputs are provisional.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..candidates.config import CandidateConfig
from ..candidates.memory import peak_rss_bytes
from ..candidates.runner import preflight
from ..candidates.store import write_json_atomic
from ..db import ReadOnlyConnection
from ..face_recognition.image_source import ManifestImageLoader
from ..face_recognition.pair_evidence import ComparisonOutcome, PairComparer, PairFaceEvidence, PairStatus, compare_pair
from ..image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL
from .attributes import AttributeStore
from .config import DECISIONS, RULE_VERSION, SCHEMA_VERSION, ScoringConfig
from .face import (
    FaceCache, FaceStageStats, FaceTask, ImageOutcomeMemory, KnownRejections, PairIndexView, SCORED_STATUSES, inputs_key, measured_detection_rates, needs_retry,
    run_face_stage,
)
from .provenance import PROVENANCE_FILE, PROVENANCE_SUMMARY_FILE, AgeProvenance, write_summary as write_provenance_summary
from .inputs import CandidateInput, ImageInput, InputError, read_jsonl, validate_candidates, validate_images
from .review import INSTRUCTIONS, INSTRUCTIONS_FILE, INSTRUCTIONS_V3_ADDENDUM, REPORT_DIR, WORKSHEET_FILE, ReportWriter, WorksheetWriter, write_report, write_worksheet  # noqa: F401
from .score import SCORE_LABEL, build_record, display_status, score_pair

log = logging.getLogger(__name__)

SCORED_FILE = "scored_candidates.jsonl"
DEFERRED_OUT_FILE = "deferred_unevaluated.jsonl"
SUMMARY_FILE = "run_summary.json"
STATE_FILE = "run_state.json"
CONFIG_FILE = "scoring_config.json"
LATEST_FILE = "latest.json"
FACE_CACHE_FILE = "face_evidence_cache.jsonl"
COVERAGE_BUCKETS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100.0001))


@dataclass
class FaceBackend:
    """How face comparisons are made. ``comparer`` is ``None`` when no service is configured (pairs are 'not_checked')."""

    comparer: Optional[PairComparer]
    loader: ManifestImageLoader
    verify: Optional[Callable[[], None]] = None  # raises if the right service is not reachable
    comparer_counter: Optional["CountingComparer"] = None
    fetch_counter: Optional["CountingFetcher"] = None
    reuse_known_image_outcomes: bool = False  # Phase 9, opt-in: skip service calls that stored answers already determine (see face.KnownRejections)
    # Phase 10: an in-process engine adds its identity to the face-cache signature (so results of another engine, model file, setting or
    # library are never reused) and describes itself for the run summary. Both stay None for the HTTP API, whose signature is unchanged.
    signature_fields: Optional[Callable[[], dict]] = None
    engine_info: Optional[Callable[[], dict]] = None


class CountingComparer:
    """Counts service calls and time without changing behaviour."""

    def __init__(self, inner: PairComparer):
        self._inner, self._lock = inner, threading.Lock()
        self.calls = 0
        self.seconds = 0.0

    def compare(self, image_a: bytes, image_b: bytes) -> ComparisonOutcome:
        started = time.perf_counter()
        try:
            return self._inner.compare(image_a, image_b)
        finally:
            with self._lock:
                self.calls += 1
                self.seconds += time.perf_counter() - started


class CountingFetcher:
    """Counts file-server downloads and bytes for the workload report; otherwise transparent."""

    def __init__(self, inner):
        self._inner, self._lock = inner, threading.Lock()
        self.requests = 0
        self.bytes = 0

    def fetch(self, url: str, max_bytes: int):
        result = self._inner.fetch(url, max_bytes)
        with self._lock:
            self.requests += 1
            self.bytes += len(result.body)
        return result


@dataclass(frozen=True)
class ScoringOptions:
    candidates_dir: Path
    image_dir: Path
    output_root: Path
    run_label: Optional[str] = None
    resume: bool = True


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def choose_run_dir(root: Path, label: Optional[str], resume: bool, signature: str) -> tuple[Path, bool]:
    """A new directory per run (previous runs are preserved), except an unfinished run with identical inputs is resumed."""
    latest = _read(root / LATEST_FILE)
    if label:
        target = root / "runs" / label
        state = _read(target / STATE_FILE)
        if state and state.get("status") == "completed":
            raise InputError(f"run directory {target.name!r} is already completed; choose another label (completed runs are never overwritten)")
        return target, bool(state)
    if resume and latest and latest.get("status") in ("running", "incomplete", "failed") and latest.get("signature") == signature:
        return root / "runs" / latest["run_dir"], True
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = stamp
    n = 1
    while (root / "runs" / name).exists():
        n += 1
        name = f"{stamp}-{n}"
    return root / "runs" / name, False


def _percentile_summary(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "min": round(ordered[0], 3), "median": round(statistics.median(ordered), 3), "max": round(ordered[-1], 3)}


def _iter_sorted(path: Path, index: list[tuple[tuple, int, int]]) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as fh:
        for _, offset, length in sorted(index, key=lambda e: e[0]):
            fh.seek(offset)
            yield json.loads(fh.read(length))


@dataclass
class Tally:
    scored: int = 0
    bands: Counter = field(default_factory=Counter)
    bands_without_hard_conflict: Counter = field(default_factory=Counter)
    by_source_and_band: dict = field(default_factory=dict)
    coverage: Counter = field(default_factory=Counter)
    face_status: Counter = field(default_factory=Counter)
    face_skip_detail: Counter = field(default_factory=Counter)
    face_threshold: Counter = field(default_factory=Counter)
    face_in_score: int = 0
    fallback_used: int = 0
    shared_image: int = 0
    insufficient: int = 0
    conflicts: Counter = field(default_factory=Counter)
    hard_conflict_in_top_bands: int = 0
    retry_needed: int = 0
    both_usable: int = 0
    face_checks_cosines_consistent: list = field(default_factory=list)
    face_checks_cosines_inconsistent: list = field(default_factory=list)
    images_unresolved: int = 0
    images_not_run: int = 0
    attribute_queries: int = 0
    attribute_rows: int = 0
    fir_rows: int = 0
    groups: Counter = field(default_factory=Counter)
    age_methods: Counter = field(default_factory=Counter)
    age_not_compared: Counter = field(default_factory=Counter)
    age_inconsistent: int = 0
    dob_status: Counter = field(default_factory=Counter)
    dob_invalid: Counter = field(default_factory=Counter)
    selection_pairs: int = 0
    selection_attempts: int = 0
    selection_demoted: int = 0
    selection_by_policy: Counter = field(default_factory=Counter)


def run_scoring(conn: ReadOnlyConnection, config: ScoringConfig, options: ScoringOptions, face: FaceBackend) -> dict[str, Any]:
    started_at, t0 = _utc(), time.perf_counter()
    tracemalloc.start()
    run_dir: Optional[Path] = None
    run_id = started_at
    try:
        # ---------------------------------------------------------------- inputs (fail closed)
        cand = validate_candidates(options.candidates_dir)
        images = validate_images(options.image_dir, cand.dataset_stamp.get("persons"))
        info, stamp = preflight(conn)
        limitations = list(images.limitations)
        if stamp != cand.dataset_stamp:
            if not config.allow_stale_candidates:
                raise InputError("the database changed since candidate generation (dataset stamp differs); regenerate candidates "
                                 "or pass allow_stale_candidates")
            limitations.append("database_changed_since_candidate_generation")
        candidate_config = CandidateConfig.from_dict(cand.summary["config"])
        if cand.deferred_blocks:
            limitations.append(f"candidate_generation_deferred_blocks:{cand.deferred_blocks}")

        # an in-process face engine identifies itself here, before anything is written: if it cannot load, the run stops cleanly
        engine_fields = face.signature_fields() if face.signature_fields is not None else {}
        signature = json.dumps({"candidates_run": cand.run_id, "images": images.summary.get("generated_at"),
                                "config": config.signature(), **({"face_engine": engine_fields} if engine_fields else {})}, sort_keys=True)
        run_dir, resumed = choose_run_dir(options.output_root, options.run_label, options.resume, signature)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        (run_dir / SUMMARY_FILE).unlink(missing_ok=True)  # the completion marker disappears first and is written last
        write_json_atomic(run_dir / STATE_FILE, {"status": "running", "run_id": run_id, "started_at": started_at, "resumed": resumed})
        write_json_atomic(options.output_root / LATEST_FILE, {"run_dir": run_dir.name, "status": "running", "signature": signature})
        write_json_atomic(run_dir / CONFIG_FILE, {"rule_version": config.rule_version, "schema_version": SCHEMA_VERSION, "reference_date": config.reference_date,
                                                  "config": config.to_dict(),
                                                  "candidate_run_id": cand.run_id, "image_summary_generated_at": images.summary.get("generated_at")})
        log.info("Run %s (%s). %d candidates; image coverage: %s", run_id, "resumed" if resumed else "new", cand.candidate_count,
                 "partial" if images.partial else "full")

        # ---------------------------------------------------------------------- face evidence
        face_signature = {**config.face_signature(), "contract": "compare-v1"}
        face_signature.update(engine_fields)
        memory = ImageOutcomeMemory.from_cache_file(options.output_root / FACE_CACHE_FILE, config.face_model,
                                                    engine_fingerprint=engine_fields.get("fingerprint"))  # frozen for the run
        cache = FaceCache(options.output_root / FACE_CACHE_FILE, face_signature)
        tasks = list(_face_tasks(cand, images, config))
        pending = [t for t in tasks if cache.get(t.candidate_id, t.inputs) is None]
        service_error: Optional[str] = None
        if pending and face.comparer is not None and face.verify is not None:
            try:
                face.verify()
            except Exception as exc:
                service_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.error("Face service check failed, face comparisons will be marked retryable: %s", service_error)
        if service_error:
            face_results, face_stats = {}, FaceStageStats(aborted=f"face service check failed ({service_error})")
            for t in tasks:
                cached = cache.get(t.candidate_id, t.inputs)
                if cached is not None:
                    face_results[t.candidate_id] = cached
                    face_stats.selections[t.candidate_id] = cache.selection(t.candidate_id)
                    face_stats.reused_from_cache += 1
        else:
            face_results, face_stats = run_face_stage(
                tasks, images.index, face.loader, face.comparer, cache, config,
                call_counter=(lambda: face.comparer_counter.calls) if face.comparer_counter else None, memory=memory,
                known_rejections=KnownRejections(memory) if face.reuse_known_image_outcomes else None)
        cache.close()
        for t in tasks:  # anything the stage never reached is a retryable error, not a result
            if t.candidate_id not in face_results:
                face_results[t.candidate_id] = PairFaceEvidence(
                    t.person_a, t.person_b, PairStatus.ERROR, reason=face_stats.aborted or "face stage did not reach this pair",
                    error_kind="unavailable")

        # ------------------------------------------------------------------------------ scoring
        provenance = AgeProvenance(run_dir) if config.rule_version == "score-v3" else None
        try:
            tally, index = _score_all(conn, cand, images, candidate_config, config, face_results, face_stats.selections, run_dir, provenance)
        except BaseException:
            if provenance is not None:
                provenance._fh.close()
            raise
        age_evidence = provenance.close(config) if provenance is not None else None
        if age_evidence is not None:
            age_evidence["fir_link_rows_read"] = tally.fir_rows
            write_provenance_summary(run_dir, age_evidence)
        deferred = _write_deferred(cand, run_dir)

        # ------------------------------------------------------------------- reviewer outputs
        retry = tally.retry_needed
        complete = retry == 0 and not face_stats.aborted
        headline = (f"{tally.scored} candidates - run {'COMPLETE' if complete else 'INCOMPLETE'}"
                    f"{'' if complete else f' ({retry} face comparisons need retry; results provisional)'}")
        banner = ("PROVISIONAL: face comparison is incomplete for some candidates. " if not complete else "") + \
                 "Match score is NOT a probability of identity. All candidates are pending human review."
        scored_path = run_dir / SCORED_FILE
        # one pass over the scored records feeds both reviewer outputs (the scored file used to be decoded once per output)
        worksheet = WorksheetWriter(run_dir / WORKSHEET_FILE, config.rule_version)
        try:
            report = ReportWriter(run_dir / REPORT_DIR, {"banner": banner, "headline": headline, "groups": dict(sorted(tally.groups.items())),
                                                          "rule_version": config.rule_version, "reference_date": config.reference_date})
            for rec in _iter_sorted(scored_path, index):
                worksheet.add(rec)
                report.add(rec)
        except BaseException:
            worksheet.abort()
            raise
        worksheet.close()
        report.close()
        (run_dir / INSTRUCTIONS_FILE).write_text(INSTRUCTIONS + (INSTRUCTIONS_V3_ADDENDUM if config.rule_version == "score-v3" else ""), encoding="utf-8")

        _, tm_peak = tracemalloc.get_traced_memory()
        summary = _summary(run_id, started_at, t0, info, stamp, cand, images, config, tally, face_stats, face_results, deferred, limitations,
                           complete, service_error, face, tm_peak, options, resumed, memory, tasks)
        if age_evidence is not None:
            summary["age_evidence_fir_anchored"] = age_evidence
            summary["outputs"] |= {"age_evidence_provenance": PROVENANCE_FILE, "age_evidence_summary": PROVENANCE_SUMMARY_FILE}
        write_json_atomic(run_dir / SUMMARY_FILE, summary)
        write_json_atomic(run_dir / STATE_FILE, {"status": summary["status"], "run_id": run_id, "started_at": started_at, "finished_at": _utc(), "resumed": resumed})
        write_json_atomic(options.output_root / LATEST_FILE, {"run_dir": run_dir.name, "status": summary["status"], "signature": signature})
        return summary
    except BaseException as exc:
        if run_dir is not None and run_dir.exists():
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            write_json_atomic(run_dir / STATE_FILE, {"status": status, "run_id": run_id, "started_at": started_at,
                                                     "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            latest = _read(options.output_root / LATEST_FILE)
            if latest and latest.get("run_dir") == run_dir.name:
                write_json_atomic(options.output_root / LATEST_FILE, {**latest, "status": status})
        raise
    finally:
        tracemalloc.stop()


# ---------------------------------------------------------------------------------- stages

def _face_tasks(cand: CandidateInput, images: ImageInput, config: ScoringConfig) -> Iterator[FaceTask]:
    for rec in read_jsonl(cand.candidates_path):
        a, b = rec["person_a_id"], rec["person_b_id"]
        if images.index.get(a).has_usable_image and images.index.get(b).has_usable_image:
            yield FaceTask(rec["candidate_id"], a, b, inputs_key(images.index, a, b, config))


def _score_all(conn, cand, images, candidate_config, config, face_results, selections, run_dir,
               provenance: Optional[AgeProvenance] = None) -> tuple[Tally, list[tuple[tuple, int, int]]]:
    store = AttributeStore(conn, config, candidate_config, on_person=provenance.record if provenance is not None else None)
    base_url = config.file_server_base_url or DEFAULT_FILE_SERVER_BASE_URL
    tally, index, offset = Tally(), [], 0
    batch: list[dict[str, Any]] = []

    def flush(batch: list[dict[str, Any]], out) -> int:
        nonlocal offset
        store.prefetch([p for c in batch for p in (c["person_a_id"], c["person_b_id"])])
        for c in batch:
            offset = _score_one(c, store, images, config, face_results, selections, tally, index, out, offset, base_url)
        return offset

    with open(run_dir / SCORED_FILE, "wb") as out:
        for rec in read_jsonl(cand.candidates_path):
            batch.append(rec)
            if len(batch) >= config.batch_size:
                flush(batch, out)
                batch = []
        if batch:
            flush(batch, out)
    tally.attribute_queries, tally.attribute_rows, tally.fir_rows = store.queries, store.rows_read, store.fir_rows_read
    return tally, index


def _score_one(c, store, images, config, face_results, selections, tally, index, out, offset, base_url) -> int:
    cid, pa, pb = c["candidate_id"], c["person_a_id"], c["person_b_id"]
    a, b = store.get(pa), store.get(pb)
    if a is None or b is None:
        raise InputError(f"candidate {cid} references a person that is not in the database")
    if (a.source_system, b.source_system) != (c["source_a"], c["source_b"]):
        raise InputError(f"candidate {cid}: source systems differ from the database (candidate file is stale or altered)")
    ia, ib = images.index.get(pa), images.index.get(pb)
    evidence = face_results.get(cid) or compare_pair(pa, pb, images.index, None, None)  # type: ignore[arg-type]
    shared = PairIndexView(images.index, pa, pb).shared_paths if ia.has_usable_image and ib.has_usable_image else set()
    limits = []
    for side, person, pid in (("a", ia, pa), ("b", ib, pb)):
        if person.person_status == "not_in_manifest":
            limits.append(f"image_resolution_not_run_for_person_{side}")
            tally.images_not_run += 1
        elif pid in images.persons_with_unresolved_media:
            limits.append(f"image_resolution_unresolved_media_person_{side}")
            tally.images_unresolved += 1
    outcome = score_pair(a, b, evidence, config)
    record = build_record(c, a, b, outcome, evidence, ia, ib, shared, config, base_url, limits, selections.get(cid))
    line = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
    out.write(line)
    score = outcome.match_score or 0.0
    group = int(record["review_group"][0])
    # score-v2: review group first (no hard conflict, hard conflict, insufficient, technical failure), then score; score-v1: score only
    index.append((((group if config.review_order != "score" else (1 if outcome.insufficient else 0)), -score, cid), offset, len(line) - 1))
    tally.groups[record["review_group"]] += 1
    age = record["age_dob_summary"]
    if age:
        if age["method"]:
            tally.age_methods[f"{age['method']}:{age['comparison_status']}"] += 1
        if not age["applicable"]:
            tally.age_not_compared[age["comparison_status"] or "unknown"] += 1
        tally.age_inconsistent += len(age["inconsistencies"])
        tally.dob_status[age["dob_comparison_status"].split(" (")[0]] += 1
        for side in ("record_a", "record_b"):
            if age[side]["dob_status"].startswith("invalid"):
                tally.dob_invalid[age[side]["dob_status"]] += 1
    sel = record["image_selection"]
    if sel.get("applicable"):
        tally.selection_pairs += 1
        tally.selection_attempts += sum(1 for side in ("a", "b") for n in sel[side] if n["attempted"])
        tally.selection_demoted += sum(1 for side in ("a", "b") for n in sel[side] if n["reason"].startswith("moved behind"))
        tally.selection_by_policy[sel["policy"]] += 1

    tally.scored += 1
    tally.bands[outcome.band] += 1
    tally.by_source_and_band.setdefault(c["source_pair"], Counter())[outcome.band] += 1
    tally.coverage[next(f"{lo}-{min(int(hi), 100)}" for lo, hi in COVERAGE_BUCKETS if lo <= outcome.coverage * 100 < hi)] += 1
    tally.face_status[display_status(evidence)] += 1
    if evidence.status in (PairStatus.MATCHED, PairStatus.NOT_MATCHED):
        tally.face_threshold["above_service_threshold" if evidence.status is PairStatus.MATCHED else "below_service_threshold"] += 1
    tally.insufficient += outcome.insufficient
    tally.shared_image += bool(shared)
    tally.retry_needed += needs_retry(evidence)
    if evidence.status is PairStatus.SKIPPED_NO_IMAGE:
        tally.face_skip_detail["no_usable_image_on_both" if not (ia.has_usable_image or ib.has_usable_image) else "no_usable_image_on_one_side"] += 1
    if ia.has_usable_image and ib.has_usable_image:
        tally.both_usable += 1
    tally.fallback_used += evidence.fallback_used
    tally.face_in_score += outcome.fields["face"].applicable
    hard = [n for n, f in outcome.fields.items() if f.conflict == "hard"]
    for n, f in outcome.fields.items():
        if f.conflict:
            tally.conflicts[f"{n}:{f.conflict}"] += 1
    if hard and outcome.band in ("A", "B"):
        tally.hard_conflict_in_top_bands += 1
    if not hard:
        tally.bands_without_hard_conflict[outcome.band] += 1
    if evidence.status in SCORED_STATUSES and evidence.face_similarity is not None and outcome.identity_score is not None:
        if outcome.identity_score >= 0.9 and not hard:
            tally.face_checks_cosines_consistent.append(evidence.face_similarity)
        elif outcome.identity_score < 0.5 or hard:
            tally.face_checks_cosines_inconsistent.append(evidence.face_similarity)
    return offset + len(line)


def _write_deferred(cand: CandidateInput, run_dir: Path) -> dict[str, Any]:
    totals = Counter()
    with open(run_dir / DEFERRED_OUT_FILE, "w", encoding="utf-8", newline="\n") as fh:
        for rec in read_jsonl(cand.deferred_path):
            out = {**rec, "evaluation_status": "never_evaluated", "interpretation":
                   "Not scored and NOT a non-match: these pairs were never evaluated because the block exceeded its size limit."}
            fh.write(json.dumps(out, ensure_ascii=True, separators=(",", ":")) + "\n")
            totals["blocks"] += 1
            totals["pairs_in_blocks"] += rec["would_be_pairs_after_policy"]
            for key in ("pairs_already_found_by_another_rule", "pairs_not_found_by_any_rule"):
                if rec.get(key) is None:
                    totals["impact_unknown_blocks"] += 1
                else:
                    totals[key] += rec[key]
    return {"blocks": totals["blocks"], "pairs_in_blocks": totals["pairs_in_blocks"],
            "pairs_also_scored_via_another_rule": totals["pairs_already_found_by_another_rule"],
            "pairs_never_evaluated_by_any_rule": totals["pairs_not_found_by_any_rule"],
            "blocks_with_unknown_impact": totals["impact_unknown_blocks"], "file": DEFERRED_OUT_FILE}


def _summary(run_id, started_at, t0, info, stamp, cand, images, config, tally, face_stats, face_results, deferred, limitations,
             complete, service_error, face, tm_peak, options, resumed, memory=None, tasks=()) -> dict[str, Any]:
    eligible = tally.both_usable
    retry = tally.retry_needed
    errors = Counter(e.error_kind or "operational" for e in face_results.values() if e.status is PairStatus.ERROR)
    if face_stats.aborted:
        limitations = [*limitations, f"face_stage_stopped:{face_stats.aborted}"]
    return {
        "status": "completed" if complete else "incomplete",
        "complete": complete,
        "review_ready": complete,
        "schema_version": SCHEMA_VERSION, "rule_version": config.rule_version,
        "run_id": run_id, "started_at": started_at, "finished_at": _utc(), "resumed": resumed,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "peak_python_heap_mb": round(tm_peak / 1048576, 1),
        "peak_process_memory_mb": round(peak_rss_bytes() / 1048576, 1) if peak_rss_bytes() else None,
        "database": info, "dataset_stamp": stamp,
        "inputs": {
            "candidates": {"directory": cand.directory.name, "run_id": cand.run_id, "candidate_pairs": cand.candidate_count,
                           "generator_version": cand.summary.get("generator_version")},
            "image_resolution": {"directory": images.directory.name, "persons": images.summary.get("persons"),
                                 "media_records": images.summary.get("media_records"), "usable_images": images.summary.get("usable_images"),
                                 "partial_run": images.partial, "persons_with_unresolved_media": len(images.persons_with_unresolved_media),
                                 "generated_at": images.summary.get("generated_at")},
        },
        "limitations": limitations,
        "reference_date": config.reference_date,
        "age_dob": None if config.rule_version == "score-v1" else {
            "reference_date": config.reference_date,
            "reference_date_policy": "one explicit date for every candidate in the run; recorded here and in scoring_config.json; never read from the clock",
            "age_tolerance_years": config.age_tolerance_years, "age_gap_factors": list(config.age_gap_factors),
            "age_conflict_gap_years": config.age_conflict_gap_years, "estimate_caps": {"age_vs_age": config.age_estimate_max_score, "dob_vs_age": config.dob_vs_age_max_score},
            "leap_day_birthday": config.leap_day_birthday,
            "comparisons_by_method_and_status": dict(sorted(tally.age_methods.items())),
            "not_compared": dict(sorted(tally.age_not_compared.items())),
            "dob_comparison_status": dict(sorted(tally.dob_status.items())), "invalid_dob_values_by_reason": dict(sorted(tally.dob_invalid.items())),
            "age_dob_inconsistencies_within_a_record": tally.age_inconsistent,
            **({"anchoring": "each recorded age is anchored to the date of the FIR it is linked to (accused_unified -> crimes_unified.fir_date); "
                             "an age with no usable FIR date is unavailable evidence, never anchored to the reference date",
                "birth_year_tolerance_years": config.birth_year_tolerance_years, "birth_year_conflict_gap_years": config.birth_year_conflict_gap_years,
                "observation_conflict_policy": config.observation_conflict_policy, "fir_timezone": config.fir_timezone}
               if config.rule_version == "score-v3" else {}),
        },
        "review_groups": {"order": config.review_order, "candidates_per_group": dict(sorted(tally.groups.items())),
                          "note": "Groups order the work only. Nothing is removed, and no group is a decision."},
        "image_selection": {
            "policy": config.image_ranking, "attempt_limit_per_person": config.face_max_images_per_person,
            "outcome_memory_images": len(memory) if memory is not None else 0, "outcome_memory_results": memory.counts() if memory is not None else {},
            "measured_face_usable_rate_by_category": measured_detection_rates(memory, images.index, {p for t in tasks for p in (t.person_a, t.person_b)}) if memory is not None else {},
            "pairs_with_image_selection": tally.selection_pairs, "images_attempted": tally.selection_attempts,
            "images_moved_behind_others_because_known_to_fail": tally.selection_demoted,
            "limitation": "No reliable portrait-vs-document signal exists in the image metadata (orientation, resolution, aspect and size did not beat the "
                          "majority class on held-out data), so the only ranking evidence is the Phase 2 category and the service's own earlier results.",
        },
        "score": {"formula": "match_score = sum(weight x field_score) / sum(applicable weights)",
                  "coverage_formula": "evidence_coverage = applicable weight / total configured weight",
                  "label": SCORE_LABEL, "weights": dict(config.weights), "face_in_score": config.face_in_score,
                  "face_transformation": {"type": "piecewise-linear", "cosine_at_or_below": config.face_different_anchor, "score_at_or_below": 0.0,
                                          "cosine_at_or_above": config.face_same_anchor, "score_at_or_above": 1.0,
                                          "basis": "FR-DOPAMS README calibration on 5 people (different <= 0.21, same 0.58-0.81)",
                                          "validated": "sanity-checked against identity-attribute pseudo-labels only; no ground-truth labels exist"},
                  "calibrated_probability": False},
        "candidates": {
            "generated": cand.candidate_count, "scored": tally.scored, "awaiting_human_review": tally.scored,
            "reviewed": 0, "insufficient_evidence": tally.insufficient,
            "score_bands_prioritisation_only": {b: tally.bands.get(b, 0) for b, _ in config.bands} | {"E_insufficient_evidence": tally.bands.get("E", 0)},
            "score_bands_without_a_hard_conflict": {b: tally.bands_without_hard_conflict.get(b, 0) for b in [*(n for n, _ in config.bands), "E"]},
            "score_bands_by_source_pair": {k: dict(sorted(v.items())) for k, v in sorted(tally.by_source_and_band.items())},
            "evidence_coverage_percent_distribution": {k: tally.coverage.get(k, 0) for k in (f"{lo}-{min(int(hi), 100)}" for lo, hi in COVERAGE_BUCKETS)},
            "conflicts": dict(sorted(tally.conflicts.items())),
            "top_band_candidates_with_a_hard_conflict": tally.hard_conflict_in_top_bands,
        },
        "face": {
            "engine": face.engine_info() if face.engine_info is not None else {"engine": "fr_dopams_api", "contract": "compare-v1"},
            "eligible_pairs_both_usable_images": eligible, "pairs_evaluated_this_run": face_stats.pairs_evaluated,
            "pairs_reused_from_cache": face_stats.reused_from_cache, "service_comparison_calls": face_stats.attempted,
            "service_calls_counted_by_client": face_stats.service_calls, "service_seconds": face_stats.seconds,
            "status_counts": dict(sorted(tally.face_status.items())),
            "compared_relative_to_service_threshold_not_a_decision": dict(sorted(tally.face_threshold.items())),
            "skipped_no_image_detail": dict(sorted(tally.face_skip_detail.items())),
            **({"reuse_known_image_outcomes": {"enabled": True, "attempts_answered_from_stored_results": face_stats.attempts_reused_from_stored_answers,
                                               "note": "service calls and image downloads skipped because stored answers fully determined the rejection; evidence is identical to making the attempt"}}
               if face_stats.attempts_reused_from_stored_answers is not None else {}),
            "compared_and_in_match_score": tally.face_in_score, "fallback_used": tally.fallback_used,
            "same_image_file_on_both_records": tally.shared_image, "retryable_errors": retry,
            "error_kinds": dict(sorted(errors.items())), "service_check_error": service_error, "stage_aborted": face_stats.aborted,
            "transformation_check_pseudo_labels": {
                "identity_fields_agree_no_hard_conflict": _percentile_summary(tally.face_checks_cosines_consistent),
                "identity_fields_disagree_or_hard_conflict": _percentile_summary(tally.face_checks_cosines_inconsistent),
                "note": "Raw cosine grouped by the non-face evidence. This is a sanity check, not validation against true labels.",
            },
        },
        "deferred": {**deferred, "interpretation": "never evaluated; not non-matches"},
        "human_review": {"status": "pending", "decisions_recorded": 0, "allowed_decisions": list(DECISIONS),
                         "worksheet": WORKSHEET_FILE, "report": f"{REPORT_DIR}/index.html", "instructions": INSTRUCTIONS_FILE},
        "outputs": {"scored_candidates": SCORED_FILE, "worksheet": WORKSHEET_FILE, "report": REPORT_DIR,
                    "deferred_unevaluated": DEFERRED_OUT_FILE, "config": CONFIG_FILE},
        "workload": {
            "face_service_requests": face.comparer_counter.calls if face.comparer_counter else None,
            "face_service_seconds_total_across_threads": round(face.comparer_counter.seconds, 1) if face.comparer_counter else None,
            "face_service_concurrency": config.face_concurrency,
            "image_downloads": face.fetch_counter.requests if face.fetch_counter else None,
            "image_bytes_downloaded": face.fetch_counter.bytes if face.fetch_counter else None,
            "attribute_queries": tally.attribute_queries, "person_attribute_rows_read": tally.attribute_rows,
        },
        "database_writes": 0,
        "notes": ["Nothing is merged, confirmed or rejected by software.", "No data is written to any database."],
    }
