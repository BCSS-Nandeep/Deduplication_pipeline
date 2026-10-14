"""Phase 10 equivalence replay: the in-process comparer against the 525 stored results of the real FR-DOPAMS service.

For every stored face case the FULL pair logic (image selection, fallback to the next image, side-specific rejections) is run again with the
local comparer, under the original conditions (empty image-outcome memory, same image order, same concurrency), and the produced
``PairFaceEvidence`` is compared with the stored one field by field. Status, rejection kinds/texts, failed sides, attempts, images used and
the service's same-person decision (status matched / not_matched) must agree EXACTLY; the raw cosine may differ by at most ``--tolerance``.

Run it in the environment that has the face packages (the FR-DOPAMS environment `.fr-venv` works; no psycopg2 is needed, the replay
never touches the database - a stub module is registered only to let the scoring package import).

    .fr-venv\\Scripts\\python tools\\phase10_replay.py --out output\\audit\\phase10\\replay_default
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # the scoring package imports the (read-only) database module, which imports the driver; the replay never connects
    import psycopg2  # noqa: F401
except ImportError:  # pragma: no cover
    stub = types.ModuleType("psycopg2")
    stub.Error = Exception
    stub.connect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the replay does not use a database"))
    sys.modules["psycopg2"] = stub

from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import LocalFacePairComparer  # noqa: E402
from dedup_pipeline.face_recognition.local_config import LocalFaceConfig  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.face import FaceCache, FaceTask, ImageOutcomeMemory, inputs_key, run_face_stage  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402
from dedup_pipeline.scoring.runner import CountingComparer, CountingFetcher  # noqa: E402

STORED = ROOT / "output" / "scoring" / "face_evidence_cache.jsonl"
EXACT_FIELDS = ("status", "reason", "failed_side", "side_failures", "error_kind", "media_id_a", "media_id_b", "fallback_used", "attempts",
                "person_status_a", "person_status_b", "unusable_reasons_a", "unusable_reasons_b", "similarity_metric", "decision_threshold", "model",
                "person_a_id", "person_b_id")


def compare_evidence(stored: dict, produced: dict, tolerance: float) -> dict:
    """Differences between two evidence dicts: exact fields, and the raw cosine against the tolerance."""
    diffs = {}
    for key in EXACT_FIELDS:
        if stored.get(key) != produced.get(key):
            diffs[key] = {"stored": stored.get(key), "local": produced.get(key)}
    a, b = stored.get("face_similarity"), produced.get("face_similarity")
    if (a is None) != (b is None):
        diffs["face_similarity"] = {"stored": a, "local": b}
    elif a is not None and abs(a - b) > tolerance:
        diffs["face_similarity"] = {"stored": a, "local": b, "abs_diff": abs(a - b)}
    unknown = (set(stored) | set(produced)) - set(EXACT_FIELDS) - {"face_similarity"}
    for key in sorted(unknown):
        if stored.get(key) != produced.get(key):
            diffs[key] = {"stored": stored.get(key), "local": produced.get(key)}
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="replay only the first N stored cases (smoke runs)")
    ap.add_argument("--per-status", type=int, help="replay at most N stored cases of each outcome status (a small sample covering every status)")
    ap.add_argument("--tolerance", type=float, default=1e-6, help="largest accepted absolute difference of the raw cosine similarity")
    ap.add_argument("--config", type=Path, help="JSON overriding LocalFaceConfig fields")
    ap.add_argument("--provider", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--stored", type=Path, default=STORED)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stored = [json.loads(line) for line in open(args.stored, encoding="utf-8")]
    if args.per_status:  # a small, deterministic sample that still covers every stored outcome type
        taken: collections.Counter = collections.Counter()
        picked = []
        for rec in stored:
            if taken[rec["evidence"]["status"]] < args.per_status:
                taken[rec["evidence"]["status"]] += 1
                picked.append(rec)
        stored = picked
    if args.limit:
        stored = stored[: args.limit]
    config = ScoringConfig(rule_version="score-v2", reference_date="2026-10-12", face_concurrency=args.concurrency)
    images = validate_images(ROOT / "output" / "image_resolution", None)
    local_config = LocalFaceConfig.from_file(args.config) if args.config else LocalFaceConfig()
    if args.provider:
        local_config = LocalFaceConfig.from_dict({**local_config.to_dict(), "provider": args.provider})
    comparer = LocalFacePairComparer(local_config)
    t = time.perf_counter()
    comparer.verify()
    init_seconds = time.perf_counter() - t
    description = comparer.describe()
    tasks = []
    for rec in stored:
        a, b = rec["evidence"]["person_a_id"], rec["evidence"]["person_b_id"]
        tasks.append(FaceTask(rec["candidate_id"], a, b, inputs_key(images.index, a, b, config)))
    # the stored results were made with these inputs; a changed image set would make the comparison meaningless, so say so
    changed_inputs = [rec["candidate_id"] for rec, task in zip(stored, tasks) if rec["inputs"] != task.inputs]
    scratch = args.out / "scratch_cache.jsonl"
    scratch.unlink(missing_ok=True)
    cache = FaceCache(scratch, {**config.face_signature(), "contract": "compare-v1", "engine": description["engine"], "fingerprint": description["fingerprint"]})
    fetcher = CountingFetcher(HttpFetcher())
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), fetcher)
    counted = CountingComparer(comparer)
    t = time.perf_counter()
    results, stats = run_face_stage(tasks, images.index, loader, counted, cache, config, call_counter=lambda: counted.calls, memory=ImageOutcomeMemory())
    seconds = time.perf_counter() - t
    cache.close()

    by_status = collections.Counter()
    exact = within_tolerance = 0
    max_diff, sims_equal, sims_total = 0.0, 0, 0
    mismatches = []
    for rec in stored:
        cid = rec["candidate_id"]
        produced = results[cid].to_dict()
        diffs = compare_evidence(rec["evidence"], produced, args.tolerance)
        by_status[rec["evidence"]["status"]] += 1
        a, b = rec["evidence"].get("face_similarity"), produced.get("face_similarity")
        if a is not None and b is not None:
            sims_total += 1
            sims_equal += a == b
            max_diff = max(max_diff, abs(a - b))
        if not diffs:
            exact += 1
            within_tolerance += 1
        else:
            mismatches.append({"candidate_id": cid, "stored_status": rec["evidence"]["status"], "local_status": produced["status"], "differences": diffs})
    result = {
        "cases_replayed": len(stored), "cases_in_which_the_inputs_key_differs_from_the_stored_one": len(changed_inputs),
        "identical_evidence_within_tolerance": len(stored) - len(mismatches), "mismatching_cases": len(mismatches),
        "exactly_equal_cosine": sims_equal, "cases_with_a_cosine": sims_total, "max_abs_cosine_difference": max_diff, "tolerance": args.tolerance,
        "stored_status_counts": dict(sorted(by_status.items())),
        "status_agreement": sum(r["evidence"]["status"] == results[r["candidate_id"]].status.value for r in stored),
        "engine": description, "model_init_seconds": round(init_seconds, 2), "face_stage_seconds": round(seconds, 1), "service_equivalent_calls": counted.calls,
        "image_downloads": fetcher.requests, "image_bytes": fetcher.bytes, "concurrency": args.concurrency,
        "mismatches": mismatches,
    }
    (args.out / "replay_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("mismatches", "engine")}, indent=1))
    print("engine:", description["engine"], description["fingerprint"][:16], description["provider"])
    for m in mismatches[:10]:
        print("MISMATCH", m["candidate_id"], m["stored_status"], "->", m["local_status"], json.dumps(m["differences"])[:300])
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
