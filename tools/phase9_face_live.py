"""Phase 9 live validation of the opt-in face reuse against ground truth, using the real FR-DOPAMS service (stateless launcher).

Takes N pairs from the stored face evidence (the evidence of the original Phase 6 run is the ground truth), re-evaluates them with a
FRESH face cache whose stored answers EXCLUDE those pairs, once without and once with ``KnownRejections``, and reports calls, seconds and
whether the produced evidence equals the stored one. Nothing existing is written: the scratch cache lives under output/perf.

python tools/phase9_face_live.py --pairs 80 --seed 9
(needs FR_DOPAMS_BASE_URL and FR_DOPAMS_API_KEY for the running service)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dedup_pipeline.face_recognition.client import FrDopamsClient  # noqa: E402
from dedup_pipeline.face_recognition.config import FrConfig  # noqa: E402
from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.pair_evidence import FrDopamsPairComparer  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.face import FaceCache, FaceTask, ImageOutcomeMemory, KnownRejections, inputs_key, run_face_stage  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402
from dedup_pipeline.scoring.runner import CountingComparer, CountingFetcher  # noqa: E402

SCRATCH = ROOT / "output" / "perf" / "face_live"


def evidence_key(evidence_dict: dict) -> str:
    """Everything except fields that legitimately vary: nothing is dropped - the whole stored record must be reproduced."""
    return json.dumps(evidence_dict, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--reuse-first", action="store_true", help="run the reuse variant first (to separate an order effect from a real difference)")
    args = ap.parse_args()
    stored_path = ROOT / "output" / "scoring" / "face_evidence_cache.jsonl"
    stored = [json.loads(line) for line in open(stored_path, encoding="utf-8")]
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(stored)), args.pairs))
    chosen_ids = {stored[i]["candidate_id"] for i in chosen}
    config = ScoringConfig(rule_version="score-v2", reference_date="2026-10-12")
    images = validate_images(ROOT / "output" / "image_resolution", None)
    tasks = []
    for i in chosen:
        a, b = stored[i]["evidence"]["person_a_id"], stored[i]["evidence"]["person_b_id"]
        tasks.append(FaceTask(stored[i]["candidate_id"], a, b, inputs_key(images.index, a, b, config)))
    result = {"pairs": len(tasks), "seed": args.seed}
    for label, reuse in ((("with_reuse", True), ("without_reuse", False)) if args.reuse_first else (("without_reuse", False), ("with_reuse", True))):
        scratch = SCRATCH / label
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        # the stored answers WITHOUT the pairs being re-evaluated: everything the run may legitimately know in advance
        with open(scratch / "known_answers.jsonl", "w", encoding="utf-8") as fh:
            for rec in stored:
                if rec["candidate_id"] not in chosen_ids:
                    fh.write(json.dumps(rec) + "\n")
        memory = ImageOutcomeMemory.from_cache_file(scratch / "known_answers.jsonl", config.face_model)
        client = FrDopamsClient(FrConfig.from_env())
        client.verify_service()
        comparer = CountingComparer(FrDopamsPairComparer(client))
        fetcher = CountingFetcher(HttpFetcher())
        loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), fetcher)
        cache = FaceCache(scratch / "fresh_cache.jsonl", {**config.face_signature(), "contract": "compare-v1"})
        known = KnownRejections(memory) if reuse else None
        t0 = time.perf_counter()
        results, stats = run_face_stage(tasks, images.index, loader, comparer, cache, config, call_counter=lambda: comparer.calls, memory=memory, known_rejections=known)
        seconds = time.perf_counter() - t0
        cache.close()
        produced = {cid: evidence_key(ev.to_dict()) for cid, ev in results.items()}
        truth = {rec["candidate_id"]: evidence_key(rec["evidence"]) for rec in stored if rec["candidate_id"] in chosen_ids}
        different = sorted(cid for cid in truth if produced.get(cid) != truth[cid])
        result[label] = {"wall_seconds": round(seconds, 2), "service_calls": comparer.calls, "image_downloads": fetcher.requests, "image_bytes": fetcher.bytes,
                         "pairs_evidence_identical_to_the_stored_original": len(truth) - len(different), "pairs_different": different,
                         "attempts_answered_from_stored_results": None if known is None else known.reused_attempts}
        print(label, json.dumps(result[label]), flush=True)
    if result["without_reuse"]["wall_seconds"]:
        result["speedup_wall"] = round(result["without_reuse"]["wall_seconds"] / max(result["with_reuse"]["wall_seconds"], 0.01), 2)
        result["service_calls_saved"] = result["without_reuse"]["service_calls"] - result["with_reuse"]["service_calls"]
    (ROOT / "output" / "perf" / "face_live_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("speedup_wall", "service_calls_saved")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
