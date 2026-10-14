"""Phase 9 profiling harness: runs one scoring run into a SCRATCH output root (a copy of the face cache; nothing existing is touched)
under cProfile and prints where the time goes. Read-only against the database.

python tools/phase9_profile.py --rule-version score-v2 --scratch <dir> [--profile]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402

PKG_ROOT = Path(os.environ.get("PHASE9_PKG_ROOT") or ROOT)  # which copy of dedup_pipeline to run (the original baseline or the optimised one)
sys.path.insert(0, str(PKG_ROOT))

from dedup_pipeline.db import connect_read_only, load_connection_settings  # noqa: E402
from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig, ScoringOptions, run_scoring  # noqa: E402
from dedup_pipeline.scoring.runner import CountingFetcher, FaceBackend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-version", default="score-v2")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--label", default="perf")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--top", type=int, default=35)
    args = ap.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    cache = args.scratch / "face_evidence_cache.jsonl"
    if not cache.exists():
        shutil.copy(ROOT / "output" / "scoring" / "face_evidence_cache.jsonl", cache)
    config = ScoringConfig(rule_version=args.rule_version, reference_date="2026-10-12") if args.rule_version != "score-v1" else ScoringConfig(rule_version="score-v1")
    fetcher = CountingFetcher(HttpFetcher())
    backend = FaceBackend(comparer=None, loader=ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), fetcher), fetch_counter=fetcher)
    options = ScoringOptions(ROOT / "output" / "candidates", ROOT / "output" / "image_resolution", args.scratch, args.label, resume=False)
    prof = cProfile.Profile() if args.profile else None
    stages: dict[str, float] = {}

    def timed(module, name, label):
        original = getattr(module, name)

        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                stages[label] = stages.get(label, 0.0) + time.perf_counter() - t
        setattr(module, name, wrapper)

    from dedup_pipeline.scoring import runner as runner_mod  # stage timers (wall clock, no behaviour change)
    from dedup_pipeline.scoring import similarity as sim
    for fn, label in (("validate_candidates", "input validation: candidates"), ("validate_images", "input validation: image manifest"), ("preflight", "database preflight"),
                      ("run_face_stage", "face stage (cache lookups only here)"), ("_score_all", "score all candidates (attributes + scoring + serialise)"),
                      ("_write_deferred", "deferred file"), ("write_worksheet", "worksheet csv"), ("write_report", "html report")):
        timed(runner_mod, fn, label)
    import gc

    if os.environ.get("PHASE9_NOGC"):
        gc.disable()  # experiment only: measures what the cyclic garbage collector costs; not enabled anywhere in the pipeline
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    with connect_read_only(load_connection_settings(None), "phase9_profile") as conn:
        if prof:
            prof.enable()
        summary = run_scoring(conn, config, options, backend)
        if prof:
            prof.disable()
    wall = time.perf_counter() - t0
    print(json.dumps({"rule_version": args.rule_version, "wall_seconds": round(wall, 2), "cpu_seconds": round(time.process_time() - cpu0, 2), "elapsed_seconds_in_summary": summary["elapsed_seconds"],
                      "peak_python_heap_mb": summary["peak_python_heap_mb"], "peak_process_memory_mb": summary["peak_process_memory_mb"],
                      "status": summary["status"], "scored": summary["candidates"]["scored"], "queries": summary["workload"]["attribute_queries"],
                      "stage_seconds": {k: round(v, 2) for k, v in stages.items()},
                      "similarity_cache": {n: dict(getattr(f, "cache_info")()._asdict()) for n, f in
                                           (("ratio", sim._ratio), ("token", sim.token_similarity), ("name", sim.name_similarity)) if hasattr(f, "cache_info")}}))
    if prof:
        for key in ("cumulative", "tottime"):
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).strip_dirs().sort_stats(key).print_stats(args.top)
            print(f"\n=== top {args.top} by {key} ===\n" + buf.getvalue()[:7000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
