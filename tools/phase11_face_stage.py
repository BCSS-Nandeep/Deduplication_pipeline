"""Phase 11: one timed run of the real face stage (downloads + pair logic + local engine) over stored cases, in a FRESH process.

The package under test is chosen with PHASE11_PKG_ROOT (the frozen pre-Phase 11 copy or the current tree), so baseline and optimised code run the
identical workload: same stored cases, empty face cache, same concurrency, same engine configuration, same machine. The produced evidence is compared with
the stored original (real service) case by case.

    $env:PHASE11_PKG_ROOT = "<root containing dedup_pipeline>"
    .face-venv\\Scripts\\python tools\\phase11_face_stage.py --cases 150 --concurrency 4 --out result.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(os.environ.get("PHASE11_PKG_ROOT") or ROOT)
sys.path.insert(0, str(PKG_ROOT))
try:
    import psycopg2  # noqa: F401
except ImportError:  # pragma: no cover
    stub = types.ModuleType("psycopg2")
    stub.Error = Exception
    sys.modules["psycopg2"] = stub

from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import LocalFacePairComparer  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.face import FaceCache, FaceTask, ImageOutcomeMemory, inputs_key, run_face_stage  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402
from dedup_pipeline.scoring.runner import CountingComparer, CountingFetcher  # noqa: E402

STORED = ROOT / "output" / "scoring" / "face_evidence_cache.jsonl"


def peak_ws_mb() -> float:
    """Peak working set of this process (Windows), in MB."""
    from ctypes import wintypes

    class C(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    kernel32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C), wintypes.DWORD]
    c = C()
    c.cb = ctypes.sizeof(c)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 1048576, 1) if ok else -1.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0, help="first N stored cases (0 = all 525)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    args = ap.parse_args()
    stored = [json.loads(line) for line in open(STORED, encoding="utf-8")]
    if args.cases:
        stored = stored[: args.cases]
    config = ScoringConfig(rule_version="score-v2", reference_date="2026-10-12", face_concurrency=args.concurrency)
    images = validate_images(ROOT / "output" / "image_resolution", None)
    tasks = [FaceTask(r["candidate_id"], r["evidence"]["person_a_id"], r["evidence"]["person_b_id"],
                      inputs_key(images.index, r["evidence"]["person_a_id"], r["evidence"]["person_b_id"], config)) for r in stored]
    comparer = LocalFacePairComparer()
    t0 = time.perf_counter()
    comparer.verify()
    init = time.perf_counter() - t0
    args.scratch.mkdir(parents=True, exist_ok=True)
    cache_file = args.scratch / "stage_cache.jsonl"
    cache_file.unlink(missing_ok=True)
    fetcher = CountingFetcher(HttpFetcher())
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), fetcher)
    counted = CountingComparer(comparer)
    cache = FaceCache(cache_file, {"phase11": True})
    cpu0 = time.process_time()
    t = time.perf_counter()
    results, stats = run_face_stage(tasks, images.index, loader, counted, cache, config, call_counter=lambda: counted.calls, memory=ImageOutcomeMemory())
    wall = time.perf_counter() - t
    cpu = time.process_time() - cpu0
    cache.close()
    cache_file.unlink(missing_ok=True)
    different = [r["candidate_id"] for r in stored if results[r["candidate_id"]].to_dict() != r["evidence"]]
    out = {"pkg_root": str(PKG_ROOT), "cases": len(stored), "concurrency": args.concurrency, "init_seconds": round(init, 2), "face_stage_wall_seconds": round(wall, 2),
           "process_cpu_seconds": round(cpu, 1), "comparer_calls": counted.calls, "image_downloads": fetcher.requests, "image_bytes": fetcher.bytes,
           "peak_working_set_mb": peak_ws_mb(), "evidence_identical_to_stored_original": len(stored) - len(different), "evidence_different": different[:10],
           "retryable_errors": stats.retryable_errors, "fingerprint": comparer.describe()["fingerprint"]}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
