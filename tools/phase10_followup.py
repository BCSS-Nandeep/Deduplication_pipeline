"""Phase 10 follow-up measurements the main benchmark cannot give:

A. Why is the API slower per pair than the local engine although HTTP costs ~1.5 ms? The service also loads the 3-D landmark model
   (landmark_3d_68) that a comparison never uses. Measure the local engine with and without it (same images, same process, alternating).
B. Memory of each side measured on its own: a fresh local process doing only comparisons, and the real service worker process(es)
   (the launcher stub is not the service).

    .fr-venv\\Scripts\\python tools\\phase10_followup.py --out output\\audit\\phase10\\followup
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    import psycopg2  # noqa: F401
except ImportError:
    stub = types.ModuleType("psycopg2")
    stub.Error = Exception
    sys.modules["psycopg2"] = stub

import phase10_benchmark as pb  # noqa: E402  (helpers: service start/stop, memory, sampling)
from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import LocalFacePairComparer  # noqa: E402
from dedup_pipeline.face_recognition.local_config import LocalFaceConfig  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402

LOCAL_MEMORY_CHILD = r"""
import sys, json, types, pickle
sys.path.insert(0, r"%(root)s"); sys.path.insert(0, r"%(root)s\tools")
try:
    import psycopg2
except ImportError:
    m = types.ModuleType("psycopg2"); m.Error = Exception; sys.modules["psycopg2"] = m
from dedup_pipeline.face_recognition import LocalFacePairComparer, LocalFaceConfig, FrPairRejected
import phase10_benchmark as pb, os
pairs = pickle.load(open(r"%(pairs)s", "rb"))
c = LocalFacePairComparer(LocalFaceConfig(provider="cpu")); c.verify()
before = pb.peak_working_set_mb(os.getpid())
for a, b in pairs:
    try: c.compare(a, b)
    except FrPairRejected: pass
print(json.dumps({"peak_mb_after_model_load_only": before, "peak_mb_after_comparisons": pb.peak_working_set_mb(os.getpid())}))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stored = [json.loads(line) for line in open(pb.STORED, encoding="utf-8")]
    images = validate_images(ROOT / "output" / "image_resolution", None)
    records = {r["media_id"]: r for rec in stored for p in (rec["evidence"]["person_a_id"], rec["evidence"]["person_b_id"]) for r in images.index.get(p).usable_images}
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), HttpFetcher())
    attempts = [(a["media_id_a"], a["media_id_b"]) for rec in stored for a in (rec["evidence"].get("attempts") or []) if a.get("outcome") in ("compared", "rejected")]
    sample = random.Random(args.seed).sample(attempts, args.pairs)
    cache = {m: loader.load(records[m]) for pair in sample for m in pair}
    pairs = [(cache[a], cache[b]) for a, b in sample]
    result: dict = {"pairs": len(pairs), "rounds": args.rounds}

    # ---- A: with and without the unused 3-D landmark model
    plain = LocalFacePairComparer(LocalFaceConfig(provider="cpu"))
    pose = LocalFacePairComparer(LocalFaceConfig(provider="cpu", load_pose_model=True))
    plain.verify(); pose.verify()
    for c in (plain, pose):
        for p in pairs[:4]:
            pb.outcome_key(c, *p)
    lat = {"without_pose_model": [], "with_pose_model": []}
    identical = 0
    for _ in range(args.rounds):
        for name, c in (("without_pose_model", plain), ("with_pose_model", pose)):
            l, _ = pb.timed_calls(c, pairs, 1)
            lat[name].append(statistics.median(l))
    identical = sum(pb.outcome_key(plain, *p) == pb.outcome_key(pose, *p) for p in pairs)
    result["local_latency_ms_median"] = {k: round(1000 * statistics.median(v), 1) for k, v in lat.items()}
    result["local_latency_ms_per_round"] = {k: [round(1000 * x, 1) for x in v] for k, v in lat.items()}
    result["results_identical_with_and_without_pose_model"] = {"pairs": len(pairs), "identical": identical}
    result["pose_model_extra_files"] = {"1k3d68.onnx_mb": round((Path("~/.insightface/models/buffalo_l/1k3d68.onnx").expanduser().stat().st_size) / 1048576, 1)}

    # ---- B: memory, each side alone
    import pickle

    pkl = args.out / "pairs.pkl"
    pkl.write_bytes(pickle.dumps(pairs))
    done = subprocess.run([sys.executable, "-c", LOCAL_MEMORY_CHILD % {"root": str(ROOT), "pairs": str(pkl)}], capture_output=True, text=True, cwd=str(ROOT))
    result["local_process_alone_mb"] = json.loads(done.stdout.strip().splitlines()[-1])
    pkl.unlink()
    proc, _ = pb.start_service()
    api = pb.api_comparer()
    for p in pairs:
        pb.outcome_key(api, *p)
    children = subprocess.run(["powershell", "-NoProfile", "-Command",
                               f"Get-CimInstance Win32_Process | Where-Object {{ $_.ParentProcessId -eq {proc.pid} -or $_.ProcessId -eq {proc.pid} }} | ForEach-Object {{ $_.ProcessId }}"],
                              capture_output=True, text=True).stdout.split()
    result["api_service_processes_peak_mb"] = {pid: pb.peak_working_set_mb(int(pid)) for pid in children}
    result["api_service_total_peak_mb"] = round(sum(v for v in result["api_service_processes_peak_mb"].values() if v == v), 1)
    pb.stop(proc)
    for pid in children:  # the launcher stub does not always take its child with it
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    (args.out / "followup_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
