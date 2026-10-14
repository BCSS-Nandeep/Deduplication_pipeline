"""Phase 10 benchmark: the in-process comparer against the FR-DOPAMS HTTP API, on identical inputs and the same machine.

Everything is measured with alternating runs and medians (the host is shared with other heavy processes, so single runs are not trusted).
Image bytes are loaded once, so inference/HTTP time is separated from download time; the full face stage is measured separately.

  1. initialisation   local: process start -> model loaded -> first comparison;  API: service process start -> /health OK -> first comparison
  2. per-pair latency sequential, one thread, preloaded bytes (realistic mix of compared and rejected attempts)
  3. throughput       the same pairs at concurrency 1/2/4/8
  4. HTTP floor       a request that the service rejects before any model work (empty files) vs the same call in-process
  5. face stage       run_face_stage (downloads + pair logic + fallback) at concurrency 4; evidence of both engines compared pair by pair
  6. memory           peak working set of the local process and of the service process

Run in the environment with the face packages:  .fr-venv\\Scripts\\python tools\\phase10_benchmark.py --out output\\audit\\phase10\\benchmark
The service is started and stopped by this tool (stateless launcher, sentinel database URL: it can reach no database).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import psycopg2  # noqa: F401
except ImportError:  # pragma: no cover - the benchmark never touches a database
    stub = types.ModuleType("psycopg2")
    stub.Error = Exception
    sys.modules["psycopg2"] = stub

from dedup_pipeline.face_recognition.client import FrDopamsClient  # noqa: E402
from dedup_pipeline.face_recognition.config import FrConfig  # noqa: E402
from dedup_pipeline.face_recognition.errors import FrPairRejected  # noqa: E402
from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import LocalFacePairComparer  # noqa: E402
from dedup_pipeline.face_recognition.local_config import LocalFaceConfig  # noqa: E402
from dedup_pipeline.face_recognition.pair_evidence import FrDopamsPairComparer  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring import ScoringConfig  # noqa: E402
from dedup_pipeline.scoring.face import FaceCache, FaceTask, ImageOutcomeMemory, inputs_key, run_face_stage  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402

KEY = "phase10-local-benchmark-key-123"
PORT = 8012
STORED = ROOT / "output" / "scoring" / "face_evidence_cache.jsonl"


# ----------------------------------------------------------------------------------------------------------------- helpers

def peak_working_set_mb(pid: int) -> float:
    """Peak working set of a process (Windows), in MB."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)  # QUERY_INFORMATION | VM_READ
    if not handle:
        return float("nan")
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    return round(counters.PeakWorkingSetSize / 1048576, 1) if ok else float("nan")


def start_service() -> tuple[subprocess.Popen, float]:
    env = {**os.environ, "DATABASE_URL": "postgresql://nodb:nodb@127.0.0.1:1/nodb", "COMPARE_API_KEY": KEY}
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "run_fr_dopams_stateless.py"), str(PORT)], env=env, cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return proc, time.perf_counter() - t0
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("the service did not start")


def stop(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait(timeout=20)


def api_comparer() -> FrDopamsPairComparer:
    return FrDopamsPairComparer(FrDopamsClient(FrConfig.from_env({"FR_DOPAMS_BASE_URL": f"http://127.0.0.1:{PORT}", "FR_DOPAMS_API_KEY": KEY})))


def outcome_key(comparer, a: bytes, b: bytes):
    """The comparable result of one call: ('compared', similarity, decision) or ('rejected', [(side, kind, reason, detail)])."""
    try:
        out = comparer.compare(a, b)
        return ("compared", out.similarity, out.decision, out.threshold, out.model, out.metric)
    except FrPairRejected as exc:
        return ("rejected", [(r.side, r.kind, r.reason, r.detail) for r in exc.rejections])


def timed_calls(comparer, pairs, workers: int) -> tuple[list[float], float]:
    latencies: list[float] = []
    lock = threading.Lock()

    def one(p):
        t = time.perf_counter()
        outcome_key(comparer, p[0], p[1])
        dt = time.perf_counter() - t
        with lock:
            latencies.append(dt)

    t0 = time.perf_counter()
    if workers == 1:
        for p in pairs:
            one(p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, pairs))
    return latencies, time.perf_counter() - t0


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


LOCAL_INIT = r"""
import sys, time, json, io
t0 = time.perf_counter()
sys.path.insert(0, r"%(root)s")
import types
try:
    import psycopg2
except ImportError:
    m = types.ModuleType("psycopg2"); m.Error = Exception; sys.modules["psycopg2"] = m
import numpy as np
from PIL import Image
from dedup_pipeline.face_recognition import LocalFacePairComparer, LocalFaceConfig, FrPairRejected
t1 = time.perf_counter()
c = LocalFacePairComparer(LocalFaceConfig(provider="cpu")); c.verify()
t2 = time.perf_counter()
b = io.BytesIO(); Image.fromarray(np.full((300, 300, 3), 128, np.uint8)).save(b, "PNG"); blank = b.getvalue()
try:
    c.compare(blank, blank)
except FrPairRejected:
    pass
t3 = time.perf_counter()
import ctypes
print(json.dumps({"import_s": t1 - t0, "model_load_s": t2 - t1, "first_compare_s": t3 - t2, "total_s": t3 - t0, "pid": __import__("os").getpid()}))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=60, help="attempts (image pairs) in the latency/throughput sample")
    ap.add_argument("--stage-pairs", type=int, default=60, help="stored cases in the full face-stage comparison")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {"machine": {"python": sys.version.split()[0], "cpu_count": os.cpu_count(), "gpu": "none (onnxruntime offers CPU only)"}, "rounds": args.rounds}

    # ---------------------------------------------------------------------------------------------------- inputs
    stored = [json.loads(line) for line in open(STORED, encoding="utf-8")]
    config = ScoringConfig(rule_version="score-v2", reference_date="2026-10-12", face_concurrency=4)
    images = validate_images(ROOT / "output" / "image_resolution", None)
    records = {r["media_id"]: r for p in {rec["evidence"]["person_a_id"] for rec in stored} | {rec["evidence"]["person_b_id"] for rec in stored}
               for r in images.index.get(p).usable_images}
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), HttpFetcher())
    rng = random.Random(args.seed)
    attempts = [(a["media_id_a"], a["media_id_b"], a["outcome"]) for rec in stored for a in (rec["evidence"].get("attempts") or []) if a.get("outcome") in ("compared", "rejected")]
    sample = rng.sample(attempts, args.pairs)
    cache: dict[str, bytes] = {}
    for ma, mb, _ in sample:
        for m in (ma, mb):
            if m not in cache:
                cache[m] = loader.load(records[m])
    pairs = [(cache[ma], cache[mb]) for ma, mb, _ in sample]
    result["sample"] = {"image_pairs": len(pairs), "stored_outcome_mix": {k: sum(1 for _, _, o in sample if o == k) for k in ("compared", "rejected")},
                        "distinct_images": len(cache), "median_image_kb": round(statistics.median(len(b) for b in cache.values()) / 1024)}

    # --------------------------------------------------------------------------------------------- 1. initialisation
    local_init, api_init = [], []
    for _ in range(args.rounds):
        done = subprocess.run([sys.executable, "-c", LOCAL_INIT % {"root": str(ROOT)}], capture_output=True, text=True, cwd=str(ROOT))
        local_init.append(json.loads(done.stdout.strip().splitlines()[-1]))
        proc, ready = start_service()
        comparer = api_comparer()
        t = time.perf_counter()
        outcome_key(comparer, pairs[0][0], pairs[0][1])
        first = time.perf_counter() - t
        api_init.append({"start_to_health_s": ready, "first_compare_s": first, "total_s": ready + first})
        stop(proc)
    med = lambda rows, k: round(statistics.median(r[k] for r in rows), 2)
    result["initialisation_seconds_median"] = {
        "local": {"import": med(local_init, "import_s"), "model_load_and_fingerprint": med(local_init, "model_load_s"), "first_compare": med(local_init, "first_compare_s"), "total": med(local_init, "total_s")},
        "api_service": {"process_start_to_health_ok": med(api_init, "start_to_health_s"), "first_compare_over_http": med(api_init, "first_compare_s"), "total": med(api_init, "total_s")},
        "note": "the service is started once and stays up; the local model is loaded once per pipeline run"}

    # ---------------------------------------------------------------------------- 2-5 with one service and one local engine
    proc, _ = start_service()
    local = LocalFacePairComparer(LocalFaceConfig(provider="cpu"))
    local.verify()
    api = api_comparer()
    for c in (local, api):  # warm-up, not measured
        for p in pairs[:4]:
            outcome_key(c, *p)

    # exact agreement on every sampled pair (the same images through both engines)
    agree = disagree = 0
    detail = []
    for i, p in enumerate(pairs):
        a, b = outcome_key(local, *p), outcome_key(api, *p)
        if a == b:
            agree += 1
        else:
            disagree += 1
            detail.append({"pair": i, "local": a, "api": b})
    result["identical_inputs_through_both_engines"] = {"pairs": len(pairs), "identical_result": agree, "different": disagree, "differences": detail[:10]}

    lat = {"local": [], "api": []}
    for _ in range(args.rounds):
        for name, c in (("local", local), ("api", api)):
            l, _ = timed_calls(c, pairs, 1)
            lat[name].append(l)
    flat = lambda name: [x for run in lat[name] for x in run]
    result["latency_ms_sequential"] = {name: {"median": round(1000 * statistics.median(flat(name)), 1), "p90": round(1000 * pct(flat(name), 0.9), 1), "mean": round(1000 * statistics.mean(flat(name)), 1),
                                              "per_round_median": [round(1000 * statistics.median(r), 1) for r in lat[name]]} for name in lat}

    thr: dict = {}
    for workers in (1, 2, 4, 8):
        rows = {"local": [], "api": []}
        for _ in range(args.rounds):
            for name, c in (("local", local), ("api", api)):
                _, wall = timed_calls(c, pairs, workers)
                rows[name].append(len(pairs) / wall)
        thr[str(workers)] = {name: {"pairs_per_second_median": round(statistics.median(v), 2), "rounds": [round(x, 2) for x in v]} for name, v in rows.items()}
    result["throughput_pairs_per_second"] = thr

    empty = [(b"", b"")] * 60  # rejected before any model work: the service still pays for HTTP, multipart parsing and authentication
    floor = {"local": [], "api": []}
    for _ in range(args.rounds):
        for name, c in (("local", local), ("api", api)):
            l, _ = timed_calls(c, empty, 1)
            floor[name].append(statistics.median(l))
    result["http_floor_ms_empty_images_median"] = {k: round(1000 * statistics.median(v), 2) for k, v in floor.items()}
    inference = statistics.median(flat("api")) - statistics.median(floor["api"]) + statistics.median(floor["local"])
    result["inference_vs_http_ms"] = {"api_total_median": result["latency_ms_sequential"]["api"]["median"], "http_floor_median": round(1000 * statistics.median(floor["api"]), 2),
                                      "model_work_inside_service_estimate": round(1000 * inference, 1), "local_total_median": result["latency_ms_sequential"]["local"]["median"],
                                      "note": "the floor is a request the service rejects before any model work; model work is the remainder"}

    # ---- full face stage: downloads, pair logic, fallback, evidence; both engines on the same stored cases
    chosen = sorted(rng.sample(range(len(stored)), args.stage_pairs))
    tasks = [FaceTask(stored[i]["candidate_id"], stored[i]["evidence"]["person_a_id"], stored[i]["evidence"]["person_b_id"],
                      inputs_key(images.index, stored[i]["evidence"]["person_a_id"], stored[i]["evidence"]["person_b_id"], config)) for i in chosen]
    stage = {"local": [], "api": []}
    evidence = {}
    for r in range(args.rounds):
        for name, c in (("local", local), ("api", api)):
            scratch = args.out / f"stage_{name}.jsonl"
            scratch.unlink(missing_ok=True)
            fc = FaceCache(scratch, {"stage": name})
            t = time.perf_counter()
            res, _ = run_face_stage(tasks, images.index, loader, c, fc, config, memory=ImageOutcomeMemory())
            stage[name].append(time.perf_counter() - t)
            fc.close()
            evidence[name] = {cid: ev.to_dict() for cid, ev in res.items()}
    same = sum(evidence["local"][t.candidate_id] == evidence["api"][t.candidate_id] for t in tasks)
    truth = sum(evidence["local"][stored[i]["candidate_id"]] == stored[i]["evidence"] for i in chosen)
    result["face_stage_seconds"] = {name: {"median": round(statistics.median(v), 1), "rounds": [round(x, 1) for x in v]} for name, v in stage.items()}
    result["face_stage_seconds"]["cases"], result["face_stage_seconds"]["concurrency"] = len(tasks), 4
    result["face_stage_evidence"] = {"cases": len(tasks), "local_equals_api": same, "local_equals_stored_original": truth}

    # ------------------------------------------------------------------------------------------------------ memory
    result["peak_memory_mb"] = {"local_process_incl_python_and_all_pipeline_imports": peak_working_set_mb(os.getpid()), "api_service_process": peak_working_set_mb(proc.pid),
                                "this_process_also_ran_the_http_client": True}
    stop(proc)
    result["initialisation_raw"] = {"local": local_init, "api": api_init}
    (args.out / "benchmark_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "initialisation_raw"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
