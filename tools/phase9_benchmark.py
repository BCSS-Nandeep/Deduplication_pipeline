"""Phase 9 benchmark: the ORIGINAL implementation (rebuilt by phase9_make_baseline.py) against the OPTIMISED one, same input, same machine,
alternating runs in fresh processes, followed by a byte-level comparison of every output with the preserved run.

python tools/phase9_benchmark.py --rule-version score-v2 --rounds 3 --reference-run output/scoring/runs/20261012T125026Z
Everything is written under output/perf/ (scratch). The preserved runs are only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "output" / "perf"
OUTPUT_FILES = ("scored_candidates.jsonl", "review_worksheet.csv", "scoring_config.json", "REVIEW_INSTRUCTIONS.md", "deferred_unevaluated.jsonl", "age_evidence_provenance.jsonl", "age_evidence_summary.json")
VOLATILE_SUMMARY_KEYS = {"run_id", "started_at", "finished_at", "elapsed_seconds", "peak_python_heap_mb", "peak_process_memory_mb", "workload", "database"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(run: Path) -> dict[str, str]:
    out = {f: sha(run / f) for f in OUTPUT_FILES if (run / f).is_file()}
    for page in sorted((run / "review_report").glob("*.html")):
        out[f"review_report/{page.name}"] = sha(page)
    return out


def normalised_summary(run: Path) -> str:
    data = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    if isinstance(data.get("face"), dict):
        data["face"] = {k: v for k, v in data["face"].items() if k != "engine"}  # Phase 10 names the face engine; the Phase 9 baseline package predates it
    return json.dumps({k: v for k, v in data.items() if k not in VOLATILE_SUMMARY_KEYS}, sort_keys=True)


def one_run(label: str, pkg_root: Path, rule_version: str, scratch: Path) -> dict:
    if scratch.exists():
        shutil.rmtree(scratch)
    sys.path.insert(0, str(ROOT))
    from dedup_pipeline.db import DEFAULT_ENV_FILE  # the baseline copy lives deeper in the tree, so its relative default would not find the file

    env = {**os.environ, "PHASE9_PKG_ROOT": str(pkg_root), "DEDUP_ENV_FILE": str(DEFAULT_ENV_FILE), "FR_DOPAMS_BASE_URL": "http://127.0.0.1:9", "FR_DOPAMS_API_KEY": "unused"}
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "phase9_profile.py"), "--rule-version", rule_version, "--scratch", str(scratch), "--label", "bench"],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        raise SystemExit(f"{label} run failed:\n{proc.stdout[-800:]}\n{proc.stderr[-1500:]}")
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-version", default="score-v2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reference-run", type=Path, required=True)
    ap.add_argument("--baseline-pkg", type=Path, default=PERF / "baseline_pkg")
    ap.add_argument("--out", type=Path, default=PERF / "benchmark_result.json")
    args = ap.parse_args()
    results = {"baseline": [], "optimised": []}
    for r in range(args.rounds):
        for label, pkg in (("baseline", args.baseline_pkg), ("optimised", ROOT)):
            res = one_run(label, pkg, args.rule_version, PERF / f"bench_{label}")
            results[label].append(res)
            print(f"round {r + 1} {label:9s}: wall {res['wall_seconds']:6.1f}s  peak RSS {res['peak_process_memory_mb']} MB  heap {res['peak_python_heap_mb']} MB", flush=True)
    # the last run of each is compared; the reference is the preserved run
    ref = tree_hashes(args.reference_run)
    comparison = {}
    for label in ("baseline", "optimised"):
        run = PERF / f"bench_{label}" / "runs" / "bench"
        got = tree_hashes(run)
        comparison[label] = {"files_compared": len(ref), "identical_to_preserved_run": sorted(k for k in ref if got.get(k) == ref[k]),
                             "different": sorted(k for k in ref if got.get(k) != ref[k]), "extra_files": sorted(set(got) - set(ref))}
    a = PERF / "bench_baseline" / "runs" / "bench"
    b = PERF / "bench_optimised" / "runs" / "bench"
    ha, hb = tree_hashes(a), tree_hashes(b)
    comparison["baseline_vs_optimised"] = {"files": len(ha), "identical": sum(ha.get(k) == hb.get(k) for k in ha), "different": sorted(k for k in ha if ha.get(k) != hb.get(k)),
                                           "summary_identical_apart_from_timings": normalised_summary(a) == normalised_summary(b)}
    med = lambda k, f: round(statistics.median(x[f] for x in results[k]), 2)
    summary = {"rule_version": args.rule_version, "rounds": args.rounds,
               "wall_seconds": {k: [x["wall_seconds"] for x in v] for k, v in results.items()},
               "median_wall_seconds": {k: med(k, "wall_seconds") for k in results}, "speedup_median": round(med("baseline", "wall_seconds") / med("optimised", "wall_seconds"), 3),
               "peak_process_memory_mb_median": {k: med(k, "peak_process_memory_mb") for k in results}, "peak_python_heap_mb_median": {k: med(k, "peak_python_heap_mb") for k in results},
               "stage_seconds_median": {k: {s: round(statistics.median(x["stage_seconds"].get(s, 0.0) for x in v), 2) for s in sorted({s for x in v for s in x["stage_seconds"]})} for k, v in results.items()},
               "output_equivalence": comparison}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("median_wall_seconds", "speedup_median", "peak_process_memory_mb_median", "peak_python_heap_mb_median")}, indent=1))
    print(json.dumps(comparison["baseline_vs_optimised"]))
    for label in ("baseline", "optimised"):
        print(label, "vs preserved run: identical", len(comparison[label]["identical_to_preserved_run"]), "different", comparison[label]["different"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
