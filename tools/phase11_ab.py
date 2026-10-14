"""Phase 11 A/B driver: alternating rounds of ``phase11_face_stage.py`` over several package variants, medians reported.

    .face-venv\\Scripts\\python tools\\phase11_ab.py --variant baseline=<root> --variant current=<root> --rounds 3 --cases 200 --concurrency 4 --out <file.json>

Every run is a fresh process with an empty face cache and the same stored cases, so only the code differs. Evidence of every run is compared with the stored
original; any difference is reported (and makes the exit code 1).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, help="label=package root (a directory containing dedup_pipeline)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=ROOT / "output" / "perf11" / "ab_scratch")
    args = ap.parse_args()
    # "label=root" or "label=root;N" (N = concurrency for that variant; default --concurrency)
    variants = []
    conc = {}
    for v in args.variant:
        label, spec = v.split("=", 1)
        root, _, c = spec.partition(";")
        variants.append((label, root))
        conc[label] = int(c) if c else args.concurrency
    runs: dict[str, list[dict]] = {label: [] for label, _ in variants}
    bad = 0
    for r in range(args.rounds):
        for label, root in variants:
            out = args.scratch / f"{label}_{r}.json"
            args.scratch.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PHASE11_PKG_ROOT": str(Path(root).resolve())}
            cmd = [sys.executable, str(ROOT / "tools" / "phase11_face_stage.py"), "--cases", str(args.cases), "--concurrency", str(conc[label]),
                   "--out", str(out), "--scratch", str(args.scratch / f"cache_{label}")]
            done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            if not out.exists():
                raise SystemExit(f"{label} round {r + 1} failed:\n{done.stderr[-1500:]}")
            res = json.loads(out.read_text())
            runs[label].append(res)
            bad += res["cases"] - res["evidence_identical_to_stored_original"]
            print(f"round {r + 1} {label:14s} wall {res['face_stage_wall_seconds']:7.1f}s cpu {res['process_cpu_seconds']:7.1f}s peak {res['peak_working_set_mb']:7.1f} MB "
                  f"calls {res['comparer_calls']} downloads {res['image_downloads']} identical {res['evidence_identical_to_stored_original']}/{res['cases']}", flush=True)
    med = lambda label, key: round(statistics.median(x[key] for x in runs[label]), 2)
    base = variants[0][0]
    summary = {"cases": runs[base][0]["cases"], "rounds": args.rounds, "variants": {}}
    for label, root in variants:
        summary["variants"][label] = {
            "root": str(root), "concurrency": conc[label], "wall_seconds_rounds": [x["face_stage_wall_seconds"] for x in runs[label]], "median_wall_seconds": med(label, "face_stage_wall_seconds"),
            "median_cpu_seconds": med(label, "process_cpu_seconds"), "median_peak_working_set_mb": med(label, "peak_working_set_mb"), "median_init_seconds": med(label, "init_seconds"),
            "image_downloads": runs[label][0]["image_downloads"], "comparer_calls": runs[label][0]["comparer_calls"], "fingerprint": runs[label][0]["fingerprint"],
            "speedup_vs_" + base: round(med(base, "face_stage_wall_seconds") / med(label, "face_stage_wall_seconds"), 3),
            "evidence_identical_to_stored_original_all_rounds": all(x["evidence_identical_to_stored_original"] == x["cases"] for x in runs[label])}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("median_wall_seconds", "median_cpu_seconds", "median_peak_working_set_mb") or kk.startswith("speedup")}
                      for k, v in summary["variants"].items()}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
