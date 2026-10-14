"""Phase 11 end-to-end: the real scoring command (face stage from an EMPTY cache + scoring + reviewer package) for two package roots, alternating rounds.

Each run writes to its own scratch output root and uses the read-only database connection. Reported: wall time, the run summary's peak memory, and the
byte equality of the outputs between the two code versions (and, with --reference, against a preserved run).

    .face-venv\\Scripts\\python tools\\phase11_e2e.py --variant baseline=output\\perf11\\baseline_pkg --variant final=. --rounds 2 --out <file.json>
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("scored_candidates.jsonl", "review_worksheet.csv", "deferred_unevaluated.jsonl", "REVIEW_INSTRUCTIONS.md")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digests(run: Path) -> dict[str, str]:
    out = {n: sha(run / n) for n in OUTPUTS}
    for page in sorted((run / "review_report").glob("*.html")):
        out[f"review_report/{page.name}"] = sha(page)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=ROOT / "output" / "perf11" / "e2e")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from dedup_pipeline.db import DEFAULT_ENV_FILE

    variants = [v.split("=", 1) for v in args.variant]
    runs: dict[str, list[dict]] = {label: [] for label, _ in variants}
    kept: dict[str, Path] = {}
    for r in range(args.rounds):
        for label, root in variants:
            out_root = args.scratch / f"{label}_{r}"
            shutil.rmtree(out_root, ignore_errors=True)
            env = {**os.environ, "DEDUP_ENV_FILE": str(DEFAULT_ENV_FILE)}
            env.pop("PYTHONPATH", None)
            cmd = [sys.executable, "-m", "dedup_pipeline.scoring", "run", "--rule-version", "score-v2", "--reference-date", "2026-10-12", "--output-root", str(out_root),
                   "--run-label", "run", "--no-resume", "--candidates-dir", str(ROOT / "output" / "candidates"), "--image-dir", str(ROOT / "output" / "image_resolution")]
            t = time.perf_counter()
            done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(Path(root).resolve()))
            wall = time.perf_counter() - t
            summary_path = out_root / "runs" / "run" / "run_summary.json"
            if not summary_path.exists():
                raise SystemExit(f"{label} round {r + 1} failed:\n{done.stderr[-1500:]}")
            s = json.loads(summary_path.read_text())
            runs[label].append({"wall": wall, "elapsed": s["elapsed_seconds"], "peak_process_memory_mb": s["peak_process_memory_mb"], "peak_python_heap_mb": s["peak_python_heap_mb"],
                                "status": s["status"], "scored": s["candidates"]["scored"], "awaiting_human_review": s["candidates"]["awaiting_human_review"],
                                "face_pairs": s["face"]["eligible_pairs_both_usable_images"], "retryable": s["face"]["retryable_errors"], "database_writes": s["database_writes"],
                                "read_only": s["database"]["transaction_read_only"], "concurrency": json.loads((out_root / "runs" / "run" / "scoring_config.json").read_text())["config"]["face_concurrency"]})
            print(f"round {r + 1} {label:9s}: wall {wall:6.1f}s  summary {s['elapsed_seconds']}s  peak {s['peak_process_memory_mb']} MB  status {s['status']}  scored {s['candidates']['scored']}", flush=True)
            kept[f"{label}_{r}"] = out_root / "runs" / "run"
    base, final = variants[0][0], variants[1][0]
    d_base, d_final = digests(kept[f"{base}_{args.rounds - 1}"]), digests(kept[f"{final}_{args.rounds - 1}"])
    result = {"rounds": args.rounds, "variants": {}}
    for label, root in variants:
        rows = runs[label]
        result["variants"][label] = {"root": str(root), "median_wall_seconds": round(statistics.median(x["wall"] for x in rows), 1), "wall_rounds": [round(x["wall"], 1) for x in rows],
                                     "median_peak_process_memory_mb": statistics.median(x["peak_process_memory_mb"] for x in rows), "face_concurrency_used": rows[0]["concurrency"],
                                     "status": {x["status"] for x in rows}.pop(), "scored": rows[0]["scored"], "all_pending": all(x["awaiting_human_review"] == x["scored"] for x in rows),
                                     "database_writes": sum(x["database_writes"] for x in rows), "read_only_session": {x["read_only"] for x in rows}.pop(), "retryable_errors": sum(x["retryable"] for x in rows)}
    result["speedup_median_wall"] = round(result["variants"][base]["median_wall_seconds"] / result["variants"][final]["median_wall_seconds"], 3)
    result["baseline_vs_final_outputs"] = {"files": len(d_base), "identical": sum(d_base.get(k) == d_final.get(k) for k in d_base), "different": sorted(k for k in d_base if d_base.get(k) != d_final.get(k)),
                                           "only_in_one": sorted(set(d_base) ^ set(d_final))}
    result["final_run_dir"] = str(kept[f"{final}_{args.rounds - 1}"])
    result["baseline_run_dir"] = str(kept[f"{base}_{args.rounds - 1}"])
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
