"""Phase 11: image-resolution worker-count measurement on a representative sample, alternating rounds, scratch output only.

Read-only against the database and the file server (the module only issues GET/HEAD-style reads). For each worker count the same sample is resolved
from scratch; wall time, the split between the database load and the file-server phase, error statuses (a sign of server stress) and a digest-free
byte comparison of the produced manifests (determinism across worker counts) are reported. Nothing in output/image_resolution is touched.

    .venv\\Scripts\\python tools\\phase11_image_resolution.py --workers 4 8 16 --rounds 3 --persons 4000 --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("image_manifest.jsonl", "person_summary.jsonl", "shared_paths.jsonl")


def stamp(line: str):
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)", line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000 if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--persons", type=int, default=4000)
    ap.add_argument("--source", default="V2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=ROOT / "output" / "perf11" / "img_scratch")
    ap.add_argument("--python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    args = ap.parse_args()
    runs: dict[int, list[dict]] = {w: [] for w in args.workers}
    first_bytes: dict[str, bytes] = {}
    identical = {w: True for w in args.workers}
    for r in range(args.rounds):
        for w in args.workers:
            out = args.scratch / f"w{w}"
            shutil.rmtree(out, ignore_errors=True)
            cmd = [args.python, "-m", "dedup_pipeline.image_resolution", "--output-dir", str(out), "--source-system", args.source, "--only-with-media",
                   "--limit-persons", str(args.persons), "--workers", str(w), "--fresh"]
            t = time.perf_counter()
            done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            wall = time.perf_counter() - t
            lines = done.stderr.splitlines()
            summary = json.loads((out / "run_summary.json").read_text())
            loaded = next((stamp(l) for l in lines if "Loaded" in l and "persons" in l), None)
            to_resolve = next((stamp(l) for l in lines if "to resolve" in l), None)
            resolved_done = [stamp(l) for l in lines if stamp(l)]
            files = {f: (out / f).read_bytes() for f in FILES if (out / f).exists()}
            for f, b in files.items():
                if f not in first_bytes:
                    first_bytes[f] = b
                elif first_bytes[f] != b:
                    identical[w] = False
            runs[w].append({"wall": wall, "summary_elapsed": summary["elapsed_seconds"], "media_records": summary["media_records"], "status": summary["resolution_status"],
                            "database_phase": (loaded - resolved_done[0]) if loaded and resolved_done else None,
                            "http_phase": (resolved_done[-1] - to_resolve) if to_resolve else None})
            print(f"round {r + 1} workers {w:2d}: wall {wall:5.1f}s  summary {summary['elapsed_seconds']}s  records {summary['media_records']}  errors "
                  f"{ {k: v for k, v in summary['resolution_status'].items() if 'error' in k} }", flush=True)
    result = {"sample": {"persons": args.persons, "source": args.source, "media_records": runs[args.workers[0]][0]["media_records"]}, "rounds": args.rounds, "workers": {}}
    for w in args.workers:
        rows = runs[w]
        result["workers"][str(w)] = {"median_wall_seconds": round(statistics.median(x["wall"] for x in rows), 2), "wall_rounds": [round(x["wall"], 2) for x in rows],
                                     "median_summary_elapsed_seconds": round(statistics.median(x["summary_elapsed"] for x in rows), 2),
                                     "resolution_status_first_round": rows[0]["status"], "manifests_byte_identical_to_the_first_run": identical[w]}
    base = str(args.workers[0])
    for w in args.workers:
        result["workers"][str(w)]["speedup_vs_" + base] = round(result["workers"][base]["median_wall_seconds"] / result["workers"][str(w)]["median_wall_seconds"], 3)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({w: {k: v for k, v in d.items() if k in ("median_wall_seconds", "manifests_byte_identical_to_the_first_run") or k.startswith("speedup")}
                      for w, d in result["workers"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
