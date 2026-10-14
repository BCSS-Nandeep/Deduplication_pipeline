"""Phase 11: where does the time of the local face engine go? Stage timings per image on the real engine, real images, CPU.

Stages (per image): file-server download; pixel-budget header check; cv2 decode; face detection (det_10g); the recognition pass that
`FaceAnalysis.get` runs on every detected face (the library embeds each face even though the comparer re-embeds with flip averaging and ignores it);
quality check (sharpness etc.); flip-averaged embedding (the one that is used). Medians over rounds; written under output/audit/phase11/.

    .face-venv\\Scripts\\python tools\\phase11_face_profile.py --out output\\audit\\phase11\\face_profile
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import psycopg2  # noqa: F401
except ImportError:  # pragma: no cover
    stub = types.ModuleType("psycopg2")
    stub.Error = Exception
    sys.modules["psycopg2"] = stub

from dedup_pipeline.face_recognition.image_source import ManifestImageLoader  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import ImageRejected, get_engine  # noqa: E402
from dedup_pipeline.face_recognition.local_config import LocalFaceConfig  # noqa: E402
from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher  # noqa: E402
from dedup_pipeline.image_resolution.resolver import ResolverConfig  # noqa: E402
from dedup_pipeline.scoring.inputs import validate_images  # noqa: E402

STORED = ROOT / "output" / "scoring" / "face_evidence_cache.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    engine = get_engine(LocalFaceConfig())
    np, cv2 = engine._np, engine._cv2
    from insightface.app.common import Face

    stored = [json.loads(line) for line in open(STORED, encoding="utf-8")]
    images = validate_images(ROOT / "output" / "image_resolution", None)
    records = {r["media_id"]: r for rec in stored for p in (rec["evidence"]["person_a_id"], rec["evidence"]["person_b_id"]) for r in images.index.get(p).usable_images}
    media = sorted({m for rec in stored for a in (rec["evidence"].get("attempts") or []) for m in (a.get("media_id_a"), a.get("media_id_b")) if m})
    chosen = random.Random(args.seed).sample(media, args.images)
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), HttpFetcher())

    download = []
    data = {}
    for m in chosen:
        t = time.perf_counter()
        data[m] = loader.load(records[m])
        download.append(time.perf_counter() - t)
    for m in chosen[:3]:  # warm-up (a rejection is a normal answer)
        try:
            engine.analyse(data[m])
        except ImageRejected:
            pass

    rows = {k: [] for k in ("pixel_budget", "decode", "detect", "library_recognition_pass", "quality_check", "flip_embedding", "analyse_total")}
    faces_per_image = []
    outcomes = {"usable": 0, "rejected": 0}
    for _ in range(args.rounds):
        for m in chosen:
            b = data[m]
            t = time.perf_counter(); engine._check_pixel_budget(b); rows["pixel_budget"].append(time.perf_counter() - t)
            t = time.perf_counter(); img = engine._decode(b); rows["decode"].append(time.perf_counter() - t)
            t = time.perf_counter(); bboxes, kpss = engine.app.det_model.detect(img, max_num=0, metric="default"); rows["detect"].append(time.perf_counter() - t)
            faces = [Face(bbox=bboxes[i, 0:4], kps=None if kpss is None else kpss[i], det_score=bboxes[i, 4]) for i in range(bboxes.shape[0])]
            faces_per_image.append(len(faces))
            t = time.perf_counter()
            for f in faces:
                engine.app.models["recognition"].get(img, f)
            rows["library_recognition_pass"].append(time.perf_counter() - t)
            usable = [f for f in faces if f.det_score >= engine.config.min_detection_confidence]
            if usable:
                best = max(usable, key=engine._face_area)
                t = time.perf_counter()
                try:
                    engine._check_quality(img, best)
                    rows["quality_check"].append(time.perf_counter() - t)
                    t = time.perf_counter(); engine._embedding(best, img); rows["flip_embedding"].append(time.perf_counter() - t)
                except ImageRejected:
                    rows["quality_check"].append(time.perf_counter() - t)
            t = time.perf_counter()
            try:
                engine.analyse(b)
                outcomes["usable"] += 1
            except ImageRejected:
                outcomes["rejected"] += 1
            rows["analyse_total"].append(time.perf_counter() - t)

    ms = lambda v: round(1000 * statistics.median(v), 2) if v else None
    result = {
        "images": len(chosen), "rounds": args.rounds, "median_kb": round(statistics.median(len(b) for b in data.values()) / 1024), "outcomes_over_rounds": outcomes,
        "faces_detected_per_image": {"mean": round(statistics.mean(faces_per_image), 2), "images_with_0": faces_per_image.count(0), "images_with_2_or_more": sum(1 for n in faces_per_image if n > 1)},
        "median_ms_per_image": {"download_file_server": ms(download), **{k: ms(v) for k, v in rows.items()}},
        "mean_ms_per_image": {"download_file_server": round(1000 * statistics.mean(download), 2), **{k: round(1000 * statistics.mean(v), 2) for k, v in rows.items() if v}},
        "share_of_analyse_total_mean_pct": {k: round(100 * statistics.mean(rows[k]) / statistics.mean(rows["analyse_total"]), 1) for k in ("pixel_budget", "decode", "detect", "library_recognition_pass", "quality_check", "flip_embedding")},
        "note": "library_recognition_pass is the recognition-model call that FaceAnalysis.get makes for every detected face; its output (face.embedding) is not used when landmarks exist and flip averaging is on",
    }
    (args.out / "face_profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
