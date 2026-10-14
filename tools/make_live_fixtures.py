"""Build image fixtures for the live /compare tests from InsightFace's own bundled sample photos.

No CCTNS/DOPAMS data is used. Run with the FR-DOPAMS virtualenv (needs cv2 + insightface):

    $env:DATABASE_URL = "postgresql://nodb:nodb@127.0.0.1:1/nodb"   # dead sentinel; nothing here uses a database
    .fr-venv\\Scripts\\python tools/make_live_fixtures.py <output-dir>

From the group photo ``t1.jpg`` (six different people) it writes one single-face crop per person, then
derives re-encoded/rescaled variants of a crop (same person, same source photo; NOT independent
photographs), plus images for each failure mode. Which failure mode each image really triggers is
checked against FR-DOPAMS's own functions and recorded in ``fixtures.json``.
"""

import io
import json
import os
import sys
from pathlib import Path

FR_DIR = Path(os.environ.get("FR_DOPAMS_SOURCE_DIR") or Path(__file__).resolve().parents[2] / "FR-DOPAMS")
if os.environ.get("DATABASE_URL") != "postgresql://nodb:nodb@127.0.0.1:1/nodb":
    sys.exit("Set DATABASE_URL to the dead sentinel first (see docstring); this script must never reach a database.")
sys.path.insert(0, str(FR_DIR))

import cv2  # noqa: E402
import insightface  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src import face_service as fs  # noqa: E402
from src.detect import detect_faces  # noqa: E402


def encode(image, ext=".png", quality=None) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    ok, buf = cv2.imencode(ext, image, params)
    assert ok
    return buf.tobytes()


def classify(data: bytes):
    """What FR-DOPAMS's own code says about this image: ("ok", None) or (code, reason)."""
    try:
        fs._analyse_image(data)
        return "ok", None
    except fs.ImageRejected as exc:
        return exc.code, exc.reason


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    samples = Path(insightface.__file__).parent / "data" / "images"
    group = cv2.imread(str(samples / "t1.jpg"))
    aligned = cv2.imread(str(samples / "Tom_Hanks_54745.png"))
    files: dict[str, bytes] = {}
    expect: dict[str, list] = {}

    faces = sorted(detect_faces(group), key=lambda f: f.bbox[0])
    assert len(faces) == 6, f"expected 6 faces in the group photo, found {len(faces)}"
    crops = []
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        pad = int(0.6 * max(x2 - x1, y2 - y1))
        crop = group[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        crops.append(crop)
        files[f"person{i}.png"] = encode(crop)

    base = crops[0]
    files["person0_flip.png"] = encode(cv2.flip(base, 1))
    files["person0_small.jpg"] = encode(cv2.resize(base, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA), ".jpg", 90)
    files["person0_jpeg60.jpg"] = encode(base, ".jpg", 60)
    files["person0_bright.png"] = encode(cv2.convertScaleAbs(base, alpha=1.15, beta=18))

    files["group_photo.jpg"] = encode(group, ".jpg", 92)  # six faces of similar size
    files["no_face_aligned_crop.png"] = encode(aligned)  # a tight 112x112 face crop the detector cannot find
    files["no_face_noise.png"] = encode(np.random.default_rng(7).integers(0, 256, (300, 300, 3), dtype=np.uint8))

    for scale in (0.30, 0.33, 0.36, 0.40, 0.25):  # shrink until the face is found but under MIN_FACE_SIZE
        small = encode(cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if classify(small) == ("low_quality", "too_small"):
            files["low_quality_small_face.png"] = small
            break
    for sigma in (3, 5, 7, 9, 12):  # blur until the face is still found but too soft
        soft = encode(cv2.GaussianBlur(base, (0, 0), sigma))
        if classify(soft) == ("low_quality", "too_blurred"):
            files["low_quality_blurred.png"] = soft
            break

    files["corrupt_truncated.jpg"] = encode(base, ".jpg", 90)[:900]
    files["not_an_image.txt"] = b"this is not an image\n" * 20
    files["empty.bin"] = b""
    files["too_large.bin"] = b"\xff\xd8\xff" + os.urandom(11 * 1024 * 1024)  # over MAX_UPLOAD_MB (10)
    bomb = io.BytesIO()
    Image.new("L", (7000, 7000)).save(bomb, "PNG")  # ~49M pixels: tiny file, huge bitmap
    files["pixel_bomb.png"] = bomb.getvalue()

    for name, data in files.items():
        (out / name).write_bytes(data)
        if len(data) <= 10 * 1024 * 1024 and name != "pixel_bomb.png":
            expect[name] = list(classify(data)) if data else ["invalid_image", "empty"]
    expect["pixel_bomb.png"] = list(classify(files["pixel_bomb.png"]))
    (out / "fixtures.json").write_text(json.dumps(expect, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in sorted(expect.items())}, indent=1))
    missing = [n for n in ("low_quality_small_face.png", "low_quality_blurred.png") if n not in files]
    if missing:
        print("NOT PRODUCED (live tests will skip these cases):", missing)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
