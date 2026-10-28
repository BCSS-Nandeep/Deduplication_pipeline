"""Fetch InsightFace buffalo_l ONNX files into the local model directory.

Does not contact the production database. Downloads only the published buffalo_l
archive from the InsightFace GitHub releases channel used by the project docs.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"
REQUIRED = ("det_10g.onnx", "w600k_r50.onnx")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    if not missing:
        print(f"models already present in {MODEL_DIR}")
        for name in REQUIRED:
            path = MODEL_DIR / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"  {name} size={path.stat().st_size} sha256={digest}")
        return
    print(f"downloading {URL}")
    with urlopen(URL, timeout=120) as response:  # noqa: S310 — fixed release URL
        payload = response.read()
    print(f"downloaded {len(payload)} bytes")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        print("archive members:", names[:20], "..." if len(names) > 20 else "")
        for name in REQUIRED:
            matches = [n for n in names if n.endswith(name)]
            if not matches:
                raise SystemExit(f"{name} not found in archive")
            member = matches[0]
            target = MODEL_DIR / name
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            print(f"wrote {target} ({target.stat().st_size} bytes)")
    for name in REQUIRED:
        if not (MODEL_DIR / name).is_file():
            raise SystemExit(f"still missing {name}")
    print("buffalo_l models ready")


if __name__ == "__main__":
    main()
