"""The identity of a face-comparison engine, for the face-cache signature: a digest of its settings, its model files and its library versions.

This is the ONLY module of the package that hashes anything, and it hashes only (a) the bytes of the engine's model files and (b) a JSON
description of its settings. It never receives an image or a person's data and is not used to compare or de-duplicate images (the
pipeline's scope guards forbid that; the one exemption they make is for this file, and a test pins what it may contain).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
