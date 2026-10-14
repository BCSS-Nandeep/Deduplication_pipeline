"""Optional link to the Phase 2 image-resolution outputs (read only; nothing is recomputed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..image_resolution.manifest import PERSON_SUMMARY_FILE
from ..image_resolution.runner import RUN_SUMMARY_FILE


@dataclass(frozen=True)
class ImageRefs:
    people: dict[str, dict[str, Any]]
    partial: bool  # the Phase 2 run covered only a subset of persons
    source_dir: str

    def for_person(self, person_id: str) -> dict[str, Any]:
        return self.people.get(person_id) or {"person_status": "not_in_manifest", "usable_images": 0, "selected_media_id": None}


def load_image_refs(directory: Optional[Path]) -> Optional[ImageRefs]:
    """Person -> image status from a *completed* Phase 2 run, or ``None`` if there is none."""
    if directory is None:
        return None
    try:
        summary = json.loads((directory / RUN_SUMMARY_FILE).read_text(encoding="utf-8"))
        parameters = summary.get("parameters", {})
        partial = any(parameters.get(k) for k in ("limit_persons", "source_system", "only_with_media"))
        people: dict[str, dict[str, Any]] = {}
        with open(directory / PERSON_SUMMARY_FILE, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                people[row["person_id"]] = {
                    "person_status": row["person_status"],
                    "usable_images": row["usable_image_count"],
                    "selected_media_id": row["selected_media_id"],
                }
    except (OSError, ValueError, KeyError):
        return None
    return ImageRefs(people, partial, directory.name)


def image_block(refs: ImageRefs, person_a: str, person_b: str) -> dict[str, Any]:
    a, b = refs.for_person(person_a), refs.for_person(person_b)
    return {"a": a, "b": b, "both_have_usable_image": a["usable_images"] > 0 and b["usable_images"] > 0}
