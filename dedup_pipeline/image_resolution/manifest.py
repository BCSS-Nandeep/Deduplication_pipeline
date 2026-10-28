"""Turn resolver results into the manifest files.

Output order and content depend only on the database rows and the results, so
a rerun over unchanged inputs produces byte-identical files.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import VERIFIED_ACCESSIBLE, MediaResult, MediaRow, Person, Status

# Output file names (also read by dedup_pipeline.face_recognition without importing the runner or any DB code).
MANIFEST_FILE = "image_manifest.jsonl"
PERSON_SUMMARY_FILE = "person_summary.jsonl"

# person_status values
HAS_USABLE_IMAGE = "usable_image_available"
NO_MEDIA_RECORDS = "no_person_media_records"
NO_ACCESSIBLE_MEDIA = "no_accessible_media"
NO_USABLE_IMAGE = "no_usable_image"


def rank_key(row: MediaRow, category_priority: tuple[str, ...]):
    """Deterministic order for one person's media: preferred category, verified first, then media_id."""
    try:
        category_rank = category_priority.index(row.attachment_category or "")
    except ValueError:
        category_rank = len(category_priority)
    return (category_rank, 0 if row.availability_status == VERIFIED_ACCESSIBLE else 1, row.media_id)


def find_shared_paths(rows: Iterable[MediaRow]) -> dict[str, dict[str, set[str]]]:
    """Paths referenced by more than one *person*.

    Plain path equality from the database: a lead for investigation, not a
    content comparison and not evidence that the people are the same.
    """
    people: dict[str, set[str]] = defaultdict(set)
    media: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        path_ref, _ = row.chosen_path()
        if path_ref:
            people[path_ref].add(row.person_id)
            media[path_ref].add(row.media_id)
    return {p: {"person_ids": people[p], "media_ids": media[p]} for p in people if len(people[p]) > 1}


def build_outputs(
    persons: list[Person],
    rows: list[MediaRow],
    results: dict[str, MediaResult],
    category_priority: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (manifest records, person summaries, shared-path records)."""
    shared = find_shared_paths(rows)
    by_person: dict[str, list[MediaRow]] = defaultdict(list)
    for row in rows:
        by_person[row.person_id].append(row)

    manifest: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for person in sorted(persons, key=lambda p: p.person_id):
        ordered = sorted(by_person.get(person.person_id, []), key=lambda r: rank_key(r, category_priority))
        selected_id = next((r.media_id for r in ordered if results[r.media_id].usable), None)
        reasons: dict[str, int] = defaultdict(int)
        for rank, row in enumerate(ordered, 1):
            result = results[row.media_id]
            path_ref, _ = row.chosen_path()
            share_count = len(shared[path_ref]["person_ids"]) if path_ref in shared else 1
            if not result.usable:
                reasons[result.status.value] += 1
            manifest.append(
                {
                    "person_id": person.person_id,
                    "source_system": person.source_system,
                    "media_id": row.media_id,
                    "attachment_category": row.attachment_category,
                    "availability_status": row.availability_status,
                    "path_ref": result.path_ref,
                    "path_source": result.path_source,
                    "resolved_suffix": result.resolved_suffix,
                    "detected_type": result.detected_type,
                    "width": result.width,
                    "height": result.height,
                    "byte_size": result.byte_size,
                    "resolution_status": result.status.value,
                    "usable_for_face_recognition": result.usable,
                    "skip_reason": result.skip_reason,
                    "face_check_status": result.face_check_status,
                    "media_class": result.media_class,
                    "media_class_confidence": result.media_class_confidence,
                    "media_class_reason": result.media_class_reason,
                    "person_rank": rank,
                    "selected_for_person": row.media_id == selected_id,
                    "shared_path_flag": share_count > 1,
                    "shared_path_person_count": share_count,
                }
            )
        if not ordered:
            person_status = NO_MEDIA_RECORDS
        elif selected_id is not None:
            person_status = HAS_USABLE_IMAGE
        elif not any(r.availability_status == VERIFIED_ACCESSIBLE for r in ordered):
            person_status = NO_ACCESSIBLE_MEDIA
        else:
            person_status = NO_USABLE_IMAGE
        summaries.append(
            {
                "person_id": person.person_id,
                "source_system": person.source_system,
                "person_status": person_status,
                "media_record_count": len(ordered),
                "usable_image_count": sum(1 for r in ordered if results[r.media_id].usable),
                "selected_media_id": selected_id,
                "unusable_reasons": dict(sorted(reasons.items())),
            }
        )

    shared_records = [
        {
            "path_ref": path,
            "person_count": len(info["person_ids"]),
            "person_ids": sorted(info["person_ids"]),
            "media_ids": sorted(info["media_ids"]),
        }
        for path, info in sorted(shared.items())
    ]
    return manifest, summaries, shared_records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=True, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
