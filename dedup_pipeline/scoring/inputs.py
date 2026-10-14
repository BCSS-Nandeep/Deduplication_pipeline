"""Validate the Phase 2 and Phase 5 outputs before anything is scored. Fails closed on anything inconsistent.

Completion markers: a Phase 5 run is complete only if ``run_summary.json`` says ``status: completed`` *and*
``run_state.json`` agrees. A Phase 2 run is complete if ``run_summary.json`` exists (it is written last); whether it
covered every person (a "full" run) is read from its recorded parameters and counts. An incomplete image run is not
an error here: it is reported as a limitation, and persons it did not cover are never assumed to have been checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from ..candidates.runner import GENERATOR_VERSION as CANDIDATE_VERSION
from ..candidates.store import CANDIDATES_FILE, DEFERRED_FILE, STATE_FILE, SUMMARY_FILE
from ..db import APPROVED_DATABASES
from ..face_recognition.image_source import ImageIndex
from ..image_resolution.manifest import MANIFEST_FILE, PERSON_SUMMARY_FILE
from ..image_resolution.runner import RUN_SUMMARY_FILE as IMAGE_SUMMARY_FILE

CANDIDATE_KEYS = {"candidate_id", "person_a_id", "person_b_id", "source_a", "source_b", "source_pair",
                  "source_record_a", "source_record_b", "rule_names", "rules"}
DEFERRED_KEYS = {"strategy", "member_count", "would_be_pairs", "would_be_pairs_after_policy", "reason", "limit"}
MANIFEST_KEYS = {"person_id", "media_id", "resolution_status", "usable_for_face_recognition", "person_rank", "path_ref"}
SUMMARY_KEYS = {"person_id", "person_status", "usable_image_count"}
TRANSIENT_RESOLUTION = {"fetch_error", "internal_error", "face_check_error"}


class InputError(RuntimeError):
    """An input is missing, incomplete, inconsistent or from an unsupported version."""


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputError(f"{what}: {path.name} not found in {path.parent} (the run did not finish or was never made)") from None
    except (OSError, ValueError) as exc:
        raise InputError(f"{what}: {path.name} is unreadable ({type(exc).__name__})") from None
    if not isinstance(data, dict):
        raise InputError(f"{what}: {path.name} is not a JSON object")
    return data


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                raise InputError(f"{path.name}: line {number} is not valid JSON") from None
            if not isinstance(record, dict):
                raise InputError(f"{path.name}: line {number} is not a JSON object")
            yield record


@dataclass
class CandidateInput:
    directory: Path
    summary: dict[str, Any]
    run_id: str
    candidates_path: Path
    deferred_path: Path
    candidate_count: int
    deferred_blocks: int
    dataset_stamp: dict[str, Any]


@dataclass
class ImageInput:
    directory: Path
    summary: dict[str, Any]
    index: ImageIndex
    partial: bool
    persons_with_unresolved_media: set[str] = field(default_factory=set)
    limitations: list[str] = field(default_factory=list)


def validate_candidates(directory: Path) -> CandidateInput:
    summary = _read_json(directory / SUMMARY_FILE, "Phase 5 candidates")
    state = _read_json(directory / STATE_FILE, "Phase 5 candidates")
    if summary.get("status") != "completed" or summary.get("complete") is not True:
        raise InputError(f"Phase 5 run is not complete (status={summary.get('status')!r})")
    if state.get("status") != "completed" or state.get("run_id") != summary.get("run_id"):
        raise InputError("Phase 5 run_state.json does not confirm the run in run_summary.json (a newer or failed run overwrote it)")
    version = str(summary.get("generator_version", ""))
    if version != CANDIDATE_VERSION:
        raise InputError(f"unsupported candidate generator version {version!r} (this scorer reads {CANDIDATE_VERSION!r})")
    if summary.get("database", {}).get("database") not in APPROVED_DATABASES:
        raise InputError("Phase 5 output was not produced from the approved database")
    candidates_path, deferred_path = directory / CANDIDATES_FILE, directory / DEFERRED_FILE
    for path in (candidates_path, deferred_path):
        if not path.is_file():
            raise InputError(f"{path.name} is missing")

    count, previous = 0, ("", "")
    for record in read_jsonl(candidates_path):
        missing = CANDIDATE_KEYS - set(record)
        if missing:
            raise InputError(f"{CANDIDATES_FILE}: record {count + 1} lacks {sorted(missing)}")
        a, b = record["person_a_id"], record["person_b_id"]
        if not (isinstance(a, str) and isinstance(b, str) and a < b):
            raise InputError(f"{CANDIDATES_FILE}: record {count + 1} is not in canonical order (person_a_id < person_b_id)")
        if record["candidate_id"] != f"{a}:{b}":
            raise InputError(f"{CANDIDATES_FILE}: record {count + 1} has an inconsistent candidate_id")
        if (a, b) <= previous:
            raise InputError(f"{CANDIDATES_FILE}: record {count + 1} is duplicated or out of order")
        previous, count = (a, b), count + 1
    expected = summary.get("union", {}).get("candidate_pairs")
    if count != expected:
        raise InputError(f"{CANDIDATES_FILE} has {count} candidates but run_summary.json says {expected}")

    deferred = 0
    for record in read_jsonl(deferred_path):
        missing = DEFERRED_KEYS - set(record)
        if missing:
            raise InputError(f"{DEFERRED_FILE}: record {deferred + 1} lacks {sorted(missing)}")
        deferred += 1
    if deferred != summary.get("deferred", {}).get("blocks"):
        raise InputError(f"{DEFERRED_FILE} has {deferred} blocks but run_summary.json says {summary.get('deferred', {}).get('blocks')}")
    return CandidateInput(directory, summary, summary["run_id"], candidates_path, deferred_path, count, deferred, summary.get("dataset_stamp", {}))


def validate_images(directory: Path, expected_persons: Optional[int]) -> ImageInput:
    summary = _read_json(directory / IMAGE_SUMMARY_FILE, "Phase 2 image resolution")
    if summary.get("database", {}).get("database") not in APPROVED_DATABASES:
        raise InputError("Phase 2 output was not produced from the approved database")
    unresolved_persons: set[str] = set()  # persons with media the image run could not settle (transient errors)
    usable: dict[str, list[dict[str, Any]]] = {}  # kept from this single parse, so the index below does not parse the files again
    person_rows: list[dict[str, Any]] = []
    for name, key, required in ((MANIFEST_FILE, "media_records", MANIFEST_KEYS), (PERSON_SUMMARY_FILE, "persons", SUMMARY_KEYS)):
        path = directory / name
        if not path.is_file():
            raise InputError(f"{name} is missing from {directory}")
        lines = 0
        for record in read_jsonl(path):
            missing = required - set(record)
            if missing:
                raise InputError(f"{name}: record {lines + 1} lacks {sorted(missing)}")
            if name == MANIFEST_FILE:
                if record["resolution_status"] in TRANSIENT_RESOLUTION:
                    unresolved_persons.add(record["person_id"])
                if record["usable_for_face_recognition"]:
                    usable.setdefault(record["person_id"], []).append(record)
            else:
                person_rows.append(record)
            lines += 1
        if lines != summary.get(key):
            raise InputError(f"{name} has {lines} records but run_summary.json says {summary.get(key)}")

    parameters = summary.get("parameters", {})
    scoped = [k for k in ("source_system", "limit_persons", "only_with_media") if parameters.get(k)]
    limitations: list[str] = []
    partial = bool(scoped)
    if partial:
        limitations.append(f"image_resolution_partial_run:{'+'.join(scoped)}")
    if expected_persons is not None and summary.get("persons") != expected_persons:
        partial = True
        limitations.append(f"image_resolution_person_count_mismatch:{summary.get('persons')}_of_{expected_persons}")
    if unresolved_persons:
        limitations.append(f"image_resolution_unresolved_media:{len(unresolved_persons)}_persons")
    return ImageInput(directory, summary, ImageIndex.from_parsed(usable, person_rows), partial, unresolved_persons, limitations)
