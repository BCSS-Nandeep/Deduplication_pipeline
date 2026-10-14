"""Orchestration: read persons and media, resolve images, write local output files."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..db import ReadOnlyConnection, assert_read_only
from .file_server import Fetcher
from .manifest import MANIFEST_FILE, PERSON_SUMMARY_FILE, build_outputs, write_json_atomic, write_jsonl_atomic
from .queries import load_person_media, load_persons
from .resolver import Checkpoint, FaceChecker, ResolverConfig, resolve_all

log = logging.getLogger(__name__)

SHARED_PATHS_FILE = "shared_paths.jsonl"
RUN_SUMMARY_FILE = "run_summary.json"
CHECKPOINT_FILE = "checkpoint.jsonl"


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path
    source_system: Optional[str] = None
    limit_persons: Optional[int] = None
    only_with_media: bool = False
    workers: int = 8
    fresh: bool = False


def run_image_resolution(
    conn: ReadOnlyConnection,
    config: ResolverConfig,
    options: RunOptions,
    fetcher: Fetcher,
    face_check: Optional[FaceChecker] = None,
) -> dict[str, Any]:
    started = time.monotonic()
    db_info = assert_read_only(conn)
    log.info("Database %s confirmed read-only.", db_info["database"])
    options.output_dir.mkdir(parents=True, exist_ok=True)

    persons = load_persons(conn, options.source_system, options.limit_persons, options.only_with_media)
    rows = load_person_media(conn, [p.person_id for p in persons])
    log.info("Loaded %d persons and %d person-media rows.", len(persons), len(rows))

    checkpoint = Checkpoint(
        options.output_dir / CHECKPOINT_FILE, config.signature(face_check is not None), fresh=options.fresh
    )
    try:
        results = resolve_all(rows, config, fetcher, checkpoint, options.workers, face_check)
    finally:
        checkpoint.close(keep=(r.media_id for r in rows))

    manifest, summaries, shared = build_outputs(persons, rows, results, config.category_priority)
    out = options.output_dir
    write_jsonl_atomic(out / MANIFEST_FILE, manifest)
    write_jsonl_atomic(out / PERSON_SUMMARY_FILE, summaries)
    write_jsonl_atomic(out / SHARED_PATHS_FILE, shared)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "database": db_info,
        "parameters": {
            "source_system": options.source_system,
            "limit_persons": options.limit_persons,
            "only_with_media": options.only_with_media,
            "workers": options.workers,
            "fresh": options.fresh,
            **{k: v for k, v in config.signature(face_check is not None).items() if k != "base_url"},
        },
        "persons": len(persons),
        "media_records": len(manifest),
        "usable_images": sum(1 for m in manifest if m["usable_for_face_recognition"]),
        "person_status": dict(sorted(Counter(s["person_status"] for s in summaries).items())),
        "resolution_status": dict(sorted(Counter(m["resolution_status"] for m in manifest).items())),
        "detected_type": dict(sorted(Counter(m["detected_type"] or "none" for m in manifest).items())),
        "shared_path_count": len(shared),
        "shared_path_scope": "persons loaded in this run only",
        "face_check": "applied" if face_check is not None else "not_run (no checker supplied)",
        "outputs": {
            "manifest": MANIFEST_FILE,
            "person_summary": PERSON_SUMMARY_FILE,
            "shared_paths": SHARED_PATHS_FILE,
            "checkpoint": CHECKPOINT_FILE,
        },
    }
    write_json_atomic(out / RUN_SUMMARY_FILE, summary)
    return summary
