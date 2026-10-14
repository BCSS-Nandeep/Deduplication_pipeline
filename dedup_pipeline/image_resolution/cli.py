"""Command line entry point:  python -m dedup_pipeline.image_resolution --help"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..db import connect_read_only, load_connection_settings
from .file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher
from .resolver import ResolverConfig
from .runner import RunOptions, run_image_resolution

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "image_resolution"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dedup_pipeline.image_resolution",
        description="Resolve person images from the unified database (read-only) and write a local manifest.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--env-file", help="file with UNIFIED_PG_* settings (or use DEDUP_PG_* variables)")
    p.add_argument("--file-server-base-url", default=os.environ.get("FILE_SERVER_BASE_URL") or DEFAULT_FILE_SERVER_BASE_URL)
    p.add_argument("--source-system", choices=["V1", "V2"])
    p.add_argument("--limit-persons", type=int, help="process only the first N persons (ordered by person_id)")
    p.add_argument("--only-with-media", action="store_true", help="restrict persons to those with person-media rows")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe-unverified", action="store_true", help="also try media not marked VERIFIED_ACCESSIBLE")
    p.add_argument("--fresh", action="store_true", help="ignore the checkpoint and re-resolve everything")
    p.add_argument("--timeout", type=float, default=15.0, help="per-request timeout in seconds")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.output_dir / "run.log", encoding="utf-8")],
    )
    config = ResolverConfig(base_url=args.file_server_base_url, probe_unverified=args.probe_unverified)
    options = RunOptions(
        output_dir=args.output_dir,
        source_system=args.source_system,
        limit_persons=args.limit_persons,
        only_with_media=args.only_with_media,
        workers=args.workers,
        fresh=args.fresh,
    )
    with connect_read_only(load_connection_settings(args.env_file), "dedup_image_resolution") as conn:
        summary = run_image_resolution(conn, config, options, HttpFetcher(timeout=args.timeout))
    print(f"persons={summary['persons']} media_records={summary['media_records']} usable_images={summary['usable_images']}")
    print("person_status:", summary["person_status"])
    print("resolution_status:", summary["resolution_status"])
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
