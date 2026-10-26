"""python -m dedup_pipeline.candidates --help"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..db import connect_read_only, load_connection_settings
from .config import SOURCE_POLICIES, CandidateConfig, final_candidate_config
from .runner import RunOptions, run_candidate_generation

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "candidates"
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parents[2] / "output" / "image_resolution"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dedup_pipeline.candidates",
        description="Generate candidate duplicate-person pairs (blocking only) from the unified database, read-only, into local files.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--env-file", help="file with UNIFIED_PG_* settings (or use DEDUP_PG_* variables)")
    p.add_argument("--config", type=Path, help="JSON file overriding any CandidateConfig field")
    p.add_argument("--source-policy", choices=SOURCE_POLICIES, help="which source-system pairs to keep (default: all)")
    p.add_argument("--name-max-block", type=int)
    p.add_argument("--phone-max-block", type=int)
    p.add_argument("--name-min-tokens", type=int)
    p.add_argument("--no-alias", action="store_true", help="disable the alias-matches-name strategy")
    p.add_argument("--profile", choices=["legacy", "final"], default="legacy",
                   help="final: metadata-cleaned names, relative-name and initial blocks, no alias. legacy is the historical default")
    p.add_argument("--export-keys", action="store_true", help="write normalised names/numbers to the output (sensitive)")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="completed Phase 2 output to link (default: output/image_resolution)")
    p.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    p.add_argument("--no-resume", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> CandidateConfig:
    values = final_candidate_config().to_dict() if getattr(args, "profile", "legacy") == "final" else {}
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    for key, flag in (("source_policy", args.source_policy), ("name_max_block_size", args.name_max_block),
                      ("phone_max_block_size", args.phone_max_block), ("name_min_tokens", args.name_min_tokens)):
        if flag is not None:
            values[key] = flag
    if args.no_alias:
        values["alias_enabled"] = False
    if args.export_keys:
        values["export_comparison_keys"] = True
    return CandidateConfig.from_dict(values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.output_dir / "run.log", encoding="utf-8")])
    config = config_from_args(args)
    options = RunOptions(output_dir=args.output_dir, image_dir=args.image_dir, resume=not args.no_resume, fresh=args.fresh)
    with connect_read_only(load_connection_settings(args.env_file), "dedup_candidate_generation") as conn:
        summary = run_candidate_generation(conn, config, options)
    print(f"status={summary['status']} candidate_pairs={summary['union']['candidate_pairs']} elapsed={summary['elapsed_seconds']}s")
    for name, s in summary["strategies"].items():
        print(f"  {name}: emitted={s['pairs_emitted']} unique={s['unique_pairs']} deferred_blocks={s['blocks_deferred']}")
    print("by source pair:", summary["union"]["by_source_pair"])
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
