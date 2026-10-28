"""python -m dedup_pipeline.scoring run | validate-worksheet"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from ..db import connect_read_only, load_connection_settings
from ..face_recognition.image_source import ManifestImageLoader
from ..face_recognition.local_comparer import LocalFacePairComparer
from ..face_recognition.local_config import LocalFaceConfig, apply_gpu_choice, gpu_flag_from_sources
from ..image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher
from ..image_resolution.resolver import ResolverConfig
from .config import ScoringConfig
from .runner import CountingComparer, CountingFetcher, FaceBackend, ScoringOptions, run_scoring
from .worksheet import WorksheetError, validate_worksheet, write_validated

ROOT = Path(__file__).resolve().parents[2]


def add_face_engine_arguments(parser: argparse.ArgumentParser) -> None:
    # The face engine is the in-process comparer (validated on CPU; needs the dedicated .face-venv, see requirements-face-local.txt). It is the only
    # engine: there is no HTTP/API engine and no fallback of any kind.
    parser.add_argument("--local-face-config", type=Path, help="JSON file overriding any LocalFaceConfig field")
    parser.add_argument("--face-provider", choices=["cpu", "auto", "cuda"],
                        help="onnxruntime provider. Overrides the GPU setting in .env. cpu is the validated path; "
                             "auto/cuda are refused unless GPU=true or --accept-unvalidated-provider is given")
    parser.add_argument("--accept-unvalidated-provider", action="store_true",
                        help="allow a provider whose results have not been replayed against the 525 stored cases (GPU); recorded in the run configuration")


def gpu_env_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    env_file = getattr(args, "env_file", None)
    if env_file:
        files.append(Path(env_file))
    files.append(ROOT / ".env")
    return files


def local_comparer_from_args(args: argparse.Namespace) -> LocalFacePairComparer:
    local_config = LocalFaceConfig.from_file(args.local_face_config) if args.local_face_config else LocalFaceConfig()
    if args.face_provider:
        local_config = replace(local_config, provider=args.face_provider)
        if args.face_provider == "cpu":
            local_config = replace(local_config, accept_unvalidated_provider=False)
    else:
        local_config = apply_gpu_choice(local_config, gpu_flag_from_sources(os.environ, gpu_env_files(args)))
    if args.accept_unvalidated_provider:
        local_config = replace(local_config, accept_unvalidated_provider=True)
    return LocalFacePairComparer(local_config)


def preflight_local_engine(args: argparse.Namespace) -> tuple[LocalFacePairComparer | None, int]:
    """Check the local engine BEFORE anything else happens (no output directory, no database connection). On any problem: a clear message and exit
    code 2. There is no other engine to fall back to."""
    try:
        comparer = local_comparer_from_args(args)
        comparer.preflight()  # packages, model files, provider: every problem listed at once
        comparer.verify()  # the model really loads (once per process; the run reuses it)
    except Exception as exc:
        detail = exc.detail if hasattr(exc, "detail") and exc.detail else f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"FACE ENGINE PREFLIGHT FAILED (local): {detail}\n"
              "Nothing was run. Fix the environment (setup: README 'Face engine setup').", file=sys.stderr)
        return None, 2
    return comparer, 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dedup_pipeline.scoring",
                                description="Score candidate pairs and prepare the human-review package. Read-only; never decides.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="score candidates and write the reviewer package")
    r.add_argument("--candidates-dir", type=Path, default=ROOT / "output" / "candidates")
    r.add_argument("--image-dir", type=Path, default=ROOT / "output" / "image_resolution")
    r.add_argument("--output-root", type=Path, default=ROOT / "output" / "scoring")
    r.add_argument("--run-label", help="name of the run directory (default: UTC timestamp)")
    r.add_argument("--env-file", help="file with UNIFIED_PG_* settings (or use DEDUP_PG_* variables)")
    r.add_argument("--config", type=Path, help="JSON file overriding any ScoringConfig field")
    r.add_argument("--no-face-service", action="store_true", help="do not run face comparison (face comparisons stay pending; run is incomplete)")
    r.add_argument("--rule-version", choices=["score-v1", "score-v2", "score-v3", "score-final"],
                   help="default score-v2 (historical). score-final is the authoritative 25/25/20/15/10/5 profile and does not use age, DOB or alias")
    r.add_argument("--reference-date", help="ISO date at which every age is evaluated (score-v2) or the validity cut-off (score-v3). If omitted, today's "
                                              "date is used ONCE, logged and recorded; pass it explicitly for reproducible reruns")
    r.add_argument("--review-order", choices=["score", "conflict_free_first"])
    r.add_argument("--image-ranking", choices=["phase2_rank", "outcome_memory"])
    r.add_argument("--face-concurrency", type=int)
    add_face_engine_arguments(r)
    r.add_argument("--reuse-known-image-outcomes", action="store_true",
                   help="Phase 9, opt-in: skip face-service calls whose rejection stored results already determine exactly (default off)")
    r.add_argument("--allow-stale-candidates", action="store_true")
    r.add_argument("--no-resume", action="store_true")
    f = sub.add_parser("face-preflight", help="check that the local face engine can run here (packages, model files, provider, model load) and print its identity")
    add_face_engine_arguments(f)
    w = sub.add_parser("validate-worksheet", help="check a completed review worksheet and keep its decisions locally")
    w.add_argument("worksheet", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-worksheet":
        try:
            result = validate_worksheet(args.worksheet)
        except WorksheetError as exc:
            print(f"INVALID: {exc}", file=sys.stderr)
            return 2
        out = write_validated(args.worksheet, result)
        print(f"rows={result['rows']} decisions={len(result['decisions'])} pending={result['pending']} problems={len(result['problems'])}")
        for problem in result["problems"][:50]:
            print("  PROBLEM:", problem)
        print(f"validated decisions kept locally in {out} (nothing is written to any database)")
        return 1 if result["problems"] else 0

    if args.command == "face-preflight":
        comparer, code = preflight_local_engine(args)
        if comparer is not None:
            print(json.dumps(comparer.describe(), indent=2))
        return code

    local: LocalFacePairComparer | None = None
    if not args.no_face_service:
        local, code = preflight_local_engine(args)
        if local is None:
            return code
    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.output_root / "run.log", encoding="utf-8")])
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for key, flag in (("rule_version", args.rule_version), ("reference_date", args.reference_date),
                      ("review_order", args.review_order), ("image_ranking", args.image_ranking)):
        if flag:
            values[key] = flag
    if values.get("rule_version", "score-v2") in ("score-v2", "score-v3") and not values.get("reference_date"):
        values["reference_date"] = date.today().isoformat()  # evaluated once here; the whole run uses this one value
        logging.getLogger(__name__).warning("No --reference-date given: using today's date %s for every age in this run (recorded in the output).",
                                            values["reference_date"])
    if args.face_concurrency:
        values["face_concurrency"] = args.face_concurrency
    if args.allow_stale_candidates:
        values["allow_stale_candidates"] = True
    config = ScoringConfig.from_dict(values)

    base_url = config.file_server_base_url or os.environ.get("FILE_SERVER_BASE_URL") or DEFAULT_FILE_SERVER_BASE_URL
    fetcher = CountingFetcher(HttpFetcher())
    loader = ManifestImageLoader(ResolverConfig(base_url=base_url), fetcher)
    backend = FaceBackend(comparer=None, loader=loader, fetch_counter=fetcher, reuse_known_image_outcomes=args.reuse_known_image_outcomes)
    if local is not None:
        counted = CountingComparer(local)
        backend = FaceBackend(counted, loader, verify=local.verify, comparer_counter=counted, fetch_counter=fetcher,
                              reuse_known_image_outcomes=args.reuse_known_image_outcomes, signature_fields=local.signature_fields, engine_info=local.describe)
    options = ScoringOptions(args.candidates_dir, args.image_dir, args.output_root, args.run_label, resume=not args.no_resume)
    with connect_read_only(load_connection_settings(args.env_file), "dedup_scoring") as conn:
        summary = run_scoring(conn, config, options, backend)
    print(f"status={summary['status']} scored={summary['candidates']['scored']} face_pairs={summary['face']['eligible_pairs_both_usable_images']} "
          f"retryable={summary['face']['retryable_errors']} elapsed={summary['elapsed_seconds']}s")
    print("score bands (prioritisation only):", summary["candidates"]["score_bands_prioritisation_only"])
    print("run directory:", args.output_root / "runs" / summary["run_id"])
    return 0 if summary["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
