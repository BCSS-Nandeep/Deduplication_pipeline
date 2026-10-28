"""Fixtures for Phase 6 tests: a fake read-only database, real Phase 5 / Phase 2 outputs built from a small world, scripted face service."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PIL import Image

from dedup_pipeline.candidates import CandidateConfig, RunOptions as CandidateRunOptions, run_candidate_generation
from dedup_pipeline.db import check_statement_is_read_only
from dedup_pipeline.face_recognition import ComparisonOutcome, FrPairRejected, SideRejection
from dedup_pipeline.face_recognition.image_source import ManifestImageLoader
from dedup_pipeline.image_resolution.manifest import MANIFEST_FILE, PERSON_SUMMARY_FILE, build_outputs, write_json_atomic, write_jsonl_atomic
from dedup_pipeline.image_resolution.models import MediaResult, MediaRow, Person, Status
from dedup_pipeline.image_resolution.resolver import ResolverConfig
from dedup_pipeline.scoring.runner import FaceBackend

from .candidates_fakes import FakePersonDb, person as cand_person
from .conftest import BASE, FakeFetcher, url_for

from dedup_pipeline.scoring import ScoringConfig  # noqa: E402


def V1CONFIG(**kw) -> ScoringConfig:
    """A score-v1 configuration: the tests that pin the earlier behaviour use this explicitly."""
    return ScoringConfig(rule_version="score-v1", **kw)


def V2CONFIG(**kw) -> ScoringConfig:
    kw.setdefault("reference_date", "2026-10-12")
    return ScoringConfig(rule_version="score-v2", **kw)


def V3CONFIG(**kw) -> ScoringConfig:
    kw.setdefault("reference_date", "2026-10-12")
    return ScoringConfig(rule_version="score-v3", **kw)


STAMP = (0, "2026-10-01T00:00:00", 0, "2026-10-01T00:00:00")


def pdict(pid: str, source: str = "V2", name: str = "Ramesh Kumar", relative: Optional[str] = "Suresh Kumar", age: Optional[int] = 30,
          gender: Optional[str] = "Male", phone: Optional[str] = None, alias: Optional[str] = None, dob: Optional[date] = None,
          address: Optional[dict] = None, record: Optional[str] = None, fir: Optional[list[tuple]] = None) -> dict[str, Any]:
    """``fir`` = [(accused_id, crime_id, fir_date)] for score-v3 (what the accused_unified -> crimes_unified query returns)."""
    return {"person_id": pid, "source_system": source, "source_record_id": record or f"rec-{pid}", "name": name, "alias": alias,
            "relative": relative, "phone": phone, "age": age, "dob": dob, "gender": gender, "address": address, "fir": fir or []}


class FakeScoringDb:
    """Answers the preflight, dataset-stamp and attribute queries from an in-memory world; refuses anything but reads."""

    def __init__(self, people: list[dict[str, Any]], read_only: str = "on", database: str = "dopams_cctns_v2", stamp: Optional[tuple] = None):
        self.people = {p["person_id"]: p for p in people}
        self.read_only, self.database = read_only, database
        self.stamp = stamp or (len(people), STAMP[1], len(people), STAMP[3])
        self.statements: list[str] = []

    def cursor(self):
        return _Cursor(self)

    def close(self) -> None:
        pass


class _Cursor:
    def __init__(self, db: FakeScoringDb):
        self.db, self._rows = db, []

    def execute(self, sql: str, params: Any = None):
        check_statement_is_read_only(sql)
        self.db.statements.append(sql)
        if sql.startswith(("SHOW transaction_read_only", "SHOW default_transaction_read_only")):
            self._rows = [(self.db.read_only,)]
        elif "inet_server_addr()" in sql:
            self._rows = [("local",)]
        elif "current_database()" in sql:
            self._rows = [(self.db.database, "PostgreSQL 16.14 fake")]
        elif "candidates:preflight_person_ids" in sql:
            self._rows = [(0,)]
        elif "candidates:dataset_stamp" in sql:
            self._rows = [self.db.stamp]
        elif "scoring:attributes" in sql:
            rows = []
            for pid in params["ids"]:
                p = self.db.people.get(pid)
                if p:
                    rows.append((pid, p["source_system"], p["source_record_id"], p["name"], p["alias"], p["relative"], p["phone"], p["age"],
                                 p["dob"], p["gender"], p["address"]))
            self._rows = rows
        elif "scoring:fir_links" in sql:
            assert params["tz"], "the FIR timezone must be passed as a parameter"
            self._rows = [(pid, *row) for pid in params["ids"] if pid in self.db.people for row in self.db.people[pid].get("fir", [])]
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")
        return self

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows

    def close(self) -> None:
        pass


def image_bytes(label: int) -> bytes:
    """A distinct, valid PNG per label (different size), so the scripted face service can tell images apart by content."""
    buffer = io.BytesIO()
    Image.new("RGB", (64 + label, 64), (label % 256, 90, 160)).save(buffer, "PNG")
    return buffer.getvalue()


@dataclass
class World:
    root: Path
    people: list[dict[str, Any]]
    images: dict[str, list[tuple[str, int]]]  # person -> [(media_id, label)] in rank order
    candidate_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    label_of: dict[bytes, str] = field(default_factory=dict)  # image bytes -> media id
    files: dict[str, bytes] = field(default_factory=dict)  # url -> bytes (file server)

    def db(self, **kw) -> FakeScoringDb:
        return FakeScoringDb(self.people, **kw)


def build_world(root: Path, people: list[dict[str, Any]], images: Optional[dict[str, list[tuple]]] = None,
                candidate_config: Optional[CandidateConfig] = None, partial_images: bool = False,
                shared_paths: Optional[dict[str, str]] = None, unusable: Optional[dict[str, list[tuple[str, Status]]]] = None) -> World:
    """Run the real candidate generator and the real manifest builder over a small world, writing both output sets.

    ``images`` items are ``(media_id, label)`` or ``(media_id, label, attachment_category)``; ``unusable`` lists media the
    image run rejected (e.g. ``Status.CORRUPT_IMAGE``), which must never reach face comparison."""
    images = images or {}
    world = World(root, people, {p: [(i[0], i[1]) for i in items] for p, items in images.items()})
    cfg = candidate_config or CandidateConfig()
    results: dict[str, MediaResult] = {}
    cand_people = [cand_person(p["person_id"], p["source_system"], name=p["name"], phone=p["phone"], alias=p["alias"], record=p["source_record_id"])
                   for p in people]
    world.candidate_dir = root / "candidates"
    run_candidate_generation(FakePersonDb(cand_people, cfg, stamp=(len(people), STAMP[1], len(people), STAMP[3])), cfg,
                             CandidateRunOptions(output_dir=world.candidate_dir))

    rows = []
    for pid, items in (unusable or {}).items():
        for media_id, status in items:
            path = f"person/media/{media_id}"
            rows.append(MediaRow(media_id, pid, "person/MEDIA", "VERIFIED_ACCESSIBLE", resolved_relative_path=path))
            results[media_id] = MediaResult(media_id, status, "VERIFIED_ACCESSIBLE", path_ref=path, path_source="resolved_relative_path",
                                            skip_reason="rejected by the image run")
    for pid, items in images.items():
        for item in items:
            media_id, label = item[0], item[1]
            category = item[2] if len(item) > 2 else "person/MEDIA"
            path = (shared_paths or {}).get(media_id, f"person/media/{media_id}")
            rows.append(MediaRow(media_id, pid, category, "VERIFIED_ACCESSIBLE", resolved_relative_path=path))
            results[media_id] = MediaResult(
                media_id, Status.USABLE, "VERIFIED_ACCESSIBLE", path_ref=path, path_source="resolved_relative_path",
                resolved_suffix=".png", detected_type="png", width=64 + label, height=64, byte_size=100,
                media_class="CANDIDATE_PORTRAIT", media_class_confidence=0.9,
                media_class_reason="test fixture portrait",
            )
            data = image_bytes(label)
            world.files[url_for(path, ".png")] = data
            world.label_of[data] = media_id
    persons = [Person(p["person_id"], p["source_system"]) for p in people]
    manifest, summaries, shared = build_outputs(persons, rows, results, ("person/MEDIA", "person/IDENTITY_DETAILS"))
    world.image_dir = root / "images"
    world.image_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(world.image_dir / MANIFEST_FILE, manifest)
    write_jsonl_atomic(world.image_dir / PERSON_SUMMARY_FILE, summaries)
    write_json_atomic(world.image_dir / "run_summary.json", {
        "generated_at": "2026-10-12T00:00:00+00:00", "database": {"database": "dopams_cctns_v2"},
        "parameters": {"limit_persons": 5 if partial_images else None, "source_system": None, "only_with_media": False},
        "persons": len(summaries), "media_records": len(manifest), "usable_images": sum(1 for r in results.values() if r.usable),
        "media_classification": "applied",
    })
    return world


FACE_SIGNATURE = {"model": "buffalo_l", "max_images_per_person": 3, "contract": "compare-v1"}


def seed_known_image_failure(root: Path, media_id: str, kind: str = "no_face", partner: str = "seed-partner") -> None:
    """Pretend an earlier run already learned that the service refuses this image (as the stored comparison cache would say)."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    record = {"candidate_id": f"seed-{media_id}", "inputs": "seed", "signature": FACE_SIGNATURE, "selection": None, "evidence": {
        "person_a_id": "seed-a", "person_b_id": "seed-b", "status": kind,
        "attempts": [{"media_id_a": media_id, "media_id_b": partner, "outcome": "rejected", "rejections": {"a": kind}}]}}
    with open(root / "face_evidence_cache.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------------------------- face backend

class FakeComparer:
    """Scripted face service. ``rule(media_a, media_b)`` returns a ComparisonOutcome, or raises what the service would."""

    def __init__(self, world: World, rule: Callable[[str, str], Union[ComparisonOutcome, Exception]]):
        self.world, self.rule, self.calls = world, rule, []

    def compare(self, image_a: bytes, image_b: bytes) -> ComparisonOutcome:
        a, b = self.world.label_of[image_a], self.world.label_of[image_b]
        self.calls.append((a, b))
        result = self.rule(a, b)
        if isinstance(result, Exception):
            raise result
        return result


def outcome(cosine: float, model: str = "buffalo_l", decision: Optional[bool] = None, threshold: float = 0.45) -> ComparisonOutcome:
    return ComparisonOutcome(cosine, "cosine", cosine >= threshold if decision is None else decision, threshold, model)


def rejection(side: str, kind: str) -> FrPairRejected:
    return FrPairRejected((SideRejection(side, kind, None, "detail"),))


def backend(world: World, comparer, verify: Optional[Callable[[], None]] = None) -> FaceBackend:
    loader = ManifestImageLoader(ResolverConfig(base_url=BASE), FakeFetcher(dict(world.files)))
    return FaceBackend(comparer=comparer, loader=loader, verify=verify)
