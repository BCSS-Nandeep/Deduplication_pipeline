"""Phase 9, opt-in reuse of stored face-service answers: identical evidence, fewer service calls, never on by default."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from dedup_pipeline.face_recognition.errors import FrPairRejected, SideRejection
from dedup_pipeline.scoring import ScoringOptions, run_scoring
from dedup_pipeline.scoring.face import ImageOutcomeMemory, KnownRejections

from .scoring_fakes import FakeComparer, V2CONFIG, backend, build_world, outcome, pdict

NO_FACE_TEXT = "No face detected in the image."


def service(world):
    """A deterministic scripted service. Images f1..f4 have no face, b1/b2 are too blurred (per-image detail in the text), the rest are fine."""
    def sides(a, b):
        found = []
        for side, media in (("a", a), ("b", b)):
            if media.startswith("f"):
                found.append(SideRejection(side, "no_face", None, NO_FACE_TEXT))
            elif media.startswith("b"):
                found.append(SideRejection(side, "low_quality", "too_blurred", f"Face is too blurred ({media}). Use a sharper image."))
        return found

    def rule(a, b):
        rejected = sides(a, b)
        if rejected:
            return FrPairRejected(tuple(rejected))
        return outcome(0.5 + (hash_free(a) + hash_free(b)) / 100.0)
    return FakeComparer(world, rule)


def hash_free(media: str) -> int:
    return sum(ord(c) for c in media) % 17


def people_and_images():
    names = ["Ramesh Kumar"] * 9
    people = [pdict(f"p{i}", name=n, relative="Suresh Kumar", age=30, phone="9876543211") for i, n in enumerate(names)]
    images = {"p0": [("f1", 1)], "p1": [("f2", 2)], "p2": [("f3", 3)], "p3": [("f4", 4)], "p4": [("b1", 5)], "p5": [("b2", 6)],
              "p6": [("ok1", 7)], "p7": [("ok2", 8)], "p8": [("ok3", 9)]}
    return people, images


def run(world, root: Path, reuse: bool, label: str):
    comparer = service(world)
    be = backend(world, comparer, None)
    be.reuse_known_image_outcomes = reuse
    summary = run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, label, False), be)
    return summary, comparer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_cache(source: Path, target: Path, keep_every: int = 3) -> None:
    """Keep the stored answers for a third of the pairs: the images are then 'known', but the other pairs still have to be evaluated."""
    target.mkdir(parents=True, exist_ok=True)
    lines = (source / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines()
    kept = [l for i, l in enumerate(lines) if i % keep_every == 0]
    (target / "face_evidence_cache.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")


def test_reuse_gives_byte_identical_outputs_with_fewer_service_calls(tmp_path):
    people, images = people_and_images()
    world = build_world(tmp_path / "w", people, images)
    first, _ = run(world, tmp_path / "full", False, "full")  # the service answers everything once; this fills the stored answers
    partial_cache(tmp_path / "full", tmp_path / "off")
    shutil.copytree(tmp_path / "off", tmp_path / "on")
    off, calls_off = run(world, tmp_path / "off", False, "r")
    on, calls_on = run(world, tmp_path / "on", True, "r")
    assert first["status"] == off["status"] == on["status"] == "completed"
    # same evidence, same scores, same review outputs: the scored file and the worksheet are byte-identical, and so is every stored evidence record
    for name in ("scored_candidates.jsonl", "review_worksheet.csv"):
        assert sha(tmp_path / "off" / "runs" / "r" / name) == sha(tmp_path / "on" / "runs" / "r" / name), name  # (the first full run differs only in its
        # image-selection notes, which record what was already known when it started - existing behaviour, unrelated to the reuse)
    stored = lambda root: sorted(json.dumps(json.loads(l)["evidence"], sort_keys=True) for l in (root / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines())
    assert stored(tmp_path / "off") == stored(tmp_path / "on")
    # ... with fewer calls to the service
    assert len(calls_on.calls) < len(calls_off.calls)
    reused = on["face"]["reuse_known_image_outcomes"]
    assert reused["enabled"] is True and reused["attempts_answered_from_stored_results"] == len(calls_off.calls) - len(calls_on.calls) > 0
    assert "reuse_known_image_outcomes" not in off["face"]  # the default summary is unchanged
    # service calls that stay: pairs of two usable faces, and anything involving a low_quality image (its text carries per-image detail)
    assert all(not (a.startswith("f") and b.startswith("f")) for a, b in calls_on.calls)
    assert any(a.startswith("b") or b.startswith("b") for a, b in calls_on.calls)


def test_the_reuse_is_off_unless_asked_for(tmp_path):
    people, images = people_and_images()
    world = build_world(tmp_path / "w", people, images)
    run(world, tmp_path / "seed", False, "seed")
    partial_cache(tmp_path / "seed", tmp_path / "default")
    _, calls_default = run(world, tmp_path / "default", False, "r")
    be = backend(world, service(world), None)
    assert be.reuse_known_image_outcomes is False  # the dataclass default
    assert len(calls_default.calls) > 0


def test_known_rejections_only_answers_when_the_rejection_can_be_rebuilt_exactly():
    memory = ImageOutcomeMemory({"nf1": "no_face", "nf2": "no_face", "lq": "low_quality", "mf": "multiple_faces", "good": "ok"})
    memory.no_face_text = NO_FACE_TEXT
    known = KnownRejections(memory)
    both = known("nf1", "nf2")
    assert [(r.side, r.kind) for r in both.rejections] == [("a", "no_face"), ("b", "no_face")] and both.rejections[0].detail == NO_FACE_TEXT
    assert [(r.side, r.kind) for r in known("good", "nf1").rejections] == [("b", "no_face")]
    assert [(r.side, r.kind) for r in known("nf1", "good").rejections] == [("a", "no_face")]
    assert [(r.side, r.kind) for r in known("nf1", "lq").rejections] == [("a", "no_face"), ("b", "low_quality")]
    assert known("good", "good") is None  # a similarity is needed: the service must answer
    assert known("lq", "good") is None and known("mf", "nf1") is None  # first failing side has per-image detail that is not stored
    assert known("nf1", "never-seen") is None and known("never-seen", "nf1") is None  # an unknown image: the service reports every unusable side
    assert known.reused_attempts == 4


def test_the_no_face_wording_must_be_unique_in_the_stored_evidence_or_the_shortcut_stays_off(tmp_path):
    def record(cid, text):
        return {"candidate_id": cid, "inputs": "i", "signature": {"model": "buffalo_l", "max_images_per_person": 3, "contract": "compare-v1"},
                "evidence": {"status": "no_face", "reason": text, "attempts": [{"media_id_a": f"{cid}a", "media_id_b": f"{cid}b", "outcome": "rejected", "rejections": {"a": "no_face"}}]}}

    same = tmp_path / "same.jsonl"
    same.write_text("\n".join(json.dumps(record(f"c{i}", NO_FACE_TEXT)) for i in range(3)) + "\n", encoding="utf-8")
    assert ImageOutcomeMemory.from_cache_file(same, "buffalo_l").no_face_text == NO_FACE_TEXT
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(record("c1", NO_FACE_TEXT)) + "\n" + json.dumps(record("c2", "No face (v2 wording).")) + "\n", encoding="utf-8")
    memory = ImageOutcomeMemory.from_cache_file(mixed, "buffalo_l")
    assert memory.no_face_text is None and KnownRejections(memory)("c1a", "c1b") is None and not KnownRejections(memory).usable
    assert ImageOutcomeMemory.from_cache_file(tmp_path / "missing.jsonl", "buffalo_l").no_face_text is None
    other_model = ImageOutcomeMemory.from_cache_file(same, "another_model")
    assert other_model.no_face_text is None and len(other_model) == 0  # answers from a different model are never reused
