"""Phase 11: the face-engine optimisations (skip the library's unused recognition pass; per-run answer cache) change no result.

The logic is tested without any machine-learning package by giving a bare engine object scripted internals; the real engine is compared with the
unoptimised path in tests/test_local_comparer_ml.py (run in .face-venv).
"""

from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from dedup_pipeline.face_recognition import local_comparer as lc
from dedup_pipeline.face_recognition.image_source import ImageBytes, ManifestImageLoader
from dedup_pipeline.face_recognition.local_comparer import ImageRejected, LocalFaceEngine
from dedup_pipeline.face_recognition.local_config import LocalFaceConfig
from dedup_pipeline.image_resolution.resolver import ResolverConfig

from .conftest import url_for


class FakeArray(list):
    """Stands in for a numpy embedding: has the one method the cache calls."""

    def setflags(self, write=True):
        self.write = write


def bare_engine(script, config=None):
    """A LocalFaceEngine without the model: ``script(data)`` returns an embedding or raises, in place of the real analysis."""
    engine = LocalFaceEngine.__new__(LocalFaceEngine)
    engine.config = config or LocalFaceConfig()
    engine._analysis, engine._analysis_lock, engine._in_flight = {}, threading.Lock(), {}
    engine.analysis_hits = engine.analysis_misses = 0
    engine.calls = []

    def uncached(data):
        engine.calls.append(bytes(data))
        return script(bytes(data))

    engine._analyse_uncached = uncached
    return engine


def labelled(data: bytes, key: str) -> ImageBytes:
    out = ImageBytes(data)
    out.cache_key = key
    return out


# ------------------------------------------------------------------------------------------------------ the answer cache

def test_a_labelled_image_is_analysed_once_and_the_same_embedding_is_returned():
    engine = bare_engine(lambda d: FakeArray([1.0, 2.0]))
    first = engine.analyse(labelled(b"img", "m1@p/a.jpg"))
    second = engine.analyse(labelled(b"img", "m1@p/a.jpg"))
    assert first is second and engine.calls == [b"img"] and first.write is False  # read-only: a cached embedding cannot be altered by a caller
    assert (engine.analysis_hits, engine.analysis_misses) == (1, 1)


def test_a_rejection_is_cached_with_its_exact_code_text_and_reason():
    def script(data):
        raise ImageRejected("low_quality", "Face is too small (30px; minimum 40px). Use a closer or higher-resolution image.", "too_small")

    engine = bare_engine(script)
    seen = []
    for _ in range(3):
        with pytest.raises(ImageRejected) as exc:
            engine.analyse(labelled(b"img", "m1@p/a.jpg"))
        seen.append((exc.value.code, str(exc.value), exc.value.reason))
    assert len(engine.calls) == 1 and len(set(seen)) == 1 == len({seen[0]}) and seen[0] == (
        "low_quality", "Face is too small (30px; minimum 40px). Use a closer or higher-resolution image.", "too_small")


def test_unlabelled_bytes_and_different_files_are_never_confused():
    engine = bare_engine(lambda d: FakeArray([float(len(d))]))
    engine.analyse(b"plain")
    engine.analyse(b"plain")
    assert len(engine.calls) == 2 and not engine._analysis  # plain bytes: always analysed, never remembered
    a = engine.analyse(labelled(b"aaaa", "m1@p/a.jpg"))
    b = engine.analyse(labelled(b"bbbb", "m2@p/b.jpg"))
    assert a is not b and len(engine._analysis) == 2
    changed = engine.analyse(labelled(b"aaaaaa", "m1@p/a.jpg"))  # same label but a different length: recomputed, not served from the cache
    assert changed[0] == 6.0 and len(engine.calls) == 5


def test_an_operational_failure_is_never_remembered():
    state = {"fail": True}

    def script(data):
        if state["fail"]:
            raise RuntimeError("onnxruntime crashed")
        return FakeArray([1.0])

    engine = bare_engine(script)
    with pytest.raises(RuntimeError):
        engine.analyse(labelled(b"img", "m1@x.jpg"))
    assert not engine._analysis and engine.analysis_misses == 0
    state["fail"] = False
    assert engine.analyse(labelled(b"img", "m1@x.jpg")) == FakeArray([1.0]) and len(engine.calls) == 2  # recomputed after the failure


def test_the_cache_is_bounded_and_stops_growing_without_evicting(monkeypatch):
    monkeypatch.setattr(lc, "ANALYSIS_CACHE_MAX", 3)
    engine = bare_engine(lambda d: FakeArray([float(len(d))]))
    for i in range(6):
        engine.analyse(labelled(b"x" * (i + 1), f"m{i}@x.jpg"))
    assert len(engine._analysis) == 3 and engine.analysis_cache_stats() == {"entries": 3, "hits": 0, "misses": 6, "max_entries": 3}
    engine.analyse(labelled(b"x", "m0@x.jpg"))  # an entry kept earlier is still served
    assert engine.analysis_hits == 1
    engine.analyse(labelled(b"x" * 6, "m5@x.jpg"))  # one that did not fit is recomputed (correct, just not faster)
    assert engine.analysis_hits == 1 and len(engine.calls) == 7


def test_concurrent_threads_get_the_same_answer_for_the_same_image():
    engine = bare_engine(lambda d: FakeArray([3.0, 4.0]))

    def one(_):
        return tuple(engine.analyse(labelled(b"img", "m1@x.jpg")))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(one, range(200)))
    assert set(results) == {(3.0, 4.0)} and len(engine._analysis) == 1 and engine.analysis_hits + engine.analysis_misses == 200


def test_simultaneous_requests_for_the_same_image_compute_it_once():
    import time

    def slow(data):
        time.sleep(0.15)
        return FakeArray([7.0])

    engine = bare_engine(slow)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: engine.analyse(labelled(b"img", "m1@x.jpg")), range(16)))
    assert len(engine.calls) == 1 and all(r is results[0] for r in results) and engine._in_flight == {}
    assert engine.analysis_cache_stats()["hits"] == 15 and engine.analysis_cache_stats()["misses"] == 1


def test_a_rejection_reaches_every_simultaneous_caller_with_the_same_wording():
    import time

    def slow(data):
        time.sleep(0.1)
        raise ImageRejected("no_face", "No face detected in the image.")

    engine = bare_engine(slow)

    def one(_):
        try:
            engine.analyse(labelled(b"img", "m1@x.jpg"))
        except ImageRejected as exc:
            return (exc.code, str(exc), exc.reason)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(8)))
    assert set(results) == {("no_face", "No face detected in the image.", None)} and len(engine.calls) == 1


def test_a_failing_computation_releases_the_waiting_threads_and_is_retried_not_remembered():
    import time

    state = {"fail": True}

    def flaky(data):
        time.sleep(0.1)
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("onnxruntime crashed")
        return FakeArray([1.0])

    engine = bare_engine(flaky)
    outcomes = []

    def one(_):
        try:
            outcomes.append(("ok", tuple(engine.analyse(labelled(b"img", "m1@x.jpg")))))
        except RuntimeError:
            outcomes.append(("failed",))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, range(6)))  # no thread may hang
    assert outcomes.count(("failed",)) == 1 and outcomes.count(("ok", (1.0,))) == 5  # the failure hit one caller only; the others computed the answer
    assert engine._in_flight == {} and len(engine._analysis) == 1


# ------------------------------------------------------------------------------------------- the loader labels, content untouched

def test_the_loader_labels_images_with_their_manifest_identity_and_keeps_the_bytes(png_bytes):
    from .conftest import BASE, FakeFetcher

    body = png_bytes
    record = {"media_id": "V2:m1", "path_ref": "person/media/abc", "resolved_suffix": ".png"}
    loaded = ManifestImageLoader(ResolverConfig(base_url=BASE), FakeFetcher({url_for("person/media/abc", ".png"): body})).load(record)
    assert isinstance(loaded, ImageBytes) and loaded == body and bytes(loaded) == body and loaded.cache_key == "V2:m1@person/media/abc.png"
    assert {body: 1}[loaded] == 1 and hash(loaded) == hash(body)  # interchangeable with plain bytes everywhere


# ------------------------------------------------------------------- concurrency changes speed, never a result

def test_the_default_concurrency_is_eight_and_never_part_of_the_cache_signature():
    from dedup_pipeline.scoring import ScoringConfig

    assert ScoringConfig(reference_date="2026-10-12").face_concurrency == 8
    a, b = ScoringConfig(reference_date="2026-10-12", face_concurrency=1), ScoringConfig(reference_date="2026-10-12", face_concurrency=16)
    assert a.face_signature() == b.face_signature()  # stored face results stay valid whatever concurrency produced them


def test_results_outputs_and_stored_evidence_do_not_depend_on_how_many_pairs_run_at_once(tmp_path):
    import hashlib
    import json

    from dedup_pipeline.scoring import ScoringOptions, run_scoring

    from .scoring_fakes import V2CONFIG, build_world
    from .test_local_comparer import FakeEngine, local_backend, people_and_images, rule

    people, images = people_and_images()
    people += [dict(p, person_id=f"x{i}") for i, p in enumerate(people)]  # more pairs than workers
    world = build_world(tmp_path / "w", people, {**images, **{f"x{i}": v for i, (k, v) in enumerate(images.items())}})
    digests, evidence = {}, {}
    for concurrency in (1, 3, 8):
        engine = FakeEngine(world, rule)
        be, _ = local_backend(world, engine)
        root = tmp_path / f"c{concurrency}"
        summary = run_scoring(world.db(), V2CONFIG(face_concurrency=concurrency), ScoringOptions(world.candidate_dir, world.image_dir, root, "r", False), be)
        run = root / "runs" / "r"
        assert summary["status"] == "completed" and summary["candidates"]["awaiting_human_review"] == summary["candidates"]["scored"]
        digests[concurrency] = {n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in ("scored_candidates.jsonl", "review_worksheet.csv")}
        evidence[concurrency] = sorted(json.dumps({k: v for k, v in json.loads(line).items() if k != "selection"}, sort_keys=True)
                                       for line in (root / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines())
    assert digests[1] == digests[3] == digests[8] and evidence[1] == evidence[3] == evidence[8]


# ----------------------------------------------------------------- skipping the library's unused recognition pass (logic)

class FakeFace:
    def __init__(self, bbox, kps, det_score):
        self.bbox, self.kps, self.det_score = bbox, kps, det_score


class FakeNp:
    """Minimal array stand-in: only shape[0] and 2-D indexing like the detector's output are needed."""


def engine_with_detector(kpss_present=True, flip=True, pose=False):
    engine = bare_engine(lambda d: None, LocalFaceConfig(flip_average=flip, load_pose_model=pose))
    engine._Face = FakeFace
    log = {"detect": 0, "get": 0, "rec": 0}

    class Boxes:
        shape = (2, 5)

        def __getitem__(self, idx):
            row, cols = idx
            values = [[10, 20, 110, 140, 0.99], [200, 50, 240, 90, 0.6]][row]
            return values[cols] if isinstance(cols, slice) else values[cols]

    def detect(image, max_num=0, metric="default"):
        log["detect"] += 1
        return Boxes(), ([[1, 1], [2, 2]] if kpss_present else None)

    engine.app = types.SimpleNamespace(det_model=types.SimpleNamespace(detect=detect), get=lambda image: (log.__setitem__("get", log["get"] + 1) or ["via-get"]),
                                       models={"recognition": types.SimpleNamespace(get=lambda *a: log.__setitem__("rec", log["rec"] + 1))})
    return engine, log


def test_with_landmarks_and_flip_averaging_the_detector_is_called_directly_and_the_recognition_pass_is_skipped():
    engine, log = engine_with_detector()
    faces = engine._detect(object())
    assert [type(f) for f in faces] == [FakeFace, FakeFace] and log == {"detect": 1, "get": 0, "rec": 0}
    assert [f.det_score for f in faces] == [0.99, 0.6] and [f.kps for f in faces] == [[1, 1], [2, 2]] and faces[0].bbox == [10, 20, 110, 140]  # detector order and values kept


@pytest.mark.parametrize("kwargs", [{"kpss_present": False}, {"flip": False}, {"pose": True}])
def test_in_every_other_case_the_library_call_is_used_unchanged(kwargs):
    engine, log = engine_with_detector(**kwargs)
    assert engine._detect(object()) == ["via-get"] and log["get"] == 1
