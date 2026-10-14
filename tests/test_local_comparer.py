"""Phase 10: the in-process face comparer - contract, configuration, fingerprint, cache separation, failure handling.

These tests need no machine-learning package: the engine is scripted. The real engine is exercised by test_local_comparer_ml.py (run in an
environment that has requirements-face-local.txt) and by tools/phase10_replay.py against the 525 stored real-service results.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from dedup_pipeline.face_recognition import FrOperationalError, FrPairRejected, LocalFaceConfig, LocalFacePairComparer
from dedup_pipeline.face_recognition.local_comparer import DETAIL_LIMIT, EngineUnavailable, ImageRejected
from dedup_pipeline.face_recognition.local_config import ENGINE_NAME, fingerprint
from dedup_pipeline.face_recognition.pair_evidence import ComparisonOutcome
from dedup_pipeline.scoring import ScoringOptions, run_scoring
from dedup_pipeline.scoring.face import ImageOutcomeMemory

from .scoring_fakes import V2CONFIG, backend, build_world, pdict

FR_CONFIG = Path(r"C:\Users\Dell\Desktop\FR-DOPAMS\src\config.py")


# ------------------------------------------------------------------------------------------------------ a scripted engine

def unit(angle: float):
    return (math.cos(angle), math.sin(angle))


class FakeEngine:
    """Same surface as LocalFaceEngine: ``analyse(bytes)``, ``cosine(a, b)`` and the identity attributes. ``rule(media)`` returns an angle
    (a usable face) or an exception to raise; the image bytes map back to media ids through the test world."""

    def __init__(self, world, rule, fingerprint_value: str = "f" * 64, config: LocalFaceConfig | None = None):
        self.world, self.rule, self.fingerprint = world, rule, fingerprint_value
        self.config = config or LocalFaceConfig()
        self.provider_used, self.providers = "CPUExecutionProvider", ["CPUExecutionProvider"]
        self.model_hashes, self.libraries = {"det_10g.onnx": "d" * 64, "w600k_r50.onnx": "w" * 64}, {"insightface": "2.0"}
        self.calls: list[str] = []

    def analyse(self, data: bytes):
        media = self.world.label_of[data]
        self.calls.append(media)
        result = self.rule(media)
        if isinstance(result, Exception):
            raise result
        return unit(result)

    def cosine(self, a, b) -> float:
        return max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))


def local_backend(world, engine, **kw):
    comparer = LocalFacePairComparer(engine.config, engine=engine)
    be = backend(world, comparer, comparer.verify)
    be.signature_fields, be.engine_info = comparer.signature_fields, comparer.describe
    return be, comparer


def people_and_images():
    people = [pdict(f"p{i}", name="Ramesh Kumar", relative="Suresh Kumar", age=30, phone="9876543211") for i in range(6)]
    images = {"p0": [("a1", 1)], "p1": [("a2", 2)], "p2": [("nf", 3)], "p3": [("lq", 4)], "p4": [("a3", 5)], "p5": [("nf2", 6)]}
    return people, images


def rule(media: str):
    if media.startswith("nf"):
        return ImageRejected("no_face", "No face detected in the image.")
    if media == "lq":
        return ImageRejected("low_quality", "Face is too small (30px; minimum 40px). Use a closer or higher-resolution image.", "too_small")
    return {"a1": 0.0, "a2": 0.1, "a3": 1.2}[media]


def run(world, root, engine, label="r", resume=False):
    be, comparer = local_backend(world, engine)
    return run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, label, resume), be), comparer


# ------------------------------------------------------------------------------------------------------------ configuration

def test_defaults_equal_the_constants_the_fr_dopams_service_used():
    if not FR_CONFIG.is_file():
        pytest.skip("FR-DOPAMS source not available")
    text = FR_CONFIG.read_text(encoding="utf-8")
    default = lambda name: re.search(rf'getenv\("{name}",\s*([^)]+)\)', text).group(1).strip()
    c = LocalFaceConfig()
    assert float(default("FACE_MIN_CONF")) == c.min_detection_confidence and int(default("MIN_FACE_SIZE")) == c.min_face_size_px
    assert float(default("MIN_FACE_SHARPNESS")) == c.min_face_sharpness and float(default("MULTI_FACE_AREA_RATIO")) == c.multi_face_area_ratio
    assert float(default("MAX_UPLOAD_MB")) == c.max_image_mb and int(default("COMPARE_MAX_PIXELS").replace("_", "")) == c.max_pixels
    assert float(default("PERSON_MATCH_THRESHOLD")) == c.same_person_threshold and default("FACE_DET_SIZE").strip('"') == "640,640" and c.det_size == (640, 640)
    assert c.model_name == "buffalo_l" and c.embedding_crop_px == 112 and c.sharpness_crop_px == 112 and c.flip_average is True


def test_configuration_is_validated_serialisable_and_rejects_unknown_keys():
    c = LocalFaceConfig(provider="cpu", same_person_threshold=0.5)
    assert LocalFaceConfig.from_dict(c.to_dict()) == c and json.loads(json.dumps(c.to_dict()))["det_size"] == [640, 640]
    for bad in ({"provider": "tpu"}, {"det_size": (10, 640)}, {"min_detection_confidence": 1.5}, {"same_person_threshold": 2}, {"min_face_size_px": 0},
                {"multi_face_area_ratio": 0}, {"max_image_mb": 0}, {"max_pixels": -1}):
        with pytest.raises(ValueError):
            LocalFaceConfig(**bad)
    with pytest.raises(ValueError):
        LocalFaceConfig.from_dict({"threshold": 0.4})


def test_the_fingerprint_changes_with_every_setting_model_file_library_and_provider_but_not_with_the_model_location():
    base = LocalFaceConfig()
    models, libs = {"det": "1", "rec": "2"}, {"insightface": "2.0", "onnxruntime": "1.23.2"}
    ref = fingerprint(base, models, libs, "CPUExecutionProvider")
    assert ref == fingerprint(base, dict(reversed(list(models.items()))), libs, "CPUExecutionProvider")  # order of mappings is irrelevant
    assert ref == fingerprint(LocalFaceConfig(model_root="D:/elsewhere"), models, libs, "CPUExecutionProvider")  # where the files are is irrelevant
    variants = {"det_size": (320, 320), "min_detection_confidence": 0.6, "min_face_size_px": 50, "min_face_sharpness": 20.0, "sharpness_crop_px": 96,
                "multi_face_area_ratio": 0.6, "max_image_mb": 5.0, "max_pixels": 1_000_000, "same_person_threshold": 0.5, "embedding_crop_px": 128,
                "flip_average": False, "load_pose_model": True, "model_name": "buffalo_s", "device_id": 1}
    seen = {ref}
    for key, value in variants.items():
        f = fingerprint(LocalFaceConfig(**{key: value}), models, libs, "CPUExecutionProvider")
        assert f not in seen, key
        seen.add(f)
    assert fingerprint(base, {**models, "rec": "3"}, libs, "CPUExecutionProvider") not in seen  # a different model file
    assert fingerprint(base, models, {**libs, "onnxruntime": "1.24.0"}, "CPUExecutionProvider") not in seen  # a different library version
    assert fingerprint(base, models, libs, "CUDAExecutionProvider") not in seen  # a different provider (different numerics)
    assert fingerprint(LocalFaceConfig(provider="cuda"), models, libs, "CPUExecutionProvider") == ref  # the provider actually USED counts, not the one asked for


# --------------------------------------------------------------------------------------------------- the PairComparer contract

@pytest.fixture
def world(tmp_path):
    people, images = people_and_images()
    return build_world(tmp_path / "w", people, images)


def test_a_comparison_returns_the_same_outcome_shape_as_the_api_comparer(world):
    engine = FakeEngine(world, rule)
    comparer = LocalFacePairComparer(engine.config, engine=engine)
    bytes_of = {media: data for data, media in world.label_of.items()}
    out = comparer.compare(bytes_of["a1"], bytes_of["a2"])
    assert isinstance(out, ComparisonOutcome) and out.metric == "cosine" and out.model == "buffalo_l" and out.threshold == 0.45
    assert out.similarity == pytest.approx(math.cos(0.1)) and out.decision is True
    far = comparer.compare(bytes_of["a1"], bytes_of["a3"])
    assert far.similarity == pytest.approx(math.cos(1.2)) and far.decision is False
    assert -1.0 <= far.similarity <= 1.0


def test_the_decision_is_taken_at_the_threshold_inclusive(world):
    engine = FakeEngine(world, rule, config=LocalFaceConfig(same_person_threshold=math.cos(0.1)))
    bytes_of = {media: data for data, media in world.label_of.items()}
    out = LocalFacePairComparer(engine.config, engine=engine).compare(bytes_of["a1"], bytes_of["a2"])
    assert out.similarity == out.threshold and out.decision is True


def test_every_unusable_side_is_reported_at_once_with_the_services_codes_and_texts(world):
    engine = FakeEngine(world, rule)
    comparer = LocalFacePairComparer(engine.config, engine=engine)
    bytes_of = {media: data for data, media in world.label_of.items()}
    with pytest.raises(FrPairRejected) as both:
        comparer.compare(bytes_of["nf"], bytes_of["lq"])
    assert [(r.side, r.kind, r.reason, r.detail) for r in both.value.rejections] == [
        ("a", "no_face", None, "No face detected in the image."),
        ("b", "low_quality", "too_small", "Face is too small (30px; minimum 40px). Use a closer or higher-resolution image.")]
    assert engine.calls == ["nf", "lq"]  # the second image was analysed although the first failed
    with pytest.raises(FrPairRejected) as one:
        comparer.compare(bytes_of["a1"], bytes_of["nf2"])
    assert [(r.side, r.kind) for r in one.value.rejections] == [("b", "no_face")]


def test_long_rejection_texts_are_cut_like_the_http_client_cuts_them(world):
    engine = FakeEngine(world, lambda m: ImageRejected("multiple_faces", "x" * 500) if m == "nf" else 0.0)
    bytes_of = {media: data for data, media in world.label_of.items()}
    with pytest.raises(FrPairRejected) as exc:
        LocalFacePairComparer(engine.config, engine=engine).compare(bytes_of["nf"], bytes_of["a1"])
    assert len(exc.value.rejections[0].detail) == DETAIL_LIMIT == 200


def test_an_engine_failure_is_an_operational_error_never_a_face_verdict(world):
    engine = FakeEngine(world, lambda m: RuntimeError("onnxruntime exploded") if m == "a2" else 0.0)
    bytes_of = {media: data for data, media in world.label_of.items()}
    with pytest.raises(FrOperationalError) as exc:
        LocalFacePairComparer(engine.config, engine=engine).compare(bytes_of["a1"], bytes_of["a2"])
    assert not isinstance(exc.value, FrPairRejected) and exc.value.kind == "operational" and "RuntimeError" in exc.value.detail


def test_a_missing_machine_learning_stack_is_reported_as_an_unavailable_engine_not_a_rejection():
    comparer = LocalFacePairComparer(LocalFaceConfig(model_root="Z:/no/such/place"))
    with pytest.raises(FrOperationalError) as exc:
        comparer.compare(b"x", b"y")  # either the packages or the model files are missing here
    assert isinstance(exc.value, EngineUnavailable) and exc.value.kind == "operational"
    with pytest.raises(EngineUnavailable):
        comparer.verify()


def test_the_module_imports_without_the_machine_learning_packages():
    import subprocess
    import sys

    code = ("import sys; sys.modules['numpy']=None; sys.modules['cv2']=None; sys.modules['onnxruntime']=None; sys.modules['insightface']=None;"
            "import dedup_pipeline.face_recognition as f; c=f.LocalFacePairComparer(); print('imported', c.config.model_name)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert done.stdout.strip() == "imported buffalo_l", done.stderr[-400:]


# --------------------------------------------------------------------------- the pipeline with the local engine (cache, summary, failures)

def test_a_run_records_the_engine_and_its_fingerprint_and_the_cache_signature_carries_it(world, tmp_path):
    engine = FakeEngine(world, rule)
    summary, comparer = run(world, tmp_path / "root", engine)
    assert summary["status"] == "completed" and summary["candidates"]["scored"] == 15
    block = summary["face"]["engine"]
    assert block["engine"] == ENGINE_NAME and block["fingerprint"] == engine.fingerprint and block["provider"] == "CPUExecutionProvider"
    assert block["settings"]["same_person_threshold"] == 0.45 and block["model"] == "buffalo_l" and set(block["model_file_digests"]) == {"det_10g.onnx", "w600k_r50.onnx"}
    lines = [json.loads(l) for l in (tmp_path / "root" / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines and all(l["signature"]["engine"] == ENGINE_NAME and l["signature"]["fingerprint"] == engine.fingerprint and l["signature"]["contract"] == "compare-v1" for l in lines)
    latest = json.loads((tmp_path / "root" / "latest.json").read_text())
    assert engine.fingerprint in latest["signature"]  # a resume across engines is not mistaken for the same run


def test_results_of_a_different_engine_fingerprint_are_never_reused_and_the_same_one_is(world, tmp_path):
    root = tmp_path / "root"
    first = FakeEngine(world, rule, "1" * 64)
    run(world, root, first, "r1")
    assert len(first.calls) > 0
    again = FakeEngine(world, rule, "1" * 64)
    summary, _ = run(world, root, again, "r2")
    assert again.calls == [] and summary["face"]["pairs_reused_from_cache"] == 15  # same fingerprint: all 15 pairs come from the cache
    other = FakeEngine(world, rule, "2" * 64)  # a different model file / setting / library
    summary, _ = run(world, root, other, "r3")
    assert len(other.calls) == len(first.calls) and summary["face"]["pairs_reused_from_cache"] == 0  # recomputed, not silently reused


def test_the_api_comparer_and_the_local_engine_keep_separate_stored_results(world, tmp_path):
    from .scoring_fakes import FakeComparer, outcome

    root = tmp_path / "root"
    api = FakeComparer(world, lambda a, b: outcome(0.7))
    api_summary = run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, "api", False), backend(world, api, None))
    assert api_summary["face"]["engine"] == {"engine": "fr_dopams_api", "contract": "compare-v1"}  # the default is named too
    api_lines = (root / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert api_lines and all("engine" not in json.loads(l)["signature"] for l in api_lines)  # the API signature is unchanged by Phase 10
    engine = FakeEngine(world, rule)
    local_summary, _ = run(world, root, engine, "local")
    assert len(engine.calls) > 0 and local_summary["face"]["pairs_reused_from_cache"] == 0  # API results are not taken as local results
    api2 = FakeComparer(world, lambda a, b: outcome(0.7))
    back = run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, root, "api2", False), backend(world, api2, None))
    assert api2.calls == [] and back["face"]["pairs_reused_from_cache"] == api_summary["face"]["eligible_pairs_both_usable_images"]  # and the other way round


def test_image_outcome_memory_ignores_results_of_other_engines(tmp_path):
    def rec(cid, fingerprint_value, media):
        sig = {"model": "buffalo_l", "max_images_per_person": 3, "contract": "compare-v1"}
        if fingerprint_value:
            sig.update({"engine": ENGINE_NAME, "fingerprint": fingerprint_value})
        return {"candidate_id": cid, "inputs": "i", "signature": sig,
                "evidence": {"status": "no_face", "reason": "No face detected in the image.", "attempts": [{"media_id_a": media, "media_id_b": media + "x", "outcome": "rejected", "rejections": {"a": "no_face"}}]}}

    path = tmp_path / "cache.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (rec("c1", None, "api-image"), rec("c2", "AAA", "engine-a-image"), rec("c3", "BBB", "engine-b-image"))) + "\n", encoding="utf-8")
    assert ImageOutcomeMemory.from_cache_file(path, "buffalo_l").failed("api-image") and not ImageOutcomeMemory.from_cache_file(path, "buffalo_l").failed("engine-a-image")
    a = ImageOutcomeMemory.from_cache_file(path, "buffalo_l", engine_fingerprint="AAA")
    assert a.failed("engine-a-image") and not a.failed("api-image") and not a.failed("engine-b-image")


def test_no_face_and_low_quality_and_shared_and_missing_images_are_handled_by_the_existing_pair_logic(world, tmp_path):
    engine = FakeEngine(world, rule)
    summary, _ = run(world, tmp_path / "root", engine)
    recs = {r["candidate_id"]: r for r in map(json.loads, (tmp_path / "root" / "runs" / "r" / "scored_candidates.jsonl").read_text(encoding="ascii").splitlines())}
    face = lambda a, b: recs[f"{a}:{b}"]["face"]
    assert face("p0", "p1")["status"] == "compared" and face("p0", "p1")["raw_cosine"] == pytest.approx(math.cos(0.1)) and face("p0", "p1")["service_threshold"] == 0.45
    assert face("p0", "p2")["status"] == "no_face" and face("p2", "p3")["status"] == "no_face"  # first failing side decides the status, as in the service
    assert face("p0", "p3")["status"] == "low_quality" and face("p0", "p3")["side_failures"] == {"b": "low_quality"}
    assert summary["status"] == "completed" and all(r["review"]["status"] == "pending" for r in recs.values())


def test_a_missing_image_file_is_an_image_load_error_and_never_reaches_the_engine(tmp_path):
    people, images = people_and_images()
    world = build_world(tmp_path / "w", people, images)
    world.files.clear()  # the file server no longer has any image
    engine = FakeEngine(world, rule)
    be, _ = local_backend(world, engine)
    summary = run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, tmp_path / "root", "r", False), be)
    # a file that is gone is a permanent fact about the record (not a transient outage), exactly as with the HTTP comparer: it never reaches the engine,
    # is recorded as an image-load error, does not make the run incomplete, and the pair is scored without face evidence
    assert engine.calls == [] and summary["status"] == "completed" and summary["face"]["retryable_errors"] == 0
    assert summary["face"]["error_kinds"] == {"image_load:file_not_found": 15}


def test_an_engine_failure_makes_the_run_incomplete_and_is_never_cached(world, tmp_path):
    engine = FakeEngine(world, lambda m: RuntimeError("model crashed") if m == "a2" else rule(m))
    summary, _ = run(world, tmp_path / "root", engine)
    assert summary["status"] == "incomplete" and summary["face"]["retryable_errors"] >= 1 and summary["face"]["error_kinds"].get("operational", 0) >= 1
    stored = [json.loads(l)["evidence"]["status"] for l in (tmp_path / "root" / "face_evidence_cache.jsonl").read_text(encoding="utf-8").splitlines()]
    healthy = FakeEngine(world, rule)
    fixed, _ = run(world, tmp_path / "root", healthy, "r2")  # the next run retries the failed pairs only
    assert fixed["status"] == "completed" and len(healthy.calls) < len(engine.calls) + 1 and "error" in stored


def test_an_engine_that_cannot_load_stops_the_run_before_anything_is_written(world, tmp_path):
    class Broken:
        config = LocalFaceConfig()

        @property
        def fingerprint(self):
            raise EngineUnavailable("the face-recognition packages are not installed here (insightface)")

    comparer = LocalFacePairComparer(Broken.config, engine=Broken())
    be = backend(world, comparer, comparer.verify)
    be.signature_fields, be.engine_info = comparer.signature_fields, comparer.describe
    with pytest.raises(EngineUnavailable):
        run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, tmp_path / "root", "r", False), be)
    assert not (tmp_path / "root" / "runs").exists() and not (tmp_path / "root" / "face_evidence_cache.jsonl").exists()
