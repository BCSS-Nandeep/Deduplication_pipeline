"""Phase 10: the REAL in-process engine. Needs the face packages and the buffalo_l model files, so it is skipped in the pipeline's own environment.

Run it where requirements-face-local.txt is installed (the FR-DOPAMS environment works):

    .fr-venv\\Scripts\\python -m pytest tests/test_local_comparer_ml.py -q

It checks the failure cases with synthetic images, load-once and thread safety, provider handling, and - when the file server and the stored
results are available - replays a sample covering every stored outcome type against the real service's recorded evidence.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("insightface")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from dedup_pipeline.face_recognition import FrPairRejected, LocalFaceConfig, LocalFacePairComparer  # noqa: E402
from dedup_pipeline.face_recognition.local_comparer import EngineUnavailable, ImageRejected, get_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HAS_CUDA = "CUDAExecutionProvider" in ort.get_available_providers()
MODEL_DIR = Path("~/.insightface/models/buffalo_l").expanduser()
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "det_10g.onnx").is_file(), reason="buffalo_l model files not installed")


def png(array) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array).save(out, "PNG")
    return out.getvalue()


BLANK = png(np.full((300, 300, 3), 128, np.uint8))
NOISE = png(np.random.default_rng(3).integers(0, 255, (300, 300, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def comparer():
    return LocalFacePairComparer(LocalFaceConfig(provider="cpu"))


def kinds(exc) -> list[tuple[str, str, str | None]]:
    return [(r.side, r.kind, r.reason) for r in exc.value.rejections]


def test_images_without_a_face_are_rejected_on_the_side_that_has_none(comparer):
    with pytest.raises(FrPairRejected) as exc:
        comparer.compare(BLANK, NOISE)
    assert kinds(exc) == [("a", "no_face", None), ("b", "no_face", None)]
    assert all(r.detail == "No face detected in the image." for r in exc.value.rejections)


def test_empty_corrupt_and_non_image_data_are_invalid_images_never_operational_errors(comparer):
    with pytest.raises(FrPairRejected) as exc:
        comparer.compare(b"", b"this is not an image")
    assert kinds(exc) == [("a", "invalid_image", "empty"), ("b", "invalid_image", "undecodable")]
    truncated = BLANK[: len(BLANK) // 2]
    with pytest.raises(FrPairRejected) as exc:
        comparer.compare(truncated, BLANK)
    assert kinds(exc)[0][:2] == ("a", "invalid_image")


def test_the_size_and_pixel_limits_are_enforced_before_decoding():
    small = LocalFacePairComparer(LocalFaceConfig(provider="cpu", max_image_mb=0.0005, max_pixels=10_000))
    with pytest.raises(FrPairRejected) as exc:
        small.compare(BLANK, png(np.zeros((90, 90, 3), np.uint8)))  # 900 bytes compressed is fine for side b only if the size limit allows it
    assert kinds(exc)[0] == ("a", "invalid_image", "too_large")
    pixels = LocalFacePairComparer(LocalFaceConfig(provider="cpu", max_pixels=10_000))
    with pytest.raises(FrPairRejected) as exc:
        pixels.compare(png(np.zeros((200, 200, 3), np.uint8)), png(np.zeros((50, 50, 3), np.uint8)))
    assert kinds(exc)[0] == ("a", "invalid_image", "too_many_pixels") and exc.value.rejections[0].detail.startswith("Image is 200x200;")


def test_the_model_is_loaded_once_per_process_and_shared(comparer):
    other = LocalFacePairComparer(LocalFaceConfig(provider="cpu"))
    assert comparer.engine is other.engine is get_engine(LocalFaceConfig(provider="cpu"))
    assert comparer.engine.model_hashes.keys() == {"det_10g.onnx", "w600k_r50.onnx"}
    assert LocalFacePairComparer(LocalFaceConfig(provider="cpu", min_face_size_px=50)).engine is not comparer.engine  # another setting, another engine


def test_concurrent_calls_give_identical_answers(comparer):
    def one(_):
        with pytest.raises(FrPairRejected) as exc:
            comparer.compare(BLANK, NOISE)
        return kinds(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(32)))
    assert all(r == results[0] for r in results)


def test_the_fingerprint_is_stable_for_a_process_and_names_the_provider_used(comparer):
    d = comparer.describe()
    assert d["fingerprint"] == comparer.signature_fields()["fingerprint"] == LocalFacePairComparer(LocalFaceConfig(provider="cpu")).describe()["fingerprint"]
    assert d["provider"] == "CPUExecutionProvider" and d["libraries"]["onnxruntime"] == ort.__version__
    assert LocalFacePairComparer(LocalFaceConfig(provider="cpu", load_pose_model=True)).describe()["fingerprint"] != d["fingerprint"]


@pytest.mark.skipif(HAS_CUDA, reason="this machine has CUDA")
def test_cuda_requested_without_a_cuda_provider_fails_clearly_instead_of_falling_back():
    cuda = LocalFacePairComparer(LocalFaceConfig(provider="cuda"))
    with pytest.raises(EngineUnavailable, match="cuda"):
        cuda.verify()
    assert LocalFacePairComparer(LocalFaceConfig(provider="auto")).describe()["provider"] == "CPUExecutionProvider"  # auto falls back, as the service did


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA provider on this machine: GPU behaviour could not be tested")
def test_cuda_provider_is_used_when_available_and_has_its_own_fingerprint():  # pragma: no cover - needs a GPU
    gpu = LocalFacePairComparer(LocalFaceConfig(provider="cuda"))
    assert gpu.describe()["provider"] == "CUDAExecutionProvider"
    assert gpu.describe()["fingerprint"] != LocalFacePairComparer(LocalFaceConfig(provider="cpu")).describe()["fingerprint"]


# ------------------------------------------------------------------------------------------- the stored real-service cases

def _replay_tool():
    spec = importlib.util.spec_from_file_location("phase10_replay", ROOT / "tools" / "phase10_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_server_up() -> bool:
    import urllib.request

    try:
        from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL

        urllib.request.urlopen(DEFAULT_FILE_SERVER_BASE_URL, timeout=3)
        return True
    except Exception as exc:  # an HTTP error still means the server answered
        return hasattr(exc, "code")


@pytest.mark.skipif(not (ROOT / "output" / "scoring" / "face_evidence_cache.jsonl").is_file() or not (ROOT / "output" / "image_resolution").is_dir(),
                    reason="the stored real-service results are not available")
def test_a_sample_of_every_stored_outcome_type_replays_to_identical_evidence(tmp_path, monkeypatch):
    if not _file_server_up():
        pytest.skip("the image file server is not reachable, so the stored cases cannot be reproduced")
    tool = _replay_tool()
    monkeypatch.setattr(sys, "argv", ["phase10_replay", "--out", str(tmp_path), "--per-status", "3", "--provider", "cpu"])
    assert tool.main() == 0  # exit code 1 would mean at least one mismatch
    import json

    result = json.loads((tmp_path / "replay_result.json").read_text())
    assert result["mismatching_cases"] == 0 and result["status_agreement"] == result["cases_replayed"] >= 15
    assert set(result["stored_status_counts"]) >= {"no_face", "low_quality", "matched", "not_matched", "multiple_faces", "skipped_shared_image"}
    assert result["max_abs_cosine_difference"] <= result["tolerance"]


# ------------------------------------------------------------------------------ Phase 11: optimised paths equal the library path

@pytest.mark.skipif(not (ROOT / "output" / "scoring" / "face_evidence_cache.jsonl").is_file() or not (ROOT / "output" / "image_resolution").is_dir(),
                    reason="the stored real-service results are not available")
def test_skipping_the_library_recognition_pass_and_the_answer_cache_change_no_result_on_real_images():
    if not _file_server_up():
        pytest.skip("the image file server is not reachable")
    import json

    from dedup_pipeline.face_recognition.image_source import ManifestImageLoader
    from dedup_pipeline.image_resolution.file_server import DEFAULT_FILE_SERVER_BASE_URL, HttpFetcher
    from dedup_pipeline.image_resolution.resolver import ResolverConfig
    from dedup_pipeline.scoring.inputs import validate_images

    engine = get_engine(LocalFaceConfig(provider="cpu"))
    images = validate_images(ROOT / "output" / "image_resolution", None)
    loader = ManifestImageLoader(ResolverConfig(base_url=DEFAULT_FILE_SERVER_BASE_URL), HttpFetcher())
    stored = [json.loads(line) for line in open(ROOT / "output" / "scoring" / "face_evidence_cache.jsonl", encoding="utf-8")]
    seen, records = set(), []
    for rec in stored:  # a spread of images: with a face, without, several faces, too small, blurred
        for a in rec["evidence"].get("attempts") or []:
            for side, key in (("a", "person_a_id"), ("b", "person_b_id")):
                media = a[f"media_id_{side}"]
                if media not in seen and len(seen) < 40:
                    seen.add(media)
                    records += [r for r in images.index.get(rec["evidence"][key]).usable_images if r["media_id"] == media]
    assert len(records) >= 30

    def reference(data):
        """The unoptimised path: the library's own `get` (including its recognition pass), no answer cache."""
        original = engine._detect
        engine._detect = lambda image: engine.app.get(image)
        try:
            return ("ok", engine._analyse_uncached(bytes(data)).tolist())
        except ImageRejected as exc:
            return ("rejected", exc.code, str(exc), exc.reason)
        finally:
            engine._detect = original

    def optimised(data):
        try:
            return ("ok", engine.analyse(data).tolist())
        except ImageRejected as exc:
            return ("rejected", exc.code, str(exc), exc.reason)

    outcomes = {"ok": 0, "rejected": 0}
    for record in records:
        data = loader.load(record)
        want = reference(data)
        assert optimised(data) == want, record["media_id"]  # first call: computed, skipping the library pass
        assert optimised(data) == want, record["media_id"]  # second call: served from the answer cache
        outcomes[want[0]] += 1
    assert outcomes["ok"] >= 5 and outcomes["rejected"] >= 5  # both kinds of answer were exercised
    assert engine.analysis_cache_stats()["hits"] >= len(records)

    # the detector's faces are the same values with and without the library's recognition pass
    for record in records[:15]:
        image = engine._decode(loader.load(record))
        via_get = sorted(engine.app.get(image), key=lambda f: -float(f.det_score))
        direct = sorted(engine._detect(image), key=lambda f: -float(f.det_score))
        assert len(via_get) == len(direct)
        for a, b in zip(via_get, direct):
            assert (a.bbox == b.bbox).all() and (a.kps == b.kps).all() and a.det_score == b.det_score
