"""In-process face comparer: the stateless comparison logic of FR-DOPAMS ``POST /compare``, without the service around it.

Reused, selectively and with the same numerics, from ``C:\\Users\\Dell\\Desktop\\FR-DOPAMS``:

* ``src/detect.py``     InsightFace ``buffalo_l`` detection, CUDA-if-available / CPU provider choice, detection-confidence filter;
* ``src/embed.py``      aligned 112x112 crop and its mirror image, embedded and averaged, L2-normalised (flip averaging);
* ``src/face_service.py``  image decoding, upload and pixel limits, single-face validation (largest face, second-face area ratio),
                        face-quality checks (size, sharpness), and the cosine comparison of the two unit embeddings;
* ``src/config.py``     the numeric constants - here explicit fields of ``LocalFaceConfig``.

Deliberately NOT taken: the HTTP routes and middleware, the API key, request-size handling, gallery enrolment and search, match logs,
the database layer, pose checks and the 3-D landmark model. Nothing here touches a database or the network.

Contract: ``LocalFacePairComparer`` implements ``PairComparer`` exactly like ``FrDopamsPairComparer``: ``compare(image_a, image_b)`` returns a
``ComparisonOutcome`` or raises ``FrPairRejected`` (every unusable side listed, texts as the service words them, cut to 200 characters as
the HTTP client does) or ``FrOperationalError``. A face problem is data; an engine problem is never reported as a face verdict.

The machine-learning packages are imported only when the engine is first used (the pipeline's own environment does not have them), and the
model is loaded once per process and shared by every comparer and thread that uses the same configuration.
"""

from __future__ import annotations

import io
import logging
import threading
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from .errors import FrOperationalError, FrPairRejected, SideRejection
from .engine_fingerprint import file_digest
from .local_config import ENGINE_NAME, LIBRARIES, MODEL_FILES, VALIDATED_LIBRARIES, VALIDATED_PROVIDERS, LocalFaceConfig, fingerprint
from .pair_evidence import ComparisonOutcome

log = logging.getLogger(__name__)
DETAIL_LIMIT = 200  # the HTTP client keeps at most this many characters of a rejection message
ANALYSIS_CACHE_MAX = 20_000  # per-process answers kept (an embedding is 2 KB; a rejection a few hundred bytes); when full, new answers are simply not kept


class EngineUnavailable(FrOperationalError):
    """The machine-learning stack or the model files are not available in this environment."""

    kind = "operational"


class ImageRejected(ValueError):
    """An image the engine cannot use (the service's ``ImageRejected``): ``code`` no_face / multiple_faces / low_quality / invalid_image."""

    def __init__(self, code: str, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.code, self.reason = code, reason


# ---------------------------------------------------------------------------------------------------- the engine (heavy part)

REQUIRED_MODULES = (("cv2", "opencv-python"), ("numpy", "numpy"), ("onnxruntime", "onnxruntime"), ("insightface", "insightface"),
                    ("PIL", "Pillow"), ("skimage", "scikit-image"), ("scipy", "scipy"))


def check_environment(config: LocalFaceConfig) -> dict[str, Any]:
    """Preflight: everything that must be true before the model is loaded, with ONE clear error listing all problems.

    Never loads the model and never falls back to anything: a missing package, a missing model file, a provider onnxruntime does not offer, or a
    provider that has not been validated stops the run (``EngineUnavailable``). Returns a description of the environment when it is fine."""
    import importlib.util

    problems: list[str] = []
    missing = [pip_name for module, pip_name in REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        problems.append(f"missing Python packages: {', '.join(missing)} (install requirements-face-local.txt into the dedicated .face-venv)")
    root = Path(config.model_root).expanduser() if config.model_root else Path("~/.insightface").expanduser()
    model_dir = root / "models" / config.model_name
    absent = [f for f in MODEL_FILES if not (model_dir / f).is_file()]
    if absent:
        problems.append(f"model files missing in {model_dir}: {', '.join(absent)}")
    available: list[str] = []
    if "onnxruntime" not in missing:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
        if config.provider == "cuda" and "CUDAExecutionProvider" not in available:
            problems.append(f"provider 'cuda' was requested but onnxruntime offers only {available}")
    used = "CUDAExecutionProvider" if config.provider != "cpu" and "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if not problems and used not in VALIDATED_PROVIDERS and not config.accept_unvalidated_provider:
        problems.append(f"provider {used} has not been validated: replay the 525 stored cases on this hardware first, then set accept_unvalidated_provider "
                        "(--accept-unvalidated-provider); use --face-provider cpu for the validated engine")
    if problems:
        raise EngineUnavailable("; ".join(problems))
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {"provider": used, "onnxruntime_providers": available, "model_dir": str(model_dir), "libraries": versions,
            "differs_from_validated_libraries": {k: {"installed": versions.get(k), "validated": v} for k, v in VALIDATED_LIBRARIES.items() if versions.get(k) != v}}


class LocalFaceEngine:
    """The loaded model plus the analysis steps. One instance per process and configuration (see ``get_engine``)."""

    def __init__(self, config: LocalFaceConfig):
        self.config = config
        environment = check_environment(config)  # raises EngineUnavailable with every problem at once
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        from PIL import Image, UnidentifiedImageError

        self._cv2, self._np, self._Image, self._UnidentifiedImageError = cv2, np, Image, UnidentifiedImageError
        use_cuda = environment["provider"] == "CUDAExecutionProvider"
        self.providers = (["CUDAExecutionProvider"] if use_cuda else []) + ["CPUExecutionProvider"]
        modules = ["detection", "recognition"] + (["landmark_3d_68"] if config.load_pose_model else [])
        model_dir = Path(environment["model_dir"])
        root = str(model_dir.parent.parent)
        self.model_dir = model_dir
        self.app = FaceAnalysis(name=config.model_name, root=root, providers=self.providers, allowed_modules=modules)
        self.app.prepare(ctx_id=config.device_id if use_cuda else -1, det_size=config.det_size)
        self.provider_used = self.providers[0]
        self.model_hashes = {f: file_digest(self.model_dir / f) for f in MODEL_FILES}
        self.libraries = dict(environment["libraries"])
        self.environment = environment
        self.fingerprint = fingerprint(config, self.model_hashes, self.libraries, self.provider_used)
        self._max_bytes = int(config.max_image_mb * 1024 * 1024)
        from insightface.app.common import Face

        self._Face = Face
        self._analysis: dict[tuple[str, int], tuple] = {}
        self._analysis_lock = threading.Lock()
        self._in_flight: dict[tuple[str, int], threading.Event] = {}
        self.analysis_hits = self.analysis_misses = 0

    # ---- face_service.py steps, same order and same thresholds

    def _decode(self, data: bytes):
        if not data:
            raise ImageRejected("invalid_image", "Empty image data provided.", "empty")
        if len(data) > self._max_bytes:
            raise ImageRejected("invalid_image", f"Image is larger than the {self.config.max_image_mb:g} MB limit.", "too_large")
        image = self._cv2.imdecode(self._np.frombuffer(data, dtype=self._np.uint8), self._cv2.IMREAD_COLOR)
        if image is None:
            raise ImageRejected("invalid_image", "Unable to decode image bytes.", "undecodable")
        return image

    def _check_pixel_budget(self, data: bytes) -> None:
        """Refuse images that would decode to an enormous bitmap, reading only the header."""
        try:
            with self._Image.open(io.BytesIO(data)) as probe:
                width, height = probe.size
        except self._Image.DecompressionBombError:
            raise ImageRejected("invalid_image", "Image has far too many pixels to decode safely.", "too_many_pixels") from None
        except (self._UnidentifiedImageError, OSError, ValueError):
            return  # not a format Pillow reads; the real decoder decides
        if width * height > self.config.max_pixels:
            raise ImageRejected("invalid_image", f"Image is {width}x{height}; more than {self.config.max_pixels} pixels is not accepted.", "too_many_pixels")

    @staticmethod
    def _face_area(face) -> float:
        x1, y1, x2, y2 = face.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _sharpness(self, image, face) -> float:
        """Laplacian variance of the face crop at a fixed size, so the score does not depend on image resolution."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if crop.size == 0:
            return 0.0
        size = self.config.sharpness_crop_px
        gray = self._cv2.cvtColor(self._cv2.resize(crop, (size, size)), self._cv2.COLOR_BGR2GRAY)
        return float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())

    def _check_quality(self, image, face) -> None:
        x1, y1, x2, y2 = face.bbox.astype(float)
        size = min(x2 - x1, y2 - y1)
        if size < self.config.min_face_size_px:
            raise ImageRejected("low_quality", f"Face is too small ({size:.0f}px; minimum {self.config.min_face_size_px}px). "
                                               "Use a closer or higher-resolution image.", "too_small")
        if self._sharpness(image, face) < self.config.min_face_sharpness:
            raise ImageRejected("low_quality", "Face is too blurred to match reliably. Use a sharper image.", "too_blurred")

    def _detect(self, image):
        """Faces with box, landmarks and detection score, in the detector's order.

        ``FaceAnalysis.get`` also runs the recognition model on every detected face. That embedding (``face.embedding``) is only read when there are no
        landmarks or flip averaging is off; otherwise ``_embedding`` embeds the aligned crop and its mirror image itself and the library's pass is thrown
        away (measured: about a quarter of the time of an image that contains a face). In that case the detector is called directly - the same
        ``det_model.detect`` call ``get`` makes, producing the same ``Face`` values - and the unused pass is skipped. Anything else uses ``get`` unchanged."""
        if self.config.flip_average and not self.config.load_pose_model:
            bboxes, kpss = self.app.det_model.detect(image, max_num=0, metric="default")
            if kpss is not None:
                return [self._Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4]) for i in range(bboxes.shape[0])]
        return self.app.get(image)

    def _single_face(self, image):
        faces = [f for f in self._detect(image) if f.det_score >= self.config.min_detection_confidence]
        if not faces:
            raise ImageRejected("no_face", "No face detected in the image.")
        faces = sorted(faces, key=self._face_area, reverse=True)
        if len(faces) > 1 and self._face_area(faces[1]) >= self._face_area(faces[0]) * self.config.multi_face_area_ratio:
            raise ImageRejected("multiple_faces", f"{len(faces)} faces of similar size detected. Crop the image to the one person to be matched.")
        self._check_quality(image, faces[0])
        return faces[0]

    def _unit(self, vec):
        vec = self._np.asarray(vec, dtype=self._np.float32).reshape(-1)
        return vec / self._np.linalg.norm(vec)

    def _embedding(self, face, image):
        """L2-normalised ArcFace embedding; with landmarks, the aligned crop and its mirror image are embedded and averaged."""
        kps = getattr(face, "kps", None)
        if image is None or kps is None or not self.config.flip_average:
            return self._unit(face.embedding)
        from insightface.utils import face_align

        crop = face_align.norm_crop(image, landmark=kps, image_size=self.config.embedding_crop_px)
        feats = self.app.models["recognition"].get_feat([crop, self._np.ascontiguousarray(crop[:, ::-1])])
        return self._unit(self._unit(feats[0]) + self._unit(feats[1]))

    def _analyse_uncached(self, data: bytes):
        if data and len(data) <= self._max_bytes:
            self._check_pixel_budget(data)
        image = self._decode(data)
        face = self._single_face(image)
        return self._embedding(face, image)

    def analyse(self, data: bytes):
        """Embedding of the one usable face in ``data`` or ``ImageRejected``.

        The answer is a pure function of the image and the engine, and the same image is compared in many pairs of one run. When the loader labelled the
        bytes with the manifest identity of the file (``ImageBytes.cache_key``: media id, path, extension), the answer - an embedding, or the rejection
        with its exact wording - is kept in memory for this process and reused. Bytes without a label are never cached. Only definite answers are kept:
        an operational failure raises out of here and is never remembered. The key includes the byte length as a guard (no hashing of image content)."""
        label = getattr(data, "cache_key", None)
        if label is None:
            return self._analyse_uncached(data)
        key = (label, len(data))
        while True:
            with self._analysis_lock:
                hit = self._analysis.get(key)
                if hit is not None:
                    self.analysis_hits += 1
                    break
                running = self._in_flight.get(key)
                if running is None:  # this thread computes the answer; others arriving meanwhile wait for it instead of repeating the work
                    self._in_flight[key] = threading.Event()
                    break
            running.wait()  # then look again: normally the answer is now in the cache; after a failure or a full cache this thread computes it itself
        if hit is not None:
            if hit[0] == "rejected":
                raise ImageRejected(hit[1], hit[2], hit[3])
            return hit[1]
        entry = None
        try:
            embedding = self._analyse_uncached(data)
            embedding.setflags(write=False)
            entry = ("ok", embedding)
        except ImageRejected as exc:
            entry = ("rejected", exc.code, str(exc), exc.reason)
        finally:  # whatever happened - even an operational failure, which is not stored - release the threads waiting for this image
            with self._analysis_lock:
                if entry is not None:
                    self.analysis_misses += 1
                    if len(self._analysis) < ANALYSIS_CACHE_MAX:
                        self._analysis[key] = entry
                self._in_flight.pop(key).set()
        if entry[0] == "rejected":
            raise ImageRejected(entry[1], entry[2], entry[3])
        return entry[1]
    def analysis_cache_stats(self) -> dict[str, int]:
        with self._analysis_lock:
            return {"entries": len(self._analysis), "hits": self.analysis_hits, "misses": self.analysis_misses, "max_entries": ANALYSIS_CACHE_MAX}

    def cosine(self, a, b) -> float:
        return max(-1.0, min(1.0, float(self._np.dot(a, b))))


_ENGINES: dict[str, LocalFaceEngine] = {}
_ENGINE_LOCK = threading.Lock()


def get_engine(config: LocalFaceConfig) -> LocalFaceEngine:
    """The process-wide engine for this configuration; the model is loaded once, on first use."""
    key = repr(sorted(config.to_dict().items()))
    with _ENGINE_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = _ENGINES[key] = LocalFaceEngine(config)
        return engine


# ------------------------------------------------------------------------------------------------- the PairComparer

class LocalFacePairComparer:
    """``PairComparer`` backed by the in-process engine. Thread-safe: the pipeline calls it from several threads."""

    def __init__(self, config: Optional[LocalFaceConfig] = None, engine: Optional[Any] = None):
        self.config = config or LocalFaceConfig()
        self._engine = engine  # injectable for tests; otherwise loaded lazily by get_engine

    @property
    def engine(self) -> LocalFaceEngine:
        if self._engine is None:
            self._engine = get_engine(self.config)
        return self._engine

    # ---- what the pipeline records and uses

    def signature_fields(self) -> dict[str, str]:
        """Added to the face-cache signature: results from a different engine, setting, model file or library are never reused."""
        engine = self.engine
        return {"engine": ENGINE_NAME, "fingerprint": engine.fingerprint}

    def describe(self) -> dict[str, Any]:
        engine = self.engine
        return {"engine": ENGINE_NAME, "fingerprint": engine.fingerprint, "provider": engine.provider_used, "providers_loaded": list(engine.providers),
                "model": self.config.model_name, "model_file_digests": dict(engine.model_hashes), "libraries": dict(engine.libraries),
                "settings": self.config.to_dict(),
                "analysis_cache": engine.analysis_cache_stats() if hasattr(engine, "analysis_cache_stats") else None,
                "validated_environment": {"provider_validated": engine.provider_used in VALIDATED_PROVIDERS,
                                          "libraries_differing_from_the_validated_set": (getattr(engine, "environment", None) or {}).get("differs_from_validated_libraries", {})}}

    def preflight(self) -> dict[str, Any]:
        """Check the environment WITHOUT loading the model (packages, model files, provider). Raises ``EngineUnavailable`` listing every problem."""
        return check_environment(self.config)

    def verify(self) -> None:
        """Raise unless the engine loads and reports the configured model (the local counterpart of ``verify_service``)."""
        engine = self.engine
        if engine.config.model_name != self.config.model_name:  # pragma: no cover - defensive
            raise EngineUnavailable(f"engine model {engine.config.model_name!r} is not the configured {self.config.model_name!r}")

    # ---- PairComparer

    def compare(self, image_a: bytes, image_b: bytes) -> ComparisonOutcome:
        try:
            engine = self.engine
        except EngineUnavailable:
            raise
        except Exception as exc:
            raise EngineUnavailable(f"{type(exc).__name__}: {str(exc)[:160]}") from None
        embeddings: dict[str, Any] = {}
        rejections: list[SideRejection] = []
        for side, data in (("a", image_a), ("b", image_b)):  # both are analysed, so every unusable side is reported at once
            try:
                embeddings[side] = engine.analyse(data)
            except ImageRejected as exc:
                rejections.append(SideRejection(side, exc.code, exc.reason, str(exc)[:DETAIL_LIMIT]))
            except FrOperationalError:
                raise
            except Exception as exc:  # an engine failure is never a face verdict
                log.exception("local face comparison failed")
                raise FrOperationalError(f"{type(exc).__name__}: {str(exc)[:160]}") from None
        if rejections:
            raise FrPairRejected(tuple(rejections))
        similarity = engine.cosine(embeddings["a"], embeddings["b"])
        return ComparisonOutcome(similarity, "cosine", similarity >= self.config.same_person_threshold, self.config.same_person_threshold, self.config.model_name)
