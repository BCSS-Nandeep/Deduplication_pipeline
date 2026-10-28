"""Explicit settings of the in-process face comparer, and the fingerprint that goes into the face-cache signature.

Every value that can change a face result is a field here. ``LocalFaceConfig()`` itself does not read the environment.
The scoring command reads ``GPU`` from the process environment or ``.env`` and applies it with ``apply_gpu_choice`` before the engine starts.
The defaults equal the values the FR-DOPAMS ``POST /compare`` endpoint used when the 525 stored results were
made (``src/config.py``: FACE_MIN_CONF 0.50, FACE_DET_SIZE 640x640, MIN_FACE_SIZE 40, MIN_FACE_SHARPNESS 15, MULTI_FACE_AREA_RATIO 0.5,
MAX_UPLOAD_MB 10, COMPARE_MAX_PIXELS 40,000,000, PERSON_MATCH_THRESHOLD 0.45, model ``buffalo_l``).

This module imports only the standard library, so the pipeline environment can import it without the machine-learning packages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .engine_fingerprint import engine_digest

ENGINE_NAME = "local-v1"  # bump when the comparison logic below (not just a setting) changes
PROVIDER_CHOICES = ("auto", "cpu", "cuda")
# Model files whose bytes decide the embeddings. The pose/landmark model of the service is not part of a comparison.
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
# Libraries whose version can change floating-point results.
LIBRARIES = ("insightface", "onnxruntime", "numpy", "opencv-python", "scikit-image", "scipy", "Pillow")
# What the Phase 10 replay (525 stored real-service cases, CPU) validated. A different version is not an error - it changes the fingerprint, so
# stored results are recomputed instead of reused - but the preflight reports the difference.
VALIDATED_PROVIDERS = ("CPUExecutionProvider",)
VALIDATED_LIBRARIES = {"insightface": "2.0", "onnxruntime": "1.23.2", "numpy": "2.2.6", "opencv-python": "4.13.0.92", "scikit-image": "0.26.0", "scipy": "1.18.1",
                       "Pillow": "12.2.0"}


@dataclass(frozen=True)
class LocalFaceConfig:
    model_name: str = "buffalo_l"
    model_root: Optional[str] = None  # None -> ~/.insightface (where InsightFace keeps downloaded models)
    provider: str = "cpu"  # "cpu" (the validated default); "auto": CUDA when available, otherwise CPU; "cuda" (fails if unavailable)
    accept_unvalidated_provider: bool = False  # a provider other than the validated CPU one is refused unless this is set after its own replay
    device_id: int = 0
    det_size: tuple[int, int] = (640, 640)
    min_detection_confidence: float = 0.50
    min_face_size_px: int = 40  # shorter side of the face box
    min_face_sharpness: float = 15.0  # Laplacian variance of the face crop resized to sharpness_crop_px
    sharpness_crop_px: int = 112
    multi_face_area_ratio: float = 0.5  # a second face this fraction of the largest one's area makes the image ambiguous
    max_image_mb: float = 10.0
    max_pixels: int = 40_000_000
    same_person_threshold: float = 0.45  # the service's PERSON_MATCH_THRESHOLD: 1:1 verification, not an identity decision
    embedding_crop_px: int = 112
    flip_average: bool = True  # embed the aligned crop and its mirror image and average (as the service does)
    load_pose_model: bool = False  # the service loads landmark_3d_68 for enrolment poses; a comparison never uses it

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_CHOICES:
            raise ValueError(f"provider must be one of {PROVIDER_CHOICES}")
        if len(self.det_size) != 2 or min(self.det_size) < 32:
            raise ValueError("det_size must be two integers of at least 32")
        if not (0.0 <= self.min_detection_confidence <= 1.0 and -1.0 <= self.same_person_threshold <= 1.0):
            raise ValueError("confidence must be in [0, 1] and the similarity threshold in [-1, 1]")
        if self.min_face_size_px < 1 or self.min_face_sharpness < 0 or not 0 < self.multi_face_area_ratio <= 1:
            raise ValueError("face quality limits out of range")
        if self.max_image_mb <= 0 or self.max_pixels <= 0:
            raise ValueError("image limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["det_size"] = list(self.det_size)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalFaceConfig":
        known = {f.name for f in fields(cls)}
        if set(data) - known:
            raise ValueError(f"unknown local face configuration keys: {sorted(set(data) - known)}")
        values = dict(data)
        if "det_size" in values:
            values["det_size"] = tuple(int(v) for v in values["det_size"])
        return cls(**values)

    @classmethod
    def from_file(cls, path: Path) -> "LocalFaceConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


_GPU_TRUE = {"1", "true", "yes", "on"}
_GPU_FALSE = {"0", "false", "no", "off"}


def parse_gpu_flag(value: Optional[str]) -> Optional[bool]:
    """``None`` when GPU is unset. ``True`` and ``False`` are the only other results."""
    if value is None:
        return None
    text = value.strip().lower()
    if text in _GPU_TRUE:
        return True
    if text in _GPU_FALSE:
        return False
    raise ValueError(f"GPU must be true or false, got {value!r}")


def _env_file_value(path: Path, name: str) -> Optional[str]:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, raw = text.partition("=")
        if key.strip() == name:
            return raw.strip().strip("'\"")
    return None


def gpu_flag_from_sources(environ: Mapping[str, str], env_files: Sequence[Path]) -> Optional[bool]:
    """Process environment wins, then the first env file that contains ``GPU``."""
    if "GPU" in environ:
        return parse_gpu_flag(environ["GPU"])
    for path in env_files:
        value = _env_file_value(path, "GPU")
        if value is not None:
            return parse_gpu_flag(value)
    return None


def apply_gpu_choice(config: LocalFaceConfig, enabled: Optional[bool]) -> LocalFaceConfig:
    """``True`` runs buffalo_l on CUDA. ``False`` forces CPU. ``None`` leaves the config unchanged."""
    if enabled is True:
        return replace(config, provider="cuda", accept_unvalidated_provider=True)
    if enabled is False:
        return replace(config, provider="cpu", accept_unvalidated_provider=False)
    return config


def fingerprint(config: LocalFaceConfig, model_hashes: Mapping[str, str], libraries: Mapping[str, str], provider_used: str) -> str:
    """Hash of everything that decides a result: all settings (the requested provider is replaced by the one actually used),
    the SHA-256 of each model file, the versions of the numeric libraries and the engine's own version."""
    settings = config.to_dict()
    settings.pop("model_root")  # where the files are is irrelevant; what they contain is hashed below
    settings.pop("accept_unvalidated_provider")  # a policy switch, not something that changes a result
    settings["provider"] = provider_used
    payload = {"engine": ENGINE_NAME, "settings": settings, "models": dict(sorted(model_hashes.items())), "libraries": dict(sorted(libraries.items()))}
    return engine_digest(payload)
