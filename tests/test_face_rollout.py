"""Phase 10.1: `local` is the default face engine; the preflight fails safely; the API engine is an explicit rollback; nothing falls back silently."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dedup_pipeline.face_recognition import FrOperationalError, LocalFaceConfig
from dedup_pipeline.face_recognition.local_comparer import REQUIRED_MODULES, EngineUnavailable, check_environment
from dedup_pipeline.face_recognition.local_config import ENGINE_NAME, VALIDATED_LIBRARIES, VALIDATED_PROVIDERS, fingerprint
from dedup_pipeline.scoring import cli


def fake_onnxruntime(monkeypatch, providers):
    module = types.ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


def all_packages_present(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: types.SimpleNamespace(name=name) if name in dict(REQUIRED_MODULES) else real(name, *a, **k))


def model_dir(tmp_path: Path, files=("det_10g.onnx", "w600k_r50.onnx")) -> Path:
    directory = tmp_path / "models" / "buffalo_l"
    directory.mkdir(parents=True)
    for name in files:
        (directory / name).write_bytes(b"x")
    return tmp_path


# ------------------------------------------------------------------------------------------------ the defaults

def test_local_is_the_only_engine_and_cpu_is_the_default_provider():
    parser = cli.build_parser()
    default = parser.parse_args(["run"])
    assert default.face_provider is None and default.accept_unvalidated_provider is False and not hasattr(default, "face_engine")
    with pytest.raises(SystemExit):  # the API engine is gone: there is no switch to select it
        parser.parse_args(["run", "--face-engine", "api"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--face-engine", "local"])
    assert not hasattr(cli, "FrDopamsClient") and not hasattr(cli, "FrDopamsPairComparer") and not hasattr(cli, "FrConfig")
    assert LocalFaceConfig().provider == "cpu" and LocalFaceConfig().accept_unvalidated_provider is False
    assert VALIDATED_PROVIDERS == ("CPUExecutionProvider",)
    from dedup_pipeline.scoring import ScoringConfig
    from dedup_pipeline.scoring.config import RULE_VERSION

    assert RULE_VERSION == "score-v2"  # the scoring default is untouched
    assert ScoringConfig(reference_date="2026-10-12").rule_version == "score-v2"


# ---------------------------------------------------------------------------------------------------- preflight

def test_missing_packages_are_all_listed_at_once_with_the_fix(monkeypatch, tmp_path):
    import importlib.util

    real = importlib.util.find_spec
    gone = {"insightface", "scipy", "skimage"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name in gone else real(name, *a, **k))
    with pytest.raises(EngineUnavailable) as exc:
        check_environment(LocalFaceConfig(model_root=str(model_dir(tmp_path))))
    message = exc.value.detail
    assert "insightface" in message and "scipy" in message and "scikit-image" in message and ".face-venv" in message


def test_missing_model_files_are_named_with_their_folder(monkeypatch, tmp_path):
    all_packages_present(monkeypatch)
    fake_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    root = model_dir(tmp_path, files=("det_10g.onnx",))
    with pytest.raises(EngineUnavailable) as exc:
        check_environment(LocalFaceConfig(model_root=str(root)))
    assert "w600k_r50.onnx" in exc.value.detail and str(root / "models" / "buffalo_l") in exc.value.detail and "det_10g.onnx" not in exc.value.detail.split("missing in")[1].split(":")[1]


def test_packages_and_model_problems_are_reported_together(monkeypatch, tmp_path):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    with pytest.raises(EngineUnavailable) as exc:
        check_environment(LocalFaceConfig(model_root=str(tmp_path / "nowhere")))
    assert "missing Python packages" in exc.value.detail and "model files missing" in exc.value.detail


def test_cuda_requested_without_cuda_fails_instead_of_falling_back(monkeypatch, tmp_path):
    all_packages_present(monkeypatch)
    fake_onnxruntime(monkeypatch, ["AzureExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(EngineUnavailable, match="cuda.*only"):
        check_environment(LocalFaceConfig(provider="cuda", model_root=str(model_dir(tmp_path))))


def test_a_gpu_provider_is_refused_until_it_is_explicitly_accepted(monkeypatch, tmp_path):
    all_packages_present(monkeypatch)
    fake_onnxruntime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    root = str(model_dir(tmp_path))
    for provider in ("auto", "cuda"):
        with pytest.raises(EngineUnavailable, match="not been validated"):
            check_environment(LocalFaceConfig(provider=provider, model_root=root))
    assert check_environment(LocalFaceConfig(provider="cpu", model_root=root))["provider"] == "CPUExecutionProvider"  # a GPU being present changes nothing for cpu
    accepted = check_environment(LocalFaceConfig(provider="cuda", accept_unvalidated_provider=True, model_root=root))
    assert accepted["provider"] == "CUDAExecutionProvider"


def test_auto_without_a_gpu_is_the_validated_cpu_engine(monkeypatch, tmp_path):
    all_packages_present(monkeypatch)
    fake_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    assert check_environment(LocalFaceConfig(provider="auto", model_root=str(model_dir(tmp_path))))["provider"] == "CPUExecutionProvider"


def test_libraries_that_differ_from_the_validated_set_are_reported_not_hidden(monkeypatch, tmp_path):
    all_packages_present(monkeypatch)
    fake_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9" if name == "scipy" else VALIDATED_LIBRARIES.get(name, "1"))
    report = check_environment(LocalFaceConfig(model_root=str(model_dir(tmp_path))))
    assert report["differs_from_validated_libraries"] == {"scipy": {"installed": "9.9.9", "validated": VALIDATED_LIBRARIES["scipy"]}}


def test_every_library_that_can_change_a_result_is_in_the_fingerprint():
    models, libs = {"det": "1", "rec": "2"}, dict(VALIDATED_LIBRARIES)
    ref = fingerprint(LocalFaceConfig(), models, libs, "CPUExecutionProvider")
    for name in ("insightface", "onnxruntime", "numpy", "opencv-python", "scikit-image", "scipy", "Pillow"):
        assert fingerprint(LocalFaceConfig(), models, {**libs, name: "0.0.1"}, "CPUExecutionProvider") != ref, name
    assert fingerprint(LocalFaceConfig(accept_unvalidated_provider=True), models, libs, "CPUExecutionProvider") == ref  # a policy switch is not a numeric setting
    assert ENGINE_NAME == "local-v1"


# ------------------------------------------------------------------------ the CLI: fail safely, never fall back

class Boom(Exception):
    pass


def forbid_api_and_database(monkeypatch):
    """The run must never reach the database or the scorer when the local preflight fails (and there is no other engine to fall back to)."""
    def boom(*a, **k):
        raise Boom("the run continued after a failed preflight")

    monkeypatch.setattr(cli, "connect_read_only", boom)
    monkeypatch.setattr(cli, "run_scoring", boom)


def broken_config(tmp_path: Path) -> Path:
    path = tmp_path / "local.json"
    path.write_text(json.dumps({"model_root": str(tmp_path / "no-models-here")}), encoding="utf-8")
    return path


def test_a_failed_preflight_stops_the_run_without_side_effects_and_without_falling_back(monkeypatch, tmp_path, capsys):
    forbid_api_and_database(monkeypatch)
    root = tmp_path / "out"
    code = cli.main(["run", "--output-root", str(root), "--local-face-config", str(broken_config(tmp_path)), "--reference-date", "2026-10-12"])
    err = capsys.readouterr().err
    assert code == 2 and "PREFLIGHT FAILED (local)" in err and "Nothing was run" in err and "api" not in err.lower().replace("preflight", "")
    assert not root.exists()  # not even the output directory or its log was created


def test_the_face_preflight_command_reports_the_same_failure(monkeypatch, tmp_path, capsys):
    forbid_api_and_database(monkeypatch)
    assert cli.main(["face-preflight", "--local-face-config", str(broken_config(tmp_path))]) == 2
    assert "PREFLIGHT FAILED (local)" in capsys.readouterr().err


class FakeLocal:
    """Stands in for LocalFacePairComparer: a healthy engine that records how the CLI used it."""

    instances: list = []

    def __init__(self, config):
        self.config = config
        FakeLocal.instances.append(self)

    def preflight(self):
        return {}

    def verify(self):
        pass

    def compare(self, a, b):  # pragma: no cover - never called here
        raise AssertionError

    def signature_fields(self):
        return {"engine": ENGINE_NAME, "fingerprint": "f" * 64}

    def describe(self):
        return {"engine": ENGINE_NAME, "fingerprint": "f" * 64}


def capture_run(monkeypatch):
    seen = {}

    class Conn:
        def __enter__(self):
            return object()

        def __exit__(self, *exc):
            return False

    def fake_run(conn, config, options, backend):
        seen["backend"], seen["config"] = backend, config
        return {"status": "completed", "complete": True, "candidates": {"scored": 0, "score_bands_prioritisation_only": {}},
                "face": {"eligible_pairs_both_usable_images": 0, "retryable_errors": 0}, "elapsed_seconds": 0, "run_id": "x"}

    monkeypatch.setattr(cli, "connect_read_only", lambda *a, **k: Conn())
    monkeypatch.setattr(cli, "load_connection_settings", lambda *a, **k: None)
    monkeypatch.setattr(cli, "run_scoring", fake_run)
    return seen


def test_without_flags_a_run_uses_the_local_engine_and_score_v2(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    seen = capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    monkeypatch.delenv("GPU", raising=False)
    monkeypatch.setattr(cli, "gpu_env_files", lambda args: [])
    assert cli.main(["run", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"]) == 0
    backend = seen["backend"]
    assert len(FakeLocal.instances) == 1 and backend.signature_fields is not None and backend.engine_info is not None
    assert seen["config"].rule_version == "score-v2" and FakeLocal.instances[0].config.provider == "cpu"


def test_the_cli_source_has_no_api_engine_and_no_fallback_code():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    for forbidden in ("FrDopamsClient", "FrDopamsPairComparer", "FrConfig", "FR_DOPAMS", "--face-engine", "verify_service"):
        assert forbidden not in source, forbidden


def test_running_without_face_comparison_needs_no_engine_at_all(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    seen = capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    assert cli.main(["run", "--no-face-service", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"]) == 0
    assert FakeLocal.instances == [] and seen["backend"].comparer is None


def test_the_unvalidated_provider_flag_reaches_the_configuration_and_is_recorded(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    cli.main(["run", "--face-provider", "cuda", "--accept-unvalidated-provider", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"])
    config = FakeLocal.instances[0].config
    assert config.provider == "cuda" and config.accept_unvalidated_provider is True
    assert config.to_dict()["accept_unvalidated_provider"] is True  # visible in the run summary's engine settings


def test_gpu_true_in_the_env_file_runs_buffalo_on_cuda(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    monkeypatch.delenv("GPU", raising=False)
    env = tmp_path / "pipeline.env"
    env.write_text("GPU=true\n", encoding="utf-8")
    monkeypatch.setattr(cli, "gpu_env_files", lambda args: [env])
    assert cli.main(["run", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"]) == 0
    config = FakeLocal.instances[0].config
    assert config.model_name == "buffalo_l" and config.provider == "cuda" and config.accept_unvalidated_provider is True


def test_gpu_false_keeps_the_face_engine_on_cpu(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    monkeypatch.setenv("GPU", "false")
    assert cli.main(["run", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"]) == 0
    config = FakeLocal.instances[0].config
    assert config.provider == "cpu" and config.accept_unvalidated_provider is False


def test_face_provider_cpu_overrides_gpu_true(monkeypatch, tmp_path):
    FakeLocal.instances.clear()
    capture_run(monkeypatch)
    monkeypatch.setattr(cli, "LocalFacePairComparer", FakeLocal)
    monkeypatch.setenv("GPU", "true")
    cli.main(["run", "--face-provider", "cpu", "--output-root", str(tmp_path / "out"), "--reference-date", "2026-10-12"])
    config = FakeLocal.instances[0].config
    assert config.provider == "cpu" and config.accept_unvalidated_provider is False


def test_a_bad_gpu_value_stops_before_the_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GPU", "maybe")
    monkeypatch.setattr(cli, "gpu_env_files", lambda args: [])
    assert cli.main(["face-preflight"]) == 2
    assert "GPU must be true or false" in capsys.readouterr().err


def test_the_process_environment_wins_over_the_env_file(tmp_path):
    from dedup_pipeline.face_recognition.local_config import gpu_flag_from_sources
    env = tmp_path / ".env"
    env.write_text("GPU=true\n", encoding="utf-8")
    assert gpu_flag_from_sources({"GPU": "false"}, [env]) is False
    assert gpu_flag_from_sources({}, [env]) is True
    assert gpu_flag_from_sources({}, []) is None
