"""FR-DOPAMS client over real HTTP against an emulator that mirrors the service's responses."""

from __future__ import annotations

import logging

import pytest

from dedup_pipeline.face_recognition import (
    FrAuthError,
    FrConfig,
    FrConfigError,
    FrContractError,
    FrDopamsClient,
    FrImageRejected,
    FrInvalidResponse,
    FrPairRejected,
    FrServerError,
    FrServiceNotReady,
    FrUnavailable,
    FrWritePolicyError,
)
from dedup_pipeline.face_recognition.client import classify_bad_request, encode_multipart

from .conftest import make_image
from .fr_emulator import (
    API_KEY, FrEmulator, compare_rejection, compare_success, openapi_document, search_success, start_emulator,
)


@pytest.fixture
def emulator():
    httpd, url = start_emulator()
    yield url, FrEmulator
    httpd.shutdown()
    httpd.server_close()


def client_for(url: str, **kw) -> FrDopamsClient:
    options = dict(connect_timeout=2, read_timeout=2, retries=0, backoff=0, allow_audit_log_writes=True, api_key=API_KEY)
    options.update(kw)
    return FrDopamsClient(FrConfig(base_url=url, **options))


# ---------------------------------------------------------------- configuration

def test_base_url_has_no_default():
    with pytest.raises(FrConfigError, match="FR_DOPAMS_BASE_URL"):
        FrConfig.from_env({})


def test_config_from_environment_and_validation():
    cfg = FrConfig.from_env(
        {"FR_DOPAMS_BASE_URL": "http://fr.test:8000", "FR_DOPAMS_READ_TIMEOUT": "9", "FR_DOPAMS_RETRIES": "1",
         "FR_DOPAMS_ALLOW_AUDIT_LOG_WRITES": "yes"}
    )
    assert (cfg.read_timeout, cfg.retries, cfg.allow_audit_log_writes) == (9.0, 1, True)
    assert FrConfig.from_env({"FR_DOPAMS_BASE_URL": "http://fr.test"}).allow_audit_log_writes is False
    for bad in ("fr.test:8000", "ftp://fr.test", "http://"):
        with pytest.raises(FrConfigError):
            FrConfig(base_url=bad)
    with pytest.raises(FrConfigError):
        FrConfig.from_env({"FR_DOPAMS_BASE_URL": "http://x", "FR_DOPAMS_RETRIES": "many"})


# ------------------------------------------------------------------ /health

def test_health_ok_and_request_shape(emulator):
    url, server = emulator
    assert client_for(url).health() == {"status": "ok"}
    request = server.requests[0]
    assert (request["method"], request["path"]) == ("GET", "/health")
    assert "authorization" not in {k.lower() for k in request["headers"]}  # the service has no auth


def test_health_with_unexpected_body_is_invalid_response(emulator):
    url, server = emulator
    server.health_body = {"status": "degraded"}
    with pytest.raises(FrInvalidResponse):
        client_for(url).health()


def test_another_service_answering_ok_on_the_same_port_is_not_accepted_as_fr_dopams(emulator):
    url, server = emulator
    # Observed for real: an unrelated application on 127.0.0.1:8000 returns status "ok" plus its own fields.
    server.health_body = {"status": "ok", "service": "Some Other Service", "version": "1.0.0", "database": "sqlite"}
    with pytest.raises(FrInvalidResponse, match="may not be FR-DOPAMS"):
        client_for(url).health()
    with pytest.raises(FrInvalidResponse):
        client_for(url).verify_service()


def test_verify_service_accepts_the_real_shape_and_makes_only_get_requests(emulator):
    url, server = emulator
    client_for(url).verify_service()
    assert [(r["method"], r["path"]) for r in server.requests] == [("GET", "/health"), ("GET", "/openapi.json")]


@pytest.mark.parametrize(
    "spec",
    [
        openapi_document(title="Interrogation Report Extraction Service"),  # a different application
        {"info": {"title": "Face Enrollment and Matching API"}, "paths": {}},  # /compare missing: an older FR-DOPAMS
        {**openapi_document(), "components": {"schemas": {"Body_compare_compare_post": {"properties": {"file_a": {}}, "required": ["file_a"]}}}},
    ],
)
def test_verify_service_detects_a_wrong_service_or_api_drift(emulator, spec):
    url, server = emulator
    server.openapi = spec
    with pytest.raises(FrContractError):
        client_for(url).verify_service()


def test_verify_service_can_check_other_endpoints(emulator):
    url, server = emulator
    client_for(url).verify_service({"/search": ("file",), "/compare": ("file_a", "file_b")})
    server.openapi = openapi_document(file_field=False)  # /search lost its 'file' field
    with pytest.raises(FrContractError, match="/search"):
        client_for(url).verify_service({"/search": ("file",)})


# ----------------------------------------------------------------- /compare

def test_compare_request_shape(emulator):
    url, server = emulator
    a, b = make_image("JPEG"), make_image("PNG")
    client_for(url).compare(a, b)
    request = server.requests[0]
    assert (request["method"], request["path"]) == ("POST", "/compare")
    assert request["fields"] == ["file_a", "file_b"]
    assert request["payloads"] == {"file_a": a, "file_b": b}
    assert request["filenames"] == ["image.jpg", "image.png"]  # neutral names; nothing identifying is sent
    assert request["headers"]["X-API-Key"] == API_KEY
    assert request["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")


def test_compare_returns_the_raw_cosine_and_the_services_decision(emulator):
    url, server = emulator
    server.compare_script = [(200, compare_success(0.6312, 0.45))]
    result = client_for(url).compare(make_image("JPEG"), make_image("JPEG"))
    assert (result.similarity, result.metric, result.same_person, result.threshold) == (0.6312, "cosine", True, 0.45)
    assert result.threshold_name == "PERSON_MATCH_THRESHOLD" and result.model == "buffalo_l"


@pytest.mark.parametrize("similarity", [-1.0, -0.2, 0.0, 0.05, 1.0])
def test_compare_keeps_negative_and_boundary_similarities_exactly(emulator, similarity):
    url, server = emulator
    server.compare_script = [(200, compare_success(similarity))]
    assert client_for(url).compare(make_image("JPEG"), make_image("JPEG")).similarity == similarity


def test_compare_rejections_list_every_unusable_side(emulator):
    url, server = emulator
    server.compare_script = [(422, compare_rejection(a="no_face", b="low_quality:too_small"))]
    with pytest.raises(FrPairRejected) as caught:
        client_for(url).compare(make_image("JPEG"), make_image("JPEG"))
    rejection = caught.value
    assert [(r.side, r.kind, r.reason) for r in rejection.rejections] == [("a", "no_face", None), ("b", "low_quality", "too_small")]
    assert rejection.for_side("a").kind == "no_face" and rejection.for_side("c") is None
    server.compare_script = [(422, compare_rejection(b="multiple_faces"))]
    with pytest.raises(FrPairRejected) as caught:
        client_for(url).compare(make_image("JPEG"), make_image("JPEG"))
    assert [r.side for r in caught.value.rejections] == ["b"]


def test_a_422_without_side_errors_is_a_contract_problem_not_a_face_verdict(emulator):
    url, server = emulator
    server.compare_script = [(422, {"detail": [{"loc": ["body", "file_b"], "msg": "Field required"}]})]
    with pytest.raises(FrContractError):
        client_for(url).compare(make_image("JPEG"), make_image("JPEG"))


def test_wrong_or_missing_key_is_an_auth_error_and_not_a_face_verdict(emulator):
    url, server = emulator
    for key in ("", "wrong-key"):
        with pytest.raises(FrAuthError):
            client_for(url, api_key=key).compare(make_image("JPEG"), make_image("JPEG"))
    assert "X-API-Key" not in {k for k in server.requests[0]["headers"]}  # no key configured, none sent


def test_disabled_endpoint_is_reported_once_without_retries(emulator):
    url, server = emulator
    server.compare_script = [(503, {"success": False, "errors": [{"code": "compare_disabled"}]})] * 3
    with pytest.raises(FrAuthError, match="disabled"):
        client_for(url, retries=2).compare(make_image("JPEG"), make_image("JPEG"))
    assert len(server.requests) == 1


def test_gateway_503_is_retried_and_ends_unavailable(emulator):
    url, server = emulator
    server.compare_script = [(503, {"detail": "starting"})] * 4
    with pytest.raises(FrUnavailable):
        client_for(url, retries=2).compare(make_image("JPEG"), make_image("JPEG"))
    assert len(server.requests) == 3
    server.requests.clear()
    server.compare_script = [(502, {}), (200, compare_success(0.5))]
    assert client_for(url, retries=2).compare(make_image("JPEG"), make_image("JPEG")).similarity == 0.5


@pytest.mark.parametrize("status,error", [(500, FrServerError), (413, FrContractError), (404, FrContractError)])
def test_compare_operational_statuses(emulator, status, error):
    url, server = emulator
    server.compare_script = [(status, {"success": False, "errors": [{"code": "internal_error"}]})]
    with pytest.raises(error):
        client_for(url).compare(make_image("JPEG"), make_image("JPEG"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(similarity=1.5),  # outside [-1, 1]
        lambda d: d.update(similarity="0.5"),
        lambda d: d.update(metric="euclidean"),
        lambda d: d.update(same_person="yes"),
        lambda d: d.update(status="matched"),
        lambda d: d.pop("threshold"),
    ],
)
def test_malformed_compare_bodies_are_invalid_responses(emulator, mutate):
    url, server = emulator
    body = compare_success()
    mutate(body["data"])
    server.compare_script = [(200, body)]
    with pytest.raises(FrInvalidResponse):
        client_for(url).compare(make_image("JPEG"), make_image("JPEG"))


def test_compare_never_logs_images_keys_or_response_bodies(emulator, caplog):
    url, server = emulator
    caplog.set_level(logging.DEBUG)
    image = make_image("JPEG")
    server.compare_script = [(503, {"detail": "x"}), (200, compare_success(0.7777))]
    client_for(url, retries=1).compare(image, image)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert API_KEY not in text and "0.7777" not in text and repr(image[:16]) not in text and "/compare" in text


def test_api_key_is_not_in_the_config_repr_or_environment_error():
    config = FrConfig.from_env({"FR_DOPAMS_BASE_URL": "http://fr.test", "FR_DOPAMS_API_KEY": "super-secret-key-value"})
    assert config.api_key == "super-secret-key-value" and "super-secret" not in repr(config) and "super-secret" not in str(config)


def test_health_works_under_a_base_path_prefix(emulator):
    url, server = emulator
    client = FrDopamsClient(FrConfig(base_url=url + "/fr/", retries=0))
    with pytest.raises(FrInvalidResponse):  # the emulator only serves /health, so /fr/health is its 404 body
        client.health()
    assert server.requests[-1]["path"] == "/fr/health"  # the prefix reaches the request line


# ------------------------------------------------------------------ /search

def test_search_sends_one_multipart_file_field_and_parses_hits(emulator):
    url, server = emulator
    image = make_image("JPEG")
    result = client_for(url).search(image)
    request = server.requests[0]
    assert (request["method"], request["path"], request["field"]) == ("POST", "/search", "file")
    assert request["part_type"] == "image/jpeg" and request["filename"] == "image.jpg"
    assert request["payload"] == image
    assert request["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert [(h.person_id, h.similarity) for h in result.hits] == [("CRIM000001", 0.8123)] and result.threshold == 0.4


def test_search_with_no_hits_is_still_a_success(emulator):
    url, server = emulator
    server.script = [(200, search_success(hits=()))]
    assert client_for(url).search(make_image("PNG")).hits == ()
    assert server.requests[0]["part_type"] == "image/png"


def test_gallery_person_details_are_discarded(emulator):
    url, _ = emulator
    result = client_for(url).search(make_image("JPEG"))
    assert "SENSITIVE" not in repr(result)
    assert not hasattr(result.hits[0], "name")


def test_search_is_refused_without_explicit_permission_and_sends_nothing(emulator):
    url, server = emulator
    with pytest.raises(FrWritePolicyError, match="match_logs"):
        client_for(url, allow_audit_log_writes=False).search(make_image("JPEG"))
    assert server.requests == []


@pytest.mark.parametrize(
    "detail,kind",
    [
        ("No face detected in the image.", "no_face"),
        ("2 faces of similar size detected. Crop the image to the one person to be matched.", "multiple_faces"),
        ("Face is too small (30px; minimum 40px). Use a closer or higher-resolution image.", "low_quality"),
        ("Face is too blurred to match reliably. Use a sharper image.", "low_quality"),
        ("Unable to decode image bytes.", "invalid_image"),
        ("Empty image data provided.", "invalid_image"),
        ("Image is larger than the 10 MB limit.", "invalid_image"),
    ],
)
def test_400_image_rejections_are_face_problems_not_operational_errors(emulator, detail, kind):
    url, server = emulator
    server.script = [(400, {"detail": detail})]
    with pytest.raises(FrImageRejected) as caught:
        client_for(url).search(make_image("JPEG"))
    assert caught.value.kind == kind


def test_empty_gallery_400_is_service_not_ready_not_no_face(emulator):
    url, server = emulator
    server.script = [(400, {"detail": "No faces are enrolled yet. Enroll faces before matching."})]
    with pytest.raises(FrServiceNotReady):
        client_for(url).search(make_image("JPEG"))


def test_unrecognised_400_is_a_contract_error_not_a_face_verdict(emulator):
    url, server = emulator
    server.script = [(400, {"detail": "Something new FR-DOPAMS started saying"})]
    with pytest.raises(FrContractError):
        client_for(url).search(make_image("JPEG"))


@pytest.mark.parametrize("status", [401, 404, 413, 422])
def test_unexpected_statuses_are_contract_errors(emulator, status):
    url, server = emulator
    server.script = [(status, {"detail": "nope"})]
    with pytest.raises(FrContractError, match=str(status)):
        client_for(url).search(make_image("JPEG"))


def test_fastapi_validation_errors_keep_field_and_message_only():
    body = b'{"detail":[{"type":"missing","loc":["body","file"],"msg":"Field required","input":"SECRET"}]}'
    assert FrDopamsClient._error_detail(body) == "body.file: Field required"


def test_500_is_a_server_error(emulator):
    url, server = emulator
    server.script = [(500, {"detail": "Search request failed: connection to server lost"})]
    with pytest.raises(FrServerError):
        client_for(url).search(make_image("JPEG"))
    assert len(server.requests) == 1  # not retried: a deterministic 500 would only repeat


@pytest.mark.parametrize("body", [{"success": False}, {"success": True}, {"success": True, "results": [{"similarity": "high"}]}, [1]])
def test_200_with_the_wrong_shape_is_an_invalid_response(emulator, body):
    url, server = emulator
    server.script = [(200, body)]
    with pytest.raises(FrInvalidResponse):
        client_for(url).search(make_image("JPEG"))


def test_200_with_a_non_json_body_is_an_invalid_response(emulator):
    url, server = emulator
    server.script = [(200, b"<html>proxy page</html>")]
    with pytest.raises(FrInvalidResponse):
        client_for(url).search(make_image("JPEG"))


# ------------------------------------------------- outages and retries

def test_service_down_is_unavailable(emulator):
    url, server = emulator
    with pytest.raises(FrUnavailable):
        client_for("http://127.0.0.1:9", retries=1, backoff=0).health()


def test_503_is_retried_then_succeeds(emulator):
    url, server = emulator
    server.script = [(503, {"detail": "starting"}), (502, {"detail": "bad gateway"}), (200, search_success())]
    result = client_for(url, retries=2).search(make_image("JPEG"))
    assert len(result.hits) == 1 and len(server.requests) == 3


def test_persistent_503_ends_as_unavailable_after_all_attempts(emulator):
    url, server = emulator
    server.script = [(503, {"detail": "down"})] * 5
    with pytest.raises(FrUnavailable):
        client_for(url, retries=2).search(make_image("JPEG"))
    assert len(server.requests) == 3


def test_read_timeout_is_unavailable(emulator):
    url, server = emulator
    server.delay = 0.6
    with pytest.raises(FrUnavailable):
        client_for(url, read_timeout=0.1, retries=0).search(make_image("JPEG"))


def test_a_400_is_never_retried(emulator):
    url, server = emulator
    server.script = [(400, {"detail": "No face detected in the image."})] * 3
    with pytest.raises(FrImageRejected):
        client_for(url, retries=3).search(make_image("JPEG"))
    assert len(server.requests) == 1


# --------------------------------------------------------------- hygiene

def test_nothing_sensitive_is_logged(emulator, caplog):
    url, server = emulator
    caplog.set_level(logging.DEBUG)
    image = make_image("JPEG")
    server.script = [(503, {"detail": "x"}), (200, search_success())]
    client_for(url, retries=1).search(image)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SENSITIVE" not in text and "CRIM000001" not in text and repr(image[:16]) not in text
    assert "/search" in text  # method, path and status are logged


def test_config_repr_never_echoes_credentials_embedded_in_the_url():
    text = repr(FrConfig(base_url="http://user:hunter2@fr.test:8000/api"))
    assert "hunter2" not in text and "user" not in text and "fr.test:8000/api" in text


def test_classify_bad_request_and_multipart_helpers():
    assert classify_bad_request("") == "unrecognised"
    body, content_type = encode_multipart("file", b"\x00\x01", "image/png", "image.png")
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(f"\r\n--{boundary}--\r\n".encode())
    assert b'name="file"; filename="image.png"' in body and b"\x00\x01" in body
