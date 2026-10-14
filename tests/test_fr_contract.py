"""Check the client against FR-DOPAMS's actual source code (no service or model needed).

The source is parsed with ``ast``; nothing from FR-DOPAMS is imported or executed. These tests fail
loudly if the service changes a route, a form field, a response key or an error message this client
depends on. Skipped when the FR-DOPAMS checkout is not present (override with FR_DOPAMS_SOURCE_DIR).
Behaviour against the *running* service is covered by tests/live/test_live_compare.py.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from dedup_pipeline.face_recognition.client import classify_bad_request
from dedup_pipeline.face_recognition.errors import IMAGE_KINDS

FR_DIR = Path(os.environ.get("FR_DOPAMS_SOURCE_DIR") or Path(__file__).resolve().parents[2] / "FR-DOPAMS")
pytestmark = pytest.mark.skipif(not (FR_DIR / "app.py").is_file(), reason="FR-DOPAMS source not found")

HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


def _routes() -> dict[tuple[str, str], dict]:
    """(METHOD, path) -> {"uploads": [...], "forms": [...], "calls": {...}} for every route in app.py and api/*.py."""
    found: dict[tuple[str, str], dict] = {}
    for path in [FR_DIR / "app.py", *sorted((FR_DIR / "api").glob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {  # module-level NAME = "string", so @router.post(COMPARE_PATH) resolves
            t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str) for t in n.targets if isinstance(t, ast.Name)
        }
        prefix = ""
        for node in ast.walk(tree):  # APIRouter(prefix="/admin")
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "APIRouter":
                for kw in node.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        prefix = kw.value.value
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in fn.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in HTTP_METHODS:
                    first = dec.args[0] if dec.args else None
                    value = first.value if isinstance(first, ast.Constant) else constants.get(getattr(first, "id", ""))
                    route = prefix + value if value is not None else None
                    args = fn.args.args
                    uploads = [a.arg for a in args if "UploadFile" in ast.unparse(a.annotation or ast.Constant(""))]
                    forms = [a.arg for a, d in zip(args[-len(fn.args.defaults):] if fn.args.defaults else [], fn.args.defaults)
                             if "Form" in ast.unparse(d)]
                    calls = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
                    found[(dec.func.attr.upper(), route)] = {"uploads": uploads, "forms": forms, "calls": calls, "file": path.name}
    return found


ROUTES = _routes() if (FR_DIR / "app.py").is_file() else {}


def _source(*parts: str) -> str:
    return (FR_DIR.joinpath(*parts)).read_text(encoding="utf-8")


# ----------------------------------------------------------- routes and fields

def test_endpoints_the_client_calls_exist_with_the_expected_shape():
    assert ROUTES[("GET", "/health")]["uploads"] == []
    compare = ROUTES[("POST", "/compare")]
    assert compare["uploads"] == ["file_a", "file_b"] and compare["forms"] == []
    assert ROUTES[("POST", "/search")]["uploads"] == ["file"]


def test_only_compare_is_authenticated_and_it_checks_the_key_by_header():
    for path in sorted((FR_DIR / "api").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert ("APIKeyHeader" in text) == (path.name == "compare_api.py"), path.name
    compare = _source("api", "compare_api.py")
    assert 'name="X-API-Key"' in compare and "hmac.compare_digest" in compare and "COMPARE_API_KEY" in compare
    assert "CompareGuardMiddleware" in _source("app.py")


def test_compare_is_stateless_and_separate_from_the_gallery():
    compare = ROUTES[("POST", "/compare")]
    assert compare["file"] == "compare_api.py"
    assert not {c for c in compare["calls"] if "create_match_log" in c or "session_scope" in c or "enroll" in c}
    api_tree = ast.parse(_source("api", "compare_api.py"))
    imported = {n.module for n in ast.walk(api_tree) if isinstance(n, ast.ImportFrom)}
    assert not any("database" in (m or "") for m in imported)
    service = ast.parse(_source("src", "face_service.py"))
    for fn in ast.walk(service):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("compare_images", "_analyse_image", "_check_pixel_budget"):
            assert "db." not in ast.unparse(fn), fn.name


def test_no_other_endpoint_takes_two_images():
    many = {key for key, info in ROUTES.items() if len(info["uploads"]) > 1}
    assert many == {("POST", "/enroll"), ("POST", "/compare")}


# --------------------------------------------------------------- response shape

def _dict_keys_returned_by(function: str, source: str) -> set[str]:
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function)
    ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)][-1]
    return {k.value for k in ret.value.keys if isinstance(k, ast.Constant)}


def test_compare_success_body_has_the_keys_the_client_parses():
    assert _dict_keys_returned_by("compare_images", _source("src", "face_service.py")) >= {
        "status", "similarity", "metric", "same_person", "threshold", "threshold_name", "model", "similarity_range"}
    assert _dict_keys_returned_by("compare", _source("api", "compare_api.py")) == {"success", "message", "data", "errors"}


def test_error_codes_are_exactly_what_the_client_understands():
    tree = ast.parse(_source("src", "face_service.py"))
    codes = {
        n.args[0].value for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ImageRejected" and n.args and isinstance(n.args[0], ast.Constant)
    }
    assert codes == set(IMAGE_KINDS)
    api = _source("api", "compare_api.py")
    for key in ('"side"', '"code"', '"reason"', '"message"', "status_code=422"):
        assert key in api
    for code in ("unauthorized", "compare_disabled", "request_too_large", "internal_error"):
        assert code in api


def test_service_identity_facts_the_preflight_relies_on():
    app = _source("app.py")
    assert 'title="Face Enrollment and Matching API"' in app  # EXPECTED_SERVICE_TITLE
    health = app.split('@app.get("/health")', 1)[1].split("\n\n\n", 1)[0]
    assert 'return {"status": "ok"}' in health and "docs_url=None" not in app and "openapi_url=None" not in app


def test_search_response_shape_the_client_parses():
    source = _source("api", "match_api.py")
    assert '"data": result.get("results", [])' in source and '**result' in source
    service = _source("src", "face_service.py")
    for key in ('"success": True', '"results": results', '"threshold": SEARCH_THRESHOLD', '"similarity": item["similarity"]', '"person_id": item["person_id"]'):
        assert key in service


def test_every_search_error_message_the_client_classifies_still_exists_in_the_service():
    service = _source("src", "face_service.py")
    for fragment in (
        "No face detected in the image.", "faces of similar size detected", "Face is too small (",
        "Face is too blurred to match reliably.", "Unable to decode image bytes.", "Empty image data provided.",
        "Image is larger than the", "No faces are enrolled yet.",
    ):
        assert fragment in service, f"FR-DOPAMS no longer says {fragment!r}; update classify_bad_request"
    samples = {
        "No face detected in the image.": "no_face",
        "3 faces of similar size detected. Crop the image to the one person to be matched.": "multiple_faces",
        "No faces are enrolled yet. Enroll faces before matching.": "service_not_ready",
    }
    assert {k: classify_bad_request(k) for k in samples} == samples


# -------------------------------------- facts that made /compare necessary (still true)

def test_one_to_one_verification_still_needs_an_enrolled_person():
    body = _source("src", "face_service.py").split("def match_against_person", 1)[1].split("\ndef ", 1)[0]
    assert "find_person_by_id(person_id)" in body and "does not exist" in body


def test_enrollment_still_requires_three_poses_and_head_turn_checks():
    service = _source("src", "face_service.py")
    assert "Missing {pose} image for enrollment." in service and "SIDE_MIN_YAW" in service


def test_gallery_image_endpoints_still_write_an_audit_row_on_every_call():
    for key in (("POST", "/match"), ("POST", "/match/person"), ("POST", "/search")):
        assert "create_match_log" in ROUTES[key]["calls"], key
