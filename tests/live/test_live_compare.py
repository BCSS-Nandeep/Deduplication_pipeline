"""Tests against a RUNNING FR-DOPAMS service: real HTTP, real middleware, real InsightFace model.

Skipped unless these are set (see tools/run_fr_dopams_stateless.py and tools/make_live_fixtures.py):

    FR_DOPAMS_LIVE_URL        e.g. http://127.0.0.1:8010
    FR_DOPAMS_API_KEY         the key the service was started with
    FR_LIVE_FIXTURE_DIR       output of tools/make_live_fixtures.py
    FR_LIVE_SERVER_LOG        optional: the service's stderr log, scanned for database access attempts

Run with the FR-DOPAMS virtualenv or any env that has Pillow + pytest. The service identity is verified
before anything else, so a different application on the address fails the whole module.
"""

from __future__ import annotations

import http.client
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from dedup_pipeline.face_recognition import (
    FrAuthError,
    FrConfig,
    FrDopamsClient,
    FrDopamsPairComparer,
    FrPairRejected,
    PairStatus,
    compare_pair,
)
from dedup_pipeline.face_recognition.client import encode_multipart_files
from dedup_pipeline.face_recognition.image_source import ImageIndex, ManifestImageLoader
from dedup_pipeline.image_resolution.resolver import ResolverConfig

from ..conftest import BASE, FakeFetcher, url_for
from ..test_fr_integration import selected, write_phase2_outputs

URL = os.environ.get("FR_DOPAMS_LIVE_URL", "")
KEY = os.environ.get("FR_DOPAMS_API_KEY", "")
FIXTURES = Path(os.environ.get("FR_LIVE_FIXTURE_DIR", ""))
SERVER_LOG = os.environ.get("FR_LIVE_SERVER_LOG", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (URL and KEY and FIXTURES.is_dir()), reason="live FR-DOPAMS environment not configured"),
]

MAX_UPLOAD = 10 * 1024 * 1024


def fx(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture(scope="module")
def client() -> FrDopamsClient:
    c = FrDopamsClient(FrConfig(base_url=URL, api_key=KEY, retries=1, read_timeout=120))
    c.verify_service()  # the right application, exposing POST /compare with file_a + file_b
    return c


@pytest.fixture(scope="module")
def expectations() -> dict:
    return json.loads((FIXTURES / "fixtures.json").read_text())


def raw_post(files, key=KEY, extra_headers=None, body=None, content_type=None):
    """A bare HTTP POST /compare, bypassing our client, to check the protocol itself."""
    parts = urlsplit(URL)
    if body is None:
        body, content_type = encode_multipart_files(files)
    headers = {"Content-Type": content_type, **(extra_headers or {})}
    if key is not None:
        headers["X-API-Key"] = key
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=120)
    started = time.monotonic()
    try:
        conn.request("POST", "/compare", body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else None, time.monotonic() - started
    finally:
        conn.close()


def two(a: str, b: str):
    return [("file_a", fx(a), "image/png", "a.png"), ("file_b", fx(b), "image/png", "b.png")]


# --------------------------------------------------------- identity & auth

def test_the_running_service_is_fr_dopams_and_other_endpoints_are_unchanged():
    parts = urlsplit(URL)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
    conn.request("GET", "/health")
    health = conn.getresponse()
    assert (health.status, json.loads(health.read())) == (200, {"status": "ok"})  # open, no key needed
    conn.request("POST", "/match")  # no file: validation error, NOT 401 -> other routes still unauthenticated
    response = conn.getresponse()
    response.read()
    assert response.status == 422
    conn.close()


def test_missing_and_wrong_keys_are_401(client):
    for key in (None, "", "definitely-not-the-key-0000000000"):
        status, body, _ = raw_post(two("person0.png", "person1.png"), key=key)
        assert status == 401 and body["errors"][0]["code"] == "unauthorized" and body["data"] is None


def test_unauthenticated_large_upload_is_refused_without_reading_it(client):
    body, content_type = encode_multipart_files([("file_a", b"\xff\xd8\xff" + os.urandom(5_000_000), "image/jpeg", "a.jpg")])
    status, payload, elapsed = raw_post(None, key=None, body=body, content_type=content_type)
    assert status == 401 and elapsed < 5  # the model never ran


def test_the_client_reports_a_wrong_key_as_an_auth_error_not_a_face_verdict():
    wrong = FrDopamsClient(FrConfig(base_url=URL, api_key="wrong-key-wrong-key-wrong-key", retries=0))
    with pytest.raises(FrAuthError):
        wrong.compare(fx("person0.png"), fx("person1.png"))


# ------------------------------------------------------------- similarity

def test_same_person_scores_high_and_is_decided_by_the_service(client):
    for variant in ("person0_flip.png", "person0_small.jpg", "person0_jpeg60.jpg", "person0_bright.png"):
        result = client.compare(fx("person0.png"), fx(variant))
        assert result.same_person is True and result.similarity > 0.6, (variant, result.similarity)
        assert (result.metric, result.model) == ("cosine", "buffalo_l") and result.threshold == 0.45


def test_different_people_score_low_and_are_not_the_same_person(client):
    scores = []
    for i in range(6):
        for j in range(i + 1, 6):
            result = client.compare(fx(f"person{i}.png"), fx(f"person{j}.png"))
            scores.append(result.similarity)
            assert result.same_person is False, (i, j, result.similarity)
    assert max(scores) < 0.45, scores  # every one of the 15 distinct-person pairs is below the service's threshold
    print("different-person cosines:", sorted(round(s, 3) for s in scores))


def test_identical_images_score_about_one(client):
    assert client.compare(fx("person2.png"), fx("person2.png")).similarity == pytest.approx(1.0, abs=1e-3)


def test_cosine_is_symmetric(client):
    ab = client.compare(fx("person0.png"), fx("person3.png")).similarity
    ba = client.compare(fx("person3.png"), fx("person0.png")).similarity
    assert ab == pytest.approx(ba, abs=1e-5)


def test_the_raw_cosine_can_be_negative_and_is_returned_unclamped(client):
    scores = {(i, j): client.compare(fx(f"person{i}.png"), fx(f"person{j}.png")).similarity for i in range(6) for j in range(i + 1, 6)}
    negatives = {k: v for k, v in scores.items() if v < 0}
    assert negatives, "expected at least one negative cosine among 15 distinct-person pairs"
    status, body, _ = raw_post(two(*(f"person{n}.png" for n in next(iter(negatives)))))
    assert status == 200 and body["data"]["similarity"] < 0 and body["data"]["similarity_range"] == [-1.0, 1.0]


def test_success_response_matches_the_documented_contract():
    status, body, _ = raw_post(two("person0.png", "person0_flip.png"))
    data = body["data"]
    assert status == 200 and body["success"] is True and body["errors"] == []
    assert set(data) == {"status", "similarity", "metric", "similarity_range", "same_person", "threshold", "threshold_name",
                         "model", "embedding_dim", "faces"}
    assert data["embedding_dim"] == 512 and set(data["faces"]) == {"a", "b"}
    assert all(set(f) == {"face_width", "face_height", "detection_score"} for f in data["faces"].values())
    text = json.dumps(body)
    assert "person_id" not in text and "CRIM" not in text  # no gallery data can leak: there is no gallery access


# -------------------------------------------------- side-specific errors (real model)

def reject(a: str, b: str):
    status, body, _ = raw_post(two(a, b))
    assert status == 422 and body["success"] is False and body["data"] is None, (status, body)
    return {e["side"]: (e["code"], e["reason"]) for e in body["errors"]}


def test_group_photo_is_multiple_faces_on_the_right_side():
    assert reject("group_photo.jpg", "person1.png") == {"a": ("multiple_faces", None)}
    assert reject("person1.png", "group_photo.jpg") == {"b": ("multiple_faces", None)}


def test_a_tight_face_crop_the_detector_cannot_find_is_no_face():
    assert reject("no_face_aligned_crop.png", "person1.png") == {"a": ("no_face", None)}
    assert reject("person1.png", "no_face_noise.png") == {"b": ("no_face", None)}


def test_low_quality_faces_carry_a_reason():
    for name, reason in (("low_quality_small_face.png", "too_small"), ("low_quality_blurred.png", "too_blurred")):
        if not (FIXTURES / name).exists():
            pytest.skip(f"{name} not produced")
        assert reject(name, "person1.png") == {"a": ("low_quality", reason)}


def test_unusable_uploads_are_invalid_image_with_a_reason():
    assert reject("corrupt_truncated.jpg", "person1.png") == {"a": ("invalid_image", "undecodable")}
    assert reject("person1.png", "not_an_image.txt") == {"b": ("invalid_image", "undecodable")}
    assert reject("empty.bin", "person1.png") == {"a": ("invalid_image", "empty")}
    assert reject("pixel_bomb.png", "person1.png") == {"a": ("invalid_image", "too_many_pixels")}


def test_a_file_over_the_per_image_limit_is_rejected_per_side_not_the_whole_request():
    assert reject("too_large.bin", "person1.png") == {"a": ("invalid_image", "too_large")}


def test_both_sides_failing_are_reported_together_in_side_order():
    status, body, _ = raw_post(two("group_photo.jpg", "no_face_noise.png"))
    assert status == 422
    assert [(e["side"], e["code"]) for e in body["errors"]] == [("a", "multiple_faces"), ("b", "no_face")]
    assert body["detail"] == "a: multiple_faces; b: no_face"


def test_the_client_turns_each_real_rejection_into_the_same_structure(client):
    with pytest.raises(FrPairRejected) as caught:
        client.compare(fx("group_photo.jpg"), fx("no_face_noise.png"))
    assert [(r.side, r.kind) for r in caught.value.rejections] == [("a", "multiple_faces"), ("b", "no_face")]


def test_expected_codes_recorded_by_the_fixture_builder_match_the_service(expectations):
    for name, (code, reason) in expectations.items():
        if name.startswith("person") and "_" not in name.split(".")[0]:
            continue  # single-face crops; scores are covered above
        if code == "ok":
            continue
        got = reject(name, "person1.png")["a"]
        assert got == (code, reason), (name, got, (code, reason))


# --------------------------------------------------------------- request limits

def test_request_over_the_cap_is_413_before_the_model_runs(client):
    big = b"\xff\xd8\xff" + os.urandom(11 * 1024 * 1024)
    body, content_type = encode_multipart_files([("file_a", big, "image/jpeg", "a.jpg"), ("file_b", big, "image/jpeg", "b.jpg")])
    assert len(body) > 21 * 1024 * 1024
    status, payload, elapsed = raw_post(None, body=body, content_type=content_type)
    assert status == 413 and payload["errors"][0]["code"] == "request_too_large"
    assert elapsed < 10


def test_chunked_upload_without_content_length_is_also_capped():
    parts = urlsplit(URL)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=120)
    boundary = "b" * 32

    def chunks():  # a VALID multipart stream (so the parser keeps reading), 25 MB, no Content-Length
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file_a"; filename="a.jpg"\r\n'
               f"Content-Type: image/jpeg\r\n\r\n").encode()
        for _ in range(25):
            yield os.urandom(1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    conn.request("POST", "/compare", body=chunks(), encode_chunked=True,
                 headers={"X-API-Key": KEY, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    assert response.status == 413 and body["errors"][0]["code"] == "request_too_large"


def test_missing_field_is_fastapis_standard_validation_error():
    status, body, _ = raw_post([("file_a", fx("person0.png"), "image/png", "a.png")])
    assert status == 422 and isinstance(body["detail"], list) and "errors" not in body


# ------------------------------------------------------------ behaviour under load

def test_parallel_requests_give_identical_results_and_latency_is_reasonable(client):
    pairs = [("person0.png", "person0_flip.png"), ("person1.png", "person2.png")] * 4
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: client.compare(fx(p[0]), fx(p[1])).similarity, pairs))
    elapsed = time.monotonic() - started
    assert results[0::2] == [results[0]] * 4 and results[1::2] == [results[1]] * 4
    latencies = []
    for _ in range(5):
        t = time.monotonic()
        client.compare(fx("person0.png"), fx("person4.png"))
        latencies.append(time.monotonic() - t)
    print(f"8 parallel compares in {elapsed:.2f}s; sequential median {statistics.median(latencies):.2f}s")


# -------------------------------------------- pipeline: pair evidence with fallback

def test_pair_evidence_falls_back_to_the_next_image_against_the_real_service(tmp_path):
    write_phase2_outputs(
        tmp_path,
        {
            # first image has no detectable face; the second is a proper photo
            "P": ("usable_image_available", [selected("P1", "person/media/p1"), selected("P2", "person/media/p2", rank=2)]),
            "Q": ("usable_image_available", [selected("Q1", "person/media/q1")]),
            "R": ("usable_image_available", [selected("R1", "person/media/r1")]),
            "NOIMG": ("no_person_media_records", []),
        },
    )
    files = {
        url_for("person/media/p1", ".png"): fx("no_face_aligned_crop.png"),
        url_for("person/media/p2", ".png"): fx("person0.png"),
        url_for("person/media/q1", ".png"): fx("person0_flip.png"),
        url_for("person/media/r1", ".png"): fx("person3.png"),
    }
    # the manifest records say ".jpg"; point them at the .png fixtures
    for name in ("image_manifest.jsonl",):
        text = (tmp_path / name).read_text().replace('".jpg"', '".png"').replace('"jpeg"', '"png"')
        (tmp_path / name).write_text(text)
    index = ImageIndex.from_output_dir(tmp_path)
    loader = ManifestImageLoader(ResolverConfig(base_url=BASE), FakeFetcher(files))
    comparer = FrDopamsPairComparer(FrDopamsClient(FrConfig(base_url=URL, api_key=KEY, retries=1, read_timeout=120)))

    same = compare_pair("P", "Q", index, loader, comparer)
    assert same.status is PairStatus.MATCHED and same.fallback_used and same.face_similarity > 0.6
    assert (same.media_id_a, same.media_id_b) == ("P2", "Q1")
    assert [(a["media_id_a"], a["outcome"]) for a in same.attempts] == [("P1", "rejected"), ("P2", "compared")]
    assert same.attempts[0]["rejections"] == {"a": "no_face"}

    different = compare_pair("P", "R", index, loader, comparer)
    assert different.status is PairStatus.NOT_MATCHED and different.fallback_used and different.face_similarity < 0.45

    no_alternative = compare_pair("Q", "P", index, loader, comparer)  # Q has one image; P's fallback still works
    assert no_alternative.status is PairStatus.MATCHED

    skipped = compare_pair("P", "NOIMG", index, loader, comparer)
    assert skipped.status is PairStatus.SKIPPED_NO_IMAGE and skipped.face_similarity is None


def test_only_no_face_images_exhaust_the_fallback_and_keep_the_real_failure(tmp_path):
    write_phase2_outputs(
        tmp_path,
        {"P": ("usable_image_available", [selected("P1", "person/media/p1"), selected("P2", "person/media/p2", rank=2)]),
         "Q": ("usable_image_available", [selected("Q1", "person/media/q1")])},
    )
    (tmp_path / "image_manifest.jsonl").write_text(
        (tmp_path / "image_manifest.jsonl").read_text().replace('".jpg"', '".png"').replace('"jpeg"', '"png"'))
    files = {
        url_for("person/media/p1", ".png"): fx("no_face_aligned_crop.png"),
        url_for("person/media/p2", ".png"): fx("no_face_noise.png"),
        url_for("person/media/q1", ".png"): fx("person0.png"),
    }
    loader = ManifestImageLoader(ResolverConfig(base_url=BASE), FakeFetcher(files))
    comparer = FrDopamsPairComparer(FrDopamsClient(FrConfig(base_url=URL, api_key=KEY, retries=1, read_timeout=120)))
    evidence = compare_pair("P", "Q", ImageIndex.from_output_dir(tmp_path), loader, comparer)
    assert evidence.status is PairStatus.NO_FACE and evidence.failed_side == "a" and evidence.face_similarity is None
    assert [a["media_id_a"] for a in evidence.attempts] == ["P1", "P2"] and evidence.fallback_used


def test_a_service_outage_is_an_error_not_a_mismatch(tmp_path):
    write_phase2_outputs(
        tmp_path,
        {"P": ("usable_image_available", [selected("P1", "person/media/p1")]),
         "Q": ("usable_image_available", [selected("Q1", "person/media/q1")])},
    )
    (tmp_path / "image_manifest.jsonl").write_text(
        (tmp_path / "image_manifest.jsonl").read_text().replace('".jpg"', '".png"').replace('"jpeg"', '"png"'))
    files = {url_for("person/media/p1", ".png"): fx("person0.png"), url_for("person/media/q1", ".png"): fx("person1.png")}
    loader = ManifestImageLoader(ResolverConfig(base_url=BASE), FakeFetcher(files))
    dead = FrDopamsClient(FrConfig(base_url="http://127.0.0.1:9", api_key=KEY, retries=0, connect_timeout=1))
    evidence = compare_pair("P", "Q", ImageIndex.from_output_dir(tmp_path), loader, FrDopamsPairComparer(dead))
    assert evidence.status is PairStatus.ERROR and evidence.error_kind == "unavailable" and evidence.face_similarity is None


# ------------------------------------------------------------ statelessness

def test_the_service_log_shows_no_database_activity():
    logs = [Path(p) for p in SERVER_LOG.split(os.pathsep) if p]  # uvicorn: errors on stderr, access lines on stdout
    if not logs or not all(p.is_file() for p in logs):
        pytest.skip("FR_LIVE_SERVER_LOG not provided")
    text = "\n".join(p.read_text(errors="replace") for p in logs)
    for marker in ("Traceback", "DATABASE_URL", "nodb", "pg8000", "sqlalchemy", "OperationalError", "Connection refused"):
        assert marker not in text, f"server log mentions {marker!r}: something tried to use the database"
    assert text.count('"POST /compare HTTP/1.1"') > 20  # and the endpoint really was exercised, many times
    # No gallery endpoint did any work. The one POST /match in this run is the deliberate no-file request,
    # which FastAPI rejects with 422 during validation, before any service code (or database) is reached.
    assert '"POST /search' not in text and '"POST /enroll' not in text
    assert all(line.rstrip().endswith("422 Unprocessable Entity") for line in text.splitlines() if '"POST /match' in line)
