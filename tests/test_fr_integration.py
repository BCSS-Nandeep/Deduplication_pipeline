"""FaceChecker integration, image loading from Phase 2 outputs, and per-pair face evidence with fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dedup_pipeline.face_recognition import (
    ComparisonOutcome,
    FallbackPolicy,
    FrAuthError,
    FrConfig,
    FrDopamsClient,
    FrDopamsFaceChecker,
    FrDopamsPairComparer,
    FrImageRejected,
    FrPairRejected,
    FrServiceNotReady,
    FrUnavailable,
    PairStatus,
    SideRejection,
    compare_pair,
)
from dedup_pipeline.face_recognition.client import SearchResult
from dedup_pipeline.face_recognition.image_source import ImageIndex, ImageLoadError, ManifestImageLoader
from dedup_pipeline.image_resolution.models import Status
from dedup_pipeline.image_resolution.resolver import resolve_media

from .conftest import FakeFetcher, media_row, url_for
from .fr_emulator import API_KEY, FrEmulator, compare_rejection, compare_success, start_emulator


class StubClient:
    """Stands in for FrDopamsClient.search; raises what is scripted."""

    def __init__(self, outcome=None):
        self.outcome, self.calls = outcome, 0

    def search(self, image_bytes):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SearchResult((), None)


# ------------------------------------------------------- FaceChecker hook

def _resolve_with(checker, jpeg_bytes, config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    return resolve_media(media_row(), config, fetcher, checker)


def test_checker_ok_leaves_the_image_usable(config, jpeg_bytes):
    result = _resolve_with(FrDopamsFaceChecker(StubClient()), jpeg_bytes, config)
    assert result.usable and result.face_check_status == "ok"


@pytest.mark.parametrize(
    "kind,status",
    [("no_face", Status.NO_FACE), ("multiple_faces", Status.MULTIPLE_FACES), ("low_quality", Status.LOW_QUALITY),
     ("invalid_image", Status.LOW_QUALITY)],
)
def test_checker_maps_fr_rejections_to_phase_2_statuses(config, jpeg_bytes, kind, status):
    checker = FrDopamsFaceChecker(StubClient(FrImageRejected(kind, "reason from FR-DOPAMS")))
    result = _resolve_with(checker, jpeg_bytes, config)
    assert result.status is status and not result.usable
    assert "FR-DOPAMS" in result.skip_reason


@pytest.mark.parametrize("error", [FrUnavailable("down"), FrServiceNotReady("empty gallery")])
def test_operational_failures_are_not_reported_as_face_verdicts(config, jpeg_bytes, error):
    result = _resolve_with(FrDopamsFaceChecker(StubClient(error)), jpeg_bytes, config)
    assert result.status is Status.FACE_CHECK_ERROR  # transient: retried on the next run
    assert result.face_check_status == "error"


def test_checker_is_not_called_for_images_that_already_failed_decoding(config):
    stub = StubClient()
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): b"\xff\xd8\xff" + b"x" * 90})
    result = resolve_media(media_row(), config, fetcher, FrDopamsFaceChecker(stub))
    assert result.status is Status.CORRUPT_IMAGE and stub.calls == 0


def test_checker_over_real_http_with_the_emulator(config, jpeg_bytes):
    httpd, url = start_emulator()
    try:
        client = FrDopamsClient(FrConfig(base_url=url, retries=0, allow_audit_log_writes=True))
        assert _resolve_with(FrDopamsFaceChecker(client), jpeg_bytes, config).usable
        FrEmulator.script = [(400, {"detail": "No face detected in the image."})]
        assert _resolve_with(FrDopamsFaceChecker(client), jpeg_bytes, config).status is Status.NO_FACE
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_checker_is_blocked_by_default_policy(config, jpeg_bytes):
    from dedup_pipeline.face_recognition import FrWritePolicyError

    client = FrDopamsClient(FrConfig(base_url="http://127.0.0.1:9"))  # allow_audit_log_writes defaults to False
    with pytest.raises(FrWritePolicyError):
        FrDopamsFaceChecker(client)(jpeg_bytes)
    assert _resolve_with(FrDopamsFaceChecker(client), jpeg_bytes, config).status is Status.FACE_CHECK_ERROR


# ------------------------------------------------------- Phase 2 outputs

def selected(media_id, path, suffix=".jpg", typ="jpeg", rank=1):
    return {"media_id": media_id, "path_ref": path, "resolved_suffix": suffix, "detected_type": typ, "person_rank": rank}


def write_phase2_outputs(directory: Path, people: dict[str, tuple[str, list[dict]]]) -> None:
    """people: person_id -> (person_status, usable manifest records in rank order)."""
    manifest, summaries = [], []
    for person_id, (status, records) in people.items():
        for record in records:
            manifest.append({"person_id": person_id, "usable_for_face_recognition": True,
                             "selected_for_person": record["person_rank"] == 1, **record})
        manifest.append({"person_id": person_id, "media_id": f"X-{person_id}", "usable_for_face_recognition": False,
                         "selected_for_person": False, "person_rank": 99, "path_ref": None, "resolved_suffix": None})
        summaries.append({"person_id": person_id, "person_status": status,
                          "unusable_reasons": {"pdf_skipped": 1} if status == "no_usable_image" else {}})
    (directory / "image_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    (directory / "person_summary.jsonl").write_text("\n".join(json.dumps(s) for s in summaries) + "\n")


class ScriptedComparer:
    """Plays one outcome per call; an Exception is raised, a ComparisonOutcome returned."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def compare(self, image_a, image_b):
        self.calls.append((image_a, image_b))
        outcome = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def rejected(**sides):
    return FrPairRejected(tuple(SideRejection(s, spec.partition(":")[0], spec.partition(":")[2] or None, "detail") for s, spec in sorted(sides.items())))


OK = ComparisonOutcome(0.4321, "cosine", decision=True, threshold=0.45)


@pytest.fixture
def world(tmp_path, config, jpeg_bytes, png_bytes):
    write_phase2_outputs(
        tmp_path,
        {
            "A": ("usable_image_available", [selected("A1", "person/media/a1"), selected("A2", "person/media/a2", rank=2),
                                              selected("A3", "person/media/a3", rank=3), selected("A4", "person/media/a4", rank=4)]),
            "B": ("usable_image_available", [selected("B1", "person/media/b1", ".png", "png"),
                                              selected("B2", "person/media/b2", ".png", "png", rank=2)]),
            "C": ("usable_image_available", [selected("C1", "person/media/c1")]),
            "NOIMG": ("no_person_media_records", []),
            "PDFONLY": ("no_usable_image", []),
            "GONE": ("usable_image_available", [selected("MG", "person/media/gone")]),
        },
    )
    files = {url_for(f"person/media/{n}", ".jpg"): jpeg_bytes for n in ("a1", "a2", "a3", "a4", "c1")}
    files.update({url_for(f"person/media/{n}", ".png"): png_bytes for n in ("b1", "b2")})
    fetcher = FakeFetcher(files)
    return ImageIndex.from_output_dir(tmp_path), ManifestImageLoader(config, fetcher), fetcher


def test_index_orders_usable_images_by_phase_2_rank_and_ignores_unusable_ones(world):
    index, _, _ = world
    assert [r["media_id"] for r in index.get("A").usable_images] == ["A1", "A2", "A3", "A4"]
    assert index.get("A").selected["media_id"] == "A1"
    assert index.get("NOIMG").person_status == "no_person_media_records" and not index.get("NOIMG").has_usable_image
    assert index.get("PDFONLY").unusable_reasons == {"pdf_skipped": 1}
    assert index.get("NEVER_SEEN").person_status == "not_in_manifest"


def test_loader_uses_the_recorded_extension_and_rejects_changed_content(world, config):
    _, loader, fetcher = world
    assert loader.load(selected("MA", "person/media/a1"))[:3] == b"\xff\xd8\xff"
    assert fetcher.urls[-1].endswith("person/media/a1.jpg")
    for bad, kind in ((404, "file_not_found"), (403, "access_denied"), (500, "fetch_error")):
        loader_bad = ManifestImageLoader(config, FakeFetcher({url_for("person/media/x", ".jpg"): bad}))
        with pytest.raises(ImageLoadError) as caught:
            loader_bad.load(selected("M", "person/media/x"))
        assert caught.value.status == kind
    swapped = ManifestImageLoader(config, FakeFetcher({url_for("person/media/x", ".jpg"): b"%PDF-1.4 " + b"x" * 80}))
    with pytest.raises(ImageLoadError) as caught:
        swapped.load(selected("M", "person/media/x"))
    assert caught.value.status == "unknown_format"


# ------------------------------------------------ pair evidence (no fallback needed)

def test_both_images_present_invokes_the_comparer_and_keeps_the_similarity_verbatim(world):
    index, loader, _ = world
    comparer = ScriptedComparer(OK)
    evidence = compare_pair("A", "B", index, loader, comparer)
    assert len(comparer.calls) == 1 and evidence.status is PairStatus.MATCHED
    assert (evidence.face_similarity, evidence.similarity_metric, evidence.decision_threshold) == (0.4321, "cosine", 0.45)
    assert (evidence.media_id_a, evidence.media_id_b, evidence.fallback_used) == ("A1", "B1", False)
    assert [a["outcome"] for a in evidence.attempts] == ["compared"]


def test_negative_cosine_is_kept_and_a_missing_decision_is_not_invented(world):
    index, loader, _ = world
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(ComparisonOutcome(-0.12, "cosine")))
    assert evidence.status is PairStatus.COMPARED and evidence.face_similarity == -0.12
    not_matched = compare_pair("A", "B", index, loader, ScriptedComparer(ComparisonOutcome(0.05, "cosine", decision=False, threshold=0.45)))
    assert not_matched.status is PairStatus.NOT_MATCHED and not_matched.face_similarity == 0.05


@pytest.mark.parametrize("a,b,side", [("A", "NOIMG", "b"), ("NOIMG", "B", "a"), ("PDFONLY", "A", "a"), ("NEVER_SEEN", "A", "a")])
def test_one_side_without_an_image_skips_the_comparison_without_a_similarity(world, a, b, side):
    index, loader, fetcher = world
    comparer = ScriptedComparer(OK)
    evidence = compare_pair(a, b, index, loader, comparer)
    assert evidence.status is PairStatus.SKIPPED_NO_IMAGE and evidence.failed_side == side
    assert evidence.face_similarity is None and comparer.calls == [] and fetcher.calls == []


def test_skip_preserves_each_sides_actual_phase_2_status(world):
    index, loader, _ = world
    evidence = compare_pair("PDFONLY", "NOIMG", index, loader, ScriptedComparer(OK))
    assert evidence.failed_side is None
    assert (evidence.person_status_a, evidence.person_status_b) == ("no_usable_image", "no_person_media_records")
    assert evidence.unusable_reasons_a == {"pdf_skipped": 1}


def test_without_a_comparer_the_pair_is_not_checked_and_has_no_similarity(world):
    index, loader, fetcher = world
    evidence = compare_pair("A", "B", index, loader, None)
    assert evidence.status is PairStatus.NOT_CHECKED and evidence.face_similarity is None and fetcher.calls == []


# ------------------------------------------------ deterministic fallback

def test_no_face_on_one_side_falls_back_to_that_persons_next_image_only(world):
    index, loader, _ = world
    comparer = ScriptedComparer(rejected(a="no_face"), OK)
    evidence = compare_pair("A", "B", index, loader, comparer)
    assert evidence.status is PairStatus.MATCHED and evidence.fallback_used
    assert (evidence.media_id_a, evidence.media_id_b) == ("A2", "B1")  # only side a moved
    assert [(a["media_id_a"], a["media_id_b"], a["outcome"]) for a in evidence.attempts] == [("A1", "B1", "rejected"), ("A2", "B1", "compared")]
    assert evidence.attempts[0]["rejections"] == {"a": "no_face"}
    assert comparer.calls[0][1] == comparer.calls[1][1]  # b's image bytes were reused unchanged


def test_both_sides_failing_move_to_their_next_images_together(world):
    index, loader, _ = world
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="no_face", b="low_quality:too_small"), OK))
    assert (evidence.media_id_a, evidence.media_id_b) == ("A2", "B2") and evidence.status is PairStatus.MATCHED


def test_fallback_walks_the_images_in_rank_order_up_to_the_cap(world):
    index, loader, _ = world
    comparer = ScriptedComparer(rejected(a="no_face"), rejected(a="multiple_faces"), rejected(a="no_face"))
    evidence = compare_pair("A", "B", index, loader, comparer)
    assert [a["media_id_a"] for a in evidence.attempts] == ["A1", "A2", "A3"]  # A4 exists but the cap is 3 images
    assert evidence.status is PairStatus.NO_FACE and evidence.failed_side == "a" and evidence.side_failures == {"a": "no_face"}
    assert evidence.face_similarity is None  # never a stand-in value
    assert len(comparer.calls) == 3


def test_fallback_is_deterministic(world):
    index, loader, _ = world
    runs = {
        json.dumps(compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="no_face"), rejected(b="no_face"), OK)).to_dict(), sort_keys=True)
        for _ in range(5)
    }
    assert len(runs) == 1


def test_person_with_a_single_image_has_nothing_to_fall_back_to(world):
    index, loader, _ = world
    comparer = ScriptedComparer(rejected(b="no_face"))
    evidence = compare_pair("A", "C", index, loader, comparer)
    assert evidence.status is PairStatus.NO_FACE and evidence.failed_side == "b" and len(comparer.calls) == 1
    assert evidence.fallback_used is False


def test_when_both_sides_are_stuck_the_statuses_of_both_are_kept(world):
    index, loader, _ = world
    evidence = compare_pair("C", "NOIMG", index, loader, ScriptedComparer(OK))
    assert evidence.status is PairStatus.SKIPPED_NO_IMAGE
    both = compare_pair("C", "C", index, loader, ScriptedComparer(rejected(a="no_face", b="multiple_faces")))
    assert both.failed_side == "both" and both.side_failures == {"a": "no_face", "b": "multiple_faces"}
    assert both.status is PairStatus.NO_FACE  # side a's failure leads, deterministically


def test_the_fallback_policy_is_configurable(world):
    index, loader, _ = world
    only_no_face = FallbackPolicy(fallback_on=frozenset({"no_face"}))
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="low_quality:too_blurred"), OK), only_no_face)
    assert evidence.status is PairStatus.LOW_QUALITY and not evidence.fallback_used  # not eligible, so no fallback
    single = FallbackPolicy(max_images_per_person=1)
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="no_face"), OK), single)
    assert evidence.status is PairStatus.NO_FACE


# --------------------------- operational failures stay separate from face results

@pytest.mark.parametrize("error,kind", [(FrUnavailable("refused"), "unavailable"), (FrAuthError("bad key"), "auth_error")])
def test_service_failures_are_errors_never_mismatches_and_never_trigger_fallback(world, error, kind):
    index, loader, _ = world
    comparer = ScriptedComparer(error)
    evidence = compare_pair("A", "B", index, loader, comparer)
    assert evidence.status is PairStatus.ERROR and evidence.error_kind == kind
    assert evidence.status is not PairStatus.NOT_MATCHED and evidence.face_similarity is None
    assert len(comparer.calls) == 1 and evidence.fallback_used is False  # an outage says nothing about the images


def test_an_outage_after_a_fallback_keeps_the_history(world):
    index, loader, _ = world
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="no_face"), FrUnavailable("down")))
    assert evidence.status is PairStatus.ERROR and evidence.fallback_used
    assert [a["outcome"] for a in evidence.attempts] == ["rejected", "error"]


def test_image_that_vanished_since_phase_2_is_an_error_with_the_real_cause_and_no_fallback(world):
    index, loader, _ = world
    comparer = ScriptedComparer(OK)
    evidence = compare_pair("A", "GONE", index, loader, comparer)
    assert evidence.status is PairStatus.ERROR
    assert (evidence.error_kind, evidence.failed_side) == ("image_load:file_not_found", "b")
    assert comparer.calls == []


def test_evidence_is_json_serialisable_and_carries_no_image_data(world):
    index, loader, _ = world
    evidence = compare_pair("A", "B", index, loader, ScriptedComparer(rejected(a="no_face"), OK))
    record = json.loads(json.dumps(evidence.to_dict()))
    assert record["status"] == "matched" and record["fallback_used"] is True and len(record["attempts"]) == 2
    assert not any(isinstance(v, (bytes, bytearray)) for v in record.values())


# --------------------- the real comparer over HTTP (emulator) end to end

def test_pair_comparer_over_http_applies_fallback_with_the_real_client(world):
    index, loader, _ = world
    httpd, url = start_emulator()
    try:
        client = FrDopamsClient(FrConfig(base_url=url, retries=0, api_key=API_KEY))
        FrEmulator.compare_script = [(422, compare_rejection(a="no_face")), (200, compare_success(0.7))]
        evidence = compare_pair("A", "B", index, loader, FrDopamsPairComparer(client))
        assert evidence.status is PairStatus.MATCHED and evidence.face_similarity == 0.7 and evidence.fallback_used
        assert (evidence.media_id_a, evidence.media_id_b) == ("A2", "B1")
        assert [r["fields"] for r in FrEmulator.requests if r["method"] == "POST"] == [["file_a", "file_b"]] * 2

        FrEmulator.compare_script = [(503, {"detail": "down"})]
        down = compare_pair("A", "B", index, loader, FrDopamsPairComparer(FrDopamsClient(FrConfig(base_url=url, retries=0, api_key=API_KEY))))
        assert down.status is PairStatus.ERROR and down.error_kind == "unavailable"
        wrong_key = compare_pair("A", "B", index, loader, FrDopamsPairComparer(FrDopamsClient(FrConfig(base_url=url, retries=0, api_key="x"))))
        assert wrong_key.status is PairStatus.ERROR and wrong_key.error_kind == "auth_error"
    finally:
        httpd.shutdown()
        httpd.server_close()
