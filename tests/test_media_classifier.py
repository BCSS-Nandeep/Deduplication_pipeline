"""Phase 1 media classification: portraits vs identity-document layouts, selection safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dedup_pipeline.image_resolution.manifest import build_outputs
from dedup_pipeline.image_resolution.media_classifier import (
    DOCUMENT_CONFIDENCE_MIN,
    PORTRAIT_CONFIDENCE_MIN,
    MediaClass,
    classify_media_image,
)
from dedup_pipeline.image_resolution.models import MediaRow, Person, Status
from dedup_pipeline.image_resolution.resolver import Checkpoint, ResolverConfig, resolve_all, resolve_media
from dedup_pipeline.image_resolution.runner import MEDIA_CLASS_REVIEW_FILE, RunOptions, run_image_resolution
from dedup_pipeline.scoring.config import FINAL_WEIGHTS, RULE_VERSION, ScoringConfig

from .conftest import (
    BASE, FakeDb, FakeFetcher, make_ambiguous_header_image, make_blank_image, make_document_image,
    make_group_image, make_portrait_image, media_row, url_for,
)


def test_confidence_floors_are_documented():
    assert DOCUMENT_CONFIDENCE_MIN == 0.70
    assert PORTRAIT_CONFIDENCE_MIN == 0.70


def test_portrait_fixture_is_candidate_portrait():
    result = classify_media_image(make_portrait_image())
    assert result.label is MediaClass.CANDIDATE_PORTRAIT
    assert result.confidence >= PORTRAIT_CONFIDENCE_MIN
    assert "portrait" in result.reason.lower() or "skin" in result.reason.lower()


def test_aadhaar_like_mock_is_identity_document_without_real_aadhaar_data():
    data = make_document_image()
    result = classify_media_image(data)
    assert result.label is MediaClass.IDENTITY_DOCUMENT
    assert result.confidence >= DOCUMENT_CONFIDENCE_MIN
    assert b"aadhaar" not in data.lower() and b"uidai" not in data.lower()


def test_group_blank_and_ambiguous_images_are_unknown():
    assert classify_media_image(make_group_image()).label is MediaClass.UNKNOWN
    assert classify_media_image(make_blank_image()).label is MediaClass.UNKNOWN
    # Near-square image with weak structure should not be forced into a confident class.
    weak = make_blank_image(size=(200, 190))
    assert classify_media_image(weak).label is MediaClass.UNKNOWN


def test_ambiguous_header_band_is_not_accepted_as_portrait():
    """D1: coloured header + central skin-like block must not become CANDIDATE_PORTRAIT."""
    result = classify_media_image(make_ambiguous_header_image())
    assert result.label is MediaClass.UNKNOWN
    assert result.label is not MediaClass.CANDIDATE_PORTRAIT
    assert "band" in result.reason.lower() or "conflict" in result.reason.lower()
    # Control cases still hold after the guard.
    assert classify_media_image(make_portrait_image()).label is MediaClass.CANDIDATE_PORTRAIT
    assert classify_media_image(make_document_image()).label is MediaClass.IDENTITY_DOCUMENT


def test_resolve_excludes_documents_and_unknown_from_face_matching(config):
    portrait = make_portrait_image()
    document = make_document_image()
    blank = make_blank_image()
    good = resolve_media(media_row("P", "A", "person/media/portrait"), config,
                         FakeFetcher({url_for("person/media/portrait", ".jpg"): portrait}))
    doc = resolve_media(media_row("D", "A", "person/id/doc", category="person/IDENTITY_DETAILS"), config,
                        FakeFetcher({url_for("person/id/doc", ".jpg"): document}))
    unk = resolve_media(media_row("U", "A", "person/media/blank"), config,
                        FakeFetcher({url_for("person/media/blank", ".jpg"): blank}))
    assert good.usable and good.status is Status.USABLE and good.media_class == "CANDIDATE_PORTRAIT"
    assert not doc.usable and doc.status is Status.IDENTITY_DOCUMENT_EXCLUDED
    assert doc.media_class == "IDENTITY_DOCUMENT"
    assert not unk.usable and unk.status is Status.MEDIA_CLASS_UNCERTAIN
    assert unk.media_class == "UNKNOWN"


def test_corrupt_and_missing_media_never_reach_classification(config):
    missing = resolve_media(media_row(path="person/media/gone"), config, FakeFetcher({}))
    assert missing.status is Status.FILE_NOT_FOUND and missing.media_class is None
    corrupt = resolve_media(
        media_row(path="person/media/bad"), config,
        FakeFetcher({url_for("person/media/bad", ".jpg"): b"\xff\xd8\xff" + b"not-a-jpeg"}),
    )
    assert corrupt.status is Status.CORRUPT_IMAGE and corrupt.media_class is None


def test_selection_does_not_fall_back_to_excluded_or_unknown_images(config, tmp_path):
    """Category priority still ranks MEDIA first, but only CANDIDATE_PORTRAIT is selectable."""
    rows = [
        media_row("M_doc", "P1", "person/media/doc", category="person/MEDIA"),
        media_row("M_blank", "P1", "person/media/blank", category="person/MEDIA"),
        media_row("M_port", "P1", "person/id/port", category="person/IDENTITY_DETAILS"),
    ]
    files = {
        url_for("person/media/doc", ".jpg"): make_document_image(),
        url_for("person/media/blank", ".jpg"): make_blank_image(),
        url_for("person/id/port", ".jpg"): make_portrait_image(),
    }
    cp = Checkpoint(tmp_path / "cp.jsonl", config.signature(False), fresh=True)
    results = resolve_all(rows, config, FakeFetcher(files), cp, workers=1)
    manifest, summaries, _ = build_outputs([Person("P1", "V2")], rows, results, config.category_priority)
    by_id = {row["media_id"]: row for row in manifest}
    assert by_id["M_doc"]["usable_for_face_recognition"] is False
    assert by_id["M_blank"]["usable_for_face_recognition"] is False
    assert by_id["M_port"]["usable_for_face_recognition"] is True
    assert by_id["M_port"]["selected_for_person"] is True
    assert summaries[0]["selected_media_id"] == "M_port"
    assert summaries[0]["usable_image_count"] == 1


def test_when_only_documents_exist_face_media_is_unavailable(config, tmp_path):
    rows = [media_row("M1", "P1", "person/media/a"), media_row("M2", "P1", "person/id/b", category="person/IDENTITY_DETAILS")]
    files = {
        url_for("person/media/a", ".jpg"): make_document_image(),
        url_for("person/id/b", ".jpg"): make_document_image(),
    }
    cp = Checkpoint(tmp_path / "cp.jsonl", config.signature(False), fresh=True)
    results = resolve_all(rows, config, FakeFetcher(files), cp, workers=1)
    _, summaries, _ = build_outputs([Person("P1", "V2")], rows, results, config.category_priority)
    assert summaries[0]["selected_media_id"] is None
    assert summaries[0]["usable_image_count"] == 0
    assert summaries[0]["person_status"] == "no_usable_image"


def test_runner_writes_local_classification_review(tmp_path):
    db = FakeDb(
        persons=[("P1", "V2")],
        media=[
            ("M1", "P1", "person/MEDIA", "VERIFIED_ACCESSIBLE", "person/media/doc", None, None),
            ("M2", "P1", "person/IDENTITY_DETAILS", "VERIFIED_ACCESSIBLE", "person/id/port", None, None),
        ],
    )
    files = {
        url_for("person/media/doc", ".jpg"): make_document_image(),
        url_for("person/id/port", ".jpg"): make_portrait_image(),
    }
    out = tmp_path / "images"
    summary = run_image_resolution(db, ResolverConfig(base_url=BASE), RunOptions(out), FakeFetcher(files))
    review_path = out / MEDIA_CLASS_REVIEW_FILE
    assert review_path.is_file()
    review = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    assert any(item["media_class"] == "IDENTITY_DOCUMENT" for item in review)
    assert summary["media_class_review_count"] >= 1
    assert summary["outputs"]["media_classification_review"] == MEDIA_CLASS_REVIEW_FILE
    # Document was preferred by category but excluded; portrait under IDENTITY_DETAILS is selected.
    assert summary["usable_images"] == 1


def test_score_final_threshold_and_historical_default_unchanged():
    assert RULE_VERSION == "score-v2"
    final = ScoringConfig(rule_version="score-final")
    assert final.face_pass_threshold == 0.85
    assert FINAL_WEIGHTS["face"] == 0.25
    assert FINAL_WEIGHTS["name"] == 0.25
