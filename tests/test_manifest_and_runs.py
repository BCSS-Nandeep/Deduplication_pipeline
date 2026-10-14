"""Multiple images, shared paths, V1 persons, checkpoint/resume and safe reruns."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dedup_pipeline.image_resolution.manifest import (
    HAS_USABLE_IMAGE,
    NO_ACCESSIBLE_MEDIA,
    NO_MEDIA_RECORDS,
    NO_USABLE_IMAGE,
    build_outputs,
    find_shared_paths,
)
from dedup_pipeline.image_resolution.models import MediaRow, Person, Status
from dedup_pipeline.image_resolution.resolver import Checkpoint, ResolverConfig, resolve_all
from dedup_pipeline.image_resolution.runner import (
    CHECKPOINT_FILE,
    MANIFEST_FILE,
    PERSON_SUMMARY_FILE,
    RUN_SUMMARY_FILE,
    SHARED_PATHS_FILE,
    RunOptions,
    run_image_resolution,
)

from .conftest import BASE, PDF_BYTES, FakeDb, FakeFetcher, media_row, url_for

PRIORITY = ("person/MEDIA", "person/IDENTITY_DETAILS")


def resolve(rows, files, config, tmp_path, **kw):
    cp = Checkpoint(tmp_path / "cp.jsonl", config.signature(False), fresh=True)
    return resolve_all(rows, config, FakeFetcher(files), cp, workers=1, **kw)


# ----------------------------------------------------------- multiple images

def test_multiple_images_are_ranked_deterministically(config, jpeg_bytes, tmp_path):
    persons = [Person("P1", "V2")]
    rows = [
        media_row("M9", "P1", "person/id/c", "person/IDENTITY_DETAILS"),
        media_row("M2", "P1", "person/media/b", "person/MEDIA"),
        media_row("M1", "P1", "person/media/a", "person/MEDIA"),
        media_row("M5", "P1", "person/media/gone", "person/MEDIA", "BOOKKEEPING_DOWNLOADED_INACCESSIBLE"),
    ]
    files = {url_for(r.resolved_relative_path, ".jpg"): jpeg_bytes for r in rows}
    runs = []
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        results = resolve(shuffled, files, config, tmp_path)
        manifest, summaries, _ = build_outputs(persons, shuffled, results, PRIORITY)
        runs.append(json.dumps([manifest, summaries], sort_keys=True))
    assert len(set(runs)) == 1  # input order never changes the output

    manifest = json.loads(runs[0])[0]
    # preferred category first, verified before unverified, then media_id
    assert [m["media_id"] for m in manifest] == ["M1", "M2", "M5", "M9"]
    by_id = {m["media_id"]: m for m in manifest}
    assert by_id["M1"]["person_rank"] == 1 and by_id["M1"]["selected_for_person"]
    assert [m["person_rank"] for m in manifest] == [1, 2, 3, 4]
    assert sum(m["selected_for_person"] for m in manifest) == 1
    assert by_id["M5"]["resolution_status"] == "availability_not_verified"


def test_selection_skips_unusable_higher_ranked_images(config, jpeg_bytes, tmp_path):
    rows = [media_row("M1", "P1", "person/media/a"), media_row("M2", "P1", "person/media/b")]
    files = {url_for("person/media/a", ".pdf"): PDF_BYTES, url_for("person/media/b", ".jpg"): jpeg_bytes}
    results = resolve(rows, files, config, tmp_path)
    manifest, summaries, _ = build_outputs([Person("P1", "V2")], rows, results, PRIORITY)
    assert [m["selected_for_person"] for m in manifest] == [False, True]
    assert summaries[0]["selected_media_id"] == "M2" and summaries[0]["usable_image_count"] == 1
    assert summaries[0]["unusable_reasons"] == {"pdf_skipped": 1}


# ------------------------------------------------------------ person status

def test_every_person_gets_a_status_including_v1_without_media(config, jpeg_bytes, tmp_path):
    persons = [Person("V1P", "V1"), Person("OK", "V2"), Person("EMPTY", "V2"), Person("BAD", "V2")]
    rows = [
        media_row("M1", "OK", "person/media/ok"),
        media_row("M2", "EMPTY", None, availability="EMPTY_AT_SOURCE"),
        media_row("M3", "BAD", "person/media/bad"),
    ]
    files = {url_for("person/media/ok", ".jpg"): jpeg_bytes, url_for("person/media/bad", ".pdf"): PDF_BYTES}
    results = resolve(rows, files, config, tmp_path)
    _, summaries, _ = build_outputs(persons, rows, results, PRIORITY)
    status = {s["person_id"]: s["person_status"] for s in summaries}
    assert status == {"V1P": NO_MEDIA_RECORDS, "OK": HAS_USABLE_IMAGE, "EMPTY": NO_ACCESSIBLE_MEDIA, "BAD": NO_USABLE_IMAGE}
    assert [s["person_id"] for s in summaries] == sorted(status)


def test_every_failure_has_a_status_and_a_reason(config, tmp_path):
    rows = [
        media_row("A", "P", None),
        media_row("B", "P", "person/media/missing"),
        media_row("C", "P", "../bad"),
        media_row("D", "P", "x", availability="EMPTY_AT_SOURCE"),
    ]
    results = resolve(rows, {}, config, tmp_path)
    manifest, _, _ = build_outputs([Person("P", "V2")], rows, results, PRIORITY)
    assert len(manifest) == 4
    for record in manifest:
        assert not record["usable_for_face_recognition"]
        assert record["resolution_status"] != "usable" and record["skip_reason"]


# ------------------------------------------------------------- shared paths

def test_shared_paths_are_flagged_but_not_treated_as_identity(config, jpeg_bytes, tmp_path):
    rows = [
        media_row("M1", "P1", "person/media/same"),
        media_row("M2", "P2", "/person/media/same"),  # same path, leading slash normalised
        media_row("M3", "P3", "person/media/own"),
        media_row("M4", "P3", "person/media/own2"),
        media_row("M5", "P4", "person/media/dup"),
        media_row("M6", "P4", "person/media/dup"),  # same person twice: not "shared"
    ]
    files = {url_for(p, ".jpg"): jpeg_bytes for p in ("person/media/same", "person/media/own", "person/media/own2", "person/media/dup")}
    results = resolve(rows, files, config, tmp_path)
    persons = [Person(f"P{i}", "V2") for i in range(1, 5)]
    manifest, summaries, shared = build_outputs(persons, rows, results, PRIORITY)

    assert [s["path_ref"] for s in shared] == ["person/media/same"]
    assert shared[0]["person_ids"] == ["P1", "P2"] and shared[0]["media_ids"] == ["M1", "M2"]
    flags = {m["media_id"]: (m["shared_path_flag"], m["shared_path_person_count"]) for m in manifest}
    assert flags["M1"] == flags["M2"] == (True, 2)
    assert flags["M3"] == flags["M5"] == flags["M6"] == (False, 1)
    # both images stay fully usable and no identity link is implied anywhere in the output
    assert all(m["usable_for_face_recognition"] for m in manifest)
    assert not any("same_person" in json.dumps(r) or "match" in json.dumps(r).lower() for r in manifest + summaries)


def test_find_shared_paths_ignores_blank_paths():
    rows = [MediaRow("A", "P1", None, "X"), MediaRow("B", "P2", None, "X")]
    assert find_shared_paths(rows) == {}


# --------------------------------------------------------------- checkpoint

def test_checkpoint_skips_finished_work_and_retries_transient_errors(config, jpeg_bytes, tmp_path):
    rows = [media_row("OK", "P", "person/media/ok"), media_row("NF", "P", "person/media/nf"), media_row("ERR", "P", "person/media/err")]
    files = {url_for("person/media/ok", ".jpg"): jpeg_bytes, url_for("person/media/err"): 500}
    cp_path = tmp_path / "cp.jsonl"

    first = FakeFetcher(files)
    cp = Checkpoint(cp_path, config.signature(False))
    resolve_all(rows, config, first, cp, workers=1)
    cp.close(keep=[r.media_id for r in rows])
    assert first.calls

    second = FakeFetcher({**files, url_for("person/media/err", ".png"): jpeg_bytes})
    cp = Checkpoint(cp_path, config.signature(False))
    results = resolve_all(rows, config, second, cp, workers=1)
    cp.close(keep=[r.media_id for r in rows])
    touched = {u.split("/files/")[1].split("/")[-1].split(".")[0] for u in second.urls}
    assert touched == {"err"}  # OK and NF came from the checkpoint; only the transient error was retried
    assert results["NF"].status is Status.FILE_NOT_FOUND


def test_checkpoint_is_invalidated_when_the_record_or_settings_change(config, jpeg_bytes, tmp_path):
    cp_path = tmp_path / "cp.jsonl"
    row = media_row("M", "P", "person/media/a")
    files = {url_for("person/media/a", ".jpg"): jpeg_bytes}
    cp = Checkpoint(cp_path, config.signature(False))
    resolve_all([row], config, FakeFetcher(files), cp, workers=1)
    cp.close(keep=["M"])

    moved = media_row("M", "P", "person/media/other")  # path changed in the database
    cp = Checkpoint(cp_path, config.signature(False))
    assert cp.get(moved) is None and cp.get(row) is not None
    cp.close(keep=["M"])

    other = ResolverConfig(base_url=BASE, min_dimension=500)  # different settings
    assert Checkpoint(cp_path, other.signature(False)).get(row) is None


def test_checkpoint_tolerates_a_half_written_last_line(config, jpeg_bytes, tmp_path):
    cp_path = tmp_path / "cp.jsonl"
    row = media_row("M", "P", "person/media/a")
    cp = Checkpoint(cp_path, config.signature(False))
    resolve_all([row], config, FakeFetcher({url_for("person/media/a", ".jpg"): jpeg_bytes}), cp, workers=1)
    cp._handle.close()
    with open(cp_path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "result", "result": {"media_id": "X", "stat')  # simulated crash mid-write
    assert Checkpoint(cp_path, config.signature(False)).get(row).usable


def test_one_crashing_record_does_not_stop_the_run(config, jpeg_bytes, tmp_path):
    class Exploding(FakeFetcher):
        def fetch(self, url, max_bytes):
            if "boom" in url:
                raise RuntimeError("unexpected")
            return super().fetch(url, max_bytes)

    rows = [media_row("B", "P", "person/media/boom"), media_row("G", "P", "person/media/good")]
    cp = Checkpoint(tmp_path / "cp.jsonl", config.signature(False), fresh=True)
    results = resolve_all(rows, config, Exploding({url_for("person/media/good", ".jpg"): jpeg_bytes}), cp, workers=2)
    assert results["B"].status is Status.INTERNAL_ERROR and results["G"].usable


# ------------------------------------------------- full run, reruns, DB use

def _db(jpeg_bytes):
    persons = [("P1", "V2"), ("P2", "V2"), ("V1X", "V1")]
    media = [
        ("M1", "P1", "person/MEDIA", "VERIFIED_ACCESSIBLE", "person/media/a", None, None),
        ("M2", "P2", "person/MEDIA", "VERIFIED_ACCESSIBLE", "person/media/a", None, None),
        ("M3", "P2", "person/IDENTITY_DETAILS", "EMPTY_AT_SOURCE", None, None, None),
    ]
    return FakeDb(persons, media), {url_for("person/media/a", ".jpg"): jpeg_bytes}


def test_run_writes_local_files_and_is_safely_rerunnable(config, jpeg_bytes, tmp_path):
    db, files = _db(jpeg_bytes)
    options = RunOptions(output_dir=tmp_path / "out", workers=2)

    fetcher = FakeFetcher(files)
    summary = run_image_resolution(db, config, options, fetcher)
    out = options.output_dir
    first = {name: (out / name).read_bytes() for name in (MANIFEST_FILE, PERSON_SUMMARY_FILE, SHARED_PATHS_FILE)}
    assert summary["persons"] == 3 and summary["media_records"] == 3 and summary["usable_images"] == 2
    assert summary["person_status"] == {"no_person_media_records": 1, "usable_image_available": 2}
    assert summary["shared_path_count"] == 1
    assert summary["face_check"].startswith("not_run")
    assert summary["database"]["transaction_read_only"] == "on"
    assert (out / RUN_SUMMARY_FILE).is_file() and (out / CHECKPOINT_FILE).is_file()

    rerun_fetcher = FakeFetcher(files)
    run_image_resolution(db, config, options, rerun_fetcher)
    assert rerun_fetcher.calls == []  # nothing re-downloaded
    assert {name: (out / name).read_bytes() for name in first} == first  # byte-identical output
    assert not list(out.glob("*.tmp"))

    run_image_resolution(db, config, RunOptions(output_dir=out, fresh=True), FakeFetcher(files))
    assert {name: (out / name).read_bytes() for name in first} == first


def test_manifest_contains_no_image_data_or_personal_fields(config, jpeg_bytes, tmp_path):
    db, files = _db(jpeg_bytes)
    options = RunOptions(output_dir=tmp_path / "out")
    run_image_resolution(db, config, options, FakeFetcher(files))
    record = json.loads((options.output_dir / MANIFEST_FILE).read_text().splitlines()[0])
    assert set(record) == {
        "person_id", "source_system", "media_id", "attachment_category", "availability_status", "path_ref", "path_source",
        "resolved_suffix", "detected_type", "width", "height", "byte_size", "resolution_status",
        "usable_for_face_recognition", "skip_reason", "face_check_status", "person_rank", "selected_for_person",
        "shared_path_flag", "shared_path_person_count",
    }
    assert "http" not in json.dumps(record)  # relative reference only, never the server address


def test_run_refuses_a_connection_that_is_not_read_only(config, jpeg_bytes, tmp_path):
    from dedup_pipeline.db import ReadOnlyViolation

    db, files = _db(jpeg_bytes)
    db.read_only = "off"
    with pytest.raises(ReadOnlyViolation):
        run_image_resolution(db, config, RunOptions(output_dir=tmp_path / "out"), FakeFetcher(files))
    assert not (tmp_path / "out" / MANIFEST_FILE).exists()


def test_run_issues_only_select_statements(config, jpeg_bytes, tmp_path):
    db, files = _db(jpeg_bytes)
    run_image_resolution(db, config, RunOptions(output_dir=tmp_path / "out"), FakeFetcher(files))
    assert db.statements and all(s.lstrip().upper().startswith(("SELECT", "SHOW")) for s in db.statements)
