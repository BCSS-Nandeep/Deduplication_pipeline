"""Per-record resolution: formats, extensionless paths, failures, face-check hook."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from dedup_pipeline.image_resolution.file_server import FetchResult, InvalidPath, build_candidates
from dedup_pipeline.image_resolution.formats import sniff_format
from dedup_pipeline.image_resolution.models import MediaRow, Status
from dedup_pipeline.image_resolution.resolver import FaceCheckResult, ResolverConfig, resolve_media

from .conftest import BASE, PDF_BYTES, FakeFetcher, make_image, media_row, url_for


# ---------------------------------------------------------- accessible images

@pytest.mark.parametrize("fmt,expected", [("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp")])
def test_accessible_images_are_usable(config, fmt, expected):
    data = make_image(fmt, (150, 100))
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): data})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.USABLE and result.usable
    assert result.detected_type == expected
    assert (result.width, result.height, result.byte_size) == (150, 100, len(data))
    assert result.skip_reason is None


def test_type_comes_from_signature_not_from_the_file_name(config, png_bytes):
    # A PNG served under a .jpg name is still reported as png.
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): png_bytes})
    assert resolve_media(media_row(), config, fetcher).detected_type == "png"


def test_pdf_under_an_image_name_is_skipped(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): PDF_BYTES})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.PDF_SKIPPED and not result.usable
    assert result.detected_type == "pdf"
    assert len(fetcher.calls) == 2  # as-is 404, then .jpg probe; a PDF is never downloaded in full


def test_pdf_found_only_via_extension_fallback(config):
    fetcher = FakeFetcher({url_for("person/identitydetails/x", ".pdf"): PDF_BYTES})
    result = resolve_media(media_row(path="person/identitydetails/x"), config, fetcher)
    assert result.status is Status.PDF_SKIPPED and result.resolved_suffix == ".pdf"


# ------------------------------------------------------------- extensionless

def test_extensionless_path_tries_as_is_then_extensions_in_proxy_order(config, png_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc", ".png"): png_bytes})
    result = resolve_media(media_row(), config, fetcher)
    assert result.usable and result.resolved_suffix == ".png"
    probed = [u.removeprefix(url_for("person/media/abc")) for u in fetcher.urls[:4]]
    assert probed == ["", ".jpg", ".jpeg", ".png"]


def test_path_with_extension_gets_no_fallbacks(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc.jpg"): jpeg_bytes})
    result = resolve_media(media_row(path="person/media/abc.jpg"), config, fetcher)
    assert result.usable and result.resolved_suffix == ""
    missing = FakeFetcher({})
    resolve_media(media_row(path="person/media/abc.jpg"), config, missing)
    assert len(missing.calls) == 1


def test_candidate_urls_encode_segments_like_the_proxy():
    urls = [c.url for c in build_candidates("/person/my file/a#b?c", BASE)[:1]]
    assert urls == [BASE + "person/my%20file/a%23b%3Fc"]
    # already-encoded input is not double-encoded
    assert build_candidates("a%20b/c", BASE)[0].url == BASE + "a%20b/c"


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/../b", "a\\b", "http://evil.test/files/x", ""])
def test_unsafe_paths_are_rejected(config, bad):
    with pytest.raises(InvalidPath):
        build_candidates(bad, BASE)
    result = resolve_media(media_row(path=bad or "/"), config, FakeFetcher({}))
    assert result.status in (Status.INVALID_PATH, Status.NO_FILE_PATH)


def test_absolute_url_inside_the_file_server_is_accepted(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc.jpg"): jpeg_bytes})
    assert resolve_media(media_row(path=BASE + "person/media/abc.jpg"), config, fetcher).usable


# ------------------------------------------------------------- inaccessible

@pytest.mark.parametrize(
    "availability", ["BOOKKEEPING_DOWNLOADED_INACCESSIBLE", "EMPTY_AT_SOURCE", "DOWNLOAD_FAILED", "UNRESOLVED_FILE_ID"]
)
def test_unverified_availability_is_skipped_without_network(config, availability):
    fetcher = FakeFetcher({})
    result = resolve_media(media_row(availability=availability), config, fetcher)
    assert result.status is Status.AVAILABILITY_NOT_VERIFIED and availability in result.skip_reason
    assert fetcher.calls == []


def test_probe_unverified_option_still_records_missing_files(config):
    cfg = ResolverConfig(base_url=BASE, probe_unverified=True)
    result = resolve_media(media_row(availability="BOOKKEEPING_DOWNLOADED_INACCESSIBLE"), cfg, FakeFetcher({}))
    assert result.status is Status.FILE_NOT_FOUND


def test_verified_but_missing_on_server(config):
    assert resolve_media(media_row(), config, FakeFetcher({})).status is Status.FILE_NOT_FOUND


def test_forbidden_is_access_denied(config):
    fetcher = FakeFetcher({url_for("person/media/abc"): 403})
    assert resolve_media(media_row(), config, fetcher).status is Status.ACCESS_DENIED


def test_server_error_or_timeout_is_a_retryable_fetch_error(config, jpeg_bytes):
    for failure in (500, TimeoutError()):
        fetcher = FakeFetcher({url_for("person/media/abc"): failure})
        assert resolve_media(media_row(), config, fetcher).status is Status.FETCH_ERROR
    # an error must not be mistaken for "not found" even if other names are simply missing
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): 502})
    assert resolve_media(media_row(), config, fetcher).status is Status.FETCH_ERROR


def test_failure_while_downloading_after_a_good_probe(config, jpeg_bytes):
    class Flaky(FakeFetcher):
        def fetch(self, url, max_bytes):
            return super().fetch(url, max_bytes) if max_bytes <= 64 else FetchResult(500, error="HTTP 500")

    fetcher = Flaky({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    assert resolve_media(media_row(), config, fetcher).status is Status.FETCH_ERROR


# ----------------------------------------------------- missing / empty / junk

def test_missing_path_columns(config):
    result = resolve_media(media_row(path=None), config, FakeFetcher({}))
    assert result.status is Status.NO_FILE_PATH
    blank = MediaRow("M", "P", "person/MEDIA", "VERIFIED_ACCESSIBLE", "  ", None, "")
    assert resolve_media(blank, config, FakeFetcher({})).status is Status.NO_FILE_PATH


def test_path_columns_follow_the_backend_precedence(config, jpeg_bytes):
    row = MediaRow("M", "P", "person/MEDIA", "VERIFIED_ACCESSIBLE", None, "person/media/local.jpg", "person/media/orig.jpg")
    fetcher = FakeFetcher({url_for("person/media/local.jpg"): jpeg_bytes})
    result = resolve_media(row, config, fetcher)
    assert result.usable and result.path_source == "source_local_path"


def test_empty_file(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): b""})
    assert resolve_media(media_row(), config, fetcher).status is Status.EMPTY_FILE


def test_html_error_page_with_http_200_is_not_an_image(config):
    page = b"<html><body>Not Found</body></html>"
    fetcher = FakeFetcher({url_for("person/media/abc"): page})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.UNKNOWN_FORMAT and result.detected_type == "html"


def test_html_at_one_name_does_not_hide_a_real_image_at_the_next(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc"): b"<html></html>", url_for("person/media/abc", ".jpg"): jpeg_bytes})
    assert resolve_media(media_row(), config, fetcher).usable


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_image_formats_are_unsupported(config, fmt):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): make_image(fmt, (60, 60))})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.UNSUPPORTED_FORMAT and result.detected_type == fmt.lower()


def test_office_documents_are_unsupported(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".docx"): b"PK\x03\x04" + b"\x00" * 100})
    assert resolve_media(media_row(), config, fetcher).status is Status.UNSUPPORTED_FORMAT


# --------------------------------------------------------------- corrupt data

def test_truncated_jpeg_is_corrupt(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes[: len(jpeg_bytes) // 2]})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.CORRUPT_IMAGE and result.detected_type == "jpeg" and not result.usable


def test_jpeg_header_followed_by_garbage_is_corrupt(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): b"\xff\xd8\xff\xe0" + b"garbage" * 40})
    assert resolve_media(media_row(), config, fetcher).status is Status.CORRUPT_IMAGE


def test_damaged_png_data_is_corrupt(config, png_bytes):
    damaged = png_bytes[:-60] + b"\x00" * 60
    fetcher = FakeFetcher({url_for("person/media/abc", ".png"): damaged})
    assert resolve_media(media_row(), config, fetcher).status is Status.CORRUPT_IMAGE


def test_riff_container_that_is_not_webp_is_corrupt(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".webp"): b"RIFF\x20\x00\x00\x00WEBPVP8 " + b"\x00" * 40})
    assert resolve_media(media_row(), config, fetcher).status is Status.CORRUPT_IMAGE


def test_tiny_image_is_rejected(config):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): make_image("JPEG", (20, 200))})
    result = resolve_media(media_row(), config, fetcher)
    assert result.status is Status.IMAGE_TOO_SMALL and (result.width, result.height) == (20, 200)


def test_oversized_file_is_rejected_without_decoding(config, jpeg_bytes):
    cfg = ResolverConfig(base_url=BASE, max_image_bytes=len(jpeg_bytes) - 1)
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    assert resolve_media(media_row(), cfg, fetcher).status is Status.FILE_TOO_LARGE


def test_decompression_bomb_is_rejected(config, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    fetcher = FakeFetcher({url_for("person/media/abc", ".png"): make_image("PNG", (400, 400))})
    assert resolve_media(media_row(), config, fetcher).status is Status.IMAGE_TOO_LARGE


# ------------------------------------------------------- face-check hook

def test_face_check_is_not_run_by_default(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    assert resolve_media(media_row(), config, fetcher).face_check_status == "not_checked"


@pytest.mark.parametrize(
    "outcome,status", [("no_face", Status.NO_FACE), ("multiple_faces", Status.MULTIPLE_FACES), ("low_quality", Status.LOW_QUALITY)]
)
def test_face_check_failures_make_an_image_unusable(config, jpeg_bytes, outcome, status):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    result = resolve_media(media_row(), config, fetcher, lambda data: FaceCheckResult(outcome, "detail"))
    assert result.status is status and not result.usable and result.face_check_status == outcome


def test_face_check_ok_and_checker_exception(config, jpeg_bytes):
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): jpeg_bytes})
    ok = resolve_media(media_row(), config, fetcher, lambda data: FaceCheckResult("ok"))
    assert ok.usable and ok.face_check_status == "ok"

    def boom(data):
        raise RuntimeError("model crashed")

    failed = resolve_media(media_row(), config, fetcher, boom)
    assert failed.status is Status.FACE_CHECK_ERROR


def test_face_check_never_sees_undecodable_images(config):
    seen = []
    fetcher = FakeFetcher({url_for("person/media/abc", ".jpg"): b"\xff\xd8\xff" + b"x" * 80})
    resolve_media(media_row(), config, fetcher, lambda d: seen.append(d) or FaceCheckResult("ok"))
    assert seen == []


def test_sniff_format_recognises_signatures():
    assert sniff_format(make_image("JPEG")[:32]) == "jpeg"
    assert sniff_format(make_image("PNG")[:32]) == "png"
    assert sniff_format(make_image("WEBP")[:32]) == "webp"
    assert sniff_format(b"\n  %PDF-1.7") == "pdf"
    assert sniff_format(b"") == "unknown"
    assert sniff_format(io.BytesIO(b"hello world, plain").read()) == "unknown"
