"""Read-only database guard, settings handling, real HTTP behaviour, and scope guarantees."""

from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dedup_pipeline.db import (
    ConnectionSettings,
    ReadOnlyViolation,
    assert_read_only,
    check_statement_is_read_only,
    load_connection_settings,
)
from dedup_pipeline.image_resolution import queries
from dedup_pipeline.image_resolution.file_server import HttpFetcher
from dedup_pipeline.image_resolution.models import MediaRow, Status
from dedup_pipeline.image_resolution.resolver import ResolverConfig, resolve_media

from .conftest import PDF_BYTES, FakeDb, make_image

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "dedup_pipeline"


# ------------------------------------------------------------ statement guard

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select person_id from public.persons_unified where source_system = %s order by 1 limit %s;",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SHOW transaction_read_only",
        "SELECT updated_at, created_by FROM t -- trailing comment",
    ],
)
def test_guard_allows_plain_reads(sql):
    check_statement_is_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO public.identity_links VALUES (1)",
        "UPDATE public.persons_unified SET full_name = 'x'",
        "DELETE FROM public.media_unified",
        "TRUNCATE public.identity_links",
        "DROP TABLE public.identity_links",
        "CREATE TABLE t (a int)",
        "CREATE INDEX i ON public.persons_unified (full_name)",
        "ALTER TABLE public.identity_links ADD COLUMN x int",
        "SELECT 1; DROP TABLE t",
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "SELECT * INTO newtable FROM public.persons_unified",
        "SELECT * FROM t FOR UPDATE",
        "SET default_transaction_read_only = off",
        "COPY t TO '/tmp/x'",
        "CALL some_procedure()",
        "/* sneaky */ UPDATE t SET a = 1",
        "",
    ],
)
def test_guard_rejects_anything_that_could_write(sql):
    with pytest.raises(ReadOnlyViolation):
        check_statement_is_read_only(sql)


def test_every_sql_string_in_the_package_passes_the_guard():
    """Static check of the real queries, so a future write statement cannot slip in."""
    db = FakeDb([("P1", "V2")], [])
    queries.load_persons(db, "V2", 5, True)
    queries.load_persons(db)
    queries.load_person_media(db, ["P1"])
    assert len(db.statements) == 3


def test_no_database_write_apis_are_used_anywhere_in_the_package():
    source = "\n".join(p.read_text(encoding="utf-8") for p in PACKAGE_ROOT.rglob("*.py"))
    for forbidden in ("executemany", "copy_from", "copy_expert", ".commit(", "execute_values"):
        assert forbidden not in source
    # A transaction is opened in exactly one place: ReadOnlyConnection.stream (a server-side cursor needs one).
    # It lives in db.py, runs inside the read-only session, ends with rollback() and never commits.
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "autocommit = False" in text:
            assert path.name == "db.py" and text.count("autocommit = False") == 1 and "raw.rollback()" in text, path
    # SQL text lives only in the two query modules, and none of it is a write
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name not in ("queries.py", "db.py"):
            assert not re.search(r"\b(INSERT INTO|UPDATE \w+ SET|DELETE FROM|CREATE TABLE|ALTER TABLE|DROP )", path.read_text())


ENGINE_FINGERPRINT_MODULE = "engine_fingerprint.py"  # Phase 10: the single, pinned exemption (see the test below)


def test_scope_guards_no_hashing_or_fingerprinting():
    """No image hashing or image fingerprinting anywhere. The only hashing in the package is the face engine's identity digest, which lives
    in one module and is checked to hash model files and a settings description, never images or people."""
    source = "\n".join(p.read_text(encoding="utf-8") for p in PACKAGE_ROOT.rglob("*.py") if p.name != ENGINE_FINGERPRINT_MODULE).lower()
    for forbidden in ("hashlib", "sha256", "imagehash", "phash", "md5", "perceptual"):
        assert forbidden not in source


def test_the_only_hashing_module_hashes_model_files_and_settings_not_images():
    path = next(PACKAGE_ROOT.rglob(ENGINE_FINGERPRINT_MODULE))
    text = path.read_text(encoding="utf-8")
    assert re.findall(r"^def (\w+)\(", text, re.M) == ["file_digest", "engine_digest"]
    assert "def file_digest(path: Path)" in text and "def engine_digest(payload: Mapping[str, Any])" in text  # a file path and a JSON mapping only
    for forbidden in ("imdecode", "PIL", "cv2", "numpy", "image_bytes", "perceptual", "phash", "imagehash", "md5"):
        assert forbidden not in text, forbidden
    users = [p.name for p in PACKAGE_ROOT.rglob("*.py") if re.search(r"from \.+(?:\w+\.)*engine_fingerprint import|import engine_fingerprint", p.read_text(encoding="utf-8"))
             and p.name != ENGINE_FINGERPRINT_MODULE]
    assert sorted(users) == ["local_comparer.py", "local_config.py"]  # nothing else may start hashing through it


def test_session_must_be_read_only_and_on_the_approved_database():
    assert assert_read_only(FakeDb([], []))["database"] == "dopams_cctns_v2"
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(FakeDb([], [], read_only="off"))
    with pytest.raises(ReadOnlyViolation, match="unapproved"):
        assert_read_only(FakeDb([], [], database="dopams_cctns"))


# ----------------------------------------------------------------- settings

def test_settings_prefer_dedup_variables_and_pin_the_database(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "UNIFIED_PG_HOST=file-host\nUNIFIED_PG_PORT=5433\nUNIFIED_PG_USER=file-user\n"
        "UNIFIED_PG_PASSWORD=file-secret\nUNIFIED_PG_DATABASE=dopams_cctns_v2 , UNIFIED_PG_DATABASE=dopams_cctns\n"
    )
    from_file = load_connection_settings(str(env), environ={})
    assert (from_file.host, from_file.port, from_file.user) == ("file-host", 5433, "file-user")
    assert from_file.database == "dopams_cctns_v2"  # the malformed line is ignored
    override = load_connection_settings(str(env), environ={"DEDUP_PG_USER": "ro_user", "DEDUP_PG_PASSWORD": "pw"})
    assert (override.user, override.password, override.host) == ("ro_user", "pw", "file-host")


def test_password_never_appears_in_repr_or_errors(tmp_path):
    settings = ConnectionSettings("h", 5432, "d", "u", "SuperSecret")
    assert "SuperSecret" not in repr(settings) and "SuperSecret" not in str(settings)
    with pytest.raises(RuntimeError):
        load_connection_settings(str(tmp_path / "missing.env"), environ={})


# ------------------------------------------------- real HTTP file server

class _FileServer(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    hits: list[str] = []

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        self.hits.append(self.path)
        if self.path == "/files/redirect/x":
            self.send_response(302)
            self.send_header("Location", "http://other-host.invalid/files/y")
            self.end_headers()
            return
        if self.path == "/files/broken":
            self.send_response(500)
            self.end_headers()
            return
        body = self.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        start, end = 0, len(body) - 1
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match and not body:
            self.send_response(416)
            self.end_headers()
            return
        status = 200
        if match:
            start, end = int(match.group(1)), min(int(match.group(2)), len(body) - 1)
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")  # as the real server does
        self.send_header("Content-Length", str(end - start + 1 if body else 0))
        self.end_headers()
        self.wfile.write(body[start:end + 1])


@pytest.fixture
def server():
    _FileServer.files, _FileServer.hits = {}, []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FileServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/files/", _FileServer
    httpd.shutdown()
    httpd.server_close()


def _row(path):
    return MediaRow("M", "P", "person/MEDIA", "VERIFIED_ACCESSIBLE", path)


def test_http_extensionless_image_resolves_despite_octet_stream(server):
    base, handler = server
    data = make_image("JPEG", (90, 90))
    handler.files["/files/person/media/abc.jpg"] = data
    config = ResolverConfig(base_url=base)
    result = resolve_media(_row("person/media/abc"), config, HttpFetcher(retries=0))
    assert result.usable and result.resolved_suffix == ".jpg" and result.byte_size == len(data)
    assert handler.hits[:2] == ["/files/person/media/abc", "/files/person/media/abc.jpg"]


def test_http_pdf_missing_and_empty_and_oversized(server):
    base, handler = server
    handler.files["/files/doc/a.pdf"] = PDF_BYTES
    handler.files["/files/empty/a.jpg"] = b""
    handler.files["/files/big/a.png"] = make_image("PNG", (300, 300))
    config = ResolverConfig(base_url=base, max_image_bytes=500)
    fetcher = HttpFetcher(retries=0)
    assert resolve_media(_row("doc/a"), config, fetcher).status is Status.PDF_SKIPPED
    assert resolve_media(_row("nothing/here"), config, fetcher).status is Status.FILE_NOT_FOUND
    assert resolve_media(_row("empty/a"), config, fetcher).status is Status.EMPTY_FILE
    assert resolve_media(_row("big/a"), config, fetcher).status is Status.FILE_TOO_LARGE


def test_http_server_error_is_retried_then_reported_as_fetch_error(server):
    base, handler = server
    config = ResolverConfig(base_url=base)
    result = resolve_media(_row("broken"), config, HttpFetcher(retries=2, backoff=0))
    assert result.status is Status.FETCH_ERROR
    assert handler.hits.count("/files/broken") == 3


def test_http_redirect_to_another_host_is_refused(server):
    base, _ = server
    result = resolve_media(_row("redirect/x"), ResolverConfig(base_url=base), HttpFetcher(retries=0))
    assert result.status in (Status.ACCESS_DENIED, Status.FILE_NOT_FOUND)  # refused, never followed
    assert result.status is not Status.USABLE


def test_http_connection_refused_is_a_fetch_error():
    config = ResolverConfig(base_url="http://127.0.0.1:9/files/")
    assert resolve_media(_row("a/b"), config, HttpFetcher(timeout=2, retries=0)).status is Status.FETCH_ERROR
