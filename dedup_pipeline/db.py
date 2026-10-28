"""Read-only PostgreSQL access.

The pipeline treats the unified database as a read-only source. Three layers
enforce that:

1. the session is opened with ``default_transaction_read_only=on``;
2. ``assert_read_only`` refuses to proceed unless the server confirms it, and
   unless the connected database is the approved one;
3. every statement goes through ``GuardedCursor``, which only allows a single
   SELECT / WITH / SHOW statement and rejects write keywords.

Credentials are read at runtime and are never logged.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import psycopg2

# The audit found two databases on the server; only v2 carries media_unified.
APPROVED_DATABASES = frozenset({"dopams_cctns_v2"})
DEFAULT_DATABASE = "dopams_cctns_v2"
# Opt-in local test DBs (comma-separated in DEDUP_APPROVED_LOCAL_DATABASES). Live connections
# may use these names only when the PostgreSQL server address is loopback.
LOCAL_APPROVED_ENV = "DEDUP_APPROVED_LOCAL_DATABASES"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT.parent / "Dopams_data_base" / ".env"


class ReadOnlyViolation(RuntimeError):
    """The connection or a statement is not provably read-only."""


def _local_approved_names(environ: Optional[dict[str, str]] = None) -> set[str]:
    environ = os.environ if environ is None else environ
    raw = (environ.get(LOCAL_APPROVED_ENV) or "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def approved_output_databases(environ: Optional[dict[str, str]] = None) -> frozenset[str]:
    """Databases allowed in local run_summary stamps (production set plus opt-in local names)."""
    return frozenset(set(APPROVED_DATABASES) | _local_approved_names(environ))


def _require_loopback_for_local_names(conn: "ReadOnlyConnection", extras: set[str]) -> None:
    """Refuse local-only approved names unless the live server is on loopback."""
    if not extras:
        return
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(inet_server_addr()::text, 'local')")
    address = cur.fetchone()[0]
    cur.close()
    if address != "local" and not str(address).startswith(("127.0.0.1", "::1")):
        raise ReadOnlyViolation(
            f"{LOCAL_APPROVED_ENV}={sorted(extras)} is set but server address {address!r} is not loopback"
        )


@dataclass(frozen=True)
class ConnectionSettings:
    host: str
    port: int
    database: str
    user: str
    password: str = ""

    def __repr__(self) -> str:  # never leak the password through logging or tracebacks
        return f"ConnectionSettings(host={self.host!r}, port={self.port}, database={self.database!r}, user={self.user!r})"


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip("'\"")
    return values


def load_connection_settings(env_file: Optional[str] = None, environ: Optional[dict[str, str]] = None) -> ConnectionSettings:
    """``DEDUP_PG_*`` variables win; otherwise ``UNIFIED_PG_*`` from an .env file.

    The database name is pinned to the approved one by default, because the
    shared .env lists two names on a single malformed line.
    """
    environ = os.environ if environ is None else environ
    file_values: dict[str, str] = {}
    candidate = env_file or environ.get("DEDUP_ENV_FILE")
    path = Path(candidate) if candidate else DEFAULT_ENV_FILE
    if path.is_file():
        file_values = _parse_env_file(path)

    def pick(name: str, fallback_key: str, default: str = "") -> str:
        return environ.get(f"DEDUP_PG_{name}") or file_values.get(f"UNIFIED_PG_{fallback_key}") or default

    user = pick("USER", "USER")
    if not user:
        raise RuntimeError(
            "No database user configured. Set DEDUP_PG_USER/DEDUP_PG_PASSWORD or pass --env-file."
        )
    return ConnectionSettings(
        host=pick("HOST", "HOST", "localhost"),
        port=int(pick("PORT", "PORT", "5432")),
        database=environ.get("DEDUP_PG_DATABASE") or DEFAULT_DATABASE,
        user=user,
        password=pick("PASSWORD", "PASSWORD"),
    )


# ------------------------------------------------------------ statement guard

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|COPY|CALL|DO|VACUUM|REINDEX|CLUSTER|"
    r"REFRESH|LOCK|COMMENT|SET|RESET|NOTIFY|LISTEN|PREPARE|EXECUTE|INTO|ANALYZE|DISCARD|IMPORT|SECURITY)\b",
    re.IGNORECASE,
)
_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def check_statement_is_read_only(sql: str) -> None:
    """Raise ``ReadOnlyViolation`` unless ``sql`` is one SELECT/WITH/SHOW statement.

    A deliberately conservative allow-list: false rejections are cheap (the
    queries here are fixed and parameterised), a missed write is not.
    """
    stripped = _COMMENTS.sub(" ", sql).strip()
    while stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if not stripped:
        raise ReadOnlyViolation("Empty SQL statement.")
    if ";" in stripped:
        raise ReadOnlyViolation("Multiple SQL statements are not allowed.")
    if stripped.split(None, 1)[0].upper() not in ("SELECT", "WITH", "SHOW"):
        raise ReadOnlyViolation("Only SELECT, WITH and SHOW statements are allowed.")
    blocked = _BLOCKED.search(stripped)
    if blocked:
        raise ReadOnlyViolation(f"Blocked keyword in statement: {blocked.group(1).upper()}")


class GuardedCursor:
    """Cursor wrapper that validates each statement before it reaches the server."""

    def __init__(self, cursor: Any):
        self._cursor = cursor

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> "GuardedCursor":
        check_statement_is_read_only(sql)
        self._cursor.execute(sql, params)
        return self

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list:
        return self._cursor.fetchall()

    def fetchmany(self, size: int) -> list:
        return self._cursor.fetchmany(size)

    def close(self) -> None:
        self._cursor.close()


class ReadOnlyConnection:
    def __init__(self, connection: Any):
        self._connection = connection

    def cursor(self) -> GuardedCursor:
        return GuardedCursor(self._connection.cursor())

    def stream(self, sql: str, params: Optional[Any] = None, batch_size: int = 2000) -> Iterator[tuple]:
        """Yield rows through a server-side cursor in bounded batches (the client never holds the full result).

        The statement is checked by the same guard as every other query, and the transaction is the
        session's read-only one. A server-side cursor needs a transaction, so autocommit is switched off
        for the duration and restored afterwards; nothing is ever committed.
        """
        check_statement_is_read_only(sql)
        raw = self._connection
        raw.autocommit = False
        cursor = raw.cursor(name=f"dedup_stream_{uuid.uuid4().hex[:12]}")
        cursor.itersize = batch_size
        try:
            cursor.execute(sql, params)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield from rows
        finally:
            try:
                cursor.close()
            finally:
                raw.rollback()
                raw.autocommit = True

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ReadOnlyConnection":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def assert_read_only(
    conn: ReadOnlyConnection,
    approved: Optional[Iterable[str]] = None,
    environ: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Ask the server whether the session is read-only; refuse to continue if not."""
    cur = conn.cursor()
    cur.execute("SHOW transaction_read_only")
    in_tx = cur.fetchone()[0]
    cur.execute("SHOW default_transaction_read_only")
    default = cur.fetchone()[0]
    cur.execute("SELECT current_database(), version()")
    database, version = cur.fetchone()
    cur.close()
    if in_tx != "on" or default != "on":
        raise ReadOnlyViolation(
            f"Session is not read-only (transaction_read_only={in_tx}, default_transaction_read_only={default})."
        )
    extras = _local_approved_names(environ)
    allowed = set(APPROVED_DATABASES if approved is None else approved) | extras
    if extras:
        _require_loopback_for_local_names(conn, extras)
    if database not in allowed:
        raise ReadOnlyViolation(f"Connected to unapproved database {database!r}.")
    return {"database": database, "transaction_read_only": in_tx, "server_version": version.split(",")[0][:40]}


def connect_read_only(settings: ConnectionSettings, application_name: str = "dedup_pipeline_readonly") -> ReadOnlyConnection:
    try:
        raw = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.database,
            user=settings.user,
            password=settings.password,
            connect_timeout=15,
            application_name=application_name,
            options="-c default_transaction_read_only=on -c statement_timeout=60000",
        )
    except psycopg2.Error as exc:
        message = str(exc).replace(settings.password, "***") if settings.password else str(exc)
        raise ConnectionError(f"Could not connect to {settings!r}: {message.strip().splitlines()[0]}") from None
    raw.set_session(readonly=True, autocommit=True)
    conn = ReadOnlyConnection(raw)
    try:
        assert_read_only(conn)
    except Exception:
        conn.close()
        raise
    return conn
