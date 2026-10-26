"""Create the local candidate-parity database and load the bounded fixture.

This script refuses any host other than loopback. It does not connect to production.
Credentials and the test database name come from the environment:

  DEDUP_LOCAL_PG_USER
  DEDUP_LOCAL_PG_PASSWORD
  DEDUP_LOCAL_PG_DATABASE

Optional: DEDUP_LOCAL_PG_PORT (default 5432). The maintenance connection uses the
local server's ``postgres`` database only long enough to create the named test database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "tests" / "fixtures" / "candidate_parity_extract.json"
SYNTHETIC = ROOT / "tests" / "fixtures" / "candidate_parity_synthetic.json"
PRODUCTION_HOST = "192.168.103.106"

SCHEMA = """
DROP TABLE IF EXISTS public.person_enrichment;
DROP TABLE IF EXISTS public.persons_unified;
CREATE TABLE public.persons_unified (
    person_id character varying(100) PRIMARY KEY,
    source_system character varying(2) NOT NULL,
    source_record_id character varying(100) NOT NULL,
    full_name character varying(500),
    alias character varying(255),
    relative_name character varying(255),
    phone_number character varying(20)
);
CREATE TABLE public.person_enrichment (
    person_id character varying(100) PRIMARY KEY REFERENCES public.persons_unified (person_id),
    cleaned_full_name text,
    cleaned_alias character varying(255),
    cleaned_relative_name text
);
"""


def _settings() -> dict:
    host = os.environ.get("DEDUP_LOCAL_PG_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} or host == PRODUCTION_HOST:
        raise SystemExit(f"Refusing local load: host {host!r} is not loopback")
    missing = [name for name in ("DEDUP_LOCAL_PG_USER", "DEDUP_LOCAL_PG_PASSWORD", "DEDUP_LOCAL_PG_DATABASE") if not os.environ.get(name)]
    if missing:
        raise SystemExit("Local PostgreSQL credentials are not configured: " + ", ".join(missing))
    return {
        "host": host,
        "port": int(os.environ.get("DEDUP_LOCAL_PG_PORT", "5432")),
        "user": os.environ["DEDUP_LOCAL_PG_USER"],
        "password": os.environ["DEDUP_LOCAL_PG_PASSWORD"],
        "database": os.environ["DEDUP_LOCAL_PG_DATABASE"],
    }


def _connect(settings: dict, database: str):
    return psycopg2.connect(
        host=settings["host"], port=settings["port"], dbname=database,
        user=settings["user"], password=settings["password"], connect_timeout=5,
        application_name="dedup_local_parity_load",
    )


def _assert_local(conn) -> tuple[str, str]:
    cur = conn.cursor()
    cur.execute("SELECT current_database(), COALESCE(inet_server_addr()::text, 'local'), current_setting('server_version'), "
                "(SELECT datcollate FROM pg_database WHERE datname = current_database())")
    database, address, version, collate = cur.fetchone()
    cur.close()
    if str(address).startswith(PRODUCTION_HOST):
        raise SystemExit(f"Refusing local load: server address is {address}")
    if address not in {"local"} and not str(address).startswith(("127.0.0.1", "::1")):
        raise SystemExit(f"Refusing local load: server address is {address}")
    return version, collate


def main() -> None:
    settings = _settings()
    extract = json.loads(EXTRACT.read_text(encoding="utf-8"))
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    records = list(extract["records"]) + list(synthetic["records"])
    admin = _connect(settings, "postgres")
    admin.autocommit = True
    _assert_local(admin)
    cur = admin.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings["database"],))
    exists = cur.fetchone() is not None
    if not exists:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings["database"])))
    cur.close()
    admin.close()

    conn = _connect(settings, settings["database"])
    version, collate = _assert_local(conn)
    cur = conn.cursor()
    cur.execute(SCHEMA)
    for record in records:
        cur.execute(
            "INSERT INTO public.persons_unified "
            "(person_id, source_system, source_record_id, full_name, alias, relative_name, phone_number) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (record["person_id"], record["source_system"], record["source_record_id"], record.get("full_name"),
             record.get("alias"), record.get("relative_name"), record.get("phone_number")),
        )
        cur.execute(
            "INSERT INTO public.person_enrichment (person_id, cleaned_full_name, cleaned_alias, cleaned_relative_name) "
            "VALUES (%s, %s, %s, %s)",
            (record["person_id"], record.get("cleaned_full_name"), record.get("cleaned_alias"), record.get("cleaned_relative_name")),
        )
    cur.execute("SELECT count(*) FROM public.persons_unified")
    persons = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM public.person_enrichment")
    enrichment = cur.fetchone()[0]
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'persons_unified' ORDER BY ordinal_position"
    )
    columns = [row[0] for row in cur.fetchall()]
    conn.commit()
    cur.close()
    conn.close()
    production = sum(1 for record in records if record.get("origin") == "production")
    print(f"local_database={settings['database']} host={settings['host']} version={version} collate={collate}")
    print(f"persons={persons} enrichment={enrichment} production_rows={production} synthetic_rows={persons - production}")
    print("columns=" + ",".join(columns))


if __name__ == "__main__":
    main()
