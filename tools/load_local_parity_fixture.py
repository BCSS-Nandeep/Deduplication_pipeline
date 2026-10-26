"""Load the bounded parity fixture into the local test database.

Refuses any host other than loopback and refuses the production database name.
Requires DEDUP_LOCAL_PG_USER, DEDUP_LOCAL_PG_PASSWORD and
DEDUP_LOCAL_PG_DATABASE=dopams_dedup_parity. The database must already exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_HOST = "192.168.103.106"
LOCAL_DATABASE = "dopams_dedup_parity"


def main() -> None:
    host = os.environ.get("DEDUP_LOCAL_PG_HOST", "127.0.0.1")
    database = os.environ.get("DEDUP_LOCAL_PG_DATABASE", "")
    user = os.environ.get("DEDUP_LOCAL_PG_USER", "")
    password = os.environ.get("DEDUP_LOCAL_PG_PASSWORD", "")
    if host in {PRODUCTION_HOST, ""} or host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(f"Refusing load: host {host!r} is not the local server")
    if database != LOCAL_DATABASE:
        raise SystemExit(f"Refusing load: database must be {LOCAL_DATABASE}")
    if not user or not password:
        raise SystemExit("Local PostgreSQL username and password are required")
    conn = psycopg2.connect(
        host=host, port=int(os.environ.get("DEDUP_LOCAL_PG_PORT", "5432")),
        dbname=database, user=user, password=password, connect_timeout=5,
        application_name="dedup_local_fixture_load",
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT current_database(), COALESCE(inet_server_addr()::text, 'local')")
    found_db, address = cur.fetchone()
    if found_db != LOCAL_DATABASE or str(address).startswith(PRODUCTION_HOST):
        conn.close()
        raise SystemExit(f"Refusing load on {address} / {found_db}")
    extract = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_extract.json").read_text(encoding="utf-8"))
    synthetic = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_synthetic.json").read_text(encoding="utf-8"))
    people = extract["records"] + synthetic["records"]
    if len(people) != len({row["person_id"] for row in people}):
        raise SystemExit("Duplicate person_id in the fixture")
    cur.execute("DROP TABLE IF EXISTS public.person_enrichment")
    cur.execute("DROP TABLE IF EXISTS public.persons_unified")
    cur.execute(
        """
        CREATE TABLE public.persons_unified (
            person_id varchar(100) PRIMARY KEY,
            source_system varchar(2) NOT NULL,
            source_record_id varchar(100) NOT NULL,
            full_name varchar(500),
            alias varchar(255),
            relative_name varchar(255),
            phone_number varchar(20)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE public.person_enrichment (
            person_id varchar(100) PRIMARY KEY,
            cleaned_full_name text,
            cleaned_alias varchar(255),
            cleaned_relative_name text
        )
        """
    )
    columns = (
        "person_id", "source_system", "source_record_id", "full_name", "alias", "relative_name", "phone_number",
    )
    cur.executemany(
        "INSERT INTO public.persons_unified (person_id, source_system, source_record_id, full_name, alias, "
        "relative_name, phone_number) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        [tuple(row[name] for name in columns) for row in people],
    )
    cur.executemany(
        "INSERT INTO public.person_enrichment (person_id, cleaned_full_name, cleaned_alias, cleaned_relative_name) "
        "VALUES (%s, %s, %s, %s)",
        [(row["person_id"], row["cleaned_full_name"], row["cleaned_alias"], row["cleaned_relative_name"]) for row in people],
    )
    cur.execute("SELECT count(*) FROM public.persons_unified")
    stored = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"local_rows={stored} production_rows={len(extract['records'])} synthetic_rows={len(synthetic['records'])}")


if __name__ == "__main__":
    main()
