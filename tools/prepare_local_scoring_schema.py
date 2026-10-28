"""Add nullable scoring attribute columns on the local test database only.

Loopback only. Never connects to production. Does not rewrite person_id values.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_HOST = "192.168.103.106"

DDL = [
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS age integer",
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS date_of_birth date",
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS gender character varying(50)",
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS present_address_text text",
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS permanent_address_text text",
    "ALTER TABLE public.persons_unified ADD COLUMN IF NOT EXISTS computed_at timestamptz",
    "ALTER TABLE public.person_enrichment ADD COLUMN IF NOT EXISTS address_resolution jsonb",
    "ALTER TABLE public.person_enrichment ADD COLUMN IF NOT EXISTS computed_at timestamptz",
    "UPDATE public.persons_unified SET computed_at = COALESCE(computed_at, NOW()) WHERE computed_at IS NULL",
    "UPDATE public.person_enrichment SET computed_at = COALESCE(computed_at, NOW()) WHERE computed_at IS NULL",
]


def _load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key.startswith("DEDUP_LOCAL_PG_"):
            os.environ.setdefault(key, value)


def main() -> None:
    _load_env()
    host = os.environ.get("DEDUP_LOCAL_PG_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} or host == PRODUCTION_HOST:
        raise SystemExit(f"Refusing schema change: host {host!r} is not loopback")
    missing = [n for n in ("DEDUP_LOCAL_PG_USER", "DEDUP_LOCAL_PG_PASSWORD", "DEDUP_LOCAL_PG_DATABASE") if not os.environ.get(n)]
    if missing:
        raise SystemExit("Missing " + ", ".join(missing))
    conn = psycopg2.connect(
        host=host,
        port=int(os.environ.get("DEDUP_LOCAL_PG_PORT", "5432")),
        dbname=os.environ["DEDUP_LOCAL_PG_DATABASE"],
        user=os.environ["DEDUP_LOCAL_PG_USER"],
        password=os.environ["DEDUP_LOCAL_PG_PASSWORD"],
        application_name="dedup_local_scoring_schema",
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT current_database(), COALESCE(inet_server_addr()::text, 'local')")
    database, address = cur.fetchone()
    if not (address == "local" or str(address).startswith(("127.0.0.1", "::1"))):
        raise SystemExit(f"Refusing schema change: address {address!r}")
    if database == "dopams_cctns_v2":
        raise SystemExit("Refusing schema change on production database name")
    for stmt in DDL:
        print("EXEC", stmt)
        cur.execute(stmt)
    cur.execute("SELECT COUNT(*) FROM public.persons_unified WHERE person_id !~ '^[0-9a-z]+$'")
    bad = cur.fetchone()[0]
    print(f"database={database} address={address} bad_person_ids={bad}")
    if bad:
        raise SystemExit("Local persons_unified still has ids outside ^[0-9a-z]+$; refuse to proceed")
    cur.close()
    conn.close()
    print("local scoring schema ready")


if __name__ == "__main__":
    main()
