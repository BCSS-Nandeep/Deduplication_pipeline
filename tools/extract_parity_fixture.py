"""One bounded read-only extract for local candidate-parity tests.

Connects only to the approved production database, checks the session is
read-only, then reads the columns the candidate SQL uses. It does not run
candidate-generation SQL and does not write to the database.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dedup_pipeline.db import connect_read_only, load_connection_settings

PRODUCTION_HOST = "192.168.103.106"
ROW_LIMIT = 80
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "candidate_parity_extract.json"

COLUMNS = (
    "person_id", "source_system", "source_record_id", "full_name", "alias", "relative_name", "phone_number",
)
ENRICHMENT = ("person_id", "cleaned_full_name", "cleaned_alias", "cleaned_relative_name")


def main() -> None:
    settings = load_connection_settings(str(Path(__file__).resolve().parents[1] / ".env"))
    if settings.host != PRODUCTION_HOST or settings.database != "dopams_cctns_v2":
        raise SystemExit(f"Refusing extract: connected target is {settings.host} / {settings.database}")
    with connect_read_only(settings, "dedup_bounded_extract") as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT current_setting('default_transaction_read_only'), current_database(), "
            "inet_server_addr()::text, current_setting('server_version'), "
            "(SELECT datcollate FROM pg_database WHERE datname = current_database())"
        )
        read_only, database, address, version, collate = cur.fetchone()
        if read_only != "on" or database != "dopams_cctns_v2" or not str(address).startswith(PRODUCTION_HOST):
            raise SystemExit(f"Refusing extract: read_only={read_only} database={database} address={address}")
        cur.execute(
            "SELECT c.relname, a.attname "
            "FROM pg_index x "
            "JOIN pg_class c ON c.oid = x.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(x.indkey) "
            "WHERE n.nspname = 'public' AND c.relname IN ('persons_unified', 'person_enrichment') "
            "AND x.indisprimary ORDER BY c.relname, a.attnum"
        )
        primary = [f"{table}.{column}" for table, column in cur.fetchall()]
        if "persons_unified.person_id" not in primary:
            raise SystemExit(f"Refusing extract: no person_id primary key ({primary})")
        cur.execute(
            "SELECT person_id, source_system, source_record_id, full_name, alias, relative_name, phone_number "
            "FROM public.persons_unified ORDER BY person_id LIMIT %(n)s",
            {"n": ROW_LIMIT},
        )
        people = [dict(zip(COLUMNS, row)) for row in cur.fetchall()]
        ids = [person["person_id"] for person in people]
        cur.execute(
            "SELECT person_id, cleaned_full_name, cleaned_alias, cleaned_relative_name "
            "FROM public.person_enrichment WHERE person_id = ANY(%(ids)s)",
            {"ids": ids},
        )
        enrichment = {row[0]: dict(zip(ENRICHMENT, row)) for row in cur.fetchall()}
        cur.close()
    for person in people:
        extra = enrichment.get(person["person_id"], {})
        person["cleaned_full_name"] = extra.get("cleaned_full_name")
        person["cleaned_alias"] = extra.get("cleaned_alias")
        person["cleaned_relative_name"] = extra.get("cleaned_relative_name")
        person["origin"] = "production"
    payload = {
        "extracted_on": date.today().isoformat(),
        "source": {
            "host": PRODUCTION_HOST,
            "port": 5432,
            "database": "dopams_cctns_v2",
            "read_only": read_only,
            "server_version": version,
            "lc_collate": collate,
            "address": address,
        },
        "extract": {
            "persons_sql": "SELECT required columns FROM public.persons_unified ORDER BY person_id LIMIT 80",
            "enrichment_sql": "SELECT required columns FROM public.person_enrichment WHERE person_id = ANY(extracted ids)",
            "primary_key": primary,
            "row_count": len(people),
        },
        "columns": list(COLUMNS) + ["cleaned_full_name", "cleaned_alias", "cleaned_relative_name"],
        "records": people,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"rows={len(people)} enrichment={len(enrichment)} version={version} collate={collate} file={OUT.name}")


if __name__ == "__main__":
    main()
