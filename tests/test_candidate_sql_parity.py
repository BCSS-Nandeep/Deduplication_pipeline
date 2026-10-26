"""Run the production candidate SQL on the local fixture database and compare pairs.

Skipped unless ``DEDUP_SQL_PARITY=1`` (``DEDUP_LOCAL_PARITY=1`` is accepted as the same
switch) and a local login is configured. A skip is not a passing result. The connection
is loopback only; this module never connects to the production host.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import psycopg2
import pytest

from dedup_pipeline.candidates.blocks import expand_block, parse_block_row
from dedup_pipeline.candidates.config import CandidateConfig, final_candidate_config
from dedup_pipeline.candidates.sql import alias_blocks, block_query
from tests.live.test_live_candidates import reference

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_HOST = "192.168.103.106"
_REQUIRED = ("DEDUP_LOCAL_PG_USER", "DEDUP_LOCAL_PG_PASSWORD", "DEDUP_LOCAL_PG_DATABASE")
def _parity_enabled() -> bool:
    return os.environ.get("DEDUP_SQL_PARITY") == "1" or os.environ.get("DEDUP_LOCAL_PARITY") == "1"


pytestmark = pytest.mark.skipif(
    not _parity_enabled() or any(not os.environ.get(name) for name in _REQUIRED),
    reason="DEDUP_SQL_PARITY=1 and local PostgreSQL credentials are not set; SQL was not executed",
)


def _local_connect():
    host = os.environ.get("DEDUP_LOCAL_PG_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(f"parity host {host!r} is not loopback")
    conn = psycopg2.connect(
        host=host, port=int(os.environ.get("DEDUP_LOCAL_PG_PORT", "5432")),
        dbname=os.environ["DEDUP_LOCAL_PG_DATABASE"], user=os.environ["DEDUP_LOCAL_PG_USER"],
        password=os.environ["DEDUP_LOCAL_PG_PASSWORD"], connect_timeout=5,
        application_name="dedup_local_sql_parity",
        options="-c default_transaction_read_only=on -c statement_timeout=60000",
    )
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    read_only = cur.fetchone()[0]
    cur.execute("SELECT current_database(), COALESCE(inet_server_addr()::text, 'local')")
    database, address = cur.fetchone()
    cur.close()
    if read_only != "on" or str(address).startswith(PRODUCTION_HOST) or database == "dopams_cctns_v2" and address.startswith(PRODUCTION_HOST):
        conn.close()
        raise RuntimeError(f"refusing parity connection read_only={read_only} database={database} address={address}")
    if address != "local" and not str(address).startswith(("127.0.0.1", "::1")):
        conn.close()
        raise RuntimeError(f"refusing parity connection to {address}")
    return conn


def _people(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT p.person_id, p.source_system, p.full_name, e.cleaned_full_name, p.alias, e.cleaned_alias, "
        "p.phone_number, p.relative_name, e.cleaned_relative_name "
        "FROM public.persons_unified p LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id "
        "ORDER BY p.person_id"
    )
    keys = ("person_id", "source_system", "full_name", "cleaned_full_name", "alias", "cleaned_alias",
            "phone_number", "relative_name", "cleaned_relative_name")
    rows = [dict(zip(keys, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def sql_pairs(conn, config):
    found = defaultdict(set)
    deferred = []
    for strategy in config.enabled_strategies():
        spec = block_query(strategy, config)
        if "public.persons_unified" not in spec.sql and strategy != "name_initial":
            raise AssertionError(f"{strategy} SQL does not read the local person table")
        assert "age" not in spec.sql and "date_of_birth" not in spec.sql
        cur = conn.cursor()
        cur.execute(spec.sql, spec.params)
        cap = config.max_block_size(strategy)
        for ordinal, row in enumerate(cur.fetchall()):
            block = parse_block_row(strategy, ordinal, row)
            if block.size > cap:
                deferred.append({
                    "strategy": strategy, "reason": "block_too_large", "member_count": block.size,
                    "person_ids": [member.person_id for member in block.members],
                })
                continue
            for left, right in expand_block(block, config.source_policy):
                found[(left.person_id, right.person_id)].add(strategy)
        cur.close()
    return found, deferred


def _assert_same_pairs(actual, expected):
    sql_only = sorted(set(actual) - set(expected))
    python_only = sorted(set(expected) - set(actual))
    assert sql_only == [] and python_only == [], {"sql_only": sql_only, "python_only": python_only}
    mismatched = {pair: {"sql": sorted(actual[pair]), "python": sorted(expected[pair])}
                  for pair in expected if actual[pair] != expected[pair]}
    assert mismatched == {}


@pytest.fixture(scope="module")
def loaded():
    conn = _local_connect()
    people = _people(conn)
    yield conn, people
    conn.close()


def test_final_profile_sql_pairs_match_the_python_reference(loaded):
    conn, people = loaded
    config = final_candidate_config()
    assert "alias_name" not in config.enabled_strategies()
    actual, deferred = sql_pairs(conn, config)
    expected = reference(people, config)
    _assert_same_pairs(actual, expected)
    assert {"token_sorted_name", "relative_exact", "phone_exact"} <= actual[("syn01", "syn02")]
    assert "name_exact" in actual[("syn01", "syn07")]
    assert "name_exact" in actual[("syn01", "syn11")]
    assert "name_initial" in actual[("syn01", "syn03")]
    assert "name_initial" not in actual.get(("syn01", "syn04"), set())
    assert "relative_token_sorted" in actual[("syn02", "syn05")]
    assert "phone_exact" in actual[("syn01", "syn06")]
    assert ("syn08", "syn09") not in actual
    assert ("syn10", "syn11") not in actual
    assert all("alias_name" not in rules for rules in actual.values())
    initial = [block for block in deferred if block["strategy"] == "name_initial"]
    assert len(initial) == 1 and initial[0]["member_count"] == 20
    assert initial[0]["person_ids"] == [f"n{i:02d}" for i in range(20)]
    assert not any("name_initial" in rules and pair[0].startswith("n") for pair, rules in actual.items())

    cur = conn.cursor()
    spec = alias_blocks(config)
    cur.execute(spec.sql, spec.params)
    alias_found = set()
    for ordinal, row in enumerate(cur.fetchall()):
        block = parse_block_row("alias_name", ordinal, row)
        for left, right in expand_block(block, config.source_policy):
            alias_found.add((left.person_id, right.person_id))
    cur.close()
    assert ("syn10", "syn11") in alias_found


def test_legacy_profile_does_not_apply_final_cleaning(loaded):
    conn, people = loaded
    config = CandidateConfig()
    assert config.metadata_name_cleaning is False
    assert "name_initial" not in config.enabled_strategies()
    actual, _deferred = sql_pairs(conn, config)
    expected = reference(people, config)
    _assert_same_pairs(actual, expected)
    assert "name_exact" not in actual.get(("syn01", "syn11"), set())
    assert "name_initial" not in {rule for rules in actual.values() for rule in rules}


def test_loaded_fixture_contains_the_extract_and_synthetic_rows(loaded):
    _conn, people = loaded
    extract = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_extract.json").read_text(encoding="utf-8"))
    synthetic = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_synthetic.json").read_text(encoding="utf-8"))
    ids = {person["person_id"] for person in people}
    assert {row["person_id"] for row in extract["records"]} <= ids
    assert {row["person_id"] for row in synthetic["records"]} <= ids
    assert len(people) == len(extract["records"]) + len(synthetic["records"])
