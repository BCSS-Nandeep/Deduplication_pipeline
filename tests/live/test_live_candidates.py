"""Candidate generation against the live ``dopams_cctns_v2`` database (read-only), checked two independent ways.

Skipped unless DEDUP_LIVE_DB=1 (credentials come from the usual DEDUP_PG_* / UNIFIED_PG_* settings).

1. The SQL normalisation expressions are run on probe strings via ``SELECT ... FROM (VALUES ...)`` (a read-only
   SELECT, creating nothing) and compared with the Python reference rules.
2. The whole pipeline is run, then a pure-Python reference implementation, which reads the raw person rows and
   applies the same rules with no SQL grouping, must produce exactly the same pairs with the same rule
   attribution.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pytest

from dedup_pipeline.candidates import CandidateConfig, RunOptions, run_candidate_generation
from dedup_pipeline.candidates import sql as sqlmod
from dedup_pipeline.candidates.config import ALIAS_NAME, NAME_EXACT, PHONE_EXACT, TOKEN_SORTED
from dedup_pipeline.candidates.normalize import (
    name_exclusion_reason, normalize_name, phone_key, pick_name, token_sorted_key,
)
from dedup_pipeline.db import connect_read_only, load_connection_settings

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("DEDUP_LIVE_DB") != "1", reason="DEDUP_LIVE_DB=1 not set")]


@pytest.fixture(scope="module")
def conn():
    with connect_read_only(load_connection_settings(), "dedup_live_candidates_test") as c:
        yield c


def reference(persons, cfg: CandidateConfig) -> dict[tuple[str, str], set[str]]:
    """Independent implementation: group rows in Python with the reference normalisers."""
    found: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(strategy, members):
        for a, b in combinations(sorted(members), 2):
            found[(a, b)].add(strategy)

    name_groups, sorted_groups, phone_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    alias_groups: dict[str, dict[str, bool]] = defaultdict(dict)
    for p in persons:
        name = normalize_name(pick_name(p["cleaned_full_name"], p["full_name"], cfg))
        if name_exclusion_reason(name, cfg.name_min_tokens, cfg.name_min_chars) is None:
            name_groups[name].append(p["person_id"])
        if name_exclusion_reason(name, cfg.token_sorted_min_tokens, cfg.name_min_chars) is None:
            sorted_groups[token_sorted_key(name)].append(p["person_id"])
        key, _ = phone_key(p["phone_number"], cfg)
        if key:
            phone_groups[key].append(p["person_id"])
        if name_exclusion_reason(name, cfg.alias_min_tokens, cfg.name_min_chars) is None:
            alias_groups[name].setdefault(p["person_id"], False)
        alias = normalize_name(pick_name(p["cleaned_alias"], p["alias"], cfg))
        if name_exclusion_reason(alias, cfg.alias_min_tokens, cfg.name_min_chars) is None:
            alias_groups[alias][p["person_id"]] = True
    for groups, strategy, cap in ((name_groups, NAME_EXACT, cfg.name_max_block_size),
                                  (sorted_groups, TOKEN_SORTED, cfg.token_sorted_max_block_size),
                                  (phone_groups, PHONE_EXACT, cfg.phone_max_block_size)):
        for members in groups.values():
            if 2 <= len(members) <= cap:
                add(strategy, members)
    for members in alias_groups.values():
        if 2 <= len(members) <= cfg.alias_max_block_size and any(members.values()):
            for a, b in combinations(sorted(members), 2):
                if members[a] or members[b]:
                    found[(a, b)].add(ALIAS_NAME)
    return found


def test_sql_normalisation_matches_the_python_rules(conn):
    probes = ["  Ramesh   KUMAR ", "O'Brien-Smith, Jr.", "Alle Sai-Shiva", "S/O Raju", "A.B.C. Reddy", "తెలుగు పేరు",
              "María  José", "x_y z", "", "   ", "Name\tWith\ttabs", "ÅNGSTRÖM Lars", "12 Main St."]
    values = ", ".join("(%s)" for _ in probes)
    cur = conn.cursor()
    cur.execute(f"SELECT {sqlmod._NORMALIZE.format(col='v.n')} FROM (VALUES {values}) AS v(n)", probes)
    assert [r[0] for r in cur.fetchall()] == [normalize_name(s) for s in probes]
    phones = ["9876543211", "+91 98765-43211", "09876543211", "919876543211", "1234567890", "9999999999", "5876543210", "98765", "", None,
              "9876543211 9876543212", "0000000000"]
    cfg = CandidateConfig()
    cte, params = sqlmod._phone_classification(cfg)  # the production classification, pointed at literal probe rows

    def literal(value):
        return "NULL" if value is None else "'" + value.replace("'", "''") + "'"

    probe_rows = ", ".join(f"({i}, {literal(ph)})" for i, ph in enumerate(phones))
    probe_sql = cte.replace(
        "p.person_id, p.source_system, p.source_record_id, p.phone_number,",
        "p.person_id, 'x' AS source_system, 'x' AS source_record_id, p.phone_number,",
    ).replace("FROM public.persons_unified p)", f"FROM (VALUES {probe_rows}) AS p(person_id, phone_number))")
    assert "VALUES" in probe_sql and "public.persons_unified" not in probe_sql
    cur.execute(f"{probe_sql} SELECT person_id, key, reason FROM c ORDER BY person_id", params)
    sql_result = {r[0]: r[2] for r in cur.fetchall()}
    for i, ph in enumerate(phones):
        key, reason = phone_key(ph, cfg)
        assert sql_result[i] == (reason or "ok"), (ph, sql_result[i], reason)


def test_sql_and_python_normalisation_agree_on_every_real_name(conn):
    """All names and aliases in the database, not just probes. Only counts are reported on failure (no names)."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT n, {sqlmod._NORMALIZE.format(col='n')} FROM (SELECT full_name AS n FROM public.persons_unified "
        "UNION SELECT alias FROM public.persons_unified UNION SELECT cleaned_full_name FROM public.person_enrichment "
        "UNION SELECT cleaned_alias FROM public.person_enrichment) AS names WHERE n IS NOT NULL")
    rows = cur.fetchall()
    assert len(rows) > 50_000
    mismatches = [1 for raw, sql_key in rows if normalize_name(raw) != sql_key]
    assert not mismatches, f"{len(mismatches)} of {len(rows)} distinct names normalise differently in SQL and Python"


@pytest.fixture(scope="module")
def run(conn, tmp_path_factory):
    out = tmp_path_factory.mktemp("live_candidates")
    cfg = CandidateConfig()
    summary = run_candidate_generation(conn, cfg, RunOptions(output_dir=out))
    pairs = {}
    for line in (out / "candidates.jsonl").read_text().splitlines():
        rec = json.loads(line)
        pairs[(rec["person_a_id"], rec["person_b_id"])] = set(rec["rule_names"])
    return cfg, summary, pairs, out


@pytest.fixture(scope="module")
def persons(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT p.person_id, p.source_system, p.full_name, e.cleaned_full_name, p.alias, e.cleaned_alias, p.phone_number "
        "FROM public.persons_unified p LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id ORDER BY p.person_id")
    keys = ("person_id", "source_system", "full_name", "cleaned_full_name", "alias", "cleaned_alias", "phone_number")
    return [dict(zip(keys, r)) for r in cur.fetchall()]


def test_pipeline_equals_the_independent_python_reference_exactly(run, persons):
    cfg, summary, pairs, _ = run
    expected = reference(persons, cfg)
    assert set(pairs) == set(expected), (len(pairs), len(expected))
    assert all(pairs[k] == expected[k] for k in pairs), "rule attribution differs for some pair"
    assert summary["union"]["candidate_pairs"] == len(expected)


def test_run_is_complete_read_only_and_ordered(run):
    cfg, summary, pairs, out = run
    assert summary["status"] == "completed" and summary["database"]["transaction_read_only"] == "on"
    assert all(a < b for a, b in pairs)
    assert list(pairs) == sorted(pairs)
    assert (out / "run_summary.json").is_file()
