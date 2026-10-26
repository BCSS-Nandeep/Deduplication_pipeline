"""Every statement the candidate stage runs: read-only SELECTs, grouped on the database side.

Each query starts with a ``/* candidates:<tag> */`` comment (ignored by the server, used by tests and logs).
Block queries return one row per group of two or more persons:
``(key, person_ids[], source_systems[], source_record_ids[] [, is_alias[]])``. Names and numbers are only
used as grouping keys in memory; they are not written to the output unless ``export_comparison_keys`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..identity.names import COUNTRY_TOKENS, ROLE_TOKENS
from .config import (
    ALIAS_NAME, NAME_EXACT, NAME_INITIAL, PHONE_EXACT, RELATIVE_EXACT, RELATIVE_SORTED, TOKEN_SORTED, CandidateConfig,
)

PERSONS_WITH_ENRICHMENT = (
    "FROM public.persons_unified p LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id"
)
# Same rule as normalize.normalize_name: non-alphanumeric characters -> space, collapse whitespace, lower-case.
_NORMALIZE = (
    "lower(btrim(regexp_replace(regexp_replace({col}, '[^[:alnum:][:space:]]+', ' ', 'g'), '[[:space:]]+', ' ', 'g')))"
)
_TOKENS = "cardinality(regexp_split_to_array({key}, ' '))"
_AGGREGATES = (
    "array_agg(person_id ORDER BY person_id), array_agg(source_system ORDER BY person_id), "
    "array_agg(source_record_id ORDER BY person_id)"
)


@dataclass(frozen=True)
class QuerySpec:
    tag: str
    sql: str
    params: dict[str, Any]


def _name_column(config: CandidateConfig) -> str:
    if config.name_source == "cleaned_then_raw":
        return "COALESCE(NULLIF(btrim(e.cleaned_full_name), ''), p.full_name)"
    return "p.full_name"


def _alias_column(config: CandidateConfig) -> str:
    if config.name_source == "cleaned_then_raw":
        return "COALESCE(NULLIF(btrim(e.cleaned_alias), ''), p.alias)"
    return "p.alias"


def _clean_params(config: CandidateConfig) -> dict[str, Any]:
    if not config.metadata_name_cleaning:
        return {}
    return {"name_roles": sorted(ROLE_TOKENS), "name_countries": sorted(COUNTRY_TOKENS)}


def _strip_case(column: str) -> str:
    """One metadata-token removal, same order as ``normalize_identity_name``."""
    return (
        f"CASE WHEN cardinality({column}) > 1 AND {column}[1] = 'late' THEN {column}[2:cardinality({column})] "
        f"WHEN cardinality({column}) >= 2 AND {column}[cardinality({column})] = ANY(%(name_roles)s) "
        f"THEN {column}[1:cardinality({column})-1] "
        f"WHEN cardinality({column}) >= 3 AND {column}[cardinality({column})] = ANY(%(name_countries)s) "
        f"THEN {column}[1:cardinality({column})-1] ELSE {column} END"
    )


def _relative_column(config: CandidateConfig) -> str:
    if config.name_source == "cleaned_then_raw":
        return "COALESCE(NULLIF(btrim(e.cleaned_relative_name), ''), p.relative_name)"
    return "p.relative_name"


def _name_keys_cte(config: CandidateConfig, column_sql: Optional[str] = None) -> str:
    column = _name_column(config) if column_sql is None else column_sql
    normalized = _NORMALIZE.format(col=column)
    if not config.metadata_name_cleaning:
        return (
            "WITH k AS (SELECT p.person_id, p.source_system, p.source_record_id, "
            f"{normalized} AS key {PERSONS_WITH_ENRICHMENT})"
        )
    # One CTE per removal. MATERIALIZED stops PostgreSQL from inlining each step into the next:
    # each step mentions the token array many times, and six inlined steps explode the planner.
    parts = [
        "WITH raw AS MATERIALIZED (SELECT p.person_id, p.source_system, p.source_record_id, "
        f"regexp_split_to_array({normalized}, ' ') AS t {PERSONS_WITH_ENRICHMENT})",
    ]
    previous = "raw"
    for index in range(1, 7):
        parts.append(
            f", c{index} AS MATERIALIZED (SELECT person_id, source_system, source_record_id, {_strip_case('t')} AS t FROM {previous})"
        )
        previous = f"c{index}"
    parts.append(
        f", k AS MATERIALIZED (SELECT person_id, source_system, source_record_id, btrim(array_to_string(t, ' ')) AS key FROM {previous})"
    )
    return "".join(parts)


# ------------------------------------------------------------------ safety / bookkeeping

def preflight_person_ids() -> QuerySpec:
    """Person ids must be lower-case alphanumeric so Python's ordering equals the database's (canonical pair order)."""
    return QuerySpec(
        "preflight_person_ids",
        "/* candidates:preflight_person_ids */ SELECT count(*) FROM public.persons_unified WHERE person_id !~ '^[0-9a-z]+$'",
        {},
    )


def dataset_stamp() -> QuerySpec:
    return QuerySpec(
        "dataset_stamp",
        "/* candidates:dataset_stamp */ SELECT "
        "(SELECT count(*) FROM public.persons_unified), (SELECT max(computed_at) FROM public.persons_unified), "
        "(SELECT count(*) FROM public.person_enrichment), (SELECT max(computed_at) FROM public.person_enrichment)",
        {},
    )


# -------------------------------------------------------------------------- block queries

def name_exact_blocks(config: CandidateConfig) -> QuerySpec:
    return QuerySpec(
        "name_exact:blocks",
        f"/* candidates:name_exact:blocks */ {_name_keys_cte(config)} "
        f"SELECT key, {_AGGREGATES} FROM k "
        f"WHERE key <> '' AND char_length(key) >= %(min_chars)s AND {_TOKENS.format(key='key')} >= %(min_tokens)s "
        "GROUP BY key HAVING count(*) > 1 ORDER BY key",
        {"min_chars": config.name_min_chars, "min_tokens": config.name_min_tokens, **_clean_params(config)},
    )


def token_sorted_blocks(config: CandidateConfig) -> QuerySpec:
    sorted_key = f"array_to_string(ARRAY(SELECT t FROM unnest(regexp_split_to_array(key, ' ')) AS t ORDER BY t COLLATE \"C\"), ' ')"
    return QuerySpec(
        "token_sorted_name:blocks",
        f"/* candidates:token_sorted_name:blocks */ {_name_keys_cte(config)}, "
        f"s AS (SELECT person_id, source_system, source_record_id, {sorted_key} AS sorted_key FROM k "
        f"WHERE key <> '' AND char_length(key) >= %(min_chars)s AND {_TOKENS.format(key='key')} >= %(min_tokens)s) "
        f"SELECT sorted_key, {_AGGREGATES} FROM s GROUP BY sorted_key HAVING count(*) > 1 ORDER BY sorted_key",
        {"min_chars": config.name_min_chars, "min_tokens": config.token_sorted_min_tokens, **_clean_params(config)},
    )


def _phone_classification(config: CandidateConfig) -> tuple[str, dict[str, Any]]:
    """CTE ``c(person_id, source_system, source_record_id, key, reason)``; reason is 'ok' for usable numbers."""
    params: dict[str, Any] = {
        "digits": config.phone_digits, "max_digits": config.phone_max_input_digits,
        "leading": list(config.phone_leading_digits), "min_distinct": config.phone_min_distinct_digits,
        "blocklist": list(config.phone_blocklist),
    }
    blocklist_test = "WHEN right(digits, %(digits)s) = ANY(%(blocklist)s) THEN 'blocklisted' " if config.phone_blocklist else ""
    cte = (
        "WITH d AS (SELECT p.person_id, p.source_system, p.source_record_id, p.phone_number, "
        "regexp_replace(p.phone_number, '[^0-9]', '', 'g') AS digits FROM public.persons_unified p), "
        "c AS (SELECT person_id, source_system, source_record_id, right(digits, %(digits)s) AS key, "
        "CASE WHEN phone_number IS NULL OR btrim(phone_number) = '' THEN 'no_phone' "
        "WHEN digits = '' THEN 'no_digits' "
        "WHEN length(digits) < %(digits)s THEN 'too_few_digits' "
        "WHEN length(digits) > %(max_digits)s THEN 'too_many_digits' "
        "WHEN left(right(digits, %(digits)s), 1) <> ALL(%(leading)s) THEN 'bad_leading_digit' "
        "WHEN (SELECT count(DISTINCT x) FROM regexp_split_to_table(right(digits, %(digits)s), '') AS x) < %(min_distinct)s "
        "THEN 'repetitive_digits' "
        f"{blocklist_test}"
        "ELSE 'ok' END AS reason FROM d)"
    )
    return cte, params


def phone_blocks(config: CandidateConfig) -> QuerySpec:
    cte, params = _phone_classification(config)
    return QuerySpec(
        "phone_exact:blocks",
        f"/* candidates:phone_exact:blocks */ {cte} SELECT key, {_AGGREGATES} FROM c "
        "WHERE reason = 'ok' GROUP BY key HAVING count(*) > 1 ORDER BY key",
        params,
    )


def alias_blocks(config: CandidateConfig) -> QuerySpec:
    name_col, alias_col = _NORMALIZE.format(col=_name_column(config)), _NORMALIZE.format(col=_alias_column(config))
    valid = "char_length({k}) >= %(min_chars)s AND " + _TOKENS.format(key="{k}") + " >= %(min_tokens)s"
    return QuerySpec(
        "alias_name:blocks",
        "/* candidates:alias_name:blocks */ "
        f"WITH base AS (SELECT p.person_id, p.source_system, p.source_record_id, {name_col} AS name_key, "
        f"{alias_col} AS alias_key {PERSONS_WITH_ENRICHMENT}), "
        "k AS ("
        f"SELECT person_id, source_system, source_record_id, name_key AS key, false AS is_alias FROM base "
        f"WHERE name_key <> '' AND {valid.format(k='name_key')} "
        "UNION ALL "
        f"SELECT person_id, source_system, source_record_id, alias_key, true FROM base "
        f"WHERE alias_key <> '' AND {valid.format(k='alias_key')}), "
        "g AS (SELECT key, person_id, source_system, source_record_id, bool_or(is_alias) AS is_alias FROM k "
        "GROUP BY key, person_id, source_system, source_record_id) "
        f"SELECT key, {_AGGREGATES}, array_agg(is_alias ORDER BY person_id) FROM g "
        "GROUP BY key HAVING count(*) > 1 AND bool_or(is_alias) ORDER BY key",
        {"min_chars": config.name_min_chars, "min_tokens": config.alias_min_tokens},
    )


# ------------------------------------------------------- what the rules exclude (aggregates only)

def name_exclusions(config: CandidateConfig, min_tokens: int, tag: str) -> list[QuerySpec]:
    params = {"min_chars": config.name_min_chars, "min_tokens": min_tokens, **_clean_params(config)}
    cte = _name_keys_cte(config)
    toks = _TOKENS.format(key="key")
    return [
        QuerySpec(
            f"{tag}:exclusions",
            f"/* candidates:{tag}:exclusions */ {cte} SELECT count(*), count(*) FILTER (WHERE key = ''), "
            f"count(*) FILTER (WHERE key <> '' AND {toks} < %(min_tokens)s), "
            f"count(*) FILTER (WHERE key <> '' AND {toks} >= %(min_tokens)s AND char_length(key) < %(min_chars)s) FROM k",
            params,
        ),
        QuerySpec(
            f"{tag}:excluded_blocks",
            f"/* candidates:{tag}:excluded_blocks */ {cte} SELECT count(*), COALESCE(sum(n), 0), COALESCE(sum(n * (n - 1) / 2), 0) "
            "FROM (SELECT count(*) AS n FROM k "
            f"WHERE key <> '' AND ({toks} < %(min_tokens)s OR char_length(key) < %(min_chars)s) GROUP BY key HAVING count(*) > 1) s",
            params,
        ),
    ]


def phone_exclusions(config: CandidateConfig) -> list[QuerySpec]:
    cte, params = _phone_classification(config)
    return [
        QuerySpec("phone_exact:exclusions", f"/* candidates:phone_exact:exclusions */ {cte} "
                  "SELECT reason, count(*) FROM c GROUP BY reason ORDER BY reason", params),
        QuerySpec("phone_exact:excluded_blocks", f"/* candidates:phone_exact:excluded_blocks */ {cte} "
                  "SELECT reason, count(*), sum(n), sum(n * (n - 1) / 2) FROM (SELECT reason, key, count(*) AS n FROM c "
                  "WHERE reason IN ('bad_leading_digit', 'repetitive_digits', 'blocklisted') GROUP BY reason, key "
                  "HAVING count(*) > 1) s GROUP BY reason ORDER BY reason", params),
    ]


def alias_exclusions(config: CandidateConfig) -> list[QuerySpec]:
    alias_norm = _NORMALIZE.format(col=_alias_column(config))
    return [
        QuerySpec(
            "alias_name:exclusions",
            f"/* candidates:alias_name:exclusions */ WITH a AS (SELECT {alias_norm} AS key {PERSONS_WITH_ENRICHMENT}) "
            "SELECT count(*) FILTER (WHERE key <> ''), "
            f"count(*) FILTER (WHERE key <> '' AND ({_TOKENS.format(key='key')} < %(min_tokens)s OR char_length(key) < %(min_chars)s)) FROM a",
            {"min_chars": config.name_min_chars, "min_tokens": config.alias_min_tokens},
        )
    ]


def relative_exact_blocks(config: CandidateConfig) -> QuerySpec:
    return QuerySpec(
        "relative_exact:blocks",
        f"/* candidates:relative_exact:blocks */ {_name_keys_cte(config, _relative_column(config))} "
        f"SELECT key, {_AGGREGATES} FROM k "
        f"WHERE key <> '' AND char_length(key) >= %(min_chars)s AND {_TOKENS.format(key='key')} >= %(min_tokens)s "
        "GROUP BY key HAVING count(*) > 1 ORDER BY key",
        {"min_chars": config.name_min_chars, "min_tokens": config.relative_min_tokens, **_clean_params(config)},
    )


def relative_token_sorted_blocks(config: CandidateConfig) -> QuerySpec:
    sorted_key = f"array_to_string(ARRAY(SELECT t FROM unnest(regexp_split_to_array(key, ' ')) AS t ORDER BY t COLLATE \"C\"), ' ')"
    return QuerySpec(
        "relative_token_sorted:blocks",
        f"/* candidates:relative_token_sorted:blocks */ {_name_keys_cte(config, _relative_column(config))}, "
        f"s AS (SELECT person_id, source_system, source_record_id, {sorted_key} AS sorted_key FROM k "
        f"WHERE key <> '' AND char_length(key) >= %(min_chars)s AND {_TOKENS.format(key='key')} >= %(min_tokens)s) "
        f"SELECT sorted_key, {_AGGREGATES} FROM s GROUP BY sorted_key HAVING count(*) > 1 ORDER BY sorted_key",
        {"min_chars": config.name_min_chars, "min_tokens": config.relative_min_tokens, **_clean_params(config)},
    )


def _relative_exclusions(config: CandidateConfig, min_tokens: int, tag: str) -> list[QuerySpec]:
    saved = config
    # name_exclusions reads the name CTE. Build the same two statements against the relative column.
    params = {"min_chars": saved.name_min_chars, "min_tokens": min_tokens, **_clean_params(saved)}
    cte = _name_keys_cte(saved, _relative_column(saved))
    toks = _TOKENS.format(key="key")
    return [
        QuerySpec(
            f"{tag}:exclusions",
            f"/* candidates:{tag}:exclusions */ {cte} SELECT count(*), count(*) FILTER (WHERE key = ''), "
            f"count(*) FILTER (WHERE key <> '' AND {toks} < %(min_tokens)s), "
            f"count(*) FILTER (WHERE key <> '' AND {toks} >= %(min_tokens)s AND char_length(key) < %(min_chars)s) FROM k",
            params,
        ),
        QuerySpec(
            f"{tag}:excluded_blocks",
            f"/* candidates:{tag}:excluded_blocks */ {cte} SELECT count(*), COALESCE(sum(n), 0), COALESCE(sum(n * (n - 1) / 2), 0) "
            "FROM (SELECT count(*) AS n FROM k "
            f"WHERE key <> '' AND ({toks} < %(min_tokens)s OR char_length(key) < %(min_chars)s) GROUP BY key HAVING count(*) > 1) s",
            params,
        ),
    ]


def name_initial_blocks(config: CandidateConfig) -> QuerySpec:
    """Index literal initials and one-token reductions of fully spelled names under the same key."""
    cte = _name_keys_cte(config)
    return QuerySpec(
        "name_initial:blocks",
        f"/* candidates:name_initial:blocks */ {cte}, "
        "toks AS (SELECT k.person_id, k.source_system, k.source_record_id, t.token, t.ord::int "
        "FROM k CROSS JOIN LATERAL regexp_split_to_table(k.key, ' ') WITH ORDINALITY AS t(token, ord) "
        "WHERE k.key <> ''), "
        "literal AS (SELECT person_id, source_system, source_record_id, "
        "(SELECT array_to_string(ARRAY(SELECT t2.token FROM toks t2 WHERE t2.person_id = toks.person_id "
        "AND char_length(t2.token) >= 2 ORDER BY t2.token COLLATE \"C\"), ' ') || '|' || "
        "(SELECT string_agg(t3.token, '' ORDER BY t3.token COLLATE \"C\") FROM toks t3 "
        "WHERE t3.person_id = toks.person_id AND char_length(t3.token) = 1)) AS key, true AS holds_initial "
        "FROM toks GROUP BY person_id, source_system, source_record_id "
        "HAVING bool_or(char_length(token) = 1) AND bool_or(char_length(token) >= %(anchor)s) "
        "AND count(*) FILTER (WHERE char_length(token) >= 2) >= 1), "
        "reduced AS (SELECT t.person_id, t.source_system, t.source_record_id, "
        "(SELECT array_to_string(ARRAY(SELECT t2.token FROM toks t2 WHERE t2.person_id = t.person_id AND t2.ord <> t.ord "
        "AND char_length(t2.token) >= 2 ORDER BY t2.token COLLATE \"C\"), ' ') || '|' || left(t.token, 1)) AS key, "
        "false AS holds_initial FROM toks t WHERE char_length(t.token) >= %(anchor)s "
        "AND NOT EXISTS (SELECT 1 FROM toks i WHERE i.person_id = t.person_id AND char_length(i.token) = 1) "
        "AND EXISTS (SELECT 1 FROM toks o WHERE o.person_id = t.person_id AND o.ord <> t.ord AND char_length(o.token) >= 2)), "
        "u AS (SELECT key, person_id, source_system, source_record_id, bool_or(holds_initial) AS holds_initial "
        "FROM (SELECT * FROM literal UNION ALL SELECT * FROM reduced) s GROUP BY key, person_id, source_system, source_record_id) "
        f"SELECT key, {_AGGREGATES}, array_agg(holds_initial ORDER BY person_id) FROM u "
        "WHERE key <> '' AND strpos(key, '|') <> 1 GROUP BY key HAVING count(*) > 1 ORDER BY key",
        {"anchor": config.name_initial_min_anchor_chars, **_clean_params(config)},
    )


def name_initial_exclusions(config: CandidateConfig) -> list[QuerySpec]:
    cte = _name_keys_cte(config)
    toks = _TOKENS.format(key="key")
    params = {"anchor": config.name_initial_min_anchor_chars, "min_tokens": 2, **_clean_params(config)}
    return [
        QuerySpec(
            "name_initial:exclusions",
            f"/* candidates:name_initial:exclusions */ {cte} SELECT count(*), "
            "count(*) FILTER (WHERE key = ''), "
            f"count(*) FILTER (WHERE key <> '' AND {toks} < %(min_tokens)s), "
            "count(*) FILTER (WHERE key <> '' AND NOT EXISTS ("
            "SELECT 1 FROM regexp_split_to_table(key, ' ') tok WHERE char_length(tok) >= %(anchor)s)) FROM k",
            params,
        ),
        QuerySpec(
            "name_initial:excluded_blocks",
            "/* candidates:name_initial:excluded_blocks */ SELECT 0, 0, 0",
            {},
        ),
    ]


def block_query(strategy: str, config: CandidateConfig) -> QuerySpec:
    return {NAME_EXACT: name_exact_blocks, TOKEN_SORTED: token_sorted_blocks, PHONE_EXACT: phone_blocks,
            ALIAS_NAME: alias_blocks, NAME_INITIAL: name_initial_blocks, RELATIVE_EXACT: relative_exact_blocks,
            RELATIVE_SORTED: relative_token_sorted_blocks}[strategy](config)


def exclusion_queries(strategy: str, config: CandidateConfig) -> list[QuerySpec]:
    if strategy == NAME_EXACT:
        return name_exclusions(config, config.name_min_tokens, "name_exact")
    if strategy == TOKEN_SORTED:
        return name_exclusions(config, config.token_sorted_min_tokens, "token_sorted_name")
    if strategy == RELATIVE_EXACT:
        return _relative_exclusions(config, config.relative_min_tokens, "relative_exact")
    if strategy == RELATIVE_SORTED:
        return _relative_exclusions(config, config.relative_min_tokens, "relative_token_sorted")
    if strategy == NAME_INITIAL:
        return name_initial_exclusions(config)
    if strategy == PHONE_EXACT:
        return phone_exclusions(config)
    return alias_exclusions(config)
