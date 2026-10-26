"""A fake read-only database for candidate-generation tests.

It answers the tagged queries from ``dedup_pipeline.candidates.sql`` using the Python reference rules over an
in-memory person list. The grouping SQL itself is verified against the live database in
tests/live/test_live_candidates.py; these fakes let the pipeline logic be tested without a server.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterator, Optional

from dedup_pipeline.candidates.config import (
    ALIAS_NAME, NAME_EXACT, NAME_INITIAL, PHONE_EXACT, RELATIVE_EXACT, RELATIVE_SORTED, TOKEN_SORTED, CandidateConfig,
)
from dedup_pipeline.candidates.normalize import (
    name_exclusion_reason, normalize_name, phone_key, pick_name, token_sorted_key,
)
from dedup_pipeline.identity.names import initial_index_keys, normalize_identity_name
from dedup_pipeline.db import check_statement_is_read_only

TAG = re.compile(r"/\* candidates:(\S+) \*/")


def person(pid: str, source: str = "V1", name: Optional[str] = None, phone: Optional[str] = None, alias: Optional[str] = None,
           cleaned: Optional[str] = None, record: Optional[str] = None, relative: Optional[str] = None,
           cleaned_relative: Optional[str] = None) -> dict[str, Any]:
    return {"person_id": pid, "source_system": source, "source_record_id": record or f"rec-{pid}", "full_name": name,
            "cleaned_full_name": cleaned, "alias": alias, "cleaned_alias": None, "phone_number": phone,
            "relative_name": relative, "cleaned_relative_name": cleaned_relative}


class SimulatedCrash(RuntimeError):
    pass


class FakePersonDb:
    def __init__(self, persons: list[dict[str, Any]], config: CandidateConfig, read_only: str = "on",
                 database: str = "dopams_cctns_v2", bad_ids: int = 0, crash_in: Optional[tuple[str, int]] = None,
                 stamp: Optional[tuple] = None):
        self.persons, self.config = persons, config
        self.read_only, self.database, self.bad_ids = read_only, database, bad_ids
        self.crash_in = crash_in  # (tag, n): raise after n rows have been delivered by that stream
        self.stamp = stamp or (len(persons), "2026-10-01T00:00:00", len(persons), "2026-10-01T00:00:00")
        self.statements: list[str] = []
        self.streams: list[str] = []

    # ------------------------------------------------------------------ connection API
    def cursor(self):
        return _Cursor(self)

    def close(self) -> None:
        pass

    def stream(self, sql: str, params: Any = None, batch_size: int = 2000) -> Iterator[tuple]:
        check_statement_is_read_only(sql)
        self.statements.append(sql)
        tag = TAG.search(sql).group(1)
        self.streams.append(tag)
        for index, row in enumerate(self._blocks(tag.split(":")[0])):
            if self.crash_in and self.crash_in[0] == tag and index >= self.crash_in[1]:
                raise SimulatedCrash(f"simulated crash in {tag} after {index} rows")
            yield row

    # ---------------------------------------------------------------------- block rows
    def _rows(self, groups: dict[str, list[dict[str, Any]]], extra: bool = False) -> list[tuple]:
        rows = []
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda m: m["person_id"])
            if len(members) < 2:
                continue
            row = (key, [m["person_id"] for m in members], [m["source_system"] for m in members],
                   [m["source_record_id"] for m in members])
            rows.append(row + ([m.get("_alias", False) for m in members],) if extra else row)
        return rows

    def _cmp(self, raw: Optional[str]) -> str:
        if self.config.metadata_name_cleaning:
            return normalize_identity_name(raw).comparison
        return normalize_name(raw)

    def _name(self, p: dict[str, Any]) -> str:
        return self._cmp(pick_name(p.get("cleaned_full_name"), p.get("full_name"), self.config))

    def _relative(self, p: dict[str, Any]) -> str:
        return self._cmp(pick_name(p.get("cleaned_relative_name"), p.get("relative_name"), self.config))

    def _blocks(self, strategy: str) -> list[tuple]:
        cfg, groups = self.config, defaultdict(list)
        if strategy == NAME_EXACT:
            for p in self.persons:
                key = self._name(p)
                if name_exclusion_reason(key, cfg.name_min_tokens, cfg.name_min_chars) is None:
                    groups[key].append(p)
            return self._rows(groups)
        if strategy == TOKEN_SORTED:
            for p in self.persons:
                key = self._name(p)
                if name_exclusion_reason(key, cfg.token_sorted_min_tokens, cfg.name_min_chars) is None:
                    groups[token_sorted_key(key)].append(p)
            return self._rows(groups)
        if strategy == PHONE_EXACT:
            for p in self.persons:
                key, _ = phone_key(p["phone_number"], cfg)
                if key:
                    groups[key].append(p)
            return self._rows(groups)
        if strategy in (RELATIVE_EXACT, RELATIVE_SORTED):
            minimum = cfg.relative_min_tokens
            for p in self.persons:
                key = self._relative(p)
                if name_exclusion_reason(key, minimum, cfg.name_min_chars) is None:
                    groups[key if strategy == RELATIVE_EXACT else token_sorted_key(key)].append(p)
            return self._rows(groups)
        if strategy == NAME_INITIAL:
            indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for p in self.persons:
                for key, holds in initial_index_keys(self._name(p), cfg.name_initial_min_anchor_chars):
                    current = indexed[key].get(p["person_id"])
                    if current is None or holds:
                        indexed[key][p["person_id"]] = {**p, "_alias": holds}
            return [row for row in self._rows({k: list(v.values()) for k, v in indexed.items()}, extra=True) if any(row[4])]
        if strategy != ALIAS_NAME:
            raise AssertionError(f"unexpected block strategy {strategy}")
        flagged: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)  # alias strategy: key -> person -> row
        for p in self.persons:
            name, alias = self._name(p), normalize_name(pick_name(p["cleaned_alias"], p["alias"], cfg))
            if name_exclusion_reason(name, cfg.alias_min_tokens, cfg.name_min_chars) is None:
                flagged[name].setdefault(p["person_id"], {**p, "_alias": False})
            if name_exclusion_reason(alias, cfg.alias_min_tokens, cfg.name_min_chars) is None:
                flagged[alias][p["person_id"]] = {**p, "_alias": True}
        return [r for r in self._rows({k: list(v.values()) for k, v in flagged.items()}, extra=True) if any(r[4])]

    # -------------------------------------------------------------- aggregate queries
    def answer(self, tag: str) -> list[tuple]:
        cfg = self.config
        if tag == "preflight_person_ids":
            return [(self.bad_ids,)]
        if tag == "dataset_stamp":
            return [self.stamp]
        if tag.endswith(":exclusions") and tag.startswith(("relative_exact", "relative_token_sorted")):
            reasons = [name_exclusion_reason(self._relative(p), cfg.relative_min_tokens, cfg.name_min_chars) for p in self.persons]
            return [(len(self.persons), reasons.count("blank_name"), reasons.count("too_few_tokens"), reasons.count("too_short"))]
        if tag.endswith(":excluded_blocks") and tag.startswith(("relative_exact", "relative_token_sorted")):
            groups: dict[str, int] = defaultdict(int)
            for p in self.persons:
                key = self._relative(p)
                if key and name_exclusion_reason(key, cfg.relative_min_tokens, cfg.name_min_chars) in ("too_few_tokens", "too_short"):
                    groups[key if tag.startswith("relative_exact") else token_sorted_key(key)] += 1
            shared = [n for n in groups.values() if n > 1]
            return [(len(shared), sum(shared), sum(n * (n - 1) // 2 for n in shared))]
        if tag == "name_initial:exclusions":
            blank = few = no_anchor = 0
            for p in self.persons:
                key = self._name(p)
                if not key:
                    blank += 1
                elif name_exclusion_reason(key, 2, 1) == "too_few_tokens":
                    few += 1
                elif not initial_index_keys(key, cfg.name_initial_min_anchor_chars):
                    no_anchor += 1
            return [(len(self.persons), blank, few, no_anchor)]
        if tag == "name_initial:excluded_blocks":
            return [(0, 0, 0)]
        if tag.endswith(":exclusions") and tag.startswith(("name_exact", "token_sorted_name")):
            min_tokens = cfg.name_min_tokens if tag.startswith("name_exact") else cfg.token_sorted_min_tokens
            reasons = [name_exclusion_reason(self._name(p), min_tokens, cfg.name_min_chars) for p in self.persons]
            return [(len(self.persons), reasons.count("blank_name"), reasons.count("too_few_tokens"), reasons.count("too_short"))]
        if tag.endswith(":excluded_blocks") and tag.startswith(("name_exact", "token_sorted_name")):
            min_tokens = cfg.name_min_tokens if tag.startswith("name_exact") else cfg.token_sorted_min_tokens
            groups = defaultdict(int)
            for p in self.persons:
                key = self._name(p)
                if key and name_exclusion_reason(key, min_tokens, cfg.name_min_chars) in ("too_few_tokens", "too_short"):
                    groups[key] += 1
            shared = [n for n in groups.values() if n > 1]
            return [(len(shared), sum(shared), sum(n * (n - 1) // 2 for n in shared))]
        if tag == "phone_exact:exclusions":
            reasons = defaultdict(int)
            for p in self.persons:
                reasons[phone_key(p["phone_number"], cfg)[1] or "ok"] += 1
            return sorted(reasons.items())
        if tag == "phone_exact:excluded_blocks":
            groups = defaultdict(int)
            for p in self.persons:
                digits = "".join(c for c in (p["phone_number"] or "") if c.isdigit())
                key, reason = phone_key(p["phone_number"], cfg)
                if reason in ("bad_leading_digit", "repetitive_digits", "blocklisted"):
                    groups[(reason, digits[-cfg.phone_digits:])] += 1
            by_reason: dict[str, list[int]] = defaultdict(list)
            for (reason, _), n in groups.items():
                if n > 1:
                    by_reason[reason].append(n)
            return [(r, len(v), sum(v), sum(n * (n - 1) // 2 for n in v)) for r, v in sorted(by_reason.items())]
        if tag == "alias_name:exclusions":
            aliases = [normalize_name(pick_name(p["cleaned_alias"], p["alias"], cfg)) for p in self.persons]
            present = [a for a in aliases if a]
            return [(len(present), sum(1 for a in present if name_exclusion_reason(a, cfg.alias_min_tokens, cfg.name_min_chars)))]
        raise AssertionError(f"unexpected query tag {tag}")


class _Cursor:
    def __init__(self, db: FakePersonDb):
        self.db, self._rows = db, []

    def execute(self, sql: str, params: Any = None):
        check_statement_is_read_only(sql)
        self.db.statements.append(sql)
        if sql.startswith("SHOW transaction_read_only") or sql.startswith("SHOW default_transaction_read_only"):
            self._rows = [(self.db.read_only,)]
        elif "current_database()" in sql:
            self._rows = [(self.db.database, "PostgreSQL 16.14 fake")]
        else:
            self._rows = self.db.answer(TAG.search(sql).group(1))
        return self

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows

    def close(self) -> None:
        pass
