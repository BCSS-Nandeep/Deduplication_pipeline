"""Person attributes needed for scoring, read from the unified database (read-only) in bounded chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional, Sequence

from ..candidates.config import CandidateConfig
from ..candidates.normalize import normalize_name, phone_key
from ..db import ReadOnlyConnection
from .config import ScoringConfig

ATTRIBUTE_QUERY = (
    "/* scoring:attributes */ SELECT p.person_id, p.source_system, p.source_record_id, "
    "COALESCE(NULLIF(btrim(e.cleaned_full_name), ''), p.full_name), "
    "COALESCE(NULLIF(btrim(e.cleaned_alias), ''), p.alias), "
    "COALESCE(NULLIF(btrim(e.cleaned_relative_name), ''), p.relative_name), "
    "p.phone_number, p.age, p.date_of_birth, p.gender, e.address_resolution "
    "FROM public.persons_unified p LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id "
    "WHERE p.person_id = ANY(%(ids)s) ORDER BY p.person_id"
)
# score-v3: every accused row of the person and the FIR it points to (read-only). The timestamp is turned into the calendar date it has
# in the configured zone; a LEFT JOIN keeps an accused row whose FIR row is missing, so the gap is reported rather than hidden.
FIR_LINK_QUERY = (
    "/* scoring:fir_links */ SELECT a.person_id, a.accused_id, a.crime_id, (c.fir_date AT TIME ZONE %(tz)s)::date "
    "FROM public.accused_unified a LEFT JOIN public.crimes_unified c ON c.crime_id = a.crime_id "
    "WHERE a.person_id = ANY(%(ids)s) ORDER BY a.person_id, a.accused_id"
)
CHUNK = 2000
DEFINITE_GENDERS = {"male": "Male", "female": "Female"}  # 'Others' / 'Unknown' / 'Transgender' are not compared


@dataclass(frozen=True)
class Address:
    state: str = ""
    district: str = ""
    mandal: str = ""

    def usable(self) -> bool:
        return bool(self.state or self.district)


@dataclass(frozen=True)
class PersonAttributes:
    person_id: str
    source_system: str
    source_record_id: str
    name: str = ""  # normalised comparison keys
    alias: str = ""
    relative_name: str = ""
    phone: Optional[str] = None  # last-10 key of a plausible number, else None
    phone_reason: Optional[str] = None  # why there is no usable phone
    age: Optional[int] = None  # only if within the meaningful range
    age_reason: Optional[str] = None
    dob: Optional[date] = None
    gender: Optional[str] = None  # "Male" / "Female" only
    addresses: tuple[Address, ...] = field(default_factory=tuple)
    age_info: Optional[Any] = None  # score-v2: age.AgeInfo evaluated at the run's reference date


def _clean(value: Any) -> str:
    return " ".join(str(value).lower().split()) if value is not None else ""


def _addresses(raw: Any) -> tuple[Address, ...]:
    out = []
    if isinstance(raw, dict):
        for kind in ("present", "permanent"):
            part = raw.get(kind)
            if isinstance(part, dict):
                address = Address(_clean(part.get("state")), _clean(part.get("district")), _clean(part.get("mandal")))
                if address.usable():
                    out.append(address)
    return tuple(out)


def build_attributes(row: Sequence[Any], scoring: ScoringConfig, candidate_config: CandidateConfig,
                     fir_rows: Optional[Sequence[tuple[Any, Any, Any]]] = None) -> PersonAttributes:
    from .age import build_age_info  # imported here: age.py needs fields.py, which needs this module

    pid, source, record, name, alias, relative, phone, age, dob, gender, address = row
    key, reason = phone_key(phone, candidate_config)  # the same plausibility rules as candidate generation
    meaningful = age is not None and scoring.age_min <= age <= scoring.age_max
    if scoring.rule_version == "score-v3":
        from .age_v3 import build_age_evidence

        age_info: Any = build_age_evidence(dob, age, fir_rows, scoring)
    else:
        age_info = build_age_info(dob, age, scoring)
    return PersonAttributes(
        person_id=pid, source_system=source, source_record_id=record,
        name=normalize_name(name), alias=normalize_name(alias), relative_name=normalize_name(relative),
        phone=key, phone_reason=reason,
        age=int(age) if meaningful else None,
        age_reason=None if meaningful else ("no_age" if age is None else "age_out_of_range"),
        dob=dob, gender=DEFINITE_GENDERS.get(_clean(gender)), addresses=_addresses(address),
        age_info=age_info,
    )


class AttributeStore:
    """Fetch-on-miss cache over the database. Bounded: it is emptied when it grows past ``attribute_cache_max``."""

    def __init__(self, conn: ReadOnlyConnection, scoring: ScoringConfig, candidate_config: CandidateConfig,
                 on_person: Optional[Callable[[PersonAttributes], None]] = None):
        self._conn, self._scoring, self._candidate_config = conn, scoring, candidate_config
        self._cache: dict[str, PersonAttributes] = {}
        self._on_person = on_person  # called once per person as it is read (the score-v3 provenance writer)
        self.queries = 0
        self.rows_read = 0
        self.fir_rows_read = 0

    def prefetch(self, person_ids: Sequence[str]) -> None:
        missing = sorted({p for p in person_ids if p not in self._cache})
        if len(self._cache) + len(missing) > self._scoring.attribute_cache_max:
            self._cache.clear()
            missing = sorted(set(person_ids))
        cur = self._conn.cursor()
        for start in range(0, len(missing), CHUNK):
            chunk = missing[start:start + CHUNK]
            cur.execute(ATTRIBUTE_QUERY, {"ids": chunk})
            self.queries += 1
            rows = cur.fetchall()
            fir: dict[str, list[tuple[Any, Any, Any]]] = {}
            if self._scoring.rule_version == "score-v3":
                cur.execute(FIR_LINK_QUERY, {"ids": chunk, "tz": self._scoring.fir_timezone})
                self.queries += 1
                for person_id, accused_id, crime_id, fir_date in cur.fetchall():
                    fir.setdefault(person_id, []).append((accused_id, crime_id, fir_date))
                    self.fir_rows_read += 1
            for row in rows:
                self._cache[row[0]] = build_attributes(row, self._scoring, self._candidate_config, fir.get(row[0]))
                self.rows_read += 1
                if self._on_person is not None:
                    self._on_person(self._cache[row[0]])
        cur.close()

    def get(self, person_id: str) -> Optional[PersonAttributes]:
        return self._cache.get(person_id)
