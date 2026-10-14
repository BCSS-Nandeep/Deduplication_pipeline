"""The only SQL this module runs. Every statement is a read-only SELECT."""

from __future__ import annotations

from typing import Optional

from ..db import ReadOnlyConnection
from .models import MediaRow, Person

PERSON_PARENT_TYPE = "person"  # media_unified.parent_entity_type for person-owned media
_MEDIA_CHUNK = 2000

_HAS_PERSON_MEDIA = (
    "EXISTS (SELECT 1 FROM public.media_unified m "
    "WHERE m.parent_entity_type = %s AND m.parent_entity_id = p.person_id)"
)


def load_persons(
    conn: ReadOnlyConnection,
    source_system: Optional[str] = None,
    limit: Optional[int] = None,
    only_with_media: bool = False,
) -> list[Person]:
    clauses: list[str] = []
    params: list = []
    if source_system:
        clauses.append("p.source_system = %s")
        params.append(source_system)
    if only_with_media:
        clauses.append(_HAS_PERSON_MEDIA)
        params.append(PERSON_PARENT_TYPE)
    sql = "SELECT p.person_id, p.source_system FROM public.persons_unified p"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY p.person_id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur = conn.cursor()
    cur.execute(sql, params)
    persons = [Person(person_id=r[0], source_system=r[1]) for r in cur.fetchall()]
    cur.close()
    return persons


def load_person_media(conn: ReadOnlyConnection, person_ids: list[str]) -> list[MediaRow]:
    """Media joined on ``parent_entity_id = person_id`` with ``parent_entity_type = 'person'``."""
    rows: list[MediaRow] = []
    cur = conn.cursor()
    for start in range(0, len(person_ids), _MEDIA_CHUNK):
        cur.execute(
            "SELECT media_id, parent_entity_id, attachment_category, availability_status, "
            "resolved_relative_path, source_local_path, original_attach_path "
            "FROM public.media_unified "
            "WHERE parent_entity_type = %s AND parent_entity_id = ANY(%s) "
            "ORDER BY media_id",
            (PERSON_PARENT_TYPE, person_ids[start:start + _MEDIA_CHUNK]),
        )
        rows.extend(MediaRow(*r) for r in cur.fetchall())
    cur.close()
    return rows
