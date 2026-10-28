"""Audit score-final photo-available vs photo-unavailable scoring on local fixture rows.

Production is contacted once, read-only, only to count media for the already-extracted
person ids. All scoring and result storage use local ``dopams_dedup_test``.

Face cosine for the photo-available scenario is a controlled test input. It is not a
real buffalo_l comparison and must not be treated as a production face match.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

from dedup_pipeline.candidates.blocks import expand_block, parse_block_row
from dedup_pipeline.candidates.config import CandidateConfig, final_candidate_config
from dedup_pipeline.candidates.sql import block_query
from dedup_pipeline.db import connect_read_only, load_connection_settings
from dedup_pipeline.face_recognition.pair_evidence import PairFaceEvidence, PairStatus
from dedup_pipeline.scoring.attributes import Address, PersonAttributes, build_attributes
from dedup_pipeline.scoring.config import FINAL_WEIGHTS, RULE_VERSION, ScoringConfig
from dedup_pipeline.scoring.face import face_field_score
from dedup_pipeline.scoring.score import score_pair

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "tests" / "fixtures" / "candidate_parity_extract.json"
PRODUCTION_HOST = "192.168.103.106"
FINAL = ScoringConfig(rule_version="score-final")


def _load_local_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if key.startswith("DEDUP_LOCAL_PG_") or key == "DEDUP_SQL_PARITY":
            os.environ[key] = value.strip().strip("\"'")


def _local_connect(writable: bool = False):
    host = os.environ.get("DEDUP_LOCAL_PG_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(f"refusing non-loopback host {host!r}")
    conn = psycopg2.connect(
        host=host,
        port=int(os.environ.get("DEDUP_LOCAL_PG_PORT", "5432")),
        dbname=os.environ["DEDUP_LOCAL_PG_DATABASE"],
        user=os.environ["DEDUP_LOCAL_PG_USER"],
        password=os.environ["DEDUP_LOCAL_PG_PASSWORD"],
        connect_timeout=10,
        application_name="dedup_score_final_photo_audit",
    )
    if not writable:
        conn.set_session(readonly=True, autocommit=True)
    return conn


def cosine_for(mapped: float) -> float:
    return 0.20 + mapped * (0.58 - 0.20)


def _face(status: PairStatus, similarity=None, **kw) -> PairFaceEvidence:
    return PairFaceEvidence(
        "a", "b", status, face_similarity=similarity,
        similarity_metric="cosine" if similarity is not None else None,
        model="buffalo_l", **kw,
    )


def probe_production_media(person_ids: list[str]) -> dict:
    """Bounded read-only counts only. No candidate SQL and no scoring on production."""
    settings = load_connection_settings(str(ROOT / ".env"))
    if settings.host != PRODUCTION_HOST or settings.database != "dopams_cctns_v2":
        raise SystemExit(f"refusing media probe: {settings.host}/{settings.database}")
    sql = (
        "SELECT count(*) AS media_rows, "
        "count(DISTINCT parent_entity_id) AS persons_with_media "
        "FROM public.media_unified "
        "WHERE parent_entity_type = %s AND parent_entity_id = ANY(%s)"
    )
    with connect_read_only(settings, "dedup_media_count_probe") as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT current_setting('default_transaction_read_only'), current_database(), "
            "COALESCE(inet_server_addr()::text, 'local')"
        )
        read_only, database, address = cur.fetchone()
        if read_only != "on" or not str(address).startswith(PRODUCTION_HOST):
            raise SystemExit(f"refusing media probe: read_only={read_only} address={address}")
        print("PRODUCTION_MEDIA_SQL:")
        print(sql)
        print("params: parent_entity_type='person', person_ids=EXTRACT_80")
        cur.execute(sql, ("person", person_ids))
        media_rows, persons_with_media = cur.fetchone()
        cur.close()
    return {
        "read_only": read_only,
        "database": database,
        "address": address,
        "person_ids_probed": len(person_ids),
        "media_rows": int(media_rows),
        "persons_with_media": int(persons_with_media),
        "limitation": (
            "Local dopams_dedup_test has no media_unified table and no resolved images. "
            "Any photo-available score below uses a controlled cosine, not a real buffalo_l match."
        ),
    }


def local_people(conn) -> dict[str, dict]:
    sql = (
        "SELECT p.person_id, p.source_system, p.source_record_id, p.full_name, p.alias, "
        "p.relative_name, p.phone_number, e.cleaned_full_name, e.cleaned_relative_name "
        "FROM public.persons_unified p "
        "LEFT JOIN public.person_enrichment e ON e.person_id = p.person_id "
        "ORDER BY p.person_id"
    )
    print("LOCAL_PEOPLE_SQL:")
    print(sql)
    cur = conn.cursor()
    cur.execute(sql)
    keys = (
        "person_id", "source_system", "source_record_id", "full_name", "alias",
        "relative_name", "phone_number", "cleaned_full_name", "cleaned_relative_name",
    )
    rows = [dict(zip(keys, row)) for row in cur.fetchall()]
    cur.close()
    print(f"LOCAL_PEOPLE_RESULT: rows={len(rows)}")
    return {row["person_id"]: row for row in rows}


def local_candidate_pairs(conn) -> list[tuple[str, str, list[str]]]:
    config = final_candidate_config()
    found = {}
    for strategy in config.enabled_strategies():
        spec = block_query(strategy, config)
        print(f"LOCAL_CANDIDATE_SQL[{strategy}]:")
        print(spec.sql)
        cur = conn.cursor()
        cur.execute(spec.sql, spec.params)
        rows = cur.fetchall()
        print(f"LOCAL_CANDIDATE_RESULT[{strategy}]: blocks={len(rows)}")
        cap = config.max_block_size(strategy)
        for ordinal, row in enumerate(rows):
            block = parse_block_row(strategy, ordinal, row)
            if block.size > cap:
                continue
            for left, right in expand_block(block, config.source_policy):
                pair = (left.person_id, right.person_id)
                found.setdefault(pair, set()).add(strategy)
        cur.close()
    return [(a, b, sorted(rules)) for (a, b), rules in sorted(found.items())]


def person_attrs(row: dict) -> PersonAttributes:
    """Build score-final attributes from the local fixture columns.

    Address/gender are not in the local minimum schema. Controlled present-address
    and Male gender are attached so the full 25/25/20/15/10/5 arithmetic can be
    verified; they are labeled in the stored result as controlled_fill.
    """
    name = row.get("cleaned_full_name") or row.get("full_name")
    relative = row.get("cleaned_relative_name") or row.get("relative_name")
    db_row = (
        row["person_id"], row["source_system"], row["source_record_id"],
        name, row.get("alias"), relative, row.get("phone_number"),
        None, None, "Male",
        {"present": {
            "state": "Telangana", "district": "Hyderabad", "mandal": "Shaikpet",
            "village": "Shaikpet", "pin": "500008", "country": "India",
        }},
        "shaikpet hyderabad telangana 500008", None,
    )
    return build_attributes(db_row, FINAL, CandidateConfig())


def ensure_result_table(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS public.score_final_photo_audit (
            audit_id bigserial PRIMARY KEY,
            run_at timestamptz NOT NULL,
            scenario text NOT NULL,
            person_a text NOT NULL,
            person_b text NOT NULL,
            candidate_rules jsonb NOT NULL,
            face_input_kind text NOT NULL,
            raw_cosine double precision,
            mapped_face_score double precision,
            face_applicable boolean NOT NULL,
            face_score double precision,
            face_reason text,
            match_score double precision,
            evidence_coverage double precision,
            applicable_fields integer NOT NULL,
            field_breakdown jsonb NOT NULL,
            weights jsonb NOT NULL,
            notes text NOT NULL
        )
        """
    )
    cur.execute("DELETE FROM public.score_final_photo_audit")
    conn.commit()
    cur.close()
    print("LOCAL_DDL: CREATE TABLE IF NOT EXISTS public.score_final_photo_audit (...)")
    print("LOCAL_SQL: DELETE FROM public.score_final_photo_audit")


def store_result(conn, payload: dict) -> None:
    sql = (
        "INSERT INTO public.score_final_photo_audit ("
        "run_at, scenario, person_a, person_b, candidate_rules, face_input_kind, "
        "raw_cosine, mapped_face_score, face_applicable, face_score, face_reason, "
        "match_score, evidence_coverage, applicable_fields, field_breakdown, weights, notes"
        ") VALUES ("
        "%(run_at)s, %(scenario)s, %(person_a)s, %(person_b)s, %(candidate_rules)s, %(face_input_kind)s, "
        "%(raw_cosine)s, %(mapped_face_score)s, %(face_applicable)s, %(face_score)s, %(face_reason)s, "
        "%(match_score)s, %(evidence_coverage)s, %(applicable_fields)s, %(field_breakdown)s, %(weights)s, %(notes)s"
        ")"
    )
    print("LOCAL_INSERT_SQL:")
    print(sql)
    cur = conn.cursor()
    cur.execute(sql, {
        **payload,
        "candidate_rules": json.dumps(payload["candidate_rules"]),
        "field_breakdown": json.dumps(payload["field_breakdown"]),
        "weights": json.dumps(payload["weights"]),
    })
    conn.commit()
    cur.close()


def breakdown(outcome) -> dict:
    out = {}
    for name, field in outcome.fields.items():
        out[name] = {
            "weight": field.weight,
            "applicable": field.applicable,
            "score": field.score,
            "reason_not_applicable": field.reason_not_applicable,
            "detail_status": (field.detail or {}).get("status"),
            "threshold_outcome": (field.detail or {}).get("threshold_outcome"),
            "mapped_score": (field.detail or {}).get("mapped_score"),
        }
    return out


def score_and_store(conn, scenario: str, left: PersonAttributes, right: PersonAttributes,
                    rules: list[str], evidence: PairFaceEvidence | None, face_kind: str, notes: str) -> dict:
    outcome = score_pair(left, right, evidence, FINAL)
    face_field = outcome.fields["face"]
    mapped = None
    raw = None
    if evidence is not None and evidence.face_similarity is not None and face_field.applicable:
        raw = float(evidence.face_similarity)
        mapped = round(face_field_score(raw, FINAL), 4)
    elif evidence is not None and evidence.face_similarity is not None:
        raw = float(evidence.face_similarity)
        mapped = round(face_field_score(raw, FINAL), 4)
    payload = {
        "run_at": datetime.now(timezone.utc),
        "scenario": scenario,
        "person_a": left.person_id,
        "person_b": right.person_id,
        "candidate_rules": rules,
        "face_input_kind": face_kind,
        "raw_cosine": raw,
        "mapped_face_score": mapped,
        "face_applicable": face_field.applicable,
        "face_score": face_field.score,
        "face_reason": face_field.reason_not_applicable,
        "match_score": outcome.match_score,
        "evidence_coverage": outcome.coverage,
        "applicable_fields": outcome.applicable_fields,
        "field_breakdown": breakdown(outcome),
        "weights": {k: v for k, v in FINAL_WEIGHTS.items() if v},
        "notes": notes,
    }
    store_result(conn, payload)
    return {"outcome": outcome, "payload": payload}


def choose_pair(people: dict[str, dict], pairs: list[tuple[str, str, list[str]]]) -> tuple[str, str, list[str]]:
    # Prefer synthetic syn01/syn02: designed same-name / same-phone multi-accused case.
    for a, b, rules in pairs:
        if {a, b} == {"syn01", "syn02"}:
            return a, b, rules
    for a, b, rules in pairs:
        if a in people and b in people and "name_exact" in rules:
            return a, b, rules
    raise SystemExit("no suitable local candidate pair")


def print_scenario(title: str, result: dict) -> None:
    payload = result["payload"]
    outcome = result["outcome"]
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"pair={payload['person_a']} <-> {payload['person_b']}")
    print(f"candidate_rules={payload['candidate_rules']}")
    print(f"face_input_kind={payload['face_input_kind']}")
    print(f"raw_cosine={payload['raw_cosine']} mapped_face_score={payload['mapped_face_score']}")
    print(
        f"face_applicable={payload['face_applicable']} face_score={payload['face_score']} "
        f"face_reason={payload['face_reason']}"
    )
    print(
        f"match_score={None if outcome.match_score is None else round(outcome.match_score, 6)} "
        f"match_score_percent={None if outcome.match_score is None else round(outcome.match_score * 100, 1)} "
        f"evidence_coverage={round(outcome.coverage, 6)} "
        f"evidence_coverage_percent={round(outcome.coverage * 100, 1)} "
        f"applicable_fields={outcome.applicable_fields}"
    )
    print("field_breakdown:")
    for name, item in payload["field_breakdown"].items():
        if item["weight"] == 0 and name in {"alias", "age_dob"}:
            continue
        print(
            f"  {name}: weight={item['weight']} applicable={item['applicable']} "
            f"score={item['score']} reason={item['reason_not_applicable']} "
            f"status={item['detail_status']} threshold={item['threshold_outcome']}"
        )
    print(f"notes={payload['notes']}")


def main() -> None:
    _load_local_env()
    extract = json.loads(EXTRACT.read_text(encoding="utf-8"))
    extract_ids = [row["person_id"] for row in extract["records"]]

    print("ACTIVE_WEIGHTS", {k: v for k, v in FINAL_WEIGHTS.items() if v})
    print("FACE_PASS_THRESHOLD", FINAL.face_pass_threshold)
    print("DEFAULT_RULE_VERSION_UNCHANGED", RULE_VERSION)

    media = probe_production_media(extract_ids)
    print("PRODUCTION_MEDIA_RESULT:", json.dumps(media, indent=2))

    read_conn = _local_connect(writable=False)
    people = local_people(read_conn)
    pairs = local_candidate_pairs(read_conn)
    print(f"LOCAL_PAIR_COUNT: {len(pairs)}")
    a_id, b_id, rules = choose_pair(people, pairs)
    read_conn.close()

    left = person_attrs(people[a_id])
    right = person_attrs(people[b_id])

    write_conn = _local_connect(writable=True)
    ensure_result_table(write_conn)

    # Scenario 1: both have usable photos -> controlled comparison above 0.85 mapped.
    mapped_pass = 0.92
    cosine = cosine_for(mapped_pass)
    assert round(face_field_score(cosine, FINAL), 4) == mapped_pass
    assert mapped_pass > FINAL.face_pass_threshold
    available = score_and_store(
        write_conn,
        "photo_available",
        left,
        right,
        rules,
        _face(PairStatus.COMPARED, cosine),
        "controlled_test_cosine_not_real_engine",
        (
            "Controlled input: raw cosine chosen so mapped score is 0.92 (> 0.85). "
            "Not a real buffalo_l comparison. Local DB has no images; production media "
            f"probe found media_rows={media['media_rows']} for the 80 extract ids."
        ),
    )
    print_scenario("SCENARIO 1: PHOTO AVAILABLE (controlled face comparison)", available)

    # Scenario 2: one/both lack usable photos -> face weight excluded, remaining renormalized.
    unavailable = score_and_store(
        write_conn,
        "photo_unavailable",
        left,
        right,
        rules,
        _face(PairStatus.SKIPPED_NO_IMAGE, reason="no usable image on one or both accused"),
        "skipped_no_image",
        (
            "Face weight 0.25 excluded from numerator and denominator. "
            "Remaining active weights renormalize over 0.75 total coverage."
        ),
    )
    print_scenario("SCENARIO 2: PHOTO UNAVAILABLE (face weight excluded)", unavailable)

    # Boundary check stored as notes only via a third controlled row: mapped exactly 0.85 fails and keeps weight.
    boundary = score_and_store(
        write_conn,
        "photo_available_boundary_0_85",
        left,
        right,
        rules,
        _face(PairStatus.COMPARED, cosine_for(0.85)),
        "controlled_test_cosine_boundary",
        "Mapped face score exactly 0.85 fails the strict >0.85 rule; face stays applicable with score 0.",
    )
    print_scenario("BOUNDARY: mapped face == 0.85 (fails, keeps 25% weight)", boundary)

    cur = write_conn.cursor()
    select_sql = (
        "SELECT scenario, person_a, person_b, face_input_kind, raw_cosine, mapped_face_score, "
        "face_applicable, face_score, match_score, evidence_coverage, applicable_fields "
        "FROM public.score_final_photo_audit ORDER BY audit_id"
    )
    print()
    print("LOCAL_SELECT_SQL:")
    print(select_sql)
    cur.execute(select_sql)
    rows = cur.fetchall()
    print("LOCAL_SELECT_RESULT:")
    for row in rows:
        print(row)
    cur.close()
    write_conn.close()

    # Arithmetic assertions for the audit itself.
    full = available["outcome"]
    bare = unavailable["outcome"]
    edge = boundary["outcome"]
    assert full.fields["face"].applicable and full.fields["face"].score == pytest_approx(0.92)
    assert full.coverage == pytest_approx(1.0)
    assert bare.fields["face"].applicable is False and bare.fields["face"].score is None
    assert bare.coverage == pytest_approx(0.75)
    assert edge.fields["face"].applicable and edge.fields["face"].score == 0.0
    assert edge.coverage == pytest_approx(1.0)
    print()
    print("AUDIT_ASSERTIONS: passed")


def pytest_approx(value: float, rel: float = 1e-9):
    class Approx:
        def __eq__(self, other):
            return other is not None and abs(float(other) - value) <= rel
    return Approx()


if __name__ == "__main__":
    main()
