"""Checks the saved fixture files. This does not execute SQL and is not a parity result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_extract_is_bounded_and_contains_no_secrets():
    extract = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_extract.json").read_text(encoding="utf-8"))
    assert extract["source"]["host"] == "192.168.103.106"
    assert extract["source"]["database"] == "dopams_cctns_v2"
    assert extract["source"]["read_only"] == "on"
    assert extract["extract"]["row_count"] == 80
    assert len(extract["records"]) == 80
    assert all(row["origin"] == "production" for row in extract["records"])
    blob = json.dumps(extract).lower()
    assert "password" not in blob and "secret" not in blob
    assert extract["source"]["server_version"].startswith("16.14")


def test_synthetic_rows_are_marked_and_cover_the_extract_gaps():
    synthetic = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_synthetic.json").read_text(encoding="utf-8"))
    assert synthetic["origin"] == "synthetic"
    assert all(row["origin"] == "synthetic" for row in synthetic["records"])
    ids = {row["person_id"] for row in synthetic["records"]}
    assert {f"n{index:02d}" for index in range(20)} <= ids
    assert "syn01" in ids and "syn10" in ids
    extract = json.loads((ROOT / "tests" / "fixtures" / "candidate_parity_extract.json").read_text(encoding="utf-8"))
    assert ids.isdisjoint({row["person_id"] for row in extract["records"]})
