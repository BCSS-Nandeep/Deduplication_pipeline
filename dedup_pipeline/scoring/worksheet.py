"""Check a completed review worksheet and keep a local copy of the explicit decisions.

A row is a decision only if a human filled ``reviewer_decision`` with an allowed value *and* ``reviewer_id`` *and* a
parseable ``reviewed_at``. Blank rows stay pending. Nothing is written to any database; the output is a local JSONL.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import DECISIONS
from .review import DECISION_COLUMNS, WORKSHEET_COLUMNS_V1, WORKSHEET_COLUMNS_V2, WORKSHEET_COLUMNS_V3

KNOWN_LAYOUTS = (WORKSHEET_COLUMNS_V3, WORKSHEET_COLUMNS_V2, WORKSHEET_COLUMNS_V1)  # worksheets from earlier runs (score-v1) stay valid

VALIDATED_FILE = "decisions_validated.jsonl"


class WorksheetError(ValueError):
    pass


def _parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_worksheet(path: Path, known_candidates: Optional[set[str]] = None) -> dict[str, Any]:
    """Return ``{"decisions": [...], "pending": n, "problems": [...]}``; ``problems`` is empty for a clean worksheet."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) not in KNOWN_LAYOUTS:
            raise WorksheetError("worksheet columns do not match the generated layout; do not edit, add or reorder columns")
        decisions, problems, pending, seen = [], [], 0, set()
        for line, row in enumerate(reader, 2):
            cid = row["candidate_id"]
            if cid in seen:
                problems.append(f"line {line}: duplicate candidate {cid}")
                continue
            seen.add(cid)
            if known_candidates is not None and cid not in known_candidates:
                problems.append(f"line {line}: unknown candidate {cid}")
                continue
            values = {c: (row[c] or "").strip() for c in DECISION_COLUMNS}
            if not any(values.values()):
                pending += 1
                continue
            issues = []
            if values["reviewer_decision"] not in DECISIONS:
                issues.append(f"reviewer_decision must be one of {', '.join(DECISIONS)}")
            if not values["reviewer_id"]:
                issues.append("reviewer_id is required")
            if not values["reviewed_at"] or _parse_time(values["reviewed_at"]) is None:
                issues.append("reviewed_at must be an ISO date/time")
            if issues:
                problems.append(f"line {line} ({cid}): " + "; ".join(issues))
                continue
            decisions.append({"candidate_id": cid, "person_a_id": row["person_a_id"], "person_b_id": row["person_b_id"],
                              "decision": values["reviewer_decision"], "reviewer_id": values["reviewer_id"],
                              "reviewed_at": values["reviewed_at"], "notes": values["reviewer_notes"] or None})
    return {"decisions": decisions, "pending": pending, "problems": problems, "rows": len(seen)}


def write_validated(path: Path, result: dict[str, Any]) -> Path:
    out = path.parent / VALIDATED_FILE
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for decision in result["decisions"]:
            fh.write(json.dumps(decision, ensure_ascii=True) + "\n")
    return out
