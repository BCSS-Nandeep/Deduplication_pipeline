"""score-v3 age-evidence provenance: for every person in the scored candidates, which recorded age was anchored to which FIR
(by id and date), how each observation was classified, and what needs a human. Local files only; no names are written."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..candidates.store import replace_with_retry, write_json_atomic
from .age_v3 import AGE_SOURCE, LINK_SOURCE, AgeEvidence
from .attributes import PersonAttributes

PROVENANCE_FILE = "age_evidence_provenance.jsonl"
PROVENANCE_SUMMARY_FILE = "age_evidence_summary.json"


class AgeProvenance:
    def __init__(self, run_dir: Path):
        self._final = run_dir / PROVENANCE_FILE
        self._tmp = run_dir / (PROVENANCE_FILE + ".tmp")
        self._fh = open(self._tmp, "wb")
        self._seen: set[str] = set()
        self.persons = Counter()
        self.by_source: dict[str, Counter] = {}
        self.observation_status = Counter()
        self.unavailable_reasons = Counter()
        self.fir_date_status = Counter()
        self.fir_year = Counter()
        self.flags = Counter()
        self.quality = 0

    def record(self, attrs: PersonAttributes) -> None:
        if attrs.person_id in self._seen:
            return
        self._seen.add(attrs.person_id)
        e = attrs.age_info
        if not isinstance(e, AgeEvidence):
            return
        line = {
            "person_id": attrs.person_id, "source_system": attrs.source_system, "source_record_id": attrs.source_record_id,
            "age_source": AGE_SOURCE, "fir_link_source": LINK_SOURCE,
            "dob_status": e.dob_status, "dob_present": e.dob is not None, "recorded_ages": e.recorded_ages(),
            "observation_status": e.observation_status, "age_evidence_state": e.side_state(),
            "merged_birth_year_range": list(e.merged_range) if e.merged_range else None,
            "observations": [o.to_dict(include_values=True) for o in e.observations],
            "own_dob_vs_age_at_fir_date_inconsistencies": list(e.quality), "review_flags": [f["issue"] for f in e.flags],
        }
        self._fh.write((json.dumps(line, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii"))
        self.persons["persons"] += 1
        src = self.by_source.setdefault(attrs.source_system, Counter())
        src["persons"] += 1
        src[f"state:{e.side_state()}"] += 1
        self.observation_status[e.observation_status] += 1
        for o in e.observations:
            self.fir_date_status[o.fir_date_status] += 1
            if o.unavailable_reason:
                self.unavailable_reasons[o.unavailable_reason] += 1
            if o.fir_date:
                self.fir_year[o.fir_date.year] += 1
        for f in e.flags:
            self.flags[f["issue"]] += 1
        self.quality += len(e.quality)
        if len(e.observations) > 1:
            self.persons["with_multiple_fir_links"] += 1

    def close(self, config: Any) -> dict[str, Any]:
        self._fh.flush()
        self._fh.close()
        replace_with_retry(self._tmp, self._final)
        summary = {
            "rule_version": config.rule_version, "reference_date": config.reference_date,
            "reference_date_use": "validation cut-off only (a FIR dated after it is invalid); never used as the date an age was recorded",
            "age_source": AGE_SOURCE, "fir_link_source": LINK_SOURCE, "fir_timezone": config.fir_timezone,
            "persons": self.persons["persons"], "persons_with_multiple_fir_links": self.persons["with_multiple_fir_links"],
            "observation_status_per_person": dict(sorted(self.observation_status.items())),
            "age_evidence_state_by_source": {s: dict(sorted(c.items())) for s, c in sorted(self.by_source.items())},
            "fir_date_status_per_observation": dict(sorted(self.fir_date_status.items())),
            "age_unavailable_reasons_per_observation": dict(sorted(self.unavailable_reasons.items())),
            "review_flags": dict(sorted(self.flags.items())),
            "own_dob_vs_age_at_fir_date_inconsistencies": self.quality,
            "fir_year_distribution": {str(y): n for y, n in sorted(self.fir_year.items())},
            "file": PROVENANCE_FILE,
        }
        return summary


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    write_json_atomic(run_dir / PROVENANCE_SUMMARY_FILE, summary)
