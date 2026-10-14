"""Local files for one candidate-generation run: state, checkpoint, per-strategy work files, final outputs.

A run is complete if and only if ``run_summary.json`` exists with ``status == "completed"``. It is deleted when
a run starts and written last, so a crashed, interrupted or still-running run can never be mistaken for a
finished one. ``run_state.json`` says which of running / failed / interrupted / completed it is.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

CANDIDATES_FILE = "candidates.jsonl"
DEFERRED_FILE = "deferred_blocks.jsonl"
SUMMARY_FILE = "run_summary.json"
STATE_FILE = "run_state.json"
CHECKPOINT_FILE = "checkpoint.json"
WORK_DIR = "work"
FINAL_FILES = (CANDIDATES_FILE, DEFERRED_FILE, SUMMARY_FILE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def replace_with_retry(tmp: Path, path: Path, attempts: int = 12) -> None:
    """``os.replace`` that survives Windows briefly locking a file (antivirus, indexer): PermissionError is retried."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.02 * 2 ** attempt, 0.5))


def write_json_atomic(path: Path, data: Any, indent: Optional[int] = 2) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=True, indent=indent, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    replace_with_retry(tmp, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
    replace_with_retry(tmp, path)
    return count


class RunStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.work = directory / WORK_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ state
    def read_state(self) -> Optional[dict[str, Any]]:
        return self._read(self.dir / STATE_FILE)

    def write_state(self, status: str, run_id: str, **extra: Any) -> None:
        write_json_atomic(self.dir / STATE_FILE, {"status": status, "run_id": run_id, "updated_at": now_iso(), **extra})

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    def clear_finals(self) -> None:
        """Remove the previous run's results, completion marker first, so nothing stale can look current."""
        for name in (SUMMARY_FILE, CANDIDATES_FILE, DEFERRED_FILE):
            (self.dir / name).unlink(missing_ok=True)
            (self.dir / (name + ".tmp")).unlink(missing_ok=True)

    # ------------------------------------------------------------- checkpoint
    def load_checkpoint(self) -> Optional[dict[str, Any]]:
        return self._read(self.work / CHECKPOINT_FILE)

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        write_json_atomic(self.work / CHECKPOINT_FILE, checkpoint, indent=None)

    def reset_work(self) -> None:
        for path in self.work.glob("*"):
            path.unlink()

    def pairs_path(self, strategy: str) -> Path:
        return self.work / f"{strategy}.pairs.jsonl"

    def truncate_pairs(self, strategy: str, keep_lines: int) -> None:
        """Make the strategy's pair file hold exactly the first ``keep_lines`` complete lines."""
        path = self.pairs_path(strategy)
        lines = path.read_text(encoding="utf-8").splitlines()[:keep_lines] if path.exists() else []
        if len(lines) < keep_lines:
            raise RuntimeError(f"work file for {strategy} has fewer than the {keep_lines} checkpointed lines")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(line + "\n" for line in lines)

    @staticmethod
    def _read(path: Path) -> Optional[dict[str, Any]]:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None
