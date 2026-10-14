"""Rebuild the PRE-Phase-9 implementation in a separate directory, for before/after benchmarks and output-equivalence checks.

The project has no version control, so the original code of every file Phase 9 changes is embedded here (verbatim) and written over a copy
of the current package. The result is checked in ``phase9_benchmark.py`` by regenerating the preserved score-v2 run byte for byte.

python tools/phase9_make_baseline.py <target directory>     # creates <target>/dedup_pipeline
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, optimised text, original text); each optimised text must occur exactly once in the current file
REVERTS: list[tuple[str, str, str]] = []


def revert(path: str, new: str, old: str) -> None:
    REVERTS.append((path, new, old))


# ---------------------------------------------------------------- similarity: no memoisation
revert("scoring/similarity.py", "def _ratio(a: str, b: str) -> float:  # not cached: measured hit rate 3%, so it would only cost memory", "def _ratio(a: str, b: str) -> float:")
revert("scoring/similarity.py", "@lru_cache(maxsize=CACHE_SIZE)\ndef token_similarity(", "def token_similarity(")
revert("scoring/similarity.py", "@lru_cache(maxsize=CACHE_SIZE)\ndef name_similarity(", "def name_similarity(")

revert("scoring/similarity.py", "    return _ratio_at_least(x, y, FUZZY_FLOOR)", "    r = _ratio(x, y)\n    return r if r >= FUZZY_FLOOR else 0.0")
revert("scoring/similarity.py", '    compact = _ratio_at_least(a.replace(" ", ""), b.replace(" ", ""), COMPACT_FLOOR)  # below the floor it is never used (0.0 fails the test below)',
       '    compact = _ratio(a.replace(" ", ""), b.replace(" ", ""))')

# ---------------------------------------------------------------- review: html escape without the fast path
revert("scoring/review.py", '''def _e(value: Any) -> str:
    text = "" if value is None else str(value)
    return html.escape(text, quote=True) if _NEEDS_ESCAPE.search(text) else text  # html.escape is the identity when none of & < > " ' occurs''',
       '''def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)''')

# ---------------------------------------------------------------- runner: worksheet and report written by two passes over the scored file
revert("scoring/runner.py", '''        # one pass over the scored records feeds both reviewer outputs (the scored file used to be decoded once per output)
        worksheet = WorksheetWriter(run_dir / WORKSHEET_FILE, config.rule_version)
        try:
            report = ReportWriter(run_dir / REPORT_DIR, {"banner": banner, "headline": headline, "groups": dict(sorted(tally.groups.items())),
                                                          "rule_version": config.rule_version, "reference_date": config.reference_date})
            for rec in _iter_sorted(scored_path, index):
                worksheet.add(rec)
                report.add(rec)
        except BaseException:
            worksheet.abort()
            raise
        worksheet.close()
        report.close()''', '''        write_worksheet(run_dir / WORKSHEET_FILE, _iter_sorted(scored_path, index), config.rule_version)
        write_report(run_dir / REPORT_DIR, _iter_sorted(scored_path, index),
                     {"banner": banner, "headline": headline, "groups": dict(sorted(tally.groups.items())), "rule_version": config.rule_version,
                      "reference_date": config.reference_date})''')

# ---------------------------------------------------------------- inputs: manifest parsed twice
revert("scoring/inputs.py", '''    usable: dict[str, list[dict[str, Any]]] = {}  # kept from this single parse, so the index below does not parse the files again
    person_rows: list[dict[str, Any]] = []
''', "")
revert("scoring/inputs.py", '''            if name == MANIFEST_FILE:
                if record["resolution_status"] in TRANSIENT_RESOLUTION:
                    unresolved_persons.add(record["person_id"])
                if record["usable_for_face_recognition"]:
                    usable.setdefault(record["person_id"], []).append(record)
            else:
                person_rows.append(record)
''', '''            if name == MANIFEST_FILE and record["resolution_status"] in TRANSIENT_RESOLUTION:
                unresolved_persons.add(record["person_id"])
''')
revert("scoring/inputs.py", "ImageIndex.from_parsed(usable, person_rows)", "ImageIndex.from_output_dir(directory)")


def build(target: Path) -> None:
    dest = target / "dedup_pipeline"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / "dedup_pipeline", dest, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, new, old in REVERTS:
        path = dest / rel
        text = path.read_text(encoding="utf-8")
        if text.count(new) != 1:
            raise SystemExit(f"{rel}: expected exactly one occurrence of the optimised text, found {text.count(new)}:\n{new[:120]}")
        path.write_text(text.replace(new, old), encoding="utf-8", newline="")
    # the original ReportWriter/WorksheetWriter classes are harmless extras; the original write_worksheet/write_report bodies are restored below
    review = dest / "scoring" / "review.py"
    text = review.read_text(encoding="utf-8")
    start = text.index("class WorksheetWriter:")
    end = text.index("def _replace(")
    text = text[:start] + ORIGINAL_WRITE_WORKSHEET + "\n\n" + text[end:]
    start = text.index("class ReportWriter:")
    end = text.index("INSTRUCTIONS_V3_ADDENDUM")
    text = text[:start] + ORIGINAL_WRITE_REPORT + "\n\n" + text[end:]
    # runner imports the writer classes; keep the names resolvable
    text += "\n\nclass WorksheetWriter:  # unused in the baseline\n    pass\n\n\nclass ReportWriter:  # unused in the baseline\n    pass\n"
    review.write_text(text, encoding="utf-8", newline="")
    print(f"baseline written to {dest}")


ORIGINAL_WRITE_WORKSHEET = '''def write_worksheet(path: Path, records: Iterable[dict[str, Any]], rule_version: str = "score-v2") -> int:
    """score-v1 runs keep writing the original layout (so that run is reproducible); score-v2 writes the extended one."""
    columns, build = {"score-v1": (WORKSHEET_COLUMNS_V1, worksheet_row), "score-v3": (WORKSHEET_COLUMNS_V3, worksheet_row_v3)}.get(
        rule_version, (WORKSHEET_COLUMNS_V2, worksheet_row_v2))
    tmp = path.with_suffix(".csv.tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads non-Latin names correctly
        writer = csv.writer(fh)
        writer.writerow(columns)
        for order, rec in enumerate(records, 1):
            writer.writerow([safe_cell(v) for v in build(order, rec)])
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)
    return count'''

ORIGINAL_WRITE_REPORT = '''def write_report(directory: Path, records: Iterable[dict[str, Any]], summary_text: dict[str, Any]) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("page_*.html"):
        old.unlink()
    banner = (f"<div class='banner'><b>{_e(summary_text['banner'])}</b><br>Sensitive: contains names and other personal values "
              "for authorised verification only. Every candidate is <b>pending</b>: the software never confirms or rejects a match.</div>")
    batch, order, number = [], 0, 1

    def flush(cards: list[str], n: int, has_next: bool) -> None:
        nav = (f"<nav>{_link('index.html', 'Index')} " + (_link(page_name(n - 1), 'Previous') if n > 1 else "") + " "
               + (_link(page_name(n + 1), "Next") if has_next else "") + "</nav>")
        body = f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Review page {n}</title><style>{CSS}</style><header><b>Candidate review</b> - page {n}</header><main>{banner}{nav}{''.join(cards)}{nav}</main></html>"
        (directory / page_name(n)).write_text(body, encoding="utf-8")

    held: list[str] = []  # a full page is written only once we know whether another page follows it
    previous_group, in_batch = None, 0
    for rec in records:
        if in_batch == PAGE_SIZE:
            if held:
                flush(held, number - 1, True)
            held, batch, number, in_batch = batch, [], number + 1, 0
            previous_group = None  # a new page restates the group at its top
        order += 1
        in_batch += 1
        if rec.get("review_group") != previous_group:
            previous_group = rec.get("review_group")
            batch.append(f"<h2 class='group'>{_e(previous_group)} &mdash; {_e(rec.get('review_group_label'))}</h2>")
        batch.append(card(order, rec))
    if held:
        flush(held, number - 1, True)
    if batch:
        flush(batch, number, False)
    pages = number if batch else 0
    index = "".join(f"<li>{_link(page_name(n), f'Page {n}')}</li>" for n in range(1, pages + 1))
    (directory / "index.html").write_text(
        f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Candidate review</title><style>{CSS}</style>"
        f"<header><b>Candidate review - index</b></header><main>{banner}<h3>{_e(summary_text['headline'])}</h3>"
        f"<p>{_e(SCORE_LABEL)}</p><p>Rule version {_e(summary_text.get('rule_version'))}; ages evaluated at reference date "
        f"<b>{_e(summary_text.get('reference_date') or 'n/a')}</b>.</p>"
        f"<p>{order} candidates in review order. Groups (an ordering aid, never a decision): "
        f"{_e(', '.join(f'{g}: {n}' for g, n in (summary_text.get('groups') or {}).items()))}.</p><ol>{index}</ol></main></html>",
        encoding="utf-8",
    )
    return order'''


if __name__ == "__main__":
    build(Path(sys.argv[1]))
