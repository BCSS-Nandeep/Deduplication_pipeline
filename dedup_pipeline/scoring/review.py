"""Reviewer-facing outputs: CSV worksheet (decision columns empty), paginated read-only HTML report, instructions.

Nothing here records or implies a decision. The HTML report is a read-only view; decisions are entered by an
authorised human in the worksheet's empty ``reviewer_decision`` / ``reviewer_id`` / ``reviewed_at`` / ``reviewer_notes``
columns and checked by ``worksheet.py``. Records that contain personal values are marked sensitive in every file.
"""

from __future__ import annotations

import csv
import html
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DECISIONS
from .score import SCORE_LABEL

WORKSHEET_FILE = "review_worksheet.csv"
REPORT_DIR = "review_report"
INSTRUCTIONS_FILE = "REVIEW_INSTRUCTIONS.md"
PAGE_SIZE = 100
DECISION_COLUMNS = ("reviewer_decision", "reviewer_id", "reviewed_at", "reviewer_notes")
WORKSHEET_COLUMNS_V2 = (
    "review_order", "review_group", "rule_version", "candidate_id", "person_a_id", "person_b_id", "source_a", "source_record_a", "source_b", "source_record_b",
    "match_score_percent_NOT_A_PROBABILITY", "evidence_coverage_percent", "score_band", "insufficient_evidence", "has_hard_conflict", "hard_conflicts",
    "dob_comparison_status", "age_a", "age_b", "reference_date", "birth_year_range_a", "birth_year_range_b", "age_tolerance_years",
    "age_comparison_method", "age_dob_inconsistency",
    "face_status", "face_raw_cosine", "face_model", "face_fallback_used", "face_in_match_score", "image_selection_rationale", "face_attempts",
    "field_scores", "score_explanation", "matching_attributes", "conflicting_attributes", "missing_evidence",
    "image_links_a", "image_links_b", "same_image_file_on_both", "candidate_rules", "processing_flags", "review_status",
    "reviewer_decision", "reviewer_id", "reviewed_at", "reviewer_notes",
)
# score-v3 adds the FIR anchoring of each age (the v2 columns stay where they are, the new ones go right after the age columns)
WORKSHEET_COLUMNS_V3 = (*WORKSHEET_COLUMNS_V2[:WORKSHEET_COLUMNS_V2.index("age_dob_inconsistency") + 1],
                        "fir_year_a", "fir_year_b", "age_evidence_a", "age_evidence_b", "age_review_flags",
                        *WORKSHEET_COLUMNS_V2[WORKSHEET_COLUMNS_V2.index("age_dob_inconsistency") + 1:])
WORKSHEET_COLUMNS = WORKSHEET_COLUMNS_V2  # the layout written for score-v2; worksheet.py also still accepts the score-v1 layout below
WORKSHEET_COLUMNS_V1 = (
    "review_order", "candidate_id", "person_a_id", "person_b_id", "source_a", "source_record_a", "source_b", "source_record_b",
    "match_score_percent_NOT_A_PROBABILITY", "evidence_coverage_percent", "score_band", "insufficient_evidence", "has_hard_conflict",
    "face_status", "face_raw_cosine", "face_model", "face_fallback_used", "face_in_match_score",
    "field_scores", "score_explanation", "matching_attributes", "conflicting_attributes", "missing_evidence",
    "image_links_a", "image_links_b", "same_image_file_on_both", "candidate_rules", "processing_flags", "review_status",
    *DECISION_COLUMNS,
)
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula injection: text starting with = + - @ would be executed by Excel."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def _join(items: Iterable[str]) -> str:
    return "; ".join(items)


def _selection_text(rec: dict[str, Any]) -> str:
    sel = rec.get("image_selection") or {}
    if not sel.get("applicable"):
        return ""
    parts = []
    for side in ("a", "b"):
        for n in sel[side]:
            result = n["service_result"] or ("not attempted" if n["within_attempt_limit"] else "beyond attempt limit")
            parts.append(f"{side.upper()}#{n['position']} {n['media_id']} [{n['category']}] {result}: {n['reason']}")
    return " | ".join(parts)


def _attempts_text(rec: dict[str, Any]) -> str:
    out = []
    for att in rec["face"].get("attempts") or []:
        detail = att["outcome"] + (f" {att['rejections']}" if att.get("rejections") else (f" {att['error_kind']}" if att.get("error_kind") else ""))
        out.append(f"{att.get('media_id_a')}+{att.get('media_id_b')}: {detail}")
    return " | ".join(out)


def worksheet_row_v2(order: int, rec: dict[str, Any]) -> list[Any]:
    base = worksheet_row(order, rec)  # the v1 cell values, re-used by name below
    columns = dict(zip(WORKSHEET_COLUMNS_V1, base))
    age = rec.get("age_dob_summary") or {}
    rng = lambda side: "-".join(str(y) for y in age.get(f"birth_year_range_{side}") or [])
    inconsistency = "; ".join(f"record {i['record'].upper()}: recorded age {i['recorded_age']} vs age implied by its DOB {i['age_implied_by_dob']}"
                              for i in age.get("inconsistencies", []))
    extra = {
        "review_group": rec.get("review_group"), "rule_version": rec.get("rule_version"),
        "hard_conflicts": _join(f"{c['field']}: {c['explanation']}" for c in rec.get("hard_conflicts", [])),
        "dob_comparison_status": age.get("dob_comparison_status"), "age_a": age.get("recorded_age_a"), "age_b": age.get("recorded_age_b"),
        "reference_date": age.get("reference_date"), "birth_year_range_a": rng("a"), "birth_year_range_b": rng("b"),
        "age_tolerance_years": age.get("age_tolerance_years"),
        "age_comparison_method": f"{age.get('method')} / {age.get('comparison_status')}" if age.get("method") else (age.get("comparison_status") or ""),
        "age_dob_inconsistency": inconsistency, "image_selection_rationale": _selection_text(rec), "face_attempts": _attempts_text(rec),
        "conflicting_attributes": _join(f"{c['field']} ({c['severity']})" for c in rec["conflicting_attributes"]),
        "has_hard_conflict": rec.get("has_hard_conflict"),
    }
    columns.update(extra)
    return [columns.get(name, "") for name in WORKSHEET_COLUMNS_V2]


def _observation_text(side: dict[str, Any], values: dict[str, Any]) -> str:
    """One cell: how this record's age was dated. FIR date and id come from ``values`` (sensitive, shown only if values are included)."""
    obs = values.get("observations") if values else None
    if obs:
        return " | ".join((f"age {o.get('recorded_age')} at FIR {o.get('fir_date')} ({o.get('crime_id')}) -> birth years "
                           f"{'-'.join(str(y) for y in o['birth_year_range'])}") if o["usable"] else
                          f"age {o.get('recorded_age')}: unavailable ({o['unavailable_reason']})" for o in obs)
    return " | ".join((f"FIR year {o['fir_year']}" if o["usable"] else f"unavailable ({o['unavailable_reason']})") for o in side["observations"])


def worksheet_row_v3(order: int, rec: dict[str, Any]) -> list[Any]:
    columns = dict(zip(WORKSHEET_COLUMNS_V2, worksheet_row_v2(order, rec)))
    age = rec.get("age_dob_summary") or {}
    years = lambda side: "/".join(str(y) for y in sorted({o["fir_year"] for o in age.get(f"record_{side}", {}).get("observations", []) if o["fir_year"]}))
    values = (rec["fields"]["age_dob"].get("values") or {})
    inconsistency = "; ".join(f"record {i['record'].upper()}: recorded age {i['recorded_age']} vs age implied by its DOB {i['age_implied_by_dob']} at its FIR date"
                              for i in age.get("inconsistencies", []))
    columns.update({
        "age_dob_inconsistency": inconsistency, "fir_year_a": years("a"), "fir_year_b": years("b"),
        "age_evidence_a": _observation_text(age["record_a"], values.get("a", {})) if age else "",
        "age_evidence_b": _observation_text(age["record_b"], values.get("b", {})) if age else "",
        "age_review_flags": "; ".join(f"record {f['record'].upper()}: {f['issue']}" for f in age.get("review_flags", [])),
    })
    return [columns.get(name, "") for name in WORKSHEET_COLUMNS_V3]


def worksheet_row(order: int, rec: dict[str, Any]) -> list[Any]:
    """The score-v1 cell values, by v1 column (``worksheet_row_v2`` builds on this)."""
    fields, face = rec["fields"], rec["face"]
    field_scores = _join(f"{n}={'n/a' if not f['applicable'] else f['score']}" for n, f in fields.items())
    explanation = " | ".join(f"{n}: {f['explanation']}" for n, f in fields.items())
    links = lambda side: _join(f"{i['file_proxy_link']}" for i in rec["images"][side])
    return [
        order, rec["candidate_id"], rec["person_a_id"], rec["person_b_id"], rec["source_a"], rec["source_record_a"],
        rec["source_b"], rec["source_record_b"], rec["match_score_percent"], rec["evidence_coverage_percent"],
        rec["score_band"], rec["insufficient_evidence"], rec["has_hard_conflict"], face["status"], face["raw_cosine"], face.get("model"),
        face.get("fallback_used"), face.get("in_match_score"), field_scores, explanation,
        _join(m["field"] for m in rec["matching_attributes"]),
        _join(f"{c['field']} ({c['severity']})" for c in rec["conflicting_attributes"]),
        _join(f"{m['field']}: {m['reason']}" for m in rec["missing_evidence"]),
        links("a"), links("b"), rec["images"]["shared_image_file"],
        _join(rec["candidate_generation"]["rule_names"]), _join(rec["processing"]["indicators"]), rec["review"]["status"],
        "", "", "", "",
    ]


class WorksheetWriter:
    """Incremental worksheet writer: ``add`` each record in review order, then ``close``. Lets the run write the worksheet and the
    HTML report in ONE pass over the scored records (Phase 9) instead of decoding the scored file once per output."""

    def __init__(self, path: Path, rule_version: str = "score-v2"):
        self._columns, self._build = {"score-v1": (WORKSHEET_COLUMNS_V1, worksheet_row), "score-v3": (WORKSHEET_COLUMNS_V3, worksheet_row_v3)}.get(
            rule_version, (WORKSHEET_COLUMNS_V2, worksheet_row_v2))
        self._path = path
        self._tmp = path.with_suffix(".csv.tmp")
        self.count = 0
        self._fh = open(self._tmp, "w", encoding="utf-8-sig", newline="")  # BOM so Excel reads non-Latin names correctly
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self._columns)

    def add(self, rec: dict[str, Any]) -> None:
        self.count += 1
        self._writer.writerow([safe_cell(v) for v in self._build(self.count, rec)])

    def close(self) -> int:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        _replace(self._tmp, self._path)
        return self.count

    def abort(self) -> None:
        self._fh.close()


def write_worksheet(path: Path, records: Iterable[dict[str, Any]], rule_version: str = "score-v2") -> int:
    """score-v1 runs keep writing the original layout (so that run is reproducible); score-v2 writes the extended one."""
    writer = WorksheetWriter(path, rule_version)
    try:
        for rec in records:
            writer.add(rec)
    except BaseException:
        writer.abort()
        raise
    return writer.close()


def _replace(tmp: Path, path: Path) -> None:
    from ..candidates.store import replace_with_retry

    replace_with_retry(tmp, path)


# ------------------------------------------------------------------------------- HTML

CSS = """body{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;background:#f4f5f7;color:#1c2330}
header{background:#1c2330;color:#fff;padding:12px 20px}header a{color:#9cc4ff}main{max-width:1100px;margin:0 auto;padding:16px}
.banner{background:#fff4d6;border:1px solid #e3c25a;padding:10px 14px;border-radius:6px;margin:12px 0}
.card{background:#fff;border:1px solid #d6dae1;border-radius:8px;margin:14px 0;padding:14px 16px}
.head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.score{font-size:20px;font-weight:700}
.sub{color:#5a6475;font-size:12px}table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border-top:1px solid #e6e9ee;padding:4px 8px;text-align:left;vertical-align:top;font-size:13px}
.tag{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;margin-right:4px;background:#e8ecf3}.hard{background:#fbd5d5}.soft{background:#ffe8c2}
.ok{background:#d9f0dc}.pending{background:#e8ecf3}.decision{border:2px dashed #9aa3b2;padding:8px;border-radius:6px;margin-top:8px}
nav{margin:10px 0}code{background:#eef0f4;padding:0 4px;border-radius:3px}
.hardbox{background:#fde3e3;border:2px solid #d14b4b;border-radius:6px;padding:8px 12px;margin:8px 0}.hardbox ul{margin:4px 0 0 18px;padding:0}
.qualitybox{background:#fff0d6;border:1px solid #d9a441;border-radius:6px;padding:6px 10px;margin:8px 0}.data_quality{background:#ffe0a8}
.group{background:#1c2330;color:#fff;padding:8px 14px;border-radius:6px;margin:22px 0 4px}"""


_NEEDS_ESCAPE = re.compile(r"[&<>\"']")


def _e(value: Any) -> str:
    text = "" if value is None else str(value)
    return html.escape(text, quote=True) if _NEEDS_ESCAPE.search(text) else text  # html.escape is the identity when none of & < > " ' occurs


_RELATIVE_PAGE = re.compile(r"[A-Za-z0-9_.-]+\.html")


def _link(url: str, text: str) -> str:
    """Only site-relative, http(s) or plain sibling-page links are emitted; anything else (javascript:, data:) becomes '#'."""
    safe = url if (url.startswith(("/", "http://", "https://")) or _RELATIVE_PAGE.fullmatch(url)) else "#"
    return f'<a href="{_e(safe)}" rel="noopener noreferrer" target="_blank">{_e(text)}</a>'


def card(order: int, rec: dict[str, Any]) -> str:
    face = rec["face"]
    rows = []
    for name, f in rec["fields"].items():
        score = "n/a" if not f["applicable"] else f"{f['score']:.2f}"
        shown = f.get("values")
        detail = f" <span class='sub'>{_e(shown)}</span>" if shown else ""
        conflict = f" <span class='tag {_e(f.get('conflict'))}'>{_e(f.get('conflict'))} conflict</span>" if f.get("conflict") else ""
        rows.append(f"<tr><td>{_e(name)}</td><td>{score}</td><td>{f['weight']:.2f}</td><td>{_e(f['explanation'])}{detail}{conflict}</td></tr>")
    tags = "".join(f"<span class='tag ok'>{_e(m['field'])}</span>" for m in rec["matching_attributes"]) or "<span class='sub'>none</span>"
    conflicts = "".join(f"<span class='tag {_e(c['severity'])}'>{_e(c['field'])} ({_e(c['severity'])})</span>" for c in rec["conflicting_attributes"]) or "<span class='sub'>none</span>"
    missing = "".join(f"<span class='tag'>{_e(m['field'])}: {_e(m['reason'])}</span>" for m in rec["missing_evidence"]) or "<span class='sub'>none</span>"
    imgs = lambda side: " ".join(_link(i["file_proxy_link"], f"image {i['person_rank']}{' (used)' if i['used_in_face_comparison'] else ''}") for i in rec["images"][side]) or "<span class='sub'>no usable image</span>"
    face_text = f"status <b>{_e(face['status'])}</b>"
    if face.get("raw_cosine") is not None:
        face_text += f"; raw cosine <b>{face['raw_cosine']:.3f}</b> (range -1 to 1; model {_e(face.get('model'))})"
        if face.get("relative_to_service_threshold"):
            face_text += f"; {_e(face['relative_to_service_threshold'])} the service threshold {_e(face.get('service_threshold'))} (not a decision)"
    if face.get("fallback_used"):
        face_text += "; <b>fallback to another image was used</b>"
    if face.get("reason") and face["status"] not in ("compared", "matched", "not_matched"):
        face_text += f"; {_e(face['reason'])}"
    flags = "".join(f"<span class='tag soft'>{_e(i)}</span>" for i in rec["processing"]["indicators"])
    score = "n/a" if rec["match_score_percent"] is None else f"{rec['match_score_percent']:.1f}"
    hard = rec.get("hard_conflicts") or []
    hard_box = ("<div class='hardbox'><b>HARD CONFLICT - read before relying on the score:</b><ul>"
                + "".join(f"<li><b>{_e(c['field'])}</b>: {_e(c['explanation'])}</li>" for c in hard) + "</ul></div>") if hard else ""
    quality_box = "".join(f"<div class='qualitybox'><b>Data quality:</b> {_e(c['explanation'])}</div>"
                          for c in rec["conflicting_attributes"] if c["severity"] == "data_quality")
    age = rec.get("age_dob_summary")
    age_text = ""
    if age:
        shown = lambda s: f"{s}: recorded age {_e(age.get(f'recorded_age_{s}'))}, DOB {_e(age.get(f'dob_{s}'))}, birth years {_e(age.get(f'birth_year_range_{s}'))}" \
            if f"recorded_age_{s}" in age else f"{s}: (values withheld)"
        if age.get("rule_version") == "score-v3":
            obs = lambda s: _e(_observation_text(age[f"record_{s}"], (rec["fields"]["age_dob"].get("values") or {}).get(s, {})))
            flags = "".join(f"<span class='tag data_quality'>{_e(f['issue'])} (record {_e(f['record'].upper())})</span>" for f in age.get("review_flags", []))
            age_text = (f"<p><b>Age evidence anchored to FIR dates</b> (score-v3; the reference date {_e(age['reference_date'])} is only a validity cut-off): "
                        f"method <b>{_e(age['method'] or 'none')}</b>, status <b>{_e(age['comparison_status'])}</b>; DOB comparison: {_e(age['dob_comparison_status'])}.<br>"
                        f"<span class='sub'>A: {obs('a')}<br>B: {obs('b')}<br>An age is a statement made on the FIR date; possible birth years are estimates, "
                        f"never dates of birth. An age with no usable FIR date is not compared. {flags}</span></p>")
        else:
            age_text = (f"<p><b>Age / DOB</b> (reference date <b>{_e(age['reference_date'])}</b>, tolerance {_e(age['age_tolerance_years'])} year(s)): "
                    f"method <b>{_e(age['method'] or 'none')}</b>, status <b>{_e(age['comparison_status'])}</b>; DOB comparison: {_e(age['dob_comparison_status'])}.<br>"
                    f"<span class='sub'>{shown('a')} &nbsp;|&nbsp; {shown('b')}. Birth years derived from an age are estimates, not dates of birth.</span></p>")
    sel = rec.get("image_selection") or {}
    sel_text = ""
    if sel.get("applicable"):
        lines = []
        for side in ("a", "b"):
            for n in sel[side]:
                result = n["service_result"] or ("not attempted" if n["within_attempt_limit"] else "beyond attempt limit")
                lines.append(f"{side.upper()}#{n['position']} <code>{_e(n['media_id'])}</code> {_e(n['category'])} &rarr; {_e(result)} <span class='sub'>({_e(n['reason'])})</span>")
        sel_text = f"<p class='sub'><b>Image selection</b> (policy {_e(sel['policy'])}, up to {_e(sel['attempt_limit_per_person'])} images per person):<br>{'<br>'.join(lines)}</p>"
    return f"""<section class="card" id="{_e(rec['candidate_id'])}">
<div class="head"><div><b>#{order}</b> <code>{_e(rec['candidate_id'])}</code><div class="sub">A: {_e(rec['source_a'])} / {_e(rec['source_record_a'])} &nbsp;|&nbsp; B: {_e(rec['source_b'])} / {_e(rec['source_record_b'])}</div></div>
<div><span class="score">Match score {score} / 100</span><div class="sub">coverage {rec['evidence_coverage_percent']}% &middot; band {_e(rec['score_band'])} &middot; {_e(rec['score_band_note'])}</div></div></div>
<div class="sub">{_e(SCORE_LABEL)}</div>
{hard_box}{quality_box}
<table><tr><th>Field</th><th>Score</th><th>Weight</th><th>Why</th></tr>{''.join(rows)}</table>
{age_text}
<p><b>Matching:</b> {tags}<br><b>Conflicting:</b> {conflicts}<br><b>Missing evidence:</b> {missing}</p>
<p><b>Face:</b> {face_text}</p>{sel_text}
<p><b>Images A:</b> {imgs('a')}<br><b>Images B:</b> {imgs('b')}{'<br><b>Note:</b> the same image file is attached to both records.' if rec['images']['shared_image_file'] else ''}</p>
<p class="sub">Found by: {_e(', '.join(rec['candidate_generation']['rule_names']))}. {flags}</p>
<div class="decision"><b>Reviewer decision:</b> <span class="tag pending">pending</span> &mdash; to be entered by an authorised human in the worksheet (<code>{_e(' / '.join(DECISIONS))}</code>).</div>
</section>"""


def page_name(number: int) -> str:
    return f"page_{number:04d}.html"


class ReportWriter:
    """Incremental HTML report writer (same pages, same bytes as the one-shot ``write_report``): ``add`` records in review order, then ``close``."""

    def __init__(self, directory: Path, summary_text: dict[str, Any]):
        self._directory, self._summary = directory, summary_text
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("page_*.html"):
            old.unlink()
        self._banner = (f"<div class='banner'><b>{_e(summary_text['banner'])}</b><br>Sensitive: contains names and other personal values "
                        "for authorised verification only. Every candidate is <b>pending</b>: the software never confirms or rejects a match.</div>")
        self._batch: list[str] = []
        self.order, self._number = 0, 1
        self._held: list[str] = []  # a full page is written only once we know whether another page follows it
        self._previous_group, self._in_batch = None, 0

    def _flush(self, cards: list[str], n: int, has_next: bool) -> None:
        nav = (f"<nav>{_link('index.html', 'Index')} " + (_link(page_name(n - 1), 'Previous') if n > 1 else "") + " "
               + (_link(page_name(n + 1), "Next") if has_next else "") + "</nav>")
        body = f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Review page {n}</title><style>{CSS}</style><header><b>Candidate review</b> - page {n}</header><main>{self._banner}{nav}{''.join(cards)}{nav}</main></html>"
        (self._directory / page_name(n)).write_text(body, encoding="utf-8")

    def add(self, rec: dict[str, Any]) -> None:
        if self._in_batch == PAGE_SIZE:
            if self._held:
                self._flush(self._held, self._number - 1, True)
            self._held, self._batch, self._number, self._in_batch = self._batch, [], self._number + 1, 0
            self._previous_group = None  # a new page restates the group at its top
        self.order += 1
        self._in_batch += 1
        if rec.get("review_group") != self._previous_group:
            self._previous_group = rec.get("review_group")
            self._batch.append(f"<h2 class='group'>{_e(self._previous_group)} &mdash; {_e(rec.get('review_group_label'))}</h2>")
        self._batch.append(card(self.order, rec))

    def close(self) -> int:
        summary_text, banner, order, number = self._summary, self._banner, self.order, self._number
        if self._held:
            self._flush(self._held, number - 1, True)
        if self._batch:
            self._flush(self._batch, number, False)
        pages = number if self._batch else 0
        index = "".join(f"<li>{_link(page_name(n), f'Page {n}')}</li>" for n in range(1, pages + 1))
        (self._directory / "index.html").write_text(
            f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Candidate review</title><style>{CSS}</style>"
            f"<header><b>Candidate review - index</b></header><main>{banner}<h3>{_e(summary_text['headline'])}</h3>"
            f"<p>{_e(SCORE_LABEL)}</p><p>Rule version {_e(summary_text.get('rule_version'))}; ages evaluated at reference date "
            f"<b>{_e(summary_text.get('reference_date') or 'n/a')}</b>.</p>"
            f"<p>{order} candidates in review order. Groups (an ordering aid, never a decision): "
            f"{_e(', '.join(f'{g}: {n}' for g, n in (summary_text.get('groups') or {}).items()))}.</p><ol>{index}</ol></main></html>",
            encoding="utf-8",
        )
        return order


def write_report(directory: Path, records: Iterable[dict[str, Any]], summary_text: dict[str, Any]) -> int:
    writer = ReportWriter(directory, summary_text)
    for rec in records:
        writer.add(rec)
    return writer.close()


INSTRUCTIONS_V3_ADDENDUM = """
## Age evidence in score-v3 (replaces the "Age and date of birth" section above for this run)
A recorded age is a statement made when the record was created, so score-v3 anchors it to the date of the **FIR the person is linked to**
(not to today). The possible birth years follow from that date: age 26 at a 2022 FIR and age 30 at a 2026 FIR are consistent (both allow
birth year 1996), because the birthday is unknown. These are estimates, never dates of birth, and never an exact DOB match.
* A recorded age with **no usable FIR date** (no FIR link, missing/invalid date) is shown as unavailable evidence. It is not compared and
  the current date is never substituted. A date of birth is still compared as before.
* If one record has several FIRs, every observation is listed with its FIR; none is preferred. If they contradict each other the card
  carries a data-quality flag: look at the record itself.
* Whether recorded ages in the older (V1) source are as of the FIR date is only weakly supported by the data; treat V1 age evidence with care.
"""

INSTRUCTIONS = f"""# Reviewing duplicate-person candidates

**Every candidate is pending until an authorised human decides.** The software never confirms, rejects or merges anything,
and no score threshold triggers any action. The match score ({'0-100'}) is a prioritisation aid built from weighted
evidence; it is **not** a probability that two records are the same person and has not been calibrated against labelled data.

## Review order (score-v2)
Candidates are grouped, and within each group listed by match score, highest first. Nothing is removed and no group is a decision:
1. `1_no_hard_conflict` - no hard conflict; 2. `2_hard_conflict` - a hard conflict is shown in red at the top of the card, because a high
weighted score can hide one strong contradiction; 3. `3_insufficient_evidence`; 4. `4_technical_failure` - a retry or investigation is needed first.

## Age and date of birth
All ages are evaluated at one **reference date** stated in the report. Birth years derived from a recorded age are estimates (an age of 30 at
a 2026 reference date means born in 1995 or 1996), so equal ages never count as an exact DOB match. A record whose own age and DOB disagree is
flagged as a data-quality issue; its DOB was used.

## What to look at
* The score *and* why: each field's score, weight and explanation, the matching and conflicting attributes, and what is missing.
* Evidence coverage: how much of the possible evidence was available. A high score with low coverage is weak.
* Face evidence: raw cosine similarity (-1 to 1), not a probability. Open the image links in your authorised DOPAMS session.
* Name-based candidate rules guarantee a similar name by construction; do not treat name agreement as independent evidence.

## Recording a decision (worksheet `{WORKSHEET_FILE}`)
Fill all of `reviewer_decision`, `reviewer_id`, `reviewed_at` (ISO date/time) for each candidate you decide. Allowed `reviewer_decision` values:

| value | meaning |
|---|---|
| `confirm_same_person` | I reviewed the evidence; both records refer to the same individual |
| `reject_match` | I reviewed the evidence; the records refer to different individuals |
| `needs_further_review` | Evidence is insufficient or conflicting |
| `unable_to_verify` | Technical or data-quality limits prevent a decision |

Leave a row blank if you have not reviewed it. Check a completed worksheet with
`python -m dedup_pipeline.scoring validate-worksheet <file.csv>`; it writes a local `decisions_validated.jsonl` and
changes nothing else. Decisions are **not** written to any database and no records are merged.
"""
