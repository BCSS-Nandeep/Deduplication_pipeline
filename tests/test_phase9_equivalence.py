"""Phase 9: the optimisations must not change any result. Each optimised component is compared with a verbatim copy of the original."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from dedup_pipeline.scoring import similarity as new

# ---------------------------------------------------------------- the ORIGINAL similarity implementation (pre-Phase 9), verbatim
INITIAL_MATCH, ABBREVIATION_MATCH, FUZZY_FLOOR, COMPACT_FLOOR = 0.9, 0.85, 0.75, 0.90


def old_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def old_is_subsequence(short, long):
    remaining = iter(long)
    return all(ch in remaining for ch in short)


def old_token_similarity(x, y):
    if x == y:
        return 1.0
    short, long = (x, y) if len(x) <= len(y) else (y, x)
    if len(short) == 1 and long.startswith(short):
        return INITIAL_MATCH
    if len(short) >= 2 and short[0] == long[0] and old_is_subsequence(short, long):
        return ABBREVIATION_MATCH
    r = old_ratio(x, y)
    return r if r >= FUZZY_FLOOR else 0.0


def old_alignment(ta, tb):
    if not ta or not tb:
        return 0.0
    pairs = sorted(((old_token_similarity(x, y), i, j) for i, x in enumerate(ta) for j, y in enumerate(tb)), key=lambda t: (-t[0], t[1], t[2]))
    used_a, used_b, total = set(), set(), 0.0
    for sim, i, j in pairs:
        if sim <= 0.0:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        total += sim
    return total / ((len(ta) + len(tb)) / 2)


def old_name_similarity(a, b):
    if not a or not b:
        return None
    if a == b:
        return 1.0, "identical"
    aligned = min(1.0, old_alignment(a.split(), b.split()))
    compact = old_ratio(a.replace(" ", ""), b.replace(" ", ""))
    if compact >= COMPACT_FLOOR and compact > aligned:
        return round(compact, 4), "spacing variant"
    return round(aligned, 4), "token alignment"


# ------------------------------------------------------------------------------------------------ data
BASE = ["ramesh kumar", "suresh babu", "mohammed ali", "md ali", "sai kumar", "saikumar", "venkata ramana reddy", "v ramana reddy", "lakshmi narayana",
        "anil kumar", "bala reddy", "chandu rao", "dinesh babu", "eswar naidu", "k srinivas", "kothapalli srinivas rao", "shaik abdul rahman", "s a rahman",
        "రమేష్ కుమార్", "సురేష్ బాబు", "रमेश कुमार", "सुरेश बाबू", "a", "b c", "mohd rafi", "mohammed rafi", "nagaraju", "nagaraj", "yellaiah", "yellaiha"]


def mutate(rng: random.Random, name: str) -> str:
    chars = list(name)
    for _ in range(rng.choice((0, 0, 1, 1, 2, 3))):
        if not chars:
            break
        op = rng.choice(("drop", "swap", "dup", "space", "unspace", "replace"))
        i = rng.randrange(len(chars))
        if op == "drop":
            del chars[i]
        elif op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "dup":
            chars.insert(i, chars[i])
        elif op == "space":
            chars.insert(i, " ")
        elif op == "unspace":
            text = "".join(chars).replace(" ", "", 1)
            chars = list(text)
        else:
            chars[i] = rng.choice("aeioustnrk")
    text = " ".join("".join(chars).split())
    if rng.random() < 0.15:
        parts = text.split()
        rng.shuffle(parts)
        text = " ".join(parts)
    return text


def test_optimised_similarity_returns_exactly_what_the_original_returned():
    rng = random.Random(9)
    checked = 0
    for _ in range(30000):
        a, b = mutate(rng, rng.choice(BASE)), mutate(rng, rng.choice(BASE))
        assert new.name_similarity(a, b) == old_name_similarity(a, b), (a, b)
        checked += 1
    for _ in range(20000):  # random short tokens: the ratio floor logic is what changed
        x = "".join(rng.choice("abcdeimnrstu") for _ in range(rng.randint(1, 9)))
        y = "".join(rng.choice("abcdeimnrstu") for _ in range(rng.randint(1, 9)))
        assert new.token_similarity(x, y) == old_token_similarity(x, y), (x, y)
        checked += 1
    assert checked == 50000


def test_the_ratio_upper_bounds_the_optimisation_relies_on_really_are_upper_bounds():
    rng = random.Random(10)
    for _ in range(20000):
        a = "".join(rng.choice("abcdefgh ") for _ in range(rng.randint(0, 14)))
        b = "".join(rng.choice("abcdefgh ") for _ in range(rng.randint(0, 14)))
        m = SequenceMatcher(None, a, b)
        r = m.ratio()
        assert m.real_quick_ratio() >= r and m.quick_ratio() >= r
        for floor in (0.75, 0.90):
            expected = r if r >= floor else 0.0
            assert new._ratio_at_least(a, b, floor) == expected


def test_the_caches_are_bounded_and_keyed_by_exact_arguments():
    assert new.name_similarity.cache_info().maxsize == new.CACHE_SIZE == new.token_similarity.cache_info().maxsize
    assert new.name_similarity("ramesh kumar", "ramesh kumar") == (1.0, "identical")
    assert new.name_similarity("ramesh kumar", "kumar ramesh") == old_name_similarity("ramesh kumar", "kumar ramesh")
    assert new.name_similarity("", "x") is None and new.name_similarity("x", "") is None  # blank stays None, and None is cached like any value


# ---------------------------------------------------------------- html escaping fast path
def test_the_html_escape_fast_path_is_the_identity_for_unescapable_text():
    import html

    from dedup_pipeline.scoring.review import _e

    rng = random.Random(11)
    alphabet = "abc xyz 0123 &<>\"' రమ र ;#-_/"
    for _ in range(20000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20)))
        assert _e(value) == html.escape(value, quote=True)
    assert _e(None) == "" and _e(5) == "5" and _e(1.5) == "1.5" and _e(True) == "True"


# ---------------------------------------------------------------- single-pass reviewer outputs
def test_one_pass_writers_produce_the_same_bytes_as_the_one_shot_writers(tmp_path):
    from dedup_pipeline.scoring.review import ReportWriter, WorksheetWriter, write_report, write_worksheet

    rec_dir = Path(__file__).resolve().parents[1] / "output" / "audit" / "phase8_1"
    source = Path(__file__).resolve().parents[1] / "output" / "scoring_v3"
    runs = sorted((source / "runs").glob("*/scored_candidates.jsonl")) if source.exists() else []
    if not runs:
        pytest.skip("no scored candidates available")
    records = []
    with open(runs[0], encoding="ascii") as fh:
        for line in fh:
            records.append(json.loads(line))
            if len(records) >= 260:  # more than two report pages
                break
    for version in ("score-v3", "score-v2", "score-v1"):
        a, b = tmp_path / f"a_{version}", tmp_path / f"b_{version}"
        a.mkdir()
        b.mkdir()
        text = {"banner": "b", "headline": "h", "groups": {"g": 1}, "rule_version": version, "reference_date": "2026-10-12"}
        assert write_worksheet(a / "w.csv", iter(records), version) == len(records)
        assert write_report(a / "report", iter(records), text) == len(records)
        ws, rp = WorksheetWriter(b / "w.csv", version), ReportWriter(b / "report", text)
        for rec in records:
            ws.add(rec)
            rp.add(rec)
        assert ws.close() == len(records) and rp.close() == len(records)
        for p in sorted(a.rglob("*")):
            if p.is_file():
                assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256((b / p.relative_to(a)).read_bytes()).hexdigest(), p.name


def test_a_full_run_equals_the_original_two_pass_method_on_the_same_scored_records(tmp_path):
    """End to end on a small world: the run's worksheet and report (written in one pass) are byte-identical to what the original method -
    the one-shot writers fed from the scored file once per output - produces from the same records; ids, scores, groups and decisions are
    the records themselves."""
    from dedup_pipeline.scoring import ScoringOptions, run_scoring
    from dedup_pipeline.scoring.review import write_report, write_worksheet

    from .scoring_fakes import FakeComparer, V2CONFIG, backend, build_world, outcome, pdict

    people = [pdict(f"p{i}", name=n, relative="Suresh Kumar", age=a, phone=ph, gender=g)
              for i, (n, a, ph, g) in enumerate([("Ramesh Kumar", 30, "9876543211", "Male"), ("Ramesh Kumar", 31, "9876543211", "Male"), ("Ramesh Kumr", 45, None, "Male"),
                                                  ("Anil Reddy", 22, None, "Male"), ("Anil Reddy", 22, None, "Female"), ("S Rao", None, None, None), ("Sai Rao", None, None, None),
                                                  ("Anil Reddy", 25, None, "Male"), ("Anil Reddy", 24, None, "Male"), ("Ramesh Kumar", 61, "9876543211", "Male")])]
    world = build_world(tmp_path / "w", people, {"p0": [("m0", 1)], "p1": [("m1", 2)]})
    summary = run_scoring(world.db(), V2CONFIG(), ScoringOptions(world.candidate_dir, world.image_dir, tmp_path / "root", "run", False),
                          backend(world, FakeComparer(world, lambda a, b: outcome(0.7)), None))
    run = tmp_path / "root" / "runs" / "run"
    by_id = {r["candidate_id"]: r for r in map(json.loads, (run / "scored_candidates.jsonl").read_text(encoding="ascii").splitlines())}
    with open(run / "review_worksheet.csv", encoding="utf-8-sig", newline="") as fh:
        order = [row["candidate_id"] for row in csv.DictReader(fh)]
    assert set(order) == set(by_id) and len(order) == summary["candidates"]["scored"] > 3
    ordered = [by_id[c] for c in order]
    text = {"banner": ("" if summary["complete"] else "PROVISIONAL: face comparison is incomplete for some candidates. ") +
            "Match score is NOT a probability of identity. All candidates are pending human review.",
            "headline": f"{len(ordered)} candidates - run COMPLETE", "groups": dict(sorted(summary["review_groups"]["candidates_per_group"].items())),
            "rule_version": "score-v2", "reference_date": "2026-10-12"}
    reference = tmp_path / "reference"
    reference.mkdir()
    write_worksheet(reference / "review_worksheet.csv", iter(ordered), "score-v2")
    write_report(reference / "review_report", iter(ordered), text)
    assert sha(run / "review_worksheet.csv") == sha(reference / "review_worksheet.csv")
    for page in sorted((reference / "review_report").glob("*.html")):
        assert sha(run / "review_report" / page.name) == sha(page), page.name
    assert all(r["review"]["status"] == "pending" and r["review"]["decision"] is None for r in ordered)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_image_index_built_from_one_parse_equals_the_one_built_from_two(tmp_path):
    from dedup_pipeline.face_recognition.image_source import ImageIndex
    from dedup_pipeline.image_resolution.manifest import MANIFEST_FILE, PERSON_SUMMARY_FILE

    rng = random.Random(12)
    manifest, people = [], []
    for p in range(40):
        pid = f"p{p:03d}"
        n = rng.randint(0, 4)
        for k in range(n):
            manifest.append({"person_id": pid, "media_id": f"m{p}-{rng.randint(0, 99)}-{k}", "resolution_status": "resolved", "usable_for_face_recognition": rng.random() < 0.7,
                             "person_rank": rng.randint(1, 3), "path_ref": f"x/{p}/{k}.jpg"})
        people.append({"person_id": pid, "person_status": "usable_image_available" if n else "no_person_media_records", "usable_image_count": n, "unusable_reasons": None})
    (tmp_path / MANIFEST_FILE).write_text("\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    (tmp_path / PERSON_SUMMARY_FILE).write_text("\n".join(json.dumps(r) for r in people) + "\n", encoding="utf-8")
    two_parses = ImageIndex.from_output_dir(tmp_path)
    usable: dict = {}
    for r in manifest:
        if r["usable_for_face_recognition"]:
            usable.setdefault(r["person_id"], []).append(r)
    one_parse = ImageIndex.from_parsed(usable, people)
    for row in people:
        assert one_parse.get(row["person_id"]) == two_parses.get(row["person_id"])
    assert one_parse.get("unknown") == two_parses.get("unknown")
