"""Candidate generation: strategies, policy, deferral, determinism, checkpoints, read-only enforcement."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from dedup_pipeline.candidates import CandidateConfig, PairBudgetExceeded, PreflightError, RunOptions, run_candidate_generation
from dedup_pipeline.candidates import sql as sqlmod
from dedup_pipeline.candidates.blocks import candidate_id, count_pairs, expand_block, parse_block_row, source_pair_label
from dedup_pipeline.candidates.normalize import normalize_name, phone_key, token_sorted_key
from dedup_pipeline.db import ReadOnlyViolation, check_statement_is_read_only

from .candidates_fakes import FakePersonDb, SimulatedCrash, person

PACKAGE = Path(__file__).resolve().parent.parent / "dedup_pipeline" / "candidates"
FINALS = ("candidates.jsonl", "deferred_blocks.jsonl", "run_summary.json")


def run(persons, tmp_path, config=None, name="out", **db_kwargs):
    config = config or CandidateConfig()
    db = FakePersonDb(persons, config, **db_kwargs)
    out = tmp_path / name
    summary = run_candidate_generation(db, config, RunOptions(output_dir=out))
    return summary, load(out), db


def load(out: Path) -> dict:
    def jsonl(name):
        path = out / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return {"dir": out, "candidates": jsonl("candidates.jsonl"), "deferred": jsonl("deferred_blocks.jsonl")}


def pairs_of(result) -> dict[tuple[str, str], list[str]]:
    return {(c["person_a_id"], c["person_b_id"]): c["rule_names"] for c in result["candidates"]}


# ------------------------------------------------------------------- normalisation

@pytest.mark.parametrize(
    "raw,expected",
    [("  Ramesh   KUMAR ", "ramesh kumar"), ("O'Brien-Smith, Jr.", "o brien smith jr"), ("S/O Raju", "s o raju"),
     ("Name\tWith\ttabs", "name with tabs"), ("A.B.C. Reddy", "a b c reddy"), ("", ""), (None, ""), ("   ", ""),
     ("తెలుగు పేరు", "తెలుగు పేరు"), ("ÅNGSTRÖM  Lars", "ångström lars"), ("x_y z", "x y z")],
)
def test_name_normalisation(raw, expected):
    assert normalize_name(raw) == expected


def test_token_sorted_key_ignores_order_only():
    assert token_sorted_key("sai shiva alle") == token_sorted_key("alle sai shiva") == "alle sai shiva"
    assert token_sorted_key("alle sai") != token_sorted_key("alle sai shiva")


@pytest.mark.parametrize(
    "raw,key,reason",
    [("9876543211", "9876543211", None), ("+91 98765-43211", "9876543211", None), ("09876543211", "9876543211", None),
     ("919876543211", "9876543211", None), ("9876543211 9876543212", None, "too_many_digits"), ("98765", None, "too_few_digits"),
     ("5876543210", None, "bad_leading_digit"), ("9999999999", None, "repetitive_digits"), ("9898989898", None, "repetitive_digits"),
     ("1234567890", None, "bad_leading_digit"), (None, None, "no_phone"), ("  ", None, "no_phone"), ("abc", None, "no_digits")],
)
def test_phone_normalisation(raw, key, reason):
    assert phone_key(raw, CandidateConfig()) == (key, reason)


def test_phone_blocklist_is_configurable():
    cfg = CandidateConfig(phone_blocklist=("9876543211",))
    assert phone_key("9876543211", cfg) == (None, "blocklisted")
    assert phone_key("9876543211", CandidateConfig(phone_blocklist=())) == ("9876543211", None)


# ------------------------------------------------------------------ strategies

def test_exact_name_candidates_ignore_case_punctuation_and_spacing(tmp_path):
    people = [person("a1", name="Ramesh  Kumar"), person("a2", name="RAMESH KUMAR."), person("a3", name="Ramesh Kumaar"),
              person("a4", name="Suresh Babu")]
    summary, result, _ = run(people, tmp_path)
    assert pairs_of(result) == {("a1", "a2"): ["name_exact", "token_sorted_name"]}
    assert summary["strategies"]["name_exact"]["pairs_emitted"] == 1


def test_cleaned_name_is_preferred_and_raw_name_is_the_fallback(tmp_path):
    people = [person("a1", name="Mr Ramesh Kumar", cleaned="Ramesh Kumar"), person("a2", name="Ramesh Kumar"),
              person("a3", name="Dr Ramesh Kumar", cleaned=None)]
    _, result, _ = run(people, tmp_path)
    assert set(pairs_of(result)) == {("a1", "a2")}
    _, raw_result, _ = run(people, tmp_path, CandidateConfig(name_source="raw"), name="raw")
    assert raw_result["candidates"] == []


def test_single_token_names_are_excluded_and_reported_not_silently_dropped(tmp_path):
    people = [person("a1", name="Kumar"), person("a2", name="kumar"), person("a3", name="Kumar"), person("a4", name="Ab")]
    summary, result, _ = run(people, tmp_path)
    assert result["candidates"] == []
    ex = summary["strategies"]["name_exact"]["excluded_by_rule"]
    assert ex["too_few_tokens"] == 4 and ex["excluded_blocks"] == {"blocks": 1, "members": 3, "pairs": 3, "reason": ex["excluded_blocks"]["reason"]}
    _, included, _ = run(people, tmp_path, CandidateConfig(name_min_tokens=1, name_min_chars=2), name="one")
    assert set(pairs_of(included)) == {("a1", "a2"), ("a1", "a3"), ("a2", "a3")}


def test_phone_candidates_match_across_formats_and_skip_junk_numbers(tmp_path):
    people = [person("a1", phone="98765 43211"), person("a2", phone="+91-9876543211"), person("a3", phone="09876543211"),
              person("b1", phone="9999999999"), person("b2", phone="9999999999"), person("c1", phone="5876543210"),
              person("c2", phone="5876543210"), person("d1", phone="9123456780")]
    summary, result, _ = run(people, tmp_path)
    assert set(pairs_of(result)) == {("a1", "a2"), ("a1", "a3"), ("a2", "a3")}
    assert all(c["rule_names"] == ["phone_exact"] for c in result["candidates"])
    reasons = summary["strategies"]["phone_exact"]["excluded_by_rule"]
    assert reasons["persons_by_reason"]["repetitive_digits"] == 2 and reasons["persons_by_reason"]["bad_leading_digit"] == 2
    assert reasons["excluded_blocks_by_reason"]["repetitive_digits"] == {"blocks": 1, "members": 2, "pairs": 1}


def test_token_sorted_name_catches_reordered_names_only_through_its_own_rule(tmp_path):
    people = [person("a1", name="Alle Sai Shiva"), person("a2", name="Sai Shiva Alle"), person("a3", name="Sai Alle Shiva")]
    summary, result, _ = run(people, tmp_path)
    assert pairs_of(result) == {("a1", "a2"): ["token_sorted_name"], ("a1", "a3"): ["token_sorted_name"], ("a2", "a3"): ["token_sorted_name"]}
    assert summary["strategies"]["name_exact"]["pairs_emitted"] == 0
    assert summary["strategies"]["token_sorted_name"]["pairs_found_only_by_this_rule"] == 3


def test_alias_matches_another_persons_name_or_alias(tmp_path):
    people = [person("a1", name="Ravi Teja", alias="Raju Bhai"), person("a2", name="Raju Bhai"), person("a3", name="Other Person", alias="Raju Bhai"),
              person("a4", name="Someone Else", alias="Raju")]  # single-token alias is excluded
    summary, result, _ = run(people, tmp_path)
    assert pairs_of(result) == {("a1", "a2"): ["alias_name"], ("a1", "a3"): ["alias_name"], ("a2", "a3"): ["alias_name"]}
    assert summary["strategies"]["alias_name"]["excluded_by_rule"] == {"persons_with_alias": 3, "alias_below_minimum": 1}
    _, off, _ = run(people, tmp_path, CandidateConfig(alias_enabled=False), name="off")
    assert off["candidates"] == []


def test_plain_name_pairs_are_not_alias_pairs(tmp_path):
    people = [person("a1", name="Ravi Teja", alias="X Y"), person("a2", name="Ravi Teja"), person("a3", name="Ravi Teja")]
    _, result, _ = run(people, tmp_path)
    assert all("alias_name" not in names for names in pairs_of(result).values())


# ---------------------------------------------- union, canonical order, evidence

def test_pairs_found_by_several_rules_appear_once_listing_every_rule(tmp_path):
    people = [person("a1", name="Ramesh Kumar", phone="9876543211", alias="Ravi Kumar"),
              person("a2", name="Ramesh Kumar", phone="9876543211"), person("a3", name="Kumar Ramesh", phone="9876543211")]
    summary, result, _ = run(people, tmp_path)
    assert len(result["candidates"]) == 3  # one record per pair, not one per rule
    by_pair = {(c["person_a_id"], c["person_b_id"]): c for c in result["candidates"]}
    assert by_pair[("a1", "a2")]["rule_names"] == ["name_exact", "token_sorted_name", "phone_exact"]
    assert [r["rule"] for r in by_pair[("a1", "a2")]["rules"]] == ["name_exact", "token_sorted_name", "phone_exact"]
    assert by_pair[("a1", "a3")]["rule_names"] == ["token_sorted_name", "phone_exact"]
    s = summary["strategies"]
    assert s["phone_exact"]["pairs_emitted"] == 3 and s["phone_exact"]["unique_pairs"] == 3
    assert s["name_exact"]["unique_pairs"] == 1 and s["token_sorted_name"]["unique_pairs"] == 3
    assert sum(x["pairs_emitted"] for x in s.values()) == 7 and summary["union"]["candidate_pairs"] == 3
    assert summary["union"]["by_rule_combination"] == {"token_sorted_name+phone_exact": 2, "name_exact+token_sorted_name+phone_exact": 1}


def test_ids_are_stored_in_canonical_order_whatever_the_input_order(tmp_path):
    people = [person("b9", name="Same Name"), person("a1", name="Same Name"), person("m5", name="Same Name")]
    _, result, _ = run(people, tmp_path)
    assert list(pairs_of(result)) == [("a1", "b9"), ("a1", "m5"), ("b9", "m5")]
    assert all(c["person_a_id"] < c["person_b_id"] for c in result["candidates"])
    assert [c["candidate_id"] for c in result["candidates"]] == ["a1:b9", "a1:m5", "b9:m5"]
    assert candidate_id("a1", "b9") == "a1:b9"


def test_each_pair_carries_source_identifiers_and_rule_evidence_without_personal_data(tmp_path):
    people = [person("a1", "V1", name="Ramesh Kumar", phone="9876543211", record="cctns-77"),
              person("a2", "V2", name="Ramesh Kumar", phone="9876543211", record="mongo-99")]
    _, result, _ = run(people, tmp_path)
    record = result["candidates"][0]
    assert (record["source_a"], record["source_b"], record["source_pair"]) == ("V1", "V2", "V1-V2")
    assert (record["source_record_a"], record["source_record_b"]) == ("cctns-77", "mongo-99")
    name_rule = next(r for r in record["rules"] if r["rule"] == "name_exact")
    assert name_rule == {"rule": "name_exact", "block_size": 2, "hint": {"tokens": 2}}
    text = (result["dir"] / "candidates.jsonl").read_text() + (result["dir"] / "run_summary.json").read_text()
    assert "ramesh" not in text.lower() and "9876543211" not in text  # keys are not exported by default


def test_comparison_keys_are_exported_only_on_request(tmp_path):
    people = [person("a1", name="Ramesh Kumar", phone="9876543211"), person("a2", name="Ramesh Kumar", phone="9876543211")]
    _, result, _ = run(people, tmp_path, CandidateConfig(export_comparison_keys=True))
    keys = {r["rule"]: r["key"] for r in result["candidates"][0]["rules"]}
    assert keys == {"name_exact": "ramesh kumar", "token_sorted_name": "kumar ramesh", "phone_exact": "9876543211"}


# --------------------------------------------------------------- source policy

def mixed_people():
    return [person("a1", "V1", name="Ramesh Kumar"), person("a2", "V1", name="Ramesh Kumar"),
            person("b1", "V2", name="Ramesh Kumar"), person("b2", "V2", name="Ramesh Kumar")]


def test_default_policy_keeps_cross_source_and_same_source_and_reports_each(tmp_path):
    summary, result, _ = run(mixed_people(), tmp_path)
    assert summary["union"]["by_source_pair"] == {"V1-V1": 1, "V1-V2": 4, "V2-V2": 1}
    assert summary["source_policy"] == "all" and len(result["candidates"]) == 6


def test_cross_source_only_policy_counts_what_it_filters(tmp_path):
    summary, result, _ = run(mixed_people(), tmp_path, CandidateConfig(source_policy="cross_source_only"))
    assert summary["union"]["by_source_pair"] == {"V1-V2": 4} and len(result["candidates"]) == 4
    s = summary["strategies"]["name_exact"]
    assert (s["pairs_generated"], s["pairs_emitted"], s["pairs_policy_filtered"]) == (6, 4, 2)  # filtered pairs are counted, not hidden
    assert s["pairs_not_strategy_eligible"] == 0 and s["emitted_by_source_pair"] == {"V1-V2": 4}


def test_alias_skips_are_reported_as_strategy_eligibility_not_as_source_policy(tmp_path):
    people = [person("a1", "V1", name="Ravi Teja", alias="Raju Bhai"), person("a2", "V2", name="Raju Bhai"),
              person("a3", "V2", name="Raju Bhai"), person("a4", "V1", name="Raju Bhai")]  # three plain-name members + one alias holder
    summary, _, _ = run(people, tmp_path)
    s = summary["strategies"]["alias_name"]
    assert (s["pairs_generated"], s["pairs_not_strategy_eligible"], s["pairs_policy_filtered"], s["pairs_emitted"]) == (6, 3, 0, 3)


def test_same_source_only_policy(tmp_path):
    summary, _, _ = run(mixed_people(), tmp_path, CandidateConfig(source_policy="same_source_only"))
    assert summary["union"]["by_source_pair"] == {"V1-V1": 1, "V2-V2": 1}


def test_source_pair_label_is_order_independent():
    assert source_pair_label("V2", "V1") == source_pair_label("V1", "V2") == "V1-V2"


# ---------------------------------------------------------- oversized blocks

def big_block(n=5, prefix="p"):
    return [person(f"{prefix}{i:02d}", "V1" if i % 2 else "V2", name="Common Name") for i in range(n)]


def test_oversized_blocks_are_deferred_reported_and_never_partially_expanded(tmp_path):
    people = big_block(5) + [person("z1", name="Rare Name"), person("z2", name="Rare Name")]
    summary, result, _ = run(people, tmp_path, CandidateConfig(name_max_block_size=4, token_sorted_max_block_size=4))
    assert set(pairs_of(result)) == {("z1", "z2")}  # nothing from the 5-member block, not even a subset
    d = next(x for x in result["deferred"] if x["strategy"] == "name_exact")
    assert (d["reason"], d["member_count"], d["limit"], d["would_be_pairs"]) == ("block_too_large", 5, 4, 10)
    assert d["person_ids"] == ["p00", "p01", "p02", "p03", "p04"] and d["person_ids_truncated"] is False
    assert d["source_counts"] == {"V1": 2, "V2": 3} and d["would_be_pairs_after_policy"] == 10
    s = summary["strategies"]["name_exact"]
    assert (s["blocks_total"], s["blocks_deferred"], s["blocks_processed"], s["deferred_pairs"]) == (2, 1, 1, 10)
    assert summary["deferred"]["blocks"] == 2 and summary["deferred"]["impact"]["pairs_in_deferred_blocks"] == 20


def test_deferral_impact_shows_pairs_another_rule_still_finds(tmp_path):
    people = big_block(5)
    for p in people[:3]:
        p["phone_number"] = "9876543211"  # three of the five also share a phone, so phone covers 3 of the 10 pairs
    summary, result, _ = run(people, tmp_path, CandidateConfig(name_max_block_size=4, token_sorted_max_block_size=4))
    d = next(x for x in result["deferred"] if x["strategy"] == "name_exact")
    assert (d["pairs_already_found_by_another_rule"], d["pairs_not_found_by_any_rule"]) == (3, 7)
    assert summary["deferred"]["impact"]["not_found_by_any_rule"] == 14  # name_exact + token_sorted deferrals, 7 each


def test_deferred_listing_is_capped_but_flagged(tmp_path):
    cfg = CandidateConfig(name_max_block_size=4, token_sorted_max_block_size=4, deferred_listing_cap=3)
    _, result, _ = run(big_block(5), tmp_path, cfg)
    d = result["deferred"][0]
    assert len(d["person_ids"]) == 3 and d["person_ids_truncated"] is True and d["member_count"] == 5


def test_pair_budget_fails_the_run_instead_of_truncating(tmp_path):
    cfg = CandidateConfig(max_total_pairs=5)
    db = FakePersonDb(big_block(6), cfg)
    out = tmp_path / "out"
    with pytest.raises(PairBudgetExceeded, match="Nothing was truncated"):
        run_candidate_generation(db, cfg, RunOptions(output_dir=out))
    assert not (out / "run_summary.json").exists() and not (out / "candidates.jsonl").exists()
    assert json.loads((out / "run_state.json").read_text())["status"] == "failed"


def test_block_limits_and_rules_are_configurable_and_validated():
    with pytest.raises(ValueError):
        CandidateConfig(phone_max_block_size=1)
    with pytest.raises(ValueError):
        CandidateConfig(source_policy="nonsense")
    with pytest.raises(ValueError, match="unknown"):
        CandidateConfig.from_dict({"name_max_block": 5})
    assert CandidateConfig.from_dict({"name_max_block_size": 7, "phone_blocklist": ["1"]}).phone_blocklist == ("1",)


def test_canonical_order_does_not_depend_on_the_order_the_server_returns_members():
    block = parse_block_row("name_exact", 0, ("k", ["c3", "a1", "b2"], ["V2", "V1", "V1"], ["rc", "ra", "rb"]))
    assert [m.person_id for m in block.members] == ["a1", "b2", "c3"]
    pairs = [(a.person_id, b.person_id) for a, b in expand_block(block, "all")]
    assert pairs == [("a1", "b2"), ("a1", "c3"), ("b2", "c3")] and all(a < b for a, b in pairs)
    first = block.members[0]  # identifiers travel with their person even after sorting
    assert (first.source_system, first.source_record_id) == ("V1", "ra")


def test_count_pairs_matches_enumeration_for_every_policy():
    block = parse_block_row("name_exact", 0, ("k", ["a", "b", "c", "d"], ["V1", "V1", "V2", "V2"], ["1", "2", "3", "4"]))
    for policy in ("all", "cross_source_only", "same_source_only"):
        total, eligible, allowed = count_pairs(block, policy)
        assert (total, eligible) == (6, 6) and allowed == len(list(expand_block(block, policy)))
    alias = parse_block_row("alias_name", 0, ("k", ["a", "b", "c"], ["V1", "V2", "V2"], ["1", "2", "3"], [True, False, False]))
    # b-c has no alias on either side so the strategy skips it; the two eligible pairs (a-b, a-c) are both V1-V2
    assert count_pairs(alias, "all") == (3, 2, 2) and count_pairs(alias, "cross_source_only") == (3, 2, 2)
    assert count_pairs(alias, "same_source_only") == (3, 2, 0)


# ----------------------------------------------------------- missing fields

def test_missing_and_blank_fields_never_crash_and_never_match(tmp_path):
    people = [person("a1"), person("a2", name="", phone="", alias=""), person("a3", name="  ", phone="  "),
              person("a4", name=None, phone=None), person("a5", name="Real Person", phone="9876543211"),
              person("a6", name="Real Person", phone="9876543211")]
    summary, result, _ = run(people, tmp_path)
    assert set(pairs_of(result)) == {("a5", "a6")}
    assert summary["strategies"]["name_exact"]["excluded_by_rule"]["blank_name"] == 4
    assert summary["strategies"]["phone_exact"]["excluded_by_rule"]["persons_by_reason"]["no_phone"] == 4


def test_no_candidates_still_completes_with_empty_files(tmp_path):
    summary, result, _ = run([person("a1", name="Only One")], tmp_path)
    assert summary["status"] == "completed" and result["candidates"] == [] and summary["union"]["candidate_pairs"] == 0


# ------------------------------------------------------------- determinism

def test_output_is_byte_identical_across_runs_and_input_orderings(tmp_path):
    people = [person(f"a{i:02d}", "V1" if i % 3 else "V2", name=f"Name {i % 7} X", phone=f"98765432{i % 11:02d}") for i in range(60)]
    outputs = []
    for seed in range(4):
        shuffled = people[:]
        random.Random(seed).shuffle(shuffled)
        _, result, _ = run(shuffled, tmp_path, name=f"run{seed}")
        outputs.append(tuple((result["dir"] / f).read_bytes() for f in ("candidates.jsonl", "deferred_blocks.jsonl")))
    assert len(set(outputs)) == 1 and outputs[0][0]


def test_rerunning_into_the_same_directory_is_idempotent(tmp_path):
    people = [person(f"a{i}", name="Same Name") for i in range(4)]
    _, first, _ = run(people, tmp_path)
    before = (first["dir"] / "candidates.jsonl").read_bytes()
    _, second, _ = run(people, tmp_path)
    assert (second["dir"] / "candidates.jsonl").read_bytes() == before
    assert json.loads((second["dir"] / "run_state.json").read_text())["status"] == "completed"


# ------------------------------------------------------- checkpoints and resumption

def crashing_setup(tmp_path):
    people = [person(f"a{i:02d}", "V1" if i % 2 else "V2", name=f"Pair {i // 2} Name", phone=f"98765432{i // 3:02d}") for i in range(40)]
    cfg = CandidateConfig(checkpoint_every_blocks=1)
    return people, cfg, tmp_path / "out"


def test_a_crashed_run_is_incomplete_and_a_resume_finishes_it_identically(tmp_path):
    people, cfg, out = crashing_setup(tmp_path)
    _, reference, _ = run(people, tmp_path, cfg, name="reference")

    crashing = FakePersonDb(people, cfg, crash_in=("token_sorted_name:blocks", 5))
    with pytest.raises(SimulatedCrash):
        run_candidate_generation(crashing, cfg, RunOptions(output_dir=out))
    assert not any((out / f).exists() for f in FINALS)  # a partial run has no results and no completion marker
    state = json.loads((out / "run_state.json").read_text())
    assert state["status"] == "failed" and "SimulatedCrash" in state["error"]
    checkpoint = json.loads((out / "work" / "checkpoint.json").read_text())
    assert checkpoint["strategies"]["name_exact"]["complete"] and not checkpoint["strategies"]["token_sorted_name"]["complete"]
    assert checkpoint["strategies"]["token_sorted_name"]["blocks_done"] == 5

    healthy = FakePersonDb(people, cfg)
    summary = run_candidate_generation(healthy, cfg, RunOptions(output_dir=out))
    assert summary["resumed_from_checkpoint"] is True
    assert "name_exact:blocks" not in healthy.streams  # the finished strategy was not queried again
    assert (out / "candidates.jsonl").read_bytes() == (reference["dir"] / "candidates.jsonl").read_bytes()
    assert {k: summary["strategies"][k]["pairs_emitted"] for k in summary["strategies"]} == \
           {k: v["pairs_emitted"] for k, v in json.loads((reference["dir"] / "run_summary.json").read_text())["strategies"].items()}


def test_resume_discards_work_written_after_the_last_checkpoint(tmp_path):
    people, cfg, out = crashing_setup(tmp_path)
    cfg = CandidateConfig(checkpoint_every_blocks=1000)  # no mid-strategy checkpoint: a crash leaves extra, uncheckpointed lines
    _, reference, _ = run(people, tmp_path, cfg, name="reference")
    with pytest.raises(SimulatedCrash):
        run_candidate_generation(FakePersonDb(people, cfg, crash_in=("name_exact:blocks", 4)), cfg, RunOptions(output_dir=out))
    assert (out / "work" / "name_exact.pairs.jsonl").read_text() != ""  # lines exist that the checkpoint does not cover
    run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=out))
    assert (out / "candidates.jsonl").read_bytes() == (reference["dir"] / "candidates.jsonl").read_bytes()  # no duplicates


def test_a_half_written_work_line_is_ignored_on_resume(tmp_path):
    people, cfg, out = crashing_setup(tmp_path)
    _, reference, _ = run(people, tmp_path, cfg, name="reference")
    with pytest.raises(SimulatedCrash):
        run_candidate_generation(FakePersonDb(people, cfg, crash_in=("name_exact:blocks", 6)), cfg, RunOptions(output_dir=out))
    with open(out / "work" / "name_exact.pairs.jsonl", "a") as fh:
        fh.write('{"a": "a01", "b": "a0')  # killed mid-write
    run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=out))
    assert (out / "candidates.jsonl").read_bytes() == (reference["dir"] / "candidates.jsonl").read_bytes()


def test_a_checkpoint_from_other_settings_or_changed_data_is_not_reused(tmp_path):
    people, cfg, out = crashing_setup(tmp_path)
    with pytest.raises(SimulatedCrash):
        run_candidate_generation(FakePersonDb(people, cfg, crash_in=("token_sorted_name:blocks", 3)), cfg, RunOptions(output_dir=out))
    other = CandidateConfig(checkpoint_every_blocks=1, phone_max_block_size=5)
    db = FakePersonDb(people, other)
    assert run_candidate_generation(db, other, RunOptions(output_dir=out))["resumed_from_checkpoint"] is False
    assert "name_exact:blocks" in db.streams  # started over

    with pytest.raises(SimulatedCrash):
        run_candidate_generation(FakePersonDb(people, cfg, crash_in=("token_sorted_name:blocks", 3)), cfg, RunOptions(output_dir=out))
    changed = FakePersonDb(people, cfg, stamp=(41, "2026-10-02T00:00:00", 41, "2026-10-02T00:00:00"))
    assert run_candidate_generation(changed, cfg, RunOptions(output_dir=out))["resumed_from_checkpoint"] is False


def test_a_completed_run_is_never_resumed_and_new_runs_clear_old_results_first(tmp_path):
    people, cfg, out = crashing_setup(tmp_path)
    run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=out))
    assert all((out / f).exists() for f in FINALS)
    crashing = FakePersonDb(people, cfg, crash_in=("name_exact:blocks", 2))
    with pytest.raises(SimulatedCrash):
        run_candidate_generation(crashing, cfg, RunOptions(output_dir=out))
    assert not any((out / f).exists() for f in FINALS)  # the old completed results cannot be mistaken for this run's
    assert "name_exact:blocks" in crashing.streams  # and the completed run was not resumed


def test_atomic_writes_survive_a_briefly_locked_destination_but_not_a_permanent_one(tmp_path, monkeypatch):
    from dedup_pipeline.candidates import store

    real_replace, failures = store.os.replace, {"left": 3}

    def flaky(src, dst):  # Windows: antivirus/indexer holds the file for a moment -> PermissionError
        if failures["left"] > 0:
            failures["left"] -= 1
            raise PermissionError("[WinError 5] Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    target = tmp_path / "x.json"
    store.write_json_atomic(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True} and failures["left"] == 0

    failures["left"] = 10_000
    with pytest.raises(PermissionError):
        store.write_json_atomic(tmp_path / "y.json", {"never": "written"})
    assert not (tmp_path / "y.json").exists()  # a permanent failure is reported, never swallowed


def test_the_summary_is_the_last_thing_written(tmp_path):
    summary, result, _ = run([person("a1", name="Same Name"), person("a2", name="Same Name")], tmp_path)
    out = result["dir"]
    assert summary["status"] == "completed" and summary["complete"] is True
    assert json.loads((out / "run_state.json").read_text())["status"] == "completed"
    assert (out / "run_summary.json").stat().st_mtime_ns >= (out / "candidates.jsonl").stat().st_mtime_ns
    assert not list(out.glob("*.tmp")) and not list((out / "work").glob("*.tmp"))


# ------------------------------------------------------- run summary content

def test_summary_reports_counts_timing_memory_and_configuration(tmp_path):
    people = [person(f"a{i:02d}", "V1" if i % 2 else "V2", name="Common Name", phone="9876543211") for i in range(5)]
    summary, _, _ = run(people, tmp_path)
    assert summary["union"]["candidate_pairs"] == 10 and summary["union"]["persons_covered"] == 5
    assert sum(summary["union"]["by_source_pair"].values()) == 10 and summary["config"]["name_max_block_size"] == 30
    assert summary["elapsed_seconds"] >= 0 and summary["peak_python_heap_mb"] >= 0 and "peak_process_memory_mb" in summary
    assert summary["database"] == {"database": "dopams_cctns_v2", "transaction_read_only": "on", "server_version": "PostgreSQL 16.14 fake"}
    for name, s in summary["strategies"].items():
        assert {"blocks_total", "blocks_deferred", "pairs_generated", "pairs_emitted", "unique_pairs", "block_size_limit", "seconds"} <= set(s)


# ---------------------------------------------------- image-resolution references

def phase2_dir(directory: Path, rows: list[dict], partial=False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "person_summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "run_summary.json").write_text(json.dumps({"parameters": {"limit_persons": 10 if partial else None}}))
    return directory


def test_image_references_are_attached_when_a_completed_phase_2_run_exists(tmp_path):
    images = phase2_dir(tmp_path / "img", [
        {"person_id": "a1", "person_status": "usable_image_available", "usable_image_count": 2, "selected_media_id": "M1"},
        {"person_id": "a2", "person_status": "usable_image_available", "usable_image_count": 1, "selected_media_id": "M2"},
        {"person_id": "a3", "person_status": "no_person_media_records", "usable_image_count": 0, "selected_media_id": None}])
    people = [person("a1", name="Same Name"), person("a2", name="Same Name"), person("a3", name="Same Name"), person("a4", name="Same Name")]
    cfg = CandidateConfig()
    out = tmp_path / "out"
    summary = run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=out, image_dir=images))
    by_pair = {(c["person_a_id"], c["person_b_id"]): c["images"] for c in load(out)["candidates"]}
    assert by_pair[("a1", "a2")] == {"a": {"person_status": "usable_image_available", "usable_images": 2, "selected_media_id": "M1"},
                                     "b": {"person_status": "usable_image_available", "usable_images": 1, "selected_media_id": "M2"},
                                     "both_have_usable_image": True}
    assert by_pair[("a1", "a3")]["both_have_usable_image"] is False and by_pair[("a1", "a4")]["b"]["person_status"] == "not_in_manifest"
    assert summary["union"]["pairs_where_both_persons_have_a_usable_image"] == 1
    assert summary["image_resolution"] == {"available": True, "source": "img", "partial_run": False, "persons": 3}


def test_image_references_are_optional_and_incomplete_phase_2_runs_are_ignored(tmp_path):
    people = [person("a1", name="Same Name"), person("a2", name="Same Name")]
    summary, result, _ = run(people, tmp_path)
    assert "images" not in result["candidates"][0] and summary["image_resolution"] == {"available": False}
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "person_summary.jsonl").write_text("")  # no run_summary.json: Phase 2 never finished
    cfg = CandidateConfig()
    s = run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=tmp_path / "o2", image_dir=broken))
    assert s["image_resolution"] == {"available": False}
    partial = phase2_dir(tmp_path / "partial", [], partial=True)
    s = run_candidate_generation(FakePersonDb(people, cfg), cfg, RunOptions(output_dir=tmp_path / "o3", image_dir=partial))
    assert s["image_resolution"]["partial_run"] is True


# ------------------------------------------------- read-only enforcement, no writes

def test_a_connection_that_is_not_read_only_fails_closed_before_anything_is_produced(tmp_path):
    cfg = CandidateConfig()
    out = tmp_path / "out"
    db = FakePersonDb([person("a1", name="Same Name")], cfg, read_only="off")
    with pytest.raises(ReadOnlyViolation):
        run_candidate_generation(db, cfg, RunOptions(output_dir=out))
    assert not any((out / f).exists() for f in FINALS) and db.streams == []  # no block was ever read


def test_an_unapproved_database_fails_closed(tmp_path):
    cfg = CandidateConfig()
    with pytest.raises(ReadOnlyViolation, match="unapproved"):
        run_candidate_generation(FakePersonDb([], cfg, database="dopams_cctns"), cfg, RunOptions(output_dir=tmp_path / "o"))


def test_ids_whose_sort_order_could_differ_from_the_database_fail_closed(tmp_path):
    cfg = CandidateConfig()
    with pytest.raises(PreflightError, match="lower-case alphanumeric"):
        run_candidate_generation(FakePersonDb([], cfg, bad_ids=3), cfg, RunOptions(output_dir=tmp_path / "o"))


def test_every_statement_issued_is_a_read_only_select(tmp_path):
    people = [person("a1", name="Same Name", phone="9876543211", alias="X Y"), person("a2", name="Same Name", phone="9876543211")]
    _, _, db = run(people, tmp_path)
    assert db.statements and all(re.match(r"^\s*/\*[^*]*\*/\s*(SELECT|WITH)|^\s*(SHOW|SELECT)", s, re.I) for s in db.statements)
    for s in db.statements:
        check_statement_is_read_only(s)


@pytest.mark.parametrize("strategy", ["name_exact", "token_sorted_name", "phone_exact", "alias_name"])
def test_the_real_queries_pass_the_read_only_guard_and_have_all_their_parameters(strategy):
    cfg = CandidateConfig()
    for spec in [sqlmod.block_query(strategy, cfg), *sqlmod.exclusion_queries(strategy, cfg)]:
        check_statement_is_read_only(spec.sql)
        assert spec.sql.lstrip().startswith("/* candidates:")
        assert set(re.findall(r"%\((\w+)\)s", spec.sql)) <= set(spec.params), spec.tag
        assert "%s" not in spec.sql.replace("%(", "")  # named parameters only
    for spec in (sqlmod.preflight_person_ids(), sqlmod.dataset_stamp()):
        check_statement_is_read_only(spec.sql)


class RecordingRaw:
    """Stands in for a psycopg2 connection to check the streaming cursor's transaction handling."""

    def __init__(self):
        self.autocommit, self.events = True, []

    def cursor(self, name=None):
        raw = self
        raw.events.append(("cursor", name, raw.autocommit))

        class Cur:
            itersize = 0
            rows = [(1,), (2,), (3,)]

            def execute(self, sql, params=None):
                raw.events.append(("execute", sql))

            def fetchmany(self, n):
                chunk, self.rows = self.rows[:n], self.rows[n:]
                return chunk

            def close(self):
                raw.events.append(("close",))

        return Cur()

    def rollback(self):
        self.events.append(("rollback",))


def test_streaming_uses_a_server_side_cursor_in_a_rolled_back_transaction():
    from dedup_pipeline.db import ReadOnlyConnection

    raw = RecordingRaw()
    rows = list(ReadOnlyConnection(raw).stream("SELECT 1", batch_size=2))
    assert rows == [(1,), (2,), (3,)]
    kinds = [e[0] for e in raw.events]
    assert kinds[0] == "cursor" and raw.events[0][1].startswith("dedup_stream_") and raw.events[0][2] is False  # named, in a transaction
    assert kinds[-2:] == ["close", "rollback"] and "commit" not in kinds and raw.autocommit is True


def test_streaming_refuses_anything_that_is_not_a_read():
    from dedup_pipeline.db import ReadOnlyConnection

    raw = RecordingRaw()
    with pytest.raises(ReadOnlyViolation):
        list(ReadOnlyConnection(raw).stream("DELETE FROM public.persons_unified"))
    assert raw.events == []  # nothing reached the (fake) server


def test_the_candidate_package_contains_no_write_paths_and_no_fingerprinting():
    source = "\n".join(p.read_text(encoding="utf-8") for p in PACKAGE.rglob("*.py"))
    lower = source.lower()
    for forbidden in ("executemany", "copy_from", "copy_expert", ".commit(", "execute_values", "insert into",
                      "hashlib", "sha256", "imagehash", "phash", "md5", "perceptual"):
        assert forbidden not in lower, forbidden
    # SQL text lives only in sql.py, and none of it touches the link/mapping tables the migration would add
    sql_text = (PACKAGE / "sql.py").read_text(encoding="utf-8").lower()
    for table in ("identity_links", "identity_dedup_run_log", "person_canonical_mapping", "media_unified"):
        assert table not in sql_text, table
