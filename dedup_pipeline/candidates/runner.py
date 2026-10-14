"""Candidate generation: blocking strategies -> deduplicated candidate pairs, written to local files only.

Flow: verify the connection is read-only (fail closed) -> for each enabled strategy stream its blocks from the
database in bounded batches, expand them into canonical-order pairs and append to a local work file, with a
checkpoint every few hundred blocks -> merge the work files into the union (one record per pair, listing every
rule that produced it) -> write the final files, ``run_summary.json`` last.

Nothing here writes to a database, scores a pair, calls FR-DOPAMS or merges identities.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..db import ReadOnlyConnection, ReadOnlyViolation, assert_read_only
from . import sql
from .blocks import (
    Member, candidate_id, count_pairs, expand_block, pair_eligible, parse_block_row, source_counts, source_pair_label,
)
from .config import ALL_STRATEGIES, CandidateConfig
from .images import ImageRefs, image_block, load_image_refs
from .memory import peak_rss_bytes
from .store import (
    CANDIDATES_FILE, DEFERRED_FILE, SUMMARY_FILE, RunStore, now_iso, write_json_atomic, write_jsonl_atomic,
)

log = logging.getLogger(__name__)
GENERATOR_VERSION = "candidates-v2"  # bumped when checkpoint contents change, so older checkpoints are never reused


class PreflightError(RuntimeError):
    """A precondition for trustworthy output does not hold; the run stops before producing anything."""


class PairBudgetExceeded(RuntimeError):
    """More candidate pairs than ``max_total_pairs``. The run fails; pairs are never silently dropped."""


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path
    image_dir: Optional[Path] = None
    resume: bool = True
    fresh: bool = False


# --------------------------------------------------------------------------- helpers

def _one(conn: ReadOnlyConnection, spec: sql.QuerySpec) -> tuple:
    cur = conn.cursor()
    cur.execute(spec.sql, spec.params)
    row = cur.fetchone()
    cur.close()
    return row


def _all(conn: ReadOnlyConnection, spec: sql.QuerySpec) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(spec.sql, spec.params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _jsonable(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def preflight(conn: ReadOnlyConnection) -> tuple[dict[str, str], dict[str, Any]]:
    """Fail closed: read-only session on the approved database, and ids whose order is unambiguous."""
    try:
        info = assert_read_only(conn)
    except ReadOnlyViolation:
        raise
    bad_ids = _one(conn, sql.preflight_person_ids())[0]
    if bad_ids:
        raise PreflightError(
            f"{bad_ids} person_id values are not lower-case alphanumeric; Python's pair ordering could then differ "
            "from the database's (identity_links requires person_a_id < person_b_id). Refusing to continue."
        )
    row = _one(conn, sql.dataset_stamp())
    stamp = {"persons": row[0], "persons_max_computed_at": _jsonable(row[1]),
             "person_enrichment": row[2], "person_enrichment_max_computed_at": _jsonable(row[3])}
    return info, stamp


def _exclusions(conn: ReadOnlyConnection, strategy: str, config: CandidateConfig) -> dict[str, Any]:
    queries = sql.exclusion_queries(strategy, config)
    if strategy == "phone_exact":
        persons = {r[0]: r[1] for r in _all(conn, queries[0])}
        blocks = {r[0]: {"blocks": r[1], "members": int(r[2]), "pairs": int(r[3])} for r in _all(conn, queries[1])}
        return {"persons_by_reason": dict(sorted(persons.items())), "excluded_blocks_by_reason": dict(sorted(blocks.items()))}
    if strategy == "alias_name":
        row = _one(conn, queries[0])
        return {"persons_with_alias": row[0], "alias_below_minimum": row[1]}
    counts, blocks = _one(conn, queries[0]), _one(conn, queries[1])
    return {
        "persons": counts[0], "blank_name": counts[1], "too_few_tokens": counts[2], "too_short": counts[3],
        "excluded_blocks": {"blocks": blocks[0], "members": int(blocks[1]), "pairs": int(blocks[2]),
                            "reason": "name below the minimum token / character count"},
    }


def _empty_stats() -> dict[str, Any]:
    return {"blocks_total": 0, "blocks_processed": 0, "blocks_deferred": 0, "members_processed": 0, "pairs_generated": 0,
            "pairs_not_strategy_eligible": 0, "pairs_policy_filtered": 0, "pairs_emitted": 0, "deferred_pairs": 0,
            "deferred_pairs_after_policy": 0, "emitted_by_source_pair": {}}


# ------------------------------------------------------------------ one strategy

def process_strategy(
    conn: ReadOnlyConnection, strategy: str, config: CandidateConfig, store: RunStore,
    checkpoint: dict[str, Any], budget: list[int],
) -> None:
    entry = checkpoint["strategies"].get(strategy)
    if entry and entry["complete"]:
        log.info("%s: complete in checkpoint, reusing %d pair lines", strategy, entry["pair_lines"])
        return
    started = time.perf_counter()
    if entry is None:
        entry = {"complete": False, "blocks_done": 0, "pair_lines": 0, "stats": _empty_stats(), "deferred": [],
                 "exclusions": _exclusions(conn, strategy, config), "seconds": 0.0}
        checkpoint["strategies"][strategy] = entry
        store.pairs_path(strategy).write_text("", encoding="utf-8")
    else:
        log.info("%s: resuming after %d blocks", strategy, entry["blocks_done"])
    store.truncate_pairs(strategy, entry["pair_lines"])  # drop anything written after the last checkpoint
    stats, cap, policy = entry["stats"], config.max_block_size(strategy), config.source_policy
    spec = sql.block_query(strategy, config)
    elapsed_before = entry["seconds"]
    skip = entry["blocks_done"]

    def commit() -> None:
        handle.flush()
        entry["seconds"] = round(elapsed_before + time.perf_counter() - started, 2)
        store.save_checkpoint(checkpoint)

    with open(store.pairs_path(strategy), "a", encoding="utf-8", newline="\n") as handle:
        for ordinal, row in enumerate(conn.stream(spec.sql, spec.params, config.stream_batch_size)):
            if ordinal < skip:
                continue
            block = parse_block_row(strategy, ordinal, row)
            stats["blocks_total"] += 1
            total_pairs, eligible_pairs, allowed_pairs = count_pairs(block, policy)
            if block.size > cap:
                stats["blocks_deferred"] += 1
                stats["deferred_pairs"] += total_pairs
                stats["deferred_pairs_after_policy"] += allowed_pairs
                entry["deferred"].append({
                    "strategy": strategy, "block_ordinal": ordinal, "reason": "block_too_large", "limit": cap,
                    "member_count": block.size, "would_be_pairs": total_pairs, "would_be_pairs_after_policy": allowed_pairs,
                    "source_counts": source_counts(block), "hint": block.hint(),
                    "members": [[m.person_id, m.source_system, m.is_alias] for m in block.members],
                })
            else:
                stats["blocks_processed"] += 1
                stats["members_processed"] += block.size
                stats["pairs_generated"] += total_pairs
                stats["pairs_not_strategy_eligible"] += total_pairs - eligible_pairs
                stats["pairs_policy_filtered"] += eligible_pairs - allowed_pairs
                for a, b in expand_block(block, policy):
                    rule: dict[str, Any] = {"rule": strategy, "block_size": block.size, "hint": block.hint()}
                    if config.export_comparison_keys:
                        rule["key"] = block.key
                    handle.write(json.dumps({"a": a.person_id, "b": b.person_id, "sa": a.source_system, "sb": b.source_system,
                                             "ra": a.source_record_id, "rb": b.source_record_id, "r": rule},
                                            ensure_ascii=True, separators=(",", ":")) + "\n")
                    label = source_pair_label(a.source_system, b.source_system)
                    stats["emitted_by_source_pair"][label] = stats["emitted_by_source_pair"].get(label, 0) + 1
                    stats["pairs_emitted"] += 1
                    entry["pair_lines"] += 1
                    budget[0] += 1
            entry["blocks_done"] = ordinal + 1  # state is consistent from here: block fully counted and written
            if budget[0] > config.max_total_pairs:
                commit()
                raise PairBudgetExceeded(
                    f"more than max_total_pairs={config.max_total_pairs} candidate pairs; lower the block limits, "
                    "tighten a rule or raise the budget. Nothing was truncated."
                )
            if entry["blocks_done"] % config.checkpoint_every_blocks == 0:
                commit()
        entry["complete"] = True
        commit()
    log.info("%s: %d blocks (%d deferred), %d pairs emitted", strategy, stats["blocks_total"], stats["blocks_deferred"], stats["pairs_emitted"])


# ------------------------------------------------------------------------- merge

def merge_outputs(
    store: RunStore, config: CandidateConfig, checkpoint: dict[str, Any], refs: Optional[ImageRefs]
) -> dict[str, Any]:
    union: dict[tuple[str, str], dict[str, Any]] = {}
    per_rule: Counter = Counter()
    for strategy in config.enabled_strategies():
        with open(store.pairs_path(strategy), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                item = union.get((rec["a"], rec["b"]))
                if item is None:
                    item = union[(rec["a"], rec["b"])] = {"sa": rec["sa"], "sb": rec["sb"], "ra": rec["ra"], "rb": rec["rb"], "rules": []}
                item["rules"].append(rec["r"])
                per_rule[strategy] += 1

    unique_per_rule: Counter = Counter()
    only_per_rule: Counter = Counter()
    by_combo: Counter = Counter()
    by_source: Counter = Counter()
    by_source_and_rules: dict[str, Counter] = {}
    covered_persons: set[str] = set()
    both_usable = 0
    records = []
    for (a, b) in sorted(union):
        item = union[(a, b)]
        item["rules"].sort(key=lambda r: (ALL_STRATEGIES.index(r["rule"]), r["block_size"]))
        names = sorted({r["rule"] for r in item["rules"]}, key=ALL_STRATEGIES.index)
        for name in names:
            unique_per_rule[name] += 1
        if len(names) == 1:
            only_per_rule[names[0]] += 1
        combo = "+".join(names)
        by_combo[combo] += 1
        label = source_pair_label(item["sa"], item["sb"])
        by_source[label] += 1
        by_source_and_rules.setdefault(label, Counter())[combo] += 1
        covered_persons.update((a, b))
        record: dict[str, Any] = {
            "candidate_id": candidate_id(a, b), "person_a_id": a, "person_b_id": b,
            "source_a": item["sa"], "source_b": item["sb"], "source_pair": label,
            "source_record_a": item["ra"], "source_record_b": item["rb"],
            "rule_names": names, "rules": item["rules"],
        }
        if refs is not None:
            record["images"] = image_block(refs, a, b)
            both_usable += record["images"]["both_have_usable_image"]
        records.append(record)

    deferred_out, impact_total = _deferred_records(checkpoint, config, union)
    candidates = write_jsonl_atomic(store.dir / CANDIDATES_FILE, records)
    write_jsonl_atomic(store.dir / DEFERRED_FILE, deferred_out)
    return {
        "candidates": candidates, "per_rule_emitted": per_rule, "unique_per_rule": unique_per_rule, "only_per_rule": only_per_rule,
        "by_combo": by_combo, "by_source": by_source, "by_source_and_rules": by_source_and_rules,
        "persons_covered": len(covered_persons), "both_usable": both_usable if refs is not None else None,
        "deferred_blocks": len(deferred_out), "deferred_impact": impact_total,
    }


def _deferred_records(
    checkpoint: dict[str, Any], config: CandidateConfig, union: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deferred blocks for the output, with how many of their pairs another rule already found."""
    out: list[dict[str, Any]] = []
    totals = {"pairs_in_deferred_blocks": 0, "already_found_by_another_rule": 0, "not_found_by_any_rule": 0, "impact_unknown_blocks": 0}
    for strategy in config.enabled_strategies():
        for d in checkpoint["strategies"][strategy]["deferred"]:
            members = [Member(pid, src, "", bool(alias)) for pid, src, alias in d["members"]]
            record = {k: v for k, v in d.items() if k != "members"}
            ids = [m.person_id for m in members]
            record["person_ids"] = ids[: config.deferred_listing_cap]
            record["person_ids_truncated"] = len(ids) > config.deferred_listing_cap
            if d["would_be_pairs_after_policy"] <= config.impact_pair_limit:
                found = missing = 0
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        if pair_eligible(a, b, strategy, config.source_policy):
                            if (a.person_id, b.person_id) in union:
                                found += 1
                            else:
                                missing += 1
                record["pairs_already_found_by_another_rule"] = found
                record["pairs_not_found_by_any_rule"] = missing
                totals["already_found_by_another_rule"] += found
                totals["not_found_by_any_rule"] += missing
            else:
                record["pairs_already_found_by_another_rule"] = record["pairs_not_found_by_any_rule"] = None
                totals["impact_unknown_blocks"] += 1
            totals["pairs_in_deferred_blocks"] += d["would_be_pairs_after_policy"]
            out.append(record)
    return out, totals


# ---------------------------------------------------------------------------- run

def run_candidate_generation(conn: ReadOnlyConnection, config: CandidateConfig, options: RunOptions) -> dict[str, Any]:
    started_at, t0 = now_iso(), time.perf_counter()
    store = RunStore(options.output_dir)
    tracemalloc.start()
    run_id = store.new_run_id()
    try:
        info, stamp = preflight(conn)  # raises before anything is written if the session is not provably read-only
        log.info("Database %s confirmed read-only.", info["database"])

        state, checkpoint = store.read_state(), store.load_checkpoint()
        resumable = bool(
            options.resume and not options.fresh and state and checkpoint
            and state.get("status") in ("running", "failed", "interrupted")
            and checkpoint.get("signature") == config.signature() and checkpoint.get("dataset_stamp") == stamp
            and checkpoint.get("version") == GENERATOR_VERSION
        )
        store.clear_finals()  # completion marker disappears first; it is written last
        if resumable:
            run_id, resumed = state["run_id"], True
            log.info("Resuming run %s from its checkpoint.", run_id)
        else:
            store.reset_work()
            checkpoint = {"version": GENERATOR_VERSION, "signature": config.signature(), "dataset_stamp": stamp, "strategies": {}}
            store.save_checkpoint(checkpoint)
            resumed = False
        store.write_state("running", run_id, started_at=started_at, resumed=resumed)

        budget = [sum(e["stats"]["pairs_emitted"] for e in checkpoint["strategies"].values())]
        for strategy in config.enabled_strategies():
            process_strategy(conn, strategy, config, store, checkpoint, budget)

        refs = load_image_refs(options.image_dir)
        merged = merge_outputs(store, config, checkpoint, refs)
        _, tm_peak = tracemalloc.get_traced_memory()
        summary = _summary(run_id, started_at, t0, info, stamp, config, checkpoint, merged, refs, resumed, tm_peak, options)
        write_json_atomic(store.dir / SUMMARY_FILE, summary)  # the completion marker, written last
        store.write_state("completed", run_id, started_at=started_at, finished_at=now_iso(), resumed=resumed)
        return summary
    except KeyboardInterrupt:
        store.write_state("interrupted", run_id, started_at=started_at)
        raise
    except BaseException as exc:
        store.write_state("failed", run_id, started_at=started_at, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    finally:
        tracemalloc.stop()


def _summary(run_id, started_at, t0, info, stamp, config, checkpoint, merged, refs, resumed, tm_peak, options) -> dict[str, Any]:
    strategies = {}
    for name in config.enabled_strategies():
        e = checkpoint["strategies"][name]
        stats = e["stats"]
        strategies[name] = {
            **{k: v for k, v in stats.items() if k != "emitted_by_source_pair"},
            "unique_pairs": merged["unique_per_rule"].get(name, 0),
            "pairs_found_only_by_this_rule": merged["only_per_rule"].get(name, 0),
            "emitted_by_source_pair": dict(sorted(stats["emitted_by_source_pair"].items())),
            "block_size_limit": config.max_block_size(name),
            "excluded_by_rule": e["exclusions"],
            "seconds": e["seconds"],
        }
    return {
        "status": "completed",
        "complete": True,
        "generator_version": GENERATOR_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": now_iso(),
        "resumed_from_checkpoint": resumed,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "peak_python_heap_mb": round(tm_peak / 1048576, 1),
        "peak_process_memory_mb": round(peak_rss_bytes() / 1048576, 1) if peak_rss_bytes() else None,
        "database": info,
        "dataset_stamp": stamp,
        "source_policy": config.source_policy,
        "config": config.to_dict(),
        "strategies": strategies,
        "union": {
            "candidate_pairs": merged["candidates"],
            "persons_covered": merged["persons_covered"],
            "by_source_pair": dict(sorted(merged["by_source"].items())),
            "by_rule_combination": dict(sorted(merged["by_combo"].items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_source_pair_and_rule_combination": {k: dict(sorted(v.items())) for k, v in sorted(merged["by_source_and_rules"].items())},
            "pairs_where_both_persons_have_a_usable_image": merged["both_usable"],
        },
        "deferred": {"blocks": merged["deferred_blocks"], "impact": merged["deferred_impact"]},
        "image_resolution": (
            {"available": True, "source": refs.source_dir, "partial_run": refs.partial, "persons": len(refs.people)}
            if refs else {"available": False}
        ),
        "outputs": {"candidates": CANDIDATES_FILE, "deferred_blocks": DEFERRED_FILE, "directory": options.output_dir.name},
        "notes": [
            "Candidates are blocking output only: no scoring, no face comparison, no identity merging.",
            "Comparison keys (names, numbers) are not exported unless export_comparison_keys is set.",
        ],
    }
