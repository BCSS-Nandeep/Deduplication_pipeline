"""Blocks (groups of persons sharing a key), pair expansion in canonical order, and the source policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterator, Optional, Sequence

from .config import ALIAS_NAME, NAME_EXACT, PHONE_EXACT, TOKEN_SORTED, CandidateConfig
from .normalize import token_count


@dataclass(frozen=True)
class Member:
    person_id: str
    source_system: str
    source_record_id: str
    is_alias: bool = False  # alias strategy only: the person matched through their alias


@dataclass
class Block:
    strategy: str
    ordinal: int  # position in the strategy's deterministic (key-ordered) result
    key: str  # grouping key; kept in memory, exported only when export_comparison_keys is set
    members: list[Member] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def hint(self) -> dict[str, Any]:
        """Structure of the key, never its value: safe to write to the output."""
        if self.strategy == PHONE_EXACT:
            return {"digits": len(self.key)}
        return {"tokens": token_count(self.key)}


def parse_block_row(strategy: str, ordinal: int, row: Sequence[Any]) -> Block:
    key, ids, sources, records = row[0], row[1], row[2], row[3]
    aliases = row[4] if strategy == ALIAS_NAME else [False] * len(ids)
    members = [Member(i, s, r, bool(a)) for i, s, r, a in zip(ids, sources, records, aliases)]
    members.sort(key=lambda m: m.person_id)  # canonical order does not depend on the server's array order
    return Block(strategy, ordinal, key, members)


# ----------------------------------------------------------------------- source policy

def source_pair_label(source_a: str, source_b: str) -> str:
    low, high = sorted((source_a, source_b))
    return f"{low}-{high}"


def policy_allows(source_a: str, source_b: str, policy: str) -> bool:
    if policy == "cross_source_only":
        return source_a != source_b
    if policy == "same_source_only":
        return source_a == source_b
    return True


def pair_eligible(a: Member, b: Member, strategy: str, policy: str) -> bool:
    if strategy == ALIAS_NAME and not (a.is_alias or b.is_alias):
        return False  # two plain-name members already agree by name; the alias rule needs an alias on one side
    return policy_allows(a.source_system, b.source_system, policy)


def count_pairs(block: Block, policy: str) -> tuple[int, int, int]:
    """``(all member pairs, pairs the strategy itself accepts, pairs that also pass the source policy)``.

    The two filters are kept apart so reports never blame the source policy for what the strategy skipped:
    the alias rule only accepts pairs where someone matched through an alias.
    """
    n = block.size
    total = n * (n - 1) // 2
    if block.strategy == ALIAS_NAME:
        eligible = [(a, b) for a, b in combinations(block.members, 2) if a.is_alias or b.is_alias]
        return total, len(eligible), sum(1 for a, b in eligible if policy_allows(a.source_system, b.source_system, policy))
    by_source: dict[str, int] = {}
    for m in block.members:
        by_source[m.source_system] = by_source.get(m.source_system, 0) + 1
    same = sum(c * (c - 1) // 2 for c in by_source.values())
    if policy == "cross_source_only":
        return total, total, total - same
    if policy == "same_source_only":
        return total, total, same
    return total, total, total


def expand_block(block: Block, policy: str) -> Iterator[tuple[Member, Member]]:
    """Every eligible pair, ``(a, b)`` with ``a.person_id < b.person_id``, in a fixed order."""
    for a, b in combinations(block.members, 2):  # members are sorted, so a.person_id < b.person_id
        if pair_eligible(a, b, block.strategy, policy):
            yield a, b


def source_counts(block: Block) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in block.members:
        counts[m.source_system] = counts.get(m.source_system, 0) + 1
    return dict(sorted(counts.items()))


def candidate_id(person_a: str, person_b: str) -> str:
    """Stable id from the ordered pair itself (reversible, no hashing, no image input)."""
    return f"{person_a}:{person_b}"
