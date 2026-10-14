"""String similarity for names. Standard library only (difflib), as in the existing IR-54 matching helpers.

All inputs are already normalised with ``dedup_pipeline.candidates.normalize.normalize_name``. Results are in
[0, 1] with the method that produced them, so the reviewer can see *why* a name scored what it did.

Token alignment is the basis: whole-string character ratios reward coincidental shared letters ("ramesh kumar" vs
"suresh babu" scores 0.52 that way), so a token only counts when it matches another token well (equal, initial,
abbreviation or a close spelling). A whole-string comparison is accepted only for near-exact spacing variants.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

# Phase 9: these functions are pure (the same two normalised strings always give the same answer), and names repeat heavily
# across candidates ("kumar", "rao", "reddy", the same person in several pairs), so their results are memoised. A cache hit returns
# the very same float/tuple the computation would have produced; the caches are bounded and keyed by the exact arguments.
CACHE_SIZE = 131_072  # measured on the full run: 54k distinct token pairs (45% hits) and 30k distinct name pairs (46% hits)

INITIAL_MATCH = 0.9  # "R" vs "Ramesh"
ABBREVIATION_MATCH = 0.85  # "Mohd" / "Md" vs "Mohammed"; "Ram" vs "Ramesh"
FUZZY_FLOOR = 0.75  # token pairs below this similarity contribute nothing to the alignment
COMPACT_FLOOR = 0.90  # spacing variants ("sai kumar" vs "saikumar") must be nearly identical


def _ratio(a: str, b: str) -> float:  # not cached: measured hit rate 3%, so it would only cost memory
    return SequenceMatcher(None, a, b).ratio()


def _ratio_at_least(a: str, b: str, floor: float) -> float:
    """``_ratio(a, b)`` if it reaches ``floor``, else 0.0 - exactly what every caller does with a ratio below its floor.

    ``real_quick_ratio`` and ``quick_ratio`` are documented upper bounds on ``ratio`` (computed with the same 2*M/T division, M only
    larger), so a pair whose bound is already under the floor cannot reach it and the expensive matching is skipped. The result is
    identical to computing the ratio and then applying the floor."""
    matcher = SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < floor or matcher.quick_ratio() < floor:
        return 0.0
    r = matcher.ratio()
    return r if r >= floor else 0.0


def _is_subsequence(short: str, long: str) -> bool:
    remaining = iter(long)
    return all(ch in remaining for ch in short)


@lru_cache(maxsize=CACHE_SIZE)
def token_similarity(x: str, y: str) -> float:
    if x == y:
        return 1.0
    short, long = (x, y) if len(x) <= len(y) else (y, x)
    if len(short) == 1 and long.startswith(short):
        return INITIAL_MATCH
    if len(short) >= 2 and short[0] == long[0] and _is_subsequence(short, long):
        return ABBREVIATION_MATCH
    return _ratio_at_least(x, y, FUZZY_FLOOR)


def _alignment(ta: list[str], tb: list[str]) -> float:
    """Greedy one-to-one token alignment, so reordering and initials do not hurt; averaged over both lengths."""
    if not ta or not tb:
        return 0.0
    pairs = sorted(
        ((token_similarity(x, y), i, j) for i, x in enumerate(ta) for j, y in enumerate(tb)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    total = 0.0
    for sim, i, j in pairs:
        if sim <= 0.0:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        total += sim
    return total / ((len(ta) + len(tb)) / 2)


@lru_cache(maxsize=CACHE_SIZE)
def name_similarity(a: str, b: str) -> Optional[tuple[float, str]]:
    """Token alignment, or a near-exact spacing variant. ``None`` if either name is blank."""
    if not a or not b:
        return None
    if a == b:
        return 1.0, "identical"
    aligned = min(1.0, _alignment(a.split(), b.split()))
    compact = _ratio_at_least(a.replace(" ", ""), b.replace(" ", ""), COMPACT_FLOOR)  # below the floor it is never used (0.0 fails the test below)
    if compact >= COMPACT_FLOOR and compact > aligned:
        return round(compact, 4), "spacing variant"
    return round(aligned, 4), "token alignment"
