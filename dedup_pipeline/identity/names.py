"""One name-normalisation and comparison framework for accused names and father/relative names.

The two fields are never joined into one signal. Cleaning removes a token only when its position shows it is
metadata: a leading ``late``, a trailing verified role such as ``peddler``, or a trailing recognised country
when at least two other tokens remain. A country or role token in any other position is kept, because it may
be part of a real name.

Initials never receive the full name score. Token-order variants of the same tokens do.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from ..candidates.normalize import name_tokens

NAME_NORM_VERSION = "name-norm-v1"
PREFIX_TOKENS = frozenset({"late"})
ROLE_TOKENS = frozenset({"peddler"})
# Single-token country names and a few abbreviations that show up as appended metadata.
# Applied only as a trailing token, and only when at least two other tokens remain.
COUNTRY_TOKENS = frozenset({
    "afghanistan", "algeria", "america", "angola", "australia", "bangladesh", "botswana", "brazil",
    "britain", "burundi", "cameroon", "canada", "china", "congo", "egypt", "england", "ethiopia",
    "france", "gabon", "germany", "ghana", "india", "indonesia", "iran", "iraq", "italy", "japan",
    "kenya", "korea", "liberia", "libya", "madagascar", "malawi", "malaysia", "mauritius", "mexico",
    "morocco", "mozambique", "myanmar", "namibia", "nepal", "nigeria", "pakistan", "philippines",
    "russia", "rwanda", "senegal", "singapore", "somalia", "spain", "srilanka", "sudan", "syria",
    "tanzania", "thailand", "tunisia", "turkey", "uae", "uganda", "uk", "usa", "yemen", "zambia",
    "zimbabwe",
})

INITIAL_TOKEN_SCORE = 0.55  # a one-letter token against a longer token that starts with it
INITIAL_NAME_CAP = 0.75  # a name that matched only through an initial cannot score as an exact name
ABBREVIATION_SCORE = 0.80  # "mohd" inside "mohammed"; length at least 3, so a single initial is not this
SPELLING_FLOOR = 0.88
_STRIP_PASSES = 6


@dataclass(frozen=True)
class NormalizedName:
    original: str
    display: str
    comparison: str
    tokens: tuple[str, ...]
    initial_signature: str
    status: str  # blank | ok
    removed: tuple[tuple[str, str], ...]  # (term, reason)
    rule_version: str = NAME_NORM_VERSION

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "display": self.display,
            "comparison": self.comparison,
            "tokens": list(self.tokens),
            "initial_signature": self.initial_signature,
            "status": self.status,
            "removed": [{"term": term, "reason": reason} for term, reason in self.removed],
            "rule_version": self.rule_version,
        }


def _strip_once(tokens: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], Optional[tuple[str, str]]]:
    """Remove at most one metadata token. ``tokens`` are ``(display, key)``."""
    if len(tokens) > 1 and tokens[0][1] in PREFIX_TOKENS:
        return tokens[1:], (tokens[0][0], "prefix_metadata")
    if len(tokens) >= 2 and tokens[-1][1] in ROLE_TOKENS:
        return tokens[:-1], (tokens[-1][0], "role_metadata")
    if len(tokens) >= 3 and tokens[-1][1] in COUNTRY_TOKENS:
        return tokens[:-1], (tokens[-1][0], "country_metadata")
    return tokens, None


def normalize_identity_name(raw: Optional[str]) -> NormalizedName:
    """Deterministic and idempotent. The original value is never overwritten; it is returned beside the cleaned form."""
    original = "" if raw is None else str(raw)
    pairs = [(token, token.lower()) for token in name_tokens(original)]
    removed: list[tuple[str, str]] = []
    for _ in range(_STRIP_PASSES):
        pairs, gone = _strip_once(pairs)
        if gone is None:
            break
        removed.append(gone)
    keys = tuple(key for _, key in pairs)
    display = " ".join(token for token, _ in pairs)
    return NormalizedName(
        original=original,
        display=display,
        comparison=" ".join(keys),
        tokens=keys,
        initial_signature="".join(key[0] for key in keys if key),
        status="ok" if keys else "blank",
        removed=tuple(removed),
    )


@dataclass(frozen=True)
class NameComparison:
    available: bool
    score: Optional[float]
    status: str  # matching | partial | conflicting | unavailable
    method: str
    reason: str
    used_initial: bool = False


def _token_score(left: str, right: str) -> tuple[float, str]:
    if left == right:
        return 1.0, "exact"
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) == 1 and len(long) >= 2 and long.startswith(short):
        return INITIAL_TOKEN_SCORE, "initial"
    if len(short) >= 3 and short[0] == long[0] and _is_subsequence(short, long):
        return ABBREVIATION_SCORE, "abbreviation"
    ratio = SequenceMatcher(None, left, right).ratio()
    if ratio >= SPELLING_FLOOR:
        return round(ratio, 4), "spelling"
    return 0.0, "none"


def _is_subsequence(short: str, long: str) -> bool:
    remaining = iter(long)
    return all(ch in remaining for ch in short)


def _align(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[float, bool]:
    pairs = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score, method = _token_score(a, b)
            if score > 0:
                pairs.append((score, method, i, j))
    pairs.sort(key=lambda item: (-item[0], item[2], item[3]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    total = 0.0
    used_initial = False
    for score, method, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        total += score
        used_initial = used_initial or method == "initial"
    if not left or not right:
        return 0.0, False
    aligned = total / ((len(left) + len(right)) / 2)
    return min(1.0, aligned), used_initial


def compare_identity_names(left: NormalizedName, right: NormalizedName) -> NameComparison:
    """Explicit rules. An initial match is capped below a full-name score. Blank input is unavailable, not a zero."""
    if left.status == "blank" or right.status == "blank":
        return NameComparison(False, None, "unavailable", "missing", "A name is missing on at least one record.")
    if left.tokens == right.tokens:
        return NameComparison(True, 1.0, "matching", "exact", "Exact cleaned-name match.")
    if sorted(left.tokens) == sorted(right.tokens):
        return NameComparison(True, 1.0, "matching", "token_order", "Same name tokens in a different order.")
    score, used_initial = _align(left.tokens, right.tokens)
    if used_initial and score > INITIAL_NAME_CAP:
        score = INITIAL_NAME_CAP
    score = round(score, 4)
    if score <= 0:
        return NameComparison(True, 0.0, "conflicting", "none", "The names do not share a comparable token.", used_initial)
    if used_initial:
        return NameComparison(
            True, score, "partial", "initial",
            "Compatible initial. An initial does not receive the full name score and does not establish identity.",
            True,
        )
    if score >= SPELLING_FLOOR:
        return NameComparison(True, score, "matching", "spelling_or_alignment", "Name tokens agree closely, including spelling variation.")
    return NameComparison(
        True, score, "partial", "partial",
        "Partial name agreement only. This is not an exact or token-order match.",
        False,
    )


def initial_index_keys(comparison: str, min_anchor: int) -> list[tuple[str, bool]]:
    """Blocking keys for compatible initials.

    A key is ``sorted full tokens | initial letters``. A record that already contains a one-letter token is the
    only side allowed to open a pair (the caller checks that flag). Fully spelled names are indexed by reducing
    one long token to its first letter, so ``Darla Ramesh`` meets ``D Ramesh`` and does not meet ``Dinesh Ramesh``
    through this rule. A block still needs one token of at least ``min_anchor`` characters, so a bare initial
    never becomes a block.
    """
    tokens = comparison.split() if comparison else []
    if len(tokens) < 2:
        return []
    full = [token for token in tokens if len(token) >= 2]
    if not full or not any(len(token) >= min_anchor for token in full):
        return []
    initials = [token for token in tokens if len(token) == 1]
    if initials:
        stem = " ".join(sorted(full))
        signature = "".join(sorted(initials))
        return [(f"{stem}|{signature}", True)] if stem and signature else []
    keys: list[tuple[str, bool]] = []
    for index, token in enumerate(tokens):
        if len(token) < min_anchor:
            continue
        rest = [other for position, other in enumerate(tokens) if position != index and len(other) >= 2]
        if not rest:
            continue
        keys.append((" ".join(sorted(rest)) + "|" + token[0], False))
    return keys
