"""Reference implementations of the normalisation rules.

The database does the grouping (``sql.py``); these Python versions define the same rules in one readable
place, are used for output hints, and are what the tests (and the live equivalence test) compare the SQL against.
"""

from __future__ import annotations

import unicodedata
from typing import Optional

from .config import CandidateConfig


def _is_name_char(ch: str) -> bool:
    # Combining marks (e.g. Telugu or Devanagari vowel signs) belong to the word; the database's
    # [[:alnum:]] class keeps them too, and splitting there would shred names written in Indic scripts.
    return ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M")


def normalize_name(raw: Optional[str]) -> str:
    """Lower-case; anything that is not a letter, digit, combining mark or whitespace becomes a space; whitespace collapsed."""
    if not raw:
        return ""
    cleaned = "".join(ch if _is_name_char(ch) else " " for ch in raw)
    return " ".join(cleaned.lower().split())


def token_count(key: str) -> int:
    return len(key.split()) if key else 0


def token_sorted_key(key: str) -> str:
    """Tokens in code-point order, the same as ``ORDER BY t COLLATE "C"`` on UTF-8 text."""
    return " ".join(sorted(key.split()))


def name_exclusion_reason(key: str, min_tokens: int, min_chars: int) -> Optional[str]:
    if not key:
        return "blank_name"
    if token_count(key) < min_tokens:
        return "too_few_tokens"
    if len(key) < min_chars:
        return "too_short"
    return None


def pick_name(cleaned: Optional[str], raw: Optional[str], config: CandidateConfig) -> Optional[str]:
    if config.name_source == "cleaned_then_raw" and cleaned and cleaned.strip():
        return cleaned
    return raw


def phone_key(raw: Optional[str], config: CandidateConfig) -> tuple[Optional[str], Optional[str]]:
    """Return ``(key, None)`` for a usable number, else ``(None, reason)``."""
    if raw is None or not raw.strip():
        return None, "no_phone"
    digits = "".join(ch for ch in raw if ch in "0123456789")
    if not digits:
        return None, "no_digits"
    if len(digits) < config.phone_digits:
        return None, "too_few_digits"
    if len(digits) > config.phone_max_input_digits:
        return None, "too_many_digits"
    key = digits[-config.phone_digits:]
    if key[0] not in config.phone_leading_digits:
        return None, "bad_leading_digit"
    if len(set(key)) < config.phone_min_distinct_digits:
        return None, "repetitive_digits"
    if key in config.phone_blocklist:
        return None, "blocklisted"
    return key, None
