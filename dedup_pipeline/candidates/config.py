"""Candidate-generation settings. Every threshold, cap and normalisation rule is configurable.

Defaults come from the measured data (Phase 1 audit plus the Phase 5 inspection of ``dopams_cctns_v2``):
single-token names produce three quarters of exact-name pairs and almost all of it is noise; the
largest phone "blocks" are junk numbers such as repeated digits; all real name blocks are small.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

SOURCE_POLICIES = ("all", "cross_source_only", "same_source_only")
NAME_SOURCES = ("cleaned_then_raw", "raw")

NAME_EXACT = "name_exact"
TOKEN_SORTED = "token_sorted_name"
NAME_INITIAL = "name_initial"
RELATIVE_EXACT = "relative_exact"
RELATIVE_SORTED = "relative_token_sorted"
PHONE_EXACT = "phone_exact"
ALIAS_NAME = "alias_name"
# Relative order of the original four rules is unchanged, so existing rule-name lists stay stable.
ALL_STRATEGIES = (NAME_EXACT, TOKEN_SORTED, NAME_INITIAL, RELATIVE_EXACT, RELATIVE_SORTED, PHONE_EXACT, ALIAS_NAME)


@dataclass(frozen=True)
class CandidateConfig:
    # --- which source-system pairs are kept. "all" keeps V1-V1, V1-V2 and V2-V2 and reports each.
    source_policy: str = "all"

    # --- name normalisation (shared by the name strategies)
    name_source: str = "cleaned_then_raw"  # person_enrichment.cleaned_full_name, else persons_unified.full_name
    name_min_chars: int = 3

    # --- strategy 1: exact normalised name
    name_exact_enabled: bool = True
    name_min_tokens: int = 2  # single-token names ("Kumar") are excluded and reported
    name_max_block_size: int = 30

    # --- strategy 2: token-sorted name ("Sai Shiva Alle" == "Alle Sai Shiva")
    token_sorted_enabled: bool = True
    token_sorted_min_tokens: int = 2
    token_sorted_max_block_size: int = 30

    # --- strategy 3: phone agreement (last N digits of a plausible number)
    phone_enabled: bool = True
    phone_digits: int = 10
    phone_max_input_digits: int = 13  # 10 + country/trunk prefix; longer strings may hold several numbers
    phone_leading_digits: str = "6789"  # valid Indian mobile numbers start with 6-9
    phone_min_distinct_digits: int = 3  # drops 9999999999, 9898989898, ...
    phone_blocklist: tuple[str, ...] = ("1234567890", "0123456789", "9876543210")
    phone_max_block_size: int = 20  # a number shared by more people than this is a shared line, not a duplicate

    # --- strategy 4: alias of one person equals the name (or alias) of another.
    # Kept for historical runs. The final profile turns it off: alias is not a candidate signal there.
    alias_enabled: bool = True

    # --- final-profile signals. Off unless selected, so existing runs stay reproducible.
    metadata_name_cleaning: bool = False  # strip leading "late", trailing role/country before blocking
    name_initial_enabled: bool = False
    name_initial_max_block_size: int = 15  # smaller than name blocks: a shared surname plus one initial is common
    name_initial_min_anchor_chars: int = 4  # a block needs one token at least this long; a bare initial is not a block
    relative_exact_enabled: bool = False
    relative_token_sorted_enabled: bool = False
    relative_min_tokens: int = 2
    relative_max_block_size: int = 30
    alias_min_tokens: int = 2
    alias_max_block_size: int = 30

    # --- safety, batching, output
    max_total_pairs: int = 5_000_000  # exceeding it FAILS the run; pairs are never silently dropped
    checkpoint_every_blocks: int = 500
    stream_batch_size: int = 2000
    deferred_listing_cap: int = 1000  # person ids listed per deferred block in the output (flagged if cut)
    impact_pair_limit: int = 2_000_000  # deferred-block impact is computed only up to this many pairs
    export_comparison_keys: bool = False  # True writes normalised names/phones into the output (sensitive)

    def __post_init__(self) -> None:
        if self.source_policy not in SOURCE_POLICIES:
            raise ValueError(f"source_policy must be one of {SOURCE_POLICIES}")
        if self.name_source not in NAME_SOURCES:
            raise ValueError(f"name_source must be one of {NAME_SOURCES}")
        for name in ("name_max_block_size", "token_sorted_max_block_size", "phone_max_block_size", "alias_max_block_size",
                     "name_initial_max_block_size", "relative_max_block_size"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.name_initial_min_anchor_chars < 2:
            raise ValueError("name_initial_min_anchor_chars must be at least 2")
        for name in ("name_min_tokens", "token_sorted_min_tokens", "alias_min_tokens", "name_min_chars", "phone_min_distinct_digits"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not (1 <= self.phone_digits <= self.phone_max_input_digits):
            raise ValueError("phone_digits must be between 1 and phone_max_input_digits")
        if not self.phone_leading_digits or not self.phone_leading_digits.isdigit():
            raise ValueError("phone_leading_digits must be a non-empty string of digits")
        if self.max_total_pairs < 1 or self.checkpoint_every_blocks < 1 or self.stream_batch_size < 1:
            raise ValueError("max_total_pairs, checkpoint_every_blocks and stream_batch_size must be positive")

    def enabled_strategies(self) -> list[str]:
        flags = {NAME_EXACT: self.name_exact_enabled, TOKEN_SORTED: self.token_sorted_enabled,
                 NAME_INITIAL: self.name_initial_enabled, RELATIVE_EXACT: self.relative_exact_enabled,
                 RELATIVE_SORTED: self.relative_token_sorted_enabled,
                 PHONE_EXACT: self.phone_enabled, ALIAS_NAME: self.alias_enabled}
        return [s for s in ALL_STRATEGIES if flags[s]]

    def max_block_size(self, strategy: str) -> int:
        return {NAME_EXACT: self.name_max_block_size, TOKEN_SORTED: self.token_sorted_max_block_size,
                NAME_INITIAL: self.name_initial_max_block_size, RELATIVE_EXACT: self.relative_max_block_size,
                RELATIVE_SORTED: self.relative_max_block_size,
                PHONE_EXACT: self.phone_max_block_size, ALIAS_NAME: self.alias_max_block_size}[strategy]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phone_blocklist"] = list(self.phone_blocklist)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        values = dict(data)
        if "phone_blocklist" in values:
            values["phone_blocklist"] = tuple(values["phone_blocklist"])
        return cls(**values)

    def signature(self) -> dict[str, Any]:
        """What a checkpoint must match to be reused. Output-only knobs are excluded."""
        data = self.to_dict()
        for name in ("checkpoint_every_blocks", "stream_batch_size", "deferred_listing_cap", "impact_pair_limit"):
            data.pop(name)
        return data


def final_candidate_config(**overrides: Any) -> CandidateConfig:
    """Candidate settings for the final identity profile.

    Alias, age and date of birth are not signals. Relative name, token order, compatible initials and phone are.
    Age and date of birth were never blocking keys.
    """
    values: dict[str, Any] = {
        "alias_enabled": False,
        "metadata_name_cleaning": True,
        "name_initial_enabled": True,
        "relative_exact_enabled": True,
        "relative_token_sorted_enabled": True,
    }
    values.update(overrides)
    return CandidateConfig.from_dict(values)
