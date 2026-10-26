"""Shared identity-name normalisation used by candidate generation and scoring."""

from .names import (
    NAME_NORM_VERSION,
    NameComparison,
    NormalizedName,
    compare_identity_names,
    initial_index_keys,
    normalize_identity_name,
)

__all__ = [
    "NAME_NORM_VERSION",
    "NameComparison",
    "NormalizedName",
    "compare_identity_names",
    "initial_index_keys",
    "normalize_identity_name",
]
