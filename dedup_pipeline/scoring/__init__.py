"""Phase 6: explainable match scoring and the human-review package (read-only; never decides, never merges)."""

from .config import DECISIONS, RULE_VERSION, ScoringConfig
from .inputs import InputError
from .runner import FaceBackend, ScoringOptions, run_scoring

__all__ = ["DECISIONS", "FaceBackend", "InputError", "RULE_VERSION", "ScoringConfig", "ScoringOptions", "run_scoring"]
