"""Phase 5: candidate generation by blocking (read-only database access, local-file output, no scoring)."""

from .config import CandidateConfig
from .runner import PairBudgetExceeded, PreflightError, RunOptions, run_candidate_generation

__all__ = ["CandidateConfig", "PairBudgetExceeded", "PreflightError", "RunOptions", "run_candidate_generation"]
