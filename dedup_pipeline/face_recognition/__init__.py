"""Phase 3/4: client for the FR-DOPAMS API and per-pair face evidence (no scoring, no database access)."""

from .client import CompareResult, FrDopamsClient, SearchHit, SearchResult
from .config import FrConfig
from .errors import (
    FrAuthError,
    FrConfigError,
    FrContractError,
    FrError,
    FrImageRejected,
    FrInvalidResponse,
    FrOperationalError,
    FrPairRejected,
    FrServerError,
    FrServiceNotReady,
    FrUnavailable,
    FrWritePolicyError,
    SideRejection,
)
from .face_checker import FrDopamsFaceChecker
from .local_comparer import LocalFacePairComparer  # imports no machine-learning package until the engine is first used
from .local_config import LocalFaceConfig
from .pair_evidence import (
    ComparisonOutcome,
    FallbackPolicy,
    FrDopamsPairComparer,
    PairComparer,
    PairFaceEvidence,
    PairStatus,
    compare_pair,
)

__all__ = [
    "CompareResult", "ComparisonOutcome", "FallbackPolicy", "FrAuthError", "FrConfig", "FrConfigError",
    "FrContractError", "FrDopamsClient", "FrDopamsFaceChecker", "FrDopamsPairComparer", "FrError", "FrImageRejected",
    "FrInvalidResponse", "FrOperationalError", "FrPairRejected", "FrServerError", "FrServiceNotReady",
    "FrUnavailable", "FrWritePolicyError", "LocalFaceConfig", "LocalFacePairComparer", "PairComparer", "PairFaceEvidence", "PairStatus", "SearchHit",
    "SearchResult", "SideRejection", "compare_pair",
]
