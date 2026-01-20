# evaluator/__init__.py
"""
Data evaluation module for NEXUS AI.

Provides:
- Source agreement checking between prediction sources
- Data freshness evaluation with date parsing
- Cross-validation between data sources
- Odds variance calculation
- Comprehensive web data evaluation
"""

from evaluator.source_agreement import (
    SourceAgreementChecker,
    SourcePrediction,
    AgreementResult,
    AgreementLevel,
    check_source_agreement,
)

from evaluator.freshness_checker import (
    FreshnessChecker,
    FreshnessResult,
    FreshnessLevel,
    check_data_freshness,
)

from evaluator.web_evaluator import (
    WebDataEvaluator,
    WebDataEvaluationResult,
    WebDataQuality,
    CrossValidationResult,
    OddsVarianceResult,
    evaluate_web_data,
)

__all__ = [
    # Source Agreement
    "SourceAgreementChecker",
    "SourcePrediction",
    "AgreementResult",
    "AgreementLevel",
    "check_source_agreement",
    # Freshness
    "FreshnessChecker",
    "FreshnessResult",
    "FreshnessLevel",
    "check_data_freshness",
    # Web Evaluator
    "WebDataEvaluator",
    "WebDataEvaluationResult",
    "WebDataQuality",
    "CrossValidationResult",
    "OddsVarianceResult",
    "evaluate_web_data",
]
