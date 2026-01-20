# evaluator/web_evaluator.py
"""
WebDataEvaluator for NEXUS AI.
Comprehensive evaluation of web-sourced data quality.

Combines:
- Source agreement check (35% weight)
- Freshness check with date parsing (30% weight)
- Cross-validation between sources (20% weight)
- Odds variance calculation (15% weight)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from evaluator.source_agreement import (
    SourceAgreementChecker,
    SourcePrediction,
    AgreementResult,
    AgreementLevel,
)
from evaluator.freshness_checker import (
    FreshnessChecker,
    FreshnessResult,
    FreshnessLevel,
)

logger = logging.getLogger(__name__)


class WebDataQuality(Enum):
    """Overall web data quality level."""
    EXCELLENT = "excellent"   # >= 0.85
    GOOD = "good"             # >= 0.70
    MODERATE = "moderate"     # >= 0.50
    POOR = "poor"             # >= 0.30
    INSUFFICIENT = "insufficient"  # < 0.30


@dataclass
class CrossValidationResult:
    """Result of cross-validation between sources."""
    is_valid: bool
    agreement_ratio: float  # 0-1
    conflicting_pairs: List[Dict[str, Any]]
    consistent_pairs: List[Dict[str, Any]]
    confidence_adjustment: float  # Multiplier for confidence


@dataclass
class OddsVarianceResult:
    """Result of odds variance calculation."""
    home_variance: float
    away_variance: float
    combined_variance: float
    variance_score: float  # 0-1, lower variance = higher score
    bookmaker_count: int
    suspicious_odds: List[Dict[str, Any]]


@dataclass
class WebDataEvaluationResult:
    """Complete web data evaluation result."""
    overall_score: float  # 0-1
    quality_level: WebDataQuality
    source_agreement: AgreementResult
    freshness: Dict[str, FreshnessResult]
    cross_validation: CrossValidationResult
    odds_variance: Optional[OddsVarianceResult]
    component_scores: Dict[str, float]
    recommendations: List[str]
    issues: List[str]
    timestamp: datetime


class WebDataEvaluator:
    """
    Comprehensive evaluator for web-sourced betting data.

    Evaluates:
    1. Source Agreement (35%) - Do different sources agree on predictions?
    2. Freshness (30%) - Is the data recent enough?
    3. Cross-validation (20%) - Do related data points validate each other?
    4. Odds Variance (15%) - Is there consensus among bookmakers?
    """

    # Component weights
    WEIGHTS = {
        "source_agreement": 0.35,
        "freshness": 0.30,
        "cross_validation": 0.20,
        "odds_variance": 0.15,
    }

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        WebDataQuality.EXCELLENT: 0.85,
        WebDataQuality.GOOD: 0.70,
        WebDataQuality.MODERATE: 0.50,
        WebDataQuality.POOR: 0.30,
    }

    # Cross-validation tolerance
    CROSS_VALIDATION_TOLERANCE = 0.15

    # Suspicious odds variance threshold
    MAX_ACCEPTABLE_VARIANCE = 0.08

    def __init__(
        self,
        agreement_checker: Optional[SourceAgreementChecker] = None,
        freshness_checker: Optional[FreshnessChecker] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the evaluator.

        Args:
            agreement_checker: Custom agreement checker
            freshness_checker: Custom freshness checker
            custom_weights: Override default component weights
        """
        self.agreement_checker = agreement_checker or SourceAgreementChecker()
        self.freshness_checker = freshness_checker or FreshnessChecker()

        self.weights = {**self.WEIGHTS}
        if custom_weights:
            # Normalize weights
            total = sum(custom_weights.values())
            self.weights = {k: v/total for k, v in custom_weights.items()}

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        odds_data: Optional[Dict[str, Dict[str, float]]] = None,
        data_timestamps: Optional[Dict[str, Any]] = None,
        match_context: Optional[Dict[str, Any]] = None
    ) -> WebDataEvaluationResult:
        """
        Perform comprehensive evaluation of web data.

        Args:
            predictions: List of predictions from various sources
                [{"source": "sofascore", "probability": 0.65, "confidence": 0.8, "timestamp": ...}]
            odds_data: Odds by bookmaker
                {"bet365": {"home": 1.85, "away": 2.10}, ...}
            data_timestamps: Timestamps for different data types
                {"news": "2024-01-15T10:00:00", "odds": "2024-01-15T11:30:00", ...}
            match_context: Optional context about the match

        Returns:
            WebDataEvaluationResult with complete evaluation
        """
        issues = []
        recommendations = []

        # 1. Source Agreement (35%)
        source_agreement = self._evaluate_source_agreement(predictions)
        agreement_score = source_agreement.agreement_score

        if source_agreement.agreement_level == AgreementLevel.NONE:
            issues.append("Sources significantly disagree on prediction")
            recommendations.append("Consider reducing stake due to source disagreement")

        # 2. Freshness (30%)
        freshness_results = self._evaluate_freshness(data_timestamps or {})
        freshness_score = self._calculate_freshness_score(freshness_results)

        stale_sources = [
            src for src, result in freshness_results.items()
            if not result.is_acceptable
        ]
        if stale_sources:
            issues.append(f"Stale data from: {', '.join(stale_sources)}")
            recommendations.append("Refresh data before betting")

        # 3. Cross-validation (20%)
        cross_validation = self._cross_validate(predictions, odds_data)
        cross_validation_score = cross_validation.agreement_ratio

        if not cross_validation.is_valid:
            issues.append("Cross-validation failed between sources")
            recommendations.append("Verify data integrity manually")

        # 4. Odds Variance (15%)
        if odds_data:
            odds_variance = self._calculate_odds_variance(odds_data)
            odds_score = odds_variance.variance_score

            if odds_variance.combined_variance > self.MAX_ACCEPTABLE_VARIANCE:
                issues.append(f"High odds variance ({odds_variance.combined_variance:.4f})")
                recommendations.append("Wait for odds to stabilize")

            if odds_variance.suspicious_odds:
                issues.append("Suspicious odds detected from some bookmakers")
        else:
            odds_variance = None
            odds_score = 0.5  # Neutral if no odds data

        # Calculate overall score
        component_scores = {
            "source_agreement": agreement_score,
            "freshness": freshness_score,
            "cross_validation": cross_validation_score,
            "odds_variance": odds_score,
        }

        overall_score = sum(
            score * self.weights[component]
            for component, score in component_scores.items()
        )

        # Determine quality level
        quality_level = self._get_quality_level(overall_score)

        # Add quality-based recommendations
        if quality_level == WebDataQuality.EXCELLENT:
            recommendations.append("High-quality data - proceed with confidence")
        elif quality_level == WebDataQuality.GOOD:
            recommendations.append("Good data quality - standard bet recommended")
        elif quality_level == WebDataQuality.MODERATE:
            recommendations.append("Moderate quality - consider reduced stake")
        elif quality_level == WebDataQuality.POOR:
            recommendations.append("Poor quality - minimal bet or skip")
        else:
            recommendations.append("Insufficient data - skip this match")

        return WebDataEvaluationResult(
            overall_score=round(overall_score, 4),
            quality_level=quality_level,
            source_agreement=source_agreement,
            freshness=freshness_results,
            cross_validation=cross_validation,
            odds_variance=odds_variance,
            component_scores=component_scores,
            recommendations=recommendations,
            issues=issues,
            timestamp=datetime.now(),
        )

    def _evaluate_source_agreement(
        self,
        predictions: List[Dict[str, Any]]
    ) -> AgreementResult:
        """Evaluate agreement between prediction sources."""
        preds = [
            SourcePrediction(
                source=p.get("source", "unknown"),
                probability=p.get("probability", 0.5),
                confidence=p.get("confidence", 0.5),
                timestamp=p.get("timestamp"),
            )
            for p in predictions
        ]

        return self.agreement_checker.check_agreement(preds)

    def _evaluate_freshness(
        self,
        timestamps: Dict[str, Any]
    ) -> Dict[str, FreshnessResult]:
        """Evaluate freshness for each data type."""
        results = {}

        # Default data types if not provided
        default_types = ["odds", "news", "stats"]
        types_to_check = list(timestamps.keys()) or default_types

        for data_type in types_to_check:
            ts = timestamps.get(data_type)
            if ts:
                results[data_type] = self.freshness_checker.check_freshness(
                    ts, data_type, data_type
                )
            else:
                # Mark as unknown/outdated if no timestamp
                results[data_type] = self.freshness_checker.check_freshness(
                    None, data_type, data_type
                )

        return results

    def _calculate_freshness_score(
        self,
        freshness_results: Dict[str, FreshnessResult]
    ) -> float:
        """Calculate aggregate freshness score."""
        if not freshness_results:
            return 0.0

        # Weight by data type importance
        importance = {
            "odds": 0.4,
            "news": 0.3,
            "stats": 0.2,
            "lineups": 0.1,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for data_type, result in freshness_results.items():
            weight = importance.get(data_type, 0.1)
            weighted_score += result.freshness_score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _cross_validate(
        self,
        predictions: List[Dict[str, Any]],
        odds_data: Optional[Dict[str, Dict[str, float]]]
    ) -> CrossValidationResult:
        """Cross-validate predictions against odds and between sources."""
        conflicting_pairs = []
        consistent_pairs = []

        # Validate predictions against each other
        for i, pred_a in enumerate(predictions):
            for pred_b in predictions[i+1:]:
                diff = abs(pred_a.get("probability", 0.5) - pred_b.get("probability", 0.5))

                pair_info = {
                    "source_a": pred_a.get("source"),
                    "source_b": pred_b.get("source"),
                    "prob_a": pred_a.get("probability"),
                    "prob_b": pred_b.get("probability"),
                    "difference": round(diff, 4),
                }

                if diff > self.CROSS_VALIDATION_TOLERANCE:
                    conflicting_pairs.append(pair_info)
                else:
                    consistent_pairs.append(pair_info)

        # Validate predictions against implied odds probability
        if odds_data and predictions:
            # Calculate average implied probability from odds
            implied_probs = []
            for bookmaker, odds in odds_data.items():
                home = odds.get("home", 2.0)
                away = odds.get("away", 2.0)
                total_implied = (1/home) + (1/away)
                implied_probs.append((1/home) / total_implied)

            if implied_probs:
                avg_implied = sum(implied_probs) / len(implied_probs)

                for pred in predictions:
                    pred_prob = pred.get("probability", 0.5)
                    diff = abs(pred_prob - avg_implied)

                    pair_info = {
                        "source_a": pred.get("source"),
                        "source_b": "odds_implied",
                        "prob_a": pred_prob,
                        "prob_b": round(avg_implied, 4),
                        "difference": round(diff, 4),
                    }

                    if diff > self.CROSS_VALIDATION_TOLERANCE * 1.5:  # Wider tolerance for odds
                        conflicting_pairs.append(pair_info)
                    else:
                        consistent_pairs.append(pair_info)

        # Calculate metrics
        total_pairs = len(conflicting_pairs) + len(consistent_pairs)
        agreement_ratio = len(consistent_pairs) / total_pairs if total_pairs > 0 else 1.0
        is_valid = agreement_ratio >= 0.6  # At least 60% agreement

        # Confidence adjustment based on agreement
        if agreement_ratio >= 0.9:
            confidence_adjustment = 1.0
        elif agreement_ratio >= 0.7:
            confidence_adjustment = 0.9
        elif agreement_ratio >= 0.5:
            confidence_adjustment = 0.7
        else:
            confidence_adjustment = 0.5

        return CrossValidationResult(
            is_valid=is_valid,
            agreement_ratio=round(agreement_ratio, 4),
            conflicting_pairs=conflicting_pairs,
            consistent_pairs=consistent_pairs,
            confidence_adjustment=confidence_adjustment,
        )

    def _calculate_odds_variance(
        self,
        odds_data: Dict[str, Dict[str, float]]
    ) -> OddsVarianceResult:
        """Calculate variance in odds across bookmakers."""
        if not odds_data:
            return OddsVarianceResult(
                home_variance=0.0,
                away_variance=0.0,
                combined_variance=0.0,
                variance_score=0.5,
                bookmaker_count=0,
                suspicious_odds=[],
            )

        home_odds = []
        away_odds = []
        suspicious = []

        for bookmaker, odds in odds_data.items():
            home = odds.get("home", 0)
            away = odds.get("away", 0)

            if home > 0:
                home_odds.append(home)
            if away > 0:
                away_odds.append(away)

        # Calculate variances
        def variance(values):
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)

        home_var = variance(home_odds) if home_odds else 0.0
        away_var = variance(away_odds) if away_odds else 0.0
        combined_var = (home_var + away_var) / 2

        # Detect suspicious odds (outliers)
        if len(home_odds) > 2:
            home_mean = sum(home_odds) / len(home_odds)
            home_std = (variance(home_odds) ** 0.5) if home_odds else 0

            for bookmaker, odds in odds_data.items():
                home = odds.get("home", 0)
                if home > 0 and home_std > 0:
                    z_score = abs(home - home_mean) / home_std
                    if z_score > 2:
                        suspicious.append({
                            "bookmaker": bookmaker,
                            "odds": home,
                            "type": "home",
                            "z_score": round(z_score, 2),
                        })

        # Variance score (lower variance = higher score)
        # Normalize assuming max acceptable variance of 0.1
        variance_score = max(0, 1 - (combined_var / 0.1))

        return OddsVarianceResult(
            home_variance=round(home_var, 6),
            away_variance=round(away_var, 6),
            combined_variance=round(combined_var, 6),
            variance_score=round(variance_score, 4),
            bookmaker_count=len(odds_data),
            suspicious_odds=suspicious,
        )

    def _get_quality_level(self, score: float) -> WebDataQuality:
        """Determine quality level from score."""
        for level, threshold in sorted(
            self.QUALITY_THRESHOLDS.items(),
            key=lambda x: -x[1]  # Descending order
        ):
            if score >= threshold:
                return level

        return WebDataQuality.INSUFFICIENT


def evaluate_web_data(
    predictions: List[Dict[str, Any]],
    odds_data: Optional[Dict[str, Dict[str, float]]] = None,
    data_timestamps: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate web data quality.

    Args:
        predictions: List of predictions from various sources
        odds_data: Odds by bookmaker
        data_timestamps: Timestamps for different data types

    Returns:
        Evaluation result as dict
    """
    evaluator = WebDataEvaluator()
    result = evaluator.evaluate(predictions, odds_data, data_timestamps)

    return {
        "overall_score": result.overall_score,
        "quality_level": result.quality_level.value,
        "component_scores": result.component_scores,
        "source_agreement": {
            "score": result.source_agreement.agreement_score,
            "level": result.source_agreement.agreement_level.value,
            "consensus_probability": result.source_agreement.consensus_probability,
        },
        "freshness_score": result.component_scores["freshness"],
        "cross_validation": {
            "is_valid": result.cross_validation.is_valid,
            "agreement_ratio": result.cross_validation.agreement_ratio,
            "conflicting_count": len(result.cross_validation.conflicting_pairs),
        },
        "odds_variance": {
            "score": result.odds_variance.variance_score if result.odds_variance else 0.5,
            "combined_variance": result.odds_variance.combined_variance if result.odds_variance else 0,
        } if result.odds_variance else None,
        "recommendations": result.recommendations,
        "issues": result.issues,
        "timestamp": result.timestamp.isoformat(),
    }
