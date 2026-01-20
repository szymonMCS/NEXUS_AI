# evaluator/source_agreement.py
"""
Source agreement evaluation for NEXUS AI.
Checks agreement between multiple data sources for cross-validation.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class AgreementLevel(Enum):
    """Level of agreement between sources."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class SourcePrediction:
    """Prediction from a single source."""
    source: str
    probability: float  # 0-1
    confidence: float  # 0-1
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgreementResult:
    """Result of source agreement analysis."""
    agreement_score: float  # 0-1, higher = better agreement
    agreement_level: AgreementLevel
    consensus_probability: float
    consensus_confidence: float
    variance: float
    std_deviation: float
    source_count: int
    outliers: List[str]  # Sources that deviate significantly
    description: str
    weights: Dict[str, float]  # Weight assigned to each source


class SourceAgreementChecker:
    """
    Evaluates agreement between multiple prediction sources.

    Uses variance and standard deviation to determine consensus level.
    Identifies outliers and calculates weighted consensus.
    """

    # Thresholds for agreement levels
    STRONG_THRESHOLD = 0.05  # std_dev < 5%
    MODERATE_THRESHOLD = 0.10  # std_dev < 10%
    WEAK_THRESHOLD = 0.15  # std_dev < 15%

    # Outlier threshold (deviates more than 2 std_dev from mean)
    OUTLIER_THRESHOLD = 2.0

    # Default source weights (can be customized)
    DEFAULT_WEIGHTS = {
        "sofascore": 1.0,
        "flashscore": 0.95,
        "odds_implied": 0.9,
        "tennis_model": 0.85,
        "basketball_model": 0.85,
        "historical_h2h": 0.8,
        "ranking_based": 0.75,
        "news_sentiment": 0.6,
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the checker.

        Args:
            custom_weights: Optional custom source weights
        """
        self.weights = {**self.DEFAULT_WEIGHTS}
        if custom_weights:
            self.weights.update(custom_weights)

    def check_agreement(
        self,
        predictions: List[SourcePrediction]
    ) -> AgreementResult:
        """
        Check agreement between multiple source predictions.

        Args:
            predictions: List of predictions from different sources

        Returns:
            AgreementResult with analysis
        """
        if not predictions:
            return self._empty_result()

        if len(predictions) == 1:
            return self._single_source_result(predictions[0])

        # Extract probabilities
        probs = [p.probability for p in predictions]
        confidences = [p.confidence for p in predictions]
        sources = [p.source for p in predictions]

        # Calculate statistics
        mean_prob = statistics.mean(probs)
        variance = statistics.variance(probs) if len(probs) > 1 else 0
        std_dev = statistics.stdev(probs) if len(probs) > 1 else 0

        mean_confidence = statistics.mean(confidences)

        # Determine agreement level
        agreement_level, description = self._get_agreement_level(std_dev)

        # Calculate agreement score (inverse of normalized variance)
        max_variance = 0.25  # Maximum theoretical variance for 0-1 range
        agreement_score = max(0, 1 - (variance / max_variance))

        # Identify outliers
        outliers = self._identify_outliers(predictions, mean_prob, std_dev)

        # Calculate weighted consensus (excluding outliers)
        valid_predictions = [p for p in predictions if p.source not in outliers]
        if not valid_predictions:
            valid_predictions = predictions  # Fall back to all

        consensus_prob, consensus_conf, weights = self._weighted_consensus(
            valid_predictions
        )

        return AgreementResult(
            agreement_score=round(agreement_score, 4),
            agreement_level=agreement_level,
            consensus_probability=round(consensus_prob, 4),
            consensus_confidence=round(consensus_conf, 4),
            variance=round(variance, 6),
            std_deviation=round(std_dev, 4),
            source_count=len(predictions),
            outliers=outliers,
            description=description,
            weights=weights,
        )

    def check_odds_agreement(
        self,
        odds_by_bookmaker: Dict[str, Dict[str, float]]
    ) -> Tuple[float, float, List[str]]:
        """
        Check agreement between bookmaker odds.

        Args:
            odds_by_bookmaker: Dict of bookmaker -> {home: odds, away: odds}

        Returns:
            Tuple of (agreement_score, variance, outliers)
        """
        if not odds_by_bookmaker:
            return 0.0, 0.0, []

        # Convert odds to implied probabilities
        predictions = []
        for bookmaker, odds in odds_by_bookmaker.items():
            home_odds = odds.get("home", 2.0)
            away_odds = odds.get("away", 2.0)

            # Implied probability (ignoring margin for simplicity)
            total_implied = (1/home_odds) + (1/away_odds)
            home_prob = (1/home_odds) / total_implied

            predictions.append(SourcePrediction(
                source=f"odds_{bookmaker}",
                probability=home_prob,
                confidence=0.8,  # Odds-based confidence
            ))

        result = self.check_agreement(predictions)
        return result.agreement_score, result.variance, result.outliers

    def cross_validate_sources(
        self,
        source_a: SourcePrediction,
        source_b: SourcePrediction,
        tolerance: float = 0.10
    ) -> Dict[str, Any]:
        """
        Cross-validate two specific sources.

        Args:
            source_a: First source prediction
            source_b: Second source prediction
            tolerance: Maximum allowed difference

        Returns:
            Validation result with agreement details
        """
        diff = abs(source_a.probability - source_b.probability)
        agrees = diff <= tolerance

        # Weighted average based on confidence
        total_conf = source_a.confidence + source_b.confidence
        if total_conf > 0:
            weighted_avg = (
                source_a.probability * source_a.confidence +
                source_b.probability * source_b.confidence
            ) / total_conf
        else:
            weighted_avg = (source_a.probability + source_b.probability) / 2

        return {
            "sources": [source_a.source, source_b.source],
            "probabilities": [source_a.probability, source_b.probability],
            "difference": round(diff, 4),
            "tolerance": tolerance,
            "agrees": agrees,
            "weighted_average": round(weighted_avg, 4),
            "confidence_weighted": True,
        }

    def _get_agreement_level(
        self,
        std_dev: float
    ) -> Tuple[AgreementLevel, str]:
        """Determine agreement level based on standard deviation."""
        if std_dev < self.STRONG_THRESHOLD:
            return AgreementLevel.STRONG, "Sources strongly agree"
        elif std_dev < self.MODERATE_THRESHOLD:
            return AgreementLevel.MODERATE, "Sources moderately agree"
        elif std_dev < self.WEAK_THRESHOLD:
            return AgreementLevel.WEAK, "Sources weakly agree"
        else:
            return AgreementLevel.NONE, "Sources disagree significantly"

    def _identify_outliers(
        self,
        predictions: List[SourcePrediction],
        mean: float,
        std_dev: float
    ) -> List[str]:
        """Identify sources that are outliers."""
        if std_dev == 0:
            return []

        outliers = []
        for pred in predictions:
            z_score = abs(pred.probability - mean) / std_dev
            if z_score > self.OUTLIER_THRESHOLD:
                outliers.append(pred.source)
                logger.debug(
                    f"Outlier detected: {pred.source} "
                    f"(prob={pred.probability:.2%}, z={z_score:.2f})"
                )

        return outliers

    def _weighted_consensus(
        self,
        predictions: List[SourcePrediction]
    ) -> Tuple[float, float, Dict[str, float]]:
        """Calculate weighted consensus probability and confidence."""
        weighted_prob_sum = 0.0
        weighted_conf_sum = 0.0
        total_weight = 0.0
        weights_used = {}

        for pred in predictions:
            # Get weight for source (default 0.5 if unknown)
            base_weight = self.weights.get(pred.source.lower(), 0.5)

            # Adjust by confidence
            adjusted_weight = base_weight * pred.confidence

            weighted_prob_sum += pred.probability * adjusted_weight
            weighted_conf_sum += pred.confidence * adjusted_weight
            total_weight += adjusted_weight
            weights_used[pred.source] = round(adjusted_weight, 4)

        if total_weight == 0:
            return 0.5, 0.5, weights_used

        consensus_prob = weighted_prob_sum / total_weight
        consensus_conf = weighted_conf_sum / total_weight

        return consensus_prob, consensus_conf, weights_used

    def _empty_result(self) -> AgreementResult:
        """Return empty result for no predictions."""
        return AgreementResult(
            agreement_score=0.0,
            agreement_level=AgreementLevel.NONE,
            consensus_probability=0.5,
            consensus_confidence=0.0,
            variance=0.0,
            std_deviation=0.0,
            source_count=0,
            outliers=[],
            description="No predictions available",
            weights={},
        )

    def _single_source_result(self, pred: SourcePrediction) -> AgreementResult:
        """Return result for single source (no agreement possible)."""
        weight = self.weights.get(pred.source.lower(), 0.5)
        return AgreementResult(
            agreement_score=0.5,  # Neutral - can't measure agreement
            agreement_level=AgreementLevel.NONE,
            consensus_probability=pred.probability,
            consensus_confidence=pred.confidence * 0.7,  # Reduced for single source
            variance=0.0,
            std_deviation=0.0,
            source_count=1,
            outliers=[],
            description="Single source - agreement not measurable",
            weights={pred.source: weight},
        )


def check_source_agreement(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to check source agreement.

    Args:
        predictions: List of dicts with source, probability, confidence

    Returns:
        Agreement analysis as dict
    """
    checker = SourceAgreementChecker()

    preds = [
        SourcePrediction(
            source=p.get("source", "unknown"),
            probability=p.get("probability", 0.5),
            confidence=p.get("confidence", 0.5),
            timestamp=p.get("timestamp"),
        )
        for p in predictions
    ]

    result = checker.check_agreement(preds)

    return {
        "agreement_score": result.agreement_score,
        "agreement_level": result.agreement_level.value,
        "consensus_probability": result.consensus_probability,
        "consensus_confidence": result.consensus_confidence,
        "variance": result.variance,
        "std_deviation": result.std_deviation,
        "source_count": result.source_count,
        "outliers": result.outliers,
        "description": result.description,
        "weights": result.weights,
    }
