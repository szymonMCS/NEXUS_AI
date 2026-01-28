"""
ML Prediction Results.

Checkpoint: 4.1
Responsibility: Result dataclasses for ML prediction service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

from core.data.schemas import DataQuality


class PredictionStatus(Enum):
    """Status of a prediction request."""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    MODEL_ERROR = "model_error"
    VALIDATION_ERROR = "validation_error"
    NO_VALUE = "no_value"  # Model found no betting value


@dataclass
class BettingRecommendation:
    """
    Single betting recommendation.
    """
    market: str  # e.g., "over_2.5", "handicap_-1.5", "1x2_home"
    selection: str  # e.g., "over", "home", "away"
    probability: float  # Model's estimated probability
    odds_required: float  # Minimum odds needed for value
    confidence: float  # 0-1, how confident the model is
    edge: Optional[float] = None  # If odds provided, the edge (prob - implied_prob)
    kelly_fraction: Optional[float] = None  # Recommended stake fraction
    reasoning: str = ""  # Brief explanation

    @property
    def has_value(self) -> bool:
        """Check if this recommendation offers value."""
        return self.edge is not None and self.edge > 0

    @property
    def confidence_level(self) -> str:
        """Human-readable confidence level."""
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        return "low"


@dataclass
class GoalsPredictionResult:
    """
    Result of goals/over-under prediction.
    """
    home_expected: float
    away_expected: float
    total_expected: float

    # Over/under probabilities
    over_15_prob: float = 0.0
    over_25_prob: float = 0.0
    over_35_prob: float = 0.0
    under_15_prob: float = 0.0
    under_25_prob: float = 0.0
    under_35_prob: float = 0.0

    # Both teams to score
    btts_yes_prob: float = 0.0
    btts_no_prob: float = 0.0

    # Score probabilities (top N most likely)
    score_probabilities: Dict[str, float] = field(default_factory=dict)

    # Model metadata
    model_version: str = ""
    confidence: float = 0.0

    def get_over_under_prob(self, line: float) -> tuple:
        """Get over/under probability for a given line."""
        if line == 1.5:
            return (self.over_15_prob, self.under_15_prob)
        elif line == 2.5:
            return (self.over_25_prob, self.under_25_prob)
        elif line == 3.5:
            return (self.over_35_prob, self.under_35_prob)
        return (0.0, 0.0)

    def most_likely_score(self, top_n: int = 3) -> List[tuple]:
        """Get most likely scores."""
        sorted_scores = sorted(
            self.score_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_scores[:top_n]


@dataclass
class HandicapPredictionResult:
    """
    Result of handicap/spread prediction.
    """
    expected_margin: float  # Home - Away

    # 1X2 probabilities
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0

    # Asian handicap probabilities
    home_minus_05_prob: float = 0.0  # Home -0.5 (same as home win)
    home_minus_15_prob: float = 0.0  # Home -1.5
    home_minus_25_prob: float = 0.0  # Home -2.5
    away_plus_05_prob: float = 0.0   # Away +0.5 (same as draw or away win)
    away_plus_15_prob: float = 0.0   # Away +1.5
    away_plus_25_prob: float = 0.0   # Away +2.5

    # Model metadata
    model_version: str = ""
    confidence: float = 0.0

    @property
    def predicted_winner(self) -> str:
        """Get predicted match winner."""
        if self.home_win_prob > self.away_win_prob and self.home_win_prob > self.draw_prob:
            return "home"
        elif self.away_win_prob > self.home_win_prob and self.away_win_prob > self.draw_prob:
            return "away"
        return "draw"

    def get_handicap_prob(self, line: float, side: str) -> float:
        """Get handicap probability for a given line and side."""
        if side == "home":
            if line == -0.5:
                return self.home_minus_05_prob
            elif line == -1.5:
                return self.home_minus_15_prob
            elif line == -2.5:
                return self.home_minus_25_prob
        elif side == "away":
            if line == 0.5:
                return self.away_plus_05_prob
            elif line == 1.5:
                return self.away_plus_15_prob
            elif line == 2.5:
                return self.away_plus_25_prob
        return 0.0


@dataclass
class MLPredictionResult:
    """
    Complete ML prediction result for a match.

    Combines goals and handicap predictions with betting recommendations.
    """
    # Identification
    match_id: str
    prediction_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Status
    status: PredictionStatus = PredictionStatus.SUCCESS
    error_message: Optional[str] = None

    # Predictions
    goals_prediction: Optional[GoalsPredictionResult] = None
    handicap_prediction: Optional[HandicapPredictionResult] = None

    # Recommendations
    recommendations: List[BettingRecommendation] = field(default_factory=list)

    # Data quality
    data_quality: Optional[DataQuality] = None
    features_used: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if prediction was successful."""
        return self.status == PredictionStatus.SUCCESS

    @property
    def has_recommendations(self) -> bool:
        """Check if there are any betting recommendations."""
        return len(self.recommendations) > 0

    @property
    def has_value_bets(self) -> bool:
        """Check if any recommendations have positive edge."""
        return any(r.has_value for r in self.recommendations)

    def get_top_recommendations(self, n: int = 3) -> List[BettingRecommendation]:
        """Get top N recommendations by confidence."""
        sorted_recs = sorted(
            self.recommendations,
            key=lambda r: (r.edge or 0, r.confidence),
            reverse=True
        )
        return sorted_recs[:n]

    def get_recommendations_by_market(self, market: str) -> List[BettingRecommendation]:
        """Get recommendations for a specific market."""
        return [r for r in self.recommendations if r.market == market]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "match_id": self.match_id,
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
        }

        if self.error_message:
            result["error_message"] = self.error_message

        if self.goals_prediction:
            result["goals"] = {
                "home_expected": round(self.goals_prediction.home_expected, 2),
                "away_expected": round(self.goals_prediction.away_expected, 2),
                "total_expected": round(self.goals_prediction.total_expected, 2),
                "over_25_prob": round(self.goals_prediction.over_25_prob, 3),
                "under_25_prob": round(self.goals_prediction.under_25_prob, 3),
                "btts_yes_prob": round(self.goals_prediction.btts_yes_prob, 3),
                "confidence": round(self.goals_prediction.confidence, 2),
                "model_version": self.goals_prediction.model_version,
            }

        if self.handicap_prediction:
            result["handicap"] = {
                "expected_margin": round(self.handicap_prediction.expected_margin, 2),
                "home_win_prob": round(self.handicap_prediction.home_win_prob, 3),
                "draw_prob": round(self.handicap_prediction.draw_prob, 3),
                "away_win_prob": round(self.handicap_prediction.away_win_prob, 3),
                "predicted_winner": self.handicap_prediction.predicted_winner,
                "confidence": round(self.handicap_prediction.confidence, 2),
                "model_version": self.handicap_prediction.model_version,
            }

        if self.recommendations:
            result["recommendations"] = [
                {
                    "market": r.market,
                    "selection": r.selection,
                    "probability": round(r.probability, 3),
                    "odds_required": round(r.odds_required, 2),
                    "confidence": r.confidence_level,
                    "edge": round(r.edge, 3) if r.edge else None,
                    "reasoning": r.reasoning,
                }
                for r in self.recommendations
            ]

        if self.data_quality:
            result["data_quality"] = {
                "completeness": self.data_quality.completeness,
                "freshness": self.data_quality.freshness,
                "sources_count": self.data_quality.sources_count,
            }

        return result


@dataclass
class BatchPredictionResult:
    """
    Result of batch prediction for multiple matches.
    """
    predictions: List[MLPredictionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_processing_time_ms: float = 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for p in self.predictions if p.is_success)

    @property
    def failure_count(self) -> int:
        return len(self.predictions) - self.success_count

    @property
    def value_bet_count(self) -> int:
        return sum(1 for p in self.predictions if p.has_value_bets)

    def get_all_recommendations(self) -> List[tuple]:
        """Get all recommendations with match info."""
        all_recs = []
        for pred in self.predictions:
            for rec in pred.recommendations:
                all_recs.append((pred.match_id, rec))
        return all_recs

    def get_top_value_bets(self, n: int = 10) -> List[tuple]:
        """Get top N value bets across all matches."""
        all_recs = self.get_all_recommendations()
        value_recs = [(m, r) for m, r in all_recs if r.has_value]
        sorted_recs = sorted(
            value_recs,
            key=lambda x: (x[1].edge or 0),
            reverse=True
        )
        return sorted_recs[:n]
