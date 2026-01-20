# core/models/base_model.py
"""
Base class for all prediction models in NEXUS AI.
Adapted from backend_draft/core/base_predictor.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Sport(Enum):
    """Supported sports."""
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    HANDBALL = "handball"
    TABLE_TENNIS = "table_tennis"


@dataclass
class PredictionResult:
    """Result of a prediction."""
    sport: str
    predicted_winner: str  # "home" or "away" or player name
    confidence: float  # 0-1
    probabilities: Dict[str, float]  # {"home": 0.6, "away": 0.4}
    model_name: str
    features_used: List[str]
    feature_values: Dict[str, Any]
    reasoning: List[str] = field(default_factory=list)
    reliability_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sport": self.sport,
            "predicted_winner": self.predicted_winner,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model_name": self.model_name,
            "features_used": self.features_used,
            "reasoning": self.reasoning,
            "reliability_score": self.reliability_score,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ModelMetrics:
    """Training metrics for a model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    inference_time: float
    validation_samples: int = 0
    training_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "validation_samples": self.validation_samples,
            "training_samples": self.training_samples
        }


@dataclass
class BettingRecommendation:
    """Betting recommendation with stake calculation."""
    match_id: str
    bet_type: str  # "moneyline", "handicap", "total"
    selection: str  # "home", "away", "over", "under"
    odds: float
    probability: float
    edge: float
    kelly_stake: float
    adjusted_stake: float
    confidence: float
    quality_score: float
    reasoning: List[str]

    @property
    def expected_value(self) -> float:
        """Calculate expected value."""
        return (self.probability * self.odds) - 1


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Provides common functionality:
    - Input validation
    - Feature engineering interface
    - Model training/prediction interface
    - Betting recommendations
    - Metrics tracking
    """

    def __init__(self, sport: Sport, model_name: str):
        """Initialize base model."""
        self.sport = sport
        self.model_name = model_name
        self.model_version = "1.0.0"
        self.is_trained = False
        self.training_metrics: Optional[ModelMetrics] = None
        self.feature_names: List[str] = []
        self.required_features: List[str] = []

    @abstractmethod
    def predict(self, match_data: Dict[str, Any]) -> PredictionResult:
        """
        Make a prediction for a match.

        Args:
            match_data: Dictionary containing match features

        Returns:
            PredictionResult with prediction details
        """
        pass

    @abstractmethod
    def predict_proba(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Get probability distribution for outcomes.

        Args:
            match_data: Dictionary containing match features

        Returns:
            Dictionary mapping outcomes to probabilities
        """
        pass

    @abstractmethod
    def validate_input(self, match_data: Dict[str, Any]) -> bool:
        """
        Validate input data has required features.

        Args:
            match_data: Dictionary containing match features

        Returns:
            True if valid, raises ValueError otherwise
        """
        pass

    @abstractmethod
    def explain_prediction(self, match_data: Dict[str, Any],
                          prediction: PredictionResult) -> List[str]:
        """
        Generate human-readable explanation for prediction.

        Args:
            match_data: Input features
            prediction: The prediction result

        Returns:
            List of explanation strings
        """
        pass

    def get_required_features(self) -> List[str]:
        """Get list of required features for this model."""
        return self.required_features

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "sport": self.sport.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "required_features": self.required_features,
            "feature_count": len(self.feature_names),
            "training_metrics": self.training_metrics.to_dict() if self.training_metrics else None
        }

    def calculate_reliability_score(self, match_data: Dict[str, Any],
                                   prediction: PredictionResult) -> float:
        """
        Calculate reliability score based on data quality.

        Args:
            match_data: Input features
            prediction: The prediction result

        Returns:
            Reliability score 0-1
        """
        score = 0.0
        weights = {
            "has_ranking": 0.20,
            "has_recent_form": 0.25,
            "has_h2h": 0.15,
            "has_odds": 0.15,
            "high_confidence": 0.15,
            "data_completeness": 0.10
        }

        # Check data availability
        if match_data.get("home_ranking") and match_data.get("away_ranking"):
            score += weights["has_ranking"]

        if match_data.get("home_recent_form") and match_data.get("away_recent_form"):
            score += weights["has_recent_form"]

        if match_data.get("h2h_record"):
            score += weights["has_h2h"]

        if match_data.get("odds"):
            score += weights["has_odds"]

        # Confidence bonus
        if prediction.confidence > 0.65:
            score += weights["high_confidence"]

        # Data completeness
        available = sum(1 for f in self.required_features if f in match_data)
        completeness = available / len(self.required_features) if self.required_features else 0
        score += weights["data_completeness"] * completeness

        return min(1.0, score)

    def generate_betting_recommendation(
        self,
        match_data: Dict[str, Any],
        prediction: PredictionResult,
        quality_score: float,
        bankroll: float = 1000.0
    ) -> Optional[BettingRecommendation]:
        """
        Generate betting recommendation with Kelly stake.

        Args:
            match_data: Match features
            prediction: Prediction result
            quality_score: Data quality score 0-1
            bankroll: Total bankroll for stake calculation

        Returns:
            BettingRecommendation or None if no value found
        """
        # Get odds
        odds = match_data.get("odds", {})
        if not odds:
            return None

        # Determine selection
        selection = prediction.predicted_winner
        prob = prediction.probabilities.get(selection, 0.5)
        bet_odds = odds.get(selection, 2.0)

        # Calculate edge
        implied_prob = 1 / bet_odds
        edge = prob - implied_prob

        # Minimum edge threshold based on quality
        min_edge = 0.03 if quality_score >= 0.7 else 0.05

        if edge < min_edge:
            return None

        # Kelly Criterion
        kelly_fraction = (prob * bet_odds - 1) / (bet_odds - 1)
        kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%

        # Quality-adjusted stake
        quality_multiplier = self._get_quality_multiplier(quality_score)
        adjusted_fraction = kelly_fraction * quality_multiplier * 0.5  # Half-Kelly

        kelly_stake = bankroll * kelly_fraction
        adjusted_stake = bankroll * adjusted_fraction

        return BettingRecommendation(
            match_id=match_data.get("match_id", "unknown"),
            bet_type="moneyline",
            selection=selection,
            odds=bet_odds,
            probability=prob,
            edge=edge,
            kelly_stake=kelly_stake,
            adjusted_stake=adjusted_stake,
            confidence=prediction.confidence,
            quality_score=quality_score,
            reasoning=prediction.reasoning
        )

    def _get_quality_multiplier(self, quality_score: float) -> float:
        """Get stake multiplier based on quality score."""
        if quality_score >= 0.85:
            return 1.0
        elif quality_score >= 0.70:
            return 0.9
        elif quality_score >= 0.50:
            return 0.7
        elif quality_score >= 0.40:
            return 0.5
        else:
            return 0.3

    @staticmethod
    def normalize_probability(prob: float) -> float:
        """Normalize probability to valid range [0.01, 0.99]."""
        return max(0.01, min(0.99, prob))

    @staticmethod
    def elo_probability(rating_a: float, rating_b: float, k: float = 400) -> float:
        """
        Calculate win probability using Elo formula.

        Args:
            rating_a: Rating of player/team A
            rating_b: Rating of player/team B
            k: Scaling factor (default 400)

        Returns:
            Probability of A winning
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / k))
