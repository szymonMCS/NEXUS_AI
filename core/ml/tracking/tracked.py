"""
Tracked prediction dataclass.

Checkpoint: 3.7
Responsibility: Track predictions for accuracy measurement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class PredictionMarket(Enum):
    """Types of betting markets."""
    OVER_UNDER_25 = "over_under_2.5"
    OVER_UNDER_15 = "over_under_1.5"
    OVER_UNDER_35 = "over_under_3.5"
    HANDICAP = "handicap"
    MATCH_WINNER = "1x2"
    BTTS = "btts"


class PredictionOutcome(Enum):
    """Outcome of a tracked prediction."""
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    VOID = "void"  # Match cancelled, etc.


@dataclass
class TrackedPrediction:
    """
    Pojedyncza śledzona predykcja.

    Łączy predykcję z rzeczywistym wynikiem dla pomiaru dokładności.
    """
    # Identification
    prediction_id: str
    match_id: str
    market: PredictionMarket
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Prediction details
    predicted_value: float = 0.0  # e.g., 0.65 for over 2.5 probability
    predicted_outcome: str = ""  # e.g., "over", "home", etc.
    confidence: float = 0.0
    model_version: str = ""

    # Odds (if available)
    odds_at_prediction: Optional[float] = None
    stake: float = 0.0  # For ROI calculation

    # Actual result
    actual_value: Optional[float] = None  # e.g., actual total goals
    actual_outcome: Optional[str] = None

    # Evaluation
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    profit_loss: float = 0.0  # For ROI

    # Metadata
    extra_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Check if prediction has been resolved."""
        return self.outcome != PredictionOutcome.PENDING

    @property
    def is_correct(self) -> bool:
        return self.outcome == PredictionOutcome.CORRECT

    @property
    def was_confident(self) -> bool:
        """Was this a high-confidence prediction."""
        return self.confidence >= 0.6

    @property
    def roi(self) -> Optional[float]:
        """Return on investment if stake was placed."""
        if self.stake <= 0:
            return None
        return self.profit_loss / self.stake

    def resolve(
        self,
        actual_value: float,
        actual_outcome: str,
    ) -> PredictionOutcome:
        """
        Resolve the prediction with actual results.

        Args:
            actual_value: Actual numeric result
            actual_outcome: Actual outcome string

        Returns:
            PredictionOutcome
        """
        self.actual_value = actual_value
        self.actual_outcome = actual_outcome

        # Determine if correct
        if self.predicted_outcome.lower() == actual_outcome.lower():
            self.outcome = PredictionOutcome.CORRECT

            # Calculate profit if odds available
            if self.odds_at_prediction and self.stake > 0:
                self.profit_loss = self.stake * (self.odds_at_prediction - 1)
        else:
            self.outcome = PredictionOutcome.INCORRECT
            self.profit_loss = -self.stake

        return self.outcome

    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "match_id": self.match_id,
            "market": self.market.value,
            "timestamp": self.timestamp.isoformat(),
            "predicted_value": self.predicted_value,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "odds_at_prediction": self.odds_at_prediction,
            "stake": self.stake,
            "actual_value": self.actual_value,
            "actual_outcome": self.actual_outcome,
            "outcome": self.outcome.value,
            "profit_loss": self.profit_loss,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrackedPrediction":
        return cls(
            prediction_id=data["prediction_id"],
            match_id=data["match_id"],
            market=PredictionMarket(data["market"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            predicted_value=data.get("predicted_value", 0.0),
            predicted_outcome=data.get("predicted_outcome", ""),
            confidence=data.get("confidence", 0.0),
            model_version=data.get("model_version", ""),
            odds_at_prediction=data.get("odds_at_prediction"),
            stake=data.get("stake", 0.0),
            actual_value=data.get("actual_value"),
            actual_outcome=data.get("actual_outcome"),
            outcome=PredictionOutcome(data.get("outcome", "pending")),
            profit_loss=data.get("profit_loss", 0.0),
            extra_data=data.get("extra_data", {}),
        )


@dataclass
class PredictionSummary:
    """Summary statistics for a set of predictions."""
    total_predictions: int = 0
    resolved_predictions: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0

    total_stake: float = 0.0
    total_profit_loss: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.resolved_predictions == 0:
            return 0.0
        return self.correct_predictions / self.resolved_predictions

    @property
    def roi(self) -> float:
        if self.total_stake == 0:
            return 0.0
        return self.total_profit_loss / self.total_stake

    @property
    def pending_count(self) -> int:
        return self.total_predictions - self.resolved_predictions
