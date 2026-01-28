"""
Prediction dataclasses for ML models.

Checkpoint: 2.1
Responsibility: Define structured prediction outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class PredictionResult:
    """Generic prediction result for any model."""
    home_win_prob: float = 0.33
    draw_prob: float = 0.34
    away_win_prob: float = 0.33
    confidence: float = 0.5
    model_version: str = ""
    reasoning: str = ""
    expected_goals: Optional[float] = None
    
    @property
    def predicted_outcome(self) -> str:
        """Get predicted outcome."""
        probs = {
            "home": self.home_win_prob,
            "draw": self.draw_prob,
            "away": self.away_win_prob,
        }
        return max(probs, key=probs.get)


@dataclass
class GoalsPrediction:
    """
    Predykcja liczby bramek (dla over/under).

    Używane przez model Poisson do predykcji total goals.
    """
    # Expected values
    home_expected: float = 1.3  # Lambda dla gospodarzy
    away_expected: float = 1.1  # Lambda dla gości
    total_expected: float = 2.4  # Suma expected goals

    # Over/under probabilities
    over_15_prob: float = 0.0
    under_15_prob: float = 0.0
    over_25_prob: float = 0.0
    under_25_prob: float = 0.0
    over_35_prob: float = 0.0
    under_35_prob: float = 0.0

    # Confidence and metadata
    confidence: float = 0.0  # 0-1, jak pewny jest model
    model_version: str = ""
    reasoning: str = ""  # Wyjaśnienie predykcji

    # Score probabilities (optional detailed breakdown)
    score_matrix: Optional[Dict[str, float]] = None  # "2-1": 0.08, etc.

    @property
    def recommended_bet(self) -> Optional[str]:
        """Sugerowany zakład jeśli confidence > 0.6."""
        if self.confidence < 0.6:
            return None

        # Find best over/under bet
        bets = [
            ("over_1.5", self.over_15_prob),
            ("under_1.5", self.under_15_prob),
            ("over_2.5", self.over_25_prob),
            ("under_2.5", self.under_25_prob),
            ("over_3.5", self.over_35_prob),
            ("under_3.5", self.under_35_prob),
        ]

        best = max(bets, key=lambda x: x[1])
        if best[1] > 0.55:  # Minimum edge
            return best[0]
        return None

    @property
    def btts_prob(self) -> float:
        """Both teams to score probability."""
        import math
        home_scores = 1 - math.exp(-self.home_expected)
        away_scores = 1 - math.exp(-self.away_expected)
        return home_scores * away_scores


@dataclass
class HandicapPrediction:
    """
    Predykcja handicap/spread.

    Używane przez model GBM do predykcji czy drużyna pokryje spread.
    """
    # Expected margin (home perspective)
    expected_margin: float = 0.0  # Positive = home wins by this

    # Cover probabilities for common lines
    home_cover_minus_15: float = 0.0  # Home -1.5
    home_cover_minus_05: float = 0.0  # Home -0.5 (home win)
    home_cover_plus_05: float = 0.0   # Home +0.5 (home not lose)
    home_cover_plus_15: float = 0.0   # Home +1.5

    # Away covers (inverse)
    away_cover_minus_15: float = 0.0
    away_cover_plus_15: float = 0.0

    # 1X2 probabilities
    home_win_prob: float = 0.33
    draw_prob: float = 0.34
    away_win_prob: float = 0.33

    # Confidence and metadata
    confidence: float = 0.0
    model_version: str = ""
    reasoning: str = ""

    @property
    def recommended_bet(self) -> Optional[str]:
        """Sugerowany zakład handicap."""
        if self.confidence < 0.6:
            return None

        bets = [
            ("home_-1.5", self.home_cover_minus_15),
            ("home_+1.5", self.home_cover_plus_15),
            ("away_-1.5", self.away_cover_minus_15),
            ("away_+1.5", self.away_cover_plus_15),
        ]

        best = max(bets, key=lambda x: x[1])
        if best[1] > 0.55:
            return best[0]
        return None

    @property
    def predicted_winner(self) -> str:
        """Przewidywany zwycięzca."""
        if self.home_win_prob > self.away_win_prob and self.home_win_prob > self.draw_prob:
            return "home"
        elif self.away_win_prob > self.home_win_prob and self.away_win_prob > self.draw_prob:
            return "away"
        return "draw"


@dataclass
class CombinedPrediction:
    """
    Połączona predykcja z wielu modeli.
    """
    match_id: str = ""
    goals: GoalsPrediction = field(default_factory=GoalsPrediction)
    handicap: HandicapPrediction = field(default_factory=HandicapPrediction)

    # Overall confidence (weighted average)
    overall_confidence: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_quality_score: float = 0.0
    models_used: List[str] = field(default_factory=list)

    def get_best_bets(self, min_confidence: float = 0.6) -> List[Dict]:
        """Zwróć najlepsze zakłady z wszystkich modeli."""
        bets = []

        if self.goals.confidence >= min_confidence:
            goals_bet = self.goals.recommended_bet
            if goals_bet:
                bets.append({
                    "market": "goals",
                    "bet": goals_bet,
                    "probability": getattr(self.goals, f"{goals_bet.replace('.', '')}_prob", 0),
                    "confidence": self.goals.confidence,
                })

        if self.handicap.confidence >= min_confidence:
            hcap_bet = self.handicap.recommended_bet
            if hcap_bet:
                bets.append({
                    "market": "handicap",
                    "bet": hcap_bet,
                    "confidence": self.handicap.confidence,
                })

        return sorted(bets, key=lambda x: x["confidence"], reverse=True)


@dataclass
class ModelInfo:
    """Informacje o modelu ML."""
    name: str = ""
    version: str = ""
    trained_at: datetime = field(default_factory=datetime.utcnow)
    training_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    @property
    def is_trained(self) -> bool:
        return self.training_samples > 0
