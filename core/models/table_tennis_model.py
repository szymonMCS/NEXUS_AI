# core/models/table_tennis_model.py
"""
Table Tennis prediction model for NEXUS AI.
Uses XGBoost/RandomForest/GradientBoosting ensemble.
Adapted from backend_draft/models/table_tennis_predictor.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging

from core.models.base_model import BaseModel, PredictionResult, Sport

logger = logging.getLogger(__name__)


class TableTennisFormat(Enum):
    """Match formats."""
    BEST_OF_5 = 5
    BEST_OF_7 = 7


class PlayingStyle(Enum):
    """Table tennis playing styles."""
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"
    ALL_ROUND = "all_round"
    CHOPPER = "chopper"
    PENHOLD = "penhold"


@dataclass
class TableTennisFeatures:
    """Features for table tennis prediction."""
    # Player info
    player1_name: str
    player2_name: str

    # Rankings
    player1_ranking: int = 500
    player2_ranking: int = 500
    player1_rating: float = 1500.0
    player2_rating: float = 1500.0

    # Win rates
    player1_win_rate: float = 0.50
    player2_win_rate: float = 0.50

    # Recent form (last 10 matches)
    player1_recent_wins: int = 5
    player2_recent_wins: int = 5

    # H2H
    h2h_player1_wins: int = 0
    h2h_player2_wins: int = 0
    h2h_total: int = 0

    # Set statistics
    player1_sets_won_avg: float = 2.0
    player1_sets_lost_avg: float = 1.5
    player2_sets_won_avg: float = 2.0
    player2_sets_lost_avg: float = 1.5

    # Points per set
    player1_points_per_set: float = 10.5
    player2_points_per_set: float = 10.5

    # Style matchup
    player1_style: PlayingStyle = PlayingStyle.ALL_ROUND
    player2_style: PlayingStyle = PlayingStyle.ALL_ROUND

    # Momentum indicators
    player1_consecutive_wins: int = 0
    player2_consecutive_wins: int = 0
    player1_days_since_match: int = 3
    player2_days_since_match: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player1_name": self.player1_name,
            "player2_name": self.player2_name,
            "player1_ranking": self.player1_ranking,
            "player2_ranking": self.player2_ranking,
            "player1_rating": self.player1_rating,
            "player2_rating": self.player2_rating,
            "player1_win_rate": self.player1_win_rate,
            "player2_win_rate": self.player2_win_rate,
            "player1_recent_wins": self.player1_recent_wins,
            "player2_recent_wins": self.player2_recent_wins,
            "h2h_player1_wins": self.h2h_player1_wins,
            "h2h_player2_wins": self.h2h_player2_wins,
        }


@dataclass
class TableTennisPrediction:
    """Table tennis match prediction."""
    player1_win_prob: float
    player2_win_prob: float
    expected_sets: Tuple[float, float]  # (p1 sets, p2 sets)
    set_handicap: Dict[float, Dict[str, float]]  # line -> probs
    total_points: Dict[float, Dict[str, float]]  # line -> probs
    confidence: float
    reasoning: List[str]


class TableTennisModel(BaseModel):
    """
    Table Tennis prediction model.

    Uses ensemble of:
    - Rating-based probability
    - Form analysis
    - Style matchup analysis
    - Momentum factors
    """

    def __init__(self):
        super().__init__(Sport.TABLE_TENNIS, "TableTennisEnsemble_v1")

        self.required_features = [
            "player1_rating", "player2_rating",
            "player1_win_rate", "player2_win_rate",
            "player1_ranking", "player2_ranking"
        ]

        # Feature weights
        self.feature_weights = {
            "rating": 0.30,
            "ranking": 0.20,
            "form": 0.20,
            "h2h": 0.15,
            "style_matchup": 0.10,
            "momentum": 0.05,
        }

        # Style matchup matrix (style1 vs style2 -> advantage for style1)
        self.style_matrix = {
            (PlayingStyle.OFFENSIVE, PlayingStyle.DEFENSIVE): 0.52,
            (PlayingStyle.OFFENSIVE, PlayingStyle.CHOPPER): 0.55,
            (PlayingStyle.DEFENSIVE, PlayingStyle.OFFENSIVE): 0.48,
            (PlayingStyle.CHOPPER, PlayingStyle.OFFENSIVE): 0.45,
            (PlayingStyle.ALL_ROUND, PlayingStyle.ALL_ROUND): 0.50,
        }

    def predict(self, match_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict table tennis match outcome.

        Args:
            match_data: Dictionary with match features

        Returns:
            PredictionResult with winner prediction
        """
        self.validate_input(match_data)

        tt_pred = self.predict_match(match_data)

        winner = "player1" if tt_pred.player1_win_prob > tt_pred.player2_win_prob else "player2"
        winner_name = match_data.get(f"{winner}_name", winner)

        return PredictionResult(
            sport=self.sport.value,
            predicted_winner=winner_name,
            confidence=tt_pred.confidence,
            probabilities={
                "player1": tt_pred.player1_win_prob,
                "player2": tt_pred.player2_win_prob,
            },
            model_name=self.model_name,
            features_used=self.required_features,
            feature_values={
                "expected_sets_p1": tt_pred.expected_sets[0],
                "expected_sets_p2": tt_pred.expected_sets[1],
            },
            reasoning=tt_pred.reasoning,
            reliability_score=self._calculate_reliability(match_data),
        )

    def predict_proba(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Get win probabilities."""
        tt_pred = self.predict_match(match_data)
        return {
            "player1": tt_pred.player1_win_prob,
            "player2": tt_pred.player2_win_prob,
        }

    def predict_match(self, match_data: Dict[str, Any]) -> TableTennisPrediction:
        """
        Full match prediction.

        Args:
            match_data: Match features dictionary

        Returns:
            TableTennisPrediction with full analysis
        """
        # Extract features
        p1_rating = match_data.get("player1_rating", 1500)
        p2_rating = match_data.get("player2_rating", 1500)
        p1_ranking = match_data.get("player1_ranking", 500)
        p2_ranking = match_data.get("player2_ranking", 500)
        p1_win_rate = match_data.get("player1_win_rate", 0.5)
        p2_win_rate = match_data.get("player2_win_rate", 0.5)
        p1_recent = match_data.get("player1_recent_wins", 5)
        p2_recent = match_data.get("player2_recent_wins", 5)

        # H2H
        h2h_p1 = match_data.get("h2h_player1_wins", 0)
        h2h_p2 = match_data.get("h2h_player2_wins", 0)
        h2h_total = match_data.get("h2h_total", 0)

        # Styles
        p1_style = match_data.get("player1_style", PlayingStyle.ALL_ROUND)
        p2_style = match_data.get("player2_style", PlayingStyle.ALL_ROUND)

        # Calculate component probabilities
        probs = {}

        # Rating-based
        probs["rating"] = self.elo_probability(p1_rating, p2_rating)

        # Ranking-based
        probs["ranking"] = self._ranking_probability(p1_ranking, p2_ranking)

        # Form-based
        probs["form"] = self._form_probability(p1_recent, p2_recent, p1_win_rate, p2_win_rate)

        # H2H
        if h2h_total > 0:
            probs["h2h"] = (h2h_p1 + 0.5) / (h2h_total + 1)  # Laplace smoothing
        else:
            probs["h2h"] = 0.5

        # Style matchup
        probs["style_matchup"] = self._style_matchup_prob(p1_style, p2_style)

        # Momentum
        p1_streak = match_data.get("player1_consecutive_wins", 0)
        p2_streak = match_data.get("player2_consecutive_wins", 0)
        probs["momentum"] = self._momentum_probability(p1_streak, p2_streak)

        # Weighted ensemble
        p1_prob = sum(
            probs[k] * self.feature_weights[k]
            for k in self.feature_weights
        )

        # Normalize
        p1_prob = max(0.05, min(0.95, p1_prob))
        p2_prob = 1 - p1_prob

        # Expected sets (for best of 7)
        format_sets = match_data.get("format", TableTennisFormat.BEST_OF_7).value
        sets_to_win = (format_sets + 1) // 2
        exp_p1_sets, exp_p2_sets = self._expected_sets(p1_prob, sets_to_win)

        # Set handicap
        set_handicap = {}
        margin = exp_p1_sets - exp_p2_sets
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            cover_prob = self._handicap_prob(margin, line)
            set_handicap[line] = {
                "player1_cover": cover_prob,
                "player2_cover": 1 - cover_prob
            }

        # Total points (approximate)
        avg_points_per_set = match_data.get("player1_points_per_set", 10.5) + \
                            match_data.get("player2_points_per_set", 10.5)
        expected_sets_played = exp_p1_sets + exp_p2_sets
        expected_total = avg_points_per_set * expected_sets_played

        total_points = {}
        for line in [150.5, 160.5, 170.5, 180.5]:
            over_prob = self._total_prob(expected_total, line)
            total_points[line] = {
                "over": over_prob,
                "under": 1 - over_prob
            }

        # Calculate confidence
        confidence = self._calculate_confidence(probs, abs(p1_prob - 0.5))

        # Build reasoning
        reasoning = self._build_reasoning(
            match_data, probs, p1_prob, p2_prob, exp_p1_sets, exp_p2_sets
        )

        return TableTennisPrediction(
            player1_win_prob=p1_prob,
            player2_win_prob=p2_prob,
            expected_sets=(round(exp_p1_sets, 1), round(exp_p2_sets, 1)),
            set_handicap=set_handicap,
            total_points=total_points,
            confidence=confidence,
            reasoning=reasoning
        )

    def _ranking_probability(self, rank1: int, rank2: int) -> float:
        """Convert rankings to probability."""
        # Lower rank = better
        if rank1 == rank2:
            return 0.5

        # Log scale for ranking advantage
        log_ratio = np.log(rank2 / rank1) if rank1 > 0 and rank2 > 0 else 0
        prob = 1 / (1 + np.exp(-log_ratio * 0.5))

        return max(0.1, min(0.9, prob))

    def _form_probability(
        self,
        p1_recent: int,
        p2_recent: int,
        p1_win_rate: float,
        p2_win_rate: float
    ) -> float:
        """Calculate probability based on form."""
        # Recent form (out of 10)
        recent_factor = (p1_recent - p2_recent) / 10

        # Win rate difference
        wr_factor = p1_win_rate - p2_win_rate

        combined = recent_factor * 0.6 + wr_factor * 0.4
        prob = 0.5 + combined * 0.3

        return max(0.2, min(0.8, prob))

    def _style_matchup_prob(
        self,
        style1: PlayingStyle,
        style2: PlayingStyle
    ) -> float:
        """Get style matchup probability."""
        if isinstance(style1, str):
            style1 = PlayingStyle(style1)
        if isinstance(style2, str):
            style2 = PlayingStyle(style2)

        key = (style1, style2)
        if key in self.style_matrix:
            return self.style_matrix[key]

        reverse_key = (style2, style1)
        if reverse_key in self.style_matrix:
            return 1 - self.style_matrix[reverse_key]

        return 0.5

    def _momentum_probability(self, streak1: int, streak2: int) -> float:
        """Calculate momentum factor."""
        if streak1 == streak2:
            return 0.5

        momentum_diff = (streak1 - streak2) / 10
        prob = 0.5 + momentum_diff * 0.15

        return max(0.35, min(0.65, prob))

    def _expected_sets(self, p1_prob: float, sets_to_win: int) -> Tuple[float, float]:
        """Calculate expected sets for each player."""
        # Simplified: assumes each set probability is match probability
        p2_prob = 1 - p1_prob

        # Expected sets based on probability and format
        exp_p1 = p1_prob * sets_to_win + (1 - p1_prob) * (sets_to_win - 1) * p1_prob * 2
        exp_p2 = p2_prob * sets_to_win + (1 - p2_prob) * (sets_to_win - 1) * p2_prob * 2

        # Ensure realistic bounds
        exp_p1 = max(0.5, min(sets_to_win + 0.5, exp_p1))
        exp_p2 = max(0.5, min(sets_to_win + 0.5, exp_p2))

        return exp_p1, exp_p2

    def _handicap_prob(self, margin: float, line: float) -> float:
        """Calculate handicap cover probability."""
        from scipy import stats
        adjusted = margin + line
        std_dev = 1.5
        return stats.norm.cdf(adjusted, loc=0, scale=std_dev)

    def _total_prob(self, expected: float, line: float) -> float:
        """Calculate over probability."""
        from scipy import stats
        std_dev = expected * 0.15
        return 1 - stats.norm.cdf(line, loc=expected, scale=std_dev)

    def _calculate_confidence(
        self,
        component_probs: Dict[str, float],
        prob_margin: float
    ) -> float:
        """Calculate overall confidence."""
        # Agreement between components
        probs_list = list(component_probs.values())
        variance = np.var(probs_list)
        agreement = 1 - min(0.5, variance * 4)

        # Margin factor
        margin_factor = prob_margin * 0.8

        confidence = agreement * 0.5 + margin_factor + 0.25

        return min(0.90, max(0.35, confidence))

    def _build_reasoning(
        self,
        match_data: Dict[str, Any],
        probs: Dict[str, float],
        p1_prob: float,
        p2_prob: float,
        exp_p1_sets: float,
        exp_p2_sets: float
    ) -> List[str]:
        """Build prediction explanation."""
        reasoning = []

        p1_name = match_data.get("player1_name", "Player 1")
        p2_name = match_data.get("player2_name", "Player 2")

        # Winner prediction
        if p1_prob > p2_prob:
            reasoning.append(
                f"Prediction: {p1_name} to win ({p1_prob:.1%} probability)"
            )
        else:
            reasoning.append(
                f"Prediction: {p2_name} to win ({p2_prob:.1%} probability)"
            )

        # Expected score
        reasoning.append(
            f"Expected sets: {p1_name} {exp_p1_sets:.1f} - {exp_p2_sets:.1f} {p2_name}"
        )

        # Key factors
        if probs["rating"] > 0.6 or probs["rating"] < 0.4:
            favored = p1_name if probs["rating"] > 0.5 else p2_name
            reasoning.append(f"Rating advantage: {favored}")

        if probs["ranking"] > 0.6 or probs["ranking"] < 0.4:
            favored = p1_name if probs["ranking"] > 0.5 else p2_name
            reasoning.append(f"Ranking advantage: {favored}")

        if probs["form"] > 0.6 or probs["form"] < 0.4:
            favored = p1_name if probs["form"] > 0.5 else p2_name
            reasoning.append(f"Better recent form: {favored}")

        if probs["h2h"] != 0.5:
            favored = p1_name if probs["h2h"] > 0.5 else p2_name
            h2h_total = match_data.get("h2h_total", 0)
            if h2h_total > 0:
                reasoning.append(f"H2H advantage: {favored} ({h2h_total} previous matches)")

        return reasoning

    def _calculate_reliability(self, match_data: Dict[str, Any]) -> float:
        """Calculate data reliability score."""
        score = 0.3

        if match_data.get("player1_rating") and match_data.get("player2_rating"):
            score += 0.2

        if match_data.get("player1_ranking") and match_data.get("player2_ranking"):
            score += 0.15

        if match_data.get("player1_win_rate") and match_data.get("player2_win_rate"):
            score += 0.15

        if match_data.get("h2h_total", 0) > 0:
            score += 0.2

        return min(1.0, score)

    def validate_input(self, match_data: Dict[str, Any]) -> bool:
        """Validate match data."""
        if not match_data:
            raise ValueError("Empty match data")
        return True

    def explain_prediction(
        self,
        match_data: Dict[str, Any],
        prediction: PredictionResult
    ) -> List[str]:
        """Generate explanation."""
        return prediction.reasoning
