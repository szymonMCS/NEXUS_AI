# core/models/handball_model.py
"""
Handball prediction model for NEXUS AI.
Uses SEL (Statistically Enhanced Learning) approach with CMP distribution.
Adapted from backend_draft/models/handball_predictor.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging
from scipy import stats

from core.models.base_model import BaseModel, PredictionResult, Sport

logger = logging.getLogger(__name__)


class HandballMarket(Enum):
    """Available handball betting markets."""
    MATCH_WINNER = "match_winner"
    HANDICAP = "handicap"
    TOTAL_GOALS = "total_goals"
    FIRST_HALF = "first_half"
    BOTH_TEAMS_SCORE = "both_teams_score"


@dataclass
class HandballFeatures:
    """Features for handball match prediction."""
    # Team info
    home_team: str
    away_team: str

    # Rankings/Ratings
    home_elo: float = 1500.0
    away_elo: float = 1500.0

    # Goals stats (per game)
    home_goals_scored: float = 25.0
    home_goals_conceded: float = 25.0
    away_goals_scored: float = 25.0
    away_goals_conceded: float = 25.0

    # First half stats
    home_first_half_goals: float = 12.0
    away_first_half_goals: float = 12.0

    # Form (last 5 matches: 1=win, 0.5=draw, 0=loss)
    home_form: List[float] = field(default_factory=list)
    away_form: List[float] = field(default_factory=list)

    # H2H
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_total_matches: int = 0

    # Home advantage factor
    home_advantage: float = 2.5  # Average goals advantage

    # Rest days
    home_rest_days: int = 4
    away_rest_days: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_elo": self.home_elo,
            "away_elo": self.away_elo,
            "home_goals_scored": self.home_goals_scored,
            "home_goals_conceded": self.home_goals_conceded,
            "away_goals_scored": self.away_goals_scored,
            "away_goals_conceded": self.away_goals_conceded,
            "home_form": self.home_form,
            "away_form": self.away_form,
            "home_rest_days": self.home_rest_days,
            "away_rest_days": self.away_rest_days,
        }


@dataclass
class HandballPrediction:
    """Handball match prediction result."""
    home_goals: float
    away_goals: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_under: Dict[float, Dict[str, float]]  # line -> {over, under}
    handicap: Dict[float, Dict[str, float]]  # line -> {home_cover, away_cover}
    confidence: float
    reasoning: List[str]


class HandballModel(BaseModel):
    """
    Handball prediction model using SEL approach.

    SEL = Statistically Enhanced Learning
    - Uses Conway-Maxwell-Poisson for goal modeling
    - Accounts for handball-specific patterns
    - Includes tactical and physical factors
    """

    def __init__(self):
        super().__init__(Sport.HANDBALL, "HandballSEL_v1")

        self.required_features = [
            "home_goals_scored", "home_goals_conceded",
            "away_goals_scored", "away_goals_conceded",
            "home_elo", "away_elo"
        ]

        # Home advantage in handball (stronger than football)
        self.home_advantage_goals = 2.5

        # Feature weights for prediction
        self.feature_weights = {
            "goals_average": 0.30,
            "elo_rating": 0.25,
            "recent_form": 0.20,
            "h2h": 0.10,
            "rest_factor": 0.10,
            "home_advantage": 0.05,
        }

        # Average goals in professional handball
        self.league_avg_goals = 55.0  # Total per match

    def predict(self, match_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict handball match outcome.

        Args:
            match_data: Dictionary with match features

        Returns:
            PredictionResult with winner prediction
        """
        self.validate_input(match_data)

        handball_pred = self.predict_match(match_data)

        # Determine predicted winner
        if handball_pred.home_win_prob > handball_pred.away_win_prob:
            if handball_pred.home_win_prob > handball_pred.draw_prob:
                winner = "home"
            else:
                winner = "draw"
        else:
            if handball_pred.away_win_prob > handball_pred.draw_prob:
                winner = "away"
            else:
                winner = "draw"

        probabilities = {
            "home": handball_pred.home_win_prob,
            "draw": handball_pred.draw_prob,
            "away": handball_pred.away_win_prob,
        }

        return PredictionResult(
            sport=self.sport.value,
            predicted_winner=winner,
            confidence=handball_pred.confidence,
            probabilities=probabilities,
            model_name=self.model_name,
            features_used=self.required_features,
            feature_values={
                "expected_home_goals": handball_pred.home_goals,
                "expected_away_goals": handball_pred.away_goals,
            },
            reasoning=handball_pred.reasoning,
            reliability_score=self._calculate_reliability(match_data),
        )

    def predict_proba(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Get outcome probabilities."""
        handball_pred = self.predict_match(match_data)
        return {
            "home": handball_pred.home_win_prob,
            "draw": handball_pred.draw_prob,
            "away": handball_pred.away_win_prob,
        }

    def predict_match(self, match_data: Dict[str, Any]) -> HandballPrediction:
        """
        Full match prediction with all markets.

        Args:
            match_data: Match features dictionary

        Returns:
            HandballPrediction with full analysis
        """
        # Extract features
        home_scored = match_data.get("home_goals_scored", 25.0)
        home_conceded = match_data.get("home_goals_conceded", 25.0)
        away_scored = match_data.get("away_goals_scored", 25.0)
        away_conceded = match_data.get("away_goals_conceded", 25.0)

        home_elo = match_data.get("home_elo", 1500)
        away_elo = match_data.get("away_elo", 1500)

        # Expected goals calculation
        exp_home_goals = self._calculate_expected_goals(
            attack=home_scored,
            defense=away_conceded,
            is_home=True,
            elo_diff=home_elo - away_elo
        )

        exp_away_goals = self._calculate_expected_goals(
            attack=away_scored,
            defense=home_conceded,
            is_home=False,
            elo_diff=away_elo - home_elo
        )

        # Rest factor adjustment
        home_rest = match_data.get("home_rest_days", 4)
        away_rest = match_data.get("away_rest_days", 4)
        exp_home_goals *= self._rest_factor(home_rest)
        exp_away_goals *= self._rest_factor(away_rest)

        # Form adjustment
        home_form = match_data.get("home_form", [])
        away_form = match_data.get("away_form", [])
        if home_form:
            form_adj = (sum(home_form) / len(home_form) - 0.5) * 2
            exp_home_goals += form_adj
        if away_form:
            form_adj = (sum(away_form) / len(away_form) - 0.5) * 2
            exp_away_goals += form_adj

        # Calculate outcome probabilities using CMP-like distribution
        home_win_prob, draw_prob, away_win_prob = self._calculate_outcome_probs(
            exp_home_goals, exp_away_goals
        )

        # Over/under lines
        total_expected = exp_home_goals + exp_away_goals
        over_under = {}
        for line in [50.5, 52.5, 54.5, 56.5, 58.5]:
            over_prob = self._calculate_over_prob(total_expected, line)
            over_under[line] = {
                "over": over_prob,
                "under": 1 - over_prob
            }

        # Handicap lines
        margin = exp_home_goals - exp_away_goals
        handicap = {}
        for line in [-5.5, -3.5, -1.5, 1.5, 3.5, 5.5]:
            cover_prob = self._calculate_handicap_prob(margin, line)
            handicap[line] = {
                "home_cover": cover_prob,
                "away_cover": 1 - cover_prob
            }

        # Calculate confidence
        confidence = self._calculate_confidence(
            home_win_prob, draw_prob, away_win_prob,
            abs(home_elo - away_elo)
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            match_data, exp_home_goals, exp_away_goals,
            home_win_prob, away_win_prob
        )

        return HandballPrediction(
            home_goals=round(exp_home_goals, 1),
            away_goals=round(exp_away_goals, 1),
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            over_under=over_under,
            handicap=handicap,
            confidence=confidence,
            reasoning=reasoning
        )

    def _calculate_expected_goals(
        self,
        attack: float,
        defense: float,
        is_home: bool,
        elo_diff: float
    ) -> float:
        """Calculate expected goals for a team."""
        # Base from attack vs defense
        base_goals = (attack + defense) / 2

        # Elo adjustment
        elo_factor = 1 + (elo_diff / 1000)
        base_goals *= elo_factor

        # Home advantage
        if is_home:
            base_goals += self.home_advantage_goals / 2

        return max(15.0, min(40.0, base_goals))

    def _rest_factor(self, rest_days: int) -> float:
        """Adjust for rest days."""
        if rest_days >= 7:
            return 1.02  # Well rested
        elif rest_days >= 4:
            return 1.0  # Normal
        elif rest_days >= 3:
            return 0.98  # Slightly tired
        else:
            return 0.95  # Very tired

    def _calculate_outcome_probs(
        self,
        home_goals: float,
        away_goals: float
    ) -> Tuple[float, float, float]:
        """Calculate 1X2 probabilities using Poisson-like approach."""
        # Simulate using normal approximation (faster than full Poisson)
        margin = home_goals - away_goals
        std_dev = np.sqrt(home_goals + away_goals) * 0.6

        # P(home win) = P(margin > 0.5)
        home_win = 1 - stats.norm.cdf(0.5, loc=margin, scale=std_dev)

        # P(away win) = P(margin < -0.5)
        away_win = stats.norm.cdf(-0.5, loc=margin, scale=std_dev)

        # P(draw) = remainder
        draw = max(0.02, 1 - home_win - away_win)

        # Normalize
        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total

    def _calculate_over_prob(self, expected_total: float, line: float) -> float:
        """Calculate probability of over."""
        std_dev = np.sqrt(expected_total) * 0.8
        return 1 - stats.norm.cdf(line, loc=expected_total, scale=std_dev)

    def _calculate_handicap_prob(self, margin: float, line: float) -> float:
        """Calculate probability of home covering handicap."""
        # Adjusted margin with line
        adjusted_margin = margin + line
        std_dev = 5.0  # Typical margin standard deviation
        return stats.norm.cdf(adjusted_margin, loc=0, scale=std_dev)

    def _calculate_confidence(
        self,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        elo_diff: float
    ) -> float:
        """Calculate prediction confidence."""
        # Max probability indicates confidence
        max_prob = max(home_prob, draw_prob, away_prob)

        # Elo difference adds confidence
        elo_factor = min(0.2, abs(elo_diff) / 500)

        confidence = max_prob * 0.7 + elo_factor

        return min(0.90, max(0.35, confidence))

    def _build_reasoning(
        self,
        match_data: Dict[str, Any],
        exp_home: float,
        exp_away: float,
        home_prob: float,
        away_prob: float
    ) -> List[str]:
        """Build explanation for prediction."""
        reasoning = []

        home_team = match_data.get("home_team", "Home")
        away_team = match_data.get("away_team", "Away")

        reasoning.append(
            f"Expected score: {home_team} {exp_home:.1f} - {exp_away:.1f} {away_team}"
        )

        # Elo comparison
        home_elo = match_data.get("home_elo", 1500)
        away_elo = match_data.get("away_elo", 1500)
        if home_elo != away_elo:
            diff = home_elo - away_elo
            if abs(diff) > 50:
                stronger = home_team if diff > 0 else away_team
                reasoning.append(f"{stronger} rated higher (Elo diff: {abs(diff):.0f})")

        # Goals analysis
        total = exp_home + exp_away
        if total > 58:
            reasoning.append("High-scoring match expected (over 58.5 likely)")
        elif total < 52:
            reasoning.append("Lower-scoring match expected (under 52.5 likely)")

        # Form
        home_form = match_data.get("home_form", [])
        away_form = match_data.get("away_form", [])
        if home_form:
            home_form_avg = sum(home_form) / len(home_form)
            if home_form_avg > 0.7:
                reasoning.append(f"{home_team} in excellent form")
            elif home_form_avg < 0.3:
                reasoning.append(f"{home_team} struggling recently")
        if away_form:
            away_form_avg = sum(away_form) / len(away_form)
            if away_form_avg > 0.7:
                reasoning.append(f"{away_team} in excellent form")
            elif away_form_avg < 0.3:
                reasoning.append(f"{away_team} struggling recently")

        # Home advantage
        reasoning.append(f"Home advantage factor: +{self.home_advantage_goals:.1f} goals")

        return reasoning

    def _calculate_reliability(self, match_data: Dict[str, Any]) -> float:
        """Calculate reliability score."""
        score = 0.3  # Base

        if match_data.get("home_elo") and match_data.get("away_elo"):
            score += 0.2

        if match_data.get("home_goals_scored") and match_data.get("away_goals_scored"):
            score += 0.2

        if match_data.get("home_form") and match_data.get("away_form"):
            score += 0.15

        if match_data.get("h2h_total_matches", 0) > 0:
            score += 0.15

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

    def predict_total(
        self,
        match_data: Dict[str, Any],
        line: float
    ) -> Dict[str, float]:
        """Predict total goals over/under."""
        pred = self.predict_match(match_data)
        return pred.over_under.get(line, {"over": 0.5, "under": 0.5})

    def predict_handicap(
        self,
        match_data: Dict[str, Any],
        line: float
    ) -> Dict[str, float]:
        """Predict handicap outcome."""
        pred = self.predict_match(match_data)
        return pred.handicap.get(line, {"home_cover": 0.5, "away_cover": 0.5})
