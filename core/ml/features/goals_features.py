"""
Goals feature extractor for over/under predictions.

Checkpoint: 1.3
Responsibility: Extract features relevant for total goals prediction (Poisson model).
Principle: Features should be interpretable and statistically meaningful.
"""

from typing import Dict, List, Set
import math

from core.data.schemas import MatchData, HistoricalMatch
from core.ml.features.base import BaseFeatureExtractor


class GoalsFeatureExtractor(BaseFeatureExtractor):
    """
    Ekstraktor cech dla predykcji liczby bramek.

    Cechy zaprojektowane dla modelu Poisson regression:
    - Średnie bramkowe (attack/defense)
    - Siła ataku/obrony względem ligi
    - Trendy bramkowe z H2H
    - Czynniki kontekstowe (dom/wyjazd, odpoczynek)
    """

    @property
    def name(self) -> str:
        return "goals"

    @property
    def required_fields(self) -> Set[str]:
        return {"home_stats", "away_stats"}

    def get_feature_names(self) -> List[str]:
        return [
            # Team averages
            "home_goals_scored_avg",
            "home_goals_conceded_avg",
            "away_goals_scored_avg",
            "away_goals_conceded_avg",
            # Attack/defense strength (relative to league)
            "home_attack_strength",
            "home_defense_strength",
            "away_attack_strength",
            "away_defense_strength",
            # Expected goals (basic Poisson lambda)
            "home_expected_goals",
            "away_expected_goals",
            "total_expected_goals",
            # H2H features
            "h2h_avg_total_goals",
            "h2h_over25_ratio",
            "h2h_matches_count",
            # Home/away specifics
            "home_home_goals_avg",
            "away_away_goals_avg",
            # Context
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            # Derived
            "goals_diff_expected",
            "both_teams_score_prob",
        ]

    def extract(self, match: MatchData) -> Dict[str, float]:
        """Extract goals-related features."""
        features = {}

        home = match.home_stats
        away = match.away_stats

        # Basic averages
        features["home_goals_scored_avg"] = home.goals_scored_avg
        features["home_goals_conceded_avg"] = home.goals_conceded_avg
        features["away_goals_scored_avg"] = away.goals_scored_avg
        features["away_goals_conceded_avg"] = away.goals_conceded_avg

        # Attack/defense strength (using property from TeamMatchStats)
        features["home_attack_strength"] = home.attack_strength
        features["home_defense_strength"] = home.defense_strength
        features["away_attack_strength"] = away.attack_strength
        features["away_defense_strength"] = away.defense_strength

        # Expected goals using Dixon-Coles style calculation
        # Home expected = home_attack * away_defense * league_avg * home_advantage
        league_avg = 1.3  # Average goals per team per match
        home_advantage = 1.1  # ~10% home advantage

        home_exp = (
            home.attack_strength *
            away.defense_strength *
            league_avg *
            home_advantage
        )
        away_exp = (
            away.attack_strength *
            home.defense_strength *
            league_avg
        )

        features["home_expected_goals"] = self._clip(home_exp, 0.1, 5.0)
        features["away_expected_goals"] = self._clip(away_exp, 0.1, 5.0)
        features["total_expected_goals"] = features["home_expected_goals"] + features["away_expected_goals"]

        # H2H features
        h2h_features = self._extract_h2h_features(match)
        features.update(h2h_features)

        # Home/away specific
        features["home_home_goals_avg"] = home.home_goals_avg if home.home_goals_avg else home.goals_scored_avg
        features["away_away_goals_avg"] = away.away_goals_avg if away.away_goals_avg else away.goals_scored_avg

        # Rest days
        features["home_rest_days"] = self._clip(home.rest_days, 0, 14)
        features["away_rest_days"] = self._clip(away.rest_days, 0, 14)
        features["rest_advantage"] = (home.rest_days - away.rest_days) / 7.0  # Normalized

        # Derived features
        features["goals_diff_expected"] = features["home_expected_goals"] - features["away_expected_goals"]

        # Both teams to score probability (simplified)
        home_score_prob = 1 - math.exp(-features["home_expected_goals"])
        away_score_prob = 1 - math.exp(-features["away_expected_goals"])
        features["both_teams_score_prob"] = home_score_prob * away_score_prob

        return features

    def _extract_h2h_features(self, match: MatchData) -> Dict[str, float]:
        """Extract features from head-to-head history."""
        features = {
            "h2h_avg_total_goals": 2.5,  # Default to league average
            "h2h_over25_ratio": 0.5,
            "h2h_matches_count": 0.0,
        }

        if not match.h2h_history:
            return features

        h2h = match.h2h_history
        features["h2h_matches_count"] = float(len(h2h))

        if len(h2h) == 0:
            return features

        # Calculate H2H stats
        total_goals = [m.total_goals for m in h2h]
        features["h2h_avg_total_goals"] = sum(total_goals) / len(total_goals)

        over25_count = sum(1 for m in h2h if m.is_over_25)
        features["h2h_over25_ratio"] = over25_count / len(h2h)

        return features

    def calculate_poisson_probability(
        self,
        expected: float,
        goals: int,
    ) -> float:
        """
        Calculate Poisson probability P(X = goals | lambda = expected).

        Useful for model validation.
        """
        if expected <= 0:
            return 0.0
        return (math.exp(-expected) * (expected ** goals)) / math.factorial(goals)

    def calculate_over_under_prob(
        self,
        home_exp: float,
        away_exp: float,
        threshold: float = 2.5,
        max_goals: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate over/under probabilities using independent Poisson.

        Args:
            home_exp: Expected home goals (lambda)
            away_exp: Expected away goals (lambda)
            threshold: Goals threshold (e.g., 2.5)
            max_goals: Maximum goals to consider

        Returns:
            Dict with 'over' and 'under' probabilities
        """
        under_prob = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h + a <= threshold:
                    prob_h = self.calculate_poisson_probability(home_exp, h)
                    prob_a = self.calculate_poisson_probability(away_exp, a)
                    under_prob += prob_h * prob_a

        return {
            "under": under_prob,
            "over": 1 - under_prob,
        }
