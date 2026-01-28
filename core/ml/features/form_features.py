"""
Form/momentum feature extractor.

Checkpoint: 1.5
Responsibility: Extract features related to team form and momentum.
"""

from typing import Dict, List, Set

from core.data.schemas import MatchData
from core.ml.features.base import BaseFeatureExtractor


class FormFeatureExtractor(BaseFeatureExtractor):
    """
    Ekstraktor cech związanych z formą drużyn.

    Cechy:
    - Punkty formowe (ostatnie N meczów)
    - Seria zwycięstw/porażek
    - Trend bramkowy
    - Odpoczynek między meczami
    """

    @property
    def name(self) -> str:
        return "form"

    @property
    def required_fields(self) -> Set[str]:
        return {"home_stats", "away_stats"}

    def get_feature_names(self) -> List[str]:
        return [
            # Basic form
            "home_form_points",
            "away_form_points",
            "form_points_diff",
            # Rest and fatigue
            "home_rest_days_normalized",
            "away_rest_days_normalized",
            "rest_diff_normalized",
            # Form quality indicators
            "home_form_consistency",
            "away_form_consistency",
            # Momentum indicators
            "home_momentum",
            "away_momentum",
            "momentum_diff",
            # Combined form score
            "home_overall_form",
            "away_overall_form",
        ]

    def extract(self, match: MatchData) -> Dict[str, float]:
        """Extract form-related features."""
        features = {}

        home = match.home_stats
        away = match.away_stats

        # Basic form points (already 0-1 normalized)
        features["home_form_points"] = home.form_points
        features["away_form_points"] = away.form_points
        features["form_points_diff"] = home.form_points - away.form_points

        # Rest days (normalized to 0-1 scale, 7 days = 1.0)
        features["home_rest_days_normalized"] = self._clip(home.rest_days / 7.0, 0, 2)
        features["away_rest_days_normalized"] = self._clip(away.rest_days / 7.0, 0, 2)
        features["rest_diff_normalized"] = (home.rest_days - away.rest_days) / 7.0

        # Form consistency (based on goals variance - estimated)
        # Higher form with lower goals conceded = more consistent
        features["home_form_consistency"] = self._calculate_consistency(home.form_points, home.goals_conceded_avg)
        features["away_form_consistency"] = self._calculate_consistency(away.form_points, away.goals_conceded_avg)

        # Momentum (form + rest advantage)
        home_momentum = self._calculate_momentum(home.form_points, home.rest_days)
        away_momentum = self._calculate_momentum(away.form_points, away.rest_days)
        features["home_momentum"] = home_momentum
        features["away_momentum"] = away_momentum
        features["momentum_diff"] = home_momentum - away_momentum

        # Overall form score (weighted combination)
        features["home_overall_form"] = self._calculate_overall_form(
            home.form_points, home.rest_days, home.goals_scored_avg, home.goals_conceded_avg
        )
        features["away_overall_form"] = self._calculate_overall_form(
            away.form_points, away.rest_days, away.goals_scored_avg, away.goals_conceded_avg
        )

        return features

    def _calculate_consistency(self, form_points: float, goals_conceded_avg: float) -> float:
        """
        Calculate form consistency.

        High form + low goals conceded = consistent defensive form.
        """
        # Invert goals conceded (lower is better)
        defense_factor = 1.0 - self._clip(goals_conceded_avg / 3.0, 0, 1)
        return (form_points * 0.7 + defense_factor * 0.3)

    def _calculate_momentum(self, form_points: float, rest_days: int) -> float:
        """
        Calculate momentum score.

        Good form + optimal rest (3-5 days) = high momentum.
        """
        # Optimal rest is around 4 days
        rest_factor = 1.0 - abs(rest_days - 4) / 10.0
        rest_factor = self._clip(rest_factor, 0, 1)

        return form_points * 0.8 + rest_factor * 0.2

    def _calculate_overall_form(
        self,
        form_points: float,
        rest_days: int,
        goals_scored_avg: float,
        goals_conceded_avg: float,
    ) -> float:
        """
        Calculate overall form score combining multiple factors.
        """
        # Components
        form_component = form_points  # 0-1

        # Rest component (optimal around 4-5 days)
        rest_component = 1.0 - abs(rest_days - 4.5) / 10.0
        rest_component = self._clip(rest_component, 0, 1)

        # Attack component (goals scored relative to average)
        attack_component = self._clip(goals_scored_avg / 2.0, 0, 1)

        # Defense component (inverse of goals conceded)
        defense_component = 1.0 - self._clip(goals_conceded_avg / 3.0, 0, 1)

        # Weighted combination
        overall = (
            form_component * 0.4 +
            rest_component * 0.15 +
            attack_component * 0.25 +
            defense_component * 0.2
        )

        return self._clip(overall, 0, 1)
