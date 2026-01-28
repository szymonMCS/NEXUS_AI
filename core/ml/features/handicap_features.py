"""
Handicap feature extractor for spread predictions.

Checkpoint: 1.4
Responsibility: Extract features relevant for handicap/spread prediction.
Principle: Focus on margin-related statistics and team strength differentials.
"""

from typing import Dict, List, Set
import math

from core.data.schemas import MatchData, HistoricalMatch
from core.ml.features.base import BaseFeatureExtractor


class HandicapFeatureExtractor(BaseFeatureExtractor):
    """
    Ekstraktor cech dla predykcji handicap (spread).

    Cechy zaprojektowane dla modelu klasyfikacyjnego (GBM):
    - Różnice w sile drużyn
    - Historyczne marże zwycięstw
    - ELO/ranking różnice
    - Trendy form
    """

    @property
    def name(self) -> str:
        return "handicap"

    @property
    def required_fields(self) -> Set[str]:
        return {"home_stats", "away_stats"}

    def get_feature_names(self) -> List[str]:
        return [
            # Strength differentials
            "attack_diff",
            "defense_diff",
            "overall_strength_diff",
            # Goal difference features
            "home_goal_diff_avg",
            "away_goal_diff_avg",
            "expected_margin",
            # Form features
            "home_form",
            "away_form",
            "form_diff",
            # H2H margin features
            "h2h_avg_margin",
            "h2h_home_win_ratio",
            "h2h_away_win_ratio",
            "h2h_draw_ratio",
            # ELO/ranking
            "elo_diff",
            "ranking_diff",
            # Odds implied
            "odds_implied_home_prob",
            "odds_implied_away_prob",
            "odds_margin",
            # Context
            "home_advantage_factor",
            "fatigue_factor",
            # Derived
            "blowout_risk",  # Risk of large margin
            "close_game_prob",  # Probability of close game
        ]

    def extract(self, match: MatchData) -> Dict[str, float]:
        """Extract handicap-related features."""
        features = {}

        home = match.home_stats
        away = match.away_stats

        # Strength differentials
        features["attack_diff"] = home.attack_strength - away.attack_strength
        features["defense_diff"] = away.defense_strength - home.defense_strength  # Lower is better
        features["overall_strength_diff"] = (
            (home.attack_strength - home.defense_strength) -
            (away.attack_strength - away.defense_strength)
        )

        # Goal difference averages
        home_gd = home.goals_scored_avg - home.goals_conceded_avg
        away_gd = away.goals_scored_avg - away.goals_conceded_avg
        features["home_goal_diff_avg"] = home_gd
        features["away_goal_diff_avg"] = away_gd

        # Expected margin (simplified model)
        home_advantage = 0.3  # Goals
        features["expected_margin"] = home_gd - away_gd + home_advantage

        # Form
        features["home_form"] = home.form_points
        features["away_form"] = away.form_points
        features["form_diff"] = home.form_points - away.form_points

        # H2H features
        h2h_features = self._extract_h2h_margin_features(match)
        features.update(h2h_features)

        # ELO/ranking (if available)
        elo_features = self._extract_elo_features(match)
        features.update(elo_features)

        # Odds implied probabilities
        odds_features = self._extract_odds_features(match)
        features.update(odds_features)

        # Context factors
        features["home_advantage_factor"] = 1.0  # Base, can be adjusted per league

        # Fatigue factor (based on rest days)
        rest_diff = home.rest_days - away.rest_days
        features["fatigue_factor"] = self._clip(rest_diff / 7.0, -1.0, 1.0)

        # Derived risk features
        features["blowout_risk"] = self._calculate_blowout_risk(features)
        features["close_game_prob"] = self._calculate_close_game_prob(features)

        return features

    def _extract_h2h_margin_features(self, match: MatchData) -> Dict[str, float]:
        """Extract margin-related features from H2H history."""
        features = {
            "h2h_avg_margin": 0.0,
            "h2h_home_win_ratio": 0.33,
            "h2h_away_win_ratio": 0.33,
            "h2h_draw_ratio": 0.34,
        }

        if not match.h2h_history:
            return features

        h2h = match.h2h_history
        if len(h2h) == 0:
            return features

        home_id = match.home_team.team_id
        margins = []
        home_wins = 0
        away_wins = 0
        draws = 0

        for m in h2h:
            # Calculate margin from home team's perspective
            if m.home_team_id == home_id:
                margin = m.goal_diff
            else:
                margin = -m.goal_diff

            margins.append(margin)

            if margin > 0:
                home_wins += 1
            elif margin < 0:
                away_wins += 1
            else:
                draws += 1

        features["h2h_avg_margin"] = sum(margins) / len(margins)
        features["h2h_home_win_ratio"] = home_wins / len(h2h)
        features["h2h_away_win_ratio"] = away_wins / len(h2h)
        features["h2h_draw_ratio"] = draws / len(h2h)

        return features

    def _extract_elo_features(self, match: MatchData) -> Dict[str, float]:
        """Extract ELO/ranking features."""
        features = {
            "elo_diff": 0.0,
            "ranking_diff": 0.0,
        }

        home_elo = match.home_team.elo_rating
        away_elo = match.away_team.elo_rating

        if home_elo and away_elo:
            features["elo_diff"] = (home_elo - away_elo) / 400.0  # Normalized

        home_rank = match.home_team.ranking
        away_rank = match.away_team.ranking

        if home_rank and away_rank:
            # Lower ranking is better, so away - home
            features["ranking_diff"] = (away_rank - home_rank) / 10.0  # Normalized

        return features

    def _extract_odds_features(self, match: MatchData) -> Dict[str, float]:
        """Extract implied probabilities from odds."""
        features = {
            "odds_implied_home_prob": 0.4,
            "odds_implied_away_prob": 0.3,
            "odds_margin": 0.0,
        }

        if not match.odds:
            return features

        odds = match.odds

        # Convert odds to implied probabilities
        if odds.home_win and odds.home_win > 1.0:
            features["odds_implied_home_prob"] = 1.0 / odds.home_win

        if odds.away_win and odds.away_win > 1.0:
            features["odds_implied_away_prob"] = 1.0 / odds.away_win

        # Odds margin (bookmaker's edge)
        if odds.home_win and odds.away_win and odds.draw:
            total = (1/odds.home_win) + (1/odds.draw) + (1/odds.away_win)
            features["odds_margin"] = total - 1.0

        return features

    def _calculate_blowout_risk(self, features: Dict[str, float]) -> float:
        """
        Calculate risk of a blowout (large margin game).

        High when teams are mismatched.
        """
        strength_diff = abs(features.get("overall_strength_diff", 0))
        form_diff = abs(features.get("form_diff", 0))

        # Combine factors
        risk = (strength_diff * 0.6 + form_diff * 0.4)
        return self._clip(risk, 0.0, 1.0)

    def _calculate_close_game_prob(self, features: Dict[str, float]) -> float:
        """
        Calculate probability of a close game (margin <= 1).

        High when teams are evenly matched.
        """
        strength_diff = abs(features.get("overall_strength_diff", 0))
        exp_margin = abs(features.get("expected_margin", 0))

        # Close game more likely when differences are small
        closeness = 1.0 - (strength_diff * 0.5 + exp_margin * 0.3)
        return self._clip(closeness, 0.1, 0.9)

    def predict_cover_probability(
        self,
        features: Dict[str, float],
        handicap_line: float,
    ) -> Dict[str, float]:
        """
        Estimate probability of covering a handicap.

        Args:
            features: Extracted features
            handicap_line: The handicap line (e.g., -1.5 for home)

        Returns:
            Dict with 'home_cover' and 'away_cover' probabilities
        """
        expected_margin = features.get("expected_margin", 0)

        # Simple model: assume normal distribution of margins
        # Std dev of football margins is typically around 1.5-2.0
        std_dev = 1.7

        # Probability home covers (margin > handicap_line)
        z_score = (expected_margin - handicap_line) / std_dev

        # Approximate normal CDF
        home_cover = self._normal_cdf(z_score)

        return {
            "home_cover": home_cover,
            "away_cover": 1 - home_cover,
        }

    def _normal_cdf(self, z: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
