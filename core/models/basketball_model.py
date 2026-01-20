# core/models/basketball_model.py
"""
Basketball prediction model for NEXUS AI.
Uses statistical features: ratings, form, rest days, home/away.
Adapted from backend_draft/models/nba_predictor.py
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from core.models.base_model import (
    BaseModel, PredictionResult, ModelMetrics, Sport
)

logger = logging.getLogger(__name__)


@dataclass
class BasketballFeatures:
    """Extracted features for basketball prediction."""
    # Home team
    home_rating: float = 100.0  # Net rating (off - def)
    home_off_rating: float = 110.0  # Points per 100 possessions
    home_def_rating: float = 110.0
    home_recent_form: float = 0.5  # Win rate last 10
    home_home_record: float = 0.5  # Home win rate
    home_rest_days: int = 2
    home_b2b: bool = False  # Back-to-back game
    home_streak: int = 0  # Positive = wins, negative = losses

    # Away team
    away_rating: float = 100.0
    away_off_rating: float = 110.0
    away_def_rating: float = 110.0
    away_recent_form: float = 0.5
    away_away_record: float = 0.5  # Away win rate
    away_rest_days: int = 2
    away_b2b: bool = False
    away_streak: int = 0

    # Head-to-head
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    season_h2h_home_wins: int = 0
    season_h2h_away_wins: int = 0

    # Context
    is_playoff: bool = False
    league: str = "NBA"  # NBA, Euroleague, etc.
    travel_distance: float = 0.0  # Miles traveled by away team


class BasketballModel(BaseModel):
    """
    Basketball prediction model using statistical analysis.

    Feature weights (from plan):
    - Offensive/Defensive Ratings: 35%
    - Recent Performance: 25%
    - Rest Days Impact: 20%
    - Home/Away Record: 15%
    - Key Player Impact: 5% (via injury adjustments)
    """

    # Feature weights
    WEIGHTS = {
        "ratings": 0.35,
        "recent_form": 0.25,
        "rest_days": 0.20,
        "home_away": 0.15,
        "h2h": 0.05
    }

    # Home court advantage by league
    HOME_ADVANTAGE = {
        "NBA": 0.035,      # ~3.5% edge
        "Euroleague": 0.04,
        "NCAA": 0.045,
        "default": 0.04
    }

    # Rest day impact
    REST_IMPACT = {
        0: -0.03,   # B2B: -3%
        1: 0.0,     # 1 day rest: neutral
        2: 0.01,    # 2 days: +1%
        3: 0.015,   # 3 days: +1.5%
        "4+": 0.02  # 4+ days: +2%
    }

    def __init__(self):
        """Initialize basketball model."""
        super().__init__(Sport.BASKETBALL, "BasketballStatisticalModel")

        self.required_features = [
            "home_rating", "away_rating",
            "home_recent_form", "away_recent_form"
        ]

        self.feature_names = [
            "rating_diff", "off_rating_diff", "def_rating_diff",
            "form_diff", "home_advantage", "rest_advantage",
            "h2h_factor", "streak_factor", "playoff_factor"
        ]

        self.is_trained = True  # Statistical model

    def validate_input(self, match_data: Dict[str, Any]) -> bool:
        """Validate input has minimum required features."""
        # Allow partial data
        return True

    def extract_features(self, match_data: Dict[str, Any]) -> BasketballFeatures:
        """Extract and normalize features from match data."""
        features = BasketballFeatures()

        # Home team features
        features.home_rating = match_data.get("home_rating", 100.0)
        features.home_off_rating = match_data.get("home_off_rating", 110.0)
        features.home_def_rating = match_data.get("home_def_rating", 110.0)
        features.home_recent_form = match_data.get("home_recent_form", 0.5)
        features.home_home_record = match_data.get("home_home_record", 0.5)
        features.home_rest_days = match_data.get("home_rest_days", 2)
        features.home_b2b = features.home_rest_days == 0
        features.home_streak = match_data.get("home_streak", 0)

        # Away team features
        features.away_rating = match_data.get("away_rating", 100.0)
        features.away_off_rating = match_data.get("away_off_rating", 110.0)
        features.away_def_rating = match_data.get("away_def_rating", 110.0)
        features.away_recent_form = match_data.get("away_recent_form", 0.5)
        features.away_away_record = match_data.get("away_away_record", 0.5)
        features.away_rest_days = match_data.get("away_rest_days", 2)
        features.away_b2b = features.away_rest_days == 0
        features.away_streak = match_data.get("away_streak", 0)

        # H2H
        h2h = match_data.get("h2h", {})
        features.h2h_home_wins = h2h.get("home_wins", 0)
        features.h2h_away_wins = h2h.get("away_wins", 0)
        features.season_h2h_home_wins = h2h.get("season_home_wins", 0)
        features.season_h2h_away_wins = h2h.get("season_away_wins", 0)

        # Context
        features.is_playoff = match_data.get("is_playoff", False)
        features.league = match_data.get("league", "NBA")
        features.travel_distance = match_data.get("travel_distance", 0.0)

        return features

    def predict(self, match_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict match outcome.

        Args:
            match_data: Dictionary with match features

        Returns:
            PredictionResult with prediction details
        """
        self.validate_input(match_data)
        features = self.extract_features(match_data)

        # Calculate component probabilities
        components = self._calculate_components(features)

        # Weighted combination
        home_prob = 0.0
        for component, (home_comp, _) in components.items():
            weight = self.WEIGHTS.get(component, 0)
            home_prob += weight * home_comp

        # Add home court advantage
        home_advantage = self.HOME_ADVANTAGE.get(features.league, self.HOME_ADVANTAGE["default"])
        home_prob += home_advantage

        # Normalize
        home_prob = self.normalize_probability(home_prob)
        away_prob = 1 - home_prob

        # Determine winner
        predicted_winner = "home" if home_prob > 0.5 else "away"
        confidence = max(home_prob, away_prob)

        # Generate reasoning
        reasoning = self._generate_reasoning(features, components, home_prob)

        result = PredictionResult(
            sport=self.sport.value,
            predicted_winner=predicted_winner,
            confidence=confidence,
            probabilities={"home": home_prob, "away": away_prob},
            model_name=self.model_name,
            features_used=list(components.keys()),
            feature_values=match_data,
            reasoning=reasoning
        )

        result.reliability_score = self.calculate_reliability_score(match_data, result)
        return result

    def predict_proba(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Get probability distribution."""
        result = self.predict(match_data)
        return result.probabilities

    def _calculate_components(self, features: BasketballFeatures) -> Dict[str, tuple]:
        """Calculate probability components."""
        components = {}

        # 1. Rating Factor (35%)
        rating_prob = self._calculate_rating_probability(features)
        components["ratings"] = (rating_prob, 1 - rating_prob)

        # 2. Recent Form (25%)
        form_prob = self._calculate_form_probability(features)
        components["recent_form"] = (form_prob, 1 - form_prob)

        # 3. Rest Days (20%)
        rest_prob = self._calculate_rest_probability(features)
        components["rest_days"] = (rest_prob, 1 - rest_prob)

        # 4. Home/Away Record (15%)
        home_away_prob = self._calculate_home_away_probability(features)
        components["home_away"] = (home_away_prob, 1 - home_away_prob)

        # 5. H2H (5%)
        h2h_prob = self._calculate_h2h_probability(features)
        components["h2h"] = (h2h_prob, 1 - h2h_prob)

        return components

    def _calculate_rating_probability(self, features: BasketballFeatures) -> float:
        """Calculate probability based on team ratings."""
        # Net rating difference
        home_net = features.home_off_rating - features.home_def_rating
        away_net = features.away_off_rating - features.away_def_rating

        rating_diff = home_net - away_net

        # Convert to probability using logistic function
        # Empirically, each point of net rating ~2.5% win probability
        prob = 1 / (1 + np.exp(-rating_diff * 0.025))

        return prob

    def _calculate_form_probability(self, features: BasketballFeatures) -> float:
        """Calculate probability based on recent form."""
        home_form = features.home_recent_form
        away_form = features.away_recent_form

        # Include streak momentum
        home_momentum = np.tanh(features.home_streak * 0.1)  # -1 to 1
        away_momentum = np.tanh(features.away_streak * 0.1)

        # Combine form and momentum
        home_adjusted = home_form * 0.7 + (home_momentum + 1) / 2 * 0.3
        away_adjusted = away_form * 0.7 + (away_momentum + 1) / 2 * 0.3

        total = home_adjusted + away_adjusted
        if total == 0:
            return 0.5

        return home_adjusted / total

    def _calculate_rest_probability(self, features: BasketballFeatures) -> float:
        """Calculate probability adjustment based on rest days."""
        # Get rest impact for each team
        home_rest = features.home_rest_days
        away_rest = features.away_rest_days

        home_impact = self.REST_IMPACT.get(home_rest, self.REST_IMPACT["4+"])
        away_impact = self.REST_IMPACT.get(away_rest, self.REST_IMPACT["4+"])

        # Travel fatigue for away team
        if features.travel_distance > 1500:
            away_impact -= 0.01  # Long travel penalty

        # B2B penalty
        if features.home_b2b:
            home_impact = self.REST_IMPACT[0]
        if features.away_b2b:
            away_impact = self.REST_IMPACT[0]

        # Convert to probability
        rest_diff = home_impact - away_impact
        return 0.5 + rest_diff

    def _calculate_home_away_probability(self, features: BasketballFeatures) -> float:
        """Calculate probability based on home/away records."""
        home_record = features.home_home_record
        away_record = features.away_away_record

        # Home team uses home record, away team uses away record
        total = home_record + (1 - away_record)
        if total == 0:
            return 0.5

        return home_record / total

    def _calculate_h2h_probability(self, features: BasketballFeatures) -> float:
        """Calculate probability based on head-to-head record."""
        total_h2h = features.h2h_home_wins + features.h2h_away_wins
        season_h2h = features.season_h2h_home_wins + features.season_h2h_away_wins

        if total_h2h == 0:
            return 0.5  # No H2H history

        # Weight season H2H more heavily
        overall_rate = features.h2h_home_wins / total_h2h
        season_rate = features.season_h2h_home_wins / season_h2h if season_h2h > 0 else overall_rate

        h2h_prob = (season_rate * 0.6 + overall_rate * 0.4)

        # Regress towards 0.5 if small sample
        sample_weight = min(1.0, total_h2h / 8)
        return 0.5 + (h2h_prob - 0.5) * sample_weight

    def _generate_reasoning(self, features: BasketballFeatures,
                           components: Dict[str, tuple],
                           home_prob: float) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []

        # Rating analysis
        home_net = features.home_off_rating - features.home_def_rating
        away_net = features.away_off_rating - features.away_def_rating
        rating_diff = home_net - away_net

        if abs(rating_diff) > 5:
            better = "Home" if rating_diff > 0 else "Away"
            reasoning.append(f"{better} team has significant rating advantage ({rating_diff:+.1f} net rating)")
        else:
            reasoning.append(f"Teams are evenly matched by rating ({rating_diff:+.1f} net rating)")

        # Form analysis
        if abs(features.home_recent_form - features.away_recent_form) > 0.15:
            better = "Home" if features.home_recent_form > features.away_recent_form else "Away"
            reasoning.append(f"{better} team in better recent form")

        # Rest analysis
        rest_diff = features.home_rest_days - features.away_rest_days
        if features.home_b2b:
            reasoning.append("Home team on back-to-back (fatigue factor)")
        elif features.away_b2b:
            reasoning.append("Away team on back-to-back (fatigue factor)")
        elif abs(rest_diff) >= 2:
            better = "Home" if rest_diff > 0 else "Away"
            reasoning.append(f"{better} team has rest advantage ({rest_diff:+d} days)")

        # Streak analysis
        if abs(features.home_streak) >= 3 or abs(features.away_streak) >= 3:
            home_desc = f"W{features.home_streak}" if features.home_streak > 0 else f"L{abs(features.home_streak)}"
            away_desc = f"W{features.away_streak}" if features.away_streak > 0 else f"L{abs(features.away_streak)}"
            reasoning.append(f"Streaks: Home {home_desc}, Away {away_desc}")

        # Home court
        reasoning.append(f"Home court advantage: {features.league}")

        # Final prediction
        winner = "Home" if home_prob > 0.5 else "Away"
        reasoning.append(f"Model prediction: {winner} ({max(home_prob, 1-home_prob):.1%} confidence)")

        return reasoning

    def explain_prediction(self, match_data: Dict[str, Any],
                          prediction: PredictionResult) -> List[str]:
        """Generate detailed explanation."""
        return prediction.reasoning

    def predict_spread(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict point spread and total.

        Returns:
            Dictionary with spread predictions
        """
        prediction = self.predict(match_data)
        features = self.extract_features(match_data)

        home_prob = prediction.probabilities["home"]

        # Estimate expected margin
        # Use rating difference as base
        home_net = features.home_off_rating - features.home_def_rating
        away_net = features.away_off_rating - features.away_def_rating
        rating_diff = home_net - away_net

        # Add home court (~3 points)
        home_advantage_pts = 3.0
        expected_margin = rating_diff * 0.3 + home_advantage_pts  # ~0.3 pts per net rating point

        # Estimate total
        avg_pace = 100  # Possessions per game
        expected_home_pts = features.home_off_rating * avg_pace / 100
        expected_away_pts = features.away_off_rating * avg_pace / 100
        expected_total = expected_home_pts + expected_away_pts

        return {
            "home_probability": home_prob,
            "away_probability": 1 - home_prob,
            "expected_margin": round(expected_margin, 1),
            "expected_total": round(expected_total, 1),
            "suggested_spread": round(expected_margin),
            "confidence": prediction.confidence
        }
