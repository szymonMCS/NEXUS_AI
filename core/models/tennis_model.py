# core/models/tennis_model.py
"""
Tennis prediction model for NEXUS AI.
Uses statistical features: ranking, form, H2H, surface stats, fatigue.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from core.models.base_model import (
    BaseModel, PredictionResult, ModelMetrics, Sport
)

logger = logging.getLogger(__name__)


@dataclass
class TennisFeatures:
    """Extracted features for tennis prediction."""
    # Player 1 (usually home/higher ranked)
    p1_ranking: int = 0
    p1_elo: float = 1500.0
    p1_surface_elo: float = 1500.0
    p1_recent_form: float = 0.5  # Win rate last 10 matches
    p1_surface_form: float = 0.5  # Win rate on surface last 10
    p1_fatigue: float = 0.0  # Matches in last 30 days
    p1_tournament_round: int = 1  # Round in current tournament

    # Player 2
    p2_ranking: int = 0
    p2_elo: float = 1500.0
    p2_surface_elo: float = 1500.0
    p2_recent_form: float = 0.5
    p2_surface_form: float = 0.5
    p2_fatigue: float = 0.0
    p2_tournament_round: int = 1

    # Head-to-head
    h2h_p1_wins: int = 0
    h2h_p2_wins: int = 0
    h2h_surface_p1_wins: int = 0
    h2h_surface_p2_wins: int = 0
    recent_h2h_p1_wins: int = 0  # Last 5 meetings
    recent_h2h_p2_wins: int = 0

    # Match context
    surface: str = "hard"  # hard, clay, grass, carpet
    tournament_category: str = "250"  # 250, 500, 1000, GS
    is_indoor: bool = False
    best_of: int = 3  # 3 or 5 sets


class TennisModel(BaseModel):
    """
    Tennis prediction model using statistical analysis.

    Feature weights (from plan):
    - Ranking Factor: 30%
    - Recent Form: 25%
    - H2H Record: 20%
    - Surface Stats: 15%
    - Fatigue Factor: 10%
    """

    # Feature weights
    WEIGHTS = {
        "ranking": 0.30,
        "recent_form": 0.25,
        "h2h": 0.20,
        "surface": 0.15,
        "fatigue": 0.10
    }

    # Surface adjustment factors
    SURFACE_ADJUSTMENT = {
        "clay": {"baseline": 1.1, "serve_and_volley": 0.85},
        "grass": {"baseline": 0.9, "serve_and_volley": 1.15},
        "hard": {"baseline": 1.0, "serve_and_volley": 1.0},
        "carpet": {"baseline": 0.95, "serve_and_volley": 1.1}
    }

    # Tournament importance multiplier
    TOURNAMENT_IMPORTANCE = {
        "GS": 1.0,      # Grand Slam
        "1000": 0.95,   # Masters 1000
        "500": 0.90,
        "250": 0.85,
        "challenger": 0.75,
        "futures": 0.60
    }

    def __init__(self):
        """Initialize tennis model."""
        super().__init__(Sport.TENNIS, "TennisStatisticalModel")

        self.required_features = [
            "p1_ranking", "p2_ranking",
            "p1_recent_form", "p2_recent_form",
            "surface"
        ]

        self.feature_names = [
            "ranking_factor", "elo_factor", "surface_elo_factor",
            "recent_form_factor", "surface_form_factor",
            "h2h_factor", "h2h_surface_factor", "recent_h2h_factor",
            "fatigue_factor", "tournament_importance",
            "home_advantage"  # For Davis Cup etc.
        ]

        self.is_trained = True  # Statistical model, no training needed

    def validate_input(self, match_data: Dict[str, Any]) -> bool:
        """Validate input has minimum required features."""
        missing = [f for f in self.required_features if f not in match_data]
        if missing:
            logger.warning(f"Missing features: {missing}")
            # Allow partial data, will use defaults
            return True
        return True

    def extract_features(self, match_data: Dict[str, Any]) -> TennisFeatures:
        """Extract and normalize features from match data."""
        features = TennisFeatures()

        # Player 1 features
        features.p1_ranking = match_data.get("p1_ranking", 100)
        features.p1_elo = match_data.get("p1_elo", 1500.0)
        features.p1_surface_elo = match_data.get("p1_surface_elo", features.p1_elo)
        features.p1_recent_form = match_data.get("p1_recent_form", 0.5)
        features.p1_surface_form = match_data.get("p1_surface_form", features.p1_recent_form)
        features.p1_fatigue = match_data.get("p1_fatigue", 0.0)
        features.p1_tournament_round = match_data.get("p1_tournament_round", 1)

        # Player 2 features
        features.p2_ranking = match_data.get("p2_ranking", 100)
        features.p2_elo = match_data.get("p2_elo", 1500.0)
        features.p2_surface_elo = match_data.get("p2_surface_elo", features.p2_elo)
        features.p2_recent_form = match_data.get("p2_recent_form", 0.5)
        features.p2_surface_form = match_data.get("p2_surface_form", features.p2_recent_form)
        features.p2_fatigue = match_data.get("p2_fatigue", 0.0)
        features.p2_tournament_round = match_data.get("p2_tournament_round", 1)

        # H2H features
        h2h = match_data.get("h2h", {})
        features.h2h_p1_wins = h2h.get("p1_wins", 0)
        features.h2h_p2_wins = h2h.get("p2_wins", 0)
        features.h2h_surface_p1_wins = h2h.get("surface_p1_wins", 0)
        features.h2h_surface_p2_wins = h2h.get("surface_p2_wins", 0)
        features.recent_h2h_p1_wins = h2h.get("recent_p1_wins", 0)
        features.recent_h2h_p2_wins = h2h.get("recent_p2_wins", 0)

        # Match context
        features.surface = match_data.get("surface", "hard").lower()
        features.tournament_category = match_data.get("tournament_category", "250")
        features.is_indoor = match_data.get("is_indoor", False)
        features.best_of = match_data.get("best_of", 3)

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
        p1_prob = 0.0
        for component, (p1_comp, _) in components.items():
            weight = self.WEIGHTS.get(component, 0)
            p1_prob += weight * p1_comp

        # Normalize
        p1_prob = self.normalize_probability(p1_prob)
        p2_prob = 1 - p1_prob

        # Determine winner
        predicted_winner = "p1" if p1_prob > 0.5 else "p2"
        confidence = max(p1_prob, p2_prob)

        # Generate reasoning
        reasoning = self._generate_reasoning(features, components, p1_prob)

        result = PredictionResult(
            sport=self.sport.value,
            predicted_winner=predicted_winner,
            confidence=confidence,
            probabilities={"p1": p1_prob, "p2": p2_prob},
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

    def _calculate_components(self, features: TennisFeatures) -> Dict[str, tuple]:
        """Calculate probability components."""
        components = {}

        # 1. Ranking Factor (30%)
        ranking_prob = self._calculate_ranking_probability(features)
        components["ranking"] = (ranking_prob, 1 - ranking_prob)

        # 2. Recent Form (25%)
        form_prob = self._calculate_form_probability(features)
        components["recent_form"] = (form_prob, 1 - form_prob)

        # 3. H2H Record (20%)
        h2h_prob = self._calculate_h2h_probability(features)
        components["h2h"] = (h2h_prob, 1 - h2h_prob)

        # 4. Surface Stats (15%)
        surface_prob = self._calculate_surface_probability(features)
        components["surface"] = (surface_prob, 1 - surface_prob)

        # 5. Fatigue Factor (10%)
        fatigue_prob = self._calculate_fatigue_probability(features)
        components["fatigue"] = (fatigue_prob, 1 - fatigue_prob)

        return components

    def _calculate_ranking_probability(self, features: TennisFeatures) -> float:
        """Calculate probability based on rankings."""
        # Use ELO-style calculation
        if features.p1_elo and features.p2_elo:
            elo_prob = self.elo_probability(features.p1_elo, features.p2_elo)
        else:
            # Fallback to ranking-based
            r1 = max(1, features.p1_ranking)
            r2 = max(1, features.p2_ranking)
            # Lower ranking = better, so invert
            elo_prob = r2 / (r1 + r2)

        return elo_prob

    def _calculate_form_probability(self, features: TennisFeatures) -> float:
        """Calculate probability based on recent form."""
        # Combine overall form and surface-specific form
        p1_form = (features.p1_recent_form * 0.6 + features.p1_surface_form * 0.4)
        p2_form = (features.p2_recent_form * 0.6 + features.p2_surface_form * 0.4)

        total = p1_form + p2_form
        if total == 0:
            return 0.5

        return p1_form / total

    def _calculate_h2h_probability(self, features: TennisFeatures) -> float:
        """Calculate probability based on head-to-head record."""
        # Weight recent H2H more heavily
        total_h2h = features.h2h_p1_wins + features.h2h_p2_wins
        recent_h2h = features.recent_h2h_p1_wins + features.recent_h2h_p2_wins
        surface_h2h = features.h2h_surface_p1_wins + features.h2h_surface_p2_wins

        if total_h2h == 0:
            return 0.5  # No H2H history

        # Calculate weighted H2H score
        overall_rate = features.h2h_p1_wins / total_h2h if total_h2h > 0 else 0.5
        recent_rate = features.recent_h2h_p1_wins / recent_h2h if recent_h2h > 0 else overall_rate
        surface_rate = features.h2h_surface_p1_wins / surface_h2h if surface_h2h > 0 else overall_rate

        # Weights: recent 50%, surface 30%, overall 20%
        h2h_prob = (recent_rate * 0.5 + surface_rate * 0.3 + overall_rate * 0.2)

        # Regress towards 0.5 if small sample size
        sample_weight = min(1.0, total_h2h / 10)  # Full weight at 10+ matches
        return 0.5 + (h2h_prob - 0.5) * sample_weight

    def _calculate_surface_probability(self, features: TennisFeatures) -> float:
        """Calculate probability based on surface performance."""
        # Use surface-specific ELO if available
        if features.p1_surface_elo and features.p2_surface_elo:
            return self.elo_probability(features.p1_surface_elo, features.p2_surface_elo)

        # Fallback to surface form
        p1_surface = features.p1_surface_form
        p2_surface = features.p2_surface_form

        total = p1_surface + p2_surface
        if total == 0:
            return 0.5

        return p1_surface / total

    def _calculate_fatigue_probability(self, features: TennisFeatures) -> float:
        """Calculate probability adjustment based on fatigue."""
        # Fatigue is 0-1 where higher = more tired
        # Impact: higher fatigue reduces win probability

        # Calculate fatigue difference
        p1_fresh = 1 - features.p1_fatigue
        p2_fresh = 1 - features.p2_fatigue

        total = p1_fresh + p2_fresh
        if total == 0:
            return 0.5

        return p1_fresh / total

    def _generate_reasoning(self, features: TennisFeatures,
                           components: Dict[str, tuple],
                           p1_prob: float) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []

        # Ranking analysis
        rank_prob = components["ranking"][0]
        if rank_prob > 0.6:
            reasoning.append(f"P1 has significant ranking advantage (#{features.p1_ranking} vs #{features.p2_ranking})")
        elif rank_prob < 0.4:
            reasoning.append(f"P2 has significant ranking advantage (#{features.p2_ranking} vs #{features.p1_ranking})")
        else:
            reasoning.append(f"Rankings are competitive (#{features.p1_ranking} vs #{features.p2_ranking})")

        # Form analysis
        form_prob = components["recent_form"][0]
        if form_prob > 0.55:
            reasoning.append(f"P1 in better recent form ({features.p1_recent_form:.0%} vs {features.p2_recent_form:.0%})")
        elif form_prob < 0.45:
            reasoning.append(f"P2 in better recent form ({features.p2_recent_form:.0%} vs {features.p1_recent_form:.0%})")

        # H2H analysis
        total_h2h = features.h2h_p1_wins + features.h2h_p2_wins
        if total_h2h > 0:
            h2h_prob = components["h2h"][0]
            if h2h_prob > 0.55:
                reasoning.append(f"P1 leads H2H ({features.h2h_p1_wins}-{features.h2h_p2_wins})")
            elif h2h_prob < 0.45:
                reasoning.append(f"P2 leads H2H ({features.h2h_p2_wins}-{features.h2h_p1_wins})")
            else:
                reasoning.append(f"H2H is balanced ({features.h2h_p1_wins}-{features.h2h_p2_wins})")

        # Surface analysis
        reasoning.append(f"Match on {features.surface} (best of {features.best_of})")

        # Final prediction
        winner = "P1" if p1_prob > 0.5 else "P2"
        reasoning.append(f"Model prediction: {winner} ({max(p1_prob, 1-p1_prob):.1%} confidence)")

        return reasoning

    def explain_prediction(self, match_data: Dict[str, Any],
                          prediction: PredictionResult) -> List[str]:
        """Generate detailed explanation."""
        return prediction.reasoning

    def predict_sets(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict expected number of sets and set scores.

        Returns:
            Dictionary with set predictions
        """
        prediction = self.predict(match_data)
        features = self.extract_features(match_data)

        p1_prob = prediction.probabilities["p1"]
        best_of = features.best_of
        sets_to_win = (best_of // 2) + 1

        # Calculate expected sets
        if best_of == 3:
            # Possible outcomes: 2-0, 2-1
            p_2_0 = p1_prob ** 2
            p_0_2 = (1 - p1_prob) ** 2
            p_2_1 = p1_prob ** 2 * (1 - p1_prob) * 2
            p_1_2 = (1 - p1_prob) ** 2 * p1_prob * 2

            expected_sets = p_2_0 * 2 + p_0_2 * 2 + p_2_1 * 3 + p_1_2 * 3
        else:  # best_of == 5
            expected_sets = 3.5 + abs(p1_prob - 0.5) * 1.5  # Rough approximation

        return {
            "p1_probability": p1_prob,
            "p2_probability": 1 - p1_prob,
            "expected_sets": round(expected_sets, 1),
            "most_likely_score": f"{sets_to_win}-{sets_to_win - 1}" if abs(p1_prob - 0.5) < 0.15 else f"{sets_to_win}-{max(0, sets_to_win - 2)}",
            "confidence": prediction.confidence
        }
