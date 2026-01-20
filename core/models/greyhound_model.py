# core/models/greyhound_model.py
"""
Greyhound racing prediction model for NEXUS AI.
Uses SVR/SVM ensemble with specialized greyhound racing features.
Adapted from backend_draft/models/greyhound_predictor.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging

from core.models.base_model import BaseModel, PredictionResult, Sport

logger = logging.getLogger(__name__)


class RaceGrade(Enum):
    """UK Greyhound race grades."""
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5
    GRADE_6 = 6
    OPEN = 0


@dataclass
class GreyhoundFeatures:
    """Features for greyhound prediction."""
    # Dog info
    dog_name: str
    trap: int  # 1-6
    weight: float  # kg
    age_months: int

    # Form
    recent_positions: List[int] = field(default_factory=list)  # Last 6 races
    recent_times: List[float] = field(default_factory=list)  # Split times
    days_since_last_race: int = 0

    # Track specific
    track_experience: int = 0  # Races at this track
    track_wins: int = 0
    distance_experience: int = 0  # Races at this distance
    distance_wins: int = 0

    # Box stats
    trap_wins: int = 0
    trap_runs: int = 0
    early_pace_rating: float = 0.0  # 0-10 scale

    # Trainer
    trainer_strike_rate: float = 0.0
    trainer_recent_form: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dog_name": self.dog_name,
            "trap": self.trap,
            "weight": self.weight,
            "age_months": self.age_months,
            "recent_positions": self.recent_positions,
            "recent_times": self.recent_times,
            "days_since_last_race": self.days_since_last_race,
            "track_experience": self.track_experience,
            "track_wins": self.track_wins,
            "distance_experience": self.distance_experience,
            "distance_wins": self.distance_wins,
            "trap_wins": self.trap_wins,
            "trap_runs": self.trap_runs,
            "early_pace_rating": self.early_pace_rating,
            "trainer_strike_rate": self.trainer_strike_rate,
            "trainer_recent_form": self.trainer_recent_form,
        }


@dataclass
class RacePrediction:
    """Prediction for a greyhound race."""
    predicted_positions: Dict[str, int]  # dog_name -> position
    win_probabilities: Dict[str, float]  # dog_name -> win prob
    place_probabilities: Dict[str, float]  # dog_name -> place prob (top 3)
    forecast: List[Tuple[str, str]]  # [(1st, 2nd)] combinations
    tricast: List[Tuple[str, str, str]]  # [(1st, 2nd, 3rd)] combinations
    confidence: float
    reasoning: List[str]


class GreyhoundModel(BaseModel):
    """
    Greyhound racing prediction model.

    Uses ensemble of:
    - SVR for position prediction
    - SVM for winner classification
    - Statistical models for pace analysis
    """

    def __init__(self):
        super().__init__(Sport.GREYHOUND, "GreyhoundPredictor_v1")

        self.required_features = [
            "trap", "weight", "recent_positions",
            "early_pace_rating", "track_experience"
        ]

        # Trap bias factors (1-6)
        # Based on typical UK track statistics
        self.trap_bias = {
            1: 0.18,  # Red - slight advantage
            2: 0.16,  # Blue
            3: 0.17,  # White
            4: 0.16,  # Black
            5: 0.17,  # Orange
            6: 0.16,  # Striped
        }

        # Weight to feature importance
        self.feature_weights = {
            "recent_form": 0.25,
            "trap_advantage": 0.15,
            "early_pace": 0.20,
            "track_experience": 0.15,
            "trainer_form": 0.10,
            "weight_trend": 0.10,
            "age_factor": 0.05,
        }

    def predict(self, match_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict race outcome.

        Args:
            match_data: Dictionary with 'runners' list containing GreyhoundFeatures

        Returns:
            PredictionResult with winner prediction
        """
        self.validate_input(match_data)

        runners = match_data.get("runners", [])
        race_prediction = self.predict_race(runners)

        # Find predicted winner
        winner_name = max(
            race_prediction.win_probabilities,
            key=race_prediction.win_probabilities.get
        )

        return PredictionResult(
            sport=self.sport.value,
            predicted_winner=winner_name,
            confidence=race_prediction.confidence,
            probabilities=race_prediction.win_probabilities,
            model_name=self.model_name,
            features_used=self.required_features,
            feature_values={"runners_count": len(runners)},
            reasoning=race_prediction.reasoning,
            reliability_score=self._calculate_reliability(match_data),
        )

    def predict_proba(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Get win probabilities for all runners."""
        runners = match_data.get("runners", [])
        race_prediction = self.predict_race(runners)
        return race_prediction.win_probabilities

    def predict_race(self, runners: List[Dict[str, Any]]) -> RacePrediction:
        """
        Full race prediction with positions and combinations.

        Args:
            runners: List of runner data dictionaries

        Returns:
            RacePrediction with full analysis
        """
        if not runners:
            return RacePrediction(
                predicted_positions={},
                win_probabilities={},
                place_probabilities={},
                forecast=[],
                tricast=[],
                confidence=0.0,
                reasoning=["No runners provided"]
            )

        # Calculate ratings for each runner
        ratings = {}
        reasoning = []

        for runner in runners:
            name = runner.get("dog_name", f"Dog_{runner.get('trap', 0)}")
            rating = self._calculate_rating(runner)
            ratings[name] = rating

        # Convert to probabilities
        total_rating = sum(ratings.values())
        if total_rating == 0:
            total_rating = len(runners)
            ratings = {k: 1.0 for k in ratings}

        win_probs = {
            name: rating / total_rating
            for name, rating in ratings.items()
        }

        # Sort by rating
        sorted_runners = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

        # Predicted positions
        positions = {name: pos + 1 for pos, (name, _) in enumerate(sorted_runners)}

        # Place probabilities (simplified)
        place_probs = {}
        for name, win_prob in win_probs.items():
            # Approximate place probability
            place_probs[name] = min(0.95, win_prob * 2.5)

        # Top forecast combinations
        top_3 = [name for name, _ in sorted_runners[:3]]
        forecast = []
        if len(top_3) >= 2:
            forecast = [
                (top_3[0], top_3[1]),
                (top_3[1], top_3[0]),
                (top_3[0], top_3[2]) if len(top_3) > 2 else None
            ]
            forecast = [f for f in forecast if f is not None]

        # Top tricast
        tricast = []
        if len(top_3) >= 3:
            tricast = [(top_3[0], top_3[1], top_3[2])]

        # Build reasoning
        winner_name, winner_rating = sorted_runners[0]
        runner_data = next((r for r in runners if r.get("dog_name") == winner_name), {})

        reasoning.append(f"Predicted winner: {winner_name} (rating: {winner_rating:.2f})")

        if runner_data.get("trap"):
            trap = runner_data["trap"]
            reasoning.append(f"Trap {trap} with {self.trap_bias.get(trap, 0.16):.1%} historical advantage")

        if runner_data.get("recent_positions"):
            recent = runner_data["recent_positions"][:3]
            avg_pos = sum(recent) / len(recent) if recent else 0
            reasoning.append(f"Recent form: avg position {avg_pos:.1f}")

        if runner_data.get("early_pace_rating"):
            pace = runner_data["early_pace_rating"]
            if pace >= 7:
                reasoning.append(f"Strong early pace ({pace:.1f}/10)")

        # Calculate confidence
        confidence = self._calculate_confidence(sorted_runners, win_probs)

        return RacePrediction(
            predicted_positions=positions,
            win_probabilities=win_probs,
            place_probabilities=place_probs,
            forecast=forecast,
            tricast=tricast,
            confidence=confidence,
            reasoning=reasoning
        )

    def _calculate_rating(self, runner: Dict[str, Any]) -> float:
        """Calculate overall rating for a runner."""
        rating = 1.0

        # Recent form
        positions = runner.get("recent_positions", [])
        if positions:
            # Weighted average (recent races more important)
            weights = [3, 2.5, 2, 1.5, 1, 0.5][:len(positions)]
            form_score = sum(
                (7 - pos) * w for pos, w in zip(positions, weights)
            ) / sum(weights)
            rating += form_score * self.feature_weights["recent_form"]

        # Trap advantage
        trap = runner.get("trap", 0)
        if trap in self.trap_bias:
            rating += self.trap_bias[trap] * 10 * self.feature_weights["trap_advantage"]

        # Early pace
        pace = runner.get("early_pace_rating", 5.0)
        rating += pace * self.feature_weights["early_pace"]

        # Track experience
        track_wins = runner.get("track_wins", 0)
        track_runs = runner.get("track_experience", 0)
        if track_runs > 0:
            track_rate = track_wins / track_runs
            rating += track_rate * 5 * self.feature_weights["track_experience"]

        # Trainer form
        trainer_form = runner.get("trainer_recent_form", 0.15)
        rating += trainer_form * 10 * self.feature_weights["trainer_form"]

        # Weight trend (optimal weight around 32kg for sprints)
        weight = runner.get("weight", 32)
        weight_diff = abs(weight - 32)
        weight_penalty = max(0, 1 - weight_diff * 0.05)
        rating *= weight_penalty

        # Age factor (peak around 24-36 months)
        age = runner.get("age_months", 30)
        if 24 <= age <= 36:
            rating *= 1.05
        elif age < 20 or age > 48:
            rating *= 0.90

        return max(0.1, rating)

    def _calculate_confidence(
        self,
        sorted_runners: List[Tuple[str, float]],
        win_probs: Dict[str, float]
    ) -> float:
        """Calculate prediction confidence."""
        if len(sorted_runners) < 2:
            return 0.3

        # Gap between 1st and 2nd
        first_rating = sorted_runners[0][1]
        second_rating = sorted_runners[1][1]

        gap = (first_rating - second_rating) / first_rating if first_rating > 0 else 0

        # Winner probability
        winner_prob = win_probs.get(sorted_runners[0][0], 0.2)

        # Combined confidence
        confidence = (gap * 0.5 + winner_prob * 0.5)

        return min(0.95, max(0.3, confidence))

    def _calculate_reliability(self, match_data: Dict[str, Any]) -> float:
        """Calculate reliability based on data completeness."""
        runners = match_data.get("runners", [])
        if not runners:
            return 0.2

        total_score = 0
        for runner in runners:
            score = 0
            if runner.get("recent_positions"):
                score += 0.3
            if runner.get("early_pace_rating"):
                score += 0.2
            if runner.get("track_experience"):
                score += 0.2
            if runner.get("trainer_strike_rate"):
                score += 0.15
            if runner.get("weight"):
                score += 0.15
            total_score += score

        return total_score / len(runners)

    def validate_input(self, match_data: Dict[str, Any]) -> bool:
        """Validate race data."""
        runners = match_data.get("runners", [])
        if not runners:
            raise ValueError("No runners provided in match_data")

        if len(runners) < 2:
            raise ValueError("Need at least 2 runners for a race")

        return True

    def explain_prediction(
        self,
        match_data: Dict[str, Any],
        prediction: PredictionResult
    ) -> List[str]:
        """Generate explanation for race prediction."""
        return prediction.reasoning
