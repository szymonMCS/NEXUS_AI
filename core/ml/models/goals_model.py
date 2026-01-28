"""
Goals prediction model using Poisson regression.

Checkpoint: 2.3
Responsibility: Predict total goals using Poisson distribution.
Principle: Statistically sound model with proper uncertainty quantification.
"""

import math
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

from core.ml.features import FeatureVector
from core.ml.models.interface import MLModelInterface
from core.ml.models.predictions import GoalsPrediction, ModelInfo


logger = logging.getLogger(__name__)


@dataclass
class PoissonParameters:
    """Parameters for Poisson model."""
    # Base rates
    league_avg_goals: float = 1.35  # Average goals per team per match

    # Home advantage
    home_advantage: float = 1.1  # Multiplier for home team

    # Feature weights for lambda calculation
    attack_weight: float = 0.4
    defense_weight: float = 0.4
    form_weight: float = 0.15
    h2h_weight: float = 0.05

    # Bounds
    min_lambda: float = 0.3
    max_lambda: float = 4.0


class GoalsModel(MLModelInterface[GoalsPrediction]):
    """
    Model Poisson dla predykcji liczby bramek.

    Używa rozkładu Poissona do modelowania liczby bramek:
    - P(goals = k) = (λ^k * e^(-λ)) / k!

    Lambda (expected goals) jest obliczana na podstawie:
    - Siły ataku drużyny strzelającej
    - Siły obrony drużyny broniącej
    - Średniej ligowej
    - Przewagi własnego boiska
    - Formy i danych H2H
    """

    VERSION = "1.0.0"

    def __init__(self, params: Optional[PoissonParameters] = None):
        self._params = params or PoissonParameters()
        self._trained = False
        self._training_samples = 0
        self._trained_at: Optional[datetime] = None
        self._metrics: Dict[str, float] = {}
        self._feature_names: List[str] = []

        # Learned adjustments (updated during training)
        self._attack_adjustments: Dict[str, float] = {}
        self._defense_adjustments: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "poisson_goals"

    @property
    def version(self) -> str:
        return self.VERSION

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, features: FeatureVector) -> GoalsPrediction:
        """
        Predict goals for a match.

        Even without training, uses base parameters for prediction.
        """
        # Calculate expected goals (lambdas)
        home_lambda, away_lambda = self._calculate_lambdas(features)

        # Calculate probabilities for various thresholds
        probs = self._calculate_probabilities(home_lambda, away_lambda)

        # Calculate confidence based on data quality and feature availability
        confidence = self._calculate_confidence(features)

        # Generate reasoning
        reasoning = self._generate_reasoning(home_lambda, away_lambda, features)

        return GoalsPrediction(
            home_expected=round(home_lambda, 3),
            away_expected=round(away_lambda, 3),
            total_expected=round(home_lambda + away_lambda, 3),
            over_15_prob=round(probs["over_1.5"], 3),
            under_15_prob=round(probs["under_1.5"], 3),
            over_25_prob=round(probs["over_2.5"], 3),
            under_25_prob=round(probs["under_2.5"], 3),
            over_35_prob=round(probs["over_3.5"], 3),
            under_35_prob=round(probs["under_3.5"], 3),
            confidence=round(confidence, 3),
            model_version=self.version,
            reasoning=reasoning,
            score_matrix=self._calculate_score_matrix(home_lambda, away_lambda),
        )

    def predict_batch(self, features_list: List[FeatureVector]) -> List[GoalsPrediction]:
        """Predict for multiple matches."""
        return [self.predict(f) for f in features_list]

    def train(
        self,
        features: List[FeatureVector],
        targets: List[Any],
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train model on historical data.

        Args:
            features: List of feature vectors
            targets: List of tuples (home_goals, away_goals)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")

        if len(features) < 10:
            raise ValueError("Need at least 10 samples to train")

        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]

        # Learn parameters using MLE approach
        self._fit_parameters(train_features, train_targets)

        # Evaluate on validation set
        metrics = self._evaluate(val_features, val_targets)

        # Store training info
        self._trained = True
        self._training_samples = len(features)
        self._trained_at = datetime.utcnow()
        self._metrics = metrics
        self._feature_names = features[0].get_feature_names() if features else []

        logger.info(f"GoalsModel trained on {len(features)} samples. Metrics: {metrics}")

        return metrics

    def save(self, path: Path) -> bool:
        """Save model to file."""
        try:
            data = {
                "version": self.version,
                "params": {
                    "league_avg_goals": self._params.league_avg_goals,
                    "home_advantage": self._params.home_advantage,
                    "attack_weight": self._params.attack_weight,
                    "defense_weight": self._params.defense_weight,
                    "form_weight": self._params.form_weight,
                    "h2h_weight": self._params.h2h_weight,
                },
                "trained": self._trained,
                "training_samples": self._training_samples,
                "trained_at": self._trained_at.isoformat() if self._trained_at else None,
                "metrics": self._metrics,
                "feature_names": self._feature_names,
                "attack_adjustments": self._attack_adjustments,
                "defense_adjustments": self._defense_adjustments,
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"GoalsModel saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Load model from file."""
        try:
            with open(path) as f:
                data = json.load(f)

            # Restore parameters
            params = data.get("params", {})
            self._params = PoissonParameters(
                league_avg_goals=params.get("league_avg_goals", 1.35),
                home_advantage=params.get("home_advantage", 1.1),
                attack_weight=params.get("attack_weight", 0.4),
                defense_weight=params.get("defense_weight", 0.4),
                form_weight=params.get("form_weight", 0.15),
                h2h_weight=params.get("h2h_weight", 0.05),
            )

            self._trained = data.get("trained", False)
            self._training_samples = data.get("training_samples", 0)
            self._trained_at = datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None
            self._metrics = data.get("metrics", {})
            self._feature_names = data.get("feature_names", [])
            self._attack_adjustments = data.get("attack_adjustments", {})
            self._defense_adjustments = data.get("defense_adjustments", {})

            logger.info(f"GoalsModel loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name=self.name,
            version=self.version,
            trained_at=self._trained_at or datetime.utcnow(),
            training_samples=self._training_samples,
            metrics=self._metrics,
            feature_names=self._feature_names,
        )

    def get_required_features(self) -> List[str]:
        """Features required for prediction."""
        return [
            "goals_home_attack_strength",
            "goals_home_defense_strength",
            "goals_away_attack_strength",
            "goals_away_defense_strength",
        ]

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _calculate_lambdas(self, features: FeatureVector) -> Tuple[float, float]:
        """Calculate expected goals (lambda) for each team."""
        f = features.features

        # Get attack/defense strengths
        home_attack = f.get("goals_home_attack_strength", 1.0)
        home_defense = f.get("goals_home_defense_strength", 1.0)
        away_attack = f.get("goals_away_attack_strength", 1.0)
        away_defense = f.get("goals_away_defense_strength", 1.0)

        # Form factors
        home_form = f.get("form_home_form_points", 0.5)
        away_form = f.get("form_away_form_points", 0.5)

        # H2H factor
        h2h_avg = f.get("goals_h2h_avg_total_goals", 2.7)
        h2h_factor = h2h_avg / 2.7  # Relative to average

        # Calculate base lambdas
        # Home expected = home_attack * away_defense * league_avg * home_advantage
        home_base = (
            home_attack *
            away_defense *
            self._params.league_avg_goals *
            self._params.home_advantage
        )

        away_base = (
            away_attack *
            home_defense *
            self._params.league_avg_goals
        )

        # Apply form adjustment
        form_adj_home = 1.0 + (home_form - 0.5) * 0.2  # ±10% based on form
        form_adj_away = 1.0 + (away_form - 0.5) * 0.2

        # Apply H2H adjustment
        h2h_adj = 1.0 + (h2h_factor - 1.0) * self._params.h2h_weight

        # Final lambdas
        home_lambda = home_base * form_adj_home * h2h_adj
        away_lambda = away_base * form_adj_away * h2h_adj

        # Clip to reasonable bounds
        home_lambda = max(self._params.min_lambda, min(self._params.max_lambda, home_lambda))
        away_lambda = max(self._params.min_lambda, min(self._params.max_lambda, away_lambda))

        return home_lambda, away_lambda

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function."""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

    def _calculate_probabilities(
        self,
        home_lambda: float,
        away_lambda: float,
        max_goals: int = 10,
    ) -> Dict[str, float]:
        """Calculate over/under probabilities."""
        # Build probability matrix
        probs = {}

        # Calculate cumulative probabilities
        for threshold in [1.5, 2.5, 3.5]:
            under_prob = 0.0

            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    total = h + a
                    if total <= threshold:
                        prob = self._poisson_pmf(h, home_lambda) * self._poisson_pmf(a, away_lambda)
                        under_prob += prob

            probs[f"under_{threshold}".replace(".", "")] = under_prob
            probs[f"over_{threshold}".replace(".", "")] = 1 - under_prob

        # Rename keys
        return {
            "over_1.5": probs["over_15"],
            "under_1.5": probs["under_15"],
            "over_2.5": probs["over_25"],
            "under_2.5": probs["under_25"],
            "over_3.5": probs["over_35"],
            "under_3.5": probs["under_35"],
        }

    def _calculate_score_matrix(
        self,
        home_lambda: float,
        away_lambda: float,
        max_goals: int = 6,
    ) -> Dict[str, float]:
        """Calculate probability matrix for exact scores."""
        matrix = {}

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = self._poisson_pmf(h, home_lambda) * self._poisson_pmf(a, away_lambda)
                if prob > 0.01:  # Only include significant probabilities
                    matrix[f"{h}-{a}"] = round(prob, 4)

        return matrix

    def _calculate_confidence(self, features: FeatureVector) -> float:
        """Calculate prediction confidence."""
        confidence = 0.5  # Base confidence

        # Boost for training
        if self._trained:
            confidence += 0.2

        # Data quality boost
        if features.quality.is_sufficient:
            confidence += 0.1

        if features.quality.has_h2h:
            confidence += 0.1

        if features.quality.has_form:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_reasoning(
        self,
        home_lambda: float,
        away_lambda: float,
        features: FeatureVector,
    ) -> str:
        """Generate human-readable reasoning."""
        f = features.features
        total = home_lambda + away_lambda

        parts = []

        # Expected goals analysis
        if total > 3.0:
            parts.append(f"High-scoring expected ({total:.1f} total)")
        elif total < 2.0:
            parts.append(f"Low-scoring expected ({total:.1f} total)")
        else:
            parts.append(f"Average scoring expected ({total:.1f} total)")

        # Team analysis
        home_attack = f.get("goals_home_attack_strength", 1.0)
        away_defense = f.get("goals_away_defense_strength", 1.0)

        if home_attack > 1.2:
            parts.append("Strong home attack")
        if away_defense > 1.3:
            parts.append("Weak away defense")

        # H2H
        h2h_over = f.get("goals_h2h_over25_ratio", 0.5)
        if h2h_over > 0.7:
            parts.append("H2H suggests over 2.5")
        elif h2h_over < 0.3:
            parts.append("H2H suggests under 2.5")

        return ". ".join(parts) + "." if parts else "Standard prediction."

    def _fit_parameters(
        self,
        features: List[FeatureVector],
        targets: List[Tuple[int, int]],
    ) -> None:
        """Fit model parameters using training data."""
        # Simple MLE: adjust league average based on actual data
        total_home_goals = sum(t[0] for t in targets)
        total_away_goals = sum(t[1] for t in targets)
        n_matches = len(targets)

        if n_matches > 0:
            self._params.league_avg_goals = (total_home_goals + total_away_goals) / (2 * n_matches)

            # Calculate home advantage from data
            home_avg = total_home_goals / n_matches
            away_avg = total_away_goals / n_matches
            if away_avg > 0:
                self._params.home_advantage = home_avg / away_avg

        logger.debug(f"Fitted params: league_avg={self._params.league_avg_goals:.3f}, home_adv={self._params.home_advantage:.3f}")

    def _evaluate(
        self,
        features: List[FeatureVector],
        targets: List[Tuple[int, int]],
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if not features:
            return {}

        # Predictions
        predictions = self.predict_batch(features)

        # Calculate metrics
        mae_home = 0.0
        mae_away = 0.0
        over25_correct = 0
        under25_correct = 0

        for pred, target in zip(predictions, targets):
            actual_home, actual_away = target
            actual_total = actual_home + actual_away

            mae_home += abs(pred.home_expected - actual_home)
            mae_away += abs(pred.away_expected - actual_away)

            # Over/under accuracy
            if actual_total > 2.5 and pred.over_25_prob > 0.5:
                over25_correct += 1
            elif actual_total <= 2.5 and pred.under_25_prob > 0.5:
                under25_correct += 1

        n = len(features)
        return {
            "mae_home": mae_home / n,
            "mae_away": mae_away / n,
            "mae_total": (mae_home + mae_away) / n,
            "over_under_accuracy": (over25_correct + under25_correct) / n,
        }
