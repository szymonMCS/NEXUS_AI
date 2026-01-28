"""
Handicap prediction model using Gradient Boosting.

Checkpoint: 2.4
Responsibility: Predict handicap/spread outcomes using GBM.
Principle: Calibrated probabilities with proper uncertainty estimation.
"""

import math
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from core.ml.features import FeatureVector
from core.ml.models.interface import MLModelInterface
from core.ml.models.predictions import HandicapPrediction, ModelInfo


logger = logging.getLogger(__name__)


@dataclass
class GBMParameters:
    """Parameters for Gradient Boosting model."""
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_samples_leaf: int = 10

    # Prediction parameters
    margin_std: float = 1.7  # Standard deviation of margins
    home_advantage_goals: float = 0.35  # Base home advantage in goals

    # Calibration
    calibrate_probabilities: bool = True


class HandicapModel(MLModelInterface[HandicapPrediction]):
    """
    Model Gradient Boosting dla predykcji handicap.

    Używa prostszego podejścia bez zewnętrznych bibliotek ML:
    - Regresja dla expected margin
    - Rozkład normalny dla prawdopodobieństw cover
    - Kalibracja na danych treningowych

    W produkcji można podmienić na sklearn/lightgbm.
    """

    VERSION = "1.0.0"

    def __init__(self, params: Optional[GBMParameters] = None):
        self._params = params or GBMParameters()
        self._trained = False
        self._training_samples = 0
        self._trained_at: Optional[datetime] = None
        self._metrics: Dict[str, float] = {}
        self._feature_names: List[str] = []

        # Learned weights (simple linear model as baseline)
        self._feature_weights: Dict[str, float] = {}
        self._intercept: float = 0.0

        # Calibration parameters
        self._calibration_a: float = 1.0  # Platt scaling
        self._calibration_b: float = 0.0

    @property
    def name(self) -> str:
        return "gbm_handicap"

    @property
    def version(self) -> str:
        return self.VERSION

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, features: FeatureVector) -> HandicapPrediction:
        """Predict handicap outcomes."""
        # Calculate expected margin
        expected_margin = self._calculate_expected_margin(features)

        # Calculate cover probabilities
        probs = self._calculate_cover_probabilities(expected_margin)

        # Calculate 1X2 probabilities
        match_probs = self._calculate_1x2_probabilities(expected_margin)

        # Confidence
        confidence = self._calculate_confidence(features)

        # Reasoning
        reasoning = self._generate_reasoning(expected_margin, features)

        return HandicapPrediction(
            expected_margin=round(expected_margin, 3),
            home_cover_minus_15=round(probs["home_-1.5"], 3),
            home_cover_minus_05=round(probs["home_-0.5"], 3),
            home_cover_plus_05=round(probs["home_+0.5"], 3),
            home_cover_plus_15=round(probs["home_+1.5"], 3),
            away_cover_minus_15=round(probs["away_-1.5"], 3),
            away_cover_plus_15=round(probs["away_+1.5"], 3),
            home_win_prob=round(match_probs["home"], 3),
            draw_prob=round(match_probs["draw"], 3),
            away_win_prob=round(match_probs["away"], 3),
            confidence=round(confidence, 3),
            model_version=self.version,
            reasoning=reasoning,
        )

    def predict_batch(self, features_list: List[FeatureVector]) -> List[HandicapPrediction]:
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
            targets: List of actual margins (home_goals - away_goals)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")

        if len(features) < 20:
            raise ValueError("Need at least 20 samples to train")

        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]

        # Learn weights using simple linear regression
        self._fit_linear_model(train_features, train_targets)

        # Calibrate probabilities
        if self._params.calibrate_probabilities:
            self._calibrate(train_features, train_targets)

        # Evaluate
        metrics = self._evaluate(val_features, val_targets)

        # Store training info
        self._trained = True
        self._training_samples = len(features)
        self._trained_at = datetime.utcnow()
        self._metrics = metrics
        self._feature_names = features[0].get_feature_names() if features else []

        logger.info(f"HandicapModel trained on {len(features)} samples. Metrics: {metrics}")

        return metrics

    def save(self, path: Path) -> bool:
        """Save model to file."""
        try:
            data = {
                "version": self.version,
                "params": {
                    "margin_std": self._params.margin_std,
                    "home_advantage_goals": self._params.home_advantage_goals,
                },
                "trained": self._trained,
                "training_samples": self._training_samples,
                "trained_at": self._trained_at.isoformat() if self._trained_at else None,
                "metrics": self._metrics,
                "feature_names": self._feature_names,
                "feature_weights": self._feature_weights,
                "intercept": self._intercept,
                "calibration_a": self._calibration_a,
                "calibration_b": self._calibration_b,
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"HandicapModel saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Load model from file."""
        try:
            with open(path) as f:
                data = json.load(f)

            params = data.get("params", {})
            self._params.margin_std = params.get("margin_std", 1.7)
            self._params.home_advantage_goals = params.get("home_advantage_goals", 0.35)

            self._trained = data.get("trained", False)
            self._training_samples = data.get("training_samples", 0)
            self._trained_at = datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None
            self._metrics = data.get("metrics", {})
            self._feature_names = data.get("feature_names", [])
            self._feature_weights = data.get("feature_weights", {})
            self._intercept = data.get("intercept", 0.0)
            self._calibration_a = data.get("calibration_a", 1.0)
            self._calibration_b = data.get("calibration_b", 0.0)

            logger.info(f"HandicapModel loaded from {path}")
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
            "handicap_attack_diff",
            "handicap_defense_diff",
            "handicap_form_diff",
        ]

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from weights."""
        if not self._feature_weights:
            return None

        # Normalize weights to importance scores
        total = sum(abs(w) for w in self._feature_weights.values())
        if total == 0:
            return None

        return {k: abs(v) / total for k, v in self._feature_weights.items()}

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _calculate_expected_margin(self, features: FeatureVector) -> float:
        """Calculate expected margin (home goals - away goals)."""
        f = features.features

        if self._trained and self._feature_weights:
            # Use learned weights
            margin = self._intercept
            for name, weight in self._feature_weights.items():
                margin += weight * f.get(name, 0.0)
        else:
            # Use heuristic approach
            attack_diff = f.get("handicap_attack_diff", 0.0)
            defense_diff = f.get("handicap_defense_diff", 0.0)
            form_diff = f.get("handicap_form_diff", 0.0)
            elo_diff = f.get("handicap_elo_diff", 0.0)

            # Combined strength
            strength = (
                attack_diff * 0.35 +
                defense_diff * 0.25 +
                form_diff * 0.2 +
                elo_diff * 0.2
            )

            # Add home advantage
            margin = self._params.home_advantage_goals + strength

        return margin

    def _calculate_cover_probabilities(self, expected_margin: float) -> Dict[str, float]:
        """Calculate probabilities for various handicap lines."""
        std = self._params.margin_std

        # Using normal distribution CDF
        probs = {}

        for line in [-1.5, -0.5, 0.5, 1.5]:
            # P(margin > line) = P(Z > (line - expected) / std)
            z = (line - expected_margin) / std
            home_cover = 1 - self._normal_cdf(z)

            # Apply calibration
            if self._params.calibrate_probabilities:
                home_cover = self._calibrate_prob(home_cover)

            probs[f"home_{line:+.1f}"] = home_cover

        # Away covers
        probs["away_-1.5"] = 1 - probs["home_+1.5"]
        probs["away_+1.5"] = 1 - probs["home_-1.5"]

        return probs

    def _calculate_1x2_probabilities(self, expected_margin: float) -> Dict[str, float]:
        """Calculate match outcome probabilities."""
        std = self._params.margin_std

        # Home win: margin > 0.5
        # Draw: -0.5 < margin < 0.5
        # Away win: margin < -0.5

        home_win = 1 - self._normal_cdf((0.5 - expected_margin) / std)
        away_win = self._normal_cdf((-0.5 - expected_margin) / std)
        draw = 1 - home_win - away_win

        # Ensure non-negative
        draw = max(0, draw)

        # Normalize
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total

        return {
            "home": home_win,
            "draw": draw,
            "away": away_win,
        }

    def _normal_cdf(self, z: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _calibrate_prob(self, prob: float) -> float:
        """Apply Platt scaling calibration."""
        # Logit transform
        if prob <= 0:
            prob = 0.001
        if prob >= 1:
            prob = 0.999

        logit = math.log(prob / (1 - prob))
        calibrated_logit = self._calibration_a * logit + self._calibration_b
        return 1 / (1 + math.exp(-calibrated_logit))

    def _calculate_confidence(self, features: FeatureVector) -> float:
        """Calculate prediction confidence."""
        confidence = 0.4  # Base

        if self._trained:
            confidence += 0.25

        # Feature availability
        has_key_features = all(
            features.has_feature(f) for f in [
                "handicap_attack_diff",
                "handicap_form_diff",
            ]
        )
        if has_key_features:
            confidence += 0.15

        # Data quality
        if features.quality.is_sufficient:
            confidence += 0.1

        if features.quality.has_h2h:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_reasoning(self, expected_margin: float, features: FeatureVector) -> str:
        """Generate human-readable reasoning."""
        f = features.features
        parts = []

        # Margin interpretation
        if expected_margin > 1.0:
            parts.append(f"Strong home favorite (exp. margin: {expected_margin:+.1f})")
        elif expected_margin > 0.3:
            parts.append(f"Slight home advantage (exp. margin: {expected_margin:+.1f})")
        elif expected_margin > -0.3:
            parts.append(f"Even match expected (exp. margin: {expected_margin:+.1f})")
        elif expected_margin > -1.0:
            parts.append(f"Slight away advantage (exp. margin: {expected_margin:+.1f})")
        else:
            parts.append(f"Strong away favorite (exp. margin: {expected_margin:+.1f})")

        # Key factors
        attack_diff = f.get("handicap_attack_diff", 0)
        if abs(attack_diff) > 0.3:
            team = "Home" if attack_diff > 0 else "Away"
            parts.append(f"{team} has stronger attack")

        form_diff = f.get("handicap_form_diff", 0)
        if abs(form_diff) > 0.2:
            team = "Home" if form_diff > 0 else "Away"
            parts.append(f"{team} in better form")

        return ". ".join(parts) + "." if parts else "Standard prediction."

    def _fit_linear_model(
        self,
        features: List[FeatureVector],
        targets: List[float],
    ) -> None:
        """Fit simple linear regression model."""
        if not features:
            return

        # Key features for linear model
        key_features = [
            "handicap_attack_diff",
            "handicap_defense_diff",
            "handicap_overall_strength_diff",
            "handicap_form_diff",
            "handicap_elo_diff",
            "handicap_h2h_avg_margin",
        ]

        # Build feature matrix
        X = []
        for fv in features:
            row = [fv.features.get(f, 0.0) for f in key_features]
            X.append(row)

        # Simple gradient descent for linear regression
        n_features = len(key_features)
        weights = [0.0] * n_features
        intercept = sum(targets) / len(targets) if targets else 0.0
        learning_rate = 0.01

        for _ in range(100):  # Iterations
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for x, y in zip(X, targets):
                pred = intercept + sum(w * xi for w, xi in zip(weights, x))
                error = pred - y

                for i in range(n_features):
                    grad_w[i] += error * x[i]
                grad_b += error

            # Update
            for i in range(n_features):
                weights[i] -= learning_rate * grad_w[i] / len(targets)
            intercept -= learning_rate * grad_b / len(targets)

        # Store
        self._feature_weights = dict(zip(key_features, weights))
        self._intercept = intercept

        logger.debug(f"Fitted weights: {self._feature_weights}")

    def _calibrate(
        self,
        features: List[FeatureVector],
        targets: List[float],
    ) -> None:
        """Calibrate probability predictions using Platt scaling."""
        # For simplicity, just adjust standard deviation
        predictions = [self._calculate_expected_margin(f) for f in features]

        if predictions and targets:
            # Calculate actual std of margins
            errors = [p - t for p, t in zip(predictions, targets)]
            if len(errors) > 1:
                mean_error = sum(errors) / len(errors)
                var = sum((e - mean_error) ** 2 for e in errors) / (len(errors) - 1)
                actual_std = math.sqrt(var) if var > 0 else 1.7
                self._params.margin_std = actual_std

    def _evaluate(
        self,
        features: List[FeatureVector],
        targets: List[float],
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if not features:
            return {}

        predictions = self.predict_batch(features)

        # MAE for margin
        mae = sum(abs(p.expected_margin - t) for p, t in zip(predictions, targets)) / len(targets)

        # 1X2 accuracy
        correct_1x2 = 0
        for pred, actual_margin in zip(predictions, targets):
            if actual_margin > 0.5 and pred.predicted_winner == "home":
                correct_1x2 += 1
            elif actual_margin < -0.5 and pred.predicted_winner == "away":
                correct_1x2 += 1
            elif -0.5 <= actual_margin <= 0.5 and pred.predicted_winner == "draw":
                correct_1x2 += 1

        # Handicap accuracy (for -1.5 line)
        correct_handicap = 0
        for pred, actual_margin in zip(predictions, targets):
            if actual_margin > 1.5 and pred.home_cover_minus_15 > 0.5:
                correct_handicap += 1
            elif actual_margin <= 1.5 and pred.home_cover_minus_15 <= 0.5:
                correct_handicap += 1

        n = len(targets)
        return {
            "mae_margin": mae,
            "accuracy_1x2": correct_1x2 / n,
            "accuracy_handicap_15": correct_handicap / n,
        }
