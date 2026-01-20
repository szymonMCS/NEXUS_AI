# core/ensemble.py
"""
Ensemble Manager for NEXUS AI.
Combines predictions from multiple models with various ensemble techniques.
Based on concepts from backend_draft/core/ensemble_manager.py
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

from core.models.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Available ensemble methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STACKING = "stacking"
    VOTING = "voting"
    BAYESIAN = "bayesian"


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction."""
    sport: str
    home_probability: float
    away_probability: float
    ensemble_confidence: float
    individual_predictions: Dict[str, PredictionResult]
    model_weights: Dict[str, float]
    method: EnsembleMethod
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport,
            "home_probability": self.home_probability,
            "away_probability": self.away_probability,
            "ensemble_confidence": self.ensemble_confidence,
            "model_weights": self.model_weights,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
        }


class EnsembleManager:
    """
    Manages multiple prediction models and combines their outputs.

    Features:
    - Register multiple models per sport
    - Multiple ensemble methods (weighted avg, confidence-weighted, bayesian)
    - Dynamic weight updates based on performance
    - Cross-model analysis
    """

    # Default model weights based on predictability
    DEFAULT_WEIGHTS = {
        "tennis": {
            "TennisModel": 0.85,
            "TennisHandicapModel": 0.75,
        },
        "basketball": {
            "BasketballModel": 0.85,
            "BasketballHandicapModel": 0.75,
        },
    }

    def __init__(self):
        """Initialize ensemble manager."""
        self.models: Dict[str, Dict[str, BaseModel]] = {}  # sport -> {name: model}
        self.weights: Dict[str, Dict[str, float]] = {}  # sport -> {name: weight}
        self.performance_history: Dict[str, List[float]] = {}
        self._prediction_count = 0

    def register_model(
        self,
        sport: str,
        model: BaseModel,
        weight: Optional[float] = None
    ):
        """
        Register a model for a sport.

        Args:
            sport: Sport type
            model: Model instance
            weight: Initial weight (0-1)
        """
        if sport not in self.models:
            self.models[sport] = {}
            self.weights[sport] = {}

        model_name = model.name
        self.models[sport][model_name] = model

        # Set weight
        if weight is not None:
            self.weights[sport][model_name] = weight
        else:
            default = self.DEFAULT_WEIGHTS.get(sport, {}).get(model_name, 0.7)
            self.weights[sport][model_name] = default

        logger.info(f"Registered {model_name} for {sport} with weight {self.weights[sport][model_name]:.2f}")

    def register_all_models(self):
        """Register all available models."""
        try:
            from core.models.tennis_model import TennisModel
            self.register_model("tennis", TennisModel())
            logger.info("Registered TennisModel")
        except Exception as e:
            logger.warning(f"Could not register TennisModel: {e}")

        try:
            from core.models.basketball_model import BasketballModel
            self.register_model("basketball", BasketballModel())
            logger.info("Registered BasketballModel")
        except Exception as e:
            logger.warning(f"Could not register BasketballModel: {e}")

        try:
            from core.models.handicap_model import TennisHandicapModel, BasketballHandicapModel
            # These are used for handicap predictions, not moneyline
            logger.info("HandicapModels available for specialized predictions")
        except Exception as e:
            logger.warning(f"Could not load handicap models: {e}")

    def predict(
        self,
        sport: str,
        match_data: Dict[str, Any],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    ) -> Optional[EnsemblePrediction]:
        """
        Generate ensemble prediction for a match.

        Args:
            sport: Sport type
            match_data: Match data dictionary
            method: Ensemble method to use

        Returns:
            EnsemblePrediction or None
        """
        if sport not in self.models or not self.models[sport]:
            logger.warning(f"No models registered for {sport}")
            return None

        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models[sport].items():
            try:
                if model.validate_input(match_data):
                    pred = model.predict(match_data)
                    predictions[model_name] = pred
            except Exception as e:
                logger.debug(f"Model {model_name} failed: {e}")

        if not predictions:
            logger.warning(f"No valid predictions for {sport}")
            return None

        # Combine predictions using selected method
        ensemble_func = self._get_ensemble_function(method)
        result = ensemble_func(sport, predictions)

        self._prediction_count += 1
        return result

    def _get_ensemble_function(self, method: EnsembleMethod) -> Callable:
        """Get ensemble function for method."""
        methods = {
            EnsembleMethod.WEIGHTED_AVERAGE: self._weighted_average,
            EnsembleMethod.CONFIDENCE_WEIGHTED: self._confidence_weighted,
            EnsembleMethod.STACKING: self._stacking,
            EnsembleMethod.VOTING: self._voting,
            EnsembleMethod.BAYESIAN: self._bayesian,
        }
        return methods.get(method, self._weighted_average)

    def _weighted_average(
        self,
        sport: str,
        predictions: Dict[str, PredictionResult]
    ) -> EnsemblePrediction:
        """Weighted average ensemble."""
        total_weight = 0
        weighted_home = 0
        weighted_away = 0
        weighted_confidence = 0

        for model_name, pred in predictions.items():
            weight = self.weights[sport].get(model_name, 0.5)
            weighted_home += pred.home_probability * weight
            weighted_away += pred.away_probability * weight
            weighted_confidence += pred.confidence * weight
            total_weight += weight

        if total_weight > 0:
            home_prob = weighted_home / total_weight
            away_prob = weighted_away / total_weight
            confidence = weighted_confidence / total_weight
        else:
            home_prob = away_prob = 0.5
            confidence = 0.5

        # Normalize probabilities
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        return EnsemblePrediction(
            sport=sport,
            home_probability=home_prob,
            away_probability=away_prob,
            ensemble_confidence=confidence,
            individual_predictions=predictions,
            model_weights=self.weights[sport].copy(),
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            reasoning=[
                f"Combined {len(predictions)} models using weighted average",
                f"Total weight: {total_weight:.2f}",
            ]
        )

    def _confidence_weighted(
        self,
        sport: str,
        predictions: Dict[str, PredictionResult]
    ) -> EnsemblePrediction:
        """Confidence-weighted ensemble."""
        total_weight = 0
        weighted_home = 0
        weighted_away = 0

        for model_name, pred in predictions.items():
            base_weight = self.weights[sport].get(model_name, 0.5)
            # Weight by confidence
            adaptive_weight = base_weight * (1 + pred.confidence) / 2
            weighted_home += pred.home_probability * adaptive_weight
            weighted_away += pred.away_probability * adaptive_weight
            total_weight += adaptive_weight

        if total_weight > 0:
            home_prob = weighted_home / total_weight
            away_prob = weighted_away / total_weight
        else:
            home_prob = away_prob = 0.5

        # Normalize
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        # Ensemble confidence is average of individual confidences
        avg_confidence = np.mean([p.confidence for p in predictions.values()])

        return EnsemblePrediction(
            sport=sport,
            home_probability=home_prob,
            away_probability=away_prob,
            ensemble_confidence=avg_confidence,
            individual_predictions=predictions,
            model_weights={
                name: self.weights[sport].get(name, 0.5) * (1 + pred.confidence) / 2
                for name, pred in predictions.items()
            },
            method=EnsembleMethod.CONFIDENCE_WEIGHTED,
            reasoning=[
                f"Confidence-weighted combination of {len(predictions)} models",
                f"Higher confidence models have more influence",
            ]
        )

    def _stacking(
        self,
        sport: str,
        predictions: Dict[str, PredictionResult]
    ) -> EnsemblePrediction:
        """
        Stacking ensemble with meta-learning.
        Uses sigmoid activation to boost high-confidence predictions.
        """
        # Create feature vector from predictions
        features = []
        for pred in predictions.values():
            features.extend([
                pred.home_probability,
                pred.away_probability,
                pred.confidence,
            ])

        features = np.array(features)

        # Simple meta-model: sigmoid transformation
        # Boost predictions that are confident and agree
        avg_home = np.mean([p.home_probability for p in predictions.values()])
        avg_away = np.mean([p.away_probability for p in predictions.values()])
        avg_conf = np.mean([p.confidence for p in predictions.values()])

        # Agreement factor
        home_std = np.std([p.home_probability for p in predictions.values()])
        agreement = 1 - min(home_std * 2, 0.5)  # Low std = high agreement

        # Meta weight
        meta_weight = 1 / (1 + np.exp(-(avg_conf - 0.5) * 4))  # Sigmoid

        # Adjusted probabilities
        home_prob = avg_home
        away_prob = avg_away

        # Normalize
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        return EnsemblePrediction(
            sport=sport,
            home_probability=home_prob,
            away_probability=away_prob,
            ensemble_confidence=avg_conf * meta_weight * agreement,
            individual_predictions=predictions,
            model_weights={name: meta_weight for name in predictions},
            method=EnsembleMethod.STACKING,
            reasoning=[
                f"Stacking meta-model with {len(predictions)} base models",
                f"Agreement factor: {agreement:.2f}",
                f"Meta weight: {meta_weight:.2f}",
            ]
        )

    def _voting(
        self,
        sport: str,
        predictions: Dict[str, PredictionResult]
    ) -> EnsemblePrediction:
        """Majority voting ensemble."""
        home_votes = 0
        away_votes = 0

        for model_name, pred in predictions.items():
            weight = self.weights[sport].get(model_name, 1.0)
            if pred.home_probability > pred.away_probability:
                home_votes += weight
            else:
                away_votes += weight

        total_votes = home_votes + away_votes
        if total_votes > 0:
            home_prob = home_votes / total_votes
            away_prob = away_votes / total_votes
        else:
            home_prob = away_prob = 0.5

        return EnsemblePrediction(
            sport=sport,
            home_probability=home_prob,
            away_probability=away_prob,
            ensemble_confidence=max(home_prob, away_prob),
            individual_predictions=predictions,
            model_weights=self.weights[sport].copy(),
            method=EnsembleMethod.VOTING,
            reasoning=[
                f"Voting: Home {home_votes:.1f} vs Away {away_votes:.1f}",
            ]
        )

    def _bayesian(
        self,
        sport: str,
        predictions: Dict[str, PredictionResult]
    ) -> EnsemblePrediction:
        """Bayesian ensemble combining prior weights with likelihood."""
        posterior_home = 0.5  # Prior
        posterior_away = 0.5

        for model_name, pred in predictions.items():
            prior = self.weights[sport].get(model_name, 0.5)
            likelihood_home = pred.home_probability
            likelihood_away = pred.away_probability

            # Bayesian update
            posterior_home = (prior * likelihood_home) / (
                prior * likelihood_home + (1 - prior) * (1 - likelihood_home)
            )
            posterior_away = (prior * likelihood_away) / (
                prior * likelihood_away + (1 - prior) * (1 - likelihood_away)
            )

        # Normalize
        total = posterior_home + posterior_away
        if total > 0:
            posterior_home /= total
            posterior_away /= total

        return EnsemblePrediction(
            sport=sport,
            home_probability=posterior_home,
            away_probability=posterior_away,
            ensemble_confidence=(max(posterior_home, posterior_away) - 0.5) * 2,
            individual_predictions=predictions,
            model_weights=self.weights[sport].copy(),
            method=EnsembleMethod.BAYESIAN,
            reasoning=[
                f"Bayesian posterior from {len(predictions)} models",
            ]
        )

    def update_weights(self, sport: str, performance: Dict[str, float]):
        """
        Update model weights based on performance.

        Args:
            sport: Sport type
            performance: Dict of model_name -> accuracy (0-1)
        """
        if sport not in self.weights:
            return

        for model_name, accuracy in performance.items():
            if model_name in self.weights[sport]:
                current = self.weights[sport][model_name]
                # Exponential moving average
                new_weight = 0.8 * current + 0.2 * accuracy
                self.weights[sport][model_name] = max(0.1, min(1.0, new_weight))

                # Track history
                key = f"{sport}_{model_name}"
                if key not in self.performance_history:
                    self.performance_history[key] = []
                self.performance_history[key].append(accuracy)

        logger.info(f"Updated weights for {sport}: {self.weights[sport]}")

    def get_info(self) -> Dict[str, Any]:
        """Get ensemble manager information."""
        return {
            "registered_sports": list(self.models.keys()),
            "models_per_sport": {
                sport: list(models.keys())
                for sport, models in self.models.items()
            },
            "current_weights": self.weights,
            "prediction_count": self._prediction_count,
            "available_methods": [m.value for m in EnsembleMethod],
        }

    def get_model_info(self, sport: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if sport not in self.models or model_name not in self.models[sport]:
            return None

        model = self.models[sport][model_name]
        return {
            "name": model.name,
            "sport": sport,
            "weight": self.weights[sport].get(model_name, 0),
            "performance_history": self.performance_history.get(f"{sport}_{model_name}", []),
        }


# Singleton instance
_ensemble_manager: Optional[EnsembleManager] = None


def get_ensemble_manager() -> EnsembleManager:
    """Get or create ensemble manager singleton."""
    global _ensemble_manager
    if _ensemble_manager is None:
        _ensemble_manager = EnsembleManager()
        _ensemble_manager.register_all_models()
    return _ensemble_manager
