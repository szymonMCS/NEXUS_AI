"""
Ensemble Model Aggregator for NEXUS AI.

Implements dynamic weighting between ML models and statistical models.
Higher weight for ML models when confident, fallback to statistical ensemble.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from core.ml import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Weight configuration for a model."""
    model_name: str
    base_weight: float
    accuracy_score: float = 0.5
    recent_performance: float = 0.5
    confidence_bias: float = 1.0
    
    @property
    def effective_weight(self) -> float:
        """Calculate effective weight based on performance."""
        # Weight = base * accuracy * recent_performance * confidence_bias
        weight = (
            self.base_weight * 
            (0.3 + 0.7 * self.accuracy_score) * 
            (0.3 + 0.7 * self.recent_performance) *
            self.confidence_bias
        )
        return max(0.05, min(0.95, weight))


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction."""
    home_win_prob: float
    away_win_prob: float
    draw_prob: Optional[float]
    confidence: float
    agreement_score: float
    model_contributions: Dict[str, float]
    primary_model: str
    uncertainty: float
    recommended_action: str  # 'bet', 'skip', 'review'


class EnsembleAggregator:
    """
    Aggregates predictions from multiple models with dynamic weighting.
    
    Strategy:
    - ML models get higher base weight (0.6) when confident
    - Statistical models provide fallback (0.4 base weight)
    - Weights adapt based on recent performance
    - Confidence reflects model agreement
    """
    
    # Base weights by model type
    BASE_WEIGHTS = {
        'xgboost': 0.70,
        'random_forest': 0.65,
        'neural_network': 0.68,
        'StatisticalEnsemble': 0.40,
        'EloPredictor': 0.30,
        'FormPredictor': 0.25,
    }
    
    def __init__(self, min_models: int = 1):
        self.min_models = min_models
        self.model_weights: Dict[str, ModelWeight] = {}
        self.performance_history: Dict[str, List[bool]] = defaultdict(list)
        self.max_history = 50
        
        logger.info("EnsembleAggregator initialized")
    
    def register_model(self, model: BasePredictor):
        """Register a model with the ensemble."""
        model_type = model.model_type if hasattr(model, 'model_type') else 'unknown'
        base_weight = self.BASE_WEIGHTS.get(model_type, 0.5)
        
        self.model_weights[model.name] = ModelWeight(
            model_name=model.name,
            base_weight=base_weight,
            accuracy_score=model.accuracy if hasattr(model, 'accuracy') else 0.5
        )
        
        logger.debug(f"Registered model: {model.name} (type={model_type}, base_weight={base_weight})")
    
    def update_performance(self, model_name: str, was_correct: bool):
        """Update model performance after result known."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(was_correct)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > self.max_history:
            self.performance_history[model_name] = self.performance_history[model_name][-self.max_history:]
        
        # Update weight
        if model_name in self.model_weights:
            history = self.performance_history[model_name]
            if len(history) >= 5:
                recent = sum(history[-10:]) / len(history[-10:])
                self.model_weights[model_name].recent_performance = recent
                
                logger.debug(f"Updated {model_name} performance: {recent:.2f} (n={len(history)})")
    
    def aggregate(
        self,
        predictions: Dict[str, PredictionResult],
        features: Dict[str, Any]
    ) -> EnsemblePrediction:
        """
        Aggregate predictions from multiple models.
        
        Args:
            predictions: Dict mapping model_name -> PredictionResult
            features: Input features for context
            
        Returns:
            EnsemblePrediction with aggregated result
        """
        if len(predictions) < self.min_models:
            logger.warning(f"Insufficient models: {len(predictions)} < {self.min_models}")
            return self._fallback_prediction()
        
        # Calculate effective weights
        weights = self._calculate_weights(predictions)
        
        # Weighted aggregation
        weighted_home = 0.0
        weighted_away = 0.0
        weighted_draw = 0.0
        total_weight = 0.0
        
        model_contributions = {}
        confidences = []
        probs_home = []
        probs_away = []
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.1)
            
            weighted_home += pred.home_win_prob * weight
            weighted_away += pred.away_win_prob * weight
            if pred.draw_prob:
                weighted_draw += pred.draw_prob * weight
            
            total_weight += weight
            confidences.append(pred.confidence)
            probs_home.append(pred.home_win_prob)
            probs_away.append(pred.away_win_prob)
            
            model_contributions[model_name] = weight
        
        # Normalize
        if total_weight > 0:
            home_prob = weighted_home / total_weight
            away_prob = weighted_away / total_weight
            draw_prob = weighted_draw / total_weight if weighted_draw > 0 else None
        else:
            home_prob = away_prob = 0.5
            draw_prob = None
        
        # Calculate agreement score (lower variance = higher agreement)
        agreement = self._calculate_agreement(probs_home, probs_away)
        
        # Calculate confidence based on model confidences and agreement
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        confidence = self._blend_confidence(avg_confidence, agreement)
        
        # Calculate uncertainty
        uncertainty = 1.0 - confidence
        
        # Determine primary model (highest weight)
        primary_model = max(weights.items(), key=lambda x: x[1])[0] if weights else "unknown"
        
        # Determine recommended action
        action = self._determine_action(home_prob, away_prob, confidence, uncertainty)
        
        result = EnsemblePrediction(
            home_win_prob=round(home_prob, 4),
            away_win_prob=round(away_prob, 4),
            draw_prob=round(draw_prob, 4) if draw_prob else None,
            confidence=round(confidence, 4),
            agreement_score=round(agreement, 4),
            model_contributions=model_contributions,
            primary_model=primary_model,
            uncertainty=round(uncertainty, 4),
            recommended_action=action
        )
        
        logger.debug(
            f"Ensemble: home={result.home_win_prob:.2f}, "
            f"conf={result.confidence:.2f}, agreement={result.agreement_score:.2f}, "
            f"action={result.recommended_action}"
        )
        
        return result
    
    def _calculate_weights(
        self,
        predictions: Dict[str, PredictionResult]
    ) -> Dict[str, float]:
        """Calculate effective weights for each model."""
        weights = {}
        
        for model_name in predictions.keys():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name].effective_weight
            else:
                # Unknown model, assign default weight
                weight = 0.3
            
            # Boost ML models when they have high confidence
            pred = predictions[model_name]
            if pred.confidence > 0.7:
                weight *= 1.2  # Boost confident predictions
            elif pred.confidence < 0.3:
                weight *= 0.8  # Reduce uncertain predictions
            
            weights[model_name] = weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calculate_agreement(self, probs_home: List[float], probs_away: List[float]) -> float:
        """Calculate agreement score between models."""
        if len(probs_home) < 2:
            return 0.5
        
        # Calculate variance
        import statistics
        try:
            variance = statistics.variance(probs_home)
            # Convert variance to agreement (0-1 scale)
            # Lower variance = higher agreement
            agreement = max(0.0, 1.0 - variance * 4)
            return agreement
        except:
            return 0.5
    
    def _blend_confidence(self, model_confidence: float, agreement: float) -> float:
        """Blend model confidence with agreement score."""
        # 60% model confidence, 40% agreement
        blended = 0.6 * model_confidence + 0.4 * agreement
        
        # Penalize low agreement
        if agreement < 0.3:
            blended *= 0.7
        
        return max(0.0, min(1.0, blended))
    
    def _determine_action(
        self,
        home_prob: float,
        away_prob: float,
        confidence: float,
        uncertainty: float
    ) -> str:
        """Determine recommended action based on prediction quality."""
        # Calculate edge
        max_prob = max(home_prob, away_prob)
        
        if confidence < 0.4 or uncertainty > 0.6:
            return 'skip'
        elif confidence > 0.7 and max_prob > 0.55:
            return 'bet'
        elif confidence > 0.5 and max_prob > 0.52:
            return 'review'
        else:
            return 'skip'
    
    def _fallback_prediction(self) -> EnsemblePrediction:
        """Return fallback prediction when insufficient models."""
        return EnsemblePrediction(
            home_win_prob=0.5,
            away_win_prob=0.5,
            draw_prob=None,
            confidence=0.0,
            agreement_score=0.0,
            model_contributions={},
            primary_model="fallback",
            uncertainty=1.0,
            recommended_action='skip'
        )
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get models ranked by effective weight."""
        rankings = []
        for name, weight in self.model_weights.items():
            perf = 0.5
            if name in self.performance_history:
                history = self.performance_history[name]
                if len(history) >= 5:
                    perf = sum(history[-10:]) / min(len(history), 10)
            
            score = weight.effective_weight * (0.5 + 0.5 * perf)
            rankings.append((name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def should_retrain(self, threshold: float = 0.55) -> List[str]:
        """Identify models that need retraining based on performance."""
        need_retrain = []
        
        for name, history in self.performance_history.items():
            if len(history) >= 20:
                recent_accuracy = sum(history[-20:]) / 20
                if recent_accuracy < threshold:
                    need_retrain.append(name)
                    logger.warning(f"Model {name} accuracy dropped to {recent_accuracy:.2f}")
        
        return need_retrain
