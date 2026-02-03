"""
High-level prediction service for NEXUS AI.
Simplifies access to ML models for agents and API.
Uses EnsembleAggregator for multi-model predictions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import logging
from core.ml.v2.registry import get_model_registry, SimpleModelRegistry
from core.ml.ensemble_aggregator import EnsembleAggregator, EnsemblePrediction

logger = logging.getLogger(__name__)
from core.ml.models.base import PredictionResult


@dataclass
class PredictionExplanation:
    """Detailed prediction with explanation."""
    home_probability: float
    away_probability: float
    draw_probability: Optional[float]
    confidence: float
    selected_outcome: str
    recommended_bet: Optional[str]
    expected_value: Optional[float]
    reasoning: List[str]
    model_used: str
    key_factors: List[Dict[str, Any]]
    agreement_score: float = 0.0
    model_contributions: Dict[str, float] = None
    recommended_action: str = "skip"


class PredictionService:
    """
    High-level service for match predictions with ensemble aggregation.
    Uses multiple models and dynamic weighting for better accuracy.
    """
    
    def __init__(self, registry: Optional['SimpleModelRegistry'] = None):
        self.registry: SimpleModelRegistry = registry or get_model_registry()
        self.aggregator = EnsembleAggregator(min_models=1)
        self._init_aggregator()
    
    def _init_aggregator(self):
        """Register all available models with the aggregator."""
        for sport in ["tennis", "basketball", "football"]:
            models = self.registry.get_active_models(sport)
            for model in models:
                self.aggregator.register_model(model)
    
    def predict(
        self,
        sport: str,
        home_player: str,
        away_player: str,
        features: Optional[Dict] = None,
        odds: Optional[Dict] = None,
        use_ensemble: bool = True
    ) -> PredictionResult:
        """
        Get prediction for a match using ensemble of models.
        
        Args:
            sport: Sport type (tennis, basketball, etc.)
            home_player: Home player/team name
            away_player: Away player/team name
            features: Optional features (rankings, form, etc.)
            odds: Optional current odds for value calculation
            use_ensemble: If True, use all models; if False, use best single model
            
        Returns:
            PredictionResult with probabilities
        """
        if features is None:
            features = {}
        
        try:
            # Add player identifiers
            features['home_player'] = home_player
            features['away_player'] = away_player
            features['home_team'] = home_player
            features['away_team'] = away_player
            
            # Get all available models for this sport
            models = self.registry.get_active_models(sport)
            
            if not models:
                logger.warning(f"No models available for {sport}, returning neutral prediction")
                return PredictionResult(
                    home_win_prob=0.5,
                    away_win_prob=0.5,
                    confidence=0.0,
                    model_name="unavailable"
                )
            
            if use_ensemble and len(models) > 1:
                # Collect predictions from all models
                predictions = {}
                for model in models:
                    try:
                        pred = model.predict(features)
                        predictions[model.name] = pred
                    except Exception as e:
                        logger.debug(f"Model {model.name} failed: {e}")
                
                if len(predictions) == 0:
                    return PredictionResult(
                        home_win_prob=0.5,
                        away_win_prob=0.5,
                        confidence=0.0,
                        model_name="all_models_failed"
                    )
                
                # Aggregate predictions
                ensemble_result = self.aggregator.aggregate(predictions, features)
                
                # Convert to PredictionResult
                result = PredictionResult(
                    home_win_prob=ensemble_result.home_win_prob,
                    away_win_prob=ensemble_result.away_win_prob,
                    draw_prob=ensemble_result.draw_prob,
                    confidence=ensemble_result.confidence,
                    model_name=f"ensemble:{ensemble_result.primary_model}",
                    features_used=list(predictions.values())[0].features_used if predictions else None
                )
                
                # Store ensemble metadata for later use
                result._ensemble_meta = {
                    'agreement_score': ensemble_result.agreement_score,
                    'model_contributions': ensemble_result.model_contributions,
                    'recommended_action': ensemble_result.recommended_action,
                    'uncertainty': ensemble_result.uncertainty
                }
                
            else:
                # Use best single model
                model = models[0]
                result = model.predict(features)
            
            logger.debug(
                f"Prediction for {home_player} vs {away_player}: "
                f"home={result.home_win_prob:.2f}, away={result.away_win_prob:.2f}, "
                f"conf={result.confidence:.2f}, model={result.model_name}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {home_player} vs {away_player}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return PredictionResult(
                home_win_prob=0.5,
                away_win_prob=0.5,
                confidence=0.0,
                model_name="error_fallback"
            )
    
    def predict_with_value(
        self,
        sport: str,
        home_player: str,
        away_player: str,
        home_odds: float,
        away_odds: float,
        features: Optional[Dict] = None,
        use_ensemble: bool = True
    ) -> PredictionExplanation:
        """
        Get prediction with value betting analysis using ensemble.
        
        Args:
            sport: Sport type
            home_player: Home player name
            away_player: Away player name
            home_odds: Bookmaker odds for home win
            away_odds: Bookmaker odds for away win
            features: Optional features
            use_ensemble: If True, use ensemble aggregation
            
        Returns:
            PredictionExplanation with value analysis
        """
        # Get base prediction with ensemble
        result = self.predict(sport, home_player, away_player, features, use_ensemble=use_ensemble)
        
        # Calculate value
        home_value = self._calculate_value(result.home_win_prob, home_odds)
        away_value = self._calculate_value(result.away_win_prob, away_odds)
        
        # Determine selection
        if home_value > away_value and home_value > 0:
            selected = "home"
            selected_prob = result.home_win_prob
            selected_odds = home_odds
            ev = home_value
        elif away_value > 0:
            selected = "away"
            selected_prob = result.away_win_prob
            selected_odds = away_odds
            ev = away_value
        else:
            selected = "none"
            selected_prob = 0
            selected_odds = 0
            ev = 0
        
        # Generate reasoning
        reasoning = self._generate_reasoning(result, selected, ev)
        
        # Add ensemble-specific reasoning if available
        if hasattr(result, '_ensemble_meta'):
            meta = result._ensemble_meta
            agreement = meta.get('agreement_score', 0)
            
            if agreement > 0.8:
                reasoning.insert(0, f"High model agreement ({agreement:.0%}) strengthens confidence")
            elif agreement < 0.4:
                reasoning.insert(0, f"Low model agreement ({agreement:.0%}) - proceed with caution")
            
            # Add model contributions info
            contributions = meta.get('model_contributions', {})
            if contributions:
                top_model = max(contributions.items(), key=lambda x: x[1])
                reasoning.append(f"Primary model: {top_model[0]} ({top_model[1]:.0%} weight)")
        
        # Get model info
        model_name = result.model_name
        
        # Build factors
        factors = []
        if hasattr(result, 'factors') and result.factors:
            for name, value in sorted(result.factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                factors.append({
                    "name": name,
                    "impact": "positive" if value > 0 else "negative",
                    "weight": abs(value)
                })
        
        # Extract ensemble metadata
        agreement_score = 0.0
        model_contributions = None
        recommended_action = "skip"
        
        if hasattr(result, '_ensemble_meta'):
            meta = result._ensemble_meta
            agreement_score = meta.get('agreement_score', 0)
            model_contributions = meta.get('model_contributions')
            recommended_action = meta.get('recommended_action', 'skip')
        
        return PredictionExplanation(
            home_probability=result.home_win_prob,
            away_probability=result.away_win_prob,
            draw_probability=result.draw_prob,
            confidence=result.confidence,
            selected_outcome=selected,
            recommended_bet=selected if ev > 0 else None,
            expected_value=ev if ev > 0 else None,
            reasoning=reasoning,
            model_used=model_name,
            key_factors=factors,
            agreement_score=agreement_score,
            model_contributions=model_contributions,
            recommended_action=recommended_action
        )
    
    def _calculate_value(self, probability: float, odds: float) -> float:
        """
        Calculate expected value of a bet.
        EV = (Probability * Odds) - 1
        """
        if odds <= 1.0 or probability <= 0:
            return -1.0
        return (probability * odds) - 1.0
    
    def _generate_reasoning(self, result: PredictionResult, selection: str, ev: float) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []
        
        # Model confidence
        if result.confidence > 0.7:
            reasoning.append("High confidence prediction based on strong data signals")
        elif result.confidence > 0.5:
            reasoning.append("Moderate confidence with clear directional signal")
        else:
            reasoning.append("Lower confidence due to limited or conflicting data")
        
        # Selection
        if selection == "home":
            prob = result.home_win_prob
            reasoning.append(f"Home win predicted with {prob:.1%} probability")
        elif selection == "away":
            prob = result.away_win_prob
            reasoning.append(f"Away win predicted with {prob:.1%} probability")
        else:
            reasoning.append("No clear value bet identified at current odds")
        
        # Value
        if ev > 0.1:
            reasoning.append(f"Strong value detected: +{ev:.1%} expected return")
        elif ev > 0.05:
            reasoning.append(f"Moderate value: +{ev:.1%} expected return")
        elif ev > 0:
            reasoning.append(f"Marginal value: +{ev:.1%} expected return")
        
        # Factors (if available)
        if hasattr(result, 'factors') and result.factors:
            top_factor = max(result.factors.items(), key=lambda x: abs(x[1]))
            reasoning.append(f"Key factor: {top_factor[0]} (impact: {top_factor[1]:+.2f})")
        
        return reasoning
    
    def batch_predict(self,
                     sport: str,
                     matches: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Predict multiple matches efficiently.
        
        Args:
            sport: Sport type
            matches: List of match dicts with home_player, away_player, features
            
        Returns:
            List of PredictionResult
        """
        model = self.registry.get_best_model(sport)
        
        if not model:
            return [
                PredictionResult(0.5, 0.5, confidence=0.0, model_name="unavailable")
                for _ in matches
            ]
        
        # Prepare features
        features_list = []
        for match in matches:
            f = match.get("features", {})
            f["home_player"] = match.get("home_player") or match.get("home_team")
            f["away_player"] = match.get("away_player") or match.get("away_team")
            f["home_team"] = match.get("home_player") or match.get("home_team")
            f["away_team"] = match.get("away_player") or match.get("away_team")
            features_list.append(f)
        
        return model.predict_batch(features_list)
    
    def get_model_info(self, sport: str) -> Dict[str, Any]:
        """Get information about available models and ensemble for a sport."""
        models = self.registry.get_active_models(sport)
        
        # Get ensemble rankings
        rankings = self.aggregator.get_model_rankings()
        
        # Filter rankings for models available for this sport
        model_names = {m.name for m in models}
        sport_rankings = [(name, score) for name, score in rankings if name in model_names]
        
        return {
            "sport": sport,
            "available_models": len(models),
            "models": [
                {
                    "name": m.name,
                    "version": m.version,
                    "type": m.model_type,
                    "accuracy": m.accuracy,
                    "created": m.created_at.isoformat() if m.created_at else None,
                    "ensemble_weight": next((score for name, score in sport_rankings if name == m.name), 0.5)
                }
                for m in models
            ],
            "ensemble": {
                "enabled": True,
                "model_count": len(models),
                "model_rankings": [
                    {"name": name, "score": round(score, 3)}
                    for name, score in sport_rankings
                ],
                "primary_model": sport_rankings[0][0] if sport_rankings else None
            },
            "best_model": models[0].name if models else None
        }
    
    def warmup(self, sport: str):
        """
        Preload model for a sport to avoid cold start.
        Call this at application startup.
        """
        try:
            model = self.registry.get_best_model(sport)
            logger.info(f"Model warmed up for {sport}: {type(model).__name__}")
        except Exception as e:
            logger.error(f"Failed to warm up model for {sport}: {e}")


# Singleton
_service_instance = None

def get_prediction_service() -> PredictionService:
    """Get singleton prediction service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance
