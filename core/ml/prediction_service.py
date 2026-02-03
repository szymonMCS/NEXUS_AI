"""
High-level prediction service for NEXUS AI.
Simplifies access to ML models for agents and API.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core.ml.registry import get_model_registry, ModelRegistry
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


class PredictionService:
    """
    High-level service for match predictions.
    Used by agents, API endpoints, and betting floor.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
    
    def predict(self,
                sport: str,
                home_player: str,
                away_player: str,
                features: Optional[Dict] = None,
                odds: Optional[Dict] = None) -> PredictionResult:
        """
        Get prediction for a match.
        
        Args:
            sport: Sport type (tennis, basketball, etc.)
            home_player: Home player/team name
            away_player: Away player/team name
            features: Optional features (rankings, form, etc.)
            odds: Optional current odds for value calculation
            
        Returns:
            PredictionResult with probabilities
        """
        if features is None:
            features = {}
        
        # Add player identifiers
        features['home_player'] = home_player
        features['away_player'] = away_player
        features['home_team'] = home_player
        features['away_team'] = away_player
        
        # Get best available model
        model = self.registry.get_best_model(sport)
        
        if not model:
            # Return neutral prediction if no model
            return PredictionResult(
                home_win_prob=0.5,
                away_win_prob=0.5,
                confidence=0.0,
                model_name="unavailable"
            )
        
        # Make prediction
        result = model.predict(features)
        
        return result
    
    def predict_with_value(self,
                          sport: str,
                          home_player: str,
                          away_player: str,
                          home_odds: float,
                          away_odds: float,
                          features: Optional[Dict] = None) -> PredictionExplanation:
        """
        Get prediction with value betting analysis.
        
        Args:
            sport: Sport type
            home_player: Home player name
            away_player: Away player name
            home_odds: Bookmaker odds for home win
            away_odds: Bookmaker odds for away win
            features: Optional features
            
        Returns:
            PredictionExplanation with value analysis
        """
        # Get base prediction
        result = self.predict(sport, home_player, away_player, features)
        
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
        
        # Get model info
        model = self.registry.get_best_model(sport)
        model_name = model.name if model else "unknown"
        
        # Build factors
        factors = []
        if result.factors:
            for name, value in sorted(result.factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                factors.append({
                    "name": name,
                    "impact": "positive" if value > 0 else "negative",
                    "weight": abs(value)
                })
        
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
            key_factors=factors
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
        
        # Factors
        if result.factors:
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
        """Get information about available models for a sport."""
        models = self.registry.get_active_models(sport)
        
        return {
            "sport": sport,
            "available_models": len(models),
            "models": [
                {
                    "name": m.name,
                    "version": m.version,
                    "type": m.model_type,
                    "accuracy": m.accuracy,
                    "created": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ],
            "best_model": models[0].name if models else None
        }
    
    def warmup(self, sport: str):
        """
        Preload model for a sport to avoid cold start.
        Call this at application startup.
        """
        _ = self.registry.get_best_model(sport)
        print(f"Model warmed up for {sport}")


# Singleton
_service_instance = None

def get_prediction_service() -> PredictionService:
    """Get singleton prediction service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance
