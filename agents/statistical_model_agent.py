# agents/statistical_model_agent.py
"""
Statistical Model Agent - Enhanced prediction using statistical models.
Provides fallback and ensemble predictions using statistical models.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from core.state import NexusState, Match, add_message
from core.models import (
    TennisModel, BasketballModel,
    GreyhoundModel, HandballModel, TableTennisModel,
    Sport
)

logger = logging.getLogger("nexus.statistical_agent")


class StatisticalModelAgent:
    """
    Agent that uses statistical models for predictions.
    
    Can be used as:
    1. Primary predictor (when LLM unavailable)
    2. Fallback predictor (when LLM fails)
    3. Ensemble component (combined with LLM)
    
    Supported sports:
    - Tennis (TennisModel)
    - Basketball (BasketballModel)
    - Greyhound Racing (GreyhoundModel)
    - Handball (HandballModel)
    - Table Tennis (TableTennisModel)
    """
    
    def __init__(self, use_ensemble: bool = True):
        """
        Initialize the statistical model agent.
        
        Args:
            use_ensemble: Whether to ensemble statistical + LLM predictions
        """
        self.use_ensemble = use_ensemble
        self.models: Dict[Sport, Any] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all statistical models."""
        try:
            self.models[Sport.TENNIS] = TennisModel()
            logger.info("TennisModel initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TennisModel: {e}")
        
        try:
            self.models[Sport.BASKETBALL] = BasketballModel()
            logger.info("BasketballModel initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize BasketballModel: {e}")
        
        try:
            self.models[Sport.GREYHOUND] = GreyhoundModel()
            logger.info("GreyhoundModel initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GreyhoundModel: {e}")
        
        try:
            self.models[Sport.HANDBALL] = HandballModel()
            logger.info("HandballModel initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HandballModel: {e}")
        
        try:
            self.models[Sport.TABLE_TENNIS] = TableTennisModel()
            logger.info("TableTennisModel initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TableTennisModel: {e}")
    
    def _get_model_for_sport(self, sport: Sport):
        """Get the appropriate model for a sport."""
        if sport in self.models:
            return self.models[sport]
        
        # Try to find a compatible model
        sport_models = {
            Sport.TENNIS: "tennis_model",
            Sport.BASKETBALL: "basketball_model",
            Sport.GREYHOUND: "greyhound_model",
            Sport.HANDBALL: "handball_model",
            Sport.TABLE_TENNIS: "table_tennis_model",
        }
        
        model_name = sport_models.get(sport)
        if model_name:
            logger.warning(f"Using {model_name} for {sport.value}")
        
        return None
    
    async def process(self, state: NexusState) -> NexusState:
        """
        Generate statistical predictions for matches.
        
        Args:
            state: Current workflow state with matches
            
        Returns:
            Updated state with predictions from statistical models
        """
        state.current_agent = "statistical_model_agent"
        
        # Filter matches without predictions
        matches_to_predict = [
            m for m in state.matches
            if m.prediction is None
        ]
        
        if not matches_to_predict:
            state = add_message(
                state,
                "statistical_model_agent",
                "All matches already have predictions"
            )
            return state
        
        state = add_message(
            state,
            "statistical_model_agent",
            f"Generating statistical predictions for {len(matches_to_predict)} matches"
        )
        
        sport = state.sport
        model = self._get_model_for_sport(sport)
        
        if not model:
            state = add_message(
                state,
                "statistical_model_agent",
                f"No statistical model available for {sport.value}"
            )
            return state
        
        # Generate predictions
        predictions_generated = 0
        for match in matches_to_predict:
            try:
                prediction = self._create_prediction_from_match(match, model, sport)
                if prediction:
                    match.prediction = prediction
                    predictions_generated += 1
                    
            except Exception as e:
                logger.error(f"Error predicting {match}: {e}")
        
        state = add_message(
            state,
            "statistical_model_agent",
            f"Generated {predictions_generated} statistical predictions"
        )
        
        return state
    
    def _create_prediction_from_match(
        self,
        match: Match,
        model,
        sport: Sport
    ) -> Optional[Any]:
        """Create a prediction object from match data using statistical model."""
        from core.state import MatchPrediction, DataQuality
        
        # Build match data for the model
        match_data = self._build_match_data(match, sport)
        
        try:
            # Run model prediction
            result = model.predict(match_data)
            
            # Create MatchPrediction object
            prediction = MatchPrediction(
                home_win_prob=result.probabilities.get("home", 0.5),
                away_win_prob=result.probabilities.get("away", 0.5),
                confidence=result.confidence,
                factors={
                    "model_used": result.model_name,
                    "statistical_factors": self._extract_factors(result),
                    "reliability_score": result.reliability_score,
                }
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return None
    
    def _build_match_data(self, match: Match, sport: Sport) -> Dict[str, Any]:
        """Build match data dict for the statistical model."""
        data = {}
        
        if sport == Sport.TENNIS:
            data = {
                "player1_name": match.home_player.name,
                "player2_name": match.away_player.name,
                "player1_ranking": getattr(match.home_player, 'ranking', 500),
                "player2_ranking": getattr(match.away_player, 'ranking', 500),
                "player1_win_rate": getattr(match.home_player, 'win_rate', 0.5),
                "player2_win_rate": getattr(match.away_player, 'win_rate', 0.5),
                "surface": getattr(match, 'surface', 'hard'),
            }
        
        elif sport == Sport.BASKETBALL:
            data = {
                "home_team": match.home_player.name,
                "away_team": match.away_player.name,
                "home_offensive_rating": 110.0,
                "home_defensive_rating": 105.0,
                "away_offensive_rating": 108.0,
                "away_defensive_rating": 106.0,
                "home_rest_days": 2,
                "away_rest_days": 2,
            }
        
        elif sport == Sport.GREYHOUND:
            data = {
                "runners": [
                    {
                        "dog_name": match.home_player.name,
                        "trap": 1,
                        "weight": 32,
                        "age_months": 30,
                        "recent_positions": [2, 1, 3],
                    },
                    {
                        "dog_name": match.away_player.name,
                        "trap": 2,
                        "weight": 31,
                        "age_months": 28,
                        "recent_positions": [3, 2, 1],
                    }
                ]
            }
        
        elif sport == Sport.HANDBALL:
            data = {
                "home_team": match.home_player.name,
                "away_team": match.away_player.name,
                "home_goals_scored": 26.0,
                "home_goals_conceded": 24.0,
                "away_goals_scored": 25.0,
                "away_goals_conceded": 25.0,
                "home_elo": 1500,
                "away_elo": 1500,
            }
        
        elif sport == Sport.TABLE_TENNIS:
            data = {
                "player1_name": match.home_player.name,
                "player2_name": match.away_player.name,
                "player1_ranking": getattr(match.home_player, 'ranking', 500),
                "player2_ranking": getattr(match.away_player, 'ranking', 500),
                "player1_rating": 1500.0,
                "player2_rating": 1500.0,
            }
        
        return data
    
    def _extract_factors(self, result) -> Dict[str, Any]:
        """Extract key factors from model prediction result."""
        factors = {}
        
        if hasattr(result, 'reasoning') and result.reasoning:
            factors["model_reasoning"] = result.reasoning[:3]
        
        if hasattr(result, 'reliability_score'):
            factors["reliability"] = result.reliability_score
        
        if hasattr(result, 'probabilities'):
            factors["probabilities"] = result.probabilities
        
        return factors
    
    def predict_match(
        self,
        sport: Sport,
        match_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Direct prediction for a single match.
        
        Args:
            sport: Sport type
            match_data: Match data dictionary
            
        Returns:
            Prediction result dict or None
        """
        model = self._get_model_for_sport(sport)
        
        if not model:
            return None
        
        try:
            result = model.predict(match_data)
            return {
                "winner": result.predicted_winner,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "model": result.model_name,
                "reliability": result.reliability_score,
                "reasoning": result.reasoning,
            }
        except Exception as e:
            logger.error(f"Direct prediction failed: {e}")
            return None


# === HELPER FUNCTIONS ===

def create_statistical_agent(use_ensemble: bool = True) -> StatisticalModelAgent:
    """Create a new statistical model agent instance."""
    return StatisticalModelAgent(use_ensemble=use_ensemble)


async def run_statistical_prediction(
    sport: str,
    match_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for single match statistical prediction.
    
    Args:
        sport: Sport type
        match_data: Match data
        
    Returns:
        Prediction result or None
    """
    agent = create_statistical_agent()
    return agent.predict_match(Sport(sport), match_data)
