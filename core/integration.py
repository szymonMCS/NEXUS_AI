"""
NEXUS AI Integration Module.

Integrates datasets, ML models, and prediction monitoring
with the main LangGraph workflow.

This module provides the glue between:
- Data collection (core/datasets)
- ML predictions (core/ml)
- Hybrid predictions (core/llm/hybrid_predictor.py)
- Quality monitoring (core/ml/tracking)
- LangGraph agents (agents/)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.datasets import DatasetManager, DatasetConfig
from core.llm.hybrid_predictor import HybridPredictor, HybridPrediction
from core.ml.tracking.prediction_monitor import PredictionMonitor, TrackedPrediction
from core.state import NexusState, Match

logger = logging.getLogger(__name__)


class NexusIntegration:
    """
    Main integration class for NEXUS AI.
    
    Provides unified interface for:
    - Data collection
    - Hybrid predictions (ML + Kimi)
    - Performance tracking
    - Model retraining triggers
    
    Usage:
        nexus = NexusIntegration()
        
        # Get prediction with full tracking
        prediction = await nexus.predict(home, away, league, sport)
        
        # Track performance
        report = nexus.get_performance_report(days=30)
    """
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.hybrid_predictor = HybridPredictor()
        self.monitor = PredictionMonitor()
        
        logger.info("NEXUS Integration initialized")
    
    async def predict(
        self,
        home_team: str,
        away_team: str,
        league: str,
        sport: str,
        match_id: Optional[str] = None,
        odds: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> HybridPrediction:
        """
        Generate hybrid prediction with tracking.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            sport: Sport type
            match_id: Optional match identifier
            odds: Market odds
            context: Additional context for analysis
            
        Returns:
            HybridPrediction with all details
        """
        # Generate prediction
        prediction = await self.hybrid_predictor.predict(
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_id=match_id,
            odds=odds,
            context=context,
        )
        
        # Track prediction
        if prediction.recommended_bets:
            for bet in prediction.recommended_bets:
                tracked = TrackedPrediction(
                    prediction_id=f"{match_id}_{bet['market']}" if match_id else f"pred_{datetime.utcnow().timestamp()}",
                    match_id=match_id or "unknown",
                    sport=sport,
                    model_version="hybrid_v1",
                    model_type="hybrid",
                    predicted_outcome=bet.get("selection", ""),
                    predicted_prob=bet.get("probability", 0.0),
                    confidence=prediction.confidence,
                    market_odds=bet.get("odds", 0.0),
                    edge=bet.get("edge", 0.0),
                    recommended_stake=0.0,  # Will be set by risk manager
                )
                self.monitor.track_prediction(tracked)
        
        return prediction
    
    async def predict_batch(
        self,
        matches: List[Dict[str, Any]],
        sport: str,
    ) -> List[HybridPrediction]:
        """
        Generate predictions for multiple matches.
        
        Args:
            matches: List of match dicts
            sport: Sport type
            
        Returns:
            List of HybridPrediction
        """
        predictions = await self.hybrid_predictor.batch_predict(matches)
        
        # Track all predictions
        for match, pred in zip(matches, predictions):
            if pred.recommended_bets:
                for bet in pred.recommended_bets:
                    tracked = TrackedPrediction(
                        prediction_id=f"{match.get('match_id', '')}_{bet['market']}",
                        match_id=match.get('match_id', 'unknown'),
                        sport=sport,
                        model_version="hybrid_v1",
                        model_type="hybrid",
                        predicted_outcome=bet.get("selection", ""),
                        predicted_prob=bet.get("probability", 0.0),
                        confidence=pred.confidence,
                        market_odds=bet.get("odds", 0.0),
                        edge=bet.get("edge", 0.0),
                        recommended_stake=0.0,
                    )
                    self.monitor.track_prediction(tracked)
        
        return predictions
    
    async def collect_training_data(
        self,
        sport: str,
        days: int = 365,
        min_matches: int = 1000,
    ) -> bool:
        """
        Collect historical data for model training.
        
        Args:
            sport: Sport to collect
            days: Days of history
            min_matches: Minimum matches required
            
        Returns:
            True if successful
        """
        from datetime import timedelta
        
        config = DatasetConfig(
            sport=sport,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now(),
            min_matches=min_matches,
        )
        
        try:
            matches = await self.dataset_manager.collect(config)
            
            if len(matches) < min_matches:
                logger.warning(f"Collected only {len(matches)} matches, need {min_matches}")
                return False
            
            # Save to training format
            output_path = f"data/historical/{sport}/training_data.csv"
            success = self.dataset_manager.save_to_training_format(
                matches, output_path, format="csv"
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return False
    
    def get_performance_report(self, days: int = 30) -> str:
        """Get formatted performance report."""
        return self.monitor.generate_report(days=days)
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get model retraining recommendations."""
        return self.monitor.get_retraining_recommendation()
    
    def resolve_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
        profit_loss: float = 0.0,
    ) -> bool:
        """
        Resolve a tracked prediction with actual result.
        
        Args:
            prediction_id: Prediction identifier
            actual_outcome: Actual outcome
            profit_loss: Profit or loss amount
            
        Returns:
            True if successful
        """
        return self.monitor.resolve_prediction(prediction_id, actual_outcome, profit_loss)
    
    def get_top_models(self, sport: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get top performing models."""
        models = []
        
        for model_type in ["goals", "handicap", "hybrid"]:
            perf = self.monitor.get_performance(
                model_type=model_type,
                sport=sport,
                days=days,
            )
            
            if perf.total_predictions > 10:
                models.append({
                    "model_type": model_type,
                    "accuracy": perf.accuracy,
                    "roi": perf.roi,
                    "predictions": perf.total_predictions,
                    "score": perf.accuracy * 0.5 + (1 + perf.roi) * 0.5,  # Combined score
                })
        
        # Sort by combined score
        models.sort(key=lambda x: x["score"], reverse=True)
        return models


# Convenience functions for use in agents

async def get_prediction_with_tracking(
    home_team: str,
    away_team: str,
    league: str,
    sport: str,
    odds: Optional[Dict[str, float]] = None,
) -> HybridPrediction:
    """Get prediction with automatic tracking."""
    integration = NexusIntegration()
    return await integration.predict(home_team, away_team, league, sport, odds=odds)


def get_performance_summary(days: int = 30) -> Dict[str, Any]:
    """Get quick performance summary."""
    integration = NexusIntegration()
    monitor = integration.monitor
    
    overall = monitor.get_performance(days=days)
    
    return {
        "total_predictions": overall.total_predictions,
        "accuracy": overall.accuracy,
        "roi": overall.roi,
        "profit": overall.profit,
        "win_rate": overall.win_rate,
        "brier_score": overall.brier_score,
        "needs_retraining": integration.get_retraining_status()["needs_retraining"],
    }


def update_prediction_results(results: List[Dict[str, Any]]) -> int:
    """
    Bulk update prediction results.
    
    Args:
        results: List of dicts with prediction_id, actual_outcome, profit_loss
        
    Returns:
        Number of successfully updated predictions
    """
    integration = NexusIntegration()
    count = 0
    
    for result in results:
        success = integration.resolve_prediction(
            prediction_id=result["prediction_id"],
            actual_outcome=result["actual_outcome"],
            profit_loss=result.get("profit_loss", 0.0),
        )
        if success:
            count += 1
    
    return count
