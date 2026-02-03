"""
Feedback Loop for ML Models.
Collects match results and triggers model retraining.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from core.ml import get_prediction_service, get_model_registry

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a completed match."""
    match_id: str
    sport: str
    home_player: str
    away_player: str
    home_score: float
    away_score: float
    match_date: datetime
    prediction_home_prob: Optional[float] = None
    prediction_away_prob: Optional[float] = None
    prediction_confidence: Optional[float] = None
    actual_result: Optional[str] = None  # 'home', 'away', 'draw'
    prediction_correct: Optional[bool] = None
    
    def __post_init__(self):
        if self.actual_result is None:
            if self.home_score > self.away_score:
                self.actual_result = 'home'
            elif self.home_score < self.away_score:
                self.actual_result = 'away'
            else:
                self.actual_result = 'draw'
        
        # Check if prediction was correct
        if self.prediction_home_prob and self.prediction_away_prob:
            predicted_winner = 'home' if self.prediction_home_prob > self.prediction_away_prob else 'away'
            self.prediction_correct = (predicted_winner == self.actual_result)


class FeedbackLoop:
    """
    Manages match results and model feedback.
    Tracks prediction accuracy and triggers retraining.
    """
    
    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.data_dir / "match_results.json"
        self.metrics_file = self.data_dir / "performance_metrics.json"
        
        self.retrain_threshold = 50  # Retrain after N new matches
        self.min_accuracy_for_retrain = 0.55  # Minimum accuracy to trigger retrain
        
        self._load_metrics()
    
    def _load_metrics(self):
        """Load performance metrics."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
                self.metrics = {}
        else:
            self.metrics = {}
    
    def _save_metrics(self):
        """Save performance metrics."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def record_result(self, result: MatchResult):
        """
        Record a match result.
        
        Args:
            result: MatchResult with actual scores
        """
        # Save to file
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        
        # Convert to dict and add
        result_dict = asdict(result)
        result_dict['recorded_at'] = datetime.now().isoformat()
        results.append(result_dict)
        
        # Keep only last 10000 results
        results = results[-10000:]
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Recorded result: {result.match_id} - {result.home_player} {result.home_score}:{result.away_score} {result.away_player}")
        
        # Update metrics
        self._update_metrics(result)
        
        # Check if retraining needed
        self._check_retrain_needed(result.sport)
    
    def _update_metrics(self, result: MatchResult):
        """Update performance metrics for a sport."""
        sport = result.sport
        
        if sport not in self.metrics:
            self.metrics[sport] = {
                "total_matches": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "last_updated": datetime.now().isoformat(),
                "recent_results": []
            }
        
        metrics = self.metrics[sport]
        metrics["total_matches"] += 1
        
        if result.prediction_correct:
            metrics["correct_predictions"] += 1
        
        # Calculate accuracy
        if metrics["total_matches"] > 0:
            metrics["accuracy"] = metrics["correct_predictions"] / metrics["total_matches"]
        
        # Track recent results (last 100)
        metrics["recent_results"].append({
            "match_id": result.match_id,
            "correct": result.prediction_correct,
            "date": result.match_date.isoformat() if isinstance(result.match_date, datetime) else result.match_date
        })
        metrics["recent_results"] = metrics["recent_results"][-100:]
        
        metrics["last_updated"] = datetime.now().isoformat()
        
        self._save_metrics()
    
    def _check_retrain_needed(self, sport: str):
        """Check if model retraining is needed."""
        # Count unprocessed results
        results = self.get_recent_results(sport, days=7)
        
        if len(results) >= self.retrain_threshold:
            logger.info(f"Retraining threshold reached for {sport}: {len(results)} new matches")
            # Trigger retraining (async)
            asyncio.create_task(self._retrain_model(sport))
    
    async def _retrain_model(self, sport: str):
        """Retrain model for a sport."""
        try:
            logger.info(f"Starting retraining for {sport}...")
            
            # Get all results for training
            results = self.get_recent_results(sport, days=365)
            
            # Convert to training format
            matches = []
            for r in results:
                matches.append({
                    "home_player": r.home_player,
                    "away_player": r.away_player,
                    "home_score": r.home_score,
                    "away_score": r.away_score,
                    "date": r.match_date,
                    "sport": r.sport
                })
            
            # Train statistical ensemble
            from core.ml import StatisticalEnsemble
            
            model = StatisticalEnsemble(sport=sport)
            model.fit(matches)
            
            # Save to registry
            registry = get_model_registry()
            registry.save_model(model)
            
            logger.info(f"Retraining complete for {sport}. Trained on {len(matches)} matches.")
            
        except Exception as e:
            logger.error(f"Retraining failed for {sport}: {e}")
    
    def get_recent_results(self, sport: str, days: int = 30) -> List[MatchResult]:
        """Get recent match results."""
        if not self.results_file.exists():
            return []
        
        with open(self.results_file, 'r') as f:
            all_results = json.load(f)
        
        # Filter by sport and date
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        results = []
        for r in all_results:
            if r.get('sport') == sport and r.get('match_date', '') > cutoff:
                # Convert dict to MatchResult
                results.append(MatchResult(
                    match_id=r['match_id'],
                    sport=r['sport'],
                    home_player=r['home_player'],
                    away_player=r['away_player'],
                    home_score=r['home_score'],
                    away_score=r['away_score'],
                    match_date=datetime.fromisoformat(r['match_date']) if isinstance(r['match_date'], str) else r['match_date'],
                    prediction_home_prob=r.get('prediction_home_prob'),
                    prediction_away_prob=r.get('prediction_away_prob'),
                    prediction_confidence=r.get('prediction_confidence')
                ))
        
        return results
    
    def get_performance_report(self, sport: Optional[str] = None) -> Dict:
        """Get performance report."""
        if sport:
            return self.metrics.get(sport, {})
        return self.metrics
    
    def record_prediction(self, match_id: str, sport: str, 
                         home_prob: float, away_prob: float, 
                         confidence: float):
        """
        Record a prediction before match starts.
        Will be linked with result later.
        """
        # Store in pending predictions
        pending_file = self.data_dir / "pending_predictions.json"
        
        pending = []
        if pending_file.exists():
            with open(pending_file, 'r') as f:
                pending = json.load(f)
        
        pending.append({
            "match_id": match_id,
            "sport": sport,
            "home_prob": home_prob,
            "away_prob": away_prob,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(pending_file, 'w') as f:
            json.dump(pending, f, indent=2)
    
    def link_prediction_with_result(self, match_id: str, result: MatchResult):
        """Link a stored prediction with actual result."""
        pending_file = self.data_dir / "pending_predictions.json"
        
        if not pending_file.exists():
            return
        
        with open(pending_file, 'r') as f:
            pending = json.load(f)
        
        # Find prediction
        for p in pending:
            if p["match_id"] == match_id:
                result.prediction_home_prob = p["home_prob"]
                result.prediction_away_prob = p["away_prob"]
                result.prediction_confidence = p["confidence"]
                break
        
        # Remove from pending
        pending = [p for p in pending if p["match_id"] != match_id]
        
        with open(pending_file, 'w') as f:
            json.dump(pending, f, indent=2)


# Singleton instance
_feedback_loop = None

def get_feedback_loop() -> FeedbackLoop:
    """Get singleton feedback loop instance."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
