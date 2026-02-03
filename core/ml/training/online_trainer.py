"""
Online training pipeline for NEXUS AI.
Handles continuous learning and model updates.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from dataclasses import dataclass

from core.ml.registry import ModelRegistry, ModelTrainer, get_model_registry
from core.ml.models.base import PredictionResult


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    model_name: str
    sport: str
    training_date: datetime
    samples_used: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    validation_accuracy: float
    training_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sport": self.sport,
            "training_date": self.training_date.isoformat(),
            "samples_used": self.samples_used,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "validation_accuracy": self.validation_accuracy,
            "training_time_seconds": self.training_time_seconds
        }


class MatchResultCollector:
    """
    Collects and stores match results for training.
    Interfaces with database to fetch completed matches.
    """
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.results_file = Path("data/match_results.json")
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def fetch_completed_matches(self, 
                                     sport: str, 
                                     days_back: int = 7,
                                     min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Fetch completed matches from database or storage.
        
        Args:
            sport: Sport type
            days_back: How many days back to look
            min_confidence: Minimum prediction confidence to include
            
        Returns:
            List of match results
        """
        matches = []
        
        # Try database first
        if self.db_session:
            try:
                from database.models import Match, SportType
                from database.db import get_db_session
                
                with get_db_session() as db:
                    since = datetime.now() - timedelta(days=days_back)
                    
                    # Map string sport to enum
                    sport_enum = getattr(SportType, sport.upper(), None)
                    if not sport_enum:
                        return matches
                    
                    db_matches = db.query(Match).filter(
                        Match.sport == sport_enum,
                        Match.is_finished == True,
                        Match.start_time >= since,
                        Match.winner.isnot(None)
                    ).all()
                    
                    for m in db_matches:
                        # Get prediction confidence if available
                        confidence = 0.0
                        if m.predictions:
                            confidence = m.predictions[0].confidence
                        
                        if confidence >= min_confidence:
                            matches.append({
                                "match_id": m.external_id,
                                "home_player": m.home_team,
                                "away_player": m.away_team,
                                "home_score": m.home_score,
                                "away_score": m.away_score,
                                "date": m.start_time,
                                "winner": m.winner,
                                "confidence": confidence
                            })
            except Exception as e:
                print(f"Database fetch error: {e}")
        
        # Fallback to file storage
        if not matches and self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    all_results = json.load(f)
                
                since = (datetime.now() - timedelta(days=days_back)).isoformat()
                
                for result in all_results.get(sport, []):
                    if result.get('date', '') >= since:
                        matches.append(result)
            except Exception as e:
                print(f"File fetch error: {e}")
        
        return matches
    
    def store_match_result(self, sport: str, result: Dict[str, Any]):
        """
        Store a match result for future training.
        
        Args:
            sport: Sport type
            result: Match result dict
        """
        try:
            data = {}
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
            
            if sport not in data:
                data[sport] = []
            
            # Add timestamp if not present
            if 'date' not in result:
                result['date'] = datetime.now().isoformat()
            
            data[sport].append(result)
            
            # Keep only last 10000 matches per sport
            data[sport] = data[sport][-10000:]
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error storing match result: {e}")


class OnlineTrainer:
    """
    Manages online training pipeline.
    Retrains models periodically as new data arrives.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
        self.trainer = ModelTrainer(self.registry)
        self.collector = MatchResultCollector()
        
        self.training_history: List[TrainingMetrics] = []
        self.last_training: Dict[str, datetime] = {}  # sport -> date
        
        self.min_matches_for_retrain = 50  # Minimum new matches to trigger retrain
        self.retrain_interval_days = 7  # Retrain at least every N days
    
    async def should_retrain(self, sport: str) -> bool:
        """
        Check if model should be retrained for a sport.
        
        Returns True if:
        - Never trained before
        - Min matches accumulated since last training
        - Interval days passed since last training
        """
        # Check if we have a model
        models = self.registry.get_active_models(sport)
        if not models:
            return True
        
        # Check time since last training
        last_train = self.last_training.get(sport)
        if last_train:
            days_since = (datetime.now() - last_train).days
            if days_since >= self.retrain_interval_days:
                return True
        
        # Check new matches count
        new_matches = await self.collector.fetch_completed_matches(
            sport, 
            days_back=self.retrain_interval_days
        )
        
        if len(new_matches) >= self.min_matches_for_retrain:
            return True
        
        return False
    
    async def train_statistical_models(self, sport: str) -> Optional[TrainingMetrics]:
        """
        Train statistical models for a sport.
        
        Args:
            sport: Sport type
            
        Returns:
            Training metrics or None
        """
        print(f"\n{'='*60}")
        print(f"Training Statistical Models for {sport.upper()}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Fetch all historical matches
        matches = await self.collector.fetch_completed_matches(sport, days_back=365)
        
        if len(matches) < 10:
            print(f"Insufficient data: {len(matches)} matches")
            return None
        
        print(f"Training on {len(matches)} matches...")
        
        # Train
        model = self.trainer.train_statistical_models(sport, matches)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Create metrics
        metrics = TrainingMetrics(
            model_name="StatisticalEnsemble",
            sport=sport,
            training_date=end_time,
            samples_used=len(matches),
            accuracy=model.metadata.accuracy if model.metadata else 0.0,
            precision=0.0,  # Would need validation set
            recall=0.0,
            f1_score=0.0,
            validation_accuracy=0.0,
            training_time_seconds=training_time
        )
        
        self.training_history.append(metrics)
        self.last_training[sport] = end_time
        
        print(f"Training complete in {training_time:.1f}s")
        print(f"Model accuracy: {metrics.accuracy:.3f}")
        
        return metrics
    
    async def train_ml_models(self, 
                             sport: str,
                             feature_extractor=None) -> Optional[TrainingMetrics]:
        """
        Train ML models if sklearn available.
        
        Args:
            sport: Sport type
            feature_extractor: Function to extract features from match data
            
        Returns:
            Training metrics or None
        """
        from core.ml.models import SKLEARN_AVAILABLE
        
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available, skipping ML training")
            return None
        
        print(f"\n{'='*60}")
        print(f"Training ML Models for {sport.upper()}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Fetch matches
        matches = await self.collector.fetch_completed_matches(sport, days_back=365)
        
        if len(matches) < 100:
            print(f"Insufficient data: {len(matches)} matches (need 100+)")
            return None
        
        # Extract features
        if feature_extractor is None:
            feature_extractor = self._default_feature_extractor
        
        X, y, feature_names = feature_extractor(matches)
        
        if X is None or len(X) < 100:
            print("Feature extraction failed")
            return None
        
        print(f"Training on {len(X)} samples with {len(feature_names)} features...")
        
        # Train Random Forest
        try:
            model = self.trainer.train_ml_model(
                sport=sport,
                X=X,
                y=y,
                feature_names=feature_names,
                model_type="random_forest",
                n_estimators=200,
                max_depth=10
            )
            
            if model:
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()
                
                metrics = TrainingMetrics(
                    model_name="RandomForest",
                    sport=sport,
                    training_date=end_time,
                    samples_used=len(X),
                    accuracy=model.metadata.accuracy if model.metadata else 0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    validation_accuracy=0.0,
                    training_time_seconds=training_time
                )
                
                self.training_history.append(metrics)
                
                print(f"ML training complete in {training_time:.1f}s")
                print(f"Model accuracy: {metrics.accuracy:.3f}")
                
                return metrics
                
        except Exception as e:
            print(f"ML training error: {e}")
            return None
    
    def _default_feature_extractor(self, matches: List[Dict]) -> tuple:
        """
        Default feature extraction for matches.
        Returns (X, y, feature_names)
        """
        features_list = []
        labels = []
        
        for match in matches:
            # Basic features
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            # Feature vector
            features = [
                home_score / 10.0 if home_score else 0,  # Normalize
                away_score / 10.0 if away_score else 0,
                1.0 if match.get('home_advantage') else 0.0,
                match.get('home_rank', 100) / 100.0,
                match.get('away_rank', 100) / 100.0,
            ]
            
            features_list.append(features)
            
            # Label: 1 if home wins, 0 if away wins
            if home_score > away_score:
                labels.append(1)
            else:
                labels.append(0)
        
        if not features_list:
            return None, None, None
        
        X = np.array(features_list)
        y = np.array(labels)
        feature_names = ['home_score', 'away_score', 'home_advantage', 'home_rank', 'away_rank']
        
        return X, y, feature_names
    
    async def run_training_cycle(self, sport: str):
        """
        Run complete training cycle for a sport.
        
        1. Check if retraining needed
        2. Train statistical models
        3. Train ML models (if data available)
        4. Update ensemble
        """
        if not await self.should_retrain(sport):
            print(f"No retraining needed for {sport}")
            return
        
        print(f"\nStarting training cycle for {sport.upper()}")
        
        # Train statistical models (always)
        stat_metrics = await self.train_statistical_models(sport)
        
        # Train ML models (if possible)
        ml_metrics = await self.train_ml_models(sport)
        
        # Create ensemble if we have multiple models
        if ml_metrics:
            print(f"\nCreating ensemble for {sport}...")
            try:
                ensemble = self.registry.create_ensemble(
                    sport=sport,
                    model_names=["StatisticalEnsemble", "RandomForest"],
                    weights=[0.3, 0.7]
                )
                if ensemble.models:  # If we successfully loaded models
                    self.registry.register(ensemble)
                    print("Ensemble created and registered")
            except Exception as e:
                print(f"Ensemble creation error: {e}")
        
        print(f"\nTraining cycle for {sport} complete!")
    
    async def run_continuous_training(self, sports: List[str], interval_hours: int = 24):
        """
        Run continuous training loop.
        
        Args:
            sports: List of sports to monitor
            interval_hours: Check interval in hours
        """
        print(f"Starting continuous training for: {', '.join(sports)}")
        print(f"Check interval: {interval_hours} hours")
        
        while True:
            for sport in sports:
                try:
                    await self.run_training_cycle(sport)
                except Exception as e:
                    print(f"Training error for {sport}: {e}")
            
            # Wait for next cycle
            print(f"\nNext training check in {interval_hours} hours...")
            await asyncio.sleep(interval_hours * 3600)
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate report of all training runs."""
        return {
            "total_training_runs": len(self.training_history),
            "last_training_per_sport": {
                sport: date.isoformat() 
                for sport, date in self.last_training.items()
            },
            "history": [m.to_dict() for m in self.training_history[-10:]]  # Last 10
        }


# Singleton
trainer_instance = None

def get_online_trainer() -> OnlineTrainer:
    """Get singleton online trainer instance."""
    global trainer_instance
    if trainer_instance is None:
        trainer_instance = OnlineTrainer()
    return trainer_instance
