"""
Base classes for ML models in NEXUS AI.
All prediction models should inherit from BasePredictor.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
import json
from pathlib import Path
import numpy as np


@dataclass
class PredictionResult:
    """Standardized prediction result."""
    home_win_prob: float
    away_win_prob: float
    draw_prob: Optional[float] = None
    confidence: float = 0.0
    model_name: str = ""
    version: str = "1.0"
    features_used: List[str] = None
    raw_output: Any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "draw_prob": self.draw_prob,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "version": self.version,
            "features_used": self.features_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class ModelMetadata:
    """Metadata for model tracking."""
    name: str
    version: str
    sport: str
    created_at: datetime
    training_samples: int = 0
    accuracy: float = 0.0
    f1_score: float = 0.0
    features: List[str] = None
    hyperparameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.hyperparameters is None:
            self.hyperparameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sport": self.sport,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "training_samples": self.training_samples,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "features": self.features,
            "hyperparameters": self.hyperparameters
        }


class BasePredictor(ABC):
    """
    Abstract base class for all prediction models.
    
    All models must implement:
    - fit: Training method
    - predict: Prediction method
    - save/load: Model persistence
    - get_feature_importance: Explainability
    """
    
    def __init__(self, name: str, version: str = "1.0", sport: str = "general"):
        self.name = name
        self.version = version
        self.sport = sport
        self.is_trained = False
        self.metadata = ModelMetadata(
            name=name,
            version=version,
            sport=sport,
            created_at=datetime.now()
        )
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Make prediction for a single match.
        
        Args:
            features: Dictionary of match features
            
        Returns:
            PredictionResult with probabilities and metadata
        """
        pass
    
    @abstractmethod
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Make predictions for multiple matches.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of PredictionResult
        """
        pass
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (default: auto-generated)
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            filepath = f"models/{self.sport}/{self.name}_v{self.version}.pkl"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self.model,
            "metadata": self.metadata,
            "name": self.name,
            "version": self.version,
            "sport": self.sport,
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Save metadata separately as JSON
        meta_path = filepath.replace('.pkl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        return filepath
    
    def load(self, filepath: str) -> bool:
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
            
            self.model = save_dict["model"]
            self.metadata = save_dict["metadata"]
            self.name = save_dict["name"]
            self.version = save_dict["version"]
            self.sport = save_dict["sport"]
            self.is_trained = save_dict["is_trained"]
            
            return True
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for model explainability.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def get_confidence_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate confidence score for prediction.
        Can be overridden by subclasses.
        
        Args:
            features: Match features
            
        Returns:
            Confidence score 0.0-1.0
        """
        # Default: use model's own confidence if available
        if hasattr(self.model, 'predict_proba'):
            return 0.7  # Placeholder
        return 0.5
    
    def update_metadata(self, **kwargs):
        """Update model metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)


class EnsemblePredictor(BasePredictor):
    """
    Ensemble of multiple predictors with weighted voting.
    """
    
    def __init__(self, name: str = "Ensemble", version: str = "1.0", sport: str = "general"):
        super().__init__(name, version, sport)
        self.models: List[Tuple[BasePredictor, float]] = []  # (model, weight)
        
    def add_model(self, model: BasePredictor, weight: float = 1.0):
        """Add model to ensemble."""
        self.models.append((model, weight))
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train all models in ensemble."""
        for model, _ in self.models:
            model.fit(X, y, **kwargs)
        self.is_trained = all(m.is_trained for m, _ in self.models)
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Weighted average of all models."""
        if not self.models:
            return PredictionResult(0.5, 0.5, confidence=0.0, model_name=self.name)
        
        predictions = []
        weights = []
        
        for model, weight in self.models:
            pred = model.predict(features)
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average
        total_weight = sum(weights)
        home_prob = sum(p.home_win_prob * w for p, w in zip(predictions, weights)) / total_weight
        away_prob = sum(p.away_win_prob * w for p, w in zip(predictions, weights)) / total_weight
        
        # Confidence based on agreement between models
        probs = [p.home_win_prob for p in predictions]
        confidence = 1.0 - (np.std(probs) * 2)  # Lower std = higher confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return PredictionResult(
            home_win_prob=home_prob,
            away_win_prob=away_prob,
            confidence=confidence,
            model_name=self.name,
            version=self.version,
            features_used=list(features.keys())
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        return [self.predict(f) for f in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Average feature importance across models."""
        if not self.models:
            return {}
        
        importance = {}
        for model, weight in self.models:
            model_imp = model.get_feature_importance()
            for feature, imp in model_imp.items():
                importance[feature] = importance.get(feature, 0) + imp * weight
        
        total_weight = sum(w for _, w in self.models)
        return {k: v / total_weight for k, v in importance.items()}
