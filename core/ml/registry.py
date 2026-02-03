"""
Model Registry for NEXUS AI.
Manages model versions, training, and inference.
"""

import json
import pickle
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict

from core.ml.models.base import BasePredictor, PredictionResult, EnsemblePredictor
from core.ml.models.statistical_models import StatisticalEnsemble, EloPredictor, FormPredictor

# Try to import ML models
try:
    from core.ml.models.sklearn_models import RandomForestPredictor, GradientBoostingPredictor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    version: str
    sport: str
    model_type: str
    filepath: str
    created_at: datetime
    accuracy: float = 0.0
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sport": self.sport,
            "model_type": self.model_type,
            "filepath": self.filepath,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accuracy": self.accuracy,
            "is_active": self.is_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """
    Central registry for all ML models.
    Handles model storage, versioning, and retrieval.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.models_dir / "registry.json"
        self.models: Dict[str, ModelInfo] = {}  # key: "sport/name:version"
        self.loaded_models: Dict[str, BasePredictor] = {}  # Cache of loaded models
        
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for key, model_data in data.items():
                    self.models[key] = ModelInfo.from_dict(model_data)
                print(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                print(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save model registry to disk."""
        data = {k: v.to_dict() for k, v in self.models.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_key(self, sport: str, name: str, version: str) -> str:
        """Generate registry key."""
        return f"{sport}/{name}:{version}"
    
    def register(self, model: BasePredictor, filepath: Optional[str] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model: Trained model to register
            filepath: Optional custom filepath
            
        Returns:
            Registry key
        """
        sport = model.sport
        name = model.name
        version = model.version
        
        if filepath is None:
            filepath = str(self.models_dir / sport / f"{name}_v{version}.pkl")
        
        # Save model
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model.save(filepath)
        
        # Register
        key = self._get_key(sport, name, version)
        self.models[key] = ModelInfo(
            name=name,
            version=version,
            sport=sport,
            model_type=type(model).__name__,
            filepath=filepath,
            created_at=datetime.now(),
            accuracy=getattr(model.metadata, 'accuracy', 0.0),
            is_active=True,
            metadata=model.metadata.to_dict() if model.metadata else {}
        )
        
        self._save_registry()
        print(f"Registered model: {key}")
        
        return key
    
    def load(self, sport: str, name: str, version: Optional[str] = None) -> Optional[BasePredictor]:
        """
        Load a model from registry.
        
        Args:
            sport: Sport type
            name: Model name
            version: Specific version (None for latest)
            
        Returns:
            Loaded model or None
        """
        # Find model info
        if version:
            key = self._get_key(sport, name, version)
            info = self.models.get(key)
        else:
            # Find latest version
            matching = [
                (k, v) for k, v in self.models.items()
                if v.sport == sport and v.name == name and v.is_active
            ]
            if not matching:
                return None
            # Sort by created_at and get latest
            matching.sort(key=lambda x: x[1].created_at, reverse=True)
            key, info = matching[0]
        
        if not info:
            return None
        
        # Check cache
        if key in self.loaded_models:
            return self.loaded_models[key]
        
        # Load based on type
        model = self._create_model_instance(info.model_type, name, info.version, sport)
        if model and model.load(info.filepath):
            self.loaded_models[key] = model
            return model
        
        return None
    
    def _create_model_instance(self, model_type: str, name: str, version: str, sport: str) -> Optional[BasePredictor]:
        """Create empty model instance based on type."""
        if model_type == "RandomForestPredictor" and SKLEARN_AVAILABLE:
            return RandomForestPredictor(name, version, sport)
        elif model_type == "GradientBoostingPredictor" and SKLEARN_AVAILABLE:
            return GradientBoostingPredictor(name, version, sport)
        elif model_type == "StatisticalEnsemble":
            return StatisticalEnsemble(name, version, sport)
        elif model_type == "EloPredictor":
            return EloPredictor(name, version, sport)
        elif model_type == "FormPredictor":
            return FormPredictor(name, version, sport)
        elif model_type == "EnsemblePredictor":
            return EnsemblePredictor(name, version, sport)
        else:
            print(f"Unknown model type: {model_type}")
            return None
    
    def get_active_models(self, sport: Optional[str] = None) -> List[ModelInfo]:
        """Get list of active models."""
        models = [m for m in self.models.values() if m.is_active]
        if sport:
            models = [m for m in models if m.sport == sport]
        return models
    
    def deactivate(self, sport: str, name: str, version: str):
        """Deactivate a model version."""
        key = self._get_key(sport, name, version)
        if key in self.models:
            self.models[key].is_active = False
            self._save_registry()
    
    def get_best_model(self, sport: str) -> Optional[BasePredictor]:
        """
        Get best active model for a sport based on accuracy.
        Falls back to statistical models if no ML models available.
        """
        active = self.get_active_models(sport)
        
        if not active:
            # Create default statistical model
            print(f"No registered models for {sport}, creating statistical ensemble...")
            model = StatisticalEnsemble(sport=sport)
            # Try to load existing statistical model
            stat_path = self.models_dir / sport / "statistical_ensemble.json"
            if stat_path.exists():
                model.load(str(stat_path))
            return model
        
        # Sort by accuracy
        active.sort(key=lambda x: x.accuracy, reverse=True)
        best = active[0]
        
        return self.load(best.sport, best.name, best.version)
    
    def list_models(self, sport: Optional[str] = None) -> Dict[str, List[ModelInfo]]:
        """List all models grouped by sport."""
        result = {}
        for info in self.models.values():
            s = info.sport
            if sport and s != sport:
                continue
            if s not in result:
                result[s] = []
            result[s].append(info)
        return result
    
    def create_ensemble(self, sport: str, model_names: List[str], weights: Optional[List[float]] = None) -> EnsemblePredictor:
        """
        Create ensemble from multiple models.
        
        Args:
            sport: Sport type
            model_names: List of model names to include
            weights: Optional weights for each model
            
        Returns:
            Ensemble predictor
        """
        ensemble = EnsemblePredictor(name=f"{sport}_ensemble", sport=sport)
        
        if weights is None:
            weights = [1.0] * len(model_names)
        
        for name, weight in zip(model_names, weights):
            model = self.load(sport, name)
            if model:
                ensemble.add_model(model, weight)
        
        return ensemble


class ModelTrainer:
    """
    Handles model training and retraining workflows.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.training_history: List[Dict] = []
    
    def train_statistical_models(self, sport: str, matches: List[Dict[str, Any]]) -> StatisticalEnsemble:
        """
        Train statistical models (ELO + Form + H2H).
        
        Args:
            sport: Sport type
            matches: Historical matches with results
            
        Returns:
            Trained StatisticalEnsemble
        """
        print(f"Training statistical models for {sport}...")
        
        model = StatisticalEnsemble(sport=sport)
        model.fit(matches)
        
        # Save and register
        filepath = self.registry.models_dir / sport / "statistical_ensemble.json"
        model.save(str(filepath))
        
        self.registry.register(model, str(filepath))
        
        print(f"Statistical models trained on {len(matches)} matches")
        
        return model
    
    def train_ml_model(self, 
                       sport: str,
                       X: np.ndarray, 
                       y: np.ndarray,
                       feature_names: List[str],
                       model_type: str = "random_forest",
                       **kwargs) -> Optional[BasePredictor]:
        """
        Train ML model from scratch.
        
        Args:
            sport: Sport type
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            model_type: "random_forest" or "gradient_boosting"
            **kwargs: Model hyperparameters
            
        Returns:
            Trained model or None if sklearn not available
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available, skipping ML training")
            return None
        
        print(f"Training {model_type} model for {sport}...")
        
        if model_type == "random_forest":
            model = RandomForestPredictor(
                sport=sport,
                **kwargs
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingPredictor(
                sport=sport,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y, feature_names=feature_names)
        
        # Register
        self.registry.register(model)
        
        print(f"Model trained: accuracy={model.metadata.accuracy:.3f}")
        
        return model
    
    def retrain_if_needed(self, sport: str, min_matches: int = 100) -> bool:
        """
        Check if model needs retraining and retrain if necessary.
        
        Args:
            sport: Sport type
            min_matches: Minimum new matches required for retraining
            
        Returns:
            True if retrained
        """
        # This would check database for new matches since last training
        # For now, placeholder
        return False


# Singleton instance
_registry = None

def get_model_registry() -> ModelRegistry:
    """Get singleton model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
