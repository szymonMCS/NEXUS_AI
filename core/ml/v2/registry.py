"""
Simple Model Registry for NEXUS AI v2.
Lightweight implementation without database dependencies.
"""

import json
import pickle
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from core.ml.v2.models_base import BasePredictor, EnsemblePredictor
from core.ml.v2.statistical_models import StatisticalEnsemble

# Try to import ML models
try:
    from core.ml.models.sklearn_models import RandomForestPredictor, GradientBoostingPredictor
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestPredictor = None
    GradientBoostingPredictor = None
    SKLEARN_AVAILABLE = False


class SimpleModelRegistry:
    """
    Simple in-memory model registry.
    Stores models in filesystem and loads on demand.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, BasePredictor] = {}
    
    def _get_model_path(self, sport: str, name: str, version: str = "latest") -> Path:
        """Get path to model file."""
        sport_dir = self.models_dir / sport
        sport_dir.mkdir(exist_ok=True)
        
        if version == "latest":
            # Find latest version
            files = list(sport_dir.glob(f"{name}_v*.pkl"))
            if files:
                return max(files, key=lambda p: p.stat().st_mtime)
            return sport_dir / f"{name}.pkl"
        
        return sport_dir / f"{name}_v{version}.pkl"
    
    def save_model(self, model: BasePredictor) -> str:
        """
        Save model to registry.
        
        Args:
            model: Trained model
            
        Returns:
            Path to saved model
        """
        sport_dir = self.models_dir / model.sport
        sport_dir.mkdir(exist_ok=True)
        
        filepath = sport_dir / f"{model.name}_v{model.version}.pkl"
        
        # Save
        model.save(str(filepath))
        
        # Update cache
        key = f"{model.sport}/{model.name}:{model.version}"
        self._cache[key] = model
        
        return str(filepath)
    
    def load_model(self, sport: str, name: str, version: str = "latest") -> Optional[BasePredictor]:
        """
        Load model from registry.
        
        Args:
            sport: Sport type
            name: Model name
            version: Model version (default: latest)
            
        Returns:
            Loaded model or None
        """
        key = f"{sport}/{name}:{version}"
        
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Find file
        filepath = self._get_model_path(sport, name, version)
        
        if not filepath.exists():
            return None
        
        # Create appropriate model type
        model = self._create_model(name, version, sport)
        if model and model.load(str(filepath)):
            self._cache[key] = model
            return model
        
        return None
    
    def _create_model(self, name: str, version: str, sport: str) -> Optional[BasePredictor]:
        """Create empty model instance."""
        # Default to statistical ensemble
        return StatisticalEnsemble(name=name, version=version, sport=sport)
    
    def get_best_model(self, sport: str) -> BasePredictor:
        """
        Get best available model for a sport.
        Returns statistical ensemble if no trained models exist.
        """
        # Try to load existing statistical model
        model = StatisticalEnsemble(sport=sport)
        
        # Check for saved statistical model
        stat_path = self.models_dir / sport / "statistical_ensemble.json"
        if stat_path.exists():
            try:
                model.load(str(stat_path))
                return model
            except Exception:
                pass
        
        # Return fresh statistical model
        return model
    
    def get_active_models(self, sport: str) -> List[Any]:
        """Get list of active models for a sport (compatibility method)."""
        # Return empty list - models are loaded on demand
        return []
    
    def list_models(self, sport: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []
        
        search_dir = self.models_dir
        if sport:
            search_dir = search_dir / sport
        
        if not search_dir.exists():
            return models
        
        for filepath in search_dir.rglob("*.pkl"):
            relative = filepath.relative_to(self.models_dir)
            models.append({
                "path": str(relative),
                "size": filepath.stat().st_size,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            })
        
        return models
    
    def clear_cache(self):
        """Clear model cache."""
        self._cache.clear()


# Singleton
_registry: Optional[SimpleModelRegistry] = None

def get_model_registry() -> SimpleModelRegistry:
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = SimpleModelRegistry()
    return _registry
