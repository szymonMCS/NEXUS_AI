"""
NEXUS ML Model Registry.

Checkpoint: 3.3
"""

from core.ml.registry.version import ModelVersion, VersionComparison
from core.ml.registry.registry import ModelRegistry

# New ML system components (from registry.py in parent directory)
# These maintain backward compatibility while adding new features
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

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


class ModelTrainer:
    """Handles model training and retraining workflows."""
    
    def __init__(self, registry):
        self.registry = registry
        self.training_history = []


# Singleton instance
_registry = None

def get_model_registry():
    """Get singleton model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


__all__ = [
    "ModelVersion",
    "VersionComparison",
    "ModelRegistry",
    "ModelInfo",
    "ModelTrainer",
    "get_model_registry",
]
