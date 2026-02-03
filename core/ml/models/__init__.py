"""
ML Models for NEXUS AI.
"""

# Import base classes directly from v2
from core.ml.v2.models_base import (
    BasePredictor,
    PredictionResult,
    ModelMetadata,
    EnsemblePredictor
)

# Statistical models
from core.ml.v2.statistical_models import (
    StatisticalEnsemble,
    EloPredictor,
    FormPredictor
)

# Try sklearn models
try:
    from core.ml.models.sklearn_models import (
        RandomForestPredictor,
        GradientBoostingPredictor
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestPredictor = None
    GradientBoostingPredictor = None
    SKLEARN_AVAILABLE = False

__all__ = [
    'BasePredictor',
    'PredictionResult',
    'ModelMetadata',
    'EnsemblePredictor',
    'StatisticalEnsemble',
    'EloPredictor',
    'FormPredictor',
    'RandomForestPredictor',
    'GradientBoostingPredictor',
    'SKLEARN_AVAILABLE',
]
