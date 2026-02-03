"""
Machine Learning module for NEXUS AI - Unified.
"""

# Import in order to avoid circular dependencies

# 1. First import base classes (no dependencies)
from core.ml.v2.models_base import (
    BasePredictor,
    PredictionResult,
    ModelMetadata,
    EnsemblePredictor
)

# 2. Import registry (depends on base)
from core.ml.v2.registry import (
    SimpleModelRegistry,
    get_model_registry
)
ModelRegistry = SimpleModelRegistry

# 3. Import prediction service (depends on registry)
from core.ml.v2.prediction_service import (
    PredictionService,
    PredictionExplanation,
    get_prediction_service
)

# 3.5 Import ensemble aggregator
from core.ml.ensemble_aggregator import (
    EnsembleAggregator,
    EnsemblePrediction,
    ModelWeight
)

# 4. Import statistical models (depends on base)
from core.ml.v2.statistical_models import (
    StatisticalEnsemble,
    EloPredictor,
    FormPredictor
)

# 5. Try ML models (optional, may fail)
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
    # Base
    'BasePredictor',
    'PredictionResult',
    'ModelMetadata',
    'EnsemblePredictor',
    # Statistical
    'StatisticalEnsemble',
    'EloPredictor',
    'FormPredictor',
    # ML
    'RandomForestPredictor',
    'GradientBoostingPredictor',
    'SKLEARN_AVAILABLE',
    # Registry
    'SimpleModelRegistry',
    'ModelRegistry',
    'get_model_registry',
    # Service
    'PredictionService',
    'PredictionExplanation',
    'get_prediction_service',
    # Ensemble
    'EnsembleAggregator',
    'EnsemblePrediction',
    'ModelWeight',
]
