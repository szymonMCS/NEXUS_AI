"""
NEXUS ML Prediction Service.

Checkpoint: 4.4
"""

from core.ml.service.results import (
    PredictionStatus,
    BettingRecommendation,
    GoalsPredictionResult,
    HandicapPredictionResult,
    MLPredictionResult,
    BatchPredictionResult,
)
from core.ml.service.prediction_service import MLPredictionService
from core.ml.service.ensemble_integration import (
    CombinationMethod,
    AgentPrediction,
    EnsembleConfig,
    EnsemblePrediction,
    EnsembleResult,
    EnsembleIntegration,
    create_default_ensemble,
    create_ml_primary_ensemble,
    create_conservative_ensemble,
)

__all__ = [
    # Results
    "PredictionStatus",
    "BettingRecommendation",
    "GoalsPredictionResult",
    "HandicapPredictionResult",
    "MLPredictionResult",
    "BatchPredictionResult",
    # Service
    "MLPredictionService",
    # Ensemble
    "CombinationMethod",
    "AgentPrediction",
    "EnsembleConfig",
    "EnsemblePrediction",
    "EnsembleResult",
    "EnsembleIntegration",
    "create_default_ensemble",
    "create_ml_primary_ensemble",
    "create_conservative_ensemble",
]
