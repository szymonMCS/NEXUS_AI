"""
ML Models Package.

Available models:
- GoalsModel: Poisson regression for goals prediction
- HandicapModel: GBM for handicap/spread prediction
- RandomForestEnsembleModel: RF ensemble (Research: 81.9% accuracy)
- MLPNeuralNetworkModel: Deep neural network (Research: 86.7% accuracy)
"""

from core.ml.models.interface import MLModelInterface
from core.ml.models.predictions import (
    PredictionResult,
    GoalsPrediction,
    HandicapPrediction,
    ModelInfo,
)
from core.ml.models.goals_model import GoalsModel, PoissonParameters
from core.ml.models.handicap_model import HandicapModel
from core.ml.models.random_forest_model import (
    RandomForestEnsembleModel,
    RFParameters,
)
from core.ml.models.mlp_model import (
    MLPNeuralNetworkModel,
    MLPParameters,
)

__all__ = [
    "MLModelInterface",
    "PredictionResult",
    "GoalsPrediction",
    "HandicapPrediction",
    "ModelInfo",
    "GoalsModel",
    "PoissonParameters",
    "HandicapModel",
    "RandomForestEnsembleModel",
    "RFParameters",
    "MLPNeuralNetworkModel",
    "MLPParameters",
]
