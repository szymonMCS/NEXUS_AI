# core/models/__init__.py
"""
Prediction models for NEXUS AI.

Includes:
- BaseModel: Abstract base class for all models
- TennisModel: Tennis match prediction (ranking, form, H2H, surface)
- BasketballModel: Basketball prediction (ratings, rest, home/away)
- HandicapModel: Handicap/spread predictions for all sports
"""

# Base model
from core.models.base_model import (
    BaseModel,
    PredictionResult,
    ModelMetrics,
    BettingRecommendation,
    Sport,
)

# Sport-specific models
from core.models.tennis_model import TennisModel, TennisFeatures
from core.models.basketball_model import BasketballModel, BasketballFeatures

# Handicap models
from core.models.handicap_model import (
    HandicapModel,
    TennisHandicapModel,
    BasketballHandicapModel,
    HandicapPrediction,
    TotalPrediction,
    HalfStats,
    MarketType,
    find_value_handicap,
)

__all__ = [
    # Base
    "BaseModel",
    "PredictionResult",
    "ModelMetrics",
    "BettingRecommendation",
    "Sport",
    # Tennis
    "TennisModel",
    "TennisFeatures",
    # Basketball
    "BasketballModel",
    "BasketballFeatures",
    # Handicaps
    "HandicapModel",
    "TennisHandicapModel",
    "BasketballHandicapModel",
    "HandicapPrediction",
    "TotalPrediction",
    "HalfStats",
    "MarketType",
    "find_value_handicap",
]
