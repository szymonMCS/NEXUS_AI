# core/models/__init__.py
"""
Prediction models for NEXUS AI.

Includes:
- BaseModel: Abstract base class for all models
- TennisModel: Tennis match prediction (ranking, form, H2H, surface)
- BasketballModel: Basketball prediction (ratings, rest, home/away)
- HandicapModel: Handicap/spread predictions for all sports
- GreyhoundModel: Greyhound racing predictions (SVR/SVM ensemble)
- HandballModel: Handball predictions (SEL with CMP distribution)
- TableTennisModel: Table tennis predictions (XGBoost ensemble)
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

# Additional sports models
from core.models.greyhound_model import (
    GreyhoundModel,
    GreyhoundFeatures,
    RacePrediction,
    RaceGrade,
)
from core.models.handball_model import (
    HandballModel,
    HandballFeatures,
    HandballPrediction,
    HandballMarket,
)
from core.models.table_tennis_model import (
    TableTennisModel,
    TableTennisFeatures,
    TableTennisPrediction,
    TableTennisFormat,
    PlayingStyle,
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
    # Greyhound
    "GreyhoundModel",
    "GreyhoundFeatures",
    "RacePrediction",
    "RaceGrade",
    # Handball
    "HandballModel",
    "HandballFeatures",
    "HandballPrediction",
    "HandballMarket",
    # Table Tennis
    "TableTennisModel",
    "TableTennisFeatures",
    "TableTennisPrediction",
    "TableTennisFormat",
    "PlayingStyle",
]
