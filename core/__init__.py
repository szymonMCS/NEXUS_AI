# core/__init__.py
"""
Core module for NEXUS AI.

Includes:
- State management for LangGraph
- Quality scoring
- Prediction models
- Value calculation
"""

from core.state import NexusState
from core.quality_scorer import QualityScorer
from core.value_calculator import ValueCalculator, ValueBet, LeagueType

# Re-export models
from core.models import (
    BaseModel,
    PredictionResult,
    ModelMetrics,
    BettingRecommendation,
    Sport,
    TennisModel,
    BasketballModel,
    HandicapModel,
    TennisHandicapModel,
    BasketballHandicapModel,
)

__all__ = [
    # State
    "NexusState",
    # Quality
    "QualityScorer",
    # Value
    "ValueCalculator",
    "ValueBet",
    "LeagueType",
    # Models
    "BaseModel",
    "PredictionResult",
    "ModelMetrics",
    "BettingRecommendation",
    "Sport",
    "TennisModel",
    "BasketballModel",
    "HandicapModel",
    "TennisHandicapModel",
    "BasketballHandicapModel",
]
