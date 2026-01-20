# core/__init__.py
"""
Core module for NEXUS AI.

Includes:
- State management for LangGraph
- Quality scoring
- Prediction models
- Value calculation
- Ensemble management
- Monitoring
"""

from core.state import NexusState
from core.quality_scorer import QualityScorer
from core.value_calculator import ValueCalculator, ValueBet, LeagueType
from core.ensemble import (
    EnsembleManager,
    EnsemblePrediction,
    EnsembleMethod,
    get_ensemble_manager,
)
from core.monitoring import (
    MonitoringService,
    PredictionLog,
    PerformanceMetrics,
    get_monitoring_service,
)

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
    # Ensemble
    "EnsembleManager",
    "EnsemblePrediction",
    "EnsembleMethod",
    "get_ensemble_manager",
    # Monitoring
    "MonitoringService",
    "PredictionLog",
    "PerformanceMetrics",
    "get_monitoring_service",
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
