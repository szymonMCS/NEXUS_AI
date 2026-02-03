"""
NEXUS AI ML v2 - Internal module.

NOTE: Use core.ml module for public interface.
This module is for internal organization only.
"""

# This module intentionally minimal to avoid circular imports
# Public exports are in core.ml

# Only export base classes for internal use
from core.ml.v2.models_base import (
    BasePredictor,
    PredictionResult,
    ModelMetadata
)

__all__ = [
    'BasePredictor',
    'PredictionResult',
    'ModelMetadata',
]
