"""
NEXUS ML Feature Engineering.

Checkpoint: 1.7
Exports all feature extraction components.
"""

from core.ml.features.vector import FeatureVector
from core.ml.features.base import BaseFeatureExtractor
from core.ml.features.goals_features import GoalsFeatureExtractor
from core.ml.features.handicap_features import HandicapFeatureExtractor
from core.ml.features.form_features import FormFeatureExtractor
from core.ml.features.pipeline import (
    FeaturePipeline,
    PipelineConfig,
    create_goals_pipeline,
    create_handicap_pipeline,
    create_full_pipeline,
    get_pipeline,
)

__all__ = [
    # Core
    "FeatureVector",
    "BaseFeatureExtractor",
    # Extractors
    "GoalsFeatureExtractor",
    "HandicapFeatureExtractor",
    "FormFeatureExtractor",
    # Pipeline
    "FeaturePipeline",
    "PipelineConfig",
    # Factory functions
    "create_goals_pipeline",
    "create_handicap_pipeline",
    "create_full_pipeline",
    "get_pipeline",
]
