"""
Feature engineering pipeline.

Checkpoint: 1.6
Responsibility: Orchestrate feature extraction from multiple extractors.
Principle: Single entry point for all feature extraction.
"""

from typing import Dict, List, Optional, Type
import logging
from dataclasses import dataclass, field

from core.data.schemas import MatchData, DataQuality
from core.ml.features.base import BaseFeatureExtractor
from core.ml.features.vector import FeatureVector
from core.ml.features.goals_features import GoalsFeatureExtractor
from core.ml.features.handicap_features import HandicapFeatureExtractor
from core.ml.features.form_features import FormFeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for feature pipeline."""
    use_goals_features: bool = True
    use_handicap_features: bool = True
    use_form_features: bool = True
    normalize_features: bool = True
    fill_missing_with_zero: bool = True
    min_quality_threshold: float = 0.3


@dataclass
class FeatureStats:
    """Statistics for feature normalization."""
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0


class FeaturePipeline:
    """
    Pipeline dla ekstrakcji i przetwarzania cech.

    Łączy wiele ekstraktorów w jeden spójny interfejs:
    1. Walidacja danych wejściowych
    2. Ekstrakcja cech z każdego ekstraktora
    3. Normalizacja (opcjonalnie)
    4. Tworzenie FeatureVector

    Usage:
        pipeline = FeaturePipeline()
        features = pipeline.extract(match)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self._extractors: List[BaseFeatureExtractor] = []
        self._feature_stats: Dict[str, FeatureStats] = {}

        # Initialize extractors based on config
        self._init_extractors()

        logger.info(f"FeaturePipeline initialized with {len(self._extractors)} extractors")

    def _init_extractors(self) -> None:
        """Initialize extractors based on config."""
        if self.config.use_goals_features:
            self._extractors.append(GoalsFeatureExtractor())

        if self.config.use_handicap_features:
            self._extractors.append(HandicapFeatureExtractor())

        if self.config.use_form_features:
            self._extractors.append(FormFeatureExtractor())

    def add_extractor(self, extractor: BaseFeatureExtractor) -> None:
        """
        Add a custom extractor to the pipeline.

        Args:
            extractor: Feature extractor instance
        """
        self._extractors.append(extractor)
        logger.info(f"Added extractor: {extractor.name}")

    def remove_extractor(self, name: str) -> bool:
        """
        Remove an extractor by name.

        Args:
            name: Extractor name

        Returns:
            True if removed, False if not found
        """
        for i, ext in enumerate(self._extractors):
            if ext.name == name:
                self._extractors.pop(i)
                logger.info(f"Removed extractor: {name}")
                return True
        return False

    def extract(self, match: MatchData) -> FeatureVector:
        """
        Extract all features from a match.

        Args:
            match: Match data

        Returns:
            FeatureVector with all extracted features
        """
        all_features: Dict[str, float] = {}
        extractor_names: List[str] = []

        for extractor in self._extractors:
            try:
                features = extractor.extract_safe(match)

                # Prefix features with extractor name for clarity
                for name, value in features.items():
                    prefixed_name = f"{extractor.name}_{name}"
                    all_features[prefixed_name] = value

                extractor_names.append(extractor.name)

            except Exception as e:
                logger.error(f"Error in extractor {extractor.name}: {e}")
                # Add zero features on error
                for name in extractor.get_feature_names():
                    all_features[f"{extractor.name}_{name}"] = 0.0

        # Normalize if configured
        if self.config.normalize_features and self._feature_stats:
            all_features = self._normalize(all_features)

        # Calculate quality based on extractors that succeeded
        quality = self._calculate_quality(match, extractor_names)

        return FeatureVector(
            features=all_features,
            match_id=match.match_id,
            quality=quality,
            extractor_names=extractor_names,
        )

    def extract_batch(self, matches: List[MatchData]) -> List[FeatureVector]:
        """
        Extract features from multiple matches.

        Args:
            matches: List of match data

        Returns:
            List of FeatureVectors
        """
        return [self.extract(match) for match in matches]

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names from all extractors.

        Returns:
            Sorted list of all feature names
        """
        names = []
        for extractor in self._extractors:
            for name in extractor.get_feature_names():
                names.append(f"{extractor.name}_{name}")
        return sorted(names)

    def get_extractor_names(self) -> List[str]:
        """Get names of all active extractors."""
        return [e.name for e in self._extractors]

    def fit_normalizer(self, feature_vectors: List[FeatureVector]) -> None:
        """
        Fit normalization statistics from training data.

        Args:
            feature_vectors: List of training feature vectors
        """
        if not feature_vectors:
            return

        # Collect all values for each feature
        feature_values: Dict[str, List[float]] = {}

        for fv in feature_vectors:
            for name, value in fv.features.items():
                if name not in feature_values:
                    feature_values[name] = []
                feature_values[name].append(value)

        # Calculate statistics
        for name, values in feature_values.items():
            if len(values) == 0:
                continue

            import statistics

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 1.0

            self._feature_stats[name] = FeatureStats(
                mean=mean,
                std=std if std > 0 else 1.0,
                min_val=min(values),
                max_val=max(values),
            )

        logger.info(f"Fitted normalizer on {len(feature_vectors)} vectors")

    def _normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features using z-score normalization.

        Args:
            features: Raw features

        Returns:
            Normalized features
        """
        normalized = {}

        for name, value in features.items():
            if name in self._feature_stats:
                stats = self._feature_stats[name]
                normalized[name] = (value - stats.mean) / stats.std
            else:
                normalized[name] = value

        return normalized

    def _calculate_quality(
        self,
        match: MatchData,
        successful_extractors: List[str],
    ) -> DataQuality:
        """
        Calculate feature quality based on extraction success.

        Args:
            match: Original match data
            successful_extractors: Names of extractors that succeeded

        Returns:
            DataQuality reflecting extraction success
        """
        # Base quality from match
        base_quality = match.data_quality

        # Adjust completeness based on extractor success
        extractor_ratio = len(successful_extractors) / max(len(self._extractors), 1)
        adjusted_completeness = base_quality.completeness * extractor_ratio

        return DataQuality(
            completeness=adjusted_completeness,
            freshness_hours=base_quality.freshness_hours,
            sources_count=base_quality.sources_count,
            has_h2h=base_quality.has_h2h,
            has_form=base_quality.has_form,
            has_odds=base_quality.has_odds,
        )

    def can_extract(self, match: MatchData) -> bool:
        """
        Check if any extractor can process this match.

        Args:
            match: Match data

        Returns:
            True if at least one extractor can extract features
        """
        for extractor in self._extractors:
            if extractor.can_extract(match):
                return True
        return False

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "extractors_count": len(self._extractors),
            "extractor_names": self.get_extractor_names(),
            "total_features": len(self.get_feature_names()),
            "normalization_fitted": len(self._feature_stats) > 0,
        }


# Convenience factory functions
def create_goals_pipeline() -> FeaturePipeline:
    """Create pipeline for goals/over-under prediction only."""
    config = PipelineConfig(
        use_goals_features=True,
        use_handicap_features=False,
        use_form_features=True,
    )
    return FeaturePipeline(config)


def create_handicap_pipeline() -> FeaturePipeline:
    """Create pipeline for handicap/spread prediction only."""
    config = PipelineConfig(
        use_goals_features=False,
        use_handicap_features=True,
        use_form_features=True,
    )
    return FeaturePipeline(config)


def create_full_pipeline() -> FeaturePipeline:
    """Create pipeline with all feature extractors."""
    return FeaturePipeline()


# Default pipeline instance
_default_pipeline: Optional[FeaturePipeline] = None


def get_pipeline() -> FeaturePipeline:
    """Get or create default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = create_full_pipeline()
    return _default_pipeline
