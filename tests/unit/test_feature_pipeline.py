"""
Tests for feature pipeline.

Checkpoint: 1.9
Integration tests for FeaturePipeline.
"""

import pytest
from datetime import datetime
from typing import Dict, List

from core.data.enums import Sport
from core.data.schemas import (
    MatchData,
    TeamData,
    TeamMatchStats,
    HistoricalMatch,
    DataQuality,
)
from core.ml.features import (
    FeaturePipeline,
    PipelineConfig,
    FeatureVector,
    BaseFeatureExtractor,
    create_goals_pipeline,
    create_handicap_pipeline,
    create_full_pipeline,
    get_pipeline,
)


@pytest.fixture
def sample_match():
    return MatchData(
        match_id="pipeline-test-1",
        sport=Sport.FOOTBALL,
        home_team=TeamData(team_id="home-1", name="Home FC"),
        away_team=TeamData(team_id="away-1", name="Away FC"),
        league="Test League",
        start_time=datetime.utcnow(),
        home_stats=TeamMatchStats(
            goals_scored_avg=1.8,
            goals_conceded_avg=1.0,
            form_points=0.7,
            rest_days=5,
        ),
        away_stats=TeamMatchStats(
            goals_scored_avg=1.3,
            goals_conceded_avg=1.5,
            form_points=0.5,
            rest_days=3,
        ),
        data_quality=DataQuality(
            completeness=0.8,
            freshness_hours=1,
            sources_count=2,
        ),
    )


@pytest.fixture
def sample_matches(sample_match):
    """Create multiple sample matches for batch testing."""
    matches = [sample_match]

    for i in range(1, 5):
        match = MatchData(
            match_id=f"pipeline-test-{i+1}",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id=f"home-{i+1}", name=f"Home {i+1}"),
            away_team=TeamData(team_id=f"away-{i+1}", name=f"Away {i+1}"),
            league="Test League",
            start_time=datetime.utcnow(),
            home_stats=TeamMatchStats(
                goals_scored_avg=1.5 + i * 0.1,
                goals_conceded_avg=1.0 + i * 0.1,
                form_points=0.5 + i * 0.05,
                rest_days=3 + i,
            ),
            away_stats=TeamMatchStats(
                goals_scored_avg=1.2 + i * 0.1,
                goals_conceded_avg=1.3 + i * 0.1,
                form_points=0.4 + i * 0.05,
                rest_days=4,
            ),
            data_quality=DataQuality(
                completeness=0.7 + i * 0.05,
                freshness_hours=1,
                sources_count=2,
            ),
        )
        matches.append(match)

    return matches


class CustomTestExtractor(BaseFeatureExtractor):
    """Custom extractor for testing."""

    @property
    def name(self) -> str:
        return "custom_test"

    def get_feature_names(self) -> List[str]:
        return ["custom_feature_1", "custom_feature_2"]

    def extract(self, match: MatchData) -> Dict[str, float]:
        return {
            "custom_feature_1": 1.0,
            "custom_feature_2": 2.0,
        }


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_init_default(self):
        pipeline = FeaturePipeline()
        stats = pipeline.get_stats()

        assert stats["extractors_count"] == 3  # goals, handicap, form
        assert "goals" in stats["extractor_names"]
        assert "handicap" in stats["extractor_names"]
        assert "form" in stats["extractor_names"]

    def test_init_with_config(self):
        config = PipelineConfig(
            use_goals_features=True,
            use_handicap_features=False,
            use_form_features=False,
        )
        pipeline = FeaturePipeline(config)

        assert pipeline.get_stats()["extractors_count"] == 1
        assert "goals" in pipeline.get_extractor_names()

    def test_extract_returns_feature_vector(self, sample_match):
        pipeline = FeaturePipeline()
        result = pipeline.extract(sample_match)

        assert isinstance(result, FeatureVector)
        assert result.match_id == "pipeline-test-1"
        assert len(result.features) > 0

    def test_extract_prefixes_features(self, sample_match):
        pipeline = FeaturePipeline()
        result = pipeline.extract(sample_match)

        # Features should be prefixed with extractor name
        feature_names = list(result.features.keys())
        assert any(n.startswith("goals_") for n in feature_names)
        assert any(n.startswith("handicap_") for n in feature_names)
        assert any(n.startswith("form_") for n in feature_names)

    def test_extract_quality_tracking(self, sample_match):
        pipeline = FeaturePipeline()
        result = pipeline.extract(sample_match)

        # Quality should reflect original match quality
        assert result.quality.completeness > 0
        assert result.quality.sources_count > 0

    def test_extract_extractor_names_tracked(self, sample_match):
        pipeline = FeaturePipeline()
        result = pipeline.extract(sample_match)

        assert "goals" in result.extractor_names
        assert "handicap" in result.extractor_names
        assert "form" in result.extractor_names

    def test_extract_batch(self, sample_matches):
        pipeline = FeaturePipeline()
        results = pipeline.extract_batch(sample_matches)

        assert len(results) == len(sample_matches)
        assert all(isinstance(r, FeatureVector) for r in results)

        # Each result should have unique match_id
        match_ids = [r.match_id for r in results]
        assert len(set(match_ids)) == len(match_ids)

    def test_get_feature_names(self):
        pipeline = FeaturePipeline()
        names = pipeline.get_feature_names()

        assert len(names) > 0
        # Should be sorted
        assert names == sorted(names)
        # All should be prefixed
        assert all("_" in n for n in names)

    def test_add_extractor(self, sample_match):
        pipeline = FeaturePipeline()
        initial_count = pipeline.get_stats()["extractors_count"]

        custom = CustomTestExtractor()
        pipeline.add_extractor(custom)

        assert pipeline.get_stats()["extractors_count"] == initial_count + 1
        assert "custom_test" in pipeline.get_extractor_names()

        # Extract should include custom features
        result = pipeline.extract(sample_match)
        assert "custom_test_custom_feature_1" in result.features

    def test_remove_extractor(self):
        pipeline = FeaturePipeline()
        initial_count = pipeline.get_stats()["extractors_count"]

        success = pipeline.remove_extractor("form")
        assert success is True
        assert pipeline.get_stats()["extractors_count"] == initial_count - 1
        assert "form" not in pipeline.get_extractor_names()

    def test_remove_extractor_not_found(self):
        pipeline = FeaturePipeline()
        success = pipeline.remove_extractor("nonexistent")
        assert success is False

    def test_can_extract(self, sample_match):
        pipeline = FeaturePipeline()
        assert pipeline.can_extract(sample_match) is True

    def test_can_extract_no_stats(self):
        pipeline = FeaturePipeline()

        match = MatchData(
            match_id="test",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id="h", name="Home"),
            away_team=TeamData(team_id="a", name="Away"),
            league="Test",
            start_time=datetime.utcnow(),
        )

        # Without stats, extractors should still work (extract_safe)
        assert pipeline.can_extract(match) is False

    def test_extract_handles_missing_data_gracefully(self):
        pipeline = FeaturePipeline()

        match = MatchData(
            match_id="incomplete",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id="h", name="Home"),
            away_team=TeamData(team_id="a", name="Away"),
            league="Test",
            start_time=datetime.utcnow(),
            data_quality=DataQuality(
                completeness=0.1,
                freshness_hours=1,
                sources_count=1,
            ),
        )

        # Should not raise, should return zeros
        result = pipeline.extract(match)
        assert isinstance(result, FeatureVector)
        assert all(v == 0.0 for v in result.features.values())


class TestFeatureNormalization:
    """Tests for feature normalization."""

    def test_fit_normalizer(self, sample_matches):
        pipeline = FeaturePipeline()

        # Extract features first
        vectors = pipeline.extract_batch(sample_matches)

        # Fit normalizer
        pipeline.fit_normalizer(vectors)

        assert pipeline._feature_stats
        assert pipeline.get_stats()["normalization_fitted"] is True

    def test_normalize_after_fit(self, sample_matches):
        # Create pipeline with normalization
        config = PipelineConfig(normalize_features=True)
        pipeline = FeaturePipeline(config)

        # Extract and fit
        vectors = pipeline.extract_batch(sample_matches)
        pipeline.fit_normalizer(vectors)

        # Extract again - should be normalized
        normalized = pipeline.extract(sample_matches[0])

        # Normalized values should be different from original
        original = vectors[0]

        # At least some values should differ
        different_count = sum(
            1 for k in original.features
            if abs(original.features[k] - normalized.features.get(k, 0)) > 0.001
        )
        assert different_count > 0


class TestFeatureVectorOperations:
    """Tests for FeatureVector operations."""

    def test_to_array(self, sample_match):
        pipeline = FeaturePipeline()
        fv = pipeline.extract(sample_match)

        arr = fv.to_array()
        assert len(arr) == len(fv.features)

    def test_to_array_with_specific_names(self, sample_match):
        pipeline = FeaturePipeline()
        fv = pipeline.extract(sample_match)

        names = ["goals_home_goals_scored_avg", "goals_away_goals_scored_avg"]
        arr = fv.to_array(names)

        assert len(arr) == 2
        assert arr[0] == fv.features["goals_home_goals_scored_avg"]

    def test_merge_vectors(self, sample_match):
        # Create two feature vectors for same match
        config1 = PipelineConfig(
            use_goals_features=True,
            use_handicap_features=False,
            use_form_features=False,
        )
        config2 = PipelineConfig(
            use_goals_features=False,
            use_handicap_features=True,
            use_form_features=False,
        )

        pipeline1 = FeaturePipeline(config1)
        pipeline2 = FeaturePipeline(config2)

        fv1 = pipeline1.extract(sample_match)
        fv2 = pipeline2.extract(sample_match)

        merged = fv1.merge(fv2)

        # Merged should have features from both
        assert any(k.startswith("goals_") for k in merged.features)
        assert any(k.startswith("handicap_") for k in merged.features)

    def test_merge_different_matches_fails(self, sample_matches):
        pipeline = FeaturePipeline()
        fv1 = pipeline.extract(sample_matches[0])
        fv2 = pipeline.extract(sample_matches[1])

        with pytest.raises(ValueError):
            fv1.merge(fv2)


class TestFactoryFunctions:
    """Tests for pipeline factory functions."""

    def test_create_goals_pipeline(self):
        pipeline = create_goals_pipeline()
        names = pipeline.get_extractor_names()

        assert "goals" in names
        assert "form" in names
        assert "handicap" not in names

    def test_create_handicap_pipeline(self):
        pipeline = create_handicap_pipeline()
        names = pipeline.get_extractor_names()

        assert "handicap" in names
        assert "form" in names
        assert "goals" not in names

    def test_create_full_pipeline(self):
        pipeline = create_full_pipeline()
        names = pipeline.get_extractor_names()

        assert "goals" in names
        assert "handicap" in names
        assert "form" in names

    def test_get_pipeline_singleton(self):
        pipeline1 = get_pipeline()
        pipeline2 = get_pipeline()

        assert pipeline1 is pipeline2


class TestPipelineStats:
    """Tests for pipeline statistics."""

    def test_get_stats(self):
        pipeline = FeaturePipeline()
        stats = pipeline.get_stats()

        assert "extractors_count" in stats
        assert "extractor_names" in stats
        assert "total_features" in stats
        assert "normalization_fitted" in stats

    def test_total_features_count(self, sample_match):
        pipeline = FeaturePipeline()
        stats = pipeline.get_stats()

        result = pipeline.extract(sample_match)
        assert stats["total_features"] == len(result.features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
