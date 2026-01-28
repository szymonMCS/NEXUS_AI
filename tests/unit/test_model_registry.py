"""
Tests for model registry.

Checkpoint: 3.11
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from core.ml.registry import ModelRegistry, ModelVersion, VersionComparison
from core.ml.models import GoalsModel


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_storage):
    return ModelRegistry(storage_path=temp_storage)


@pytest.fixture
def trained_model():
    from core.data.enums import Sport
    from core.data.schemas import MatchData, TeamData, TeamMatchStats, DataQuality
    from core.ml.features import FeaturePipeline

    model = GoalsModel()
    pipeline = FeaturePipeline()

    features = []
    targets = []

    for i in range(30):
        match = MatchData(
            match_id=f"train-{i}",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id=f"h{i}", name=f"Home {i}"),
            away_team=TeamData(team_id=f"a{i}", name=f"Away {i}"),
            league="Test",
            start_time=datetime.utcnow(),
            home_stats=TeamMatchStats(goals_scored_avg=1.5, goals_conceded_avg=1.0),
            away_stats=TeamMatchStats(goals_scored_avg=1.2, goals_conceded_avg=1.3),
            data_quality=DataQuality(completeness=0.7, freshness_hours=1, sources_count=1),
        )
        features.append(pipeline.extract(match))
        targets.append((i % 4, i % 3))

    model.train(features, targets)
    return model


class TestModelVersion:
    def test_create(self):
        v = ModelVersion(name="test_model", version="v1.0.0")
        assert v.full_name == "test_model:v1.0.0"

    def test_age(self):
        v = ModelVersion(name="test", version="v1.0.0")
        assert v.age_hours < 1
        assert v.is_recent

    def test_to_from_dict(self):
        v = ModelVersion(
            name="test",
            version="v1.0.0",
            metrics={"accuracy": 0.75},
            tags=["test"],
        )
        d = v.to_dict()
        v2 = ModelVersion.from_dict(d)
        assert v2.name == v.name
        assert v2.metrics == v.metrics


class TestVersionComparison:
    def test_comparison(self):
        old = ModelVersion(name="m", version="v1", metrics={"accuracy": 0.70})
        new = ModelVersion(name="m", version="v2", metrics={"accuracy": 0.75})

        comp = VersionComparison(old, new, comparison_metric="accuracy")
        assert comp.is_improvement
        assert comp.metric_changes["accuracy"] == 0.05


class TestModelRegistry:
    def test_register(self, registry, trained_model):
        version = registry.register(
            trained_model,
            metrics={"mae": 0.5},
            description="Test version",
        )

        assert version.name == "poisson_goals"
        assert "v1" in version.version
        assert version.is_active

    def test_list_models(self, registry, trained_model):
        registry.register(trained_model, metrics={})
        models = registry.list_models()
        assert "poisson_goals" in models

    def test_list_versions(self, registry, trained_model):
        registry.register(trained_model, metrics={})
        registry.register(trained_model, metrics={})

        versions = registry.list_versions("poisson_goals")
        assert len(versions) == 2

    def test_get_active_version(self, registry, trained_model):
        registry.register(trained_model, metrics={})
        active = registry.get_active_version("poisson_goals")
        assert active is not None
        assert active.is_active

    def test_activate_version(self, registry, trained_model):
        v1 = registry.register(trained_model, metrics={})
        v2 = registry.register(trained_model, metrics={})

        assert registry.activate("poisson_goals", v1.version)

        active = registry.get_active_version("poisson_goals")
        assert active.version == v1.version

    def test_rollback(self, registry, trained_model):
        v1 = registry.register(trained_model, metrics={})
        v2 = registry.register(trained_model, metrics={})

        rolled = registry.rollback("poisson_goals", steps=1)
        assert rolled.version == v1.version

    def test_deprecate(self, registry, trained_model):
        v = registry.register(trained_model, metrics={})
        assert registry.deprecate("poisson_goals", v.version)

        versions = registry.list_versions("poisson_goals", include_deprecated=False)
        assert len(versions) == 0

    def test_load_model(self, registry, trained_model):
        registry.register(trained_model, metrics={})

        loaded = registry.load_model("poisson_goals", GoalsModel)
        assert loaded is not None
        assert loaded.is_trained

    def test_get_best_version(self, registry, trained_model):
        registry.register(trained_model, metrics={"accuracy": 0.70})
        registry.register(trained_model, metrics={"accuracy": 0.80})
        registry.register(trained_model, metrics={"accuracy": 0.75})

        best = registry.get_best_version("poisson_goals", "accuracy")
        assert best.metrics["accuracy"] == 0.80

    def test_persistence(self, temp_storage, trained_model):
        registry1 = ModelRegistry(storage_path=temp_storage)
        registry1.register(trained_model, metrics={"acc": 0.8})

        registry2 = ModelRegistry(storage_path=temp_storage)
        versions = registry2.list_versions("poisson_goals")
        assert len(versions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
