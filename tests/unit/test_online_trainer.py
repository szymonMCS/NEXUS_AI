"""
Tests for online trainer.

Checkpoint: 3.12
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from core.ml.registry import ModelRegistry
from core.ml.training import (
    OnlineTrainer,
    TrainingExample,
    TrainingBatch,
    TrainingConfig,
    TrainingResult,
    DegradationAlert,
)
from core.ml.models import GoalsModel


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_storage):
    return ModelRegistry(storage_path=temp_storage)


@pytest.fixture
def trainer(registry):
    config = TrainingConfig(min_samples=10)
    return OnlineTrainer(registry, config)


@pytest.fixture
def sample_examples():
    examples = []
    for i in range(20):
        ex = TrainingExample(
            example_id=f"ex-{i}",
            match_id=f"match-{i}",
            features={
                "goals_home_attack_strength": 1.0 + i * 0.05,
                "goals_away_defense_strength": 1.0 + i * 0.03,
                "handicap_form_diff": 0.1 * (i % 5),
            },
            actual_home_goals=i % 4,
            actual_away_goals=i % 3,
        )
        examples.append(ex)
    return examples


class TestTrainingExample:
    def test_create(self):
        ex = TrainingExample(
            example_id="ex-1",
            match_id="m-1",
            features={"f1": 1.0},
            actual_home_goals=2,
            actual_away_goals=1,
        )
        assert ex.actual_total_goals == 3
        assert ex.actual_margin == 1
        assert ex.actual_winner == "home"
        assert ex.is_over_25 is True

    def test_incomplete(self):
        ex = TrainingExample(example_id="ex-1", match_id="m-1")
        assert not ex.is_complete
        assert ex.actual_total_goals is None


class TestTrainingBatch:
    def test_create_batch(self, sample_examples):
        batch = TrainingBatch(examples=sample_examples)
        assert len(batch) == 20

    def test_get_complete(self, sample_examples):
        batch = TrainingBatch(examples=sample_examples)
        complete = batch.get_complete_examples()
        assert len(complete) == 20

    def test_get_targets(self, sample_examples):
        batch = TrainingBatch(examples=sample_examples)
        goals = batch.get_goals_targets()
        assert len(goals) == 20
        assert all(isinstance(t, tuple) for t in goals)


class TestOnlineTrainer:
    def test_add_example(self, trainer, sample_examples):
        for ex in sample_examples[:5]:
            trainer.add_example("poisson_goals", ex)

        status = trainer.get_buffer_status()
        assert status.get("poisson_goals", 0) == 5

    def test_train_incremental_insufficient(self, trainer, sample_examples):
        for ex in sample_examples[:5]:
            trainer.add_example("poisson_goals", ex)

        model = GoalsModel()
        result = trainer.train_incremental(model)

        assert not result.success
        assert "Not enough" in result.error_message

    def test_should_retrain(self, trainer, sample_examples):
        for ex in sample_examples:
            trainer.add_example("poisson_goals", ex)

        assert trainer.should_retrain("poisson_goals", min_examples=10)
        assert not trainer.should_retrain("poisson_goals", min_examples=100)


class TestDegradation:
    def test_check_degradation(self, trainer):
        trainer._baseline_metrics["test_model"] = {"accuracy": 0.80}

        alerts = trainer.check_degradation(
            "test_model",
            {"accuracy": 0.70},
            threshold_pct=5.0,
        )

        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_no_degradation(self, trainer):
        trainer._baseline_metrics["test_model"] = {"accuracy": 0.80}

        alerts = trainer.check_degradation(
            "test_model",
            {"accuracy": 0.79},
            threshold_pct=10.0,
        )

        assert len(alerts) == 0

    def test_callback(self, trainer):
        alerts_received = []
        trainer.set_degradation_callback(lambda a: alerts_received.append(a))
        trainer._baseline_metrics["test"] = {"acc": 0.80}

        trainer.check_degradation("test", {"acc": 0.50}, threshold_pct=10)

        assert len(alerts_received) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
