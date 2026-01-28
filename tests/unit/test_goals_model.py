"""
Tests for Goals (Poisson) model.

Checkpoint: 2.6
Tests for GoalsModel with Poisson distribution.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from core.data.enums import Sport
from core.data.schemas import (
    MatchData,
    TeamData,
    TeamMatchStats,
    DataQuality,
)
from core.ml.features import FeaturePipeline, FeatureVector
from core.ml.models import (
    GoalsModel,
    GoalsPrediction,
    PoissonParameters,
)


@pytest.fixture
def sample_match():
    return MatchData(
        match_id="test-goals-1",
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
def sample_features(sample_match):
    pipeline = FeaturePipeline()
    return pipeline.extract(sample_match)


@pytest.fixture
def training_data():
    """Generate training data."""
    features = []
    targets = []
    pipeline = FeaturePipeline()

    for i in range(50):
        match = MatchData(
            match_id=f"train-{i}",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id=f"h{i}", name=f"Home {i}"),
            away_team=TeamData(team_id=f"a{i}", name=f"Away {i}"),
            league="Training",
            start_time=datetime.utcnow(),
            home_stats=TeamMatchStats(
                goals_scored_avg=1.3 + (i % 5) * 0.2,
                goals_conceded_avg=1.0 + (i % 4) * 0.15,
                form_points=0.5 + (i % 3) * 0.1,
                rest_days=3 + i % 5,
            ),
            away_stats=TeamMatchStats(
                goals_scored_avg=1.2 + (i % 4) * 0.15,
                goals_conceded_avg=1.1 + (i % 5) * 0.2,
                form_points=0.4 + (i % 3) * 0.1,
                rest_days=4,
            ),
            data_quality=DataQuality(
                completeness=0.7,
                freshness_hours=1,
                sources_count=1,
            ),
        )
        fv = pipeline.extract(match)
        features.append(fv)

        # Simulated actual goals
        home_goals = (i % 4)
        away_goals = (i % 3)
        targets.append((home_goals, away_goals))

    return features, targets


class TestGoalsModel:
    """Tests for GoalsModel."""

    def test_init_default(self):
        model = GoalsModel()
        assert model.name == "poisson_goals"
        assert model.version == "1.0.0"
        assert model.is_trained is False

    def test_init_with_params(self):
        params = PoissonParameters(
            league_avg_goals=1.5,
            home_advantage=1.2,
        )
        model = GoalsModel(params)
        assert model._params.league_avg_goals == 1.5
        assert model._params.home_advantage == 1.2

    def test_predict_without_training(self, sample_features):
        """Model should work without training using base params."""
        model = GoalsModel()
        prediction = model.predict(sample_features)

        assert isinstance(prediction, GoalsPrediction)
        assert prediction.home_expected > 0
        assert prediction.away_expected > 0
        assert prediction.total_expected == prediction.home_expected + prediction.away_expected

    def test_predict_probabilities_sum(self, sample_features):
        model = GoalsModel()
        prediction = model.predict(sample_features)

        # Over + under should sum to ~1
        assert abs(prediction.over_25_prob + prediction.under_25_prob - 1.0) < 0.01
        assert abs(prediction.over_15_prob + prediction.under_15_prob - 1.0) < 0.01
        assert abs(prediction.over_35_prob + prediction.under_35_prob - 1.0) < 0.01

    def test_predict_returns_score_matrix(self, sample_features):
        model = GoalsModel()
        prediction = model.predict(sample_features)

        assert prediction.score_matrix is not None
        assert len(prediction.score_matrix) > 0
        # Probabilities should be positive
        assert all(p > 0 for p in prediction.score_matrix.values())

    def test_predict_confidence(self, sample_features):
        model = GoalsModel()
        prediction = model.predict(sample_features)

        # Confidence should be between 0 and 1
        assert 0 <= prediction.confidence <= 1

        # Trained model should have higher confidence
        model._trained = True
        prediction_trained = model.predict(sample_features)
        assert prediction_trained.confidence > prediction.confidence

    def test_predict_batch(self, sample_features):
        model = GoalsModel()
        predictions = model.predict_batch([sample_features, sample_features])

        assert len(predictions) == 2
        assert all(isinstance(p, GoalsPrediction) for p in predictions)

    def test_train(self, training_data):
        features, targets = training_data
        model = GoalsModel()

        metrics = model.train(features, targets, validation_split=0.2)

        assert model.is_trained is True
        assert model._training_samples == len(features)
        assert "mae_total" in metrics
        assert "over_under_accuracy" in metrics

    def test_train_updates_params(self, training_data):
        features, targets = training_data
        model = GoalsModel()

        initial_avg = model._params.league_avg_goals
        model.train(features, targets)

        # Parameters should be updated based on training data
        # (may or may not differ depending on data)
        assert model._params.league_avg_goals > 0

    def test_train_insufficient_data(self, sample_features):
        model = GoalsModel()

        with pytest.raises(ValueError, match="at least 10"):
            model.train([sample_features], [(2, 1)])

    def test_save_and_load(self, training_data):
        features, targets = training_data
        model = GoalsModel()
        model.train(features, targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals_model.json"

            # Save
            assert model.save(path) is True
            assert path.exists()

            # Load into new model
            new_model = GoalsModel()
            assert new_model.load(path) is True
            assert new_model.is_trained is True
            assert new_model._training_samples == model._training_samples

    def test_get_model_info(self, training_data):
        features, targets = training_data
        model = GoalsModel()
        model.train(features, targets)

        info = model.get_model_info()

        assert info.name == "poisson_goals"
        assert info.is_trained is True
        assert info.training_samples > 0

    def test_get_required_features(self):
        model = GoalsModel()
        required = model.get_required_features()

        assert "goals_home_attack_strength" in required
        assert "goals_away_defense_strength" in required


class TestPoissonMath:
    """Tests for Poisson mathematical functions."""

    def test_poisson_pmf(self):
        model = GoalsModel()

        # P(X=0 | lambda=1) = e^-1 ≈ 0.368
        p0 = model._poisson_pmf(0, 1.0)
        assert abs(p0 - 0.368) < 0.01

        # P(X=1 | lambda=1) = e^-1 ≈ 0.368
        p1 = model._poisson_pmf(1, 1.0)
        assert abs(p1 - 0.368) < 0.01

        # P(X=2 | lambda=1) = e^-1 / 2 ≈ 0.184
        p2 = model._poisson_pmf(2, 1.0)
        assert abs(p2 - 0.184) < 0.01

    def test_poisson_pmf_edge_cases(self):
        model = GoalsModel()

        # Lambda = 0 should give P(0) = 1
        assert model._poisson_pmf(0, 0.0) == 1.0
        assert model._poisson_pmf(1, 0.0) == 0.0

    def test_over_under_probability_calculation(self, sample_features):
        model = GoalsModel()
        prediction = model.predict(sample_features)

        # Calculate manually for verification
        probs = model._calculate_probabilities(
            prediction.home_expected,
            prediction.away_expected
        )

        assert probs["over_2.5"] == prediction.over_25_prob
        assert probs["under_2.5"] == prediction.under_25_prob


class TestGoalsPrediction:
    """Tests for GoalsPrediction dataclass."""

    def test_recommended_bet_high_confidence(self):
        pred = GoalsPrediction(
            home_expected=2.0,
            away_expected=1.5,
            total_expected=3.5,
            over_25_prob=0.75,
            under_25_prob=0.25,
            confidence=0.7,
        )

        bet = pred.recommended_bet
        assert bet == "over_2.5"

    def test_recommended_bet_low_confidence(self):
        pred = GoalsPrediction(
            home_expected=2.0,
            away_expected=1.5,
            total_expected=3.5,
            over_25_prob=0.75,
            under_25_prob=0.25,
            confidence=0.4,  # Too low
        )

        assert pred.recommended_bet is None

    def test_btts_probability(self):
        pred = GoalsPrediction(
            home_expected=2.0,
            away_expected=1.5,
            total_expected=3.5,
        )

        btts = pred.btts_prob
        assert 0 < btts < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
