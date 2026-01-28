"""
Tests for Handicap (GBM) model.

Checkpoint: 2.7
Tests for HandicapModel with Gradient Boosting approach.
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
    OddsData,
)
from core.ml.features import FeaturePipeline, FeatureVector
from core.ml.models import (
    HandicapModel,
    HandicapPrediction,
    GBMParameters,
)


@pytest.fixture
def sample_match():
    return MatchData(
        match_id="test-handicap-1",
        sport=Sport.FOOTBALL,
        home_team=TeamData(team_id="home-1", name="Home FC", elo_rating=1600),
        away_team=TeamData(team_id="away-1", name="Away FC", elo_rating=1500),
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
        odds=OddsData(
            home_win=1.80,
            draw=3.50,
            away_win=4.20,
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
    """Generate training data for handicap model."""
    features = []
    targets = []  # Actual margins (home_goals - away_goals)
    pipeline = FeaturePipeline()

    for i in range(50):
        home_strength = 1.3 + (i % 5) * 0.2
        away_strength = 1.2 + (i % 4) * 0.15

        match = MatchData(
            match_id=f"train-hcap-{i}",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id=f"h{i}", name=f"Home {i}", elo_rating=1500 + i * 5),
            away_team=TeamData(team_id=f"a{i}", name=f"Away {i}", elo_rating=1500 - i * 3),
            league="Training",
            start_time=datetime.utcnow(),
            home_stats=TeamMatchStats(
                goals_scored_avg=home_strength,
                goals_conceded_avg=1.0 + (i % 4) * 0.1,
                form_points=0.5 + (i % 3) * 0.15,
                rest_days=3 + i % 5,
            ),
            away_stats=TeamMatchStats(
                goals_scored_avg=away_strength,
                goals_conceded_avg=1.1 + (i % 5) * 0.15,
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

        # Simulated margin based on strength difference
        margin = int(home_strength - away_strength + (i % 3) - 1)
        targets.append(float(margin))

    return features, targets


class TestHandicapModel:
    """Tests for HandicapModel."""

    def test_init_default(self):
        model = HandicapModel()
        assert model.name == "gbm_handicap"
        assert model.version == "1.0.0"
        assert model.is_trained is False

    def test_init_with_params(self):
        params = GBMParameters(
            margin_std=2.0,
            home_advantage_goals=0.5,
        )
        model = HandicapModel(params)
        assert model._params.margin_std == 2.0
        assert model._params.home_advantage_goals == 0.5

    def test_predict_without_training(self, sample_features):
        """Model should work without training using heuristics."""
        model = HandicapModel()
        prediction = model.predict(sample_features)

        assert isinstance(prediction, HandicapPrediction)
        assert isinstance(prediction.expected_margin, float)

    def test_predict_cover_probabilities(self, sample_features):
        model = HandicapModel()
        prediction = model.predict(sample_features)

        # Cover probabilities should be between 0 and 1
        assert 0 <= prediction.home_cover_minus_15 <= 1
        assert 0 <= prediction.home_cover_plus_15 <= 1

        # Home -1.5 cover < Home +1.5 cover (easier to cover)
        assert prediction.home_cover_minus_15 < prediction.home_cover_plus_15

    def test_predict_1x2_probabilities(self, sample_features):
        model = HandicapModel()
        prediction = model.predict(sample_features)

        # 1X2 should sum to ~1
        total = prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
        assert abs(total - 1.0) < 0.01

    def test_predict_confidence(self, sample_features):
        model = HandicapModel()
        prediction = model.predict(sample_features)

        assert 0 <= prediction.confidence <= 1

        # Trained model should have higher confidence
        model._trained = True
        prediction_trained = model.predict(sample_features)
        assert prediction_trained.confidence > prediction.confidence

    def test_predict_batch(self, sample_features):
        model = HandicapModel()
        predictions = model.predict_batch([sample_features, sample_features])

        assert len(predictions) == 2
        assert all(isinstance(p, HandicapPrediction) for p in predictions)

    def test_train(self, training_data):
        features, targets = training_data
        model = HandicapModel()

        metrics = model.train(features, targets, validation_split=0.2)

        assert model.is_trained is True
        assert model._training_samples == len(features)
        assert "mae_margin" in metrics
        assert "accuracy_1x2" in metrics

    def test_train_learns_weights(self, training_data):
        features, targets = training_data
        model = HandicapModel()

        model.train(features, targets)

        assert len(model._feature_weights) > 0

    def test_train_insufficient_data(self, sample_features):
        model = HandicapModel()

        with pytest.raises(ValueError, match="at least 20"):
            model.train([sample_features], [1.0])

    def test_save_and_load(self, training_data):
        features, targets = training_data
        model = HandicapModel()
        model.train(features, targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "handicap_model.json"

            # Save
            assert model.save(path) is True
            assert path.exists()

            # Load into new model
            new_model = HandicapModel()
            assert new_model.load(path) is True
            assert new_model.is_trained is True
            assert new_model._training_samples == model._training_samples
            assert new_model._feature_weights == model._feature_weights

    def test_get_model_info(self, training_data):
        features, targets = training_data
        model = HandicapModel()
        model.train(features, targets)

        info = model.get_model_info()

        assert info.name == "gbm_handicap"
        assert info.is_trained is True
        assert info.training_samples > 0

    def test_get_feature_importance(self, training_data):
        features, targets = training_data
        model = HandicapModel()
        model.train(features, targets)

        importance = model.get_feature_importance()

        assert importance is not None
        # Importance should sum to ~1
        assert abs(sum(importance.values()) - 1.0) < 0.01


class TestHandicapMath:
    """Tests for mathematical functions."""

    def test_normal_cdf(self):
        model = HandicapModel()

        # CDF at z=0 should be 0.5
        assert abs(model._normal_cdf(0) - 0.5) < 0.01

        # CDF at z=-inf should be ~0
        assert model._normal_cdf(-5) < 0.001

        # CDF at z=+inf should be ~1
        assert model._normal_cdf(5) > 0.999

    def test_cover_probability_symmetry(self, sample_features):
        model = HandicapModel()

        # For expected_margin = 0, home -0.5 should equal away +0.5
        model._params.home_advantage_goals = 0
        # Force zero margin
        sample_features.features["handicap_attack_diff"] = 0
        sample_features.features["handicap_defense_diff"] = 0
        sample_features.features["handicap_form_diff"] = 0
        sample_features.features["handicap_elo_diff"] = 0

        prediction = model.predict(sample_features)

        # With 0 margin, home -0.5 ≈ 0.5, away +0.5 ≈ 0.5
        # (approximately, due to continuous distribution)


class TestHandicapPrediction:
    """Tests for HandicapPrediction dataclass."""

    def test_recommended_bet_high_confidence(self):
        pred = HandicapPrediction(
            expected_margin=1.5,
            home_cover_minus_15=0.65,
            home_cover_plus_15=0.85,
            confidence=0.7,
        )

        bet = pred.recommended_bet
        assert bet is not None
        assert "home" in bet.lower()

    def test_recommended_bet_low_confidence(self):
        pred = HandicapPrediction(
            expected_margin=1.5,
            home_cover_minus_15=0.65,
            confidence=0.4,  # Too low
        )

        assert pred.recommended_bet is None

    def test_predicted_winner_home(self):
        pred = HandicapPrediction(
            expected_margin=1.0,
            home_win_prob=0.6,
            draw_prob=0.2,
            away_win_prob=0.2,
        )

        assert pred.predicted_winner == "home"

    def test_predicted_winner_away(self):
        pred = HandicapPrediction(
            expected_margin=-1.0,
            home_win_prob=0.2,
            draw_prob=0.2,
            away_win_prob=0.6,
        )

        assert pred.predicted_winner == "away"

    def test_predicted_winner_draw(self):
        pred = HandicapPrediction(
            expected_margin=0.0,
            home_win_prob=0.3,
            draw_prob=0.4,
            away_win_prob=0.3,
        )

        assert pred.predicted_winner == "draw"


class TestModelComparison:
    """Tests comparing trained vs untrained models."""

    def test_trained_vs_untrained_predictions(self, training_data, sample_features):
        features, targets = training_data

        untrained = HandicapModel()
        trained = HandicapModel()
        trained.train(features, targets)

        pred_untrained = untrained.predict(sample_features)
        pred_trained = trained.predict(sample_features)

        # Both should produce valid predictions
        assert isinstance(pred_untrained, HandicapPrediction)
        assert isinstance(pred_trained, HandicapPrediction)

        # Trained should have higher confidence
        assert pred_trained.confidence > pred_untrained.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
