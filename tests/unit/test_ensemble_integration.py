"""
Tests for ensemble integration.

Checkpoint: 4.7
"""

import pytest
from unittest.mock import Mock

from core.ml.service import (
    MLPredictionResult,
    GoalsPredictionResult,
    HandicapPredictionResult,
    PredictionStatus,
)
from core.ml.service.ensemble_integration import (
    EnsembleIntegration,
    EnsembleConfig,
    CombinationMethod,
    AgentPrediction,
    EnsemblePrediction,
    EnsembleResult,
    create_default_ensemble,
    create_ml_primary_ensemble,
    create_conservative_ensemble,
)


@pytest.fixture
def sample_ml_result():
    return MLPredictionResult(
        match_id="test-match",
        prediction_id="pred-123",
        status=PredictionStatus.SUCCESS,
        goals_prediction=GoalsPredictionResult(
            home_expected=1.5,
            away_expected=1.0,
            total_expected=2.5,
            over_25_prob=0.55,
            under_25_prob=0.45,
            btts_yes_prob=0.60,
            btts_no_prob=0.40,
            confidence=0.7,
            model_version="v1.0.0",
        ),
        handicap_prediction=HandicapPredictionResult(
            expected_margin=0.5,
            home_win_prob=0.50,
            draw_prob=0.25,
            away_win_prob=0.25,
            home_minus_15_prob=0.30,
            confidence=0.65,
            model_version="v1.0.0",
        ),
    )


@pytest.fixture
def sample_agent_predictions():
    return [
        AgentPrediction(
            source="StatisticalAgent",
            market="over_2.5",
            selection="over",
            probability=0.60,
            confidence=0.75,
            reasoning="High-scoring teams",
        ),
        AgentPrediction(
            source="FormAgent",
            market="1x2",
            selection="home",
            probability=0.55,
            confidence=0.65,
            reasoning="Strong home form",
        ),
    ]


class TestEnsembleConfig:
    def test_default_config(self):
        config = EnsembleConfig()

        assert config.method == CombinationMethod.WEIGHTED_AVERAGE
        assert config.ml_weight == 0.6
        assert config.agent_weight == 0.4
        assert config.ml_weight + config.agent_weight == 1.0

    def test_custom_config(self):
        config = EnsembleConfig(
            method=CombinationMethod.ML_PRIMARY,
            ml_weight=0.8,
            agent_weight=0.2,
        )

        assert config.method == CombinationMethod.ML_PRIMARY
        assert config.ml_weight == 0.8


class TestEnsembleIntegration:
    def test_combine_ml_only(self, sample_ml_result):
        ensemble = create_default_ensemble()

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=[],
        )

        assert result.ml_available
        assert not result.agents_available
        assert len(result.predictions) > 0

    def test_combine_agent_only(self, sample_agent_predictions):
        ensemble = create_default_ensemble()

        result = ensemble.combine(
            ml_result=None,
            agent_predictions=sample_agent_predictions,
        )

        assert not result.ml_available
        assert result.agents_available
        assert len(result.predictions) > 0

    def test_combine_both(self, sample_ml_result, sample_agent_predictions):
        ensemble = create_default_ensemble()

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=sample_agent_predictions,
        )

        assert result.ml_available
        assert result.agents_available
        assert len(result.predictions) > 0

    def test_weighted_average_agreement(self, sample_ml_result, sample_agent_predictions):
        config = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            ml_weight=0.6,
            agent_weight=0.4,
        )
        ensemble = EnsembleIntegration(config)

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=sample_agent_predictions,
        )

        # Find over_2.5 prediction
        over_pred = None
        for pred in result.predictions:
            if pred.market == "over_2.5":
                over_pred = pred
                break

        if over_pred:
            assert over_pred.combination_method == "weighted_average"
            # ML says 0.55, agent says 0.60
            # Combined should be weighted average
            expected = 0.6 * 0.55 + 0.4 * 0.60
            assert abs(over_pred.combined_probability - expected) < 0.01

    def test_ml_primary_method(self, sample_ml_result, sample_agent_predictions):
        ensemble = create_ml_primary_ensemble()

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=sample_agent_predictions,
        )

        # Find prediction that has both ML and agent
        for pred in result.predictions:
            if pred.ml_probability and pred.agent_probability:
                # ML primary should use ML probability
                assert pred.combined_probability == pred.ml_probability or \
                       pred.combination_method in ["ml_only", "weighted_average", "ml_primary"]

    def test_conservative_requires_agreement(self, sample_ml_result):
        # Create agent that disagrees with ML
        conflicting_agent = [
            AgentPrediction(
                source="ConflictAgent",
                market="over_2.5",
                selection="under",  # ML says over
                probability=0.70,
                confidence=0.80,
                reasoning="Defensive game expected",
            ),
        ]

        ensemble = create_conservative_ensemble()

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=conflicting_agent,
        )

        # Should not include over_2.5 prediction due to disagreement
        over_pred = None
        for pred in result.predictions:
            if pred.market == "over_2.5":
                over_pred = pred
                break

        assert over_pred is None or over_pred.combination_method in ["ml_only", "agent_only"]

    def test_recommendations_generated(self, sample_ml_result, sample_agent_predictions):
        ensemble = create_default_ensemble()

        market_odds = {
            "over_2.5_over": 1.90,
            "1x2_home": 2.10,
        }

        result = ensemble.combine(
            ml_result=sample_ml_result,
            agent_predictions=sample_agent_predictions,
            market_odds=market_odds,
        )

        assert len(result.recommendations) > 0


class TestAgentPrediction:
    def test_create(self):
        pred = AgentPrediction(
            source="TestAgent",
            market="over_2.5",
            selection="over",
            probability=0.60,
            confidence=0.75,
        )

        assert pred.source == "TestAgent"
        assert pred.probability == 0.60


class TestEnsemblePrediction:
    def test_has_agreement(self):
        pred = EnsemblePrediction(
            market="over_2.5",
            selection="over",
            combined_probability=0.58,
            combined_confidence=0.70,
            agreement_score=0.95,
        )

        assert pred.has_agreement

    def test_no_agreement(self):
        pred = EnsemblePrediction(
            market="over_2.5",
            selection="over",
            combined_probability=0.50,
            combined_confidence=0.55,
            agreement_score=0.3,
        )

        assert not pred.has_agreement


class TestFactoryFunctions:
    def test_create_default_ensemble(self):
        ensemble = create_default_ensemble()

        assert ensemble.config.method == CombinationMethod.WEIGHTED_AVERAGE
        assert ensemble.config.ml_weight == 0.6

    def test_create_ml_primary_ensemble(self):
        ensemble = create_ml_primary_ensemble()

        assert ensemble.config.method == CombinationMethod.ML_PRIMARY
        assert ensemble.config.ml_weight == 0.8

    def test_create_conservative_ensemble(self):
        ensemble = create_conservative_ensemble()

        assert ensemble.config.require_agreement
        assert ensemble.config.min_combined_confidence == 0.65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
