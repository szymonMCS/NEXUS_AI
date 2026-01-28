"""
Tests for ML prediction service.

Checkpoint: 4.6
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from core.data.schemas import MatchData, DataQuality, Sport
from core.ml.features import FeatureVector
from core.ml.models import GoalsPrediction, HandicapPrediction
from core.ml.service import (
    MLPredictionService,
    MLPredictionResult,
    GoalsPredictionResult,
    HandicapPredictionResult,
    BettingRecommendation,
    PredictionStatus,
)


@pytest.fixture
def mock_repository():
    repo = Mock()
    repo.enrich_match_data.return_value = MatchData(
        match_id="test-match",
        sport=Sport.FOOTBALL,
        league="Test League",
        home_team="Home FC",
        away_team="Away FC",
        quality=DataQuality(completeness=0.8, freshness=1.0, sources_count=3),
    )
    return repo


@pytest.fixture
def mock_goals_model():
    model = Mock()
    model.version = "v1.0.0"
    model.predict.return_value = GoalsPrediction(
        home_expected=1.5,
        away_expected=1.0,
        total_expected=2.5,
        over_25_prob=0.45,
        under_25_prob=0.55,
        confidence=0.7,
        score_matrix={"1-0": 0.15, "1-1": 0.20, "2-1": 0.12},
    )
    return model


@pytest.fixture
def mock_handicap_model():
    model = Mock()
    model.version = "v1.0.0"
    model.predict.return_value = HandicapPrediction(
        expected_margin=0.5,
        home_win_prob=0.45,
        draw_prob=0.25,
        away_win_prob=0.30,
        home_cover_minus_15=0.25,
        confidence=0.65,
    )
    return model


@pytest.fixture
def sample_match():
    return MatchData(
        match_id="test-match",
        sport=Sport.FOOTBALL,
        league="Test League",
        home_team="Home FC",
        away_team="Away FC",
        quality=DataQuality(completeness=0.7, freshness=1.0, sources_count=2),
    )


class TestMLPredictionService:
    def test_predict_success(self, mock_repository, mock_goals_model, mock_handicap_model, sample_match):
        service = MLPredictionService(repository=mock_repository)
        service._goals_model = mock_goals_model
        service._handicap_model = mock_handicap_model

        result = service.predict(sample_match)

        assert result.is_success
        assert result.match_id == "test-match"
        assert result.goals_prediction is not None
        assert result.handicap_prediction is not None

    def test_predict_insufficient_data(self, mock_repository, sample_match):
        # Set low completeness
        sample_match.quality = DataQuality(completeness=0.3, freshness=1.0, sources_count=1)
        mock_repository.enrich_match_data.return_value = sample_match

        service = MLPredictionService(repository=mock_repository)

        result = service.predict(sample_match)

        assert not result.is_success
        assert result.status == PredictionStatus.INSUFFICIENT_DATA

    def test_predict_with_recommendations(self, mock_repository, mock_goals_model, mock_handicap_model, sample_match):
        service = MLPredictionService(repository=mock_repository)
        service._goals_model = mock_goals_model
        service._handicap_model = mock_handicap_model

        result = service.predict(sample_match, include_recommendations=True)

        assert result.is_success
        # Should have some recommendations based on probabilities
        assert isinstance(result.recommendations, list)

    def test_predict_with_market_odds(self, mock_repository, mock_goals_model, mock_handicap_model, sample_match):
        service = MLPredictionService(repository=mock_repository)
        service._goals_model = mock_goals_model
        service._handicap_model = mock_handicap_model

        market_odds = {
            "under_2.5": 1.80,
            "over_2.5": 2.00,
        }

        result = service.predict(sample_match, market_odds=market_odds)

        assert result.is_success
        # Check if edge is calculated for recommendations
        for rec in result.recommendations:
            if rec.market in ["over_2.5", "under_2.5"]:
                assert rec.edge is not None

    def test_predict_batch(self, mock_repository, mock_goals_model, mock_handicap_model, sample_match):
        service = MLPredictionService(repository=mock_repository)
        service._goals_model = mock_goals_model
        service._handicap_model = mock_handicap_model

        matches = [
            MatchData(
                match_id=f"match-{i}",
                sport=Sport.FOOTBALL,
                league="Test League",
                home_team=f"Home {i}",
                away_team=f"Away {i}",
                quality=DataQuality(completeness=0.8, freshness=1.0, sources_count=2),
            )
            for i in range(3)
        ]

        result = service.predict_batch(matches)

        assert len(result.predictions) == 3
        assert result.success_count >= 0
        assert result.total_processing_time_ms > 0


class TestGoalsPredictionResult:
    def test_get_over_under_prob(self):
        result = GoalsPredictionResult(
            home_expected=1.5,
            away_expected=1.0,
            total_expected=2.5,
            over_15_prob=0.75,
            under_15_prob=0.25,
            over_25_prob=0.45,
            under_25_prob=0.55,
            over_35_prob=0.20,
            under_35_prob=0.80,
        )

        over_25, under_25 = result.get_over_under_prob(2.5)
        assert over_25 == 0.45
        assert under_25 == 0.55

    def test_most_likely_score(self):
        result = GoalsPredictionResult(
            home_expected=1.5,
            away_expected=1.0,
            total_expected=2.5,
            score_probabilities={
                "1-1": 0.20,
                "1-0": 0.15,
                "2-1": 0.12,
                "0-0": 0.10,
            },
        )

        top_scores = result.most_likely_score(top_n=2)

        assert len(top_scores) == 2
        assert top_scores[0][0] == "1-1"
        assert top_scores[0][1] == 0.20


class TestHandicapPredictionResult:
    def test_predicted_winner(self):
        result = HandicapPredictionResult(
            expected_margin=0.5,
            home_win_prob=0.50,
            draw_prob=0.25,
            away_win_prob=0.25,
        )

        assert result.predicted_winner == "home"

    def test_predicted_winner_draw(self):
        result = HandicapPredictionResult(
            expected_margin=0.0,
            home_win_prob=0.30,
            draw_prob=0.40,
            away_win_prob=0.30,
        )

        assert result.predicted_winner == "draw"


class TestBettingRecommendation:
    def test_has_value(self):
        rec = BettingRecommendation(
            market="over_2.5",
            selection="over",
            probability=0.55,
            odds_required=1.82,
            confidence=0.7,
            edge=0.05,
        )

        assert rec.has_value

    def test_no_value_negative_edge(self):
        rec = BettingRecommendation(
            market="over_2.5",
            selection="over",
            probability=0.45,
            odds_required=2.22,
            confidence=0.6,
            edge=-0.05,
        )

        assert not rec.has_value

    def test_confidence_level(self):
        high = BettingRecommendation(
            market="test",
            selection="test",
            probability=0.8,
            odds_required=1.25,
            confidence=0.85,
        )
        medium = BettingRecommendation(
            market="test",
            selection="test",
            probability=0.6,
            odds_required=1.67,
            confidence=0.65,
        )
        low = BettingRecommendation(
            market="test",
            selection="test",
            probability=0.5,
            odds_required=2.0,
            confidence=0.45,
        )

        assert high.confidence_level == "high"
        assert medium.confidence_level == "medium"
        assert low.confidence_level == "low"


class TestMLPredictionResult:
    def test_to_dict(self):
        result = MLPredictionResult(
            match_id="test-match",
            prediction_id="pred-123",
            status=PredictionStatus.SUCCESS,
            goals_prediction=GoalsPredictionResult(
                home_expected=1.5,
                away_expected=1.0,
                total_expected=2.5,
                over_25_prob=0.45,
                under_25_prob=0.55,
                confidence=0.7,
                model_version="v1.0.0",
            ),
            processing_time_ms=50.5,
        )

        data = result.to_dict()

        assert data["match_id"] == "test-match"
        assert data["status"] == "success"
        assert "goals" in data
        assert data["goals"]["total_expected"] == 2.5

    def test_has_recommendations(self):
        result = MLPredictionResult(
            match_id="test",
            prediction_id="pred",
            recommendations=[
                BettingRecommendation(
                    market="over_2.5",
                    selection="over",
                    probability=0.55,
                    odds_required=1.82,
                    confidence=0.7,
                )
            ],
        )

        assert result.has_recommendations

    def test_get_top_recommendations(self):
        result = MLPredictionResult(
            match_id="test",
            prediction_id="pred",
            recommendations=[
                BettingRecommendation(
                    market="over_2.5",
                    selection="over",
                    probability=0.55,
                    odds_required=1.82,
                    confidence=0.7,
                    edge=0.05,
                ),
                BettingRecommendation(
                    market="under_2.5",
                    selection="under",
                    probability=0.60,
                    odds_required=1.67,
                    confidence=0.8,
                    edge=0.08,
                ),
                BettingRecommendation(
                    market="1x2",
                    selection="home",
                    probability=0.45,
                    odds_required=2.22,
                    confidence=0.6,
                    edge=0.02,
                ),
            ],
        )

        top = result.get_top_recommendations(n=2)

        assert len(top) == 2
        # Should be sorted by edge
        assert top[0].edge >= top[1].edge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
