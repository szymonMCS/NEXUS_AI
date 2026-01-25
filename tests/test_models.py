# tests/test_models.py
"""
Unit tests for NEXUS AI prediction models.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestTennisModel:
    """Tests for TennisModel."""

    @pytest.fixture
    def model(self):
        """Create TennisModel instance."""
        from core.models.tennis_model import TennisModel
        return TennisModel()

    @pytest.fixture
    def valid_match_data(self):
        """Sample valid match data."""
        return {
            "p1_ranking": 1,
            "p1_elo": 1550.0,
            "p1_recent_form": 0.8,
            "p1_surface_form": 0.85,
            "p1_fatigue": 0.2,
            "p2_ranking": 4,
            "p2_elo": 1520.0,
            "p2_recent_form": 0.75,
            "p2_surface_form": 0.70,
            "p2_fatigue": 0.1,
            "surface": "hard",
            "tournament_category": "1000",
            "h2h": {
                "p1_wins": 5,
                "p2_wins": 3,
            },
        }

    def test_model_name(self, model):
        """Test model has correct name."""
        assert model.model_name == "TennisStatisticalModel"

    def test_validate_input_valid(self, model, valid_match_data):
        """Test validation passes for valid input."""
        assert model.validate_input(valid_match_data) is True

    def test_validate_input_missing_p1_ranking(self, model):
        """Test validation with missing p1_ranking still returns True (allows partial data)."""
        data = {"p2_ranking": 1, "p1_recent_form": 0.5}
        # The model allows partial data, returns True
        assert model.validate_input(data) is True

    def test_predict_returns_result(self, model, valid_match_data):
        """Test predict returns PredictionResult."""
        result = model.predict(valid_match_data)

        assert hasattr(result, "probabilities")
        assert hasattr(result, "confidence")
        assert "p1" in result.probabilities
        assert "p2" in result.probabilities

    def test_predict_probabilities_sum_to_one(self, model, valid_match_data):
        """Test probabilities sum to approximately 1."""
        result = model.predict(valid_match_data)
        total = result.probabilities["p1"] + result.probabilities["p2"]

        assert abs(total - 1.0) < 0.01

    def test_predict_higher_ranked_favored(self, model, valid_match_data):
        """Test that higher ranked player is favored."""
        result = model.predict(valid_match_data)

        # P1 (rank 1) should be favored over P2 (rank 4)
        assert result.probabilities["p1"] > result.probabilities["p2"]

    def test_predict_confidence_range(self, model, valid_match_data):
        """Test confidence is in valid range."""
        result = model.predict(valid_match_data)

        assert 0 <= result.confidence <= 1

    def test_explain_prediction(self, model, valid_match_data):
        """Test explanation generation."""
        result = model.predict(valid_match_data)
        explanations = model.explain_prediction(valid_match_data, result)

        assert isinstance(explanations, list)
        assert len(explanations) > 0
        assert all(isinstance(e, str) for e in explanations)


class TestBasketballModel:
    """Tests for BasketballModel."""

    @pytest.fixture
    def model(self):
        """Create BasketballModel instance."""
        from core.models.basketball_model import BasketballModel
        return BasketballModel()

    @pytest.fixture
    def valid_match_data(self):
        """Sample valid match data."""
        return {
            "home_rating": 1600,
            "away_rating": 1580,
            "home_rest_days": 2,
            "away_rest_days": 1,
            "h2h_home_wins": 2,
            "h2h_away_wins": 3,
        }

    def test_model_name(self, model):
        """Test model has correct name."""
        assert model.model_name == "BasketballStatisticalModel"

    def test_validate_input_valid(self, model, valid_match_data):
        """Test validation passes for valid input."""
        assert model.validate_input(valid_match_data) is True

    def test_predict_returns_result(self, model, valid_match_data):
        """Test predict returns valid result."""
        result = model.predict(valid_match_data)

        assert hasattr(result, "probabilities")
        assert hasattr(result, "confidence")
        assert "home" in result.probabilities
        assert "away" in result.probabilities

    def test_predict_probabilities_valid(self, model, valid_match_data):
        """Test probabilities are valid."""
        result = model.predict(valid_match_data)

        assert 0 <= result.probabilities["home"] <= 1
        assert 0 <= result.probabilities["away"] <= 1
        assert abs(result.probabilities["home"] + result.probabilities["away"] - 1.0) < 0.01

    def test_home_advantage(self, model, valid_match_data):
        """Test home team has slight advantage with equal teams."""
        # Make teams equal in ratings
        valid_match_data["home_rating"] = 1500
        valid_match_data["away_rating"] = 1500
        valid_match_data["home_rest_days"] = 2
        valid_match_data["away_rest_days"] = 2

        result = model.predict(valid_match_data)

        # Home team should have slight advantage
        assert result.probabilities["home"] >= 0.50


class TestValueCalculator:
    """Tests for ValueCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create ValueCalculator instance."""
        from core.value_calculator import ValueCalculator
        return ValueCalculator()

    def test_calculate_edge_positive(self, calculator):
        """Test positive edge calculation."""
        # True probability 55%, odds 2.0 (implied 50%)
        edge = calculator.calculate_edge(0.55, 2.0)
        assert edge > 0  # Should show positive edge

    def test_calculate_edge_negative(self, calculator):
        """Test negative edge calculation."""
        # True probability 45%, odds 2.0 (implied 50%)
        edge = calculator.calculate_edge(0.45, 2.0)
        assert edge < 0  # Should show negative edge

    def test_calculate_edge_breakeven(self, calculator):
        """Test breakeven edge."""
        # True probability equals implied probability
        edge = calculator.calculate_edge(0.50, 2.0)
        assert abs(edge) < 0.01

    def test_kelly_stake_positive_edge(self, calculator):
        """Test Kelly stake with positive edge."""
        # 55% win probability, 2.0 odds
        kelly = calculator.calculate_kelly_stake(0.55, 2.0)
        assert kelly > 0

    def test_kelly_stake_negative_edge(self, calculator):
        """Test Kelly stake with negative edge."""
        # 45% win probability, 2.0 odds
        kelly = calculator.calculate_kelly_stake(0.45, 2.0)
        assert kelly == 0  # Should not bet

    def test_kelly_stake_capped(self, calculator):
        """Test Kelly stake is capped at max."""
        # Very high edge scenario
        kelly = calculator.calculate_kelly_stake(0.95, 2.0)
        assert kelly <= calculator.MAX_STAKE_PCT

    def test_fractional_kelly(self, calculator):
        """Test fractional Kelly multiplier."""
        # Use a low probability scenario where the cap doesn't apply
        # MAX_STAKE_PCT = 0.05, so use prob and odds that give kelly < 0.05
        # Kelly for p=0.52, odds=2.0: (1*0.52-0.48)/1 = 0.04 < 0.05
        half_kelly = calculator.calculate_kelly_stake(0.52, 2.0, fraction=0.5)
        full_kelly = calculator.calculate_kelly_stake(0.52, 2.0, fraction=1.0)

        assert half_kelly < full_kelly
        # When cap doesn't apply, fractional kelly should be proportional
        assert abs(half_kelly - full_kelly * 0.5) < 0.001

    def test_quality_multiplier(self, calculator):
        """Test quality multiplier lookup."""
        # Excellent quality (0.85-1.00) -> 1.0
        assert calculator.get_quality_multiplier(0.90) == 1.0
        # Good quality (0.70-0.85) -> 0.9
        assert calculator.get_quality_multiplier(0.75) == 0.9
        # Moderate quality (0.50-0.70) -> 0.7
        assert calculator.get_quality_multiplier(0.60) == 0.7
        # Low quality (0.40-0.50) -> 0.5
        assert calculator.get_quality_multiplier(0.45) == 0.5
        # Very low quality (0.00-0.40) -> 0.3
        assert calculator.get_quality_multiplier(0.30) == 0.3


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_elo_probability(self):
        """Test ELO probability calculation."""
        from core.models.base_model import BaseModel

        # Higher rated should win more often
        prob = BaseModel.elo_probability(1600, 1400)
        assert prob > 0.5

        # Equal ratings
        prob = BaseModel.elo_probability(1500, 1500)
        assert abs(prob - 0.5) < 0.01

        # Lower rated should lose more often
        prob = BaseModel.elo_probability(1400, 1600)
        assert prob < 0.5

    def test_calculate_reliability_score(self):
        """Test reliability score calculation."""
        from core.models.tennis_model import TennisModel
        from core.models.base_model import PredictionResult

        # Create a prediction result
        prediction = PredictionResult(
            sport="tennis",
            predicted_winner="p1",
            confidence=0.85,
            probabilities={"p1": 0.65, "p2": 0.35},
            model_name="test",
            features_used=["ranking"],
            feature_values={}
        )

        # Create match data - use field names expected by base_model
        match_data = {
            "home_ranking": 1,
            "away_ranking": 4,
            "home_recent_form": 0.8,
            "away_recent_form": 0.75,
            "h2h_record": {"home_wins": 5, "away_wins": 3},
        }

        # Use TennisModel which is a concrete implementation
        model = TennisModel()

        # High confidence, good quality
        reliability = model.calculate_reliability_score(match_data, prediction)
        assert reliability > 0.7

        # Low confidence prediction
        prediction_low_conf = PredictionResult(
            sport="tennis",
            predicted_winner="p1",
            confidence=0.5,
            probabilities={"p1": 0.55, "p2": 0.45},
            model_name="test",
            features_used=["ranking"],
            feature_values={}
        )
        reliability = model.calculate_reliability_score(match_data, prediction_low_conf)
        assert reliability < 0.7

        # High confidence, poor quality data (missing features)
        prediction_high = PredictionResult(
            sport="tennis",
            predicted_winner="p1",
            confidence=0.85,
            probabilities={"p1": 0.65, "p2": 0.35},
            model_name="test",
            features_used=["ranking"],
            feature_values={}
        )
        match_data_poor = {}
        reliability = model.calculate_reliability_score(match_data_poor, prediction_high)
        assert reliability < 0.5


# === Test fixtures for pytest ===

@pytest.fixture(scope="session")
def setup_test_env():
    """Setup test environment."""
    import os
    os.environ["NEXUS_ENV"] = "test"
    os.environ["NEXUS_MODE"] = "lite"
    yield
    # Cleanup if needed


# Run with: pytest tests/test_models.py -v
