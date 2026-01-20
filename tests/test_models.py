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
            "home_player": {
                "name": "Djokovic N.",
                "ranking": 1,
                "recent_matches": [
                    {"won": True, "opponent_ranking": 10},
                    {"won": True, "opponent_ranking": 5},
                    {"won": False, "opponent_ranking": 3},
                    {"won": True, "opponent_ranking": 8},
                    {"won": True, "opponent_ranking": 12},
                ],
            },
            "away_player": {
                "name": "Sinner J.",
                "ranking": 4,
                "recent_matches": [
                    {"won": True, "opponent_ranking": 15},
                    {"won": True, "opponent_ranking": 20},
                    {"won": True, "opponent_ranking": 7},
                    {"won": False, "opponent_ranking": 2},
                    {"won": True, "opponent_ranking": 18},
                ],
            },
            "surface": "hard",
            "tournament_level": "Grand Slam",
            "h2h": {
                "home_wins": 5,
                "away_wins": 3,
            },
        }

    def test_model_name(self, model):
        """Test model has correct name."""
        assert model.name == "TennisModel"

    def test_validate_input_valid(self, model, valid_match_data):
        """Test validation passes for valid input."""
        assert model.validate_input(valid_match_data) is True

    def test_validate_input_missing_home(self, model):
        """Test validation fails without home player."""
        data = {"away_player": {"name": "Player", "ranking": 1}}
        assert model.validate_input(data) is False

    def test_validate_input_missing_away(self, model):
        """Test validation fails without away player."""
        data = {"home_player": {"name": "Player", "ranking": 1}}
        assert model.validate_input(data) is False

    def test_predict_returns_result(self, model, valid_match_data):
        """Test predict returns PredictionResult."""
        result = model.predict(valid_match_data)

        assert hasattr(result, "home_probability")
        assert hasattr(result, "away_probability")
        assert hasattr(result, "confidence")

    def test_predict_probabilities_sum_to_one(self, model, valid_match_data):
        """Test probabilities sum to approximately 1."""
        result = model.predict(valid_match_data)
        total = result.home_probability + result.away_probability

        assert abs(total - 1.0) < 0.01

    def test_predict_higher_ranked_favored(self, model, valid_match_data):
        """Test that higher ranked player is favored."""
        result = model.predict(valid_match_data)

        # Djokovic (rank 1) should be favored over Sinner (rank 4)
        assert result.home_probability > result.away_probability

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

    def test_ranking_factor(self, model):
        """Test ranking factor calculation."""
        # Large ranking difference
        factor = model._calculate_ranking_factor(1, 100)
        assert factor > 0.6

        # Equal rankings
        factor = model._calculate_ranking_factor(50, 50)
        assert 0.45 <= factor <= 0.55

    def test_form_factor(self, model):
        """Test form factor calculation."""
        # All wins
        matches = [{"won": True} for _ in range(5)]
        factor = model._calculate_form_factor(matches)
        assert factor > 0.8

        # All losses
        matches = [{"won": False} for _ in range(5)]
        factor = model._calculate_form_factor(matches)
        assert factor < 0.3


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
            "home_team": {
                "name": "Lakers",
                "rating": 1600,
                "recent_matches": [
                    {"won": True, "points_scored": 115, "points_against": 102},
                    {"won": True, "points_scored": 120, "points_against": 110},
                    {"won": False, "points_scored": 98, "points_against": 105},
                    {"won": True, "points_scored": 108, "points_against": 100},
                    {"won": True, "points_scored": 112, "points_against": 99},
                ],
                "days_rest": 2,
            },
            "away_team": {
                "name": "Celtics",
                "rating": 1580,
                "recent_matches": [
                    {"won": True, "points_scored": 118, "points_against": 105},
                    {"won": True, "points_scored": 110, "points_against": 102},
                    {"won": True, "points_scored": 125, "points_against": 115},
                    {"won": False, "points_against": 95, "points_scored": 100},
                    {"won": True, "points_scored": 108, "points_against": 98},
                ],
                "days_rest": 1,
            },
            "h2h": {
                "home_wins": 2,
                "away_wins": 3,
            },
        }

    def test_model_name(self, model):
        """Test model has correct name."""
        assert model.name == "BasketballModel"

    def test_validate_input_valid(self, model, valid_match_data):
        """Test validation passes for valid input."""
        assert model.validate_input(valid_match_data) is True

    def test_predict_returns_result(self, model, valid_match_data):
        """Test predict returns valid result."""
        result = model.predict(valid_match_data)

        assert hasattr(result, "home_probability")
        assert hasattr(result, "away_probability")

    def test_predict_probabilities_valid(self, model, valid_match_data):
        """Test probabilities are valid."""
        result = model.predict(valid_match_data)

        assert 0 <= result.home_probability <= 1
        assert 0 <= result.away_probability <= 1
        assert abs(result.home_probability + result.away_probability - 1.0) < 0.01

    def test_home_advantage(self, model, valid_match_data):
        """Test home team has slight advantage with equal teams."""
        # Make teams equal in ratings
        valid_match_data["home_team"]["rating"] = 1500
        valid_match_data["away_team"]["rating"] = 1500
        valid_match_data["home_team"]["days_rest"] = 2
        valid_match_data["away_team"]["days_rest"] = 2

        result = model.predict(valid_match_data)

        # Home team should have slight advantage
        assert result.home_probability >= 0.50

    def test_rest_days_factor(self, model):
        """Test rest days advantage calculation."""
        # Well rested (3+ days)
        factor = model._calculate_rest_factor(3)
        assert factor > 0.5

        # Back-to-back (0 days)
        factor = model._calculate_rest_factor(0)
        assert factor < 0.5

    def test_rating_factor(self, model):
        """Test rating factor calculation."""
        # Large rating advantage
        factor = model._calculate_rating_factor(1700, 1400)
        assert factor > 0.6

        # Equal ratings
        factor = model._calculate_rating_factor(1500, 1500)
        assert 0.45 <= factor <= 0.55


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
        assert kelly <= calculator.max_stake

    def test_fractional_kelly(self, calculator):
        """Test fractional Kelly multiplier."""
        full_kelly = calculator.calculate_kelly_stake(0.60, 2.0, fractional=1.0)
        quarter_kelly = calculator.calculate_kelly_stake(0.60, 2.0, fractional=0.25)

        assert quarter_kelly < full_kelly
        assert abs(quarter_kelly - full_kelly * 0.25) < 0.001

    def test_quality_adjusted_stake(self, calculator):
        """Test quality adjustment."""
        base_stake = calculator.calculate_kelly_stake(0.60, 2.0)
        adjusted = calculator.apply_quality_adjustment(base_stake, 0.8)

        assert adjusted < base_stake
        assert adjusted == base_stake * 0.8

    def test_is_value_bet(self, calculator):
        """Test value bet identification."""
        # Clear value
        assert calculator.is_value_bet(0.60, 2.0, min_edge=0.03) is True

        # No value
        assert calculator.is_value_bet(0.45, 2.0, min_edge=0.03) is False


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

    def test_calculate_reliability(self):
        """Test reliability score calculation."""
        from core.models.base_model import BaseModel

        # High confidence, good quality
        reliability = BaseModel.calculate_reliability(0.85, 0.90)
        assert reliability > 0.7

        # Low confidence, good quality
        reliability = BaseModel.calculate_reliability(0.5, 0.90)
        assert reliability < 0.7

        # High confidence, poor quality
        reliability = BaseModel.calculate_reliability(0.85, 0.4)
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
