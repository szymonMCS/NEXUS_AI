# tests/test_new_sports_models.py
"""
Tests for additional sports models: Greyhound, Handball, Table Tennis.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGreyhoundModel:
    """Tests for GreyhoundModel."""
    
    def test_greyhound_model_initialization(self):
        """Test that GreyhoundModel initializes correctly."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        assert model.sport.value == "greyhound"
        assert model.model_name == "GreyhoundPredictor_v1"
        assert "trap" in model.required_features
        assert "weight" in model.required_features
    
    def test_greyhound_trap_bias(self):
        """Test that trap bias values are set correctly."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        # All traps should have bias values
        for trap in range(1, 7):
            assert trap in model.trap_bias
    
    def test_greyhound_feature_weights(self):
        """Test that feature weights sum to 1."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        total_weight = sum(model.feature_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_greyhound_rating_calculation(self):
        """Test runner rating calculation."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        runner = {
            "dog_name": "Test Dog",
            "trap": 1,
            "weight": 32,
            "age_months": 30,
            "recent_positions": [2, 1, 3, 2, 4, 1],
            "early_pace_rating": 7.5,
            "track_wins": 5,
            "track_experience": 20,
            "trainer_recent_form": 0.2,
        }
        
        rating = model._calculate_rating(runner)
        
        assert rating > 0
        assert rating > 1.0  # Should be higher than base rating
    
    def test_greyhound_predict_race(self):
        """Test race prediction with multiple runners."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        runners = [
            {"dog_name": "Dog A", "trap": 1, "weight": 32, "age_months": 30, "recent_positions": [1, 2, 1]},
            {"dog_name": "Dog B", "trap": 2, "weight": 31, "age_months": 36, "recent_positions": [2, 1, 3]},
            {"dog_name": "Dog C", "trap": 3, "weight": 33, "age_months": 24, "recent_positions": [3, 2, 2]},
        ]
        
        prediction = model.predict_race(runners)
        
        assert len(prediction.win_probabilities) == 3
        assert prediction.confidence > 0
        assert len(prediction.forecast) > 0
        assert len(prediction.reasoning) > 0
    
    def test_greyhound_validate_input(self):
        """Test input validation."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        # Should raise for empty runners
        with pytest.raises(ValueError, match="No runners provided"):
            model.validate_input({"runners": []})
        
        # Should raise for single runner
        with pytest.raises(ValueError, match="Need at least 2 runners"):
            model.validate_input({"runners": [{"dog_name": "Solo"}]})
    
    def test_greyhound_predict(self):
        """Test full prediction method."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        match_data = {
            "runners": [
                {"dog_name": "Fast Dog", "trap": 1, "weight": 30, "age_months": 28, "recent_positions": [1, 1]},
                {"dog_name": "Slow Dog", "trap": 2, "weight": 35, "age_months": 40, "recent_positions": [4, 3]},
            ]
        }
        
        result = model.predict(match_data)
        
        assert result.sport == "greyhound"
        assert result.predicted_winner in ["Fast Dog", "Slow Dog"]
        assert result.confidence > 0
        assert result.reliability_score > 0
    
    def test_greyhound_confidence_calculation(self):
        """Test confidence calculation based on ratings gap."""
        from core.models.greyhound_model import GreyhoundModel
        
        model = GreyhoundModel()
        
        # High gap should give higher confidence
        sorted_runners = [("Leader", 10.0), ("Follower", 5.0)]
        win_probs = {"Leader": 0.8, "Follower": 0.2}
        
        confidence = model._calculate_confidence(sorted_runners, win_probs)
        assert confidence > 0.5
        
        # Low gap should give lower confidence
        sorted_runners_tight = [("Leader", 5.1), ("Follower", 5.0)]
        win_probs_tight = {"Leader": 0.52, "Follower": 0.48}
        
        confidence_tight = model._calculate_confidence(sorted_runners_tight, win_probs_tight)
        assert confidence_tight < confidence


class TestHandballModel:
    """Tests for HandballModel."""
    
    def test_handball_model_initialization(self):
        """Test that HandballModel initializes correctly."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        assert model.sport.value == "handball"
        assert model.model_name == "HandballSEL_v1"
        assert "home_goals_scored" in model.required_features
    
    def test_handball_feature_weights(self):
        """Test that feature weights sum to 1."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        total_weight = sum(model.feature_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_handball_expected_goals(self):
        """Test expected goals calculation."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        # Average teams
        exp_goals = model._calculate_expected_goals(
            attack=25.0,
            defense=25.0,
            is_home=True,
            elo_diff=0
        )
        assert 20 <= exp_goals <= 35
        
        # Strong team at home
        strong_goals = model._calculate_expected_goals(
            attack=30.0,
            defense=22.0,
            is_home=True,
            elo_diff=100
        )
        assert strong_goals > exp_goals
    
    def test_handball_rest_factor(self):
        """Test rest factor calculation."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        # Well rested
        assert model._rest_factor(7) > 1.0
        
        # Normal rest
        assert model._rest_factor(4) == 1.0
        
        # Short rest
        assert model._rest_factor(2) < 1.0
    
    def test_handball_outcome_probabilities(self):
        """Test 1X2 probability calculation."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        # Strong home team
        home_prob, draw_prob, away_prob = model._calculate_outcome_probs(30, 22)
        assert home_prob > draw_prob
        assert home_prob > away_prob
        assert abs(home_prob + draw_prob + away_prob - 1.0) < 0.01
        
        # Even match
        home_prob_eq, draw_prob_eq, away_prob_eq = model._calculate_outcome_probs(25, 25)
        # Even match should have higher draw probability
        assert draw_prob_eq > 0.05
    
    def test_handball_predict_match(self):
        """Test full match prediction."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        match_data = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_goals_scored": 28.0,
            "home_goals_conceded": 24.0,
            "away_goals_scored": 26.0,
            "away_goals_conceded": 26.0,
            "home_elo": 1550,
            "away_elo": 1450,
        }
        
        prediction = model.predict_match(match_data)
        
        assert prediction.home_goals > 0
        assert prediction.away_goals > 0
        assert prediction.confidence > 0
        assert len(prediction.reasoning) > 0
        assert "over" in prediction.over_under.get(50.5, {})
        assert "home_cover" in prediction.handicap.get(1.5, {})
    
    def test_handball_predict(self):
        """Test full predict method."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        match_data = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_goals_scored": 28.0,
            "home_goals_conceded": 24.0,
            "away_goals_scored": 26.0,
            "away_goals_conceded": 26.0,
            "home_elo": 1550,
            "away_elo": 1450,
        }
        
        result = model.predict(match_data)
        
        assert result.sport == "handball"
        assert result.predicted_winner in ["home", "away", "draw"]
        assert result.confidence > 0
    
    def test_handball_confidence(self):
        """Test confidence calculation."""
        from core.models.handball_model import HandballModel
        
        model = HandballModel()
        
        # Clear favorite
        confidence_clear = model._calculate_confidence(
            home_prob=0.75,
            draw_prob=0.15,
            away_prob=0.10,
            elo_diff=100
        )
        
        # Even match
        confidence_even = model._calculate_confidence(
            home_prob=0.40,
            draw_prob=0.30,
            away_prob=0.30,
            elo_diff=0
        )
        
        assert confidence_clear > confidence_even


class TestTableTennisModel:
    """Tests for TableTennisModel."""
    
    def test_table_tennis_model_initialization(self):
        """Test that TableTennisModel initializes correctly."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        assert model.sport.value == "table_tennis"
        assert model.model_name == "TableTennisEnsemble_v1"
        assert "player1_rating" in model.required_features
    
    def test_table_tennis_feature_weights(self):
        """Test that feature weights sum to 1."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        total_weight = sum(model.feature_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_table_tennis_ranking_probability(self):
        """Test ranking to probability conversion."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        # Same ranking
        prob_same = model._ranking_probability(100, 100)
        assert prob_same == 0.5
        
        # Better ranking (lower number)
        prob_better = model._ranking_probability(50, 100)
        assert prob_better > 0.5
        
        # Much better ranking
        prob_much_better = model._ranking_probability(10, 500)
        assert prob_much_better > 0.8
    
    def test_table_tennis_form_probability(self):
        """Test form probability calculation."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        # Player 1 better form
        prob_p1_better = model._form_probability(
            p1_recent=8, p2_recent=3,
            p1_win_rate=0.70, p2_win_rate=0.45
        )
        assert prob_p1_better > 0.5
    
    def test_table_tennis_style_matchup(self):
        """Test style matchup probability."""
        from core.models.table_tennis_model import TableTennisModel, PlayingStyle
        
        model = TableTennisModel()
        
        # Offensive vs Defensive
        prob = model._style_matchup_prob(PlayingStyle.OFFENSIVE, PlayingStyle.DEFENSIVE)
        assert prob == 0.52
        
        # Unknown styles
        prob_unknown = model._style_matchup_prob(PlayingStyle.PENHOLD, PlayingStyle.ALL_ROUND)
        assert prob_unknown == 0.5
    
    def test_table_tennis_momentum(self):
        """Test momentum probability calculation."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        # Equal momentum
        prob_equal = model._momentum_probability(0, 0)
        assert prob_equal == 0.5
        
        # Player 1 has streak
        prob_streak = model._momentum_probability(3, 0)
        assert prob_streak > 0.5
    
    def test_table_tennis_predict_match(self):
        """Test full match prediction."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        match_data = {
            "player1_name": "Player A",
            "player2_name": "Player B",
            "player1_rating": 1600,
            "player2_rating": 1400,
            "player1_win_rate": 0.65,
            "player2_win_rate": 0.45,
            "player1_ranking": 50,
            "player2_ranking": 200,
            "player1_recent_wins": 7,
            "player2_recent_wins": 4,
        }
        
        prediction = model.predict_match(match_data)
        
        assert prediction.player1_win_prob > 0.5
        assert prediction.player2_win_prob < 0.5
        assert prediction.confidence > 0
        assert len(prediction.reasoning) > 0
        assert "over" in prediction.total_points.get(160.5, {})
    
    def test_table_tennis_predict(self):
        """Test full predict method."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        match_data = {
            "player1_name": "Player A",
            "player2_name": "Player B",
            "player1_rating": 1600,
            "player2_rating": 1400,
            "player1_win_rate": 0.65,
            "player2_win_rate": 0.45,
        }
        
        result = model.predict(match_data)
        
        assert result.sport == "table_tennis"
        assert result.predicted_winner in ["Player A", "Player B"]
        assert result.confidence > 0
    
    def test_table_tennis_confidence(self):
        """Test confidence calculation based on component agreement."""
        from core.models.table_tennis_model import TableTennisModel
        
        model = TableTennisModel()
        
        # All components agree
        agreed_probs = {"rating": 0.8, "ranking": 0.78, "form": 0.82, "h2h": 0.75}
        confidence_agreed = model._calculate_confidence(agreed_probs, 0.3)
        
        # Components disagree
        disagreed_probs = {"rating": 0.8, "ranking": 0.35, "form": 0.82, "h2h": 0.4}
        confidence_disagreed = model._calculate_confidence(disagreed_probs, 0.3)
        
        assert confidence_agreed > confidence_disagreed


class TestModelIntegration:
    """Integration tests for all models."""
    
    def test_all_models_have_base_interface(self):
        """Test that all models implement BaseModel interface."""
        from core.models.greyhound_model import GreyhoundModel
        from core.models.handball_model import HandballModel
        from core.models.table_tennis_model import TableTennisModel
        
        models = [GreyhoundModel(), HandballModel(), TableTennisModel()]
        
        for model in models:
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            assert hasattr(model, 'validate_input')
            assert hasattr(model, 'explain_prediction')
            assert hasattr(model, 'sport')
            assert hasattr(model, 'model_name')
    
    def test_all_models_produce_valid_predictions(self):
        """Test that all models produce valid predictions."""
        from core.models.greyhound_model import GreyhoundModel
        from core.models.handball_model import HandballModel
        from core.models.table_tennis_model import TableTennisModel
        
        # Greyhound
        greyhound = GreyhoundModel()
        greyhound_result = greyhound.predict({
            "runners": [
                {"dog_name": "A", "trap": 1, "weight": 32, "age_months": 30},
                {"dog_name": "B", "trap": 2, "weight": 31, "age_months": 28},
            ]
        })
        assert greyhound_result.confidence > 0
        assert greyhound_result.reliability_score > 0
        
        # Handball
        handball = HandballModel()
        handball_result = handball.predict({
            "home_team": "A", "away_team": "B",
            "home_goals_scored": 25, "home_goals_conceded": 23,
            "away_goals_scored": 24, "away_goals_conceded": 26,
        })
        assert handball_result.confidence > 0
        assert handball_result.reliability_score > 0
        
        # Table Tennis
        tt = TableTennisModel()
        tt_result = tt.predict({
            "player1_name": "A", "player2_name": "B",
            "player1_rating": 1500, "player2_rating": 1500,
        })
        assert tt_result.confidence > 0
        assert tt_result.reliability_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
