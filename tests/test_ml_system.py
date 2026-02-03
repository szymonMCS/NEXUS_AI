"""
Unit tests for ML system.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml import (
    get_prediction_service,
    get_model_registry,
    StatisticalEnsemble,
    EloPredictor,
    FormPredictor,
    PredictionResult
)


class TestStatisticalModels(unittest.TestCase):
    """Test statistical prediction models."""
    
    def test_elo_predictor_creation(self):
        """Test ELO predictor initialization."""
        elo = EloPredictor(sport="tennis")
        self.assertEqual(elo.sport, "tennis")
        self.assertEqual(elo.name, "ELO")
        self.assertFalse(elo.is_trained)
    
    def test_elo_rating_calculation(self):
        """Test ELO rating calculations."""
        elo = EloPredictor(sport="tennis")
        
        # Create ratings
        rating1 = elo.get_or_create_rating("Player A")
        rating2 = elo.get_or_create_rating("Player B")
        
        # Both should start at 1500
        self.assertEqual(rating1.rating, 1500.0)
        self.assertEqual(rating2.rating, 1500.0)
        
        # Expected score should be 0.5 for equal ratings
        expected = rating1.expected_score(rating2.rating)
        self.assertAlmostEqual(expected, 0.5, places=2)
    
    def test_form_predictor(self):
        """Test form predictor."""
        form = FormPredictor(sport="tennis")
        
        # Add some results
        form.add_match_result("Player A", "W")
        form.add_match_result("Player A", "W")
        form.add_match_result("Player A", "L")
        
        # Form should be between 0 and 1
        form_score = form.calculate_form_score("Player A")
        self.assertGreaterEqual(form_score, 0.0)
        self.assertLessEqual(form_score, 1.0)
    
    def test_statistical_ensemble_predict(self):
        """Test StatisticalEnsemble prediction."""
        ensemble = StatisticalEnsemble(sport="tennis")
        
        # Test prediction without training
        features = {
            "home_player": "Player A",
            "away_player": "Player B"
        }
        
        result = ensemble.predict(features)
        
        # Check result structure
        self.assertIsInstance(result, PredictionResult)
        self.assertGreaterEqual(result.home_win_prob, 0.0)
        self.assertLessEqual(result.home_win_prob, 1.0)
        self.assertGreaterEqual(result.away_win_prob, 0.0)
        self.assertLessEqual(result.away_win_prob, 1.0)
        self.assertAlmostEqual(result.home_win_prob + result.away_win_prob, 1.0, places=5)
    
    def test_ensemble_fit(self):
        """Test training ensemble with sample data."""
        ensemble = StatisticalEnsemble(sport="tennis")
        
        from datetime import datetime
        
        # Sample match data
        matches = [
            {
                "home_player": "Player A",
                "away_player": "Player B",
                "home_score": 2,
                "away_score": 1,
                "date": datetime.now()
            },
            {
                "home_player": "Player B",
                "away_player": "Player A",
                "home_score": 0,
                "away_score": 2,
                "date": datetime.now()
            },
        ]
        
        ensemble.fit(matches)
        
        # Should be marked as trained
        self.assertTrue(ensemble.is_trained)
        self.assertEqual(ensemble.metadata.training_samples, 2)


class TestPredictionService(unittest.TestCase):
    """Test prediction service."""
    
    def test_service_creation(self):
        """Test prediction service initialization."""
        service = get_prediction_service()
        self.assertIsNotNone(service)
    
    def test_basic_prediction(self):
        """Test basic prediction."""
        service = get_prediction_service()
        
        result = service.predict("tennis", "Player A", "Player B")
        
        # Verify result
        self.assertIsInstance(result, PredictionResult)
        self.assertGreaterEqual(result.home_win_prob, 0.0)
        self.assertLessEqual(result.home_win_prob, 1.0)
        self.assertEqual(result.model_name, "StatisticalEnsemble")
    
    def test_prediction_with_features(self):
        """Test prediction with additional features."""
        service = get_prediction_service()
        
        features = {
            "home_rank": 5,
            "away_rank": 10,
            "home_win_rate": 0.7,
            "away_win_rate": 0.6
        }
        
        result = service.predict("tennis", "Player A", "Player B", features)
        
        self.assertIsInstance(result, PredictionResult)
        self.assertGreaterEqual(result.confidence, 0.0)
    
    def test_value_prediction(self):
        """Test value betting prediction."""
        service = get_prediction_service()
        
        explanation = service.predict_with_value(
            "tennis", "Player A", "Player B",
            home_odds=1.85,
            away_odds=2.10
        )
        
        # Check explanation structure
        self.assertIn("selected_outcome", dir(explanation))
        self.assertIn("expected_value", dir(explanation))
        self.assertIn("reasoning", dir(explanation))
        self.assertIsInstance(explanation.reasoning, list)
    
    def test_model_info(self):
        """Test getting model info."""
        service = get_prediction_service()
        
        info = service.get_model_info("tennis")
        
        self.assertIn("sport", info)
        self.assertIn("available_models", info)
        self.assertEqual(info["sport"], "tennis")
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        service = get_prediction_service()
        
        matches = [
            {"home_player": "A", "away_player": "B"},
            {"home_player": "C", "away_player": "D"},
        ]
        
        results = service.batch_predict("tennis", matches)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, PredictionResult)


class TestModelRegistry(unittest.TestCase):
    """Test model registry."""
    
    def test_registry_creation(self):
        """Test registry initialization."""
        registry = get_model_registry()
        self.assertIsNotNone(registry)
    
    def test_get_best_model(self):
        """Test getting best model for sport."""
        registry = get_model_registry()
        
        model = registry.get_best_model("tennis")
        
        self.assertIsNotNone(model)
        self.assertIsInstance(model, (StatisticalEnsemble,))
    
    def test_list_models(self):
        """Test listing models."""
        registry = get_model_registry()
        
        models = registry.list_models("tennis")
        
        # Should return list (may be empty if no models saved)
        self.assertIsInstance(models, list)


class TestPredictionResult(unittest.TestCase):
    """Test prediction result data structure."""
    
    def test_result_creation(self):
        """Test creating prediction result."""
        result = PredictionResult(
            home_win_prob=0.6,
            away_win_prob=0.4,
            confidence=0.75,
            model_name="TestModel"
        )
        
        self.assertEqual(result.home_win_prob, 0.6)
        self.assertEqual(result.away_win_prob, 0.4)
        self.assertEqual(result.confidence, 0.75)
        self.assertEqual(result.model_name, "TestModel")
    
    def test_result_probabilities_sum(self):
        """Test that probabilities sum to 1."""
        result = PredictionResult(
            home_win_prob=0.55,
            away_win_prob=0.45,
            confidence=0.8
        )
        
        self.assertAlmostEqual(
            result.home_win_prob + result.away_win_prob,
            1.0,
            places=5
        )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionService))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionResult))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
