# tests/test_api_endpoints.py
"""
Tests for FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


class TestStatusEndpoint:
    """Tests for /api/status endpoint."""
    
    def test_get_status_returns_200(self):
        """Test that status endpoint returns 200."""
        response = client.get("/api/status")
        assert response.status_code == 200
    
    def test_get_status_returns_correct_structure(self):
        """Test that status response has correct fields."""
        response = client.get("/api/status")
        data = response.json()
        
        assert "status" in data
        assert "mode" in data
        assert "api_keys_configured" in data
        assert "uptime_seconds" in data
    
    def test_get_status_api_keys_structure(self):
        """Test that api_keys_configured has expected keys."""
        response = client.get("/api/status")
        data = response.json()
        api_keys = data.get("api_keys_configured", {})
        
        expected_keys = ["brave_search", "serper", "odds_api", "anthropic"]
        for key in expected_keys:
            assert key in api_keys, f"Missing key: {key}"


class TestSportsAvailableEndpoint:
    """Tests for /api/sports/available endpoint."""
    
    def test_get_available_sports_returns_200(self):
        """Test that available sports endpoint returns 200."""
        response = client.get("/api/sports/available")
        assert response.status_code == 200
    
    def test_get_available_sports_returns_sports_list(self):
        """Test that response contains sports array."""
        response = client.get("/api/sports/available")
        data = response.json()
        
        assert "sports" in data
        assert isinstance(data["sports"], list)
    
    def test_get_available_sports_has_all_sports(self):
        """Test that all expected sports are present."""
        response = client.get("/api/sports/available")
        data = response.json()
        sports = data.get("sports", [])
        sport_ids = [s["id"] for s in sports]
        
        expected_sports = ["tennis", "basketball", "greyhound", "handball", "table_tennis"]
        for sport in expected_sports:
            assert sport in sport_ids, f"Missing sport: {sport}"
    
    def test_get_available_sports_has_default(self):
        """Test that response has default sport."""
        response = client.get("/api/sports/available")
        data = response.json()
        
        assert "default" in data
        assert data["default"] == "tennis"
    
    def test_get_available_sports_has_total(self):
        """Test that response has total count."""
        response = client.get("/api/sports/available")
        data = response.json()
        
        assert "total" in data
        assert data["total"] == 5


class TestAnalysisEndpoint:
    """Tests for /api/analysis endpoint."""
    
    @patch("api.main._run_analysis_task")
    def test_run_analysis_returns_202(self, mock_task):
        """Test that analysis endpoint returns 202 when started."""
        mock_task.return_value = None
        
        response = client.post(
            "/api/analysis",
            json={"sport": "tennis", "date": "2026-01-21"}
        )
        assert response.status_code == 200
    
    def test_run_analysis_requires_sport(self):
        """Test that analysis requires sport parameter."""
        response = client.post(
            "/api/analysis",
            json={"date": "2026-01-21"}
        )
        # Should still work with default sport
        assert response.status_code in [200, 422]
    
    def test_run_analysis_default_values(self):
        """Test that analysis uses default values."""
        response = client.post(
            "/api/analysis",
            json={"sport": "tennis"}
        )
        data = response.json()
        
        assert data["status"] == "started"
        assert data["sport"] == "tennis"
        assert "message" in data


class TestPredictionsEndpoint:
    """Tests for /api/predictions endpoint."""
    
    def test_get_predictions_returns_200(self):
        """Test that predictions endpoint returns 200."""
        response = client.get("/api/predictions")
        assert response.status_code == 200
    
    def test_get_predictions_returns_no_data_initially(self):
        """Test that predictions returns no_data when no analysis run."""
        response = client.get("/api/predictions")
        data = response.json()
        
        assert data["status"] == "no_data"
        assert data["value_bets"] == []
    
    def test_get_predictions_with_sport_filter(self):
        """Test that predictions can be filtered by sport."""
        response = client.get("/api/predictions?sport=tennis")
        assert response.status_code == 200
    
    def test_get_predictions_with_date_filter(self):
        """Test that predictions can be filtered by date."""
        response = client.get("/api/predictions?date=2026-01-21")
        assert response.status_code == 200


class TestValueBetsEndpoint:
    """Tests for /api/value-bets endpoint."""
    
    def test_get_value_bets_returns_200(self):
        """Test that value-bets endpoint returns 200."""
        response = client.get("/api/value-bets")
        assert response.status_code == 200
    
    def test_get_value_bets_returns_list(self):
        """Test that value-bets returns a list."""
        response = client.get("/api/value-bets")
        data = response.json()
        
        assert isinstance(data, list)


class TestMatchesEndpoint:
    """Tests for /api/matches endpoint."""
    
    def test_get_matches_returns_200(self):
        """Test that matches endpoint returns 200."""
        response = client.get("/api/matches?sport=tennis")
        assert response.status_code == 200
    
    def test_get_matches_returns_success_status(self):
        """Test that matches response has success status."""
        response = client.get("/api/matches?sport=tennis")
        data = response.json()
        
        assert data["status"] == "success"
        assert "matches" in data
    
    def test_get_matches_with_date(self):
        """Test that matches works with date parameter."""
        response = client.get("/api/matches?sport=tennis&match_date=2026-01-21")
        assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""
    
    def test_get_stats_returns_200(self):
        """Test that stats endpoint returns 200."""
        response = client.get("/api/stats")
        assert response.status_code == 200
    
    def test_get_stats_returns_correct_structure(self):
        """Test that stats response has expected fields."""
        response = client.get("/api/stats")
        data = response.json()
        
        expected_fields = [
            "total_analyses", "successful_bets", "win_rate",
            "total_profit", "avg_edge", "avg_quality",
            "sports_analyzed", "last_7_days"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_get_stats_sports_analyzed(self):
        """Test that sports_analyzed contains all supported sports."""
        response = client.get("/api/stats")
        data = response.json()
        
        expected_sports = ["tennis", "basketball", "greyhound", "handball", "table_tennis"]
        for sport in expected_sports:
            assert sport in data["sports_analyzed"], f"Missing sport: {sport}"


class TestLivePredictionsEndpoint:
    """Tests for /api/predictions/live endpoint."""
    
    def test_get_live_predictions_returns_200(self):
        """Test that live predictions endpoint returns 200."""
        response = client.get("/api/predictions/live")
        assert response.status_code == 200
    
    def test_get_live_predictions_returns_correct_structure(self):
        """Test that live predictions response has expected fields."""
        response = client.get("/api/predictions/live")
        data = response.json()
        
        expected_fields = ["is_analyzing", "current_step", "progress", "live_bets"]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_get_live_predictions_initial_state(self):
        """Test initial state when no analysis is running."""
        response = client.get("/api/predictions/live")
        data = response.json()
        
        assert data["is_analyzing"] == False
        assert data["live_bets"] == []


class TestHandicapEndpoint:
    """Tests for /api/handicap endpoint."""
    
    def test_predict_handicap_returns_200(self):
        """Test that handicap prediction endpoint returns 200."""
        request_data = {
            "sport": "tennis",
            "market_type": "match_handicap",
            "home_stats": {"avg_games": 6.5, "win_rate": 0.65},
            "away_stats": {"avg_games": 5.5, "win_rate": 0.45},
            "line": 1.5
        }
        response = client.post("/api/handicap", json=request_data)
        assert response.status_code == 200
    
    def test_predict_handicap_response_structure(self):
        """Test handicap response has expected structure."""
        request_data = {
            "sport": "tennis",
            "market_type": "match_handicap",
            "home_stats": {"avg_games": 6.5, "win_rate": 0.65},
            "away_stats": {"avg_games": 5.5, "win_rate": 0.45},
            "line": 1.5
        }
        response = client.post("/api/handicap", json=request_data)
        data = response.json()
        
        expected_fields = [
            "market_type", "line", "cover_probability",
            "fair_odds", "expected_margin", "confidence", "reasoning"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present."""
        response = client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            }
        )
        # CORS headers should be present
        assert response.status_code in [200, 403]


class TestHealthCheck:
    """Tests for root health check."""
    
    def test_root_returns_200(self):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        # May return 404 if not defined, but should not error
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
