"""
Integration tests for ML API endpoints.

Checkpoint: 4.8
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


# Fixtures for API testing
@pytest.fixture
def mock_data_modules():
    """Mock the data modules to avoid import errors."""
    with patch.dict("sys.modules", {
        "core.data.schemas": Mock(),
        "core.data.repository": Mock(),
        "core.ml.service": Mock(),
        "core.ml.registry": Mock(),
        "core.ml.tracking": Mock(),
        "core.ml.training": Mock(),
        "core.ml.models": Mock(),
    }):
        yield


class TestMLAPIEndpoints:
    """Test ML API endpoints structure and response format."""

    def test_ml_predict_request_model(self):
        """Test MLPredictRequest model validation."""
        from api.routers import MLPredictRequest

        # Valid request
        request = MLPredictRequest(
            match_id="test-123",
            sport="football",
            home_team_name="Home FC",
            away_team_name="Away FC",
        )

        assert request.match_id == "test-123"
        assert request.sport == "football"
        assert request.include_recommendations is True  # Default

    def test_ml_predict_request_with_odds(self):
        """Test MLPredictRequest with market odds."""
        from api.routers import MLPredictRequest

        request = MLPredictRequest(
            match_id="test-123",
            sport="football",
            market_odds={
                "over_2.5": 1.90,
                "under_2.5": 2.00,
            },
        )

        assert request.market_odds["over_2.5"] == 1.90

    def test_ml_predict_response_model(self):
        """Test MLPredictResponse model structure."""
        from api.routers import MLPredictResponse
        from datetime import datetime

        response = MLPredictResponse(
            match_id="test-123",
            prediction_id="pred-456",
            status="success",
            goals={
                "home_expected": 1.5,
                "away_expected": 1.0,
                "total_expected": 2.5,
            },
            processing_time_ms=45.5,
            timestamp=datetime.now().isoformat(),
        )

        assert response.status == "success"
        assert response.goals["total_expected"] == 2.5

    def test_ml_batch_request_model(self):
        """Test MLBatchPredictRequest model."""
        from api.routers import MLBatchPredictRequest, MLPredictRequest

        request = MLBatchPredictRequest(
            matches=[
                MLPredictRequest(match_id="match-1", sport="football"),
                MLPredictRequest(match_id="match-2", sport="football"),
            ],
            include_recommendations=True,
        )

        assert len(request.matches) == 2

    def test_ml_model_status_response(self):
        """Test MLModelStatusResponse model."""
        from api.routers import MLModelStatusResponse

        response = MLModelStatusResponse(
            models={
                "poisson_goals": {
                    "versions_count": 3,
                    "active_version": "v1.2.0",
                },
                "gbm_handicap": {
                    "versions_count": 2,
                    "active_version": "v1.1.0",
                },
            },
            registry_path="data/ml/registry",
            total_predictions=1500,
            last_trained="2025-01-25T10:00:00",
        )

        assert response.models["poisson_goals"]["versions_count"] == 3

    def test_ml_train_request_model(self):
        """Test MLTrainRequest model."""
        from api.routers import MLTrainRequest

        request = MLTrainRequest(
            model_name="poisson_goals",
            force=False,
        )

        assert request.model_name == "poisson_goals"
        assert request.force is False

    def test_ml_train_response_model(self):
        """Test MLTrainResponse model."""
        from api.routers import MLTrainResponse

        # Success response
        response = MLTrainResponse(
            success=True,
            model_name="poisson_goals",
            version="v1.3.0",
            metrics={"accuracy": 0.75, "rmse": 0.8},
            error_message=None,
        )

        assert response.success
        assert response.metrics["accuracy"] == 0.75

        # Failure response
        error_response = MLTrainResponse(
            success=False,
            model_name="poisson_goals",
            version=None,
            metrics=None,
            error_message="Not enough training examples",
        )

        assert not error_response.success
        assert error_response.error_message is not None


class TestMLRouterRegistration:
    """Test ML router is properly registered."""

    def test_ml_router_exists(self):
        """Test ml_router is defined."""
        from api.routers import ml_router

        assert ml_router is not None
        assert ml_router.prefix == "/api/ml"

    def test_ml_router_tags(self):
        """Test ml_router has correct tags."""
        from api.routers import ml_router

        assert "ml" in ml_router.tags

    def test_register_routers_includes_ml(self):
        """Test register_routers includes ML router."""
        from api.routers import register_routers, ml_router
        from unittest.mock import Mock

        mock_app = Mock()
        register_routers(mock_app)

        # Check ml_router was included
        calls = mock_app.include_router.call_args_list
        routers_included = [call[0][0] for call in calls]

        assert ml_router in routers_included


class TestMLRouterEndpointStructure:
    """Test that ML router endpoints are defined correctly."""

    def test_predict_endpoint_exists(self):
        """Test /predict endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/predict" in routes

    def test_batch_predict_endpoint_exists(self):
        """Test /predict/batch endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/predict/batch" in routes

    def test_models_status_endpoint_exists(self):
        """Test /models/status endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/models/status" in routes

    def test_models_train_endpoint_exists(self):
        """Test /models/train endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/models/train" in routes

    def test_recommendations_endpoint_exists(self):
        """Test /recommendations endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/recommendations" in routes

    def test_tracking_summary_endpoint_exists(self):
        """Test /tracking/summary endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/tracking/summary" in routes

    def test_tracking_roi_endpoint_exists(self):
        """Test /tracking/roi endpoint is defined."""
        from api.routers import ml_router

        routes = [route.path for route in ml_router.routes]
        assert "/tracking/roi" in routes


class TestAPIIntegration:
    """Integration tests requiring full app setup."""

    @pytest.mark.skip(reason="Requires full app setup with all dependencies")
    def test_full_prediction_flow(self):
        """Test full prediction flow from request to response."""
        # This test would require:
        # 1. Setting up test data
        # 2. Training or loading models
        # 3. Making prediction
        # 4. Verifying response structure
        pass

    @pytest.mark.skip(reason="Requires full app setup with all dependencies")
    def test_batch_prediction_performance(self):
        """Test batch prediction handles multiple matches efficiently."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
