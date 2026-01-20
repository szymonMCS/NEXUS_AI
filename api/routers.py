# api/routers.py
"""
API Routers for NEXUS AI.
Organized endpoints for predictions, ensemble, monitoring, and admin.
Based on concepts from backend_draft/api/routers.py
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# === PYDANTIC MODELS ===

class EnsemblePredictRequest(BaseModel):
    """Request for ensemble prediction."""
    sport: str = "tennis"
    match_data: Dict[str, Any]
    method: str = "weighted_average"  # weighted_average, confidence_weighted, stacking, voting, bayesian


class EnsemblePredictResponse(BaseModel):
    """Response for ensemble prediction."""
    sport: str
    home_probability: float
    away_probability: float
    ensemble_confidence: float
    method: str
    model_weights: Dict[str, float]
    reasoning: List[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response for model information."""
    registered_sports: List[str]
    models_per_sport: Dict[str, List[str]]
    current_weights: Dict[str, Dict[str, float]]
    prediction_count: int
    available_methods: List[str]


class MonitoringStatsResponse(BaseModel):
    """Response for monitoring stats."""
    total_predictions: int
    uptime_seconds: float
    by_sport: Dict[str, Any]
    by_model: Dict[str, Any]
    predictions_today: int


class PerformanceMetricsResponse(BaseModel):
    """Response for performance metrics."""
    period_days: int
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    avg_edge: float
    by_sport: Dict[str, Any]
    by_model: Dict[str, Any]


class AlertsResponse(BaseModel):
    """Response for alerts."""
    alerts: List[Dict[str, Any]]


# === ROUTERS ===

# Ensemble router
ensemble_router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])

# Monitoring router
monitoring_router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Admin router
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


# === ENSEMBLE ENDPOINTS ===

@ensemble_router.get("/info", response_model=ModelInfoResponse)
async def get_ensemble_info():
    """Get ensemble manager information."""
    try:
        from core.ensemble import get_ensemble_manager
        manager = get_ensemble_manager()
        info = manager.get_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting ensemble info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ensemble_router.post("/predict", response_model=EnsemblePredictResponse)
async def predict_ensemble(request: EnsemblePredictRequest):
    """
    Generate ensemble prediction for a match.

    Combines predictions from multiple models using selected method.
    """
    try:
        from core.ensemble import get_ensemble_manager, EnsembleMethod

        manager = get_ensemble_manager()

        # Parse method
        method_map = {
            "weighted_average": EnsembleMethod.WEIGHTED_AVERAGE,
            "confidence_weighted": EnsembleMethod.CONFIDENCE_WEIGHTED,
            "stacking": EnsembleMethod.STACKING,
            "voting": EnsembleMethod.VOTING,
            "bayesian": EnsembleMethod.BAYESIAN,
        }
        method = method_map.get(request.method, EnsembleMethod.WEIGHTED_AVERAGE)

        # Get prediction
        result = manager.predict(request.sport, request.match_data, method)

        if not result:
            raise HTTPException(
                status_code=400,
                detail=f"No models available for {request.sport}"
            )

        return EnsemblePredictResponse(
            sport=result.sport,
            home_probability=result.home_probability,
            away_probability=result.away_probability,
            ensemble_confidence=result.ensemble_confidence,
            method=result.method.value,
            model_weights=result.model_weights,
            reasoning=result.reasoning,
            timestamp=result.timestamp.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ensemble_router.get("/methods")
async def get_ensemble_methods():
    """Get available ensemble methods."""
    return {
        "methods": [
            {
                "name": "weighted_average",
                "description": "Weighted average based on model weights"
            },
            {
                "name": "confidence_weighted",
                "description": "Weights adjusted by prediction confidence"
            },
            {
                "name": "stacking",
                "description": "Meta-learning with sigmoid activation"
            },
            {
                "name": "voting",
                "description": "Majority voting by models"
            },
            {
                "name": "bayesian",
                "description": "Bayesian posterior combination"
            },
        ]
    }


@ensemble_router.post("/update-weights")
async def update_ensemble_weights(
    sport: str,
    performance: Dict[str, float]
):
    """
    Update model weights based on performance.

    Args:
        sport: Sport type
        performance: Dict of model_name -> accuracy (0-1)
    """
    try:
        from core.ensemble import get_ensemble_manager

        manager = get_ensemble_manager()
        manager.update_weights(sport, performance)

        return {
            "status": "success",
            "message": f"Updated weights for {sport}",
            "new_weights": manager.weights.get(sport, {}),
        }
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === MONITORING ENDPOINTS ===

@monitoring_router.get("/stats", response_model=MonitoringStatsResponse)
async def get_monitoring_stats():
    """Get monitoring statistics."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        stats = service.get_stats()

        return MonitoringStatsResponse(
            total_predictions=stats["total_predictions"],
            uptime_seconds=stats["uptime_seconds"],
            by_sport=stats["by_sport"],
            by_model=stats["by_model"],
            predictions_today=stats["predictions_today"],
        )
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/performance")
async def get_performance_metrics(days: int = 7):
    """Get performance metrics for a period."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        metrics = service.get_performance_metrics(days)

        return {
            "period_days": days,
            "total_predictions": metrics.total_predictions,
            "correct_predictions": metrics.correct_predictions,
            "accuracy": metrics.accuracy,
            "avg_confidence": metrics.avg_confidence,
            "avg_edge": metrics.avg_edge,
            "by_sport": metrics.by_sport,
            "by_model": metrics.by_model,
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/alerts", response_model=AlertsResponse)
async def get_alerts():
    """Get performance alerts."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        alerts = service.check_alerts()

        return AlertsResponse(alerts=alerts)
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        predictions = service.get_recent_predictions(limit)

        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.post("/record-result")
async def record_prediction_result(
    prediction_id: str,
    actual_result: str
):
    """Record actual result for a prediction."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        success = service.record_result(prediction_id, actual_result)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction {prediction_id} not found"
            )

        return {
            "status": "success",
            "message": f"Result recorded for {prediction_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === ADMIN ENDPOINTS ===

@admin_router.get("/system-info")
async def get_system_info():
    """Get comprehensive system information."""
    try:
        from core.ensemble import get_ensemble_manager
        from core.monitoring import get_monitoring_service

        ensemble = get_ensemble_manager()
        monitoring = get_monitoring_service()

        return {
            "ensemble": ensemble.get_info(),
            "monitoring": monitoring.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/export-data")
async def export_monitoring_data(format: str = "json"):
    """Export monitoring data."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        data = service.export_data(format)

        return {
            "format": format,
            "data": data,
            "exported_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/save-monitoring")
async def save_monitoring_data():
    """Save monitoring data to file."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        filepath = service.save_to_file()

        return {
            "status": "success",
            "filepath": str(filepath),
        }
    except Exception as e:
        logger.error(f"Error saving monitoring data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/clear-old-data")
async def clear_old_monitoring_data(days: int = 30):
    """Clear monitoring data older than specified days."""
    try:
        from core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        service.clear_old_data(days)

        return {
            "status": "success",
            "message": f"Cleared data older than {days} days",
        }
    except Exception as e:
        logger.error(f"Error clearing old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/health")
async def health_check():
    """Comprehensive health check."""
    try:
        from core.ensemble import get_ensemble_manager
        from core.monitoring import get_monitoring_service

        # Check ensemble
        ensemble = get_ensemble_manager()
        ensemble_healthy = len(ensemble.models) > 0

        # Check monitoring
        monitoring = get_monitoring_service()
        monitoring_healthy = True  # Always healthy if instantiated

        return {
            "status": "healthy" if (ensemble_healthy and monitoring_healthy) else "degraded",
            "components": {
                "ensemble": {
                    "healthy": ensemble_healthy,
                    "registered_sports": list(ensemble.models.keys()),
                },
                "monitoring": {
                    "healthy": monitoring_healthy,
                    "total_predictions": monitoring._stats["total_predictions"],
                },
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def register_routers(app):
    """Register all routers with the app."""
    app.include_router(ensemble_router)
    app.include_router(monitoring_router)
    app.include_router(admin_router)
    logger.info("Registered API routers: ensemble, monitoring, admin")
