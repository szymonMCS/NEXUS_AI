# api/routers.py
"""
API Routers for NEXUS AI.
Organized endpoints for predictions, ensemble, monitoring, and admin.
Based on concepts from backend_draft/api/routers.py

Checkpoint: 4.5 - Added ML predictions router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import uuid

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# === PYDANTIC MODELS ===

# --- ML Prediction Models ---

class MLPredictRequest(BaseModel):
    """Request for ML prediction."""
    match_id: str
    sport: str = "football"
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    league: Optional[str] = None
    match_date: Optional[str] = None
    include_recommendations: bool = True
    market_odds: Optional[Dict[str, float]] = None


class MLPredictResponse(BaseModel):
    """Response for ML prediction."""
    match_id: str
    prediction_id: str
    status: str
    goals: Optional[Dict[str, Any]] = None
    handicap: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    data_quality: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    timestamp: str


class MLBatchPredictRequest(BaseModel):
    """Request for batch ML prediction."""
    matches: List[MLPredictRequest]
    include_recommendations: bool = True


class MLBatchPredictResponse(BaseModel):
    """Response for batch ML prediction."""
    predictions: List[MLPredictResponse]
    success_count: int
    failure_count: int
    value_bet_count: int
    total_processing_time_ms: float
    timestamp: str


class MLModelStatusResponse(BaseModel):
    """Response for ML model status."""
    models: Dict[str, Dict[str, Any]]
    registry_path: Optional[str]
    total_predictions: int
    last_trained: Optional[str]


class MLTrainRequest(BaseModel):
    """Request for model training."""
    model_name: str
    force: bool = False


class MLTrainResponse(BaseModel):
    """Response for model training."""
    success: bool
    model_name: str
    version: Optional[str]
    metrics: Optional[Dict[str, float]]
    error_message: Optional[str]


# --- Ensemble Models ---

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

# ML Predictions router
ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

# Ensemble router
ensemble_router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])

# Monitoring router
monitoring_router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Admin router
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


# === ML PREDICTION ENDPOINTS ===

@ml_router.post("/predict", response_model=MLPredictResponse)
async def ml_predict(request: MLPredictRequest):
    """
    Generate ML prediction for a match.

    Uses Poisson model for goals/over-under and GBM for handicap.
    """
    try:
        from core.data.schemas import MatchData, DataQuality, Sport
        from core.data.repository import UnifiedDataRepository
        from core.ml.service import MLPredictionService

        # Create match data from request
        sport = Sport(request.sport) if request.sport in [s.value for s in Sport] else Sport.FOOTBALL

        match = MatchData(
            match_id=request.match_id,
            sport=sport,
            league=request.league or "",
            home_team=request.home_team_name,
            away_team=request.away_team_name,
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            quality=DataQuality(completeness=0.5, freshness=1.0, sources_count=1),
        )

        # Get prediction service
        repository = UnifiedDataRepository()
        service = MLPredictionService(repository=repository)

        # Generate prediction
        result = service.predict(
            match=match,
            include_recommendations=request.include_recommendations,
            market_odds=request.market_odds,
        )

        # Convert to response
        response_data = result.to_dict()
        return MLPredictResponse(
            match_id=response_data["match_id"],
            prediction_id=response_data["prediction_id"],
            status=response_data["status"],
            goals=response_data.get("goals"),
            handicap=response_data.get("handicap"),
            recommendations=response_data.get("recommendations"),
            data_quality=response_data.get("data_quality"),
            processing_time_ms=response_data["processing_time_ms"],
            timestamp=response_data["timestamp"],
        )

    except Exception as e:
        logger.exception(f"ML prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/predict/batch", response_model=MLBatchPredictResponse)
async def ml_predict_batch(request: MLBatchPredictRequest):
    """
    Generate ML predictions for multiple matches.
    """
    try:
        from core.data.schemas import MatchData, DataQuality, Sport
        from core.data.repository import UnifiedDataRepository
        from core.ml.service import MLPredictionService

        repository = UnifiedDataRepository()
        service = MLPredictionService(repository=repository)

        # Convert requests to match data
        matches = []
        for req in request.matches:
            sport = Sport(req.sport) if req.sport in [s.value for s in Sport] else Sport.FOOTBALL
            match = MatchData(
                match_id=req.match_id,
                sport=sport,
                league=req.league or "",
                home_team=req.home_team_name,
                away_team=req.away_team_name,
                home_team_id=req.home_team_id,
                away_team_id=req.away_team_id,
                quality=DataQuality(completeness=0.5, freshness=1.0, sources_count=1),
            )
            matches.append(match)

        # Generate batch predictions
        batch_result = service.predict_batch(
            matches=matches,
            include_recommendations=request.include_recommendations,
        )

        # Convert to responses
        predictions = []
        for result in batch_result.predictions:
            response_data = result.to_dict()
            predictions.append(MLPredictResponse(
                match_id=response_data["match_id"],
                prediction_id=response_data["prediction_id"],
                status=response_data["status"],
                goals=response_data.get("goals"),
                handicap=response_data.get("handicap"),
                recommendations=response_data.get("recommendations"),
                data_quality=response_data.get("data_quality"),
                processing_time_ms=response_data["processing_time_ms"],
                timestamp=response_data["timestamp"],
            ))

        return MLBatchPredictResponse(
            predictions=predictions,
            success_count=batch_result.success_count,
            failure_count=batch_result.failure_count,
            value_bet_count=batch_result.value_bet_count,
            total_processing_time_ms=batch_result.total_processing_time_ms,
            timestamp=batch_result.timestamp.isoformat(),
        )

    except Exception as e:
        logger.exception(f"ML batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/models/status", response_model=MLModelStatusResponse)
async def get_ml_models_status():
    """
    Get status of ML models.
    """
    try:
        from core.ml.registry import ModelRegistry
        from pathlib import Path

        # Try to load registry
        registry_path = Path("data/ml/registry")
        registry = ModelRegistry(storage_path=registry_path) if registry_path.exists() else None

        models_info = {}
        total_predictions = 0
        last_trained = None

        if registry:
            for model_name in ["poisson_goals", "gbm_handicap"]:
                versions = registry.get_versions(model_name)
                active = registry.get_active_version(model_name)
                models_info[model_name] = {
                    "versions_count": len(versions),
                    "active_version": active.version if active else None,
                    "active_metrics": active.metrics if active else {},
                    "last_updated": active.created_at.isoformat() if active else None,
                }
                if active and (last_trained is None or active.created_at > datetime.fromisoformat(last_trained)):
                    last_trained = active.created_at.isoformat()
        else:
            models_info = {
                "poisson_goals": {"status": "not_initialized", "versions_count": 0},
                "gbm_handicap": {"status": "not_initialized", "versions_count": 0},
            }

        return MLModelStatusResponse(
            models=models_info,
            registry_path=str(registry_path) if registry else None,
            total_predictions=total_predictions,
            last_trained=last_trained,
        )

    except Exception as e:
        logger.exception(f"Error getting ML models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/models/train", response_model=MLTrainResponse)
async def train_ml_model(request: MLTrainRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain an ML model.

    Available models: poisson_goals, gbm_handicap
    """
    try:
        from core.ml.registry import ModelRegistry
        from core.ml.training import OnlineTrainer, TrainingConfig
        from core.ml.models import GoalsModel, HandicapModel
        from pathlib import Path

        registry_path = Path("data/ml/registry")
        registry_path.mkdir(parents=True, exist_ok=True)
        registry = ModelRegistry(storage_path=registry_path)

        config = TrainingConfig(min_samples=10)
        trainer = OnlineTrainer(registry, config)

        # Check buffer status
        buffer_status = trainer.get_buffer_status()
        examples_count = buffer_status.get(request.model_name, 0)

        if examples_count < config.min_samples and not request.force:
            return MLTrainResponse(
                success=False,
                model_name=request.model_name,
                version=None,
                metrics=None,
                error_message=f"Not enough training examples ({examples_count}/{config.min_samples}). Use force=True to train anyway.",
            )

        # Select model
        if request.model_name == "poisson_goals":
            model = GoalsModel()
        elif request.model_name == "gbm_handicap":
            model = HandicapModel()
        else:
            return MLTrainResponse(
                success=False,
                model_name=request.model_name,
                version=None,
                metrics=None,
                error_message=f"Unknown model: {request.model_name}",
            )

        # Train
        result = trainer.train_incremental(model, force=request.force)

        if result.success:
            # Register new version
            version = registry.register(
                model=model,
                metrics=result.metrics,
                model_name=request.model_name,
            )
            return MLTrainResponse(
                success=True,
                model_name=request.model_name,
                version=version.version,
                metrics=result.metrics,
                error_message=None,
            )
        else:
            return MLTrainResponse(
                success=False,
                model_name=request.model_name,
                version=None,
                metrics=None,
                error_message=result.error_message,
            )

    except Exception as e:
        logger.exception(f"ML training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/recommendations")
async def get_ml_recommendations(
    sport: str = "football",
    min_confidence: float = 0.55,
    min_edge: float = 0.02,
    limit: int = 10,
):
    """
    Get top ML-based betting recommendations.

    Filters by confidence and edge thresholds.
    """
    try:
        from core.ml.tracking import AccuracyTracker
        from pathlib import Path

        tracker_path = Path("data/ml/tracking")
        tracker = AccuracyTracker(storage_path=tracker_path) if tracker_path.exists() else None

        if not tracker:
            return {
                "recommendations": [],
                "message": "No tracking data available",
            }

        # Get recent predictions with good performance
        summary = tracker.get_summary()

        return {
            "recommendations": [],  # Would come from live predictions
            "summary": {
                "total_predictions": summary.total_predictions,
                "accuracy": summary.accuracy,
                "roi": summary.roi,
            },
            "filters": {
                "sport": sport,
                "min_confidence": min_confidence,
                "min_edge": min_edge,
            },
        }

    except Exception as e:
        logger.exception(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/tracking/summary")
async def get_tracking_summary(
    model_version: Optional[str] = None,
    market: Optional[str] = None,
    days: int = 30,
):
    """
    Get prediction tracking summary.
    """
    try:
        from core.ml.tracking import AccuracyTracker
        from pathlib import Path

        tracker_path = Path("data/ml/tracking")
        if not tracker_path.exists():
            return {
                "status": "no_data",
                "message": "Tracking not initialized",
            }

        tracker = AccuracyTracker(storage_path=tracker_path)
        summary = tracker.get_summary(model_version=model_version)

        return {
            "total_predictions": summary.total_predictions,
            "resolved_predictions": summary.resolved_predictions,
            "correct_predictions": summary.correct_predictions,
            "accuracy": summary.accuracy,
            "roi": summary.roi,
            "calibration_error": tracker.calibration_error() if summary.resolved_predictions > 0 else None,
        }

    except Exception as e:
        logger.exception(f"Error getting tracking summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/tracking/roi")
async def get_roi_tracking(days: int = 30):
    """
    Get ROI tracking data.
    """
    try:
        from core.ml.tracking import ROITracker
        from pathlib import Path

        tracker_path = Path("data/ml/roi")
        if not tracker_path.exists():
            return {
                "status": "no_data",
                "message": "ROI tracking not initialized",
            }

        # ROI tracker is in-memory, so we return placeholder
        return {
            "initial_bankroll": 1000.0,
            "current_bankroll": 1000.0,
            "total_bets": 0,
            "roi": 0.0,
            "daily_roi": [],
        }

    except Exception as e:
        logger.exception(f"Error getting ROI tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    app.include_router(ml_router)
    app.include_router(ensemble_router)
    app.include_router(monitoring_router)
    app.include_router(admin_router)
    logger.info("Registered API routers: ml, ensemble, monitoring, admin")
