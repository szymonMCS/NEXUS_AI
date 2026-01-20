# api/metrics.py
"""
Prometheus metrics for NEXUS AI API.
Provides monitoring for predictions, model performance, and system health.
"""

import time
from typing import Callable
from functools import wraps

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# === HTTP METRICS ===

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# === PREDICTION METRICS ===

PREDICTIONS_TOTAL = Counter(
    "nexus_predictions_total",
    "Total number of predictions made",
    ["sport", "model"]
)

PREDICTION_CONFIDENCE = Histogram(
    "nexus_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["sport"],
    buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
)

VALUE_BETS_FOUND = Counter(
    "nexus_value_bets_found",
    "Number of value bets identified",
    ["sport"]
)

MATCHES_ANALYZED = Counter(
    "nexus_matches_analyzed",
    "Number of matches analyzed",
    ["sport"]
)

# === MODEL METRICS ===

MODEL_ACCURACY = Gauge(
    "nexus_model_accuracy",
    "Current model accuracy",
    ["model", "sport"]
)

MODEL_INFERENCE_TIME = Histogram(
    "nexus_model_inference_seconds",
    "Model inference time in seconds",
    ["model"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
)

# === DATA QUALITY METRICS ===

DATA_QUALITY_SCORE = Gauge(
    "nexus_data_quality_score",
    "Current data quality score (0-1)",
    ["sport", "source"]
)

DATA_FRESHNESS = Gauge(
    "nexus_data_freshness_seconds",
    "Age of data in seconds",
    ["source"]
)

# === BETTING METRICS ===

AVG_EDGE = Gauge(
    "nexus_avg_edge",
    "Average betting edge",
    ["sport"]
)

KELLY_STAKE = Histogram(
    "nexus_kelly_stake_percent",
    "Distribution of Kelly stake recommendations",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0)
)

# === SYSTEM METRICS ===

ACTIVE_ANALYSES = Gauge(
    "nexus_active_analyses",
    "Number of currently running analyses"
)

WEBSOCKET_CONNECTIONS = Gauge(
    "nexus_websocket_connections",
    "Number of active WebSocket connections"
)

# === INFO METRIC ===

APP_INFO = Info(
    "nexus_app",
    "NEXUS AI application information"
)


def setup_metrics(app: FastAPI, version: str = "2.2.0"):
    """
    Setup Prometheus metrics middleware and endpoint for FastAPI.

    Args:
        app: FastAPI application
        version: Application version
    """
    # Set app info
    APP_INFO.info({
        "version": version,
        "mode": "lite",
        "name": "NEXUS AI"
    })

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        """Record HTTP request metrics."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Get endpoint (normalize path parameters)
        path = request.url.path
        if path.startswith("/api/"):
            endpoint = path
        else:
            endpoint = "other"

        # Record metrics
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)

        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )


def record_prediction(sport: str, model: str, confidence: float):
    """Record a prediction metric."""
    PREDICTIONS_TOTAL.labels(sport=sport, model=model).inc()
    PREDICTION_CONFIDENCE.labels(sport=sport).observe(confidence)


def record_value_bet(sport: str, edge: float, stake_percent: float):
    """Record a value bet finding."""
    VALUE_BETS_FOUND.labels(sport=sport).inc()
    KELLY_STAKE.observe(stake_percent)
    AVG_EDGE.labels(sport=sport).set(edge)


def record_analysis(sport: str, matches_count: int, quality_score: float):
    """Record an analysis run."""
    MATCHES_ANALYZED.labels(sport=sport).inc(matches_count)
    DATA_QUALITY_SCORE.labels(sport=sport, source="combined").set(quality_score)


def record_model_performance(model: str, sport: str, accuracy: float, inference_time: float):
    """Record model performance metrics."""
    MODEL_ACCURACY.labels(model=model, sport=sport).set(accuracy)
    MODEL_INFERENCE_TIME.labels(model=model).observe(inference_time)


def set_active_analyses(count: int):
    """Set number of active analyses."""
    ACTIVE_ANALYSES.set(count)


def set_websocket_connections(count: int):
    """Set number of active WebSocket connections."""
    WEBSOCKET_CONNECTIONS.set(count)


def track_inference_time(model: str):
    """Decorator to track model inference time."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            MODEL_INFERENCE_TIME.labels(model=model).observe(duration)
            return result
        return wrapper
    return decorator
