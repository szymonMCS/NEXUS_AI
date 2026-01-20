# core/monitoring.py
"""
Monitoring service for NEXUS AI.
Tracks predictions, performance, and system metrics.
Based on concepts from backend_draft/services/monitoring.py
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    """Log entry for a prediction."""
    prediction_id: str
    sport: str
    match_id: str
    timestamp: datetime
    home_probability: float
    away_probability: float
    confidence: float
    model_name: str
    actual_result: Optional[str] = None  # "home", "away", or None
    was_correct: Optional[bool] = None
    odds_used: Optional[float] = None
    edge: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "sport": self.sport,
            "match_id": self.match_id,
            "timestamp": self.timestamp.isoformat(),
            "home_probability": self.home_probability,
            "away_probability": self.away_probability,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "actual_result": self.actual_result,
            "was_correct": self.was_correct,
            "odds_used": self.odds_used,
            "edge": self.edge,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a period."""
    period_start: datetime
    period_end: datetime
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_edge: float = 0.0
    by_sport: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MonitoringService:
    """
    Service for monitoring predictions and system performance.

    Features:
    - Log all predictions
    - Track accuracy by sport/model
    - Calculate performance metrics
    - Export data for analysis
    - Alert on performance degradation
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize monitoring service.

        Args:
            log_dir: Directory for storing logs
        """
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.predictions: List[PredictionLog] = []
        self.prediction_counter = 0

        # Aggregated stats
        self._stats = {
            "total_predictions": 0,
            "by_sport": defaultdict(lambda: {"total": 0, "correct": 0}),
            "by_model": defaultdict(lambda: {"total": 0, "correct": 0}),
            "daily_predictions": defaultdict(int),
            "hourly_predictions": defaultdict(int),
        }

        # Performance thresholds for alerts
        self.alert_thresholds = {
            "min_accuracy": 0.50,
            "min_confidence": 0.60,
            "max_prediction_latency_ms": 5000,
        }

        self._start_time = datetime.now()

    def log_prediction(
        self,
        sport: str,
        match_id: str,
        home_probability: float,
        away_probability: float,
        confidence: float,
        model_name: str,
        odds_used: Optional[float] = None,
        edge: Optional[float] = None,
    ) -> str:
        """
        Log a prediction.

        Args:
            sport: Sport type
            match_id: Match identifier
            home_probability: Home win probability
            away_probability: Away win probability
            confidence: Model confidence
            model_name: Name of the model
            odds_used: Odds if betting
            edge: Calculated edge

        Returns:
            Prediction ID
        """
        self.prediction_counter += 1
        pred_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.prediction_counter:04d}"

        log_entry = PredictionLog(
            prediction_id=pred_id,
            sport=sport,
            match_id=match_id,
            timestamp=datetime.now(),
            home_probability=home_probability,
            away_probability=away_probability,
            confidence=confidence,
            model_name=model_name,
            odds_used=odds_used,
            edge=edge,
        )

        self.predictions.append(log_entry)

        # Update stats
        self._stats["total_predictions"] += 1
        self._stats["by_sport"][sport]["total"] += 1
        self._stats["by_model"][model_name]["total"] += 1
        self._stats["daily_predictions"][datetime.now().strftime("%Y-%m-%d")] += 1
        self._stats["hourly_predictions"][datetime.now().strftime("%Y-%m-%d %H")] += 1

        logger.debug(f"Logged prediction {pred_id} for {sport}")
        return pred_id

    def record_result(
        self,
        prediction_id: str,
        actual_result: str
    ) -> bool:
        """
        Record actual result for a prediction.

        Args:
            prediction_id: Prediction ID
            actual_result: "home" or "away"

        Returns:
            True if found and updated
        """
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.actual_result = actual_result

                # Determine if correct
                predicted = "home" if pred.home_probability > pred.away_probability else "away"
                pred.was_correct = (predicted == actual_result)

                # Update stats
                if pred.was_correct:
                    self._stats["by_sport"][pred.sport]["correct"] += 1
                    self._stats["by_model"][pred.model_name]["correct"] += 1

                logger.info(f"Recorded result for {prediction_id}: {'correct' if pred.was_correct else 'incorrect'}")
                return True

        logger.warning(f"Prediction not found: {prediction_id}")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds()

        # Calculate accuracies
        sport_accuracy = {}
        for sport, stats in self._stats["by_sport"].items():
            if stats["total"] > 0:
                sport_accuracy[sport] = {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                }

        model_accuracy = {}
        for model, stats in self._stats["by_model"].items():
            if stats["total"] > 0:
                model_accuracy[model] = {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                }

        return {
            "total_predictions": self._stats["total_predictions"],
            "uptime_seconds": uptime,
            "start_time": self._start_time.isoformat(),
            "by_sport": sport_accuracy,
            "by_model": model_accuracy,
            "recent_predictions": len([p for p in self.predictions if (datetime.now() - p.timestamp).days < 1]),
            "predictions_today": self._stats["daily_predictions"].get(datetime.now().strftime("%Y-%m-%d"), 0),
        }

    def get_performance_metrics(
        self,
        days: int = 7
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a period.

        Args:
            days: Number of days to analyze

        Returns:
            PerformanceMetrics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = [p for p in self.predictions if p.timestamp >= cutoff]

        if not recent:
            return PerformanceMetrics(
                period_start=cutoff,
                period_end=datetime.now(),
            )

        settled = [p for p in recent if p.was_correct is not None]

        total = len(recent)
        correct = sum(1 for p in settled if p.was_correct)
        accuracy = correct / len(settled) if settled else 0

        avg_confidence = sum(p.confidence for p in recent) / total if total > 0 else 0
        edges = [p.edge for p in recent if p.edge is not None]
        avg_edge = sum(edges) / len(edges) if edges else 0

        # By sport
        by_sport = defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0})
        for p in recent:
            by_sport[p.sport]["total"] += 1
            if p.was_correct:
                by_sport[p.sport]["correct"] += 1
        for sport in by_sport:
            if by_sport[sport]["total"] > 0:
                by_sport[sport]["accuracy"] = by_sport[sport]["correct"] / by_sport[sport]["total"]

        # By model
        by_model = defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0})
        for p in recent:
            by_model[p.model_name]["total"] += 1
            if p.was_correct:
                by_model[p.model_name]["correct"] += 1
        for model in by_model:
            if by_model[model]["total"] > 0:
                by_model[model]["accuracy"] = by_model[model]["correct"] / by_model[model]["total"]

        return PerformanceMetrics(
            period_start=cutoff,
            period_end=datetime.now(),
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_edge=avg_edge,
            by_sport=dict(by_sport),
            by_model=dict(by_model),
        )

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        metrics = self.get_performance_metrics(days=7)

        # Check overall accuracy
        if metrics.total_predictions >= 10 and metrics.accuracy < self.alert_thresholds["min_accuracy"]:
            alerts.append({
                "type": "low_accuracy",
                "message": f"Overall accuracy ({metrics.accuracy:.1%}) below threshold ({self.alert_thresholds['min_accuracy']:.1%})",
                "severity": "warning",
            })

        # Check per-sport accuracy
        for sport, stats in metrics.by_sport.items():
            if stats["total"] >= 5 and stats["accuracy"] < self.alert_thresholds["min_accuracy"]:
                alerts.append({
                    "type": "low_sport_accuracy",
                    "message": f"{sport} accuracy ({stats['accuracy']:.1%}) below threshold",
                    "severity": "warning",
                    "sport": sport,
                })

        # Check average confidence
        if metrics.avg_confidence < self.alert_thresholds["min_confidence"]:
            alerts.append({
                "type": "low_confidence",
                "message": f"Average confidence ({metrics.avg_confidence:.1%}) is low",
                "severity": "info",
            })

        return alerts

    def export_data(self, format: str = "json") -> Any:
        """
        Export monitoring data.

        Args:
            format: "json" or "csv"

        Returns:
            Exported data
        """
        if format == "json":
            return {
                "predictions": [p.to_dict() for p in self.predictions],
                "stats": self.get_stats(),
                "metrics": self.get_performance_metrics().__dict__,
                "exported_at": datetime.now().isoformat(),
            }
        elif format == "csv":
            # CSV header and rows
            headers = [
                "prediction_id", "sport", "match_id", "timestamp",
                "home_probability", "away_probability", "confidence",
                "model_name", "actual_result", "was_correct", "odds_used", "edge"
            ]
            rows = [",".join(headers)]
            for p in self.predictions:
                row = [
                    p.prediction_id, p.sport, p.match_id,
                    p.timestamp.isoformat(),
                    str(p.home_probability), str(p.away_probability),
                    str(p.confidence), p.model_name,
                    str(p.actual_result or ""), str(p.was_correct or ""),
                    str(p.odds_used or ""), str(p.edge or ""),
                ]
                rows.append(",".join(row))
            return "\n".join(rows)
        else:
            raise ValueError(f"Unknown format: {format}")

    def save_to_file(self, filename: Optional[str] = None):
        """Save monitoring data to file."""
        if not filename:
            filename = f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.log_dir / filename
        data = self.export_data("json")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Monitoring data saved to {filepath}")
        return filepath

    def load_from_file(self, filepath: Path):
        """Load monitoring data from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        for pred_data in data.get("predictions", []):
            pred = PredictionLog(
                prediction_id=pred_data["prediction_id"],
                sport=pred_data["sport"],
                match_id=pred_data["match_id"],
                timestamp=datetime.fromisoformat(pred_data["timestamp"]),
                home_probability=pred_data["home_probability"],
                away_probability=pred_data["away_probability"],
                confidence=pred_data["confidence"],
                model_name=pred_data["model_name"],
                actual_result=pred_data.get("actual_result"),
                was_correct=pred_data.get("was_correct"),
                odds_used=pred_data.get("odds_used"),
                edge=pred_data.get("edge"),
            )
            self.predictions.append(pred)

        logger.info(f"Loaded {len(self.predictions)} predictions from {filepath}")

    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions."""
        recent = sorted(self.predictions, key=lambda p: p.timestamp, reverse=True)[:limit]
        return [p.to_dict() for p in recent]

    def clear_old_data(self, days: int = 30):
        """Clear predictions older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        old_count = len(self.predictions)
        self.predictions = [p for p in self.predictions if p.timestamp >= cutoff]
        removed = old_count - len(self.predictions)
        logger.info(f"Cleared {removed} old predictions (older than {days} days)")


# Singleton instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get or create monitoring service singleton."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service
