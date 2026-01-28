"""
Accuracy tracker for predictions.

Checkpoint: 3.8
Responsibility: Track and analyze prediction accuracy.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from core.ml.tracking.tracked import (
    TrackedPrediction,
    PredictionMarket,
    PredictionOutcome,
    PredictionSummary,
)


logger = logging.getLogger(__name__)


class AccuracyTracker:
    """
    Tracker dokładności predykcji.

    Funkcje:
    - Śledzenie predykcji i ich wyników
    - Obliczanie metryk dokładności
    - Analiza per model, per market
    - Wykrywanie trendów
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize tracker.

        Args:
            storage_path: Path for persisting predictions
        """
        self._storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

        self._predictions: Dict[str, TrackedPrediction] = {}
        self._by_match: Dict[str, List[str]] = defaultdict(list)  # match_id -> prediction_ids
        self._by_model: Dict[str, List[str]] = defaultdict(list)  # model_version -> prediction_ids
        self._by_market: Dict[PredictionMarket, List[str]] = defaultdict(list)

        # Load existing data
        self._load()

    def track(self, prediction: TrackedPrediction) -> None:
        """
        Add a prediction to track.

        Args:
            prediction: Prediction to track
        """
        self._predictions[prediction.prediction_id] = prediction
        self._by_match[prediction.match_id].append(prediction.prediction_id)
        self._by_model[prediction.model_version].append(prediction.prediction_id)
        self._by_market[prediction.market].append(prediction.prediction_id)

        self._save()
        logger.debug(f"Tracking prediction {prediction.prediction_id}")

    def resolve(
        self,
        prediction_id: str,
        actual_value: float,
        actual_outcome: str,
    ) -> Optional[PredictionOutcome]:
        """
        Resolve a prediction with actual result.

        Args:
            prediction_id: ID of prediction to resolve
            actual_value: Actual numeric result
            actual_outcome: Actual outcome

        Returns:
            PredictionOutcome or None if not found
        """
        pred = self._predictions.get(prediction_id)
        if not pred:
            return None

        outcome = pred.resolve(actual_value, actual_outcome)
        self._save()

        logger.info(f"Resolved {prediction_id}: {outcome.value}")
        return outcome

    def resolve_match(
        self,
        match_id: str,
        home_goals: int,
        away_goals: int,
    ) -> int:
        """
        Resolve all predictions for a match.

        Args:
            match_id: Match ID
            home_goals: Actual home goals
            away_goals: Actual away goals

        Returns:
            Number of predictions resolved
        """
        pred_ids = self._by_match.get(match_id, [])
        resolved = 0

        total_goals = home_goals + away_goals
        margin = home_goals - away_goals

        for pred_id in pred_ids:
            pred = self._predictions.get(pred_id)
            if not pred or pred.is_resolved:
                continue

            # Determine actual outcome based on market
            actual_value = 0.0
            actual_outcome = ""

            if pred.market in [PredictionMarket.OVER_UNDER_15, PredictionMarket.OVER_UNDER_25, PredictionMarket.OVER_UNDER_35]:
                actual_value = float(total_goals)
                threshold = float(pred.market.value.split("_")[-1])
                actual_outcome = "over" if total_goals > threshold else "under"

            elif pred.market == PredictionMarket.MATCH_WINNER:
                actual_value = float(margin)
                if margin > 0:
                    actual_outcome = "home"
                elif margin < 0:
                    actual_outcome = "away"
                else:
                    actual_outcome = "draw"

            elif pred.market == PredictionMarket.HANDICAP:
                actual_value = float(margin)
                # Handicap outcome depends on the line in predicted_outcome
                actual_outcome = f"margin_{margin}"

            elif pred.market == PredictionMarket.BTTS:
                actual_value = 1.0 if (home_goals > 0 and away_goals > 0) else 0.0
                actual_outcome = "yes" if actual_value == 1.0 else "no"

            pred.resolve(actual_value, actual_outcome)
            resolved += 1

        self._save()
        logger.info(f"Resolved {resolved} predictions for match {match_id}")
        return resolved

    def get_summary(
        self,
        model_version: Optional[str] = None,
        market: Optional[PredictionMarket] = None,
        since: Optional[datetime] = None,
        min_confidence: float = 0.0,
    ) -> PredictionSummary:
        """
        Get accuracy summary with filters.

        Args:
            model_version: Filter by model
            market: Filter by market type
            since: Only include predictions after this time
            min_confidence: Minimum confidence threshold

        Returns:
            PredictionSummary
        """
        summary = PredictionSummary()

        for pred in self._predictions.values():
            # Apply filters
            if model_version and pred.model_version != model_version:
                continue
            if market and pred.market != market:
                continue
            if since and pred.timestamp < since:
                continue
            if pred.confidence < min_confidence:
                continue

            summary.total_predictions += 1
            summary.total_stake += pred.stake

            if pred.is_resolved:
                summary.resolved_predictions += 1
                summary.total_profit_loss += pred.profit_loss

                if pred.is_correct:
                    summary.correct_predictions += 1
                else:
                    summary.incorrect_predictions += 1

        return summary

    def get_accuracy_by_confidence(
        self,
        bins: List[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy broken down by confidence levels.

        Args:
            bins: Confidence bin boundaries (default: [0.5, 0.6, 0.7, 0.8, 0.9])

        Returns:
            Dict mapping confidence range to accuracy stats
        """
        bins = bins or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        results = {}

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            key = f"{low:.1f}-{high:.1f}"

            matching = [
                p for p in self._predictions.values()
                if low <= p.confidence < high and p.is_resolved
            ]

            if matching:
                correct = sum(1 for p in matching if p.is_correct)
                results[key] = {
                    "count": len(matching),
                    "accuracy": correct / len(matching),
                    "avg_confidence": sum(p.confidence for p in matching) / len(matching),
                }
            else:
                results[key] = {"count": 0, "accuracy": 0.0, "avg_confidence": 0.0}

        return results

    def get_model_comparison(self) -> Dict[str, PredictionSummary]:
        """Get accuracy summary per model version."""
        results = {}
        for model_version in self._by_model.keys():
            results[model_version] = self.get_summary(model_version=model_version)
        return results

    def get_market_comparison(self) -> Dict[str, PredictionSummary]:
        """Get accuracy summary per market type."""
        results = {}
        for market in self._by_market.keys():
            results[market.value] = self.get_summary(market=market)
        return results

    def get_recent_accuracy(
        self,
        days: int = 7,
        model_version: Optional[str] = None,
    ) -> float:
        """Get accuracy for recent predictions."""
        since = datetime.utcnow() - timedelta(days=days)
        summary = self.get_summary(model_version=model_version, since=since)
        return summary.accuracy

    def get_prediction(self, prediction_id: str) -> Optional[TrackedPrediction]:
        """Get a specific prediction."""
        return self._predictions.get(prediction_id)

    def get_pending_predictions(self) -> List[TrackedPrediction]:
        """Get all pending (unresolved) predictions."""
        return [p for p in self._predictions.values() if not p.is_resolved]

    def get_predictions_for_match(self, match_id: str) -> List[TrackedPrediction]:
        """Get all predictions for a match."""
        pred_ids = self._by_match.get(match_id, [])
        return [self._predictions[pid] for pid in pred_ids if pid in self._predictions]

    def calibration_error(
        self,
        n_bins: int = 10,
        model_version: Optional[str] = None,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Lower is better - means predicted probabilities match actual outcomes.
        """
        resolved = [
            p for p in self._predictions.values()
            if p.is_resolved and (model_version is None or p.model_version == model_version)
        ]

        if not resolved:
            return 0.0

        bin_size = 1.0 / n_bins
        total_error = 0.0
        total_samples = 0

        for i in range(n_bins):
            low = i * bin_size
            high = (i + 1) * bin_size

            bin_preds = [p for p in resolved if low <= p.predicted_value < high]
            if not bin_preds:
                continue

            avg_confidence = sum(p.predicted_value for p in bin_preds) / len(bin_preds)
            actual_accuracy = sum(1 for p in bin_preds if p.is_correct) / len(bin_preds)

            total_error += len(bin_preds) * abs(avg_confidence - actual_accuracy)
            total_samples += len(bin_preds)

        return total_error / total_samples if total_samples > 0 else 0.0

    def _save(self) -> None:
        """Save predictions to disk."""
        if not self._storage_path:
            return

        data = {
            "predictions": {
                pid: pred.to_dict()
                for pid, pred in self._predictions.items()
            }
        }

        file_path = self._storage_path / "accuracy_tracker.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load predictions from disk."""
        if not self._storage_path:
            return

        file_path = self._storage_path / "accuracy_tracker.json"
        if not file_path.exists():
            return

        try:
            with open(file_path) as f:
                data = json.load(f)

            for pid, pred_data in data.get("predictions", {}).items():
                pred = TrackedPrediction.from_dict(pred_data)
                self._predictions[pid] = pred
                self._by_match[pred.match_id].append(pid)
                self._by_model[pred.model_version].append(pid)
                self._by_market[pred.market].append(pid)

            logger.info(f"Loaded {len(self._predictions)} tracked predictions")

        except Exception as e:
            logger.error(f"Error loading accuracy tracker: {e}")
