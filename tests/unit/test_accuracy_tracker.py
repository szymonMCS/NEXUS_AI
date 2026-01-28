"""
Tests for accuracy and ROI trackers.

Checkpoint: 3.13
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import uuid

from core.ml.tracking import (
    TrackedPrediction,
    PredictionMarket,
    PredictionOutcome,
    PredictionSummary,
    AccuracyTracker,
    ROITracker,
    ROISummary,
)


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def accuracy_tracker(temp_storage):
    return AccuracyTracker(storage_path=temp_storage)


@pytest.fixture
def roi_tracker():
    return ROITracker(initial_bankroll=1000.0)


@pytest.fixture
def sample_predictions():
    preds = []
    for i in range(10):
        pred = TrackedPrediction(
            prediction_id=f"pred-{i}",
            match_id=f"match-{i}",
            market=PredictionMarket.OVER_UNDER_25,
            predicted_value=0.6 + i * 0.02,
            predicted_outcome="over" if i % 2 == 0 else "under",
            confidence=0.5 + i * 0.05,
            model_version="v1.0.0",
            odds_at_prediction=1.90,
            stake=10.0,
        )
        preds.append(pred)
    return preds


class TestTrackedPrediction:
    def test_create(self):
        pred = TrackedPrediction(
            prediction_id="p1",
            match_id="m1",
            market=PredictionMarket.OVER_UNDER_25,
            predicted_value=0.65,
            predicted_outcome="over",
            confidence=0.7,
        )
        assert not pred.is_resolved
        assert pred.was_confident

    def test_resolve_correct(self):
        pred = TrackedPrediction(
            prediction_id="p1",
            match_id="m1",
            market=PredictionMarket.OVER_UNDER_25,
            predicted_outcome="over",
            odds_at_prediction=1.90,
            stake=10.0,
        )
        outcome = pred.resolve(3.0, "over")

        assert outcome == PredictionOutcome.CORRECT
        assert pred.is_correct
        assert pred.profit_loss == 9.0  # 10 * 0.9

    def test_resolve_incorrect(self):
        pred = TrackedPrediction(
            prediction_id="p1",
            match_id="m1",
            market=PredictionMarket.OVER_UNDER_25,
            predicted_outcome="over",
            stake=10.0,
        )
        outcome = pred.resolve(2.0, "under")

        assert outcome == PredictionOutcome.INCORRECT
        assert pred.profit_loss == -10.0


class TestAccuracyTracker:
    def test_track_prediction(self, accuracy_tracker, sample_predictions):
        accuracy_tracker.track(sample_predictions[0])
        pred = accuracy_tracker.get_prediction("pred-0")
        assert pred is not None

    def test_resolve_prediction(self, accuracy_tracker, sample_predictions):
        accuracy_tracker.track(sample_predictions[0])
        outcome = accuracy_tracker.resolve("pred-0", 3.0, "over")

        assert outcome == PredictionOutcome.CORRECT

    def test_resolve_match(self, accuracy_tracker, sample_predictions):
        for pred in sample_predictions[:3]:
            pred.match_id = "match-same"
            accuracy_tracker.track(pred)

        resolved = accuracy_tracker.resolve_match("match-same", 2, 1)
        assert resolved == 3

    def test_get_summary(self, accuracy_tracker, sample_predictions):
        for i, pred in enumerate(sample_predictions):
            accuracy_tracker.track(pred)
            accuracy_tracker.resolve(
                pred.prediction_id,
                3.0 if i % 2 == 0 else 2.0,
                "over" if i % 2 == 0 else "under",
            )

        summary = accuracy_tracker.get_summary()
        assert summary.total_predictions == 10
        assert summary.resolved_predictions == 10

    def test_accuracy_by_confidence(self, accuracy_tracker, sample_predictions):
        for i, pred in enumerate(sample_predictions):
            accuracy_tracker.track(pred)
            accuracy_tracker.resolve(pred.prediction_id, 3.0, "over")

        by_conf = accuracy_tracker.get_accuracy_by_confidence()
        assert len(by_conf) > 0

    def test_model_comparison(self, accuracy_tracker):
        for i in range(5):
            pred = TrackedPrediction(
                prediction_id=f"p{i}",
                match_id=f"m{i}",
                market=PredictionMarket.OVER_UNDER_25,
                predicted_outcome="over",
                model_version="v1" if i < 3 else "v2",
            )
            accuracy_tracker.track(pred)
            accuracy_tracker.resolve(pred.prediction_id, 3.0, "over")

        comparison = accuracy_tracker.get_model_comparison()
        assert "v1" in comparison
        assert "v2" in comparison

    def test_persistence(self, temp_storage):
        tracker1 = AccuracyTracker(storage_path=temp_storage)
        pred = TrackedPrediction(
            prediction_id="test",
            match_id="m1",
            market=PredictionMarket.OVER_UNDER_25,
        )
        tracker1.track(pred)

        tracker2 = AccuracyTracker(storage_path=temp_storage)
        assert tracker2.get_prediction("test") is not None


class TestROITracker:
    def test_initial_bankroll(self, roi_tracker):
        assert roi_tracker.bankroll == 1000.0

    def test_record_bet(self, roi_tracker, sample_predictions):
        pred = sample_predictions[0]
        assert roi_tracker.record_bet(pred, stake=50.0)
        assert roi_tracker.bankroll == 950.0

    def test_settle_bet_win(self, roi_tracker, sample_predictions):
        pred = sample_predictions[0]
        pred.odds_at_prediction = 2.0
        roi_tracker.record_bet(pred, stake=50.0)

        profit = roi_tracker.settle_bet(pred.prediction_id, PredictionOutcome.CORRECT)

        assert profit == 50.0  # 50 * (2.0 - 1)
        assert roi_tracker.bankroll == 1050.0

    def test_settle_bet_loss(self, roi_tracker, sample_predictions):
        pred = sample_predictions[0]
        roi_tracker.record_bet(pred, stake=50.0)

        profit = roi_tracker.settle_bet(pred.prediction_id, PredictionOutcome.INCORRECT)

        assert profit == -50.0
        assert roi_tracker.bankroll == 950.0

    def test_kelly_stake(self, roi_tracker):
        stake = roi_tracker.calculate_kelly_stake(
            win_probability=0.55,
            decimal_odds=2.0,
        )
        assert stake > 0
        assert stake < roi_tracker.bankroll * 0.1

    def test_kelly_negative_edge(self, roi_tracker):
        stake = roi_tracker.calculate_kelly_stake(
            win_probability=0.40,
            decimal_odds=2.0,
        )
        assert stake == 0  # Don't bet with negative edge

    def test_session_management(self, roi_tracker, sample_predictions):
        session = roi_tracker.start_session("session-1")
        assert session.is_active

        pred = sample_predictions[0]
        roi_tracker.record_bet(pred, stake=50.0)

        session = roi_tracker.end_session()
        assert not session.is_active
        assert len(session.predictions) == 1

    def test_roi_summary(self, roi_tracker, sample_predictions):
        for i, pred in enumerate(sample_predictions[:5]):
            pred.odds_at_prediction = 2.0
            roi_tracker.record_bet(pred, stake=20.0)
            roi_tracker.settle_bet(
                pred.prediction_id,
                PredictionOutcome.CORRECT if i % 2 == 0 else PredictionOutcome.INCORRECT,
            )

        summary = roi_tracker.get_roi_summary()
        assert summary.total_bets == 5
        assert summary.total_stake == 100.0

    def test_daily_roi(self, roi_tracker, sample_predictions):
        pred = sample_predictions[0]
        roi_tracker.record_bet(pred, stake=10.0)
        roi_tracker.settle_bet(pred.prediction_id, PredictionOutcome.CORRECT)

        daily = roi_tracker.get_daily_roi(days=7)
        assert len(daily) > 0


class TestPredictionSummary:
    def test_accuracy(self):
        summary = PredictionSummary(
            total_predictions=100,
            resolved_predictions=80,
            correct_predictions=50,
            incorrect_predictions=30,
        )
        assert summary.accuracy == 50 / 80

    def test_roi(self):
        summary = PredictionSummary(
            total_stake=1000.0,
            total_profit_loss=150.0,
        )
        assert summary.roi == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
