"""
ROI tracker for betting performance.

Checkpoint: 3.9
Responsibility: Track return on investment for predictions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from core.ml.tracking.tracked import (
    TrackedPrediction,
    PredictionMarket,
    PredictionOutcome,
)


logger = logging.getLogger(__name__)


@dataclass
class ROISummary:
    """ROI summary statistics."""
    total_bets: int = 0
    total_stake: float = 0.0
    total_returns: float = 0.0
    total_profit_loss: float = 0.0

    winning_bets: int = 0
    losing_bets: int = 0

    @property
    def roi_percentage(self) -> float:
        """ROI as percentage."""
        if self.total_stake == 0:
            return 0.0
        return (self.total_profit_loss / self.total_stake) * 100

    @property
    def win_rate(self) -> float:
        """Win rate as fraction."""
        if self.total_bets == 0:
            return 0.0
        return self.winning_bets / self.total_bets

    @property
    def average_stake(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.total_stake / self.total_bets

    @property
    def average_profit_per_bet(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.total_profit_loss / self.total_bets


@dataclass
class BettingSession:
    """A betting session for grouping bets."""
    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    initial_bankroll: float = 0.0
    current_bankroll: float = 0.0
    predictions: List[str] = field(default_factory=list)  # prediction_ids

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def profit_loss(self) -> float:
        return self.current_bankroll - self.initial_bankroll

    @property
    def roi_percentage(self) -> float:
        if self.initial_bankroll == 0:
            return 0.0
        return (self.profit_loss / self.initial_bankroll) * 100


class ROITracker:
    """
    Tracker ROI (Return on Investment) dla zakładów.

    Funkcje:
    - Śledzenie zwrotów z predykcji
    - Analiza rentowności per model/market
    - Zarządzanie bankrollem
    - Kelly criterion dla optymalnych stawek
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        """
        Initialize ROI tracker.

        Args:
            initial_bankroll: Starting bankroll
        """
        self._initial_bankroll = initial_bankroll
        self._current_bankroll = initial_bankroll

        self._predictions: Dict[str, TrackedPrediction] = {}
        self._sessions: Dict[str, BettingSession] = {}
        self._current_session: Optional[str] = None

    def record_bet(
        self,
        prediction: TrackedPrediction,
        stake: Optional[float] = None,
    ) -> bool:
        """
        Record a bet placement.

        Args:
            prediction: The prediction being bet on
            stake: Stake amount (uses Kelly if None)

        Returns:
            True if bet was recorded
        """
        if stake is None:
            stake = self.calculate_kelly_stake(
                prediction.predicted_value,
                prediction.odds_at_prediction or 2.0,
            )

        if stake > self._current_bankroll:
            logger.warning(f"Insufficient bankroll for stake {stake}")
            return False

        prediction.stake = stake
        self._predictions[prediction.prediction_id] = prediction
        self._current_bankroll -= stake

        if self._current_session:
            self._sessions[self._current_session].predictions.append(prediction.prediction_id)
            self._sessions[self._current_session].current_bankroll = self._current_bankroll

        logger.debug(f"Recorded bet: {stake} on {prediction.prediction_id}")
        return True

    def settle_bet(
        self,
        prediction_id: str,
        outcome: PredictionOutcome,
        actual_odds: Optional[float] = None,
    ) -> float:
        """
        Settle a bet with its outcome.

        Args:
            prediction_id: Prediction ID
            outcome: Bet outcome
            actual_odds: Actual odds (if different from prediction time)

        Returns:
            Profit/loss amount
        """
        pred = self._predictions.get(prediction_id)
        if not pred:
            return 0.0

        odds = actual_odds or pred.odds_at_prediction or 2.0

        if outcome == PredictionOutcome.CORRECT:
            profit = pred.stake * (odds - 1)
            self._current_bankroll += pred.stake + profit
            pred.profit_loss = profit
        elif outcome == PredictionOutcome.INCORRECT:
            pred.profit_loss = -pred.stake
        else:  # VOID
            self._current_bankroll += pred.stake
            pred.profit_loss = 0.0

        pred.outcome = outcome

        if self._current_session:
            self._sessions[self._current_session].current_bankroll = self._current_bankroll

        logger.info(f"Settled bet {prediction_id}: {pred.profit_loss:+.2f}")
        return pred.profit_loss

    def get_roi_summary(
        self,
        model_version: Optional[str] = None,
        market: Optional[PredictionMarket] = None,
        since: Optional[datetime] = None,
    ) -> ROISummary:
        """Get ROI summary with optional filters."""
        summary = ROISummary()

        for pred in self._predictions.values():
            if not pred.is_resolved:
                continue
            if model_version and pred.model_version != model_version:
                continue
            if market and pred.market != market:
                continue
            if since and pred.timestamp < since:
                continue

            summary.total_bets += 1
            summary.total_stake += pred.stake
            summary.total_profit_loss += pred.profit_loss

            if pred.is_correct:
                summary.winning_bets += 1
                summary.total_returns += pred.stake + pred.profit_loss
            else:
                summary.losing_bets += 1

        return summary

    def get_daily_roi(self, days: int = 30) -> Dict[str, float]:
        """Get ROI per day for last N days."""
        daily = defaultdict(lambda: {"stake": 0.0, "profit": 0.0})

        for pred in self._predictions.values():
            if not pred.is_resolved:
                continue

            day_key = pred.timestamp.strftime("%Y-%m-%d")
            daily[day_key]["stake"] += pred.stake
            daily[day_key]["profit"] += pred.profit_loss

        results = {}
        for day, data in sorted(daily.items())[-days:]:
            if data["stake"] > 0:
                results[day] = (data["profit"] / data["stake"]) * 100
            else:
                results[day] = 0.0

        return results

    def calculate_kelly_stake(
        self,
        win_probability: float,
        decimal_odds: float,
        fraction: float = 0.25,  # Use quarter Kelly for safety
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.

        Args:
            win_probability: Estimated probability of winning
            decimal_odds: Decimal odds (e.g., 2.0 for even money)
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Recommended stake amount
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        if decimal_odds <= 1:
            return 0.0

        # Kelly formula: f* = (bp - q) / b
        # where b = decimal_odds - 1, p = win_prob, q = 1 - p
        b = decimal_odds - 1
        p = win_probability
        q = 1 - p

        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0.0  # Negative edge, don't bet

        # Apply fraction and bankroll
        stake = self._current_bankroll * kelly * fraction

        # Cap at reasonable percentage of bankroll
        max_stake = self._current_bankroll * 0.1  # Max 10%
        return min(stake, max_stake)

    def start_session(
        self,
        session_id: str,
        bankroll: Optional[float] = None,
    ) -> BettingSession:
        """Start a new betting session."""
        if bankroll:
            self._current_bankroll = bankroll

        session = BettingSession(
            session_id=session_id,
            start_time=datetime.utcnow(),
            initial_bankroll=self._current_bankroll,
            current_bankroll=self._current_bankroll,
        )

        self._sessions[session_id] = session
        self._current_session = session_id

        logger.info(f"Started session {session_id} with bankroll {self._current_bankroll}")
        return session

    def end_session(self, session_id: Optional[str] = None) -> Optional[BettingSession]:
        """End a betting session."""
        sid = session_id or self._current_session
        if not sid or sid not in self._sessions:
            return None

        session = self._sessions[sid]
        session.end_time = datetime.utcnow()
        session.current_bankroll = self._current_bankroll

        if sid == self._current_session:
            self._current_session = None

        logger.info(f"Ended session {sid}: ROI = {session.roi_percentage:.1f}%")
        return session

    def get_session_summary(self, session_id: str) -> Optional[ROISummary]:
        """Get ROI summary for a specific session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        summary = ROISummary()
        for pred_id in session.predictions:
            pred = self._predictions.get(pred_id)
            if not pred or not pred.is_resolved:
                continue

            summary.total_bets += 1
            summary.total_stake += pred.stake
            summary.total_profit_loss += pred.profit_loss

            if pred.is_correct:
                summary.winning_bets += 1

        return summary

    @property
    def bankroll(self) -> float:
        """Current bankroll."""
        return self._current_bankroll

    @property
    def total_profit_loss(self) -> float:
        """Total P/L from initial bankroll."""
        return self._current_bankroll - self._initial_bankroll

    @property
    def overall_roi(self) -> float:
        """Overall ROI percentage."""
        return (self.total_profit_loss / self._initial_bankroll) * 100

    def reset_bankroll(self, amount: float) -> None:
        """Reset bankroll to specified amount."""
        self._initial_bankroll = amount
        self._current_bankroll = amount
        logger.info(f"Bankroll reset to {amount}")
