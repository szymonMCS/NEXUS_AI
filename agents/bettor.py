# agents/bettor.py
"""
Bettor Agent - Executes betting decisions.
Handles bet placement, tracking, and result monitoring.

Note: This is an optional agent. Actual bet placement requires
integration with bookmaker APIs which may have legal restrictions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

from config.settings import settings
from core.state import NexusState, Match, BetDecision, add_message

logger = logging.getLogger(__name__)


class BetStatus(Enum):
    """Status of a placed bet."""
    PENDING = "pending"          # Bet created, not placed
    PLACED = "placed"            # Successfully placed with bookmaker
    CONFIRMED = "confirmed"      # Confirmed by bookmaker
    WON = "won"                  # Bet won
    LOST = "lost"                # Bet lost
    VOID = "void"                # Bet cancelled/void
    PARTIAL = "partial"          # Partial result (e.g., push)
    FAILED = "failed"            # Failed to place
    CASHED_OUT = "cashed_out"    # Early cash out


@dataclass
class PlacedBet:
    """Record of a placed bet."""
    bet_id: str
    match_id: str
    match_name: str
    sport: str
    selection: str              # "home", "away", "over", "under"
    bet_type: str               # "moneyline", "spread", "total"
    stake: float
    odds: float
    bookmaker: str
    status: BetStatus = BetStatus.PENDING
    placed_at: datetime = field(default_factory=datetime.now)
    settled_at: Optional[datetime] = None
    profit_loss: float = 0.0
    notes: str = ""

    # Prediction info
    predicted_probability: float = 0.0
    edge: float = 0.0
    confidence: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bet_id": self.bet_id,
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "selection": self.selection,
            "bet_type": self.bet_type,
            "stake": self.stake,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "status": self.status.value,
            "placed_at": self.placed_at.isoformat(),
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "profit_loss": self.profit_loss,
            "predicted_probability": self.predicted_probability,
            "edge": self.edge,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "notes": self.notes
        }

    @property
    def potential_profit(self) -> float:
        """Calculate potential profit if bet wins."""
        return self.stake * (self.odds - 1)

    @property
    def expected_value(self) -> float:
        """Calculate expected value."""
        return (self.predicted_probability * self.odds - 1) * self.stake


@dataclass
class BettingSession:
    """Represents a betting session."""
    session_id: str
    date: str
    sport: str
    starting_bankroll: float
    current_bankroll: float
    bets: List[PlacedBet] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_staked(self) -> float:
        """Total amount staked."""
        return sum(b.stake for b in self.bets if b.status != BetStatus.FAILED)

    @property
    def total_profit_loss(self) -> float:
        """Total profit/loss from settled bets."""
        return sum(b.profit_loss for b in self.bets if b.status in [BetStatus.WON, BetStatus.LOST])

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        settled = [b for b in self.bets if b.status in [BetStatus.WON, BetStatus.LOST]]
        if not settled:
            return 0.0
        wins = sum(1 for b in settled if b.status == BetStatus.WON)
        return wins / len(settled)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "date": self.date,
            "sport": self.sport,
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": self.current_bankroll,
            "bet_count": len(self.bets),
            "total_staked": self.total_staked,
            "total_profit_loss": self.total_profit_loss,
            "win_rate": self.win_rate,
            "bets": [b.to_dict() for b in self.bets]
        }


class BettorAgent:
    """
    Bettor Agent executes betting decisions.

    Features:
    - Bet placement (simulated or via bookmaker APIs)
    - Bet tracking and history
    - Result monitoring
    - Profit/loss calculation
    - Session management

    Note: By default operates in simulation mode.
    Real betting requires bookmaker API integration.
    """

    def __init__(
        self,
        simulation_mode: bool = True,
        default_bookmaker: str = "simulation"
    ):
        """
        Initialize Bettor Agent.

        Args:
            simulation_mode: If True, bets are simulated (not actually placed)
            default_bookmaker: Default bookmaker for bet placement
        """
        self.simulation_mode = simulation_mode
        self.default_bookmaker = default_bookmaker

        # Active session
        self.session: Optional[BettingSession] = None

        # Bet history (all sessions)
        self.bet_history: List[PlacedBet] = []

        # Bookmaker API clients (to be implemented)
        self.bookmaker_clients: Dict[str, Any] = {}

        # Bet counter for ID generation
        self._bet_counter = 0

    async def process(self, state: NexusState) -> NexusState:
        """
        Process approved bets and place them.

        Args:
            state: Current workflow state with approved bets

        Returns:
            Updated state with placed bets info
        """
        state.current_agent = "bettor"

        if not state.approved_bets:
            state = add_message(state, "bettor", "No approved bets to place")
            return state

        # Start or continue session
        if not self.session:
            self.session = BettingSession(
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                date=state.date,
                sport=state.sport,
                starting_bankroll=state.current_bankroll,
                current_bankroll=state.current_bankroll
            )

        state = add_message(
            state,
            "bettor",
            f"Processing {len(state.approved_bets)} approved bets "
            f"({'SIMULATION' if self.simulation_mode else 'LIVE'})"
        )

        placed_bets = []

        for match in state.approved_bets:
            if not match.value_bet:
                continue

            # Create bet record
            bet = self._create_bet(match, state.current_bankroll)

            # Place bet (simulated or real)
            success = await self._place_bet(bet)

            if success:
                placed_bets.append(bet)
                self.session.bets.append(bet)
                self.bet_history.append(bet)

                state = add_message(
                    state,
                    "bettor",
                    f"PLACED: {bet.match_name} | {bet.selection.upper()} @ {bet.odds:.2f} | "
                    f"Stake: ${bet.stake:.2f} | Edge: {bet.edge:.1%}"
                )
            else:
                bet.status = BetStatus.FAILED
                state = add_message(
                    state,
                    "bettor",
                    f"FAILED: {bet.match_name} - Could not place bet"
                )

        # Update session bankroll
        total_staked = sum(b.stake for b in placed_bets)
        self.session.current_bankroll = state.current_bankroll - total_staked

        # Summary
        state = add_message(
            state,
            "bettor",
            f"Session: {len(placed_bets)} bets placed | "
            f"Total stake: ${total_staked:.2f} | "
            f"Remaining bankroll: ${self.session.current_bankroll:.2f}"
        )

        # Store session info in state
        state.betting_session = self.session.to_dict()

        return state

    def _create_bet(self, match: Match, bankroll: float) -> PlacedBet:
        """Create PlacedBet record from match."""
        self._bet_counter += 1
        bet_id = f"bet_{datetime.now().strftime('%Y%m%d')}_{self._bet_counter:04d}"

        value_bet = match.value_bet
        stake = value_bet.kelly_stake * bankroll

        return PlacedBet(
            bet_id=bet_id,
            match_id=match.match_id,
            match_name=f"{match.home_player.name} vs {match.away_player.name}",
            sport=match.sport,
            selection=value_bet.bet_on,
            bet_type="moneyline",
            stake=stake,
            odds=value_bet.odds,
            bookmaker=self.default_bookmaker,
            predicted_probability=value_bet.true_probability,
            edge=value_bet.edge,
            confidence=match.prediction.confidence if match.prediction else 0,
            quality_score=match.data_quality.overall_score if match.data_quality else 0
        )

    async def _place_bet(self, bet: PlacedBet) -> bool:
        """
        Place a bet with bookmaker.

        Args:
            bet: Bet to place

        Returns:
            True if successful
        """
        if self.simulation_mode:
            # Simulate bet placement
            await asyncio.sleep(0.1)  # Simulate API call
            bet.status = BetStatus.PLACED
            bet.notes = "Simulated bet"
            logger.info(f"Simulated bet placed: {bet.bet_id}")
            return True

        # Real bet placement would go here
        # Example:
        # client = self.bookmaker_clients.get(bet.bookmaker)
        # if client:
        #     result = await client.place_bet(bet)
        #     return result.success

        logger.warning(f"No bookmaker client for {bet.bookmaker}")
        return False

    async def settle_bet(self, bet_id: str, won: bool) -> Optional[PlacedBet]:
        """
        Settle a bet with result.

        Args:
            bet_id: Bet ID to settle
            won: Whether bet won

        Returns:
            Updated bet or None
        """
        # Find bet
        bet = None
        for b in self.bet_history:
            if b.bet_id == bet_id:
                bet = b
                break

        if not bet:
            logger.warning(f"Bet not found: {bet_id}")
            return None

        # Update status
        bet.status = BetStatus.WON if won else BetStatus.LOST
        bet.settled_at = datetime.now()

        # Calculate profit/loss
        if won:
            bet.profit_loss = bet.stake * (bet.odds - 1)
        else:
            bet.profit_loss = -bet.stake

        # Update session bankroll if active
        if self.session:
            self.session.current_bankroll += bet.stake + bet.profit_loss

        logger.info(f"Bet settled: {bet_id} - {'WON' if won else 'LOST'} (P/L: ${bet.profit_loss:.2f})")
        return bet

    async def check_results(self) -> List[PlacedBet]:
        """
        Check results for pending bets.

        Returns:
            List of newly settled bets
        """
        settled = []

        for bet in self.bet_history:
            if bet.status not in [BetStatus.PLACED, BetStatus.CONFIRMED]:
                continue

            # In real implementation, would check bookmaker API or results feed
            # For now, just log pending bets
            logger.debug(f"Pending bet: {bet.bet_id} - {bet.match_name}")

        return settled

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if not self.session:
            return {"error": "No active session"}

        return {
            "session_id": self.session.session_id,
            "date": self.session.date,
            "sport": self.session.sport,
            "starting_bankroll": self.session.starting_bankroll,
            "current_bankroll": self.session.current_bankroll,
            "profit_loss": self.session.current_bankroll - self.session.starting_bankroll,
            "roi": ((self.session.current_bankroll / self.session.starting_bankroll) - 1) * 100,
            "bet_count": len(self.session.bets),
            "total_staked": self.session.total_staked,
            "win_rate": self.session.win_rate,
            "pending_bets": sum(1 for b in self.session.bets if b.status in [BetStatus.PLACED, BetStatus.CONFIRMED]),
            "settled_bets": sum(1 for b in self.session.bets if b.status in [BetStatus.WON, BetStatus.LOST])
        }

    def get_history_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get historical betting statistics.

        Args:
            days: Number of days to include

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        recent_bets = [
            b for b in self.bet_history
            if (cutoff - b.placed_at).days <= days
        ]

        if not recent_bets:
            return {"error": "No bets in period", "days": days}

        settled = [b for b in recent_bets if b.status in [BetStatus.WON, BetStatus.LOST]]
        won = [b for b in settled if b.status == BetStatus.WON]

        total_staked = sum(b.stake for b in recent_bets)
        total_pl = sum(b.profit_loss for b in settled)

        return {
            "period_days": days,
            "total_bets": len(recent_bets),
            "settled_bets": len(settled),
            "pending_bets": len(recent_bets) - len(settled),
            "wins": len(won),
            "losses": len(settled) - len(won),
            "win_rate": len(won) / len(settled) if settled else 0,
            "total_staked": total_staked,
            "total_profit_loss": total_pl,
            "roi": (total_pl / total_staked * 100) if total_staked > 0 else 0,
            "avg_stake": total_staked / len(recent_bets) if recent_bets else 0,
            "avg_odds": sum(b.odds for b in recent_bets) / len(recent_bets) if recent_bets else 0,
            "avg_edge": sum(b.edge for b in recent_bets) / len(recent_bets) if recent_bets else 0
        }

    def export_bets(self, format: str = "dict") -> Any:
        """
        Export bet history.

        Args:
            format: Export format ("dict", "csv", "json")

        Returns:
            Exported data
        """
        if format == "dict":
            return [b.to_dict() for b in self.bet_history]
        elif format == "csv":
            # CSV string
            headers = ["bet_id", "match_name", "selection", "stake", "odds", "status", "profit_loss"]
            lines = [",".join(headers)]
            for b in self.bet_history:
                lines.append(f"{b.bet_id},{b.match_name},{b.selection},{b.stake:.2f},{b.odds:.2f},{b.status.value},{b.profit_loss:.2f}")
            return "\n".join(lines)
        else:
            import json
            return json.dumps([b.to_dict() for b in self.bet_history], indent=2)


# === HELPER FUNCTIONS ===

async def place_bets(
    matches: List[Match],
    bankroll: float = 1000.0,
    simulation: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to place bets on approved matches.

    Args:
        matches: List of approved matches
        bankroll: Current bankroll
        simulation: Use simulation mode

    Returns:
        Betting session summary
    """
    agent = BettorAgent(simulation_mode=simulation)

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        approved_bets=matches,
        current_bankroll=bankroll
    )

    result_state = await agent.process(state)
    return agent.get_session_stats()
