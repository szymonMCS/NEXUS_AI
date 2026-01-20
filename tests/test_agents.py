# tests/test_agents.py
"""
Unit tests for NEXUS AI agents.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


class TestBettorAgent:
    """Tests for BettorAgent."""

    @pytest.fixture
    def agent(self):
        """Create BettorAgent in simulation mode."""
        from agents.bettor import BettorAgent
        return BettorAgent(simulation_mode=True)

    @pytest.fixture
    def sample_state(self):
        """Create sample NexusState with approved bets."""
        from core.state import NexusState, Match, Player, ValueBet, Prediction

        # Create players
        home_player = Player(name="Djokovic N.", ranking=1)
        away_player = Player(name="Sinner J.", ranking=4)

        # Create prediction
        prediction = Prediction(
            home_probability=0.65,
            away_probability=0.35,
            confidence=0.8
        )

        # Create value bet
        value_bet = ValueBet(
            bet_on="home",
            odds=1.80,
            probability=0.65,
            edge=0.17,
            kelly_fraction=0.15,
            kelly_stake=0.04,
            quality_multiplier=0.85
        )

        # Create match
        match = Match(
            match_id="test_match_1",
            sport="tennis",
            league="ATP 500",
            home_player=home_player,
            away_player=away_player,
            start_time=datetime.now(),
            prediction=prediction,
            value_bet=value_bet
        )

        return NexusState(
            sport="tennis",
            date=datetime.now().strftime("%Y-%m-%d"),
            approved_bets=[match],
            current_bankroll=1000.0
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.simulation_mode is True
        assert agent.default_bookmaker == "simulation"
        assert agent.session is None
        assert len(agent.bet_history) == 0

    @pytest.mark.asyncio
    async def test_process_creates_session(self, agent, sample_state):
        """Test process creates betting session."""
        result_state = await agent.process(sample_state)

        assert agent.session is not None
        assert agent.session.sport == "tennis"
        assert agent.session.starting_bankroll == 1000.0

    @pytest.mark.asyncio
    async def test_process_places_bets(self, agent, sample_state):
        """Test process places bets."""
        result_state = await agent.process(sample_state)

        assert len(agent.bet_history) > 0
        assert len(agent.session.bets) > 0

    @pytest.mark.asyncio
    async def test_bet_record_created(self, agent, sample_state):
        """Test bet record has correct data."""
        await agent.process(sample_state)

        bet = agent.bet_history[0]

        assert bet.match_id == "test_match_1"
        assert bet.sport == "tennis"
        assert bet.selection == "home"
        assert bet.odds == 1.80
        assert bet.bookmaker == "simulation"

    @pytest.mark.asyncio
    async def test_simulation_mode(self, agent, sample_state):
        """Test simulation mode marks bet as placed."""
        await agent.process(sample_state)

        from agents.bettor import BetStatus
        bet = agent.bet_history[0]

        assert bet.status == BetStatus.PLACED
        assert "Simulated" in bet.notes

    @pytest.mark.asyncio
    async def test_empty_approved_bets(self, agent):
        """Test handling of no approved bets."""
        from core.state import NexusState

        state = NexusState(
            sport="tennis",
            date="2024-01-01",
            approved_bets=[],
            current_bankroll=1000.0
        )

        result_state = await agent.process(state)

        assert len(agent.bet_history) == 0
        assert agent.session is None

    @pytest.mark.asyncio
    async def test_settle_bet_won(self, agent, sample_state):
        """Test settling a winning bet."""
        await agent.process(sample_state)

        bet = agent.bet_history[0]
        stake = bet.stake

        settled_bet = await agent.settle_bet(bet.bet_id, won=True)

        from agents.bettor import BetStatus

        assert settled_bet.status == BetStatus.WON
        assert settled_bet.profit_loss > 0
        assert settled_bet.profit_loss == stake * (bet.odds - 1)

    @pytest.mark.asyncio
    async def test_settle_bet_lost(self, agent, sample_state):
        """Test settling a losing bet."""
        await agent.process(sample_state)

        bet = agent.bet_history[0]
        stake = bet.stake

        settled_bet = await agent.settle_bet(bet.bet_id, won=False)

        from agents.bettor import BetStatus

        assert settled_bet.status == BetStatus.LOST
        assert settled_bet.profit_loss < 0
        assert settled_bet.profit_loss == -stake

    @pytest.mark.asyncio
    async def test_settle_nonexistent_bet(self, agent):
        """Test settling non-existent bet returns None."""
        result = await agent.settle_bet("fake_bet_id", won=True)
        assert result is None

    def test_session_stats(self, agent):
        """Test session stats before any session."""
        stats = agent.get_session_stats()
        assert "error" in stats

    @pytest.mark.asyncio
    async def test_session_stats_active(self, agent, sample_state):
        """Test session stats with active session."""
        await agent.process(sample_state)

        stats = agent.get_session_stats()

        assert "session_id" in stats
        assert stats["starting_bankroll"] == 1000.0
        assert stats["bet_count"] > 0

    def test_placed_bet_potential_profit(self):
        """Test PlacedBet potential profit calculation."""
        from agents.bettor import PlacedBet

        bet = PlacedBet(
            bet_id="test",
            match_id="match_1",
            match_name="Test Match",
            sport="tennis",
            selection="home",
            bet_type="moneyline",
            stake=100.0,
            odds=2.0,
            bookmaker="test"
        )

        # Potential profit = stake * (odds - 1) = 100 * 1 = 100
        assert bet.potential_profit == 100.0

    def test_placed_bet_expected_value(self):
        """Test PlacedBet expected value calculation."""
        from agents.bettor import PlacedBet

        bet = PlacedBet(
            bet_id="test",
            match_id="match_1",
            match_name="Test Match",
            sport="tennis",
            selection="home",
            bet_type="moneyline",
            stake=100.0,
            odds=2.0,
            bookmaker="test",
            predicted_probability=0.60
        )

        # EV = (prob * odds - 1) * stake = (0.6 * 2 - 1) * 100 = 20
        assert bet.expected_value == 20.0


class TestBettingSession:
    """Tests for BettingSession."""

    @pytest.fixture
    def session(self):
        """Create BettingSession."""
        from agents.bettor import BettingSession
        return BettingSession(
            session_id="test_session",
            date="2024-01-01",
            sport="tennis",
            starting_bankroll=1000.0,
            current_bankroll=1000.0
        )

    def test_session_initialization(self, session):
        """Test session initializes correctly."""
        assert session.session_id == "test_session"
        assert session.starting_bankroll == 1000.0
        assert len(session.bets) == 0

    def test_total_staked_empty(self, session):
        """Test total staked with no bets."""
        assert session.total_staked == 0

    def test_total_profit_loss_empty(self, session):
        """Test profit/loss with no bets."""
        assert session.total_profit_loss == 0

    def test_win_rate_empty(self, session):
        """Test win rate with no bets."""
        assert session.win_rate == 0.0

    def test_win_rate_with_bets(self, session):
        """Test win rate calculation."""
        from agents.bettor import PlacedBet, BetStatus

        # Add winning bet
        bet1 = PlacedBet(
            bet_id="bet1", match_id="m1", match_name="Match 1",
            sport="tennis", selection="home", bet_type="moneyline",
            stake=100, odds=2.0, bookmaker="test"
        )
        bet1.status = BetStatus.WON
        bet1.profit_loss = 100

        # Add losing bet
        bet2 = PlacedBet(
            bet_id="bet2", match_id="m2", match_name="Match 2",
            sport="tennis", selection="away", bet_type="moneyline",
            stake=100, odds=2.0, bookmaker="test"
        )
        bet2.status = BetStatus.LOST
        bet2.profit_loss = -100

        session.bets = [bet1, bet2]

        assert session.win_rate == 0.5  # 1 win / 2 total


class TestBetStatus:
    """Tests for BetStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        from agents.bettor import BetStatus

        expected = [
            "PENDING", "PLACED", "CONFIRMED",
            "WON", "LOST", "VOID", "PARTIAL",
            "FAILED", "CASHED_OUT"
        ]

        for status in expected:
            assert hasattr(BetStatus, status)

    def test_status_values(self):
        """Test status string values."""
        from agents.bettor import BetStatus

        assert BetStatus.PENDING.value == "pending"
        assert BetStatus.WON.value == "won"
        assert BetStatus.LOST.value == "lost"


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.asyncio
    async def test_place_bets_helper(self):
        """Test place_bets convenience function."""
        from agents.bettor import place_bets
        from core.state import Match, Player, ValueBet, Prediction
        from datetime import datetime

        # Create test match
        match = Match(
            match_id="test",
            sport="tennis",
            league="ATP",
            home_player=Player(name="Player A", ranking=1),
            away_player=Player(name="Player B", ranking=2),
            start_time=datetime.now(),
            prediction=Prediction(
                home_probability=0.6,
                away_probability=0.4,
                confidence=0.8
            ),
            value_bet=ValueBet(
                bet_on="home",
                odds=1.8,
                probability=0.6,
                edge=0.08,
                kelly_fraction=0.1,
                kelly_stake=0.03,
                quality_multiplier=0.9
            )
        )

        result = await place_bets([match], bankroll=1000.0, simulation=True)

        assert "session_id" in result
        assert result["bet_count"] > 0


# === Mock fixtures for testing without real dependencies ===

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic API client."""
    with patch("langchain_anthropic.ChatAnthropic") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_state():
    """Create mock NexusState."""
    state = MagicMock()
    state.sport = "tennis"
    state.date = "2024-01-01"
    state.current_bankroll = 1000.0
    state.approved_bets = []
    return state


# Run with: pytest tests/test_agents.py -v
