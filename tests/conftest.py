# tests/conftest.py
"""
Pytest configuration and shared fixtures for NEXUS AI tests.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# === Environment Setup ===

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ["NEXUS_ENV"] = "test"
    os.environ["NEXUS_MODE"] = "lite"
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    os.environ["DATABASE_URL"] = "sqlite:///./test_nexus.db"

    yield

    # Cleanup
    if os.path.exists("./test_nexus.db"):
        os.remove("./test_nexus.db")


# === Async Support ===

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# === Mock External Services ===

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic API client."""
    with patch("langchain_anthropic.ChatAnthropic") as mock:
        mock_client = MagicMock()
        mock_client.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        mock.return_value = mock_client
        yield mock


@pytest.fixture
def mock_httpx():
    """Mock httpx for API requests."""
    with patch("httpx.AsyncClient") as mock:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock.return_value = mock_client
        yield mock


@pytest.fixture
def mock_playwright():
    """Mock Playwright for web scraping."""
    with patch("playwright.async_api.async_playwright") as mock:
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_playwright.stop = AsyncMock()

        mock.return_value.start = AsyncMock(return_value=mock_playwright)
        yield mock


# === State Fixtures ===

@pytest.fixture
def sample_player():
    """Create sample Player."""
    from core.state import Player

    return Player(
        name="Djokovic N.",
        ranking=1,
        rating=2100
    )


@pytest.fixture
def sample_team():
    """Create sample Team."""
    from core.state import Team

    return Team(
        name="Lakers",
        rating=1650
    )


@pytest.fixture
def sample_prediction():
    """Create sample Prediction."""
    from core.state import Prediction

    return Prediction(
        home_probability=0.65,
        away_probability=0.35,
        confidence=0.8
    )


@pytest.fixture
def sample_value_bet():
    """Create sample ValueBet."""
    from core.state import ValueBet

    return ValueBet(
        bet_on="home",
        odds=1.80,
        probability=0.65,
        edge=0.17,
        kelly_fraction=0.15,
        kelly_stake=0.04,
        quality_multiplier=0.85
    )


@pytest.fixture
def sample_data_quality():
    """Create sample DataQuality."""
    from core.state import DataQuality

    return DataQuality(
        overall_score=0.85,
        completeness=0.90,
        freshness=0.95,
        source_agreement=0.80
    )


@pytest.fixture
def sample_match(sample_player, sample_prediction, sample_value_bet, sample_data_quality):
    """Create sample Match."""
    from core.state import Match, Player

    away_player = Player(name="Sinner J.", ranking=4)

    return Match(
        match_id="test_match_001",
        sport="tennis",
        league="ATP 500",
        home_player=sample_player,
        away_player=away_player,
        start_time=datetime.now(),
        prediction=sample_prediction,
        value_bet=sample_value_bet,
        data_quality=sample_data_quality
    )


@pytest.fixture
def sample_state(sample_match):
    """Create sample NexusState."""
    from core.state import NexusState

    return NexusState(
        sport="tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        matches=[sample_match],
        approved_bets=[sample_match],
        current_bankroll=1000.0
    )


# === Model Fixtures ===

@pytest.fixture
def tennis_model():
    """Create TennisModel instance."""
    from core.models.tennis_model import TennisModel
    return TennisModel()


@pytest.fixture
def basketball_model():
    """Create BasketballModel instance."""
    from core.models.basketball_model import BasketballModel
    return BasketballModel()


@pytest.fixture
def value_calculator():
    """Create ValueCalculator instance."""
    from core.value_calculator import ValueCalculator
    return ValueCalculator()


# === Agent Fixtures ===

@pytest.fixture
def bettor_agent():
    """Create BettorAgent in simulation mode."""
    from agents.bettor import BettorAgent
    return BettorAgent(simulation_mode=True)


# === Data Fixtures ===

@pytest.fixture
def sample_fixture_data():
    """Sample fixture data from API."""
    return {
        "match_id": "flashscore_abc123",
        "sport": "tennis",
        "league": "ATP 500 - Dubai",
        "home_team": "Djokovic N.",
        "away_team": "Sinner J.",
        "start_time": datetime.now(),
        "home_odds": 1.75,
        "away_odds": 2.10,
        "source": "sofascore"
    }


@pytest.fixture
def sample_historical_match():
    """Sample historical match for backtesting."""
    return {
        "id": "hist_001",
        "sport": "tennis",
        "match_date": "2024-01-15",
        "home_team": "Djokovic N.",
        "away_team": "Sinner J.",
        "home_odds": 1.75,
        "away_odds": 2.10,
        "winner": "home",
        "home_score": 2,
        "away_score": 1
    }


# === Helper Functions ===

def create_mock_response(data, status_code=200):
    """Create mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = data
    response.text = str(data)
    return response


def create_mock_match(**kwargs):
    """Create mock Match with custom attributes."""
    from core.state import Match, Player, Prediction, ValueBet

    defaults = {
        "match_id": "test_match",
        "sport": "tennis",
        "league": "ATP",
        "home_player": Player(name="Player A", ranking=1),
        "away_player": Player(name="Player B", ranking=2),
        "start_time": datetime.now(),
        "prediction": Prediction(
            home_probability=0.6,
            away_probability=0.4,
            confidence=0.75
        ),
        "value_bet": ValueBet(
            bet_on="home",
            odds=1.90,
            probability=0.6,
            edge=0.14,
            kelly_fraction=0.12,
            kelly_stake=0.03,
            quality_multiplier=0.85
        )
    }

    defaults.update(kwargs)
    return Match(**defaults)


# === Pytest Configuration ===

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require external API access"
    )


# === Skip conditions ===

def requires_anthropic_key():
    """Skip if no Anthropic API key."""
    return pytest.mark.skipif(
        os.environ.get("ANTHROPIC_API_KEY", "test_key") == "test_key",
        reason="Requires real Anthropic API key"
    )


def requires_playwright():
    """Skip if Playwright not installed."""
    try:
        import playwright
        return lambda x: x
    except ImportError:
        return pytest.mark.skip(reason="Playwright not installed")
