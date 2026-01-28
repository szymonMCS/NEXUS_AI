"""
Tests for Historical Data Collector.

Checkpoint: 5.6
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from core.data.enums import Sport
from data.collectors.results import (
    CollectionStatus,
    SourceResult,
    CollectedMatch,
    CollectionResult,
    CollectionConfig,
    DEFAULT_LEAGUES,
)
from data.collectors.historical_collector import (
    HistoricalDataCollector,
    FootballDataAdapter,
    APISportsFootballAdapter,
    APISportsBasketballAdapter,
)
from data.apis import APIResponse


# =============================================================================
# Test CollectedMatch
# =============================================================================

class TestCollectedMatch:
    """Tests for CollectedMatch dataclass."""

    def test_create_match(self):
        """Test basic match creation."""
        match = CollectedMatch(
            match_id="test_123",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime(2024, 1, 15, 15, 0),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=1,
        )

        assert match.match_id == "test_123"
        assert match.total_goals == 3
        assert match.goal_difference == 1
        assert match.result == "H"
        assert not match.is_over_25  # 3 goals is not > 2.5
        assert match.btts  # Both teams scored

    def test_match_properties(self):
        """Test computed properties."""
        # Draw
        match = CollectedMatch(
            match_id="draw_match",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=1,
            away_goals=1,
        )
        assert match.result == "D"
        assert match.goal_difference == 0

        # Away win, over 2.5
        match2 = CollectedMatch(
            match_id="away_win",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=1,
            away_goals=3,
        )
        assert match2.result == "A"
        assert match2.is_over_25  # 4 goals > 2.5
        assert match2.btts

        # No BTTS
        match3 = CollectedMatch(
            match_id="no_btts",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=0,
        )
        assert not match3.btts

    def test_match_with_stats(self):
        """Test match with detailed statistics."""
        match = CollectedMatch(
            match_id="with_stats",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=1,
            home_shots=15,
            away_shots=10,
            home_possession=0.55,
        )
        assert match.has_stats
        assert not match.has_odds

    def test_match_with_odds(self):
        """Test match with odds."""
        match = CollectedMatch(
            match_id="with_odds",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=1,
            odds_home=1.85,
            odds_draw=3.50,
            odds_away=4.20,
        )
        assert match.has_odds

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        match = CollectedMatch(
            match_id="serialize_test",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime(2024, 1, 15, 15, 0),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=1,
            home_shots=15,
            odds_home=1.85,
        )

        data = match.to_dict()
        restored = CollectedMatch.from_dict(data)

        assert restored.match_id == match.match_id
        assert restored.home_goals == match.home_goals
        assert restored.away_goals == match.away_goals
        assert restored.home_shots == match.home_shots
        assert restored.odds_home == match.odds_home


# =============================================================================
# Test CollectionResult
# =============================================================================

class TestCollectionResult:
    """Tests for CollectionResult dataclass."""

    def test_create_result(self):
        """Test basic result creation."""
        result = CollectionResult(
            collection_id="test_001",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert result.collection_id == "test_001"
        assert result.total_collected == 0
        assert result.total_errors == 0

    def test_add_matches(self):
        """Test adding matches to result."""
        result = CollectionResult(
            collection_id="test_002",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        match = CollectedMatch(
            match_id="m1",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="Team A",
            away_team_id="2",
            away_team_name="Team B",
            home_goals=2,
            away_goals=1,
        )

        result.add_match(match)
        assert result.total_collected == 1

    def test_finalize_success(self):
        """Test finalization with successful collection."""
        result = CollectionResult(
            collection_id="test_003",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        # Add a match
        result.add_match(CollectedMatch(
            match_id="m1",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="A",
            away_team_id="2",
            away_team_name="B",
            home_goals=1,
            away_goals=1,
        ))

        result.finalize()
        assert result.status == CollectionStatus.SUCCESS
        assert result.completed_at is not None

    def test_finalize_partial(self):
        """Test finalization with partial success."""
        result = CollectionResult(
            collection_id="test_004",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        result.add_match(CollectedMatch(
            match_id="m1",
            source="test",
            sport=Sport.FOOTBALL,
            league="PL",
            season="2024",
            match_date=datetime.utcnow(),
            home_team_id="1",
            home_team_name="A",
            away_team_id="2",
            away_team_name="B",
            home_goals=1,
            away_goals=1,
        ))
        result.add_error("Some API failed")

        result.finalize()
        assert result.status == CollectionStatus.PARTIAL

    def test_finalize_failed(self):
        """Test finalization with complete failure."""
        result = CollectionResult(
            collection_id="test_005",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        result.add_error("All APIs failed")
        result.finalize()
        assert result.status == CollectionStatus.FAILED

    def test_finalize_no_data(self):
        """Test finalization with no data and no errors."""
        result = CollectionResult(
            collection_id="test_006",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        result.finalize()
        assert result.status == CollectionStatus.NO_DATA

    def test_get_summary(self):
        """Test summary generation."""
        result = CollectionResult(
            collection_id="test_007",
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        result.add_source_result(SourceResult(
            source_name="api_football",
            status=CollectionStatus.SUCCESS,
            records_collected=100,
        ))

        summary = result.get_summary()
        assert summary["collection_id"] == "test_007"
        assert summary["league"] == "PL"
        assert "api_football" in summary["sources_used"]


# =============================================================================
# Test SourceResult
# =============================================================================

class TestSourceResult:
    """Tests for SourceResult dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = SourceResult(
            source_name="test",
            status=CollectionStatus.SUCCESS,
            records_collected=80,
            records_failed=20,
        )
        assert result.success_rate == 0.8

    def test_success_rate_no_records(self):
        """Test success rate with no records."""
        result = SourceResult(
            source_name="test",
            status=CollectionStatus.NO_DATA,
        )
        assert result.success_rate == 0.0


# =============================================================================
# Test CollectionConfig
# =============================================================================

class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CollectionConfig()
        assert Sport.FOOTBALL in config.sports
        assert config.min_matches_per_league == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = CollectionConfig(
            sports=[Sport.BASKETBALL],
            min_matches_per_league=50,
            require_odds=True,
        )
        assert Sport.BASKETBALL in config.sports
        assert Sport.FOOTBALL not in config.sports
        assert config.require_odds


# =============================================================================
# Test Adapters (with mocks)
# =============================================================================

class TestFootballDataAdapter:
    """Tests for FootballDataAdapter."""

    @pytest.mark.asyncio
    async def test_fetch_non_football(self):
        """Test that adapter returns empty for non-football sports."""
        adapter = FootballDataAdapter()
        api_manager = MagicMock()

        matches = await adapter.fetch_matches(
            api_manager=api_manager,
            sport=Sport.BASKETBALL,
            league="NBA",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert matches == []

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful fetch."""
        adapter = FootballDataAdapter()

        # Mock API manager
        api_manager = MagicMock()
        api_manager.football_data = AsyncMock()
        api_manager.football_data.get_matches = AsyncMock(return_value=APIResponse(
            success=True,
            data={
                "matches": [
                    {
                        "id": 123,
                        "status": "FINISHED",
                        "utcDate": "2024-01-15T15:00:00Z",
                        "homeTeam": {"id": 1, "name": "Team A"},
                        "awayTeam": {"id": 2, "name": "Team B"},
                        "score": {
                            "fullTime": {"home": 2, "away": 1},
                            "halfTime": {"home": 1, "away": 0},
                        },
                        "season": {"startDate": "2024-08-01"},
                    }
                ]
            },
            source="football_data",
        ))

        matches = await adapter.fetch_matches(
            api_manager=api_manager,
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert len(matches) == 1
        assert matches[0].home_goals == 2
        assert matches[0].away_goals == 1
        assert matches[0].home_goals_ht == 1

    @pytest.mark.asyncio
    async def test_fetch_skips_unfinished(self):
        """Test that unfinished matches are skipped."""
        adapter = FootballDataAdapter()

        api_manager = MagicMock()
        api_manager.football_data = AsyncMock()
        api_manager.football_data.get_matches = AsyncMock(return_value=APIResponse(
            success=True,
            data={
                "matches": [
                    {
                        "id": 123,
                        "status": "SCHEDULED",  # Not finished
                        "utcDate": "2024-01-15T15:00:00Z",
                        "homeTeam": {"id": 1, "name": "Team A"},
                        "awayTeam": {"id": 2, "name": "Team B"},
                        "score": {"fullTime": {"home": None, "away": None}},
                    }
                ]
            },
            source="football_data",
        ))

        matches = await adapter.fetch_matches(
            api_manager=api_manager,
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert len(matches) == 0


# =============================================================================
# Test HistoricalDataCollector
# =============================================================================

class TestHistoricalDataCollector:
    """Tests for HistoricalDataCollector."""

    def test_create_collector(self):
        """Test collector creation."""
        collector = HistoricalDataCollector()
        assert collector is not None
        assert collector.config is not None

    def test_create_with_custom_config(self):
        """Test collector with custom config."""
        config = CollectionConfig(
            sports=[Sport.BASKETBALL],
            min_matches_per_league=50,
        )
        collector = HistoricalDataCollector(config=config)
        assert collector.config.min_matches_per_league == 50


# =============================================================================
# Test DEFAULT_LEAGUES
# =============================================================================

class TestDefaultLeagues:
    """Tests for DEFAULT_LEAGUES constant."""

    def test_football_leagues(self):
        """Test that default football leagues are defined."""
        assert Sport.FOOTBALL in DEFAULT_LEAGUES
        assert "PL" in DEFAULT_LEAGUES[Sport.FOOTBALL]
        assert "LaLiga" in DEFAULT_LEAGUES[Sport.FOOTBALL]

    def test_basketball_leagues(self):
        """Test that default basketball leagues are defined."""
        assert Sport.BASKETBALL in DEFAULT_LEAGUES
        assert "NBA" in DEFAULT_LEAGUES[Sport.BASKETBALL]
