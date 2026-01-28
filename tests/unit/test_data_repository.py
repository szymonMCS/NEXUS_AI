"""
Tests for NEXUS ML data repository.

Checkpoint: 0.12
Integration tests for UnifiedDataRepository with mocking.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import Mock, patch

from core.data.enums import Sport
from core.data.schemas import (
    DataQuality,
    TeamMatchStats,
    HistoricalMatch,
    MatchData,
    TeamData,
    OddsData,
)
from core.data.repository import (
    UnifiedDataRepository,
    IMatchDataProvider,
    IOddsProvider,
    get_repository,
    reset_repository,
)
from core.data.validators import ValidationResult


class MockMatchDataProvider(IMatchDataProvider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock_provider"):
        self._name = name
        self._matches: List[MatchData] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def supported_sports(self) -> List[Sport]:
        return [Sport.FOOTBALL, Sport.TENNIS]

    def fetch_upcoming_matches(
        self,
        sport: Sport,
        date: Optional[datetime] = None,
    ) -> List[MatchData]:
        return [m for m in self._matches if m.sport == sport]

    def fetch_match_details(
        self,
        match_id: str,
        sport: Sport,
    ) -> Optional[MatchData]:
        for m in self._matches:
            if m.match_id == match_id:
                return m
        return None

    def fetch_team_stats(
        self,
        team_id: str,
        sport: Sport,
    ) -> Optional[TeamMatchStats]:
        return TeamMatchStats(
            goals_scored_avg=1.5,
            goals_conceded_avg=1.2,
        )

    def is_available(self) -> bool:
        return True

    def add_match(self, match: MatchData) -> None:
        """Helper to add test matches."""
        self._matches.append(match)


class MockOddsProvider(IOddsProvider):
    """Mock odds provider for testing."""

    @property
    def provider_name(self) -> str:
        return "mock_odds"

    def fetch_odds(
        self,
        match_id: str,
        sport: Sport,
    ) -> Optional[dict]:
        return {
            "home_win": 1.85,
            "draw": 3.40,
            "away_win": 4.00,
            "over_25": 1.90,
            "under_25": 1.90,
            "bookmaker": "mock_bookie",
        }


@pytest.fixture
def repository():
    """Create a fresh repository for each test."""
    return UnifiedDataRepository()


@pytest.fixture
def repository_with_providers():
    """Create repository with mock providers."""
    provider = MockMatchDataProvider()
    odds_provider = MockOddsProvider()
    return UnifiedDataRepository(
        providers=[provider],
        odds_providers=[odds_provider],
    )


@pytest.fixture
def sample_team_home():
    return TeamData(team_id="home-1", name="Home FC")


@pytest.fixture
def sample_team_away():
    return TeamData(team_id="away-1", name="Away FC")


@pytest.fixture
def sample_match(sample_team_home, sample_team_away):
    return MatchData(
        match_id="match-123",
        sport=Sport.FOOTBALL,
        home_team=sample_team_home,
        away_team=sample_team_away,
        league="Test League",
        start_time=datetime.utcnow() + timedelta(hours=2),
        data_quality=DataQuality(
            completeness=0.5,
            freshness_hours=0,
            sources_count=1,
        ),
    )


@pytest.fixture
def sample_historical():
    return HistoricalMatch(
        match_id="hist-1",
        date=datetime(2024, 1, 15),
        home_team_id="home-1",
        away_team_id="away-1",
        home_goals=2,
        away_goals=1,
        league="Test League",
    )


class TestUnifiedDataRepository:
    """Tests for UnifiedDataRepository."""

    def test_init_default(self):
        """Test initialization with defaults."""
        repo = UnifiedDataRepository()
        assert repo.get_stats()["providers"] == 0
        assert repo.get_stats()["odds_providers"] == 0

    def test_init_with_providers(self):
        """Test initialization with providers."""
        provider = MockMatchDataProvider()
        odds = MockOddsProvider()
        repo = UnifiedDataRepository(
            providers=[provider],
            odds_providers=[odds],
        )
        assert repo.get_stats()["providers"] == 1
        assert repo.get_stats()["odds_providers"] == 1

    def test_store_and_get_match(self, repository, sample_match):
        """Test storing and retrieving match data."""
        # Store via enrich (which calls _store_match)
        enriched = repository.enrich_match_data(sample_match, include_h2h=False, include_form=False, include_odds=False)

        # Retrieve
        retrieved = repository.get_match_data("match-123", Sport.FOOTBALL)
        assert retrieved is not None
        assert retrieved.match_id == "match-123"
        assert retrieved.home_team.name == "Home FC"

    def test_get_match_not_found(self, repository):
        """Test getting non-existent match."""
        result = repository.get_match_data("nonexistent")
        assert result is None

    def test_store_historical_match(self, repository, sample_historical):
        """Test storing historical match."""
        result = repository.store_historical_match(sample_historical, Sport.FOOTBALL)
        assert result is True

        # Verify via training data
        training = repository.get_training_data(Sport.FOOTBALL)
        assert len(training) == 1
        assert training[0].home_goals == 2

    def test_store_historical_no_id_fails(self, repository):
        """Test storing match without ID fails."""
        match = HistoricalMatch(
            match_id="",  # Empty ID
            date=datetime.utcnow(),
            home_team_id="a",
            away_team_id="b",
            home_goals=1,
            away_goals=1,
        )
        result = repository.store_historical_match(match, Sport.FOOTBALL)
        assert result is False

    def test_get_h2h_history(self, repository):
        """Test getting H2H history."""
        # Store some H2H matches
        h2h1 = HistoricalMatch(
            match_id="h2h-1",
            date=datetime(2024, 1, 1),
            home_team_id="team-a",
            away_team_id="team-b",
            home_goals=2,
            away_goals=1,
        )
        h2h2 = HistoricalMatch(
            match_id="h2h-2",
            date=datetime(2023, 6, 1),
            home_team_id="team-b",
            away_team_id="team-a",
            home_goals=0,
            away_goals=0,
        )
        h2h3 = HistoricalMatch(
            match_id="h2h-3",
            date=datetime(2023, 1, 1),
            home_team_id="team-a",
            away_team_id="team-c",  # Different team
            home_goals=3,
            away_goals=0,
        )

        repository.store_historical_match(h2h1, Sport.FOOTBALL)
        repository.store_historical_match(h2h2, Sport.FOOTBALL)
        repository.store_historical_match(h2h3, Sport.FOOTBALL)

        # Get H2H between team-a and team-b
        h2h = repository.get_h2h_history("team-a", "team-b", Sport.FOOTBALL)
        assert len(h2h) == 2
        assert h2h[0].match_id == "h2h-1"  # Most recent first

    def test_get_team_stats_from_historical(self, repository):
        """Test team stats calculation from historical data."""
        # Store matches for team-a
        matches = [
            HistoricalMatch(
                match_id=f"m{i}",
                date=datetime(2024, 1, i + 1),
                home_team_id="team-a" if i % 2 == 0 else "team-x",
                away_team_id="team-x" if i % 2 == 0 else "team-a",
                home_goals=2,
                away_goals=1,
            )
            for i in range(5)
        ]

        for m in matches:
            repository.store_historical_match(m, Sport.FOOTBALL)

        stats = repository.get_team_stats("team-a", Sport.FOOTBALL, num_matches=5)
        assert stats is not None
        assert stats.goals_scored_avg > 0
        assert stats.form_points > 0

    def test_get_team_stats_no_data(self, repository):
        """Test team stats with no data."""
        stats = repository.get_team_stats("nonexistent", Sport.FOOTBALL)
        assert stats is None

    def test_get_training_data_with_filters(self, repository):
        """Test training data with date filters."""
        matches = [
            HistoricalMatch(
                match_id=f"m{i}",
                date=datetime(2024, i + 1, 1),
                home_team_id="a",
                away_team_id="b",
                home_goals=1,
                away_goals=1,
                league="League A" if i < 3 else "League B",
            )
            for i in range(6)
        ]

        for m in matches:
            repository.store_historical_match(m, Sport.FOOTBALL)

        # All matches
        all_data = repository.get_training_data(Sport.FOOTBALL)
        assert len(all_data) == 6

        # Date filter
        filtered = repository.get_training_data(
            Sport.FOOTBALL,
            start_date=datetime(2024, 3, 1),
            end_date=datetime(2024, 5, 1),
        )
        assert len(filtered) == 2

        # League filter
        league_a = repository.get_training_data(
            Sport.FOOTBALL,
            league="League A",
        )
        assert len(league_a) == 3

    def test_get_upcoming_matches(self, repository, sample_match):
        """Test getting upcoming matches."""
        # Store a match starting in 2 hours
        repository.enrich_match_data(sample_match, include_h2h=False, include_form=False, include_odds=False)

        upcoming = repository.get_upcoming_matches(Sport.FOOTBALL, hours_ahead=24)
        assert len(upcoming) == 1

        # No upcoming in 1 hour (match is in 2 hours)
        soon = repository.get_upcoming_matches(Sport.FOOTBALL, hours_ahead=1)
        assert len(soon) == 0

    def test_update_match_result(self, repository, sample_match):
        """Test updating match result."""
        repository.enrich_match_data(sample_match, include_h2h=False, include_form=False, include_odds=False)

        # Update result
        success = repository.update_match_result(
            "match-123",
            home_goals=3,
            away_goals=1,
            home_goals_ht=1,
            away_goals_ht=0,
        )
        assert success is True

        # Verify historical record created
        training = repository.get_training_data(Sport.FOOTBALL)
        assert len(training) == 1
        assert training[0].home_goals == 3
        assert training[0].home_goals_ht == 1

        # Match should no longer be in upcoming
        upcoming = repository.get_upcoming_matches(Sport.FOOTBALL)
        assert len(upcoming) == 0

    def test_update_match_result_not_found(self, repository):
        """Test updating non-existent match."""
        success = repository.update_match_result(
            "nonexistent",
            home_goals=1,
            away_goals=0,
        )
        assert success is False

    def test_get_data_quality(self, repository, sample_match):
        """Test getting data quality."""
        repository.enrich_match_data(sample_match, include_h2h=False, include_form=False, include_odds=False)

        quality = repository.get_data_quality("match-123")
        assert quality.completeness >= 0

    def test_get_data_quality_not_found(self, repository):
        """Test data quality for non-existent match."""
        quality = repository.get_data_quality("nonexistent")
        assert quality.completeness == 0.0
        assert quality.sources_count == 0

    def test_enrich_match_data(self, repository, sample_match):
        """Test match enrichment."""
        # Store some H2H data first
        h2h = HistoricalMatch(
            match_id="h2h-1",
            date=datetime(2024, 1, 1),
            home_team_id="home-1",
            away_team_id="away-1",
            home_goals=1,
            away_goals=1,
        )
        repository.store_historical_match(h2h, Sport.FOOTBALL)

        # Store some historical for form
        for i in range(5):
            m = HistoricalMatch(
                match_id=f"form-{i}",
                date=datetime(2024, 1, i + 1),
                home_team_id="home-1" if i % 2 == 0 else "other",
                away_team_id="other" if i % 2 == 0 else "home-1",
                home_goals=2,
                away_goals=1,
            )
            repository.store_historical_match(m, Sport.FOOTBALL)

        enriched = repository.enrich_match_data(
            sample_match,
            include_h2h=True,
            include_form=True,
            include_odds=False,
        )

        assert enriched.h2h_history is not None
        assert len(enriched.h2h_history) >= 1
        assert enriched.data_quality.has_h2h is True

    def test_count_matches(self, repository):
        """Test match counting."""
        for i in range(10):
            m = HistoricalMatch(
                match_id=f"m{i}",
                date=datetime(2024, (i % 12) + 1, 1),
                home_team_id="a",
                away_team_id="b",
                home_goals=1,
                away_goals=1,
            )
            repository.store_historical_match(m, Sport.FOOTBALL)

        count = repository.count_matches(Sport.FOOTBALL)
        assert count == 10

        # With date filter
        filtered_count = repository.count_matches(
            Sport.FOOTBALL,
            start_date=datetime(2024, 6, 1),
        )
        assert filtered_count < 10

    def test_refresh_from_providers(self, repository_with_providers):
        """Test refreshing from providers."""
        # Add a match to the mock provider
        mock_provider = repository_with_providers._providers[0]
        match = MatchData(
            match_id="provider-match-1",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id="t1", name="Team 1"),
            away_team=TeamData(team_id="t2", name="Team 2"),
            league="Test",
            start_time=datetime.utcnow() + timedelta(hours=1),
            data_quality=DataQuality(
                completeness=0.3,
                freshness_hours=0,
                sources_count=1,
            ),
        )
        mock_provider.add_match(match)

        # Refresh
        count = repository_with_providers.refresh_from_providers(Sport.FOOTBALL)
        assert count == 1

        # Verify match is stored
        retrieved = repository_with_providers.get_match_data("provider-match-1")
        assert retrieved is not None

    def test_add_provider_at_runtime(self, repository):
        """Test adding provider at runtime."""
        assert repository.get_stats()["providers"] == 0

        provider = MockMatchDataProvider("dynamic")
        repository.add_provider(provider)

        assert repository.get_stats()["providers"] == 1

    def test_clear_cache(self, repository, sample_historical):
        """Test cache clearing."""
        # Build up some cache
        repository.store_historical_match(sample_historical, Sport.FOOTBALL)
        repository.get_team_stats("home-1", Sport.FOOTBALL)

        assert repository.get_stats()["cache_entries"] > 0

        repository.clear_cache()
        assert repository.get_stats()["cache_entries"] == 0

    def test_validate_for_prediction(self, repository, sample_match):
        """Test prediction validation."""
        repository.enrich_match_data(sample_match, include_h2h=False, include_form=False, include_odds=False)

        can_predict, reason = repository.validate_for_prediction("match-123")
        # Result depends on data quality
        assert isinstance(can_predict, bool)
        assert isinstance(reason, str)

    def test_validate_for_prediction_not_found(self, repository):
        """Test validation for non-existent match."""
        can_predict, reason = repository.validate_for_prediction("nonexistent")
        assert can_predict is False
        assert "not found" in reason.lower()


class TestSingletonHelpers:
    """Tests for singleton helpers."""

    def test_get_repository_singleton(self):
        """Test get_repository returns same instance."""
        reset_repository()  # Start fresh

        repo1 = get_repository()
        repo2 = get_repository()

        assert repo1 is repo2

    def test_reset_repository(self):
        """Test reset creates new instance."""
        repo1 = get_repository()
        reset_repository()
        repo2 = get_repository()

        assert repo1 is not repo2


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_reads(self, repository, sample_historical):
        """Test concurrent reads don't cause issues."""
        import threading

        repository.store_historical_match(sample_historical, Sport.FOOTBALL)

        errors = []

        def read_data():
            try:
                for _ in range(100):
                    repository.get_h2h_history("home-1", "away-1", Sport.FOOTBALL)
                    repository.get_training_data(Sport.FOOTBALL)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_data) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_writes(self, repository):
        """Test concurrent writes don't corrupt data."""
        import threading

        errors = []
        written = []

        def write_data(thread_id):
            try:
                for i in range(20):
                    match = HistoricalMatch(
                        match_id=f"thread-{thread_id}-match-{i}",
                        date=datetime.utcnow(),
                        home_team_id="a",
                        away_team_id="b",
                        home_goals=1,
                        away_goals=1,
                    )
                    repository.store_historical_match(match, Sport.FOOTBALL)
                    written.append(match.match_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_data, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all writes succeeded
        training = repository.get_training_data(Sport.FOOTBALL)
        assert len(training) == 100  # 5 threads * 20 matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
