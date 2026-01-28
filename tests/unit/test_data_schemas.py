"""
Tests for NEXUS ML data schemas.

Checkpoint: 0.11
Tests for dataclasses: creation, properties, validation.
"""

import pytest
from datetime import datetime, timedelta

from core.data.enums import Sport
from core.data.schemas import (
    DataQuality,
    TeamMatchStats,
    HistoricalMatch,
    MatchData,
    TeamData,
    OddsData,
)


class TestSportEnum:
    """Tests for Sport enum."""

    def test_sport_values(self):
        """Test all sport values exist."""
        assert Sport.TENNIS.value == "tennis"
        assert Sport.BASKETBALL.value == "basketball"
        assert Sport.FOOTBALL.value == "football"
        assert Sport.HANDBALL.value == "handball"
        assert Sport.TABLE_TENNIS.value == "table_tennis"
        assert Sport.GREYHOUND.value == "greyhound"

    def test_sport_str(self):
        """Test __str__ returns value."""
        assert str(Sport.FOOTBALL) == "football"
        assert str(Sport.TENNIS) == "tennis"


class TestDataQuality:
    """Tests for DataQuality dataclass."""

    def test_create_default(self):
        """Test creating with minimal args."""
        dq = DataQuality(
            completeness=0.8,
            freshness_hours=2,
            sources_count=2,
        )
        assert dq.completeness == 0.8
        assert dq.freshness_hours == 2
        assert dq.sources_count == 2
        assert dq.has_h2h is False
        assert dq.has_form is False
        assert dq.has_odds is False

    def test_is_sufficient_true(self):
        """Test is_sufficient with good data."""
        dq = DataQuality(
            completeness=0.7,
            freshness_hours=1,
            sources_count=2,
        )
        assert dq.is_sufficient is True

    def test_is_sufficient_false_low_completeness(self):
        """Test is_sufficient with low completeness."""
        dq = DataQuality(
            completeness=0.5,
            freshness_hours=1,
            sources_count=2,
        )
        assert dq.is_sufficient is False

    def test_is_sufficient_false_no_sources(self):
        """Test is_sufficient with no sources."""
        dq = DataQuality(
            completeness=0.9,
            freshness_hours=1,
            sources_count=0,
        )
        assert dq.is_sufficient is False

    def test_quality_level_excellent(self):
        """Test quality_level excellent."""
        dq = DataQuality(
            completeness=0.95,
            freshness_hours=1,
            sources_count=3,
        )
        assert dq.quality_level == "excellent"

    def test_quality_level_good(self):
        """Test quality_level good."""
        dq = DataQuality(
            completeness=0.75,
            freshness_hours=1,
            sources_count=2,
        )
        assert dq.quality_level == "good"

    def test_quality_level_moderate(self):
        """Test quality_level moderate."""
        dq = DataQuality(
            completeness=0.65,
            freshness_hours=1,
            sources_count=1,
        )
        assert dq.quality_level == "moderate"

    def test_quality_level_insufficient(self):
        """Test quality_level insufficient."""
        dq = DataQuality(
            completeness=0.3,
            freshness_hours=1,
            sources_count=1,
        )
        assert dq.quality_level == "insufficient"


class TestTeamMatchStats:
    """Tests for TeamMatchStats dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        stats = TeamMatchStats(
            goals_scored_avg=1.5,
            goals_conceded_avg=1.2,
        )
        assert stats.goals_scored_avg == 1.5
        assert stats.goals_conceded_avg == 1.2
        assert stats.home_goals_avg is None
        assert stats.away_goals_avg is None
        assert stats.form_points == 0.0
        assert stats.rest_days == 0

    def test_create_full(self):
        """Test creating with all args."""
        stats = TeamMatchStats(
            goals_scored_avg=2.0,
            goals_conceded_avg=1.0,
            home_goals_avg=2.5,
            away_goals_avg=1.5,
            form_points=0.8,
            rest_days=5,
        )
        assert stats.home_goals_avg == 2.5
        assert stats.away_goals_avg == 1.5
        assert stats.form_points == 0.8
        assert stats.rest_days == 5

    def test_attack_strength(self):
        """Test attack_strength property."""
        stats = TeamMatchStats(
            goals_scored_avg=2.6,  # 2x league avg (1.3)
            goals_conceded_avg=1.0,
        )
        assert stats.attack_strength == 2.0

    def test_defense_strength(self):
        """Test defense_strength property."""
        stats = TeamMatchStats(
            goals_scored_avg=1.0,
            goals_conceded_avg=0.65,  # 0.5x league avg
        )
        assert stats.defense_strength == 0.5


class TestHistoricalMatch:
    """Tests for HistoricalMatch dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        match = HistoricalMatch(
            match_id="test-123",
            date=datetime(2024, 1, 15),
            home_team_id="team-a",
            away_team_id="team-b",
            home_goals=2,
            away_goals=1,
        )
        assert match.match_id == "test-123"
        assert match.home_goals == 2
        assert match.away_goals == 1
        assert match.league == ""
        assert match.season == ""

    def test_total_goals(self):
        """Test total_goals property."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=3,
            away_goals=2,
        )
        assert match.total_goals == 5

    def test_goal_diff(self):
        """Test goal_diff property (from home perspective)."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=1,
            away_goals=3,
        )
        assert match.goal_diff == -2

    def test_is_over_25_true(self):
        """Test is_over_25 when total > 2.5."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=2,
            away_goals=1,
        )
        assert match.is_over_25 is True

    def test_is_over_25_false(self):
        """Test is_over_25 when total <= 2.5."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=1,
            away_goals=1,
        )
        assert match.is_over_25 is False

    def test_home_win(self):
        """Test home_win property."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=2,
            away_goals=0,
        )
        assert match.home_win is True
        assert match.away_win is False
        assert match.draw is False

    def test_away_win(self):
        """Test away_win property."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=0,
            away_goals=1,
        )
        assert match.home_win is False
        assert match.away_win is True
        assert match.draw is False

    def test_draw(self):
        """Test draw property."""
        match = HistoricalMatch(
            match_id="test-1",
            date=datetime(2024, 1, 1),
            home_team_id="a",
            away_team_id="b",
            home_goals=1,
            away_goals=1,
        )
        assert match.home_win is False
        assert match.away_win is False
        assert match.draw is True


class TestTeamData:
    """Tests for TeamData dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        team = TeamData(
            team_id="team-123",
            name="Test FC",
        )
        assert team.team_id == "team-123"
        assert team.name == "Test FC"
        assert team.ranking is None
        assert team.elo_rating is None

    def test_create_full(self):
        """Test creating with all args."""
        team = TeamData(
            team_id="team-123",
            name="Test FC",
            ranking=5,
            elo_rating=1850.5,
        )
        assert team.ranking == 5
        assert team.elo_rating == 1850.5


class TestOddsData:
    """Tests for OddsData dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        odds = OddsData(
            home_win=1.80,
            away_win=2.10,
        )
        assert odds.home_win == 1.80
        assert odds.away_win == 2.10
        assert odds.draw is None
        assert odds.bookmaker == "unknown"

    def test_create_full(self):
        """Test creating with all args."""
        odds = OddsData(
            home_win=1.75,
            draw=3.50,
            away_win=4.20,
            over_25=1.85,
            under_25=1.95,
            handicap_line=-1.5,
            handicap_home=2.10,
            handicap_away=1.70,
            bookmaker="test_bookie",
        )
        assert odds.draw == 3.50
        assert odds.over_25 == 1.85
        assert odds.handicap_line == -1.5
        assert odds.bookmaker == "test_bookie"


class TestMatchData:
    """Tests for MatchData dataclass."""

    @pytest.fixture
    def home_team(self):
        return TeamData(team_id="home-1", name="Home FC")

    @pytest.fixture
    def away_team(self):
        return TeamData(team_id="away-1", name="Away FC")

    def test_create_minimal(self, home_team, away_team):
        """Test creating with minimal args."""
        match = MatchData(
            match_id="match-123",
            sport=Sport.FOOTBALL,
            home_team=home_team,
            away_team=away_team,
            league="Premier League",
            start_time=datetime(2024, 3, 15, 15, 0),
        )
        assert match.match_id == "match-123"
        assert match.sport == Sport.FOOTBALL
        assert match.home_team.name == "Home FC"
        assert match.away_team.name == "Away FC"
        assert match.league == "Premier League"
        assert match.home_stats is None
        assert match.away_stats is None
        assert match.h2h_history is None
        assert match.odds is None

    def test_default_data_quality(self, home_team, away_team):
        """Test default data_quality is empty."""
        match = MatchData(
            match_id="match-123",
            sport=Sport.FOOTBALL,
            home_team=home_team,
            away_team=away_team,
            league="League",
            start_time=datetime.utcnow(),
        )
        assert match.data_quality.completeness == 0.0
        assert match.data_quality.sources_count == 0

    def test_create_with_stats(self, home_team, away_team):
        """Test creating with stats."""
        home_stats = TeamMatchStats(goals_scored_avg=1.8, goals_conceded_avg=1.0)
        away_stats = TeamMatchStats(goals_scored_avg=1.2, goals_conceded_avg=1.5)

        match = MatchData(
            match_id="match-123",
            sport=Sport.FOOTBALL,
            home_team=home_team,
            away_team=away_team,
            league="League",
            start_time=datetime.utcnow(),
            home_stats=home_stats,
            away_stats=away_stats,
        )
        assert match.home_stats.goals_scored_avg == 1.8
        assert match.away_stats.goals_conceded_avg == 1.5

    def test_create_with_h2h(self, home_team, away_team):
        """Test creating with h2h history."""
        h2h = [
            HistoricalMatch(
                match_id="h2h-1",
                date=datetime(2023, 12, 1),
                home_team_id="home-1",
                away_team_id="away-1",
                home_goals=2,
                away_goals=1,
            ),
            HistoricalMatch(
                match_id="h2h-2",
                date=datetime(2023, 6, 1),
                home_team_id="away-1",
                away_team_id="home-1",
                home_goals=0,
                away_goals=0,
            ),
        ]

        match = MatchData(
            match_id="match-123",
            sport=Sport.FOOTBALL,
            home_team=home_team,
            away_team=away_team,
            league="League",
            start_time=datetime.utcnow(),
            h2h_history=h2h,
        )
        assert len(match.h2h_history) == 2
        assert match.h2h_history[0].home_goals == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
