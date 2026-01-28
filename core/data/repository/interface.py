"""
Repository interface for NEXUS ML.

Checkpoint: 0.8
Responsibility: Define abstract interface for data access.
Principle: Single source of truth - all data access through this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

from core.data.enums import Sport
from core.data.schemas import (
    MatchData,
    TeamMatchStats,
    HistoricalMatch,
    DataQuality,
)


class IDataRepository(ABC):
    """
    Abstract interface for unified data repository.

    All data access in NEXUS ML should go through implementations
    of this interface to ensure single source of truth.
    """

    @abstractmethod
    def get_match_data(
        self,
        match_id: str,
        sport: Optional[Sport] = None,
    ) -> Optional[MatchData]:
        """
        Get complete match data by ID.

        Args:
            match_id: Unique match identifier
            sport: Optional sport filter for disambiguation

        Returns:
            MatchData if found, None otherwise
        """
        pass

    @abstractmethod
    def get_team_stats(
        self,
        team_id: str,
        sport: Sport,
        num_matches: int = 5,
    ) -> Optional[TeamMatchStats]:
        """
        Get aggregated team statistics from recent matches.

        Args:
            team_id: Unique team identifier
            sport: Sport type
            num_matches: Number of recent matches to aggregate

        Returns:
            TeamMatchStats if data available, None otherwise
        """
        pass

    @abstractmethod
    def get_h2h_history(
        self,
        team1_id: str,
        team2_id: str,
        sport: Sport,
        limit: int = 10,
    ) -> List[HistoricalMatch]:
        """
        Get head-to-head history between two teams.

        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            sport: Sport type
            limit: Maximum number of matches to return

        Returns:
            List of historical matches (empty if none found)
        """
        pass

    @abstractmethod
    def store_historical_match(
        self,
        match: HistoricalMatch,
        sport: Sport,
    ) -> bool:
        """
        Store a historical match result for ML training.

        Args:
            match: Historical match data
            sport: Sport type

        Returns:
            True if stored successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_training_data(
        self,
        sport: Sport,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        league: Optional[str] = None,
        min_quality: float = 0.6,
    ) -> List[HistoricalMatch]:
        """
        Get historical matches for ML training.

        Args:
            sport: Sport type
            start_date: Filter - matches after this date
            end_date: Filter - matches before this date
            league: Optional league filter
            min_quality: Minimum data quality threshold

        Returns:
            List of historical matches meeting criteria
        """
        pass

    @abstractmethod
    def get_upcoming_matches(
        self,
        sport: Sport,
        hours_ahead: int = 24,
        league: Optional[str] = None,
    ) -> List[MatchData]:
        """
        Get upcoming matches for prediction.

        Args:
            sport: Sport type
            hours_ahead: How many hours ahead to look
            league: Optional league filter

        Returns:
            List of upcoming matches with available data
        """
        pass

    @abstractmethod
    def update_match_result(
        self,
        match_id: str,
        home_goals: int,
        away_goals: int,
        home_goals_ht: Optional[int] = None,
        away_goals_ht: Optional[int] = None,
    ) -> bool:
        """
        Update match result after completion.

        Args:
            match_id: Match identifier
            home_goals: Final home team goals
            away_goals: Final away team goals
            home_goals_ht: Half-time home goals (optional)
            away_goals_ht: Half-time away goals (optional)

        Returns:
            True if updated successfully, False if match not found
        """
        pass

    @abstractmethod
    def get_data_quality(self, match_id: str) -> DataQuality:
        """
        Get data quality metrics for a match.

        Args:
            match_id: Match identifier

        Returns:
            DataQuality with completeness and freshness info
        """
        pass

    @abstractmethod
    def enrich_match_data(
        self,
        match: MatchData,
        include_h2h: bool = True,
        include_form: bool = True,
        include_odds: bool = True,
    ) -> MatchData:
        """
        Enrich match data with additional information.

        Fetches and attaches team stats, H2H history, and odds
        to the match data object.

        Args:
            match: Base match data
            include_h2h: Include head-to-head history
            include_form: Include form/stats data
            include_odds: Include betting odds

        Returns:
            Enriched MatchData with updated data_quality
        """
        pass

    @abstractmethod
    def count_matches(
        self,
        sport: Sport,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count available matches (for statistics/debugging).

        Args:
            sport: Sport type
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Number of matches matching criteria
        """
        pass


class IMatchDataProvider(ABC):
    """
    Interface for external match data providers (APIs).

    Implementations wrap specific data sources like SofaScore,
    API-Football, etc.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the data provider."""
        pass

    @property
    @abstractmethod
    def supported_sports(self) -> List[Sport]:
        """List of sports this provider supports."""
        pass

    @abstractmethod
    def fetch_upcoming_matches(
        self,
        sport: Sport,
        date: Optional[datetime] = None,
    ) -> List[MatchData]:
        """
        Fetch upcoming matches from the provider.

        Args:
            sport: Sport type
            date: Optional specific date

        Returns:
            List of matches (may have incomplete data)
        """
        pass

    @abstractmethod
    def fetch_match_details(
        self,
        match_id: str,
        sport: Sport,
    ) -> Optional[MatchData]:
        """
        Fetch detailed data for a specific match.

        Args:
            match_id: Provider-specific match ID
            sport: Sport type

        Returns:
            MatchData with details, or None if not found
        """
        pass

    @abstractmethod
    def fetch_team_stats(
        self,
        team_id: str,
        sport: Sport,
    ) -> Optional[TeamMatchStats]:
        """
        Fetch team statistics from the provider.

        Args:
            team_id: Provider-specific team ID
            sport: Sport type

        Returns:
            TeamMatchStats or None
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is currently available (health check).

        Returns:
            True if API is responding, False otherwise
        """
        pass


class IOddsProvider(ABC):
    """
    Interface for betting odds providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the odds provider."""
        pass

    @abstractmethod
    def fetch_odds(
        self,
        match_id: str,
        sport: Sport,
    ) -> Optional[dict]:
        """
        Fetch current odds for a match.

        Args:
            match_id: Match identifier
            sport: Sport type

        Returns:
            Dict with odds data or None
        """
        pass
