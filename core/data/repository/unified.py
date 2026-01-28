"""
Unified Data Repository for NEXUS ML.

Checkpoint: 0.9
Responsibility: Single source of truth for all match data.
Principle: Centralized data access with quality tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
from threading import Lock

from core.data.enums import Sport
from core.data.schemas import (
    MatchData,
    TeamData,
    TeamMatchStats,
    HistoricalMatch,
    DataQuality,
    OddsData,
)
from core.data.repository.interface import (
    IDataRepository,
    IMatchDataProvider,
    IOddsProvider,
)
from core.data.validators import validate_match, can_predict


logger = logging.getLogger(__name__)


@dataclass
class StoredMatch:
    """Internal storage wrapper for match data."""
    match: MatchData
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_completed: bool = False
    result_home_goals: Optional[int] = None
    result_away_goals: Optional[int] = None


class UnifiedDataRepository(IDataRepository):
    """
    Central repository for all match data in NEXUS ML.

    This implementation uses in-memory storage with optional
    persistence. All data access should go through this class
    to ensure consistency and quality tracking.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        providers: Optional[List[IMatchDataProvider]] = None,
        odds_providers: Optional[List[IOddsProvider]] = None,
    ):
        """
        Initialize the repository.

        Args:
            providers: List of data providers (APIs)
            odds_providers: List of odds providers
        """
        # Internal storage (by sport, then by match_id)
        self._matches: Dict[Sport, Dict[str, StoredMatch]] = {
            sport: {} for sport in Sport
        }
        self._historical: Dict[Sport, Dict[str, HistoricalMatch]] = {
            sport: {} for sport in Sport
        }
        self._team_stats_cache: Dict[str, TeamMatchStats] = {}

        # External providers
        self._providers: List[IMatchDataProvider] = providers or []
        self._odds_providers: List[IOddsProvider] = odds_providers or []

        # Thread safety
        self._lock = Lock()

        logger.info(
            f"UnifiedDataRepository initialized with "
            f"{len(self._providers)} data providers, "
            f"{len(self._odds_providers)} odds providers"
        )

    def get_match_data(
        self,
        match_id: str,
        sport: Optional[Sport] = None,
    ) -> Optional[MatchData]:
        """Get complete match data by ID."""
        with self._lock:
            if sport:
                stored = self._matches[sport].get(match_id)
                return stored.match if stored else None

            # Search all sports
            for sport_matches in self._matches.values():
                if match_id in sport_matches:
                    return sport_matches[match_id].match

        return None

    def get_team_stats(
        self,
        team_id: str,
        sport: Sport,
        num_matches: int = 5,
    ) -> Optional[TeamMatchStats]:
        """Get aggregated team statistics from recent matches."""
        cache_key = f"{sport.value}:{team_id}:{num_matches}"

        # Check cache
        if cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]

        # Calculate from historical data
        team_matches = self._get_team_matches(team_id, sport, num_matches)
        if not team_matches:
            return None

        stats = self._calculate_team_stats(team_id, team_matches)
        self._team_stats_cache[cache_key] = stats
        return stats

    def get_h2h_history(
        self,
        team1_id: str,
        team2_id: str,
        sport: Sport,
        limit: int = 10,
    ) -> List[HistoricalMatch]:
        """Get head-to-head history between two teams."""
        h2h_matches = []

        with self._lock:
            for match in self._historical[sport].values():
                is_h2h = (
                    (match.home_team_id == team1_id and match.away_team_id == team2_id) or
                    (match.home_team_id == team2_id and match.away_team_id == team1_id)
                )
                if is_h2h:
                    h2h_matches.append(match)

        # Sort by date descending and limit
        h2h_matches.sort(key=lambda m: m.date, reverse=True)
        return h2h_matches[:limit]

    def store_historical_match(
        self,
        match: HistoricalMatch,
        sport: Sport,
    ) -> bool:
        """Store a historical match result for ML training."""
        if not match.match_id:
            logger.warning("Cannot store match without match_id")
            return False

        with self._lock:
            self._historical[sport][match.match_id] = match

            # Invalidate related cache
            self._invalidate_team_cache(match.home_team_id, sport)
            self._invalidate_team_cache(match.away_team_id, sport)

        logger.debug(f"Stored historical match: {match.match_id}")
        return True

    def get_training_data(
        self,
        sport: Sport,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        league: Optional[str] = None,
        min_quality: float = 0.6,
    ) -> List[HistoricalMatch]:
        """Get historical matches for ML training."""
        matches = []

        with self._lock:
            for match in self._historical[sport].values():
                # Date filter
                if start_date and match.date < start_date:
                    continue
                if end_date and match.date > end_date:
                    continue

                # League filter
                if league and match.league != league:
                    continue

                matches.append(match)

        # Sort by date for time-series consistency
        matches.sort(key=lambda m: m.date)
        return matches

    def get_upcoming_matches(
        self,
        sport: Sport,
        hours_ahead: int = 24,
        league: Optional[str] = None,
    ) -> List[MatchData]:
        """Get upcoming matches for prediction."""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []

        with self._lock:
            for stored in self._matches[sport].values():
                if stored.is_completed:
                    continue

                match = stored.match
                if not (now <= match.start_time <= cutoff):
                    continue

                if league and match.league != league:
                    continue

                upcoming.append(match)

        # Sort by start time
        upcoming.sort(key=lambda m: m.start_time)
        return upcoming

    def update_match_result(
        self,
        match_id: str,
        home_goals: int,
        away_goals: int,
        home_goals_ht: Optional[int] = None,
        away_goals_ht: Optional[int] = None,
    ) -> bool:
        """Update match result after completion."""
        with self._lock:
            for sport in Sport:
                if match_id in self._matches[sport]:
                    stored = self._matches[sport][match_id]
                    stored.is_completed = True
                    stored.result_home_goals = home_goals
                    stored.result_away_goals = away_goals
                    stored.updated_at = datetime.utcnow()

                    # Also create historical record
                    match = stored.match
                    historical = HistoricalMatch(
                        match_id=match_id,
                        date=match.start_time,
                        home_team_id=match.home_team.team_id,
                        away_team_id=match.away_team.team_id,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        home_goals_ht=home_goals_ht,
                        away_goals_ht=away_goals_ht,
                        league=match.league,
                    )
                    self._historical[sport][match_id] = historical

                    logger.info(f"Updated match result: {match_id} = {home_goals}-{away_goals}")
                    return True

        logger.warning(f"Match not found for result update: {match_id}")
        return False

    def get_data_quality(self, match_id: str) -> DataQuality:
        """Get data quality metrics for a match."""
        match = self.get_match_data(match_id)
        if match:
            return match.data_quality

        return DataQuality(
            completeness=0.0,
            freshness_hours=999,
            sources_count=0,
        )

    def enrich_match_data(
        self,
        match: MatchData,
        include_h2h: bool = True,
        include_form: bool = True,
        include_odds: bool = True,
    ) -> MatchData:
        """Enrich match data with additional information."""
        enriched = match
        sources = set()

        # H2H History
        if include_h2h and not match.h2h_history:
            h2h = self.get_h2h_history(
                match.home_team.team_id,
                match.away_team.team_id,
                match.sport,
            )
            if h2h:
                enriched.h2h_history = h2h
                sources.add("h2h")

        # Team stats/form
        if include_form:
            if not match.home_stats:
                home_stats = self.get_team_stats(
                    match.home_team.team_id,
                    match.sport,
                )
                if home_stats:
                    enriched.home_stats = home_stats
                    sources.add("form")

            if not match.away_stats:
                away_stats = self.get_team_stats(
                    match.away_team.team_id,
                    match.sport,
                )
                if away_stats:
                    enriched.away_stats = away_stats
                    sources.add("form")

        # Odds
        if include_odds and not match.odds:
            for provider in self._odds_providers:
                odds = provider.fetch_odds(match.match_id, match.sport)
                if odds:
                    enriched.odds = self._parse_odds(odds)
                    sources.add("odds")
                    break

        # Update quality metrics
        enriched.data_quality = self._calculate_data_quality(enriched, sources)

        # Store enriched version
        self._store_match(enriched)

        return enriched

    def count_matches(
        self,
        sport: Sport,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count available matches."""
        count = 0
        with self._lock:
            for match in self._historical[sport].values():
                if start_date and match.date < start_date:
                    continue
                if end_date and match.date > end_date:
                    continue
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Additional public methods
    # -------------------------------------------------------------------------

    def add_provider(self, provider: IMatchDataProvider) -> None:
        """Add a data provider at runtime."""
        self._providers.append(provider)
        logger.info(f"Added provider: {provider.provider_name}")

    def add_odds_provider(self, provider: IOddsProvider) -> None:
        """Add an odds provider at runtime."""
        self._odds_providers.append(provider)
        logger.info(f"Added odds provider: {provider.provider_name}")

    def refresh_from_providers(
        self,
        sport: Sport,
        date: Optional[datetime] = None,
    ) -> int:
        """
        Fetch and store new matches from all providers.

        Returns:
            Number of new matches stored
        """
        new_count = 0

        for provider in self._providers:
            if sport not in provider.supported_sports:
                continue

            try:
                matches = provider.fetch_upcoming_matches(sport, date)
                for match in matches:
                    if self._store_match(match):
                        new_count += 1
            except Exception as e:
                logger.error(f"Error fetching from {provider.provider_name}: {e}")

        logger.info(f"Refreshed {new_count} new matches for {sport.value}")
        return new_count

    def get_stats(self) -> Dict[str, int]:
        """Get repository statistics."""
        with self._lock:
            return {
                "total_upcoming": sum(
                    len([m for m in matches.values() if not m.is_completed])
                    for matches in self._matches.values()
                ),
                "total_historical": sum(
                    len(matches) for matches in self._historical.values()
                ),
                "providers": len(self._providers),
                "odds_providers": len(self._odds_providers),
                "cache_entries": len(self._team_stats_cache),
            }

    def validate_for_prediction(self, match_id: str) -> tuple:
        """
        Check if match has sufficient data for ML prediction.

        Returns:
            (can_predict: bool, reason: str)
        """
        match = self.get_match_data(match_id)
        if not match:
            return False, f"Match not found: {match_id}"

        return can_predict(match)

    def clear_cache(self) -> None:
        """Clear all caches (for testing/refresh)."""
        with self._lock:
            self._team_stats_cache.clear()
        logger.info("Cache cleared")

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _store_match(self, match: MatchData) -> bool:
        """Store or update a match."""
        with self._lock:
            existing = self._matches[match.sport].get(match.match_id)

            # Don't overwrite more complete data with less complete
            if existing and existing.match.data_quality.completeness > match.data_quality.completeness:
                return False

            self._matches[match.sport][match.match_id] = StoredMatch(
                match=match,
                created_at=existing.created_at if existing else datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            return True

    def _get_team_matches(
        self,
        team_id: str,
        sport: Sport,
        limit: int,
    ) -> List[HistoricalMatch]:
        """Get recent matches for a team."""
        matches = []
        with self._lock:
            for match in self._historical[sport].values():
                if match.home_team_id == team_id or match.away_team_id == team_id:
                    matches.append(match)

        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]

    def _calculate_team_stats(
        self,
        team_id: str,
        matches: List[HistoricalMatch],
    ) -> TeamMatchStats:
        """Calculate team stats from historical matches."""
        if not matches:
            return TeamMatchStats(
                goals_scored_avg=0.0,
                goals_conceded_avg=0.0,
            )

        goals_scored = []
        goals_conceded = []
        home_goals = []
        away_goals = []
        points = []

        for match in matches:
            is_home = match.home_team_id == team_id

            if is_home:
                scored = match.home_goals
                conceded = match.away_goals
                home_goals.append(scored)
            else:
                scored = match.away_goals
                conceded = match.home_goals
                away_goals.append(scored)

            goals_scored.append(scored)
            goals_conceded.append(conceded)

            # Points: win=3, draw=1, loss=0
            if scored > conceded:
                points.append(3)
            elif scored == conceded:
                points.append(1)
            else:
                points.append(0)

        # Calculate rest days from most recent match
        rest_days = 0
        if matches:
            last_match = max(matches, key=lambda m: m.date)
            rest_days = (datetime.utcnow() - last_match.date).days

        max_points = len(matches) * 3
        form_points = sum(points) / max_points if max_points > 0 else 0.0

        return TeamMatchStats(
            goals_scored_avg=sum(goals_scored) / len(goals_scored),
            goals_conceded_avg=sum(goals_conceded) / len(goals_conceded),
            home_goals_avg=sum(home_goals) / len(home_goals) if home_goals else None,
            away_goals_avg=sum(away_goals) / len(away_goals) if away_goals else None,
            form_points=form_points,
            rest_days=rest_days,
        )

    def _calculate_data_quality(
        self,
        match: MatchData,
        sources: Set[str],
    ) -> DataQuality:
        """Calculate data quality metrics for a match."""
        completeness = 0.0
        fields_present = 0
        total_fields = 6

        if match.home_team:
            fields_present += 1
        if match.away_team:
            fields_present += 1
        if match.home_stats:
            fields_present += 1
        if match.away_stats:
            fields_present += 1
        if match.h2h_history:
            fields_present += 1
        if match.odds:
            fields_present += 1

        completeness = fields_present / total_fields

        return DataQuality(
            completeness=completeness,
            freshness_hours=0,  # Just calculated
            sources_count=len(sources) + 1,  # +1 for base data
            has_h2h=match.h2h_history is not None and len(match.h2h_history) > 0,
            has_form=match.home_stats is not None and match.away_stats is not None,
            has_odds=match.odds is not None,
        )

    def _invalidate_team_cache(self, team_id: str, sport: Sport) -> None:
        """Invalidate cached stats for a team."""
        prefix = f"{sport.value}:{team_id}:"
        keys_to_remove = [k for k in self._team_stats_cache if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._team_stats_cache[key]

    def _parse_odds(self, odds_dict: dict) -> OddsData:
        """Parse odds dictionary into OddsData."""
        return OddsData(
            home_win=odds_dict.get("home_win", 0.0),
            draw=odds_dict.get("draw"),
            away_win=odds_dict.get("away_win", 0.0),
            over_25=odds_dict.get("over_25"),
            under_25=odds_dict.get("under_25"),
            handicap_line=odds_dict.get("handicap_line"),
            handicap_home=odds_dict.get("handicap_home"),
            handicap_away=odds_dict.get("handicap_away"),
            bookmaker=odds_dict.get("bookmaker", "unknown"),
            timestamp=datetime.utcnow(),
        )


# Singleton instance for convenience
_default_repository: Optional[UnifiedDataRepository] = None


def get_repository() -> UnifiedDataRepository:
    """Get or create the default repository instance."""
    global _default_repository
    if _default_repository is None:
        _default_repository = UnifiedDataRepository()
    return _default_repository


def reset_repository() -> None:
    """Reset the default repository (for testing)."""
    global _default_repository
    _default_repository = None
