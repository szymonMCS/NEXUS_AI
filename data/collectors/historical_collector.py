"""
Historical Data Collector.

Checkpoint: 5.2
Responsibility: Collect historical match data for ML training.

Sources (priority):
1. API-Football (via API-Sports) - detailed stats
2. Football-Data.org - European leagues
3. MLB Stats API - Baseball
4. API-Sports Basketball - NBA and leagues
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

from core.data.enums import Sport
from data.collectors.results import (
    CollectionResult,
    CollectionStatus,
    CollectedMatch,
    SourceResult,
    CollectionConfig,
    DEFAULT_LEAGUES,
)
from data.apis import APITierManager, APIResponse

logger = logging.getLogger(__name__)


class DataSourceAdapter(ABC):
    """Abstract adapter for converting API responses to CollectedMatch."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source."""
        pass

    @abstractmethod
    async def fetch_matches(
        self,
        api_manager: APITierManager,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CollectedMatch]:
        """Fetch matches from the source."""
        pass


class FootballDataAdapter(DataSourceAdapter):
    """Adapter for Football-Data.org API."""

    # Mapping from our league codes to Football-Data.org codes
    LEAGUE_CODES = {
        "PL": "PL",           # Premier League
        "LaLiga": "PD",       # La Liga (Primera División)
        "SerieA": "SA",       # Serie A
        "Bundesliga": "BL1",  # Bundesliga
        "Ligue1": "FL1",      # Ligue 1
        "CL": "CL",           # Champions League
        "EL": "EL",           # Europa League
        "EC": "EC",           # European Championship
        "WC": "WC",           # World Cup
    }

    @property
    def source_name(self) -> str:
        return "football_data"

    async def fetch_matches(
        self,
        api_manager: APITierManager,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CollectedMatch]:
        if sport != Sport.FOOTBALL:
            return []

        # Map league code to Football-Data.org format
        fd_league = self.LEAGUE_CODES.get(league, league)

        matches = []
        response = await api_manager.football_data.get_matches(
            competition=fd_league,
            date_from=start_date.strftime("%Y-%m-%d"),
            date_to=end_date.strftime("%Y-%m-%d"),
        )

        if not response.success or not response.data:
            logger.warning(f"Football-Data.org failed for {league}: {response.error}")
            return []

        raw_matches = response.data.get("matches", [])
        for raw in raw_matches:
            try:
                # Only collect finished matches
                if raw.get("status") != "FINISHED":
                    continue

                score = raw.get("score", {})
                full_time = score.get("fullTime", {})
                half_time = score.get("halfTime", {})

                match = CollectedMatch(
                    match_id=f"fd_{raw.get('id')}",
                    source=self.source_name,
                    sport=Sport.FOOTBALL,
                    league=league,
                    season=str(raw.get("season", {}).get("startDate", "")[:4]),
                    match_date=datetime.fromisoformat(raw.get("utcDate", "").replace("Z", "+00:00")),
                    home_team_id=str(raw.get("homeTeam", {}).get("id", "")),
                    home_team_name=raw.get("homeTeam", {}).get("name", ""),
                    away_team_id=str(raw.get("awayTeam", {}).get("id", "")),
                    away_team_name=raw.get("awayTeam", {}).get("name", ""),
                    home_goals=full_time.get("home", 0) or 0,
                    away_goals=full_time.get("away", 0) or 0,
                    home_goals_ht=half_time.get("home"),
                    away_goals_ht=half_time.get("away"),
                )
                matches.append(match)
            except Exception as e:
                logger.debug(f"Error parsing match: {e}")
                continue

        return matches


class APISportsFootballAdapter(DataSourceAdapter):
    """Adapter for API-Sports Football (via API-Football)."""

    LEAGUE_IDS = {
        "PL": 39,       # Premier League
        "LaLiga": 140,  # La Liga
        "SerieA": 135,  # Serie A
        "Bundesliga": 78,  # Bundesliga
        "Ligue1": 61,   # Ligue 1
        "CL": 2,        # Champions League
        "EL": 3,        # Europa League
    }

    @property
    def source_name(self) -> str:
        return "api_football"

    async def fetch_matches(
        self,
        api_manager: APITierManager,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CollectedMatch]:
        if sport != Sport.FOOTBALL:
            return []

        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            logger.warning(f"Unknown league for API-Football: {league}")
            return []

        matches = []

        # API-Football uses api_football_pro client
        if not api_manager.api_football_pro.is_configured:
            # Fall back to api_sports for football
            response = await api_manager.api_sports.get_games(
                sport="football",
                league_id=league_id,
                season=str(start_date.year),
            )
        else:
            response = await api_manager.api_football_pro.get_fixtures(
                league_id=league_id,
                season=start_date.year,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )

        if not response.success or not response.data:
            logger.warning(f"API-Football failed for {league}: {response.error}")
            return []

        raw_matches = response.data if isinstance(response.data, list) else []
        for raw in raw_matches:
            try:
                fixture = raw.get("fixture", {})
                teams = raw.get("teams", {})
                goals = raw.get("goals", {})
                score = raw.get("score", {})

                # Only finished matches
                status = fixture.get("status", {}).get("short", "")
                if status != "FT":
                    continue

                ht_score = score.get("halftime", {})
                stats = raw.get("statistics", [])

                match = CollectedMatch(
                    match_id=f"apif_{fixture.get('id')}",
                    source=self.source_name,
                    sport=Sport.FOOTBALL,
                    league=league,
                    season=str(raw.get("league", {}).get("season", "")),
                    match_date=datetime.fromisoformat(fixture.get("date", "").replace("Z", "+00:00")),
                    home_team_id=str(teams.get("home", {}).get("id", "")),
                    home_team_name=teams.get("home", {}).get("name", ""),
                    away_team_id=str(teams.get("away", {}).get("id", "")),
                    away_team_name=teams.get("away", {}).get("name", ""),
                    home_goals=goals.get("home", 0) or 0,
                    away_goals=goals.get("away", 0) or 0,
                    home_goals_ht=ht_score.get("home"),
                    away_goals_ht=ht_score.get("away"),
                )

                # Extract stats if available
                if stats:
                    match = self._add_stats(match, stats)

                matches.append(match)
            except Exception as e:
                logger.debug(f"Error parsing API-Football match: {e}")
                continue

        return matches

    def _add_stats(self, match: CollectedMatch, stats: List[Dict]) -> CollectedMatch:
        """Add detailed statistics to match."""
        home_stats = stats[0] if len(stats) > 0 else {}
        away_stats = stats[1] if len(stats) > 1 else {}

        def get_stat(team_stats: Dict, stat_type: str) -> Optional[Any]:
            for stat in team_stats.get("statistics", []):
                if stat.get("type") == stat_type:
                    return stat.get("value")
            return None

        match.home_shots = get_stat(home_stats, "Total Shots")
        match.away_shots = get_stat(away_stats, "Total Shots")
        match.home_shots_on_target = get_stat(home_stats, "Shots on Goal")
        match.away_shots_on_target = get_stat(away_stats, "Shots on Goal")

        possession_home = get_stat(home_stats, "Ball Possession")
        if possession_home:
            match.home_possession = float(str(possession_home).replace("%", "")) / 100

        possession_away = get_stat(away_stats, "Ball Possession")
        if possession_away:
            match.away_possession = float(str(possession_away).replace("%", "")) / 100

        match.home_corners = get_stat(home_stats, "Corner Kicks")
        match.away_corners = get_stat(away_stats, "Corner Kicks")
        match.home_fouls = get_stat(home_stats, "Fouls")
        match.away_fouls = get_stat(away_stats, "Fouls")
        match.home_yellow_cards = get_stat(home_stats, "Yellow Cards")
        match.away_yellow_cards = get_stat(away_stats, "Yellow Cards")
        match.home_red_cards = get_stat(home_stats, "Red Cards")
        match.away_red_cards = get_stat(away_stats, "Red Cards")

        return match


class APISportsBasketballAdapter(DataSourceAdapter):
    """Adapter for API-Sports Basketball."""

    LEAGUE_IDS = {
        "NBA": 12,
        "EuroLeague": 120,
    }

    @property
    def source_name(self) -> str:
        return "api_basketball"

    def _get_nba_seasons(self, start_date: datetime, end_date: datetime) -> List[str]:
        """
        Get NBA season strings covering a date range.
        NBA season runs October-June, so 2024-2025 covers Oct 2024 to June 2025.
        """
        # Helper to get season for a date
        def date_to_season_year(date: datetime) -> int:
            if date.month >= 10:  # Oct-Dec -> current year season
                return date.year
            else:  # Jan-Sep -> previous year season
                return date.year - 1

        # Get start and end season years
        start_year = date_to_season_year(start_date)
        end_year = date_to_season_year(end_date)

        # Add all seasons in range
        seasons = []
        for year in range(start_year, end_year + 1):
            seasons.append(f"{year}-{year + 1}")

        return seasons

    async def fetch_matches(
        self,
        api_manager: APITierManager,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CollectedMatch]:
        if sport != Sport.BASKETBALL:
            return []

        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []

        matches = []

        # Get all relevant NBA seasons for the date range
        seasons = self._get_nba_seasons(start_date, end_date)
        logger.info(f"Fetching NBA seasons: {seasons}")

        for season in seasons:
            response = await api_manager.api_sports.get_games(
                sport="basketball",
                league_id=league_id,
                season=season,
            )

            if not response.success or not response.data:
                continue

            for raw in response.data:
                try:
                    status = raw.get("status", {}).get("short", "")
                    if status != "FT":
                        continue

                    # Filter by date range
                    match_date = datetime.fromisoformat(raw.get("date", "").replace("Z", "+00:00"))
                    match_date_naive = match_date.replace(tzinfo=None)
                    if match_date_naive < start_date or match_date_naive > end_date:
                        continue

                    scores = raw.get("scores", {})
                    home_score = scores.get("home", {}).get("total", 0) or 0
                    away_score = scores.get("away", {}).get("total", 0) or 0

                    match = CollectedMatch(
                        match_id=f"apib_{raw.get('id')}",
                        source=self.source_name,
                        sport=Sport.BASKETBALL,
                        league=league,
                        season=str(raw.get("league", {}).get("season", "")),
                        match_date=match_date,
                        home_team_id=str(raw.get("teams", {}).get("home", {}).get("id", "")),
                        home_team_name=raw.get("teams", {}).get("home", {}).get("name", ""),
                        away_team_id=str(raw.get("teams", {}).get("away", {}).get("id", "")),
                        away_team_name=raw.get("teams", {}).get("away", {}).get("name", ""),
                        home_goals=home_score,
                        away_goals=away_score,
                    )
                    matches.append(match)
                except Exception as e:
                    logger.debug(f"Error parsing basketball match: {e}")
                    continue

        return matches


class MLBAdapter(DataSourceAdapter):
    """Adapter for MLB Stats API."""

    @property
    def source_name(self) -> str:
        return "mlb_stats"

    async def fetch_matches(
        self,
        api_manager: APITierManager,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CollectedMatch]:
        # MLB uses a different sport type - we'll treat it as special case
        if sport != Sport.FOOTBALL:  # Using FOOTBALL as placeholder for "other"
            return []

        if league != "MLB":
            return []

        matches = []
        response = await api_manager.mlb_stats.get_schedule(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if not response.success or not response.data:
            return []

        for date_entry in response.data:
            for game in date_entry.get("games", []):
                try:
                    status = game.get("status", {}).get("codedGameState", "")
                    if status != "F":  # Final
                        continue

                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})

                    match = CollectedMatch(
                        match_id=f"mlb_{game.get('gamePk')}",
                        source=self.source_name,
                        sport=Sport.FOOTBALL,  # Placeholder
                        league="MLB",
                        season=str(game.get("season", "")),
                        match_date=datetime.fromisoformat(game.get("gameDate", "").replace("Z", "+00:00")),
                        home_team_id=str(home.get("team", {}).get("id", "")),
                        home_team_name=home.get("team", {}).get("name", ""),
                        away_team_id=str(away.get("team", {}).get("id", "")),
                        away_team_name=away.get("team", {}).get("name", ""),
                        home_goals=home.get("score", 0) or 0,
                        away_goals=away.get("score", 0) or 0,
                    )
                    matches.append(match)
                except Exception as e:
                    logger.debug(f"Error parsing MLB game: {e}")
                    continue

        return matches


class HistoricalDataCollector:
    """
    Collector for historical match data.

    Uses multiple data sources with automatic fallback.
    Deduplicates matches and stores for ML training.

    Usage:
        collector = HistoricalDataCollector()
        result = await collector.collect(
            sport=Sport.FOOTBALL,
            league="PL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )
    """

    def __init__(self, config: Optional[CollectionConfig] = None):
        self.config = config or CollectionConfig()
        self._api_manager: Optional[APITierManager] = None

        # Initialize adapters
        self._adapters: Dict[Sport, List[DataSourceAdapter]] = {
            Sport.FOOTBALL: [
                APISportsFootballAdapter(),
                FootballDataAdapter(),
            ],
            Sport.BASKETBALL: [
                APISportsBasketballAdapter(),
            ],
        }

    async def _get_api_manager(self) -> APITierManager:
        """Get or create API manager."""
        if self._api_manager is None:
            self._api_manager = APITierManager()
        return self._api_manager

    async def close(self):
        """Close connections."""
        if self._api_manager:
            await self._api_manager.close()
            self._api_manager = None

    async def collect(
        self,
        sport: Sport,
        league: str,
        start_date: datetime,
        end_date: datetime,
    ) -> CollectionResult:
        """
        Collect historical data for a sport/league.

        Args:
            sport: Sport type
            league: League code (e.g., "PL", "NBA")
            start_date: Start of date range
            end_date: End of date range

        Returns:
            CollectionResult with collected matches
        """
        collection_id = str(uuid.uuid4())[:8]
        result = CollectionResult(
            collection_id=collection_id,
            sport=sport,
            league=league,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Starting collection {collection_id}: {sport.value}/{league} "
                    f"from {start_date.date()} to {end_date.date()}")

        api_manager = await self._get_api_manager()
        adapters = self._adapters.get(sport, [])

        if not adapters:
            result.add_error(f"No adapters configured for sport: {sport}")
            result.finalize()
            return result

        collected_ids = set()

        for adapter in adapters:
            source_start = datetime.utcnow()
            source_result = SourceResult(source_name=adapter.source_name, status=CollectionStatus.SUCCESS)

            try:
                matches = await adapter.fetch_matches(
                    api_manager=api_manager,
                    sport=sport,
                    league=league,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Deduplicate and add matches
                for match in matches:
                    # Create unique key for deduplication
                    match_key = f"{match.home_team_name}_{match.away_team_name}_{match.match_date.date()}"

                    if match_key not in collected_ids:
                        collected_ids.add(match_key)
                        result.add_match(match)
                        source_result.records_collected += 1
                    else:
                        logger.debug(f"Duplicate match skipped: {match_key}")

                source_result.duration_seconds = (datetime.utcnow() - source_start).total_seconds()
                logger.info(f"  {adapter.source_name}: collected {source_result.records_collected} matches")

            except Exception as e:
                source_result.status = CollectionStatus.FAILED
                source_result.error_message = str(e)
                result.add_error(f"{adapter.source_name}: {e}")
                logger.error(f"  {adapter.source_name}: error - {e}")

            result.add_source_result(source_result)

            # Rate limiting between sources
            await asyncio.sleep(self.config.delay_between_leagues)

        result.finalize()
        logger.info(f"Collection {collection_id} complete: {result.total_collected} matches, "
                    f"{result.total_errors} errors, {result.duration_seconds:.1f}s")

        return result

    async def collect_multiple_leagues(
        self,
        sport: Sport,
        leagues: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CollectionResult]:
        """
        Collect data for multiple leagues.

        Args:
            sport: Sport type
            leagues: List of league codes (uses defaults if None)
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)

        Returns:
            List of CollectionResults
        """
        if leagues is None:
            leagues = DEFAULT_LEAGUES.get(sport, [])

        if end_date is None:
            end_date = datetime.utcnow()

        if start_date is None:
            start_date = end_date - timedelta(days=365)

        results = []
        for league in leagues:
            result = await self.collect(sport, league, start_date, end_date)
            results.append(result)

            # Rate limiting between leagues
            await asyncio.sleep(self.config.delay_between_leagues)

        return results

    async def collect_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[Sport, List[CollectionResult]]:
        """
        Collect data for all configured sports and leagues.

        Returns:
            Dictionary mapping Sport to list of CollectionResults
        """
        all_results = {}

        for sport in self.config.sports:
            leagues = self.config.leagues.get(sport) or DEFAULT_LEAGUES.get(sport, [])
            results = await self.collect_multiple_leagues(
                sport=sport,
                leagues=leagues,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[sport] = results

        return all_results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions
async def collect_football_history(
    leagues: Optional[List[str]] = None,
    days_back: int = 365,
) -> List[CollectionResult]:
    """Quick function to collect football history."""
    async with HistoricalDataCollector() as collector:
        return await collector.collect_multiple_leagues(
            sport=Sport.FOOTBALL,
            leagues=leagues,
            start_date=datetime.utcnow() - timedelta(days=days_back),
            end_date=datetime.utcnow(),
        )


async def collect_basketball_history(
    leagues: Optional[List[str]] = None,
    days_back: int = 365,
) -> List[CollectionResult]:
    """Quick function to collect basketball history."""
    async with HistoricalDataCollector() as collector:
        return await collector.collect_multiple_leagues(
            sport=Sport.BASKETBALL,
            leagues=leagues,
            start_date=datetime.utcnow() - timedelta(days=days_back),
            end_date=datetime.utcnow(),
        )
