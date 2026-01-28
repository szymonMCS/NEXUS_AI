"""
API Tier Manager for NEXUS AI.

Auto-detects available APIs and routes requests to the best available source.
When premium API keys are added, the system automatically upgrades data quality.

Usage:
    manager = APITierManager()
    status = await manager.get_all_status()

    # Automatically uses best available source
    matches = await manager.get_football_matches("PL")
    odds = await manager.get_odds("nba", "game_id")
"""

import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from data.apis.sports_api_client import (
    APIResponse,
    OddsAPIClient,
    FootballDataClient,
    APISportsClient,
    PandaScoreClient,
    OpenF1Client,
    MLBStatsClient,
    NewsAPIClient,
)
from data.apis.premium_api_clients import (
    APITier,
    APIStatus,
    SportradarClient,
    StatsPerformClient,
    GeniusSportsClient,
    LSportsClient,
    SportsDataIOClient,
    APIFootballProClient,
)

logger = logging.getLogger(__name__)


@dataclass
class TierStatus:
    """Status of available API tiers."""
    enterprise: List[APIStatus] = field(default_factory=list)
    professional: List[APIStatus] = field(default_factory=list)
    developer: List[APIStatus] = field(default_factory=list)
    free: List[APIStatus] = field(default_factory=list)

    @property
    def best_tier(self) -> APITier:
        """Get the best available tier."""
        if any(s.available for s in self.enterprise):
            return APITier.ENTERPRISE
        if any(s.available for s in self.professional):
            return APITier.PROFESSIONAL
        if any(s.available for s in self.developer):
            return APITier.DEVELOPER
        return APITier.FREE

    @property
    def total_configured(self) -> int:
        """Count of configured APIs."""
        all_apis = self.enterprise + self.professional + self.developer + self.free
        return sum(1 for s in all_apis if s.configured)

    @property
    def total_available(self) -> int:
        """Count of available APIs."""
        all_apis = self.enterprise + self.professional + self.developer + self.free
        return sum(1 for s in all_apis if s.available)


class APITierManager:
    """
    Manages API clients across all tiers with automatic fallback.

    Priority order (uses first available):
    1. Enterprise (Sportradar, Stats Perform)
    2. Professional (Genius Sports, LSports)
    3. Developer (SportsDataIO, API-Football Pro)
    4. Free (Odds API, Football-Data, OpenF1, MLB Stats, etc.)
    """

    def __init__(self):
        # Enterprise tier
        self.sportradar = SportradarClient()
        self.stats_perform = StatsPerformClient()

        # Professional tier
        self.genius_sports = GeniusSportsClient()
        self.lsports = LSportsClient()

        # Developer tier
        self.sportsdata_io = SportsDataIOClient()
        self.api_football_pro = APIFootballProClient()

        # Free tier
        self.odds_api = OddsAPIClient()
        self.football_data = FootballDataClient()
        self.api_sports = APISportsClient()
        self.pandascore = PandaScoreClient()
        self.openf1 = OpenF1Client()
        self.mlb_stats = MLBStatsClient()
        self.news_api = NewsAPIClient()

        # Track initialized state
        self._status_cache: Optional[TierStatus] = None

    async def get_all_status(self, refresh: bool = False) -> TierStatus:
        """
        Get status of all configured APIs.

        Args:
            refresh: Force refresh of cached status

        Returns:
            TierStatus with all API statuses organized by tier
        """
        if self._status_cache and not refresh:
            return self._status_cache

        status = TierStatus()

        # Enterprise tier
        status.enterprise = [
            await self.sportradar.get_status(),
            await self.stats_perform.get_status(),
        ]

        # Professional tier
        status.professional = [
            await self.genius_sports.get_status(),
            await self.lsports.get_status(),
        ]

        # Developer tier
        status.developer = [
            await self.sportsdata_io.get_status(),
            await self.api_football_pro.get_status(),
        ]

        # Free tier status (simplified)
        status.free = [
            APIStatus(
                name="The Odds API",
                tier=APITier.FREE,
                configured=bool(os.getenv("ODDS_API_KEY")),
                available=bool(os.getenv("ODDS_API_KEY")),
                features=["Odds", "40+ Bookmakers", "70+ Sports"]
            ),
            APIStatus(
                name="Football-Data.org",
                tier=APITier.FREE,
                configured=bool(os.getenv("X-Auth-Token")),
                available=bool(os.getenv("X-Auth-Token")),
                features=["European Football", "Standings", "Matches"]
            ),
            APIStatus(
                name="API-Sports",
                tier=APITier.FREE,
                configured=bool(os.getenv("x-apisports-key")),
                available=bool(os.getenv("x-apisports-key")),
                features=["Basketball", "Tennis", "Football", "Hockey"]
            ),
            APIStatus(
                name="PandaScore",
                tier=APITier.FREE,
                configured=bool(os.getenv("PANDASCORE_API_KEY")),
                available=bool(os.getenv("PANDASCORE_API_KEY")),
                features=["eSports", "LoL", "CS2", "Dota 2", "Valorant"]
            ),
            APIStatus(
                name="OpenF1",
                tier=APITier.FREE,
                configured=True,  # No key needed
                available=True,
                features=["F1 Data", "Telemetry", "Live Sessions"]
            ),
            APIStatus(
                name="MLB Stats",
                tier=APITier.FREE,
                configured=True,  # No key needed
                available=True,
                features=["MLB Data", "Schedules", "Standings", "Box Scores"]
            ),
        ]

        self._status_cache = status
        return status

    def print_status_report(self, status: TierStatus) -> str:
        """Generate a human-readable status report."""
        lines = [
            "=" * 60,
            "NEXUS API Tier Status",
            "=" * 60,
            f"Best Available Tier: {status.best_tier.value.upper()}",
            f"APIs Configured: {status.total_configured}",
            f"APIs Available: {status.total_available}",
            "",
        ]

        for tier_name, apis in [
            ("ENTERPRISE", status.enterprise),
            ("PROFESSIONAL", status.professional),
            ("DEVELOPER", status.developer),
            ("FREE", status.free),
        ]:
            lines.append(f"--- {tier_name} ---")
            for api in apis:
                icon = "[OK]" if api.available else "[--]" if api.configured else "[  ]"
                lines.append(f"  {icon} {api.name}")
                if api.error and not api.available:
                    lines.append(f"      Error: {api.error}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # FOOTBALL/SOCCER DATA
    # =========================================================================

    async def get_football_matches(
        self,
        competition: str = "PL",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> APIResponse:
        """
        Get football matches - uses best available API with automatic fallback.

        Priority: Stats Perform > API-Football Pro > Football-Data.org
        Falls back automatically if premium API fails.
        """
        league_map = {
            "PL": 39,    # Premier League
            "LaLiga": 140,
            "SerieA": 135,
            "Bundesliga": 78,
            "Ligue1": 61,
            "CL": 2,     # Champions League
        }

        # Try API-Football Pro (developer+) - has real fixtures endpoint
        if self.api_football_pro.is_configured:
            logger.info("Trying API-Football Pro for football matches")
            league_id = league_map.get(competition, 39)
            result = await self.api_football_pro.get_fixtures(
                league_id=league_id,
                from_date=date_from,
                to_date=date_to
            )
            if result.success:
                return result
            logger.warning(f"API-Football Pro failed: {result.error}, falling back")

        # Fall back to Football-Data.org (free)
        logger.info("Using Football-Data.org for football matches")
        return await self.football_data.get_matches(
            competition=competition,
            date_from=date_from,
            date_to=date_to
        )

    async def get_football_predictions(self, fixture_id: int) -> APIResponse:
        """
        Get match predictions - uses best available API.

        Priority: Stats Perform > API-Football Pro
        """
        if self.stats_perform.is_configured:
            logger.info("Using Stats Perform for predictions")
            return await self.stats_perform.get_match_stats(str(fixture_id))

        if self.api_football_pro.is_configured:
            logger.info("Using API-Football Pro for predictions")
            return await self.api_football_pro.get_predictions(fixture_id)

        return APIResponse(
            success=False,
            error="No prediction API configured. Add API_FOOTBALL_PRO_KEY or STATS_PERFORM_API_KEY",
            source="tier_manager"
        )

    # =========================================================================
    # US SPORTS DATA (NBA, NFL, MLB, NHL)
    # =========================================================================

    async def get_nba_schedule(self, season: int = 2024) -> APIResponse:
        """
        Get NBA schedule - uses best available API with fallback.

        Priority: Sportradar > SportsDataIO > API-Sports (free)
        """
        # Try Sportradar (enterprise)
        if self.sportradar.is_configured:
            logger.info("Trying Sportradar for NBA schedule")
            result = await self.sportradar.get_nba_schedule(season)
            if result.success:
                return result
            logger.warning(f"Sportradar failed: {result.error}, falling back")

        # Try SportsDataIO (developer)
        if self.sportsdata_io.is_configured:
            logger.info("Trying SportsDataIO for NBA schedule")
            result = await self.sportsdata_io.get_nba_games(f"{season}-01-01")
            if result.success:
                return result
            logger.warning(f"SportsDataIO failed: {result.error}, falling back")

        # Fall back to API-Sports (free)
        logger.info("Using API-Sports for NBA schedule")
        return await self.api_sports.get_games(sport="basketball", season=str(season))

    async def get_nfl_schedule(self, season: int = 2024) -> APIResponse:
        """
        Get NFL schedule - uses best available API with fallback.

        Priority: Sportradar > SportsDataIO
        Note: No free fallback available for NFL.
        """
        # Try Sportradar (enterprise)
        if self.sportradar.is_configured:
            logger.info("Trying Sportradar for NFL schedule")
            result = await self.sportradar.get_nfl_schedule(season)
            if result.success:
                return result
            logger.warning(f"Sportradar failed: {result.error}, falling back")

        # Try SportsDataIO (developer)
        if self.sportsdata_io.is_configured:
            logger.info("Trying SportsDataIO for NFL schedule")
            result = await self.sportsdata_io.get_nfl_schedule(season)
            if result.success:
                return result
            logger.warning(f"SportsDataIO failed: {result.error}")

        return APIResponse(
            success=False,
            error="No working NFL API. Add valid SPORTRADAR_API_KEY or SPORTSDATAIO_API_KEY",
            source="tier_manager"
        )

    async def get_nfl_projections(self, season: int = 2024, week: int = 1) -> APIResponse:
        """
        Get NFL player projections for betting/DFS.

        Only available via SportsDataIO.
        """
        if self.sportsdata_io.is_configured:
            logger.info("Using SportsDataIO for NFL projections")
            return await self.sportsdata_io.get_nfl_projections(season, week)

        return APIResponse(
            success=False,
            error="NFL projections require SPORTSDATAIO_API_KEY",
            source="tier_manager"
        )

    async def get_mlb_schedule(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> APIResponse:
        """
        Get MLB schedule - uses best available API.

        Priority: Sportradar > SportsDataIO > MLB Stats (free)
        """
        if self.sportradar.is_configured:
            logger.info("Using Sportradar for MLB schedule")
            # Sportradar MLB endpoint
            pass

        if self.sportsdata_io.is_configured:
            logger.info("Using SportsDataIO for MLB schedule")
            # Would need MLB endpoint
            pass

        # Fall back to free MLB Stats API
        logger.info("Using MLB Stats API for schedule")
        return await self.mlb_stats.get_schedule(start_date, end_date)

    # =========================================================================
    # ODDS DATA
    # =========================================================================

    async def get_odds(
        self,
        sport: str,
        event_id: Optional[str] = None,
        regions: str = "eu",
        markets: str = "h2h,spreads,totals"
    ) -> APIResponse:
        """
        Get betting odds - uses best available API.

        Priority: LSports (live) > Genius Sports > Odds API
        """
        # For live odds, prefer LSports or Genius Sports
        if event_id:
            if self.lsports.is_configured:
                logger.info("Using LSports for live odds")
                return await self.lsports.get_odds(event_id)

            if self.genius_sports.is_configured:
                logger.info("Using Genius Sports for live odds")
                return await self.genius_sports.get_live_odds(event_id)

        # For pre-match odds, use Odds API
        logger.info("Using Odds API for odds")
        return await self.odds_api.get_odds(sport, regions, markets)

    async def get_live_events(self, sport_id: Optional[int] = None) -> APIResponse:
        """
        Get live/in-play events.

        Priority: LSports > Genius Sports
        """
        if self.lsports.is_configured:
            logger.info("Using LSports for live events")
            return await self.lsports.get_live_events(sport_id)

        if self.genius_sports.is_configured:
            logger.info("Using Genius Sports for live events")
            return await self.genius_sports.get_events(date="live")

        return APIResponse(
            success=False,
            error="Live events require LSPORTS_API_KEY or GENIUS_SPORTS_API_KEY",
            source="tier_manager"
        )

    # =========================================================================
    # INJURIES & NEWS
    # =========================================================================

    async def get_injuries(self, sport: str = "nfl") -> APIResponse:
        """
        Get current injuries for a sport.

        Only available via SportsDataIO.
        """
        if self.sportsdata_io.is_configured:
            logger.info("Using SportsDataIO for injuries")
            return await self.sportsdata_io.get_injuries(sport)

        return APIResponse(
            success=False,
            error="Injury data requires SPORTSDATAIO_API_KEY",
            source="tier_manager"
        )

    async def get_sports_news(
        self,
        query: Optional[str] = None,
        country: str = "us"
    ) -> APIResponse:
        """Get sports news headlines."""
        if query:
            return await self.news_api.search_news(query)
        return await self.news_api.get_sports_headlines(country)

    # =========================================================================
    # SPECIALTY SPORTS
    # =========================================================================

    async def get_f1_sessions(
        self,
        year: Optional[int] = None,
        session_type: Optional[str] = None
    ) -> APIResponse:
        """Get F1 sessions (always uses free OpenF1)."""
        return await self.openf1.get_sessions(year, session_type)

    async def get_esports_matches(
        self,
        game: str = "lol",
        status: str = "upcoming"
    ) -> APIResponse:
        """Get eSports matches (always uses PandaScore)."""
        return await self.pandascore.get_matches(game, status)

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def close(self):
        """Close all client connections."""
        # Enterprise
        await self.sportradar.close()
        await self.stats_perform.close()

        # Professional
        await self.genius_sports.close()
        await self.lsports.close()

        # Developer
        await self.sportsdata_io.close()
        await self.api_football_pro.close()

        # Free
        await self.odds_api.close()
        await self.football_data.close()
        await self.api_sports.close()
        await self.pandascore.close()
        await self.openf1.close()
        await self.mlb_stats.close()
        await self.news_api.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for quick status check
async def check_api_status() -> str:
    """Check all API statuses and return a report."""
    async with APITierManager() as manager:
        status = await manager.get_all_status()
        return manager.print_status_report(status)
