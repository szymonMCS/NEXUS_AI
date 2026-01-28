"""
Premium Sports API Clients.

Enterprise-grade paid API integrations for NEXUS AI.
These clients are designed to work automatically when API keys are configured.

Tier 1 (Enterprise): Sportradar, Stats Perform
Tier 2 (Professional): Genius Sports, LSports
Tier 3 (Developer): SportsDataIO, API-Football Pro
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import httpx

from data.apis.sports_api_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


class APITier(Enum):
    """API pricing/quality tiers."""
    FREE = "free"
    DEVELOPER = "developer"       # $50-200/mo
    PROFESSIONAL = "professional"  # $300-1000/mo
    ENTERPRISE = "enterprise"     # $1000+/mo


@dataclass
class APIStatus:
    """Status of an API client."""
    name: str
    tier: APITier
    configured: bool
    available: bool
    error: Optional[str] = None
    features: List[str] = field(default_factory=list)


# =============================================================================
# TIER 1: ENTERPRISE APIs
# =============================================================================

class SportradarClient(BaseAPIClient):
    """
    Sportradar API client.
    Documentation: https://developer.sportradar.com/

    Official data partner for NBA, NHL, MLB, NFL, UEFA.
    Features: Real-time scores, play-by-play, advanced statistics.

    Pricing: Enterprise contracts, typically $10,000+/year
    """

    # API endpoints by sport
    ENDPOINTS = {
        "nba": "https://api.sportradar.com/nba/trial/v8/en",
        "nfl": "https://api.sportradar.com/nfl/official/trial/v7/en",
        "mlb": "https://api.sportradar.com/mlb/trial/v7/en",
        "nhl": "https://api.sportradar.com/nhl/trial/v7/en",
        "soccer": "https://api.sportradar.com/soccer/trial/v4/en",
    }

    TIER = APITier.ENTERPRISE

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SPORTRADAR_API_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_params(self) -> Dict[str, str]:
        return {"api_key": self.api_key} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        if not self.api_key:
            return APIStatus(
                name="Sportradar",
                tier=self.TIER,
                configured=False,
                available=False,
                error="SPORTRADAR_API_KEY not configured",
                features=["NBA", "NFL", "MLB", "NHL", "Soccer", "Real-time", "Play-by-play"]
            )

        # Test connection
        try:
            result = await self.get_nba_schedule()
            return APIStatus(
                name="Sportradar",
                tier=self.TIER,
                configured=True,
                available=result.success,
                error=result.error if not result.success else None,
                features=["NBA", "NFL", "MLB", "NHL", "Soccer", "Real-time", "Play-by-play"]
            )
        except Exception as e:
            return APIStatus(
                name="Sportradar",
                tier=self.TIER,
                configured=True,
                available=False,
                error=str(e),
                features=["NBA", "NFL", "MLB", "NHL", "Soccer", "Real-time", "Play-by-play"]
            )

    async def get_nba_schedule(self, season: int = 2024, season_type: str = "REG") -> APIResponse:
        """Get NBA schedule."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTRADAR_API_KEY not configured", source="sportradar")

        client = await self._get_client()
        try:
            url = f"{self.ENDPOINTS['nba']}/games/{season}/{season_type}/schedule.json"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportradar")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportradar")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportradar")

    async def get_nba_game_summary(self, game_id: str) -> APIResponse:
        """Get detailed NBA game summary with play-by-play."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTRADAR_API_KEY not configured", source="sportradar")

        client = await self._get_client()
        try:
            url = f"{self.ENDPOINTS['nba']}/games/{game_id}/summary.json"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportradar")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportradar")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportradar")

    async def get_nfl_schedule(self, season: int = 2024, season_type: str = "REG") -> APIResponse:
        """Get NFL schedule."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTRADAR_API_KEY not configured", source="sportradar")

        client = await self._get_client()
        try:
            url = f"{self.ENDPOINTS['nfl']}/games/{season}/{season_type}/schedule.json"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportradar")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportradar")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportradar")

    async def get_soccer_matches(self, competition_id: str) -> APIResponse:
        """Get soccer matches for a competition."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTRADAR_API_KEY not configured", source="sportradar")

        client = await self._get_client()
        try:
            url = f"{self.ENDPOINTS['soccer']}/competitions/{competition_id}/schedule.json"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportradar")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportradar")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportradar")


class StatsPerformClient(BaseAPIClient):
    """
    Stats Perform (Opta) API client.
    Documentation: https://developer.statsperform.com/

    AI/ML analytics, expected goals (xG), predictive modeling.
    Strongest for European football (soccer).

    Pricing: Enterprise only, custom quotes
    """

    BASE_URL = "https://api.statsperform.com/v1"
    TIER = APITier.ENTERPRISE

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("STATS_PERFORM_API_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        return APIStatus(
            name="Stats Perform (Opta)",
            tier=self.TIER,
            configured=self.is_configured,
            available=self.is_configured,  # Can't test without valid key
            error="STATS_PERFORM_API_KEY not configured" if not self.is_configured else None,
            features=["xG", "AI Analytics", "Predictive Models", "European Football", "Tennis"]
        )

    async def get_match_stats(self, match_id: str) -> APIResponse:
        """Get detailed match statistics including xG."""
        if not self.api_key:
            return APIResponse(success=False, error="STATS_PERFORM_API_KEY not configured", source="stats_perform")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URL}/football/matches/{match_id}/stats"
            response = await client.get(url, headers=self._get_headers())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="stats_perform")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="stats_perform")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="stats_perform")

    async def get_player_xg(self, player_id: str, season: str = "2024") -> APIResponse:
        """Get player expected goals (xG) statistics."""
        if not self.api_key:
            return APIResponse(success=False, error="STATS_PERFORM_API_KEY not configured", source="stats_perform")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URL}/football/players/{player_id}/xg"
            response = await client.get(url, headers=self._get_headers(), params={"season": season})
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="stats_perform")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="stats_perform")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="stats_perform")


# =============================================================================
# TIER 2: PROFESSIONAL APIs
# =============================================================================

class GeniusSportsClient(BaseAPIClient):
    """
    Genius Sports (formerly BetGenius) API client.
    Documentation: https://developer.geniussports.com/

    Real-time odds, trading tools, integrity monitoring.
    150,000+ events/year, NFL, NCAA, soccer, tennis.

    Pricing: Starts ~$500/month for basic plans
    """

    BASE_URL = "https://api.geniussports.com/v2"
    TIER = APITier.PROFESSIONAL

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GENIUS_SPORTS_API_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        return APIStatus(
            name="Genius Sports",
            tier=self.TIER,
            configured=self.is_configured,
            available=self.is_configured,
            error="GENIUS_SPORTS_API_KEY not configured" if not self.is_configured else None,
            features=["Real-time Odds", "NFL", "NCAA", "Trading Tools", "Integrity"]
        )

    async def get_live_odds(self, event_id: str) -> APIResponse:
        """Get live odds for an event."""
        if not self.api_key:
            return APIResponse(success=False, error="GENIUS_SPORTS_API_KEY not configured", source="genius_sports")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URL}/odds/live/{event_id}"
            response = await client.get(url, headers=self._get_headers())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="genius_sports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="genius_sports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="genius_sports")

    async def get_events(self, sport: str = "football", date: Optional[str] = None) -> APIResponse:
        """Get events for a sport."""
        if not self.api_key:
            return APIResponse(success=False, error="GENIUS_SPORTS_API_KEY not configured", source="genius_sports")

        client = await self._get_client()
        params = {"sport": sport}
        if date:
            params["date"] = date

        try:
            url = f"{self.BASE_URL}/events"
            response = await client.get(url, headers=self._get_headers(), params=params)
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="genius_sports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="genius_sports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="genius_sports")


class LSportsClient(BaseAPIClient):
    """
    LSports API client.
    Documentation: https://www.lsports.eu/documentation/

    Ultra-low latency data for betting/trading environments.
    60+ sports, 100+ leagues, pre-match and in-play feeds.

    Pricing: ~$300-1,000/month depending on coverage
    """

    BASE_URL = "https://api.lsports.eu/v2"
    TIER = APITier.PROFESSIONAL

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("LSPORTS_API_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        return APIStatus(
            name="LSports",
            tier=self.TIER,
            configured=self.is_configured,
            available=self.is_configured,
            error="LSPORTS_API_KEY not configured" if not self.is_configured else None,
            features=["Ultra-low Latency", "60+ Sports", "In-play", "Pre-match", "Trading"]
        )

    async def get_fixtures(self, sport_id: int, from_date: Optional[str] = None) -> APIResponse:
        """Get fixtures for a sport."""
        if not self.api_key:
            return APIResponse(success=False, error="LSPORTS_API_KEY not configured", source="lsports")

        client = await self._get_client()
        params = {"sportId": sport_id}
        if from_date:
            params["fromDate"] = from_date

        try:
            url = f"{self.BASE_URL}/fixtures"
            response = await client.get(url, headers=self._get_headers(), params=params)
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="lsports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="lsports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="lsports")

    async def get_live_events(self, sport_id: Optional[int] = None) -> APIResponse:
        """Get live events (in-play)."""
        if not self.api_key:
            return APIResponse(success=False, error="LSPORTS_API_KEY not configured", source="lsports")

        client = await self._get_client()
        params = {}
        if sport_id:
            params["sportId"] = sport_id

        try:
            url = f"{self.BASE_URL}/inplay"
            response = await client.get(url, headers=self._get_headers(), params=params)
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="lsports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="lsports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="lsports")

    async def get_odds(self, fixture_id: str) -> APIResponse:
        """Get odds for a fixture."""
        if not self.api_key:
            return APIResponse(success=False, error="LSPORTS_API_KEY not configured", source="lsports")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URL}/odds/{fixture_id}"
            response = await client.get(url, headers=self._get_headers())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="lsports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="lsports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="lsports")


# =============================================================================
# TIER 3: DEVELOPER-FRIENDLY APIs
# =============================================================================

class SportsDataIOClient(BaseAPIClient):
    """
    SportsDataIO API client.
    Documentation: https://sportsdata.io/developers/api-documentation

    Comprehensive US sports coverage with projections and DFS data.
    NFL, NBA, MLB, NHL, PGA, UFC, NASCAR, MLS.

    Pricing: $50-500/month depending on sports/features
    """

    BASE_URLS = {
        "nfl": "https://api.sportsdata.io/v3/nfl",
        "nba": "https://api.sportsdata.io/v3/nba",
        "mlb": "https://api.sportsdata.io/v3/mlb",
        "nhl": "https://api.sportsdata.io/v3/nhl",
        "mls": "https://api.sportsdata.io/v4/soccer",
        "golf": "https://api.sportsdata.io/golf/v2",
    }

    TIER = APITier.DEVELOPER

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SPORTSDATAIO_API_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_params(self) -> Dict[str, str]:
        return {"key": self.api_key} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        if not self.is_configured:
            return APIStatus(
                name="SportsDataIO",
                tier=self.TIER,
                configured=False,
                available=False,
                error="SPORTSDATAIO_API_KEY not configured",
                features=["NFL", "NBA", "MLB", "NHL", "Projections", "DFS", "Injuries"]
            )

        # Test connection
        result = await self.get_nfl_teams()
        return APIStatus(
            name="SportsDataIO",
            tier=self.TIER,
            configured=True,
            available=result.success,
            error=result.error if not result.success else None,
            features=["NFL", "NBA", "MLB", "NHL", "Projections", "DFS", "Injuries"]
        )

    async def get_nfl_teams(self) -> APIResponse:
        """Get NFL teams."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URLS['nfl']}/scores/json/Teams"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")

    async def get_nfl_schedule(self, season: int = 2024) -> APIResponse:
        """Get NFL schedule."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URLS['nfl']}/scores/json/Schedules/{season}"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")

    async def get_nfl_projections(self, season: int = 2024, week: int = 1) -> APIResponse:
        """Get NFL player projections for DFS/betting."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URLS['nfl']}/projections/json/PlayerGameProjectionStatsByWeek/{season}REG/{week}"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")

    async def get_nba_games(self, date: str) -> APIResponse:
        """Get NBA games for a date (format: YYYY-MM-DD)."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        client = await self._get_client()
        try:
            # Convert date format
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%Y-%b-%d").upper()

            url = f"{self.BASE_URLS['nba']}/scores/json/GamesByDate/{date_str}"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")

    async def get_nba_projections(self, date: str) -> APIResponse:
        """Get NBA player projections for a date."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        client = await self._get_client()
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%Y-%b-%d").upper()

            url = f"{self.BASE_URLS['nba']}/projections/json/PlayerGameProjectionStatsByDate/{date_str}"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")

    async def get_injuries(self, sport: str = "nfl") -> APIResponse:
        """Get current injuries for a sport."""
        if not self.api_key:
            return APIResponse(success=False, error="SPORTSDATAIO_API_KEY not configured", source="sportsdata_io")

        if sport not in self.BASE_URLS:
            return APIResponse(success=False, error=f"Unknown sport: {sport}", source="sportsdata_io")

        client = await self._get_client()
        try:
            url = f"{self.BASE_URLS[sport]}/scores/json/Injuries"
            response = await client.get(url, params=self._get_params())
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="sportsdata_io")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="sportsdata_io")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="sportsdata_io")


class APIFootballProClient(BaseAPIClient):
    """
    API-Football Pro client (upgraded tier).
    Documentation: https://www.api-football.com/documentation-v3

    860+ football leagues, 140+ countries.
    Upgrade from free tier for more requests and features.

    Pricing: 19-199 EUR/month
    """

    BASE_URL = "https://v3.football.api-sports.io"
    TIER = APITier.DEVELOPER

    def __init__(self):
        super().__init__()
        # Can use either pro key or standard API-Sports key
        self.api_key = os.getenv("API_FOOTBALL_PRO_KEY") or os.getenv("x-apisports-key")
        self.is_pro = bool(os.getenv("API_FOOTBALL_PRO_KEY"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        return {"x-apisports-key": self.api_key} if self.api_key else {}

    async def get_status(self) -> APIStatus:
        """Check API status and availability."""
        if not self.is_configured:
            return APIStatus(
                name="API-Football" + (" Pro" if self.is_pro else ""),
                tier=self.TIER,
                configured=False,
                available=False,
                error="API_FOOTBALL_PRO_KEY not configured",
                features=["860+ Leagues", "Live Scores", "Odds", "Predictions", "Statistics"]
            )

        result = await self.get_leagues()
        return APIStatus(
            name="API-Football" + (" Pro" if self.is_pro else ""),
            tier=self.TIER,
            configured=True,
            available=result.success,
            error=result.error if not result.success else None,
            features=["860+ Leagues", "Live Scores", "Odds", "Predictions", "Statistics"]
        )

    async def get_leagues(self) -> APIResponse:
        """Get all available leagues."""
        if not self.api_key:
            return APIResponse(success=False, error="API_FOOTBALL_PRO_KEY not configured", source="api_football")

        client = await self._get_client()
        try:
            response = await client.get(f"{self.BASE_URL}/leagues", headers=self._get_headers())
            if response.status_code == 200:
                data = response.json()
                remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
                logger.info(f"API-Football: {remaining} requests remaining")
                return APIResponse(success=True, data=data.get("response", []), source="api_football")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_football")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_football")

    async def get_fixtures(
        self,
        league_id: int,
        season: int = 2024,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> APIResponse:
        """Get fixtures for a league."""
        if not self.api_key:
            return APIResponse(success=False, error="API_FOOTBALL_PRO_KEY not configured", source="api_football")

        client = await self._get_client()
        params = {"league": league_id, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            response = await client.get(
                f"{self.BASE_URL}/fixtures",
                headers=self._get_headers(),
                params=params
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("response", []), source="api_football")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_football")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_football")

    async def get_predictions(self, fixture_id: int) -> APIResponse:
        """Get AI predictions for a fixture."""
        if not self.api_key:
            return APIResponse(success=False, error="API_FOOTBALL_PRO_KEY not configured", source="api_football")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/predictions",
                headers=self._get_headers(),
                params={"fixture": fixture_id}
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("response", []), source="api_football")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_football")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_football")

    async def get_odds(self, fixture_id: int) -> APIResponse:
        """Get odds for a fixture from multiple bookmakers."""
        if not self.api_key:
            return APIResponse(success=False, error="API_FOOTBALL_PRO_KEY not configured", source="api_football")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/odds",
                headers=self._get_headers(),
                params={"fixture": fixture_id}
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("response", []), source="api_football")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_football")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_football")

    async def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> APIResponse:
        """Get head-to-head history between two teams."""
        if not self.api_key:
            return APIResponse(success=False, error="API_FOOTBALL_PRO_KEY not configured", source="api_football")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/fixtures/headtohead",
                headers=self._get_headers(),
                params={"h2h": f"{team1_id}-{team2_id}", "last": last}
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("response", []), source="api_football")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_football")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_football")
