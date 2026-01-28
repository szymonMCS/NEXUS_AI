"""
Unified Sports API Client.

Provides access to multiple sports data APIs with unified interface.
Based on official documentation from each provider.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, date
from dataclasses import dataclass, field
import httpx

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    source: str = ""
    cached: bool = False


class BaseAPIClient:
    """Base class for API clients."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class OddsAPIClient(BaseAPIClient):
    """
    The Odds API client.
    Documentation: https://the-odds-api.com/liveapi/guides/v4/

    Provides odds from 40+ bookmakers for 70+ sports.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ODDS_API_KEY")

    async def get_sports(self) -> APIResponse:
        """Get list of available sports."""
        if not self.api_key:
            return APIResponse(success=False, error="ODDS_API_KEY not configured", source="odds_api")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/sports",
                params={"apiKey": self.api_key}
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="odds_api")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="odds_api")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="odds_api")

    async def get_odds(
        self,
        sport: str,
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "decimal"
    ) -> APIResponse:
        """
        Get odds for a sport.

        Args:
            sport: Sport key (e.g., "soccer_epl", "basketball_nba")
            regions: Comma-separated regions (us, us2, uk, eu, au)
            markets: Comma-separated markets (h2h, spreads, totals)
            odds_format: decimal or american
        """
        if not self.api_key:
            return APIResponse(success=False, error="ODDS_API_KEY not configured", source="odds_api")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/sports/{sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format
                }
            )
            if response.status_code == 200:
                remaining = response.headers.get("x-requests-remaining", "?")
                logger.info(f"Odds API: {remaining} requests remaining")
                return APIResponse(success=True, data=response.json(), source="odds_api")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="odds_api")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="odds_api")


class FootballDataClient(BaseAPIClient):
    """
    Football-Data.org client.
    Documentation: https://www.football-data.org/documentation/quickstart

    Free tier: 10 req/min, major European leagues.
    Header: X-Auth-Token
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("X-Auth-Token")

    def _get_headers(self) -> Dict[str, str]:
        return {"X-Auth-Token": self.api_key} if self.api_key else {}

    async def get_competitions(self) -> APIResponse:
        """Get list of competitions."""
        if not self.api_key:
            return APIResponse(success=False, error="X-Auth-Token not configured", source="football_data")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/competitions",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="football_data")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="football_data")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="football_data")

    async def get_matches(
        self,
        competition: str = "PL",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> APIResponse:
        """
        Get matches for a competition.

        Args:
            competition: Competition code (PL, LaLiga, CL, etc.)
            date_from: YYYY-MM-DD
            date_to: YYYY-MM-DD
        """
        if not self.api_key:
            return APIResponse(success=False, error="X-Auth-Token not configured", source="football_data")

        client = await self._get_client()
        params = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        try:
            response = await client.get(
                f"{self.BASE_URL}/competitions/{competition}/matches",
                headers=self._get_headers(),
                params=params
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="football_data")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="football_data")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="football_data")

    async def get_standings(self, competition: str = "PL") -> APIResponse:
        """Get standings for a competition."""
        if not self.api_key:
            return APIResponse(success=False, error="X-Auth-Token not configured", source="football_data")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/competitions/{competition}/standings",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="football_data")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="football_data")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="football_data")


class APISportsClient(BaseAPIClient):
    """
    API-Sports.io client.
    Documentation: https://api-sports.io/documentation

    Covers: Basketball, Tennis, Football, Baseball, Hockey, Rugby
    Header: x-apisports-key
    Free tier: 100 req/day
    """

    ENDPOINTS = {
        "basketball": "https://v1.basketball.api-sports.io",
        "tennis": "https://v1.tennis.api-sports.io",
        "football": "https://v3.football.api-sports.io",
        "baseball": "https://v1.baseball.api-sports.io",
        "hockey": "https://v1.hockey.api-sports.io",
        "rugby": "https://v1.rugby.api-sports.io",
    }

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("x-apisports-key")

    def _get_headers(self) -> Dict[str, str]:
        return {"x-apisports-key": self.api_key} if self.api_key else {}

    async def get_leagues(self, sport: str = "basketball") -> APIResponse:
        """Get leagues for a sport."""
        if not self.api_key:
            return APIResponse(success=False, error="x-apisports-key not configured", source="api_sports")

        base_url = self.ENDPOINTS.get(sport)
        if not base_url:
            return APIResponse(success=False, error=f"Unknown sport: {sport}", source="api_sports")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{base_url}/leagues",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
                logger.info(f"API-Sports: {remaining} requests remaining")
                return APIResponse(success=True, data=data.get("response", []), source="api_sports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_sports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_sports")

    async def get_games(
        self,
        sport: str = "basketball",
        league_id: Optional[int] = None,
        date: Optional[str] = None,
        season: Optional[str] = None
    ) -> APIResponse:
        """Get games for a sport."""
        if not self.api_key:
            return APIResponse(success=False, error="x-apisports-key not configured", source="api_sports")

        base_url = self.ENDPOINTS.get(sport)
        if not base_url:
            return APIResponse(success=False, error=f"Unknown sport: {sport}", source="api_sports")

        client = await self._get_client()
        params = {}
        if league_id:
            params["league"] = league_id
        if date:
            params["date"] = date
        if season:
            params["season"] = season

        try:
            response = await client.get(
                f"{base_url}/games",
                headers=self._get_headers(),
                params=params
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("response", []), source="api_sports")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="api_sports")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="api_sports")


class PandaScoreClient(BaseAPIClient):
    """
    PandaScore client for eSports.
    Documentation: https://developers.pandascore.co/docs

    Covers: LoL, CS2, Dota 2, Valorant, Overwatch
    Header: Authorization: Bearer <token>
    Free tier: 1000 req/hour
    """

    BASE_URL = "https://api.pandascore.co"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("PANDASCORE_API_KEY")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def get_matches(
        self,
        game: str = "lol",
        status: str = "upcoming",
        per_page: int = 10
    ) -> APIResponse:
        """
        Get matches for a game.

        Args:
            game: lol, csgo, dota2, valorant, overwatch
            status: upcoming, running, past
            per_page: Results per page
        """
        if not self.api_key:
            return APIResponse(success=False, error="PANDASCORE_API_KEY not configured", source="pandascore")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/{game}/matches/{status}",
                headers=self._get_headers(),
                params={"per_page": per_page}
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="pandascore")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="pandascore")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="pandascore")

    async def get_tournaments(self, game: str = "lol") -> APIResponse:
        """Get tournaments for a game."""
        if not self.api_key:
            return APIResponse(success=False, error="PANDASCORE_API_KEY not configured", source="pandascore")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/{game}/tournaments",
                headers=self._get_headers(),
                params={"per_page": 20}
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="pandascore")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="pandascore")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="pandascore")


class OpenF1Client(BaseAPIClient):
    """
    OpenF1 API client.
    Documentation: https://openf1.org

    Free, no API key needed.
    Real-time F1 data, telemetry, sessions.
    """

    BASE_URL = "https://api.openf1.org/v1"

    async def get_sessions(
        self,
        year: Optional[int] = None,
        session_type: Optional[str] = None
    ) -> APIResponse:
        """
        Get F1 sessions.

        Args:
            year: Year (e.g., 2024)
            session_type: Race, Qualifying, Practice, Sprint
        """
        client = await self._get_client()
        params = {}
        if year:
            params["year"] = year
        if session_type:
            params["session_type"] = session_type

        try:
            response = await client.get(f"{self.BASE_URL}/sessions", params=params)
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="openf1")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="openf1")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="openf1")

    async def get_drivers(self, session_key: Optional[int] = None) -> APIResponse:
        """Get F1 drivers."""
        client = await self._get_client()
        params = {}
        if session_key:
            params["session_key"] = session_key

        try:
            response = await client.get(f"{self.BASE_URL}/drivers", params=params)
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="openf1")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="openf1")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="openf1")

    async def get_position(self, session_key: int, driver_number: int) -> APIResponse:
        """Get position data for a driver in a session."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/position",
                params={"session_key": session_key, "driver_number": driver_number}
            )
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="openf1")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="openf1")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="openf1")


class MLBStatsClient(BaseAPIClient):
    """
    MLB Stats API client.
    Documentation: https://statsapi.mlb.com/docs

    Free, no API key needed.
    Official MLB data.
    """

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    async def get_teams(self, sport_id: int = 1) -> APIResponse:
        """Get MLB teams. sport_id=1 is MLB."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.BASE_URL}/teams", params={"sportId": sport_id})
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("teams", []), source="mlb_stats")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="mlb_stats")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="mlb_stats")

    async def get_schedule(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sport_id: int = 1
    ) -> APIResponse:
        """
        Get MLB schedule.

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            sport_id: 1 = MLB
        """
        client = await self._get_client()
        params = {"sportId": sport_id}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        try:
            response = await client.get(f"{self.BASE_URL}/schedule", params=params)
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("dates", []), source="mlb_stats")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="mlb_stats")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="mlb_stats")

    async def get_game(self, game_pk: int) -> APIResponse:
        """Get detailed game data."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.BASE_URL}/game/{game_pk}/boxscore")
            if response.status_code == 200:
                return APIResponse(success=True, data=response.json(), source="mlb_stats")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="mlb_stats")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="mlb_stats")

    async def get_standings(self, league_id: int = 103, season: int = 2024) -> APIResponse:
        """Get MLB standings. league_id: 103=AL, 104=NL"""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/standings",
                params={"leagueId": league_id, "season": season}
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("records", []), source="mlb_stats")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="mlb_stats")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="mlb_stats")


class NewsAPIClient(BaseAPIClient):
    """
    NewsAPI client.
    Documentation: https://newsapi.org/docs

    Free tier: 100 req/day, development only.
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("NEWSAPI_KEY")

    async def get_sports_headlines(
        self,
        country: str = "us",
        page_size: int = 10
    ) -> APIResponse:
        """Get sports headlines."""
        if not self.api_key:
            return APIResponse(success=False, error="NEWSAPI_KEY not configured", source="newsapi")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/top-headlines",
                params={
                    "apiKey": self.api_key,
                    "category": "sports",
                    "country": country,
                    "pageSize": page_size
                }
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("articles", []), source="newsapi")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="newsapi")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="newsapi")

    async def search_news(
        self,
        query: str,
        sort_by: str = "relevancy",
        page_size: int = 10
    ) -> APIResponse:
        """Search for news articles."""
        if not self.api_key:
            return APIResponse(success=False, error="NEWSAPI_KEY not configured", source="newsapi")

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.BASE_URL}/everything",
                params={
                    "apiKey": self.api_key,
                    "q": query,
                    "sortBy": sort_by,
                    "pageSize": page_size
                }
            )
            if response.status_code == 200:
                data = response.json()
                return APIResponse(success=True, data=data.get("articles", []), source="newsapi")
            return APIResponse(success=False, error=f"HTTP {response.status_code}", source="newsapi")
        except Exception as e:
            return APIResponse(success=False, error=str(e), source="newsapi")


class UnifiedSportsAPI:
    """
    Unified interface for all sports APIs.

    Basic usage (free tier only):
        api = UnifiedSportsAPI()
        odds = await api.odds.get_odds("basketball_nba")
        matches = await api.football.get_matches("PL")
        await api.close()

    For auto-tiered access (uses best available):
        from data.apis.api_tier_manager import APITierManager
        manager = APITierManager()
        matches = await manager.get_football_matches("PL")
    """

    def __init__(self):
        # Free tier clients
        self.odds = OddsAPIClient()
        self.football = FootballDataClient()
        self.api_sports = APISportsClient()
        self.pandascore = PandaScoreClient()
        self.f1 = OpenF1Client()
        self.mlb = MLBStatsClient()
        self.news = NewsAPIClient()

        # Premium clients (lazy loaded when keys available)
        self._premium_clients = None

    def _init_premium(self):
        """Lazy initialize premium clients if keys are configured."""
        if self._premium_clients is not None:
            return

        from data.apis.premium_api_clients import (
            SportradarClient,
            StatsPerformClient,
            GeniusSportsClient,
            LSportsClient,
            SportsDataIOClient,
            APIFootballProClient,
        )

        self._premium_clients = {
            "sportradar": SportradarClient(),
            "stats_perform": StatsPerformClient(),
            "genius_sports": GeniusSportsClient(),
            "lsports": LSportsClient(),
            "sportsdata_io": SportsDataIOClient(),
            "api_football_pro": APIFootballProClient(),
        }

    @property
    def sportradar(self):
        """Access Sportradar client (enterprise tier)."""
        self._init_premium()
        return self._premium_clients["sportradar"]

    @property
    def stats_perform(self):
        """Access Stats Perform client (enterprise tier)."""
        self._init_premium()
        return self._premium_clients["stats_perform"]

    @property
    def genius_sports(self):
        """Access Genius Sports client (professional tier)."""
        self._init_premium()
        return self._premium_clients["genius_sports"]

    @property
    def lsports(self):
        """Access LSports client (professional tier)."""
        self._init_premium()
        return self._premium_clients["lsports"]

    @property
    def sportsdata_io(self):
        """Access SportsDataIO client (developer tier)."""
        self._init_premium()
        return self._premium_clients["sportsdata_io"]

    @property
    def api_football_pro(self):
        """Access API-Football Pro client (developer tier)."""
        self._init_premium()
        return self._premium_clients["api_football_pro"]

    def get_available_tiers(self) -> dict:
        """Check which API tiers are available."""
        self._init_premium()
        return {
            "enterprise": {
                "sportradar": self._premium_clients["sportradar"].is_configured,
                "stats_perform": self._premium_clients["stats_perform"].is_configured,
            },
            "professional": {
                "genius_sports": self._premium_clients["genius_sports"].is_configured,
                "lsports": self._premium_clients["lsports"].is_configured,
            },
            "developer": {
                "sportsdata_io": self._premium_clients["sportsdata_io"].is_configured,
                "api_football_pro": self._premium_clients["api_football_pro"].is_configured,
            },
            "free": {
                "odds_api": bool(os.getenv("ODDS_API_KEY")),
                "football_data": bool(os.getenv("X-Auth-Token")),
                "api_sports": bool(os.getenv("x-apisports-key")),
                "pandascore": bool(os.getenv("PANDASCORE_API_KEY")),
                "openf1": True,  # No key needed
                "mlb_stats": True,  # No key needed
            },
        }

    async def close(self):
        """Close all client connections."""
        # Free tier
        await self.odds.close()
        await self.football.close()
        await self.api_sports.close()
        await self.pandascore.close()
        await self.f1.close()
        await self.mlb.close()
        await self.news.close()

        # Premium tier (if initialized)
        if self._premium_clients:
            for client in self._premium_clients.values():
                await client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
