# data/apis/thesportsdb_client.py
"""
TheSportsDB API client for NEXUS AI.
Free API (key=3) for fixtures, teams, players, and events.

Documentation: https://www.thesportsdb.com/api.php
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TheSportsDBClient:
    """
    Client for TheSportsDB free API.

    Free tier features (API key = 3):
    - Events by date/league
    - Team/player search
    - League listings
    - Past results

    Rate limit: ~100 requests/minute (be respectful)
    """

    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"

    # Sport IDs in TheSportsDB
    SPORT_IDS = {
        "tennis": 102,
        "basketball": 104,
        "football": 100,
        "baseball": 103,
        "ice_hockey": 105,
        "volleyball": 106,
    }

    # League IDs for popular leagues
    POPULAR_LEAGUES = {
        "tennis": {
            "ATP": "4464",
            "WTA": "4462",
            "Australian Open": "4585",
            "French Open": "4586",
            "Wimbledon": "4587",
            "US Open": "4588",
        },
        "basketball": {
            "NBA": "4387",
            "Euroleague": "4416",
            "NCAA": "4607",
            "Spanish Liga ACB": "4423",
            "French Pro A": "4424",
        }
    }

    def __init__(self, rate_limit: float = 1.5):
        """
        Initialize TheSportsDB client.

        Args:
            rate_limit: Requests per second (default: 1.5)
        """
        self.rate_limit = rate_limit
        self.session: Optional[httpx.AsyncClient] = None
        self._last_request = 0.0

    async def __aenter__(self):
        """Initialize HTTP session."""
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "NEXUS-AI/1.0",
                "Accept": "application/json"
            },
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()

    async def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make rate-limited API request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response or None
        """
        # Rate limiting
        now = asyncio.get_event_loop().time()
        time_since_last = now - self._last_request
        if time_since_last < 1.0 / self.rate_limit:
            await asyncio.sleep(1.0 / self.rate_limit - time_since_last)

        if not self.session:
            await self.__aenter__()

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = await self.session.get(url, params=params)
            self._last_request = asyncio.get_event_loop().time()

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TheSportsDB request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"TheSportsDB request error: {e}")
            return None

    async def get_events_by_date(
        self,
        date: str,
        sport: str = None
    ) -> List[Dict]:
        """
        Get all events for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            sport: Optional sport filter (Tennis, Basketball, etc.)

        Returns:
            List of event dicts
        """
        # TheSportsDB uses eventsday.php for events by date
        data = await self._request("eventsday.php", {"d": date, "s": sport})

        if not data:
            return []

        events = data.get("events") or []
        return events

    async def get_events_by_league(
        self,
        league_id: str,
        season: str = None
    ) -> List[Dict]:
        """
        Get events for a specific league.

        Args:
            league_id: TheSportsDB league ID
            season: Season (e.g., "2024-2025")

        Returns:
            List of event dicts
        """
        if not season:
            # Current season
            year = datetime.now().year
            season = f"{year}-{year + 1}"

        data = await self._request("eventsseason.php", {"id": league_id, "s": season})

        if not data:
            return []

        return data.get("events") or []

    async def get_next_events_by_league(
        self,
        league_id: str,
        limit: int = 15
    ) -> List[Dict]:
        """
        Get next upcoming events for a league.

        Args:
            league_id: TheSportsDB league ID
            limit: Maximum events to return

        Returns:
            List of upcoming events
        """
        data = await self._request("eventsnextleague.php", {"id": league_id})

        if not data:
            return []

        events = data.get("events") or []
        return events[:limit]

    async def get_past_events_by_league(
        self,
        league_id: str,
        limit: int = 15
    ) -> List[Dict]:
        """
        Get past events for a league.

        Args:
            league_id: TheSportsDB league ID
            limit: Maximum events to return

        Returns:
            List of past events
        """
        data = await self._request("eventspastleague.php", {"id": league_id})

        if not data:
            return []

        events = data.get("events") or []
        return events[:limit]

    async def search_team(self, team_name: str) -> Optional[Dict]:
        """
        Search for a team by name.

        Args:
            team_name: Team name to search

        Returns:
            Team dict or None
        """
        data = await self._request("searchteams.php", {"t": team_name})

        if not data:
            return None

        teams = data.get("teams") or []
        return teams[0] if teams else None

    async def search_player(self, player_name: str) -> Optional[Dict]:
        """
        Search for a player by name.

        Args:
            player_name: Player name to search

        Returns:
            Player dict or None
        """
        data = await self._request("searchplayers.php", {"p": player_name})

        if not data:
            return None

        players = data.get("player") or []
        return players[0] if players else None

    async def get_team(self, team_id: str) -> Optional[Dict]:
        """
        Get team details by ID.

        Args:
            team_id: TheSportsDB team ID

        Returns:
            Team dict or None
        """
        data = await self._request("lookupteam.php", {"id": team_id})

        if not data:
            return None

        teams = data.get("teams") or []
        return teams[0] if teams else None

    async def get_player(self, player_id: str) -> Optional[Dict]:
        """
        Get player details by ID.

        Args:
            player_id: TheSportsDB player ID

        Returns:
            Player dict or None
        """
        data = await self._request("lookupplayer.php", {"id": player_id})

        if not data:
            return None

        players = data.get("players") or []
        return players[0] if players else None

    async def get_all_leagues_by_sport(self, sport: str) -> List[Dict]:
        """
        Get all leagues for a sport.

        Args:
            sport: Sport name (e.g., "Tennis", "Basketball")

        Returns:
            List of league dicts
        """
        data = await self._request("search_all_leagues.php", {"s": sport})

        if not data:
            return []

        return data.get("countrys") or []

    async def get_league_table(self, league_id: str, season: str = None) -> List[Dict]:
        """
        Get league standings/table.

        Args:
            league_id: TheSportsDB league ID
            season: Season (e.g., "2024-2025")

        Returns:
            List of standing dicts
        """
        if not season:
            year = datetime.now().year
            season = f"{year}-{year + 1}"

        data = await self._request("lookuptable.php", {"l": league_id, "s": season})

        if not data:
            return []

        return data.get("table") or []

    async def get_event_details(self, event_id: str) -> Optional[Dict]:
        """
        Get detailed event information.

        Args:
            event_id: TheSportsDB event ID

        Returns:
            Event dict or None
        """
        data = await self._request("lookupevent.php", {"id": event_id})

        if not data:
            return None

        events = data.get("events") or []
        return events[0] if events else None

    async def get_event_results(self, event_id: str) -> Optional[Dict]:
        """
        Get event results/scores.

        Args:
            event_id: TheSportsDB event ID

        Returns:
            Event result dict or None
        """
        # Same as event details for completed events
        return await self.get_event_details(event_id)

    async def get_team_last_events(
        self,
        team_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get team's recent past events.

        Args:
            team_id: TheSportsDB team ID
            limit: Maximum events

        Returns:
            List of past events
        """
        data = await self._request("eventslast.php", {"id": team_id})

        if not data:
            return []

        events = data.get("results") or []
        return events[:limit]

    async def get_team_next_events(
        self,
        team_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get team's upcoming events.

        Args:
            team_id: TheSportsDB team ID
            limit: Maximum events

        Returns:
            List of upcoming events
        """
        data = await self._request("eventsnext.php", {"id": team_id})

        if not data:
            return []

        events = data.get("events") or []
        return events[:limit]


# === HELPER FUNCTIONS ===

async def get_tennis_fixtures(date: str = None) -> List[Dict]:
    """
    Get tennis fixtures for a date.

    Args:
        date: Date in YYYY-MM-DD (defaults to today)

    Returns:
        List of tennis events
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    async with TheSportsDBClient() as client:
        return await client.get_events_by_date(date, "Tennis")


async def get_basketball_fixtures(date: str = None) -> List[Dict]:
    """
    Get basketball fixtures for a date.

    Args:
        date: Date in YYYY-MM-DD (defaults to today)

    Returns:
        List of basketball events
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    async with TheSportsDBClient() as client:
        return await client.get_events_by_date(date, "Basketball")


async def get_nba_fixtures() -> List[Dict]:
    """Get upcoming NBA fixtures."""
    async with TheSportsDBClient() as client:
        return await client.get_next_events_by_league(
            TheSportsDBClient.POPULAR_LEAGUES["basketball"]["NBA"]
        )


async def search_and_get_team_stats(team_name: str) -> Optional[Dict]:
    """
    Search for team and get its recent results.

    Args:
        team_name: Team name to search

    Returns:
        Dict with team info and recent results
    """
    async with TheSportsDBClient() as client:
        team = await client.search_team(team_name)
        if not team:
            return None

        team_id = team.get("idTeam")
        if not team_id:
            return None

        # Get recent results
        results = await client.get_team_last_events(team_id, 5)

        return {
            "team": team,
            "recent_results": results
        }
