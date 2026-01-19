# data/basketball/bets_api_client.py
"""
Client for Bets API (https://the-bets-api.com/) for basketball data.
Provides NBA, EuroLeague, and other basketball league data.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from config.settings import settings


class BetsAPIBasketballClient:
    """
    Client for Bets API basketball endpoints.

    Features:
    - NBA, EuroLeague, and international leagues
    - Match schedules and results
    - Team statistics
    - Player stats
    - Head-to-head records
    - Live scores
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.BETS_API_KEY
        self.base_url = "https://api.the-bets-api.com/v1/basketball"
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def get_matches_by_date(
        self,
        date: str,
        league: str = "nba"
    ) -> List[Dict]:
        """
        Get basketball matches for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            league: League identifier (nba, euroleague, etc.)

        Returns:
            List of match dicts
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/{league}/matches/{date}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            return data.get("matches", [])
        except Exception as e:
            print(f"Error fetching basketball matches for {date}: {e}")
            return []

    async def get_team_stats(
        self,
        team_id: int,
        league: str = "nba"
    ) -> Optional[Dict]:
        """
        Get team statistics.

        Args:
            team_id: Team ID from API
            league: League identifier

        Returns:
            Team stats dict or None
        """
        if not self.api_key:
            return None

        endpoint = f"{self.base_url}/{league}/team/{team_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return None

    async def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        league: str = "nba"
    ) -> Optional[Dict]:
        """
        Get head-to-head record between two teams.

        Args:
            team1_id: First team ID
            team2_id: Second team ID
            league: League identifier

        Returns:
            H2H dict with match history
        """
        if not self.api_key:
            return None

        endpoint = f"{self.base_url}/{league}/h2h/{team1_id}/{team2_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching H2H: {e}")
            return None

    async def get_team_form(
        self,
        team_id: int,
        league: str = "nba",
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get recent matches for a team.

        Args:
            team_id: Team ID
            league: League identifier
            num_matches: Number of recent matches

        Returns:
            List of recent match dicts
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/{league}/team/{team_id}/matches/recent"
        params = {"limit": num_matches}

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            return data.get("matches", [])
        except Exception as e:
            print(f"Error fetching team form: {e}")
            return []

    async def get_standings(self, league: str = "nba") -> List[Dict]:
        """
        Get league standings.

        Args:
            league: League identifier

        Returns:
            List of team standings
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/{league}/standings"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            return data.get("standings", [])
        except Exception as e:
            print(f"Error fetching standings: {e}")
            return []

    def parse_match_to_nexus_format(self, match_data: Dict, league: str = "nba") -> Dict:
        """
        Parse Bets API match data to NEXUS standard format.

        Args:
            match_data: Match dict from Bets API
            league: League name

        Returns:
            Match dict in NEXUS format
        """
        # Extract team info
        home_team = match_data.get("home_team", {})
        away_team = match_data.get("away_team", {})

        # Parse start time
        start_time_str = match_data.get("start_time")
        start_time = None
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            except Exception:
                pass

        return {
            "external_id": str(match_data.get("id")),
            "sport": "basketball",
            "home_team": home_team.get("name", ""),
            "away_team": away_team.get("name", ""),
            "league": league.upper(),
            "country": match_data.get("country", ""),
            "start_time": start_time,
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_ranking": home_team.get("ranking"),
            "away_ranking": away_team.get("ranking"),
        }

    async def get_match_details(
        self,
        home_team: str,
        away_team: str,
        league: str = "nba",
        date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get detailed match information including team stats.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League identifier
            date: Match date (YYYY-MM-DD), defaults to today

        Returns:
            Complete match dict with team stats
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get matches for the date
        matches = await self.get_matches_by_date(date, league)

        # Find matching event
        for match in matches:
            h_team = match.get("home_team", {}).get("name", "").lower()
            a_team = match.get("away_team", {}).get("name", "").lower()

            if home_team.lower() in h_team and away_team.lower() in a_team:
                # Get additional team stats
                home_id = match.get("home_team", {}).get("id")
                away_id = match.get("away_team", {}).get("id")

                home_stats = None
                away_stats = None
                h2h = None

                if home_id and away_id:
                    # Fetch in parallel
                    results = await asyncio.gather(
                        self.get_team_stats(home_id, league),
                        self.get_team_stats(away_id, league),
                        self.get_head_to_head(home_id, away_id, league),
                        return_exceptions=True
                    )

                    home_stats = results[0] if not isinstance(results[0], Exception) else None
                    away_stats = results[1] if not isinstance(results[1], Exception) else None
                    h2h = results[2] if not isinstance(results[2], Exception) else None

                # Combine data
                match_info = self.parse_match_to_nexus_format(match, league)
                match_info["home_stats"] = home_stats
                match_info["away_stats"] = away_stats
                match_info["h2h"] = h2h

                return match_info

        return None


# === HELPER FUNCTIONS ===

async def get_basketball_match_data(
    home_team: str,
    away_team: str,
    league: str = "nba",
    date: Optional[str] = None
) -> Optional[Dict]:
    """
    Convenience function to get complete basketball match data.

    Args:
        home_team: Home team name
        away_team: Away team name
        league: League (nba, euroleague, etc.)
        date: Match date (YYYY-MM-DD)

    Returns:
        Complete match dict with stats
    """
    async with BetsAPIBasketballClient() as client:
        return await client.get_match_details(home_team, away_team, league, date)


async def get_basketball_standings(league: str = "nba") -> List[Dict]:
    """
    Get basketball league standings.

    Args:
        league: League identifier

    Returns:
        List of team standings
    """
    async with BetsAPIBasketballClient() as client:
        return await client.get_standings(league)


async def get_upcoming_basketball_matches(
    league: str = "nba",
    date: Optional[str] = None
) -> List[Dict]:
    """
    Get upcoming basketball matches.

    Args:
        league: League identifier
        date: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        List of matches in NEXUS format
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    async with BetsAPIBasketballClient() as client:
        matches = await client.get_matches_by_date(date, league)
        return [client.parse_match_to_nexus_format(m, league) for m in matches]
