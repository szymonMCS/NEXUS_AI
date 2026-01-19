# data/tennis/api_tennis_client.py
"""
Client for API-Tennis (https://api-tennis.com/).
Provides comprehensive tennis match data, player stats, and rankings.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from config.settings import settings


class TennisAPIClient:
    """
    Client for API-Tennis.

    Features:
    - ATP/WTA rankings
    - Match schedules and results
    - Player statistics
    - Head-to-head records
    - Tournament information
    - Live scores
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.API_TENNIS_KEY
        self.base_url = "https://api-tennis.com/tennis"
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"X-RapidAPI-Key": self.api_key} if self.api_key else {}
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def get_rankings(self, tour: str = "atp") -> List[Dict]:
        """
        Get current ATP/WTA rankings.

        Args:
            tour: "atp" or "wta"

        Returns:
            List of player ranking dicts
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/rankings/{tour}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            return data.get("rankings", [])
        except Exception as e:
            print(f"Error fetching {tour.upper()} rankings: {e}")
            return []

    async def get_matches_by_date(
        self,
        date: str,
        tour: str = "atp"
    ) -> List[Dict]:
        """
        Get matches scheduled for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            tour: "atp" or "wta"

        Returns:
            List of match dicts
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/matches/{tour}/{date}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            return data.get("matches", [])
        except Exception as e:
            print(f"Error fetching matches for {date}: {e}")
            return []

    async def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """
        Get detailed player statistics.

        Args:
            player_id: Player ID from API

        Returns:
            Player stats dict or None
        """
        if not self.api_key:
            return None

        endpoint = f"{self.base_url}/player/{player_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return None

    async def get_head_to_head(
        self,
        player1_id: int,
        player2_id: int
    ) -> Optional[Dict]:
        """
        Get head-to-head record between two players.

        Args:
            player1_id: First player ID
            player2_id: Second player ID

        Returns:
            H2H dict with match history
        """
        if not self.api_key:
            return None

        endpoint = f"{self.base_url}/h2h/{player1_id}/{player2_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching H2H: {e}")
            return None

    async def get_player_form(
        self,
        player_id: int,
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get recent match results for a player.

        Args:
            player_id: Player ID
            num_matches: Number of recent matches to fetch

        Returns:
            List of recent match dicts
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/player/{player_id}/matches/recent"
        params = {"limit": num_matches}

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            return data.get("matches", [])
        except Exception as e:
            print(f"Error fetching player form: {e}")
            return []

    def parse_match_to_nexus_format(self, match_data: Dict) -> Dict:
        """
        Parse API-Tennis match data to NEXUS standard format.

        Args:
            match_data: Match dict from API-Tennis

        Returns:
            Match dict in NEXUS format
        """
        # Extract player info
        player1 = match_data.get("player1", {})
        player2 = match_data.get("player2", {})

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
            "sport": "tennis",
            "home_team": player1.get("name", ""),
            "away_team": player2.get("name", ""),
            "league": match_data.get("tournament", {}).get("name", ""),
            "country": match_data.get("tournament", {}).get("country", ""),
            "start_time": start_time,
            "home_player_id": player1.get("id"),
            "away_player_id": player2.get("id"),
            "home_ranking": player1.get("ranking"),
            "away_ranking": player2.get("ranking"),
        }

    async def get_match_details(
        self,
        player1_name: str,
        player2_name: str,
        date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get detailed match information including player stats.

        Args:
            player1_name: First player name
            player2_name: Second player name
            date: Match date (YYYY-MM-DD), defaults to today

        Returns:
            Complete match dict with player stats
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get matches for the date
        atp_matches = await self.get_matches_by_date(date, "atp")
        wta_matches = await self.get_matches_by_date(date, "wta")
        all_matches = atp_matches + wta_matches

        # Find matching event
        for match in all_matches:
            p1 = match.get("player1", {}).get("name", "").lower()
            p2 = match.get("player2", {}).get("name", "").lower()

            if (player1_name.lower() in p1 and player2_name.lower() in p2) or \
               (player1_name.lower() in p2 and player2_name.lower() in p1):

                # Get additional player stats
                player1_id = match.get("player1", {}).get("id")
                player2_id = match.get("player2", {}).get("id")

                player1_stats = None
                player2_stats = None
                h2h = None

                if player1_id and player2_id:
                    # Fetch in parallel
                    results = await asyncio.gather(
                        self.get_player_stats(player1_id),
                        self.get_player_stats(player2_id),
                        self.get_head_to_head(player1_id, player2_id),
                        return_exceptions=True
                    )

                    player1_stats = results[0] if not isinstance(results[0], Exception) else None
                    player2_stats = results[1] if not isinstance(results[1], Exception) else None
                    h2h = results[2] if not isinstance(results[2], Exception) else None

                # Combine data
                match_info = self.parse_match_to_nexus_format(match)
                match_info["player1_stats"] = player1_stats
                match_info["player2_stats"] = player2_stats
                match_info["h2h"] = h2h

                return match_info

        return None


# === HELPER FUNCTIONS ===

async def get_tennis_match_data(
    player1: str,
    player2: str,
    date: Optional[str] = None
) -> Optional[Dict]:
    """
    Convenience function to get complete tennis match data.

    Args:
        player1: First player name
        player2: Second player name
        date: Match date (YYYY-MM-DD)

    Returns:
        Complete match dict with stats
    """
    async with TennisAPIClient() as client:
        return await client.get_match_details(player1, player2, date)


async def get_tennis_rankings(tour: str = "atp") -> List[Dict]:
    """
    Get current tennis rankings.

    Args:
        tour: "atp" or "wta"

    Returns:
        List of player rankings
    """
    async with TennisAPIClient() as client:
        return await client.get_rankings(tour)


async def get_upcoming_tennis_matches(date: Optional[str] = None) -> List[Dict]:
    """
    Get upcoming tennis matches.

    Args:
        date: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        List of matches in NEXUS format
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    async with TennisAPIClient() as client:
        atp_matches = await client.get_matches_by_date(date, "atp")
        wta_matches = await client.get_matches_by_date(date, "wta")

        all_matches = []
        for match in atp_matches + wta_matches:
            all_matches.append(client.parse_match_to_nexus_format(match))

        return all_matches
