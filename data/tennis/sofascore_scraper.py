# data/tennis/sofascore_scraper.py
"""
Sofascore scraper for free tennis data.
Alternative to paid APIs for Lite mode.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

from config.free_apis import SOFASCORE_CONFIG


class SofascoreTennisScraper:
    """
    Scraper for Sofascore tennis data.

    Sofascore provides free API endpoints (no key required) with:
    - ATP/WTA match schedules
    - Player statistics
    - Live scores
    - Rankings
    - H2H records
    """

    def __init__(self):
        self.base_url = SOFASCORE_CONFIG["base_url"]
        self.rate_limit = SOFASCORE_CONFIG["rate_limit"]
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        # Full browser-like headers to avoid 403 blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def get_matches_by_date(self, date: str) -> List[Dict]:
        """
        Get tennis matches for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of match dicts
        """
        # Sofascore uses sport ID 5 for tennis
        endpoint = f"{self.base_url}/sport/tennis/scheduled-events/{date}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            matches = []
            events = data.get("events", [])

            for event in events:
                matches.append(self._parse_event(event))

            # Rate limiting
            await asyncio.sleep(1.0 / self.rate_limit)

            return matches

        except Exception as e:
            print(f"Error fetching Sofascore matches for {date}: {e}")
            return []

    async def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """
        Get player statistics from Sofascore.

        Args:
            player_id: Sofascore player ID

        Returns:
            Player stats dict or None
        """
        endpoint = f"{self.base_url}/player/{player_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            await asyncio.sleep(1.0 / self.rate_limit)

            return data.get("player", {})

        except Exception as e:
            print(f"Error fetching player stats from Sofascore: {e}")
            return None

    async def get_h2h(self, player1_id: int, player2_id: int) -> Optional[Dict]:
        """
        Get head-to-head record between two players.

        Args:
            player1_id: First player Sofascore ID
            player2_id: Second player Sofascore ID

        Returns:
            H2H dict or None
        """
        # Note: Sofascore H2H endpoint may require event ID
        # This is a simplified implementation
        endpoint = f"{self.base_url}/h2h/tennis/player/{player1_id}/player/{player2_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            await asyncio.sleep(1.0 / self.rate_limit)

            return data

        except Exception as e:
            print(f"Error fetching H2H from Sofascore: {e}")
            return None

    async def get_player_form(
        self,
        player_id: int,
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get recent matches for a player.

        Args:
            player_id: Sofascore player ID
            num_matches: Number of matches to fetch

        Returns:
            List of recent match dicts
        """
        endpoint = f"{self.base_url}/player/{player_id}/events/last/{num_matches}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            await asyncio.sleep(1.0 / self.rate_limit)

            events = data.get("events", [])
            return [self._parse_event(event) for event in events]

        except Exception as e:
            print(f"Error fetching player form from Sofascore: {e}")
            return []

    async def search_player(self, player_name: str) -> Optional[Dict]:
        """
        Search for a player by name.

        Args:
            player_name: Player name to search

        Returns:
            Player dict with ID or None
        """
        endpoint = f"{self.base_url}/search/{player_name}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            # Look for tennis players in results
            for result in data.get("results", []):
                if result.get("type") == "player" and result.get("sport") == "tennis":
                    return result.get("entity", {})

            await asyncio.sleep(1.0 / self.rate_limit)

            return None

        except Exception as e:
            print(f"Error searching for player on Sofascore: {e}")
            return None

    def _parse_event(self, event: Dict) -> Dict:
        """
        Parse Sofascore event to NEXUS format.

        Args:
            event: Sofascore event dict

        Returns:
            Match dict in NEXUS format
        """
        home_team = event.get("homeTeam", {})
        away_team = event.get("awayTeam", {})
        tournament = event.get("tournament", {})

        # Parse start time
        start_timestamp = event.get("startTimestamp")
        start_time = None
        if start_timestamp:
            start_time = datetime.fromtimestamp(start_timestamp)

        return {
            "external_id": f"sofascore_{event.get('id')}",
            "sport": "tennis",
            "home_team": home_team.get("name", ""),
            "away_team": away_team.get("name", ""),
            "league": tournament.get("name", ""),
            "country": tournament.get("category", {}).get("name", ""),
            "start_time": start_time,
            "home_player_id": home_team.get("id"),
            "away_player_id": away_team.get("id"),
            "home_ranking": home_team.get("ranking"),
            "away_ranking": away_team.get("ranking"),
            "status": event.get("status", {}).get("description", ""),
        }

    async def get_match_details(
        self,
        player1_name: str,
        player2_name: str,
        date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get complete match details for two players.

        Args:
            player1_name: First player name
            player2_name: Second player name
            date: Match date (YYYY-MM-DD)

        Returns:
            Complete match dict with stats
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get matches for the date
        matches = await self.get_matches_by_date(date)

        # Find matching event
        for match in matches:
            home = match.get("home_team", "").lower()
            away = match.get("away_team", "").lower()

            if (player1_name.lower() in home and player2_name.lower() in away) or \
               (player1_name.lower() in away and player2_name.lower() in home):

                # Get additional stats
                player1_id = match.get("home_player_id")
                player2_id = match.get("away_player_id")

                if player1_id and player2_id:
                    # Fetch player stats and H2H
                    results = await asyncio.gather(
                        self.get_player_stats(player1_id),
                        self.get_player_stats(player2_id),
                        self.get_h2h(player1_id, player2_id),
                        self.get_player_form(player1_id, 5),
                        self.get_player_form(player2_id, 5),
                        return_exceptions=True
                    )

                    match["player1_stats"] = results[0] if not isinstance(results[0], Exception) else None
                    match["player2_stats"] = results[1] if not isinstance(results[1], Exception) else None
                    match["h2h"] = results[2] if not isinstance(results[2], Exception) else None
                    match["player1_form"] = results[3] if not isinstance(results[3], Exception) else []
                    match["player2_form"] = results[4] if not isinstance(results[4], Exception) else []

                return match

        return None


# === HELPER FUNCTIONS ===

async def scrape_tennis_match_data(
    player1: str,
    player2: str,
    date: Optional[str] = None
) -> Optional[Dict]:
    """
    Convenience function to scrape tennis match data from Sofascore.

    Args:
        player1: First player name
        player2: Second player name
        date: Match date (YYYY-MM-DD)

    Returns:
        Complete match dict with stats
    """
    async with SofascoreTennisScraper() as scraper:
        return await scraper.get_match_details(player1, player2, date)


async def scrape_upcoming_tennis_matches(date: Optional[str] = None) -> List[Dict]:
    """
    Scrape upcoming tennis matches from Sofascore with TheSportsDB fallback.

    Args:
        date: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        List of matches in NEXUS format
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # Try Sofascore first
    async with SofascoreTennisScraper() as scraper:
        matches = await scraper.get_matches_by_date(date)
        if matches:
            return matches

    # Fallback to TheSportsDB if Sofascore fails
    print("Sofascore unavailable, falling back to TheSportsDB...")
    try:
        from data.apis.thesportsdb_client import TheSportsDBClient

        async with TheSportsDBClient() as client:
            events = await client.get_events_by_date(date, "Tennis")

            # Convert TheSportsDB format to NEXUS format
            matches = []
            for event in events:
                matches.append({
                    "external_id": f"thesportsdb_{event.get('idEvent')}",
                    "sport": "tennis",
                    "home_team": event.get("strHomeTeam", ""),
                    "away_team": event.get("strAwayTeam", ""),
                    "league": event.get("strLeague", ""),
                    "country": event.get("strCountry", ""),
                    "start_time": datetime.strptime(
                        f"{event.get('dateEvent', '')} {event.get('strTime', '00:00')}",
                        "%Y-%m-%d %H:%M"
                    ) if event.get("dateEvent") else None,
                    "status": event.get("strStatus", ""),
                })
            return matches
    except Exception as e:
        print(f"TheSportsDB fallback also failed: {e}")
        return []
