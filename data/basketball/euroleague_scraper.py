# data/basketball/euroleague_scraper.py
"""
Scraper for EuroLeague and basketball data from Sofascore.
Free alternative to paid APIs for Lite mode.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import re

from config.free_apis import SOFASCORE_CONFIG


class SofascoreBasketballScraper:
    """
    Scraper for basketball data from Sofascore.

    Provides:
    - NBA, EuroLeague, and other league schedules
    - Team statistics
    - Live scores
    - Standings
    """

    def __init__(self):
        self.base_url = SOFASCORE_CONFIG["base_url"]
        self.rate_limit = SOFASCORE_CONFIG["rate_limit"]
        self.session: Optional[httpx.AsyncClient] = None

        # Basketball is sport ID 2 on Sofascore
        self.sport_id = 2

    async def __aenter__(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def get_matches_by_date(self, date: str, league: str = "nba") -> List[Dict]:
        """
        Get basketball matches for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            league: League filter (nba, euroleague, etc.)

        Returns:
            List of match dicts
        """
        endpoint = f"{self.base_url}/sport/basketball/scheduled-events/{date}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            matches = []
            events = data.get("events", [])

            for event in events:
                # Filter by league if specified
                tournament_name = event.get("tournament", {}).get("name", "").lower()
                if league.lower() in tournament_name or league == "all":
                    matches.append(self._parse_event(event))

            # Rate limiting
            await asyncio.sleep(1.0 / self.rate_limit)

            return matches

        except Exception as e:
            print(f"Error fetching Sofascore basketball matches for {date}: {e}")
            return []

    async def get_team_stats(self, team_id: int) -> Optional[Dict]:
        """
        Get team statistics from Sofascore.

        Args:
            team_id: Sofascore team ID

        Returns:
            Team stats dict or None
        """
        endpoint = f"{self.base_url}/team/{team_id}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            await asyncio.sleep(1.0 / self.rate_limit)

            return data.get("team", {})

        except Exception as e:
            print(f"Error fetching team stats from Sofascore: {e}")
            return None

    async def get_h2h(self, team1_id: int, team2_id: int) -> Optional[Dict]:
        """
        Get head-to-head record between two teams.

        Args:
            team1_id: First team Sofascore ID
            team2_id: Second team Sofascore ID

        Returns:
            H2H dict or None
        """
        endpoint = f"{self.base_url}/h2h/basketball/team/{team1_id}/team/{team2_id}"

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

    async def get_team_form(
        self,
        team_id: int,
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get recent matches for a team.

        Args:
            team_id: Sofascore team ID
            num_matches: Number of matches to fetch

        Returns:
            List of recent match dicts
        """
        endpoint = f"{self.base_url}/team/{team_id}/events/last/{num_matches}"

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
            print(f"Error fetching team form from Sofascore: {e}")
            return []

    async def get_standings(self, tournament_id: int) -> List[Dict]:
        """
        Get league standings.

        Args:
            tournament_id: Sofascore tournament ID
                          (e.g., 132 for NBA, 7166 for EuroLeague)

        Returns:
            List of team standings
        """
        endpoint = f"{self.base_url}/tournament/{tournament_id}/standings"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            await asyncio.sleep(1.0 / self.rate_limit)

            # Sofascore returns standings in rows
            standings = []
            for row in data.get("standings", [])[0].get("rows", []):
                standings.append({
                    "team": row.get("team", {}).get("name"),
                    "position": row.get("position"),
                    "wins": row.get("wins"),
                    "losses": row.get("losses"),
                    "percentage": row.get("percentage"),
                })

            return standings

        except Exception as e:
            print(f"Error fetching standings from Sofascore: {e}")
            return []

    async def search_team(self, team_name: str) -> Optional[Dict]:
        """
        Search for a team by name.

        Args:
            team_name: Team name to search

        Returns:
            Team dict with ID or None
        """
        endpoint = f"{self.base_url}/search/{team_name}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(endpoint)
            response.raise_for_status()
            data = response.json()

            # Look for basketball teams in results
            for result in data.get("results", []):
                if result.get("type") == "team" and result.get("sport") == "basketball":
                    return result.get("entity", {})

            await asyncio.sleep(1.0 / self.rate_limit)

            return None

        except Exception as e:
            print(f"Error searching for team on Sofascore: {e}")
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
            "external_id": f"sofascore_bball_{event.get('id')}",
            "sport": "basketball",
            "home_team": home_team.get("name", ""),
            "away_team": away_team.get("name", ""),
            "league": tournament.get("name", ""),
            "country": tournament.get("category", {}).get("name", ""),
            "start_time": start_time,
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_score": event.get("homeScore", {}).get("current"),
            "away_score": event.get("awayScore", {}).get("current"),
            "status": event.get("status", {}).get("description", ""),
        }

    async def get_match_details(
        self,
        home_team: str,
        away_team: str,
        league: str = "nba",
        date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get complete match details for two teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League filter
            date: Match date (YYYY-MM-DD)

        Returns:
            Complete match dict with stats
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get matches for the date
        matches = await self.get_matches_by_date(date, league)

        # Find matching event
        for match in matches:
            home = match.get("home_team", "").lower()
            away = match.get("away_team", "").lower()

            if home_team.lower() in home and away_team.lower() in away:
                # Get additional stats
                home_id = match.get("home_team_id")
                away_id = match.get("away_team_id")

                if home_id and away_id:
                    # Fetch team stats and H2H
                    results = await asyncio.gather(
                        self.get_team_stats(home_id),
                        self.get_team_stats(away_id),
                        self.get_h2h(home_id, away_id),
                        self.get_team_form(home_id, 5),
                        self.get_team_form(away_id, 5),
                        return_exceptions=True
                    )

                    match["home_stats"] = results[0] if not isinstance(results[0], Exception) else None
                    match["away_stats"] = results[1] if not isinstance(results[1], Exception) else None
                    match["h2h"] = results[2] if not isinstance(results[2], Exception) else None
                    match["home_form"] = results[3] if not isinstance(results[3], Exception) else []
                    match["away_form"] = results[4] if not isinstance(results[4], Exception) else []

                return match

        return None


# === HELPER FUNCTIONS ===

async def scrape_basketball_match_data(
    home_team: str,
    away_team: str,
    league: str = "nba",
    date: Optional[str] = None
) -> Optional[Dict]:
    """
    Convenience function to scrape basketball match data from Sofascore.

    Args:
        home_team: Home team name
        away_team: Away team name
        league: League filter
        date: Match date (YYYY-MM-DD)

    Returns:
        Complete match dict with stats
    """
    async with SofascoreBasketballScraper() as scraper:
        return await scraper.get_match_details(home_team, away_team, league, date)


async def scrape_upcoming_basketball_matches(
    league: str = "nba",
    date: Optional[str] = None
) -> List[Dict]:
    """
    Scrape upcoming basketball matches from Sofascore.

    Args:
        league: League filter (nba, euroleague, etc.)
        date: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        List of matches in NEXUS format
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    async with SofascoreBasketballScraper() as scraper:
        return await scraper.get_matches_by_date(date, league)


async def scrape_basketball_standings(tournament_id: int = 132) -> List[Dict]:
    """
    Scrape basketball standings.

    Args:
        tournament_id: Tournament ID (132 for NBA, 7166 for EuroLeague)

    Returns:
        List of team standings
    """
    async with SofascoreBasketballScraper() as scraper:
        return await scraper.get_standings(tournament_id)
