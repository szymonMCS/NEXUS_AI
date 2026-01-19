# data/odds/odds_api_client.py
"""
Client for The Odds API (https://the-odds-api.com/).
Fetches odds from 40+ bookmakers for various sports.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timezone
from enum import Enum

from config.settings import settings


class OddsMarket(str, Enum):
    """Available odds markets"""
    H2H = "h2h"  # Head to head (moneyline)
    SPREADS = "spreads"  # Point spreads / handicap
    TOTALS = "totals"  # Over/under

class OddsFormat(str, Enum):
    """Odds format"""
    DECIMAL = "decimal"
    AMERICAN = "american"


class OddsAPIClient:
    """
    Client for The Odds API.

    Supports:
    - Tennis, Basketball, American Football, Ice Hockey, Baseball
    - Multiple markets (H2H, Spreads, Totals)
    - 40+ bookmakers
    - Live and upcoming matches
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session: Optional[httpx.AsyncClient] = None

        # Sports mapping
        self.sports_map = {
            "tennis": "tennis_atp",  # Also: tennis_wta, tennis_atp_doubles
            "basketball": "basketball_nba",  # Also: basketball_euroleague
            "ice_hockey": "icehockey_nhl",
            "baseball": "baseball_mlb",
            "football": "americanfootball_nfl",
        }

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def get_sports(self) -> List[Dict]:
        """
        Get list of available sports.

        Returns:
            List of sport dicts with keys: key, group, title, active
        """
        if not self.api_key:
            return []

        endpoint = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}

        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)

            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching sports list: {e}")
            return []

    async def get_upcoming_matches(
        self,
        sport: str,
        regions: str = "eu,us",
        markets: str = "h2h",
        odds_format: OddsFormat = OddsFormat.DECIMAL,
    ) -> List[Dict]:
        """
        Get upcoming matches with odds.

        Args:
            sport: Sport key (e.g., "tennis", "basketball")
            regions: Comma-separated regions (eu, us, uk, au)
            markets: Comma-separated markets (h2h, spreads, totals)
            odds_format: Odds format (decimal or american)

        Returns:
            List of match dicts with odds from multiple bookmakers
        """
        if not self.api_key:
            return []

        # Map sport to API sport key
        sport_key = self.sports_map.get(sport, sport)

        endpoint = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format.value,
            "dateFormat": "iso",
        }

        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)

            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()

            matches = response.json()

            # Log remaining quota
            quota_remaining = response.headers.get("x-requests-remaining")
            if quota_remaining:
                print(f"The Odds API quota remaining: {quota_remaining}")

            return matches
        except Exception as e:
            print(f"Error fetching odds for {sport}: {e}")
            return []

    async def get_match_odds(
        self,
        sport: str,
        match_id: str,
        regions: str = "eu,us",
        markets: str = "h2h,spreads,totals",
        odds_format: OddsFormat = OddsFormat.DECIMAL,
    ) -> Optional[Dict]:
        """
        Get odds for a specific match.

        Args:
            sport: Sport key
            match_id: Match event ID from The Odds API
            regions: Bookmaker regions
            markets: Markets to fetch
            odds_format: Odds format

        Returns:
            Match dict with odds or None
        """
        if not self.api_key:
            return None

        sport_key = self.sports_map.get(sport, sport)

        endpoint = f"{self.base_url}/sports/{sport_key}/events/{match_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format.value,
            "dateFormat": "iso",
        }

        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)

            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching match odds: {e}")
            return None

    def parse_odds_to_standard_format(self, match_data: Dict) -> List[Dict]:
        """
        Parse The Odds API response to standard NEXUS format.

        Args:
            match_data: Match dict from The Odds API

        Returns:
            List of odds dicts in standard format
        """
        odds_list = []

        match_id = match_data.get("id")
        home_team = match_data.get("home_team")
        away_team = match_data.get("away_team")
        commence_time = match_data.get("commence_time")

        bookmakers = match_data.get("bookmakers", [])

        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get("key")
            markets = bookmaker.get("markets", [])

            for market in markets:
                market_key = market.get("key")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h":
                    # Moneyline odds
                    home_odds = None
                    away_odds = None

                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            home_odds = outcome.get("price")
                        elif outcome.get("name") == away_team:
                            away_odds = outcome.get("price")

                    if home_odds and away_odds:
                        odds_list.append({
                            "external_match_id": match_id,
                            "bookmaker": bookmaker_name,
                            "odds_type": "moneyline",
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                elif market_key == "spreads":
                    # Handicap odds
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            handicap_line = outcome.get("point")
                            handicap_home_odds = outcome.get("price")
                        elif outcome.get("name") == away_team:
                            handicap_away_odds = outcome.get("price")

                    odds_list.append({
                        "external_match_id": match_id,
                        "bookmaker": bookmaker_name,
                        "odds_type": "handicap",
                        "handicap_line": handicap_line,
                        "handicap_home_odds": handicap_home_odds,
                        "handicap_away_odds": handicap_away_odds,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                elif market_key == "totals":
                    # Over/Under odds
                    total_line = None
                    over_odds = None
                    under_odds = None

                    for outcome in outcomes:
                        if outcome.get("name") == "Over":
                            total_line = outcome.get("point")
                            over_odds = outcome.get("price")
                        elif outcome.get("name") == "Under":
                            under_odds = outcome.get("price")

                    if total_line and over_odds and under_odds:
                        odds_list.append({
                            "external_match_id": match_id,
                            "bookmaker": bookmaker_name,
                            "odds_type": "totals",
                            "total_line": total_line,
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

        return odds_list

    async def fetch_sport_odds(
        self,
        sport: str,
        markets: List[OddsMarket] = None
    ) -> Dict[str, List[Dict]]:
        """
        Fetch all odds for a sport.

        Args:
            sport: Sport name (tennis, basketball, etc.)
            markets: List of markets to fetch

        Returns:
            Dict mapping match IDs to list of odds
        """
        if markets is None:
            markets = [OddsMarket.H2H, OddsMarket.SPREADS, OddsMarket.TOTALS]

        markets_str = ",".join(m.value for m in markets)

        matches = await self.get_upcoming_matches(
            sport=sport,
            markets=markets_str,
            regions="eu,us"
        )

        odds_by_match = {}

        for match in matches:
            match_id = match.get("id")
            odds_list = self.parse_odds_to_standard_format(match)
            odds_by_match[match_id] = odds_list

        return odds_by_match


# === HELPER FUNCTIONS ===

async def get_odds_for_match(
    sport: str,
    home_team: str,
    away_team: str,
    match_date: Optional[datetime] = None
) -> List[Dict]:
    """
    Get odds for a specific match.

    Args:
        sport: Sport name
        home_team: Home team/player name
        away_team: Away team/player name
        match_date: Match date (for filtering)

    Returns:
        List of odds dicts from multiple bookmakers
    """
    async with OddsAPIClient() as client:
        matches = await client.get_upcoming_matches(
            sport=sport,
            markets="h2h,spreads,totals"
        )

        # Find matching event
        for match in matches:
            if (match.get("home_team") == home_team and
                match.get("away_team") == away_team):

                return client.parse_odds_to_standard_format(match)

        return []


async def get_best_odds(sport: str, selection: str = "home") -> Dict:
    """
    Get best available odds for a sport.

    Args:
        sport: Sport name
        selection: "home" or "away"

    Returns:
        Dict with best odds information
    """
    async with OddsAPIClient() as client:
        odds_by_match = await client.fetch_sport_odds(sport)

        best_odds = {
            "bookmaker": None,
            "odds": 0.0,
            "match_id": None,
        }

        for match_id, odds_list in odds_by_match.items():
            for odds in odds_list:
                if odds.get("odds_type") == "moneyline":
                    current_odds = odds.get(f"{selection}_odds", 0)
                    if current_odds > best_odds["odds"]:
                        best_odds = {
                            "bookmaker": odds.get("bookmaker"),
                            "odds": current_odds,
                            "match_id": match_id,
                        }

        return best_odds
