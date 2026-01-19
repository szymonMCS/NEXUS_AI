# data/odds/pl_scraper.py
"""
Web scraper for Polish bookmakers (Fortuna, STS, Betclic).
Provides free odds data for Lite mode.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import re

from config.free_apis import PL_BOOKMAKERS_CONFIG


class PolishBookmakerScraper:
    """
    Scrapes odds from Polish bookmakers.

    Supported bookmakers:
    - Fortuna.pl
    - STS.pl
    - Betclic.pl
    """

    def __init__(self):
        self.config = PL_BOOKMAKERS_CONFIG
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
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

    async def scrape_fortuna(self, sport: str) -> List[Dict]:
        """
        Scrape odds from Fortuna.pl

        Args:
            sport: Sport name (tennis, basketball, etc.)

        Returns:
            List of odds dicts
        """
        fortuna_config = self.config.get("fortuna", {})
        base_url = fortuna_config.get("base_url", "")
        sport_path = fortuna_config.get("sports", {}).get(sport, "")

        if not base_url or not sport_path:
            return []

        url = f"{base_url}{sport_path}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            odds_list = []

            # Fortuna-specific parsing
            # This is a simplified example - actual implementation would need
            # to inspect the current HTML structure of Fortuna.pl
            matches = soup.find_all("div", class_=re.compile("match|event"))

            for match in matches:
                try:
                    # Extract team names
                    teams = match.find_all("span", class_=re.compile("team|player"))
                    if len(teams) >= 2:
                        home_team = teams[0].get_text(strip=True)
                        away_team = teams[1].get_text(strip=True)
                    else:
                        continue

                    # Extract odds
                    odds_elements = match.find_all("span", class_=re.compile("odd|coefficient"))
                    if len(odds_elements) >= 2:
                        home_odds = self._parse_odds(odds_elements[0].get_text(strip=True))
                        away_odds = self._parse_odds(odds_elements[1].get_text(strip=True))
                    else:
                        continue

                    if home_odds and away_odds:
                        odds_list.append({
                            "bookmaker": "fortuna",
                            "odds_type": "moneyline",
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                except Exception as e:
                    print(f"Error parsing Fortuna match: {e}")
                    continue

            # Rate limiting
            await asyncio.sleep(fortuna_config.get("rate_limit", 1))

            return odds_list

        except Exception as e:
            print(f"Error scraping Fortuna: {e}")
            return []

    async def scrape_sts(self, sport: str) -> List[Dict]:
        """
        Scrape odds from STS.pl

        Args:
            sport: Sport name

        Returns:
            List of odds dicts
        """
        sts_config = self.config.get("sts", {})
        base_url = sts_config.get("base_url", "")
        sport_path = sts_config.get("sports", {}).get(sport, "")

        if not base_url or not sport_path:
            return []

        url = f"{base_url}{sport_path}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            odds_list = []

            # STS-specific parsing
            # Note: This is a template - actual selectors depend on current STS.pl structure
            matches = soup.find_all("div", class_=re.compile("match|event|game"))

            for match in matches:
                try:
                    # Extract match info
                    teams_container = match.find("div", class_=re.compile("teams|participants"))
                    if not teams_container:
                        continue

                    teams = teams_container.find_all("span", class_=re.compile("team|name"))
                    if len(teams) < 2:
                        continue

                    home_team = teams[0].get_text(strip=True)
                    away_team = teams[1].get_text(strip=True)

                    # Extract odds
                    odds_container = match.find("div", class_=re.compile("odds|rates"))
                    if not odds_container:
                        continue

                    odds_values = odds_container.find_all("button", class_=re.compile("odd|rate"))
                    if len(odds_values) < 2:
                        continue

                    home_odds = self._parse_odds(odds_values[0].get_text(strip=True))
                    away_odds = self._parse_odds(odds_values[1].get_text(strip=True))

                    if home_odds and away_odds:
                        odds_list.append({
                            "bookmaker": "sts",
                            "odds_type": "moneyline",
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                except Exception as e:
                    print(f"Error parsing STS match: {e}")
                    continue

            await asyncio.sleep(sts_config.get("rate_limit", 1))

            return odds_list

        except Exception as e:
            print(f"Error scraping STS: {e}")
            return []

    async def scrape_betclic(self, sport: str) -> List[Dict]:
        """
        Scrape odds from Betclic.pl

        Args:
            sport: Sport name

        Returns:
            List of odds dicts
        """
        betclic_config = self.config.get("betclic", {})
        base_url = betclic_config.get("base_url", "")
        sport_path = betclic_config.get("sports", {}).get(sport, "")

        if not base_url or not sport_path:
            return []

        url = f"{base_url}{sport_path}"

        try:
            if not self.session:
                await self.__aenter__()

            response = await self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            odds_list = []

            # Betclic-specific parsing
            matches = soup.find_all("div", class_=re.compile("match|event|sport-event"))

            for match in matches:
                try:
                    # Extract teams
                    team_elements = match.find_all("span", class_=re.compile("team|competitor"))
                    if len(team_elements) < 2:
                        continue

                    home_team = team_elements[0].get_text(strip=True)
                    away_team = team_elements[1].get_text(strip=True)

                    # Extract odds
                    odds_buttons = match.find_all("button", class_=re.compile("odd|selection|button"))

                    # Filter for main moneyline odds (usually first 2)
                    main_odds = [btn for btn in odds_buttons if self._parse_odds(btn.get_text(strip=True))][:2]

                    if len(main_odds) >= 2:
                        home_odds = self._parse_odds(main_odds[0].get_text(strip=True))
                        away_odds = self._parse_odds(main_odds[1].get_text(strip=True))

                        if home_odds and away_odds:
                            odds_list.append({
                                "bookmaker": "betclic",
                                "odds_type": "moneyline",
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_odds": home_odds,
                                "away_odds": away_odds,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })

                except Exception as e:
                    print(f"Error parsing Betclic match: {e}")
                    continue

            await asyncio.sleep(betclic_config.get("rate_limit", 1))

            return odds_list

        except Exception as e:
            print(f"Error scraping Betclic: {e}")
            return []

    def _parse_odds(self, odds_text: str) -> Optional[float]:
        """
        Parse odds value from text.

        Args:
            odds_text: Odds string (e.g., "1.85", "2,50")

        Returns:
            Float odds value or None
        """
        try:
            # Clean text
            cleaned = odds_text.strip().replace(",", ".")

            # Extract number
            match = re.search(r"(\d+\.?\d*)", cleaned)
            if match:
                odds = float(match.group(1))
                # Validate odds range (typically 1.01 - 100.00)
                if 1.01 <= odds <= 100.0:
                    return odds
        except Exception:
            pass

        return None

    async def scrape_all_bookmakers(self, sport: str) -> List[Dict]:
        """
        Scrape all configured Polish bookmakers.

        Args:
            sport: Sport name

        Returns:
            Combined list of odds from all bookmakers
        """
        tasks = []

        if self.config.get("fortuna", {}).get("enabled"):
            tasks.append(self.scrape_fortuna(sport))

        if self.config.get("sts", {}).get("enabled"):
            tasks.append(self.scrape_sts(sport))

        if self.config.get("betclic", {}).get("enabled"):
            tasks.append(self.scrape_betclic(sport))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_odds = []
        for result in results:
            if isinstance(result, list):
                all_odds.extend(result)

        return all_odds

    def match_teams_fuzzy(self, team1: str, team2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy match team names (handles slight variations).

        Args:
            team1: First team name
            team2: Second team name
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            bool: True if teams match
        """
        # Simple normalization
        t1 = team1.lower().strip()
        t2 = team2.lower().strip()

        if t1 == t2:
            return True

        # Check if one contains the other
        if t1 in t2 or t2 in t1:
            return True

        # More sophisticated fuzzy matching could use libraries like:
        # - difflib.SequenceMatcher
        # - fuzzywuzzy
        # - rapidfuzz

        return False


# === HELPER FUNCTIONS ===

async def scrape_polish_odds(sport: str) -> List[Dict]:
    """
    Convenience function to scrape Polish bookmakers.

    Args:
        sport: Sport name (tennis, basketball, etc.)

    Returns:
        List of odds dicts
    """
    async with PolishBookmakerScraper() as scraper:
        return await scraper.scrape_all_bookmakers(sport)


async def find_match_odds(
    sport: str,
    home_team: str,
    away_team: str
) -> List[Dict]:
    """
    Find odds for a specific match from Polish bookmakers.

    Args:
        sport: Sport name
        home_team: Home team/player name
        away_team: Away team/player name

    Returns:
        List of matching odds dicts
    """
    async with PolishBookmakerScraper() as scraper:
        all_odds = await scraper.scrape_all_bookmakers(sport)

        matching_odds = []
        for odds in all_odds:
            if (scraper.match_teams_fuzzy(odds.get("home_team", ""), home_team) and
                scraper.match_teams_fuzzy(odds.get("away_team", ""), away_team)):
                matching_odds.append(odds)

        return matching_odds
