"""
Baseball Reference Scraper.

Web scraping for baseball data from baseball-reference.com.
Uses BeautifulSoup for parsing HTML.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, date
from dataclasses import dataclass
import httpx
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Team statistics."""
    team_name: str
    team_abbr: str
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    runs_scored: int = 0
    runs_allowed: int = 0
    run_diff: int = 0
    home_record: str = ""
    away_record: str = ""
    last_10: str = ""
    streak: str = ""


@dataclass
class PlayerStats:
    """Player statistics."""
    name: str
    team: str
    position: str = ""
    games: int = 0
    at_bats: int = 0
    runs: int = 0
    hits: int = 0
    home_runs: int = 0
    rbi: int = 0
    batting_avg: float = 0.0
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0


@dataclass
class GameResult:
    """Game result."""
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: str
    venue: str = ""


class BaseballReferenceScraper:
    """
    Scraper for baseball-reference.com.

    Note: Please be respectful of rate limits.
    Recommended: 1-2 requests per second max.
    """

    BASE_URL = "https://www.baseball-reference.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_standings(self, year: int = 2024) -> Dict[str, List[TeamStats]]:
        """
        Get MLB standings for a year.

        Returns dict with 'AL' and 'NL' keys containing team stats.
        """
        client = await self._get_client()
        url = f"{self.BASE_URL}/leagues/MLB/{year}-standings.shtml"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch standings: HTTP {response.status_code}")
                return {"AL": [], "NL": []}

            soup = BeautifulSoup(response.text, "html.parser")
            standings = {"AL": [], "NL": []}

            # Find standings tables
            for league in ["AL", "NL"]:
                table_id = f"standings_{league.upper()}"
                table = soup.find("table", {"id": table_id})
                if not table:
                    continue

                tbody = table.find("tbody")
                if not tbody:
                    continue

                for row in tbody.find_all("tr"):
                    if "thead" in row.get("class", []):
                        continue

                    cells = row.find_all(["th", "td"])
                    if len(cells) < 5:
                        continue

                    try:
                        team_link = cells[0].find("a")
                        team_name = team_link.text.strip() if team_link else cells[0].text.strip()

                        stats = TeamStats(
                            team_name=team_name,
                            team_abbr=self._extract_team_abbr(team_link.get("href", "") if team_link else ""),
                            wins=int(cells[1].text.strip() or 0),
                            losses=int(cells[2].text.strip() or 0),
                            win_pct=float(cells[3].text.strip() or 0),
                        )
                        standings[league].append(stats)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing row: {e}")
                        continue

            return standings

        except Exception as e:
            logger.exception(f"Error fetching standings: {e}")
            return {"AL": [], "NL": []}

    async def get_team_schedule(
        self,
        team_abbr: str,
        year: int = 2024
    ) -> List[GameResult]:
        """
        Get team schedule/results.

        Args:
            team_abbr: Team abbreviation (NYY, BOS, LAD, etc.)
            year: Season year
        """
        client = await self._get_client()
        url = f"{self.BASE_URL}/teams/{team_abbr}/{year}-schedule-scores.shtml"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch schedule: HTTP {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            games = []

            table = soup.find("table", {"id": "team_schedule"})
            if not table:
                return []

            tbody = table.find("tbody")
            if not tbody:
                return []

            for row in tbody.find_all("tr"):
                if "thead" in row.get("class", []):
                    continue

                cells = row.find_all(["th", "td"])
                if len(cells) < 10:
                    continue

                try:
                    # Parse game result
                    game_date = cells[0].text.strip()
                    opponent = cells[4].text.strip()
                    result = cells[6].text.strip()  # W or L
                    runs_scored = cells[7].text.strip()
                    runs_allowed = cells[8].text.strip()

                    if not runs_scored or not runs_allowed:
                        continue

                    # Determine home/away
                    is_home = "@" not in cells[3].text
                    opponent_clean = opponent.replace("@", "").strip()

                    if is_home:
                        home_team = team_abbr
                        away_team = opponent_clean
                        home_score = int(runs_scored)
                        away_score = int(runs_allowed)
                    else:
                        home_team = opponent_clean
                        away_team = team_abbr
                        home_score = int(runs_allowed)
                        away_score = int(runs_scored)

                    games.append(GameResult(
                        date=game_date,
                        home_team=home_team,
                        away_team=away_team,
                        home_score=home_score,
                        away_score=away_score,
                        winner=home_team if home_score > away_score else away_team
                    ))
                except (ValueError, IndexError) as e:
                    continue

            return games

        except Exception as e:
            logger.exception(f"Error fetching schedule: {e}")
            return []

    async def get_player_stats(
        self,
        player_id: str,
        year: int = 2024
    ) -> Optional[PlayerStats]:
        """
        Get player stats for a season.

        Args:
            player_id: Baseball reference player ID (e.g., "troutmi01")
            year: Season year
        """
        client = await self._get_client()
        url = f"{self.BASE_URL}/players/{player_id[0]}/{player_id}.shtml"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch player: HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Get player name
            name_elem = soup.find("h1", {"itemprop": "name"})
            player_name = name_elem.text.strip() if name_elem else player_id

            # Find batting stats table
            table = soup.find("table", {"id": "batting_standard"})
            if not table:
                return None

            tbody = table.find("tbody")
            if not tbody:
                return None

            # Find row for requested year
            for row in tbody.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 15:
                    continue

                year_cell = cells[0].text.strip()
                if str(year) not in year_cell:
                    continue

                try:
                    return PlayerStats(
                        name=player_name,
                        team=cells[2].text.strip(),
                        games=int(cells[5].text.strip() or 0),
                        at_bats=int(cells[7].text.strip() or 0),
                        runs=int(cells[8].text.strip() or 0),
                        hits=int(cells[9].text.strip() or 0),
                        home_runs=int(cells[13].text.strip() or 0),
                        rbi=int(cells[14].text.strip() or 0),
                        batting_avg=float(cells[18].text.strip() or 0),
                        obp=float(cells[19].text.strip() or 0),
                        slg=float(cells[20].text.strip() or 0),
                        ops=float(cells[21].text.strip() or 0),
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing player stats: {e}")
                    return None

            return None

        except Exception as e:
            logger.exception(f"Error fetching player stats: {e}")
            return None

    async def get_league_leaders(
        self,
        stat: str = "batting_avg",
        year: int = 2024
    ) -> List[Dict[str, Any]]:
        """
        Get league leaders for a stat.

        Args:
            stat: Stat to get leaders for (batting_avg, home_runs, rbi, etc.)
            year: Season year
        """
        client = await self._get_client()
        url = f"{self.BASE_URL}/leagues/MLB/{year}-batting-leaders.shtml"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch leaders: HTTP {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            leaders = []

            # Find the specific stat table
            tables = soup.find_all("table", {"class": "stats_table"})
            for table in tables:
                caption = table.find("caption")
                if caption and stat.replace("_", " ").lower() in caption.text.lower():
                    tbody = table.find("tbody")
                    if not tbody:
                        continue

                    for row in tbody.find_all("tr")[:10]:
                        cells = row.find_all(["th", "td"])
                        if len(cells) < 3:
                            continue

                        try:
                            player_link = cells[1].find("a")
                            leaders.append({
                                "rank": len(leaders) + 1,
                                "player": player_link.text.strip() if player_link else cells[1].text.strip(),
                                "team": cells[2].text.strip() if len(cells) > 2 else "",
                                "value": cells[-1].text.strip()
                            })
                        except (ValueError, IndexError):
                            continue

                    break

            return leaders

        except Exception as e:
            logger.exception(f"Error fetching leaders: {e}")
            return []

    def _extract_team_abbr(self, href: str) -> str:
        """Extract team abbreviation from URL."""
        match = re.search(r"/teams/(\w+)/", href)
        return match.group(1) if match else ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
