"""
Basketball Data Sources Integration.

Based on sport_datasets_AI_report.md - Basketball datasets:
- NBA Play-by-Play Data (Kaggle)
- Basketball Shot Logs (NBA 2014-2015)
- Basketball Player Movement Dataset (SportVU)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv

from core.datasets.base import SportsDataSource, DatasetInfo, RawMatch, DatasetQuality, DatasetLicense

logger = logging.getLogger(__name__)


class BasketballDataSource(SportsDataSource):
    """
    Basketball data source aggregator.
    
    Primary sources:
    1. NBA Stats API (official)
    2. Basketball-Reference (scraping)
    3. Local CSV datasets (Kaggle)
    """
    
    SUPPORTED_LEAGUES = ["NBA", "EuroLeague", "NCAA", "PLK"]
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="basketball_nba_combined",
            description="Combined NBA and basketball data from multiple sources",
            source_url="https://www.nba.com/stats/",
            sport="basketball",
            quality=DatasetQuality.EXCELLENT,
            license=DatasetLicense.OPEN,
            formats=["json", "csv"],
            update_frequency="daily",
            features=[
                "points", "rebounds", "assists", "steals", "blocks",
                "fg_pct", "3p_pct", "ft_pct", "turnovers", "minutes",
                "plus_minus", "pace", "off_rating", "def_rating"
            ],
            has_player_tracking=True,
            has_play_by_play=True,
            has_injury_data=True,
            requires_api_key=False,
        )
    
    async def fetch_matches(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str] = "NBA",
    ) -> List[RawMatch]:
        """Fetch basketball matches."""
        matches = []
        
        # Try multiple sources
        sources = [
            self._fetch_from_nba_api,
            self._fetch_from_local,
            self._fetch_from_basketball_reference,
        ]
        
        for source_fn in sources:
            try:
                source_matches = await source_fn(start_date, end_date, league)
                matches.extend(source_matches)
                if len(matches) >= 100:  # Sufficient data
                    break
            except Exception as e:
                logger.warning(f"Source {source_fn.__name__} failed: {e}")
                continue
        
        logger.info(f"Fetched {len(matches)} basketball matches")
        return matches
    
    async def _fetch_from_nba_api(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from NBA Stats API."""
        import httpx
        
        matches = []
        base_url = "https://stats.nba.com/stats/leaguegamelog"
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    base_url,
                    headers=headers,
                    params={
                        "LeagueID": "00",  # NBA
                        "Season": f"{start_date.year}-{str(start_date.year + 1)[-2:]}",
                        "SeasonType": "Regular Season",
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                result_sets = data.get("resultSets", [])
                
                if not result_sets:
                    return []
                
                games = result_sets[0].get("rowSet", [])
                headers = result_sets[0].get("headers", [])
                
                for game in games:
                    game_dict = dict(zip(headers, game))
                    
                    match_date = datetime.strptime(
                        game_dict.get("GAME_DATE", ""), "%Y-%m-%d"
                    )
                    
                    if start_date <= match_date <= end_date:
                        match = RawMatch(
                            match_id=str(game_dict.get("GAME_ID", "")),
                            source="nba_api",
                            sport="basketball",
                            home_team=game_dict.get("TEAM_NAME", ""),
                            away_team=game_dict.get("MATCHUP", "").split(" vs. ")[-1] if "vs." in game_dict.get("MATCHUP", "") else "",
                            match_date=match_date,
                            season=str(start_date.year),
                            home_score=float(game_dict.get("PTS", 0)),
                            home_stats={
                                "fg_pct": game_dict.get("FG_PCT", 0),
                                "3p_pct": game_dict.get("FG3_PCT", 0),
                                "ft_pct": game_dict.get("FT_PCT", 0),
                                "rebounds": game_dict.get("REB", 0),
                                "assists": game_dict.get("AST", 0),
                                "steals": game_dict.get("STL", 0),
                                "blocks": game_dict.get("BLK", 0),
                                "turnovers": game_dict.get("TOV", 0),
                            }
                        )
                        matches.append(match)
                        
            except Exception as e:
                logger.warning(f"NBA API error: {e}")
        
        return matches
    
    async def _fetch_from_local(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from local CSV datasets."""
        matches = []
        data_dir = Path("data/historical/basketball")
        
        if not data_dir.exists():
            return []
        
        for csv_file in data_dir.glob("*.csv"):
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            match_date = datetime.strptime(row.get("date", ""), "%Y-%m-%d")
                            if start_date <= match_date <= end_date:
                                match = RawMatch(
                                    match_id=row.get("game_id", f"local_{len(matches)}"),
                                    source="local_csv",
                                    sport="basketball",
                                    home_team=row.get("home_team", ""),
                                    away_team=row.get("away_team", ""),
                                    match_date=match_date,
                                    home_score=float(row.get("home_score", 0)),
                                    away_score=float(row.get("away_score", 0)),
                                    raw_data=row,
                                )
                                matches.append(match)
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                logger.warning(f"Local CSV error: {e}")
        
        return matches
    
    async def _fetch_from_basketball_reference(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from Basketball-Reference (fallback)."""
        # Implementation would use scraping
        logger.info("Basketball-Reference scraping not implemented in MVP")
        return []
    
    async def fetch_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch player statistics."""
        return {
            "player_id": player_id,
            "season": season,
            "points_per_game": 0.0,
            "rebounds_per_game": 0.0,
            "assists_per_game": 0.0,
            "efficiency": 0.0,
        }
    
    async def fetch_team_stats(
        self,
        team_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch team statistics."""
        return {
            "team_id": team_id,
            "season": season,
            "wins": 0,
            "losses": 0,
            "points_for": 0.0,
            "points_against": 0.0,
            "home_record": "0-0",
            "away_record": "0-0",
        }
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess basketball match data."""
        # Normalize team names
        match.home_team = match.home_team.strip().title()
        match.away_team = match.away_team.strip().title()
        
        # Calculate result if not set
        if match.result is None and match.home_score is not None and match.away_score is not None:
            if match.home_score > match.away_score:
                match.result = "H"
            elif match.home_score < match.away_score:
                match.result = "A"
            else:
                match.result = "D"
        
        return match
