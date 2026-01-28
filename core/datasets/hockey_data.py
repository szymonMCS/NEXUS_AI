"""
Hockey Data Sources Integration.

Based on sport_datasets_AI_report.md - Hockey datasets:
- NHL Play-by-Play Data
- Kaggle NHL datasets
- Hockey statistics
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

from core.datasets.base import SportsDataSource, DatasetInfo, RawMatch, DatasetQuality, DatasetLicense

logger = logging.getLogger(__name__)


class HockeyDataSource(SportsDataSource):
    """
    Hockey data source aggregator.
    
    Primary sources:
    1. NHL API
    2. Kaggle NHL datasets
    3. Local CSV files
    """
    
    SUPPORTED_LEAGUES = ["NHL", "KHL", "SHL", "Liiga", "DEL"]
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="hockey_nhl_combined",
            description="Combined NHL and hockey data from multiple sources",
            source_url="https://api.nhle.com/",
            sport="hockey",
            quality=DatasetQuality.GOOD,
            license=DatasetLicense.OPEN,
            formats=["json", "csv"],
            update_frequency="daily",
            features=[
                "goals", "assists", "shots", "saves", "save_pct",
                "power_play", "penalty_kill", "hits", "blocks",
                "faceoff_pct", "time_on_ice"
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
        league: Optional[str] = "NHL",
    ) -> List[RawMatch]:
        """Fetch hockey matches."""
        matches = []
        
        # Try NHL API
        try:
            nhl_matches = await self._fetch_from_nhl_api(start_date, end_date)
            matches.extend(nhl_matches)
        except Exception as e:
            logger.warning(f"NHL API failed: {e}")
        
        # Try local data
        try:
            local_matches = await self._fetch_from_local(start_date, end_date, league)
            matches.extend(local_matches)
        except Exception as e:
            logger.warning(f"Local fetch failed: {e}")
        
        logger.info(f"Fetched {len(matches)} hockey matches")
        return matches
    
    async def _fetch_from_nhl_api(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[RawMatch]:
        """Fetch from NHL API."""
        import httpx
        
        matches = []
        base_url = "https://api.nhle.com/stats/rest/en/game"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    base_url,
                    params={
                        "cayenneExp": f"gameDate>='{start_date.strftime('%Y-%m-%d')}' and gameDate<='{end_date.strftime('%Y-%m-%d')}'",
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                games = data.get("data", [])
                
                for game in games:
                    try:
                        match_date = datetime.strptime(
                            game.get("gameDate", ""), "%Y-%m-%d"
                        )
                        
                        match = RawMatch(
                            match_id=str(game.get("id", "")),
                            source="nhl_api",
                            sport="hockey",
                            home_team=game.get("homeTeam", {}).get("name", ""),
                            away_team=game.get("awayTeam", {}).get("name", ""),
                            match_date=match_date,
                            season=str(start_date.year),
                            home_score=float(game.get("homeTeam", {}).get("score", 0)),
                            away_score=float(game.get("awayTeam", {}).get("score", 0)),
                            raw_data=game,
                        )
                        matches.append(match)
                    except (ValueError, KeyError):
                        continue
                        
            except Exception as e:
                logger.warning(f"NHL API error: {e}")
        
        return matches
    
    async def _fetch_from_local(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from local CSV files."""
        matches = []
        data_dir = Path("data/historical/hockey")
        
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
                                    sport="hockey",
                                    home_team=row.get("home_team", ""),
                                    away_team=row.get("away_team", ""),
                                    match_date=match_date,
                                    home_score=float(row.get("home_goals", 0)),
                                    away_score=float(row.get("away_goals", 0)),
                                    raw_data=row,
                                )
                                matches.append(match)
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                logger.warning(f"Local CSV error: {e}")
        
        return matches
    
    async def fetch_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch player statistics."""
        return {
            "player_id": player_id,
            "season": season,
            "goals": 0,
            "assists": 0,
            "points": 0,
            "plus_minus": 0,
            "save_pct": 0.0,
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
            "ot_losses": 0,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
        }
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess hockey match data."""
        match.home_team = match.home_team.strip()
        match.away_team = match.away_team.strip()
        return match
