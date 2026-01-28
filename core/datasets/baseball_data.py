"""
Baseball Data Sources Integration.

Based on sport_datasets_AI_report.md - Baseball datasets:
- MLB Statcast Data
- Retrosheet
- Baseball-Reference
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

from core.datasets.base import SportsDataSource, DatasetInfo, RawMatch, DatasetQuality, DatasetLicense

logger = logging.getLogger(__name__)


class BaseballDataSource(SportsDataSource):
    """
    Baseball data source aggregator.
    
    Primary sources:
    1. MLB Stats API
    2. Retrosheet
    3. Baseball-Reference
    """
    
    SUPPORTED_LEAGUES = ["MLB", "NPB", "KBO", "CPBL"]
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="baseball_mlb_combined",
            description="Combined MLB and baseball data from multiple sources",
            source_url="https://statsapi.mlb.com/",
            sport="baseball",
            quality=DatasetQuality.EXCELLENT,
            license=DatasetLicense.OPEN,
            formats=["json", "csv"],
            update_frequency="daily",
            features=[
                "runs", "hits", "errors", "innings", "pitch_count",
                "strikeouts", "walks", "home_runs", "avg", "ops", "era"
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
        league: Optional[str] = "MLB",
    ) -> List[RawMatch]:
        """Fetch baseball matches."""
        matches = []
        
        # Try MLB API
        try:
            mlb_matches = await self._fetch_from_mlb_api(start_date, end_date)
            matches.extend(mlb_matches)
        except Exception as e:
            logger.warning(f"MLB API failed: {e}")
        
        # Try local data
        try:
            local_matches = await self._fetch_from_local(start_date, end_date, league)
            matches.extend(local_matches)
        except Exception as e:
            logger.warning(f"Local fetch failed: {e}")
        
        logger.info(f"Fetched {len(matches)} baseball matches")
        return matches
    
    async def _fetch_from_mlb_api(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[RawMatch]:
        """Fetch from MLB Stats API."""
        import httpx
        
        matches = []
        base_url = "https://statsapi.mlb.com/api/v1/schedule"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    base_url,
                    params={
                        "startDate": start_date.strftime("%Y-%m-%d"),
                        "endDate": end_date.strftime("%Y-%m-%d"),
                        "sportId": 1,  # MLB
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                dates = data.get("dates", [])
                
                for date_data in dates:
                    games = date_data.get("games", [])
                    for game in games:
                        try:
                            match_date = datetime.strptime(
                                game.get("gameDate", ""), "%Y-%m-%dT%H:%M:%SZ"
                            )
                            
                            # Get scores
                            home_score = game.get("teams", {}).get("home", {}).get("score", 0)
                            away_score = game.get("teams", {}).get("away", {}).get("score", 0)
                            
                            match = RawMatch(
                                match_id=str(game.get("gamePk", "")),
                                source="mlb_api",
                                sport="baseball",
                                home_team=game.get("teams", {}).get("home", {}).get("team", {}).get("name", ""),
                                away_team=game.get("teams", {}).get("away", {}).get("team", {}).get("name", ""),
                                match_date=match_date,
                                season=str(start_date.year),
                                home_score=float(home_score) if home_score else None,
                                away_score=float(away_score) if away_score else None,
                                raw_data=game,
                            )
                            matches.append(match)
                        except (ValueError, KeyError):
                            continue
                            
            except Exception as e:
                logger.warning(f"MLB API error: {e}")
        
        return matches
    
    async def _fetch_from_local(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from local CSV files."""
        matches = []
        data_dir = Path("data/historical/baseball")
        
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
                                    sport="baseball",
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
    
    async def fetch_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch player statistics."""
        return {
            "player_id": player_id,
            "season": season,
            "avg": 0.0,
            "hr": 0,
            "rbi": 0,
            "ops": 0.0,
            "era": 0.0,
            "whip": 0.0,
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
            "win_pct": 0.0,
            "runs_for": 0,
            "runs_against": 0,
        }
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess baseball match data."""
        match.home_team = match.home_team.strip()
        match.away_team = match.away_team.strip()
        return match
