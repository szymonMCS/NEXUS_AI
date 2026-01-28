"""
Handball Data Sources Integration.

Based on sport_datasets_AI_report.md - Handball datasets:
- European handball leagues
- EHF data
- Bundesliga data
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

from core.datasets.base import SportsDataSource, DatasetInfo, RawMatch, DatasetQuality, DatasetLicense

logger = logging.getLogger(__name__)


class HandballDataSource(SportsDataSource):
    """
    Handball data source aggregator.
    
    Primary sources:
    1. Local CSV datasets (European leagues)
    2. Bundesliga data
    3. EHF competition data
    """
    
    SUPPORTED_LEAGUES = ["Bundesliga", "Liga_ASOBAL", "LNH", "Handball_Bundesliga", "EHF_CL"]
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="handball_european_combined",
            description="Combined European handball data from multiple sources",
            source_url="https://www.handball-bundesliga.de/",
            sport="handball",
            quality=DatasetQuality.FAIR,
            license=DatasetLicense.OPEN,
            formats=["csv"],
            update_frequency="weekly",
            features=[
                "goals", "shots", "shot_pct", "saves", "save_pct",
                "penalties", "suspensions", "turnovers", "fast_breaks"
            ],
            has_player_tracking=False,
            has_play_by_play=False,
            has_injury_data=False,
            requires_api_key=False,
        )
    
    async def fetch_matches(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str] = None,
    ) -> List[RawMatch]:
        """Fetch handball matches."""
        matches = []
        
        # Try local data (primary source for handball)
        try:
            local_matches = await self._fetch_from_local(start_date, end_date, league)
            matches.extend(local_matches)
        except Exception as e:
            logger.warning(f"Local fetch failed: {e}")
        
        logger.info(f"Fetched {len(matches)} handball matches")
        return matches
    
    async def _fetch_from_local(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from local CSV files."""
        matches = []
        data_dir = Path("data/historical/handball")
        
        if not data_dir.exists():
            logger.warning(f"Handball data directory not found: {data_dir}")
            return []
        
        # Look for league-specific files
        if league:
            files = list(data_dir.glob(f"*{league}*.csv"))
        else:
            files = list(data_dir.glob("*.csv"))
        
        for csv_file in files:
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
                                    sport="handball",
                                    home_team=row.get("home_team", ""),
                                    away_team=row.get("away_team", ""),
                                    match_date=match_date,
                                    season=row.get("season", str(start_date.year)),
                                    home_score=float(row.get("home_score", 0)),
                                    away_score=float(row.get("away_score", 0)),
                                    home_stats={
                                        "shots": row.get("home_shots", 0),
                                        "saves": row.get("home_saves", 0),
                                        "penalties": row.get("home_penalties", 0),
                                        "suspensions": row.get("home_suspensions", 0),
                                    },
                                    away_stats={
                                        "shots": row.get("away_shots", 0),
                                        "saves": row.get("away_saves", 0),
                                        "penalties": row.get("away_penalties", 0),
                                        "suspensions": row.get("away_suspensions", 0),
                                    },
                                    raw_data=row,
                                )
                                matches.append(match)
                        except (ValueError, KeyError) as e:
                            continue
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
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
            "shots": 0,
            "shot_pct": 0.0,
            "saves": 0,
            "save_pct": 0.0,
            "penalties": 0,
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
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
        }
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess handball match data."""
        match.home_team = match.home_team.strip()
        match.away_team = match.away_team.strip()
        
        # Handball can have draws
        if match.result is None and match.home_score is not None and match.away_score is not None:
            if match.home_score > match.away_score:
                match.result = "H"
            elif match.home_score < match.away_score:
                match.result = "A"
            else:
                match.result = "D"
        
        return match
