"""
Tennis Data Sources Integration.

Based on sport_datasets_AI_report.md - Tennis datasets:
- ATP/WTA match data
- Tennis rankings
- Historical match statistics
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

from core.datasets.base import SportsDataSource, DatasetInfo, RawMatch, DatasetQuality, DatasetLicense

logger = logging.getLogger(__name__)


class TennisDataSource(SportsDataSource):
    """
    Tennis data source aggregator.
    
    Primary sources:
    1. ATP/WTA official data
    2. Jeff Sackmann's tennis datasets (GitHub)
    3. Tennis-Abstract
    """
    
    SURFACES = ["hard", "clay", "grass", "carpet"]
    LEVELS = ["G", "M", "A", "2", "3"]  # Grand Slam, Masters, ATP, ATP250, Challenger
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="tennis_atp_wta_combined",
            description="Combined ATP and WTA tennis data from multiple sources",
            source_url="https://github.com/JeffSackmann/tennis_atp",
            sport="tennis",
            quality=DatasetQuality.EXCELLENT,
            license=DatasetLicense.OPEN,
            formats=["csv"],
            update_frequency="weekly",
            features=[
                "surface", "tourney_level", "winner_rank", "loser_rank",
                "score", "minutes", "aces", "double_faults", "serve_points",
                "first_serve_pct", "break_points", "return_points"
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
        league: Optional[str] = None,  # ATP, WTA, or None for both
    ) -> List[RawMatch]:
        """Fetch tennis matches."""
        matches = []
        
        # Try Jeff Sackmann's datasets (GitHub)
        for tour in ([league] if league else ["atp", "wta"]):
            try:
                tour_matches = await self._fetch_from_sackmann(start_date, end_date, tour)
                matches.extend(tour_matches)
            except Exception as e:
                logger.warning(f"Sackmann dataset failed for {tour}: {e}")
        
        # Try local data
        try:
            local_matches = await self._fetch_from_local(start_date, end_date, league)
            matches.extend(local_matches)
        except Exception as e:
            logger.warning(f"Local fetch failed: {e}")
        
        logger.info(f"Fetched {len(matches)} tennis matches")
        return matches
    
    async def _fetch_from_sackmann(
        self,
        start_date: datetime,
        end_date: datetime,
        tour: str,
    ) -> List[RawMatch]:
        """Fetch from Jeff Sackmann's tennis datasets."""
        import httpx
        
        matches = []
        base_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master"
        
        # Determine which files to fetch based on year range
        years = range(start_date.year, end_date.year + 1)
        
        async with httpx.AsyncClient() as client:
            for year in years:
                url = f"{base_url}/{tour}_matches_{year}.csv"
                try:
                    response = await client.get(url, timeout=30.0)
                    if response.status_code != 200:
                        continue
                    
                    # Parse CSV
                    content = response.text
                    lines = content.strip().split('\n')
                    if len(lines) < 2:
                        continue
                    
                    headers = lines[0].split(',')
                    
                    for line in lines[1:]:
                        try:
                            values = line.split(',')
                            row = dict(zip(headers, values))
                            
                            match_date = datetime.strptime(
                                row.get("tourney_date", "20000101"), 
                                "%Y%m%d"
                            )
                            
                            if start_date <= match_date <= end_date:
                                match = RawMatch(
                                    match_id=f"{tour}_{row.get('match_num', len(matches))}_{year}",
                                    source=f"sackmann_{tour}",
                                    sport="tennis",
                                    home_team=row.get("winner_name", ""),
                                    away_team=row.get("loser_name", ""),
                                    match_date=match_date,
                                    season=str(year),
                                    home_score=1,  # Winner
                                    away_score=0,  # Loser
                                    result="H",
                                    home_stats={
                                        "rank": row.get("winner_rank", ""),
                                        "rank_points": row.get("winner_rank_points", ""),
                                        "ace": row.get("w_ace", 0),
                                        "df": row.get("w_df", 0),
                                        "svpt": row.get("w_svpt", 0),
                                        "first_in": row.get("w_1stIn", 0),
                                        "first_won": row.get("w_1stWon", 0),
                                        "second_won": row.get("w_2ndWon", 0),
                                        "bp_saved": row.get("w_bpSaved", 0),
                                        "bp_faced": row.get("w_bpFaced", 0),
                                    },
                                    away_stats={
                                        "rank": row.get("loser_rank", ""),
                                        "rank_points": row.get("loser_rank_points", ""),
                                        "ace": row.get("l_ace", 0),
                                        "df": row.get("l_df", 0),
                                        "svpt": row.get("l_svpt", 0),
                                        "first_in": row.get("l_1stIn", 0),
                                        "first_won": row.get("l_1stWon", 0),
                                        "second_won": row.get("l_2ndWon", 0),
                                        "bp_saved": row.get("l_bpSaved", 0),
                                        "bp_faced": row.get("l_bpFaced", 0),
                                    },
                                    raw_data={
                                        "surface": row.get("surface", ""),
                                        "tourney_level": row.get("tourney_level", ""),
                                        "tourney_name": row.get("tourney_name", ""),
                                        "round": row.get("round", ""),
                                        "score": row.get("score", ""),
                                        "minutes": row.get("minutes", ""),
                                    }
                                )
                                matches.append(match)
                        except (ValueError, IndexError):
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error fetching {year}: {e}")
        
        return matches
    
    async def _fetch_from_local(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str],
    ) -> List[RawMatch]:
        """Fetch from local CSV files."""
        matches = []
        data_dir = Path("data/historical/tennis")
        
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
                                    match_id=row.get("match_id", f"local_{len(matches)}"),
                                    source="local_csv",
                                    sport="tennis",
                                    home_team=row.get("player1", ""),
                                    away_team=row.get("player2", ""),
                                    match_date=match_date,
                                    home_score=float(row.get("p1_games", 0)),
                                    away_score=float(row.get("p2_games", 0)),
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
            "rank": 0,
            "rank_points": 0,
            "win_rate": 0.0,
            "surface_win_rates": {s: 0.0 for s in self.SURFACES},
        }
    
    async def fetch_team_stats(
        self,
        team_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch team statistics - not applicable for tennis."""
        return {}
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess tennis match data."""
        # Normalize player names
        match.home_team = match.home_team.strip().title()
        match.away_team = match.away_team.strip().title()
        
        return match
