"""
Dataset Manager.

Orchestrates data collection from multiple sports sources.
Provides unified interface for fetching training data.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import json

from core.datasets.base import SportsDataSource, RawMatch, DatasetInfo
from core.datasets.basketball_data import BasketballDataSource
from core.datasets.tennis_data import TennisDataSource
from core.datasets.hockey_data import HockeyDataSource
from core.datasets.baseball_data import BaseballDataSource
from core.datasets.handball_data import HandballDataSource

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset collection."""
    sport: str
    start_date: datetime
    end_date: datetime
    leagues: List[str] = field(default_factory=list)
    min_matches: int = 1000
    max_concurrent: int = 3


@dataclass
class CollectionReport:
    """Report from dataset collection."""
    sport: str
    total_matches: int
    sources_used: List[str]
    date_range: tuple
    leagues_covered: List[str]
    quality_score: float  # 0-1
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport,
            "total_matches": self.total_matches,
            "sources_used": self.sources_used,
            "date_range": [d.isoformat() for d in self.date_range],
            "leagues_covered": self.leagues_covered,
            "quality_score": self.quality_score,
            "errors": self.errors,
        }


class DatasetManager:
    """
    Manages data collection from multiple sports sources.
    
    Usage:
        manager = DatasetManager()
        
        # Collect basketball data
        config = DatasetConfig(
            sport="basketball",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            leagues=["NBA"],
        )
        matches = await manager.collect(config)
        
        # Save for training
        manager.save_to_training_format(matches, "data/training/basketball_2023.csv")
    """
    
    # Registry of data sources
    SOURCES: Dict[str, Type[SportsDataSource]] = {
        "basketball": BasketballDataSource,
        "tennis": TennisDataSource,
        "hockey": HockeyDataSource,
        "baseball": BaseballDataSource,
        "handball": HandballDataSource,
    }
    
    def __init__(self):
        self._sources: Dict[str, SportsDataSource] = {}
        self._cache: Dict[str, List[RawMatch]] = {}
        
    def get_source(self, sport: str) -> Optional[SportsDataSource]:
        """Get or create data source for sport."""
        if sport not in self._sources:
            source_class = self.SOURCES.get(sport)
            if source_class:
                self._sources[sport] = source_class()
        return self._sources.get(sport)
    
    def list_available_sports(self) -> List[str]:
        """List all available sports."""
        return list(self.SOURCES.keys())
    
    def get_dataset_info(self, sport: str) -> Optional[DatasetInfo]:
        """Get metadata about a dataset."""
        source = self.get_source(sport)
        return source.info if source else None
    
    async def collect(
        self,
        config: DatasetConfig,
        use_cache: bool = True,
    ) -> List[RawMatch]:
        """
        Collect matches for a sport.
        
        Args:
            config: Collection configuration
            use_cache: Whether to use cached data
            
        Returns:
            List of raw matches
        """
        cache_key = f"{config.sport}_{config.start_date.date()}_{config.end_date.date()}"
        
        if use_cache and cache_key in self._cache:
            logger.info(f"Using cached data for {config.sport}")
            return self._cache[cache_key]
        
        source = self.get_source(config.sport)
        if not source:
            raise ValueError(f"No data source available for sport: {config.sport}")
        
        logger.info(f"Collecting {config.sport} data from {config.start_date.date()} to {config.end_date.date()}")
        
        matches = []
        errors = []
        
        # Collect for each league
        if config.leagues:
            for league in config.leagues:
                try:
                    league_matches = await source.fetch_matches(
                        start_date=config.start_date,
                        end_date=config.end_date,
                        league=league,
                    )
                    matches.extend(league_matches)
                    logger.info(f"Collected {len(league_matches)} matches from {league}")
                except Exception as e:
                    errors.append(f"{league}: {str(e)}")
                    logger.error(f"Error collecting {league}: {e}")
        else:
            # Collect all leagues
            try:
                matches = await source.fetch_matches(
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error collecting {config.sport}: {e}")
        
        # Preprocess matches
        matches = [source.preprocess(m) for m in matches]
        
        # Validate
        matches = [m for m in matches if source.validate_data(m)]
        
        logger.info(f"Total valid matches collected: {len(matches)}")
        
        # Cache results
        self._cache[cache_key] = matches
        
        return matches
    
    async def collect_multiple(
        self,
        configs: List[DatasetConfig],
    ) -> Dict[str, List[RawMatch]]:
        """
        Collect data for multiple sports concurrently.
        
        Args:
            configs: List of collection configurations
            
        Returns:
            Dictionary mapping sport to matches
        """
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent collections
        
        async def collect_with_limit(config: DatasetConfig) -> tuple:
            async with semaphore:
                matches = await self.collect(config)
                return config.sport, matches
        
        results = await asyncio.gather(*[
            collect_with_limit(config) for config in configs
        ])
        
        return {sport: matches for sport, matches in results}
    
    def generate_report(
        self,
        sport: str,
        matches: List[RawMatch],
    ) -> CollectionReport:
        """Generate collection report."""
        sources = list(set(m.source for m in matches))
        leagues = list(set(m.raw_data.get("league", "unknown") for m in matches if hasattr(m, 'raw_data')))
        
        if matches:
            dates = [m.match_date for m in matches]
            date_range = (min(dates), max(dates))
        else:
            date_range = (datetime.now(), datetime.now())
        
        # Calculate quality score based on data completeness
        if matches:
            completeness_scores = []
            for m in matches:
                score = 0
                if m.home_score is not None and m.away_score is not None:
                    score += 0.4
                if m.home_stats:
                    score += 0.3
                if m.away_stats:
                    score += 0.3
                completeness_scores.append(score)
            quality_score = sum(completeness_scores) / len(completeness_scores)
        else:
            quality_score = 0.0
        
        return CollectionReport(
            sport=sport,
            total_matches=len(matches),
            sources_used=sources,
            date_range=date_range,
            leagues_covered=leagues,
            quality_score=quality_score,
        )
    
    def save_to_training_format(
        self,
        matches: List[RawMatch],
        output_path: str,
        format: str = "csv",
    ) -> bool:
        """
        Save matches to training format.
        
        Args:
            matches: List of matches
            output_path: Output file path
            format: Output format (csv, json)
            
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "csv":
                import csv
                
                # Flatten match data for CSV
                rows = []
                for m in matches:
                    row = {
                        "match_id": m.match_id,
                        "source": m.source,
                        "sport": m.sport,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "match_date": m.match_date.strftime("%Y-%m-%d"),
                        "season": m.season,
                        "home_score": m.home_score,
                        "away_score": m.away_score,
                        "result": m.result,
                        **{f"home_{k}": v for k, v in m.home_stats.items()},
                        **{f"away_{k}": v for k, v in m.away_stats.items()},
                    }
                    rows.append(row)
                
                if rows:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                        
            elif format == "json":
                data = [
                    {
                        "match_id": m.match_id,
                        "source": m.source,
                        "sport": m.sport,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "match_date": m.match_date.isoformat(),
                        "season": m.season,
                        "home_score": m.home_score,
                        "away_score": m.away_score,
                        "result": m.result,
                        "home_stats": m.home_stats,
                        "away_stats": m.away_stats,
                        "raw_data": m.raw_data,
                    }
                    for m in matches
                ]
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(matches)} matches to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving matches: {e}")
            return False
    
    def clear_cache(self):
        """Clear cached data."""
        self._cache.clear()
        logger.info("Cache cleared")
