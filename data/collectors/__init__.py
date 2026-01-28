# data/collectors/__init__.py
"""
Data collectors for NEXUS AI.

Multi-source data collection with deduplication and merging.

Usage:
    # Collect fixtures (upcoming matches)
    from data.collectors import collect_fixtures
    fixtures = await collect_fixtures(sport="football")

    # Collect historical data for ML training
    from data.collectors import HistoricalDataCollector, collect_football_history
    results = await collect_football_history(leagues=["PL", "LaLiga"], days_back=365)
"""

from data.collectors.fixture_collector import (
    FixtureCollector,
    collect_fixtures,
    collect_and_enrich_fixtures,
)
from data.collectors.results import (
    CollectionStatus,
    SourceResult,
    CollectedMatch,
    CollectionResult,
    CollectionConfig,
    DEFAULT_LEAGUES,
)
from data.collectors.historical_collector import (
    HistoricalDataCollector,
    DataSourceAdapter,
    FootballDataAdapter,
    APISportsFootballAdapter,
    APISportsBasketballAdapter,
    MLBAdapter,
    collect_football_history,
    collect_basketball_history,
)

__all__ = [
    # Fixture collection
    "FixtureCollector",
    "collect_fixtures",
    "collect_and_enrich_fixtures",
    # Results dataclasses
    "CollectionStatus",
    "SourceResult",
    "CollectedMatch",
    "CollectionResult",
    "CollectionConfig",
    "DEFAULT_LEAGUES",
    # Historical collection
    "HistoricalDataCollector",
    "DataSourceAdapter",
    "FootballDataAdapter",
    "APISportsFootballAdapter",
    "APISportsBasketballAdapter",
    "MLBAdapter",
    "collect_football_history",
    "collect_basketball_history",
]
