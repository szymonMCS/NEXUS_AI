# data/collectors/__init__.py
"""
Data collectors for NEXUS AI.

Multi-source data collection with deduplication and merging.
"""

from data.collectors.fixture_collector import (
    FixtureCollector,
    collect_fixtures,
    collect_and_enrich_fixtures,
)

__all__ = [
    "FixtureCollector",
    "collect_fixtures",
    "collect_and_enrich_fixtures",
]
