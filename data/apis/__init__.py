# data/apis/__init__.py
"""
API clients for NEXUS AI.

Free/public API integrations + unified sports API.
Premium tier APIs auto-activate when keys are configured.

Usage:
    # Basic (free tier)
    from data.apis import UnifiedSportsAPI
    api = UnifiedSportsAPI()

    # Auto-tiered (best available)
    from data.apis import APITierManager
    manager = APITierManager()
    matches = await manager.get_football_matches("PL")

    # Check status
    from data.apis import check_api_status
    print(await check_api_status())
"""

from data.apis.thesportsdb_client import TheSportsDBClient
from data.apis.sports_api_client import (
    APIResponse,
    BaseAPIClient,
    OddsAPIClient,
    FootballDataClient,
    APISportsClient,
    PandaScoreClient,
    OpenF1Client,
    MLBStatsClient,
    NewsAPIClient,
    UnifiedSportsAPI,
)
from data.apis.baseball_scraper import (
    BaseballReferenceScraper,
    TeamStats,
    PlayerStats,
    GameResult,
)
from data.apis.premium_api_clients import (
    APITier,
    APIStatus,
    SportradarClient,
    StatsPerformClient,
    GeniusSportsClient,
    LSportsClient,
    SportsDataIOClient,
    APIFootballProClient,
)
from data.apis.api_tier_manager import (
    APITierManager,
    TierStatus,
    check_api_status,
)

__all__ = [
    # Legacy
    "TheSportsDBClient",
    # Response type
    "APIResponse",
    # Base
    "BaseAPIClient",
    # Free tier clients
    "OddsAPIClient",
    "FootballDataClient",
    "APISportsClient",
    "PandaScoreClient",
    "OpenF1Client",
    "MLBStatsClient",
    "NewsAPIClient",
    # Premium tier clients
    "SportradarClient",
    "StatsPerformClient",
    "GeniusSportsClient",
    "LSportsClient",
    "SportsDataIOClient",
    "APIFootballProClient",
    # Tier management
    "APITier",
    "APIStatus",
    "TierStatus",
    "APITierManager",
    "check_api_status",
    # Unified interface
    "UnifiedSportsAPI",
    # Scraper
    "BaseballReferenceScraper",
    "TeamStats",
    "PlayerStats",
    "GameResult",
]
