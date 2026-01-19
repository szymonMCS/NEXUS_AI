# data/basketball/__init__.py
"""
Basketball data module - API client and scraper for basketball match data.
"""

from data.basketball.bets_api_client import (
    BetsAPIBasketballClient,
    get_basketball_match_data,
    get_basketball_standings,
    get_upcoming_basketball_matches,
)

from data.basketball.euroleague_scraper import (
    SofascoreBasketballScraper,
    scrape_basketball_match_data,
    scrape_upcoming_basketball_matches,
    scrape_basketball_standings,
)

__all__ = [
    # API Client
    "BetsAPIBasketballClient",
    "get_basketball_match_data",
    "get_basketball_standings",
    "get_upcoming_basketball_matches",
    # Sofascore Scraper
    "SofascoreBasketballScraper",
    "scrape_basketball_match_data",
    "scrape_upcoming_basketball_matches",
    "scrape_basketball_standings",
]
