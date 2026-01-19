# data/tennis/__init__.py
"""
Tennis data module - API client and scraper for tennis match data.
"""

from data.tennis.api_tennis_client import (
    TennisAPIClient,
    get_tennis_match_data,
    get_tennis_rankings,
    get_upcoming_tennis_matches,
)

from data.tennis.sofascore_scraper import (
    SofascoreTennisScraper,
    scrape_tennis_match_data,
    scrape_upcoming_tennis_matches,
)

__all__ = [
    # API Client
    "TennisAPIClient",
    "get_tennis_match_data",
    "get_tennis_rankings",
    "get_upcoming_tennis_matches",
    # Sofascore Scraper
    "SofascoreTennisScraper",
    "scrape_tennis_match_data",
    "scrape_upcoming_tennis_matches",
]
