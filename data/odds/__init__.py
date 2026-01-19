# data/odds/__init__.py
"""
Odds module - fetching, scraping, and analyzing betting odds.
"""

from data.odds.odds_api_client import (
    OddsAPIClient,
    OddsMarket,
    OddsFormat,
    get_odds_for_match,
    get_best_odds,
)

from data.odds.pl_scraper import (
    PolishBookmakerScraper,
    scrape_polish_odds,
    find_match_odds,
)

from data.odds.odds_merger import (
    OddsMerger,
    get_merged_odds_analysis,
)

__all__ = [
    # API Client
    "OddsAPIClient",
    "OddsMarket",
    "OddsFormat",
    "get_odds_for_match",
    "get_best_odds",
    # Polish Scraper
    "PolishBookmakerScraper",
    "scrape_polish_odds",
    "find_match_odds",
    # Merger
    "OddsMerger",
    "get_merged_odds_analysis",
]
