# data/scrapers/__init__.py
"""
Web scrapers for NEXUS AI.

Playwright-based scrapers for JS-heavy sites.
"""

try:
    from data.scrapers.flashscore_scraper import (
        FlashscoreScraper, 
        get_upcoming_matches,
        scrape_flashscore_fixtures
    )
    __all__ = ["FlashscoreScraper", "get_upcoming_matches", "scrape_flashscore_fixtures"]
except ImportError:
    # Playwright not installed
    async def get_upcoming_matches(sport: str, league: str = None):
        return []
    __all__ = ["get_upcoming_matches"]
