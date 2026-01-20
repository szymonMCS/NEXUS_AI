# data/scrapers/__init__.py
"""
Web scrapers for NEXUS AI.

Playwright-based scrapers for JS-heavy sites.
"""

try:
    from data.scrapers.flashscore_scraper import FlashscoreScraper
    __all__ = ["FlashscoreScraper"]
except ImportError:
    # Playwright not installed
    __all__ = []
