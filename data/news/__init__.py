# data/news/__init__.py
"""
News module - aggregation, validation, and injury extraction.
"""

from data.news.aggregator import (
    NewsAggregator,
    NewsArticle,
    get_match_news,
)

from data.news.validator import (
    NewsSourceValidator,
    validate_news_quality,
    get_confirmed_injuries,
)

from data.news.injury_extractor import (
    InjuryExtractor,
    InjuryInfo,
    extract_match_injuries,
)

__all__ = [
    # Aggregator
    "NewsAggregator",
    "NewsArticle",
    "get_match_news",
    # Validator
    "NewsSourceValidator",
    "validate_news_quality",
    "get_confirmed_injuries",
    # Injury Extractor
    "InjuryExtractor",
    "InjuryInfo",
    "extract_match_injuries",
]
