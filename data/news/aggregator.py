# data/news/aggregator.py
"""
News aggregator - collects news from multiple sources (Brave, Serper, NewsAPI).
Deduplicates and ranks by relevance.
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

from config.settings import settings, NEWS_CONFIG


class NewsArticle:
    """Represents a news article"""

    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        snippet: str = "",
        published_date: Optional[datetime] = None,
    ):
        self.title = title
        self.url = url
        self.source = source
        self.snippet = snippet
        self.published_date = published_date or datetime.now()
        self.relevance_score = 0.0
        self.content_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate content hash for deduplication"""
        content = f"{self.title.lower()}{self.url}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "relevance_score": self.relevance_score,
        }


class NewsAggregator:
    """
    Aggregates news from multiple sources.
    Handles rate limiting, deduplication, and ranking.
    """

    def __init__(self):
        self.brave_config = NEWS_CONFIG.get("brave_search", {})
        self.serper_config = NEWS_CONFIG.get("serper", {})
        self.newsapi_config = NEWS_CONFIG.get("newsapi", {})

    async def search_match_news(
        self,
        player1: str,
        player2: str,
        sport: str,
        max_results: int = 20,
    ) -> List[NewsArticle]:
        """
        Search for news about a specific match.

        Args:
            player1: First player/team name
            player2: Second player/team name
            sport: Sport type (tennis, basketball, etc.)
            max_results: Maximum number of results to return

        Returns:
            List of deduplicated and ranked NewsArticle objects
        """
        # Build search queries
        queries = self._build_search_queries(player1, player2, sport)

        # Collect from all sources in parallel
        tasks = []

        if self.brave_config.get("enabled"):
            tasks.append(self._search_brave(queries))

        if self.serper_config.get("enabled"):
            tasks.append(self._search_serper(queries))

        if self.newsapi_config.get("enabled") and settings.is_pro_mode:
            tasks.append(self._search_newsapi(queries))

        # Execute all searches
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and deduplicate
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Deduplicate
        articles = self._deduplicate(all_articles)

        # Rank by relevance
        articles = self._rank_articles(articles, player1, player2)

        # Return top results
        return articles[:max_results]

    def _build_search_queries(self, player1: str, player2: str, sport: str) -> List[str]:
        """Build search query variations"""
        queries = [
            f"{player1} vs {player2} {sport}",
            f"{player1} {player2} prediction",
            f"{player1} news injury",
            f"{player2} news injury",
            f"{player1} {player2} preview",
        ]
        return queries

    async def _search_brave(self, queries: List[str]) -> List[NewsArticle]:
        """Search using Brave Search API"""
        if not self.brave_config.get("api_key"):
            return []

        articles = []
        endpoint = self.brave_config["endpoint"]
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_config["api_key"],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries[:3]:  # Limit to 3 queries to save API calls
                try:
                    params = {
                        "q": query,
                        "count": 5,
                        "freshness": "pw",  # Past week
                    }

                    response = await client.get(endpoint, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()

                    # Parse Brave results
                    for result in data.get("web", {}).get("results", []):
                        article = NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            source="brave",
                            snippet=result.get("description", ""),
                            published_date=self._parse_date(result.get("age")),
                        )
                        articles.append(article)

                    # Rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"Error searching Brave: {e}")
                    continue

        return articles

    async def _search_serper(self, queries: List[str]) -> List[NewsArticle]:
        """Search using Serper (Google Search) API"""
        if not self.serper_config.get("api_key"):
            return []

        articles = []
        endpoint = self.serper_config["endpoint"]
        headers = {
            "X-API-KEY": self.serper_config["api_key"],
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries[:3]:
                try:
                    payload = {
                        "q": query,
                        "num": 5,
                        "tbm": "nws",  # News search
                    }

                    response = await client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    # Parse Serper results
                    for result in data.get("news", []):
                        article = NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("link", ""),
                            source="serper",
                            snippet=result.get("snippet", ""),
                            published_date=self._parse_date(result.get("date")),
                        )
                        articles.append(article)

                    await asyncio.sleep(0.3)

                except Exception as e:
                    print(f"Error searching Serper: {e}")
                    continue

        return articles

    async def _search_newsapi(self, queries: List[str]) -> List[NewsArticle]:
        """Search using NewsAPI (Pro mode only)"""
        if not self.newsapi_config.get("api_key"):
            return []

        articles = []
        endpoint = self.newsapi_config["endpoint"]
        headers = {"X-Api-Key": self.newsapi_config["api_key"]}

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries[:2]:
                try:
                    params = {
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 5,
                    }

                    response = await client.get(endpoint, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()

                    for result in data.get("articles", []):
                        article = NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            source="newsapi",
                            snippet=result.get("description", ""),
                            published_date=self._parse_date(result.get("publishedAt")),
                        )
                        articles.append(article)

                    await asyncio.sleep(1.0)

                except Exception as e:
                    print(f"Error searching NewsAPI: {e}")
                    continue

        return articles

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on content hash"""
        seen_hashes = set()
        unique_articles = []

        for article in articles:
            if article.content_hash not in seen_hashes:
                seen_hashes.add(article.content_hash)
                unique_articles.append(article)

        return unique_articles

    def _rank_articles(
        self,
        articles: List[NewsArticle],
        player1: str,
        player2: str,
    ) -> List[NewsArticle]:
        """
        Rank articles by relevance.

        Scoring factors:
        - Mentions both players (high priority)
        - Recent publication date
        - Keywords: injury, preview, prediction, analysis
        """
        keywords_high = ["injury", "injured", "out", "doubtful"]
        keywords_medium = ["preview", "prediction", "analysis", "vs", "match"]

        for article in articles:
            score = 0.0
            text = f"{article.title.lower()} {article.snippet.lower()}"

            # Both players mentioned
            if player1.lower() in text and player2.lower() in text:
                score += 0.5

            # Either player mentioned
            elif player1.lower() in text or player2.lower() in text:
                score += 0.3

            # High priority keywords
            for keyword in keywords_high:
                if keyword in text:
                    score += 0.3

            # Medium priority keywords
            for keyword in keywords_medium:
                if keyword in text:
                    score += 0.1

            # Recency bonus (last 24h)
            if article.published_date:
                age_hours = (datetime.now() - article.published_date).total_seconds() / 3600
                if age_hours < 24:
                    score += 0.2
                elif age_hours < 72:
                    score += 0.1

            article.relevance_score = min(score, 1.0)

        # Sort by relevance
        articles.sort(key=lambda x: x.relevance_score, reverse=True)

        return articles

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None

        try:
            # ISO format
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

            # Relative dates from Brave (e.g., "2 days ago")
            if "ago" in date_str.lower():
                parts = date_str.lower().split()
                if len(parts) >= 2:
                    value = int(parts[0])
                    unit = parts[1]

                    if "hour" in unit:
                        return datetime.now() - timedelta(hours=value)
                    elif "day" in unit:
                        return datetime.now() - timedelta(days=value)
                    elif "week" in unit:
                        return datetime.now() - timedelta(weeks=value)

        except Exception:
            pass

        return None


# === HELPER FUNCTIONS ===

async def get_match_news(
    home_player: str,
    away_player: str,
    sport: str,
    max_results: int = 20,
) -> List[Dict]:
    """
    Convenience function to get news for a match.

    Returns:
        List of article dictionaries
    """
    aggregator = NewsAggregator()
    articles = await aggregator.search_match_news(
        home_player, away_player, sport, max_results
    )
    return [article.to_dict() for article in articles]
