## 4. WARSTWA DANYCH - NEWS AGGREGATOR

### 4.1 `data/news/aggregator.py` - Agregacja newsów z wielu źródeł

```python
# data/news/aggregator.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Awaitable
import json
import hashlib
from dataclasses import dataclass
from config.settings import NEWS_CONFIG, settings
from core.utils.rate_limiter import RateLimiter

@dataclass
class NewsArticle:
    """Struktura pojedynczego artykułu"""
    title: str
    url: str
    snippet: Optional[str]
    source: str
    api_source: str
    published: Optional[str]
    timestamp: str
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "api_source": self.api_source,
            "published": self.published,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score
        }

    @property
    def hash(self) -> str:
        """Unikalny hash do deduplikacji"""
        return hashlib.md5(self.url.encode()).hexdigest()


class NewsAggregator:
    """
    Agreguje newsy z wielu źródeł: Brave Search, Serper, NewsAPI.
    Implementuje rate limiting, deduplikację i sortowanie.
    """

    def __init__(self):
        self.config = NEWS_CONFIG
        self.rate_limiter = RateLimiter()
        self.sources: List[Callable[[str], Awaitable[List[NewsArticle]]]] = []

        # Dodaj aktywne źródła w kolejności priorytetu
        if self.config["brave_search"]["enabled"]:
            self.sources.append(self._fetch_brave)

        if self.config["serper"]["enabled"]:
            self.sources.append(self._fetch_serper)

        if self.config["newsapi"]["enabled"]:
            self.sources.append(self._fetch_newsapi)

        if not self.sources:
            raise ValueError("No news sources configured! Set at least one API key.")

    async def get_match_news(
        self,
        home: str,
        away: str,
        sport: str,
        include_injury_search: bool = True
    ) -> Dict:
        """
        Pobierz wszystkie newsy o meczu z wielu źródeł.

        Args:
            home: Nazwa zawodnika/drużyny gospodarza
            away: Nazwa zawodnika/drużyny gościa
            sport: "tennis" lub "basketball"
            include_injury_search: Czy dodatkowo szukać newsów o kontuzjach

        Returns:
            Dict z articles, count, sources, quality metrics
        """
        queries = self._generate_queries(home, away, sport, include_injury_search)

        # Fetch from all sources in parallel
        all_articles: List[NewsArticle] = []
        successful_sources = 0

        for query in queries:
            tasks = [source(query) for source in self.sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"News source error: {result}")
                    continue

                successful_sources += 1
                all_articles.extend(result)

        # Deduplikacja
        unique_articles = self._deduplicate(all_articles)

        # Sortowanie po relevance i freshness
        sorted_articles = self._sort_by_relevance(unique_articles, home, away)

        return {
            "articles": [a.to_dict() for a in sorted_articles],
            "count": len(sorted_articles),
            "sources_tried": len(self.sources) * len(queries),
            "sources_successful": successful_sources,
            "queries_used": queries,
            "fetched_at": datetime.now().isoformat()
        }

    def _generate_queries(
        self,
        home: str,
        away: str,
        sport: str,
        include_injury: bool
    ) -> List[str]:
        """Generuje zapytania do wyszukiwania"""
        queries = [
            f"{home} vs {away} {sport}",
            f"{home} {away} prediction",
            f"{home} form {sport} 2026",
            f"{away} form {sport} 2026"
        ]

        if include_injury:
            queries.extend([
                f"{home} injury news",
                f"{away} injury news"
            ])

        return queries

    async def _fetch_brave(self, query: str) -> List[NewsArticle]:
        """Brave Search API"""
        await self.rate_limiter.acquire("brave_search")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.config["brave_search"]["api_key"]
        }

        params = {
            "q": query,
            "count": 10,
            "freshness": "pd",  # Past day
            "search_lang": "en"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config["brave_search"]["endpoint"],
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()

                    articles = []
                    for result in data.get("web", {}).get("results", []):
                        articles.append(NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            snippet=result.get("description", ""),
                            source=self._extract_domain(result.get("url", "")),
                            api_source="brave",
                            published=result.get("age", ""),
                            timestamp=datetime.now().isoformat()
                        ))

                    return articles
        except Exception as e:
            print(f"Brave Search error: {e}")
            return []

    async def _fetch_serper(self, query: str) -> List[NewsArticle]:
        """Serper (Google Search) API"""
        await self.rate_limiter.acquire("serper")

        headers = {
            "X-API-KEY": self.config["serper"]["api_key"],
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            "num": 10,
            "tbs": "qdr:d"  # Past day
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config["serper"]["endpoint"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()

                    articles = []
                    for result in data.get("organic", []):
                        articles.append(NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("link", ""),
                            snippet=result.get("snippet", ""),
                            source=self._extract_domain(result.get("link", "")),
                            api_source="serper",
                            published=result.get("date", ""),
                            timestamp=datetime.now().isoformat()
                        ))

                    return articles
        except Exception as e:
            print(f"Serper error: {e}")
            return []

    async def _fetch_newsapi(self, query: str) -> List[NewsArticle]:
        """NewsAPI.org"""
        await self.rate_limiter.acquire("newsapi")

        params = {
            "q": query,
            "apiKey": self.config["newsapi"]["api_key"],
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config["newsapi"]["endpoint"],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()

                    articles = []
                    for article in data.get("articles", []):
                        articles.append(NewsArticle(
                            title=article.get("title", ""),
                            url=article.get("url", ""),
                            snippet=article.get("description", ""),
                            source=article.get("source", {}).get("name", ""),
                            api_source="newsapi",
                            published=article.get("publishedAt", ""),
                            timestamp=datetime.now().isoformat()
                        ))

                    return articles
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Usuwa duplikaty na podstawie URL hash"""
        seen_hashes = set()
        unique = []

        for article in articles:
            if article.hash not in seen_hashes:
                seen_hashes.add(article.hash)
                unique.append(article)

        return unique

    def _sort_by_relevance(
        self,
        articles: List[NewsArticle],
        home: str,
        away: str
    ) -> List[NewsArticle]:
        """Sortuje artykuły po relevance score"""
        for article in articles:
            score = 0.0
            title_lower = article.title.lower()

            # Czy zawiera nazwiska/drużyny
            if home.lower() in title_lower:
                score += 0.3
            if away.lower() in title_lower:
                score += 0.3

            # Czy to wiarygodne źródło
            from config.thresholds import RELIABLE_NEWS_SOURCES
            if article.source in RELIABLE_NEWS_SOURCES["tier1"]:
                score += 0.3
            elif article.source in RELIABLE_NEWS_SOURCES["tier2"]:
                score += 0.2
            elif article.source in RELIABLE_NEWS_SOURCES["tier3"]:
                score += 0.1

            # Kluczowe słowa
            keywords = ["injury", "lineup", "prediction", "preview", "analysis"]
            for kw in keywords:
                if kw in title_lower:
                    score += 0.1

            article.relevance_score = min(score, 1.0)

        return sorted(articles, key=lambda x: x.relevance_score, reverse=True)

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Wyciąga domenę z URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "unknown"


# === HELPER: Search for specific injury news ===
async def search_injury_news(team: str, sport: str) -> List[Dict]:
    """Pomocnicza funkcja do szukania newsów o kontuzjach"""
    aggregator = NewsAggregator()
    result = await aggregator.get_match_news(team, "injury", sport, include_injury_search=True)

    # Filtruj tylko te z "injury" w tytule
    injury_articles = [
        a for a in result["articles"]
        if "injury" in a["title"].lower() or "injured" in a["title"].lower()
    ]

    return injury_articles
```

### 4.2 `data/news/validator.py` - Walidacja jakości newsów

```python
# data/news/validator.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from config.thresholds import thresholds, RELIABLE_NEWS_SOURCES

@dataclass
class NewsQualityReport:
    """Raport jakości newsów dla meczu"""
    quality_score: float          # 0-1 overall
    freshness_score: float        # 0-1
    reliability_score: float      # 0-1
    diversity_score: float        # 0-1 (różnorodność źródeł)
    article_count: int
    reliable_sources_count: int
    avg_age_hours: Optional[float]
    issues: List[str]
    is_sufficient: bool


class NewsValidator:
    """
    Ocenia jakość i wiarygodność zebranych newsów.
    """

    def __init__(self):
        self.thresholds = thresholds

    def validate_news_quality(self, news_data: Dict) -> NewsQualityReport:
        """
        Kompleksowa walidacja jakości newsów.

        Args:
            news_data: Output z NewsAggregator.get_match_news()

        Returns:
            NewsQualityReport z pełną oceną
        """
        articles = news_data.get("articles", [])
        issues = []

        # 1. Liczba artykułów
        article_count = len(articles)
        if article_count < self.thresholds.minimum_news_articles:
            issues.append(f"Too few articles: {article_count}/{self.thresholds.minimum_news_articles}")

        # 2. Freshness score
        freshness_score, avg_age = self._calculate_freshness(articles)
        if freshness_score < 0.5:
            issues.append(f"Stale news: avg age {avg_age:.1f}h")

        # 3. Reliability score
        reliability_score, reliable_count = self._calculate_reliability(articles)
        if reliability_score < 0.3:
            issues.append("No reliable news sources found")

        # 4. Diversity score (różnorodność źródeł)
        diversity_score = self._calculate_diversity(articles)
        if diversity_score < 0.3:
            issues.append("News from too few unique sources")

        # 5. Overall quality score (weighted average)
        weights = {
            "articles": 0.25,
            "freshness": 0.30,
            "reliability": 0.25,
            "diversity": 0.20
        }

        # Normalize article count to 0-1
        article_score = min(article_count / self.thresholds.minimum_news_articles, 1.0)

        quality_score = (
            article_score * weights["articles"] +
            freshness_score * weights["freshness"] +
            reliability_score * weights["reliability"] +
            diversity_score * weights["diversity"]
        )

        return NewsQualityReport(
            quality_score=round(quality_score, 3),
            freshness_score=round(freshness_score, 3),
            reliability_score=round(reliability_score, 3),
            diversity_score=round(diversity_score, 3),
            article_count=article_count,
            reliable_sources_count=reliable_count,
            avg_age_hours=avg_age,
            issues=issues,
            is_sufficient=quality_score >= 0.5 and article_count >= 1
        )

    def _calculate_freshness(self, articles: List[Dict]) -> tuple[float, Optional[float]]:
        """
        Oblicza score świeżości newsów.

        Returns:
            (freshness_score 0-1, average_age_hours)
        """
        if not articles:
            return 0.0, None

        now = datetime.now()
        ages_hours = []

        for article in articles:
            timestamp = article.get("timestamp") or article.get("published")
            if not timestamp:
                continue

            try:
                # Parsuj różne formaty
                if isinstance(timestamp, str):
                    if "T" in timestamp:
                        article_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00").replace("+00:00", ""))
                    else:
                        # Próbuj różne formaty
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d %b %Y"]:
                            try:
                                article_time = datetime.strptime(timestamp, fmt)
                                break
                            except:
                                continue
                        else:
                            continue
                else:
                    continue

                age_hours = (now - article_time).total_seconds() / 3600
                if age_hours >= 0:  # Ignore future dates
                    ages_hours.append(age_hours)
            except Exception as e:
                continue

        if not ages_hours:
            return 0.5, None  # Neutral score if can't determine

        avg_age = sum(ages_hours) / len(ages_hours)

        # Score based on average age
        max_hours = self.thresholds.news_freshness_hours

        if avg_age < 6:
            score = 1.0
        elif avg_age < 12:
            score = 0.8
        elif avg_age < max_hours:
            score = 1.0 - (avg_age / max_hours) * 0.5
        else:
            score = max(0.1, 0.5 - (avg_age - max_hours) / (max_hours * 2))

        return score, avg_age

    def _calculate_reliability(self, articles: List[Dict]) -> tuple[float, int]:
        """
        Oblicza score wiarygodności źródeł.

        Returns:
            (reliability_score 0-1, reliable_sources_count)
        """
        if not articles:
            return 0.0, 0

        reliable_count = 0
        total_score = 0.0

        for article in articles:
            source = article.get("source", "")

            if source in RELIABLE_NEWS_SOURCES["tier1"]:
                total_score += 1.0
                reliable_count += 1
            elif source in RELIABLE_NEWS_SOURCES["tier2"]:
                total_score += 0.7
                reliable_count += 1
            elif source in RELIABLE_NEWS_SOURCES["tier3"]:
                total_score += 0.4
            else:
                total_score += 0.2  # Unknown source

        avg_score = total_score / len(articles)
        return avg_score, reliable_count

    def _calculate_diversity(self, articles: List[Dict]) -> float:
        """
        Oblicza score różnorodności źródeł.
        Im więcej unikalnych źródeł, tym lepiej.
        """
        if not articles:
            return 0.0

        # Unikalne domeny
        unique_sources = set()
        unique_api_sources = set()

        for article in articles:
            source = article.get("source", "")
            api_source = article.get("api_source", "")

            if source:
                unique_sources.add(source)
            if api_source:
                unique_api_sources.add(api_source)

        # Score: max przy 5+ unikalnych źródłach
        source_score = min(len(unique_sources) / 5, 1.0)
        api_score = min(len(unique_api_sources) / 3, 1.0)  # Max 3 API

        return (source_score * 0.7 + api_score * 0.3)

    def detect_suspicious_patterns(self, news_data: Dict) -> List[str]:
        """
        Wykrywa podejrzane wzorce w newsach.

        Returns:
            Lista ostrzeżeń
        """
        warnings = []
        articles = news_data.get("articles", [])

        if not articles:
            return ["No news articles found - high uncertainty"]

        # 1. Wszystkie newsy z jednego źródła
        sources = [a.get("source", "") for a in articles]
        if len(set(sources)) == 1 and len(sources) > 2:
            warnings.append(f"All news from single source: {sources[0]}")

        # 2. Bardzo stare newsy
        for article in articles:
            published = article.get("published", "")
            if "week" in published.lower() or "month" in published.lower():
                warnings.append(f"Very old article detected: {article.get('title', '')[:50]}")

        # 3. Sprzeczne informacje (prostą heurystyka)
        titles = [a.get("title", "").lower() for a in articles]
        contradictions = [
            ("will play", "will not play"),
            ("fit", "injured"),
            ("confirmed", "doubt"),
            ("winning", "losing")
        ]

        for pos, neg in contradictions:
            has_pos = any(pos in t for t in titles)
            has_neg = any(neg in t for t in titles)
            if has_pos and has_neg:
                warnings.append(f"Contradictory news detected: '{pos}' vs '{neg}'")

        return warnings
```

### 4.3 `data/news/injury_extractor.py` - Ekstrakcja kontuzji przez LLM

```python
# data/news/injury_extractor.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from langchain_anthropic import ChatAnthropic
from config.settings import settings

@dataclass
class InjuryInfo:
    """Informacja o kontuzji"""
    player: str
    team: str
    injury_type: str
    status: str  # "out", "doubtful", "questionable", "probable"
    return_date: Optional[str]
    source: str
    confidence: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "team": self.team,
            "injury_type": self.injury_type,
            "status": self.status,
            "return_date": self.return_date,
            "source": self.source,
            "confidence": self.confidence
        }


class InjuryExtractor:
    """
    Wykorzystuje LLM do ekstrakcji informacji o kontuzjach z newsów.
    """

    EXTRACTION_PROMPT = """Analyze the following sports news article and extract injury information.

ARTICLE:
Title: {title}
Content: {snippet}
Source: {source}

TASK: Extract any injury information mentioned. If there's injury news, return JSON:
{{
    "has_injury": true,
    "player": "Full player name",
    "team": "Team/Country name",
    "injury_type": "Type of injury (e.g., ankle sprain, muscle strain)",
    "status": "out" | "doubtful" | "questionable" | "probable",
    "return_date": "Expected return date or null",
    "confidence": 0.0-1.0 (how confident based on source reliability)
}}

If NO injury information is found, return:
{{
    "has_injury": false
}}

IMPORTANT:
- Only extract CONFIRMED injury news, not speculation
- "out" = definitely not playing
- "doubtful" = unlikely to play (>75% chance missing)
- "questionable" = uncertain (50/50)
- "probable" = likely to play but not 100%

Return ONLY valid JSON, no explanation."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1  # Low temperature for factual extraction
        )

    async def extract_injuries(self, articles: List[Dict]) -> List[InjuryInfo]:
        """
        Ekstrahuje informacje o kontuzjach z listy artykułów.

        Args:
            articles: Lista artykułów z NewsAggregator

        Returns:
            Lista InjuryInfo
        """
        injuries = []

        for article in articles:
            # Skip if obviously not about injuries
            title = article.get("title", "").lower()
            snippet = article.get("snippet", "").lower()

            injury_keywords = ["injury", "injured", "out", "miss", "sidelined",
                            "withdraw", "pulled out", "ruled out", "doubt"]

            if not any(kw in title or kw in snippet for kw in injury_keywords):
                continue

            # Extract with LLM
            try:
                injury_info = await self._extract_single(article)
                if injury_info:
                    injuries.append(injury_info)
            except Exception as e:
                print(f"Injury extraction error: {e}")
                continue

        # Deduplicate by player name
        unique_injuries = self._deduplicate_injuries(injuries)

        return unique_injuries

    async def _extract_single(self, article: Dict) -> Optional[InjuryInfo]:
        """Ekstrahuje z pojedynczego artykułu"""
        prompt = self.EXTRACTION_PROMPT.format(
            title=article.get("title", ""),
            snippet=article.get("snippet", "")[:500],  # Limit length
            source=article.get("source", "unknown")
        )

        response = await self.llm.ainvoke(prompt)

        try:
            # Parse JSON response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            if not data.get("has_injury"):
                return None

            return InjuryInfo(
                player=data.get("player", "Unknown"),
                team=data.get("team", "Unknown"),
                injury_type=data.get("injury_type", "Unknown"),
                status=data.get("status", "questionable"),
                return_date=data.get("return_date"),
                source=article.get("source", "unknown"),
                confidence=data.get("confidence", 0.5)
            )
        except json.JSONDecodeError:
            return None

    def _deduplicate_injuries(self, injuries: List[InjuryInfo]) -> List[InjuryInfo]:
        """
        Usuwa duplikaty, zachowując najnowszą/najbardziej wiarygodną informację.
        """
        by_player = {}

        for injury in injuries:
            player_key = injury.player.lower().strip()

            if player_key not in by_player:
                by_player[player_key] = injury
            else:
                # Zachowaj tę z wyższą confidence
                if injury.confidence > by_player[player_key].confidence:
                    by_player[player_key] = injury

        return list(by_player.values())


# === Convenience function ===
async def extract_injuries_from_news(articles: List[Dict]) -> List[Dict]:
    """
    Pomocnicza funkcja do ekstrakcji kontuzji.

    Returns:
        Lista słowników z informacjami o kontuzjach
    """
    extractor = InjuryExtractor()
    injuries = await extractor.extract_injuries(articles)
    return [i.to_dict() for i in injuries]
```

---
