# 🎯 NEXUS AI v2.0 - Kompletna Implementacja Systemu Predykcji Sportowej

## Zintegrowana Specyfikacja z Pełnym Kodem Produkcyjnym

**Wersja:** 2.0  
**Data:** Styczeń 2026  
**Stack:** LangGraph + MCP + Gradio + Python 3.11+

---

## 📋 SPIS TREŚCI

1. [Architektura Systemu](#1-architektura-systemu)
2. [Struktura Projektu](#2-struktura-projektu)
3. [Konfiguracja i Środowisko](#3-konfiguracja-i-środowisko)
4. [Warstwa Danych - News Aggregator](#4-warstwa-danych---news-aggregator)
5. [Agent Ewaluujący Jakość Danych](#5-agent-ewaluujący-jakość-danych)
6. [MCP Servers - Pełna Implementacja](#6-mcp-servers---pełna-implementacja)
7. [Modele Predykcji](#7-modele-predykcji)
8. [System Rankingu Top 3](#8-system-rankingu-top-3)
9. [Interfejs Gradio](#9-interfejs-gradio)
10. [Główna Pętla Systemu](#10-główna-pętla-systemu)
11. [Deployment i Monitoring](#11-deployment-i-monitoring)

---

## 1. ARCHITEKTURA SYSTEMU

### 1.1 Diagram Przepływu Danych

```mermaid
flowchart TD
    A[Start: Sprawdź mecze na dziś] --> B[Pobierz newsy Brave/Serper/NewsAPI]
    B --> C[News Aggregator - deduplikacja]
    C --> D[News Validator - quality score]
    D --> E{Data Evaluator Agent}
    E -->|Quality > 60%| F[Pobierz kursy z API + Scraping]
    E -->|Quality 40-60%| G[Oznacz jako HIGH RISK]
    E -->|Quality < 40%| H[Odrzuć mecz - INSUFFICIENT_DATA]
    F --> I[Analyst Agent - oblicz prawdopodobieństwa]
    G --> I
    I --> J[Value Calculator - znajdź edge]
    J --> K[Match Ranker - sortuj po composite score]
    K --> L[Wybierz Top 3 mecze]
    L --> M[Decision Agent - final verification]
    M --> N[Execute bets z Kelly Criterion]
    N --> O[Update bankroll + Send alerts]
    H --> P[Log to database: skipped_matches]
    
    subgraph "MCP Servers Layer"
        B
        F
    end
    
    subgraph "Quality Control"
        D
        E
    end
    
    subgraph "Analysis Engine"
        I
        J
        K
    end
```

### 1.2 Architektura Komponentów

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GRADIO INTERFACE                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Dashboard    │ │ Top 3 Tab   │ │ Live News   │ │ History     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH ORCHESTRATOR                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Supervisor  │◀─│ NewsAnalyst │──│ DataEvaluator│──│ Analyst    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                                                   │                    │
│         ▼                                                   ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ MatchRanker │──│ RiskManager │──│DecisionMaker│──│ BettorAgent │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVERS LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ news_server │  │evaluation_  │  │ odds_server │  │ tennis_     │             │
│  │ (Brave+     │  │server       │  │ (TheOddsAPI │  │ server      │             │
│  │  Serper)    │  │             │  │  + Scraping)│  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                              │
│  │ basketball_ │  │ alerts_     │  │ pl_bookies_ │                              │
│  │ server      │  │ server      │  │ scraper     │                              │
│  └─────────────┘  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
│  [Brave Search] [Serper] [NewsAPI] [The Odds API] [api-tennis] [BetsAPI]        │
│  [Sofascore Scraping] [Fortuna Scraping] [STS Scraping] [Betclic Scraping]      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. STRUKTURA PROJEKTU

```
nexus-ai/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── app.py                              # Gradio entry point
├── betting_floor.py                    # Główna pętla systemu
│
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Wszystkie ustawienia + API keys
│   ├── thresholds.py                   # Progi jakości danych
│   └── leagues.py                      # Konfiguracja lig (popular/medium/unpopular)
│
├── data/
│   ├── __init__.py
│   ├── sports_api.py                   # Bazowy klient API
│   │
│   ├── news/
│   │   ├── __init__.py
│   │   ├── aggregator.py               # Agregacja newsów z wielu źródeł
│   │   ├── validator.py                # Walidacja wiarygodności źródeł
│   │   └── injury_extractor.py         # Ekstrakcja kontuzji przez LLM
│   │
│   ├── tennis/
│   │   ├── __init__.py
│   │   ├── api_tennis_client.py        # api-tennis.com
│   │   └── sofascore_scraper.py        # Fallback scraper
│   │
│   ├── basketball/
│   │   ├── __init__.py
│   │   ├── bets_api_client.py          # BetsAPI
│   │   └── euroleague_scraper.py       # Scraper dla EuroLeague
│   │
│   ├── odds/
│   │   ├── __init__.py
│   │   ├── odds_api_client.py          # The Odds API
│   │   ├── pl_scraper.py               # Fortuna/STS/Betclic scraper
│   │   └── odds_merger.py              # Łączenie kursów z wielu źródeł
│   │
│   └── quality/
│       ├── __init__.py
│       └── metrics.py                  # DataQualityMetrics class
│
├── database/
│   ├── __init__.py
│   ├── models.py                       # SQLAlchemy models
│   ├── crud.py                         # CRUD operations
│   └── db.py                           # Database connection + init
│
├── mcp_servers/
│   ├── __init__.py
│   ├── news_server.py                  # Brave + Serper + NewsAPI
│   ├── evaluation_server.py            # Agent ewaluujący
│   ├── odds_server.py                  # Kursy (API + scraping)
│   ├── tennis_server.py                # Dane tenisowe
│   ├── basketball_server.py            # Dane koszykarskie
│   └── alerts_server.py                # Powiadomienia
│
├── agents/
│   ├── __init__.py
│   ├── supervisor.py                   # Koordynator LangGraph
│   ├── news_analyst.py                 # Analiza newsów
│   ├── data_evaluator.py               # Ewaluacja jakości danych
│   ├── analyst.py                      # Analiza meczów
│   ├── ranker.py                       # Ranking Top 3
│   ├── risk_manager.py                 # Ocena ryzyka
│   ├── decision_maker.py               # Decyzje końcowe
│   └── bettor.py                       # Wykonywanie zakładów
│
├── core/
│   ├── __init__.py
│   ├── state.py                        # Pydantic state models
│   ├── quality_scorer.py               # Quality score 0-1
│   ├── value_calculator.py             # Value + Kelly Criterion
│   ├── ranker_engine.py                # Logika rankingu
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tennis_model.py             # Model predykcji tenisa
│   │   └── basketball_model.py         # Model predykcji koszykówki
│   │
│   └── utils/
│       ├── __init__.py
│       ├── rate_limiter.py             # Rate limiting dla API
│       └── cache.py                    # Redis cache wrapper
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py                    # Główny dashboard
│   ├── top3_tab.py                     # Zakładka Top 3
│   ├── news_feed.py                    # Live news feed
│   ├── history_tab.py                  # Historia zakładów
│   └── components.py                   # Reusable Gradio components
│
├── tests/
│   ├── __init__.py
│   ├── test_news_aggregator.py
│   ├── test_data_evaluator.py
│   ├── test_value_calculator.py
│   └── test_ranker.py
│
├── scripts/
│   ├── setup_mcp.py
│   ├── init_db.py
│   ├── backtest.py
│   └── run_daily.py
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── .env.docker
```

---

## 3. KONFIGURACJA I ŚRODOWISKO

### 3.1 `config/settings.py` - Centralna konfiguracja

```python
# config/settings.py
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Centralna konfiguracja aplikacji NEXUS AI"""
    
    # === PODSTAWOWE ===
    APP_NAME: str = "NEXUS AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # === BAZA DANYCH ===
    DATABASE_URL: str = Field(
        default="sqlite:///./nexus.db",
        env="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # === API KEYS - NEWS ===
    BRAVE_API_KEY: Optional[str] = Field(default=None, env="BRAVE_API_KEY")
    SERPER_API_KEY: Optional[str] = Field(default=None, env="SERPER_API_KEY")
    NEWSAPI_KEY: Optional[str] = Field(default=None, env="NEWSAPI_KEY")
    
    # === API KEYS - SPORTS DATA ===
    ODDS_API_KEY: Optional[str] = Field(default=None, env="ODDS_API_KEY")
    API_TENNIS_KEY: Optional[str] = Field(default=None, env="API_TENNIS_KEY")
    BETS_API_KEY: Optional[str] = Field(default=None, env="BETS_API_KEY")
    
    # === API KEYS - LLM ===
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # === SYSTEM ===
    RUN_EVERY_N_MINUTES: int = Field(default=30, env="RUN_EVERY_N_MINUTES")
    ENABLE_LIVE_ODDS: bool = Field(default=False, env="ENABLE_LIVE_ODDS")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()


# === NEWS API CONFIG ===
NEWS_CONFIG = {
    "brave_search": {
        "enabled": bool(settings.BRAVE_API_KEY),
        "api_key": settings.BRAVE_API_KEY,
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "rate_limit": 10,  # requests per minute
        "priority": 1  # Highest priority
    },
    "serper": {
        "enabled": bool(settings.SERPER_API_KEY),
        "api_key": settings.SERPER_API_KEY,
        "endpoint": "https://google.serper.dev/search",
        "rate_limit": 100,
        "priority": 2
    },
    "newsapi": {
        "enabled": bool(settings.NEWSAPI_KEY),
        "api_key": settings.NEWSAPI_KEY,
        "endpoint": "https://newsapi.org/v2/everything",
        "rate_limit": 50,
        "priority": 3
    }
}


# === SPORTS API CONFIG ===
SPORTS_API_CONFIG = {
    "the_odds_api": {
        "enabled": bool(settings.ODDS_API_KEY),
        "api_key": settings.ODDS_API_KEY,
        "base_url": "https://api.the-odds-api.com/v4",
        "rate_limit": 10  # per second
    },
    "api_tennis": {
        "enabled": bool(settings.API_TENNIS_KEY),
        "api_key": settings.API_TENNIS_KEY,
        "base_url": "https://api.api-tennis.com/tennis",
        "rate_limit": 100  # per minute
    },
    "bets_api": {
        "enabled": bool(settings.BETS_API_KEY),
        "api_key": settings.BETS_API_KEY,
        "base_url": "https://api.betsapi.com/v1",
        "rate_limit": 50
    }
}


# === POLISH BOOKMAKERS CONFIG (Scraping) ===
PL_BOOKMAKERS_CONFIG = {
    "fortuna": {
        "enabled": True,
        "base_url": "https://www.efortuna.pl",
        "tennis_path": "/zaklady-bukmacherskie/tenis",
        "basketball_path": "/zaklady-bukmacherskie/koszykowka",
        "selectors": {
            "match_row": ".event-row",
            "teams": ".event-name",
            "odds": ".odds-value",
            "time": ".event-time"
        },
        "rate_limit": 2  # requests per minute (be gentle)
    },
    "sts": {
        "enabled": True,
        "base_url": "https://www.sts.pl/pl",
        "tennis_path": "/zaklady-bukmacherskie/tenis",
        "basketball_path": "/zaklady-bukmacherskie/koszykowka",
        "selectors": {
            "match_row": ".match-row",
            "teams": ".match-name",
            "odds": ".odds-button__odds",
            "time": ".match-time"
        },
        "rate_limit": 2
    },
    "betclic": {
        "enabled": True,
        "base_url": "https://www.betclic.pl",
        "tennis_path": "/tenis-s2",
        "basketball_path": "/koszykowka-s4",
        "selectors": {
            "match_row": ".match",
            "teams": ".match-entry",
            "odds": ".oddValue",
            "time": ".match-time"
        },
        "rate_limit": 2
    }
}
```

### 3.2 `config/thresholds.py` - Progi jakości danych

```python
# config/thresholds.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class DataQualityThresholds:
    """Progi dla oceny jakości danych"""
    
    # === NEWS QUALITY ===
    minimum_news_articles: int = 3          # Min artykułów o meczu
    news_freshness_hours: int = 24          # Max wiek newsa w godzinach
    reliable_sources_bonus: float = 0.2     # Bonus za wiarygodne źródła
    
    # === ODDS QUALITY ===
    odds_sources_required: int = 2          # Min liczba bukmacherów
    max_odds_variance: float = 0.05         # Max rozrzut kursów (5%)
    
    # === STATS QUALITY ===
    stats_completeness: float = 0.7         # Min % dostępnych statystyk
    min_historical_matches: int = 5         # Min meczów w historii
    
    # === OVERALL THRESHOLDS ===
    quality_excellent: float = 0.85         # Score > 85% = EXCELLENT
    quality_good: float = 0.70              # Score > 70% = GOOD
    quality_moderate: float = 0.50          # Score > 50% = MODERATE
    quality_high_risk: float = 0.40         # Score > 40% = HIGH RISK
    quality_reject: float = 0.30            # Score < 30% = REJECT
    
    # === VALUE THRESHOLDS ===
    min_edge_popular_league: float = 0.03   # 3% dla popularnych lig
    min_edge_medium_league: float = 0.04    # 4% dla średnich lig
    min_edge_unpopular_league: float = 0.05 # 5% dla niepopularnych lig


@dataclass
class LeagueQualityRequirements:
    """Wymagania jakościowe per typ ligi"""
    
    min_bookmakers: int
    min_matches_history: int
    min_stats_fields: int
    min_news_articles: int


LEAGUE_REQUIREMENTS: Dict[str, LeagueQualityRequirements] = {
    "popular": LeagueQualityRequirements(
        min_bookmakers=2,
        min_matches_history=3,
        min_stats_fields=5,
        min_news_articles=2
    ),
    "medium": LeagueQualityRequirements(
        min_bookmakers=2,
        min_matches_history=5,
        min_stats_fields=8,
        min_news_articles=3
    ),
    "unpopular": LeagueQualityRequirements(
        min_bookmakers=3,      # Więcej źródeł wymaganych!
        min_matches_history=8,
        min_stats_fields=10,
        min_news_articles=1    # Mniej newsów dostępnych
    )
}


# Wiarygodne źródła newsowe
RELIABLE_NEWS_SOURCES = {
    # Tier 1 - Najwyższa wiarygodność
    "tier1": {
        "BBC Sport", "ESPN", "Sky Sports", "Eurosport",
        "ATP Tour", "WTA Tour", "NBA.com", "EuroLeague.net",
        "Reuters Sports", "Associated Press"
    },
    # Tier 2 - Wysoka wiarygodność
    "tier2": {
        "The Guardian Sport", "Sports Illustrated",
        "Bleacher Report", "Tennis.com", "Tennis World USA",
        "BasketNews.com", "Eurohoops"
    },
    # Tier 3 - Średnia wiarygodność
    "tier3": {
        "SportoweFakty", "Przegląd Sportowy", "WP Sportowe Fakty",
        "Onet Sport", "Interia Sport"
    }
}


# Eksport
thresholds = DataQualityThresholds()
```

### 3.3 `config/leagues.py` - Klasyfikacja lig

```python
# config/leagues.py
from typing import Dict, List, Literal

LeagueType = Literal["popular", "medium", "unpopular"]

# === TENNIS LEAGUES ===
TENNIS_LEAGUES: Dict[LeagueType, List[str]] = {
    "popular": [
        # Grand Slams
        "australian_open", "french_open", "wimbledon", "us_open",
        # ATP Finals
        "atp_finals", "wta_finals",
        # Masters 1000 / WTA 1000
        "indian_wells", "miami", "monte_carlo", "madrid", "rome",
        "canadian_open", "cincinnati", "shanghai", "paris",
        "wta_indian_wells", "wta_miami", "wta_madrid", "wta_rome",
        "wta_canadian", "wta_cincinnati", "wta_beijing"
    ],
    "medium": [
        # ATP 500
        "rotterdam", "dubai", "acapulco", "barcelona", "queen_s",
        "halle", "hamburg", "washington", "tokyo", "basel", "vienna",
        # ATP 250
        "atp_250",
        # WTA 500
        "wta_500",
        # Challenger 125+
        "challenger_125"
    ],
    "unpopular": [
        # ITF
        "itf_men", "itf_women", "itf_m25", "itf_w25", "itf_m15", "itf_w15",
        # Lower Challengers
        "challenger_75", "challenger_50",
        # Futures
        "futures"
    ]
}

# === BASKETBALL LEAGUES ===
BASKETBALL_LEAGUES: Dict[LeagueType, List[str]] = {
    "popular": [
        # NBA
        "nba", "nba_playoffs",
        # European Top
        "euroleague", "eurocup",
        # College
        "ncaa_men", "ncaa_tournament"
    ],
    "medium": [
        # European national leagues - Top
        "liga_acb",          # Spain
        "lega_basket",       # Italy
        "betclic_elite",     # France (LNB)
        "bbl",               # Germany
        "vtb_league",        # Russia/Europe
        # Other
        "wnba",
        "nbl_australia"
    ],
    "unpopular": [
        # Polish leagues
        "plk",               # Polska Liga Koszykówki
        "1_liga_poland",     # 1. Liga (Poland)
        # Other European
        "greek_basket",
        "turkish_bsl",
        "adriatic_league",
        "liga_portugal",
        # Lower divisions
        "g_league",          # NBA G-League
        "college_other"
    ]
}


def classify_league(league_name: str, sport: str) -> LeagueType:
    """
    Klasyfikuje ligę jako popular/medium/unpopular.
    
    Args:
        league_name: Nazwa ligi (np. "ATP Masters 1000 Madrid")
        sport: "tennis" lub "basketball"
    
    Returns:
        LeagueType: "popular", "medium", lub "unpopular"
    """
    league_lower = league_name.lower()
    
    leagues = TENNIS_LEAGUES if sport == "tennis" else BASKETBALL_LEAGUES
    
    # Sprawdź popular
    for league in leagues["popular"]:
        if league in league_lower or league_lower in league:
            return "popular"
    
    # Sprawdź medium
    for league in leagues["medium"]:
        if league in league_lower or league_lower in league:
            return "medium"
    
    # Default: unpopular
    return "unpopular"


def get_league_requirements(league_type: LeagueType) -> dict:
    """Zwraca wymagania dla danego typu ligi"""
    from config.thresholds import LEAGUE_REQUIREMENTS
    return LEAGUE_REQUIREMENTS[league_type]
```

---

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

## 5. AGENT EWALUUJĄCY JAKOŚĆ DANYCH

### 5.1 `agents/data_evaluator.py` - Główny agent ewaluujący

```python
# agents/data_evaluator.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from data.news.aggregator import NewsAggregator
from data.news.validator import NewsValidator, NewsQualityReport
from data.news.injury_extractor import extract_injuries_from_news
from config.thresholds import thresholds, LEAGUE_REQUIREMENTS
from config.leagues import classify_league

@dataclass
class DataQualityReport:
    """Pełny raport jakości danych dla meczu"""
    
    # Identyfikacja
    match_id: str
    match_name: str
    sport: str
    league: str
    league_type: str  # popular/medium/unpopular
    
    # Scores (0-100)
    overall_score: float
    news_quality_score: float
    stats_completeness_score: float
    odds_quality_score: float
    
    # Sub-reports
    news_report: Optional[NewsQualityReport] = None
    
    # Metadata
    odds_sources_count: int = 0
    stats_fields_available: int = 0
    injuries_found: List[Dict] = field(default_factory=list)
    
    # Issues & Warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Final verdict
    is_ready: bool = False
    recommendation: str = "SKIP"  # PROCEED / CAUTION / SKIP
    
    @property
    def overall_score_text(self) -> str:
        return f"{self.overall_score:.1f}/100"
    
    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "league": self.league,
            "league_type": self.league_type,
            "overall_score": self.overall_score,
            "overall_score_text": self.overall_score_text,
            "news_quality_score": self.news_quality_score,
            "stats_completeness_score": self.stats_completeness_score,
            "odds_quality_score": self.odds_quality_score,
            "odds_sources_count": self.odds_sources_count,
            "stats_fields_available": self.stats_fields_available,
            "injuries_found": self.injuries_found,
            "issues": self.issues,
            "warnings": self.warnings,
            "is_ready": self.is_ready,
            "recommendation": self.recommendation
        }


class DataEvaluator:
    """
    Agent ewaluujący jakość wszystkich danych przed analizą meczu.
    
    KLUCZOWY KOMPONENT SYSTEMU - chroni przed złymi predykcjami
    wynikającymi z niewystarczających lub nierzetelnych danych.
    """
    
    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.news_validator = NewsValidator()
        self.thresholds = thresholds
    
    async def evaluate_match(
        self,
        match_id: str,
        home: str,
        away: str,
        sport: str,
        league: str,
        stats: Optional[Dict] = None,
        odds: Optional[Dict] = None
    ) -> DataQualityReport:
        """
        Przeprowadza pełną ewaluację jakości danych dla meczu.
        
        Args:
            match_id: Unikalny identyfikator meczu
            home: Nazwa zawodnika/drużyny gospodarza
            away: Nazwa zawodnika/drużyny gościa
            sport: "tennis" lub "basketball"
            league: Nazwa ligi
            stats: Opcjonalne - już pobrane statystyki
            odds: Opcjonalne - już pobrane kursy
        
        Returns:
            DataQualityReport z pełną oceną
        """
        match_name = f"{home} vs {away}"
        league_type = classify_league(league, sport)
        requirements = LEAGUE_REQUIREMENTS[league_type]
        
        issues = []
        warnings = []
        
        # === 1. EWALUACJA NEWSÓW ===
        print(f"[DataEvaluator] Fetching news for {match_name}...")
        news_data = await self.news_aggregator.get_match_news(home, away, sport)
        news_report = self.news_validator.validate_news_quality(news_data)
        
        # Sprawdź podejrzane wzorce
        suspicious = self.news_validator.detect_suspicious_patterns(news_data)
        warnings.extend(suspicious)
        
        # Ekstrakcja kontuzji
        injuries = []
        if news_data["articles"]:
            injuries = await extract_injuries_from_news(news_data["articles"])
            if injuries:
                for inj in injuries:
                    if inj["status"] == "out":
                        warnings.append(f"⚠️ {inj['player']} is OUT: {inj['injury_type']}")
        
        # News quality score (0-100)
        news_quality_score = news_report.quality_score * 100
        
        if not news_report.is_sufficient:
            issues.append("Insufficient news coverage")
        
        # === 2. EWALUACJA STATYSTYK ===
        stats_completeness_score = self._evaluate_stats(stats, sport, requirements)
        
        if stats_completeness_score < 50:
            issues.append(f"Stats completeness only {stats_completeness_score:.0f}%")
        
        # === 3. EWALUACJA KURSÓW ===
        odds_quality_score, odds_count = self._evaluate_odds(odds, requirements)
        
        if odds_count < requirements.min_bookmakers:
            issues.append(f"Only {odds_count}/{requirements.min_bookmakers} bookmakers")
        
        # === 4. OBLICZ OVERALL SCORE ===
        # Wagi zależne od typu ligi
        if league_type == "unpopular":
            # Dla niepopularnych lig bardziej liczy się ilość danych
            weights = {
                "news": 0.25,
                "stats": 0.40,  # Wyższa waga!
                "odds": 0.35
            }
        elif league_type == "medium":
            weights = {
                "news": 0.30,
                "stats": 0.35,
                "odds": 0.35
            }
        else:  # popular
            weights = {
                "news": 0.30,
                "stats": 0.30,
                "odds": 0.40
            }
        
        overall_score = (
            news_quality_score * weights["news"] +
            stats_completeness_score * weights["stats"] +
            odds_quality_score * weights["odds"]
        )
        
        # === 5. OKREŚL RECOMMENDATION ===
        if overall_score >= thresholds.quality_good * 100:
            recommendation = "PROCEED"
            is_ready = True
        elif overall_score >= thresholds.quality_moderate * 100:
            recommendation = "CAUTION"
            is_ready = True
            warnings.append("Moderate data quality - proceed with caution")
        elif overall_score >= thresholds.quality_high_risk * 100:
            recommendation = "HIGH_RISK"
            is_ready = True  # Still allow but with warnings
            warnings.append("HIGH RISK - minimal stake recommended")
        else:
            recommendation = "SKIP"
            is_ready = False
            issues.append(f"Quality score {overall_score:.1f} below threshold {thresholds.quality_reject * 100}")
        
        # Dodatkowe penalizacje dla niepopularnych lig
        if league_type == "unpopular" and overall_score < 60:
            recommendation = "SKIP"
            is_ready = False
            issues.append("Unpopular league requires higher data quality")
        
        return DataQualityReport(
            match_id=match_id,
            match_name=match_name,
            sport=sport,
            league=league,
            league_type=league_type,
            overall_score=round(overall_score, 1),
            news_quality_score=round(news_quality_score, 1),
            stats_completeness_score=round(stats_completeness_score, 1),
            odds_quality_score=round(odds_quality_score, 1),
            news_report=news_report,
            odds_sources_count=odds_count,
            stats_fields_available=len(stats) if stats else 0,
            injuries_found=injuries,
            issues=issues,
            warnings=warnings,
            is_ready=is_ready,
            recommendation=recommendation
        )
    
    def _evaluate_stats(
        self,
        stats: Optional[Dict],
        sport: str,
        requirements
    ) -> float:
        """
        Ocenia kompletność statystyk.
        
        Returns:
            Score 0-100
        """
        if not stats:
            return 0.0
        
        # Wymagane pola per sport
        if sport == "tennis":
            required_fields = [
                "home_ranking", "away_ranking",
                "home_recent_form", "away_recent_form",
                "surface", "home_surface_win_pct", "away_surface_win_pct",
                "h2h_matches", "home_last5_wins", "away_last5_wins"
            ]
        else:  # basketball
            required_fields = [
                "home_offensive_rating", "home_defensive_rating",
                "away_offensive_rating", "away_defensive_rating",
                "home_recent_games", "away_recent_games",
                "home_rest_days", "away_rest_days",
                "home_home_record", "away_away_record"
            ]
        
        # Policz dostępne pola
        available = sum(1 for field in required_fields if field in stats and stats[field] is not None)
        
        completeness = (available / len(required_fields)) * 100
        
        # Bonus za dodatkowe pola
        extra_fields = len([k for k in stats.keys() if k not in required_fields])
        bonus = min(extra_fields * 2, 20)  # Max 20% bonus
        
        return min(completeness + bonus, 100)
    
    def _evaluate_odds(
        self,
        odds: Optional[Dict],
        requirements
    ) -> tuple[float, int]:
        """
        Ocenia jakość kursów.
        
        Returns:
            (score 0-100, bookmakers_count)
        """
        if not odds:
            return 0.0, 0
        
        # Policz bukmacherów
        bookmakers_count = len(odds)
        
        if bookmakers_count == 0:
            return 0.0, 0
        
        # Bazowy score za liczbę bukmacherów
        min_required = requirements.min_bookmakers
        bookmaker_score = min(bookmakers_count / min_required, 1.0) * 50
        
        # Sprawdź spójność kursów (variance)
        consistency_score = self._calculate_odds_consistency(odds) * 50
        
        return bookmaker_score + consistency_score, bookmakers_count
    
    def _calculate_odds_consistency(self, odds: Dict) -> float:
        """
        Oblicza spójność kursów między bukmacherami.
        Niski rozrzut = wysoka spójność = lepiej.
        
        Returns:
            Score 0-1
        """
        if len(odds) < 2:
            return 0.5  # Can't compare
        
        # Zbierz wszystkie kursy na "home"
        home_odds = []
        for bookie, markets in odds.items():
            if isinstance(markets, dict):
                home_odd = markets.get("home") or markets.get("h2h_home")
                if home_odd:
                    home_odds.append(float(home_odd))
        
        if len(home_odds) < 2:
            return 0.5
        
        # Oblicz variance
        mean = sum(home_odds) / len(home_odds)
        variance = sum((x - mean) ** 2 for x in home_odds) / len(home_odds)
        std_dev = variance ** 0.5
        
        # Coefficient of variation
        cv = std_dev / mean if mean > 0 else 1
        
        # Convert to score (lower CV = higher score)
        # CV < 0.02 = excellent, CV > 0.10 = poor
        if cv < 0.02:
            return 1.0
        elif cv < 0.05:
            return 0.8
        elif cv < 0.10:
            return 0.5
        else:
            return 0.2
    
    async def batch_evaluate(
        self,
        matches: List[Dict],
        sport: str
    ) -> List[DataQualityReport]:
        """
        Ewaluuje wiele meczów równolegle.
        
        Args:
            matches: Lista słowników z danymi meczów
            sport: "tennis" lub "basketball"
        
        Returns:
            Lista DataQualityReport
        """
        tasks = []
        
        for match in matches:
            task = self.evaluate_match(
                match_id=match.get("id", "unknown"),
                home=match.get("home", ""),
                away=match.get("away", ""),
                sport=sport,
                league=match.get("league", ""),
                stats=match.get("stats"),
                odds=match.get("odds")
            )
            tasks.append(task)
        
        # Execute in parallel with semaphore to limit concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        reports = []
        for result in results:
            if isinstance(result, DataQualityReport):
                reports.append(result)
            else:
                print(f"Evaluation error: {result}")
        
        return reports


# === UTILITY FUNCTIONS ===

def print_quality_report(report: DataQualityReport) -> str:
    """
    Formatuje raport jakości do czytelnego formatu.
    """
    status_emoji = {
        "PROCEED": "✅",
        "CAUTION": "⚠️",
        "HIGH_RISK": "🔶",
        "SKIP": "❌"
    }
    
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║  DATA QUALITY REPORT: {report.match_name[:40]:<40}
╠══════════════════════════════════════════════════════════════╣
║  League: {report.league} ({report.league_type})
║  Sport:  {report.sport}
╠══════════════════════════════════════════════════════════════╣
║  OVERALL SCORE: {report.overall_score_text:>8}  {status_emoji.get(report.recommendation, '?')} {report.recommendation}
╠──────────────────────────────────────────────────────────────╣
║  📰 News Quality:      {report.news_quality_score:>6.1f}/100
║  📊 Stats Complete:    {report.stats_completeness_score:>6.1f}/100
║  💰 Odds Quality:      {report.odds_quality_score:>6.1f}/100
╠──────────────────────────────────────────────────────────────╣
║  Bookmakers: {report.odds_sources_count}  |  Stats fields: {report.stats_fields_available}
"""
    
    if report.injuries_found:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  🏥 INJURIES:\n"
        for inj in report.injuries_found:
            output += f"║    - {inj['player']}: {inj['status'].upper()} ({inj['injury_type']})\n"
    
    if report.warnings:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  ⚠️ WARNINGS:\n"
        for warning in report.warnings[:5]:
            output += f"║    - {warning[:55]}\n"
    
    if report.issues:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  ❌ ISSUES:\n"
        for issue in report.issues[:5]:
            output += f"║    - {issue[:55]}\n"
    
    output += "╚══════════════════════════════════════════════════════════════╝"
    
    return output
```

### 5.2 `core/quality_scorer.py` - Obliczanie composite quality score

```python
# core/quality_scorer.py
from typing import Dict, Optional
from dataclasses import dataclass
from agents.data_evaluator import DataQualityReport
from config.thresholds import thresholds

@dataclass
class QualityAdjustedPrediction:
    """Predykcja z korektą za jakość danych"""
    raw_probability: float
    adjusted_probability: float
    raw_edge: float
    adjusted_edge: float
    quality_multiplier: float
    quality_penalty: float
    confidence: float


class QualityScorer:
    """
    Oblicza composite quality score i stosuje korekty do predykcji.
    
    KLUCZOWE: Niższa jakość danych = niższy efektywny edge
    """
    
    def __init__(self):
        self.thresholds = thresholds
    
    def calculate_quality_multiplier(self, quality_report: DataQualityReport) -> float:
        """
        Oblicza mnożnik jakości dla edge.
        
        Quality 85-100%: multiplier = 1.0 (pełny edge)
        Quality 70-85%:  multiplier = 0.9
        Quality 50-70%:  multiplier = 0.7
        Quality 40-50%:  multiplier = 0.5
        Quality <40%:    multiplier = 0.3 (minimal)
        
        Returns:
            Multiplier 0.3-1.0
        """
        score = quality_report.overall_score / 100  # Normalize to 0-1
        
        if score >= self.thresholds.quality_excellent:
            return 1.0
        elif score >= self.thresholds.quality_good:
            return 0.9
        elif score >= self.thresholds.quality_moderate:
            return 0.7
        elif score >= self.thresholds.quality_high_risk:
            return 0.5
        else:
            return 0.3
    
    def adjust_prediction(
        self,
        raw_probability: float,
        best_odds: float,
        quality_report: DataQualityReport
    ) -> QualityAdjustedPrediction:
        """
        Dostosowuje predykcję na podstawie jakości danych.
        
        Args:
            raw_probability: Surowe prawdopodobieństwo z modelu
            best_odds: Najlepszy dostępny kurs
            quality_report: Raport jakości danych
        
        Returns:
            QualityAdjustedPrediction z adjusted values
        """
        quality_multiplier = self.calculate_quality_multiplier(quality_report)
        quality_penalty = 1 - quality_multiplier
        
        # Raw edge
        raw_edge = (raw_probability * best_odds) - 1
        
        # Adjusted edge - mnożnik za jakość
        adjusted_edge = raw_edge * quality_multiplier
        
        # Adjusted probability (conservative adjustment)
        # Przy niskiej jakości przesuwamy prawdopodobieństwo bliżej 0.5
        if quality_multiplier < 0.7:
            # Move towards 0.5
            adjustment_strength = (0.7 - quality_multiplier) / 0.4  # 0-1
            adjusted_probability = raw_probability + (0.5 - raw_probability) * adjustment_strength * 0.3
        else:
            adjusted_probability = raw_probability
        
        # Confidence = quality * (1 - variance_penalty)
        confidence = quality_report.overall_score / 100
        if quality_report.news_report and quality_report.news_report.diversity_score < 0.3:
            confidence *= 0.8  # Penalty for low diversity
        
        return QualityAdjustedPrediction(
            raw_probability=raw_probability,
            adjusted_probability=adjusted_probability,
            raw_edge=raw_edge,
            adjusted_edge=adjusted_edge,
            quality_multiplier=quality_multiplier,
            quality_penalty=quality_penalty,
            confidence=confidence
        )
    
    def should_bet(
        self,
        adjusted_prediction: QualityAdjustedPrediction,
        league_type: str
    ) -> tuple[bool, str]:
        """
        Określa czy zakład powinien zostać postawiony.
        
        Returns:
            (should_bet, reason)
        """
        # Minimum edge per league type
        min_edges = {
            "popular": self.thresholds.min_edge_popular_league,
            "medium": self.thresholds.min_edge_medium_league,
            "unpopular": self.thresholds.min_edge_unpopular_league
        }
        
        min_edge = min_edges.get(league_type, 0.05)
        
        # Check adjusted edge
        if adjusted_prediction.adjusted_edge < min_edge:
            return False, f"Adjusted edge {adjusted_prediction.adjusted_edge:.2%} < min {min_edge:.2%}"
        
        # Check confidence
        if adjusted_prediction.confidence < 0.4:
            return False, f"Confidence {adjusted_prediction.confidence:.2%} too low"
        
        # Check quality penalty
        if adjusted_prediction.quality_penalty > 0.7:
            return False, f"Quality penalty {adjusted_prediction.quality_penalty:.2%} too high"
        
        return True, "All checks passed"


# === INTEGRATION WITH VALUE CALCULATOR ===

class QualityAwareValueCalculator:
    """
    Value Calculator z uwzględnieniem jakości danych.
    """
    
    def __init__(self, sport: str):
        self.sport = sport
        self.quality_scorer = QualityScorer()
    
    async def calculate_value_with_quality(
        self,
        match_id: str,
        raw_probability: float,
        odds_by_bookmaker: Dict,
        quality_report: DataQualityReport
    ) -> Dict:
        """
        Oblicza value z pełnym uwzględnieniem jakości danych.
        """
        # Find best odds
        best_odds = 0
        best_bookmaker = None
        
        for bookie, markets in odds_by_bookmaker.items():
            if isinstance(markets, dict):
                odds = markets.get("home", 0)
                if odds > best_odds:
                    best_odds = odds
                    best_bookmaker = bookie
        
        if best_odds == 0:
            return {
                "has_value": False,
                "reason": "No odds available"
            }
        
        # Calculate adjusted prediction
        adjusted = self.quality_scorer.adjust_prediction(
            raw_probability=raw_probability,
            best_odds=best_odds,
            quality_report=quality_report
        )
        
        # Check if should bet
        should_bet, reason = self.quality_scorer.should_bet(
            adjusted,
            quality_report.league_type
        )
        
        return {
            "has_value": should_bet,
            "reason": reason,
            "raw_probability": adjusted.raw_probability,
            "adjusted_probability": adjusted.adjusted_probability,
            "best_odds": best_odds,
            "best_bookmaker": best_bookmaker,
            "raw_edge": adjusted.raw_edge,
            "adjusted_edge": adjusted.adjusted_edge,
            "quality_multiplier": adjusted.quality_multiplier,
            "quality_penalty": adjusted.quality_penalty,
            "confidence": adjusted.confidence,
            "quality_score": quality_report.overall_score
        }
```

---

## 6. MCP SERVERS - PEŁNA IMPLEMENTACJA

### 6.1 `mcp_servers/news_server.py` - Serwer newsów

```python
# mcp_servers/news_server.py
from mcp.server.fastmcp import FastMCP
from data.news.aggregator import NewsAggregator, search_injury_news
from data.news.validator import NewsValidator
from data.news.injury_extractor import extract_injuries_from_news
import json

mcp = FastMCP("news_server")

@mcp.tool()
async def search_match_news(home: str, away: str, sport: str) -> dict:
    """
    Wyszukaj najnowsze informacje o meczu z wielu źródeł (Brave, Serper, NewsAPI).
    
    Args:
        home: Nazwa drużyny/zawodnika gospodarza
        away: Nazwa drużyny/zawodnika gościa
        sport: tennis/basketball
    
    Returns:
        dict z artykułami, liczbą, quality scores
    """
    aggregator = NewsAggregator()
    news_data = await aggregator.get_match_news(home, away, sport)
    
    # Walidacja jakości
    validator = NewsValidator()
    quality_report = validator.validate_news_quality(news_data)
    
    # Ekstrakcja kontuzji
    injuries = []
    if news_data["articles"]:
        injuries = await extract_injuries_from_news(news_data["articles"])
    
    return {
        "match": f"{home} vs {away}",
        "sport": sport,
        "articles": news_data["articles"][:20],  # Limit to 20
        "total_articles": news_data["count"],
        "quality_score": quality_report.quality_score,
        "freshness_score": quality_report.freshness_score,
        "reliability_score": quality_report.reliability_score,
        "is_sufficient": quality_report.is_sufficient,
        "injuries": injuries,
        "issues": quality_report.issues
    }


@mcp.tool()
async def get_injury_news(team: str, sport: str) -> list:
    """
    Pobierz newsy o kontuzjach dla drużyny/zawodnika.
    
    Args:
        team: Nazwa drużyny lub zawodnika
        sport: tennis/basketball
    """
    articles = await search_injury_news(team, sport)
    return articles


@mcp.tool()
async def validate_news_quality(articles: list) -> dict:
    """
    Waliduj jakość zebranych artykułów.
    
    Args:
        articles: Lista artykułów do walidacji
    """
    validator = NewsValidator()
    news_data = {"articles": articles, "count": len(articles)}
    report = validator.validate_news_quality(news_data)
    
    return {
        "quality_score": report.quality_score,
        "freshness_score": report.freshness_score,
        "reliability_score": report.reliability_score,
        "diversity_score": report.diversity_score,
        "article_count": report.article_count,
        "is_sufficient": report.is_sufficient,
        "issues": report.issues
    }


@mcp.resource("news://{home}/{away}/summary")
async def get_news_summary(home: str, away: str) -> str:
    """Zasób: Podsumowanie newsów dla meczu"""
    result = await search_match_news(home, away, "unknown")
    
    summary = f"""
NEWS SUMMARY: {home} vs {away}
===================================
Articles found: {result['total_articles']}
Quality Score: {result['quality_score']:.2f}
Freshness: {result['freshness_score']:.2f}
Reliability: {result['reliability_score']:.2f}

Top Headlines:
"""
    for i, article in enumerate(result['articles'][:5], 1):
        summary += f"{i}. {article['title'][:80]}...\n"
    
    if result['injuries']:
        summary += f"\nINJURIES DETECTED:\n"
        for inj in result['injuries']:
            summary += f"- {inj['player']}: {inj['status']} ({inj['injury_type']})\n"
    
    return summary


if __name__ == "__main__":
    mcp.run(transport='stdio')
```

### 6.2 `mcp_servers/evaluation_server.py` - Serwer ewaluacyjny

```python
# mcp_servers/evaluation_server.py
from mcp.server.fastmcp import FastMCP
from agents.data_evaluator import DataEvaluator, print_quality_report
from core.quality_scorer import QualityAwareValueCalculator
import json

mcp = FastMCP("evaluation_server")

# Global evaluator instance
_evaluator = DataEvaluator()


@mcp.tool()
async def evaluate_data_quality(
    match_id: str,
    home: str,
    away: str,
    sport: str,
    league: str,
    stats: dict = None,
    odds: dict = None
) -> dict:
    """
    Agent ewaluujący: Sprawdź jakość wszystkich danych o meczu.
    
    KLUCZOWY TOOL - używaj PRZED każdą analizą!
    
    Args:
        match_id: Unikalny ID meczu
        home: Nazwa gospodarza
        away: Nazwa gościa
        sport: tennis/basketball
        league: Nazwa ligi
        stats: Opcjonalne statystyki
        odds: Opcjonalne kursy
    
    Returns:
        dict z overall_score, scores szczegółowymi, recommendation
    """
    report = await _evaluator.evaluate_match(
        match_id=match_id,
        home=home,
        away=away,
        sport=sport,
        league=league,
        stats=stats,
        odds=odds
    )
    
    return report.to_dict()


@mcp.tool()
async def batch_evaluate_matches(matches: list, sport: str) -> list:
    """
    Ewaluuj wiele meczów równolegle.
    
    Args:
        matches: Lista meczów [{id, home, away, league, stats, odds}, ...]
        sport: tennis/basketball
    
    Returns:
        Lista raportów jakości
    """
    reports = await _evaluator.batch_evaluate(matches, sport)
    return [r.to_dict() for r in reports]


@mcp.tool()
async def get_quality_recommendation(overall_score: float, league_type: str) -> dict:
    """
    Uzyskaj rekomendację na podstawie score i typu ligi.
    
    Args:
        overall_score: Score 0-100
        league_type: popular/medium/unpopular
    """
    from config.thresholds import thresholds
    
    # Adjust thresholds for unpopular leagues
    if league_type == "unpopular":
        min_score = 60  # Higher threshold
    elif league_type == "medium":
        min_score = 50
    else:
        min_score = 40
    
    if overall_score >= 70:
        return {
            "recommendation": "PROCEED",
            "stake_modifier": 1.0,
            "message": "Good data quality - normal stake"
        }
    elif overall_score >= min_score:
        return {
            "recommendation": "CAUTION",
            "stake_modifier": 0.5,
            "message": "Moderate quality - reduce stake by 50%"
        }
    else:
        return {
            "recommendation": "SKIP",
            "stake_modifier": 0,
            "message": f"Quality {overall_score:.1f} below minimum {min_score} for {league_type} league"
        }


@mcp.tool()
async def calculate_adjusted_value(
    raw_probability: float,
    best_odds: float,
    quality_score: float,
    league_type: str
) -> dict:
    """
    Oblicz value z korektą za jakość danych.
    
    Args:
        raw_probability: Prawdopodobieństwo z modelu (0-1)
        best_odds: Najlepszy kurs
        quality_score: Score jakości (0-100)
        league_type: popular/medium/unpopular
    """
    # Quality multiplier
    if quality_score >= 85:
        multiplier = 1.0
    elif quality_score >= 70:
        multiplier = 0.9
    elif quality_score >= 50:
        multiplier = 0.7
    elif quality_score >= 40:
        multiplier = 0.5
    else:
        multiplier = 0.3
    
    raw_edge = (raw_probability * best_odds) - 1
    adjusted_edge = raw_edge * multiplier
    
    # Min edge per league type
    min_edges = {"popular": 0.03, "medium": 0.04, "unpopular": 0.05}
    min_edge = min_edges.get(league_type, 0.05)
    
    has_value = adjusted_edge >= min_edge
    
    return {
        "raw_edge": round(raw_edge, 4),
        "adjusted_edge": round(adjusted_edge, 4),
        "quality_multiplier": multiplier,
        "quality_penalty": round(1 - multiplier, 2),
        "min_edge_required": min_edge,
        "has_value": has_value,
        "value_margin": round(adjusted_edge - min_edge, 4) if has_value else None
    }


@mcp.resource("evaluation://{match_id}/report")
async def get_evaluation_report(match_id: str) -> str:
    """Zasób: Pełny raport ewaluacji w formacie tekstowym"""
    # Note: This would need actual match data in real implementation
    return f"Evaluation report for match {match_id} - use evaluate_data_quality tool with full data"


@mcp.prompt()
def evaluation_prompt() -> str:
    """Prompt do ewaluacji jakości danych"""
    return """
SYSTEM: Jesteś agentem ewaluującym jakość danych sportowych.

Twoje zadanie to ocena czy dane są wystarczające do wiarygodnej predykcji.

KRYTERIA OCENY:
1. NEWS QUALITY (30%):
   - Minimum 3 artykuły o meczu
   - Świeżość < 24h
   - Wiarygodne źródła (ESPN, BBC Sport, ATP/WTA, NBA.com)

2. STATS COMPLETENESS (30-40%):
   - Rankingi obu stron
   - Ostatnia forma (5+ meczów)
   - H2H jeśli dostępne
   - Dla tenisa: stats na nawierzchni
   - Dla koszykówki: offensive/defensive ratings

3. ODDS QUALITY (30-40%):
   - Minimum 2 bukmacherów
   - Spójność kursów (variance < 5%)
   - Świeżość kursów

REKOMENDACJE:
- PROCEED: Quality >= 70%
- CAUTION: Quality 50-70%
- HIGH_RISK: Quality 40-50%
- SKIP: Quality < 40%

Dla lig NIEPOPULARNYCH (ITF, niższe ligi):
- Wymagana wyższa jakość (min 60%)
- Więcej źródeł (min 3 bukmacherów)
"""


if __name__ == "__main__":
    mcp.run(transport='stdio')
```

### 6.3 `mcp_servers/odds_server.py` - Serwer kursów

```python
# mcp_servers/odds_server.py
from mcp.server.fastmcp import FastMCP
from data.odds.odds_api_client import TheOddsAPIClient
from data.odds.pl_scraper import PLBookieScraper
from data.odds.odds_merger import OddsMerger
from config.settings import settings, SPORTS_API_CONFIG, PL_BOOKMAKERS_CONFIG
import asyncio

mcp = FastMCP("odds_server")

# Initialize clients
_odds_api = TheOddsAPIClient(SPORTS_API_CONFIG["the_odds_api"]["api_key"]) if SPORTS_API_CONFIG["the_odds_api"]["enabled"] else None
_pl_scraper = PLBookieScraper()
_merger = OddsMerger()


@mcp.tool()
async def get_tennis_odds(
    regions: str = "eu",
    markets: str = "h2h,spreads,totals",
    include_pl_bookies: bool = True
) -> dict:
    """
    Pobierz kursy tenisowe z The Odds API + polskich bukmacherów.
    
    Args:
        regions: Region bukmacherów (eu, us, uk, au)
        markets: Typy rynków (h2h, spreads, totals)
        include_pl_bookies: Czy uwzględnić Fortuna/STS/Betclic (scraping)
    """
    all_odds = {}
    
    # 1. The Odds API
    if _odds_api:
        try:
            api_odds = await _odds_api.get_tennis_odds(regions=regions, markets=markets)
            all_odds["api_odds"] = api_odds
        except Exception as e:
            print(f"Odds API error: {e}")
    
    # 2. Polish bookmakers (scraping)
    if include_pl_bookies:
        try:
            pl_odds = await _pl_scraper.scrape_odds(
                sport="tennis",
                bookmakers=["fortuna", "sts", "betclic"]
            )
            all_odds["pl_odds"] = pl_odds
        except Exception as e:
            print(f"PL scraping error: {e}")
    
    # 3. Merge odds
    merged = _merger.merge_all_sources(all_odds)
    
    return {
        "sport": "tennis",
        "matches": merged,
        "sources_count": len([k for k in all_odds if all_odds[k]]),
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
async def get_basketball_odds(
    leagues: list = None,
    regions: str = "eu",
    include_pl_bookies: bool = True
) -> dict:
    """
    Pobierz kursy koszykarskie.
    
    Args:
        leagues: Lista lig (None = wszystkie)
        regions: Region bukmacherów
        include_pl_bookies: Czy uwzględnić polskich bukmacherów
    """
    all_odds = {}
    
    # Similar implementation to tennis
    if _odds_api:
        try:
            api_odds = await _odds_api.get_basketball_odds(leagues=leagues, regions=regions)
            all_odds["api_odds"] = api_odds
        except Exception as e:
            print(f"Odds API error: {e}")
    
    if include_pl_bookies:
        try:
            pl_odds = await _pl_scraper.scrape_odds(
                sport="basketball",
                bookmakers=["fortuna", "sts", "betclic"]
            )
            all_odds["pl_odds"] = pl_odds
        except Exception as e:
            print(f"PL scraping error: {e}")
    
    merged = _merger.merge_all_sources(all_odds)
    
    return {
        "sport": "basketball",
        "matches": merged,
        "sources_count": len([k for k in all_odds if all_odds[k]]),
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
async def compare_bookmaker_odds(match_name: str, sport: str) -> dict:
    """
    Porównaj kursy między bukmacherami dla konkretnego meczu.
    
    Args:
        match_name: Nazwa meczu (np. "Djokovic vs Nadal")
        sport: tennis/basketball
    
    Returns:
        dict z kursami od każdego bukmachera i najlepszym kursem
    """
    # Get odds from all sources
    if sport == "tennis":
        odds_data = await get_tennis_odds(include_pl_bookies=True)
    else:
        odds_data = await get_basketball_odds(include_pl_bookies=True)
    
    # Find match
    match_odds = _merger.find_match_odds(odds_data["matches"], match_name)
    
    if not match_odds:
        return {"error": f"Match '{match_name}' not found"}
    
    # Find best odds
    best_home = max(
        [(bookie, data.get("home", 0)) for bookie, data in match_odds.items()],
        key=lambda x: x[1]
    )
    best_away = max(
        [(bookie, data.get("away", 0)) for bookie, data in match_odds.items()],
        key=lambda x: x[1]
    )
    
    return {
        "match": match_name,
        "odds_by_bookmaker": match_odds,
        "best_home": {"bookmaker": best_home[0], "odds": best_home[1]},
        "best_away": {"bookmaker": best_away[0], "odds": best_away[1]},
        "bookmakers_count": len(match_odds)
    }


@mcp.tool()
async def scrape_polish_bookmakers(sport: str, bookmakers: list = None) -> dict:
    """
    Scrapuj kursy tylko z polskich bukmacherów.
    
    Args:
        sport: tennis/basketball
        bookmakers: Lista bukmacherów (None = wszystkie: fortuna, sts, betclic)
    """
    if bookmakers is None:
        bookmakers = ["fortuna", "sts", "betclic"]
    
    result = await _pl_scraper.scrape_odds(sport=sport, bookmakers=bookmakers)
    
    return {
        "sport": sport,
        "bookmakers_scraped": bookmakers,
        "matches": result,
        "timestamp": datetime.now().isoformat()
    }


@mcp.resource("odds://{sport}/live")
async def get_live_odds_resource(sport: str) -> str:
    """Zasób: Aktualne kursy dla sportu"""
    if sport == "tennis":
        data = await get_tennis_odds()
    else:
        data = await get_basketball_odds()
    
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    from datetime import datetime
    import json
    mcp.run(transport='stdio')
```

---

## 8. SYSTEM RANKINGU TOP 3

### 8.1 `agents/ranker.py` - Agent rankingowy

```python
# agents/ranker.py
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from agents.data_evaluator import DataEvaluator, DataQualityReport
from core.quality_scorer import QualityAwareValueCalculator, QualityAdjustedPrediction
from core.value_calculator import ValueCalculator, KellyCriterion
from config.thresholds import thresholds
from config.leagues import classify_league

@dataclass
class RankedMatch:
    """Pojedynczy mecz w rankingu"""
    rank: int
    match_id: str
    match_name: str
    sport: str
    league: str
    league_type: str
    
    # Value metrics
    best_edge: float
    adjusted_edge: float
    best_odds: float
    best_bookmaker: str
    selection: str  # "home" / "away" / "over" / "under"
    
    # Quality metrics
    quality_score: float
    quality_report: DataQualityReport
    
    # Recommendation
    composite_score: float
    stake_recommendation: str
    risk_level: str  # "LOW" / "MEDIUM" / "HIGH"
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "league": self.league,
            "league_type": self.league_type,
            "best_edge": f"{self.best_edge:.2%}",
            "adjusted_edge": f"{self.adjusted_edge:.2%}",
            "best_odds": self.best_odds,
            "best_bookmaker": self.best_bookmaker,
            "selection": self.selection,
            "quality_score": f"{self.quality_score:.1f}/100",
            "composite_score": f"{self.composite_score:.3f}",
            "stake": self.stake_recommendation,
            "risk": self.risk_level,
            "reasoning": self.reasoning,
            "warnings": self.warnings
        }


class MatchRanker:
    """
    Agent rankingowy - wybiera Top 3 mecze dnia na podstawie:
    1. Value Edge (adjusted for quality)
    2. Data Quality Score
    3. Confidence Score
    
    KLUCZOWE: Composite score = edge * quality * confidence
    """
    
    def __init__(self, sport: str):
        self.sport = sport
        self.evaluator = DataEvaluator()
        self.value_calculator = QualityAwareValueCalculator(sport)
        self.kelly = KellyCriterion()
    
    async def rank_top_3_matches(
        self,
        date: str,
        fixtures: List[Dict],
        max_unpopular: int = 1
    ) -> List[RankedMatch]:
        """
        Znajdź 3 najlepsze mecze do zagrania.
        
        Args:
            date: Data w formacie YYYY-MM-DD
            fixtures: Lista meczów z danymi
            max_unpopular: Max meczów z lig niepopularnych (default: 1)
        
        Returns:
            Lista max 3 RankedMatch posortowanych po composite_score
        """
        print(f"[Ranker] Analyzing {len(fixtures)} matches for {date}...")
        
        # 1. Ewaluacja jakości wszystkich meczów
        quality_reports = await self.evaluator.batch_evaluate(fixtures, self.sport)
        
        # 2. Filtruj mecze z wystarczającą jakością
        valid_matches = []
        for fixture, report in zip(fixtures, quality_reports):
            if report.is_ready:
                valid_matches.append((fixture, report))
            else:
                print(f"[Ranker] Skipping {fixture.get('home')} vs {fixture.get('away')}: {report.recommendation}")
        
        print(f"[Ranker] {len(valid_matches)} matches passed quality filter")
        
        if not valid_matches:
            return []
        
        # 3. Oblicz value dla każdego meczu
        analyzed_matches = []
        
        for fixture, quality_report in valid_matches:
            # Oblicz prawdopodobieństwo (z modelu)
            raw_prob = await self._calculate_probability(fixture)
            
            # Oblicz value z korektą jakości
            value_result = await self.value_calculator.calculate_value_with_quality(
                match_id=fixture.get("id", ""),
                raw_probability=raw_prob,
                odds_by_bookmaker=fixture.get("odds", {}),
                quality_report=quality_report
            )
            
            if not value_result["has_value"]:
                continue
            
            # Oblicz composite score
            composite = self._calculate_composite_score(
                edge=value_result["adjusted_edge"],
                quality=quality_report.overall_score / 100,
                confidence=value_result["confidence"]
            )
            
            # Określ risk level
            risk = self._determine_risk_level(quality_report)
            
            # Stake recommendation
            stake = self._calculate_stake_recommendation(
                edge=value_result["adjusted_edge"],
                quality=quality_report.overall_score,
                league_type=quality_report.league_type
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(fixture, value_result, quality_report)
            
            analyzed_matches.append(RankedMatch(
                rank=0,  # Will be set after sorting
                match_id=fixture.get("id", ""),
                match_name=f"{fixture.get('home')} vs {fixture.get('away')}",
                sport=self.sport,
                league=fixture.get("league", ""),
                league_type=quality_report.league_type,
                best_edge=value_result["raw_edge"],
                adjusted_edge=value_result["adjusted_edge"],
                best_odds=value_result["best_odds"],
                best_bookmaker=value_result["best_bookmaker"],
                selection="home",  # Simplified - could be any market
                quality_score=quality_report.overall_score,
                quality_report=quality_report,
                composite_score=composite,
                stake_recommendation=stake,
                risk_level=risk,
                reasoning=reasoning,
                warnings=quality_report.warnings
            ))
        
        # 4. Sortuj po composite score
        analyzed_matches.sort(key=lambda x: x.composite_score, reverse=True)
        
        # 5. Wybierz Top 3 z ograniczeniami
        top_3 = self._select_top_3(analyzed_matches, max_unpopular)
        
        # 6. Ustaw ranki
        for i, match in enumerate(top_3):
            match.rank = i + 1
        
        return top_3
    
    def _calculate_composite_score(
        self,
        edge: float,
        quality: float,
        confidence: float
    ) -> float:
        """
        Oblicza composite score.
        
        Wzór: edge^0.5 * quality^0.3 * confidence^0.2
        
        Edge ma większą wagę, ale quality i confidence też są istotne.
        """
        # Normalize edge (cap at 20%)
        normalized_edge = min(edge, 0.20) / 0.20
        
        # Weighted geometric mean
        composite = (
            (normalized_edge ** 0.5) *
            (quality ** 0.3) *
            (confidence ** 0.2)
        )
        
        return composite
    
    def _determine_risk_level(self, quality_report: DataQualityReport) -> str:
        """Określa poziom ryzyka"""
        score = quality_report.overall_score
        
        if score >= 70 and quality_report.league_type == "popular":
            return "LOW"
        elif score >= 60 or quality_report.league_type == "popular":
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_stake_recommendation(
        self,
        edge: float,
        quality: float,
        league_type: str
    ) -> str:
        """Oblicza zalecaną stawkę"""
        # Base stake from Kelly
        base_stake = self.kelly.calculate_stake(edge, quality / 100)
        
        # Adjust for league type
        if league_type == "unpopular":
            base_stake *= 0.5
        elif league_type == "medium":
            base_stake *= 0.75
        
        # Cap and format
        stake_pct = min(base_stake, 0.02) * 100  # Max 2%
        
        if stake_pct >= 1.5:
            return "2% bankroll"
        elif stake_pct >= 1.0:
            return "1.5% bankroll"
        else:
            return "1% bankroll"
    
    def _generate_reasoning(
        self,
        fixture: Dict,
        value_result: Dict,
        quality_report: DataQualityReport
    ) -> List[str]:
        """Generuje uzasadnienie dla rekomendacji"""
        reasons = []
        
        # Edge reasoning
        if value_result["adjusted_edge"] >= 0.05:
            reasons.append(f"Strong value: {value_result['adjusted_edge']:.1%} edge")
        else:
            reasons.append(f"Positive value: {value_result['adjusted_edge']:.1%} edge")
        
        # Quality reasoning
        if quality_report.overall_score >= 70:
            reasons.append(f"Good data quality ({quality_report.overall_score:.0f}/100)")
        
        # News reasoning
        if quality_report.news_report and quality_report.news_report.article_count >= 5:
            reasons.append(f"Strong news coverage ({quality_report.news_report.article_count} articles)")
        
        # Injury reasoning
        if quality_report.injuries_found:
            for inj in quality_report.injuries_found:
                if inj["status"] == "out":
                    reasons.append(f"Key absence: {inj['player']} OUT")
        
        return reasons[:4]  # Max 4 reasons
    
    def _select_top_3(
        self,
        matches: List[RankedMatch],
        max_unpopular: int
    ) -> List[RankedMatch]:
        """
        Wybiera Top 3 z ograniczeniami:
        - Max 1 mecz z ligi niepopularnej
        - Nie więcej niż 1 mecz z tego samego turnieju
        """
        selected = []
        unpopular_count = 0
        tournaments_used = set()
        
        for match in matches:
            if len(selected) >= 3:
                break
            
            # Check unpopular limit
            if match.league_type == "unpopular":
                if unpopular_count >= max_unpopular:
                    continue
                unpopular_count += 1
            
            # Check tournament uniqueness
            tournament = self._extract_tournament(match.league)
            if tournament in tournaments_used:
                continue
            tournaments_used.add(tournament)
            
            selected.append(match)
        
        return selected
    
    def _extract_tournament(self, league: str) -> str:
        """Wyciąga nazwę turnieju z ligi"""
        # Simplified - in real implementation would parse properly
        return league.split()[0] if league else "unknown"
    
    async def _calculate_probability(self, fixture: Dict) -> float:
        """
        Oblicza prawdopodobieństwo z modelu.
        Placeholder - w rzeczywistej implementacji użyłby modelu predykcji.
        """
        # Import model based on sport
        if self.sport == "tennis":
            from core.models.tennis_model import TennisPredictionModel
            model = TennisPredictionModel()
        else:
            from core.models.basketball_model import BasketballPredictionModel
            model = BasketballPredictionModel()
        
        return model.predict(fixture)


# === FORMATTING FUNCTIONS ===

def format_top_3_report(matches: List[RankedMatch], date: str) -> str:
    """
    Formatuje Top 3 do czytelnego raportu.
    """
    if not matches:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  🏆 TOP 3 VALUE BETS - {date}
╠══════════════════════════════════════════════════════════════╣
║  ❌ No value bets found today
║  All matches either lack sufficient data or have no edge
╚══════════════════════════════════════════════════════════════╝
"""
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  🏆 TOP 3 VALUE BETS - {date}
╠══════════════════════════════════════════════════════════════════════════╣
"""
    
    rank_emojis = ["🥇", "🥈", "🥉"]
    
    for match in matches:
        emoji = rank_emojis[match.rank - 1] if match.rank <= 3 else "  "
        
        report += f"""║
║  {emoji} #{match.rank}: {match.match_name}
║  ────────────────────────────────────────────────────────────────────────
║  League: {match.league} ({match.league_type.upper()})
║  
║  Edge: {match.adjusted_edge:.1%} (raw: {match.best_edge:.1%})
║  Odds: {match.best_odds:.2f} @ {match.best_bookmaker}
║  Quality: {match.quality_score:.0f}/100 | Composite: {match.composite_score:.3f}
║  
║  Stake: {match.stake_recommendation} | Risk: {match.risk_level}
║  
║  Reasoning:
"""
        for reason in match.reasoning:
            report += f"║    • {reason}\n"
        
        if match.warnings:
            report += "║  \n║  ⚠️ Warnings:\n"
            for warning in match.warnings[:2]:
                report += f"║    • {warning[:60]}\n"
    
    report += """║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    return report
```

---

## 9. INTERFEJS GRADIO

### 9.1 `ui/top3_tab.py` - Zakładka Top 3

```python
# ui/top3_tab.py
import gradio as gr
import pandas as pd
from datetime import datetime, date
from typing import List, Tuple
import asyncio

from agents.ranker import MatchRanker, RankedMatch, format_top_3_report

class Top3Tab:
    """
    Zakładka Gradio z Top 3 Value Bets.
    """
    
    def __init__(self):
        self.rankers = {
            "tennis": MatchRanker("tennis"),
            "basketball": MatchRanker("basketball")
        }
        self._last_results = {}
    
    def render(self) -> gr.Column:
        """Renderuje zakładkę Top 3"""
        with gr.Column() as tab:
            gr.HTML("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #2e7d32;'>🏆 TOP 3 VALUE BETS DZIŚ</h2>
                <p style='color: #666;'>System automatycznie wybiera 3 najlepsze mecze na podstawie Value Edge × Data Quality × Confidence</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    sport_dropdown = gr.Dropdown(
                        choices=["tennis", "basketball", "both"],
                        value="tennis",
                        label="🏅 Sport"
                    )
                    
                    date_picker = gr.Textbox(
                        value=str(date.today()),
                        label="📅 Data (YYYY-MM-DD)"
                    )
                    
                    min_quality = gr.Slider(
                        minimum=30,
                        maximum=80,
                        value=50,
                        step=5,
                        label="📊 Minimalny próg jakości danych (%)"
                    )
                    
                    generate_btn = gr.Button(
                        "🔍 Generuj Ranking Top 3",
                        variant="primary"
                    )
                
                with gr.Column(scale=2):
                    status_output = gr.Markdown("*Kliknij 'Generuj Ranking' aby rozpocząć analizę...*")
            
            gr.Markdown("---")
            
            # Top 3 Cards
            with gr.Row():
                top3_html = gr.HTML()
            
            # Detailed table
            with gr.Accordion("📋 Szczegółowa tabela wszystkich meczów", open=False):
                all_matches_table = gr.Dataframe(
                    headers=[
                        "Mecz", "Liga", "Typ", "Edge", "Quality",
                        "Kurs", "Bukmacher", "Stake", "Risk"
                    ],
                    interactive=False
                )
            
            # Quality reports
            with gr.Accordion("📈 Raporty jakości danych", open=False):
                quality_reports_md = gr.Markdown()
            
            # Event handler
            generate_btn.click(
                fn=self._generate_top3_async,
                inputs=[sport_dropdown, date_picker, min_quality],
                outputs=[top3_html, all_matches_table, quality_reports_md, status_output]
            )
        
        return tab
    
    def _generate_top3_async(
        self,
        sport: str,
        date_str: str,
        min_quality: float
    ) -> Tuple[str, pd.DataFrame, str, str]:
        """
        Wrapper async dla generowania Top 3.
        """
        return asyncio.run(self._generate_top3(sport, date_str, min_quality))
    
    async def _generate_top3(
        self,
        sport: str,
        date_str: str,
        min_quality: float
    ) -> Tuple[str, pd.DataFrame, str, str]:
        """
        Generuje Top 3 dla wybranego sportu.
        """
        status = f"🔄 Analizuję mecze {sport} na {date_str}..."
        
        try:
            # Get fixtures (placeholder - would fetch from API)
            fixtures = await self._get_fixtures(sport, date_str)
            
            if not fixtures:
                return (
                    self._render_no_matches_html(),
                    pd.DataFrame(),
                    "Brak meczów do analizy",
                    "❌ Nie znaleziono meczów na ten dzień"
                )
            
            status = f"🔍 Analizuję {len(fixtures)} meczów..."
            
            # Run ranker
            if sport == "both":
                tennis_top3 = await self.rankers["tennis"].rank_top_3_matches(date_str, fixtures["tennis"])
                basketball_top3 = await self.rankers["basketball"].rank_top_3_matches(date_str, fixtures["basketball"])
                all_matches = tennis_top3 + basketball_top3
                all_matches.sort(key=lambda x: x.composite_score, reverse=True)
                top3 = all_matches[:3]
            else:
                top3 = await self.rankers[sport].rank_top_3_matches(date_str, fixtures)
            
            # Render outputs
            html = self._render_top3_html(top3)
            df = self._create_dataframe(top3)
            quality_md = self._render_quality_reports(top3)
            
            status = f"✅ Analiza zakończona! Znaleziono {len(top3)} value bets."
            
            return html, df, quality_md, status
            
        except Exception as e:
            return (
                f"<div style='color: red;'>Error: {str(e)}</div>",
                pd.DataFrame(),
                f"Error: {str(e)}",
                f"❌ Błąd: {str(e)}"
            )
    
    def _render_top3_html(self, matches: List[RankedMatch]) -> str:
        """Renderuje Top 3 jako HTML cards"""
        if not matches:
            return self._render_no_matches_html()
        
        rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]  # Gold, Silver, Bronze
        risk_colors = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#f44336"}
        
        html = "<div style='display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;'>"
        
        for match in matches:
            color = rank_colors[match.rank - 1] if match.rank <= 3 else "#666"
            risk_color = risk_colors.get(match.risk_level, "#666")
            
            html += f"""
            <div style='
                background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color};
                border-radius: 15px;
                padding: 20px;
                width: 300px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <div style='text-align: center; margin-bottom: 15px;'>
                    <span style='font-size: 2em;'>{"🥇🥈🥉"[match.rank-1] if match.rank <= 3 else "🏅"}</span>
                    <h3 style='margin: 5px 0; color: #333;'>#{match.rank}</h3>
                </div>
                
                <h4 style='margin: 10px 0; color: #222; text-align: center;'>
                    {match.match_name}
                </h4>
                
                <p style='color: #666; text-align: center; font-size: 0.9em;'>
                    {match.league}
                </p>
                
                <hr style='border: 1px solid #eee; margin: 15px 0;'>
                
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #2e7d32;'>
                            +{match.adjusted_edge:.1%}
                        </div>
                        <div style='font-size: 0.8em; color: #666;'>Edge</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #1976d2;'>
                            {match.quality_score:.0f}
                        </div>
                        <div style='font-size: 0.8em; color: #666;'>Quality</div>
                    </div>
                </div>
                
                <div style='margin-top: 15px; padding: 10px; background: #f5f5f5; border-radius: 8px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Kurs:</span>
                        <strong>{match.best_odds:.2f}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Bukmacher:</span>
                        <strong>{match.best_bookmaker}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Stake:</span>
                        <strong>{match.stake_recommendation}</strong>
                    </div>
                </div>
                
                <div style='
                    margin-top: 15px;
                    padding: 5px 10px;
                    background: {risk_color}22;
                    border: 1px solid {risk_color};
                    border-radius: 20px;
                    text-align: center;
                    color: {risk_color};
                    font-weight: bold;
                '>
                    Risk: {match.risk_level}
                </div>
                
                <div style='margin-top: 15px; font-size: 0.85em;'>
                    <strong>Reasoning:</strong>
                    <ul style='margin: 5px 0; padding-left: 20px;'>
                        {"".join(f"<li>{r}</li>" for r in match.reasoning[:3])}
                    </ul>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _render_no_matches_html(self) -> str:
        """Renderuje komunikat o braku meczów"""
        return """
        <div style='
            text-align: center;
            padding: 40px;
            background: #fff3e0;
            border-radius: 15px;
            border: 2px solid #ff9800;
        '>
            <span style='font-size: 3em;'>🔍</span>
            <h3 style='color: #e65100;'>Brak value bets na dziś</h3>
            <p style='color: #666;'>
                Wszystkie mecze albo nie mają wystarczającej jakości danych,
                albo nie oferują dodatniego edge.
            </p>
        </div>
        """
    
    def _create_dataframe(self, matches: List[RankedMatch]) -> pd.DataFrame:
        """Tworzy DataFrame z meczami"""
        if not matches:
            return pd.DataFrame()
        
        data = []
        for m in matches:
            data.append({
                "Mecz": m.match_name,
                "Liga": m.league,
                "Typ": m.selection,
                "Edge": f"+{m.adjusted_edge:.1%}",
                "Quality": f"{m.quality_score:.0f}%",
                "Kurs": f"{m.best_odds:.2f}",
                "Bukmacher": m.best_bookmaker,
                "Stake": m.stake_recommendation,
                "Risk": m.risk_level
            })
        
        return pd.DataFrame(data)
    
    def _render_quality_reports(self, matches: List[RankedMatch]) -> str:
        """Renderuje raporty jakości jako Markdown"""
        if not matches:
            return "Brak danych"
        
        md = ""
        for m in matches:
            qr = m.quality_report
            md += f"""
### {m.match_name}

| Metryka | Wartość |
|---------|---------|
| Overall Score | {qr.overall_score:.1f}/100 |
| News Quality | {qr.news_quality_score:.1f}/100 |
| Stats Complete | {qr.stats_completeness_score:.1f}/100 |
| Odds Quality | {qr.odds_quality_score:.1f}/100 |
| Bookmakers | {qr.odds_sources_count} |
| Recommendation | **{qr.recommendation}** |

"""
            if qr.injuries_found:
                md += "**Kontuzje:**\n"
                for inj in qr.injuries_found:
                    md += f"- {inj['player']}: {inj['status']} ({inj['injury_type']})\n"
                md += "\n"
            
            if qr.warnings:
                md += "**Ostrzeżenia:**\n"
                for w in qr.warnings[:3]:
                    md += f"- ⚠️ {w}\n"
                md += "\n"
            
            md += "---\n\n"
        
        return md
    
    async def _get_fixtures(self, sport: str, date_str: str) -> List[Dict]:
        """
        Pobiera fixtures z API.
        Placeholder - w rzeczywistej implementacji użyłby MCP servers.
        """
        # This would call the MCP servers to get real data
        # For now, return mock data
        return []
```

---

## 10. GŁÓWNA PĘTLA SYSTEMU

### 10.1 `betting_floor.py` - Zintegrowana pętla

```python
# betting_floor.py
"""
NEXUS AI - Główna pętla systemu predykcji sportowej.

Cykl działania:
1. Sprawdź czy są mecze na dziś
2. Pobierz newsy (Brave/Serper)
3. Ewaluuj jakość danych
4. Oblicz value dla meczów z quality > threshold
5. Wygeneruj Top 3 ranking
6. Wykonaj zakłady (opcjonalnie)
7. Sleep i powtórz
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import logging

from config.settings import settings
from agents.data_evaluator import DataEvaluator
from agents.ranker import MatchRanker, format_top_3_report
from agents.bettor import BettorAgent
from data.news.aggregator import NewsAggregator
from database.db import get_session, save_daily_report
from mcp_servers.alerts_server import send_alert

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BettingFloor")


class BettingFloor:
    """
    Główna klasa orkiestrująca system.
    """
    
    def __init__(self):
        self.evaluator = DataEvaluator()
        self.rankers = {
            "tennis": MatchRanker("tennis"),
            "basketball": MatchRanker("basketball")
        }
        self.news_aggregator = NewsAggregator()
        self.bettor = BettorAgent() if settings.ENABLE_LIVE_BETTING else None
        
        self.is_running = False
        self.last_run = None
    
    async def run_forever(self):
        """
        Uruchamia główną pętlę systemu.
        """
        self.is_running = True
        logger.info("🚀 NEXUS AI Betting Floor started")
        
        while self.is_running:
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await send_alert(f"❌ Betting Floor error: {e}")
            
            # Sleep before next cycle
            sleep_minutes = settings.RUN_EVERY_N_MINUTES
            logger.info(f"💤 Sleeping {sleep_minutes} minutes until next cycle...")
            await asyncio.sleep(sleep_minutes * 60)
    
    async def _run_cycle(self):
        """
        Pojedynczy cykl analizy.
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        logger.info(f"═══════════════════════════════════════════════════")
        logger.info(f"🔄 Starting analysis cycle for {today}")
        logger.info(f"═══════════════════════════════════════════════════")
        
        # Check if sports markets are open
        if not self._is_market_open():
            logger.info("📴 Sports markets closed, skipping...")
            return
        
        all_top3 = []
        
        # Analyze each sport
        for sport in ["tennis", "basketball"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 Analyzing {sport.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                top3 = await self._analyze_sport(sport, today)
                all_top3.extend(top3)
            except Exception as e:
                logger.error(f"Error analyzing {sport}: {e}")
                continue
        
        # Generate combined report
        if all_top3:
            # Sort by composite score
            all_top3.sort(key=lambda x: x.composite_score, reverse=True)
            final_top3 = all_top3[:3]
            
            report = format_top_3_report(final_top3, today)
            logger.info(report)
            
            # Save to database
            await save_daily_report(today, final_top3)
            
            # Send alert
            await self._send_top3_alert(final_top3, today)
            
            # Execute bets if enabled
            if self.bettor:
                for match in final_top3:
                    await self._execute_bet(match)
        else:
            logger.info("❌ No value bets found today")
        
        self.last_run = now
    
    async def _analyze_sport(self, sport: str, date_str: str) -> List:
        """
        Analizuje wszystkie mecze dla danego sportu.
        """
        # 1. Get fixtures
        fixtures = await self._get_fixtures(sport, date_str)
        logger.info(f"Found {len(fixtures)} {sport} fixtures")
        
        if not fixtures:
            return []
        
        # 2. Evaluate data quality for each match
        logger.info("Evaluating data quality...")
        quality_reports = await self.evaluator.batch_evaluate(fixtures, sport)
        
        # 3. Filter by quality
        valid_fixtures = []
        for fixture, report in zip(fixtures, quality_reports):
            if report.is_ready:
                fixture["quality_report"] = report
                valid_fixtures.append(fixture)
            else:
                logger.debug(f"Skipping {fixture.get('home')} vs {fixture.get('away')}: "
                           f"{report.recommendation} (score: {report.overall_score:.1f})")
        
        logger.info(f"{len(valid_fixtures)}/{len(fixtures)} matches passed quality filter")
        
        # 4. Generate Top 3
        ranker = self.rankers[sport]
        top3 = await ranker.rank_top_3_matches(date_str, valid_fixtures)
        
        logger.info(f"Top 3 {sport} matches selected")
        for match in top3:
            logger.info(f"  #{match.rank}: {match.match_name} "
                       f"(edge: {match.adjusted_edge:.1%}, quality: {match.quality_score:.0f})")
        
        return top3
    
    async def _get_fixtures(self, sport: str, date_str: str) -> List[Dict]:
        """
        Pobiera wszystkie mecze na dany dzień.
        Używa MCP servers do pobrania danych.
        """
        from mcp_servers.tennis_server import get_fixtures as get_tennis_fixtures
        from mcp_servers.basketball_server import get_fixtures as get_basketball_fixtures
        from mcp_servers.odds_server import get_tennis_odds, get_basketball_odds
        
        if sport == "tennis":
            fixtures = await get_tennis_fixtures(date_str)
            odds = await get_tennis_odds()
        else:
            fixtures = await get_basketball_fixtures(date_str)
            odds = await get_basketball_odds()
        
        # Merge odds with fixtures
        for fixture in fixtures:
            match_key = f"{fixture.get('home')} vs {fixture.get('away')}"
            fixture["odds"] = odds.get("matches", {}).get(match_key, {})
        
        return fixtures
    
    async def _execute_bet(self, match):
        """
        Wykonuje zakład na mecz.
        """
        if not self.bettor:
            return
        
        logger.info(f"💰 Placing bet on {match.match_name}")
        
        try:
            result = await self.bettor.place_bet(
                match_id=match.match_id,
                selection=match.selection,
                stake=match.stake_recommendation,
                odds=match.best_odds,
                bookmaker=match.best_bookmaker
            )
            
            if result["success"]:
                logger.info(f"✅ Bet placed successfully: {result['bet_id']}")
            else:
                logger.warning(f"⚠️ Bet failed: {result['error']}")
        except Exception as e:
            logger.error(f"❌ Bet execution error: {e}")
    
    async def _send_top3_alert(self, top3: List, date_str: str):
        """
        Wysyła alert z Top 3.
        """
        message = f"🏆 TOP 3 VALUE BETS - {date_str}\n\n"
        
        for match in top3:
            emoji = "🥇🥈🥉"[match.rank - 1] if match.rank <= 3 else "🏅"
            message += f"{emoji} {match.match_name}\n"
            message += f"   Edge: +{match.adjusted_edge:.1%} | "
            message += f"Odds: {match.best_odds:.2f} @ {match.best_bookmaker}\n"
            message += f"   Quality: {match.quality_score:.0f}/100 | "
            message += f"Stake: {match.stake_recommendation}\n\n"
        
        await send_alert(message)
    
    def _is_market_open(self) -> bool:
        """
        Sprawdza czy rynki sportowe są otwarte.
        """
        now = datetime.now()
        hour = now.hour
        
        # Markets typically active 8:00 - 23:00
        return 8 <= hour <= 23
    
    def stop(self):
        """Zatrzymuje system"""
        self.is_running = False
        logger.info("🛑 Betting Floor stopped")


# === ENTRY POINT ===

async def main():
    """Main entry point"""
    floor = BettingFloor()
    
    try:
        await floor.run_forever()
    except KeyboardInterrupt:
        floor.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 11. DEPLOYMENT I MONITORING

### 11.1 `docker-compose.yml`

```yaml
version: '3.8'

services:
  nexus-app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "7860:7860"
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ODDS_API_KEY=${ODDS_API_KEY}
      - API_TENNIS_KEY=${API_TENNIS_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://nexus:${DB_PASSWORD}@postgres:5432/nexus
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  betting-floor:
    build:
      context: .
      dockerfile: docker/Dockerfile
    command: python betting_floor.py
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ODDS_API_KEY=${ODDS_API_KEY}
      - API_TENNIS_KEY=${API_TENNIS_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://nexus:${DB_PASSWORD}@postgres:5432/nexus
      - REDIS_URL=redis://redis:6379/0
      - RUN_EVERY_N_MINUTES=30
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  mcp-news:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/news_server.py
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
    restart: unless-stopped

  mcp-evaluation:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/evaluation_server.py
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped

  mcp-odds:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/odds_server.py
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=nexus
      - POSTGRES_USER=nexus
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafanadata:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
  grafanadata:
```

### 11.2 Szacowane Koszty Miesięczne

| Serwis | Plan | Koszt | Limit |
|--------|------|-------|-------|
| **Brave Search API** | Free | $0 | 2,000 req/mies |
| **Serper API** | Free | $0 | 2,500 req/mies |
| **NewsAPI** | Free | $0 | 100 req/dzień |
| **The Odds API** | Starter | $79 | 10,000 req/mies |
| **api-tennis.com** | Basic | ~$30 | 5,000 req/mies |
| **Anthropic Claude** | Pay-as-you-go | ~$50-100 | ~500k tokens/dzień |
| **Hosting (VPS)** | Basic | ~$20 | 4GB RAM |
| **Redis Cloud** | Free | $0 | 30MB |
| **PostgreSQL** | Self-hosted | $0 | Included in VPS |

**RAZEM: ~$180-230/miesiąc**

---

## PODSUMOWANIE

System **NEXUS AI v2.0** integruje najlepsze rozwiązania z obu specyfikacji:

### Kluczowe Komponenty:

1. **NewsAggregator** - agregacja z Brave, Serper, NewsAPI z deduplikacją i sortowaniem
2. **NewsValidator** - quality, freshness, reliability, diversity scores
3. **InjuryExtractor** - ekstrakcja kontuzji przez LLM
4. **DataEvaluator** - kompleksowa ocena jakości przed analizą (KLUCZOWY!)
5. **QualityScorer** - korekty predykcji na podstawie jakości danych
6. **MatchRanker** - ranking Top 3 po composite score
7. **MCP Servers** - modularnebackendy dla każdej funkcjonalności
8. **Gradio UI** - interaktywny interfejs z Top 3 cards
9. **BettingFloor** - główna pętla z pełnym workflow

### Innowacje:

- **Quality-Aware Value** - edge jest mnożony przez quality multiplier
- **Adaptive Thresholds** - wyższe wymagania dla lig niepopularnych
- **Multi-Source News** - 3 źródła newsów + ekstrakcja kontuzji
- **Composite Scoring** - edge × quality × confidence
- **Risk Levels** - automatyczne określanie poziomu ryzyka

### Następne Kroki:

1. Implementacja `data/odds/pl_scraper.py` (Playwright scraper)
2. Implementacja `core/models/tennis_model.py` i `basketball_model.py`
3. Integracja z rzeczywistymi API
4. Testy backtestingowe
5. Deployment na VPS
