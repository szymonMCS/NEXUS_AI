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
