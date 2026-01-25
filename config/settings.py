# config/settings.py
import os
from pathlib import Path
from typing import Optional, Literal, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

AppMode = Literal["lite", "pro"]

class Settings(BaseSettings):
    """Centralna konfiguracja aplikacji NEXUS AI"""

    # === PODSTAWOWE ===
    APP_NAME: str = "NEXUS AI"
    APP_VERSION: str = "2.0.0"
    APP_MODE: AppMode = "lite"  # "lite" lub "pro"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # === BAZA DANYCH ===
    DATABASE_URL: str = "sqlite:///./nexus.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # === API KEYS - NEWS ===
    BRAVE_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None
    NEWSAPI_KEY: Optional[str] = None

    # === API KEYS - SPORTS DATA ===
    ODDS_API_KEY: Optional[str] = None
    API_TENNIS_KEY: Optional[str] = None
    BETS_API_KEY: Optional[str] = None

    # === API KEYS - LLM ===
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    MINIMAX_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "anthropic"  # "anthropic", "openai", or "minimax"
    MODEL_NAME: str = "claude-sonnet-4-20250506"

    # === SYSTEM ===
    RUN_EVERY_N_MINUTES: int = 30
    ENABLE_LIVE_ODDS: bool = False
    MAX_CONCURRENT_REQUESTS: int = 10

    # === LITE MODE SETTINGS ===
    USE_WEB_SCRAPING: bool = True
    USE_FREE_APIS: bool = True
    ENABLE_CACHE: bool = True
    CACHE_TTL_HOURS: int = 1

    # === API-SPORTS FREE TIER (opcjonalne) ===
    API_SPORTS_BASKETBALL_KEY: Optional[str] = None
    API_SPORTS_TENNIS_KEY: Optional[str] = None

    # === BETTING CONFIGURATION ===
    DEFAULT_BANKROLL: float = 1000.0
    KELLY_FRACTION: float = 0.25
    MAX_STAKE_PERCENT: float = 0.05
    MAX_DAILY_STAKE_PERCENT: float = 0.20
    MIN_EDGE_POPULAR: float = 0.03
    MIN_EDGE_MEDIUM: float = 0.05
    MIN_EDGE_UNPOPULAR: float = 0.07

    # === SECURITY ===
    SECRET_KEY: str = "nexus_secret_key_change_in_production"
    CORS_ORIGINS: str = '["http://localhost:3000", "http://localhost:8000"]'

    @property
    def get_cors_origins(self) -> List[str]:
        """Parse CORS_ORIGINS from JSON string."""
        import json
        try:
            return json.loads(self.CORS_ORIGINS)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:3000", "http://localhost:8000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def is_lite_mode(self) -> bool:
        """Czy aplikacja działa w trybie Lite"""
        return self.APP_MODE == "lite"

    @property
    def is_pro_mode(self) -> bool:
        """Czy aplikacja działa w trybie Pro"""
        return self.APP_MODE == "pro"


# Singleton instance
settings = Settings()


# === NEWS API CONFIG ===
# W trybie Lite: używaj tylko Brave/Serper (darmowe limity)
# W trybie Pro: używaj wszystkich dostępnych źródeł
NEWS_CONFIG = {
    "brave_search": {
        "enabled": bool(settings.BRAVE_API_KEY) and (settings.is_lite_mode or settings.is_pro_mode),
        "api_key": settings.BRAVE_API_KEY,
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "rate_limit": 10,  # requests per minute
        "monthly_limit": 2000,  # Free tier limit
        "priority": 1  # Highest priority
    },
    "serper": {
        "enabled": bool(settings.SERPER_API_KEY) and (settings.is_lite_mode or settings.is_pro_mode),
        "api_key": settings.SERPER_API_KEY,
        "endpoint": "https://google.serper.dev/search",
        "rate_limit": 100,
        "monthly_limit": 2500,  # Free tier limit
        "priority": 2
    },
    "newsapi": {
        "enabled": bool(settings.NEWSAPI_KEY) and settings.is_pro_mode,  # Tylko Pro
        "api_key": settings.NEWSAPI_KEY,
        "endpoint": "https://newsapi.org/v2/everything",
        "rate_limit": 50,
        "priority": 3
    }
}


# === SPORTS API CONFIG (PRO MODE) ===
# W trybie Lite: te API są wyłączone, używamy darmowych alternatyw
# W trybie Pro: płatne API z pełnym dostępem
SPORTS_API_CONFIG = {
    "the_odds_api": {
        "enabled": bool(settings.ODDS_API_KEY) and settings.is_pro_mode,
        "api_key": settings.ODDS_API_KEY,
        "base_url": "https://api.the-odds-api.com/v4",
        "rate_limit": 10,  # per second
        "mode": "pro"
    },
    "api_tennis": {
        "enabled": bool(settings.API_TENNIS_KEY) and settings.is_pro_mode,
        "api_key": settings.API_TENNIS_KEY,
        "base_url": "https://api.api-tennis.com/tennis",
        "rate_limit": 100,  # per minute
        "mode": "pro"
    },
    "bets_api": {
        "enabled": bool(settings.BETS_API_KEY) and settings.is_pro_mode,
        "api_key": settings.BETS_API_KEY,
        "base_url": "https://api.betsapi.com/v1",
        "rate_limit": 50,
        "mode": "pro"
    }
}

# === FREE SPORTS API CONFIG (LITE MODE) ===
# Importuj konfigurację darmowych API z free_apis.py
if settings.is_lite_mode or (settings.is_pro_mode and settings.USE_FREE_APIS):
    from config.free_apis import (
        THESPORTSDB_CONFIG,
        API_SPORTS_CONFIG,
        SOFASCORE_CONFIG,
        FLASHSCORE_CONFIG,
        PL_BOOKMAKERS_CONFIG,
        NEWS_SEARCH_CONFIG,
        WEB_CACHE_CONFIG,
        RATE_LIMIT_CONFIG,
    )


# PL_BOOKMAKERS_CONFIG jest teraz importowana z free_apis.py w trybie Lite
# W trybie Pro również można używać scrapingu jako backup
