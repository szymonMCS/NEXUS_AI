"""Secure settings configuration for NEXUS AI"""
import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from .secrets_manager import secrets_manager

AppMode = Literal["lite", "pro"]

class Settings(BaseSettings):
    """Secure central configuration for NEXUS AI"""
    
    # === BASIC SETTINGS ===
    APP_NAME: str = "NEXUS AI"
    APP_VERSION: str = "2.0.0"
    APP_MODE: AppMode = Field(default="lite", env="APP_MODE")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # === DATABASE ===
    DATABASE_URL: str = Field(default="sqlite:///./nexus.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # === SYSTEM SETTINGS ===
    RUN_EVERY_N_MINUTES: int = Field(default=30, env="RUN_EVERY_N_MINUTES")
    ENABLE_LIVE_ODDS: bool = Field(default=False, env="ENABLE_LIVE_ODDS")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # === LITE MODE SETTINGS ===
    USE_WEB_SCRAPING: bool = Field(default=True, env="USE_WEB_SCRAPING")
    USE_FREE_APIS: bool = Field(default=True, env="USE_FREE_APIS")
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")
    CACHE_TTL_HOURS: int = Field(default=1, env="CACHE_TTL_HOURS")
    
    # === SECURITY SETTINGS ===
    SECURE_MODE: bool = Field(default=True, env="SECURE_MODE")
    ENCRYPT_SECRETS: bool = Field(default=True, env="ENCRYPT_SECRETS")
    
    @property
    def is_lite_mode(self) -> bool:
        """Check if app is in Lite mode"""
        return self.APP_MODE == "lite"
    
    @property
    def is_pro_mode(self) -> bool:
        """Check if app is in Pro mode"""
        return self.APP_MODE == "pro"
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Securely get API key or secret"""
        if not self.SECURE_MODE:
            # Fallback to environment variables for development
            return os.getenv(f"NEXUS_{key.upper()}", default)
        
        # Use secrets manager in secure mode
        return secrets_manager.get_secret(key.lower(), default)
    
    # === API KEYS (Secure) ===
    @property
    def BRAVE_API_KEY(self) -> Optional[str]:
        return self.get_secret("brave_api_key")
    
    @property
    def SERPER_API_KEY(self) -> Optional[str]:
        return self.get_secret("serper_api_key")
    
    @property
    def NEWSAPI_KEY(self) -> Optional[str]:
        return self.get_secret("newsapi_key")
    
    @property
    def ODDS_API_KEY(self) -> Optional[str]:
        return self.get_secret("odds_api_key")
    
    @property
    def API_TENNIS_KEY(self) -> Optional[str]:
        return self.get_secret("api_tennis_key")
    
    @property
    def BETS_API_KEY(self) -> Optional[str]:
        return self.get_secret("bets_api_key")
    
    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        return self.get_secret("anthropic_api_key")
    
    @property
    def API_SPORTS_BASKETBALL_KEY(self) -> Optional[str]:
        return self.get_secret("api_sports_basketball_key")
    
    @property
    def API_SPORTS_TENNIS_KEY(self) -> Optional[str]:
        return self.get_secret("api_sports_tennis_key")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global instance
settings = Settings()

# === NEWS API CONFIG ===
def get_news_config() -> Dict[str, Any]:
    """Get news API configuration based on app mode"""
    base_config = {
        "brave_search": {
            "enabled": bool(settings.BRAVE_API_KEY) and (settings.is_lite_mode or settings.is_pro_mode),
            "api_key": settings.BRAVE_API_KEY,
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "rate_limit": 10,
            "monthly_limit": 2000,
            "priority": 1
        },
        "serper": {
            "enabled": bool(settings.SERPER_API_KEY) and (settings.is_lite_mode or settings.is_pro_mode),
            "api_key": settings.SERPER_API_KEY,
            "endpoint": "https://google.serper.dev/search",
            "rate_limit": 100,
            "monthly_limit": 2500,
            "priority": 2
        },
        "newsapi": {
            "enabled": bool(settings.NEWSAPI_KEY) and settings.is_pro_mode,
            "api_key": settings.NEWSAPI_KEY,
            "endpoint": "https://newsapi.org/v2/everything",
            "rate_limit": 50,
            "priority": 3
        }
    }
    return base_config

# === SPORTS API CONFIG ===
def get_sports_api_config() -> Dict[str, Any]:
    """Get sports API configuration based on app mode"""
    return {
        "the_odds_api": {
            "enabled": bool(settings.ODDS_API_KEY) and settings.is_pro_mode,
            "api_key": settings.ODDS_API_KEY,
            "base_url": "https://api.the-odds-api.com/v4",
            "rate_limit": 10,
            "mode": "pro"
        },
        "api_tennis": {
            "enabled": bool(settings.API_TENNIS_KEY) and settings.is_pro_mode,
            "api_key": settings.API_TENNIS_KEY,
            "base_url": "https://api.api-tennis.com/tennis",
            "rate_limit": 100,
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
