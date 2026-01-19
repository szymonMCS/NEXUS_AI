# config/free_apis.py
"""
Konfiguracja darmowych źródeł danych dla NEXUS AI Lite.
Używane zamiast płatnych API podczas development/testing.
"""

# === THESPORTSDB - Darmowe API ===
THESPORTSDB_CONFIG = {
    "base_url": "https://www.thesportsdb.com/api/v1/json",
    "api_key": "3",  # Darmowy klucz publiczny (lub "123")
    "enabled": True,
    "endpoints": {
        "all_leagues": "/all_leagues.php",
        "league_events": "/eventsseason.php",  # ?id={league_id}&s={season}
        "next_events": "/eventsnextleague.php",  # ?id={league_id}
        "team_details": "/lookupteam.php",  # ?id={team_id}
        "player_details": "/lookupplayer.php",  # ?id={player_id}
        "search_team": "/searchteams.php",  # ?t={team_name}
        "search_player": "/searchplayers.php",  # ?p={player_name}
        "event_details": "/lookupevent.php",  # ?id={event_id}
    },
    # Dostępne ligi tenisa i koszykówki
    "tennis_leagues": {
        "4464": "ATP Tour",
        "4465": "WTA Tour",
    },
    "basketball_leagues": {
        "4387": "NBA",
        "4424": "EuroLeague",
        "4710": "PLK",  # Polska Liga Koszykówki
    }
}

# === API-SPORTS FREE TIER ===
# 100 requests/day per API (basketball, tennis osobno)
API_SPORTS_CONFIG = {
    "basketball": {
        "base_url": "https://v1.basketball.api-sports.io",
        "enabled": False,  # Włącz jeśli masz klucz
        "daily_limit": 100,
        "endpoints": {
            "games": "/games",
            "standings": "/standings",
            "teams": "/teams",
            "statistics": "/statistics",
            "h2h": "/h2h",
        }
    },
    "tennis": {
        "base_url": "https://v1.tennis.api-sports.io",
        "enabled": False,  # Włącz jeśli masz klucz
        "daily_limit": 100,
        "endpoints": {
            "games": "/games",
            "rankings": "/rankings",
            "h2h": "/h2h",
            "statistics": "/statistics",
        }
    }
}

# === ALLSPORTSAPI - Darmowe (260 req/h) ===
ALLSPORTSAPI_CONFIG = {
    "base_url": "https://allsportsapi.com",
    "enabled": False,  # Wymagana rejestracja
    "hourly_limit": 260,
    "endpoints": {
        "tennis": "/tennis",
        "basketball": "/basketball",
    }
}

# === SOFASCORE - Web Scraping/API ===
SOFASCORE_CONFIG = {
    "base_url": "https://api.sofascore.com/api/v1",
    "enabled": True,
    "rate_limit": 2,  # requests per second (be gentle!)
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sofascore.com/",
    },
    "sport_ids": {
        "tennis": 5,
        "basketball": 2,
    },
    "endpoints": {
        "events_by_date": "/sport/{sport}/scheduled-events/{date}",
        "event_details": "/event/{event_id}",
        "event_statistics": "/event/{event_id}/statistics",
        "h2h": "/event/{event_id}/h2h/events",
        "player_statistics": "/player/{player_id}/statistics",
        "team_statistics": "/team/{team_id}/statistics",
        "rankings": "/rankings/type/{ranking_type}",
    }
}

# === FLASHSCORE - Web Scraping ===
FLASHSCORE_CONFIG = {
    "base_url": "https://www.flashscore.pl",
    "enabled": True,
    "rate_limit": 1,  # requests per second (be gentle!)
    "use_playwright": True,  # Requires dynamic JS rendering
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
    "sport_paths": {
        "tennis": "/tenis/",
        "basketball": "/koszykowka/",
    },
    "selectors": {
        "match_row": ".event__match",
        "home_team": ".event__participant--home",
        "away_team": ".event__participant--away",
        "match_time": ".event__time",
        "odds_row": ".ui-table__row",
        "bookmaker": ".oddsCell__bookmaker",
        "odds_value": ".oddsCell__odd",
    }
}

# === POLISH BOOKMAKERS - Web Scraping ===
PL_BOOKMAKERS_CONFIG = {
    "fortuna": {
        "base_url": "https://www.efortuna.pl",
        "enabled": True,
        "rate_limit": 1,  # requests per minute
        "use_playwright": True,
        "paths": {
            "tennis": "/zaklady-bukmacherskie/tenis",
            "basketball": "/zaklady-bukmacherskie/koszykowka",
        },
        "selectors": {
            "match_row": "[data-testid='event-row']",
            "teams": ".event-name",
            "odds": ".odds-value",
            "time": ".event-time",
        }
    },
    "sts": {
        "base_url": "https://www.sts.pl",
        "enabled": True,
        "rate_limit": 1,
        "use_playwright": True,
        "paths": {
            "tennis": "/pl/zaklady-bukmacherskie/tenis",
            "basketball": "/pl/zaklady-bukmacherskie/koszykowka",
        },
        "selectors": {
            "match_row": ".match-row",
            "teams": ".participant-name",
            "odds": ".odds-button__value",
            "time": ".match-time",
        }
    },
    "betclic": {
        "base_url": "https://www.betclic.pl",
        "enabled": True,
        "rate_limit": 1,
        "use_playwright": True,
        "paths": {
            "tennis": "/sport/tenis",
            "basketball": "/sport/koszykowka",
        },
        "selectors": {
            "match_row": ".match",
            "teams": ".scoreboard_contestantLabel",
            "odds": ".oddValue",
            "time": ".match-time",
        }
    }
}

# === NEWS SEARCH - Brave + Serper (darmowe limity) ===
NEWS_SEARCH_CONFIG = {
    "brave": {
        "enabled": True,  # Requires BRAVE_API_KEY in .env
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "monthly_limit": 2000,  # Free tier
    },
    "serper": {
        "enabled": True,  # Requires SERPER_API_KEY in .env
        "endpoint": "https://google.serper.dev/search",
        "monthly_limit": 2500,  # Free tier
    }
}

# === CACHE SETTINGS (dla web scraping) ===
WEB_CACHE_CONFIG = {
    "enabled": True,
    "ttl_seconds": 3600,  # 1 hour cache
    "max_size_mb": 100,
}

# === RATE LIMITING (ochrona przed banem) ===
RATE_LIMIT_CONFIG = {
    "sofascore": {
        "requests_per_second": 2,
        "burst": 5,
    },
    "flashscore": {
        "requests_per_second": 1,
        "burst": 3,
    },
    "bookmakers": {
        "requests_per_minute": 5,
        "delay_between_requests": 2,  # seconds
    }
}
