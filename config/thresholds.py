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
