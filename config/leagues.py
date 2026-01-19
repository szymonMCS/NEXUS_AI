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
