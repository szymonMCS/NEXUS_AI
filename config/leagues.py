# config/leagues.py
"""
League configuration for all supported sports in NEXUS AI.
Classifies leagues into popularity tiers for data quality requirements.
"""
from typing import Dict, List, Literal, Optional

LeagueType = Literal["popular", "medium", "unpopular"]
SportType = Literal["tennis", "basketball", "greyhound", "handball", "table_tennis"]

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


# === GREYHOUND RACING ===
GREYHOUND_LEAGUES: Dict[LeagueType, List[str]] = {
    "popular": [
        # UK Grade 1 & 2
        "uk_grade1", "uk_grade2",
        "english_greyhound_derby", "scottish_derby", "irish_derby",
        "tv_trophy", "arc",
        # Irish Top
        "irish_classic", "shelbourne_open",
    ],
    "medium": [
        # UK Grade 3-4
        "uk_grade3", "uk_grade4",
        # Australian
        "aus_group1", "aus_group2", "aus_group3",
        "melbourne_cup", "sandown_cup",
        # Other UK tracks
        "romford", "monmore", "crayford", "belle_vue",
    ],
    "unpopular": [
        # Lower grades
        "uk_grade5", "uk_grade6",
        # Regional
        "regional_uk", "regional_ireland",
        # Trials
        "trials", "puppy_stakes",
    ]
}

# === HANDBALL LEAGUES ===
HANDBALL_LEAGUES: Dict[LeagueType, List[str]] = {
    "popular": [
        # International
        "ehf_champions_league", "ehf_european_league",
        "world_championship", "european_championship", "olympics",
        # Top national leagues
        "bundesliga_men", "bundesliga_women",  # Germany
        "lidl_starligue",  # France
        "liga_asobal",  # Spain
        "danish_handboldliga",  # Denmark
    ],
    "medium": [
        # Other European leagues
        "polish_superliga",  # PGNiG Superliga
        "hungarian_nb1",
        "norwegian_eliteserien",
        "swedish_handbollsligan",
        "portuguese_andebol",
        # EHF Lower
        "ehf_cup",
    ],
    "unpopular": [
        # Lower divisions
        "2_bundesliga",
        "polish_1_liga",
        "regional_leagues",
        # Youth
        "u21_championships",
        "youth_leagues",
    ]
}

# === TABLE TENNIS LEAGUES ===
TABLE_TENNIS_LEAGUES: Dict[LeagueType, List[str]] = {
    "popular": [
        # World Tour
        "wtt_grand_smash", "wtt_champions",
        "world_championships", "world_cup", "olympics",
        # Top national
        "china_super_league",
        "bundesliga_tt",  # Germany
        "t_league_japan",
    ],
    "medium": [
        # WTT
        "wtt_contender", "wtt_star_contender",
        # European leagues
        "ettu_champions_league",
        "polish_superliga_tt",
        "french_pro_a",
        "russian_premier",
    ],
    "unpopular": [
        # Lower WTT
        "wtt_feeder",
        # Regional
        "regional_tt",
        "club_championships",
        # Amateur
        "amateur_leagues",
    ]
}

# All leagues by sport
ALL_LEAGUES: Dict[SportType, Dict[LeagueType, List[str]]] = {
    "tennis": TENNIS_LEAGUES,
    "basketball": BASKETBALL_LEAGUES,
    "greyhound": GREYHOUND_LEAGUES,
    "handball": HANDBALL_LEAGUES,
    "table_tennis": TABLE_TENNIS_LEAGUES,
}


def classify_league(league_name: str, sport: str) -> LeagueType:
    """
    Klasyfikuje ligę jako popular/medium/unpopular.

    Args:
        league_name: Nazwa ligi (np. "ATP Masters 1000 Madrid")
        sport: Typ sportu

    Returns:
        LeagueType: "popular", "medium", lub "unpopular"
    """
    league_lower = league_name.lower()

    # Get leagues for sport
    leagues = ALL_LEAGUES.get(sport, TENNIS_LEAGUES)

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


def get_supported_sports() -> List[SportType]:
    """Returns list of all supported sports."""
    return list(ALL_LEAGUES.keys())


def get_leagues_for_sport(sport: SportType) -> Optional[Dict[LeagueType, List[str]]]:
    """Returns league configuration for a specific sport."""
    return ALL_LEAGUES.get(sport)


def get_league_requirements(league_type: LeagueType) -> dict:
    """Zwraca wymagania dla danego typu ligi"""
    from config.thresholds import LEAGUE_REQUIREMENTS
    return LEAGUE_REQUIREMENTS[league_type]
