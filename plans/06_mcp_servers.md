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
from datetime import datetime
import json

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
    mcp.run(transport='stdio')
```

---
