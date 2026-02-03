# data/persistence.py
"""
Data persistence service - bridges data collectors to the database.
Saves fixtures, odds, news, and match results to the DB.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from database.db import get_db_session
from database.models import Match, Odds, News, MatchStats, SportType, OddsType, BetStatus
from database.crud import (
    create_match,
    get_match_by_external_id,
    create_odds,
    create_news,
    create_or_update_match_stats,
    update_match_result,
    settle_bet,
    get_pending_bets,
)

logger = logging.getLogger(__name__)


class DataPersistenceService:
    """
    Saves collected data (fixtures, odds, news) into the database.
    Handles deduplication via external_id lookups.
    """

    # Map string sport names to SportType enum
    SPORT_MAP = {
        "tennis": SportType.TENNIS,
        "basketball": SportType.BASKETBALL,
        "football": SportType.FOOTBALL,
        "soccer": SportType.FOOTBALL,
        "ice_hockey": SportType.ICE_HOCKEY,
        "hockey": SportType.ICE_HOCKEY,
        "baseball": SportType.BASEBALL,
        "esports": SportType.ESPORTS,
    }

    def save_fixtures(self, fixtures: List[Any], sport: str) -> int:
        """
        Save a list of Fixture objects to the database.
        Skips duplicates (by external_id / fixture_id).

        Returns:
            Number of new fixtures saved.
        """
        saved = 0
        sport_enum = self.SPORT_MAP.get(sport, SportType.TENNIS)

        with get_db_session() as db:
            for fix in fixtures:
                # Support both Fixture dataclass and dict
                if hasattr(fix, "fixture_id"):
                    ext_id = fix.fixture_id
                    data = {
                        "external_id": ext_id,
                        "sport": sport_enum,
                        "home_team": fix.home_team,
                        "away_team": fix.away_team,
                        "league": fix.league or "",
                        "country": getattr(fix, "country", "") or "",
                        "start_time": fix.start_time if isinstance(fix.start_time, datetime) else datetime.fromisoformat(str(fix.start_time)),
                        "quality_score": getattr(fix, "data_confidence", 0.5),
                        "data_sources_count": len(getattr(fix, "sources", [])),
                    }
                elif isinstance(fix, dict):
                    ext_id = fix.get("fixture_id") or fix.get("external_id") or fix.get("match_id", "")
                    data = {
                        "external_id": ext_id,
                        "sport": sport_enum,
                        "home_team": fix.get("home_team", ""),
                        "away_team": fix.get("away_team", ""),
                        "league": fix.get("league", ""),
                        "country": fix.get("country", ""),
                        "start_time": datetime.fromisoformat(fix["start_time"]) if isinstance(fix.get("start_time"), str) else fix.get("start_time", datetime.now()),
                        "quality_score": fix.get("data_confidence", 0.5),
                        "data_sources_count": fix.get("data_sources_count", 1),
                    }
                else:
                    continue

                if not data["external_id"] or not data["home_team"]:
                    continue

                # Skip if already exists
                existing = get_match_by_external_id(db, data["external_id"])
                if existing:
                    continue

                try:
                    create_match(db, data)
                    saved += 1
                except Exception as e:
                    logger.warning("Failed to save fixture %s: %s", data["external_id"], e)

        logger.info("Saved %d new fixtures (sport=%s)", saved, sport)
        return saved

    def save_odds(self, match_external_id: str, odds_list: List[Dict[str, Any]]) -> int:
        """
        Save odds for a match.

        Args:
            match_external_id: External match ID.
            odds_list: List of odds dicts with keys:
                bookmaker, home_odds, away_odds, draw_odds, etc.

        Returns:
            Number of odds records saved.
        """
        saved = 0
        with get_db_session() as db:
            match = get_match_by_external_id(db, match_external_id)
            if not match:
                logger.warning("Cannot save odds - match not found: %s", match_external_id)
                return 0

            for odds_data in odds_list:
                try:
                    record = {
                        "match_id": match.id,
                        "bookmaker": odds_data.get("bookmaker", "unknown"),
                        "odds_type": OddsType.MONEYLINE,
                        "home_odds": odds_data.get("home_odds"),
                        "away_odds": odds_data.get("away_odds"),
                        "draw_odds": odds_data.get("draw_odds"),
                        "is_current": True,
                    }

                    # Handicap data
                    if odds_data.get("handicap_line") is not None:
                        record["odds_type"] = OddsType.HANDICAP
                        record["handicap_line"] = odds_data["handicap_line"]
                        record["handicap_home_odds"] = odds_data.get("handicap_home_odds")
                        record["handicap_away_odds"] = odds_data.get("handicap_away_odds")

                    # Totals data
                    if odds_data.get("total_line") is not None:
                        record["odds_type"] = OddsType.TOTALS
                        record["total_line"] = odds_data["total_line"]
                        record["over_odds"] = odds_data.get("over_odds")
                        record["under_odds"] = odds_data.get("under_odds")

                    create_odds(db, record)
                    saved += 1
                except Exception as e:
                    logger.warning("Failed to save odds for %s: %s", match_external_id, e)

        return saved

    def save_news(self, match_external_id: str, articles: List[Dict[str, Any]]) -> int:
        """
        Save news articles linked to a match.

        Returns:
            Number of articles saved.
        """
        saved = 0
        with get_db_session() as db:
            match = get_match_by_external_id(db, match_external_id)
            match_id = match.id if match else None

            for article in articles:
                try:
                    create_news(db, {
                        "match_id": match_id,
                        "title": article.get("title", "")[:500],
                        "url": article.get("url", article.get("link", "")),
                        "source": article.get("source", "unknown"),
                        "snippet": article.get("snippet", article.get("description", "")),
                        "relevance_score": article.get("relevance_score", 0.5),
                        "mentions_injury": article.get("mentions_injury", False),
                        "published_at": (
                            datetime.fromisoformat(article["published_at"])
                            if isinstance(article.get("published_at"), str)
                            else article.get("published_at")
                        ),
                    })
                    saved += 1
                except Exception as e:
                    logger.warning("Failed to save news article: %s", e)

        return saved

    def save_match_stats(self, match_external_id: str, stats: Dict[str, Any]) -> bool:
        """Save match statistics (form, rankings, H2H)."""
        with get_db_session() as db:
            match = get_match_by_external_id(db, match_external_id)
            if not match:
                return False

            try:
                create_or_update_match_stats(db, match.id, stats)
                return True
            except Exception as e:
                logger.warning("Failed to save match stats: %s", e)
                return False

    def update_results(self, results: List[Dict[str, Any]]) -> int:
        """
        Update match results and settle pending bets.

        Args:
            results: List of dicts with keys:
                external_id, home_score, away_score

        Returns:
            Number of matches updated.
        """
        updated = 0
        with get_db_session() as db:
            for result in results:
                ext_id = result.get("external_id", "")
                home_score = result.get("home_score")
                away_score = result.get("away_score")

                if not ext_id or home_score is None or away_score is None:
                    continue

                match = get_match_by_external_id(db, ext_id)
                if not match or match.is_finished:
                    continue

                # Determine winner
                if home_score > away_score:
                    winner = "home"
                elif away_score > home_score:
                    winner = "away"
                else:
                    winner = "draw"

                update_match_result(db, match.id, home_score, away_score, winner)

                # Settle related bets
                for bet in match.bets:
                    if bet.status.value == BetStatus.PENDING.value:
                        if bet.selection == winner:
                            profit = bet.stake * (bet.odds - 1)
                            settle_bet(db, bet.id, BetStatus.WON, profit)
                        else:
                            settle_bet(db, bet.id, BetStatus.LOST, -bet.stake)

                updated += 1

        logger.info("Updated %d match results", updated)
        return updated

    def get_unfinished_match_ids(self) -> List[str]:
        """Get external IDs of matches that haven't finished yet."""
        with get_db_session() as db:
            matches = db.query(Match).filter(
                Match.is_finished == False,
                Match.start_time < datetime.now(),
            ).all()
            return [m.external_id for m in matches]
