# data/odds/odds_merger.py
"""
Odds merger - combines odds from multiple sources and finds best value.
Implements Kelly Criterion for optimal stake sizing.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import statistics

from config.settings import settings
from data.odds.odds_api_client import OddsAPIClient, get_odds_for_match as get_api_odds
from data.odds.pl_scraper import PolishBookmakerScraper, find_match_odds as get_scraped_odds


class OddsMerger:
    """
    Merges odds from multiple sources and provides analytics.

    Features:
    - Combine odds from The Odds API and web scrapers
    - Find best odds across bookmakers
    - Calculate implied probability
    - Detect arbitrage opportunities
    - Calculate value bets
    - Kelly Criterion stake sizing
    """

    def __init__(self):
        self.odds_api_client = None
        self.scraper = None

    async def get_all_odds(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get odds from all available sources.

        Args:
            sport: Sport name
            home_team: Home team/player name
            away_team: Away team/player name
            match_date: Match date (optional)

        Returns:
            Combined list of odds from all sources
        """
        all_odds = []

        # Fetch from The Odds API (if API key available)
        if settings.ODDS_API_KEY and settings.is_pro_mode:
            try:
                api_odds = await get_api_odds(sport, home_team, away_team, match_date)
                all_odds.extend(api_odds)
            except Exception as e:
                print(f"Error fetching API odds: {e}")

        # Fetch from Polish scrapers (Lite mode or as backup)
        if settings.is_lite_mode or settings.USE_WEB_SCRAPING:
            try:
                scraped_odds = await get_scraped_odds(sport, home_team, away_team)
                all_odds.extend(scraped_odds)
            except Exception as e:
                print(f"Error scraping odds: {e}")

        return all_odds

    def find_best_odds(
        self,
        odds_list: List[Dict],
        selection: str = "home"
    ) -> Optional[Dict]:
        """
        Find best odds for a selection across all bookmakers.

        Args:
            odds_list: List of odds dicts
            selection: "home" or "away"

        Returns:
            Dict with best odds info or None
        """
        if not odds_list:
            return None

        best = {
            "bookmaker": None,
            "odds": 0.0,
            "odds_type": None,
        }

        odds_field = f"{selection}_odds"

        for odds in odds_list:
            if odds.get("odds_type") == "moneyline":
                current_odds = odds.get(odds_field, 0)
                if current_odds > best["odds"]:
                    best = {
                        "bookmaker": odds.get("bookmaker"),
                        "odds": current_odds,
                        "odds_type": odds.get("odds_type"),
                    }

        return best if best["bookmaker"] else None

    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calculate implied probability from decimal odds.

        Args:
            odds: Decimal odds (e.g., 2.50)

        Returns:
            Implied probability (0.0 - 1.0)
        """
        if odds <= 1.0:
            return 0.0

        return 1.0 / odds

    def calculate_bookmaker_margin(self, home_odds: float, away_odds: float) -> float:
        """
        Calculate bookmaker's margin (overround).

        Args:
            home_odds: Home odds
            away_odds: Away odds

        Returns:
            Margin percentage (e.g., 0.05 = 5%)
        """
        home_prob = self.calculate_implied_probability(home_odds)
        away_prob = self.calculate_implied_probability(away_odds)

        total_prob = home_prob + away_prob

        # Margin = total probability - 1.0
        return total_prob - 1.0

    def get_odds_statistics(self, odds_list: List[Dict]) -> Dict:
        """
        Calculate statistics for odds from multiple bookmakers.

        Args:
            odds_list: List of odds dicts

        Returns:
            Dict with statistics (avg, min, max, variance)
        """
        home_odds_values = []
        away_odds_values = []

        for odds in odds_list:
            if odds.get("odds_type") == "moneyline":
                if odds.get("home_odds"):
                    home_odds_values.append(odds["home_odds"])
                if odds.get("away_odds"):
                    away_odds_values.append(odds["away_odds"])

        stats = {
            "home": self._calculate_stats(home_odds_values),
            "away": self._calculate_stats(away_odds_values),
            "num_bookmakers": len(odds_list),
        }

        return stats

    def _calculate_stats(self, values: List[float]) -> Dict:
        """Calculate statistics for a list of values"""
        if not values:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "variance": 0.0,
            }

        return {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        }

    def detect_arbitrage(self, odds_list: List[Dict]) -> Optional[Dict]:
        """
        Detect arbitrage opportunities (guaranteed profit).

        An arbitrage exists when:
        1/best_home_odds + 1/best_away_odds < 1.0

        Args:
            odds_list: List of odds dicts

        Returns:
            Dict with arbitrage info or None
        """
        best_home = self.find_best_odds(odds_list, "home")
        best_away = self.find_best_odds(odds_list, "away")

        if not best_home or not best_away:
            return None

        home_odds = best_home["odds"]
        away_odds = best_away["odds"]

        # Calculate arbitrage percentage
        arb_percentage = (1 / home_odds) + (1 / away_odds)

        if arb_percentage < 1.0:
            # Arbitrage exists!
            profit_percentage = (1 / arb_percentage - 1) * 100

            return {
                "exists": True,
                "profit_percentage": profit_percentage,
                "home_bookmaker": best_home["bookmaker"],
                "home_odds": home_odds,
                "away_bookmaker": best_away["bookmaker"],
                "away_odds": away_odds,
                # Stake distribution for $100 total stake
                "home_stake": (100 / home_odds) / arb_percentage,
                "away_stake": (100 / away_odds) / arb_percentage,
            }

        return None

    def calculate_value_bet(
        self,
        predicted_probability: float,
        best_odds: float,
        min_edge: float = 0.03
    ) -> Optional[Dict]:
        """
        Calculate if a bet offers value (expected value > 0).

        Value exists when:
        predicted_probability * odds > 1.0

        Args:
            predicted_probability: AI predicted probability (0.0 - 1.0)
            best_odds: Best available odds
            min_edge: Minimum edge required (default 3%)

        Returns:
            Dict with value bet info or None
        """
        if predicted_probability <= 0 or best_odds <= 1.0:
            return None

        # Calculate expected value
        expected_value = (predicted_probability * best_odds) - 1.0

        # Edge percentage
        edge = expected_value * 100

        if edge >= min_edge * 100:
            return {
                "has_value": True,
                "edge_percentage": edge,
                "expected_value": expected_value,
                "predicted_probability": predicted_probability,
                "best_odds": best_odds,
                "implied_probability": self.calculate_implied_probability(best_odds),
            }

        return None

    def kelly_criterion(
        self,
        probability: float,
        odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.

        Kelly % = (probability * odds - 1) / (odds - 1)

        Args:
            probability: Win probability (0.0 - 1.0)
            odds: Decimal odds
            bankroll: Current bankroll
            kelly_fraction: Fractional Kelly (default 0.25 = quarter Kelly)

        Returns:
            Recommended stake amount
        """
        if probability <= 0 or odds <= 1.0:
            return 0.0

        # Kelly formula
        kelly_percentage = (probability * odds - 1) / (odds - 1)

        # Apply fractional Kelly for risk management
        kelly_percentage *= kelly_fraction

        # Ensure non-negative
        kelly_percentage = max(0, kelly_percentage)

        # Cap at reasonable maximum (e.g., 10% of bankroll)
        kelly_percentage = min(kelly_percentage, 0.10)

        stake = bankroll * kelly_percentage

        return round(stake, 2)

    def merge_and_analyze(
        self,
        odds_list: List[Dict],
        predicted_home_prob: float,
        predicted_away_prob: float,
        bankroll: float = 1000.0
    ) -> Dict:
        """
        Comprehensive odds analysis.

        Args:
            odds_list: List of odds from all sources
            predicted_home_prob: AI predicted home win probability
            predicted_away_prob: AI predicted away win probability
            bankroll: Current bankroll

        Returns:
            Dict with complete analysis
        """
        # Find best odds
        best_home = self.find_best_odds(odds_list, "home")
        best_away = self.find_best_odds(odds_list, "away")

        # Statistics
        stats = self.get_odds_statistics(odds_list)

        # Arbitrage detection
        arbitrage = self.detect_arbitrage(odds_list)

        # Value bets
        value_home = None
        value_away = None
        recommended_bet = None

        if best_home:
            value_home = self.calculate_value_bet(
                predicted_home_prob,
                best_home["odds"]
            )
            if value_home and value_home["has_value"]:
                kelly_stake = self.kelly_criterion(
                    predicted_home_prob,
                    best_home["odds"],
                    bankroll
                )
                value_home["kelly_stake"] = kelly_stake
                recommended_bet = "home"

        if best_away:
            value_away = self.calculate_value_bet(
                predicted_away_prob,
                best_away["odds"]
            )
            if value_away and value_away["has_value"]:
                kelly_stake = self.kelly_criterion(
                    predicted_away_prob,
                    best_away["odds"],
                    bankroll
                )
                value_away["kelly_stake"] = kelly_stake

                # If both have value, pick higher edge
                if not recommended_bet or (value_away["edge_percentage"] > value_home["edge_percentage"]):
                    recommended_bet = "away"

        return {
            "best_odds": {
                "home": best_home,
                "away": best_away,
            },
            "statistics": stats,
            "arbitrage": arbitrage,
            "value_bets": {
                "home": value_home,
                "away": value_away,
            },
            "recommended_bet": recommended_bet,
            "num_sources": len(odds_list),
        }


# === HELPER FUNCTIONS ===

async def get_merged_odds_analysis(
    sport: str,
    home_team: str,
    away_team: str,
    predicted_home_prob: float,
    predicted_away_prob: float,
    bankroll: float = 1000.0
) -> Dict:
    """
    Convenience function for complete odds analysis.

    Args:
        sport: Sport name
        home_team: Home team/player
        away_team: Away team/player
        predicted_home_prob: Predicted home win probability
        predicted_away_prob: Predicted away win probability
        bankroll: Current bankroll

    Returns:
        Complete odds analysis dict
    """
    merger = OddsMerger()

    # Fetch all odds
    odds_list = await merger.get_all_odds(sport, home_team, away_team)

    # Analyze
    analysis = merger.merge_and_analyze(
        odds_list,
        predicted_home_prob,
        predicted_away_prob,
        bankroll
    )

    return analysis
