# agents/risk_manager.py
"""
Risk Manager Agent - Manages betting risk and calculates optimal stakes.
Implements Kelly Criterion and bankroll management.
"""

from typing import List, Dict, Optional
from datetime import datetime

from config.settings import settings
from config.llm_config import get_llm
from config.thresholds import thresholds
from core.state import (
    NexusState, Match, MatchOdds, ValueBet,
    add_message
)
from data.odds import OddsMerger, get_merged_odds_analysis


class RiskManagerAgent:
    """
    Risk Manager handles betting risk assessment and stake calculation.

    Responsibilities:
    - Fetch best odds for top matches
    - Calculate value bets (expected edge)
    - Apply Kelly Criterion for stake sizing
    - Enforce bankroll management rules
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = get_llm(model_name=self.model_name, temperature=0.1)
        self.merger = OddsMerger()

        # Risk parameters
        self.kelly_fraction = 0.25  # Quarter Kelly (conservative)
        self.max_stake_percent = 0.10  # Max 10% of bankroll per bet
        self.min_edge_required = 0.03  # Minimum 3% edge

    async def process(self, state: NexusState) -> NexusState:
        """
        Assess risk and calculate stakes for top matches.

        Args:
            state: Current workflow state

        Returns:
            Updated state with value bets calculated
        """
        state.current_agent = "risk_manager"

        if not state.top_matches:
            state = add_message(
                state,
                "risk_manager",
                "No top matches to assess"
            )
            return state

        state = add_message(
            state,
            "risk_manager",
            f"Assessing risk for {len(state.top_matches)} top matches"
        )

        value_bets_found = 0

        for match in state.top_matches:
            try:
                value_bet = await self._assess_match_risk(match, state.current_bankroll)
                match.value_bet = value_bet

                if value_bet:
                    value_bets_found += 1
                    state = add_message(
                        state,
                        "risk_manager",
                        f"Value bet found: {match.home_player.name} vs {match.away_player.name} | "
                        f"Bet: {value_bet.bet_on.upper()} @ {value_bet.odds:.2f} | "
                        f"Edge: {value_bet.edge:.1%} | Stake: {value_bet.kelly_stake:.2%}"
                    )

            except Exception as e:
                state = add_message(
                    state,
                    "risk_manager",
                    f"Error assessing {match.home_player.name} vs {match.away_player.name}: {str(e)}"
                )

        state = add_message(
            state,
            "risk_manager",
            f"Found {value_bets_found} value bets"
        )

        return state

    async def _assess_match_risk(
        self,
        match: Match,
        bankroll: float
    ) -> Optional[ValueBet]:
        """
        Assess risk for a single match.

        Args:
            match: Match to assess
            bankroll: Current bankroll

        Returns:
            ValueBet if value exists, None otherwise
        """
        if not match.prediction:
            return None

        home_prob = match.prediction.home_win_probability
        away_prob = match.prediction.away_win_probability

        # Fetch odds
        odds_analysis = await get_merged_odds_analysis(
            sport=match.sport.value,
            home_team=match.home_player.name,
            away_team=match.away_player.name,
            predicted_home_prob=home_prob,
            predicted_away_prob=away_prob,
            bankroll=bankroll
        )

        if not odds_analysis:
            # Use mock odds if no real odds available
            return self._assess_with_mock_odds(match, bankroll)

        # Extract best value bet
        value_bets = odds_analysis.get("value_bets", {})
        recommended = odds_analysis.get("recommended_bet")

        if not recommended:
            return None

        value_info = value_bets.get(recommended)
        if not value_info or not value_info.get("has_value"):
            return None

        best_odds = odds_analysis["best_odds"].get(recommended, {})

        # Get league-specific minimum edge
        min_edge = self._get_min_edge(match.league)

        edge = value_info.get("edge_percentage", 0) / 100
        if edge < min_edge:
            return None

        # Calculate Kelly stake
        if recommended == "home":
            prob = home_prob
        else:
            prob = away_prob

        kelly_stake = self._calculate_kelly_stake(
            prob,
            best_odds.get("odds", 2.0),
            bankroll
        )

        return ValueBet(
            bet_on=recommended,
            odds=best_odds.get("odds", 2.0),
            true_probability=prob,
            edge=edge,
            kelly_stake=kelly_stake / bankroll if bankroll > 0 else 0,
            confidence=match.prediction.confidence
        )

    def _assess_with_mock_odds(
        self,
        match: Match,
        bankroll: float
    ) -> Optional[ValueBet]:
        """
        Assess with mock odds when real odds unavailable.

        Used for testing and when odds APIs are unavailable.

        Args:
            match: Match to assess
            bankroll: Current bankroll

        Returns:
            ValueBet if value exists
        """
        if not match.prediction:
            return None

        home_prob = match.prediction.home_win_probability
        away_prob = match.prediction.away_win_probability

        # Generate mock fair odds
        if home_prob > 0:
            mock_home_odds = 1 / home_prob * 1.05  # 5% margin
        else:
            mock_home_odds = 10.0

        if away_prob > 0:
            mock_away_odds = 1 / away_prob * 1.05
        else:
            mock_away_odds = 10.0

        # Determine best bet
        home_edge = (home_prob * mock_home_odds) - 1
        away_edge = (away_prob * mock_away_odds) - 1

        min_edge = self._get_min_edge(match.league)

        if home_edge > away_edge and home_edge >= min_edge:
            bet_on = "home"
            odds = mock_home_odds
            prob = home_prob
            edge = home_edge
        elif away_edge >= min_edge:
            bet_on = "away"
            odds = mock_away_odds
            prob = away_prob
            edge = away_edge
        else:
            return None

        kelly_stake = self._calculate_kelly_stake(prob, odds, bankroll)

        return ValueBet(
            bet_on=bet_on,
            odds=odds,
            true_probability=prob,
            edge=edge,
            kelly_stake=kelly_stake / bankroll if bankroll > 0 else 0,
            confidence=match.prediction.confidence
        )

    def _get_min_edge(self, league: str) -> float:
        """
        Get minimum edge required for a league.

        Args:
            league: League name

        Returns:
            Minimum edge (0.0 - 1.0)
        """
        league_lower = league.lower()

        # Popular leagues need less edge (more liquidity)
        popular = ["atp 1000", "grand slam", "nba"]
        for p in popular:
            if p in league_lower:
                return thresholds.min_edge_popular_league

        # Medium leagues
        medium = ["atp 500", "atp 250", "euroleague"]
        for m in medium:
            if m in league_lower:
                return thresholds.min_edge_medium_league

        # Unpopular leagues need more edge
        return thresholds.min_edge_unpopular_league

    def _calculate_kelly_stake(
        self,
        probability: float,
        odds: float,
        bankroll: float
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.

        Kelly % = (probability * odds - 1) / (odds - 1)

        Args:
            probability: Win probability
            odds: Decimal odds
            bankroll: Current bankroll

        Returns:
            Recommended stake amount
        """
        if probability <= 0 or odds <= 1:
            return 0

        # Full Kelly
        kelly = (probability * odds - 1) / (odds - 1)

        # Apply fraction
        kelly *= self.kelly_fraction

        # Ensure non-negative and cap at max
        kelly = max(0, min(kelly, self.max_stake_percent))

        return kelly * bankroll


# === HELPER FUNCTIONS ===

async def assess_betting_risk(
    matches: List[Match],
    bankroll: float = 1000.0
) -> List[Match]:
    """
    Convenience function to assess betting risk.

    Args:
        matches: List of top matches
        bankroll: Current bankroll

    Returns:
        List of matches with value bets calculated
    """
    agent = RiskManagerAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        top_matches=matches,
        current_bankroll=bankroll
    )

    result_state = await agent.process(state)
    return result_state.top_matches
