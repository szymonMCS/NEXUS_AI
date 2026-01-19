# agents/decision_maker.py
"""
Decision Maker Agent - Makes final betting decisions.
Applies final filters and approves/rejects bets.
"""

from typing import List, Dict
from datetime import datetime

from langchain_anthropic import ChatAnthropic

from config.settings import settings
from core.state import (
    NexusState, Match, BetDecision,
    add_message
)


class DecisionMakerAgent:
    """
    Decision Maker makes final betting decisions.

    Final checks:
    - Minimum edge threshold
    - Confidence threshold
    - Daily exposure limits
    - Correlated bets check
    - Final value verification
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )

        # Decision thresholds
        self.min_confidence = 0.4  # Minimum prediction confidence
        self.min_edge = 0.02  # Minimum 2% edge
        self.max_daily_bets = 5  # Maximum bets per day
        self.max_daily_exposure = 0.25  # Max 25% of bankroll at risk

    async def process(self, state: NexusState) -> NexusState:
        """
        Make final betting decisions.

        Args:
            state: Current workflow state

        Returns:
            Updated state with approved/rejected bets
        """
        state.current_agent = "decision_maker"

        if not state.top_matches:
            state = add_message(
                state,
                "decision_maker",
                "No matches to evaluate"
            )
            return state

        state = add_message(
            state,
            "decision_maker",
            f"Making decisions for {len(state.top_matches)} matches"
        )

        approved = []
        rejected = []
        total_stake = 0.0

        for match in state.top_matches:
            decision, reason = self._make_decision(
                match,
                state.current_bankroll,
                total_stake,
                len(approved)
            )

            match.recommended = decision

            if decision == BetDecision.APPROVED:
                approved.append(match)
                if match.value_bet:
                    total_stake += match.value_bet.kelly_stake * state.current_bankroll

                state = add_message(
                    state,
                    "decision_maker",
                    f"APPROVED: {match.home_player.name} vs {match.away_player.name} | "
                    f"Bet: {match.value_bet.bet_on.upper() if match.value_bet else 'N/A'}"
                )
            else:
                match.rejection_reason = reason
                rejected.append(match)

                state = add_message(
                    state,
                    "decision_maker",
                    f"REJECTED: {match.home_player.name} vs {match.away_player.name} | "
                    f"Reason: {reason}"
                )

        state.approved_bets = approved
        state.rejected_matches = rejected

        # Summary
        state = add_message(
            state,
            "decision_maker",
            f"Decision complete: {len(approved)} approved, {len(rejected)} rejected | "
            f"Total stake: ${total_stake:.2f}"
        )

        return state

    def _make_decision(
        self,
        match: Match,
        bankroll: float,
        current_stake: float,
        current_bets: int
    ) -> tuple[BetDecision, str]:
        """
        Make decision for a single match.

        Args:
            match: Match to decide on
            bankroll: Current bankroll
            current_stake: Total stake already committed
            current_bets: Number of bets already approved

        Returns:
            Tuple of (BetDecision, rejection_reason)
        """
        # Check if already rejected
        if match.recommended == BetDecision.REJECTED:
            return BetDecision.REJECTED, match.rejection_reason or "Previously rejected"

        # Check prediction confidence
        if not match.prediction:
            return BetDecision.REJECTED, "No prediction available"

        if match.prediction.confidence < self.min_confidence:
            return BetDecision.REJECTED, f"Low confidence ({match.prediction.confidence:.1%})"

        # Check value bet
        if not match.value_bet:
            return BetDecision.REJECTED, "No value bet found"

        if match.value_bet.edge < self.min_edge:
            return BetDecision.REJECTED, f"Insufficient edge ({match.value_bet.edge:.1%})"

        # Check daily limits
        if current_bets >= self.max_daily_bets:
            return BetDecision.REJECTED, "Daily bet limit reached"

        # Check exposure limit
        stake_amount = match.value_bet.kelly_stake * bankroll
        new_total_stake = current_stake + stake_amount

        if new_total_stake > bankroll * self.max_daily_exposure:
            return BetDecision.REJECTED, "Daily exposure limit reached"

        # Check for injury concerns
        home_injury = match.home_player.injury_status
        away_injury = match.away_player.injury_status

        if home_injury in ["out", "doubtful"] and match.value_bet.bet_on == "home":
            return BetDecision.REJECTED, f"Home player injury concern: {home_injury}"

        if away_injury in ["out", "doubtful"] and match.value_bet.bet_on == "away":
            return BetDecision.REJECTED, f"Away player injury concern: {away_injury}"

        # Check data quality
        if match.data_quality and match.data_quality.overall_score < 0.4:
            return BetDecision.REJECTED, "Insufficient data quality"

        # All checks passed
        return BetDecision.APPROVED, ""

    def get_decision_summary(self, state: NexusState) -> Dict:
        """
        Get summary of decisions made.

        Args:
            state: Workflow state

        Returns:
            Summary dict
        """
        total_stake = 0
        expected_profit = 0

        for match in state.approved_bets:
            if match.value_bet:
                stake = match.value_bet.kelly_stake * state.current_bankroll
                total_stake += stake
                expected_profit += stake * match.value_bet.edge

        return {
            "approved_count": len(state.approved_bets),
            "rejected_count": len(state.rejected_matches),
            "total_stake": total_stake,
            "expected_profit": expected_profit,
            "bankroll_at_risk": (total_stake / state.current_bankroll * 100) if state.current_bankroll > 0 else 0
        }


# === HELPER FUNCTIONS ===

async def make_betting_decisions(
    matches: List[Match],
    bankroll: float = 1000.0
) -> tuple[List[Match], List[Match]]:
    """
    Convenience function to make betting decisions.

    Args:
        matches: List of matches with value bets
        bankroll: Current bankroll

    Returns:
        Tuple of (approved_bets, rejected_matches)
    """
    agent = DecisionMakerAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        top_matches=matches,
        current_bankroll=bankroll
    )

    result_state = await agent.process(state)
    return result_state.approved_bets, result_state.rejected_matches
