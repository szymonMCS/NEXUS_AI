# agents/ranker.py
"""
Ranker Agent - Ranks matches by betting value potential.
Selects top opportunities based on composite score.
"""

from typing import List, Dict
from datetime import datetime

from langchain_anthropic import ChatAnthropic

from config.settings import settings
from config.thresholds import thresholds
from core.state import (
    NexusState, Match, BetDecision,
    DataQualityLevel, add_message
)


class RankerAgent:
    """
    Ranker scores and ranks matches by betting value potential.

    Ranking factors:
    - Prediction confidence
    - Data quality score
    - Edge potential (predicted prob vs implied prob)
    - League popularity
    - News coverage quality
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )

        # Ranking weights
        self.weights = {
            "confidence": 0.25,
            "data_quality": 0.25,
            "edge_potential": 0.30,
            "league_factor": 0.10,
            "news_coverage": 0.10
        }

    async def process(self, state: NexusState) -> NexusState:
        """
        Rank matches and select top opportunities.

        Args:
            state: Current workflow state

        Returns:
            Updated state with ranked top matches
        """
        state.current_agent = "ranker"

        # Filter matches with predictions
        predicted_matches = [
            m for m in state.matches
            if m.prediction is not None and m.recommended != BetDecision.REJECTED
        ]

        state = add_message(
            state,
            "ranker",
            f"Ranking {len(predicted_matches)} predicted matches"
        )

        # Calculate composite scores
        for match in predicted_matches:
            match.composite_score = self._calculate_composite_score(match)

        # Sort by composite score
        ranked_matches = sorted(
            predicted_matches,
            key=lambda m: m.composite_score,
            reverse=True
        )

        # Select top matches (configurable, default 5)
        max_top_matches = 5
        state.top_matches = ranked_matches[:max_top_matches]

        # Log ranking details
        if state.top_matches:
            state = add_message(
                state,
                "ranker",
                f"Top {len(state.top_matches)} matches selected:"
            )

            for i, match in enumerate(state.top_matches, 1):
                home_prob = match.prediction.home_win_probability
                away_prob = match.prediction.away_win_probability

                fav = "HOME" if home_prob > away_prob else "AWAY"
                prob = max(home_prob, away_prob)

                state = add_message(
                    state,
                    "ranker",
                    f"  {i}. {match.home_player.name} vs {match.away_player.name} | "
                    f"Score: {match.composite_score:.3f} | {fav}: {prob:.1%}"
                )

        return state

    def _calculate_composite_score(self, match: Match) -> float:
        """
        Calculate composite ranking score for a match.

        Args:
            match: Match with prediction

        Returns:
            Composite score (0.0 - 1.0)
        """
        score = 0.0

        # 1. Confidence score
        if match.prediction:
            confidence_score = match.prediction.confidence * self.weights["confidence"]
            score += confidence_score

        # 2. Data quality score
        if match.data_quality:
            quality_score = match.data_quality.overall_score * self.weights["data_quality"]
            score += quality_score

        # 3. Edge potential (based on prediction strength)
        if match.prediction:
            # Higher confidence in one outcome = higher edge potential
            home_prob = match.prediction.home_win_probability
            away_prob = match.prediction.away_win_probability
            max_prob = max(home_prob, away_prob)

            # Edge potential: how far from 50-50
            edge_potential = (max_prob - 0.5) * 2  # Normalize to 0-1
            edge_score = edge_potential * self.weights["edge_potential"]
            score += edge_score

        # 4. League factor
        league_score = self._get_league_factor(match.league) * self.weights["league_factor"]
        score += league_score

        # 5. News coverage
        if match.news_articles:
            coverage = min(len(match.news_articles) / 10, 1.0)  # Cap at 10 articles
            news_score = coverage * self.weights["news_coverage"]
            score += news_score

        return min(score, 1.0)

    def _get_league_factor(self, league: str) -> float:
        """
        Get league quality factor.

        Higher-profile leagues = lower required edge (more efficient market)
        Lower-profile leagues = higher potential edge but need more data

        Args:
            league: League name

        Returns:
            League factor (0.0 - 1.0)
        """
        league_lower = league.lower()

        # Popular leagues - efficient markets, lower edge potential
        popular = ["atp 1000", "grand slam", "wta 1000", "nba", "euroleague"]
        for p in popular:
            if p in league_lower:
                return 0.6

        # Medium leagues - good balance
        medium = ["atp 500", "atp 250", "wta 500", "challenger"]
        for m in medium:
            if m in league_lower:
                return 0.8

        # Less followed leagues - potentially higher edge
        return 0.9

    def get_ranking_criteria(self) -> Dict[str, float]:
        """
        Get the ranking criteria weights.

        Returns:
            Dict of criteria and weights
        """
        return self.weights.copy()


# === HELPER FUNCTIONS ===

async def rank_matches(matches: List[Match], max_results: int = 5) -> List[Match]:
    """
    Convenience function to rank matches.

    Args:
        matches: List of matches with predictions
        max_results: Maximum number of top matches to return

    Returns:
        List of top ranked matches
    """
    agent = RankerAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        matches=matches
    )

    result_state = await agent.process(state)
    return result_state.top_matches[:max_results]
