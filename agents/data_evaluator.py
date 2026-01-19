# agents/data_evaluator.py
"""
Data Evaluator Agent - Assesses data quality for each match.
Filters matches based on quality thresholds.
"""

from typing import List, Dict
from datetime import datetime

from langchain_anthropic import ChatAnthropic

from config.settings import settings
from config.thresholds import thresholds, LEAGUE_REQUIREMENTS
from core.state import (
    NexusState, Match, DataQualityMetrics, DataQualityLevel,
    BetDecision, add_message
)
from core.quality_scorer import QualityScorer, score_match_quality


class DataEvaluatorAgent:
    """
    Data Evaluator assesses the quality of collected data for each match.

    Responsibilities:
    - Score news quality (quantity, freshness, relevance)
    - Score odds quality (number of sources, variance)
    - Score stats quality (completeness, historical data)
    - Calculate overall quality score
    - Filter matches below quality threshold
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )
        self.scorer = QualityScorer()

    async def process(self, state: NexusState) -> NexusState:
        """
        Evaluate data quality for all matches.

        Args:
            state: Current workflow state

        Returns:
            Updated state with quality assessments
        """
        state.current_agent = "data_evaluator"
        state = add_message(
            state,
            "data_evaluator",
            f"Evaluating data quality for {len(state.matches)} matches"
        )

        evaluated_matches = []
        high_quality_count = 0
        rejected_count = 0

        for match in state.matches:
            # Determine league type based on league name
            league_type = self._determine_league_type(match.league)

            # Calculate quality scores
            quality = score_match_quality(match, league_type)

            # Create DataQualityMetrics object
            match.data_quality = DataQualityMetrics(
                news_score=quality.news_score,
                odds_score=quality.odds_score,
                stats_score=quality.stats_score,
                overall_score=quality.overall_score,
                quality_level=quality.quality_level,
                issues=quality.issues,
                sources_count=quality.sources_count
            )

            # Check if should reject
            if self._should_reject(quality):
                match.recommended = BetDecision.REJECTED
                match.rejection_reason = self._get_rejection_reason(quality)
                rejected_count += 1
            else:
                if quality.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]:
                    high_quality_count += 1

            evaluated_matches.append(match)

        state.matches = evaluated_matches

        # Summary message
        state = add_message(
            state,
            "data_evaluator",
            f"Evaluation complete: {high_quality_count} high quality, {rejected_count} rejected"
        )

        # Add quality distribution
        quality_dist = self._get_quality_distribution(evaluated_matches)
        state = add_message(
            state,
            "data_evaluator",
            f"Quality distribution: {quality_dist}"
        )

        return state

    def _determine_league_type(self, league: str) -> str:
        """
        Determine league type (popular, medium, unpopular) based on league name.

        Args:
            league: League name

        Returns:
            League type string
        """
        league_lower = league.lower()

        # Popular leagues
        popular = [
            "atp", "wta", "grand slam", "masters",
            "nba", "euroleague", "ncaa",
            "premier league", "la liga", "bundesliga", "serie a"
        ]

        # Medium leagues
        medium = [
            "atp 250", "atp 500", "challenger",
            "euroleague", "acb", "lnb",
        ]

        for p in popular:
            if p in league_lower:
                return "popular"

        for m in medium:
            if m in league_lower:
                return "medium"

        return "unpopular"

    def _should_reject(self, quality) -> bool:
        """
        Determine if match should be rejected based on quality.

        Args:
            quality: QualityScoreComponents

        Returns:
            bool: True if should reject
        """
        # Reject if quality is insufficient
        if quality.quality_level == DataQualityLevel.INSUFFICIENT:
            return True

        # Reject if overall score below threshold
        if quality.overall_score < thresholds.quality_reject:
            return True

        # Reject if critical issues
        critical_issues = ["No news data available", "No odds data available"]
        if any(issue in quality.issues for issue in critical_issues):
            return True

        return False

    def _get_rejection_reason(self, quality) -> str:
        """
        Get human-readable rejection reason.

        Args:
            quality: QualityScoreComponents

        Returns:
            Rejection reason string
        """
        if quality.quality_level == DataQualityLevel.INSUFFICIENT:
            return f"Insufficient data quality ({quality.overall_score:.2%})"

        if quality.issues:
            return f"Quality issues: {', '.join(quality.issues[:3])}"

        return f"Quality below threshold ({quality.overall_score:.2%})"

    def _get_quality_distribution(self, matches: List[Match]) -> Dict[str, int]:
        """
        Get distribution of quality levels.

        Args:
            matches: List of evaluated matches

        Returns:
            Dict mapping quality level to count
        """
        distribution = {}

        for match in matches:
            if match.data_quality:
                level = match.data_quality.quality_level.value
                distribution[level] = distribution.get(level, 0) + 1

        return distribution


# === HELPER FUNCTIONS ===

async def evaluate_match_quality(matches: List[Match]) -> List[Match]:
    """
    Convenience function to evaluate match quality.

    Args:
        matches: List of matches with collected data

    Returns:
        List of matches with quality assessments
    """
    agent = DataEvaluatorAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        matches=matches
    )

    result_state = await agent.process(state)
    return result_state.matches
