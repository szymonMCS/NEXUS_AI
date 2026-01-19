# agents/analyst.py
"""
Analyst Agent - Makes predictions using AI analysis.
Combines statistical factors with news sentiment.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.state import (
    NexusState, Match, PredictionResult,
    DataQualityLevel, add_message
)


class AnalystAgent:
    """
    Analyst uses AI to predict match outcomes.

    Factors considered:
    - Player/team rankings
    - Recent form
    - Head-to-head record
    - Injury information
    - News sentiment
    - Historical performance
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.3  # Slightly higher for prediction creativity
        )

    async def process(self, state: NexusState) -> NexusState:
        """
        Generate predictions for qualified matches.

        Args:
            state: Current workflow state

        Returns:
            Updated state with predictions
        """
        state.current_agent = "analyst"

        # Filter matches with sufficient quality
        qualified_matches = [
            m for m in state.matches
            if m.data_quality and m.data_quality.quality_level in [
                DataQualityLevel.EXCELLENT,
                DataQualityLevel.GOOD,
                DataQualityLevel.MODERATE
            ]
        ]

        state = add_message(
            state,
            "analyst",
            f"Analyzing {len(qualified_matches)} qualified matches"
        )

        predictions_made = 0

        for match in state.matches:
            if match not in qualified_matches:
                continue

            try:
                prediction = await self._analyze_match(match)
                match.prediction = prediction
                predictions_made += 1
            except Exception as e:
                state = add_message(
                    state,
                    "analyst",
                    f"Error analyzing {match.home_player.name} vs {match.away_player.name}: {str(e)}"
                )

        state = add_message(
            state,
            "analyst",
            f"Generated {predictions_made} predictions"
        )

        return state

    async def _analyze_match(self, match: Match) -> PredictionResult:
        """
        Analyze a single match and generate prediction.

        Args:
            match: Match to analyze

        Returns:
            PredictionResult
        """
        # Prepare analysis context
        context = self._prepare_analysis_context(match)

        # Generate prediction using LLM
        system_prompt = """You are an expert sports analyst AI specializing in betting predictions.
Analyze the provided match data and return a JSON prediction with:
- home_win_probability: float (0.0-1.0)
- away_win_probability: float (0.0-1.0)
- confidence: float (0.0-1.0) - how confident you are in this prediction
- factors: dict - key factors and their weights

Consider:
1. Rankings (higher ranked = slight advantage)
2. Recent form (W/L streaks matter)
3. Head-to-head history
4. Injury concerns (major factor if present)
5. News sentiment
6. Surface/venue factors (for tennis)

Be conservative with probabilities - avoid extreme values unless data strongly supports them.
Return ONLY valid JSON, no explanation."""

        user_prompt = f"""Analyze this match and predict the outcome:

{context}

Return JSON with home_win_probability, away_win_probability, confidence, and factors."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # Parse JSON response
            response_text = response.content.strip()

            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            prediction_data = json.loads(response_text)

            # Validate and normalize probabilities
            home_prob = float(prediction_data.get("home_win_probability", 0.5))
            away_prob = float(prediction_data.get("away_win_probability", 0.5))

            # Ensure probabilities sum to ~1.0
            total = home_prob + away_prob
            if total > 0:
                home_prob = home_prob / total
                away_prob = away_prob / total

            return PredictionResult(
                home_win_probability=home_prob,
                away_win_probability=away_prob,
                confidence=float(prediction_data.get("confidence", 0.5)),
                model_version="v1.0",
                factors=prediction_data.get("factors", {})
            )

        except Exception as e:
            # Fallback to basic statistical prediction
            return self._fallback_prediction(match)

    def _prepare_analysis_context(self, match: Match) -> str:
        """
        Prepare analysis context string for LLM.

        Args:
            match: Match to analyze

        Returns:
            Context string
        """
        home = match.home_player
        away = match.away_player

        context_parts = [
            f"MATCH: {home.name} vs {away.name}",
            f"Sport: {match.sport.value}",
            f"League: {match.league}",
            f"Date: {match.date}",
            "",
            "HOME PLAYER:",
            f"  Name: {home.name}",
            f"  Ranking: {home.ranking or 'Unknown'}",
            f"  Form: {home.form or 'Unknown'}",
            f"  Win Rate: {home.win_rate or 'Unknown'}",
            f"  H2H Wins: {home.h2h_wins}",
            f"  Injury Status: {home.injury_status or 'None reported'}",
            "",
            "AWAY PLAYER:",
            f"  Name: {away.name}",
            f"  Ranking: {away.ranking or 'Unknown'}",
            f"  Form: {away.form or 'Unknown'}",
            f"  Win Rate: {away.win_rate or 'Unknown'}",
            f"  H2H Wins: {away.h2h_losses}",  # Opposite perspective
            f"  Injury Status: {away.injury_status or 'None reported'}",
        ]

        # Add news summary
        if match.news_articles:
            context_parts.append("")
            context_parts.append(f"NEWS ARTICLES: {len(match.news_articles)} found")

            injury_articles = [a for a in match.news_articles if a.mentions_injury]
            if injury_articles:
                context_parts.append(f"  Injury-related articles: {len(injury_articles)}")
                for article in injury_articles[:3]:
                    context_parts.append(f"  - {article.title[:100]}")

        # Add data quality
        if match.data_quality:
            context_parts.append("")
            context_parts.append(f"DATA QUALITY: {match.data_quality.quality_level.value}")
            context_parts.append(f"  Overall Score: {match.data_quality.overall_score:.2%}")
            if match.data_quality.issues:
                context_parts.append(f"  Issues: {', '.join(match.data_quality.issues[:3])}")

        return "\n".join(context_parts)

    def _fallback_prediction(self, match: Match) -> PredictionResult:
        """
        Generate fallback prediction based on basic statistics.

        Args:
            match: Match to predict

        Returns:
            PredictionResult
        """
        home = match.home_player
        away = match.away_player

        home_prob = 0.5
        away_prob = 0.5
        factors = {}

        # Ranking factor
        if home.ranking and away.ranking:
            ranking_diff = away.ranking - home.ranking
            ranking_factor = max(-0.2, min(0.2, ranking_diff / 100))
            home_prob += ranking_factor
            factors["ranking"] = ranking_factor

        # H2H factor
        total_h2h = home.h2h_wins + home.h2h_losses
        if total_h2h > 0:
            h2h_factor = (home.h2h_wins - home.h2h_losses) / total_h2h * 0.1
            home_prob += h2h_factor
            factors["h2h"] = h2h_factor

        # Injury factor
        if home.injury_status and home.injury_status in ["out", "doubtful"]:
            home_prob -= 0.15
            factors["home_injury"] = -0.15

        if away.injury_status and away.injury_status in ["out", "doubtful"]:
            home_prob += 0.15
            factors["away_injury"] = 0.15

        # Normalize
        home_prob = max(0.1, min(0.9, home_prob))
        away_prob = 1 - home_prob

        return PredictionResult(
            home_win_probability=home_prob,
            away_win_probability=away_prob,
            confidence=0.4,  # Lower confidence for fallback
            model_version="v1.0-fallback",
            factors=factors
        )


# === HELPER FUNCTIONS ===

async def analyze_matches(matches: List[Match]) -> List[Match]:
    """
    Convenience function to analyze matches.

    Args:
        matches: List of qualified matches

    Returns:
        List of matches with predictions
    """
    agent = AnalystAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        matches=matches
    )

    result_state = await agent.process(state)
    return result_state.matches
