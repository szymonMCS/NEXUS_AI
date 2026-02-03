"""
Analyst Agent - Makes predictions using AI + ML ensemble.
Combines statistical factors with ML predictions and news sentiment.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from config.llm_config import get_llm
from core.state import (
    NexusState, Match, PredictionResult,
    DataQualityLevel, add_message
)
from core.ml import get_prediction_service


class AnalystAgent:
    """
    Analyst uses ensemble of ML models and LLM to predict match outcomes.
    
    Prediction flow:
    1. Get baseline prediction from ML models (ELO, Form, Random Forest, etc.)
    2. Enrich with LLM analysis for context (news, injuries, surface)
    3. Combine into final prediction with confidence score
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = get_llm(model_name=self.model_name, temperature=0.3)
        self.prediction_service = get_prediction_service()

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
            f"Analyzing {len(qualified_matches)} qualified matches using ML + LLM ensemble"
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
            f"Generated {predictions_made} predictions using ensemble models"
        )

        return state

    async def _analyze_match(self, match: Match) -> PredictionResult:
        """
        Analyze a single match using ML + LLM ensemble.

        Args:
            match: Match to analyze

        Returns:
            PredictionResult with ML + LLM combined prediction
        """
        # Step 1: Get ML prediction
        ml_prediction = self._get_ml_prediction(match)
        
        # Step 2: Get LLM analysis
        llm_prediction = await self._get_llm_prediction(match)
        
        # Step 3: Combine predictions
        combined = self._combine_predictions(ml_prediction, llm_prediction, match)
        
        return combined

    def _get_ml_prediction(self, match: Match) -> PredictionResult:
        """
        Get prediction from ML models.
        Uses prediction service which selects best available model.
        """
        # Extract features from match data
        features = self._extract_features(match)
        
        # Get prediction
        result = self.prediction_service.predict(
            sport=match.sport.value,
            home_player=match.home_player.name,
            away_player=match.away_player.name,
            features=features
        )
        
        return result

    def _extract_features(self, match: Match) -> Dict[str, Any]:
        """Extract numerical features from match data."""
        features = {}
        
        # Player stats
        if match.home_player.ranking:
            features['home_rank'] = match.home_player.ranking
        if match.away_player.ranking:
            features['away_rank'] = match.away_player.ranking
        
        if match.home_player.win_rate:
            features['home_win_rate'] = match.home_player.win_rate
        if match.away_player.win_rate:
            features['away_win_rate'] = match.away_player.win_rate
        
        # H2H
        features['h2h_wins'] = match.home_player.h2h_wins
        features['h2h_losses'] = match.home_player.h2h_losses
        
        # News sentiment (count articles)
        if match.news_articles:
            features['news_count'] = len(match.news_articles)
            injury_mentions = sum(1 for a in match.news_articles if a.mentions_injury)
            features['injury_mentions'] = injury_mentions
        
        # Data quality
        if match.data_quality:
            features['data_quality_score'] = match.data_quality.overall_score
        
        return features

    async def _get_llm_prediction(self, match: Match) -> Dict[str, Any]:
        """
        Get prediction from LLM for contextual analysis.
        Uses news, injuries, surface conditions, etc.
        """
        context = self._prepare_analysis_context(match)

        system_prompt = """You are an expert sports analyst AI specializing in betting predictions.
Your task is to analyze the provided match data and return a JSON prediction.

Consider:
1. Player rankings and form
2. Recent news and injury reports
3. Head-to-head history
4. Surface/venue conditions (for tennis)
5. Schedule/rest factors
6. Psychological factors (rivalries, pressure)

Return JSON with:
- home_win_probability: float (0.0-1.0)
- away_win_probability: float (0.0-1.0)
- confidence: float (0.0-1.0)
- factors: dict with key insights
- reasoning: list of key points

Be conservative with probabilities - avoid extreme values unless data strongly supports them.
Return ONLY valid JSON, no explanation outside JSON."""

        user_prompt = f"""Analyze this match and predict the outcome:

{context}

Return JSON with home_win_probability, away_win_probability, confidence, factors, and reasoning."""

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
            
            return {
                "home_win_probability": float(prediction_data.get("home_win_probability", 0.5)),
                "away_win_probability": float(prediction_data.get("away_win_probability", 0.5)),
                "confidence": float(prediction_data.get("confidence", 0.5)),
                "factors": prediction_data.get("factors", {}),
                "reasoning": prediction_data.get("reasoning", [])
            }

        except Exception as e:
            # Fallback to neutral prediction
            return {
                "home_win_probability": 0.5,
                "away_win_probability": 0.5,
                "confidence": 0.0,
                "factors": {},
                "reasoning": [f"LLM analysis failed: {str(e)}"]
            }

    def _combine_predictions(self, 
                            ml_pred: PredictionResult, 
                            llm_pred: Dict[str, Any],
                            match: Match) -> PredictionResult:
        """
        Combine ML and LLM predictions.
        
        Weighting:
        - ML: 60% (data-driven, objective)
        - LLM: 40% (contextual, subjective)
        """
        ml_weight = 0.6
        llm_weight = 0.4
        
        # Weighted average of probabilities
        home_prob = (
            ml_pred.home_win_prob * ml_weight +
            llm_pred["home_win_probability"] * llm_weight
        )
        away_prob = (
            ml_pred.away_win_prob * llm_weight +
            llm_pred["away_win_probability"] * llm_weight
        )
        
        # Normalize to sum to 1
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total
        
        # Combined confidence (weighted by individual confidences)
        ml_conf = ml_pred.confidence
        llm_conf = llm_pred["confidence"]
        combined_confidence = (ml_conf * ml_weight + llm_conf * llm_weight)
        
        # Boost confidence if ML and LLM agree
        ml_favorite = "home" if ml_pred.home_win_prob > ml_pred.away_win_prob else "away"
        llm_favorite = "home" if llm_pred["home_win_probability"] > llm_pred["away_win_probability"] else "away"
        
        if ml_favorite == llm_favorite:
            # Agreement boost
            combined_confidence = min(1.0, combined_confidence * 1.15)
        else:
            # Disagreement penalty
            combined_confidence *= 0.85
        
        # Combine factors
        all_factors = {}
        if ml_pred.factors:
            all_factors.update({f"ml_{k}": v for k, v in ml_pred.factors.items()})
        if llm_pred.get("factors"):
            all_factors.update({f"llm_{k}": v for k, v in llm_pred["factors"].items()})
        
        return PredictionResult(
            home_win_probability=round(home_prob, 3),
            away_win_probability=round(away_prob, 3),
            confidence=round(combined_confidence, 3),
            model_version="2.0-ensemble",
            factors=all_factors
        )

    def _prepare_analysis_context(self, match: Match) -> str:
        """Prepare analysis context string for LLM."""
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
            f"  H2H Wins: {away.h2h_losses}",
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
        
        # Add ML prediction for reference
        ml_pred = self._get_ml_prediction(match)
        context_parts.append("")
        context_parts.append("ML MODEL PREDICTION (for reference):")
        context_parts.append(f"  Home win: {ml_pred.home_win_prob:.1%}")
        context_parts.append(f"  Away win: {ml_pred.away_win_prob:.1%}")
        context_parts.append(f"  Model: {ml_pred.model_name}")
        context_parts.append(f"  Confidence: {ml_pred.confidence:.1%}")

        return "\n".join(context_parts)

    def _fallback_prediction(self, match: Match) -> PredictionResult:
        """
        Generate fallback prediction based on basic statistics.
        Used when ML or LLM fails.
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
            confidence=0.4,
            model_version="v2.0-fallback",
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
