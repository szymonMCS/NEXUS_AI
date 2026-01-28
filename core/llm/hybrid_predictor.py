# core/llm/hybrid_predictor.py
"""
Hybrid Prediction System: ML Models + Kimi Reasoning.

Checkpoint: 7.4

Combines local ML models with Kimi LLM for optimal predictions:
- ML models provide statistical predictions (60% weight)
- Kimi provides contextual reasoning (40% weight)
- Ensemble produces final prediction with confidence

Architecture:
┌─────────────────┐    ┌─────────────────┐
│   ML Models     │    │  Kimi Analysis  │
│  (60% weight)   │    │  (40% weight)   │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────┐
         │  Hybrid Ensemble │
         │   Integration    │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Final Prediction │
         │ + Recommendation │
         └──────────────────┘
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.llm.kimi_client import KimiClient, KimiResponse
from core.llm.match_analyzer import MatchAnalyzer, MatchAnalysis
from core.llm.injury_extractor import InjuryExtractor, TeamAvailability

logger = logging.getLogger(__name__)


class PredictionSource(str, Enum):
    """Source of prediction component."""
    ML_GOALS = "ml_goals"
    ML_HANDICAP = "ml_handicap"
    KIMI_ANALYSIS = "kimi_analysis"
    HYBRID = "hybrid"


class RecommendationType(str, Enum):
    """Types of betting recommendations."""
    STRONG_BET = "strong_bet"        # High confidence, recommended
    MODERATE_BET = "moderate_bet"    # Medium confidence, consider
    VALUE_BET = "value_bet"          # Edge vs odds, risky but value
    AVOID = "avoid"                  # Low confidence or high risk
    NO_RECOMMENDATION = "no_recommendation"


@dataclass
class MLPrediction:
    """ML model prediction."""
    source: PredictionSource
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float
    confidence: float
    features_used: List[str] = field(default_factory=list)

    @property
    def expected_total_goals(self) -> float:
        return self.expected_home_goals + self.expected_away_goals

    @property
    def predicted_outcome(self) -> str:
        probs = {
            "home": self.home_win_prob,
            "draw": self.draw_prob,
            "away": self.away_win_prob,
        }
        return max(probs, key=probs.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "home_win_prob": self.home_win_prob,
            "draw_prob": self.draw_prob,
            "away_win_prob": self.away_win_prob,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "expected_total_goals": self.expected_total_goals,
            "over_25_prob": self.over_25_prob,
            "confidence": self.confidence,
            "predicted_outcome": self.predicted_outcome,
        }


@dataclass
class HybridPrediction:
    """Combined prediction from ML and Kimi."""
    match_id: Optional[str]
    home_team: str
    away_team: str
    league: str

    # Probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Goals
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float
    btts_prob: float

    # Confidence
    confidence: float
    ml_confidence: float
    kimi_confidence: float

    # Components
    ml_prediction: Optional[MLPrediction] = None
    kimi_analysis: Optional[MatchAnalysis] = None

    # Recommendation
    recommendation: RecommendationType = RecommendationType.NO_RECOMMENDATION
    recommended_bets: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""

    # Weights used
    ml_weight: float = 0.6
    kimi_weight: float = 0.4

    # Meta
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def expected_total_goals(self) -> float:
        return self.expected_home_goals + self.expected_away_goals

    @property
    def predicted_outcome(self) -> str:
        probs = {
            "home": self.home_win_prob,
            "draw": self.draw_prob,
            "away": self.away_win_prob,
        }
        return max(probs, key=probs.get)

    @property
    def max_outcome_prob(self) -> float:
        return max(self.home_win_prob, self.draw_prob, self.away_win_prob)

    def get_value_bets(self, odds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Find value bets by comparing prediction to market odds.

        Args:
            odds: Market odds {"home": 2.0, "draw": 3.5, "away": 3.0}

        Returns:
            List of value bet opportunities
        """
        value_bets = []

        comparisons = [
            ("home", self.home_win_prob, odds.get("home", 0)),
            ("draw", self.draw_prob, odds.get("draw", 0)),
            ("away", self.away_win_prob, odds.get("away", 0)),
        ]

        for outcome, prob, market_odds in comparisons:
            if market_odds > 0:
                implied_prob = 1 / market_odds
                edge = prob - implied_prob

                if edge > 0.03:  # 3% edge threshold
                    value_bets.append({
                        "market": "1X2",
                        "selection": outcome,
                        "predicted_prob": prob,
                        "implied_prob": implied_prob,
                        "edge": edge,
                        "odds": market_odds,
                        "kelly_fraction": edge / (market_odds - 1) if market_odds > 1 else 0,
                    })

        # Over/Under 2.5
        over_odds = odds.get("over_25", 0)
        under_odds = odds.get("under_25", 0)

        if over_odds > 0:
            implied = 1 / over_odds
            edge = self.over_25_prob - implied
            if edge > 0.03:
                value_bets.append({
                    "market": "over_under",
                    "selection": "over_2.5",
                    "predicted_prob": self.over_25_prob,
                    "implied_prob": implied,
                    "edge": edge,
                    "odds": over_odds,
                })

        if under_odds > 0:
            implied = 1 / under_odds
            under_prob = 1 - self.over_25_prob
            edge = under_prob - implied
            if edge > 0.03:
                value_bets.append({
                    "market": "over_under",
                    "selection": "under_2.5",
                    "predicted_prob": under_prob,
                    "implied_prob": implied,
                    "edge": edge,
                    "odds": under_odds,
                })

        return sorted(value_bets, key=lambda x: x["edge"], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "probabilities": {
                "home_win": self.home_win_prob,
                "draw": self.draw_prob,
                "away_win": self.away_win_prob,
            },
            "goals": {
                "expected_home": self.expected_home_goals,
                "expected_away": self.expected_away_goals,
                "expected_total": self.expected_total_goals,
                "over_25_prob": self.over_25_prob,
                "btts_prob": self.btts_prob,
            },
            "predicted_outcome": self.predicted_outcome,
            "confidence": {
                "overall": self.confidence,
                "ml": self.ml_confidence,
                "kimi": self.kimi_confidence,
            },
            "weights": {
                "ml": self.ml_weight,
                "kimi": self.kimi_weight,
            },
            "recommendation": self.recommendation.value,
            "recommended_bets": self.recommended_bets,
            "reasoning": self.reasoning,
            "ml_prediction": self.ml_prediction.to_dict() if self.ml_prediction else None,
            "kimi_summary": self.kimi_analysis.summary if self.kimi_analysis else None,
            "created_at": self.created_at.isoformat(),
        }


class HybridPredictor:
    """
    Combines ML models with Kimi LLM for hybrid predictions.

    The ensemble approach:
    1. ML models predict based on historical patterns
    2. Kimi analyzes contextual factors (injuries, news, tactics)
    3. Weighted combination produces final prediction
    4. Generate betting recommendations

    Usage:
        predictor = HybridPredictor()
        prediction = await predictor.predict(
            home_team="Arsenal",
            away_team="Chelsea",
            league="Premier League",
            features=feature_dict,
            context=context_dict,
        )
    """

    DEFAULT_ML_WEIGHT = 0.6
    DEFAULT_KIMI_WEIGHT = 0.4

    def __init__(
        self,
        ml_weight: float = DEFAULT_ML_WEIGHT,
        kimi_weight: float = DEFAULT_KIMI_WEIGHT,
        goals_model=None,
        handicap_model=None,
        kimi_client: Optional[KimiClient] = None,
    ):
        """
        Initialize hybrid predictor.

        Args:
            ml_weight: Weight for ML predictions (0-1)
            kimi_weight: Weight for Kimi analysis (0-1)
            goals_model: Pre-loaded GoalsModel (optional)
            handicap_model: Pre-loaded HandicapModel (optional)
            kimi_client: Pre-configured KimiClient (optional)
        """
        # Normalize weights
        total = ml_weight + kimi_weight
        self.ml_weight = ml_weight / total
        self.kimi_weight = kimi_weight / total

        self._goals_model = goals_model
        self._handicap_model = handicap_model
        self._kimi_client = kimi_client
        self._match_analyzer = MatchAnalyzer(kimi_client)
        self._injury_extractor = InjuryExtractor(kimi_client)

        self._models_loaded = False

    async def _load_models(self):
        """Load ML models if not already loaded."""
        if self._models_loaded:
            return

        try:
            from core.ml.models.goals_model import GoalsModel
            from core.ml.models.handicap_model import HandicapModel

            if self._goals_model is None:
                self._goals_model = GoalsModel()
                self._goals_model.load("models/goals_model")

            if self._handicap_model is None:
                self._handicap_model = HandicapModel()
                self._handicap_model.load("models/handicap_model")

            self._models_loaded = True
            logger.info("ML models loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load ML models: {e}")
            # Continue without ML models - will use Kimi only

    async def predict(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_id: Optional[str] = None,
        features: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        odds: Optional[Dict[str, float]] = None,
        use_kimi: bool = True,
    ) -> HybridPrediction:
        """
        Generate hybrid prediction for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League/competition name
            match_id: Optional match identifier
            features: Feature dictionary for ML models
            context: Context for Kimi analysis (form, injuries, news)
            odds: Market odds for value calculation
            use_kimi: Whether to include Kimi analysis

        Returns:
            HybridPrediction with combined results
        """
        await self._load_models()

        # Get ML prediction
        ml_prediction = await self._get_ml_prediction(features)

        # Get Kimi analysis
        kimi_analysis = None
        if use_kimi:
            kimi_analysis = await self._get_kimi_analysis(
                home_team=home_team,
                away_team=away_team,
                league=league,
                context=context,
            )

        # Combine predictions
        hybrid = self._combine_predictions(
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_id=match_id,
            ml_prediction=ml_prediction,
            kimi_analysis=kimi_analysis,
        )

        # Generate recommendations
        hybrid = self._generate_recommendations(hybrid, odds)

        return hybrid

    async def _get_ml_prediction(
        self,
        features: Optional[Dict[str, float]],
    ) -> Optional[MLPrediction]:
        """Get prediction from ML models."""
        if features is None or not self._models_loaded:
            return None

        try:
            # Convert features to list
            feature_list = list(features.values())

            # Goals prediction
            goals_pred = None
            if self._goals_model and self._goals_model.is_trained:
                goals_pred = self._goals_model.predict(feature_list)

            # Handicap prediction
            handicap_pred = None
            if self._handicap_model and self._handicap_model.is_trained:
                handicap_pred = self._handicap_model.predict(feature_list)

            if not goals_pred and not handicap_pred:
                return None

            # Combine into MLPrediction
            home_goals = goals_pred.expected_home_goals if goals_pred else 1.3
            away_goals = goals_pred.expected_away_goals if goals_pred else 1.1
            over_prob = goals_pred.over_probability if goals_pred else 0.5

            # Estimate outcome probs from goals (Poisson-based)
            probs = self._goals_to_probs(home_goals, away_goals)

            # Adjust with handicap model if available
            if handicap_pred:
                spread_adj = handicap_pred.predicted_spread / 10  # Scale adjustment
                probs["home"] = min(0.95, max(0.05, probs["home"] + spread_adj * 0.1))
                probs["away"] = min(0.95, max(0.05, probs["away"] - spread_adj * 0.1))
                probs["draw"] = 1 - probs["home"] - probs["away"]
                probs["draw"] = max(0.05, probs["draw"])

                # Renormalize
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}

            confidence = 0.7  # Default ML confidence
            if goals_pred:
                confidence = min(0.9, goals_pred.confidence * 1.1)

            return MLPrediction(
                source=PredictionSource.ML_GOALS,
                home_win_prob=probs["home"],
                draw_prob=probs["draw"],
                away_win_prob=probs["away"],
                expected_home_goals=home_goals,
                expected_away_goals=away_goals,
                over_25_prob=over_prob,
                confidence=confidence,
                features_used=list(features.keys()),
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    def _goals_to_probs(
        self,
        home_goals: float,
        away_goals: float,
    ) -> Dict[str, float]:
        """Convert expected goals to outcome probabilities using Poisson."""
        import math

        def poisson_prob(k: int, lambda_: float) -> float:
            return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)

        max_goals = 10
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for h in range(max_goals):
            for a in range(max_goals):
                prob = poisson_prob(h, home_goals) * poisson_prob(a, away_goals)
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob

        # Normalize
        total = home_win + draw + away_win
        return {
            "home": home_win / total,
            "draw": draw / total,
            "away": away_win / total,
        }

    async def _get_kimi_analysis(
        self,
        home_team: str,
        away_team: str,
        league: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[MatchAnalysis]:
        """Get analysis from Kimi."""
        try:
            return await self._match_analyzer.analyze_match(
                home_team=home_team,
                away_team=away_team,
                league=league,
                context=context,
            )
        except Exception as e:
            logger.error(f"Kimi analysis failed: {e}")
            return None

    def _combine_predictions(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_id: Optional[str],
        ml_prediction: Optional[MLPrediction],
        kimi_analysis: Optional[MatchAnalysis],
    ) -> HybridPrediction:
        """Combine ML and Kimi predictions."""
        # Calculate effective weights based on availability
        ml_available = ml_prediction is not None
        kimi_available = kimi_analysis is not None

        if ml_available and kimi_available:
            eff_ml_weight = self.ml_weight
            eff_kimi_weight = self.kimi_weight
        elif ml_available:
            eff_ml_weight = 1.0
            eff_kimi_weight = 0.0
        elif kimi_available:
            eff_ml_weight = 0.0
            eff_kimi_weight = 1.0
        else:
            # No predictions available - return defaults
            return HybridPrediction(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                league=league,
                home_win_prob=0.35,
                draw_prob=0.30,
                away_win_prob=0.35,
                expected_home_goals=1.3,
                expected_away_goals=1.2,
                over_25_prob=0.5,
                btts_prob=0.5,
                confidence=0.3,
                ml_confidence=0.0,
                kimi_confidence=0.0,
            )

        # ML values (with defaults)
        ml_home = ml_prediction.home_win_prob if ml_prediction else 0.35
        ml_draw = ml_prediction.draw_prob if ml_prediction else 0.30
        ml_away = ml_prediction.away_win_prob if ml_prediction else 0.35
        ml_home_goals = ml_prediction.expected_home_goals if ml_prediction else 1.3
        ml_away_goals = ml_prediction.expected_away_goals if ml_prediction else 1.2
        ml_over = ml_prediction.over_25_prob if ml_prediction else 0.5
        ml_conf = ml_prediction.confidence if ml_prediction else 0.0

        # Kimi values (with defaults)
        kimi_home = 0.35
        kimi_draw = 0.30
        kimi_away = 0.35
        kimi_goals = 2.5
        kimi_conf = 0.0

        if kimi_analysis:
            kimi_conf = kimi_analysis.outcome_confidence

            outcome = kimi_analysis.predicted_outcome.lower()
            # Convert outcome to probabilities
            if outcome == "home":
                base_prob = min(0.9, kimi_conf + 0.2)
                kimi_home = base_prob
                kimi_draw = (1 - base_prob) * 0.4
                kimi_away = (1 - base_prob) * 0.6
            elif outcome == "away":
                base_prob = min(0.9, kimi_conf + 0.2)
                kimi_away = base_prob
                kimi_draw = (1 - base_prob) * 0.4
                kimi_home = (1 - base_prob) * 0.6
            else:  # draw
                base_prob = min(0.7, kimi_conf + 0.1)
                kimi_draw = base_prob
                kimi_home = (1 - base_prob) * 0.5
                kimi_away = (1 - base_prob) * 0.5

            kimi_goals = kimi_analysis.predicted_goals

        # Weighted combination
        home_prob = eff_ml_weight * ml_home + eff_kimi_weight * kimi_home
        draw_prob = eff_ml_weight * ml_draw + eff_kimi_weight * kimi_draw
        away_prob = eff_ml_weight * ml_away + eff_kimi_weight * kimi_away

        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        # Goals
        expected_total = eff_ml_weight * (ml_home_goals + ml_away_goals) + eff_kimi_weight * kimi_goals
        home_ratio = ml_home_goals / (ml_home_goals + ml_away_goals) if ml_prediction else 0.52
        expected_home = expected_total * home_ratio
        expected_away = expected_total * (1 - home_ratio)

        # Over/Under
        over_prob = eff_ml_weight * ml_over + eff_kimi_weight * (0.6 if kimi_goals > 2.5 else 0.4)

        # BTTS (both teams to score) - estimate from goals
        btts_prob = self._estimate_btts_prob(expected_home, expected_away)

        # Combined confidence
        confidence = eff_ml_weight * ml_conf + eff_kimi_weight * kimi_conf

        return HybridPrediction(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            home_win_prob=home_prob,
            draw_prob=draw_prob,
            away_win_prob=away_prob,
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            over_25_prob=over_prob,
            btts_prob=btts_prob,
            confidence=confidence,
            ml_confidence=ml_conf,
            kimi_confidence=kimi_conf,
            ml_prediction=ml_prediction,
            kimi_analysis=kimi_analysis,
            ml_weight=eff_ml_weight,
            kimi_weight=eff_kimi_weight,
        )

    def _estimate_btts_prob(
        self,
        home_goals: float,
        away_goals: float,
    ) -> float:
        """Estimate BTTS probability from expected goals."""
        import math

        # P(home scores) = 1 - P(home=0)
        p_home_scores = 1 - math.exp(-home_goals)
        p_away_scores = 1 - math.exp(-away_goals)

        # P(BTTS) = P(home scores) * P(away scores)
        return p_home_scores * p_away_scores

    def _generate_recommendations(
        self,
        prediction: HybridPrediction,
        odds: Optional[Dict[str, float]],
    ) -> HybridPrediction:
        """Generate betting recommendations."""
        recommended_bets = []
        reasoning_parts = []

        # Determine recommendation type based on confidence
        if prediction.confidence >= 0.7:
            rec_type = RecommendationType.STRONG_BET
            reasoning_parts.append("High confidence prediction")
        elif prediction.confidence >= 0.55:
            rec_type = RecommendationType.MODERATE_BET
            reasoning_parts.append("Moderate confidence - consider carefully")
        else:
            rec_type = RecommendationType.AVOID
            reasoning_parts.append("Low confidence - avoid or small stake only")

        # Main outcome bet
        outcome = prediction.predicted_outcome
        prob = prediction.max_outcome_prob

        if prob >= 0.45:
            bet = {
                "market": "1X2",
                "selection": outcome,
                "probability": prob,
                "confidence": prediction.confidence,
            }

            if odds and outcome in odds:
                market_odds = odds[outcome]
                implied = 1 / market_odds
                edge = prob - implied
                bet["odds"] = market_odds
                bet["edge"] = edge
                bet["is_value"] = edge > 0.03

                if edge > 0.05:
                    reasoning_parts.append(f"Value edge of {edge:.1%} on {outcome}")

            recommended_bets.append(bet)

        # Over/Under bet
        if prediction.over_25_prob >= 0.6:
            bet = {
                "market": "over_under",
                "selection": "over_2.5",
                "probability": prediction.over_25_prob,
            }
            if odds and "over_25" in odds:
                bet["odds"] = odds["over_25"]
            recommended_bets.append(bet)
            reasoning_parts.append(f"Expected goals: {prediction.expected_total_goals:.1f}")
        elif prediction.over_25_prob <= 0.4:
            bet = {
                "market": "over_under",
                "selection": "under_2.5",
                "probability": 1 - prediction.over_25_prob,
            }
            if odds and "under_25" in odds:
                bet["odds"] = odds["under_25"]
            recommended_bets.append(bet)

        # Add Kimi insights to reasoning
        if prediction.kimi_analysis and prediction.kimi_analysis.summary:
            reasoning_parts.append(prediction.kimi_analysis.summary)

        prediction.recommendation = rec_type
        prediction.recommended_bets = recommended_bets
        prediction.reasoning = " | ".join(reasoning_parts)

        return prediction

    async def batch_predict(
        self,
        matches: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[HybridPrediction]:
        """
        Generate predictions for multiple matches.

        Args:
            matches: List of match dicts with required fields
            max_concurrent: Maximum concurrent predictions

        Returns:
            List of HybridPrediction results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def predict_one(match: Dict[str, Any]) -> HybridPrediction:
            async with semaphore:
                return await self.predict(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    league=match.get("league", "Unknown"),
                    match_id=match.get("match_id"),
                    features=match.get("features"),
                    context=match.get("context"),
                    odds=match.get("odds"),
                )

        results = await asyncio.gather(
            *[predict_one(m) for m in matches],
            return_exceptions=True,
        )

        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Prediction failed for match {i}: {result}")
                # Create fallback
                match = matches[i]
                predictions.append(HybridPrediction(
                    match_id=match.get("match_id"),
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    league=match.get("league", "Unknown"),
                    home_win_prob=0.35,
                    draw_prob=0.30,
                    away_win_prob=0.35,
                    expected_home_goals=1.3,
                    expected_away_goals=1.2,
                    over_25_prob=0.5,
                    btts_prob=0.5,
                    confidence=0.3,
                    ml_confidence=0.0,
                    kimi_confidence=0.0,
                    reasoning=f"Prediction failed: {result}",
                ))
            else:
                predictions.append(result)

        return predictions


# Convenience function
async def get_hybrid_prediction(
    home_team: str,
    away_team: str,
    league: str,
    features: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    odds: Optional[Dict[str, float]] = None,
) -> HybridPrediction:
    """
    Quick function to get a hybrid prediction.

    Usage:
        prediction = await get_hybrid_prediction(
            "Arsenal", "Chelsea", "Premier League",
            features={"home_form": 0.8, "away_form": 0.6},
            context={"injuries": {...}},
            odds={"home": 1.9, "draw": 3.5, "away": 4.0},
        )
        print(prediction.reasoning)
    """
    predictor = HybridPredictor()
    return await predictor.predict(
        home_team=home_team,
        away_team=away_team,
        league=league,
        features=features,
        context=context,
        odds=odds,
    )
