"""
Ensemble Integration.

Checkpoint: 4.3
Responsibility: Combine ML predictions with agent predictions.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum

from core.ml.service.results import (
    MLPredictionResult,
    GoalsPredictionResult,
    HandicapPredictionResult,
    BettingRecommendation,
)

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """Method for combining predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    ML_PRIMARY = "ml_primary"
    AGENT_PRIMARY = "agent_primary"
    MAX_CONFIDENCE = "max_confidence"
    BAYESIAN = "bayesian"


@dataclass
class AgentPrediction:
    """
    Prediction from an agent (non-ML source).
    """
    source: str  # Agent name
    market: str
    selection: str
    probability: float
    confidence: float
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble combination.
    """
    method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE

    # Weights for weighted average (must sum to 1)
    ml_weight: float = 0.6
    agent_weight: float = 0.4

    # Minimum confidence thresholds
    min_ml_confidence: float = 0.5
    min_agent_confidence: float = 0.5
    min_combined_confidence: float = 0.55

    # Disagreement handling
    max_disagreement: float = 0.3  # Max difference allowed
    require_agreement: bool = False  # If True, skip if predictions disagree


@dataclass
class EnsemblePrediction:
    """
    Combined prediction from ML and agents.
    """
    market: str
    selection: str
    combined_probability: float
    combined_confidence: float

    # Sources
    ml_probability: Optional[float] = None
    ml_confidence: Optional[float] = None
    agent_probability: Optional[float] = None
    agent_confidence: Optional[float] = None

    # Metadata
    combination_method: str = ""
    agreement_score: float = 1.0  # 1.0 = full agreement, 0.0 = opposite
    reasoning: str = ""

    @property
    def has_agreement(self) -> bool:
        """Check if ML and agent agree on selection."""
        return self.agreement_score >= 0.7


@dataclass
class EnsembleResult:
    """
    Result of ensemble prediction.
    """
    match_id: str
    predictions: List[EnsemblePrediction] = field(default_factory=list)
    recommendations: List[BettingRecommendation] = field(default_factory=list)

    # Diagnostics
    ml_available: bool = True
    agents_available: bool = True
    combination_notes: List[str] = field(default_factory=list)


class EnsembleIntegration:
    """
    Combines ML predictions with agent predictions.

    Provides multiple combination strategies and handles
    disagreements between sources.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def combine(
        self,
        ml_result: Optional[MLPredictionResult],
        agent_predictions: List[AgentPrediction],
        market_odds: Optional[Dict[str, float]] = None,
    ) -> EnsembleResult:
        """
        Combine ML and agent predictions.

        Args:
            ml_result: ML prediction result
            agent_predictions: List of agent predictions
            market_odds: Optional market odds for value calculation

        Returns:
            Combined ensemble result
        """
        match_id = ml_result.match_id if ml_result else "unknown"
        result = EnsembleResult(
            match_id=match_id,
            ml_available=ml_result is not None and ml_result.is_success,
            agents_available=len(agent_predictions) > 0,
        )

        # Group agent predictions by market
        agent_by_market = self._group_by_market(agent_predictions)

        # Extract ML predictions by market
        ml_by_market = self._extract_ml_markets(ml_result)

        # Combine for each market
        all_markets = set(ml_by_market.keys()) | set(agent_by_market.keys())

        for market in all_markets:
            ml_pred = ml_by_market.get(market)
            agent_preds = agent_by_market.get(market, [])

            combined = self._combine_market(
                market,
                ml_pred,
                agent_preds,
            )

            if combined:
                result.predictions.append(combined)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(
            result.predictions,
            market_odds,
        )

        return result

    def _group_by_market(
        self,
        predictions: List[AgentPrediction],
    ) -> Dict[str, List[AgentPrediction]]:
        """Group agent predictions by market."""
        grouped = {}
        for pred in predictions:
            if pred.market not in grouped:
                grouped[pred.market] = []
            grouped[pred.market].append(pred)
        return grouped

    def _extract_ml_markets(
        self,
        ml_result: Optional[MLPredictionResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract ML predictions organized by market."""
        markets = {}

        if not ml_result or not ml_result.is_success:
            return markets

        # Over/Under 2.5
        if ml_result.goals_prediction:
            goals = ml_result.goals_prediction
            if goals.over_25_prob > 0:
                markets["over_2.5"] = {
                    "selection": "over" if goals.over_25_prob > 0.5 else "under",
                    "probability": max(goals.over_25_prob, goals.under_25_prob),
                    "confidence": goals.confidence,
                }

            # BTTS
            if goals.btts_yes_prob > 0:
                markets["btts"] = {
                    "selection": "yes" if goals.btts_yes_prob > 0.5 else "no",
                    "probability": max(goals.btts_yes_prob, goals.btts_no_prob),
                    "confidence": goals.confidence * 0.9,
                }

        # 1X2
        if ml_result.handicap_prediction:
            hcp = ml_result.handicap_prediction
            max_prob = max(hcp.home_win_prob, hcp.draw_prob, hcp.away_win_prob)
            if hcp.home_win_prob == max_prob:
                selection = "home"
            elif hcp.away_win_prob == max_prob:
                selection = "away"
            else:
                selection = "draw"

            markets["1x2"] = {
                "selection": selection,
                "probability": max_prob,
                "confidence": hcp.confidence,
            }

            # Handicap -1.5
            if hcp.home_minus_15_prob > 0:
                markets["handicap_-1.5"] = {
                    "selection": "home" if hcp.home_minus_15_prob > 0.5 else "away",
                    "probability": max(hcp.home_minus_15_prob, 1 - hcp.home_minus_15_prob),
                    "confidence": hcp.confidence * 0.9,
                }

        return markets

    def _combine_market(
        self,
        market: str,
        ml_pred: Optional[Dict[str, Any]],
        agent_preds: List[AgentPrediction],
    ) -> Optional[EnsemblePrediction]:
        """Combine predictions for a single market."""

        # Get best agent prediction (highest confidence)
        best_agent = None
        if agent_preds:
            best_agent = max(agent_preds, key=lambda p: p.confidence)

        # Check minimum thresholds
        ml_valid = (
            ml_pred is not None and
            ml_pred.get("confidence", 0) >= self.config.min_ml_confidence
        )
        agent_valid = (
            best_agent is not None and
            best_agent.confidence >= self.config.min_agent_confidence
        )

        if not ml_valid and not agent_valid:
            return None

        # ML only
        if ml_valid and not agent_valid:
            return EnsemblePrediction(
                market=market,
                selection=ml_pred["selection"],
                combined_probability=ml_pred["probability"],
                combined_confidence=ml_pred["confidence"],
                ml_probability=ml_pred["probability"],
                ml_confidence=ml_pred["confidence"],
                combination_method="ml_only",
                reasoning="ML prediction only (no valid agent prediction)",
            )

        # Agent only
        if agent_valid and not ml_valid:
            return EnsemblePrediction(
                market=market,
                selection=best_agent.selection,
                combined_probability=best_agent.probability,
                combined_confidence=best_agent.confidence,
                agent_probability=best_agent.probability,
                agent_confidence=best_agent.confidence,
                combination_method="agent_only",
                reasoning=f"Agent prediction only ({best_agent.source})",
            )

        # Both valid - combine
        return self._combine_both(market, ml_pred, best_agent)

    def _combine_both(
        self,
        market: str,
        ml_pred: Dict[str, Any],
        agent_pred: AgentPrediction,
    ) -> Optional[EnsemblePrediction]:
        """Combine ML and agent predictions."""
        ml_prob = ml_pred["probability"]
        ml_conf = ml_pred["confidence"]
        ml_selection = ml_pred["selection"]

        agent_prob = agent_pred.probability
        agent_conf = agent_pred.confidence
        agent_selection = agent_pred.selection

        # Calculate agreement
        same_selection = ml_selection == agent_selection
        if same_selection:
            agreement = 1.0 - abs(ml_prob - agent_prob)
        else:
            agreement = 0.0

        # Check disagreement threshold
        if not same_selection and self.config.require_agreement:
            logger.info(f"Skipping {market}: ML and agent disagree")
            return None

        # Apply combination method
        if self.config.method == CombinationMethod.WEIGHTED_AVERAGE:
            combined_prob, combined_conf, selection = self._weighted_average(
                ml_prob, ml_conf, ml_selection,
                agent_prob, agent_conf, agent_selection,
            )
        elif self.config.method == CombinationMethod.ML_PRIMARY:
            combined_prob = ml_prob
            combined_conf = ml_conf
            selection = ml_selection
        elif self.config.method == CombinationMethod.AGENT_PRIMARY:
            combined_prob = agent_prob
            combined_conf = agent_conf
            selection = agent_selection
        elif self.config.method == CombinationMethod.MAX_CONFIDENCE:
            if ml_conf >= agent_conf:
                combined_prob = ml_prob
                combined_conf = ml_conf
                selection = ml_selection
            else:
                combined_prob = agent_prob
                combined_conf = agent_conf
                selection = agent_selection
        else:
            # Default to weighted average
            combined_prob, combined_conf, selection = self._weighted_average(
                ml_prob, ml_conf, ml_selection,
                agent_prob, agent_conf, agent_selection,
            )

        # Check combined threshold
        if combined_conf < self.config.min_combined_confidence:
            return None

        return EnsemblePrediction(
            market=market,
            selection=selection,
            combined_probability=combined_prob,
            combined_confidence=combined_conf,
            ml_probability=ml_prob,
            ml_confidence=ml_conf,
            agent_probability=agent_prob,
            agent_confidence=agent_conf,
            combination_method=self.config.method.value,
            agreement_score=agreement,
            reasoning=f"Combined ML ({ml_selection}: {ml_prob:.0%}) and "
                      f"{agent_pred.source} ({agent_selection}: {agent_prob:.0%})",
        )

    def _weighted_average(
        self,
        ml_prob: float,
        ml_conf: float,
        ml_selection: str,
        agent_prob: float,
        agent_conf: float,
        agent_selection: str,
    ) -> tuple:
        """Calculate weighted average of predictions."""
        ml_w = self.config.ml_weight
        agent_w = self.config.agent_weight

        # If same selection, average probabilities
        if ml_selection == agent_selection:
            combined_prob = ml_w * ml_prob + agent_w * agent_prob
            combined_conf = ml_w * ml_conf + agent_w * agent_conf
            selection = ml_selection
        else:
            # Disagreement - use higher weighted probability
            ml_weighted = ml_w * ml_prob
            agent_weighted = agent_w * agent_prob

            if ml_weighted >= agent_weighted:
                combined_prob = ml_prob
                combined_conf = ml_conf * (1 - agent_w * 0.3)  # Reduce confidence
                selection = ml_selection
            else:
                combined_prob = agent_prob
                combined_conf = agent_conf * (1 - ml_w * 0.3)
                selection = agent_selection

        return combined_prob, combined_conf, selection

    def _generate_recommendations(
        self,
        predictions: List[EnsemblePrediction],
        market_odds: Optional[Dict[str, float]],
    ) -> List[BettingRecommendation]:
        """Generate recommendations from ensemble predictions."""
        recommendations = []
        market_odds = market_odds or {}

        for pred in predictions:
            if pred.combined_confidence < self.config.min_combined_confidence:
                continue

            odds_required = 1 / pred.combined_probability if pred.combined_probability > 0 else 100
            market_key = f"{pred.market}_{pred.selection}"
            market_odd = market_odds.get(market_key)

            edge = None
            if market_odd and market_odd > 1:
                edge = pred.combined_probability - (1 / market_odd)

            recommendations.append(BettingRecommendation(
                market=pred.market,
                selection=pred.selection,
                probability=pred.combined_probability,
                odds_required=round(odds_required, 2),
                confidence=pred.combined_confidence,
                edge=round(edge, 4) if edge else None,
                reasoning=pred.reasoning,
            ))

        # Sort by confidence and edge
        recommendations.sort(
            key=lambda r: (r.edge or 0, r.confidence),
            reverse=True,
        )

        return recommendations


def create_default_ensemble() -> EnsembleIntegration:
    """Create ensemble integration with default config."""
    return EnsembleIntegration(EnsembleConfig())


def create_ml_primary_ensemble() -> EnsembleIntegration:
    """Create ensemble that prioritizes ML predictions."""
    config = EnsembleConfig(
        method=CombinationMethod.ML_PRIMARY,
        ml_weight=0.8,
        agent_weight=0.2,
    )
    return EnsembleIntegration(config)


def create_conservative_ensemble() -> EnsembleIntegration:
    """Create ensemble that requires agreement."""
    config = EnsembleConfig(
        method=CombinationMethod.WEIGHTED_AVERAGE,
        require_agreement=True,
        min_combined_confidence=0.65,
    )
    return EnsembleIntegration(config)
