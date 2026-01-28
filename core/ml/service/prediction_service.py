"""
ML Prediction Service.

Checkpoint: 4.2
Responsibility: Orchestrate feature extraction and model predictions.
"""

import logging
import uuid
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from core.data.schemas import MatchData, DataQuality, Sport
from core.data.repository import IDataRepository
from core.ml.features import FeaturePipeline, FeatureVector
from core.ml.models import GoalsModel, HandicapModel, GoalsPrediction, HandicapPrediction
from core.ml.registry import ModelRegistry
from core.ml.service.results import (
    MLPredictionResult,
    GoalsPredictionResult,
    HandicapPredictionResult,
    BettingRecommendation,
    BatchPredictionResult,
    PredictionStatus,
)

logger = logging.getLogger(__name__)


class MLPredictionService:
    """
    Main ML prediction service.

    Orchestrates:
    - Data enrichment via repository
    - Feature extraction via pipeline
    - Model predictions (goals, handicap)
    - Betting recommendation generation
    """

    # Minimum data quality for predictions
    MIN_COMPLETENESS = 0.5
    MIN_CONFIDENCE_FOR_RECOMMENDATION = 0.55
    MIN_EDGE_FOR_VALUE = 0.02  # 2% edge minimum

    def __init__(
        self,
        repository: IDataRepository,
        registry: Optional[ModelRegistry] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
    ):
        self.repository = repository
        self.registry = registry
        self.feature_pipeline = feature_pipeline or FeaturePipeline()

        # Models (lazy loaded from registry or created fresh)
        self._goals_model: Optional[GoalsModel] = None
        self._handicap_model: Optional[HandicapModel] = None

        # Configuration
        self._goals_model_name = "poisson_goals"
        self._handicap_model_name = "gbm_handicap"

    @property
    def goals_model(self) -> GoalsModel:
        """Get or load goals model."""
        if self._goals_model is None:
            if self.registry:
                loaded = self.registry.load_model(
                    self._goals_model_name,
                    GoalsModel,
                )
                if loaded:
                    self._goals_model = loaded
                    logger.info(f"Loaded goals model from registry")
            if self._goals_model is None:
                self._goals_model = GoalsModel()
                logger.info("Created new goals model (not trained)")
        return self._goals_model

    @property
    def handicap_model(self) -> HandicapModel:
        """Get or load handicap model."""
        if self._handicap_model is None:
            if self.registry:
                loaded = self.registry.load_model(
                    self._handicap_model_name,
                    HandicapModel,
                )
                if loaded:
                    self._handicap_model = loaded
                    logger.info(f"Loaded handicap model from registry")
            if self._handicap_model is None:
                self._handicap_model = HandicapModel()
                logger.info("Created new handicap model (not trained)")
        return self._handicap_model

    def predict(
        self,
        match: MatchData,
        include_recommendations: bool = True,
        market_odds: Optional[Dict[str, float]] = None,
    ) -> MLPredictionResult:
        """
        Generate ML predictions for a match.

        Args:
            match: Match data (will be enriched if needed)
            include_recommendations: Whether to generate betting recommendations
            market_odds: Optional market odds for value calculation

        Returns:
            Complete ML prediction result
        """
        start_time = time.time()
        prediction_id = str(uuid.uuid4())

        try:
            # Step 1: Enrich match data
            enriched_match = self._enrich_match(match)

            # Step 2: Check data quality
            quality_check = self._check_data_quality(enriched_match)
            if not quality_check["pass"]:
                return MLPredictionResult(
                    match_id=match.match_id,
                    prediction_id=prediction_id,
                    status=PredictionStatus.INSUFFICIENT_DATA,
                    error_message=quality_check["reason"],
                    data_quality=enriched_match.quality,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Step 3: Extract features
            features = self.feature_pipeline.extract(enriched_match)

            # Step 4: Get predictions from models
            goals_result = self._predict_goals(features)
            handicap_result = self._predict_handicap(features)

            # Step 5: Generate recommendations if requested
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_recommendations(
                    goals_result,
                    handicap_result,
                    market_odds,
                )

            # Step 6: Build result
            processing_time = (time.time() - start_time) * 1000

            return MLPredictionResult(
                match_id=match.match_id,
                prediction_id=prediction_id,
                status=PredictionStatus.SUCCESS,
                goals_prediction=goals_result,
                handicap_prediction=handicap_result,
                recommendations=recommendations,
                data_quality=enriched_match.quality,
                features_used=list(features.features.keys()),
                processing_time_ms=processing_time,
                model_versions={
                    "goals": goals_result.model_version if goals_result else "",
                    "handicap": handicap_result.model_version if handicap_result else "",
                },
            )

        except Exception as e:
            logger.exception(f"Prediction error for match {match.match_id}")
            return MLPredictionResult(
                match_id=match.match_id,
                prediction_id=prediction_id,
                status=PredictionStatus.MODEL_ERROR,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def predict_batch(
        self,
        matches: List[MatchData],
        include_recommendations: bool = True,
    ) -> BatchPredictionResult:
        """
        Generate predictions for multiple matches.

        Args:
            matches: List of match data
            include_recommendations: Whether to generate betting recommendations

        Returns:
            Batch prediction result
        """
        start_time = time.time()
        predictions = []

        for match in matches:
            result = self.predict(match, include_recommendations)
            predictions.append(result)

        total_time = (time.time() - start_time) * 1000

        return BatchPredictionResult(
            predictions=predictions,
            total_processing_time_ms=total_time,
        )

    def _enrich_match(self, match: MatchData) -> MatchData:
        """Enrich match with additional data from repository."""
        try:
            enriched = self.repository.enrich_match_data(
                match,
                include_team_stats=True,
                include_h2h=True,
            )
            return enriched
        except Exception as e:
            logger.warning(f"Failed to enrich match {match.match_id}: {e}")
            return match

    def _check_data_quality(self, match: MatchData) -> Dict[str, Any]:
        """Check if match data quality is sufficient for prediction."""
        if match.quality is None:
            return {"pass": False, "reason": "No data quality information"}

        if match.quality.completeness < self.MIN_COMPLETENESS:
            return {
                "pass": False,
                "reason": f"Data completeness {match.quality.completeness:.0%} "
                          f"below minimum {self.MIN_COMPLETENESS:.0%}",
            }

        # Check essential data
        if match.home_team is None or match.away_team is None:
            return {"pass": False, "reason": "Missing team information"}

        return {"pass": True, "reason": ""}

    def _predict_goals(self, features: FeatureVector) -> Optional[GoalsPredictionResult]:
        """Generate goals prediction."""
        try:
            prediction: GoalsPrediction = self.goals_model.predict(features)

            # Build score probabilities (top 10)
            score_probs = {}
            if prediction.score_matrix:
                sorted_scores = sorted(
                    prediction.score_matrix.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
                score_probs = dict(sorted_scores)

            return GoalsPredictionResult(
                home_expected=prediction.home_expected,
                away_expected=prediction.away_expected,
                total_expected=prediction.total_expected,
                over_15_prob=prediction.over_15_prob if hasattr(prediction, 'over_15_prob') else 0.0,
                over_25_prob=prediction.over_25_prob,
                over_35_prob=prediction.over_35_prob if hasattr(prediction, 'over_35_prob') else 0.0,
                under_15_prob=1 - (prediction.over_15_prob if hasattr(prediction, 'over_15_prob') else 0.0),
                under_25_prob=prediction.under_25_prob,
                under_35_prob=1 - (prediction.over_35_prob if hasattr(prediction, 'over_35_prob') else 0.0),
                btts_yes_prob=prediction.btts_prob,
                btts_no_prob=1 - prediction.btts_prob,
                score_probabilities=score_probs,
                model_version=self.goals_model.version,
                confidence=prediction.confidence,
            )
        except Exception as e:
            logger.warning(f"Goals prediction failed: {e}")
            return None

    def _predict_handicap(self, features: FeatureVector) -> Optional[HandicapPredictionResult]:
        """Generate handicap prediction."""
        try:
            prediction: HandicapPrediction = self.handicap_model.predict(features)

            return HandicapPredictionResult(
                expected_margin=prediction.expected_margin,
                home_win_prob=prediction.home_win_prob,
                draw_prob=prediction.draw_prob,
                away_win_prob=prediction.away_win_prob,
                home_minus_05_prob=prediction.home_win_prob,  # Home -0.5 = home win
                home_minus_15_prob=prediction.home_cover_minus_15,
                home_minus_25_prob=prediction.home_cover_minus_25 if hasattr(prediction, 'home_cover_minus_25') else 0.0,
                away_plus_05_prob=prediction.draw_prob + prediction.away_win_prob,
                away_plus_15_prob=1 - prediction.home_cover_minus_15,
                away_plus_25_prob=1 - (prediction.home_cover_minus_25 if hasattr(prediction, 'home_cover_minus_25') else 0.0),
                model_version=self.handicap_model.version,
                confidence=prediction.confidence,
            )
        except Exception as e:
            logger.warning(f"Handicap prediction failed: {e}")
            return None

    def _generate_recommendations(
        self,
        goals: Optional[GoalsPredictionResult],
        handicap: Optional[HandicapPredictionResult],
        market_odds: Optional[Dict[str, float]] = None,
    ) -> List[BettingRecommendation]:
        """Generate betting recommendations based on predictions."""
        recommendations = []
        market_odds = market_odds or {}

        # Over/Under 2.5 recommendations
        if goals:
            recommendations.extend(self._goals_recommendations(goals, market_odds))

        # 1X2 and handicap recommendations
        if handicap:
            recommendations.extend(self._handicap_recommendations(handicap, market_odds))

        # Sort by confidence and edge
        recommendations.sort(
            key=lambda r: (r.edge or 0, r.confidence),
            reverse=True,
        )

        return recommendations

    def _goals_recommendations(
        self,
        goals: GoalsPredictionResult,
        market_odds: Dict[str, float],
    ) -> List[BettingRecommendation]:
        """Generate goals/over-under recommendations."""
        recs = []

        # Over 2.5
        if goals.over_25_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / goals.over_25_prob if goals.over_25_prob > 0 else 100
            market_odd = market_odds.get("over_2.5")
            edge = self._calculate_edge(goals.over_25_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="over_2.5",
                    selection="over",
                    probability=goals.over_25_prob,
                    odds_required=round(odds_required, 2),
                    confidence=goals.confidence,
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(goals.over_25_prob, market_odd),
                    reasoning=f"Expected {goals.total_expected:.1f} goals",
                ))

        # Under 2.5
        if goals.under_25_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / goals.under_25_prob if goals.under_25_prob > 0 else 100
            market_odd = market_odds.get("under_2.5")
            edge = self._calculate_edge(goals.under_25_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="under_2.5",
                    selection="under",
                    probability=goals.under_25_prob,
                    odds_required=round(odds_required, 2),
                    confidence=goals.confidence,
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(goals.under_25_prob, market_odd),
                    reasoning=f"Expected {goals.total_expected:.1f} goals",
                ))

        # BTTS Yes
        if goals.btts_yes_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / goals.btts_yes_prob if goals.btts_yes_prob > 0 else 100
            market_odd = market_odds.get("btts_yes")
            edge = self._calculate_edge(goals.btts_yes_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="btts",
                    selection="yes",
                    probability=goals.btts_yes_prob,
                    odds_required=round(odds_required, 2),
                    confidence=goals.confidence * 0.9,  # Slightly less confident on BTTS
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(goals.btts_yes_prob, market_odd),
                    reasoning=f"Home {goals.home_expected:.1f}, Away {goals.away_expected:.1f} expected",
                ))

        return recs

    def _handicap_recommendations(
        self,
        handicap: HandicapPredictionResult,
        market_odds: Dict[str, float],
    ) -> List[BettingRecommendation]:
        """Generate handicap/1X2 recommendations."""
        recs = []

        # Home win
        if handicap.home_win_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / handicap.home_win_prob if handicap.home_win_prob > 0 else 100
            market_odd = market_odds.get("1x2_home")
            edge = self._calculate_edge(handicap.home_win_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="1x2",
                    selection="home",
                    probability=handicap.home_win_prob,
                    odds_required=round(odds_required, 2),
                    confidence=handicap.confidence,
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(handicap.home_win_prob, market_odd),
                    reasoning=f"Expected margin {handicap.expected_margin:+.1f}",
                ))

        # Away win
        if handicap.away_win_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / handicap.away_win_prob if handicap.away_win_prob > 0 else 100
            market_odd = market_odds.get("1x2_away")
            edge = self._calculate_edge(handicap.away_win_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="1x2",
                    selection="away",
                    probability=handicap.away_win_prob,
                    odds_required=round(odds_required, 2),
                    confidence=handicap.confidence,
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(handicap.away_win_prob, market_odd),
                    reasoning=f"Expected margin {handicap.expected_margin:+.1f}",
                ))

        # Draw (only if high probability)
        if handicap.draw_prob >= 0.35:  # Draws need higher threshold
            odds_required = 1 / handicap.draw_prob if handicap.draw_prob > 0 else 100
            market_odd = market_odds.get("1x2_draw")
            edge = self._calculate_edge(handicap.draw_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="1x2",
                    selection="draw",
                    probability=handicap.draw_prob,
                    odds_required=round(odds_required, 2),
                    confidence=handicap.confidence * 0.85,  # Draws are harder to predict
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(handicap.draw_prob, market_odd),
                    reasoning=f"Expected margin {handicap.expected_margin:+.1f}",
                ))

        # Asian handicap -1.5
        if handicap.home_minus_15_prob >= self.MIN_CONFIDENCE_FOR_RECOMMENDATION:
            odds_required = 1 / handicap.home_minus_15_prob if handicap.home_minus_15_prob > 0 else 100
            market_odd = market_odds.get("handicap_home_-1.5")
            edge = self._calculate_edge(handicap.home_minus_15_prob, market_odd)

            if edge is None or edge >= self.MIN_EDGE_FOR_VALUE:
                recs.append(BettingRecommendation(
                    market="handicap_-1.5",
                    selection="home",
                    probability=handicap.home_minus_15_prob,
                    odds_required=round(odds_required, 2),
                    confidence=handicap.confidence * 0.9,
                    edge=edge,
                    kelly_fraction=self._kelly_fraction(handicap.home_minus_15_prob, market_odd),
                    reasoning=f"Home to win by 2+ goals",
                ))

        return recs

    def _calculate_edge(
        self,
        probability: float,
        market_odds: Optional[float],
    ) -> Optional[float]:
        """Calculate edge (probability - implied probability)."""
        if market_odds is None or market_odds <= 1:
            return None
        implied_prob = 1 / market_odds
        return probability - implied_prob

    def _kelly_fraction(
        self,
        probability: float,
        market_odds: Optional[float],
        fraction: float = 0.25,  # Quarter Kelly
    ) -> Optional[float]:
        """Calculate Kelly criterion stake fraction."""
        if market_odds is None or market_odds <= 1:
            return None

        # Kelly formula: (bp - q) / b
        # b = decimal_odds - 1, p = probability, q = 1 - p
        b = market_odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        # Return fractional Kelly (typically 25%)
        return max(0, kelly * fraction)

    def set_goals_model(self, model: GoalsModel) -> None:
        """Set the goals model."""
        self._goals_model = model

    def set_handicap_model(self, model: HandicapModel) -> None:
        """Set the handicap model."""
        self._handicap_model = model

    def reload_models(self) -> bool:
        """Reload models from registry."""
        if not self.registry:
            return False

        self._goals_model = None
        self._handicap_model = None

        # Trigger lazy loading
        _ = self.goals_model
        _ = self.handicap_model

        return True
