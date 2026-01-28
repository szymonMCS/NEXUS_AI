"""
Tests for LLM Integration Module.

Checkpoint: 7.5
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from core.llm.kimi_client import KimiClient, KimiResponse, KimiModel, KimiMode, KimiAgentSwarm, AgentTask
from core.llm.injury_extractor import (
    InjuryExtractor,
    PlayerInjury,
    PlayerStatus,
    TeamAvailability,
)
from core.llm.match_analyzer import (
    MatchAnalyzer,
    MatchAnalysis,
    MatchFactor,
    AnalysisConfidence,
)
from core.llm.hybrid_predictor import (
    HybridPredictor,
    HybridPrediction,
    MLPrediction,
    PredictionSource,
    RecommendationType,
)


# =============================================================================
# Test KimiClient
# =============================================================================

class TestKimiClient:
    """Tests for KimiClient."""

    def test_create_client(self):
        """Test client creation."""
        client = KimiClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.model == KimiModel.K2_5_PREVIEW

    def test_is_available_with_key(self):
        """Test availability check with valid key."""
        client = KimiClient(api_key="real_api_key_123")
        assert client.is_available

    def test_is_available_without_key(self):
        """Test availability check without key."""
        client = KimiClient(api_key="")  # Empty string, not None
        assert not client.is_available

    def test_is_available_with_placeholder(self):
        """Test availability check with placeholder key."""
        client = KimiClient(api_key="your_kimi_api_key_here")
        assert not client.is_available

    @pytest.mark.asyncio
    async def test_chat_without_key(self):
        """Test chat returns error without API key."""
        client = KimiClient(api_key="")  # Empty string, not None
        response = await client.chat("Hello")

        assert not response.success
        assert "MOONSHOT_API_KEY" in response.error

    def test_kimi_response_properties(self):
        """Test KimiResponse dataclass."""
        response = KimiResponse(
            success=True,
            content="Test response",
            model="kimi-k2.5-preview",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        assert response.total_tokens == 30
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20

    def test_kimi_response_with_reasoning(self):
        """Test KimiResponse with reasoning content (thinking mode)."""
        response = KimiResponse(
            success=True,
            content="Final answer",
            model="kimi-k2-thinking",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
            reasoning_content="Step 1: Analyze... Step 2: Consider..."
        )

        assert response.has_reasoning
        assert response.reasoning_content == "Step 1: Analyze... Step 2: Consider..."

    def test_kimi_mode_enum(self):
        """Test KimiMode enum values."""
        assert KimiMode.THINKING.value == "thinking"
        assert KimiMode.INSTANT.value == "instant"

    def test_kimi_model_k25(self):
        """Test K2.5 model enum."""
        assert KimiModel.K2_5_PREVIEW.value == "kimi-k2.5-preview"
        assert KimiModel.K2_THINKING.value == "kimi-k2-thinking"

    def test_agent_task_dataclass(self):
        """Test AgentTask dataclass."""
        task = AgentTask(
            task_id="task_1",
            description="Analyze match data",
            agent_type="analyzer",
        )
        assert task.status == "pending"
        assert task.result is None

        task.status = "completed"
        task.result = "Analysis complete"
        assert task.status == "completed"


# =============================================================================
# Test InjuryExtractor
# =============================================================================

class TestInjuryExtractor:
    """Tests for InjuryExtractor."""

    def test_player_injury_to_dict(self):
        """Test PlayerInjury serialization."""
        injury = PlayerInjury(
            player_name="John Doe",
            team="Arsenal",
            injury_type="Hamstring",
            status=PlayerStatus.OUT,
            expected_return="2 weeks",
        )

        data = injury.to_dict()
        assert data["player_name"] == "John Doe"
        assert data["team"] == "Arsenal"
        assert data["status"] == "out"

    def test_player_injury_from_dict(self):
        """Test PlayerInjury deserialization."""
        data = {
            "player_name": "Jane Smith",
            "team": "Chelsea",
            "injury_type": "Knee",
            "status": "doubtful",
            "last_updated": datetime.utcnow().isoformat(),
        }

        injury = PlayerInjury.from_dict(data)
        assert injury.player_name == "Jane Smith"
        assert injury.status == PlayerStatus.DOUBTFUL

    def test_team_availability_properties(self):
        """Test TeamAvailability computed properties."""
        team = TeamAvailability(team_name="Arsenal")

        team.injuries.append(PlayerInjury(
            player_name="Player1",
            team="Arsenal",
            injury_type="Hamstring",
            status=PlayerStatus.OUT,
        ))
        team.injuries.append(PlayerInjury(
            player_name="Player2",
            team="Arsenal",
            injury_type="Knee",
            status=PlayerStatus.QUESTIONABLE,
        ))

        assert "Player1" in team.players_out
        assert "Player2" in team.players_doubtful
        assert team.total_unavailable == 1
        assert team.injury_severity_score > 0

    def test_parse_status(self):
        """Test status string parsing."""
        extractor = InjuryExtractor()

        assert extractor._parse_status("out") == PlayerStatus.OUT
        assert extractor._parse_status("ruled out") == PlayerStatus.OUT
        assert extractor._parse_status("doubtful") == PlayerStatus.DOUBTFUL
        assert extractor._parse_status("questionable") == PlayerStatus.QUESTIONABLE
        assert extractor._parse_status("50-50") == PlayerStatus.QUESTIONABLE
        assert extractor._parse_status("probable") == PlayerStatus.PROBABLE
        assert extractor._parse_status("unknown text") == PlayerStatus.UNKNOWN


# =============================================================================
# Test MatchAnalyzer
# =============================================================================

class TestMatchAnalyzer:
    """Tests for MatchAnalyzer."""

    def test_match_factor(self):
        """Test MatchFactor dataclass."""
        factor = MatchFactor(
            name="Home Form",
            description="Strong home record",
            impact="positive_home",
            weight=0.8,
            confidence=0.9,
        )

        data = factor.to_dict()
        assert data["name"] == "Home Form"
        assert data["impact"] == "positive_home"

    def test_match_analysis_confidence_level(self):
        """Test confidence level categorization."""
        analysis = MatchAnalysis(
            match_id=None,
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            predicted_outcome="home",
            outcome_confidence=0.9,
            predicted_goals=2.5,
            goals_confidence=0.7,
        )
        assert analysis.confidence_level == AnalysisConfidence.VERY_HIGH

        analysis.outcome_confidence = 0.75
        assert analysis.confidence_level == AnalysisConfidence.HIGH

        analysis.outcome_confidence = 0.6
        assert analysis.confidence_level == AnalysisConfidence.MEDIUM

        analysis.outcome_confidence = 0.45
        assert analysis.confidence_level == AnalysisConfidence.LOW

        analysis.outcome_confidence = 0.3
        assert analysis.confidence_level == AnalysisConfidence.VERY_LOW

    def test_home_advantage_score(self):
        """Test home advantage calculation."""
        analysis = MatchAnalysis(
            match_id=None,
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            predicted_outcome="home",
            outcome_confidence=0.7,
            predicted_goals=2.5,
            goals_confidence=0.6,
            key_factors=[
                MatchFactor("Home form", "Strong", "positive_home", 0.8, 0.9),
                MatchFactor("Away form", "Weak", "positive_home", 0.6, 0.8),
            ],
        )

        score = analysis.home_advantage_score
        assert 0.5 < score < 1.0  # Should favor home

    def test_fallback_analysis(self):
        """Test fallback analysis creation."""
        analyzer = MatchAnalyzer()
        analysis = analyzer._create_fallback_analysis(
            "Arsenal", "Chelsea", "PL", "match_123", "API error"
        )

        assert analysis.predicted_outcome == "draw"
        assert analysis.outcome_confidence == 0.33
        assert "API error" in analysis.risks[0]


# =============================================================================
# Test HybridPredictor
# =============================================================================

class TestHybridPredictor:
    """Tests for HybridPredictor."""

    def test_weight_normalization(self):
        """Test that weights are normalized."""
        predictor = HybridPredictor(ml_weight=0.8, kimi_weight=0.2)
        assert predictor.ml_weight == 0.8
        assert predictor.kimi_weight == 0.2

        predictor2 = HybridPredictor(ml_weight=3, kimi_weight=2)
        assert predictor2.ml_weight == 0.6
        assert predictor2.kimi_weight == 0.4

    def test_ml_prediction_properties(self):
        """Test MLPrediction computed properties."""
        pred = MLPrediction(
            source=PredictionSource.ML_GOALS,
            home_win_prob=0.5,
            draw_prob=0.25,
            away_win_prob=0.25,
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            over_25_prob=0.6,
            confidence=0.7,
        )

        assert pred.expected_total_goals == 2.5
        assert pred.predicted_outcome == "home"

    def test_hybrid_prediction_properties(self):
        """Test HybridPrediction computed properties."""
        pred = HybridPrediction(
            match_id="test_123",
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            home_win_prob=0.45,
            draw_prob=0.30,
            away_win_prob=0.25,
            expected_home_goals=1.5,
            expected_away_goals=1.2,
            over_25_prob=0.55,
            btts_prob=0.6,
            confidence=0.65,
            ml_confidence=0.7,
            kimi_confidence=0.6,
        )

        assert pred.expected_total_goals == 2.7
        assert pred.predicted_outcome == "home"
        assert pred.max_outcome_prob == 0.45

    def test_get_value_bets(self):
        """Test value bet detection."""
        pred = HybridPrediction(
            match_id=None,
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            home_win_prob=0.55,  # Our prediction
            draw_prob=0.25,
            away_win_prob=0.20,
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            over_25_prob=0.6,
            btts_prob=0.5,
            confidence=0.7,
            ml_confidence=0.7,
            kimi_confidence=0.7,
        )

        # Market implies 40% (1/2.5) for home
        odds = {"home": 2.5, "draw": 3.5, "away": 4.0}
        value_bets = pred.get_value_bets(odds)

        # Should find value on home (55% vs 40%)
        assert len(value_bets) > 0
        home_bet = next((b for b in value_bets if b["selection"] == "home"), None)
        assert home_bet is not None
        assert home_bet["edge"] > 0.1  # 55% - 40% = 15% edge

    def test_goals_to_probs(self):
        """Test Poisson probability calculation."""
        predictor = HybridPredictor()

        # Higher home goals should give higher home win prob
        probs = predictor._goals_to_probs(2.0, 1.0)
        assert probs["home"] > probs["away"]
        assert probs["home"] > probs["draw"]

        # Equal goals should favor draw somewhat
        probs = predictor._goals_to_probs(1.5, 1.5)
        assert abs(probs["home"] - probs["away"]) < 0.01

    def test_estimate_btts_prob(self):
        """Test BTTS probability estimation."""
        predictor = HybridPredictor()

        # High scoring match should have high BTTS
        btts = predictor._estimate_btts_prob(2.0, 2.0)
        assert btts > 0.7

        # Low scoring match should have lower BTTS
        btts_low = predictor._estimate_btts_prob(0.5, 0.5)
        assert btts_low < btts

    def test_combine_predictions_ml_only(self):
        """Test combination with only ML prediction."""
        predictor = HybridPredictor()

        ml_pred = MLPrediction(
            source=PredictionSource.ML_GOALS,
            home_win_prob=0.5,
            draw_prob=0.3,
            away_win_prob=0.2,
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            over_25_prob=0.6,
            confidence=0.7,
        )

        hybrid = predictor._combine_predictions(
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            match_id=None,
            ml_prediction=ml_pred,
            kimi_analysis=None,
        )

        # Should use 100% ML weights
        assert hybrid.ml_weight == 1.0
        assert hybrid.kimi_weight == 0.0
        assert abs(hybrid.home_win_prob - 0.5) < 0.01

    def test_combine_predictions_no_sources(self):
        """Test combination with no predictions."""
        predictor = HybridPredictor()

        hybrid = predictor._combine_predictions(
            home_team="Arsenal",
            away_team="Chelsea",
            league="PL",
            match_id=None,
            ml_prediction=None,
            kimi_analysis=None,
        )

        # Should return defaults
        assert hybrid.confidence == 0.3
        assert abs(hybrid.home_win_prob - 0.35) < 0.01


# =============================================================================
# Test Integration (with mocks)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked external calls."""

    @pytest.mark.asyncio
    async def test_hybrid_predict_without_models(self):
        """Test hybrid prediction falls back gracefully."""
        predictor = HybridPredictor()

        # Without loading models and with mocked Kimi
        with patch.object(predictor._match_analyzer, 'analyze_match') as mock_kimi:
            mock_kimi.return_value = MatchAnalysis(
                match_id=None,
                home_team="Arsenal",
                away_team="Chelsea",
                league="PL",
                predicted_outcome="home",
                outcome_confidence=0.65,
                predicted_goals=2.8,
                goals_confidence=0.6,
                summary="Arsenal favored at home",
            )

            prediction = await predictor.predict(
                home_team="Arsenal",
                away_team="Chelsea",
                league="Premier League",
                use_kimi=True,
            )

            assert prediction.home_team == "Arsenal"
            assert prediction.away_team == "Chelsea"
            assert prediction.kimi_analysis is not None
