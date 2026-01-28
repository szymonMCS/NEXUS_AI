# core/llm/__init__.py
"""
LLM Integration Module for NEXUS AI.

Checkpoint: 7.5 - Updated for Kimi K2.5

This module provides LLM-powered analysis and prediction enhancement using
Moonshot Kimi K2.5 with Agent Swarm capabilities.

Features:
- Kimi K2.5 multimodal agentic model
- Thinking mode with reasoning traces (kimi-k2-thinking)
- Agent Swarm for complex task decomposition
- Injury extraction from news
- Deep match analysis
- Hybrid ML + LLM predictions

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    core/llm/                         │
    ├─────────────────────────────────────────────────────┤
    │  kimi_client.py      → Moonshot K2.5 API client     │
    │                        + Agent Swarm orchestrator   │
    │  injury_extractor.py → News parsing & extraction    │
    │  match_analyzer.py   → Deep match analysis          │
    │  hybrid_predictor.py → ML + LLM ensemble            │
    └─────────────────────────────────────────────────────┘

API Configuration:
    MOONSHOT_API_KEY - Your Moonshot API key
    Get key at: https://platform.moonshot.ai/console/api-keys

Usage:
    # Quick prediction with K2.5
    from core.llm import get_hybrid_prediction
    prediction = await get_hybrid_prediction("Arsenal", "Chelsea", "PL")

    # Full analysis with thinking mode
    from core.llm import KimiClient, KimiMode
    async with KimiClient() as client:
        response = await client.chat_thinking("Complex analysis...")
        print(response.reasoning_content)  # Reasoning traces

    # Agent Swarm for complex tasks
    from core.llm import get_swarm_analysis
    result = await get_swarm_analysis("Analyze all PL matches with context")

    # Extract injuries
    from core.llm import get_team_injuries
    injuries = await get_team_injuries("Arsenal", news_texts)
"""

# Kimi Client (Moonshot K2.5)
from core.llm.kimi_client import (
    KimiClient,
    KimiResponse,
    KimiModel,
    KimiMode,
    KimiAgentSwarm,
    AgentTask,
    get_kimi_analysis,
    get_swarm_analysis,
)

# Injury Extractor
from core.llm.injury_extractor import (
    InjuryExtractor,
    PlayerInjury,
    PlayerSuspension,
    PlayerStatus,
    TeamAvailability,
    get_team_injuries,
)

# Match Analyzer
from core.llm.match_analyzer import (
    MatchAnalyzer,
    MatchAnalysis,
    MatchFactor,
    AnalysisConfidence,
    analyze_match,
)

# Hybrid Predictor
from core.llm.hybrid_predictor import (
    HybridPredictor,
    HybridPrediction,
    MLPrediction,
    PredictionSource,
    RecommendationType,
    get_hybrid_prediction,
)

__all__ = [
    # Kimi Client (Moonshot K2.5)
    "KimiClient",
    "KimiResponse",
    "KimiModel",
    "KimiMode",
    "KimiAgentSwarm",
    "AgentTask",
    "get_kimi_analysis",
    "get_swarm_analysis",
    # Injury Extractor
    "InjuryExtractor",
    "PlayerInjury",
    "PlayerSuspension",
    "PlayerStatus",
    "TeamAvailability",
    "get_team_injuries",
    # Match Analyzer
    "MatchAnalyzer",
    "MatchAnalysis",
    "MatchFactor",
    "AnalysisConfidence",
    "analyze_match",
    # Hybrid Predictor
    "HybridPredictor",
    "HybridPrediction",
    "MLPrediction",
    "PredictionSource",
    "RecommendationType",
    "get_hybrid_prediction",
]

__version__ = "1.0.0"
