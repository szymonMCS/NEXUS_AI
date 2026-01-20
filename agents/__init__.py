# agents/__init__.py
"""
LangGraph agents for NEXUS AI betting analysis workflow.

Agent Pipeline:
1. Supervisor - Orchestrates the workflow
2. News Analyst - Collects and analyzes news
3. Data Evaluator - Assesses data quality
4. Analyst - Makes predictions using AI
5. Ranker - Ranks matches by value potential
6. Risk Manager - Calculates stakes using Kelly Criterion
7. Decision Maker - Makes final betting decisions
8. Bettor - Executes betting decisions (optional)
"""

from agents.supervisor import (
    SupervisorAgent,
    create_supervisor,
    run_betting_analysis,
)

from agents.news_analyst import (
    NewsAnalystAgent,
    collect_match_news,
)

from agents.data_evaluator import (
    DataEvaluatorAgent,
    evaluate_match_quality,
)

from agents.analyst import (
    AnalystAgent,
    analyze_matches,
)

from agents.ranker import (
    RankerAgent,
    rank_matches,
)

from agents.risk_manager import (
    RiskManagerAgent,
    assess_betting_risk,
)

from agents.decision_maker import (
    DecisionMakerAgent,
    make_betting_decisions,
)

from agents.bettor import (
    BettorAgent,
    BetStatus,
    PlacedBet,
    BettingSession,
    place_bets,
)

__all__ = [
    # Supervisor
    "SupervisorAgent",
    "create_supervisor",
    "run_betting_analysis",
    # News Analyst
    "NewsAnalystAgent",
    "collect_match_news",
    # Data Evaluator
    "DataEvaluatorAgent",
    "evaluate_match_quality",
    # Analyst
    "AnalystAgent",
    "analyze_matches",
    # Ranker
    "RankerAgent",
    "rank_matches",
    # Risk Manager
    "RiskManagerAgent",
    "assess_betting_risk",
    # Decision Maker
    "DecisionMakerAgent",
    "make_betting_decisions",
    # Bettor
    "BettorAgent",
    "BetStatus",
    "PlacedBet",
    "BettingSession",
    "place_bets",
]
