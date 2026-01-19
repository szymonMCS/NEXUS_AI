# agents/supervisor.py
"""
Supervisor Agent - Main orchestrator for the NEXUS AI workflow.
Coordinates all other agents and manages the betting analysis pipeline.
"""

from typing import Literal, Dict, List, Any
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config.settings import settings
from core.state import NexusState, Match, add_message


class SupervisorAgent:
    """
    Supervisor coordinates the multi-agent betting analysis workflow.

    Workflow:
    1. Collect matches → News Analyst
    2. Evaluate data quality → Data Evaluator
    3. Analyze and predict → Analyst
    4. Rank matches → Ranker
    5. Risk assessment → Risk Manager
    6. Final decision → Decision Maker
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )

    def create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for betting analysis.

        Returns:
            Compiled LangGraph StateGraph
        """
        from agents.news_analyst import NewsAnalystAgent
        from agents.data_evaluator import DataEvaluatorAgent
        from agents.analyst import AnalystAgent
        from agents.ranker import RankerAgent
        from agents.risk_manager import RiskManagerAgent
        from agents.decision_maker import DecisionMakerAgent

        # Create agent instances
        news_analyst = NewsAnalystAgent()
        data_evaluator = DataEvaluatorAgent()
        analyst = AnalystAgent()
        ranker = RankerAgent()
        risk_manager = RiskManagerAgent()
        decision_maker = DecisionMakerAgent()

        # Define state graph
        workflow = StateGraph(NexusState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("news_analyst", news_analyst.process)
        workflow.add_node("data_evaluator", data_evaluator.process)
        workflow.add_node("analyst", analyst.process)
        workflow.add_node("ranker", ranker.process)
        workflow.add_node("risk_manager", risk_manager.process)
        workflow.add_node("decision_maker", decision_maker.process)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add edges based on routing
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next,
            {
                "news_analyst": "news_analyst",
                "data_evaluator": "data_evaluator",
                "analyst": "analyst",
                "ranker": "ranker",
                "risk_manager": "risk_manager",
                "decision_maker": "decision_maker",
                "end": END
            }
        )

        # Connect all agents back to supervisor
        workflow.add_edge("news_analyst", "supervisor")
        workflow.add_edge("data_evaluator", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("ranker", "supervisor")
        workflow.add_edge("risk_manager", "supervisor")
        workflow.add_edge("decision_maker", "supervisor")

        return workflow.compile()

    def _supervisor_node(self, state: NexusState) -> NexusState:
        """
        Supervisor node - tracks progress and determines next step.

        Args:
            state: Current workflow state

        Returns:
            Updated state with next agent assignment
        """
        state.iteration += 1

        # Log progress
        current = state.current_agent
        matches_count = len(state.matches)
        top_count = len(state.top_matches)
        approved_count = len(state.approved_bets)

        message = (
            f"Iteration {state.iteration}: "
            f"Matches={matches_count}, Top={top_count}, Approved={approved_count}"
        )
        state = add_message(state, "supervisor", message)

        return state

    def _route_next(self, state: NexusState) -> str:
        """
        Determine the next agent to call based on current state.

        Args:
            state: Current workflow state

        Returns:
            Next agent name or "end"
        """
        # Safety: max iterations
        if state.iteration > 10:
            return "end"

        # No matches yet → collect news
        if not state.matches:
            return "news_analyst"

        # Have matches but not evaluated → evaluate quality
        has_quality_scores = all(m.data_quality is not None for m in state.matches)
        if not has_quality_scores:
            return "data_evaluator"

        # Have quality but no predictions → analyze
        has_predictions = all(m.prediction is not None for m in state.matches if m.data_quality and m.data_quality.overall_score >= 0.5)
        if not has_predictions:
            return "analyst"

        # Have predictions but not ranked → rank
        if not state.top_matches:
            return "ranker"

        # Have rankings but no risk assessment → assess risk
        has_value_bets = all(m.value_bet is not None for m in state.top_matches)
        if not has_value_bets:
            return "risk_manager"

        # Have risk assessment but no decisions → decide
        if not state.approved_bets and not state.rejected_matches:
            return "decision_maker"

        # All done
        state.completed_at = datetime.now()
        return "end"

    async def run(
        self,
        sport: str,
        date: str,
        bankroll: float = 1000.0
    ) -> NexusState:
        """
        Run the complete betting analysis workflow.

        Args:
            sport: Sport to analyze (tennis, basketball)
            date: Date to analyze (YYYY-MM-DD)
            bankroll: Current bankroll amount

        Returns:
            Final workflow state with recommendations
        """
        from core.state import Sport

        # Initialize state
        initial_state = NexusState(
            sport=Sport(sport),
            date=date,
            current_bankroll=bankroll,
            started_at=datetime.now()
        )

        # Create and run workflow
        workflow = self.create_workflow()

        # Execute workflow
        final_state = await workflow.ainvoke(initial_state)

        return final_state


# === HELPER FUNCTIONS ===

def create_supervisor() -> SupervisorAgent:
    """Create a new supervisor agent instance."""
    return SupervisorAgent()


async def run_betting_analysis(
    sport: str,
    date: str,
    bankroll: float = 1000.0
) -> Dict[str, Any]:
    """
    Convenience function to run full betting analysis.

    Args:
        sport: Sport type
        date: Analysis date
        bankroll: Current bankroll

    Returns:
        Dict with analysis results
    """
    supervisor = SupervisorAgent()
    state = await supervisor.run(sport, date, bankroll)

    return {
        "sport": sport,
        "date": date,
        "matches_analyzed": len(state.matches),
        "top_matches": len(state.top_matches),
        "approved_bets": [m.dict() for m in state.approved_bets],
        "rejected_matches": len(state.rejected_matches),
        "messages": state.messages,
        "duration_seconds": (state.completed_at - state.started_at).total_seconds() if state.completed_at else None
    }
