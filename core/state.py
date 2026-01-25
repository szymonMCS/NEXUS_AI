# core/state.py
"""
Pydantic state models for LangGraph workflow.
These models define the shared state passed between agents.
"""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# === ENUMS ===

class Sport(str, Enum):
    """Supported sports"""
    TENNIS = "tennis"
    BASKETBALL = "basketball"


class DataQualityLevel(str, Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"  # > 85%
    GOOD = "good"            # 70-85%
    MODERATE = "moderate"    # 50-70%
    HIGH_RISK = "high_risk"  # 40-50%
    INSUFFICIENT = "insufficient"  # < 40%


class BetDecision(str, Enum):
    """Bet decision status"""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


# === DATA MODELS ===

class NewsArticle(BaseModel):
    """Single news article"""
    title: str
    url: str
    source: str
    published_date: datetime
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    mentions_player1: bool = False
    mentions_player2: bool = False
    mentions_injury: bool = False


class MatchOdds(BaseModel):
    """Odds from a single bookmaker"""
    bookmaker: str
    home_odds: float = Field(gt=1.0)
    away_odds: float = Field(gt=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class PlayerStats(BaseModel):
    """Player/Team statistics"""
    name: str
    ranking: Optional[int] = None
    form: Optional[str] = None  # "W-L-W-W-L" format
    win_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_matches: List[Dict] = Field(default_factory=list)
    h2h_wins: int = 0
    h2h_losses: int = 0
    injury_status: Optional[str] = None


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment"""
    news_score: float = Field(ge=0.0, le=1.0)
    odds_score: float = Field(ge=0.0, le=1.0)
    stats_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    quality_level: DataQualityLevel
    issues: List[str] = Field(default_factory=list)
    sources_count: int = 0


class PredictionResult(BaseModel):
    """Prediction from analyst agent"""
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = "v1.0"
    factors: Dict[str, float] = Field(default_factory=dict)


class ValueBet(BaseModel):
    """Value bet calculation"""
    bet_on: Literal["home", "away"]
    odds: float
    true_probability: float
    edge: float = Field(ge=0.0)  # Expected value percentage
    kelly_stake: float = Field(ge=0.0, le=1.0)  # Fraction of bankroll
    confidence: float = Field(ge=0.0, le=1.0)


class Match(BaseModel):
    """Complete match information"""
    match_id: str
    sport: Sport
    date: datetime
    league: str
    home_player: PlayerStats
    away_player: PlayerStats

    # Data collection
    news_articles: List[NewsArticle] = Field(default_factory=list)
    odds: List[MatchOdds] = Field(default_factory=list)

    # Analysis results
    data_quality: Optional[DataQualityMetrics] = None
    prediction: Optional[PredictionResult] = None
    value_bet: Optional[ValueBet] = None

    # Decision
    recommended: BetDecision = BetDecision.PENDING
    rejection_reason: Optional[str] = None
    composite_score: float = 0.0  # For ranking


# === LANGGRAPH STATE ===

class NexusState(BaseModel):
    """
    Main state object passed between LangGraph agents.
    This is the shared context for the entire workflow.
    """

    # Input
    sport: Sport
    date: str  # YYYY-MM-DD format

    # Workflow status
    current_agent: str = "supervisor"
    iteration: int = 0

    # Collected data
    matches: List[Match] = Field(default_factory=list)

    # Analysis results
    top_matches: List[Match] = Field(default_factory=list)  # Top 3-5

    # Decision tracking
    approved_bets: List[Match] = Field(default_factory=list)
    rejected_matches: List[Match] = Field(default_factory=list)

    # Agent messages/logs
    messages: List[Dict[str, str]] = Field(default_factory=list)

    # Bankroll management
    current_bankroll: float = 1000.0
    max_stake_per_bet: float = 50.0

    # Betting session
    betting_session: Optional[Dict[str, Any]] = None

    # Metadata
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


# === AGENT-SPECIFIC MODELS ===

class NewsAnalystOutput(BaseModel):
    """Output from News Analyst agent"""
    matches_with_news: List[Match]
    total_articles_found: int
    average_relevance: float


class DataEvaluatorOutput(BaseModel):
    """Output from Data Evaluator agent"""
    evaluated_matches: List[Match]
    high_quality_count: int
    rejected_count: int


class AnalystOutput(BaseModel):
    """Output from Analyst agent"""
    analyzed_matches: List[Match]
    predictions_made: int


class RankerOutput(BaseModel):
    """Output from Match Ranker agent"""
    top_matches: List[Match]
    ranking_criteria: Dict[str, float]


class DecisionMakerOutput(BaseModel):
    """Output from Decision Maker agent"""
    approved_bets: List[Match]
    rejected_bets: List[Match]
    total_stake: float


# === UTILITY FUNCTIONS ===

def add_message(state: NexusState, agent: str, message: str) -> NexusState:
    """Helper to add agent message to state"""
    state.messages.append({
        "agent": agent,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    return state


def get_match_by_id(state: NexusState, match_id: str) -> Optional[Match]:
    """Helper to find match by ID"""
    for match in state.matches:
        if match.match_id == match_id:
            return match
    return None
