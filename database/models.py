# database/models.py
"""
SQLAlchemy ORM models for NEXUS AI.
Stores matches, odds, predictions, bets, and performance metrics.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text, Enum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


# === ENUMS ===

class SportType(str, enum.Enum):
    """Sport types"""
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    FOOTBALL = "football"
    ICE_HOCKEY = "ice_hockey"
    BASEBALL = "baseball"
    ESPORTS = "esports"


class OddsType(str, enum.Enum):
    """Types of odds"""
    MONEYLINE = "moneyline"
    HANDICAP = "handicap"
    TOTALS = "totals"  # Over/Under
    DRAW_NO_BET = "draw_no_bet"


class BetStatus(str, enum.Enum):
    """Bet lifecycle status"""
    PENDING = "pending"
    PLACED = "placed"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    CASHOUT = "cashout"


class QualityLevel(str, enum.Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    HIGH_RISK = "high_risk"
    INSUFFICIENT = "insufficient"


# === CORE MODELS ===

class Match(Base):
    """Match/Event entity"""
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)  # From API
    sport = Column(Enum(SportType), nullable=False, index=True)

    # Match details
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False, index=True)
    country = Column(String, nullable=True)

    # Timing
    start_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Status
    is_live = Column(Boolean, default=False)
    is_finished = Column(Boolean, default=False)

    # Result (after match)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    winner = Column(String, nullable=True)  # "home", "away", "draw"

    # Data quality
    quality_score = Column(Float, nullable=True)  # 0.0 - 1.0
    quality_level = Column(Enum(QualityLevel), nullable=True)
    data_sources_count = Column(Integer, default=0)

    # Relationships
    odds = relationship("Odds", back_populates="match", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="match", cascade="all, delete-orphan")
    bets = relationship("Bet", back_populates="match", cascade="all, delete-orphan")
    news = relationship("News", back_populates="match", cascade="all, delete-orphan")
    stats = relationship("MatchStats", back_populates="match", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Match {self.home_team} vs {self.away_team} @ {self.start_time}>"


class Odds(Base):
    """Odds from bookmakers - supports multiple odds types"""
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    # Bookmaker info
    bookmaker = Column(String, nullable=False, index=True)
    odds_type = Column(Enum(OddsType), nullable=False)

    # Moneyline odds
    home_odds = Column(Float, nullable=True)
    away_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)

    # Handicap/Spread
    handicap_line = Column(Float, nullable=True)  # e.g., -1.5, +2.5
    handicap_home_odds = Column(Float, nullable=True)
    handicap_away_odds = Column(Float, nullable=True)

    # Totals (Over/Under)
    total_line = Column(Float, nullable=True)  # e.g., 2.5 goals
    over_odds = Column(Float, nullable=True)
    under_odds = Column(Float, nullable=True)

    # Metadata
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    max_stake = Column(Float, nullable=True)  # Max bet allowed
    is_current = Column(Boolean, default=True)  # Latest odds

    # Relationships
    match = relationship("Match", back_populates="odds")

    def __repr__(self):
        return f"<Odds {self.bookmaker} - {self.odds_type} for Match {self.match_id}>"


class Prediction(Base):
    """AI prediction for a match"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    # Probabilities
    home_win_prob = Column(Float, nullable=False)  # 0.0 - 1.0
    away_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=True)  # For sports with draws

    # Prediction details
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    model_version = Column(String, default="v1.0")

    # Value calculation
    recommended_bet = Column(String, nullable=True)  # "home", "away", "over", etc.
    edge_percentage = Column(Float, nullable=True)  # Expected value %
    kelly_stake = Column(Float, nullable=True)  # Kelly criterion stake (fraction)

    # Factors (JSON)
    factors = Column(JSON, nullable=True)  # {"ranking": 0.3, "form": 0.5, ...}

    # Metadata
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    match = relationship("Match", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction Match {self.match_id} - {self.home_win_prob:.2%} vs {self.away_win_prob:.2%}>"


class Bet(Base):
    """Placed bet record"""
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    # Bet details
    bet_type = Column(Enum(OddsType), nullable=False)
    selection = Column(String, nullable=False)  # "home", "away", "over", etc.
    odds = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)

    # Outcome
    status = Column(Enum(BetStatus), default=BetStatus.PENDING, index=True)
    profit_loss = Column(Float, nullable=True)  # Actual P/L

    # Risk management
    bankroll_at_time = Column(Float, nullable=True)
    stake_percentage = Column(Float, nullable=True)  # % of bankroll
    confidence_at_time = Column(Float, nullable=True)
    edge_at_time = Column(Float, nullable=True)

    # Bookmaker
    bookmaker = Column(String, nullable=True)
    bet_reference = Column(String, nullable=True)  # External bet ID

    # Timing
    placed_at = Column(DateTime, server_default=func.now(), index=True)
    settled_at = Column(DateTime, nullable=True)

    # Notes
    notes = Column(Text, nullable=True)

    # Relationships
    match = relationship("Match", back_populates="bets")

    def __repr__(self):
        return f"<Bet {self.selection} @ {self.odds} - {self.status.value}>"


class News(Base):
    """News articles related to matches"""
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=True, index=True)

    # Article details
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False, index=True)
    snippet = Column(Text, nullable=True)

    # Analysis
    relevance_score = Column(Float, nullable=True)  # 0.0 - 1.0
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    mentions_injury = Column(Boolean, default=False)
    mentions_lineup = Column(Boolean, default=False)

    # Metadata
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, server_default=func.now())

    # Relationships
    match = relationship("Match", back_populates="news")

    def __repr__(self):
        return f"<News {self.source} - {self.title[:50]}>"


class MatchStats(Base):
    """Detailed statistics for a match"""
    __tablename__ = "match_stats"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True, index=True)

    # Home team stats
    home_ranking = Column(Integer, nullable=True)
    home_form = Column(String, nullable=True)  # "WWLWD"
    home_win_rate = Column(Float, nullable=True)
    home_recent_goals_avg = Column(Float, nullable=True)

    # Away team stats
    away_ranking = Column(Integer, nullable=True)
    away_form = Column(String, nullable=True)
    away_win_rate = Column(Float, nullable=True)
    away_recent_goals_avg = Column(Float, nullable=True)

    # Head-to-head
    h2h_matches = Column(Integer, default=0)
    h2h_home_wins = Column(Integer, default=0)
    h2h_away_wins = Column(Integer, default=0)
    h2h_draws = Column(Integer, default=0)

    # Additional data (JSON)
    extra_stats = Column(JSON, nullable=True)

    # Metadata
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    match = relationship("Match", back_populates="stats")

    def __repr__(self):
        return f"<MatchStats for Match {self.match_id}>"


class BettingSession(Base):
    """Tracking betting sessions for ROI calculation"""
    __tablename__ = "betting_sessions"

    id = Column(Integer, primary_key=True, index=True)

    # Session details
    session_name = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=True)

    # Bankroll
    starting_bankroll = Column(Float, nullable=False)
    current_bankroll = Column(Float, nullable=False)

    # Performance metrics
    total_bets = Column(Integer, default=0)
    winning_bets = Column(Integer, default=0)
    losing_bets = Column(Integer, default=0)
    void_bets = Column(Integer, default=0)

    total_staked = Column(Float, default=0.0)
    total_profit = Column(Float, default=0.0)
    roi_percentage = Column(Float, default=0.0)  # Return on Investment

    # Win rates by sport
    sport_performance = Column(JSON, nullable=True)  # {"tennis": {"roi": 15.2, ...}, ...}

    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Notes
    description = Column(Text, nullable=True)

    def __repr__(self):
        return f"<BettingSession {self.session_name} - ROI: {self.roi_percentage:.2f}%>"


class SystemMetrics(Base):
    """System-wide performance metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Date
    date = Column(DateTime, nullable=False, unique=True, index=True)

    # Match processing
    matches_analyzed = Column(Integer, default=0)
    matches_predicted = Column(Integer, default=0)
    matches_rejected = Column(Integer, default=0)

    # Quality metrics
    avg_data_quality = Column(Float, nullable=True)
    avg_confidence = Column(Float, nullable=True)

    # Predictions
    predictions_made = Column(Integer, default=0)
    avg_edge = Column(Float, nullable=True)

    # Bets
    bets_placed = Column(Integer, default=0)
    bets_won = Column(Integer, default=0)
    bets_lost = Column(Integer, default=0)
    daily_roi = Column(Float, nullable=True)

    # News & data
    news_articles_fetched = Column(Integer, default=0)
    odds_updates_received = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<SystemMetrics {self.date.date()} - {self.matches_analyzed} matches>"
