# database/crud.py
"""
CRUD operations for database models.
Provides high-level functions for common database operations.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from database.models import (
    Match, Odds, Prediction, Bet, News, MatchStats,
    BettingSession, SystemMetrics,
    SportType, OddsType, BetStatus, QualityLevel
)
from core.state import Match as PydanticMatch


# === MATCH CRUD ===

def create_match(db: Session, match_data: Dict[str, Any]) -> Match:
    """Create a new match"""
    match = Match(**match_data)
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


def get_match_by_id(db: Session, match_id: int) -> Optional[Match]:
    """Get match by ID"""
    return db.query(Match).filter(Match.id == match_id).first()


def get_match_by_external_id(db: Session, external_id: str) -> Optional[Match]:
    """Get match by external ID (from API)"""
    return db.query(Match).filter(Match.external_id == external_id).first()


def get_upcoming_matches(
    db: Session,
    sport: Optional[SportType] = None,
    hours_ahead: int = 24,
    min_quality: Optional[QualityLevel] = None
) -> List[Match]:
    """Get upcoming matches"""
    now = datetime.now()
    future = now + timedelta(hours=hours_ahead)

    query = db.query(Match).filter(
        and_(
            Match.start_time >= now,
            Match.start_time <= future,
            Match.is_finished == False
        )
    )

    if sport:
        query = query.filter(Match.sport == sport)

    if min_quality:
        quality_order = ["excellent", "good", "moderate", "high_risk", "insufficient"]
        min_index = quality_order.index(min_quality.value)
        allowed_qualities = quality_order[:min_index + 1]
        query = query.filter(Match.quality_level.in_(allowed_qualities))

    return query.order_by(Match.start_time).all()


def update_match_quality(
    db: Session,
    match_id: int,
    quality_score: float,
    quality_level: QualityLevel,
    sources_count: int
) -> Match:
    """Update match quality metrics"""
    match = get_match_by_id(db, match_id)
    if match:
        match.quality_score = quality_score
        match.quality_level = quality_level
        match.data_sources_count = sources_count
        db.commit()
        db.refresh(match)
    return match


def update_match_result(
    db: Session,
    match_id: int,
    home_score: int,
    away_score: int,
    winner: str
) -> Match:
    """Update match result after completion"""
    match = get_match_by_id(db, match_id)
    if match:
        match.home_score = home_score
        match.away_score = away_score
        match.winner = winner
        match.is_finished = True
        db.commit()
        db.refresh(match)
    return match


# === ODDS CRUD ===

def create_odds(db: Session, odds_data: Dict[str, Any]) -> Odds:
    """Create odds record"""
    # Mark previous odds as not current
    if "match_id" in odds_data and "bookmaker" in odds_data and "odds_type" in odds_data:
        db.query(Odds).filter(
            and_(
                Odds.match_id == odds_data["match_id"],
                Odds.bookmaker == odds_data["bookmaker"],
                Odds.odds_type == odds_data["odds_type"]
            )
        ).update({"is_current": False})

    odds = Odds(**odds_data)
    db.add(odds)
    db.commit()
    db.refresh(odds)
    return odds


def get_current_odds(
    db: Session,
    match_id: int,
    odds_type: Optional[OddsType] = None
) -> List[Odds]:
    """Get current odds for a match"""
    query = db.query(Odds).filter(
        and_(
            Odds.match_id == match_id,
            Odds.is_current == True
        )
    )

    if odds_type:
        query = query.filter(Odds.odds_type == odds_type)

    return query.all()


def get_best_odds(db: Session, match_id: int, selection: str) -> Optional[Odds]:
    """
    Get best odds for a selection (highest odds).

    Args:
        match_id: Match ID
        selection: "home" or "away"
    """
    odds_list = get_current_odds(db, match_id, OddsType.MONEYLINE)

    if not odds_list:
        return None

    if selection == "home":
        return max(odds_list, key=lambda x: x.home_odds or 0)
    elif selection == "away":
        return max(odds_list, key=lambda x: x.away_odds or 0)

    return None


# === PREDICTION CRUD ===

def create_prediction(db: Session, prediction_data: Dict[str, Any]) -> Prediction:
    """Create prediction"""
    prediction = Prediction(**prediction_data)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_predictions_for_match(db: Session, match_id: int) -> List[Prediction]:
    """Get all predictions for a match"""
    return db.query(Prediction).filter(Prediction.match_id == match_id).all()


def get_latest_prediction(db: Session, match_id: int) -> Optional[Prediction]:
    """Get latest prediction for a match"""
    return db.query(Prediction).filter(
        Prediction.match_id == match_id
    ).order_by(desc(Prediction.created_at)).first()


# === BET CRUD ===

def create_bet(db: Session, bet_data: Dict[str, Any]) -> Bet:
    """Create bet record"""
    bet = Bet(**bet_data)
    db.add(bet)
    db.commit()
    db.refresh(bet)
    return bet


def get_pending_bets(db: Session) -> List[Bet]:
    """Get all pending bets"""
    return db.query(Bet).filter(Bet.status == BetStatus.PENDING).all()


def settle_bet(
    db: Session,
    bet_id: int,
    status: BetStatus,
    profit_loss: float
) -> Bet:
    """Settle a bet with result"""
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    if bet:
        bet.status = status
        bet.profit_loss = profit_loss
        bet.settled_at = datetime.now()
        db.commit()
        db.refresh(bet)
    return bet


def get_bets_for_session(
    db: Session,
    session_id: int,
    status: Optional[BetStatus] = None
) -> List[Bet]:
    """Get bets for a betting session"""
    # Note: You'd need to add session_id to Bet model if tracking sessions
    query = db.query(Bet)

    if status:
        query = query.filter(Bet.status == status)

    return query.all()


# === NEWS CRUD ===

def create_news(db: Session, news_data: Dict[str, Any]) -> News:
    """Create news article"""
    news = News(**news_data)
    db.add(news)
    db.commit()
    db.refresh(news)
    return news


def get_news_for_match(db: Session, match_id: int, min_relevance: float = 0.5) -> List[News]:
    """Get relevant news for a match"""
    return db.query(News).filter(
        and_(
            News.match_id == match_id,
            News.relevance_score >= min_relevance
        )
    ).order_by(desc(News.published_at)).all()


# === MATCH STATS CRUD ===

def create_or_update_match_stats(db: Session, match_id: int, stats_data: Dict[str, Any]) -> MatchStats:
    """Create or update match statistics"""
    stats = db.query(MatchStats).filter(MatchStats.match_id == match_id).first()

    if stats:
        # Update existing
        for key, value in stats_data.items():
            setattr(stats, key, value)
    else:
        # Create new
        stats = MatchStats(match_id=match_id, **stats_data)
        db.add(stats)

    db.commit()
    db.refresh(stats)
    return stats


# === BETTING SESSION CRUD ===

def create_betting_session(db: Session, session_data: Dict[str, Any]) -> BettingSession:
    """Create new betting session"""
    session = BettingSession(**session_data)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_active_session(db: Session) -> Optional[BettingSession]:
    """Get currently active betting session"""
    return db.query(BettingSession).filter(
        BettingSession.is_active == True
    ).first()


def update_session_metrics(db: Session, session_id: int) -> BettingSession:
    """Recalculate session metrics from bets"""
    session = db.query(BettingSession).filter(BettingSession.id == session_id).first()

    if not session:
        return None

    # Get all bets in session date range
    bets = db.query(Bet).filter(
        and_(
            Bet.placed_at >= session.start_date,
            Bet.placed_at <= (session.end_date or datetime.now())
        )
    ).all()

    # Calculate metrics
    session.total_bets = len(bets)
    session.winning_bets = sum(1 for b in bets if b.status == BetStatus.WON)
    session.losing_bets = sum(1 for b in bets if b.status == BetStatus.LOST)
    session.void_bets = sum(1 for b in bets if b.status == BetStatus.VOID)

    session.total_staked = sum(b.stake for b in bets)
    session.total_profit = sum(b.profit_loss or 0 for b in bets)

    if session.total_staked > 0:
        session.roi_percentage = (session.total_profit / session.total_staked) * 100

    session.current_bankroll = session.starting_bankroll + session.total_profit

    db.commit()
    db.refresh(session)
    return session


# === SYSTEM METRICS CRUD ===

def create_or_update_daily_metrics(db: Session, date: datetime, metrics: Dict[str, Any]) -> SystemMetrics:
    """Create or update daily system metrics"""
    existing = db.query(SystemMetrics).filter(
        func.date(SystemMetrics.date) == date.date()
    ).first()

    if existing:
        for key, value in metrics.items():
            setattr(existing, key, value)
        metric = existing
    else:
        metric = SystemMetrics(date=date, **metrics)
        db.add(metric)

    db.commit()
    db.refresh(metric)
    return metric


def get_metrics_range(db: Session, start_date: datetime, end_date: datetime) -> List[SystemMetrics]:
    """Get system metrics for a date range"""
    return db.query(SystemMetrics).filter(
        and_(
            SystemMetrics.date >= start_date,
            SystemMetrics.date <= end_date
        )
    ).order_by(SystemMetrics.date).all()


# === ANALYTICS ===

def get_roi_by_sport(db: Session, days: int = 30) -> Dict[str, float]:
    """Calculate ROI by sport for last N days"""
    since = datetime.now() - timedelta(days=days)

    results = db.query(
        Match.sport,
        func.sum(Bet.stake).label("total_staked"),
        func.sum(Bet.profit_loss).label("total_profit")
    ).join(
        Bet, Match.id == Bet.match_id
    ).filter(
        and_(
            Bet.placed_at >= since,
            Bet.status.in_([BetStatus.WON, BetStatus.LOST])
        )
    ).group_by(Match.sport).all()

    roi_by_sport = {}
    for sport, staked, profit in results:
        if staked and staked > 0:
            roi_by_sport[sport.value] = (profit / staked) * 100

    return roi_by_sport


def get_performance_summary(db: Session, days: int = 30) -> Dict[str, Any]:
    """Get comprehensive performance summary"""
    since = datetime.now() - timedelta(days=days)

    bets = db.query(Bet).filter(
        and_(
            Bet.placed_at >= since,
            Bet.status.in_([BetStatus.WON, BetStatus.LOST])
        )
    ).all()

    if not bets:
        return {
            "total_bets": 0,
            "win_rate": 0,
            "roi": 0,
            "total_profit": 0,
        }

    total_bets = len(bets)
    won = sum(1 for b in bets if b.status == BetStatus.WON)
    total_staked = sum(b.stake for b in bets)
    total_profit = sum(b.profit_loss or 0 for b in bets)

    return {
        "total_bets": total_bets,
        "winning_bets": won,
        "losing_bets": total_bets - won,
        "win_rate": (won / total_bets * 100) if total_bets > 0 else 0,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": (total_profit / total_staked * 100) if total_staked > 0 else 0,
        "avg_stake": total_staked / total_bets if total_bets > 0 else 0,
        "avg_profit_per_bet": total_profit / total_bets if total_bets > 0 else 0,
    }
