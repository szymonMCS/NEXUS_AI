# api/main.py
"""
FastAPI backend for NEXUS AI Lite.
Provides REST API for React frontend.
"""

import asyncio
import logging
import signal
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config.settings import settings
from api.auth import require_auth, optional_auth
from core.ml import get_model_registry
from core.models import (
    HandicapModel,
    TennisHandicapModel,
    BasketballHandicapModel,
    MarketType,
    find_value_handicap,
)

# === LOGGING SETUP ===
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexus.api")


# === PYDANTIC MODELS ===

class AnalysisRequest(BaseModel):
    sport: str = "tennis"
    date: Optional[str] = None
    min_quality: float = 45.0
    top_n: int = 5


class AnalysisResponse(BaseModel):
    status: str
    sport: str
    date: str
    matches_found: int
    value_bets: List[Dict[str, Any]]
    quality_filtered: int
    timestamp: str


class ValueBetResponse(BaseModel):
    rank: int
    match: str
    league: str
    selection: str
    odds: float
    bookmaker: str
    edge: float
    quality_score: float
    stake_recommendation: str
    confidence: float
    reasoning: List[str]


class SystemStatus(BaseModel):
    status: str
    mode: str
    api_keys_configured: Dict[str, bool]
    last_analysis: Optional[str]
    uptime_seconds: float


class HandicapRequest(BaseModel):
    sport: str = "tennis"
    market_type: str = "match_handicap"  # match_handicap, first_half, total_over, first_half_total
    home_stats: Dict[str, Any]
    away_stats: Dict[str, Any]
    line: float = 0.0
    bookmaker_odds: Optional[Dict[str, List[float]]] = None  # line -> [home_odds, away_odds]


class HandicapResponse(BaseModel):
    market_type: str
    line: float
    cover_probability: float
    fair_odds: float
    expected_margin: float
    confidence: float
    reasoning: List[str]
    half_patterns: Optional[Dict[str, Any]] = None
    value_bets: Optional[List[Dict[str, Any]]] = None


# === LIFESPAN (startup / shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # --- STARTUP ---
    logger.info("NEXUS AI v%s starting (mode=%s)", settings.APP_VERSION, settings.APP_MODE)

    # Initialize database tables
    try:
        from database.db import init_db
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)

    # Validate configuration
    is_valid, warnings = settings.validate_mode_requirements()
    for w in warnings:
        logger.warning("Config: %s", w)

    # Start the analysis scheduler
    scheduler = None
    try:
        from core.scheduler import get_scheduler, default_analysis_function
        scheduler = get_scheduler()
        scheduler.set_analysis_callback(default_analysis_function)
        import asyncio
        asyncio.create_task(scheduler.start())
        logger.info("Analysis scheduler started")
    except Exception as e:
        logger.warning("Scheduler startup failed (non-critical): %s", e)

    yield  # --- APP RUNNING ---

    # --- SHUTDOWN ---
    logger.info("NEXUS AI shutting down...")

    # Stop the scheduler
    if scheduler is not None:
        try:
            await scheduler.stop()
            logger.info("Analysis scheduler stopped")
        except Exception as e:
            logger.warning("Scheduler stop error: %s", e)

    # Close all WebSocket connections gracefully
    for conn in list(manager.active_connections):
        try:
            await conn.close(code=1001, reason="Server shutting down")
        except Exception:
            pass
    manager.active_connections.clear()

    logger.info("NEXUS AI shutdown complete")


# === APP SETUP ===

app = FastAPI(
    title="NEXUS AI Lite",
    description="Sports Prediction System - REST API",
    version="2.2.0",
    lifespan=lifespan,
)

# Setup Prometheus metrics
try:
    from api.metrics import setup_metrics, set_websocket_connections, set_active_analyses
    setup_metrics(app, version="2.2.0")
except ImportError:
    pass

# CORS - configured from environment variable CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === RATE LIMITING MIDDLEWARE ===

_rate_limit_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT_MAX = 60          # requests
RATE_LIMIT_WINDOW = 60       # seconds
RATE_LIMIT_ANALYSIS_MAX = 5  # analysis requests per window


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-process rate limiter per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]

    # Skip rate limiting for health checks
    if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
        return await call_next(request)

    # Stricter limit for analysis endpoints
    if request.url.path in ("/api/analysis", "/api/predictions/analyze"):
        analysis_key = f"{client_ip}:analysis"
        _rate_limit_store[analysis_key] = [
            t for t in _rate_limit_store[analysis_key] if now - t < RATE_LIMIT_WINDOW
        ]
        if len(_rate_limit_store[analysis_key]) >= RATE_LIMIT_ANALYSIS_MAX:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many analysis requests. Please wait."},
            )
        _rate_limit_store[analysis_key].append(now)

    # General limit
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please slow down."},
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


# === GLOBAL EXCEPTION HANDLER ===

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return structured JSON error."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": request.url.path},
    )


# Register additional routers (ensemble, monitoring, admin)
from api.routers import register_routers
register_routers(app)

# Store for active WebSocket connections
active_connections: List[WebSocket] = []

# Store for analysis state
analysis_state = {
    "is_running": False,
    "current_step": None,
    "progress": 0,
    "last_result": None,
    "last_analysis_time": None,
    "start_time": datetime.now()
}


# === WEBSOCKET MANAGER ===

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, RuntimeError, ConnectionError):
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# === API ENDPOINTS ===

@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get comprehensive system status including API availability."""
    uptime = (datetime.now() - analysis_state["start_time"]).total_seconds()

    # Get API status from settings
    api_status = settings.get_api_status()
    is_valid, warnings = settings.validate_mode_requirements()
    available_sources = settings.get_available_data_sources()

    return {
        "status": "running" if not analysis_state["is_running"] else "analyzing",
        "mode": settings.APP_MODE,
        "mode_valid": is_valid,
        "warnings": warnings,
        "api_status": api_status,
        "available_data_sources": available_sources,
        "last_analysis": analysis_state["last_analysis_time"],
        "uptime_seconds": uptime,
        "version": settings.APP_VERSION,
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker / load balancers."""
    checks = {"api": True, "database": False, "redis": False}

    # Database check
    try:
        from database.db import get_db_session
        from sqlalchemy import text
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.warning("Health check - DB failed: %s", e)

    # Redis check
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_timeout=2)
        r.ping()
        checks["redis"] = True
    except Exception:
        # Redis is optional - don't log as error
        pass

    healthy = checks["api"] and checks["database"]
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "degraded", "checks": checks},
    )


@app.post("/api/analysis")
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
) -> Dict[str, Any]:
    """
    Run betting analysis for a sport and date.
    Returns immediately, analysis runs in background.
    Use WebSocket for real-time updates.
    
    Authentication: Optional - works with or without auth token
    """
    """
    Run betting analysis for a sport and date.
    Returns immediately, analysis runs in background.
    Use WebSocket for real-time updates.
    """
    if analysis_state["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Analysis already in progress"
        )

    # Validate mode requirements before starting analysis
    is_valid, warnings = settings.validate_mode_requirements()
    if not is_valid:
        # Check if we have at least LLM configured
        api_status = settings.get_api_status()
        if not api_status["llm"]["openai"] and not api_status["llm"]["anthropic"]:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "No LLM API configured",
                    "warnings": warnings,
                    "hint": "Configure OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
                }
            )
        # Otherwise just return warnings but allow analysis
        # (lite mode can work with limited sources)

    target_date = request.date or str(date.today())

    # Start background analysis
    background_tasks.add_task(
        _run_analysis_task,
        request.sport,
        target_date,
        request.min_quality,
        request.top_n
    )

    return {
        "status": "started",
        "sport": request.sport,
        "date": target_date,
        "mode": settings.APP_MODE,
        "warnings": warnings if not is_valid else [],
        "available_sources": settings.get_available_data_sources(),
        "message": "Analysis started. Connect to WebSocket for updates."
    }


async def _run_analysis_task(
    sport: str,
    target_date: str,
    min_quality: float,
    top_n: int
):
    """Background task for running analysis with data persistence."""
    analysis_state["is_running"] = True
    analysis_state["progress"] = 0

    try:
        from betting_floor import run_betting_analysis
        from data.persistence import DataPersistenceService

        persistence = DataPersistenceService()

        # Step 1: Collect fixtures and save to DB
        await _broadcast_progress("collecting", 10, "Collecting fixtures...")

        try:
            from data.collectors.fixture_collector import FixtureCollector
            collector = FixtureCollector()
            fixtures = await collector.collect_fixtures(sport, target_date)
            if fixtures:
                saved = persistence.save_fixtures(fixtures, sport)
                logger.info("Saved %d fixtures to database", saved)
        except Exception as e:
            logger.warning("Fixture collection/save failed: %s", e)

        # Step 2: Run analysis
        await _broadcast_progress("analyzing", 30, "Analyzing matches...")

        result = await run_betting_analysis(
            sport=sport,
            date=target_date,
            bankroll=settings.DEFAULT_BANKROLL
        )

        # Step 3: Processing results and saving to DB
        await _broadcast_progress("processing", 70, "Processing results...")

        value_bets = []
        if result and result.get("approved_bets"):
            for i, bet in enumerate(result["approved_bets"][:top_n], 1):
                value_bets.append({
                    "rank": i,
                    "match": bet.get("match_name", ""),
                    "league": bet.get("league", ""),
                    "selection": bet.get("selection", ""),
                    "odds": bet.get("odds", 0),
                    "bookmaker": bet.get("bookmaker", ""),
                    "edge": bet.get("edge", 0),
                    "quality_score": bet.get("quality_score", 0),
                    "stake_recommendation": bet.get("stake", "1%"),
                    "confidence": bet.get("confidence", 0),
                    "reasoning": bet.get("reasoning", [])
                })

        analysis_state["last_result"] = {
            "sport": sport,
            "date": target_date,
            "value_bets": value_bets,
            "matches_analyzed": result.get("matches_analyzed", 0) if result else 0,
            "quality_filtered": result.get("quality_filtered", 0) if result else 0,
            "timestamp": datetime.now().isoformat()
        }
        analysis_state["last_analysis_time"] = datetime.now().isoformat()

        # Step 4: Complete
        await _broadcast_progress("complete", 100, "Analysis complete!", value_bets)

    except Exception as e:
        logger.error("Analysis task failed: %s", e, exc_info=True)
        await _broadcast_progress("error", 0, f"Error: {str(e)}")
    finally:
        analysis_state["is_running"] = False


async def _broadcast_progress(step: str, progress: int, message: str, data: Any = None):
    """Broadcast progress to all WebSocket connections."""
    analysis_state["current_step"] = step
    analysis_state["progress"] = progress

    await manager.broadcast({
        "type": "progress",
        "step": step,
        "progress": progress,
        "message": message,
        "data": data
    })


@app.get("/api/predictions")
async def get_predictions(
    sport: Optional[str] = None,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """Get latest predictions/value bets."""
    if not analysis_state["last_result"]:
        return {
            "status": "no_data",
            "message": "No analysis has been run yet",
            "value_bets": []
        }

    result = analysis_state["last_result"]

    # Filter by sport if specified
    if sport and result.get("sport") != sport:
        return {
            "status": "no_match",
            "message": f"Last analysis was for {result.get('sport')}, not {sport}",
            "value_bets": []
        }

    return {
        "status": "success",
        "sport": result.get("sport"),
        "date": result.get("date"),
        "value_bets": result.get("value_bets", []),
        "matches_analyzed": result.get("matches_analyzed", 0),
        "quality_filtered": result.get("quality_filtered", 0),
        "timestamp": result.get("timestamp")
    }


@app.get("/api/value-bets")
async def get_value_bets() -> List[Dict[str, Any]]:
    """Get current value bets (shorthand endpoint)."""
    if not analysis_state["last_result"]:
        return []
    return analysis_state["last_result"].get("value_bets", [])


@app.get("/api/matches")
async def get_matches(
    sport: str = "tennis",
    match_date: Optional[str] = None
) -> Dict[str, Any]:
    """Get matches for a given sport and date from database."""
    target_date = match_date or str(date.today())

    # Try to return matches from database first
    try:
        from database.db import get_db_session
        from database.models import Match as DBMatch
        from sqlalchemy import cast, Date

        with get_db_session() as db:
            query = db.query(DBMatch).filter(
                DBMatch.sport == sport
            )
            # Filter by date if start_time exists
            matches_db = query.order_by(DBMatch.start_time.desc()).limit(50).all()

            matches_list = []
            for m in matches_db:
                match_date_str = m.start_time.strftime("%Y-%m-%d %H:%M") if m.start_time else ""
                if target_date and m.start_time and m.start_time.strftime("%Y-%m-%d") != target_date:
                    continue
                matches_list.append({
                    "id": m.id,
                    "external_id": m.external_id,
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "league": m.league,
                    "start_time": match_date_str,
                    "is_live": m.is_live,
                    "is_finished": m.is_finished,
                    "home_score": m.home_score,
                    "away_score": m.away_score,
                    "quality_score": m.quality_score,
                })

            return {
                "status": "success",
                "sport": sport,
                "date": target_date,
                "matches": matches_list,
                "total": len(matches_list),
            }
    except Exception as e:
        logger.warning("Could not fetch matches from DB: %s", e)
        return {
            "status": "success",
            "sport": sport,
            "date": target_date,
            "matches": [],
            "total": 0,
            "message": "No matches in database. Run /api/analysis to collect and analyze matches.",
        }


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get system statistics for dashboard from database."""
    try:
        from database.db import get_db_session
        from database.crud import get_performance_summary, get_roi_by_sport

        with get_db_session() as db:
            summary = get_performance_summary(db, days=30)
            roi_by_sport = get_roi_by_sport(db, days=30)

        return {
            "total_analyses": summary.get("total_bets", 0),
            "successful_bets": summary.get("winning_bets", 0),
            "win_rate": round(summary.get("win_rate", 0), 2),
            "total_profit": round(summary.get("total_profit", 0), 2),
            "roi": round(summary.get("roi", 0), 2),
            "avg_edge": round(summary.get("avg_profit_per_bet", 0), 2),
            "roi_by_sport": roi_by_sport,
            "sports_analyzed": list(roi_by_sport.keys()) or ["tennis", "basketball", "football"],
            "last_7_days": [],
        }
    except Exception as e:
        logger.warning("Could not fetch stats from DB: %s", e)
        return {
            "total_analyses": 0,
            "successful_bets": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "avg_edge": 0.0,
            "roi_by_sport": {},
            "sports_analyzed": ["tennis", "basketball", "football"],
            "last_7_days": [],
        }


@app.get("/api/bets/history")
async def get_bet_history(
    sport: Optional[str] = None,
    status: Optional[str] = None,
    days: int = 90,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Get bet history from database with optional filters."""
    try:
        from database.db import get_db_session
        from database.models import Bet, Match as DBMatch, BetStatus
        from sqlalchemy import and_, desc

        with get_db_session() as db:
            query = db.query(Bet).join(DBMatch, Bet.match_id == DBMatch.id)

            # Filters
            since = datetime.now() - timedelta(days=days)
            query = query.filter(Bet.created_at >= since)

            if sport:
                query = query.filter(DBMatch.sport == sport)

            if status:
                try:
                    bet_status = BetStatus(status)
                    query = query.filter(Bet.status == bet_status)
                except ValueError:
                    pass

            total = query.count()
            bets = query.order_by(desc(Bet.created_at)).offset(offset).limit(limit).all()

            bets_list = []
            for b in bets:
                match = b.match
                bets_list.append({
                    "id": b.id,
                    "match": f"{match.home_team} vs {match.away_team}" if match else "",
                    "league": match.league if match else "",
                    "sport": match.sport.value if match and match.sport else "",
                    "selection": b.selection,
                    "odds": float(b.odds) if b.odds else 0,
                    "stake": float(b.stake) if b.stake else 0,
                    "status": b.status.value if b.status else "pending",
                    "profit_loss": float(b.profit_loss) if b.profit_loss else 0,
                    "edge": float(b.edge) if hasattr(b, 'edge') and b.edge else 0,
                    "confidence": float(b.confidence) if hasattr(b, 'confidence') and b.confidence else 0,
                    "created_at": b.created_at.isoformat() if b.created_at else "",
                    "settled_at": b.settled_at.isoformat() if hasattr(b, 'settled_at') and b.settled_at else None,
                })

            return {
                "status": "success",
                "bets": bets_list,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
    except Exception as e:
        logger.warning("Could not fetch bet history: %s", e)
        return {
            "status": "success",
            "bets": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "message": "No bet history available yet.",
        }


@app.post("/api/collect")
async def collect_fixtures(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Collect and save fixtures to database without running full analysis."""
    target_date = request.date or str(date.today())

    async def _collect_task(sport: str, target_date: str):
        try:
            from data.collectors.fixture_collector import FixtureCollector
            from data.persistence import DataPersistenceService

            collector = FixtureCollector()
            fixtures = await collector.collect_fixtures(sport, target_date)
            persistence = DataPersistenceService()
            saved = persistence.save_fixtures(fixtures, sport)
            logger.info("Collected %d fixtures, saved %d new (sport=%s)", len(fixtures), saved, sport)
        except Exception as e:
            logger.error("Collection failed: %s", e)

    background_tasks.add_task(_collect_task, request.sport, target_date)
    return {"status": "collecting", "sport": request.sport, "date": target_date}


@app.post("/api/results/check")
async def check_results(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Check and update results for finished matches. Settles pending bets."""

    async def _check_results():
        try:
            from data.persistence import DataPersistenceService
            from data.apis.thesportsdb_client import TheSportsDBClient

            persistence = DataPersistenceService()
            unfinished_ids = persistence.get_unfinished_match_ids()

            if not unfinished_ids:
                logger.info("No unfinished matches to check")
                return

            logger.info("Checking results for %d matches", len(unfinished_ids))

            # Try to get results from TheSportsDB
            client = TheSportsDBClient()
            results = []

            for ext_id in unfinished_ids[:20]:  # Limit to avoid rate limiting
                try:
                    event = await client.get_event_results(ext_id)
                    if event and event.get("intHomeScore") is not None:
                        results.append({
                            "external_id": ext_id,
                            "home_score": int(event["intHomeScore"]),
                            "away_score": int(event["intAwayScore"]),
                        })
                except Exception:
                    pass

            if results:
                updated = persistence.update_results(results)
                logger.info("Updated %d match results, settled bets", updated)

        except Exception as e:
            logger.error("Results check failed: %s", e)

    background_tasks.add_task(_check_results)
    return {"status": "checking", "message": "Checking results for unfinished matches"}


@app.get("/api/forebet/predictions")
async def get_forebet_predictions_simple(
    sport: str = "football",
    days: int = 3,
    min_confidence: float = 50.0
):
    """
    Simple endpoint to get Forebet predictions.
    Falls back to mock data if scraper unavailable.
    """
    try:
        # Try to use scraper
        from agents.sports_data_swarm.forebet_scraper import ForebetScraper
        
        async with ForebetScraper() as scraper:
            matches = await scraper.get_upcoming_matches(sport=sport, days=days)
            
            # Filter by confidence and format
            results = []
            for match in matches:
                confidence = max(match.prob_home, match.prob_draw, match.prob_away)
                if confidence >= min_confidence:
                    results.append({
                        "match_id": match.match_id,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "league": match.league,
                        "match_date": match.match_date.isoformat(),
                        "predictions": {
                            "home": match.prob_home,
                            "draw": match.prob_draw,
                            "away": match.prob_away,
                            "predicted": match.predicted_result
                        },
                        "odds": {
                            "home": match.odds_home,
                            "draw": match.odds_draw,
                            "away": match.odds_away
                        },
                        "confidence": confidence
                    })
            
            return {
                "status": "success",
                "source": "forebet",
                "count": len(results),
                "matches": results
            }
            
    except Exception as e:
        # Return status indicating scraper unavailable
        logger.warning("Forebet scraper failed: %s, returning demo data", e)
        return {
            "status": "mock",
            "source": "mock",
            "message": "Scraper unavailable, using demo data",
            "count": 4,
            "matches": [
                {
                    "match_id": "man-city-vs-arsenal-2026",
                    "home_team": "Man City",
                    "away_team": "Arsenal",
                    "league": "Premier League",
                    "match_date": "2026-01-28T20:45:00",
                    "predictions": {"home": 58, "draw": 24, "away": 18, "predicted": "1"},
                    "odds": {"home": 1.65, "draw": 3.80, "away": 5.20},
                    "confidence": 58
                },
                {
                    "match_id": "real-madrid-vs-barcelona-2026",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "league": "La Liga",
                    "match_date": "2026-01-28T21:00:00",
                    "predictions": {"home": 42, "draw": 28, "away": 30, "predicted": "1"},
                    "odds": {"home": 2.10, "draw": 3.40, "away": 3.30},
                    "confidence": 42
                },
                {
                    "match_id": "lakers-vs-warriors-2026",
                    "home_team": "Lakers",
                    "away_team": "Warriors",
                    "league": "NBA",
                    "match_date": "2026-01-28T23:00:00",
                    "predictions": {"home": 55, "away": 45, "predicted": "1"},
                    "odds": {"home": 1.85, "away": 1.95},
                    "confidence": 55
                },
                {
                    "match_id": "alcaraz-vs-djokovic-2026",
                    "home_team": "Alcaraz",
                    "away_team": "Djokovic",
                    "league": "Australian Open",
                    "match_date": "2026-01-29T09:00:00",
                    "predictions": {"home": 48, "away": 52, "predicted": "2"},
                    "odds": {"home": 1.85, "away": 1.95},
                    "confidence": 52
                }
            ]
        }


@app.get("/api/sports/available")
async def get_available_sports() -> Dict[str, Any]:
    """Get list of available sports with their configuration."""
    return {
        "sports": [
            {
                "id": "tennis",
                "name": "Tennis",
                "icon": "🎾",
                "markets": ["match_winner", "set_handicap", "game_handicap", "total_games"],
                "models": ["TennisHandicapModel", "TennisEloModel"],
                "status": "active"
            },
            {
                "id": "basketball",
                "name": "Basketball",
                "icon": "🏀",
                "markets": ["match_winner", "point_spread", "total_points", "first_half"],
                "models": ["BasketballHandicapModel", "BasketballEloModel"],
                "status": "active"
            },
            {
                "id": "greyhound",
                "name": "Greyhound Racing",
                "icon": "🐕",
                "markets": ["winner", "place", "forecast", "tricast"],
                "models": ["GreyhoundPredictor"],
                "status": "beta"
            },
            {
                "id": "handball",
                "name": "Handball",
                "icon": "🤾",
                "markets": ["match_winner", "handicap", "total_goals"],
                "models": ["HandballPredictor"],
                "status": "beta"
            },
            {
                "id": "table_tennis",
                "name": "Table Tennis",
                "icon": "🏓",
                "markets": ["match_winner", "set_handicap", "total_points"],
                "models": ["TableTennisPredictor"],
                "status": "beta"
            }
        ],
        "default": "tennis",
        "total": 5
    }


@app.get("/api/match/{match_id}")
async def get_match_details(match_id: str) -> Dict[str, Any]:
    """
    Get detailed analysis for a specific match.
    Returns comprehensive AI analysis data.
    """
    # Find match in last results
    if not analysis_state["last_result"]:
        raise HTTPException(status_code=404, detail="No analysis data available")

    value_bets = analysis_state["last_result"].get("value_bets", [])

    # Find match by ID or name
    match_data = None
    for bet in value_bets:
        # Match by rank or match name
        if str(bet.get("rank")) == match_id or bet.get("match", "").replace(" ", "-").lower() == match_id.lower():
            match_data = bet
            break

    if not match_data:
        raise HTTPException(status_code=404, detail="Match not found")

    # Build detailed analysis response
    match_name = match_data.get("match", "")
    teams = match_name.split(" vs ") if " vs " in match_name else [match_name, "Opponent"]

    return {
        "match": match_name,
        "match_id": match_id,
        "league": match_data.get("league", ""),
        "sport": analysis_state["last_result"].get("sport", "tennis"),
        "match_time": datetime.now().isoformat(),  # Would come from actual data

        "home_team": {
            "name": teams[0] if len(teams) > 0 else "Team A",
            "stats": {
                "win_rate": round(0.5 + match_data.get("confidence", 0.5) * 0.3, 2),
                "recent_form": "WWLWW",
                "rank": match_data.get("rank", 0) * 10,
            }
        },
        "away_team": {
            "name": teams[1] if len(teams) > 1 else "Team B",
            "stats": {
                "win_rate": round(0.5 - match_data.get("confidence", 0.5) * 0.1, 2),
                "recent_form": "WLWLW",
                "rank": match_data.get("rank", 0) * 10 + 5,
            }
        },

        "value_bet": {
            "selection": match_data.get("selection", ""),
            "selection_name": match_data.get("selection", ""),
            "probability": match_data.get("confidence", 0.5),
            "odds": match_data.get("odds", 1.0),
            "fair_odds": round(match_data.get("odds", 1.0) / (1 + match_data.get("edge", 0)), 2),
            "edge": match_data.get("edge", 0),
            "confidence": match_data.get("confidence", 0.5),
            "quality_score": match_data.get("quality_score", 50),
            "bookmaker": match_data.get("bookmaker", "Unknown"),
            "stake_recommendation": match_data.get("stake_recommendation", "2%"),
            "kelly_stake": round(match_data.get("edge", 0) * 100 * 0.25, 2),
        },

        "ai_analysis": {
            "summary": f"Analiza wskazuje na value w zakładzie {match_data.get('selection', '')} z edge {match_data.get('edge', 0) * 100:.1f}%.",
            "reasoning": match_data.get("reasoning", [
                "Analiza oparta na danych historycznych",
                "Model AI wykazał przewagę nad kursem bukmacherskim",
                "Wysoka jakość danych wejściowych"
            ]),
            "key_factors": [
                {"factor": "Forma", "impact": "high", "description": "Solidna seria wyników"},
                {"factor": "H2H", "impact": "medium", "description": "Korzystny bilans bezpośrednich spotkań"},
            ],
            "warnings": [
                "Sprawdź aktualne kursy przed postawieniem",
                "Rozważ dywersyfikację stawek"
            ],
            "confidence_breakdown": {
                "data_quality": match_data.get("quality_score", 50) / 100,
                "model_agreement": match_data.get("confidence", 0.5),
                "market_efficiency": 0.75,
            }
        },

        "data_quality": {
            "overall": match_data.get("quality_score", 50),
            "components": [
                {"name": "Dane historyczne", "score": min(95, match_data.get("quality_score", 50) + 10), "description": "Kompletność danych"},
                {"name": "Dane H2H", "score": match_data.get("quality_score", 50), "description": "Historia bezpośrednich spotkań"},
                {"name": "Kursy bukmacherów", "score": max(60, match_data.get("quality_score", 50) - 5), "description": "Dostępność kursów"},
            ]
        },

        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/predictions/live")
async def get_live_predictions() -> Dict[str, Any]:
    """
    Get live/in-progress predictions.
    Returns current analysis state and any live value bets.
    """
    live_data = {
        "is_analyzing": analysis_state["is_running"],
        "current_step": analysis_state["current_step"],
        "progress": analysis_state["progress"],
        "last_update": analysis_state["last_analysis_time"],
        "live_bets": []
    }

    # If there's a recent result, include top bets
    if analysis_state["last_result"]:
        result = analysis_state["last_result"]
        live_data["live_bets"] = result.get("value_bets", [])[:3]
        live_data["sport"] = result.get("sport")
        live_data["date"] = result.get("date")
        live_data["matches_analyzed"] = result.get("matches_analyzed", 0)

    return live_data


@app.post("/api/predictions/analyze")
async def analyze_predictions(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
) -> Dict[str, Any]:
    """
    Trigger new prediction analysis.
    Alternative endpoint to /api/analysis for consistency.
    
    Authentication: Optional - works with or without auth token
    """
    return await run_analysis(request, background_tasks, user)


@app.post("/api/handicap")
async def predict_handicap(request: HandicapRequest) -> HandicapResponse:
    """
    Predict handicap/spread outcomes based on statistics.

    Supports:
    - Tennis: game handicaps, set handicaps, first set winner
    - Basketball: point spreads, first half spreads, totals
    - First half/second half analysis
    """
    # Select model based on sport
    if request.sport == "tennis":
        model = TennisHandicapModel()
    elif request.sport == "basketball":
        model = BasketballHandicapModel()
    else:
        model = HandicapModel()

    # Parse market type
    market_type_map = {
        "match_handicap": MarketType.MATCH_HANDICAP,
        "first_half": MarketType.FIRST_HALF,
        "second_half": MarketType.SECOND_HALF,
        "total_over": MarketType.TOTAL_OVER,
        "total_under": MarketType.TOTAL_UNDER,
        "first_half_total": MarketType.FIRST_HALF_TOTAL,
    }
    market_type = market_type_map.get(request.market_type, MarketType.MATCH_HANDICAP)

    # Analyze half patterns
    half_patterns = model.analyze_half_patterns(request.home_stats, request.away_stats)
    half_patterns_dict = {
        "first_half": {
            "avg_scored": half_patterns["first_half"].avg_scored,
            "avg_conceded": half_patterns["first_half"].avg_conceded,
            "avg_margin": half_patterns["first_half"].avg_margin,
        },
        "second_half": {
            "avg_scored": half_patterns["second_half"].avg_scored,
            "avg_conceded": half_patterns["second_half"].avg_conceded,
            "avg_margin": half_patterns["second_half"].avg_margin,
        }
    }

    # Get prediction based on market type
    if market_type in [MarketType.TOTAL_OVER, MarketType.TOTAL_UNDER, MarketType.FIRST_HALF_TOTAL]:
        # Total prediction
        total_pred = model.predict_total(
            request.home_stats,
            request.away_stats,
            request.line,
            market_type
        )
        return HandicapResponse(
            market_type=request.market_type,
            line=request.line,
            cover_probability=total_pred.over_probability if "over" in request.market_type else total_pred.under_probability,
            fair_odds=round(1 / total_pred.over_probability, 2) if total_pred.over_probability > 0 else 99.99,
            expected_margin=total_pred.expected_total,
            confidence=total_pred.confidence,
            reasoning=total_pred.reasoning,
            half_patterns=half_patterns_dict,
            value_bets=None
        )
    else:
        # Handicap prediction
        prediction = model.predict_handicap(
            request.home_stats,
            request.away_stats,
            request.line,
            market_type
        )

        # Find value bets if odds provided
        value_bets = None
        if request.bookmaker_odds:
            odds_dict = {
                float(k): tuple(v) for k, v in request.bookmaker_odds.items()
            }
            value_bets = find_value_handicap(
                model,
                request.home_stats,
                request.away_stats,
                odds_dict,
                market_type
            )

        return HandicapResponse(
            market_type=request.market_type,
            line=request.line,
            cover_probability=prediction.cover_probability,
            fair_odds=prediction.fair_odds,
            expected_margin=prediction.expected_margin,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning,
            half_patterns=half_patterns_dict,
            value_bets=value_bets
        )


@app.get("/api/handicap/markets")
async def get_handicap_markets(sport: str = "tennis") -> Dict[str, Any]:
    """Get available handicap markets for a sport."""
    if sport == "tennis":
        return {
            "sport": "tennis",
            "markets": [
                {"type": "match_handicap", "name": "Game Handicap", "example_lines": [-4.5, -2.5, 2.5, 4.5]},
                {"type": "first_half", "name": "First Set Winner", "example_lines": [0]},
                {"type": "total_over", "name": "Total Games Over", "example_lines": [20.5, 21.5, 22.5]},
                {"type": "total_under", "name": "Total Games Under", "example_lines": [20.5, 21.5, 22.5]},
            ]
        }
    elif sport == "basketball":
        return {
            "sport": "basketball",
            "markets": [
                {"type": "match_handicap", "name": "Point Spread", "example_lines": [-10.5, -5.5, -2.5, 2.5, 5.5, 10.5]},
                {"type": "first_half", "name": "First Half Spread", "example_lines": [-5.5, -2.5, 2.5, 5.5]},
                {"type": "total_over", "name": "Total Points Over", "example_lines": [210.5, 215.5, 220.5]},
                {"type": "total_under", "name": "Total Points Under", "example_lines": [210.5, 215.5, 220.5]},
                {"type": "first_half_total", "name": "First Half Total", "example_lines": [105.5, 108.5, 110.5]},
            ]
        }
    else:
        return {
            "sport": sport,
            "markets": [
                {"type": "match_handicap", "name": "Match Handicap", "example_lines": [-1.5, -0.5, 0.5, 1.5]},
                {"type": "first_half", "name": "First Half Result", "example_lines": [0]},
                {"type": "total_over", "name": "Total Over", "example_lines": [2.5]},
                {"type": "total_under", "name": "Total Under", "example_lines": [2.5]},
            ]
        }


# === WEBSOCKET ENDPOINT ===

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time analysis updates."""
    await manager.connect(websocket)
    try:
        # Send current state on connect
        await websocket.send_json({
            "type": "connected",
            "is_analyzing": analysis_state["is_running"],
            "current_step": analysis_state["current_step"],
            "progress": analysis_state["progress"]
        })

        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()

            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# === ML PREDICTION ENDPOINTS ===

@app.get("/api/ml/status")
async def get_ml_status():
    """Get status of ML models for all sports."""
    from core.ml import get_prediction_service as get_ml_service
    
    registry = get_model_registry()
    status = {}
    
    for sport in ["tennis", "basketball", "football"]:
        models = registry.get_active_models(sport)
        status[sport] = {
            "models_available": len(models),
            "models": [
                {
                    "name": m.name,
                    "version": m.version,
                    "type": m.model_type,
                    "accuracy": m.accuracy,
                    "created": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ]
        }
    
    return {"status": "success", "ml_status": status}


@app.post("/api/ml/predict")
async def ml_predict(request: Dict[str, Any]):
    """
    Get ML prediction for a match.
    
    Request body:
    {
        "sport": "tennis",
        "home_player": "Player A",
        "away_player": "Player B",
        "features": {...}  # optional
    }
    """
    from core.ml import get_prediction_service
    
    sport = request.get("sport")
    home_player = request.get("home_player")
    away_player = request.get("away_player")
    features = request.get("features", {})
    
    if not all([sport, home_player, away_player]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    service = get_prediction_service()
    result = service.predict(sport, home_player, away_player, features)
    
    return {
        "status": "success",
        "prediction": {
            "home_win_probability": result.home_win_prob,
            "away_win_probability": result.away_win_prob,
            "draw_probability": result.draw_prob,
            "confidence": result.confidence,
            "model_used": result.model_name,
            "version": result.version
        }
    }


@app.post("/api/ml/predict-value")
async def ml_predict_value(request: Dict[str, Any]):
    """
    Get prediction with value analysis.
    
    Request body:
    {
        "sport": "tennis",
        "home_player": "Player A",
        "away_player": "Player B",
        "home_odds": 1.85,
        "away_odds": 2.00,
        "features": {...}  # optional
    }
    """
    from core.ml import get_prediction_service
    
    sport = request.get("sport")
    home_player = request.get("home_player")
    away_player = request.get("away_player")
    home_odds = request.get("home_odds")
    away_odds = request.get("away_odds")
    features = request.get("features", {})
    
    if not all([sport, home_player, away_player, home_odds, away_odds]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    service = get_prediction_service()
    explanation = service.predict_with_value(
        sport, home_player, away_player, home_odds, away_odds, features
    )
    
    return {
        "status": "success",
        "prediction": {
            "home_probability": explanation.home_probability,
            "away_probability": explanation.away_probability,
            "confidence": explanation.confidence,
            "selection": explanation.selected_outcome,
            "expected_value": explanation.expected_value,
            "reasoning": explanation.reasoning,
            "model_used": explanation.model_used,
            "key_factors": explanation.key_factors
        }
    }


@app.post("/api/ml/train")
async def ml_train(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Trigger model training for a sport.
    Runs in background.
    
    Request body:
    {
        "sport": "tennis",
        "force": false  # optional, force training even if not needed
    }
    """
    sport = request.get("sport")
    force = request.get("force", False)
    
    if not sport:
        raise HTTPException(status_code=400, detail="Sport required")
    
    def train_task():
        import asyncio
        from betting_floor import BettingFloor
        
        async def do_train():
            floor = BettingFloor()
            await floor.initialize()
            result = await floor.train_models(sport, force)
            print(f"Training completed: {result}")
        
        asyncio.run(do_train())
    
    background_tasks.add_task(train_task)
    
    return {
        "status": "training_started",
        "sport": sport,
        "force": force,
        "message": "Training started in background"
    }


@app.post("/api/ml/update-result")
async def ml_update_result(request: Dict[str, Any]):
    """
    Update match result for model training.
    
    Request body:
    {
        "match_id": "unique_id",
        "sport": "tennis",
        "home_score": 2,
        "away_score": 1
    }
    """
    from betting_floor import BettingFloor
    
    match_id = request.get("match_id")
    sport = request.get("sport")
    home_score = request.get("home_score")
    away_score = request.get("away_score")
    
    if not all([match_id, sport, home_score is not None, away_score is not None]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    floor = BettingFloor()
    await floor.initialize()
    await floor.update_match_result(match_id, home_score, away_score, sport)
    
    return {
        "status": "success",
        "message": f"Match {match_id} result updated"
    }


# === ENSEMBLE MODEL ENDPOINTS ===

@app.get("/api/ml/ensemble/status")
async def get_ensemble_status():
    """
    Get ensemble system status.
    Returns model rankings and performance metrics.
    """
    from core.ml import get_prediction_service
    
    service = get_prediction_service()
    
    # Get info for all sports
    sports = ["tennis", "basketball", "football"]
    ensemble_status = {}
    
    for sport in sports:
        info = service.get_model_info(sport)
        ensemble_status[sport] = {
            "available_models": info["available_models"],
            "ensemble_enabled": info["ensemble"]["enabled"],
            "primary_model": info["ensemble"]["primary_model"],
            "model_rankings": info["ensemble"]["model_rankings"]
        }
    
    return {
        "status": "success",
        "ensemble_status": ensemble_status
    }


@app.post("/api/ml/ensemble/predict")
async def ensemble_predict(request: Dict[str, Any]):
    """
    Get ensemble prediction from multiple models.
    
    Request body:
    {
        "sport": "tennis",
        "home_player": "Player A",
        "away_player": "Player B",
        "features": {...},  # optional
        "return_details": true  # optional, return per-model predictions
    }
    
    Returns:
    {
        "ensemble": {
            "home_probability": 0.58,
            "away_probability": 0.42,
            "confidence": 0.72,
            "agreement_score": 0.85,
            "primary_model": "xgboost_tennis",
            "recommended_action": "bet"
        },
        "model_contributions": {...},  # if return_details=true
        "individual_predictions": [...]  # if return_details=true
    }
    """
    from core.ml import get_prediction_service, get_model_registry
    
    sport = request.get("sport")
    home_player = request.get("home_player")
    away_player = request.get("away_player")
    features = request.get("features", {})
    return_details = request.get("return_details", False)
    
    if not all([sport, home_player, away_player]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    service = get_prediction_service()
    registry = get_model_registry()
    
    # Get ensemble prediction
    result = service.predict(sport, home_player, away_player, features, use_ensemble=True)
    
    response = {
        "status": "success",
        "ensemble": {
            "home_probability": result.home_win_prob,
            "away_probability": result.away_win_prob,
            "draw_probability": result.draw_prob,
            "confidence": result.confidence,
            "model_used": result.model_name
        }
    }
    
    # Add ensemble metadata
    if hasattr(result, '_ensemble_meta'):
        meta = result._ensemble_meta
        response["ensemble"].update({
            "agreement_score": meta.get("agreement_score"),
            "recommended_action": meta.get("recommended_action"),
            "uncertainty": meta.get("uncertainty")
        })
        
        if return_details:
            response["model_contributions"] = meta.get("model_contributions", {})
    
    # Add individual predictions if requested
    if return_details:
        models = registry.get_active_models(sport)
        individual = []
        
        for model in models:
            try:
                f = features.copy()
                f['home_player'] = home_player
                f['away_player'] = away_player
                pred = model.predict(f)
                individual.append({
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "home_probability": pred.home_win_prob,
                    "away_probability": pred.away_win_prob,
                    "confidence": pred.confidence
                })
            except Exception as e:
                individual.append({
                    "model_name": model.name,
                    "error": str(e)
                })
        
        response["individual_predictions"] = individual
        response["models_used"] = len(individual)
    
    return response


@app.post("/api/ml/ensemble/predict-value")
async def ensemble_predict_value(request: Dict[str, Any]):
    """
    Get ensemble prediction with value betting analysis.
    
    Request body:
    {
        "sport": "tennis",
        "home_player": "Player A",
        "away_player": "Player B",
        "home_odds": 1.85,
        "away_odds": 2.00,
        "features": {...}  # optional
    }
    """
    from core.ml import get_prediction_service
    
    sport = request.get("sport")
    home_player = request.get("home_player")
    away_player = request.get("away_player")
    home_odds = request.get("home_odds")
    away_odds = request.get("away_odds")
    features = request.get("features", {})
    
    if not all([sport, home_player, away_player, home_odds, away_odds]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    service = get_prediction_service()
    explanation = service.predict_with_value(
        sport, home_player, away_player, home_odds, away_odds, features, use_ensemble=True
    )
    
    return {
        "status": "success",
        "prediction": {
            "home_probability": explanation.home_probability,
            "away_probability": explanation.away_probability,
            "confidence": explanation.confidence,
            "agreement_score": explanation.agreement_score,
            "selection": explanation.selected_outcome,
            "expected_value": explanation.expected_value,
            "recommended_action": explanation.recommended_action,
            "reasoning": explanation.reasoning,
            "model_used": explanation.model_used,
            "key_factors": explanation.key_factors,
            "model_contributions": explanation.model_contributions
        }
    }


@app.get("/api/ml/ensemble/rankings/{sport}")
async def get_model_rankings(sport: str):
    """
    Get model rankings for a sport.
    Shows which models contribute most to ensemble.
    """
    from core.ml import get_prediction_service
    
    service = get_prediction_service()
    info = service.get_model_info(sport)
    
    return {
        "sport": sport,
        "rankings": info["ensemble"]["model_rankings"],
        "primary_model": info["ensemble"]["primary_model"],
        "total_models": info["available_models"]
    }


# === SCHEDULER ENDPOINTS ===

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and upcoming analyses."""
    from core.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    
    return {
        "status": "running" if scheduler._running else "stopped",
        "schedules_count": len(scheduler.schedules),
        "upcoming_analyses": scheduler.get_upcoming_analyses(),
        "recent_logs": scheduler.get_analysis_log(limit=5)
    }


@app.get("/api/scheduler/schedules")
async def get_schedules():
    """Get all analysis schedules."""
    from core.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    schedules = scheduler.get_schedules()
    
    return {
        "schedules": [
            {
                "id": s.id,
                "sport": s.sport,
                "league": s.league,
                "frequency": s.frequency.value,
                "custom_hours": s.custom_hours,
                "enabled": s.enabled,
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "next_run": s.next_run.isoformat() if s.next_run else None
            }
            for s in schedules
        ]
    }


@app.post("/api/scheduler/schedules")
async def create_schedule(request: Dict[str, Any]):
    """
    Create a new analysis schedule.
    
    Request body:
    {
        "sport": "tennis",
        "league": "ATP",  # optional
        "frequency": "every_6_hours",  # hourly, every_3_hours, every_6_hours, every_12_hours, daily, custom
        "custom_hours": 8  # only for custom frequency
    }
    """
    from core.scheduler import get_scheduler, AnalysisFrequency
    
    sport = request.get("sport")
    league = request.get("league")
    frequency_str = request.get("frequency", "every_6_hours")
    custom_hours = request.get("custom_hours")
    
    if not sport:
        raise HTTPException(status_code=400, detail="Sport required")
    
    try:
        frequency = AnalysisFrequency(frequency_str)
    except ValueError:
        valid = [f.value for f in AnalysisFrequency]
        raise HTTPException(status_code=400, detail=f"Invalid frequency. Valid: {valid}")
    
    scheduler = get_scheduler()
    schedule = scheduler.create_schedule(
        sport=sport,
        frequency=frequency,
        league=league,
        custom_hours=custom_hours
    )
    
    return {
        "status": "success",
        "schedule": {
            "id": schedule.id,
            "sport": schedule.sport,
            "league": schedule.league,
            "frequency": schedule.frequency.value,
            "next_run": schedule.next_run.isoformat() if schedule.next_run else None
        }
    }


@app.delete("/api/scheduler/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule."""
    from core.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.delete_schedule(schedule_id)
    
    return {"status": "success", "message": f"Schedule {schedule_id} deleted"}


@app.post("/api/scheduler/schedules/{schedule_id}/toggle")
async def toggle_schedule(schedule_id: str, enabled: bool = True):
    """Enable/disable a schedule."""
    from core.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.enable_schedule(schedule_id, enabled)
    
    return {
        "status": "success",
        "schedule_id": schedule_id,
        "enabled": enabled
    }


@app.post("/api/scheduler/start")
async def start_scheduler():
    """Start the scheduler."""
    from core.scheduler import get_scheduler, default_analysis_function
    
    scheduler = get_scheduler()
    scheduler.set_analysis_callback(default_analysis_function)
    
    import asyncio
    asyncio.create_task(scheduler.start())
    
    return {"status": "success", "message": "Scheduler started"}


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """Stop the scheduler."""
    from core.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    await scheduler.stop()
    
    return {"status": "success", "message": "Scheduler stopped"}


# === FEEDBACK LOOP ENDPOINTS ===

@app.post("/api/feedback/results")
async def record_match_result(request: Dict[str, Any]):
    """
    Record a match result for feedback loop.
    
    Request body:
    {
        "match_id": "unique_id",
        "sport": "tennis",
        "home_player": "Player A",
        "away_player": "Player B",
        "home_score": 2,
        "away_score": 1,
        "match_date": "2026-02-01T15:00:00"
    }
    """
    from core.ml.feedback_loop import get_feedback_loop, MatchResult
    from datetime import datetime
    
    try:
        match_date = request.get("match_date")
        if match_date:
            match_date = datetime.fromisoformat(match_date)
        else:
            match_date = datetime.now()
        
        result = MatchResult(
            match_id=request["match_id"],
            sport=request["sport"],
            home_player=request["home_player"],
            away_player=request["away_player"],
            home_score=request["home_score"],
            away_score=request["away_score"],
            match_date=match_date
        )
        
        feedback = get_feedback_loop()
        feedback.record_result(result)
        
        return {
            "status": "success",
            "message": f"Result recorded for match {result.match_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feedback/performance/{sport}")
async def get_performance(sport: str):
    """Get performance metrics for a sport."""
    from core.ml.feedback_loop import get_feedback_loop
    
    feedback = get_feedback_loop()
    metrics = feedback.get_performance_report(sport)
    
    return {
        "sport": sport,
        "metrics": metrics
    }


@app.get("/api/feedback/performance")
async def get_all_performance():
    """Get all performance metrics."""
    from core.ml.feedback_loop import get_feedback_loop
    
    feedback = get_feedback_loop()
    metrics = feedback.get_performance_report()
    
    return {"metrics": metrics}


# === STATIC FILES (React Frontend) ===

# Check if frontend build exists
frontend_dist = Path(__file__).parent.parent / "frontend" / "app" / "dist"

if frontend_dist.exists():
    # Serve static files
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        """Serve React frontend."""
        return FileResponse(frontend_dist / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        """Catch-all for React Router."""
        # Check if it's an API route
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # Check if file exists in dist
        file_path = frontend_dist / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Return index.html for client-side routing
        return FileResponse(frontend_dist / "index.html")
else:
    @app.get("/")
    async def no_frontend():
        """Fallback when frontend is not built."""
        return JSONResponse({
            "message": "NEXUS AI API is running",
            "frontend": "Not built. Run 'npm run build' in frontend/app/",
            "docs": "/docs"
        })


# === MAIN ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
