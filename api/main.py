# api/main.py
"""
FastAPI backend for NEXUS AI Lite.
Provides REST API for React frontend.
"""

import asyncio
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from core.models import (
    HandicapModel,
    TennisHandicapModel,
    BasketballHandicapModel,
    MarketType,
    find_value_handicap,
)


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


# === APP SETUP ===

app = FastAPI(
    title="NEXUS AI Lite",
    description="Sports Prediction System - REST API",
    version="2.2.0"
)

# Setup Prometheus metrics
try:
    from api.metrics import setup_metrics, set_websocket_connections, set_active_analyses
    setup_metrics(app, version="2.2.0")
except ImportError:
    # prometheus_client not installed
    pass

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# === API ENDPOINTS ===

@app.get("/api/status")
async def get_status() -> SystemStatus:
    """Get system status."""
    uptime = (datetime.now() - analysis_state["start_time"]).total_seconds()

    return SystemStatus(
        status="running" if not analysis_state["is_running"] else "analyzing",
        mode=settings.APP_MODE,
        api_keys_configured={
            "brave_search": bool(settings.BRAVE_API_KEY),
            "serper": bool(settings.SERPER_API_KEY),
            "odds_api": bool(settings.ODDS_API_KEY),
            "anthropic": bool(settings.ANTHROPIC_API_KEY),
        },
        last_analysis=analysis_state["last_analysis_time"],
        uptime_seconds=uptime
    )


@app.post("/api/analysis")
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
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
        "message": "Analysis started. Connect to WebSocket for updates."
    }


async def _run_analysis_task(
    sport: str,
    target_date: str,
    min_quality: float,
    top_n: int
):
    """Background task for running analysis."""
    analysis_state["is_running"] = True
    analysis_state["progress"] = 0

    try:
        # Import here to avoid circular imports
        from betting_floor import run_betting_analysis

        # Step 1: Starting
        await _broadcast_progress("collecting", 10, "Collecting fixtures...")

        # Step 2: Run analysis
        await _broadcast_progress("analyzing", 30, "Analyzing matches...")

        result = await run_betting_analysis(
            sport=sport,
            date=target_date,
            bankroll=settings.DEFAULT_BANKROLL
        )

        # Step 3: Processing results
        await _broadcast_progress("processing", 70, "Processing results...")

        # Format results for frontend
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
    """Get matches for a given sport and date."""
    target_date = match_date or str(date.today())

    # This would call the fixture collector
    # For now, return placeholder
    return {
        "status": "success",
        "sport": sport,
        "date": target_date,
        "matches": [],
        "message": "Use /api/analysis to run full analysis"
    }


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get system statistics for dashboard."""
    return {
        "total_analyses": 0,  # TODO: Track in database
        "successful_bets": 0,
        "win_rate": 0.0,
        "total_profit": 0.0,
        "avg_edge": 0.0,
        "avg_quality": 0.0,
        "sports_analyzed": ["tennis", "basketball", "greyhound", "handball", "table_tennis"],
        "last_7_days": []
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
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Trigger new prediction analysis.
    Alternative endpoint to /api/analysis for consistency.
    """
    return await run_analysis(request, background_tasks)


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
