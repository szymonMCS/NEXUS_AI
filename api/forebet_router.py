"""
Forebet.com API Routes

Provides REST endpoints for accessing Forebet predictions and statistics.
Includes Asian Handicap and European Handicap support.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Import scraper
try:
    from agents.sports_data_swarm.forebet_scraper import ForebetScraper, fetch_forebet_predictions
except ImportError:
    # Fallback if running from different path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.sports_data_swarm.forebet_scraper import ForebetScraper, fetch_forebet_predictions


router = APIRouter(prefix="/api/forebet", tags=["forebet"])


# === Pydantic Models ===

class AsianHandicapLineResponse(BaseModel):
    type: str = "asian"
    line: float
    line_display: str
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    home_edge: Optional[float] = None
    away_edge: Optional[float] = None
    recommendation: Optional[str] = None
    description: str


class EuropeanHandicapLineResponse(BaseModel):
    type: str = "european"
    line: str
    line_display: str
    home_advantage: int
    away_advantage: int
    home_odds: float
    draw_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    draw_prob: Optional[float] = None
    away_prob: Optional[float] = None
    description: str


class HandicapPredictionsResponse(BaseModel):
    asian: List[AsianHandicapLineResponse]
    european: List[EuropeanHandicapLineResponse]
    best_value: Optional[Dict[str, Any]] = None


class MatchPredictionResponse(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_result: str
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    confidence: float
    handicaps: Optional[HandicapPredictionsResponse] = None
    source: str = "forebet"


class MatchDetailsResponse(BaseModel):
    match_id: str
    title: str
    home_team: str
    away_team: str
    predictions: Dict[str, Any]
    handicaps: HandicapPredictionsResponse
    home_stats: Dict[str, Any]
    away_stats: Dict[str, Any]
    h2h: Dict[str, Any]
    standings: List[Dict[str, Any]]
    home_form: List[str]
    away_form: List[str]
    injuries: List[Dict[str, str]]
    scraped_at: str


class LiveMatchResponse(BaseModel):
    teams: str
    score: str
    time: str
    last_updated: str


class HandicapValueBet(BaseModel):
    match_id: str
    match: str
    league: str
    handicap_type: str  # 'asian' or 'european'
    line: str
    selection: str  # 'home', 'away', 'draw'
    odds: float
    probability: float
    edge: float
    confidence: float
    match_date: str


# === API Endpoints ===

@router.get("/predictions/upcoming", response_model=List[MatchPredictionResponse])
async def get_upcoming_predictions(
    sport: str = Query("football", description="Sport type (football, basketball, tennis)"),
    days: int = Query(3, ge=1, le=7, description="Number of days ahead"),
    league: Optional[str] = Query(None, description="Filter by league"),
    include_handicaps: bool = Query(True, description="Include handicap predictions")
):
    """
    Get upcoming matches with Forebet predictions.
    
    Returns matches with probability predictions for 1X2 outcomes
    and optional Asian/European handicap analysis.
    """
    try:
        async with ForebetScraper() as scraper:
            matches = await scraper.get_upcoming_matches(
                sport=sport,
                days=days,
                league=league
            )
            
            # Convert to response format
            response = []
            for match in matches:
                match_data = {
                    "match_id": match.match_id or f"{match.home_team}-{match.away_team}",
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "league": match.league,
                    "match_date": match.match_date.isoformat(),
                    "prob_home": match.prob_home,
                    "prob_draw": match.prob_draw,
                    "prob_away": match.prob_away,
                    "predicted_result": match.predicted_result,
                    "odds_home": match.odds_home,
                    "odds_draw": match.odds_draw,
                    "odds_away": match.odds_away,
                    "confidence": max(match.prob_home, match.prob_draw, match.prob_away)
                }
                
                if include_handicaps and match.handicaps:
                    match_data["handicaps"] = match.handicaps.to_dict()
                
                response.append(MatchPredictionResponse(**match_data))
            
            return response
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching predictions: {str(e)}"
        )


@router.get("/match/{match_id}", response_model=MatchDetailsResponse)
async def get_match_details(match_id: str):
    """
    Get detailed statistics for a specific match.
    
    Includes:
    - Predictions (1X2, Over/Under, BTTS)
    - Asian Handicap lines with probabilities
    - European Handicap lines with probabilities
    - Team statistics
    - Head-to-head record
    - League standings
    - Recent form
    - Injury reports
    """
    try:
        async with ForebetScraper() as scraper:
            details = await scraper.get_match_details(match_id)
            
            if not details:
                raise HTTPException(
                    status_code=404,
                    detail=f"Match {match_id} not found"
                )
            
            return MatchDetailsResponse(
                match_id=details['match_id'],
                title=details['title'],
                home_team=details['home_team'],
                away_team=details['away_team'],
                predictions=details['predictions'],
                handicaps=details['handicaps'],
                home_stats=details['home_stats'],
                away_stats=details['away_stats'],
                h2h=details['h2h'],
                standings=details['standings'],
                home_form=details['home_form'],
                away_form=details['away_form'],
                injuries=details['injuries'],
                scraped_at=details['scraped_at']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching match details: {str(e)}"
        )


@router.get("/match/{match_id}/handicaps", response_model=HandicapPredictionsResponse)
async def get_match_handicaps(match_id: str):
    """
    Get Asian and European Handicap predictions for a specific match.
    
    Returns:
    - Asian Handicap lines (-2.5 to +2.5) with odds and probabilities
    - European Handicap lines (0:0, 1:0, 0:1, etc.) with 3-way odds
    - Best value recommendation based on edge calculation
    """
    try:
        async with ForebetScraper() as scraper:
            handicaps = await scraper.get_handicap_predictions(match_id)
            
            if not handicaps:
                raise HTTPException(
                    status_code=404,
                    detail=f"Handicap predictions for match {match_id} not found"
                )
            
            return HandicapPredictionsResponse(
                asian=[AsianHandicapLineResponse(**line.to_dict()) for line in handicaps.asian_lines],
                european=[EuropeanHandicapLineResponse(**line.to_dict()) for line in handicaps.european_lines],
                best_value=handicaps.best_value
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching handicap predictions: {str(e)}"
        )


@router.get("/live", response_model=List[LiveMatchResponse])
async def get_live_matches():
    """
    Get currently live matches with scores.
    
    Real-time data for in-progress matches.
    """
    try:
        async with ForebetScraper() as scraper:
            matches = await scraper.get_live_matches()
            return [LiveMatchResponse(**match) for match in matches]
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching live matches: {str(e)}"
        )


@router.get("/league/{league_name}/standings")
async def get_league_standings(league_name: str):
    """
    Get standings table for a specific league.
    """
    try:
        async with ForebetScraper() as scraper:
            standings = await scraper.get_league_standings(league_name)
            return {
                "league": league_name,
                "standings": standings,
                "updated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching standings: {str(e)}"
        )


@router.get("/search")
async def search_matches(
    q: str = Query(..., min_length=2, description="Search query (team name)")
):
    """
    Search for matches by team name.
    """
    try:
        async with ForebetScraper() as scraper:
            matches = await scraper.search_matches(q)
            return {
                "query": q,
                "results": [
                    {
                        "match_id": m.match_id,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "league": m.league,
                        "match_date": m.match_date.isoformat(),
                        "predicted_result": m.predicted_result,
                        "confidence": max(m.prob_home, m.prob_draw, m.prob_away),
                        "handicaps": m.handicaps.to_dict() if m.handicaps else None
                    }
                    for m in matches[:20]  # Limit results
                ],
                "total": len(matches)
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching matches: {str(e)}"
        )


@router.get("/value-bets")
async def get_value_bets(
    min_confidence: float = Query(60.0, ge=0, le=100, description="Minimum confidence %"),
    sport: str = Query("football", description="Sport type"),
    days: int = Query(1, ge=1, le=3, description="Days ahead"),
    include_handicaps: bool = Query(True, description="Include handicap value bets")
):
    """
    Get high-confidence predictions that may represent value bets.
    
    Filters matches by confidence threshold and calculates edge.
    Optionally includes handicap lines with positive expected value.
    """
    try:
        async with ForebetScraper() as scraper:
            matches = await scraper.get_upcoming_matches(sport=sport, days=days)
            
            value_bets = []
            for match in matches:
                confidence = max(match.prob_home, match.prob_draw, match.prob_away)
                
                if confidence >= min_confidence:
                    # Calculate implied probability from odds
                    if match.odds_home and match.odds_draw and match.odds_away:
                        market_prob = 100 / match.odds_home if match.predicted_result == '1' else \
                                     100 / match.odds_draw if match.predicted_result == 'X' else \
                                     100 / match.odds_away
                        
                        # Edge = predicted probability - market implied probability
                        edge = confidence - market_prob
                        
                        if edge > 0:
                            value_bets.append({
                                "match_id": match.match_id,
                                "match": f"{match.home_team} vs {match.away_team}",
                                "league": match.league,
                                "selection": match.predicted_result,
                                "confidence": round(confidence, 1),
                                "odds": match.odds_home if match.predicted_result == '1' else \
                                       match.odds_draw if match.predicted_result == 'X' else \
                                       match.odds_away,
                                "edge": round(edge, 1),
                                "value_score": round(confidence * edge / 100, 2),
                                "match_date": match.match_date.isoformat(),
                                "type": "1x2"
                            })
                
                # Check handicap value bets
                if include_handicaps and match.handicaps:
                    for ah_line in match.handicaps.asian_lines:
                        if ah_line.home_edge and ah_line.home_edge > 0.05:
                            value_bets.append({
                                "match_id": match.match_id,
                                "match": f"{match.home_team} vs {match.away_team}",
                                "league": match.league,
                                "selection": f"Home AH {ah_line.line:+.1f}",
                                "confidence": round(ah_line.home_prob or 0, 1),
                                "odds": ah_line.home_odds,
                                "edge": round(ah_line.home_edge * 100, 1),
                                "value_score": round((ah_line.home_prob or 0) * ah_line.home_edge, 2),
                                "match_date": match.match_date.isoformat(),
                                "type": "asian_handicap"
                            })
                        if ah_line.away_edge and ah_line.away_edge > 0.05:
                            value_bets.append({
                                "match_id": match.match_id,
                                "match": f"{match.home_team} vs {match.away_team}",
                                "league": match.league,
                                "selection": f"Away AH {ah_line.line:+.1f}",
                                "confidence": round(ah_line.away_prob or 0, 1),
                                "odds": ah_line.away_odds,
                                "edge": round(ah_line.away_edge * 100, 1),
                                "value_score": round((ah_line.away_prob or 0) * ah_line.away_edge, 2),
                                "match_date": match.match_date.isoformat(),
                                "type": "asian_handicap"
                            })
            
            # Sort by value score
            value_bets.sort(key=lambda x: x["value_score"], reverse=True)
            
            return {
                "value_bets": value_bets[:30],  # Top 30
                "total_found": len(value_bets),
                "filters": {
                    "min_confidence": min_confidence,
                    "sport": sport,
                    "days": days,
                    "include_handicaps": include_handicaps
                }
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating value bets: {str(e)}"
        )


@router.get("/handicaps/best-value")
async def get_best_handicap_bets(
    sport: str = Query("football", description="Sport type"),
    days: int = Query(1, ge=1, le=3, description="Days ahead"),
    min_edge: float = Query(0.08, ge=0.0, le=0.5, description="Minimum edge threshold (0.08 = 8%)")
):
    """
    Get the best handicap value bets across all matches.
    
    Filters for Asian Handicap lines with significant positive expected value.
    """
    try:
        async with ForebetScraper() as scraper:
            matches = await scraper.get_upcoming_matches(sport=sport, days=days)
            
            best_bets = []
            for match in matches:
                if not match.handicaps:
                    continue
                    
                for ah in match.handicaps.asian_lines:
                    if ah.home_edge and ah.home_edge >= min_edge:
                        best_bets.append({
                            "match_id": match.match_id,
                            "match": f"{match.home_team} vs {match.away_team}",
                            "league": match.league,
                            "handicap_type": "asian",
                            "line": f"AH {ah.line:+.1f}",
                            "selection": "home",
                            "odds": ah.home_odds,
                            "probability": round(ah.home_prob, 1) if ah.home_prob else None,
                            "edge": round(ah.home_edge * 100, 1),
                            "confidence": round(ah.home_prob, 1) if ah.home_prob else 50,
                            "match_date": match.match_date.isoformat()
                        })
                    
                    if ah.away_edge and ah.away_edge >= min_edge:
                        best_bets.append({
                            "match_id": match.match_id,
                            "match": f"{match.home_team} vs {match.away_team}",
                            "league": match.league,
                            "handicap_type": "asian",
                            "line": f"AH {ah.line:+.1f}",
                            "selection": "away",
                            "odds": ah.away_odds,
                            "probability": round(ah.away_prob, 1) if ah.away_prob else None,
                            "edge": round(ah.away_edge * 100, 1),
                            "confidence": round(ah.away_prob, 1) if ah.away_prob else 50,
                            "match_date": match.match_date.isoformat()
                        })
            
            # Sort by edge
            best_bets.sort(key=lambda x: x["edge"], reverse=True)
            
            return {
                "bets": best_bets[:20],
                "total_found": len(best_bets),
                "filters": {
                    "sport": sport,
                    "days": days,
                    "min_edge": min_edge
                }
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching best handicap bets: {str(e)}"
        )


# === Integration Endpoint ===

@router.post("/refresh-cache")
async def refresh_cache(background_tasks: BackgroundTasks):
    """
    Refresh the prediction cache in background.
    
    Call this periodically to keep data fresh.
    """
    async def _refresh():
        async with ForebetScraper() as scraper:
            # Clear cache by making new requests
            await scraper.get_upcoming_matches(days=1)
            await scraper.get_live_matches()
    
    background_tasks.add_task(_refresh)
    
    return {
        "status": "started",
        "message": "Cache refresh initiated in background"
    }


# === Educational Endpoints ===

@router.get("/handicaps/explain")
async def explain_handicaps():
    """
    Get explanation of Asian and European Handicap betting.
    
    Educational endpoint for understanding handicap types.
    """
    return {
        "asian_handicap": {
            "description": "Asian Handicap eliminates the draw option by using fractional goal lines.",
            "key_features": [
                "Two outcomes only (home or away win)",
                "Fractional lines (-0.5, +1.5, etc.) eliminate draw",
                "Whole number lines (-1, 0, +1) can result in void/push",
                "Stake refunded on void bets"
            ],
            "common_lines": {
                "0.0": "Draw No Bet - stake refunded if draw",
                "-0.5": "Team must win outright",
                "+0.5": "Team wins if they win or draw",
                "-1.0": "Team must win by 2+ goals (stake back if win by 1)",
                "+1.0": "Team wins if they win, draw, or lose by 1",
                "-1.5": "Team must win by 2+ goals",
                "+1.5": "Team wins if they win, draw, or lose by 1"
            }
        },
        "european_handicap": {
            "description": "European Handicap adds goal advantage before match starts with 3 outcomes.",
            "key_features": [
                "Three outcomes (home win, draw, away win)",
                "Integer goal advantages (1:0, 0:1, etc.)",
                "Draw is possible after handicap applied",
                "No stake refunds"
            ],
            "example": "EH 1:0 means Home team starts with 1 goal advantage"
        },
        "calculations": {
            "probability": "Calculated from 1X2 predictions using Poisson distribution",
            "edge": "(Odds / Fair_Odds) - 1, where Fair_Odds = 100 / Probability",
            "value_bet": "When predicted probability > market implied probability"
        }
    }
