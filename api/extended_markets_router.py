"""
Extended Markets API Routes

Provides REST endpoints for alternative betting markets:
- Football: Cards, fouls, goal difference
- Basketball: Win margin, fouls
- Tennis: Set betting, game handicaps
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Import extended markets calculator
try:
    from agents.sports_data_swarm.extended_markets import (
        ExtendedMarketsCalculator,
        calculate_all_extended_markets,
        SportType,
        CardMarket,
        FoulMarket,
        GoalDifferenceMarket,
        WinMarginMarket,
        SetBettingMarket,
        GameHandicapMarket,
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.sports_data_swarm.extended_markets import (
        ExtendedMarketsCalculator,
        calculate_all_extended_markets,
        SportType,
        CardMarket,
        FoulMarket,
        GoalDifferenceMarket,
        WinMarginMarket,
        SetBettingMarket,
        GameHandicapMarket,
    )


router = APIRouter(prefix="/api/markets", tags=["extended-markets"])


# === Pydantic Response Models ===

class CardMarketResponse(BaseModel):
    type: str
    line: float
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    expected_cards: Optional[float] = None


class FoulMarketResponse(BaseModel):
    type: str
    line: float
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    expected_fouls: Optional[float] = None


class GoalDifferenceResponse(BaseModel):
    selection: str
    odds: float
    probability: Optional[float] = None
    edge: Optional[float] = None


class WinMarginResponse(BaseModel):
    range: str
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    recommendation: Optional[str] = None


class SetBettingResponse(BaseModel):
    score: str
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    edge: Optional[float] = None


class GameHandicapResponse(BaseModel):
    line: float
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    recommendation: Optional[str] = None


class ExtendedMarketsResponse(BaseModel):
    sport: str
    card_markets: List[CardMarketResponse] = []
    foul_markets: List[FoulMarketResponse] = []
    goal_diff_markets: List[GoalDifferenceResponse] = []
    win_margin_markets: List[WinMarginResponse] = []
    set_betting_markets: List[SetBettingResponse] = []
    game_handicap_markets: List[GameHandicapResponse] = []


# === API Endpoints ===

@router.get("/extended/{sport}")
async def get_extended_markets(
    sport: str,
    prob_home: float = Query(..., ge=0, le=100, description="Home win probability %"),
    prob_draw: float = Query(33.3, ge=0, le=100, description="Draw probability %"),
    prob_away: float = Query(..., ge=0, le=100, description="Away win probability %"),
    home_aggression: float = Query(0.5, ge=0, le=1, description="Home team aggression (0-1)"),
    away_aggression: float = Query(0.5, ge=0, le=1, description="Away team aggression (0-1)"),
    home_xg: float = Query(1.5, ge=0, description="Home expected goals/points"),
    away_xg: float = Query(1.2, ge=0, description="Away expected goals/points"),
    spread: float = Query(0, description="Point spread (basketball)"),
    sets_format: int = Query(3, ge=2, le=5, description="Tennis sets format (2 or 3)")
):
    """
    Get all extended markets for a match.
    
    Calculates alternative betting markets based on match probabilities
    and team characteristics.
    """
    try:
        if sport not in ['football', 'basketball', 'tennis']:
            raise HTTPException(status_code=400, detail="Sport must be football, basketball, or tennis")
        
        markets = calculate_all_extended_markets(
            sport=sport,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            home_aggression=home_aggression,
            away_aggression=away_aggression,
            home_xg=home_xg,
            away_xg=away_xg,
            spread=spread,
            sets_format=sets_format
        )
        
        return ExtendedMarketsResponse(
            sport=sport,
            card_markets=[CardMarketResponse(**m.to_dict()) for m in markets.card_markets],
            foul_markets=[FoulMarketResponse(**m.to_dict()) for m in markets.foul_markets],
            goal_diff_markets=[GoalDifferenceResponse(**m.to_dict()) for m in markets.goal_diff_markets],
            win_margin_markets=[WinMarginResponse(**m.to_dict()) for m in markets.win_margin_markets],
            set_betting_markets=[SetBettingResponse(**m.to_dict()) for m in markets.set_betting_markets],
            game_handicap_markets=[GameHandicapResponse(**m.to_dict()) for m in markets.game_handicap_markets],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating markets: {str(e)}")


@router.get("/football/cards")
async def get_football_cards(
    home_aggression: float = Query(0.5, ge=0, le=1),
    away_aggression: float = Query(0.5, ge=0, le=1),
    match_importance: float = Query(0.5, ge=0, le=1),
    referee_strictness: float = Query(0.5, ge=0, le=1)
):
    """
    Get football cards markets (Over/Under, Asian handicap).
    
    Factors:
    - Team aggression levels
    - Match importance (derby, playoff)
    - Referee strictness
    """
    try:
        markets = ExtendedMarketsCalculator.calculate_football_cards(
            home_aggression=home_aggression,
            away_aggression=away_aggression,
            match_importance=match_importance,
            referee_strictness=referee_strictness
        )
        
        return {
            "sport": "football",
            "market_type": "cards",
            "expected_total_cards": markets[0].expected_cards if markets else None,
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating cards: {str(e)}")


@router.get("/football/fouls")
async def get_football_fouls(
    home_foul_tendency: float = Query(0.5, ge=0, le=1),
    away_foul_tendency: float = Query(0.5, ge=0, le=1),
    match_intensity: float = Query(0.5, ge=0, le=1)
):
    """Get football fouls markets."""
    try:
        markets = ExtendedMarketsCalculator.calculate_football_fouls(
            home_foul_tendency=home_foul_tendency,
            away_foul_tendency=away_foul_tendency,
            match_intensity=match_intensity
        )
        
        return {
            "sport": "football",
            "market_type": "fouls",
            "expected_total_fouls": markets[0].expected_fouls if markets else None,
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fouls: {str(e)}")


@router.get("/football/goal-difference")
async def get_goal_difference(
    prob_home: float = Query(..., ge=0, le=100),
    prob_draw: float = Query(..., ge=0, le=100),
    prob_away: float = Query(..., ge=0, le=100),
    home_xg: float = Query(1.5, ge=0),
    away_xg: float = Query(1.2, ge=0)
):
    """
    Get goal difference / win margin markets for football.
    
    Includes: Win by 1, 2+, 3+, exactly 1, any win, etc.
    """
    try:
        markets = ExtendedMarketsCalculator.calculate_goal_difference(
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            home_xg=home_xg,
            away_xg=away_xg
        )
        
        return {
            "sport": "football",
            "market_type": "goal_difference",
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating goal difference: {str(e)}")


@router.get("/basketball/win-margin")
async def get_basketball_win_margin(
    spread: float = Query(..., description="Expected point spread"),
    total_points: float = Query(210, ge=150, le=250)
):
    """
    Get basketball win margin ranges (1-5, 6-10, 11+, etc.).
    """
    try:
        markets = ExtendedMarketsCalculator.calculate_basketball_win_margin(
            spread=spread,
            total_points=total_points
        )
        
        return {
            "sport": "basketball",
            "market_type": "win_margin",
            "spread": spread,
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating win margin: {str(e)}")


@router.get("/basketball/fouls")
async def get_basketball_fouls(
    avg_team_fouls: float = Query(20, ge=15, le=25),
    pace_factor: float = Query(1.0, ge=0.8, le=1.2)
):
    """Get basketball team fouls markets."""
    try:
        markets = ExtendedMarketsCalculator.calculate_basketball_fouls(
            avg_team_fouls=avg_team_fouls,
            pace_factor=pace_factor
        )
        
        return {
            "sport": "basketball",
            "market_type": "fouls",
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fouls: {str(e)}")


@router.get("/tennis/set-betting")
async def get_tennis_set_betting(
    home_win_prob: float = Query(..., ge=0, le=100, description="Match win probability %"),
    sets_format: int = Query(3, ge=2, le=3, description="2 for best-of-3, 3 for best-of-5")
):
    """
    Get tennis set betting (correct score) markets.
    
    Returns probabilities for scores like 2-0, 2-1, 3-0, 3-1, 3-2.
    """
    try:
        markets = ExtendedMarketsCalculator.calculate_tennis_set_betting(
            home_win_prob=home_win_prob,
            sets_format=sets_format
        )
        
        return {
            "sport": "tennis",
            "market_type": "set_betting",
            "sets_format": "best_of_5" if sets_format == 3 else "best_of_3",
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating set betting: {str(e)}")


@router.get("/tennis/game-handicap")
async def get_tennis_game_handicap(
    home_win_prob: float = Query(..., ge=0, le=100),
    total_games_estimate: int = Query(22, ge=18, le=40)
):
    """
    Get tennis game handicap markets.
    
    Lines typically range from -4.5 to +4.5 games.
    """
    try:
        markets = ExtendedMarketsCalculator.calculate_tennis_game_handicap(
            home_win_prob=home_win_prob,
            total_games_estimate=total_games_estimate
        )
        
        return {
            "sport": "tennis",
            "market_type": "game_handicap",
            "markets": [m.to_dict() for m in markets]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating game handicap: {str(e)}")


@router.get("/value-bets/extended")
async def get_extended_value_bets(
    sport: str = Query(..., description="football, basketball, or tennis"),
    min_edge: float = Query(0.05, ge=0, le=0.5, description="Minimum edge threshold")
):
    """
    Find value bets across all extended markets for a sport.
    
    Returns markets where predicted probability > market implied probability
    by at least the threshold.
    """
    try:
        # Generate sample markets to find value
        if sport == "football":
            markets = calculate_all_extended_markets(
                sport="football",
                prob_home=55,
                prob_draw=25,
                prob_away=20,
                home_aggression=0.7,
                away_aggression=0.5
            )
            
            value_bets = []
            
            # Check cards
            for m in markets.card_markets:
                if m.over_prob and m.over_odds:
                    implied = 100 / m.over_odds
                    edge = (m.over_prob - implied) / 100
                    if edge >= min_edge:
                        value_bets.append({
                            "market": f"Cards Over {m.line}",
                            "odds": m.over_odds,
                            "probability": m.over_prob,
                            "edge": round(edge, 3)
                        })
            
            # Check goal difference
            for m in markets.goal_diff_markets:
                if m.edge and m.edge >= min_edge:
                    value_bets.append({
                        "market": f"Win by {m.selection}",
                        "odds": m.odds,
                        "probability": m.probability,
                        "edge": round(m.edge, 3)
                    })
        
        elif sport == "basketball":
            markets = calculate_all_extended_markets(
                sport="basketball",
                prob_home=60,
                prob_draw=0,
                prob_away=40,
                spread=-4.5
            )
            
            value_bets = []
            for m in markets.win_margin_markets:
                if m.home_prob and m.home_odds:
                    implied = 100 / m.home_odds
                    edge = (m.home_prob - implied) / 100
                    if edge >= min_edge:
                        value_bets.append({
                            "market": f"Home wins by {m.range}",
                            "odds": m.home_odds,
                            "probability": m.home_prob,
                            "edge": round(edge, 3)
                        })
        
        elif sport == "tennis":
            markets = calculate_all_extended_markets(
                sport="tennis",
                prob_home=65,
                prob_draw=0,
                prob_away=35,
                sets_format=3
            )
            
            value_bets = []
            for m in markets.set_betting_markets:
                if m.home_prob and m.home_odds:
                    implied = 100 / m.home_odds
                    edge = (m.home_prob - implied) / 100
                    if edge >= min_edge:
                        value_bets.append({
                            "market": f"Home wins {m.score}",
                            "odds": m.home_odds,
                            "probability": m.home_prob,
                            "edge": round(edge, 3)
                        })
        
        else:
            raise HTTPException(status_code=400, detail="Invalid sport")
        
        # Sort by edge
        value_bets.sort(key=lambda x: x["edge"], reverse=True)
        
        return {
            "sport": sport,
            "min_edge": min_edge,
            "value_bets": value_bets[:20],
            "total_found": len(value_bets)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding value bets: {str(e)}")


@router.get("/explain/{market_type}")
async def explain_market(market_type: str):
    """
    Get explanation of how a specific market is calculated.
    
    Educational endpoint for understanding the models.
    """
    explanations = {
        "cards": {
            "description": "Football cards prediction based on team aggression and match context",
            "factors": [
                "Team aggression levels (tackles, duels won)",
                "Match importance (derbies have more cards)",
                "Referee strictness history",
                "League average cards per match"
            ],
            "model": "Poisson distribution with context adjustments",
            "typical_range": "3.5 - 6.5 cards per match"
        },
        "fouls": {
            "description": "Fouls prediction for football and basketball",
            "factors": [
                "Team playing style (physical vs technical)",
                "Match intensity",
                "Pace of play (basketball)"
            ],
            "model": "Poisson distribution",
            "typical_range_football": "18-24 fouls per match",
            "typical_range_basketball": "18-22 fouls per team"
        },
        "goal_difference": {
            "description": "Football win margin prediction",
            "factors": [
                "1X2 probabilities",
                "Expected goals (xG) difference",
                "Team scoring patterns"
            ],
            "model": "Normal distribution of goal difference",
            "note": "Most wins are by 1 goal (~45%), fewer by 2+ (~40%), rarely 3+ (~15%)"
        },
        "win_margin": {
            "description": "Basketball win margin ranges",
            "factors": [
                "Point spread",
                "Total points expectation",
                "Team scoring variance"
            ],
            "model": "Normal distribution of point differential",
            "typical_ranges": ["1-2 points", "3-6 points", "7-9 points", "10-14 points", "15+ points"]
        },
        "set_betting": {
            "description": "Tennis correct set score prediction",
            "factors": [
                "Match win probability",
                "Sets format (best-of-3 vs best-of-5)",
                "Player consistency"
            ],
            "model": "Binomial probability with set win probability",
            "note": "Straight set wins more likely for heavy favorites"
        },
        "game_handicap": {
            "description": "Tennis game handicap lines",
            "factors": [
                "Match win probability",
                "Expected total games",
                "Set scores distribution"
            ],
            "model": "Normal distribution of game differential",
            "typical_lines": "-4.5 to +4.5 games"
        }
    }
    
    if market_type not in explanations:
        raise HTTPException(status_code=404, detail=f"Unknown market type: {market_type}")
    
    return explanations[market_type]
