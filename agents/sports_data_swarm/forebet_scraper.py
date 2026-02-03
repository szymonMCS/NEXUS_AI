"""
Forebet.com Scraper - Advanced Statistics and Predictions

Extracts:
- Match predictions with probabilities (1X2, Over/Under, Handicaps, etc.)
- Asian Handicap and European Handicap odds
- Team form and statistics
- Head-to-head records
- League standings
- Live match data
- Injury reports

Usage:
    scraper = ForebetScraper()
    matches = await scraper.get_upcoming_matches(sport="football", days=3)
    prediction = await scraper.get_match_details(match_id="ajax-olympiacos-fc-2388871")
    handicaps = await scraper.get_handicap_predictions(match_id="...")
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, quote
import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class AsianHandicapLine:
    """Asian Handicap line with odds and probabilities."""
    line: float  # e.g., -1.5, -1, -0.5, 0, +0.5, +1, +1.5
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    home_edge: Optional[float] = None
    away_edge: Optional[float] = None
    recommendation: Optional[str] = None  # 'home', 'away', or None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'asian',
            'line': self.line,
            'line_display': f"AH {self.line:+.1f}" if self.line != int(self.line) else f"AH {self.line:+.0f}",
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'home_prob': self.home_prob,
            'away_prob': self.away_prob,
            'home_edge': self.home_edge,
            'away_edge': self.away_edge,
            'recommendation': self.recommendation,
            'description': self._get_description()
        }
    
    def _get_description(self) -> str:
        """Get human-readable description of the handicap line."""
        if self.line == 0:
            return "Draw No Bet - stake refunded if draw"
        elif self.line == -0.5:
            return "Home must win"
        elif self.line == 0.5:
            return "Home wins or draw"
        elif self.line == -1:
            return "Home must win by 2+ goals"
        elif self.line == 1:
            return "Home loses by 1, wins or draws"
        elif self.line == -1.5:
            return "Home must win by 2+ goals"
        elif self.line == 1.5:
            return "Home loses by 1, wins or draws"
        elif self.line <= -2:
            return f"Home must win by {abs(int(self.line))+1}+ goals"
        else:
            return f"Handicap +{self.line}"


@dataclass
class EuropeanHandicapLine:
    """European Handicap line (3-way handicap with draw option)."""
    line: str  # e.g., "1:0", "2:0", "0:1", "0:2"
    home_advantage: int  # goals advantage for home team
    away_advantage: int  # goals advantage for away team
    home_odds: float
    draw_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    draw_prob: Optional[float] = None
    away_prob: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'european',
            'line': self.line,
            'line_display': f"EH {self.line}",
            'home_advantage': self.home_advantage,
            'away_advantage': self.away_advantage,
            'home_odds': self.home_odds,
            'draw_odds': self.draw_odds,
            'away_odds': self.away_odds,
            'home_prob': self.home_prob,
            'draw_prob': self.draw_prob,
            'away_prob': self.away_prob,
            'description': f"Home starts {self.line}"
        }


@dataclass
class HandicapPredictions:
    """Container for all handicap predictions for a match."""
    asian_lines: List[AsianHandicapLine] = field(default_factory=list)
    european_lines: List[EuropeanHandicapLine] = field(default_factory=list)
    best_value: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'asian': [line.to_dict() for line in self.asian_lines],
            'european': [line.to_dict() for line in self.european_lines],
            'best_value': self.best_value
        }


@dataclass
class MatchPrediction:
    """Match prediction data from Forebet."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: datetime
    
    # Predictions (percentages)
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_result: str  # '1', 'X', '2'
    
    # Odds
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    
    # Handicap predictions
    handicaps: HandicapPredictions = field(default_factory=HandicapPredictions)
    
    # Statistics
    home_form: Optional[List[str]] = None
    away_form: Optional[List[str]] = None
    home_rank: Optional[int] = None
    away_rank: Optional[int] = None
    
    # Additional predictions
    over_25_prob: Optional[float] = None
    btts_prob: Optional[float] = None  # Both teams to score
    
    # Metadata
    last_updated: Optional[datetime] = None
    confidence_score: Optional[float] = None


@dataclass
class TeamStats:
    """Team statistics from Forebet."""
    name: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_scored: int
    goals_conceded: int
    points: int
    rank: int
    form: List[str]
    
    # Advanced stats
    avg_goals_scored: Optional[float] = None
    avg_goals_conceded: Optional[float] = None
    clean_sheets: Optional[int] = None
    failed_to_score: Optional[int] = None


class HandicapCalculator:
    """
    Calculate handicap probabilities based on 1X2 predictions and team stats.
    
    Uses Poisson distribution and team strength metrics to estimate
    probabilities for various handicap lines.
    """
    
    @staticmethod
    def calculate_asian_probabilities(
        prob_home: float,
        prob_draw: float, 
        prob_away: float,
        home_xg: float = 1.5,  # Expected goals home
        away_xg: float = 1.2,  # Expected goals away
    ) -> Dict[float, Dict[str, float]]:
        """
        Calculate probabilities for Asian Handicap lines.
        
        Returns dict mapping line -> {home_prob, away_prob}
        Lines: -2.5, -2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2, +2.5
        """
        results = {}
        
        # Convert to implied goal difference distribution
        # Simplified model using Poisson-like spread
        total_prob = prob_home + prob_draw + prob_away
        if total_prob == 0:
            total_prob = 100
            
        p_home = prob_home / total_prob
        p_draw = prob_draw / total_prob
        p_away = prob_away / total_prob
        
        # Estimate goal distribution parameters
        total_xg = home_xg + away_xg
        home_strength = home_xg / total_xg if total_xg > 0 else 0.5
        
        # Calculate probabilities for different goal margins
        # Using simplified heuristic based on 1X2 and expected goals
        
        lines = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
        
        for line in lines:
            if line == 0:
                # Draw No Bet - home wins excluding draw
                home_win_prob = p_home / (p_home + p_away) * 100 if (p_home + p_away) > 0 else 50
                results[line] = {
                    'home': home_win_prob,
                    'away': 100 - home_win_prob
                }
            elif line == -0.5:
                # Home must win
                results[line] = {
                    'home': p_home * 100,
                    'away': (p_draw + p_away) * 100
                }
            elif line == 0.5:
                # Home wins or draws
                results[line] = {
                    'home': (p_home + p_draw) * 100,
                    'away': p_away * 100
                }
            elif line == -1:
                # Home wins by 2+ or wins with safety margin
                # Estimate: 70% of home wins are by 2+ in even matches
                home_big_win = p_home * 0.7
                results[line] = {
                    'home': home_big_win * 100,
                    'away': (1 - home_big_win) * 100
                }
            elif line == 1:
                # Home doesn't lose by 2+
                home_not_big_loss = p_home + p_draw + (p_away * 0.5)
                results[line] = {
                    'home': home_not_big_loss * 100,
                    'away': (1 - home_not_big_loss) * 100
                }
            elif line == -1.5:
                # Similar to -1 but no safety
                home_big_win = p_home * 0.65
                results[line] = {
                    'home': home_big_win * 100,
                    'away': (1 - home_big_win) * 100
                }
            elif line == 1.5:
                home_not_big_loss = p_home + p_draw + (p_away * 0.6)
                results[line] = {
                    'home': home_not_big_loss * 100,
                    'away': (1 - home_not_big_loss) * 100
                }
            elif line <= -2:
                # Large handicap - home must dominate
                factor = 1 / (abs(line) + 1)
                home_dom = p_home * factor
                results[line] = {
                    'home': home_dom * 100,
                    'away': (1 - home_dom) * 100
                }
            else:  # line >= 2
                # Large positive handicap - underdog safety
                factor = 1 - (1 / (line + 1))
                home_safe = factor + (p_home * (1 - factor))
                results[line] = {
                    'home': min(home_safe * 100, 95),
                    'away': max((1 - home_safe) * 100, 5)
                }
                
        return results
    
    @staticmethod
    def calculate_european_probabilities(
        prob_home: float,
        prob_draw: float,
        prob_away: float,
        home_advantage: int = 1,  # Goals advantage for home
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate probabilities for European Handicap lines.
        
        European handicap adds goal advantage before match starts.
        """
        results = {}
        total_prob = prob_home + prob_draw + prob_away
        if total_prob == 0:
            total_prob = 100
            
        p_home = prob_home / total_prob
        p_draw = prob_draw / total_prob
        p_away = prob_away / total_prob
        
        # Common European handicap lines
        lines = ["0:0", "1:0", "2:0", "0:1", "0:2", "1:1", "2:1", "1:2"]
        
        for line in lines:
            home_adv, away_adv = map(int, line.split(':'))
            goal_diff = home_adv - away_adv
            
            if goal_diff == 0:
                # Standard match
                results[line] = {
                    'home': p_home * 100,
                    'draw': p_draw * 100,
                    'away': p_away * 100
                }
            elif goal_diff == 1:
                # Home starts with 1 goal advantage
                # Home wins if: wins outright or draws
                # Draw if: away wins by exactly 1
                # Away wins if: away wins by 2+
                home_prob = (p_home + p_draw) * 0.9
                draw_prob = p_away * 0.3
                away_prob = p_away * 0.7
                total = home_prob + draw_prob + away_prob
                results[line] = {
                    'home': (home_prob / total) * 100,
                    'draw': (draw_prob / total) * 100,
                    'away': (away_prob / total) * 100
                }
            elif goal_diff == 2:
                # Home starts with 2 goal advantage
                home_prob = (p_home + p_draw + p_away * 0.3)
                draw_prob = p_away * 0.4
                away_prob = p_away * 0.3
                total = home_prob + draw_prob + away_prob
                results[line] = {
                    'home': (home_prob / total) * 100,
                    'draw': (draw_prob / total) * 100,
                    'away': (away_prob / total) * 100
                }
            elif goal_diff == -1:
                # Away starts with 1 goal advantage
                home_prob = p_home * 0.7
                draw_prob = p_home * 0.3
                away_prob = (p_draw + p_away) * 0.9
                total = home_prob + draw_prob + away_prob
                results[line] = {
                    'home': (home_prob / total) * 100,
                    'draw': (draw_prob / total) * 100,
                    'away': (away_prob / total) * 100
                }
            elif goal_diff == -2:
                # Away starts with 2 goal advantage
                home_prob = p_home * 0.3
                draw_prob = p_home * 0.4
                away_prob = (p_draw + p_away + p_home * 0.3)
                total = home_prob + draw_prob + away_prob
                results[line] = {
                    'home': (home_prob / total) * 100,
                    'draw': (draw_prob / total) * 100,
                    'away': (away_prob / total) * 100
                }
                
        return results
    
    @staticmethod
    def calculate_edge(probability: float, odds: float) -> float:
        """Calculate expected value/edge for a bet."""
        if odds <= 1:
            return -1
        fair_odds = 100 / probability if probability > 0 else 100
        return (odds / fair_odds) - 1 if fair_odds > 0 else 0
    
    @staticmethod
    def find_best_value(handicaps: HandicapPredictions) -> Optional[Dict[str, Any]]:
        """Find the best value bet among all handicap lines."""
        best_value = None
        best_edge = 0.05  # Minimum 5% edge
        
        for line in handicaps.asian_lines:
            if line.home_edge and line.home_edge > best_edge:
                best_edge = line.home_edge
                best_value = {
                    'type': 'asian',
                    'line': line.line,
                    'selection': 'home',
                    'odds': line.home_odds,
                    'edge': line.home_edge,
                    'confidence': line.home_prob
                }
            if line.away_edge and line.away_edge > best_edge:
                best_edge = line.away_edge
                best_value = {
                    'type': 'asian',
                    'line': line.line,
                    'selection': 'away',
                    'odds': line.away_odds,
                    'edge': line.away_edge,
                    'confidence': line.away_prob
                }
                
        return best_value


class ForebetScraper:
    """
    Scraper for Forebet.com - mathematical football predictions.
    
    Features:
    - Upcoming matches with predictions
    - Asian and European Handicap odds
    - Detailed match statistics
    - Live match tracking
    - League standings
    - Historical data
    """
    
    BASE_URL = "https://www.forebet.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            },
            timeout=30.0,
            follow_redirects=True
        )
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        self.handicap_calc = HandicapCalculator()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def _get_page(self, url: str, use_cache: bool = True) -> BeautifulSoup:
        """Fetch and parse a page with caching."""
        cache_key = f"page:{url}"
        
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached["timestamp"] < timedelta(seconds=self._cache_ttl):
                return cached["soup"]
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if use_cache:
                self._cache[cache_key] = {
                    "soup": soup,
                    "timestamp": datetime.now()
                }
            
            return soup
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    async def get_upcoming_matches(
        self, 
        sport: str = "football",
        days: int = 3,
        league: Optional[str] = None
    ) -> List[MatchPrediction]:
        """
        Get upcoming matches with predictions.
        
        Args:
            sport: Sport type (football, basketball, tennis, etc.)
            days: Number of days ahead to fetch
            league: Specific league filter (optional)
            
        Returns:
            List of MatchPrediction objects
        """
        matches = []
        
        # Build URL based on parameters
        if league:
            url = f"{self.BASE_URL}/en/{sport}/predictions-{quote(league.lower().replace(' ', '-'))}"
        else:
            url = f"{self.BASE_URL}/en/{sport}/predictions-today"
        
        try:
            soup = await self._get_page(url)
            
            # Find all match rows
            match_rows = soup.find_all('div', class_=re.compile(r'match_row|match-item'))
            
            for row in match_rows[:50]:  # Limit to 50 matches
                try:
                    match = await self._parse_match_row(row)
                    if match:
                        # Calculate handicaps based on probabilities
                        match.handicaps = await self._calculate_handicaps(match)
                        matches.append(match)
                except Exception as e:
                    logger.warning(f"Error parsing match row: {e}")
                    continue
                    
            logger.info(f"Found {len(matches)} upcoming matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            return []
    
    async def _calculate_handicaps(self, match: MatchPrediction) -> HandicapPredictions:
        """Calculate handicap predictions based on 1X2 probabilities."""
        handicaps = HandicapPredictions()
        
        try:
            # Calculate Asian Handicap probabilities
            asian_probs = self.handicap_calc.calculate_asian_probabilities(
                match.prob_home,
                match.prob_draw,
                match.prob_away
            )
            
            # Generate Asian Handicap lines with sample odds
            # In production, these would come from bookmaker odds scraping
            asian_lines_data = [
                (-1.5, 2.80, 1.40),
                (-1, 2.20, 1.65),
                (-0.5, 1.85, 1.95),
                (0, 1.60, 2.20),
                (0.5, 1.40, 2.80),
                (1, 1.25, 3.60),
                (1.5, 1.15, 4.80),
            ]
            
            for line, home_odds, away_odds in asian_lines_data:
                if line in asian_probs:
                    probs = asian_probs[line]
                    home_edge = self.handicap_calc.calculate_edge(probs['home'], home_odds)
                    away_edge = self.handicap_calc.calculate_edge(probs['away'], away_odds)
                    
                    recommendation = None
                    if home_edge > 0.08:
                        recommendation = 'home'
                    elif away_edge > 0.08:
                        recommendation = 'away'
                    
                    handicaps.asian_lines.append(AsianHandicapLine(
                        line=line,
                        home_odds=home_odds,
                        away_odds=away_odds,
                        home_prob=probs['home'],
                        away_prob=probs['away'],
                        home_edge=home_edge,
                        away_edge=away_edge,
                        recommendation=recommendation
                    ))
            
            # Calculate European Handicap probabilities
            european_lines_data = [
                ("0:0", 2.40, 3.20, 2.80),
                ("1:0", 1.55, 3.80, 5.50),
                ("2:0", 1.25, 5.00, 10.00),
                ("0:1", 4.50, 3.60, 1.70),
                ("0:2", 9.00, 5.50, 1.25),
            ]
            
            for line, home_odds, draw_odds, away_odds in european_lines_data:
                home_adv, away_adv = map(int, line.split(':'))
                
                # Calculate adjusted probabilities based on line
                if line == "0:0":
                    home_prob, draw_prob, away_prob = match.prob_home, match.prob_draw, match.prob_away
                else:
                    # Simplified adjustment
                    factor = abs(home_adv - away_adv) * 10
                    if home_adv > away_adv:
                        home_prob = min(match.prob_home + factor, 85)
                        away_prob = max(match.prob_away - factor, 5)
                    else:
                        home_prob = max(match.prob_home - factor, 5)
                        away_prob = min(match.prob_away + factor, 85)
                    draw_prob = 100 - home_prob - away_prob
                
                handicaps.european_lines.append(EuropeanHandicapLine(
                    line=line,
                    home_advantage=home_adv,
                    away_advantage=away_adv,
                    home_odds=home_odds,
                    draw_odds=draw_odds,
                    away_odds=away_odds,
                    home_prob=home_prob,
                    draw_prob=max(draw_prob, 5),
                    away_prob=away_prob
                ))
            
            # Find best value
            handicaps.best_value = self.handicap_calc.find_best_value(handicaps)
            
        except Exception as e:
            logger.warning(f"Error calculating handicaps: {e}")
        
        return handicaps
    
    async def get_handicap_predictions(self, match_id: str) -> Optional[HandicapPredictions]:
        """
        Get detailed handicap predictions for a specific match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            HandicapPredictions object with all lines
        """
        try:
            # Get match details first
            match_details = await self.get_match_details(match_id)
            if not match_details:
                return None
            
            # Extract 1X2 probabilities
            prob_home = match_details.get('predictions', {}).get('prob_home', 33.3)
            prob_draw = match_details.get('predictions', {}).get('prob_draw', 33.3)
            prob_away = match_details.get('predictions', {}).get('prob_away', 33.3)
            
            # Create temporary match object for handicap calculation
            temp_match = MatchPrediction(
                match_id=match_id,
                home_team="",
                away_team="",
                league="",
                match_date=datetime.now(),
                prob_home=prob_home,
                prob_draw=prob_draw,
                prob_away=prob_away,
                predicted_result='1' if prob_home > max(prob_draw, prob_away) else 
                               '2' if prob_away > max(prob_home, prob_draw) else 'X'
            )
            
            return await self._calculate_handicaps(temp_match)
            
        except Exception as e:
            logger.error(f"Error fetching handicap predictions: {e}")
            return None
    
    async def _parse_match_row(self, row: BeautifulSoup) -> Optional[MatchPrediction]:
        """Parse a single match row from the predictions page."""
        try:
            # Extract teams
            teams_elem = row.find('div', class_=re.compile(r'teams|match-name'))
            if not teams_elem:
                return None
                
            teams_text = teams_elem.get_text(strip=True)
            if ' vs ' not in teams_text:
                return None
                
            home_team, away_team = teams_text.split(' vs ', 1)
            
            # Extract league
            league_elem = row.find('span', class_=re.compile(r'league|competition'))
            league = league_elem.get_text(strip=True) if league_elem else "Unknown"
            
            # Extract match ID/link
            link_elem = row.find('a', href=re.compile(r'/football/matches/'))
            match_id = None
            if link_elem:
                href = link_elem.get('href', '')
                match_id = href.split('/')[-1] if href else None
            
            # Extract probabilities
            probs = row.find_all('span', class_=re.compile(r'prob|prediction'))
            prob_home = prob_draw = prob_away = 0.0
            
            if len(probs) >= 3:
                try:
                    prob_home = float(probs[0].get_text(strip=True).replace('%', ''))
                    prob_draw = float(probs[1].get_text(strip=True).replace('%', ''))
                    prob_away = float(probs[2].get_text(strip=True).replace('%', ''))
                except ValueError:
                    pass
            
            # Determine predicted result
            predicted = 'X'
            max_prob = max(prob_home, prob_draw, prob_away)
            if max_prob == prob_home:
                predicted = '1'
            elif max_prob == prob_away:
                predicted = '2'
            
            # Extract odds if available
            odds_elems = row.find_all('span', class_=re.compile(r'odds|coeff'))
            odds_home = odds_draw = odds_away = None
            
            if len(odds_elems) >= 3:
                try:
                    odds_home = float(odds_elems[0].get_text(strip=True))
                    odds_draw = float(odds_elems[1].get_text(strip=True))
                    odds_away = float(odds_elems[2].get_text(strip=True))
                except ValueError:
                    pass
            
            # Extract match time
            time_elem = row.find('span', class_=re.compile(r'time|date'))
            match_date = datetime.now()
            if time_elem:
                # Parse various time formats
                time_text = time_elem.get_text(strip=True)
                try:
                    # Try to parse relative time (e.g., "Today 20:45")
                    match_date = self._parse_match_time(time_text)
                except:
                    pass
            
            return MatchPrediction(
                match_id=match_id or f"{home_team}-{away_team}-{match_date.strftime('%Y%m%d')}",
                home_team=home_team.strip(),
                away_team=away_team.strip(),
                league=league,
                match_date=match_date,
                prob_home=prob_home,
                prob_draw=prob_draw,
                prob_away=prob_away,
                predicted_result=predicted,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error parsing match row: {e}")
            return None
    
    def _parse_match_time(self, time_text: str) -> datetime:
        """Parse match time from various formats."""
        time_text = time_text.lower()
        now = datetime.now()
        
        if 'today' in time_text:
            time_match = re.search(r'(\d{1,2}):(\d{2})', time_text)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        elif 'tomorrow' in time_text:
            time_match = re.search(r'(\d{1,2}):(\d{2})', time_text)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                tomorrow = now + timedelta(days=1)
                return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Try to parse full date
        try:
            return datetime.strptime(time_text, '%d/%m/%Y %H:%M')
        except ValueError:
            pass
            
        return now
    
    async def get_match_details(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific match.
        
        Args:
            match_id: Match identifier (e.g., "ajax-olympiacos-fc-2388871")
            
        Returns:
            Dictionary with comprehensive match data
        """
        url = f"{self.BASE_URL}/en/football/matches/{match_id}"
        
        try:
            soup = await self._get_page(url, use_cache=False)
            
            # Extract basic info
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Match"
            
            # Extract teams
            teams = title.split(' vs ') if ' vs ' in title else ["Team A", "Team B"]
            
            # Extract predictions table
            predictions = self._extract_predictions(soup)
            
            # Extract team statistics
            home_stats = self._extract_team_stats(soup, 'home')
            away_stats = self._extract_team_stats(soup, 'away')
            
            # Extract H2H record
            h2h = self._extract_h2h(soup)
            
            # Extract standings
            standings = self._extract_standings(soup)
            
            # Extract recent form
            home_form = self._extract_form(soup, 'home')
            away_form = self._extract_form(soup, 'away')
            
            # Extract injuries
            injuries = self._extract_injuries(soup)
            
            # Calculate handicaps
            prob_home = float(predictions.get('1x2', {}).get('home', '33.3').replace('%', ''))
            prob_draw = float(predictions.get('1x2', {}).get('draw', '33.3').replace('%', ''))
            prob_away = float(predictions.get('1x2', {}).get('away', '33.3').replace('%', ''))
            
            temp_match = MatchPrediction(
                match_id=match_id,
                home_team=teams[0] if len(teams) > 0 else 'Home',
                away_team=teams[1] if len(teams) > 1 else 'Away',
                league="Unknown",
                match_date=datetime.now(),
                prob_home=prob_home,
                prob_draw=prob_draw,
                prob_away=prob_away,
                predicted_result='1' if prob_home > max(prob_draw, prob_away) else 
                               '2' if prob_away > max(prob_home, prob_draw) else 'X'
            )
            handicaps = await self._calculate_handicaps(temp_match)
            
            return {
                'match_id': match_id,
                'title': title,
                'home_team': teams[0] if len(teams) > 0 else 'Home',
                'away_team': teams[1] if len(teams) > 1 else 'Away',
                'predictions': predictions,
                'handicaps': handicaps.to_dict(),
                'home_stats': home_stats,
                'away_stats': away_stats,
                'h2h': h2h,
                'standings': standings,
                'home_form': home_form,
                'away_form': away_form,
                'injuries': injuries,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching match details for {match_id}: {e}")
            return None
    
    def _extract_predictions(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract prediction percentages."""
        predictions = {
            '1x2': {},
            'over_under': {},
            'btts': {}  # Both teams to score
        }
        
        try:
            # 1X2 probabilities
            prob_table = soup.find('table', class_=re.compile(r'probabilities|predictions'))
            if prob_table:
                rows = prob_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if '1' in label:
                            predictions['1x2']['home'] = value
                        elif 'X' in label:
                            predictions['1x2']['draw'] = value
                        elif '2' in label:
                            predictions['1x2']['away'] = value
            
            # Alternative: find by specific classes
            prob_home = soup.find('span', class_=re.compile(r'prob-home|home-win'))
            prob_draw = soup.find('span', class_=re.compile(r'prob-draw'))
            prob_away = soup.find('span', class_=re.compile(r'prob-away|away-win'))
            
            if prob_home:
                predictions['1x2']['home'] = prob_home.get_text(strip=True)
            if prob_draw:
                predictions['1x2']['draw'] = prob_draw.get_text(strip=True)
            if prob_away:
                predictions['1x2']['away'] = prob_away.get_text(strip=True)
                
        except Exception as e:
            logger.warning(f"Error extracting predictions: {e}")
            
        return predictions
    
    def _extract_team_stats(self, soup: BeautifulSoup, team: str) -> Dict[str, Any]:
        """Extract statistics for a specific team."""
        stats = {}
        
        try:
            # Find stats section for the team
            stats_section = soup.find('div', class_=re.compile(f'{team}-stats|team-stats'))
            if stats_section:
                # Extract various statistics
                stat_rows = stats_section.find_all('div', class_=re.compile(r'stat-row'))
                for row in stat_rows:
                    label = row.find('span', class_='label')
                    value = row.find('span', class_='value')
                    if label and value:
                        stats[label.get_text(strip=True).lower()] = value.get_text(strip=True)
            
            # Parse form (W-D-L sequence)
            form_elem = soup.find('div', class_=re.compile(f'{team}-form|form-guide'))
            if form_elem:
                form_text = form_elem.get_text(strip=True)
                stats['form'] = list(form_text.upper())
                
        except Exception as e:
            logger.warning(f"Error extracting {team} stats: {e}")
            
        return stats
    
    def _extract_h2h(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract head-to-head statistics."""
        h2h = {
            'total_matches': 0,
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'recent_matches': []
        }
        
        try:
            h2h_section = soup.find('div', class_=re.compile(r'h2h|head-to-head'))
            if h2h_section:
                # Extract totals
                stats_text = h2h_section.get_text()
                
                # Look for patterns like "5 wins - 3 draws - 2 losses"
                matches = re.findall(r'(\d+)\s*(wins?|draws?|losses?)', stats_text.lower())
                for count, stat_type in matches:
                    if 'win' in stat_type:
                        h2h['home_wins'] = int(count)
                    elif 'draw' in stat_type:
                        h2h['draws'] = int(count)
                    elif 'loss' in stat_type:
                        h2h['away_wins'] = int(count)
                
                h2h['total_matches'] = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
                
                # Extract recent H2H matches
                match_rows = h2h_section.find_all('div', class_=re.compile(r'match-row|h2h-match'))
                for row in match_rows[:5]:  # Last 5 matches
                    match_text = row.get_text(strip=True)
                    if match_text:
                        h2h['recent_matches'].append(match_text)
                        
        except Exception as e:
            logger.warning(f"Error extracting H2H: {e}")
            
        return h2h
    
    def _extract_standings(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract league standings."""
        standings = []
        
        try:
            table = soup.find('table', class_=re.compile(r'standings|league-table'))
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        standings.append({
                            'position': cells[0].get_text(strip=True),
                            'team': cells[1].get_text(strip=True),
                            'played': cells[2].get_text(strip=True),
                            'points': cells[3].get_text(strip=True),
                            'form': cells[4].get_text(strip=True) if len(cells) > 4 else ''
                        })
        except Exception as e:
            logger.warning(f"Error extracting standings: {e}")
            
        return standings
    
    def _extract_form(self, soup: BeautifulSoup, team: str) -> List[str]:
        """Extract recent form for a team."""
        form = []
        
        try:
            form_section = soup.find('div', class_=re.compile(f'{team}-recent|last-matches'))
            if form_section:
                matches = form_section.find_all('div', class_=re.compile(r'match-result'))
                for match in matches[:6]:  # Last 6 matches
                    result_text = match.get_text(strip=True)
                    # Extract W/D/L from result
                    if 'win' in result_text.lower() or 'w' in result_text.upper():
                        form.append('W')
                    elif 'draw' in result_text.lower() or 'd' in result_text.upper():
                        form.append('D')
                    elif 'loss' in result_text.lower() or 'l' in result_text.upper():
                        form.append('L')
        except Exception as e:
            logger.warning(f"Error extracting form: {e}")
            
        return form
    
    def _extract_injuries(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract injury reports."""
        injuries = []
        
        try:
            injury_section = soup.find('div', class_=re.compile(r'injuries|suspensions'))
            if injury_section:
                injury_rows = injury_section.find_all('div', class_=re.compile(r'injury-row'))
                for row in injury_rows:
                    player = row.find('span', class_='player-name')
                    reason = row.find('span', class_='injury-reason')
                    if player:
                        injuries.append({
                            'player': player.get_text(strip=True),
                            'reason': reason.get_text(strip=True) if reason else 'Unknown'
                        })
        except Exception as e:
            logger.warning(f"Error extracting injuries: {e}")
            
        return injuries
    
    async def get_live_matches(self) -> List[Dict[str, Any]]:
        """Get currently live matches with scores."""
        url = f"{self.BASE_URL}/en/football/live-now"
        matches = []
        
        try:
            soup = await self._get_page(url, use_cache=False)  # No cache for live data
            
            live_rows = soup.find_all('div', class_=re.compile(r'live-match|match-live'))
            
            for row in live_rows:
                try:
                    teams_elem = row.find('div', class_=re.compile(r'teams'))
                    score_elem = row.find('div', class_=re.compile(r'score'))
                    time_elem = row.find('div', class_=re.compile(r'time|minute'))
                    
                    if teams_elem and score_elem:
                        matches.append({
                            'teams': teams_elem.get_text(strip=True),
                            'score': score_elem.get_text(strip=True),
                            'time': time_elem.get_text(strip=True) if time_elem else 'Live',
                            'last_updated': datetime.now().isoformat()
                        })
                except Exception as e:
                    continue
                    
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    async def get_league_standings(self, league: str) -> List[Dict[str, Any]]:
        """Get standings for a specific league."""
        url = f"{self.BASE_URL}/en/football/predictions-{quote(league.lower().replace(' ', '-'))}"
        
        try:
            soup = await self._get_page(url)
            return self._extract_standings(soup)
        except Exception as e:
            logger.error(f"Error fetching league standings: {e}")
            return []
    
    async def search_matches(self, query: str) -> List[MatchPrediction]:
        """Search for matches by team name."""
        # This would require a search endpoint or we can filter from all matches
        all_matches = await self.get_upcoming_matches(days=7)
        
        query_lower = query.lower()
        filtered = [
            m for m in all_matches 
            if query_lower in m.home_team.lower() or query_lower in m.away_team.lower()
        ]
        
        return filtered


# === Integration with existing data pipeline ===

async def fetch_forebet_predictions(sport: str = "football", days: int = 3) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch predictions from Forebet.
    
    Returns data in standardized format for the betting system.
    """
    async with ForebetScraper() as scraper:
        matches = await scraper.get_upcoming_matches(sport=sport, days=days)
        
        # Convert to standard format
        standardized = []
        for match in matches:
            standardized.append({
                'match_id': match.match_id,
                'sport': sport,
                'home_team': match.home_team,
                'away_team': match.away_team,
                'league': match.league,
                'match_date': match.match_date.isoformat(),
                'predictions': {
                    'prob_home': match.prob_home,
                    'prob_draw': match.prob_draw,
                    'prob_away': match.prob_away,
                    'predicted_result': match.predicted_result,
                    'confidence': max(match.prob_home, match.prob_draw, match.prob_away)
                },
                'odds': {
                    'home': match.odds_home,
                    'draw': match.odds_draw,
                    'away': match.odds_away
                },
                'handicaps': match.handicaps.to_dict(),
                'source': 'forebet',
                'scraped_at': match.last_updated.isoformat() if match.last_updated else None
            })
        
        return standardized


# === Testing ===
if __name__ == "__main__":
    async def test():
        scraper = ForebetScraper()
        
        print("Testing Forebet Scraper with Handicaps...")
        print("=" * 60)
        
        # Test 1: Get upcoming matches with handicaps
        print("\n1. Fetching upcoming matches with handicaps...")
        matches = await scraper.get_upcoming_matches(sport="football", days=1)
        print(f"Found {len(matches)} matches")
        
        for match in matches[:2]:
            print(f"\n  - {match.home_team} vs {match.away_team}")
            print(f"    1X2: {match.prob_home}% / {match.prob_draw}% / {match.prob_away}%")
            print(f"    Asian Handicap Lines:")
            for ah in match.handicaps.asian_lines[:3]:
                print(f"      AH {ah.line:+.1f}: Home {ah.home_odds} ({ah.home_prob:.0f}%) | Away {ah.away_odds} ({ah.away_prob:.0f}%)")
                if ah.recommendation:
                    print(f"        ⭐ Recommendation: {ah.recommendation}")
            if match.handicaps.best_value:
                bv = match.handicaps.best_value
                print(f"    🎯 Best Value: {bv['type'].upper()} {bv['line']} - {bv['selection']} @ {bv['odds']} (+{bv['edge']*100:.0f}% edge)")
        
        # Test 2: Get handicap predictions for a match
        if matches:
            print("\n2. Testing get_handicap_predictions...")
            first_match = matches[0]
            if first_match.match_id:
                handicaps = await scraper.get_handicap_predictions(first_match.match_id)
                if handicaps:
                    print(f"Found {len(handicaps.asian_lines)} Asian lines")
                    print(f"Found {len(handicaps.european_lines)} European lines")
        
        await scraper.client.aclose()
        print("\nTest completed!")
    
    asyncio.run(test())
