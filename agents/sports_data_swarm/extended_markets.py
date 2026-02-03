"""
Extended Markets Calculator

Calculates probabilities for alternative betting markets:
- Football: Cards, Fouls, Goal difference
- Basketball: Fouls, Win margin
- Tennis: Set betting, Game handicaps
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum


class SportType(Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"


# ==================== FOOTBALL EXTENDED MARKETS ====================

@dataclass
class CardMarket:
    """Football cards market."""
    type: str  # 'total_cards', 'asian_cards', 'team_cards'
    line: float
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    expected_cards: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'line': self.line,
            'over_odds': self.over_odds,
            'under_odds': self.under_odds,
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'over_prob': self.over_prob,
            'under_prob': self.under_prob,
            'expected_cards': self.expected_cards
        }


@dataclass
class FoulMarket:
    """Fouls market for any sport."""
    type: str  # 'total_fouls', 'foul_handicap', 'team_fouls'
    line: float
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    expected_fouls: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'line': self.line,
            'over_odds': self.over_odds,
            'under_odds': self.under_odds,
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'over_prob': self.over_prob,
            'under_prob': self.under_prob,
            'expected_fouls': self.expected_fouls
        }


@dataclass
class GoalDifferenceMarket:
    """Football goal difference / win margin."""
    selection: str  # e.g., "1 goal", "2+ goals", "exactly 1"
    odds: float
    probability: Optional[float] = None
    edge: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'selection': self.selection,
            'odds': self.odds,
            'probability': self.probability,
            'edge': self.edge
        }


# ==================== BASKETBALL EXTENDED MARKETS ====================

@dataclass
class WinMarginMarket:
    """Basketball win margin ranges."""
    range: str  # e.g., "1-5", "6-10", "11+", "1-2", "3-6"
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'range': self.range,
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'home_prob': self.home_prob,
            'away_prob': self.away_prob,
            'recommendation': self.recommendation
        }


# ==================== TENNIS EXTENDED MARKETS ====================

@dataclass
class SetBettingMarket:
    """Tennis correct set score."""
    score: str  # e.g., "2-0", "2-1", "3-0", "3-1", "3-2"
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    edge: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'home_prob': self.home_prob,
            'away_prob': self.away_prob,
            'edge': self.edge
        }


@dataclass
class GameHandicapMarket:
    """Tennis game handicap."""
    line: float
    home_odds: float
    away_odds: float
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line': self.line,
            'home_odds': self.home_odds,
            'away_odds': self.away_odds,
            'home_prob': self.home_prob,
            'away_prob': self.away_prob,
            'recommendation': self.recommendation
        }


@dataclass
class ExtendedMarkets:
    """Container for all extended markets."""
    sport: SportType
    card_markets: List[CardMarket] = None
    foul_markets: List[FoulMarket] = None
    goal_diff_markets: List[GoalDifferenceMarket] = None
    win_margin_markets: List[WinMarginMarket] = None
    set_betting_markets: List[SetBettingMarket] = None
    game_handicap_markets: List[GameHandicapMarket] = None
    
    def __post_init__(self):
        if self.card_markets is None:
            self.card_markets = []
        if self.foul_markets is None:
            self.foul_markets = []
        if self.goal_diff_markets is None:
            self.goal_diff_markets = []
        if self.win_margin_markets is None:
            self.win_margin_markets = []
        if self.set_betting_markets is None:
            self.set_betting_markets = []
        if self.game_handicap_markets is None:
            self.game_handicap_markets = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sport': self.sport.value,
            'card_markets': [m.to_dict() for m in self.card_markets],
            'foul_markets': [m.to_dict() for m in self.foul_markets],
            'goal_diff_markets': [m.to_dict() for m in self.goal_diff_markets],
            'win_margin_markets': [m.to_dict() for m in self.win_margin_markets],
            'set_betting_markets': [m.to_dict() for m in self.set_betting_markets],
            'game_handicap_markets': [m.to_dict() for m in self.game_handicap_markets],
        }


class ExtendedMarketsCalculator:
    """
    Calculate probabilities for extended/alternative betting markets.
    """
    
    @staticmethod
    def calculate_football_cards(
        home_aggression: float = 0.5,  # 0-1 scale
        away_aggression: float = 0.5,
        match_importance: float = 0.5,  # derby, playoff, etc.
        referee_strictness: float = 0.5,
    ) -> List[CardMarket]:
        """
        Calculate expected cards based on team aggression and match context.
        
        Args:
            home_aggression: How aggressive home team plays (0-1)
            away_aggression: How aggressive away team plays (0-1)
            match_importance: Importance of match (0-1)
            referee_strictness: Referee card tendency (0-1)
        
        Returns:
            List of CardMarket objects
        """
        # Base cards per team per match
        base_cards = 2.5
        
        # Calculate expected total cards
        expected_cards = (
            base_cards * 2 +
            (home_aggression + away_aggression) * 2 +
            match_importance * 1.5 +
            referee_strictness * 1
        )
        
        markets = []
        
        # Total cards lines
        total_lines = [3.5, 4.5, 5.5, 6.5]
        for line in total_lines:
            # Poisson-based probability calculation
            over_prob = ExtendedMarketsCalculator._poisson_over(expected_cards, line)
            under_prob = 100 - over_prob
            
            # Sample odds (would come from bookmakers in production)
            over_odds = 100 / over_prob * 0.9 if over_prob > 0 else 10
            under_odds = 100 / under_prob * 0.9 if under_prob > 0 else 10
            
            markets.append(CardMarket(
                type='total_cards',
                line=line,
                over_odds=round(over_odds, 2),
                under_odds=round(under_odds, 2),
                over_prob=round(over_prob, 1),
                under_prob=round(under_prob, 1),
                expected_cards=round(expected_cards, 1)
            ))
        
        # Asian cards handicap (typically team cards)
        home_expected = base_cards + home_aggression * 2
        away_expected = base_cards + away_aggression * 2
        diff = home_expected - away_expected
        
        if abs(diff) > 0.5:
            line = round(diff * 2) / 2  # Round to nearest 0.5
            home_prob = 50 + diff * 15
            away_prob = 100 - home_prob
            
            markets.append(CardMarket(
                type='asian_cards',
                line=line,
                home_odds=round(100 / home_prob * 0.9, 2) if home_prob > 0 else 10,
                away_odds=round(100 / away_prob * 0.9, 2) if away_prob > 0 else 10,
                expected_cards=round(expected_cards, 1)
            ))
        
        return markets
    
    @staticmethod
    def calculate_football_fouls(
        home_foul_tendency: float = 0.5,
        away_foul_tendency: float = 0.5,
        match_intensity: float = 0.5,
    ) -> List[FoulMarket]:
        """Calculate foul markets for football."""
        base_fouls = 11  # Average fouls per team per match
        
        expected_home = base_fouls + home_foul_tendency * 4 + match_intensity * 2
        expected_away = base_fouls + away_foul_tendency * 4 + match_intensity * 2
        expected_total = expected_home + expected_away
        
        markets = []
        
        # Total fouls
        for line in [19.5, 21.5, 23.5]:
            over_prob = ExtendedMarketsCalculator._poisson_over(expected_total, line)
            under_prob = 100 - over_prob
            
            markets.append(FoulMarket(
                type='total_fouls',
                line=line,
                over_odds=round(100 / over_prob * 0.9, 2) if over_prob > 0 else 10,
                under_odds=round(100 / under_prob * 0.9, 2) if under_prob > 0 else 10,
                over_prob=round(over_prob, 1),
                under_prob=round(under_prob, 1),
                expected_fouls=round(expected_total, 1)
            ))
        
        # Team fouls handicap
        diff = expected_home - expected_away
        if abs(diff) > 1:
            line = round(diff)
            home_prob = 50 + diff * 5
            away_prob = 100 - home_prob
            
            markets.append(FoulMarket(
                type='foul_handicap',
                line=line,
                home_odds=round(100 / home_prob * 0.9, 2),
                away_odds=round(100 / away_prob * 0.9, 2),
                expected_fouls=round(expected_total, 1)
            ))
        
        return markets
    
    @staticmethod
    def calculate_goal_difference(
        prob_home: float,
        prob_draw: float,
        prob_away: float,
        home_xg: float = 1.5,
        away_xg: float = 1.2,
    ) -> List[GoalDifferenceMarket]:
        """
        Calculate goal difference / win margin markets.
        
        Args:
            prob_home: Probability of home win (%)
            prob_draw: Probability of draw (%)
            prob_away: Probability of away win (%)
            home_xg: Home expected goals
            away_xg: Away expected goals
        """
        markets = []
        
        # Normalize probabilities
        total = prob_home + prob_draw + prob_away
        p_home = prob_home / total
        p_draw = prob_draw / total
        p_away = prob_away / total
        
        # Expected goal difference distribution
        xg_diff = home_xg - away_xg
        variance = (home_xg + away_xg) * 0.5  # Approximate
        
        # Win by exactly 1 goal
        home_by_1 = p_home * 0.45  # ~45% of wins are by 1 goal
        away_by_1 = p_away * 0.45
        
        # Win by 2+ goals
        home_by_2plus = p_home * 0.40
        away_by_2plus = p_away * 0.40
        
        # Win by 3+ goals
        home_by_3plus = p_home * 0.15
        away_by_3plus = p_away * 0.15
        
        # Draw (exactly 0 difference)
        draw_prob = p_draw
        
        selections = [
            ("Home by 1", home_by_1),
            ("Home by 2+", home_by_2plus),
            ("Home by 3+", home_by_3plus),
            ("Draw", draw_prob),
            ("Away by 1", away_by_1),
            ("Away by 2+", away_by_2plus),
            ("Away by 3+", away_by_3plus),
            ("Any Home Win", p_home),
            ("Any Away Win", p_away),
        ]
        
        for selection, prob in selections:
            if prob > 0.05:  # Only include if probability > 5%
                odds = 100 / (prob * 100) * 0.9
                markets.append(GoalDifferenceMarket(
                    selection=selection,
                    odds=round(odds, 2),
                    probability=round(prob * 100, 1),
                    edge=round((odds - 100/(prob*100)) / (100/(prob*100)), 3) if prob > 0 else 0
                ))
        
        return markets
    
    @staticmethod
    def calculate_basketball_win_margin(
        spread: float,  # Expected point spread
        total_points: float = 210,
    ) -> List[WinMarginMarket]:
        """
        Calculate basketball win margin ranges.
        
        Args:
            spread: Expected point difference (positive = home favored)
            total_points: Expected total points
        """
        markets = []
        
        # Standard NBA margin ranges
        ranges = [
            ("1-2", 1, 2),
            ("3-6", 3, 6),
            ("7-9", 7, 9),
            ("10-14", 10, 14),
            ("15+", 15, 50),
        ]
        
        # Simplified model - normal distribution of margin
        import statistics
        std_dev = math.sqrt(total_points) * 0.4  # Approximate standard deviation
        
        for range_name, low, high in ranges:
            # Calculate probability of margin in this range
            # Using normal distribution approximation
            
            # Home wins by this margin
            home_prob = ExtendedMarketsCalculator._normal_cdf(high, spread, std_dev) - \
                       ExtendedMarketsCalculator._normal_cdf(low, spread, std_dev)
            
            # Away wins by this margin (mirror)
            away_prob = ExtendedMarketsCalculator._normal_cdf(-low, -spread, std_dev) - \
                       ExtendedMarketsCalculator._normal_cdf(-high, -spread, std_dev)
            
            if home_prob > 0.02 or away_prob > 0.02:
                markets.append(WinMarginMarket(
                    range=range_name,
                    home_odds=round(100 / (home_prob * 100) * 0.9, 2) if home_prob > 0 else None,
                    away_odds=round(100 / (away_prob * 100) * 0.9, 2) if away_prob > 0 else None,
                    home_prob=round(home_prob * 100, 1) if home_prob > 0 else None,
                    away_prob=round(away_prob * 100, 1) if away_prob > 0 else None,
                ))
        
        return markets
    
    @staticmethod
    def calculate_basketball_fouls(
        avg_team_fouls: float = 20,  # NBA average
        pace_factor: float = 1.0,  # Team pace adjustment
    ) -> List[FoulMarket]:
        """Calculate basketball foul markets."""
        expected = avg_team_fouls * pace_factor
        
        markets = []
        lines = [18.5, 20.5, 22.5]
        
        for line in lines:
            over_prob = ExtendedMarketsCalculator._poisson_over(expected, line)
            under_prob = 100 - over_prob
            
            markets.append(FoulMarket(
                type='team_fouls',
                line=line,
                over_odds=round(100 / over_prob * 0.9, 2),
                under_odds=round(100 / under_prob * 0.9, 2),
                over_prob=round(over_prob, 1),
                under_prob=round(under_prob, 1),
                expected_fouls=round(expected, 1)
            ))
        
        return markets
    
    @staticmethod
    def calculate_tennis_set_betting(
        home_win_prob: float,  # Probability home player wins match
        sets_format: int = 3,  # 3 for best-of-5, 2 for best-of-3
    ) -> List[SetBettingMarket]:
        """
        Calculate tennis set betting (correct score).
        
        Args:
            home_win_prob: Probability home player wins match (%)
            sets_format: 2 or 3 sets
        """
        markets = []
        p = home_win_prob / 100
        
        if sets_format == 2:  # Best of 3
            # 2-0: win first two sets
            prob_2_0 = p * p * 0.6  # 60% of wins are 2-0
            prob_0_2 = (1-p) * (1-p) * 0.6
            prob_2_1 = p * (1-p) * p + p * p * (1-p)  # Win in 3 sets
            prob_1_2 = (1-p) * p * (1-p) + (1-p) * (1-p) * p
            
            scores = [
                ("2-0", prob_2_0, prob_0_2),
                ("2-1", prob_2_1, prob_1_2),
            ]
        else:  # Best of 5 (Grand Slam)
            # Simplified model
            prob_3_0 = p ** 3 * 0.4
            prob_3_1 = p ** 3 * (1-p) * 3 * 0.35
            prob_3_2 = p ** 3 * ((1-p) ** 2) * 6 * 0.25
            prob_0_3 = (1-p) ** 3 * 0.4
            prob_1_3 = (1-p) ** 3 * p * 3 * 0.35
            prob_2_3 = (1-p) ** 3 * (p ** 2) * 6 * 0.25
            
            scores = [
                ("3-0", prob_3_0, prob_0_3),
                ("3-1", prob_3_1, prob_1_3),
                ("3-2", prob_3_2, prob_2_3),
            ]
        
        for score, home_p, away_p in scores:
            home_odds = 100 / (home_p * 100) * 0.9 if home_p > 0 else 50
            away_odds = 100 / (away_p * 100) * 0.9 if away_p > 0 else 50
            
            markets.append(SetBettingMarket(
                score=score,
                home_odds=round(home_odds, 2),
                away_odds=round(away_odds, 2),
                home_prob=round(home_p * 100, 1),
                away_prob=round(away_p * 100, 1),
            ))
        
        return markets
    
    @staticmethod
    def calculate_tennis_game_handicap(
        home_win_prob: float,
        total_games_estimate: int = 22,  # Estimated total games
    ) -> List[GameHandicapMarket]:
        """
        Calculate tennis game handicaps.
        
        Args:
            home_win_prob: Probability home player wins (%)
            total_games_estimate: Estimated total games in match
        """
        markets = []
        p = home_win_prob / 100
        
        # Estimate game difference based on match win probability
        # Higher win probability = bigger game margin
        expected_margin = (p - 0.5) * total_games_estimate * 0.4
        
        # Standard handicap lines
        lines = [-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5]
        
        for line in lines:
            # Probability of covering handicap
            required_margin = line
            
            # Simplified - probability based on normal distribution
            std_dev = math.sqrt(total_games_estimate) * 0.5
            
            home_cover_prob = ExtendedMarketsCalculator._normal_cdf(
                100, expected_margin - required_margin, std_dev
            ) * 100
            away_cover_prob = 100 - home_cover_prob
            
            home_odds = 100 / home_cover_prob * 0.9 if home_cover_prob > 0 else 10
            away_odds = 100 / away_cover_prob * 0.9 if away_cover_prob > 0 else 10
            
            recommendation = None
            if home_cover_prob > 60:
                recommendation = 'home'
            elif away_cover_prob > 60:
                recommendation = 'away'
            
            markets.append(GameHandicapMarket(
                line=line,
                home_odds=round(home_odds, 2),
                away_odds=round(away_odds, 2),
                home_prob=round(home_cover_prob, 1),
                away_prob=round(away_cover_prob, 1),
                recommendation=recommendation
            ))
        
        return markets
    
    # ==================== HELPER METHODS ====================
    
    @staticmethod
    def _poisson_over(lambda_val: float, threshold: float) -> float:
        """Calculate probability of Poisson variable exceeding threshold."""
        # P(X > threshold) = 1 - P(X <= threshold)
        prob_under = 0
        for k in range(int(threshold) + 1):
            prob_under += (lambda_val ** k) * math.exp(-lambda_val) / math.factorial(k)
        return (1 - prob_under) * 100
    
    @staticmethod
    def _normal_cdf(x: float, mean: float, std: float) -> float:
        """Cumulative distribution function for normal distribution."""
        return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


# Convenience function to get all extended markets
def calculate_all_extended_markets(
    sport: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    **kwargs
) -> ExtendedMarkets:
    """
    Calculate all extended markets for a match.
    
    Args:
        sport: 'football', 'basketball', 'tennis'
        prob_home: Home win probability
        prob_draw: Draw probability  
        prob_away: Away win probability
        **kwargs: Sport-specific parameters
    
    Returns:
        ExtendedMarkets object
    """
    sport_type = SportType(sport)
    markets = ExtendedMarkets(sport=sport_type)
    
    if sport_type == SportType.FOOTBALL:
        # Cards
        home_agg = kwargs.get('home_aggression', 0.5)
        away_agg = kwargs.get('away_aggression', 0.5)
        markets.card_markets = ExtendedMarketsCalculator.calculate_football_cards(
            home_agg, away_agg
        )
        
        # Fouls
        markets.foul_markets = ExtendedMarketsCalculator.calculate_football_fouls(
            home_agg, away_agg
        )
        
        # Goal difference
        home_xg = kwargs.get('home_xg', 1.5)
        away_xg = kwargs.get('away_xg', 1.2)
        markets.goal_diff_markets = ExtendedMarketsCalculator.calculate_goal_difference(
            prob_home, prob_draw, prob_away, home_xg, away_xg
        )
        
    elif sport_type == SportType.BASKETBALL:
        # Win margin
        spread = kwargs.get('spread', 0)
        markets.win_margin_markets = ExtendedMarketsCalculator.calculate_basketball_win_margin(
            spread
        )
        
        # Fouls
        markets.foul_markets = ExtendedMarketsCalculator.calculate_basketball_fouls()
        
    elif sport_type == SportType.TENNIS:
        # Set betting
        sets_format = kwargs.get('sets_format', 3)
        markets.set_betting_markets = ExtendedMarketsCalculator.calculate_tennis_set_betting(
            prob_home, sets_format
        )
        
        # Game handicap
        markets.game_handicap_markets = ExtendedMarketsCalculator.calculate_tennis_game_handicap(
            prob_home
        )
    
    return markets


if __name__ == "__main__":
    # Test the calculator
    print("Testing Extended Markets Calculator...")
    print("=" * 60)
    
    # Test Football
    print("\n1. Football Extended Markets")
    print("-" * 40)
    football = calculate_all_extended_markets(
        'football',
        prob_home=55,
        prob_draw=25,
        prob_away=20,
        home_aggression=0.7,
        away_aggression=0.5,
        home_xg=1.8,
        away_xg=1.0
    )
    print(f"Cards markets: {len(football.card_markets)}")
    for m in football.card_markets[:2]:
        print(f"  {m.type} {m.line}: Over {m.over_odds} ({m.over_prob}%) / Under {m.under_odds}")
    
    print(f"\nGoal diff markets: {len(football.goal_diff_markets)}")
    for m in football.goal_diff_markets[:3]:
        print(f"  {m.selection}: {m.odds} ({m.probability}%)")
    
    # Test Basketball
    print("\n2. Basketball Extended Markets")
    print("-" * 40)
    basketball = calculate_all_extended_markets(
        'basketball',
        prob_home=60,
        prob_draw=0,
        prob_away=40,
        spread=-4.5
    )
    print(f"Win margin markets: {len(basketball.win_margin_markets)}")
    for m in basketball.win_margin_markets[:3]:
        print(f"  {m.range}: Home {m.home_odds} ({m.home_prob}%) / Away {m.away_odds}")
    
    # Test Tennis
    print("\n3. Tennis Extended Markets")
    print("-" * 40)
    tennis = calculate_all_extended_markets(
        'tennis',
        prob_home=65,
        prob_draw=0,
        prob_away=35,
        sets_format=3
    )
    print(f"Set betting markets: {len(tennis.set_betting_markets)}")
    for m in tennis.set_betting_markets:
        print(f"  {m.score}: Home {m.home_odds} ({m.home_prob}%) / Away {m.away_odds}")
    
    print(f"\nGame handicap markets: {len(tennis.game_handicap_markets)}")
    for m in tennis.game_handicap_markets[:3]:
        print(f"  {m.line:+}: Home {m.home_odds} ({m.home_prob}%) / Away {m.away_odds}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
