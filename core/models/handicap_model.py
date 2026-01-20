# core/models/handicap_model.py
"""
Handicap Prediction Model for NEXUS AI.

Predicts:
- Game/Set handicaps (tennis)
- Point spreads (basketball)
- First half / Second half performance
- Over/Under totals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class MarketType(str, Enum):
    """Supported handicap market types."""
    MATCH_HANDICAP = "match_handicap"  # Full match handicap
    FIRST_HALF = "first_half"  # First half result
    SECOND_HALF = "second_half"  # Second half result
    TOTAL_OVER = "total_over"  # Over X goals/points
    TOTAL_UNDER = "total_under"  # Under X goals/points
    FIRST_HALF_TOTAL = "first_half_total"  # First half over/under
    EXACT_SCORE = "exact_score"  # Exact final score


@dataclass
class HalfStats:
    """Statistics for a half/period."""
    avg_scored: float = 0.0
    avg_conceded: float = 0.0
    clean_sheets_pct: float = 0.0  # % of games with 0 conceded
    scoring_pct: float = 0.0  # % of games with at least 1 scored
    avg_margin: float = 0.0


@dataclass
class HandicapPrediction:
    """Result of handicap prediction."""
    market_type: MarketType
    line: float  # Handicap line (e.g., -1.5, +2.5)
    cover_probability: float  # Probability of covering the spread
    push_probability: float = 0.0  # Probability of exact match (for whole numbers)
    confidence: float = 0.5
    expected_margin: float = 0.0
    reasoning: List[str] = field(default_factory=list)

    @property
    def fair_odds(self) -> float:
        """Calculate fair odds based on probability."""
        if self.cover_probability <= 0:
            return 99.99
        return round(1 / self.cover_probability, 2)


@dataclass
class TotalPrediction:
    """Prediction for over/under markets."""
    line: float  # Total line (e.g., 2.5 goals, 210.5 points)
    over_probability: float
    under_probability: float
    expected_total: float
    confidence: float = 0.5
    reasoning: List[str] = field(default_factory=list)


class HandicapModel:
    """
    Base handicap prediction model.

    Uses statistical analysis of:
    - First half vs second half performance
    - Home/away scoring patterns
    - Recent form (goals/points scored and conceded)
    - Head-to-head margins
    """

    def __init__(self):
        # Default weights for factors
        self.weights = {
            "recent_form": 0.35,
            "h2h_margin": 0.20,
            "home_away": 0.15,
            "half_pattern": 0.20,
            "fatigue": 0.10
        }

    def predict_handicap(
        self,
        home_stats: Dict,
        away_stats: Dict,
        line: float,
        market_type: MarketType = MarketType.MATCH_HANDICAP
    ) -> HandicapPrediction:
        """
        Predict probability of covering a handicap.

        Args:
            home_stats: Statistics for home team/player
            away_stats: Statistics for away team/player
            line: Handicap line (negative = home gives, positive = home receives)
            market_type: Type of handicap market

        Returns:
            HandicapPrediction with probabilities
        """
        reasoning = []

        # Calculate expected margin
        expected_margin = self._calculate_expected_margin(home_stats, away_stats)
        reasoning.append(f"Expected margin: {expected_margin:+.1f}")

        # Adjust for market type (first half, second half)
        if market_type == MarketType.FIRST_HALF:
            expected_margin = self._adjust_for_first_half(
                expected_margin, home_stats, away_stats
            )
            reasoning.append(f"First half adjusted margin: {expected_margin:+.1f}")
        elif market_type == MarketType.SECOND_HALF:
            expected_margin = self._adjust_for_second_half(
                expected_margin, home_stats, away_stats
            )
            reasoning.append(f"Second half adjusted margin: {expected_margin:+.1f}")

        # Calculate cover probability using normal distribution
        std_dev = self._calculate_std_dev(home_stats, away_stats)
        cover_prob = self._normal_cdf(expected_margin - line, std_dev)

        # Push probability for whole number lines
        push_prob = 0.0
        if line == int(line):
            push_prob = self._push_probability(expected_margin, line, std_dev)
            cover_prob -= push_prob / 2  # Adjust cover probability

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(home_stats, away_stats)

        return HandicapPrediction(
            market_type=market_type,
            line=line,
            cover_probability=round(max(0.01, min(0.99, cover_prob)), 3),
            push_probability=round(push_prob, 3),
            confidence=round(confidence, 2),
            expected_margin=round(expected_margin, 2),
            reasoning=reasoning
        )

    def predict_total(
        self,
        home_stats: Dict,
        away_stats: Dict,
        line: float,
        market_type: MarketType = MarketType.TOTAL_OVER
    ) -> TotalPrediction:
        """
        Predict over/under probabilities.

        Args:
            home_stats: Statistics for home team/player
            away_stats: Statistics for away team/player
            line: Total line (e.g., 2.5 goals)
            market_type: TOTAL_OVER or FIRST_HALF_TOTAL

        Returns:
            TotalPrediction with probabilities
        """
        reasoning = []

        # Calculate expected total
        expected_total = self._calculate_expected_total(home_stats, away_stats)
        reasoning.append(f"Expected total: {expected_total:.1f}")

        # Adjust for first half if applicable
        if market_type == MarketType.FIRST_HALF_TOTAL:
            first_half_ratio = self._get_first_half_ratio(home_stats, away_stats)
            expected_total *= first_half_ratio
            reasoning.append(f"First half expected: {expected_total:.1f}")

        # Calculate probabilities
        std_dev = self._calculate_total_std_dev(home_stats, away_stats)
        over_prob = self._normal_cdf(expected_total - line, std_dev)
        under_prob = 1 - over_prob

        confidence = self._calculate_confidence(home_stats, away_stats)

        return TotalPrediction(
            line=line,
            over_probability=round(max(0.01, min(0.99, over_prob)), 3),
            under_probability=round(max(0.01, min(0.99, under_prob)), 3),
            expected_total=round(expected_total, 2),
            confidence=round(confidence, 2),
            reasoning=reasoning
        )

    def analyze_half_patterns(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> Dict[str, HalfStats]:
        """
        Analyze first half vs second half scoring patterns.

        Returns:
            Dict with 'first_half' and 'second_half' HalfStats
        """
        result = {}

        # First half analysis
        home_1h = home_stats.get("first_half", {})
        away_1h = away_stats.get("first_half", {})

        result["first_half"] = HalfStats(
            avg_scored=(home_1h.get("avg_scored", 0) + away_1h.get("avg_conceded", 0)) / 2,
            avg_conceded=(home_1h.get("avg_conceded", 0) + away_1h.get("avg_scored", 0)) / 2,
            clean_sheets_pct=home_1h.get("clean_sheets_pct", 0),
            scoring_pct=home_1h.get("scoring_pct", 0),
            avg_margin=home_1h.get("avg_margin", 0) - away_1h.get("avg_margin", 0)
        )

        # Second half analysis
        home_2h = home_stats.get("second_half", {})
        away_2h = away_stats.get("second_half", {})

        result["second_half"] = HalfStats(
            avg_scored=(home_2h.get("avg_scored", 0) + away_2h.get("avg_conceded", 0)) / 2,
            avg_conceded=(home_2h.get("avg_conceded", 0) + away_2h.get("avg_scored", 0)) / 2,
            clean_sheets_pct=home_2h.get("clean_sheets_pct", 0),
            scoring_pct=home_2h.get("scoring_pct", 0),
            avg_margin=home_2h.get("avg_margin", 0) - away_2h.get("avg_margin", 0)
        )

        return result

    def _calculate_expected_margin(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Calculate expected margin (home - away)."""
        factors = []

        # Recent form: avg scored - avg conceded
        home_form = home_stats.get("avg_scored", 0) - home_stats.get("avg_conceded", 0)
        away_form = away_stats.get("avg_scored", 0) - away_stats.get("avg_conceded", 0)
        form_diff = home_form - away_form
        factors.append(("recent_form", form_diff))

        # H2H margin
        h2h = home_stats.get("h2h", {})
        h2h_margin = h2h.get("avg_margin", 0)  # Positive = home favored historically
        factors.append(("h2h_margin", h2h_margin))

        # Home advantage
        home_advantage = home_stats.get("home_advantage", 0.5)  # Extra points/goals at home
        factors.append(("home_away", home_advantage))

        # Weighted sum
        margin = sum(
            value * self.weights.get(name, 0)
            for name, value in factors
        )

        return margin

    def _calculate_expected_total(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Calculate expected total (home + away)."""
        home_attack = home_stats.get("avg_scored", 0)
        home_defense = home_stats.get("avg_conceded", 0)
        away_attack = away_stats.get("avg_scored", 0)
        away_defense = away_stats.get("avg_conceded", 0)

        # Expected home score = avg of (home attack, away defense)
        expected_home = (home_attack + away_defense) / 2
        # Expected away score = avg of (away attack, home defense)
        expected_away = (away_attack + home_defense) / 2

        return expected_home + expected_away

    def _adjust_for_first_half(
        self,
        full_match_margin: float,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """
        Adjust margin for first half prediction.

        Some teams/players are stronger in first half.
        """
        home_1h = home_stats.get("first_half", {})
        away_1h = away_stats.get("first_half", {})

        # First half ratio (how much of margin comes in first half)
        home_ratio = home_1h.get("margin_ratio", 0.5)  # Default: even split
        away_ratio = away_1h.get("margin_ratio", 0.5)

        avg_ratio = (home_ratio + (1 - away_ratio)) / 2

        return full_match_margin * avg_ratio

    def _adjust_for_second_half(
        self,
        full_match_margin: float,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Adjust margin for second half prediction."""
        first_half_margin = self._adjust_for_first_half(
            full_match_margin, home_stats, away_stats
        )
        return full_match_margin - first_half_margin

    def _get_first_half_ratio(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Get ratio of scoring that happens in first half."""
        home_1h = home_stats.get("first_half", {})
        away_1h = away_stats.get("first_half", {})

        home_ratio = home_1h.get("scoring_ratio", 0.45)  # Default: slightly less in 1H
        away_ratio = away_1h.get("scoring_ratio", 0.45)

        return (home_ratio + away_ratio) / 2

    def _calculate_std_dev(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Calculate standard deviation for margin prediction."""
        # Use historical variance if available
        home_var = home_stats.get("margin_variance", 4.0)
        away_var = away_stats.get("margin_variance", 4.0)

        return math.sqrt(home_var + away_var)

    def _calculate_total_std_dev(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Calculate standard deviation for total prediction."""
        home_var = home_stats.get("scoring_variance", 2.0)
        away_var = away_stats.get("scoring_variance", 2.0)

        return math.sqrt(home_var + away_var)

    def _normal_cdf(self, x: float, std_dev: float) -> float:
        """
        Cumulative distribution function for normal distribution.
        Returns probability that value is less than x.
        """
        if std_dev <= 0:
            return 0.5

        z = x / std_dev
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _push_probability(
        self,
        expected: float,
        line: float,
        std_dev: float
    ) -> float:
        """Calculate probability of exact push (for whole number lines)."""
        # Approximate: probability density at the line
        if std_dev <= 0:
            return 0

        z = (expected - line) / std_dev
        density = math.exp(-0.5 * z * z) / (std_dev * math.sqrt(2 * math.pi))

        # Approximate probability of landing exactly on the line
        return min(0.15, density * 0.5)

    def _calculate_confidence(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> float:
        """Calculate confidence based on data availability."""
        confidence = 0.3  # Base

        # Bonus for sample size
        home_games = home_stats.get("games_played", 0)
        away_games = away_stats.get("games_played", 0)

        if home_games >= 10 and away_games >= 10:
            confidence += 0.2
        elif home_games >= 5 and away_games >= 5:
            confidence += 0.1

        # Bonus for first half data
        if home_stats.get("first_half") and away_stats.get("first_half"):
            confidence += 0.15

        # Bonus for H2H data
        if home_stats.get("h2h", {}).get("games", 0) >= 3:
            confidence += 0.15

        # Bonus for recent data (last 30 days)
        if home_stats.get("recent_games", 0) >= 3:
            confidence += 0.1

        return min(0.95, confidence)


class TennisHandicapModel(HandicapModel):
    """
    Tennis-specific handicap model.

    Predicts:
    - Game handicaps (e.g., Player A -4.5 games)
    - Set handicaps (e.g., Player A -1.5 sets)
    - First set winner
    - Tiebreak probability
    """

    def __init__(self):
        super().__init__()
        self.weights = {
            "recent_form": 0.30,
            "h2h_margin": 0.25,
            "surface": 0.20,
            "ranking": 0.15,
            "fatigue": 0.10
        }

    def predict_games_handicap(
        self,
        player1_stats: Dict,
        player2_stats: Dict,
        line: float,
        surface: str = "hard"
    ) -> HandicapPrediction:
        """
        Predict game handicap for tennis match.

        Args:
            player1_stats: Stats for player 1 (server)
            player2_stats: Stats for player 2
            line: Game handicap (e.g., -4.5)
            surface: Court surface (hard, clay, grass)
        """
        # Adjust for surface
        p1_surface = player1_stats.get(f"{surface}_stats", player1_stats)
        p2_surface = player2_stats.get(f"{surface}_stats", player2_stats)

        # Service game stats
        p1_hold = p1_surface.get("service_hold_pct", 0.80)
        p2_hold = p2_surface.get("service_hold_pct", 0.80)
        p1_break = p1_surface.get("return_break_pct", 0.20)
        p2_break = p2_surface.get("return_break_pct", 0.20)

        # Expected games won per set
        p1_games_per_set = 6 * p1_hold + 6 * p1_break
        p2_games_per_set = 6 * p2_hold + 6 * p2_break

        # Estimate total sets (based on expected match length)
        expected_sets = player1_stats.get("avg_sets", 2.3)

        # Expected total games margin
        expected_margin = (p1_games_per_set - p2_games_per_set) * expected_sets

        # Standard deviation for tennis (typically higher variance)
        std_dev = 4.0 * math.sqrt(expected_sets)

        cover_prob = self._normal_cdf(expected_margin - line, std_dev)

        return HandicapPrediction(
            market_type=MarketType.MATCH_HANDICAP,
            line=line,
            cover_probability=round(max(0.01, min(0.99, cover_prob)), 3),
            expected_margin=round(expected_margin, 1),
            confidence=self._calculate_confidence(player1_stats, player2_stats),
            reasoning=[
                f"P1 service hold: {p1_hold:.0%}",
                f"P2 service hold: {p2_hold:.0%}",
                f"Expected margin: {expected_margin:+.1f} games"
            ]
        )

    def predict_first_set(
        self,
        player1_stats: Dict,
        player2_stats: Dict
    ) -> HandicapPrediction:
        """Predict first set winner."""
        # First set performance data
        p1_first = player1_stats.get("first_set", {})
        p2_first = player2_stats.get("first_set", {})

        p1_win_rate = p1_first.get("win_pct", 0.5)
        p2_win_rate = p2_first.get("win_pct", 0.5)

        # Combine with overall strength
        p1_overall = player1_stats.get("win_pct", 0.5)
        p2_overall = player2_stats.get("win_pct", 0.5)

        # Weighted first set probability
        p1_prob = 0.6 * p1_first.get("win_pct", p1_overall) + 0.4 * p1_overall
        p2_prob = 0.6 * p2_first.get("win_pct", p2_overall) + 0.4 * p2_overall

        # Normalize
        total = p1_prob + p2_prob
        p1_prob = p1_prob / total if total > 0 else 0.5

        return HandicapPrediction(
            market_type=MarketType.FIRST_HALF,  # First set = "first half" in tennis
            line=0,
            cover_probability=round(p1_prob, 3),
            confidence=0.7 if p1_first and p2_first else 0.5,
            reasoning=[
                f"P1 first set win rate: {p1_win_rate:.0%}",
                f"P2 first set win rate: {p2_win_rate:.0%}"
            ]
        )


class BasketballHandicapModel(HandicapModel):
    """
    Basketball-specific handicap model.

    Predicts:
    - Point spreads (e.g., Lakers -5.5)
    - First half spreads
    - Quarter handicaps
    - Total points over/under
    """

    def __init__(self):
        super().__init__()
        self.weights = {
            "offensive_rating": 0.25,
            "defensive_rating": 0.25,
            "recent_form": 0.20,
            "rest_days": 0.15,
            "home_court": 0.15
        }

        # NBA specific adjustments
        self.home_court_advantage = 3.0  # Points
        self.first_half_ratio = 0.48  # Slightly less scoring in first half

    def predict_spread(
        self,
        home_stats: Dict,
        away_stats: Dict,
        line: float
    ) -> HandicapPrediction:
        """
        Predict point spread for basketball game.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            line: Point spread (negative = home favorite)
        """
        reasoning = []

        # Offensive and defensive ratings
        home_ortg = home_stats.get("offensive_rating", 110)
        home_drtg = home_stats.get("defensive_rating", 110)
        away_ortg = away_stats.get("offensive_rating", 110)
        away_drtg = away_stats.get("defensive_rating", 110)

        # Expected points
        home_expected = (home_ortg + away_drtg) / 2
        away_expected = (away_ortg + home_drtg) / 2

        # Home court advantage
        home_expected += self.home_court_advantage / 2
        away_expected -= self.home_court_advantage / 2

        # Rest days adjustment
        home_rest = home_stats.get("rest_days", 1)
        away_rest = away_stats.get("rest_days", 1)
        rest_diff = min(3, home_rest) - min(3, away_rest)
        home_expected += rest_diff * 0.5

        expected_margin = home_expected - away_expected
        reasoning.append(f"Expected margin: {expected_margin:+.1f} pts")

        # Standard deviation for NBA games (typically 10-12 points)
        std_dev = 11.0

        cover_prob = self._normal_cdf(expected_margin - line, std_dev)

        return HandicapPrediction(
            market_type=MarketType.MATCH_HANDICAP,
            line=line,
            cover_probability=round(max(0.01, min(0.99, cover_prob)), 3),
            expected_margin=round(expected_margin, 1),
            confidence=self._calculate_confidence(home_stats, away_stats),
            reasoning=reasoning + [
                f"Home ORTG: {home_ortg:.1f}, DRTG: {home_drtg:.1f}",
                f"Away ORTG: {away_ortg:.1f}, DRTG: {away_drtg:.1f}"
            ]
        )

    def predict_first_half_spread(
        self,
        home_stats: Dict,
        away_stats: Dict,
        line: float
    ) -> HandicapPrediction:
        """Predict first half point spread."""
        # Get first half specific stats
        home_1h = home_stats.get("first_half", {})
        away_1h = away_stats.get("first_half", {})

        # First half scoring patterns
        home_1h_scored = home_1h.get("avg_scored", home_stats.get("avg_scored", 55) * 0.48)
        home_1h_allowed = home_1h.get("avg_allowed", home_stats.get("avg_allowed", 55) * 0.48)
        away_1h_scored = away_1h.get("avg_scored", away_stats.get("avg_scored", 55) * 0.48)
        away_1h_allowed = away_1h.get("avg_allowed", away_stats.get("avg_allowed", 55) * 0.48)

        # Expected first half margin
        home_expected = (home_1h_scored + away_1h_allowed) / 2
        away_expected = (away_1h_scored + home_1h_allowed) / 2

        # Reduced home court in first half (teams still adjusting)
        home_expected += self.home_court_advantage * 0.4 / 2
        away_expected -= self.home_court_advantage * 0.4 / 2

        expected_margin = home_expected - away_expected

        # Lower variance in first half
        std_dev = 8.0

        cover_prob = self._normal_cdf(expected_margin - line, std_dev)

        return HandicapPrediction(
            market_type=MarketType.FIRST_HALF,
            line=line,
            cover_probability=round(max(0.01, min(0.99, cover_prob)), 3),
            expected_margin=round(expected_margin, 1),
            confidence=0.7 if home_1h and away_1h else 0.5,
            reasoning=[
                f"Home 1H avg: {home_1h_scored:.1f} scored, {home_1h_allowed:.1f} allowed",
                f"Away 1H avg: {away_1h_scored:.1f} scored, {away_1h_allowed:.1f} allowed",
                f"Expected 1H margin: {expected_margin:+.1f}"
            ]
        )

    def predict_total_points(
        self,
        home_stats: Dict,
        away_stats: Dict,
        line: float,
        first_half: bool = False
    ) -> TotalPrediction:
        """
        Predict over/under for total points.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            line: Total points line
            first_half: If True, predict first half total
        """
        # Pace factor (possessions per game)
        home_pace = home_stats.get("pace", 100)
        away_pace = away_stats.get("pace", 100)
        expected_pace = (home_pace + away_pace) / 2

        # Offensive ratings
        home_ortg = home_stats.get("offensive_rating", 110)
        away_ortg = away_stats.get("offensive_rating", 110)
        home_drtg = home_stats.get("defensive_rating", 110)
        away_drtg = away_stats.get("defensive_rating", 110)

        # Expected points per 100 possessions
        home_pts_per_100 = (home_ortg + away_drtg) / 2
        away_pts_per_100 = (away_ortg + home_drtg) / 2

        # Scale to actual pace
        home_expected = home_pts_per_100 * (expected_pace / 100)
        away_expected = away_pts_per_100 * (expected_pace / 100)

        expected_total = home_expected + away_expected

        if first_half:
            expected_total *= self.first_half_ratio
            std_dev = 12.0
        else:
            std_dev = 18.0

        over_prob = self._normal_cdf(expected_total - line, std_dev)
        under_prob = 1 - over_prob

        return TotalPrediction(
            line=line,
            over_probability=round(max(0.01, min(0.99, over_prob)), 3),
            under_probability=round(max(0.01, min(0.99, under_prob)), 3),
            expected_total=round(expected_total, 1),
            confidence=self._calculate_confidence(home_stats, away_stats),
            reasoning=[
                f"Expected total: {expected_total:.1f}",
                f"Combined pace: {expected_pace:.1f}",
                f"Line: {line}"
            ]
        )


# === HELPER FUNCTIONS ===

def find_value_handicap(
    model: HandicapModel,
    home_stats: Dict,
    away_stats: Dict,
    bookmaker_odds: Dict[float, Tuple[float, float]],
    market_type: MarketType = MarketType.MATCH_HANDICAP
) -> List[Dict]:
    """
    Find value bets in handicap markets.

    Args:
        model: HandicapModel instance
        home_stats: Home team/player stats
        away_stats: Away team/player stats
        bookmaker_odds: Dict mapping line -> (home_odds, away_odds)
        market_type: Type of handicap market

    Returns:
        List of value bets with edge > 0
    """
    value_bets = []

    for line, (home_odds, away_odds) in bookmaker_odds.items():
        prediction = model.predict_handicap(
            home_stats, away_stats, line, market_type
        )

        # Check home side value
        implied_home = 1 / home_odds
        edge_home = prediction.cover_probability - implied_home

        if edge_home > 0.02:  # Minimum 2% edge
            value_bets.append({
                "line": line,
                "side": "home",
                "odds": home_odds,
                "fair_odds": prediction.fair_odds,
                "probability": prediction.cover_probability,
                "edge": round(edge_home, 4),
                "confidence": prediction.confidence,
                "reasoning": prediction.reasoning
            })

        # Check away side value (inverse line)
        implied_away = 1 / away_odds
        edge_away = (1 - prediction.cover_probability) - implied_away

        if edge_away > 0.02:
            value_bets.append({
                "line": -line,
                "side": "away",
                "odds": away_odds,
                "fair_odds": round(1 / (1 - prediction.cover_probability), 2),
                "probability": 1 - prediction.cover_probability,
                "edge": round(edge_away, 4),
                "confidence": prediction.confidence,
                "reasoning": prediction.reasoning
            })

    return sorted(value_bets, key=lambda x: x["edge"], reverse=True)
