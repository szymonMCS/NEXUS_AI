# core/value_calculator.py
"""
Value calculation and stake management for NEXUS AI.
Implements Kelly Criterion and quality-adjusted betting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LeagueType(Enum):
    """League classification for edge thresholds."""
    POPULAR = "popular"      # Top leagues: ATP/WTA, NBA, NFL, Premier League
    MEDIUM = "medium"        # Mid-tier: Challenger, G-League, Championship
    UNPOPULAR = "unpopular"  # Lower tiers: Futures, local leagues


@dataclass
class ValueBet:
    """Represents a value bet opportunity."""
    match_id: str
    match_name: str
    sport: str
    league: str
    selection: str  # "home", "away", "over", "under", etc.
    bet_type: str   # "moneyline", "spread", "total"

    # Odds and probabilities
    bookmaker_odds: float
    fair_odds: float
    implied_probability: float
    estimated_probability: float

    # Value metrics
    edge: float
    raw_edge: float
    quality_adjusted_edge: float

    # Stake recommendations
    kelly_fraction: float
    adjusted_kelly: float
    recommended_stake: float
    stake_pct: float

    # Quality and confidence
    quality_score: float
    confidence: float
    reliability_score: float

    # Metadata
    reasoning: List[str] = field(default_factory=list)
    timestamp: str = ""

    @property
    def expected_value(self) -> float:
        """Calculate expected value per unit."""
        return (self.estimated_probability * self.bookmaker_odds) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "league": self.league,
            "selection": self.selection,
            "bet_type": self.bet_type,
            "bookmaker_odds": self.bookmaker_odds,
            "fair_odds": self.fair_odds,
            "implied_probability": self.implied_probability,
            "estimated_probability": self.estimated_probability,
            "edge": self.edge,
            "quality_adjusted_edge": self.quality_adjusted_edge,
            "kelly_fraction": self.kelly_fraction,
            "recommended_stake": self.recommended_stake,
            "stake_pct": self.stake_pct,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp
        }


class ValueCalculator:
    """
    Calculates value and optimal stakes for betting opportunities.

    Features:
    - Edge calculation with quality adjustment
    - Kelly Criterion stake sizing
    - Minimum edge thresholds by league type
    - Conservative stake management
    """

    # Minimum edge thresholds by league type
    MIN_EDGE = {
        LeagueType.POPULAR: 0.03,      # 3% minimum edge
        LeagueType.MEDIUM: 0.05,       # 5% for medium leagues
        LeagueType.UNPOPULAR: 0.07,    # 7% for unpopular leagues
    }

    # Quality multipliers
    QUALITY_MULTIPLIERS = {
        (0.85, 1.00): 1.0,   # Excellent quality
        (0.70, 0.85): 0.9,   # Good quality
        (0.50, 0.70): 0.7,   # Moderate quality
        (0.40, 0.50): 0.5,   # Low quality
        (0.00, 0.40): 0.3,   # Very low quality
    }

    # Kelly fraction multiplier (conservative betting)
    KELLY_FRACTION = 0.25  # Quarter Kelly

    # Maximum stake as percentage of bankroll
    MAX_STAKE_PCT = 0.05  # 5% max

    def __init__(self, bankroll: float = 1000.0):
        """
        Initialize value calculator.

        Args:
            bankroll: Total bankroll for stake calculations
        """
        self.bankroll = bankroll

    def calculate_edge(
        self,
        estimated_probability: float,
        bookmaker_odds: float
    ) -> float:
        """
        Calculate raw edge (expected value).

        Edge = (probability × odds) - 1

        Args:
            estimated_probability: Our estimated win probability
            bookmaker_odds: Decimal odds offered

        Returns:
            Edge as decimal (0.05 = 5% edge)
        """
        return (estimated_probability * bookmaker_odds) - 1

    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calculate implied probability from decimal odds.

        Args:
            odds: Decimal odds (e.g., 2.0)

        Returns:
            Implied probability (0-1)
        """
        if odds <= 1:
            return 1.0
        return 1 / odds

    def calculate_fair_odds(self, probability: float) -> float:
        """
        Calculate fair odds from probability.

        Args:
            probability: Estimated probability (0-1)

        Returns:
            Fair decimal odds
        """
        if probability <= 0:
            return 99.99
        if probability >= 1:
            return 1.01
        return 1 / probability

    def get_quality_multiplier(self, quality_score: float) -> float:
        """
        Get stake multiplier based on data quality.

        Args:
            quality_score: Quality score (0-1)

        Returns:
            Multiplier for stake adjustment
        """
        for (low, high), multiplier in self.QUALITY_MULTIPLIERS.items():
            if low <= quality_score < high:
                return multiplier
        return 0.3  # Default to minimum

    def adjust_probability_for_quality(
        self,
        probability: float,
        quality_score: float
    ) -> float:
        """
        Adjust probability towards 0.5 for low quality data.

        Args:
            probability: Original probability estimate
            quality_score: Data quality score (0-1)

        Returns:
            Adjusted probability (more conservative)
        """
        if quality_score >= 0.70:
            return probability

        # Move towards 0.5 for lower quality
        adjustment_strength = (0.70 - quality_score) / 0.70
        adjustment = (0.5 - probability) * adjustment_strength * 0.5

        return probability + adjustment

    def calculate_kelly_stake(
        self,
        probability: float,
        odds: float,
        fraction: float = None
    ) -> float:
        """
        Calculate Kelly Criterion stake.

        Kelly % = (bp - q) / b
        where:
        - b = decimal odds - 1
        - p = probability of winning
        - q = probability of losing (1-p)

        Args:
            probability: Estimated win probability
            odds: Decimal odds
            fraction: Kelly fraction (default: KELLY_FRACTION)

        Returns:
            Recommended stake as fraction of bankroll
        """
        if fraction is None:
            fraction = self.KELLY_FRACTION

        b = odds - 1
        p = probability
        q = 1 - p

        if b <= 0:
            return 0

        kelly = (b * p - q) / b

        # Apply fraction and cap
        kelly = max(0, kelly * fraction)
        kelly = min(kelly, self.MAX_STAKE_PCT)

        return kelly

    def find_value(
        self,
        match_id: str,
        match_name: str,
        sport: str,
        league: str,
        selection: str,
        bet_type: str,
        estimated_probability: float,
        bookmaker_odds: float,
        quality_score: float,
        confidence: float,
        reliability_score: float = 0.5,
        league_type: LeagueType = LeagueType.POPULAR,
        reasoning: List[str] = None
    ) -> Optional[ValueBet]:
        """
        Find value bet opportunity.

        Args:
            match_id: Unique match identifier
            match_name: Human-readable match name
            sport: Sport type
            league: League name
            selection: Bet selection
            bet_type: Type of bet
            estimated_probability: Our probability estimate
            bookmaker_odds: Offered odds
            quality_score: Data quality score
            confidence: Model confidence
            reliability_score: Data reliability score
            league_type: Classification of league
            reasoning: Explanation of prediction

        Returns:
            ValueBet if value found, None otherwise
        """
        # Calculate raw edge
        raw_edge = self.calculate_edge(estimated_probability, bookmaker_odds)

        # Adjust probability for quality
        adjusted_prob = self.adjust_probability_for_quality(
            estimated_probability, quality_score
        )

        # Quality-adjusted edge
        quality_multiplier = self.get_quality_multiplier(quality_score)
        quality_adjusted_edge = raw_edge * quality_multiplier

        # Check minimum edge threshold
        min_edge = self.MIN_EDGE.get(league_type, self.MIN_EDGE[LeagueType.POPULAR])

        if quality_adjusted_edge < min_edge:
            logger.debug(
                f"No value in {match_name}: edge {quality_adjusted_edge:.2%} < min {min_edge:.2%}"
            )
            return None

        # Calculate stakes
        kelly_fraction = self.calculate_kelly_stake(adjusted_prob, bookmaker_odds)
        adjusted_kelly = kelly_fraction * quality_multiplier

        # Cap at max stake
        stake_pct = min(adjusted_kelly, self.MAX_STAKE_PCT)
        recommended_stake = self.bankroll * stake_pct

        # Build value bet
        value_bet = ValueBet(
            match_id=match_id,
            match_name=match_name,
            sport=sport,
            league=league,
            selection=selection,
            bet_type=bet_type,
            bookmaker_odds=bookmaker_odds,
            fair_odds=self.calculate_fair_odds(estimated_probability),
            implied_probability=self.calculate_implied_probability(bookmaker_odds),
            estimated_probability=estimated_probability,
            edge=quality_adjusted_edge,
            raw_edge=raw_edge,
            quality_adjusted_edge=quality_adjusted_edge,
            kelly_fraction=kelly_fraction,
            adjusted_kelly=adjusted_kelly,
            recommended_stake=recommended_stake,
            stake_pct=stake_pct,
            quality_score=quality_score,
            confidence=confidence,
            reliability_score=reliability_score,
            reasoning=reasoning or []
        )

        logger.info(
            f"Value found: {match_name} - {selection} @ {bookmaker_odds:.2f} "
            f"(edge: {quality_adjusted_edge:.1%}, stake: ${recommended_stake:.2f})"
        )

        return value_bet

    def rank_value_bets(
        self,
        value_bets: List[ValueBet],
        max_bets: int = 5
    ) -> List[ValueBet]:
        """
        Rank and select top value bets.

        Uses composite score: edge^0.5 × quality^0.3 × confidence^0.2

        Args:
            value_bets: List of value bets
            max_bets: Maximum bets to return

        Returns:
            Top ranked value bets
        """
        if not value_bets:
            return []

        # Calculate composite scores
        scored_bets = []
        for bet in value_bets:
            # Normalize edge (cap at 20%)
            normalized_edge = min(bet.edge, 0.20) / 0.20

            # Composite score
            score = (
                (normalized_edge ** 0.5) *
                (bet.quality_score ** 0.3) *
                (bet.confidence ** 0.2)
            )

            scored_bets.append((score, bet))

        # Sort by score descending
        scored_bets.sort(key=lambda x: x[0], reverse=True)

        # Apply diversification (max 1 bet per tournament/league)
        selected = []
        seen_leagues = set()

        for score, bet in scored_bets:
            if len(selected) >= max_bets:
                break

            # Allow max 1 from same league (diversification)
            if bet.league in seen_leagues and len(selected) >= 3:
                continue

            selected.append(bet)
            seen_leagues.add(bet.league)

        # Add ranking info to reasoning
        for i, bet in enumerate(selected, 1):
            rank_label = {1: "GOLD", 2: "SILVER", 3: "BRONZE"}.get(i, f"#{i}")
            bet.reasoning.insert(0, f"Ranked {rank_label} by composite score")

        return selected

    def calculate_portfolio_risk(
        self,
        value_bets: List[ValueBet]
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.

        Args:
            value_bets: Selected value bets

        Returns:
            Risk analysis dictionary
        """
        if not value_bets:
            return {
                "total_stake": 0,
                "total_stake_pct": 0,
                "expected_profit": 0,
                "max_loss": 0,
                "correlation_risk": "low",
                "diversification_score": 0
            }

        total_stake = sum(bet.recommended_stake for bet in value_bets)
        total_stake_pct = total_stake / self.bankroll

        # Expected profit
        expected_profit = sum(
            bet.recommended_stake * bet.expected_value
            for bet in value_bets
        )

        # Max possible loss
        max_loss = total_stake

        # Diversification score (more sports/leagues = better)
        unique_sports = len(set(bet.sport for bet in value_bets))
        unique_leagues = len(set(bet.league for bet in value_bets))
        diversification = (unique_sports + unique_leagues) / (2 * len(value_bets))

        # Correlation risk
        if len(set(bet.sport for bet in value_bets)) == 1:
            correlation_risk = "high"
        elif unique_leagues < len(value_bets) / 2:
            correlation_risk = "medium"
        else:
            correlation_risk = "low"

        return {
            "total_stake": total_stake,
            "total_stake_pct": total_stake_pct,
            "expected_profit": expected_profit,
            "max_loss": max_loss,
            "correlation_risk": correlation_risk,
            "diversification_score": diversification,
            "bet_count": len(value_bets),
            "avg_edge": np.mean([bet.edge for bet in value_bets]),
            "avg_quality": np.mean([bet.quality_score for bet in value_bets])
        }

    def set_bankroll(self, bankroll: float):
        """Update bankroll amount."""
        self.bankroll = bankroll
        logger.info(f"Bankroll updated to ${bankroll:.2f}")
