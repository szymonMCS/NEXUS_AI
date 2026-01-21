# agents/ranker.py
"""
Ranker Agent - Ranks matches by betting value potential.
Selects top opportunities based on composite score.

Implements:
- RankedMatch dataclass with detailed metrics
- Composite score formula: edge^0.5 * quality^0.3 * confidence^0.2
- Tournament constraint: max 1 bet per tournament
- Detailed reasoning generation
- format_top_3_report() for text output
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from langchain_anthropic import ChatAnthropic

from config.settings import settings
from config.thresholds import thresholds
from core.state import (
    NexusState, Match, BetDecision,
    DataQualityLevel, add_message
)


class RankingTier(Enum):
    """Ranking tier for top matches."""
    GOLD = 1
    SILVER = 2
    BRONZE = 3
    TOP_5 = 4
    RANKED = 5


@dataclass
class RankedMatch:
    """
    Represents a ranked match with detailed metrics.

    Contains all information needed for betting decision
    and report generation.
    """
    # Core match info
    match: Match
    rank: int
    tier: RankingTier

    # Composite score components
    composite_score: float
    edge_component: float      # edge^0.5 contribution
    quality_component: float   # quality^0.3 contribution
    confidence_component: float  # confidence^0.2 contribution

    # Raw values
    edge: float               # Raw edge value (0-1)
    quality_score: float      # Data quality (0-1)
    confidence: float         # Prediction confidence (0-1)

    # Selection details
    selection: str            # "home" or "away"
    selection_name: str       # Player/team name
    probability: float        # Win probability
    odds: float              # Best available odds
    bookmaker: str           # Bookmaker with best odds

    # Stake recommendation
    stake_recommendation: str  # e.g., "1.5-2% bankroll"
    kelly_stake: float        # Kelly criterion stake %

    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    key_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    tournament: str = ""
    league: str = ""
    match_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/JSON."""
        return {
            "rank": self.rank,
            "tier": self.tier.name,
            "match": f"{self.match.home_player.name} vs {self.match.away_player.name}",
            "tournament": self.tournament,
            "league": self.league,
            "composite_score": round(self.composite_score, 4),
            "selection": self.selection,
            "selection_name": self.selection_name,
            "probability": round(self.probability, 3),
            "odds": round(self.odds, 2),
            "bookmaker": self.bookmaker,
            "edge": round(self.edge, 4),
            "quality_score": round(self.quality_score, 3),
            "confidence": round(self.confidence, 3),
            "stake_recommendation": self.stake_recommendation,
            "kelly_stake": round(self.kelly_stake, 4),
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "warnings": self.warnings,
            "match_time": self.match_time
        }


class RankerAgent:
    """
    Ranker scores and ranks matches by betting value potential.

    Uses composite score formula from plan:
    composite_score = (normalized_edge^0.5) * (quality^0.3) * (confidence^0.2)

    Features:
    - Weighted geometric mean ranking
    - Tournament diversification (max 1 bet per tournament)
    - Detailed reasoning generation
    - Top 3 report formatting
    """

    # Composite score exponents (from plan)
    EDGE_EXPONENT = 0.5
    QUALITY_EXPONENT = 0.3
    CONFIDENCE_EXPONENT = 0.2

    # Edge normalization cap
    MAX_EDGE_CAP = 0.20  # 20% max edge for normalization

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )

    async def process(self, state: NexusState) -> NexusState:
        """
        Rank matches and select top opportunities.

        Args:
            state: Current workflow state

        Returns:
            Updated state with ranked top matches
        """
        state.current_agent = "ranker"

        # Filter matches with predictions
        predicted_matches = [
            m for m in state.matches
            if m.prediction is not None and m.recommended != BetDecision.REJECTED
        ]

        state = add_message(
            state,
            "ranker",
            f"Ranking {len(predicted_matches)} predicted matches"
        )

        # Create RankedMatch objects with scores
        ranked_matches: List[RankedMatch] = []
        for match in predicted_matches:
            ranked_match = self._create_ranked_match(match)
            if ranked_match:
                ranked_matches.append(ranked_match)

        # Sort by composite score
        ranked_matches.sort(key=lambda rm: rm.composite_score, reverse=True)

        # Apply tournament constraint (max 1 bet per tournament)
        diversified_matches = self._apply_tournament_constraint(ranked_matches)

        # Assign ranks and tiers
        for i, rm in enumerate(diversified_matches):
            rm.rank = i + 1
            if i == 0:
                rm.tier = RankingTier.GOLD
            elif i == 1:
                rm.tier = RankingTier.SILVER
            elif i == 2:
                rm.tier = RankingTier.BRONZE
            elif i < 5:
                rm.tier = RankingTier.TOP_5
            else:
                rm.tier = RankingTier.RANKED

        # Store in state (top 5)
        state.top_matches = [rm.match for rm in diversified_matches[:5]]

        # Store ranked matches for reporting
        state.ranked_matches = diversified_matches[:5]

        # Log ranking details
        if diversified_matches:
            state = add_message(
                state,
                "ranker",
                f"Top {min(len(diversified_matches), 5)} matches selected (diversified by tournament):"
            )

            for rm in diversified_matches[:5]:
                state = add_message(
                    state,
                    "ranker",
                    f"  #{rm.rank} [{rm.tier.name}] {rm.match.home_player.name} vs {rm.match.away_player.name} | "
                    f"Score: {rm.composite_score:.4f} | {rm.selection.upper()}: {rm.probability:.1%} @ {rm.odds:.2f}"
                )

        return state

    def _create_ranked_match(self, match: Match) -> Optional[RankedMatch]:
        """
        Create a RankedMatch from a Match with all metrics.

        Args:
            match: Match with prediction

        Returns:
            RankedMatch with composite score and details
        """
        if not match.prediction:
            return None

        # Determine selection (home or away)
        home_prob = match.prediction.home_win_probability
        away_prob = match.prediction.away_win_probability

        if home_prob > away_prob:
            selection = "home"
            selection_name = match.home_player.name
            probability = home_prob
        else:
            selection = "away"
            selection_name = match.away_player.name
            probability = away_prob

        # Get odds (use best available or estimate)
        odds = self._get_best_odds(match, selection)
        bookmaker = self._get_bookmaker(match, selection)

        # Calculate edge
        implied_prob = 1 / odds if odds > 1 else 0.5
        edge = max(0, probability - implied_prob)

        # Get quality and confidence
        quality_score = match.data_quality.overall_score if match.data_quality else 0.5
        confidence = match.prediction.confidence

        # Calculate composite score using geometric mean formula
        normalized_edge = min(edge, self.MAX_EDGE_CAP) / self.MAX_EDGE_CAP

        # Prevent zero values
        normalized_edge = max(normalized_edge, 0.01)
        quality_score = max(quality_score, 0.01)
        confidence = max(confidence, 0.01)

        # Composite score: edge^0.5 * quality^0.3 * confidence^0.2
        edge_component = math.pow(normalized_edge, self.EDGE_EXPONENT)
        quality_component = math.pow(quality_score, self.QUALITY_EXPONENT)
        confidence_component = math.pow(confidence, self.CONFIDENCE_EXPONENT)

        composite_score = edge_component * quality_component * confidence_component

        # Calculate Kelly stake
        kelly_stake = self._calculate_kelly_stake(probability, odds, quality_score)
        stake_recommendation = self._get_stake_recommendation(kelly_stake, quality_score)

        # Generate reasoning
        reasoning = self._generate_reasoning(match, edge, quality_score, confidence)
        key_factors = self._extract_key_factors(match)
        warnings = self._generate_warnings(match, quality_score, edge)

        # Extract tournament from league
        tournament = self._extract_tournament(match.league)

        return RankedMatch(
            match=match,
            rank=0,  # Will be assigned after sorting
            tier=RankingTier.RANKED,  # Will be updated
            composite_score=composite_score,
            edge_component=edge_component,
            quality_component=quality_component,
            confidence_component=confidence_component,
            edge=edge,
            quality_score=quality_score,
            confidence=confidence,
            selection=selection,
            selection_name=selection_name,
            probability=probability,
            odds=odds,
            bookmaker=bookmaker,
            stake_recommendation=stake_recommendation,
            kelly_stake=kelly_stake,
            reasoning=reasoning,
            key_factors=key_factors,
            warnings=warnings,
            tournament=tournament,
            league=match.league,
            match_time=match.match_time if hasattr(match, 'match_time') else ""
        )

    def _apply_tournament_constraint(self, ranked_matches: List[RankedMatch]) -> List[RankedMatch]:
        """
        Apply max 1 bet per tournament constraint.

        Args:
            ranked_matches: Sorted list of ranked matches

        Returns:
            Filtered list with max 1 bet per tournament
        """
        seen_tournaments = set()
        diversified = []

        for rm in ranked_matches:
            tournament_key = rm.tournament.lower().strip()

            # Skip if we already have a bet from this tournament
            if tournament_key and tournament_key in seen_tournaments:
                continue

            diversified.append(rm)
            if tournament_key:
                seen_tournaments.add(tournament_key)

        return diversified

    def _extract_tournament(self, league: str) -> str:
        """Extract tournament name from league string."""
        if not league:
            return ""
        # Common tournament patterns
        parts = league.split(" - ")
        if len(parts) > 1:
            return parts[0].strip()
        return league.strip()

    def _get_best_odds(self, match: Match, selection: str) -> float:
        """Get best available odds for selection."""
        # Try to get from odds data
        if hasattr(match, 'odds') and match.odds:
            if selection == "home" and hasattr(match.odds, 'home'):
                return match.odds.home
            elif selection == "away" and hasattr(match.odds, 'away'):
                return match.odds.away

        # Estimate from probability
        prob = match.prediction.home_win_probability if selection == "home" else match.prediction.away_win_probability
        if prob > 0:
            return round(1 / prob * 0.95, 2)  # Add 5% margin
        return 2.0

    def _get_bookmaker(self, match: Match, selection: str) -> str:
        """Get bookmaker with best odds."""
        if hasattr(match, 'odds') and match.odds and hasattr(match.odds, 'bookmaker'):
            return match.odds.bookmaker
        return "Best Available"

    def _calculate_kelly_stake(self, probability: float, odds: float, quality: float) -> float:
        """
        Calculate Kelly Criterion stake with quality adjustment.

        Formula: (p * (odds - 1) - (1 - p)) / (odds - 1)
        Adjusted by quality and using quarter Kelly.
        """
        if odds <= 1:
            return 0.0

        # Kelly formula
        q = 1 - probability
        kelly = (probability * (odds - 1) - q) / (odds - 1)

        # Apply quality multiplier
        quality_mult = 0.3 + (quality * 0.7)  # 0.3 to 1.0

        # Quarter Kelly for conservative betting
        adjusted = kelly * 0.25 * quality_mult

        # Cap at 5% max stake
        return min(max(adjusted, 0), 0.05)

    def _get_stake_recommendation(self, kelly_stake: float, quality: float) -> str:
        """Generate human-readable stake recommendation."""
        pct = kelly_stake * 100

        if pct < 0.5:
            return "0.5% bankroll (minimum)"
        elif pct < 1.0:
            return f"0.5-1% bankroll"
        elif pct < 2.0:
            return f"1-1.5% bankroll"
        elif pct < 3.0:
            return f"1.5-2% bankroll"
        elif pct < 4.0:
            return f"2-3% bankroll"
        else:
            return f"3-5% bankroll (high confidence)"

    def _generate_reasoning(
        self,
        match: Match,
        edge: float,
        quality: float,
        confidence: float
    ) -> List[str]:
        """Generate detailed reasoning for the bet."""
        reasoning = []

        # Edge reasoning
        if edge > 0.10:
            reasoning.append(f"Strong edge of {edge*100:.1f}% detected")
        elif edge > 0.05:
            reasoning.append(f"Moderate edge of {edge*100:.1f}% identified")
        elif edge > 0.03:
            reasoning.append(f"Small but viable edge of {edge*100:.1f}%")

        # Quality reasoning
        if quality >= 0.80:
            reasoning.append("Excellent data quality from multiple sources")
        elif quality >= 0.60:
            reasoning.append("Good data quality with consistent information")
        elif quality >= 0.45:
            reasoning.append("Acceptable data quality (proceed with caution)")

        # Confidence reasoning
        if confidence >= 0.75:
            reasoning.append("High confidence prediction based on strong signals")
        elif confidence >= 0.60:
            reasoning.append("Moderate confidence with clear directional signal")

        # Form reasoning
        if match.prediction and hasattr(match.prediction, 'factors'):
            factors = match.prediction.factors
            if factors.get('form_advantage'):
                reasoning.append(f"Form advantage: {factors['form_advantage']}")
            if factors.get('h2h_advantage'):
                reasoning.append(f"H2H record favors selection")

        # News reasoning
        if match.news_articles and len(match.news_articles) > 0:
            reasoning.append(f"Analysis includes {len(match.news_articles)} relevant news articles")

        return reasoning

    def _extract_key_factors(self, match: Match) -> List[str]:
        """Extract key factors that influenced the prediction."""
        factors = []

        if not match.prediction:
            return factors

        pred = match.prediction
        home_prob = pred.home_win_probability
        away_prob = pred.away_win_probability

        # Ranking factor
        if hasattr(match, 'home_player') and hasattr(match, 'away_player'):
            home_rank = getattr(match.home_player, 'ranking', 0)
            away_rank = getattr(match.away_player, 'ranking', 0)
            if home_rank and away_rank:
                if home_rank < away_rank:
                    factors.append(f"Ranking advantage (#{home_rank} vs #{away_rank})")
                elif away_rank < home_rank:
                    factors.append(f"Ranking advantage (#{away_rank} vs #{home_rank})")

        # Probability strength
        max_prob = max(home_prob, away_prob)
        if max_prob > 0.70:
            factors.append(f"Strong probability signal ({max_prob:.0%})")
        elif max_prob > 0.60:
            factors.append(f"Clear favorite ({max_prob:.0%})")

        # Quality data
        if match.data_quality and match.data_quality.overall_score >= 0.75:
            factors.append("High quality data sources")

        return factors

    def _generate_warnings(
        self,
        match: Match,
        quality: float,
        edge: float
    ) -> List[str]:
        """Generate warnings about potential risks."""
        warnings = []

        # Low quality warning
        if quality < 0.50:
            warnings.append("Low data quality - consider reducing stake")

        # Small edge warning
        if edge < 0.03:
            warnings.append("Edge below 3% - marginal value")

        # Close match warning
        if match.prediction:
            home_prob = match.prediction.home_win_probability
            away_prob = match.prediction.away_win_probability
            if abs(home_prob - away_prob) < 0.10:
                warnings.append("Close match - higher variance expected")

        # News concerns
        if match.news_articles:
            for article in match.news_articles:
                if hasattr(article, 'sentiment') and article.sentiment == 'negative':
                    warnings.append("Negative news detected - verify before betting")
                    break

        return warnings

    def get_ranking_criteria(self) -> Dict[str, float]:
        """Get the ranking criteria (exponents)."""
        return {
            "edge_exponent": self.EDGE_EXPONENT,
            "quality_exponent": self.QUALITY_EXPONENT,
            "confidence_exponent": self.CONFIDENCE_EXPONENT,
            "max_edge_cap": self.MAX_EDGE_CAP
        }


# === REPORT FORMATTING ===

def format_top_3_report(ranked_matches: List[RankedMatch], include_details: bool = True) -> str:
    """
    Format top 3 value bets as a text report.

    Args:
        ranked_matches: List of RankedMatch objects (sorted by rank)
        include_details: Whether to include detailed reasoning

    Returns:
        Formatted text report
    """
    if not ranked_matches:
        return "No value bets found for this analysis.\n"

    lines = []
    lines.append("=" * 60)
    lines.append("           🏆 TOP VALUE BETS - NEXUS AI 🏆")
    lines.append("=" * 60)
    lines.append("")

    tier_emoji = {
        RankingTier.GOLD: "🥇",
        RankingTier.SILVER: "🥈",
        RankingTier.BRONZE: "🥉",
        RankingTier.TOP_5: "📊",
        RankingTier.RANKED: "📈"
    }

    for rm in ranked_matches[:3]:
        emoji = tier_emoji.get(rm.tier, "")

        lines.append(f"{emoji} #{rm.rank} {rm.tier.name}")
        lines.append("-" * 40)
        lines.append(f"Match: {rm.match.home_player.name} vs {rm.match.away_player.name}")
        lines.append(f"Tournament: {rm.tournament}")
        lines.append(f"Selection: {rm.selection_name} ({rm.selection.upper()})")
        lines.append(f"Probability: {rm.probability:.1%}")
        lines.append(f"Odds: {rm.odds:.2f} @ {rm.bookmaker}")
        lines.append(f"Edge: {rm.edge*100:.2f}%")
        lines.append(f"Quality Score: {rm.quality_score:.0%}")
        lines.append(f"Confidence: {rm.confidence:.0%}")
        lines.append(f"Composite Score: {rm.composite_score:.4f}")
        lines.append(f"Stake: {rm.stake_recommendation}")

        if include_details:
            if rm.key_factors:
                lines.append("")
                lines.append("Key Factors:")
                for factor in rm.key_factors:
                    lines.append(f"  ✓ {factor}")

            if rm.reasoning:
                lines.append("")
                lines.append("Reasoning:")
                for reason in rm.reasoning[:3]:  # Limit to 3
                    lines.append(f"  • {reason}")

            if rm.warnings:
                lines.append("")
                lines.append("⚠️ Warnings:")
                for warning in rm.warnings:
                    lines.append(f"  ⚠ {warning}")

        lines.append("")
        lines.append("")

    # Summary
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("-" * 40)

    total_kelly = sum(rm.kelly_stake for rm in ranked_matches[:3])
    avg_edge = sum(rm.edge for rm in ranked_matches[:3]) / min(len(ranked_matches), 3)
    avg_quality = sum(rm.quality_score for rm in ranked_matches[:3]) / min(len(ranked_matches), 3)

    lines.append(f"Total Suggested Stake: {total_kelly*100:.1f}% bankroll")
    lines.append(f"Average Edge: {avg_edge*100:.2f}%")
    lines.append(f"Average Quality: {avg_quality:.0%}")
    lines.append(f"Tournaments: {len(set(rm.tournament for rm in ranked_matches[:3]))}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_top_3_html(ranked_matches: List[RankedMatch]) -> str:
    """
    Format top 3 value bets as HTML cards.

    Args:
        ranked_matches: List of RankedMatch objects

    Returns:
        HTML string with styled cards
    """
    if not ranked_matches:
        return "<p>No value bets found.</p>"

    tier_colors = {
        RankingTier.GOLD: "linear-gradient(135deg, #FFD700, #FFA500)",
        RankingTier.SILVER: "linear-gradient(135deg, #C0C0C0, #A0A0A0)",
        RankingTier.BRONZE: "linear-gradient(135deg, #CD7F32, #8B4513)"
    }

    html_parts = ['<div class="top-3-container">']

    for rm in ranked_matches[:3]:
        gradient = tier_colors.get(rm.tier, "linear-gradient(135deg, #667eea, #764ba2)")

        html_parts.append(f'''
        <div class="value-bet-card" style="background: {gradient};">
            <div class="rank-badge">#{rm.rank} {rm.tier.name}</div>
            <h3>{rm.match.home_player.name} vs {rm.match.away_player.name}</h3>
            <p class="tournament">{rm.tournament}</p>
            <div class="selection">
                <span class="label">Selection:</span>
                <span class="value">{rm.selection_name}</span>
            </div>
            <div class="stats-grid">
                <div class="stat">
                    <span class="stat-value">{rm.odds:.2f}</span>
                    <span class="stat-label">Odds</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{rm.edge*100:.1f}%</span>
                    <span class="stat-label">Edge</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{rm.quality_score:.0%}</span>
                    <span class="stat-label">Quality</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{rm.confidence:.0%}</span>
                    <span class="stat-label">Confidence</span>
                </div>
            </div>
            <div class="stake-recommendation">
                <span class="label">Recommended Stake:</span>
                <span class="value">{rm.stake_recommendation}</span>
            </div>
        </div>
        ''')

    html_parts.append('</div>')
    return '\n'.join(html_parts)


# === HELPER FUNCTIONS ===

async def rank_matches(matches: List[Match], max_results: int = 5) -> List[RankedMatch]:
    """
    Convenience function to rank matches.

    Args:
        matches: List of matches with predictions
        max_results: Maximum number of top matches to return

    Returns:
        List of top ranked matches as RankedMatch objects
    """
    agent = RankerAgent()

    state = NexusState(
        sport=matches[0].sport if matches else "tennis",
        date=datetime.now().strftime("%Y-%m-%d"),
        matches=matches
    )

    result_state = await agent.process(state)

    # Return RankedMatch objects if available
    if hasattr(result_state, 'ranked_matches') and result_state.ranked_matches:
        return result_state.ranked_matches[:max_results]

    # Fallback: create basic RankedMatch objects
    ranked = []
    for i, match in enumerate(result_state.top_matches[:max_results]):
        rm = RankedMatch(
            match=match,
            rank=i + 1,
            tier=RankingTier.GOLD if i == 0 else (RankingTier.SILVER if i == 1 else RankingTier.BRONZE),
            composite_score=match.composite_score if hasattr(match, 'composite_score') else 0.5,
            edge_component=0.5,
            quality_component=0.5,
            confidence_component=0.5,
            edge=0.05,
            quality_score=0.7,
            confidence=0.65,
            selection="home",
            selection_name=match.home_player.name,
            probability=0.6,
            odds=2.0,
            bookmaker="Best Available",
            stake_recommendation="1-2% bankroll",
            kelly_stake=0.015,
            tournament=match.league,
            league=match.league
        )
        ranked.append(rm)

    return ranked
