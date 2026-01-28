# core/llm/match_analyzer.py
"""
Deep Match Analysis using Kimi LLM.

Checkpoint: 7.3

Provides comprehensive match analysis combining:
- Statistical data
- Form analysis
- Head-to-head records
- Injury/availability information
- News and sentiment
- Tactical considerations

Returns structured analysis for prediction enhancement.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.llm.kimi_client import KimiClient, KimiResponse, KimiModel
from core.llm.injury_extractor import TeamAvailability, InjuryExtractor

logger = logging.getLogger(__name__)


class AnalysisConfidence(str, Enum):
    """Confidence level of analysis."""
    VERY_HIGH = "very_high"   # >85%
    HIGH = "high"             # 70-85%
    MEDIUM = "medium"         # 55-70%
    LOW = "low"               # 40-55%
    VERY_LOW = "very_low"     # <40%


@dataclass
class MatchFactor:
    """A factor affecting match outcome."""
    name: str
    description: str
    impact: str  # "positive_home", "positive_away", "neutral"
    weight: float  # 0.0 to 1.0
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "impact": self.impact,
            "weight": self.weight,
            "confidence": self.confidence,
        }


@dataclass
class MatchAnalysis:
    """Comprehensive match analysis result."""
    match_id: Optional[str]
    home_team: str
    away_team: str
    league: str

    # Core predictions
    predicted_outcome: str  # "home", "draw", "away"
    outcome_confidence: float
    predicted_goals: float
    goals_confidence: float

    # Detailed analysis
    key_factors: List[MatchFactor] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # Narrative
    summary: str = ""
    tactical_insight: str = ""

    # Market suggestions
    suggested_bets: List[Dict[str, Any]] = field(default_factory=list)

    # Meta
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    model_used: str = ""
    raw_response: str = ""

    @property
    def confidence_level(self) -> AnalysisConfidence:
        """Get confidence level category."""
        if self.outcome_confidence > 0.85:
            return AnalysisConfidence.VERY_HIGH
        elif self.outcome_confidence > 0.70:
            return AnalysisConfidence.HIGH
        elif self.outcome_confidence > 0.55:
            return AnalysisConfidence.MEDIUM
        elif self.outcome_confidence > 0.40:
            return AnalysisConfidence.LOW
        else:
            return AnalysisConfidence.VERY_LOW

    @property
    def home_advantage_score(self) -> float:
        """Calculate home team advantage from factors."""
        home_score = 0.0
        total_weight = 0.0

        for factor in self.key_factors:
            if factor.impact == "positive_home":
                home_score += factor.weight * factor.confidence
            elif factor.impact == "positive_away":
                home_score -= factor.weight * factor.confidence
            total_weight += factor.weight

        if total_weight == 0:
            return 0.5

        # Normalize to 0-1 range
        return (home_score / total_weight + 1) / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "predicted_outcome": self.predicted_outcome,
            "outcome_confidence": self.outcome_confidence,
            "confidence_level": self.confidence_level.value,
            "predicted_goals": self.predicted_goals,
            "goals_confidence": self.goals_confidence,
            "key_factors": [f.to_dict() for f in self.key_factors],
            "risks": self.risks,
            "opportunities": self.opportunities,
            "summary": self.summary,
            "tactical_insight": self.tactical_insight,
            "suggested_bets": self.suggested_bets,
            "home_advantage_score": self.home_advantage_score,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "model_used": self.model_used,
        }


class MatchAnalyzer:
    """
    Comprehensive match analyzer using Kimi LLM.

    Combines multiple data sources for deep analysis:
    - Statistical features
    - Injury data
    - News context
    - Historical patterns

    Usage:
        analyzer = MatchAnalyzer()
        analysis = await analyzer.analyze_match(
            home_team="Arsenal",
            away_team="Chelsea",
            league="Premier League",
            context=match_context,
        )
    """

    def __init__(
        self,
        kimi_client: Optional[KimiClient] = None,
        injury_extractor: Optional[InjuryExtractor] = None,
    ):
        self._kimi = kimi_client
        self._injury_extractor = injury_extractor or InjuryExtractor()
        self._analysis_cache: Dict[str, MatchAnalysis] = {}

    async def analyze_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_injuries: bool = True,
        model: str = KimiModel.V1_8K,
    ) -> MatchAnalysis:
        """
        Perform comprehensive match analysis.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League/competition name
            match_id: Optional match identifier
            context: Additional context (form, h2h, stats, news)
            include_injuries: Whether to include injury analysis
            model: Kimi model to use

        Returns:
            MatchAnalysis with detailed predictions and insights
        """
        # Build comprehensive context
        full_context = await self._build_context(
            home_team=home_team,
            away_team=away_team,
            league=league,
            context=context or {},
            include_injuries=include_injuries,
        )

        # Generate analysis prompt
        prompt = self._build_analysis_prompt(
            home_team=home_team,
            away_team=away_team,
            league=league,
            context=full_context,
        )

        # Call Kimi for analysis
        async with KimiClient(model=model) as kimi:
            response = await kimi.chat(
                message=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.2,
                max_tokens=2048,
            )

        if not response.success:
            logger.error(f"Kimi analysis failed: {response.error}")
            return self._create_fallback_analysis(
                home_team, away_team, league, match_id, response.error
            )

        # Parse response into MatchAnalysis
        analysis = self._parse_analysis_response(
            response=response,
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_id=match_id,
        )

        return analysis

    async def _build_context(
        self,
        home_team: str,
        away_team: str,
        league: str,
        context: Dict[str, Any],
        include_injuries: bool,
    ) -> Dict[str, Any]:
        """Build comprehensive context for analysis."""
        full_context = dict(context)

        # Add injury information if requested
        if include_injuries:
            injuries = await self._get_injury_context(home_team, away_team)
            if injuries:
                full_context["injuries"] = injuries

        # Add default context structure
        if "home_form" not in full_context:
            full_context["home_form"] = context.get("home_recent_form", "Unknown")

        if "away_form" not in full_context:
            full_context["away_form"] = context.get("away_recent_form", "Unknown")

        full_context["league"] = league

        return full_context

    async def _get_injury_context(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[Dict[str, Any]]:
        """Get injury context for both teams."""
        # Check cache first
        home_avail = self._injury_extractor.get_cached_availability(home_team)
        away_avail = self._injury_extractor.get_cached_availability(away_team)

        result = {}

        if home_avail:
            result["home"] = {
                "players_out": home_avail.players_out,
                "players_doubtful": home_avail.players_doubtful,
                "severity_score": home_avail.injury_severity_score,
            }

        if away_avail:
            result["away"] = {
                "players_out": away_avail.players_out,
                "players_doubtful": away_avail.players_doubtful,
                "severity_score": away_avail.injury_severity_score,
            }

        return result if result else None

    def _get_system_prompt(self) -> str:
        """Get system prompt for match analysis."""
        return """You are an expert sports analyst specializing in football/soccer match prediction.

Your analysis should be:
1. Data-driven and objective
2. Consider multiple factors (form, h2h, injuries, tactics)
3. Identify key differentiators
4. Assess risks and uncertainties
5. Provide actionable betting insights

IMPORTANT: Return your analysis as valid JSON with this exact structure:
{
    "predicted_outcome": "home" | "draw" | "away",
    "outcome_confidence": float (0.0 to 1.0),
    "predicted_total_goals": float,
    "goals_confidence": float (0.0 to 1.0),
    "key_factors": [
        {
            "name": "Factor name",
            "description": "Brief explanation",
            "impact": "positive_home" | "positive_away" | "neutral",
            "weight": float (0.0 to 1.0),
            "confidence": float (0.0 to 1.0)
        }
    ],
    "risks": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"],
    "summary": "2-3 sentence match summary",
    "tactical_insight": "Key tactical consideration",
    "suggested_bets": [
        {
            "type": "1X2" | "over_under" | "btts" | "asian_handicap",
            "selection": "specific selection",
            "confidence": float,
            "reasoning": "brief reasoning"
        }
    ]
}

Be conservative with confidence levels. Only use >0.7 confidence when factors strongly support the prediction."""

    def _build_analysis_prompt(
        self,
        home_team: str,
        away_team: str,
        league: str,
        context: Dict[str, Any],
    ) -> str:
        """Build the analysis prompt."""
        prompt_parts = [
            f"Analyze this {league} match:",
            f"{home_team} (HOME) vs {away_team} (AWAY)",
            "",
        ]

        # Add context sections
        if context.get("home_form"):
            prompt_parts.append(f"Home team recent form: {context['home_form']}")

        if context.get("away_form"):
            prompt_parts.append(f"Away team recent form: {context['away_form']}")

        if context.get("h2h"):
            prompt_parts.append(f"Head-to-head: {context['h2h']}")

        if context.get("injuries"):
            injuries = context["injuries"]
            if injuries.get("home"):
                home_inj = injuries["home"]
                prompt_parts.append(
                    f"Home team injuries: {', '.join(home_inj.get('players_out', ['None']))}"
                )
            if injuries.get("away"):
                away_inj = injuries["away"]
                prompt_parts.append(
                    f"Away team injuries: {', '.join(away_inj.get('players_out', ['None']))}"
                )

        if context.get("odds"):
            odds = context["odds"]
            prompt_parts.append(
                f"Market odds - Home: {odds.get('home', '?')}, "
                f"Draw: {odds.get('draw', '?')}, Away: {odds.get('away', '?')}"
            )

        if context.get("stats"):
            stats = context["stats"]
            prompt_parts.append(f"Key stats: {json.dumps(stats, indent=2)}")

        if context.get("news"):
            prompt_parts.append(f"Recent news: {context['news']}")

        prompt_parts.extend([
            "",
            "Provide comprehensive analysis in JSON format.",
        ])

        return "\n".join(prompt_parts)

    def _parse_analysis_response(
        self,
        response: KimiResponse,
        home_team: str,
        away_team: str,
        league: str,
        match_id: Optional[str],
    ) -> MatchAnalysis:
        """Parse Kimi response into MatchAnalysis."""
        try:
            data = self._extract_json(response.content)

            if not data:
                logger.warning("Could not parse JSON from Kimi response")
                return self._create_fallback_analysis(
                    home_team, away_team, league, match_id, "JSON parse error"
                )

            # Parse key factors
            key_factors = []
            for factor_data in data.get("key_factors", []):
                key_factors.append(MatchFactor(
                    name=factor_data.get("name", ""),
                    description=factor_data.get("description", ""),
                    impact=factor_data.get("impact", "neutral"),
                    weight=float(factor_data.get("weight", 0.5)),
                    confidence=float(factor_data.get("confidence", 0.5)),
                ))

            return MatchAnalysis(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                league=league,
                predicted_outcome=data.get("predicted_outcome", "draw"),
                outcome_confidence=float(data.get("outcome_confidence", 0.5)),
                predicted_goals=float(data.get("predicted_total_goals", 2.5)),
                goals_confidence=float(data.get("goals_confidence", 0.5)),
                key_factors=key_factors,
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                summary=data.get("summary", ""),
                tactical_insight=data.get("tactical_insight", ""),
                suggested_bets=data.get("suggested_bets", []),
                model_used=response.model,
                raw_response=response.content,
            )

        except Exception as e:
            logger.error(f"Failed to parse analysis: {e}")
            return self._create_fallback_analysis(
                home_team, away_team, league, match_id, str(e)
            )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{[\s\S]*\})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _create_fallback_analysis(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_id: Optional[str],
        error: Optional[str],
    ) -> MatchAnalysis:
        """Create fallback analysis when Kimi fails."""
        return MatchAnalysis(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            predicted_outcome="draw",
            outcome_confidence=0.33,
            predicted_goals=2.5,
            goals_confidence=0.4,
            key_factors=[],
            risks=[f"Analysis unavailable: {error}" if error else "Analysis unavailable"],
            opportunities=[],
            summary=f"Unable to generate detailed analysis for {home_team} vs {away_team}.",
            tactical_insight="",
            suggested_bets=[],
            model_used="fallback",
        )

    async def batch_analyze(
        self,
        matches: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[MatchAnalysis]:
        """
        Analyze multiple matches.

        Args:
            matches: List of match dicts with home_team, away_team, league, etc.
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of MatchAnalysis results
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(match: Dict[str, Any]) -> MatchAnalysis:
            async with semaphore:
                return await self.analyze_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    league=match.get("league", "Unknown"),
                    match_id=match.get("match_id"),
                    context=match.get("context"),
                )

        results = await asyncio.gather(
            *[analyze_one(m) for m in matches],
            return_exceptions=True,
        )

        # Handle exceptions
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Match analysis failed: {result}")
                match = matches[i]
                analyses.append(self._create_fallback_analysis(
                    match["home_team"],
                    match["away_team"],
                    match.get("league", "Unknown"),
                    match.get("match_id"),
                    str(result),
                ))
            else:
                analyses.append(result)

        return analyses


# Convenience function
async def analyze_match(
    home_team: str,
    away_team: str,
    league: str,
    context: Optional[Dict[str, Any]] = None,
) -> MatchAnalysis:
    """
    Quick function to analyze a single match.

    Usage:
        analysis = await analyze_match("Arsenal", "Chelsea", "Premier League")
        print(analysis.summary)
    """
    analyzer = MatchAnalyzer()
    return await analyzer.analyze_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
        context=context,
    )
