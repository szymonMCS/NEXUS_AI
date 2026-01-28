# core/llm/injury_extractor.py
"""
Injury Information Extractor using Kimi LLM.

Checkpoint: 7.2

Extracts injury, suspension, and availability information from:
- News articles
- Social media
- Team announcements

Integrates with existing news scrapers to provide structured injury data.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from core.llm.kimi_client import KimiClient, KimiResponse

logger = logging.getLogger(__name__)


class PlayerStatus(str, Enum):
    """Player availability status."""
    OUT = "out"                    # Definitely unavailable
    DOUBTFUL = "doubtful"          # Likely unavailable
    QUESTIONABLE = "questionable"  # 50/50 chance
    PROBABLE = "probable"          # Likely available
    AVAILABLE = "available"        # Confirmed available
    UNKNOWN = "unknown"


@dataclass
class PlayerInjury:
    """Represents a player's injury status."""
    player_name: str
    team: str
    injury_type: str
    status: PlayerStatus
    expected_return: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "team": self.team,
            "injury_type": self.injury_type,
            "status": self.status.value,
            "expected_return": self.expected_return,
            "last_updated": self.last_updated.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerInjury":
        return cls(
            player_name=data["player_name"],
            team=data["team"],
            injury_type=data.get("injury_type", "unknown"),
            status=PlayerStatus(data.get("status", "unknown")),
            expected_return=data.get("expected_return"),
            last_updated=datetime.fromisoformat(data["last_updated"])
            if "last_updated" in data else datetime.utcnow(),
            source=data.get("source"),
            confidence=data.get("confidence", 0.8),
        )


@dataclass
class PlayerSuspension:
    """Represents a player's suspension."""
    player_name: str
    team: str
    reason: str
    matches_remaining: int
    competition: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "team": self.team,
            "reason": self.reason,
            "matches_remaining": self.matches_remaining,
            "competition": self.competition,
            "last_updated": self.last_updated.isoformat(),
            "source": self.source,
        }


@dataclass
class TeamAvailability:
    """Team's current injury/suspension situation."""
    team_name: str
    injuries: List[PlayerInjury] = field(default_factory=list)
    suspensions: List[PlayerSuspension] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def players_out(self) -> List[str]:
        """Players definitely unavailable."""
        return [
            inj.player_name for inj in self.injuries
            if inj.status in (PlayerStatus.OUT, PlayerStatus.DOUBTFUL)
        ] + [
            sus.player_name for sus in self.suspensions
            if sus.matches_remaining > 0
        ]

    @property
    def players_doubtful(self) -> List[str]:
        """Players questionable/doubtful."""
        return [
            inj.player_name for inj in self.injuries
            if inj.status == PlayerStatus.QUESTIONABLE
        ]

    @property
    def total_unavailable(self) -> int:
        """Total number of unavailable players."""
        return len(self.players_out)

    @property
    def injury_severity_score(self) -> float:
        """
        Score indicating impact of injuries (0-1).
        Higher = more impactful injuries (key players out).
        """
        if not self.injuries and not self.suspensions:
            return 0.0

        score = 0.0
        for inj in self.injuries:
            if inj.status == PlayerStatus.OUT:
                score += 0.15
            elif inj.status == PlayerStatus.DOUBTFUL:
                score += 0.10
            elif inj.status == PlayerStatus.QUESTIONABLE:
                score += 0.05

        for sus in self.suspensions:
            if sus.matches_remaining > 0:
                score += 0.15

        return min(1.0, score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_name": self.team_name,
            "injuries": [inj.to_dict() for inj in self.injuries],
            "suspensions": [sus.to_dict() for sus in self.suspensions],
            "players_out": self.players_out,
            "players_doubtful": self.players_doubtful,
            "total_unavailable": self.total_unavailable,
            "injury_severity_score": self.injury_severity_score,
            "last_updated": self.last_updated.isoformat(),
        }


class InjuryExtractor:
    """
    Extracts injury information from text using Kimi LLM.

    Usage:
        extractor = InjuryExtractor()
        injuries = await extractor.extract_from_news(news_text, ["Arsenal", "Chelsea"])
    """

    def __init__(self, kimi_client: Optional[KimiClient] = None):
        """
        Initialize extractor.

        Args:
            kimi_client: Optional pre-configured KimiClient
        """
        self._kimi = kimi_client
        self._cache: Dict[str, TeamAvailability] = {}
        self._cache_ttl = timedelta(hours=2)

    async def _get_client(self) -> KimiClient:
        """Get or create Kimi client."""
        if self._kimi is None:
            self._kimi = KimiClient()
        return self._kimi

    async def extract_from_news(
        self,
        news_text: str,
        team_names: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, TeamAvailability]:
        """
        Extract injury information from news text.

        Args:
            news_text: News article text
            team_names: Teams to focus on (optional)
            source: Source of the news

        Returns:
            Dict mapping team name to TeamAvailability
        """
        client = await self._get_client()

        async with KimiClient() as kimi:
            response = await kimi.extract_injuries(
                news_text=news_text,
                team_names=team_names,
            )

        if not response.success:
            logger.warning(f"Kimi extraction failed: {response.error}")
            return {}

        return self._parse_extraction_response(response.content, source)

    def _parse_extraction_response(
        self,
        content: str,
        source: Optional[str] = None,
    ) -> Dict[str, TeamAvailability]:
        """Parse Kimi's JSON response into structured data."""
        try:
            # Try to extract JSON from response
            data = self._extract_json(content)

            if not data:
                logger.warning("No JSON found in Kimi response")
                return {}

            result: Dict[str, TeamAvailability] = {}

            # Process injuries
            for injury_data in data.get("injuries", []):
                team = injury_data.get("team", "Unknown")

                if team not in result:
                    result[team] = TeamAvailability(team_name=team)

                status_str = injury_data.get("status", "unknown").lower()
                status = self._parse_status(status_str)

                injury = PlayerInjury(
                    player_name=injury_data.get("player", "Unknown"),
                    team=team,
                    injury_type=injury_data.get("injury_type", "unspecified"),
                    status=status,
                    expected_return=injury_data.get("expected_return"),
                    source=source,
                )
                result[team].injuries.append(injury)

            # Process suspensions
            for sus_data in data.get("suspensions", []):
                team = sus_data.get("team", "Unknown")

                if team not in result:
                    result[team] = TeamAvailability(team_name=team)

                suspension = PlayerSuspension(
                    player_name=sus_data.get("player", "Unknown"),
                    team=team,
                    reason=sus_data.get("reason", "unspecified"),
                    matches_remaining=sus_data.get("matches_remaining", 1),
                    source=source,
                )
                result[team].suspensions.append(suspension)

            return result

        except Exception as e:
            logger.error(f"Failed to parse Kimi response: {e}")
            return {}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that may contain other content."""
        import re

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _parse_status(self, status_str: str) -> PlayerStatus:
        """Parse status string to enum."""
        status_map = {
            "out": PlayerStatus.OUT,
            "definitely out": PlayerStatus.OUT,
            "ruled out": PlayerStatus.OUT,
            "doubtful": PlayerStatus.DOUBTFUL,
            "unlikely": PlayerStatus.DOUBTFUL,
            "questionable": PlayerStatus.QUESTIONABLE,
            "50-50": PlayerStatus.QUESTIONABLE,
            "day-to-day": PlayerStatus.QUESTIONABLE,
            "probable": PlayerStatus.PROBABLE,
            "likely": PlayerStatus.PROBABLE,
            "available": PlayerStatus.AVAILABLE,
            "fit": PlayerStatus.AVAILABLE,
        }

        for key, value in status_map.items():
            if key in status_str.lower():
                return value

        return PlayerStatus.UNKNOWN

    async def extract_from_multiple_sources(
        self,
        news_items: List[Dict[str, str]],
        team_names: Optional[List[str]] = None,
    ) -> Dict[str, TeamAvailability]:
        """
        Extract injuries from multiple news items and merge results.

        Args:
            news_items: List of {"text": "...", "source": "...", "date": "..."}
            team_names: Teams to focus on

        Returns:
            Merged TeamAvailability dict
        """
        all_availability: Dict[str, TeamAvailability] = {}

        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)

        async def process_item(item: Dict[str, str]):
            async with semaphore:
                return await self.extract_from_news(
                    news_text=item.get("text", ""),
                    team_names=team_names,
                    source=item.get("source"),
                )

        results = await asyncio.gather(
            *[process_item(item) for item in news_items],
            return_exceptions=True,
        )

        # Merge results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Extraction failed: {result}")
                continue

            for team, availability in result.items():
                if team not in all_availability:
                    all_availability[team] = availability
                else:
                    # Merge injuries and suspensions, avoiding duplicates
                    existing = all_availability[team]
                    existing_players = {inj.player_name for inj in existing.injuries}

                    for inj in availability.injuries:
                        if inj.player_name not in existing_players:
                            existing.injuries.append(inj)
                            existing_players.add(inj.player_name)

                    existing_sus_players = {sus.player_name for sus in existing.suspensions}
                    for sus in availability.suspensions:
                        if sus.player_name not in existing_sus_players:
                            existing.suspensions.append(sus)

        return all_availability

    def get_cached_availability(self, team_name: str) -> Optional[TeamAvailability]:
        """Get cached availability for a team."""
        if team_name in self._cache:
            cached = self._cache[team_name]
            if datetime.utcnow() - cached.last_updated < self._cache_ttl:
                return cached
        return None

    def cache_availability(self, team_name: str, availability: TeamAvailability):
        """Cache team availability."""
        self._cache[team_name] = availability

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()


# Convenience function
async def get_team_injuries(
    team_name: str,
    news_texts: List[str],
) -> TeamAvailability:
    """
    Quick function to get injury report for a team.

    Args:
        team_name: Team to analyze
        news_texts: List of news articles

    Returns:
        TeamAvailability for the team
    """
    extractor = InjuryExtractor()
    news_items = [{"text": text, "source": "news"} for text in news_texts]

    result = await extractor.extract_from_multiple_sources(
        news_items=news_items,
        team_names=[team_name],
    )

    return result.get(team_name, TeamAvailability(team_name=team_name))
