# data/news/injury_extractor.py
"""
Injury information extractor using LLM.
Analyzes news articles to extract structured injury data.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import anthropic
import json

from config.settings import settings


class InjuryInfo(BaseModel):
    """Structured injury information"""

    player_name: str
    injury_type: str = Field(description="Type of injury (e.g., ankle sprain, hamstring)")
    severity: str = Field(description="low, medium, high, or unknown")
    status: str = Field(description="out, doubtful, questionable, day-to-day, or recovering")
    expected_return: Optional[str] = Field(None, description="Expected return timeframe")
    affects_performance: bool = Field(default=True, description="Whether it affects performance")
    source: str = Field(description="News source that reported it")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class InjuryExtractor:
    """
    Extracts injury information from news articles using Claude.
    """

    def __init__(self):
        self.client = None
        if settings.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def extract_injuries_from_article(
        self,
        article: Dict,
        player1: str,
        player2: str
    ) -> List[InjuryInfo]:
        """
        Extract injury information from a single news article.

        Args:
            article: Article dict with title, snippet, etc.
            player1: First player/team name
            player2: Second player/team name

        Returns:
            List of InjuryInfo objects
        """
        if not self.client:
            # Fallback to keyword-based extraction
            return self._fallback_extraction(article, player1, player2)

        try:
            # Prepare article text
            text = f"{article.get('title', '')} {article.get('snippet', '')}"

            # LLM prompt
            prompt = f"""Analyze this sports news article and extract any injury information about the players.

Article:
{text}

Players to check:
- {player1}
- {player2}

Extract injury information if mentioned. For each injury, provide:
1. player_name: Exact name of the injured player
2. injury_type: Specific injury (e.g., "ankle sprain", "hamstring tear")
3. severity: "low", "medium", "high", or "unknown"
4. status: "out", "doubtful", "questionable", "day-to-day", or "recovering"
5. expected_return: Time until return (e.g., "2 weeks", "unknown")
6. affects_performance: true/false - does it affect their play?
7. confidence: 0.0-1.0 - how confident are you in this extraction?

Return JSON array of injuries. If no injuries mentioned, return empty array [].
Example:
[
  {{
    "player_name": "Rafael Nadal",
    "injury_type": "knee injury",
    "severity": "high",
    "status": "out",
    "expected_return": "3 weeks",
    "affects_performance": true,
    "confidence": 0.9
  }}
]"""

            # Call Claude
            response = self.client.messages.create(
                model=settings.MODEL_NAME,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            content = response.content[0].text.strip()

            # Extract JSON from response (may have markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            injuries_data = json.loads(content)

            # Convert to InjuryInfo objects
            injuries = []
            for injury_dict in injuries_data:
                injury_dict["source"] = article.get("source", "unknown")
                injuries.append(InjuryInfo(**injury_dict))

            return injuries

        except Exception as e:
            print(f"Error extracting injuries with LLM: {e}")
            return self._fallback_extraction(article, player1, player2)

    def _fallback_extraction(
        self,
        article: Dict,
        player1: str,
        player2: str
    ) -> List[InjuryInfo]:
        """
        Fallback keyword-based injury extraction.

        Args:
            article: Article dict
            player1: First player name
            player2: Second player name

        Returns:
            List of InjuryInfo objects (may be empty)
        """
        text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
        injuries = []

        # Simple keyword detection
        injury_keywords = {
            "out": ["out", "ruled out", "sidelined", "withdrawn"],
            "doubtful": ["doubtful", "uncertain", "questionable"],
            "recovering": ["recovering", "recovery", "rehab", "rehabilitation"],
        }

        severity_keywords = {
            "high": ["serious", "severe", "major", "torn", "rupture", "break"],
            "medium": ["strain", "sprain", "minor"],
            "low": ["slight", "niggle", "discomfort"],
        }

        # Check each player
        for player in [player1, player2]:
            player_lower = player.lower()

            if player_lower not in text:
                continue

            # Check if injury-related keywords near player name
            if any(keyword in text for keyword_list in injury_keywords.values() for keyword in keyword_list):
                # Determine status
                status = "unknown"
                for status_key, keywords in injury_keywords.items():
                    if any(kw in text for kw in keywords):
                        status = status_key
                        break

                # Determine severity
                severity = "unknown"
                for sev_key, keywords in severity_keywords.items():
                    if any(kw in text for kw in keywords):
                        severity = sev_key
                        break

                # Create injury info
                injury = InjuryInfo(
                    player_name=player,
                    injury_type="unspecified",
                    severity=severity,
                    status=status,
                    expected_return=None,
                    affects_performance=True,
                    source=article.get("source", "unknown"),
                    confidence=0.5  # Lower confidence for keyword-based
                )
                injuries.append(injury)

        return injuries

    def extract_injuries_from_articles(
        self,
        articles: List[Dict],
        player1: str,
        player2: str
    ) -> Dict[str, List[InjuryInfo]]:
        """
        Extract injuries from multiple news articles.

        Args:
            articles: List of article dicts
            player1: First player name
            player2: Second player name

        Returns:
            Dict mapping player names to list of injury reports
        """
        all_injuries = {
            player1: [],
            player2: [],
        }

        for article in articles:
            # Only process articles that mention injuries
            text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
            if not self._has_injury_mention(text):
                continue

            injuries = self.extract_injuries_from_article(article, player1, player2)

            for injury in injuries:
                if injury.player_name == player1:
                    all_injuries[player1].append(injury)
                elif injury.player_name == player2:
                    all_injuries[player2].append(injury)

        return all_injuries

    def _has_injury_mention(self, text: str) -> bool:
        """Check if text mentions injuries"""
        keywords = [
            "injury", "injured", "hurt", "pain", "out", "doubtful",
            "questionable", "sidelined", "withdrawn", "kontuzja", "uraz"
        ]
        return any(keyword in text for keyword in keywords)

    def consolidate_injury_reports(
        self,
        injuries: List[InjuryInfo]
    ) -> Optional[InjuryInfo]:
        """
        Consolidate multiple injury reports about the same player.

        Takes the most recent, highest confidence report.

        Args:
            injuries: List of InjuryInfo objects for same player

        Returns:
            Consolidated InjuryInfo or None
        """
        if not injuries:
            return None

        # Sort by confidence (descending)
        sorted_injuries = sorted(injuries, key=lambda x: x.confidence, reverse=True)

        # Return highest confidence report
        return sorted_injuries[0]

    def get_injury_impact_score(self, injury: InjuryInfo) -> float:
        """
        Calculate impact score of an injury on match prediction (0.0 - 1.0).

        Args:
            injury: InjuryInfo object

        Returns:
            float: Impact score (higher = more impact on match)
        """
        score = 0.0

        # Status impact
        status_impact = {
            "out": 1.0,
            "doubtful": 0.7,
            "questionable": 0.5,
            "day-to-day": 0.3,
            "recovering": 0.2,
        }
        score += status_impact.get(injury.status, 0.3)

        # Severity impact
        severity_impact = {
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1,
        }
        score += severity_impact.get(injury.severity, 0.15)

        # Confidence adjustment
        score *= injury.confidence

        return min(score, 1.0)


# === HELPER FUNCTIONS ===

def extract_match_injuries(
    articles: List[Dict],
    player1: str,
    player2: str
) -> Dict:
    """
    Convenience function to extract and consolidate injuries for a match.

    Args:
        articles: List of news articles
        player1: First player name
        player2: Second player name

    Returns:
        Dict with injury information for both players
    """
    extractor = InjuryExtractor()
    injuries_by_player = extractor.extract_injuries_from_articles(
        articles, player1, player2
    )

    # Consolidate reports
    result = {}

    for player, injury_list in injuries_by_player.items():
        consolidated = extractor.consolidate_injury_reports(injury_list)
        if consolidated:
            result[player] = {
                "injury": consolidated.dict(),
                "impact_score": extractor.get_injury_impact_score(consolidated),
                "num_reports": len(injury_list),
            }
        else:
            result[player] = {
                "injury": None,
                "impact_score": 0.0,
                "num_reports": 0,
            }

    return result
