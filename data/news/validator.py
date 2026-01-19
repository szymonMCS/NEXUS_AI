# data/news/validator.py
"""
News validator - validates news sources and article quality.
Assigns reliability scores based on source tier and cross-validation.
"""

from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

from config.thresholds import RELIABLE_NEWS_SOURCES, thresholds


class NewsSourceValidator:
    """
    Validates news sources and assigns reliability scores.
    Implements cross-validation across multiple sources.
    """

    def __init__(self):
        self.tier1_sources = RELIABLE_NEWS_SOURCES["tier1"]
        self.tier2_sources = RELIABLE_NEWS_SOURCES["tier2"]
        self.tier3_sources = RELIABLE_NEWS_SOURCES["tier3"]

    def get_source_tier(self, source: str) -> int:
        """
        Get the tier level of a news source.

        Args:
            source: Source name (domain or publisher)

        Returns:
            int: 1 (highest), 2, 3, or 4 (unknown)
        """
        source_lower = source.lower()

        # Check exact matches first
        for tier1 in self.tier1_sources:
            if tier1.lower() in source_lower:
                return 1

        for tier2 in self.tier2_sources:
            if tier2.lower() in source_lower:
                return 2

        for tier3 in self.tier3_sources:
            if tier3.lower() in source_lower:
                return 3

        # Unknown source
        return 4

    def get_reliability_score(self, source: str) -> float:
        """
        Get reliability score for a source (0.0 - 1.0).

        Scoring:
        - Tier 1: 1.0
        - Tier 2: 0.8
        - Tier 3: 0.6
        - Unknown: 0.4
        """
        tier = self.get_source_tier(source)

        tier_scores = {
            1: 1.0,
            2: 0.8,
            3: 0.6,
            4: 0.4,
        }

        return tier_scores.get(tier, 0.4)

    def is_fresh(self, published_date: Optional[datetime]) -> bool:
        """
        Check if news article is fresh (within threshold).

        Args:
            published_date: Article publication date

        Returns:
            bool: True if fresh, False otherwise
        """
        if not published_date:
            return False

        max_age = timedelta(hours=thresholds.news_freshness_hours)
        age = datetime.now() - published_date

        return age <= max_age

    def validate_article(self, article: Dict) -> Dict:
        """
        Validate a single news article.

        Args:
            article: Article dict with title, source, published_date, etc.

        Returns:
            Dict with validation results
        """
        source = article.get("source", "")
        published_date = article.get("published_date")

        tier = self.get_source_tier(source)
        reliability_score = self.get_reliability_score(source)
        is_fresh = self.is_fresh(published_date)

        # Calculate overall article quality score
        quality_score = reliability_score

        # Freshness bonus
        if is_fresh:
            quality_score = min(quality_score + 0.1, 1.0)

        # Relevance bonus (if article already has relevance_score)
        relevance = article.get("relevance_score", 0.0)
        if relevance > 0.7:
            quality_score = min(quality_score + 0.05, 1.0)

        return {
            "source": source,
            "tier": tier,
            "reliability_score": reliability_score,
            "is_fresh": is_fresh,
            "quality_score": quality_score,
            "validated": True,
        }

    def cross_validate_articles(self, articles: List[Dict]) -> Dict:
        """
        Cross-validate information across multiple news sources.

        Checks if key information (injuries, lineup changes) is confirmed
        by multiple reliable sources.

        Args:
            articles: List of article dicts

        Returns:
            Dict with cross-validation results
        """
        # Extract key information mentions
        injury_mentions = defaultdict(list)
        lineup_mentions = defaultdict(list)

        for article in articles:
            source = article.get("source", "")
            tier = self.get_source_tier(source)
            text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()

            # Injury detection
            if self._contains_injury_keywords(text):
                # Extract player names mentioned
                players = self._extract_player_names(text, article)
                for player in players:
                    injury_mentions[player].append({
                        "source": source,
                        "tier": tier,
                        "article": article
                    })

            # Lineup changes detection
            if self._contains_lineup_keywords(text):
                players = self._extract_player_names(text, article)
                for player in players:
                    lineup_mentions[player].append({
                        "source": source,
                        "tier": tier,
                        "article": article
                    })

        # Verify cross-validation
        confirmed_injuries = self._verify_mentions(injury_mentions)
        confirmed_lineup_changes = self._verify_mentions(lineup_mentions)

        return {
            "confirmed_injuries": confirmed_injuries,
            "confirmed_lineup_changes": confirmed_lineup_changes,
            "injury_mentions": dict(injury_mentions),
            "lineup_mentions": dict(lineup_mentions),
        }

    def _contains_injury_keywords(self, text: str) -> bool:
        """Check if text contains injury-related keywords"""
        keywords = [
            "injury", "injured", "hurt", "pain", "out", "doubtful",
            "questionable", "sidelined", "withdrawn", "withdrew",
            "kontuzja", "kontuzjowany", "uraz", "kontuzji"
        ]
        return any(keyword in text for keyword in keywords)

    def _contains_lineup_keywords(self, text: str) -> bool:
        """Check if text contains lineup change keywords"""
        keywords = [
            "lineup", "line-up", "starting", "bench", "dropped",
            "rotation", "squad", "team news", "skład", "składzie"
        ]
        return any(keyword in text for keyword in keywords)

    def _extract_player_names(self, text: str, article: Dict) -> Set[str]:
        """
        Extract player names from article.

        This is a simplified implementation. In production, you might want
        to use NER (Named Entity Recognition) or match against known player lists.

        Args:
            text: Article text
            article: Article dict (may contain metadata about players)

        Returns:
            Set of player names found
        """
        players = set()

        # Check if article metadata has player information
        if article.get("mentions_player1"):
            # Would need player names from context
            pass

        if article.get("mentions_player2"):
            pass

        # Simple capitalized words extraction (very basic)
        # In production: use NER or player database matching
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                # Check if next word is also capitalized (First Last name pattern)
                if i + 1 < len(words) and words[i + 1] and words[i + 1][0].isupper():
                    full_name = f"{word} {words[i + 1]}"
                    players.add(full_name)

        return players

    def _verify_mentions(self, mentions: Dict[str, List[Dict]]) -> List[str]:
        """
        Verify if information is confirmed by multiple sources.

        Criteria for confirmation:
        - At least 2 sources mention it
        - OR at least 1 Tier 1 source mentions it

        Args:
            mentions: Dict mapping entity (player) to list of mention dicts

        Returns:
            List of confirmed entities
        """
        confirmed = []

        for entity, mention_list in mentions.items():
            # Count sources and check tiers
            num_sources = len(mention_list)
            has_tier1 = any(m["tier"] == 1 for m in mention_list)
            has_multiple_tier2 = sum(1 for m in mention_list if m["tier"] == 2) >= 2

            # Confirmation logic
            if has_tier1:
                confirmed.append(entity)
            elif num_sources >= 2 and has_multiple_tier2:
                confirmed.append(entity)
            elif num_sources >= 3:  # Multiple sources confirm
                confirmed.append(entity)

        return confirmed

    def validate_match_news(
        self,
        articles: List[Dict],
        player1: str,
        player2: str
    ) -> Dict:
        """
        Validate all news articles for a match.

        Args:
            articles: List of article dicts
            player1: First player/team name
            player2: Second player/team name

        Returns:
            Dict with validation summary
        """
        if not articles:
            return {
                "valid": False,
                "reason": "No news articles found",
                "validated_articles": [],
                "cross_validation": {},
                "quality_score": 0.0,
            }

        # Validate each article
        validated_articles = []
        total_quality = 0.0
        fresh_count = 0
        tier1_count = 0
        tier2_count = 0

        for article in articles:
            validation = self.validate_article(article)
            validated_articles.append({**article, **validation})

            total_quality += validation["quality_score"]
            if validation["is_fresh"]:
                fresh_count += 1
            if validation["tier"] == 1:
                tier1_count += 1
            elif validation["tier"] == 2:
                tier2_count += 1

        # Cross-validate
        cross_validation = self.cross_validate_articles(articles)

        # Calculate overall news quality score
        avg_quality = total_quality / len(articles)

        # Bonus for multiple fresh articles
        if fresh_count >= thresholds.minimum_news_articles:
            avg_quality = min(avg_quality + 0.1, 1.0)

        # Bonus for tier 1 sources
        if tier1_count >= 1:
            avg_quality = min(avg_quality + thresholds.reliable_sources_bonus, 1.0)

        # Determine if news quality is sufficient
        valid = (
            len(articles) >= thresholds.minimum_news_articles
            and avg_quality >= 0.5
        )

        return {
            "valid": valid,
            "validated_articles": validated_articles,
            "cross_validation": cross_validation,
            "quality_score": avg_quality,
            "total_articles": len(articles),
            "fresh_articles": fresh_count,
            "tier1_sources": tier1_count,
            "tier2_sources": tier2_count,
            "avg_relevance": sum(a.get("relevance_score", 0) for a in articles) / len(articles),
        }


# === HELPER FUNCTIONS ===

def validate_news_quality(articles: List[Dict], player1: str, player2: str) -> Dict:
    """
    Convenience function to validate news articles for a match.

    Args:
        articles: List of article dicts
        player1: First player/team name
        player2: Second player/team name

    Returns:
        Dict with validation results
    """
    validator = NewsSourceValidator()
    return validator.validate_match_news(articles, player1, player2)


def get_confirmed_injuries(articles: List[Dict]) -> List[str]:
    """
    Get list of confirmed injuries from news articles.

    Args:
        articles: List of article dicts

    Returns:
        List of player names with confirmed injuries
    """
    validator = NewsSourceValidator()
    cross_validation = validator.cross_validate_articles(articles)
    return cross_validation.get("confirmed_injuries", [])
