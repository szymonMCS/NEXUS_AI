# core/quality_scorer.py
"""
Data quality scorer for NEXUS AI.
Evaluates completeness and reliability of collected data (news, odds, stats).
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.state import DataQualityLevel, Match, NewsArticle, MatchOdds
from config.thresholds import thresholds, LEAGUE_REQUIREMENTS


@dataclass
class QualityScoreComponents:
    """Individual components of quality score"""
    news_score: float = 0.0
    odds_score: float = 0.0
    stats_score: float = 0.0
    overall_score: float = 0.0
    quality_level: DataQualityLevel = DataQualityLevel.INSUFFICIENT
    issues: List[str] = None
    sources_count: int = 0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class QualityScorer:
    """
    Evaluates data quality for matches.

    Scoring criteria:
    - News quality (0-100): Number, freshness, source reliability
    - Odds quality (0-100): Number of bookmakers, variance
    - Stats quality (0-100): Completeness, historical data
    - Overall score: Weighted average
    """

    def __init__(self):
        self.weights = {
            "news": 0.3,
            "odds": 0.4,
            "stats": 0.3,
        }

    def score_news_quality(
        self,
        news_articles: List[NewsArticle],
        min_articles: int = None
    ) -> float:
        """
        Score news data quality (0.0 - 1.0).

        Args:
            news_articles: List of NewsArticle objects
            min_articles: Minimum required articles (defaults to threshold)

        Returns:
            Quality score 0.0 - 1.0
        """
        if min_articles is None:
            min_articles = thresholds.minimum_news_articles

        if not news_articles:
            return 0.0

        score = 0.0

        # Factor 1: Quantity (0-0.4)
        num_articles = len(news_articles)
        quantity_score = min(num_articles / (min_articles * 2), 1.0) * 0.4
        score += quantity_score

        # Factor 2: Freshness (0-0.3)
        fresh_articles = 0
        max_age = timedelta(hours=thresholds.news_freshness_hours)

        for article in news_articles:
            age = datetime.now() - article.published_date
            if age <= max_age:
                fresh_articles += 1

        freshness_score = min(fresh_articles / min_articles, 1.0) * 0.3
        score += freshness_score

        # Factor 3: Relevance (0-0.3)
        avg_relevance = sum(a.relevance_score for a in news_articles) / len(news_articles)
        relevance_score = avg_relevance * 0.3
        score += relevance_score

        return min(score, 1.0)

    def score_odds_quality(
        self,
        odds: List[MatchOdds],
        min_bookmakers: int = None
    ) -> float:
        """
        Score odds data quality (0.0 - 1.0).

        Args:
            odds: List of MatchOdds objects
            min_bookmakers: Minimum required bookmakers

        Returns:
            Quality score 0.0 - 1.0
        """
        if min_bookmakers is None:
            min_bookmakers = thresholds.odds_sources_required

        if not odds:
            return 0.0

        score = 0.0

        # Factor 1: Number of bookmakers (0-0.5)
        num_bookmakers = len(odds)
        bookmaker_score = min(num_bookmakers / (min_bookmakers * 2), 1.0) * 0.5
        score += bookmaker_score

        # Factor 2: Odds consistency (0-0.3)
        # Lower variance = higher quality
        home_odds_values = [o.home_odds for o in odds]
        away_odds_values = [o.away_odds for o in odds]

        if len(home_odds_values) > 1:
            home_variance = self._calculate_variance(home_odds_values)
            away_variance = self._calculate_variance(away_odds_values)
            avg_variance = (home_variance + away_variance) / 2

            # Lower variance is better
            max_allowed_variance = thresholds.max_odds_variance
            variance_score = max(1.0 - (avg_variance / max_allowed_variance), 0.0) * 0.3
            score += variance_score
        else:
            score += 0.15  # Partial credit for single source

        # Factor 3: Recency (0-0.2)
        recent_odds = 0
        max_age = timedelta(hours=1)

        for odd in odds:
            age = datetime.now() - odd.timestamp
            if age <= max_age:
                recent_odds += 1

        recency_score = (recent_odds / len(odds)) * 0.2
        score += recency_score

        return min(score, 1.0)

    def score_stats_quality(
        self,
        match: Match
    ) -> float:
        """
        Score player/team statistics quality (0.0 - 1.0).

        Args:
            match: Match object with player stats

        Returns:
            Quality score 0.0 - 1.0
        """
        score = 0.0

        # Check home player stats completeness
        home_player = match.home_player
        away_player = match.away_player

        # Factor 1: Rankings available (0-0.3)
        if home_player.ranking and away_player.ranking:
            score += 0.3
        elif home_player.ranking or away_player.ranking:
            score += 0.15

        # Factor 2: Form data available (0-0.3)
        if home_player.form and away_player.form:
            score += 0.3
        elif home_player.form or away_player.form:
            score += 0.15

        # Factor 3: Win rate available (0-0.2)
        if home_player.win_rate is not None and away_player.win_rate is not None:
            score += 0.2
        elif home_player.win_rate is not None or away_player.win_rate is not None:
            score += 0.1

        # Factor 4: Head-to-head data (0-0.2)
        total_h2h = home_player.h2h_wins + home_player.h2h_losses
        if total_h2h >= thresholds.min_historical_matches:
            score += 0.2
        elif total_h2h > 0:
            score += 0.1

        return min(score, 1.0)

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of odds values"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5  # Standard deviation

    def calculate_overall_quality(
        self,
        match: Match,
        league_type: str = "popular"
    ) -> QualityScoreComponents:
        """
        Calculate overall data quality for a match.

        Args:
            match: Match object with all collected data
            league_type: "popular", "medium", or "unpopular"

        Returns:
            QualityScoreComponents with detailed breakdown
        """
        # Get league-specific requirements
        requirements = LEAGUE_REQUIREMENTS.get(league_type, LEAGUE_REQUIREMENTS["popular"])

        # Score individual components
        news_score = self.score_news_quality(
            match.news_articles,
            requirements.min_news_articles
        )

        odds_score = self.score_odds_quality(
            match.odds,
            requirements.min_bookmakers
        )

        stats_score = self.score_stats_quality(match)

        # Calculate weighted overall score
        overall_score = (
            news_score * self.weights["news"] +
            odds_score * self.weights["odds"] +
            stats_score * self.weights["stats"]
        )

        # Determine quality level
        quality_level = self._get_quality_level(overall_score)

        # Identify issues
        issues = self._identify_issues(
            news_score, odds_score, stats_score,
            match, requirements
        )

        # Count unique data sources
        sources_count = self._count_data_sources(match)

        return QualityScoreComponents(
            news_score=news_score,
            odds_score=odds_score,
            stats_score=stats_score,
            overall_score=overall_score,
            quality_level=quality_level,
            issues=issues,
            sources_count=sources_count
        )

    def _get_quality_level(self, score: float) -> DataQualityLevel:
        """Map score to quality level"""
        if score >= thresholds.quality_excellent:
            return DataQualityLevel.EXCELLENT
        elif score >= thresholds.quality_good:
            return DataQualityLevel.GOOD
        elif score >= thresholds.quality_moderate:
            return DataQualityLevel.MODERATE
        elif score >= thresholds.quality_high_risk:
            return DataQualityLevel.HIGH_RISK
        else:
            return DataQualityLevel.INSUFFICIENT

    def _identify_issues(
        self,
        news_score: float,
        odds_score: float,
        stats_score: float,
        match: Match,
        requirements
    ) -> List[str]:
        """Identify specific data quality issues"""
        issues = []

        # News issues
        if len(match.news_articles) < requirements.min_news_articles:
            issues.append(f"Insufficient news coverage ({len(match.news_articles)} articles)")

        if news_score < 0.5:
            issues.append("Low news quality or relevance")

        # Odds issues
        if len(match.odds) < requirements.min_bookmakers:
            issues.append(f"Insufficient bookmaker coverage ({len(match.odds)} sources)")

        if odds_score < 0.5:
            issues.append("High odds variance or outdated odds")

        # Stats issues
        if not match.home_player.ranking or not match.away_player.ranking:
            issues.append("Missing player rankings")

        if stats_score < 0.5:
            issues.append("Incomplete player statistics")

        if not match.news_articles:
            issues.append("No news data available")

        if not match.odds:
            issues.append("No odds data available")

        return issues

    def _count_data_sources(self, match: Match) -> int:
        """Count number of unique data sources used"""
        sources = set()

        # News sources
        for article in match.news_articles:
            sources.add(f"news:{article.source}")

        # Odds sources
        for odds in match.odds:
            sources.add(f"odds:{odds.bookmaker}")

        # Stats sources (implicit from presence of data)
        if match.home_player.ranking or match.away_player.ranking:
            sources.add("stats:rankings")

        if match.home_player.form or match.away_player.form:
            sources.add("stats:form")

        return len(sources)

    def should_reject_match(
        self,
        quality: QualityScoreComponents,
        min_level: DataQualityLevel = DataQualityLevel.MODERATE
    ) -> bool:
        """
        Determine if match should be rejected based on quality.

        Args:
            quality: QualityScoreComponents
            min_level: Minimum acceptable quality level

        Returns:
            True if should be rejected
        """
        quality_order = ["excellent", "good", "moderate", "high_risk", "insufficient"]
        min_index = quality_order.index(min_level.value)
        current_index = quality_order.index(quality.quality_level.value)

        return current_index > min_index


# === HELPER FUNCTIONS ===

def score_match_quality(match: Match, league_type: str = "popular") -> QualityScoreComponents:
    """
    Convenience function to score match quality.

    Args:
        match: Match object
        league_type: League category

    Returns:
        QualityScoreComponents
    """
    scorer = QualityScorer()
    return scorer.calculate_overall_quality(match, league_type)


def filter_high_quality_matches(
    matches: List[Match],
    min_level: DataQualityLevel = DataQualityLevel.GOOD,
    league_type: str = "popular"
) -> List[Match]:
    """
    Filter matches by minimum quality level.

    Args:
        matches: List of matches
        min_level: Minimum quality level
        league_type: League category

    Returns:
        Filtered list of high-quality matches
    """
    scorer = QualityScorer()
    high_quality = []

    for match in matches:
        quality = scorer.calculate_overall_quality(match, league_type)
        if not scorer.should_reject_match(quality, min_level):
            # Add quality metrics to match
            match.data_quality = quality
            high_quality.append(match)

    return high_quality
