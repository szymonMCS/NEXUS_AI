# agents/news_analyst.py
"""
News Analyst Agent - Collects and analyzes news for upcoming matches.
"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.state import (
    NexusState, Match, NewsArticle, PlayerStats,
    Sport, add_message
)
from data.news.aggregator import NewsAggregator
from data.news.validator import NewsSourceValidator
from data.news.injury_extractor import InjuryExtractor


class NewsAnalystAgent:
    """
    News Analyst collects and processes news for each match.

    Responsibilities:
    - Fetch matches for the target date
    - Collect news from multiple sources
    - Validate news quality and relevance
    - Extract injury information
    - Update match objects with news data
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1
        )
        self.aggregator = NewsAggregator()
        self.validator = NewsSourceValidator()
        self.injury_extractor = InjuryExtractor()

    async def process(self, state: NexusState) -> NexusState:
        """
        Process news for all matches.

        Args:
            state: Current workflow state

        Returns:
            Updated state with news data
        """
        state.current_agent = "news_analyst"
        state = add_message(state, "news_analyst", f"Starting news collection for {state.sport.value} on {state.date}")

        # Get matches for the date
        matches = await self._fetch_matches(state.sport, state.date)

        if not matches:
            state = add_message(state, "news_analyst", "No matches found for the specified date")
            return state

        state = add_message(state, "news_analyst", f"Found {len(matches)} matches, collecting news...")

        # Process each match
        processed_matches = []
        for match in matches:
            processed_match = await self._process_match_news(match)
            processed_matches.append(processed_match)

        state.matches = processed_matches

        # Summary
        total_articles = sum(len(m.news_articles) for m in processed_matches)
        state = add_message(
            state,
            "news_analyst",
            f"Collected {total_articles} articles for {len(processed_matches)} matches"
        )

        return state

    async def _fetch_matches(self, sport: Sport, date: str) -> List[Match]:
        """
        Fetch matches for the given sport and date.

        Args:
            sport: Sport type
            date: Date string (YYYY-MM-DD)

        Returns:
            List of Match objects
        """
        matches = []

        if sport == Sport.TENNIS:
            from data.tennis import get_upcoming_tennis_matches, scrape_upcoming_tennis_matches

            if settings.is_pro_mode and settings.API_TENNIS_KEY:
                raw_matches = await get_upcoming_tennis_matches(date)
            else:
                raw_matches = await scrape_upcoming_tennis_matches(date)

        elif sport == Sport.BASKETBALL:
            from data.basketball import get_upcoming_basketball_matches, scrape_upcoming_basketball_matches

            if settings.is_pro_mode and settings.BETS_API_KEY:
                raw_matches = await get_upcoming_basketball_matches("nba", date)
            else:
                raw_matches = await scrape_upcoming_basketball_matches("nba", date)
        else:
            raw_matches = []

        # Convert to Match objects
        for raw in raw_matches:
            match = self._convert_to_match(raw, sport)
            matches.append(match)

        return matches

    def _convert_to_match(self, raw: Dict, sport: Sport) -> Match:
        """Convert raw match data to Match object."""
        home_player = PlayerStats(
            name=raw.get("home_team", "Unknown"),
            ranking=raw.get("home_ranking"),
        )

        away_player = PlayerStats(
            name=raw.get("away_team", "Unknown"),
            ranking=raw.get("away_ranking"),
        )

        return Match(
            match_id=raw.get("external_id", f"{home_player.name}_{away_player.name}"),
            sport=sport,
            date=raw.get("start_time") or datetime.now(),
            league=raw.get("league", "Unknown"),
            home_player=home_player,
            away_player=away_player
        )

    async def _process_match_news(self, match: Match) -> Match:
        """
        Collect and process news for a single match.

        Args:
            match: Match object

        Returns:
            Match with news data populated
        """
        player1 = match.home_player.name
        player2 = match.away_player.name
        sport = match.sport.value

        # Fetch news articles
        articles = await self.aggregator.search_match_news(
            player1=player1,
            player2=player2,
            sport=sport,
            max_results=20
        )

        # Convert to NewsArticle objects
        news_articles = []
        for article in articles:
            news_article = NewsArticle(
                title=article.get("title", ""),
                url=article.get("url", ""),
                source=article.get("source", "unknown"),
                published_date=article.get("published_date") or datetime.now(),
                snippet=article.get("snippet", ""),
                relevance_score=article.get("relevance_score", 0.5),
                mentions_player1=player1.lower() in article.get("title", "").lower(),
                mentions_player2=player2.lower() in article.get("title", "").lower(),
                mentions_injury=self._check_injury_keywords(article)
            )
            news_articles.append(news_article)

        match.news_articles = news_articles

        # Extract injury information
        injuries = self.injury_extractor.extract_injuries_from_articles(
            [{"title": a.title, "snippet": a.snippet, "source": a.source} for a in news_articles],
            player1,
            player2
        )

        # Update player injury status
        if player1 in injuries and injuries[player1]:
            consolidated = self.injury_extractor.consolidate_injury_reports(injuries[player1])
            if consolidated:
                match.home_player.injury_status = consolidated.status

        if player2 in injuries and injuries[player2]:
            consolidated = self.injury_extractor.consolidate_injury_reports(injuries[player2])
            if consolidated:
                match.away_player.injury_status = consolidated.status

        return match

    def _check_injury_keywords(self, article: Dict) -> bool:
        """Check if article mentions injuries."""
        text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
        keywords = ["injury", "injured", "hurt", "out", "doubtful", "withdrawn"]
        return any(keyword in text for keyword in keywords)


# === HELPER FUNCTIONS ===

async def collect_match_news(sport: str, date: str) -> List[Match]:
    """
    Convenience function to collect news for matches.

    Args:
        sport: Sport type
        date: Date string

    Returns:
        List of matches with news
    """
    agent = NewsAnalystAgent()

    state = NexusState(
        sport=Sport(sport),
        date=date
    )

    result_state = await agent.process(state)
    return result_state.matches
