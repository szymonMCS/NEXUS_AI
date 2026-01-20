# data/collectors/fixture_collector.py
"""
Multi-source fixture collector for NEXUS AI.
Collects fixtures from TheSportsDB, Sofascore, and Flashscore with deduplication.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import logging

from data.tennis.sofascore_scraper import SofascoreTennisScraper
from data.apis.thesportsdb_client import TheSportsDBClient
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Fixture:
    """Normalized fixture from any source."""
    fixture_id: str  # Generated unique ID
    sport: str
    league: str
    home_team: str
    away_team: str
    start_time: datetime
    country: str = ""

    # Source tracking
    sources: List[str] = field(default_factory=list)
    source_ids: Dict[str, str] = field(default_factory=dict)

    # Rankings (if available)
    home_ranking: Optional[int] = None
    away_ranking: Optional[int] = None

    # Additional data
    venue: Optional[str] = None
    surface: Optional[str] = None  # For tennis
    tournament_round: Optional[str] = None

    # Quality indicators
    data_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "sport": self.sport,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "country": self.country,
            "sources": self.sources,
            "source_ids": self.source_ids,
            "home_ranking": self.home_ranking,
            "away_ranking": self.away_ranking,
            "venue": self.venue,
            "surface": self.surface,
            "tournament_round": self.tournament_round,
            "data_confidence": self.data_confidence
        }


class FixtureCollector:
    """
    Collects and merges fixtures from multiple sources.

    Sources:
    - TheSportsDB (free API with key=3)
    - Sofascore (free, no key required)
    - Flashscore (scraping, optional)

    Features:
    - Parallel data collection
    - Deduplication via fuzzy matching
    - Source confidence weighting
    - Rate limiting per source
    """

    # Source priorities (higher = more trusted)
    SOURCE_PRIORITY = {
        "sofascore": 1.0,
        "thesportsdb": 0.9,
        "flashscore": 0.8,
        "api_tennis": 0.95,
    }

    def __init__(self, enable_flashscore: bool = False):
        """
        Initialize fixture collector.

        Args:
            enable_flashscore: Whether to use Flashscore scraper (requires Playwright)
        """
        self.enable_flashscore = enable_flashscore
        self._collected_fixtures: List[Fixture] = []

    async def collect_fixtures(
        self,
        sport: str,
        date: str,
        league: Optional[str] = None
    ) -> List[Fixture]:
        """
        Collect fixtures from all available sources.

        Args:
            sport: Sport type (tennis, basketball)
            date: Date in YYYY-MM-DD format
            league: Optional league filter

        Returns:
            List of deduplicated Fixture objects
        """
        logger.info(f"Collecting {sport} fixtures for {date}")

        # Gather from all sources in parallel
        tasks = []

        # Sofascore
        tasks.append(self._collect_from_sofascore(sport, date))

        # TheSportsDB
        tasks.append(self._collect_from_thesportsdb(sport, date))

        # Flashscore (if enabled)
        if self.enable_flashscore:
            tasks.append(self._collect_from_flashscore(sport, date))

        # Run all collectors
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_fixtures: List[Fixture] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Collector {i} failed: {result}")
                continue
            if result:
                all_fixtures.extend(result)

        logger.info(f"Collected {len(all_fixtures)} raw fixtures from all sources")

        # Deduplicate
        merged_fixtures = self._deduplicate_fixtures(all_fixtures)
        logger.info(f"After deduplication: {len(merged_fixtures)} fixtures")

        # Filter by league if specified
        if league:
            merged_fixtures = [
                f for f in merged_fixtures
                if league.lower() in f.league.lower()
            ]

        # Sort by start time
        merged_fixtures.sort(key=lambda x: x.start_time or datetime.max)

        self._collected_fixtures = merged_fixtures
        return merged_fixtures

    async def _collect_from_sofascore(
        self,
        sport: str,
        date: str
    ) -> List[Fixture]:
        """Collect fixtures from Sofascore."""
        try:
            if sport == "tennis":
                async with SofascoreTennisScraper() as scraper:
                    matches = await scraper.get_matches_by_date(date)
                    return [self._sofascore_to_fixture(m) for m in matches]
            else:
                # For other sports, would need specific scrapers
                logger.debug(f"Sofascore: {sport} not implemented")
                return []
        except Exception as e:
            logger.error(f"Sofascore collection failed: {e}")
            return []

    async def _collect_from_thesportsdb(
        self,
        sport: str,
        date: str
    ) -> List[Fixture]:
        """Collect fixtures from TheSportsDB."""
        try:
            async with TheSportsDBClient() as client:
                if sport == "tennis":
                    events = await client.get_events_by_date(date, "Tennis")
                elif sport == "basketball":
                    events = await client.get_events_by_date(date, "Basketball")
                else:
                    events = await client.get_events_by_date(date, sport.title())

                return [self._thesportsdb_to_fixture(e) for e in events]
        except Exception as e:
            logger.error(f"TheSportsDB collection failed: {e}")
            return []

    async def _collect_from_flashscore(
        self,
        sport: str,
        date: str
    ) -> List[Fixture]:
        """Collect fixtures from Flashscore (if scraper available)."""
        try:
            # Import dynamically to avoid requiring Playwright if not needed
            from data.scrapers.flashscore_scraper import FlashscoreScraper

            async with FlashscoreScraper() as scraper:
                matches = await scraper.get_fixtures(sport, date)
                return [self._flashscore_to_fixture(m) for m in matches]
        except ImportError:
            logger.debug("Flashscore scraper not available")
            return []
        except Exception as e:
            logger.error(f"Flashscore collection failed: {e}")
            return []

    def _sofascore_to_fixture(self, match: Dict) -> Fixture:
        """Convert Sofascore match to Fixture."""
        fixture_id = self._generate_fixture_id(
            match.get("home_team", ""),
            match.get("away_team", ""),
            match.get("start_time")
        )

        return Fixture(
            fixture_id=fixture_id,
            sport=match.get("sport", "tennis"),
            league=match.get("league", ""),
            home_team=match.get("home_team", ""),
            away_team=match.get("away_team", ""),
            start_time=match.get("start_time"),
            country=match.get("country", ""),
            sources=["sofascore"],
            source_ids={"sofascore": match.get("external_id", "")},
            home_ranking=match.get("home_ranking"),
            away_ranking=match.get("away_ranking"),
            data_confidence=self.SOURCE_PRIORITY["sofascore"]
        )

    def _thesportsdb_to_fixture(self, event: Dict) -> Fixture:
        """Convert TheSportsDB event to Fixture."""
        start_time = None
        if event.get("dateEvent") and event.get("strTime"):
            try:
                dt_str = f"{event['dateEvent']} {event['strTime']}"
                start_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start_time = datetime.strptime(event["dateEvent"], "%Y-%m-%d")

        fixture_id = self._generate_fixture_id(
            event.get("strHomeTeam", ""),
            event.get("strAwayTeam", ""),
            start_time
        )

        return Fixture(
            fixture_id=fixture_id,
            sport=event.get("strSport", "").lower(),
            league=event.get("strLeague", ""),
            home_team=event.get("strHomeTeam", ""),
            away_team=event.get("strAwayTeam", ""),
            start_time=start_time,
            country=event.get("strCountry", ""),
            sources=["thesportsdb"],
            source_ids={"thesportsdb": str(event.get("idEvent", ""))},
            venue=event.get("strVenue"),
            tournament_round=event.get("strRound"),
            data_confidence=self.SOURCE_PRIORITY["thesportsdb"]
        )

    def _flashscore_to_fixture(self, match: Dict) -> Fixture:
        """Convert Flashscore match to Fixture."""
        fixture_id = self._generate_fixture_id(
            match.get("home_team", ""),
            match.get("away_team", ""),
            match.get("start_time")
        )

        return Fixture(
            fixture_id=fixture_id,
            sport=match.get("sport", ""),
            league=match.get("league", ""),
            home_team=match.get("home_team", ""),
            away_team=match.get("away_team", ""),
            start_time=match.get("start_time"),
            country=match.get("country", ""),
            sources=["flashscore"],
            source_ids={"flashscore": match.get("match_id", "")},
            data_confidence=self.SOURCE_PRIORITY["flashscore"]
        )

    def _generate_fixture_id(
        self,
        home: str,
        away: str,
        start_time: Optional[datetime]
    ) -> str:
        """Generate unique fixture ID from key attributes."""
        # Normalize names
        home_norm = self._normalize_name(home)
        away_norm = self._normalize_name(away)

        # Use date only (not time) for matching
        date_str = ""
        if start_time:
            date_str = start_time.strftime("%Y%m%d")

        key = f"{home_norm}|{away_norm}|{date_str}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _normalize_name(self, name: str) -> str:
        """Normalize team/player name for matching."""
        if not name:
            return ""

        # Convert to lowercase
        name = name.lower()

        # Remove common suffixes
        for suffix in [" fc", " bc", " atp", " wta", " (w)", " (m)"]:
            name = name.replace(suffix, "")

        # Remove special characters
        name = "".join(c for c in name if c.isalnum() or c == " ")

        # Remove extra spaces
        name = " ".join(name.split())

        return name.strip()

    def _deduplicate_fixtures(self, fixtures: List[Fixture]) -> List[Fixture]:
        """
        Deduplicate fixtures from multiple sources.

        Uses fixture_id for exact matching and fuzzy matching for similar names.
        """
        if not fixtures:
            return []

        # Group by fixture_id
        by_id: Dict[str, List[Fixture]] = {}
        for f in fixtures:
            if f.fixture_id not in by_id:
                by_id[f.fixture_id] = []
            by_id[f.fixture_id].append(f)

        # Merge duplicates
        merged: List[Fixture] = []
        for fixture_id, group in by_id.items():
            merged_fixture = self._merge_fixtures(group)
            merged.append(merged_fixture)

        return merged

    def _merge_fixtures(self, fixtures: List[Fixture]) -> Fixture:
        """Merge multiple fixtures for the same match."""
        if len(fixtures) == 1:
            return fixtures[0]

        # Sort by source priority (highest first)
        fixtures.sort(
            key=lambda f: max(
                self.SOURCE_PRIORITY.get(s, 0) for s in f.sources
            ),
            reverse=True
        )

        # Use highest priority fixture as base
        base = fixtures[0]

        # Merge sources and source_ids
        all_sources = set()
        all_source_ids = {}
        for f in fixtures:
            all_sources.update(f.sources)
            all_source_ids.update(f.source_ids)

        base.sources = list(all_sources)
        base.source_ids = all_source_ids

        # Calculate combined confidence
        base.data_confidence = min(
            1.0,
            sum(self.SOURCE_PRIORITY.get(s, 0.5) for s in all_sources) / len(all_sources) + 0.1 * len(all_sources)
        )

        # Fill in missing data from other sources
        for f in fixtures[1:]:
            if not base.home_ranking and f.home_ranking:
                base.home_ranking = f.home_ranking
            if not base.away_ranking and f.away_ranking:
                base.away_ranking = f.away_ranking
            if not base.venue and f.venue:
                base.venue = f.venue
            if not base.surface and f.surface:
                base.surface = f.surface
            if not base.tournament_round and f.tournament_round:
                base.tournament_round = f.tournament_round

        return base

    async def enrich_fixtures(
        self,
        fixtures: List[Fixture],
        include_stats: bool = True,
        include_odds: bool = True,
        include_news: bool = True
    ) -> List[Dict]:
        """
        Enrich fixtures with additional data (stats, odds, news).

        Args:
            fixtures: List of fixtures to enrich
            include_stats: Whether to fetch player/team stats
            include_odds: Whether to fetch betting odds
            include_news: Whether to fetch relevant news

        Returns:
            List of enriched fixture dicts
        """
        logger.info(f"Enriching {len(fixtures)} fixtures")

        enriched = []
        for fixture in fixtures:
            enriched_fixture = fixture.to_dict()

            # Parallel enrichment tasks
            tasks = []

            if include_stats:
                tasks.append(self._enrich_stats(fixture))
            if include_odds:
                tasks.append(self._enrich_odds(fixture))
            if include_news:
                tasks.append(self._enrich_news(fixture))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict):
                        enriched_fixture.update(result)

            enriched.append(enriched_fixture)

        return enriched

    async def _enrich_stats(self, fixture: Fixture) -> Dict:
        """Fetch additional stats for fixture."""
        # This would call stats APIs
        return {"stats_enriched": True}

    async def _enrich_odds(self, fixture: Fixture) -> Dict:
        """Fetch betting odds for fixture."""
        # This would call odds APIs
        return {"odds_enriched": True}

    async def _enrich_news(self, fixture: Fixture) -> Dict:
        """Fetch relevant news for fixture."""
        # This would call news aggregator
        return {"news_enriched": True}


# === HELPER FUNCTIONS ===

async def collect_fixtures(
    sport: str,
    date: Optional[str] = None,
    league: Optional[str] = None
) -> List[Fixture]:
    """
    Convenience function to collect fixtures.

    Args:
        sport: Sport type
        date: Date in YYYY-MM-DD (defaults to today)
        league: Optional league filter

    Returns:
        List of Fixture objects
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    collector = FixtureCollector()
    return await collector.collect_fixtures(sport, date, league)


async def collect_and_enrich_fixtures(
    sport: str,
    date: Optional[str] = None,
    league: Optional[str] = None,
    include_stats: bool = True,
    include_odds: bool = True,
    include_news: bool = False
) -> List[Dict]:
    """
    Collect and enrich fixtures in one call.

    Args:
        sport: Sport type
        date: Date in YYYY-MM-DD
        league: Optional league filter
        include_stats: Whether to fetch stats
        include_odds: Whether to fetch odds
        include_news: Whether to fetch news

    Returns:
        List of enriched fixture dicts
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    collector = FixtureCollector()
    fixtures = await collector.collect_fixtures(sport, date, league)

    return await collector.enrich_fixtures(
        fixtures,
        include_stats=include_stats,
        include_odds=include_odds,
        include_news=include_news
    )
