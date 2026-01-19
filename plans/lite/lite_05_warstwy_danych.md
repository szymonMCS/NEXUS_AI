## 5. IMPLEMENTACJA - WARSTWY DANYCH

### 5.1 Fixture Collector - Zbieranie meczów

```python
# data/collectors/fixture_collector.py

import asyncio
from typing import List, Dict
from datetime import datetime

from data.apis.thesportsdb_client import TheSportsDBClient
from data.scrapers.sofascore_scraper import SofascoreScraper
from data.scrapers.flashscore_scraper import FlashscoreScraper

class FixtureCollector:
    """
    Zbiera fixtures z wielu źródeł i łączy je.
    """

    def __init__(self):
        self.thesportsdb = TheSportsDBClient()
        self.sofascore = SofascoreScraper()
        self.flashscore = FlashscoreScraper()

    async def collect_fixtures(
        self,
        sport: str,
        date: str
    ) -> List[Dict]:
        """
        Zbierz wszystkie mecze na dany dzień.

        Args:
            sport: "tennis" lub "basketball"
            date: Format YYYY-MM-DD

        Returns:
            Lista meczów ze wszystkich źródeł (merged)
        """
        print(f"📅 Collecting {sport} fixtures for {date}...")

        # Parallel fetch from all sources
        tasks = [
            self._fetch_thesportsdb(sport, date),
            self._fetch_sofascore(sport, date),
            self._fetch_flashscore(sport, date)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_fixtures = []
        sources_used = []

        for source_name, result in zip(
            ["thesportsdb", "sofascore", "flashscore"],
            results
        ):
            if isinstance(result, Exception):
                print(f"  ⚠️ {source_name} failed: {result}")
                continue

            if result:
                all_fixtures.extend(result)
                sources_used.append(source_name)
                print(f"  ✅ {source_name}: {len(result)} matches")

        # Merge duplicates
        merged = self._merge_fixtures(all_fixtures)

        print(f"📊 Total unique matches: {len(merged)} from {sources_used}")

        return merged

    async def _fetch_thesportsdb(self, sport: str, date: str) -> List[Dict]:
        """Pobierz z TheSportsDB"""
        try:
            events = await self.thesportsdb.get_events_for_date(sport, date)
            return [
                {
                    "home": e.get("strHomeTeam", ""),
                    "away": e.get("strAwayTeam", ""),
                    "league": e.get("strLeague", ""),
                    "time": e.get("strTime", ""),
                    "date": e.get("dateEvent", ""),
                    "source": "thesportsdb",
                    "id_thesportsdb": e.get("idEvent")
                }
                for e in events
            ]
        except Exception as e:
            raise Exception(f"TheSportsDB: {e}")

    async def _fetch_sofascore(self, sport: str, date: str) -> List[Dict]:
        """Pobierz z Sofascore"""
        try:
            events = await self.sofascore.get_events_for_date(date, sport)
            return [
                {
                    "home": e.get("homeTeam", {}).get("name", ""),
                    "away": e.get("awayTeam", {}).get("name", ""),
                    "league": e.get("tournament", {}).get("name", ""),
                    "time": e.get("startTimestamp", ""),
                    "source": "sofascore",
                    "id_sofascore": e.get("id")
                }
                for e in events
            ]
        except Exception as e:
            raise Exception(f"Sofascore: {e}")

    async def _fetch_flashscore(self, sport: str, date: str) -> List[Dict]:
        """Pobierz z Flashscore"""
        try:
            if sport == "tennis":
                events = await self.flashscore.get_tennis_matches(date.replace("-", ""))
            else:
                events = await self.flashscore.get_basketball_matches(date.replace("-", ""))

            return events
        except Exception as e:
            raise Exception(f"Flashscore: {e}")

    def _merge_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """
        Łączy duplikaty meczów z różnych źródeł.

        Mecze są dopasowywane po nazwach zawodników/drużyn.
        """
        merged = {}

        for fix in fixtures:
            # Utwórz klucz do matchowania
            key = self._create_match_key(fix)

            if key not in merged:
                merged[key] = {
                    "home": fix["home"],
                    "away": fix["away"],
                    "league": fix.get("league", "Unknown"),
                    "time": fix.get("time", ""),
                    "date": fix.get("date", ""),
                    "sources": [fix["source"]],
                    "ids": {fix["source"]: fix.get(f"id_{fix['source']}")}
                }
            else:
                # Dodaj źródło
                merged[key]["sources"].append(fix["source"])
                if f"id_{fix['source']}" in fix:
                    merged[key]["ids"][fix["source"]] = fix.get(f"id_{fix['source']}")

        return list(merged.values())

    def _create_match_key(self, fixture: Dict) -> str:
        """Tworzy unikalny klucz dla meczu"""
        home = self._normalize_name(fixture.get("home", ""))
        away = self._normalize_name(fixture.get("away", ""))

        # Sortuj alfabetycznie dla spójności
        names = sorted([home, away])
        return f"{names[0]}::{names[1]}"

    def _normalize_name(self, name: str) -> str:
        """Normalizuje nazwę zawodnika/drużyny"""
        import re
        # Usuń znaki specjalne, zamień na lowercase
        name = re.sub(r'[^\w\s]', '', name.lower())
        # Usuń podwójne spacje
        name = ' '.join(name.split())
        return name
```

### 5.2 Data Enricher - Wzbogacanie danych

```python
# data/collectors/data_enricher.py

import asyncio
from typing import Dict, List

from data.scrapers.sofascore_scraper import SofascoreScraper
from data.scrapers.pl_bookies_scraper import PLBookiesScraper
from data.news.web_search import WebNewsSearch
from data.news.injury_extractor import InjuryExtractor

class DataEnricher:
    """
    Wzbogaca dane o meczu o statystyki, kursy i newsy.
    """

    def __init__(self):
        self.sofascore = SofascoreScraper()
        self.pl_bookies = PLBookiesScraper()
        self.news_search = WebNewsSearch()
        self.injury_extractor = InjuryExtractor()

    async def enrich_match(self, match: Dict, sport: str) -> Dict:
        """
        Wzbogaca pojedynczy mecz o dodatkowe dane.

        Równolegle pobiera:
        - Statystyki zawodników/drużyn
        - Kursy od bukmacherów
        - Newsy i informacje o kontuzjach
        """
        home = match["home"]
        away = match["away"]

        print(f"  🔍 Enriching: {home} vs {away}")

        # Parallel enrichment
        tasks = [
            self._get_statistics(match, sport),
            self._get_odds(match, sport),
            self._get_news(home, away, sport)
        ]

        stats, odds, news = await asyncio.gather(*tasks, return_exceptions=True)

        # Add to match
        match["stats"] = stats if not isinstance(stats, Exception) else {}
        match["odds"] = odds if not isinstance(odds, Exception) else {}
        match["news"] = news if not isinstance(news, Exception) else {}

        # Extract injuries from news
        if match["news"].get("combined"):
            injuries = await self.injury_extractor.extract(match["news"]["combined"])
            match["injuries"] = injuries
        else:
            match["injuries"] = []

        return match

    async def enrich_all(
        self,
        matches: List[Dict],
        sport: str,
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Wzbogaca wszystkie mecze z limitem równoległości.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def enrich_with_limit(match):
            async with semaphore:
                return await self.enrich_match(match, sport)

        enriched = await asyncio.gather(*[
            enrich_with_limit(m) for m in matches
        ])

        return enriched

    async def _get_statistics(self, match: Dict, sport: str) -> Dict:
        """Pobierz statystyki z Sofascore"""
        stats = {}

        sofascore_id = match.get("ids", {}).get("sofascore")

        if sport == "tennis":
            # Dla tenisa - statystyki zawodników
            # TODO: Implementacja pobierania statystyk per player
            pass
        else:
            # Dla koszykówki - statystyki drużyn
            pass

        return stats

    async def _get_odds(self, match: Dict, sport: str) -> Dict:
        """Pobierz kursy z polskich bukmacherów"""
        try:
            all_odds = await self.pl_bookies.scrape_all(sport)

            # Znajdź kursy dla tego meczu
            match_odds = {}
            home_norm = self._normalize(match["home"])
            away_norm = self._normalize(match["away"])

            for bookie, bookie_matches in all_odds.items():
                for bm in bookie_matches:
                    bm_home = self._normalize(bm.get("home", ""))
                    bm_away = self._normalize(bm.get("away", ""))

                    # Sprawdź dopasowanie
                    if self._names_match(home_norm, bm_home) and \
                       self._names_match(away_norm, bm_away):
                        match_odds[bookie] = bm["odds"]
                        break

            return match_odds
        except Exception as e:
            print(f"    ⚠️ Odds error: {e}")
            return {}

    async def _get_news(self, home: str, away: str, sport: str) -> Dict:
        """Wyszukaj newsy o meczu"""
        try:
            return await self.news_search.search_match_news(home, away, sport)
        except Exception as e:
            print(f"    ⚠️ News error: {e}")
            return {}

    def _normalize(self, name: str) -> str:
        """Normalizuj nazwę"""
        import re
        return re.sub(r'[^\w\s]', '', name.lower()).strip()

    def _names_match(self, name1: str, name2: str) -> bool:
        """Sprawdź czy nazwy pasują (fuzzy)"""
        # Prosty check - czy jedno zawiera się w drugim
        return name1 in name2 or name2 in name1 or \
               self._levenshtein_ratio(name1, name2) > 0.8

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Oblicz podobieństwo Levenshteina"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
```
