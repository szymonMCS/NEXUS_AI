## 3. DARMOWE ŹRÓDŁA DANYCH

### 3.1 Podsumowanie Źródeł

| Źródło | Typ | Koszt | Limit | Dane |
|--------|-----|-------|-------|------|
| **TheSportsDB** | API | FREE | Bez limitu* | Fixtures, teams, leagues |
| **API-Sports** | API | FREE | 100 req/dzień/API | Fixtures, stats, odds |
| **AllSportsAPI** | API | FREE | 260 req/godz | Tennis, basketball |
| **Sofascore** | Scraping | FREE | Be gentle | Stats, H2H, rankings |
| **Flashscore** | Scraping | FREE | Be gentle | Fixtures, odds, results |
| **Brave Search** | API | FREE | 2000 req/mies | News, injuries |
| **Serper** | API | FREE | 2500 req/mies | Google search results |
| **OddsPortal** | Scraping | FREE | Be gentle | Historical odds |
| **Fortuna/STS/Betclic** | Scraping | FREE | Be gentle | Polish odds |

*TheSportsDB: key="123" dla darmowego dostępu

### 3.2 TheSportsDB - Darmowe API

```python
# config/free_apis.py

THESPORTSDB_CONFIG = {
    "base_url": "https://www.thesportsdb.com/api/v1/json",
    "api_key": "3",  # Darmowy klucz (lub "123")
    "endpoints": {
        "all_leagues": "/all_leagues.php",
        "league_events": "/eventsseason.php",  # ?id={league_id}&s={season}
        "next_events": "/eventsnextleague.php",  # ?id={league_id}
        "team_details": "/lookupteam.php",  # ?id={team_id}
        "player_details": "/lookupplayer.php",  # ?id={player_id}
        "search_team": "/searchteams.php",  # ?t={team_name}
        "search_player": "/searchplayers.php",  # ?p={player_name}
    },
    # Dostępne ligi tenisa i koszykówki
    "tennis_leagues": {
        "4464": "ATP",
        "4465": "WTA",
    },
    "basketball_leagues": {
        "4387": "NBA",
        "4424": "EuroLeague",
        "4710": "PLK",  # Polska Liga Koszykówki
    }
}
```

### 3.3 API-Sports Free Tier

```python
# API-Sports: 100 requests/day per API (basketball, tennis osobno)

API_SPORTS_CONFIG = {
    "basketball": {
        "base_url": "https://v1.basketball.api-sports.io",
        "headers": {"x-apisports-key": "YOUR_FREE_KEY"},
        "daily_limit": 100,
        "endpoints": {
            "games": "/games",
            "standings": "/standings",
            "teams": "/teams",
            "odds": "/odds",
        }
    },
    "tennis": {
        "base_url": "https://v1.tennis.api-sports.io",
        "headers": {"x-apisports-key": "YOUR_FREE_KEY"},
        "daily_limit": 100,
        "endpoints": {
            "games": "/games",
            "rankings": "/rankings",
            "h2h": "/h2h",
        }
    }
}
```

### 3.4 Sofascore Scraping

```python
# data/scrapers/sofascore_scraper.py

import httpx
from typing import Dict, List, Optional

class SofascoreScraper:
    """
    Scraper dla Sofascore - statystyki, H2H, rankingi.

    UWAGA: Sofascore ma internal API które można użyć
    zamiast parsowania HTML.
    """

    BASE_URL = "https://api.sofascore.com/api/v1"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sofascore.com/",
    }

    async def get_events_for_date(self, date: str, sport: str = "tennis") -> List[Dict]:
        """
        Pobierz mecze na dany dzień.

        Args:
            date: Format YYYY-MM-DD
            sport: "tennis" lub "basketball"
        """
        sport_id = {"tennis": 13, "basketball": 2}[sport]

        url = f"{self.BASE_URL}/sport/{sport}/scheduled-events/{date}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.HEADERS)
            data = response.json()

            return data.get("events", [])

    async def get_match_statistics(self, match_id: int) -> Dict:
        """Pobierz statystyki meczu"""
        url = f"{self.BASE_URL}/event/{match_id}/statistics"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.HEADERS)
            return response.json()

    async def get_h2h(self, team1_id: int, team2_id: int) -> Dict:
        """Pobierz historię H2H"""
        url = f"{self.BASE_URL}/event/{team1_id}/h2h/events"
        params = {"teamId": team2_id}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.HEADERS, params=params)
            return response.json()

    async def get_player_statistics(self, player_id: int) -> Dict:
        """Pobierz statystyki zawodnika (tenis)"""
        url = f"{self.BASE_URL}/player/{player_id}/statistics"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.HEADERS)
            return response.json()

    async def get_tennis_rankings(self, ranking_type: str = "atp") -> List[Dict]:
        """
        Pobierz ranking tenisowy.

        Args:
            ranking_type: "atp" lub "wta"
        """
        ranking_id = {"atp": 3, "wta": 4}[ranking_type]
        url = f"{self.BASE_URL}/rankings/type/{ranking_id}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.HEADERS)
            data = response.json()
            return data.get("rankings", [])
```

### 3.5 Flashscore Scraping

```python
# data/scrapers/flashscore_scraper.py

from playwright.async_api import async_playwright
from typing import Dict, List
import re

class FlashscoreScraper:
    """
    Scraper dla Flashscore - fixtures, odds, wyniki.

    Używa Playwright ze względu na dynamiczny JS.
    """

    BASE_URL = "https://www.flashscore.pl"

    async def get_tennis_matches(self, date: str = None) -> List[Dict]:
        """
        Pobierz mecze tenisowe z Flashscore.

        Args:
            date: None = dziś, lub format YYYYMMDD
        """
        url = f"{self.BASE_URL}/tenis/"
        if date:
            url += f"?d={date}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(url)
            await page.wait_for_selector(".event__match", timeout=10000)

            matches = []
            match_elements = await page.query_selector_all(".event__match")

            for elem in match_elements:
                try:
                    match = await self._parse_match_element(elem)
                    if match:
                        matches.append(match)
                except Exception as e:
                    continue

            await browser.close()
            return matches

    async def _parse_match_element(self, elem) -> Dict:
        """Parsuj pojedynczy element meczu"""
        home = await elem.query_selector(".event__participant--home")
        away = await elem.query_selector(".event__participant--away")
        time_elem = await elem.query_selector(".event__time")

        home_name = await home.inner_text() if home else ""
        away_name = await away.inner_text() if away else ""
        match_time = await time_elem.inner_text() if time_elem else ""

        # Pobierz link do szczegółów
        link = await elem.get_attribute("id")
        match_id = link.replace("g_1_", "") if link else ""

        return {
            "home": home_name.strip(),
            "away": away_name.strip(),
            "time": match_time.strip(),
            "flashscore_id": match_id,
            "source": "flashscore"
        }

    async def get_match_odds(self, match_id: str) -> Dict:
        """Pobierz kursy dla meczu"""
        url = f"{self.BASE_URL}/mecz/{match_id}/#/zestawienie-kursow/kursy-1x2/koniec-meczu"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(url)
            await page.wait_for_selector(".ui-table", timeout=10000)

            odds_data = {}
            rows = await page.query_selector_all(".ui-table__row")

            for row in rows:
                try:
                    bookie = await row.query_selector(".oddsCell__bookmaker")
                    odds_cells = await row.query_selector_all(".oddsCell__odd")

                    if bookie and len(odds_cells) >= 2:
                        bookie_name = await bookie.inner_text()
                        home_odd = await odds_cells[0].inner_text()
                        away_odd = await odds_cells[1].inner_text()

                        odds_data[bookie_name.strip()] = {
                            "home": float(home_odd.replace(",", ".")),
                            "away": float(away_odd.replace(",", "."))
                        }
                except:
                    continue

            await browser.close()
            return odds_data
```

### 3.6 Polish Bookmakers Scraper

```python
# data/scrapers/pl_bookies_scraper.py

from playwright.async_api import async_playwright
from typing import Dict, List
import asyncio

class PLBookiesScraper:
    """
    Scraper dla polskich bukmacherów: Fortuna, STS, Betclic.
    """

    BOOKIES = {
        "fortuna": {
            "base_url": "https://www.efortuna.pl",
            "tennis": "/zaklady-bukmacherskie/tenis",
            "basketball": "/zaklady-bukmacherskie/koszykowka"
        },
        "sts": {
            "base_url": "https://www.sts.pl",
            "tennis": "/pl/zaklady-bukmacherskie/tenis",
            "basketball": "/pl/zaklady-bukmacherskie/koszykowka"
        },
        "betclic": {
            "base_url": "https://www.betclic.pl",
            "tennis": "/sport/tenis",
            "basketball": "/sport/koszykowka"
        }
    }

    async def scrape_all(self, sport: str) -> Dict[str, List[Dict]]:
        """
        Scrapuj wszystkich polskich bukmacherów równolegle.
        """
        tasks = [
            self._scrape_bookie(bookie, sport)
            for bookie in self.BOOKIES.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            bookie: result
            for bookie, result in zip(self.BOOKIES.keys(), results)
            if not isinstance(result, Exception)
        }

    async def _scrape_bookie(self, bookie: str, sport: str) -> List[Dict]:
        """Scrapuj pojedynczego bukmachera"""
        config = self.BOOKIES[bookie]
        url = config["base_url"] + config[sport]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=30000)
                await asyncio.sleep(3)  # Daj czas na załadowanie

                matches = await self._parse_bookie_page(page, bookie)
                return matches
            except Exception as e:
                print(f"Error scraping {bookie}: {e}")
                return []
            finally:
                await browser.close()

    async def _parse_bookie_page(self, page, bookie: str) -> List[Dict]:
        """
        Parsuj stronę bukmachera.
        Selektory różnią się między bukmacherami.
        """
        matches = []

        # Selektory per bookie (uproszczone - w rzeczywistości bardziej złożone)
        selectors = {
            "fortuna": {
                "match_row": "[data-testid='event-row']",
                "teams": ".event-name",
                "odds": ".odds-value"
            },
            "sts": {
                "match_row": ".match-row",
                "teams": ".participant-name",
                "odds": ".odds-button__value"
            },
            "betclic": {
                "match_row": ".match",
                "teams": ".scoreboard_contestantLabel",
                "odds": ".oddValue"
            }
        }

        sel = selectors.get(bookie, selectors["fortuna"])

        rows = await page.query_selector_all(sel["match_row"])

        for row in rows[:20]:  # Limit do 20 meczów
            try:
                teams = await row.query_selector_all(sel["teams"])
                odds = await row.query_selector_all(sel["odds"])

                if len(teams) >= 2 and len(odds) >= 2:
                    home = await teams[0].inner_text()
                    away = await teams[1].inner_text()
                    home_odd = await odds[0].inner_text()
                    away_odd = await odds[1].inner_text()

                    matches.append({
                        "home": home.strip(),
                        "away": away.strip(),
                        "odds": {
                            "home": self._parse_odd(home_odd),
                            "away": self._parse_odd(away_odd)
                        },
                        "bookmaker": bookie
                    })
            except:
                continue

        return matches

    def _parse_odd(self, odd_str: str) -> float:
        """Parsuj kurs do float"""
        try:
            return float(odd_str.replace(",", ".").strip())
        except:
            return 0.0
```

### 3.7 News Search (Brave + Serper)

```python
# data/news/web_search.py

import httpx
from typing import List, Dict
import os

class WebNewsSearch:
    """
    Wyszukiwanie newsów przez Brave Search i Serper.
    """

    def __init__(self):
        self.brave_key = os.getenv("BRAVE_API_KEY")
        self.serper_key = os.getenv("SERPER_API_KEY")

    async def search_match_news(
        self,
        player1: str,
        player2: str,
        sport: str
    ) -> Dict:
        """
        Wyszukaj newsy o meczu z obu źródeł.
        """
        results = {
            "brave": [],
            "serper": [],
            "combined": []
        }

        queries = [
            f"{player1} vs {player2} {sport}",
            f"{player1} injury news",
            f"{player2} injury news",
            f"{player1} {player2} prediction"
        ]

        # Brave Search
        if self.brave_key:
            for query in queries[:2]:  # Limit queries
                brave_results = await self._search_brave(query)
                results["brave"].extend(brave_results)

        # Serper
        if self.serper_key:
            for query in queries[:2]:
                serper_results = await self._search_serper(query)
                results["serper"].extend(serper_results)

        # Combine and dedupe
        results["combined"] = self._combine_and_dedupe(
            results["brave"],
            results["serper"]
        )

        return results

    async def _search_brave(self, query: str) -> List[Dict]:
        """Brave Search API"""
        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key
        }

        params = {
            "q": query,
            "count": 5,
            "freshness": "pd"  # Past day
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                data = response.json()

                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "source": "brave",
                        "age": r.get("age", "")
                    }
                    for r in data.get("web", {}).get("results", [])
                ]
            except Exception as e:
                print(f"Brave search error: {e}")
                return []

    async def _search_serper(self, query: str) -> List[Dict]:
        """Serper (Google Search) API"""
        url = "https://google.serper.dev/search"

        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            "num": 5
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                data = response.json()

                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "description": r.get("snippet", ""),
                        "source": "serper"
                    }
                    for r in data.get("organic", [])
                ]
            except Exception as e:
                print(f"Serper search error: {e}")
                return []

    def _combine_and_dedupe(
        self,
        brave: List[Dict],
        serper: List[Dict]
    ) -> List[Dict]:
        """Łączy wyniki i usuwa duplikaty"""
        seen_urls = set()
        combined = []

        for item in brave + serper:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(item)

        return combined
```
