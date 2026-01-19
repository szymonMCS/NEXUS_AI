# 🎯 NEXUS AI Lite - System Predykcji Sportowej On-Demand

## Wersja uproszczona oparta na Web Search i darmowych źródłach danych

**Wersja:** 1.0 Lite  
**Tryb:** On-Demand (uruchom → raport → koniec)  
**Koszty:** ~$0/miesiąc (opcjonalnie ~$50 na Claude API)

---

## 📋 SPIS TREŚCI

1. [Założenia Projektu](#1-założenia-projektu)
2. [Architektura Web-First](#2-architektura-web-first)
3. [Darmowe Źródła Danych](#3-darmowe-źródła-danych)
4. [Struktura Projektu](#4-struktura-projektu)
5. [Implementacja - Warstwy Danych](#5-implementacja---warstwy-danych)
6. [Ewaluator Jakości Danych Web](#6-ewaluator-jakości-danych-web)
7. [System Predykcji](#7-system-predykcji)
8. [Generator Raportów](#8-generator-raportów)
9. [Interfejs CLI + Gradio](#9-interfejs-cli--gradio)
10. [Uruchomienie i Użycie](#10-uruchomienie-i-użycie)

---

## 1. ZAŁOŻENIA PROJEKTU

### 1.1 Czym jest NEXUS AI Lite?

System on-demand do generowania dziennych raportów z rekomendacjami zakładów sportowych.
**Uruchamiasz → System zbiera dane z internetu → Ewaluuje jakość → Generuje raport Top 3-5 betów → Koniec.**

### 1.2 Kluczowe Różnice vs Wersja Pro

| Aspekt | NEXUS AI Pro | NEXUS AI Lite |
|--------|--------------|---------------|
| Tryb działania | Ciągły (background) | On-demand |
| Źródła danych | Płatne API ($150+/mies) | Web scraping + darmowe API |
| Live tracking | ✅ | ❌ |
| Deployment | Docker + VPS | Lokalnie / jeden plik |
| Koszt | ~$200/mies | ~$0-50/mies |
| Złożoność | Wysoka | Niska-średnia |

### 1.3 Flow Działania

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEXUS AI Lite - Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [USER] ──► python nexus.py --sport tennis --date 2026-01-19       │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  1. FIXTURE COLLECTOR                                      │     │
│  │     - TheSportsDB (darmowe)                               │     │
│  │     - Sofascore scraping                                  │     │
│  │     - Flashscore scraping                                 │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  2. DATA ENRICHMENT (parallel)                            │     │
│  │     - Brave Search → newsy, kontuzje                      │     │
│  │     - Serper → dodatkowe źródła                           │     │
│  │     - Sofascore → statystyki, H2H                         │     │
│  │     - Flashscore → kursy                                  │     │
│  │     - PL bookies scraping → Fortuna/STS/Betclic          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  3. WEB DATA EVALUATOR (KLUCZOWY!)                        │     │
│  │     - Czy dane z internetu są spójne?                     │     │
│  │     - Czy mamy wystarczająco źródeł?                      │     │
│  │     - Czy informacje są świeże?                           │     │
│  │     - Cross-validation między źródłami                    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│         [Odrzuć mecze z quality < 40%]                             │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  4. PREDICTION ENGINE                                     │     │
│  │     - Model tenisowy (ranking, forma, nawierzchnia)       │     │
│  │     - Model koszykarski (ratings, rest, home advantage)   │     │
│  │     - Value calculation vs kursy                          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  5. RANKING & REPORT                                      │     │
│  │     - Sortuj po: edge × quality × confidence              │     │
│  │     - Wybierz Top 3-5                                     │     │
│  │     - Wygeneruj raport MD/HTML                            │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  [OUTPUT] ──► raport_2026-01-19_tennis.md                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. ARCHITEKTURA WEB-FIRST

### 2.1 Filozofia: Internet jako baza danych

Zamiast polegać na płatnych API, traktujemy internet jako źródło prawdy:
- **Web Search** (Brave/Serper) → newsy, kontuzje, forma
- **Web Scraping** (Sofascore, Flashscore) → statystyki, kursy
- **Darmowe API** (TheSportsDB, API-Sports free tier) → fixtures, podstawowe dane

### 2.2 Ewaluator Jakości Danych Web

**KLUCZOWY KOMPONENT** - Ponieważ dane z internetu mogą być niespójne/nieaktualne:

```python
class WebDataEvaluator:
    """
    Ocenia jakość danych zebranych z internetu.
    
    Kryteria:
    1. SOURCE_AGREEMENT - Czy różne źródła zgadzają się?
    2. FRESHNESS - Jak stare są dane?
    3. COMPLETENESS - Czy mamy wszystkie potrzebne pola?
    4. RELIABILITY - Czy źródło jest wiarygodne?
    """
    
    def evaluate(self, match_data: dict) -> QualityReport:
        scores = {
            "source_agreement": self._check_source_agreement(match_data),
            "freshness": self._check_freshness(match_data),
            "completeness": self._check_completeness(match_data),
            "reliability": self._check_source_reliability(match_data)
        }
        
        overall = weighted_average(scores, weights={
            "source_agreement": 0.35,  # Najważniejsze!
            "freshness": 0.25,
            "completeness": 0.25,
            "reliability": 0.15
        })
        
        return QualityReport(
            overall_score=overall,
            is_trustworthy=overall >= 0.5,
            issues=self._identify_issues(scores)
        )
```

---

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

---

## 4. STRUKTURA PROJEKTU

```
nexus-ai-lite/
│
├── nexus.py                          # 🚀 Główny entry point CLI
├── requirements.txt
├── .env.example
│
├── config/
│   ├── __init__.py
│   ├── settings.py                   # Konfiguracja główna
│   ├── free_apis.py                  # Konfiguracja darmowych API
│   └── leagues.py                    # Klasyfikacja lig
│
├── data/
│   ├── __init__.py
│   │
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── fixture_collector.py      # Zbiera fixtures z wielu źródeł
│   │   └── data_enricher.py          # Wzbogaca dane o statystyki/news
│   │
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── sofascore_scraper.py      # Sofascore (stats, H2H)
│   │   ├── flashscore_scraper.py     # Flashscore (fixtures, odds)
│   │   └── pl_bookies_scraper.py     # Fortuna/STS/Betclic
│   │
│   ├── apis/
│   │   ├── __init__.py
│   │   ├── thesportsdb_client.py     # TheSportsDB (darmowe)
│   │   └── api_sports_client.py      # API-Sports free tier
│   │
│   └── news/
│       ├── __init__.py
│       ├── web_search.py             # Brave + Serper
│       └── injury_extractor.py       # Ekstrakcja kontuzji z newsów
│
├── evaluator/
│   ├── __init__.py
│   ├── web_data_evaluator.py         # 🔑 Ewaluator jakości danych web
│   ├── source_agreement.py           # Sprawdza zgodność źródeł
│   └── freshness_checker.py          # Sprawdza świeżość danych
│
├── prediction/
│   ├── __init__.py
│   ├── tennis_model.py               # Model predykcji tenisa
│   ├── basketball_model.py           # Model predykcji koszykówki
│   └── value_calculator.py           # Obliczanie value vs kursy
│
├── ranking/
│   ├── __init__.py
│   └── match_ranker.py               # Ranking i selekcja Top betów
│
├── reports/
│   ├── __init__.py
│   ├── report_generator.py           # Generator raportów
│   └── templates/
│       ├── report_template.md
│       └── report_template.html
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py                 # Opcjonalny interfejs Gradio
│
└── outputs/                          # Wygenerowane raporty
    └── .gitkeep
```

---

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

---

## 6. EWALUATOR JAKOŚCI DANYCH WEB

### 6.1 Web Data Evaluator - Główny komponent

```python
# evaluator/web_data_evaluator.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class WebQualityReport:
    """Raport jakości danych z internetu"""
    
    overall_score: float  # 0-100
    
    # Sub-scores (0-1)
    source_agreement_score: float
    freshness_score: float
    completeness_score: float
    reliability_score: float
    
    # Metadata
    sources_found: int
    sources_agreeing: int
    data_age_hours: Optional[float]
    missing_fields: List[str] = field(default_factory=list)
    
    # Verdict
    is_trustworthy: bool = False
    recommendation: str = "SKIP"  # PROCEED / CAUTION / SKIP
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class WebDataEvaluator:
    """
    Ewaluator jakości danych zebranych z internetu.
    
    KLUCZOWY KOMPONENT - chroni przed złymi predykcjami
    wynikającymi z niespójnych lub nieaktualnych danych web.
    
    Kryteria:
    1. SOURCE_AGREEMENT (35%) - Czy różne źródła zgadzają się?
    2. FRESHNESS (25%) - Jak stare są dane?
    3. COMPLETENESS (25%) - Czy mamy wszystkie potrzebne dane?
    4. RELIABILITY (15%) - Czy źródła są wiarygodne?
    """
    
    # Wagi dla composite score
    WEIGHTS = {
        "source_agreement": 0.35,
        "freshness": 0.25,
        "completeness": 0.25,
        "reliability": 0.15
    }
    
    # Progi jakości
    THRESHOLD_PROCEED = 0.65   # >= 65% = PROCEED
    THRESHOLD_CAUTION = 0.45   # >= 45% = CAUTION
    # < 45% = SKIP
    
    # Wymagane pola per sport
    REQUIRED_FIELDS = {
        "tennis": [
            "home", "away", "league", "time",
            "odds",  # Kursy od min. 1 bukmachera
        ],
        "basketball": [
            "home", "away", "league", "time",
            "odds",
        ]
    }
    
    # Pożądane pola (bonus)
    DESIRED_FIELDS = {
        "tennis": ["stats", "news", "h2h", "ranking_home", "ranking_away"],
        "basketball": ["stats", "news", "home_form", "away_form"]
    }
    
    def evaluate(self, match: Dict, sport: str) -> WebQualityReport:
        """
        Przeprowadza pełną ewaluację jakości danych meczu.
        
        Args:
            match: Słownik z danymi meczu (z DataEnricher)
            sport: "tennis" lub "basketball"
        
        Returns:
            WebQualityReport z oceną i rekomendacją
        """
        issues = []
        warnings = []
        
        # 1. SOURCE AGREEMENT - Czy źródła się zgadzają?
        agreement_score, agreement_issues = self._check_source_agreement(match)
        issues.extend(agreement_issues)
        
        # 2. FRESHNESS - Jak świeże są dane?
        freshness_score, data_age, freshness_issues = self._check_freshness(match)
        issues.extend(freshness_issues)
        
        # 3. COMPLETENESS - Czy mamy wszystkie dane?
        completeness_score, missing, completeness_issues = self._check_completeness(
            match, sport
        )
        issues.extend(completeness_issues)
        
        # 4. RELIABILITY - Czy źródła są wiarygodne?
        reliability_score, reliability_warnings = self._check_reliability(match)
        warnings.extend(reliability_warnings)
        
        # OVERALL SCORE (weighted average)
        overall = (
            agreement_score * self.WEIGHTS["source_agreement"] +
            freshness_score * self.WEIGHTS["freshness"] +
            completeness_score * self.WEIGHTS["completeness"] +
            reliability_score * self.WEIGHTS["reliability"]
        ) * 100
        
        # RECOMMENDATION
        if overall >= self.THRESHOLD_PROCEED * 100:
            recommendation = "PROCEED"
            is_trustworthy = True
        elif overall >= self.THRESHOLD_CAUTION * 100:
            recommendation = "CAUTION"
            is_trustworthy = True
            warnings.append("Moderate data quality - smaller stake recommended")
        else:
            recommendation = "SKIP"
            is_trustworthy = False
            issues.append(f"Quality score {overall:.1f}% below minimum threshold")
        
        return WebQualityReport(
            overall_score=round(overall, 1),
            source_agreement_score=round(agreement_score, 3),
            freshness_score=round(freshness_score, 3),
            completeness_score=round(completeness_score, 3),
            reliability_score=round(reliability_score, 3),
            sources_found=len(match.get("sources", [])),
            sources_agreeing=self._count_agreeing_sources(match),
            data_age_hours=data_age,
            missing_fields=missing,
            is_trustworthy=is_trustworthy,
            recommendation=recommendation,
            issues=issues,
            warnings=warnings
        )
    
    def _check_source_agreement(self, match: Dict) -> tuple[float, List[str]]:
        """
        Sprawdza czy różne źródła zgadzają się co do danych.
        
        Sprawdzane elementy:
        - Nazwy zawodników/drużyn
        - Czas meczu
        - Kursy (variance < 10%)
        """
        issues = []
        score = 0.0
        
        sources = match.get("sources", [])
        
        # Jeśli tylko jedno źródło - średni score
        if len(sources) <= 1:
            issues.append("Only one data source - cannot verify")
            return 0.5, issues
        
        # Bonus za wiele źródeł
        source_bonus = min(len(sources) / 3, 1.0) * 0.3
        score += source_bonus
        
        # Sprawdź spójność kursów
        odds = match.get("odds", {})
        if len(odds) >= 2:
            odds_variance = self._calculate_odds_variance(odds)
            if odds_variance < 0.05:
                score += 0.4  # Bardzo spójne
            elif odds_variance < 0.10:
                score += 0.25
            else:
                issues.append(f"High odds variance: {odds_variance:.1%}")
                score += 0.1
        else:
            score += 0.2  # Neutralne
        
        # Sprawdź czy mamy dane z różnych typów źródeł
        source_types = set()
        for s in sources:
            if s in ["sofascore", "flashscore"]:
                source_types.add("scraping")
            elif s in ["thesportsdb", "api-sports"]:
                source_types.add("api")
        
        if len(source_types) >= 2:
            score += 0.3  # Różnorodność źródeł
        
        return min(score, 1.0), issues
    
    def _check_freshness(self, match: Dict) -> tuple[float, Optional[float], List[str]]:
        """
        Sprawdza świeżość danych.
        
        Priorytet: newsy < 24h, kursy < 1h
        """
        issues = []
        score = 0.5  # Domyślnie neutralne
        data_age = None
        
        # Sprawdź wiek newsów
        news = match.get("news", {}).get("combined", [])
        if news:
            # Jeśli mamy newsy, to dobrze
            score += 0.3
            
            # Sprawdź czy newsy są świeże (< 48h)
            # TODO: Parsowanie dat z newsów
        else:
            issues.append("No recent news found")
        
        # Sprawdź czy mamy świeże kursy
        odds = match.get("odds", {})
        if odds:
            score += 0.2
        else:
            issues.append("No odds data found")
        
        return score, data_age, issues
    
    def _check_completeness(
        self, 
        match: Dict, 
        sport: str
    ) -> tuple[float, List[str], List[str]]:
        """
        Sprawdza kompletność danych.
        """
        issues = []
        missing = []
        
        required = self.REQUIRED_FIELDS.get(sport, [])
        desired = self.DESIRED_FIELDS.get(sport, [])
        
        # Sprawdź wymagane pola
        required_present = 0
        for field in required:
            if field == "odds":
                if match.get("odds"):
                    required_present += 1
                else:
                    missing.append("odds")
            elif match.get(field):
                required_present += 1
            else:
                missing.append(field)
        
        required_score = required_present / len(required) if required else 1.0
        
        # Sprawdź pożądane pola (bonus)
        desired_present = sum(1 for f in desired if match.get(f))
        desired_score = desired_present / len(desired) if desired else 0
        
        # Combined score
        score = required_score * 0.7 + desired_score * 0.3
        
        if missing:
            issues.append(f"Missing required fields: {missing}")
        
        return score, missing, issues
    
    def _check_reliability(self, match: Dict) -> tuple[float, List[str]]:
        """
        Sprawdza wiarygodność źródeł.
        """
        warnings = []
        score = 0.5
        
        sources = match.get("sources", [])
        
        # Wiarygodne źródła
        reliable_sources = ["sofascore", "flashscore", "thesportsdb"]
        reliable_count = sum(1 for s in sources if s in reliable_sources)
        
        if reliable_count >= 2:
            score = 0.9
        elif reliable_count >= 1:
            score = 0.7
        else:
            warnings.append("No reliable sources found")
            score = 0.3
        
        # Sprawdź wiarygodność newsów
        news = match.get("news", {})
        brave_count = len(news.get("brave", []))
        serper_count = len(news.get("serper", []))
        
        if brave_count > 0 and serper_count > 0:
            score += 0.1  # Bonus za oba źródła newsów
        
        return min(score, 1.0), warnings
    
    def _calculate_odds_variance(self, odds: Dict) -> float:
        """Oblicza variance kursów między bukmacherami"""
        home_odds = []
        
        for bookie, odds_data in odds.items():
            if isinstance(odds_data, dict) and "home" in odds_data:
                home_odds.append(odds_data["home"])
        
        if len(home_odds) < 2:
            return 0.0
        
        mean = sum(home_odds) / len(home_odds)
        variance = sum((x - mean) ** 2 for x in home_odds) / len(home_odds)
        
        # Coefficient of variation
        cv = (variance ** 0.5) / mean if mean > 0 else 0
        return cv
    
    def _count_agreeing_sources(self, match: Dict) -> int:
        """Liczy ile źródeł ma zgodne dane"""
        return len(match.get("sources", []))


# === BATCH EVALUATION ===

async def evaluate_all_matches(
    matches: List[Dict], 
    sport: str
) -> List[tuple[Dict, WebQualityReport]]:
    """
    Ewaluuje wszystkie mecze i zwraca z raportami.
    """
    evaluator = WebDataEvaluator()
    
    results = []
    for match in matches:
        report = evaluator.evaluate(match, sport)
        results.append((match, report))
    
    return results


def filter_by_quality(
    matches_with_reports: List[tuple[Dict, WebQualityReport]],
    min_quality: float = 45.0
) -> List[tuple[Dict, WebQualityReport]]:
    """
    Filtruje mecze po minimalnej jakości.
    """
    return [
        (match, report) 
        for match, report in matches_with_reports
        if report.overall_score >= min_quality
    ]
```

---

## 7. SYSTEM PREDYKCJI

### 7.1 Model Tenisowy

```python
# prediction/tennis_model.py

from dataclasses import dataclass
from typing import Dict, Optional
import math

@dataclass
class TennisPrediction:
    """Wynik predykcji meczu tenisowego"""
    home_win_prob: float
    away_win_prob: float
    confidence: float
    factors: Dict[str, float]
    reasoning: str


class TennisModel:
    """
    Model predykcji tenisa oparty na:
    - Rankingu (ELO-like)
    - Formie (ostatnie 5 meczów)
    - Nawierzchni
    - H2H
    """
    
    # Wagi czynników
    WEIGHTS = {
        "ranking": 0.35,
        "form": 0.25,
        "surface": 0.20,
        "h2h": 0.15,
        "fatigue": 0.05
    }
    
    def predict(self, match: Dict) -> TennisPrediction:
        """
        Oblicza prawdopodobieństwo wygranej.
        """
        stats = match.get("stats", {})
        
        factors = {}
        reasoning_parts = []
        
        # 1. RANKING FACTOR
        ranking_prob = self._ranking_probability(match, stats)
        factors["ranking"] = ranking_prob
        
        # 2. FORM FACTOR (jeśli dostępne)
        form_prob = self._form_probability(stats)
        factors["form"] = form_prob if form_prob else 0.5
        
        # 3. SURFACE FACTOR (jeśli dostępne)
        surface_prob = self._surface_probability(match, stats)
        factors["surface"] = surface_prob if surface_prob else 0.5
        
        # 4. H2H FACTOR (jeśli dostępne)
        h2h_prob = self._h2h_probability(stats)
        factors["h2h"] = h2h_prob if h2h_prob else 0.5
        
        # 5. FATIGUE FACTOR
        fatigue_adj = self._fatigue_adjustment(stats)
        factors["fatigue"] = 0.5 + fatigue_adj
        
        # WEIGHTED AVERAGE
        home_prob = sum(
            factors[k] * self.WEIGHTS[k] 
            for k in self.WEIGHTS.keys()
        )
        
        # Normalize to 0-1
        home_prob = max(0.1, min(0.9, home_prob))
        away_prob = 1 - home_prob
        
        # CONFIDENCE based on data quality
        confidence = self._calculate_confidence(match, factors)
        
        # REASONING
        reasoning = self._generate_reasoning(match, factors)
        
        return TennisPrediction(
            home_win_prob=round(home_prob, 3),
            away_win_prob=round(away_prob, 3),
            confidence=round(confidence, 2),
            factors=factors,
            reasoning=reasoning
        )
    
    def _ranking_probability(self, match: Dict, stats: Dict) -> float:
        """
        Oblicza prawdopodobieństwo na podstawie rankingu.
        
        Używa formuły ELO-like:
        P(home) = 1 / (1 + 10^(-diff/50))
        
        gdzie diff = strength_home - strength_away
        i strength = 1000 / (1 + log10(rank))
        """
        home_rank = stats.get("ranking_home", 100)
        away_rank = stats.get("ranking_away", 100)
        
        # Fallback z newsów jeśli brak danych
        if not home_rank or not away_rank:
            return 0.5
        
        # Oblicz strength
        home_strength = 1000 / (1 + math.log10(max(home_rank, 1)))
        away_strength = 1000 / (1 + math.log10(max(away_rank, 1)))
        
        diff = home_strength - away_strength
        
        # ELO formula
        prob = 1 / (1 + 10 ** (-diff / 50))
        
        return prob
    
    def _form_probability(self, stats: Dict) -> Optional[float]:
        """
        Oblicza prawdopodobieństwo na podstawie formy.
        """
        home_form = stats.get("home_last5_wins", None)
        away_form = stats.get("away_last5_wins", None)
        
        if home_form is None or away_form is None:
            return None
        
        # Prosta proporcja
        total = home_form + away_form
        if total == 0:
            return 0.5
        
        return home_form / total
    
    def _surface_probability(self, match: Dict, stats: Dict) -> Optional[float]:
        """
        Korekta za nawierzchnię.
        """
        surface = match.get("surface", "").lower()
        
        home_surface_pct = stats.get(f"home_{surface}_win_pct")
        away_surface_pct = stats.get(f"away_{surface}_win_pct")
        
        if home_surface_pct and away_surface_pct:
            total = home_surface_pct + away_surface_pct
            return home_surface_pct / total if total > 0 else 0.5
        
        return None
    
    def _h2h_probability(self, stats: Dict) -> Optional[float]:
        """
        Korekta za historię H2H.
        """
        h2h = stats.get("h2h", {})
        
        home_wins = h2h.get("home_wins", 0)
        away_wins = h2h.get("away_wins", 0)
        
        total = home_wins + away_wins
        if total < 2:  # Za mało danych
            return None
        
        return home_wins / total
    
    def _fatigue_adjustment(self, stats: Dict) -> float:
        """
        Korekta za zmęczenie (mecze w ostatnich 14 dniach).
        """
        home_matches = stats.get("home_matches_14d", 3)
        away_matches = stats.get("away_matches_14d", 3)
        
        # Optymalnie: 3-5 meczów
        def fatigue_penalty(matches):
            if matches < 2:
                return -0.05  # Za mało gier = brak rytmu
            elif matches > 6:
                return -0.05 * (matches - 6)  # Zmęczenie
            return 0
        
        home_penalty = fatigue_penalty(home_matches)
        away_penalty = fatigue_penalty(away_matches)
        
        return home_penalty - away_penalty
    
    def _calculate_confidence(self, match: Dict, factors: Dict) -> float:
        """
        Oblicza confidence na podstawie ilości dostępnych danych.
        """
        base = 0.3
        
        # Bonus za każdy dostępny factor
        available_factors = sum(1 for v in factors.values() if v != 0.5)
        base += available_factors * 0.1
        
        # Bonus za wiele źródeł
        sources = len(match.get("sources", []))
        base += min(sources * 0.1, 0.2)
        
        # Bonus za kursy
        if match.get("odds"):
            base += 0.1
        
        return min(base, 0.95)
    
    def _generate_reasoning(self, match: Dict, factors: Dict) -> str:
        """
        Generuje tekstowe uzasadnienie.
        """
        parts = []
        
        home = match.get("home", "Home")
        away = match.get("away", "Away")
        
        if factors["ranking"] > 0.6:
            parts.append(f"{home} has ranking advantage")
        elif factors["ranking"] < 0.4:
            parts.append(f"{away} has ranking advantage")
        
        if factors.get("form", 0.5) > 0.6:
            parts.append(f"{home} in better recent form")
        elif factors.get("form", 0.5) < 0.4:
            parts.append(f"{away} in better recent form")
        
        return "; ".join(parts) if parts else "Balanced match"
```

### 7.2 Value Calculator

```python
# prediction/value_calculator.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ValueBet:
    """Reprezentacja value bet"""
    has_value: bool
    selection: str  # "home" lub "away"
    probability: float
    best_odds: float
    best_bookmaker: str
    edge: float  # (prob * odds) - 1
    kelly_stake: float  # % bankroll
    reasoning: str


class ValueCalculator:
    """
    Oblicza value bety na podstawie prawdopodobieństw i kursów.
    """
    
    # Minimalne edge per typ ligi
    MIN_EDGE = {
        "popular": 0.03,    # 3%
        "medium": 0.04,     # 4%
        "unpopular": 0.05   # 5%
    }
    
    # Kelly fraction (conservative)
    KELLY_FRACTION = 0.25  # 1/4 Kelly
    
    # Max stake
    MAX_STAKE = 0.03  # 3% bankroll
    
    def calculate_value(
        self,
        home_prob: float,
        away_prob: float,
        odds: Dict,
        league_type: str = "medium",
        quality_score: float = 100.0
    ) -> Optional[ValueBet]:
        """
        Szuka value betu.
        
        Args:
            home_prob: Prawdopodobieństwo wygranej home (0-1)
            away_prob: Prawdopodobieństwo wygranej away (0-1)
            odds: Dict {bookmaker: {home: float, away: float}}
            league_type: "popular", "medium", "unpopular"
            quality_score: Score jakości danych (0-100)
        
        Returns:
            ValueBet lub None jeśli brak value
        """
        if not odds:
            return None
        
        min_edge = self.MIN_EDGE.get(league_type, 0.04)
        
        # Quality adjustment - wymagaj większego edge przy niższej jakości
        if quality_score < 60:
            min_edge *= 1.5
        
        # Znajdź najlepsze kursy
        best_home_odds = 0
        best_home_bookie = None
        best_away_odds = 0
        best_away_bookie = None
        
        for bookie, bookie_odds in odds.items():
            if not isinstance(bookie_odds, dict):
                continue
                
            home_odd = bookie_odds.get("home", 0)
            away_odd = bookie_odds.get("away", 0)
            
            if home_odd > best_home_odds:
                best_home_odds = home_odd
                best_home_bookie = bookie
            
            if away_odd > best_away_odds:
                best_away_odds = away_odd
                best_away_bookie = bookie
        
        if best_home_odds == 0 and best_away_odds == 0:
            return None
        
        # Oblicz edge dla obu stron
        home_edge = (home_prob * best_home_odds) - 1
        away_edge = (away_prob * best_away_odds) - 1
        
        # Wybierz lepszy bet
        if home_edge >= min_edge and home_edge >= away_edge:
            selection = "home"
            probability = home_prob
            best_odds = best_home_odds
            best_bookmaker = best_home_bookie
            edge = home_edge
        elif away_edge >= min_edge:
            selection = "away"
            probability = away_prob
            best_odds = best_away_odds
            best_bookmaker = best_away_bookie
            edge = away_edge
        else:
            return None  # Brak value
        
        # Kelly Criterion
        kelly_stake = self._kelly_criterion(probability, best_odds)
        
        # Quality adjustment for stake
        quality_multiplier = quality_score / 100
        adjusted_stake = kelly_stake * quality_multiplier
        
        # Cap stake
        final_stake = min(adjusted_stake, self.MAX_STAKE)
        
        return ValueBet(
            has_value=True,
            selection=selection,
            probability=probability,
            best_odds=best_odds,
            best_bookmaker=best_bookmaker,
            edge=round(edge, 4),
            kelly_stake=round(final_stake, 4),
            reasoning=self._generate_reasoning(
                selection, probability, best_odds, edge
            )
        )
    
    def _kelly_criterion(self, prob: float, odds: float) -> float:
        """
        Oblicza optymalną stawkę Kelly.
        
        f* = (bp - q) / b
        gdzie:
        - b = odds - 1
        - p = probability of winning
        - q = 1 - p
        """
        b = odds - 1
        p = prob
        q = 1 - p
        
        if b <= 0:
            return 0
        
        kelly = (b * p - q) / b
        
        # Apply fraction
        return max(0, kelly * self.KELLY_FRACTION)
    
    def _generate_reasoning(
        self, 
        selection: str, 
        prob: float, 
        odds: float, 
        edge: float
    ) -> str:
        """Generuje uzasadnienie"""
        return (
            f"{selection.upper()} at {odds:.2f} "
            f"(prob: {prob:.1%}, edge: {edge:.1%})"
        )
```

---

## 8. GENERATOR RAPORTÓW

### 8.1 Match Ranker

```python
# ranking/match_ranker.py

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from evaluator.web_data_evaluator import WebQualityReport
from prediction.value_calculator import ValueBet

@dataclass
class RankedBet:
    """Bet z pełnym rankingiem"""
    rank: int
    match_name: str
    league: str
    sport: str
    
    # Value
    selection: str
    probability: float
    odds: float
    bookmaker: str
    edge: float
    stake_recommendation: str
    
    # Quality
    quality_score: float
    data_sources: int
    
    # Composite
    composite_score: float
    
    # Reasoning
    prediction_reasoning: str
    value_reasoning: str
    quality_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MatchRanker:
    """
    Rankuje mecze i wybiera Top N betów.
    
    Composite Score = Edge × Quality × Confidence
    """
    
    def rank_bets(
        self,
        matches_with_predictions: List[Tuple[Dict, WebQualityReport, ValueBet]],
        top_n: int = 5,
        max_unpopular: int = 1
    ) -> List[RankedBet]:
        """
        Rankuje bety i wybiera Top N.
        
        Args:
            matches_with_predictions: Lista (match, quality_report, value_bet)
            top_n: Ile betów zwrócić
            max_unpopular: Max betów z niepopularnych lig
        
        Returns:
            Lista RankedBet posortowana po composite_score
        """
        scored = []
        
        for match, quality_report, value_bet in matches_with_predictions:
            if not value_bet or not value_bet.has_value:
                continue
            
            if not quality_report.is_trustworthy:
                continue
            
            # Calculate composite score
            composite = self._calculate_composite(
                edge=value_bet.edge,
                quality=quality_report.overall_score / 100,
                confidence=value_bet.probability  # Use probability as proxy
            )
            
            # Stake recommendation
            stake = self._stake_recommendation(value_bet.kelly_stake)
            
            scored.append(RankedBet(
                rank=0,  # Set later
                match_name=f"{match['home']} vs {match['away']}",
                league=match.get("league", "Unknown"),
                sport=match.get("sport", "unknown"),
                selection=value_bet.selection,
                probability=value_bet.probability,
                odds=value_bet.best_odds,
                bookmaker=value_bet.best_bookmaker,
                edge=value_bet.edge,
                stake_recommendation=stake,
                quality_score=quality_report.overall_score,
                data_sources=quality_report.sources_found,
                composite_score=composite,
                prediction_reasoning=match.get("prediction", {}).get("reasoning", ""),
                value_reasoning=value_bet.reasoning,
                quality_issues=quality_report.issues,
                warnings=quality_report.warnings
            ))
        
        # Sort by composite score
        scored.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply constraints and select top N
        selected = self._select_top_n(scored, top_n, max_unpopular)
        
        # Set ranks
        for i, bet in enumerate(selected):
            bet.rank = i + 1
        
        return selected
    
    def _calculate_composite(
        self, 
        edge: float, 
        quality: float, 
        confidence: float
    ) -> float:
        """
        Oblicza composite score.
        
        Używa geometric mean z wagami.
        """
        # Normalize edge (cap at 15%)
        norm_edge = min(edge, 0.15) / 0.15
        
        # Geometric weighted mean
        composite = (
            (norm_edge ** 0.4) *
            (quality ** 0.35) *
            (confidence ** 0.25)
        )
        
        return composite
    
    def _stake_recommendation(self, kelly_stake: float) -> str:
        """Konwertuje Kelly stake na rekomendację"""
        pct = kelly_stake * 100
        
        if pct >= 2.5:
            return "2.5% bankroll (HIGH CONFIDENCE)"
        elif pct >= 1.5:
            return "1.5-2% bankroll"
        elif pct >= 1.0:
            return "1% bankroll"
        else:
            return "0.5-1% bankroll (LOW)"
    
    def _select_top_n(
        self, 
        scored: List[RankedBet], 
        top_n: int,
        max_unpopular: int
    ) -> List[RankedBet]:
        """
        Wybiera Top N z ograniczeniami:
        - Max N betów z niepopularnych lig
        - Max 1 bet na turniej
        """
        selected = []
        unpopular_count = 0
        tournaments_used = set()
        
        for bet in scored:
            if len(selected) >= top_n:
                break
            
            # Check unpopular limit
            league_lower = bet.league.lower()
            is_unpopular = any(
                term in league_lower 
                for term in ["itf", "challenger", "futures", "2. liga", "3. liga"]
            )
            
            if is_unpopular:
                if unpopular_count >= max_unpopular:
                    continue
                unpopular_count += 1
            
            # Check tournament uniqueness
            tournament = league_lower.split()[0] if league_lower else "unknown"
            if tournament in tournaments_used:
                continue
            tournaments_used.add(tournament)
            
            selected.append(bet)
        
        return selected
```

### 8.2 Report Generator

```python
# reports/report_generator.py

from datetime import datetime
from typing import List
from pathlib import Path
from ranking.match_ranker import RankedBet

class ReportGenerator:
    """
    Generuje raporty w formatach MD i HTML.
    """
    
    def generate_markdown(
        self, 
        bets: List[RankedBet], 
        sport: str, 
        date: str
    ) -> str:
        """
        Generuje raport w formacie Markdown.
        """
        if not bets:
            return self._generate_no_bets_report(sport, date)
        
        lines = [
            f"# 🎯 NEXUS AI - Raport Predykcji",
            f"",
            f"**Sport:** {sport.upper()}  ",
            f"**Data:** {date}  ",
            f"**Wygenerowano:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## 🏆 TOP {len(bets)} VALUE BETS",
            f""
        ]
        
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        
        for bet in bets:
            emoji = rank_emoji[bet.rank - 1] if bet.rank <= 5 else f"{bet.rank}."
            
            lines.extend([
                f"### {emoji} {bet.match_name}",
                f"",
                f"**Liga:** {bet.league}  ",
                f"**Typ:** {bet.selection.upper()}  ",
                f"**Kurs:** {bet.odds:.2f} @ {bet.bookmaker}  ",
                f"**Edge:** +{bet.edge:.1%}  ",
                f"**Jakość danych:** {bet.quality_score:.0f}/100  ",
                f"**Stawka:** {bet.stake_recommendation}",
                f"",
                f"**Uzasadnienie:**",
                f"> {bet.value_reasoning}",
                f""
            ])
            
            if bet.warnings:
                lines.append(f"**⚠️ Ostrzeżenia:**")
                for w in bet.warnings:
                    lines.append(f"- {w}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Summary
        avg_edge = sum(b.edge for b in bets) / len(bets)
        avg_quality = sum(b.quality_score for b in bets) / len(bets)
        
        lines.extend([
            f"## 📊 Podsumowanie",
            f"",
            f"- **Znaleziono betów:** {len(bets)}",
            f"- **Średni edge:** {avg_edge:.1%}",
            f"- **Średnia jakość danych:** {avg_quality:.0f}/100",
            f"",
            f"---",
            f"",
            f"*Raport wygenerowany przez NEXUS AI Lite v1.0*"
        ])
        
        return "\n".join(lines)
    
    def _generate_no_bets_report(self, sport: str, date: str) -> str:
        """Raport gdy brak betów"""
        return f"""# 🎯 NEXUS AI - Raport Predykcji

**Sport:** {sport.upper()}  
**Data:** {date}  
**Wygenerowano:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## ❌ Brak Value Betów

System nie znalazł żadnych betów spełniających kryteria jakości i value.

**Możliwe przyczyny:**
- Niewystarczająca jakość danych z internetu
- Brak meczów z dodatnim edge
- Kursy bukmacherów są zbyt efektywne

**Zalecenie:** Spróbuj ponownie później lub sprawdź inny sport.

---

*Raport wygenerowany przez NEXUS AI Lite v1.0*
"""
    
    def save_report(
        self, 
        content: str, 
        sport: str, 
        date: str,
        output_dir: str = "outputs"
    ) -> str:
        """
        Zapisuje raport do pliku.
        
        Returns:
            Ścieżka do zapisanego pliku
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"raport_{date}_{sport}.md"
        filepath = output_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return str(filepath)
```

---

## 9. INTERFEJS CLI + GRADIO

### 9.1 Główny Entry Point (CLI)

```python
#!/usr/bin/env python3
# nexus.py - Główny entry point NEXUS AI Lite

import asyncio
import argparse
from datetime import datetime, date
from typing import Optional

from data.collectors.fixture_collector import FixtureCollector
from data.collectors.data_enricher import DataEnricher
from evaluator.web_data_evaluator import WebDataEvaluator, filter_by_quality
from prediction.tennis_model import TennisModel
from prediction.basketball_model import BasketballModel
from prediction.value_calculator import ValueCalculator
from ranking.match_ranker import MatchRanker
from reports.report_generator import ReportGenerator


async def run_analysis(
    sport: str,
    target_date: str,
    min_quality: float = 45.0,
    top_n: int = 5,
    verbose: bool = True
) -> str:
    """
    Przeprowadza pełną analizę i generuje raport.
    
    Args:
        sport: "tennis" lub "basketball"
        target_date: Data w formacie YYYY-MM-DD
        min_quality: Minimalny próg jakości danych
        top_n: Ile betów w raporcie
        verbose: Czy wyświetlać progress
    
    Returns:
        Ścieżka do wygenerowanego raportu
    """
    if verbose:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 NEXUS AI Lite - Analiza On-Demand                        ║
╠══════════════════════════════════════════════════════════════╣
║  Sport: {sport.upper():<10}  Data: {target_date:<15}              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 1. COLLECT FIXTURES
    if verbose:
        print("📅 [1/5] Zbieranie meczów z internetu...")
    
    collector = FixtureCollector()
    fixtures = await collector.collect_fixtures(sport, target_date)
    
    if not fixtures:
        print("❌ Nie znaleziono meczów na ten dzień!")
        return None
    
    if verbose:
        print(f"   ✅ Znaleziono {len(fixtures)} meczów\n")
    
    # 2. ENRICH DATA
    if verbose:
        print("🔍 [2/5] Wzbogacanie danych (newsy, statystyki, kursy)...")
    
    enricher = DataEnricher()
    enriched = await enricher.enrich_all(fixtures, sport, max_concurrent=3)
    
    if verbose:
        print(f"   ✅ Wzbogacono {len(enriched)} meczów\n")
    
    # 3. EVALUATE DATA QUALITY
    if verbose:
        print("📊 [3/5] Ewaluacja jakości danych z internetu...")
    
    evaluator = WebDataEvaluator()
    matches_with_quality = []
    
    for match in enriched:
        report = evaluator.evaluate(match, sport)
        matches_with_quality.append((match, report))
        
        if verbose and report.overall_score < min_quality:
            print(f"   ⚠️ {match['home']} vs {match['away']}: "
                  f"quality {report.overall_score:.0f}% (SKIP)")
    
    # Filter by quality
    quality_matches = filter_by_quality(matches_with_quality, min_quality)
    
    if verbose:
        print(f"   ✅ {len(quality_matches)}/{len(matches_with_quality)} "
              f"meczów przeszło filtr jakości (>= {min_quality}%)\n")
    
    if not quality_matches:
        print("❌ Żaden mecz nie przeszedł filtra jakości!")
        generator = ReportGenerator()
        content = generator._generate_no_bets_report(sport, target_date)
        return generator.save_report(content, sport, target_date)
    
    # 4. PREDICTIONS & VALUE
    if verbose:
        print("🧠 [4/5] Obliczanie predykcji i szukanie value...")
    
    model = TennisModel() if sport == "tennis" else BasketballModel()
    value_calc = ValueCalculator()
    
    matches_with_predictions = []
    
    for match, quality_report in quality_matches:
        # Predict
        prediction = model.predict(match)
        match["prediction"] = prediction.__dict__
        
        # Calculate value
        value_bet = value_calc.calculate_value(
            home_prob=prediction.home_win_prob,
            away_prob=prediction.away_win_prob,
            odds=match.get("odds", {}),
            league_type="medium",  # TODO: Klasyfikacja ligi
            quality_score=quality_report.overall_score
        )
        
        matches_with_predictions.append((match, quality_report, value_bet))
        
        if verbose and value_bet and value_bet.has_value:
            print(f"   💰 {match['home']} vs {match['away']}: "
                  f"edge +{value_bet.edge:.1%}")
    
    if verbose:
        value_count = sum(1 for _, _, v in matches_with_predictions if v and v.has_value)
        print(f"   ✅ Znaleziono {value_count} value betów\n")
    
    # 5. RANK & GENERATE REPORT
    if verbose:
        print("📝 [5/5] Generowanie raportu...")
    
    ranker = MatchRanker()
    top_bets = ranker.rank_bets(matches_with_predictions, top_n=top_n)
    
    generator = ReportGenerator()
    content = generator.generate_markdown(top_bets, sport, target_date)
    filepath = generator.save_report(content, sport, target_date)
    
    if verbose:
        print(f"   ✅ Raport zapisany: {filepath}\n")
        print("=" * 60)
        print(content)
        print("=" * 60)
    
    return filepath


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NEXUS AI Lite - System predykcji sportowych on-demand"
    )
    
    parser.add_argument(
        "--sport", "-s",
        choices=["tennis", "basketball"],
        default="tennis",
        help="Sport do analizy (default: tennis)"
    )
    
    parser.add_argument(
        "--date", "-d",
        default=str(date.today()),
        help="Data do analizy YYYY-MM-DD (default: dziś)"
    )
    
    parser.add_argument(
        "--min-quality", "-q",
        type=float,
        default=45.0,
        help="Minimalny próg jakości danych 0-100 (default: 45)"
    )
    
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Liczba betów w raporcie (default: 5)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Tryb cichy - tylko output raportu"
    )
    
    args = parser.parse_args()
    
    # Run async
    filepath = asyncio.run(run_analysis(
        sport=args.sport,
        target_date=args.date,
        min_quality=args.min_quality,
        top_n=args.top,
        verbose=not args.quiet
    ))
    
    if filepath:
        print(f"\n✅ Gotowe! Raport: {filepath}")
    else:
        print("\n❌ Nie udało się wygenerować raportu")


if __name__ == "__main__":
    main()
```

### 9.2 Gradio App (Opcjonalnie)

```python
# ui/gradio_app.py

import gradio as gr
import asyncio
from datetime import date

# Import main analysis function
import sys
sys.path.append("..")
from nexus import run_analysis

def run_sync(sport: str, target_date: str, min_quality: float, top_n: int):
    """Wrapper synchroniczny dla Gradio"""
    filepath = asyncio.run(run_analysis(
        sport=sport,
        target_date=target_date,
        min_quality=min_quality,
        top_n=int(top_n),
        verbose=False
    ))
    
    if filepath:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return "❌ Nie udało się wygenerować raportu"


def create_app():
    """Tworzy aplikację Gradio"""
    
    with gr.Blocks(title="NEXUS AI Lite", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎯 NEXUS AI Lite
        ### System predykcji sportowych on-demand
        
        Wygeneruj raport z najlepszymi betami na dany dzień.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                sport = gr.Dropdown(
                    choices=["tennis", "basketball"],
                    value="tennis",
                    label="🏅 Sport"
                )
                
                target_date = gr.Textbox(
                    value=str(date.today()),
                    label="📅 Data (YYYY-MM-DD)"
                )
                
                min_quality = gr.Slider(
                    minimum=30,
                    maximum=80,
                    value=45,
                    step=5,
                    label="📊 Min. jakość danych (%)"
                )
                
                top_n = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="🏆 Liczba betów w raporcie"
                )
                
                btn = gr.Button("🚀 Generuj Raport", variant="primary")
            
            with gr.Column(scale=2):
                output = gr.Markdown(label="Raport")
        
        btn.click(
            fn=run_sync,
            inputs=[sport, target_date, min_quality, top_n],
            outputs=output
        )
        
        gr.Markdown("""
        ---
        **Uwagi:**
        - System zbiera dane z internetu (Sofascore, Flashscore, newsy)
        - Jakość danych jest weryfikowana przed analizą
        - Minimalna jakość 45% jest zalecana dla wiarygodnych predykcji
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
```

---

## 10. URUCHOMIENIE I UŻYCIE

### 10.1 Instalacja

```bash
# 1. Sklonuj/utwórz projekt
mkdir nexus-ai-lite
cd nexus-ai-lite

# 2. Utwórz virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Skonfiguruj .env (opcjonalnie dla newsów)
cp .env.example .env
# Edytuj .env i dodaj klucze API (Brave, Serper)
```

### 10.2 Requirements.txt

```
# requirements.txt

# HTTP & Async
httpx>=0.24.0
aiohttp>=3.9.0

# Web Scraping
playwright>=1.40.0
beautifulsoup4>=4.12.0

# Data Processing
pandas>=2.0.0
pydantic>=2.0.0

# UI (opcjonalnie)
gradio>=4.0.0

# Utilities
python-dotenv>=1.0.0
tenacity>=8.2.0  # Retry logic

# Dev
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### 10.3 .env.example

```bash
# .env.example

# === NEWS APIs (opcjonalne - system działa też bez nich) ===
BRAVE_API_KEY=your_brave_key_here
SERPER_API_KEY=your_serper_key_here

# === LLM (opcjonalne - do ekstrakcji kontuzji) ===
ANTHROPIC_API_KEY=your_anthropic_key_here

# === API-Sports (opcjonalne) ===
API_SPORTS_KEY=your_api_sports_key_here
```

### 10.4 Przykłady Użycia

```bash
# Analiza tenisa na dziś
python nexus.py --sport tennis

# Analiza koszykówki na konkretny dzień
python nexus.py --sport basketball --date 2026-01-20

# Z wyższym progiem jakości
python nexus.py -s tennis -q 60

# Więcej betów w raporcie
python nexus.py -s tennis -n 7

# Tryb cichy (tylko raport)
python nexus.py -s tennis --quiet

# Uruchom interfejs Gradio
python ui/gradio_app.py
```

### 10.5 Przykładowy Output

```
╔══════════════════════════════════════════════════════════════╗
║  🎯 NEXUS AI Lite - Analiza On-Demand                        ║
╠══════════════════════════════════════════════════════════════╣
║  Sport: TENNIS      Data: 2026-01-19                         ║
╚══════════════════════════════════════════════════════════════╝

📅 [1/5] Zbieranie meczów z internetu...
  ✅ thesportsdb: 12 matches
  ✅ sofascore: 45 matches
  ✅ flashscore: 38 matches
   ✅ Znaleziono 52 meczów

🔍 [2/5] Wzbogacanie danych (newsy, statystyki, kursy)...
   ✅ Wzbogacono 52 meczów

📊 [3/5] Ewaluacja jakości danych z internetu...
   ⚠️ Qualifier A vs Qualifier B: quality 32% (SKIP)
   ⚠️ Unknown Player vs Unknown: quality 28% (SKIP)
   ✅ 34/52 meczów przeszło filtr jakości (>= 45%)

🧠 [4/5] Obliczanie predykcji i szukanie value...
   💰 Sinner J. vs Alcaraz C.: edge +4.2%
   💰 Sabalenka A. vs Swiatek I.: edge +3.8%
   ✅ Znaleziono 5 value betów

📝 [5/5] Generowanie raportu...
   ✅ Raport zapisany: outputs/raport_2026-01-19_tennis.md

============================================================
# 🎯 NEXUS AI - Raport Predykcji

**Sport:** TENNIS  
**Data:** 2026-01-19  
**Wygenerowano:** 2026-01-19 14:35

---

## 🏆 TOP 5 VALUE BETS

### 🥇 Sinner J. vs Alcaraz C.

**Liga:** Australian Open  
**Typ:** HOME  
**Kurs:** 2.15 @ Fortuna  
**Edge:** +4.2%  
**Jakość danych:** 78/100  
**Stawka:** 1.5-2% bankroll

**Uzasadnienie:**
> HOME at 2.15 (prob: 52.3%, edge: 4.2%)

---
...
============================================================

✅ Gotowe! Raport: outputs/raport_2026-01-19_tennis.md
```

---

## PODSUMOWANIE

### Co zawiera NEXUS AI Lite:

| Komponent | Opis |
|-----------|------|
| **FixtureCollector** | Zbiera mecze z TheSportsDB, Sofascore, Flashscore |
| **DataEnricher** | Wzbogaca o kursy (PL bookies), newsy (Brave/Serper) |
| **WebDataEvaluator** | 🔑 Ewaluuje jakość danych web (agreement, freshness, completeness) |
| **TennisModel** | Predykcja na podstawie rankingu, formy, nawierzchni, H2H |
| **BasketballModel** | Predykcja na podstawie ratings, rest, home advantage |
| **ValueCalculator** | Oblicza edge i Kelly stake |
| **MatchRanker** | Composite score = edge × quality × confidence |
| **ReportGenerator** | Generuje raporty MD/HTML |

### Koszty:

| Serwis | Koszt | Użycie |
|--------|-------|--------|
| TheSportsDB | $0 | Fixtures (key=3) |
| API-Sports | $0 | 100 req/dzień/API |
| Sofascore | $0 | Scraping stats |
| Flashscore | $0 | Scraping odds |
| Brave Search | $0 | 2000 req/mies |
| Serper | $0 | 2500 req/mies |
| **RAZEM** | **~$0/mies** | |

### Następne Kroki:

1. Utwórz strukturę katalogów
2. Zaimplementuj scrapers (Sofascore, Flashscore, PL bookies)
3. Zaimplementuj WebDataEvaluator
4. Dodaj modele predykcji
5. Testuj na rzeczywistych danych
