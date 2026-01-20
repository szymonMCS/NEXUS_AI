# data/scrapers/flashscore_scraper.py
"""
Flashscore scraper for NEXUS AI.
Uses Playwright for JavaScript-rendered content.

Note: Run 'playwright install chromium' after pip install playwright
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import logging

try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


class FlashscoreScraper:
    """
    Scraper for Flashscore fixtures and odds.

    Flashscore provides:
    - Live scores
    - Fixtures by date/sport
    - Odds from multiple bookmakers
    - Statistics
    - Head-to-head records

    Requires Playwright with Chromium installed.
    """

    BASE_URL = "https://www.flashscore.com"

    # Sport paths
    SPORT_PATHS = {
        "tennis": "/tennis/",
        "basketball": "/basketball/",
        "football": "/football/",
        "ice_hockey": "/ice-hockey/",
        "handball": "/handball/",
        "volleyball": "/volleyball/",
    }

    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize Flashscore scraper.

        Args:
            headless: Run browser in headless mode
            timeout: Default timeout in milliseconds
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        """Initialize Playwright and browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close browser and Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _get_page(self) -> Page:
        """Create new page with default settings."""
        if not self._browser:
            raise RuntimeError("Browser not initialized. Use async context manager.")

        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()
        page.set_default_timeout(self.timeout)
        return page

    async def get_fixtures(
        self,
        sport: str,
        date: str = None
    ) -> List[Dict]:
        """
        Get fixtures for a sport and date.

        Args:
            sport: Sport type (tennis, basketball, etc.)
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            List of fixture dicts
        """
        if sport not in self.SPORT_PATHS:
            logger.warning(f"Unknown sport: {sport}")
            return []

        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        page = await self._get_page()

        try:
            # Navigate to sport page
            url = f"{self.BASE_URL}{self.SPORT_PATHS[sport]}"
            await page.goto(url, wait_until="networkidle")

            # Accept cookies if dialog appears
            try:
                await page.click("#onetrust-accept-btn-handler", timeout=3000)
            except:
                pass

            # Wait for matches to load
            await page.wait_for_selector(".event__match", timeout=10000)

            # Extract fixtures
            fixtures = await self._extract_fixtures(page, sport, date)

            return fixtures

        except Exception as e:
            logger.error(f"Flashscore scraping failed: {e}")
            return []

        finally:
            await page.close()

    async def _extract_fixtures(
        self,
        page: Page,
        sport: str,
        date: str
    ) -> List[Dict]:
        """Extract fixture data from page."""
        fixtures = []

        # Get all match elements
        matches = await page.query_selector_all(".event__match")

        for match in matches:
            try:
                fixture = await self._parse_match_element(match, sport)
                if fixture:
                    # Filter by date if needed
                    if fixture.get("start_time"):
                        match_date = fixture["start_time"].strftime("%Y-%m-%d")
                        if match_date == date:
                            fixtures.append(fixture)
                    else:
                        fixtures.append(fixture)

            except Exception as e:
                logger.debug(f"Error parsing match: {e}")
                continue

        return fixtures

    async def _parse_match_element(
        self,
        element,
        sport: str
    ) -> Optional[Dict]:
        """Parse single match element."""
        try:
            # Match ID
            match_id = await element.get_attribute("id")
            if match_id:
                match_id = match_id.replace("g_1_", "")

            # Teams/Players
            home_el = await element.query_selector(".event__participant--home")
            away_el = await element.query_selector(".event__participant--away")

            home_team = await home_el.inner_text() if home_el else ""
            away_team = await away_el.inner_text() if away_el else ""

            # Time
            time_el = await element.query_selector(".event__time")
            time_str = await time_el.inner_text() if time_el else ""

            start_time = self._parse_time(time_str)

            # League (from parent section)
            league = ""
            try:
                league_header = await element.evaluate(
                    "el => el.closest('.sportName').querySelector('.event__title--name')?.innerText"
                )
                league = league_header or ""
            except:
                pass

            # Scores (if live/finished)
            home_score_el = await element.query_selector(".event__score--home")
            away_score_el = await element.query_selector(".event__score--away")

            home_score = await home_score_el.inner_text() if home_score_el else None
            away_score = await away_score_el.inner_text() if away_score_el else None

            return {
                "match_id": f"flashscore_{match_id}",
                "sport": sport,
                "league": league.strip(),
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                "start_time": start_time,
                "home_score": home_score,
                "away_score": away_score,
                "source": "flashscore"
            }

        except Exception as e:
            logger.debug(f"Match parse error: {e}")
            return None

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse Flashscore time string to datetime."""
        if not time_str:
            return None

        time_str = time_str.strip()

        # Handle "HH:MM" format
        if re.match(r"^\d{2}:\d{2}$", time_str):
            today = datetime.now().date()
            try:
                time_obj = datetime.strptime(time_str, "%H:%M").time()
                return datetime.combine(today, time_obj)
            except:
                pass

        # Handle "DD.MM. HH:MM" format
        if re.match(r"^\d{2}\.\d{2}\. \d{2}:\d{2}$", time_str):
            try:
                year = datetime.now().year
                return datetime.strptime(f"{time_str} {year}", "%d.%m. %H:%M %Y")
            except:
                pass

        # Handle live indicators
        if any(x in time_str.lower() for x in ["live", "set", "half", "qt"]):
            return datetime.now()

        return None

    async def get_match_odds(self, match_id: str) -> Dict[str, Any]:
        """
        Get odds for a specific match.

        Args:
            match_id: Flashscore match ID

        Returns:
            Dict with odds from multiple bookmakers
        """
        page = await self._get_page()

        try:
            # Navigate to match odds page
            url = f"{self.BASE_URL}/match/{match_id}/#/odds-comparison/1x2-odds/full-time"
            await page.goto(url, wait_until="networkidle")

            # Wait for odds table
            await page.wait_for_selector(".ui-table__body", timeout=10000)

            # Extract odds
            odds = await self._extract_odds(page)

            return {
                "match_id": match_id,
                "odds": odds,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get odds: {e}")
            return {"match_id": match_id, "odds": [], "error": str(e)}

        finally:
            await page.close()

    async def _extract_odds(self, page: Page) -> List[Dict]:
        """Extract odds from odds comparison page."""
        odds = []

        rows = await page.query_selector_all(".ui-table__row")

        for row in rows:
            try:
                # Bookmaker name
                bookie_el = await row.query_selector(".oddsCell__bookmaker")
                bookie = await bookie_el.inner_text() if bookie_el else "Unknown"

                # Odds values
                odds_cells = await row.query_selector_all(".oddsCell__odd")

                if len(odds_cells) >= 2:
                    home_odds = await odds_cells[0].inner_text()
                    away_odds = await odds_cells[-1].inner_text()

                    # For 1X2 markets (football etc.)
                    draw_odds = None
                    if len(odds_cells) == 3:
                        draw_odds = await odds_cells[1].inner_text()

                    odds.append({
                        "bookmaker": bookie.strip(),
                        "home_odds": self._parse_odds(home_odds),
                        "draw_odds": self._parse_odds(draw_odds) if draw_odds else None,
                        "away_odds": self._parse_odds(away_odds),
                    })

            except Exception as e:
                logger.debug(f"Error extracting odds row: {e}")
                continue

        return odds

    def _parse_odds(self, odds_str: str) -> Optional[float]:
        """Parse odds string to float."""
        if not odds_str:
            return None

        try:
            # Remove any non-numeric chars except decimal point
            cleaned = re.sub(r"[^\d.]", "", odds_str.strip())
            return float(cleaned) if cleaned else None
        except:
            return None

    async def get_h2h(self, match_id: str) -> Dict[str, Any]:
        """
        Get head-to-head record for a match.

        Args:
            match_id: Flashscore match ID

        Returns:
            Dict with H2H data
        """
        page = await self._get_page()

        try:
            # Navigate to H2H page
            url = f"{self.BASE_URL}/match/{match_id}/#/h2h/overall"
            await page.goto(url, wait_until="networkidle")

            # Wait for H2H section
            await page.wait_for_selector(".h2h", timeout=10000)

            # Extract H2H data
            h2h_data = await self._extract_h2h(page)

            return {
                "match_id": match_id,
                "h2h": h2h_data,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get H2H: {e}")
            return {"match_id": match_id, "h2h": None, "error": str(e)}

        finally:
            await page.close()

    async def _extract_h2h(self, page: Page) -> Dict:
        """Extract H2H data from page."""
        h2h = {
            "total_matches": 0,
            "home_wins": 0,
            "away_wins": 0,
            "draws": 0,
            "recent_meetings": []
        }

        try:
            # Get summary stats
            summary = await page.query_selector(".h2h__section")
            if summary:
                text = await summary.inner_text()
                # Parse "Team A: X wins, Draws: Y, Team B: Z wins"
                numbers = re.findall(r"\d+", text)
                if len(numbers) >= 3:
                    h2h["home_wins"] = int(numbers[0])
                    h2h["draws"] = int(numbers[1])
                    h2h["away_wins"] = int(numbers[2])
                    h2h["total_matches"] = sum(int(n) for n in numbers[:3])

            # Get recent meetings
            meetings = await page.query_selector_all(".h2h__row")
            for meeting in meetings[:5]:
                try:
                    date_el = await meeting.query_selector(".h2h__date")
                    home_el = await meeting.query_selector(".h2h__homeParticipant")
                    away_el = await meeting.query_selector(".h2h__awayParticipant")
                    score_el = await meeting.query_selector(".h2h__result")

                    h2h["recent_meetings"].append({
                        "date": await date_el.inner_text() if date_el else "",
                        "home": await home_el.inner_text() if home_el else "",
                        "away": await away_el.inner_text() if away_el else "",
                        "score": await score_el.inner_text() if score_el else ""
                    })
                except:
                    continue

        except Exception as e:
            logger.debug(f"H2H extraction error: {e}")

        return h2h


# === HELPER FUNCTIONS ===

async def scrape_flashscore_fixtures(
    sport: str,
    date: str = None
) -> List[Dict]:
    """
    Convenience function to scrape fixtures.

    Args:
        sport: Sport type
        date: Date in YYYY-MM-DD

    Returns:
        List of fixture dicts
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available, skipping Flashscore")
        return []

    async with FlashscoreScraper() as scraper:
        return await scraper.get_fixtures(sport, date)


async def scrape_flashscore_odds(match_id: str) -> Dict:
    """
    Convenience function to get odds.

    Args:
        match_id: Flashscore match ID

    Returns:
        Odds dict
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available")
        return {"error": "Playwright not installed"}

    async with FlashscoreScraper() as scraper:
        return await scraper.get_match_odds(match_id)
