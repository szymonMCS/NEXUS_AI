## 10. GŁÓWNA PĘTLA SYSTEMU

### 10.1 `betting_floor.py` - Zintegrowana pętla

```python
# betting_floor.py
"""
NEXUS AI - Main system loop for sports prediction.

Cycle:
1. Check if there are matches today
2. Fetch news (Brave/Serper)
3. Evaluate data quality
4. Calculate value for matches with quality > threshold
5. Generate Top 3 ranking
6. Execute bets (optional)
7. Sleep and repeat
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import logging

from config.settings import settings
from agents.data_evaluator import DataEvaluator
from agents.ranker import MatchRanker, format_top_3_report
from agents.bettor import BettorAgent
from data.news.aggregator import NewsAggregator
from database.db import get_session, save_daily_report
from mcp_servers.alerts_server import send_alert

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BettingFloor")


class BettingFloor:
    """
    Main class orchestrating the system.
    """

    def __init__(self):
        self.evaluator = DataEvaluator()
        self.rankers = {
            "tennis": MatchRanker("tennis"),
            "basketball": MatchRanker("basketball")
        }
        self.news_aggregator = NewsAggregator()
        self.bettor = BettorAgent() if settings.ENABLE_LIVE_BETTING else None

        self.is_running = False
        self.last_run = None

    async def run_forever(self):
        """
        Runs the main system loop.
        """
        self.is_running = True
        logger.info("Starting NEXUS AI Betting Floor")

        while self.is_running:
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await send_alert(f"Betting Floor error: {e}")

            # Sleep before next cycle
            sleep_minutes = settings.RUN_EVERY_N_MINUTES
            logger.info(f"Sleeping {sleep_minutes} minutes until next cycle...")
            await asyncio.sleep(sleep_minutes * 60)

    async def _run_cycle(self):
        """
        Single analysis cycle.
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        logger.info(f"═══════════════════════════════════════════════════")
        logger.info(f"Starting analysis cycle for {today}")
        logger.info(f"═══════════════════════════════════════════════════")

        # Check if sports markets are open
        if not self._is_market_open():
            logger.info("Sports markets closed, skipping...")
            return

        all_top3 = []

        # Analyze each sport
        for sport in ["tennis", "basketball"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing {sport.upper()}")
            logger.info(f"{'='*50}")

            try:
                top3 = await self._analyze_sport(sport, today)
                all_top3.extend(top3)
            except Exception as e:
                logger.error(f"Error analyzing {sport}: {e}")
                continue

        # Generate combined report
        if all_top3:
            # Sort by composite score
            all_top3.sort(key=lambda x: x.composite_score, reverse=True)
            final_top3 = all_top3[:3]

            report = format_top_3_report(final_top3, today)
            logger.info(report)

            # Save to database
            await save_daily_report(today, final_top3)

            # Send alert
            await self._send_top3_alert(final_top3, today)

            # Execute bets if enabled
            if self.bettor:
                for match in final_top3:
                    await self._execute_bet(match)
        else:
            logger.info("No value bets found today")

        self.last_run = now

    async def _analyze_sport(self, sport: str, date_str: str) -> List:
        """
        Analyzes all matches for given sport.
        """
        # 1. Get fixtures
        fixtures = await self._get_fixtures(sport, date_str)
        logger.info(f"Found {len(fixtures)} {sport} fixtures")

        if not fixtures:
            return []

        # 2. Evaluate data quality for each match
        logger.info("Evaluating data quality...")
        quality_reports = await self.evaluator.batch_evaluate(fixtures, sport)

        # 3. Filter by quality
        valid_fixtures = []
        for fixture, report in zip(fixtures, quality_reports):
            if report.is_ready:
                fixture["quality_report"] = report
                valid_fixtures.append(fixture)
            else:
                logger.debug(f"Skipping {fixture.get('home')} vs {fixture.get('away')}: "
                           f"{report.recommendation} (score: {report.overall_score:.1f})")

        logger.info(f"{len(valid_fixtures)}/{len(fixtures)} matches passed quality filter")

        # 4. Generate Top 3
        ranker = self.rankers[sport]
        top3 = await ranker.rank_top_3_matches(date_str, valid_fixtures)

        logger.info(f"Top 3 {sport} matches selected")
        for match in top3:
            logger.info(f"  #{match.rank}: {match.match_name} "
                       f"(edge: {match.adjusted_edge:.1%}, quality: {match.quality_score:.0f})")

        return top3

    async def _get_fixtures(self, sport: str, date_str: str) -> List[Dict]:
        """
        Fetches all matches for given day.
        Uses MCP servers to fetch data.
        """
        from mcp_servers.tennis_server import get_fixtures as get_tennis_fixtures
        from mcp_servers.basketball_server import get_fixtures as get_basketball_fixtures
        from mcp_servers.odds_server import get_tennis_odds, get_basketball_odds

        if sport == "tennis":
            fixtures = await get_tennis_fixtures(date_str)
            odds = await get_tennis_odds()
        else:
            fixtures = await get_basketball_fixtures(date_str)
            odds = await get_basketball_odds()

        # Merge odds with fixtures
        for fixture in fixtures:
            match_key = f"{fixture.get('home')} vs {fixture.get('away')}"
            fixture["odds"] = odds.get("matches", {}).get(match_key, {})

        return fixtures

    async def _execute_bet(self, match):
        """
        Executes a bet on match.
        """
        if not self.bettor:
            return

        logger.info(f"Placing bet on {match.match_name}")

        try:
            result = await self.bettor.place_bet(
                match_id=match.match_id,
                selection=match.selection,
                stake=match.stake_recommendation,
                odds=match.best_odds,
                bookmaker=match.best_bookmaker
            )

            if result["success"]:
                logger.info(f"Bet placed successfully: {result['bet_id']}")
            else:
                logger.warning(f"Bet failed: {result['error']}")
        except Exception as e:
            logger.error(f"Bet execution error: {e}")

    async def _send_top3_alert(self, top3: List, date_str: str):
        """
        Sends alert with Top 3.
        """
        message = f"TOP 3 VALUE BETS - {date_str}\n\n"

        for match in top3:
            emoji = "GoldMedalSilverMedalBronzeMedal"[match.rank - 1] if match.rank <= 3 else "Medal"
            message += f"{emoji} {match.match_name}\n"
            message += f"   Edge: +{match.adjusted_edge:.1%} | "
            message += f"Odds: {match.best_odds:.2f} @ {match.best_bookmaker}\n"
            message += f"   Quality: {match.quality_score:.0f}/100 | "
            message += f"Stake: {match.stake_recommendation}\n\n"

        await send_alert(message)

    def _is_market_open(self) -> bool:
        """
        Checks if sports markets are open.
        """
        now = datetime.now()
        hour = now.hour

        # Markets typically active 8:00 - 23:00
        return 8 <= hour <= 23

    def stop(self):
        """Stops the system"""
        self.is_running = False
        logger.info("Betting Floor stopped")


# === ENTRY POINT ===

async def main():
    """Main entry point"""
    floor = BettingFloor()

    try:
        await floor.run_forever()
    except KeyboardInterrupt:
        floor.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
```

---
