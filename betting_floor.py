# betting_floor.py
"""
NEXUS AI Betting Floor - Main orchestration system.
Runs the complete betting analysis pipeline on a schedule.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from config.settings import settings
from core.state import NexusState, Sport
from database.db import init_db, get_db_session
from database.crud import (
    create_match, create_prediction, create_bet,
    get_active_session, create_betting_session,
    update_session_metrics, get_performance_summary
)
from agents import SupervisorAgent, run_betting_analysis

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("nexus.betting_floor")


class BettingFloor:
    """
    Main orchestration class for NEXUS AI.

    Responsibilities:
    - Initialize system components
    - Run scheduled analysis loops
    - Track performance metrics
    - Manage betting sessions
    """

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.is_running = False
        self.current_session = None
        self.last_run: Optional[datetime] = None

    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing NEXUS AI Betting Floor...")

        # Initialize database
        init_db()
        logger.info("Database initialized")

        # Get or create active betting session
        with get_db_session() as db:
            self.current_session = get_active_session(db)

            if not self.current_session:
                self.current_session = create_betting_session(db, {
                    "session_name": f"Session_{datetime.now().strftime('%Y%m%d')}",
                    "start_date": datetime.now(),
                    "starting_bankroll": 1000.0,
                    "current_bankroll": 1000.0,
                    "is_active": True
                })
                logger.info(f"Created new betting session: {self.current_session.session_name}")
            else:
                logger.info(f"Resuming session: {self.current_session.session_name}")

        logger.info("NEXUS AI Betting Floor initialized successfully")

    async def run_analysis(
        self,
        sport: str = "tennis",
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single analysis cycle.

        Args:
            sport: Sport to analyze
            date: Date to analyze (YYYY-MM-DD), defaults to today

        Returns:
            Analysis results dict
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Starting analysis for {sport} on {date}")

        try:
            # Run the agent workflow
            bankroll = self.current_session.current_bankroll if self.current_session else 1000.0

            results = await run_betting_analysis(
                sport=sport,
                date=date,
                bankroll=bankroll
            )

            # Process results
            await self._process_results(results, sport, date)

            self.last_run = datetime.now()

            logger.info(
                f"Analysis complete: {results['approved_bets'].__len__()} approved bets, "
                f"{results['rejected_matches']} rejected"
            )

            return results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    async def _process_results(
        self,
        results: Dict[str, Any],
        sport: str,
        date: str
    ):
        """
        Process analysis results and store in database.

        Args:
            results: Analysis results from workflow
            sport: Sport analyzed
            date: Date analyzed
        """
        approved_bets = results.get("approved_bets", [])

        if not approved_bets:
            logger.info("No approved bets to process")
            return

        with get_db_session() as db:
            for match_data in approved_bets:
                try:
                    # Create match record
                    match = create_match(db, {
                        "external_id": match_data.get("match_id", ""),
                        "sport": sport,
                        "home_team": match_data.get("home_player", {}).get("name", ""),
                        "away_team": match_data.get("away_player", {}).get("name", ""),
                        "league": match_data.get("league", ""),
                        "start_time": datetime.fromisoformat(date),
                        "quality_score": match_data.get("data_quality", {}).get("overall_score", 0),
                    })

                    # Create prediction record
                    prediction_data = match_data.get("prediction", {})
                    if prediction_data:
                        create_prediction(db, {
                            "match_id": match.id,
                            "home_win_prob": prediction_data.get("home_win_probability", 0.5),
                            "away_win_prob": prediction_data.get("away_win_probability", 0.5),
                            "confidence": prediction_data.get("confidence", 0),
                            "factors": prediction_data.get("factors", {}),
                        })

                    # Create bet record (pending)
                    value_bet = match_data.get("value_bet", {})
                    if value_bet:
                        create_bet(db, {
                            "match_id": match.id,
                            "bet_type": "moneyline",
                            "selection": value_bet.get("bet_on", ""),
                            "odds": value_bet.get("odds", 1.0),
                            "stake": value_bet.get("kelly_stake", 0) * (self.current_session.current_bankroll if self.current_session else 1000),
                            "confidence_at_time": prediction_data.get("confidence", 0),
                            "edge_at_time": value_bet.get("edge", 0),
                        })

                    logger.info(f"Stored bet: {match.home_team} vs {match.away_team}")

                except Exception as e:
                    logger.error(f"Error storing bet: {str(e)}")

    async def run_scheduled_loop(
        self,
        sports: List[str] = None,
        interval_minutes: int = None
    ):
        """
        Run analysis on a schedule.

        Args:
            sports: List of sports to analyze
            interval_minutes: Minutes between runs
        """
        if sports is None:
            sports = ["tennis", "basketball"]

        if interval_minutes is None:
            interval_minutes = settings.RUN_EVERY_N_MINUTES

        self.is_running = True
        logger.info(f"Starting scheduled loop: {sports} every {interval_minutes} minutes")

        while self.is_running:
            try:
                today = datetime.now().strftime("%Y-%m-%d")

                for sport in sports:
                    await self.run_analysis(sport, today)

                # Wait for next run
                logger.info(f"Waiting {interval_minutes} minutes until next run...")
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in scheduled loop: {str(e)}")
                # Wait before retrying
                await asyncio.sleep(60)

    def stop(self):
        """Stop the scheduled loop."""
        self.is_running = False
        logger.info("Stopping betting floor...")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Status dict
        """
        with get_db_session() as db:
            performance = get_performance_summary(db)

        return {
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "current_session": {
                "name": self.current_session.session_name if self.current_session else None,
                "bankroll": self.current_session.current_bankroll if self.current_session else 0,
                "roi": self.current_session.roi_percentage if self.current_session else 0,
            } if self.current_session else None,
            "performance": performance,
            "mode": settings.APP_MODE,
        }


# === MAIN ENTRY POINTS ===

async def run_once(sport: str = "tennis", date: str = None) -> Dict:
    """
    Run a single analysis cycle.

    Args:
        sport: Sport to analyze
        date: Date to analyze

    Returns:
        Analysis results
    """
    floor = BettingFloor()
    await floor.initialize()
    return await floor.run_analysis(sport, date)


async def run_continuous(sports: List[str] = None):
    """
    Run continuous analysis loop.

    Args:
        sports: List of sports to analyze
    """
    floor = BettingFloor()
    await floor.initialize()
    await floor.run_scheduled_loop(sports)


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="NEXUS AI Betting Floor")
    parser.add_argument(
        "--mode",
        choices=["once", "continuous"],
        default="once",
        help="Run mode: once or continuous"
    )
    parser.add_argument(
        "--sport",
        choices=["tennis", "basketball"],
        default="tennis",
        help="Sport to analyze"
    )
    parser.add_argument(
        "--date",
        help="Date to analyze (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    if args.mode == "once":
        results = asyncio.run(run_once(args.sport, args.date))
        print(json.dumps(results, indent=2, default=str))
    else:
        asyncio.run(run_continuous([args.sport]))


if __name__ == "__main__":
    main()
