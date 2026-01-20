# scripts/run_daily.py
"""
Daily execution script for NEXUS AI.
Runs the complete betting analysis pipeline.

Usage:
    python scripts/run_daily.py --sport tennis
    python scripts/run_daily.py --sport basketball --mode lite --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyRunner:
    """
    Daily execution runner for NEXUS AI betting pipeline.

    Steps:
    1. Collect fixtures for today
    2. Gather data from all sources
    3. Evaluate data quality
    4. Generate predictions
    5. Identify value bets
    6. Apply risk management
    7. Output recommendations
    """

    def __init__(
        self,
        sport: str,
        mode: str = "lite",
        bankroll: float = 1000.0,
        dry_run: bool = False,
    ):
        self.sport = sport
        self.mode = mode
        self.bankroll = bankroll
        self.dry_run = dry_run
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.results = {
            "date": self.date,
            "sport": sport,
            "mode": mode,
            "fixtures": [],
            "predictions": [],
            "value_bets": [],
            "recommendations": [],
        }

    async def run(self) -> dict:
        """Run complete daily pipeline."""
        logger.info(f"Starting daily run: {self.sport} ({self.mode} mode)")
        logger.info(f"Date: {self.date}")
        logger.info(f"Bankroll: ${self.bankroll:,.2f}")

        try:
            # Step 1: Collect fixtures
            await self._collect_fixtures()

            if not self.results["fixtures"]:
                logger.warning("No fixtures found for today")
                return self.results

            # Step 2: Gather detailed data
            await self._gather_data()

            # Step 3: Evaluate data quality
            await self._evaluate_quality()

            # Step 4: Generate predictions
            await self._generate_predictions()

            # Step 5: Identify value bets
            await self._identify_value_bets()

            # Step 6: Apply risk management
            await self._apply_risk_management()

            # Step 7: Generate recommendations
            await self._generate_recommendations()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.results["error"] = str(e)
            return self.results

    async def _collect_fixtures(self):
        """Collect today's fixtures."""
        logger.info("Step 1: Collecting fixtures...")

        try:
            from data.collectors.fixture_collector import FixtureCollector

            async with FixtureCollector(mode=self.mode) as collector:
                fixtures = await collector.collect_fixtures(
                    sport=self.sport,
                    date=self.date
                )
                self.results["fixtures"] = fixtures
                logger.info(f"Found {len(fixtures)} fixtures")

        except ImportError:
            logger.warning("FixtureCollector not available, using mock data")
            self.results["fixtures"] = self._mock_fixtures()

    async def _gather_data(self):
        """Gather detailed data for each fixture."""
        logger.info("Step 2: Gathering detailed data...")

        for fixture in self.results["fixtures"]:
            try:
                # Add player/team data
                fixture["data_complete"] = True

                # In production, would fetch:
                # - Rankings
                # - Recent form
                # - H2H records
                # - News/injuries

            except Exception as e:
                logger.debug(f"Data gathering error for {fixture.get('match_id')}: {e}")
                fixture["data_complete"] = False

    async def _evaluate_quality(self):
        """Evaluate data quality for each fixture."""
        logger.info("Step 3: Evaluating data quality...")

        for fixture in self.results["fixtures"]:
            # Simple quality score
            quality = {
                "overall_score": 0.7,
                "completeness": 0.8,
                "freshness": 0.9,
                "source_count": 2,
            }

            if fixture.get("data_complete"):
                quality["overall_score"] = 0.85

            fixture["quality"] = quality

    async def _generate_predictions(self):
        """Generate predictions for qualified fixtures."""
        logger.info("Step 4: Generating predictions...")

        try:
            if self.sport == "tennis":
                from core.models.tennis_model import TennisModel
                model = TennisModel()
            elif self.sport == "basketball":
                from core.models.basketball_model import BasketballModel
                model = BasketballModel()
            else:
                model = None
        except ImportError:
            model = None

        for fixture in self.results["fixtures"]:
            # Skip low quality data
            if fixture.get("quality", {}).get("overall_score", 0) < 0.5:
                continue

            try:
                if model:
                    match_data = self._prepare_match_data(fixture)
                    prediction = model.predict(match_data)

                    self.results["predictions"].append({
                        "match_id": fixture["match_id"],
                        "home_probability": prediction.home_probability,
                        "away_probability": prediction.away_probability,
                        "confidence": prediction.confidence,
                        "model": model.name,
                    })
                else:
                    # Fallback prediction
                    self.results["predictions"].append({
                        "match_id": fixture["match_id"],
                        "home_probability": 0.5,
                        "away_probability": 0.5,
                        "confidence": 0.5,
                        "model": "fallback",
                    })

            except Exception as e:
                logger.debug(f"Prediction error: {e}")

        logger.info(f"Generated {len(self.results['predictions'])} predictions")

    async def _identify_value_bets(self):
        """Identify value betting opportunities."""
        logger.info("Step 5: Identifying value bets...")

        try:
            from core.value_calculator import ValueCalculator
            calculator = ValueCalculator()
        except ImportError:
            calculator = None

        min_edge = 0.03 if self.mode == "pro" else 0.05

        for prediction in self.results["predictions"]:
            match_id = prediction["match_id"]

            # Find fixture
            fixture = next(
                (f for f in self.results["fixtures"] if f["match_id"] == match_id),
                None
            )

            if not fixture:
                continue

            # Get odds (mock if not available)
            home_odds = fixture.get("home_odds", 2.0)
            away_odds = fixture.get("away_odds", 2.0)

            # Calculate value
            home_edge = prediction["home_probability"] * home_odds - 1
            away_edge = prediction["away_probability"] * away_odds - 1

            quality_mult = fixture.get("quality", {}).get("overall_score", 0.7)

            if home_edge >= min_edge:
                kelly = self._calculate_kelly(
                    prediction["home_probability"],
                    home_odds
                )
                self.results["value_bets"].append({
                    "match_id": match_id,
                    "bet_on": "home",
                    "odds": home_odds,
                    "probability": prediction["home_probability"],
                    "edge": home_edge,
                    "kelly_stake": kelly * quality_mult,
                    "quality_multiplier": quality_mult,
                    "confidence": prediction["confidence"],
                })

            if away_edge >= min_edge:
                kelly = self._calculate_kelly(
                    prediction["away_probability"],
                    away_odds
                )
                self.results["value_bets"].append({
                    "match_id": match_id,
                    "bet_on": "away",
                    "odds": away_odds,
                    "probability": prediction["away_probability"],
                    "edge": away_edge,
                    "kelly_stake": kelly * quality_mult,
                    "quality_multiplier": quality_mult,
                    "confidence": prediction["confidence"],
                })

        logger.info(f"Found {len(self.results['value_bets'])} value bets")

    async def _apply_risk_management(self):
        """Apply risk management rules."""
        logger.info("Step 6: Applying risk management...")

        max_daily_stake = 0.20  # Max 20% of bankroll per day
        max_single_stake = 0.05  # Max 5% per bet

        total_stake = 0

        for bet in self.results["value_bets"]:
            # Cap individual stake
            stake_pct = min(bet["kelly_stake"], max_single_stake)

            # Check daily limit
            if total_stake + stake_pct > max_daily_stake:
                stake_pct = max(0, max_daily_stake - total_stake)

            bet["adjusted_stake"] = stake_pct
            bet["stake_amount"] = stake_pct * self.bankroll
            total_stake += stake_pct

        # Sort by edge (best first)
        self.results["value_bets"].sort(
            key=lambda x: x["edge"],
            reverse=True
        )

    async def _generate_recommendations(self):
        """Generate final betting recommendations."""
        logger.info("Step 7: Generating recommendations...")

        for bet in self.results["value_bets"]:
            if bet["adjusted_stake"] <= 0:
                continue

            match_id = bet["match_id"]
            fixture = next(
                (f for f in self.results["fixtures"] if f["match_id"] == match_id),
                None
            )

            if not fixture:
                continue

            recommendation = {
                "match": f"{fixture.get('home_team', 'Home')} vs {fixture.get('away_team', 'Away')}",
                "league": fixture.get("league", "Unknown"),
                "start_time": fixture.get("start_time"),
                "bet_on": bet["bet_on"].upper(),
                "odds": bet["odds"],
                "stake": f"${bet['stake_amount']:.2f}",
                "stake_percent": f"{bet['adjusted_stake']*100:.1f}%",
                "edge": f"{bet['edge']*100:.1f}%",
                "confidence": f"{bet['confidence']*100:.0f}%",
                "potential_profit": f"${bet['stake_amount'] * (bet['odds'] - 1):.2f}",
            }

            self.results["recommendations"].append(recommendation)

        logger.info(f"Generated {len(self.results['recommendations'])} recommendations")

    def _prepare_match_data(self, fixture: dict) -> dict:
        """Prepare match data for model."""
        return {
            "home_player": {
                "name": fixture.get("home_team", "Player A"),
                "ranking": fixture.get("home_ranking", 50),
            },
            "away_player": {
                "name": fixture.get("away_team", "Player B"),
                "ranking": fixture.get("away_ranking", 50),
            },
            "home_team": {
                "name": fixture.get("home_team", "Team A"),
                "rating": fixture.get("home_rating", 1500),
            },
            "away_team": {
                "name": fixture.get("away_team", "Team B"),
                "rating": fixture.get("away_rating", 1500),
            },
        }

    def _calculate_kelly(self, probability: float, odds: float) -> float:
        """Calculate Kelly stake."""
        edge = probability * odds - 1
        if edge <= 0:
            return 0
        kelly = (probability * odds - 1) / (odds - 1)
        return kelly * 0.25  # Quarter Kelly

    def _mock_fixtures(self) -> list:
        """Generate mock fixtures for testing."""
        if self.sport == "tennis":
            return [
                {
                    "match_id": "mock_1",
                    "sport": "tennis",
                    "league": "ATP 500",
                    "home_team": "Djokovic N.",
                    "away_team": "Sinner J.",
                    "start_time": f"{self.date}T14:00:00",
                    "home_odds": 1.85,
                    "away_odds": 2.05,
                },
                {
                    "match_id": "mock_2",
                    "sport": "tennis",
                    "league": "ATP 250",
                    "home_team": "Alcaraz C.",
                    "away_team": "Medvedev D.",
                    "start_time": f"{self.date}T16:00:00",
                    "home_odds": 1.75,
                    "away_odds": 2.15,
                },
            ]
        else:
            return [
                {
                    "match_id": "mock_1",
                    "sport": "basketball",
                    "league": "NBA",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "start_time": f"{self.date}T19:30:00",
                    "home_odds": 1.90,
                    "away_odds": 1.95,
                },
            ]


def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "="*70)
    print(f"NEXUS AI - Daily Analysis Results")
    print(f"Date: {results['date']} | Sport: {results['sport'].upper()} | Mode: {results['mode'].upper()}")
    print("="*70)

    print(f"\nFixtures analyzed: {len(results['fixtures'])}")
    print(f"Predictions generated: {len(results['predictions'])}")
    print(f"Value bets found: {len(results['value_bets'])}")

    if results["recommendations"]:
        print("\n" + "-"*70)
        print("BETTING RECOMMENDATIONS")
        print("-"*70)

        for i, rec in enumerate(results["recommendations"], 1):
            print(f"\n{i}. {rec['match']}")
            print(f"   League: {rec['league']}")
            print(f"   Bet: {rec['bet_on']} @ {rec['odds']}")
            print(f"   Stake: {rec['stake']} ({rec['stake_percent']})")
            print(f"   Edge: {rec['edge']} | Confidence: {rec['confidence']}")
            print(f"   Potential profit: {rec['potential_profit']}")

    else:
        print("\nNo value bets identified for today.")

    print("\n" + "="*70 + "\n")


async def main_async(args):
    """Main async function."""
    runner = DailyRunner(
        sport=args.sport,
        mode=args.mode,
        bankroll=args.bankroll,
        dry_run=args.dry_run,
    )

    results = await runner.run()
    print_results(results)

    # Save results if output specified
    if args.output:
        # Convert datetime objects to strings
        output_results = json.loads(
            json.dumps(results, default=str)
        )
        with open(args.output, 'w') as f:
            json.dump(output_results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return len(results.get("recommendations", [])) > 0


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS AI Daily Execution"
    )
    parser.add_argument(
        "--sport",
        type=str,
        required=True,
        choices=["tennis", "basketball"],
        help="Sport to analyze"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pro", "lite"],
        default="lite",
        help="Operating mode (pro=paid APIs, lite=free sources)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Current bankroll"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation mode (no actual bets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("Running in DRY-RUN mode (simulation only)")

    success = asyncio.run(main_async(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
