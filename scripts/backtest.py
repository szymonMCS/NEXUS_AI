# scripts/backtest.py
"""
Backtesting script for NEXUS AI prediction models.
Tests model performance on historical data.

Usage:
    python scripts/backtest.py --model tennis --start 2024-01-01 --end 2024-12-31
    python scripts/backtest.py --model basketball --bankroll 10000 --kelly 0.25
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestBet:
    """Record of a simulated bet."""
    match_id: str
    date: str
    home_team: str
    away_team: str
    bet_on: str  # 'home' or 'away'
    odds: float
    stake: float
    predicted_prob: float
    actual_winner: str  # 'home' or 'away'
    won: bool = False
    profit_loss: float = 0.0

    def __post_init__(self):
        self.won = self.bet_on == self.actual_winner
        if self.won:
            self.profit_loss = self.stake * (self.odds - 1)
        else:
            self.profit_loss = -self.stake


@dataclass
class BacktestResult:
    """Backtest results summary."""
    model_name: str
    sport: str
    start_date: str
    end_date: str
    initial_bankroll: float
    final_bankroll: float
    total_bets: int
    wins: int
    losses: int
    total_staked: float
    total_profit: float
    roi: float
    win_rate: float
    avg_odds: float
    avg_edge: float
    max_drawdown: float
    sharpe_ratio: float
    bets: List[BacktestBet] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sport": self.sport,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.final_bankroll,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "total_staked": self.total_staked,
            "total_profit": self.total_profit,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "avg_odds": self.avg_odds,
            "avg_edge": self.avg_edge,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
        }


class Backtester:
    """
    Backtesting engine for prediction models.

    Features:
    - Historical data simulation
    - Kelly Criterion stake sizing
    - Performance metrics calculation
    - Drawdown analysis
    """

    def __init__(
        self,
        model_name: str,
        sport: str,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        max_stake_pct: float = 0.05,
    ):
        self.model_name = model_name
        self.sport = sport
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct

        self.bankroll = initial_bankroll
        self.bets: List[BacktestBet] = []
        self.bankroll_history: List[float] = [initial_bankroll]

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load prediction model."""
        try:
            if self.sport == "tennis":
                from core.models.tennis_model import TennisModel
                return TennisModel()
            elif self.sport == "basketball":
                from core.models.basketball_model import BasketballModel
                return BasketballModel()
            else:
                logger.warning(f"Unknown sport: {self.sport}, using base model")
                return None
        except ImportError as e:
            logger.error(f"Could not load model: {e}")
            return None

    def calculate_kelly_stake(
        self,
        probability: float,
        odds: float
    ) -> float:
        """Calculate Kelly Criterion stake."""
        edge = probability * odds - 1

        if edge <= 0:
            return 0.0

        kelly = (probability * odds - 1) / (odds - 1)
        kelly *= self.kelly_fraction  # Fractional Kelly

        # Cap at max stake
        stake_pct = min(kelly, self.max_stake_pct)
        stake = stake_pct * self.bankroll

        return max(0, stake)

    async def load_historical_data(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """Load historical match data."""
        # Try database first
        try:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)

            query = """
            SELECT * FROM historical_matches
            WHERE sport = $1
            AND match_date BETWEEN $2 AND $3
            ORDER BY match_date
            """

            rows = await conn.fetch(query, self.sport, start_date, end_date)
            await conn.close()

            if rows:
                return [dict(row) for row in rows]

        except Exception as e:
            logger.debug(f"Database not available: {e}")

        # Fall back to sample data for demonstration
        logger.info("Using sample historical data for demonstration")
        return self._generate_sample_data(start_date, end_date)

    def _generate_sample_data(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """Generate sample historical data for testing."""
        import random

        matches = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Sample teams/players
        if self.sport == "tennis":
            entities = [
                ("Djokovic N.", "Sinner J."),
                ("Alcaraz C.", "Medvedev D."),
                ("Zverev A.", "Rune H."),
                ("Ruud C.", "Fritz T."),
                ("Tsitsipas S.", "Tiafoe F."),
            ]
        else:
            entities = [
                ("Lakers", "Celtics"),
                ("Warriors", "Bucks"),
                ("Nuggets", "Heat"),
                ("Suns", "76ers"),
                ("Mavs", "Clippers"),
            ]

        match_id = 0
        while current <= end:
            # Generate 2-5 matches per day
            for _ in range(random.randint(2, 5)):
                match_id += 1
                home, away = random.choice(entities)

                # Random but realistic odds
                home_implied = random.uniform(0.3, 0.7)
                margin = 1.05  # 5% bookmaker margin

                home_odds = round(margin / home_implied, 2)
                away_odds = round(margin / (1 - home_implied), 2)

                # Determine winner (slightly favor home)
                home_win_prob = home_implied + random.uniform(-0.1, 0.1)
                winner = "home" if random.random() < home_win_prob else "away"

                matches.append({
                    "id": f"backtest_{match_id}",
                    "sport": self.sport,
                    "match_date": current.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "winner": winner,
                    "home_score": random.randint(60, 120) if self.sport == "basketball" else random.randint(0, 3),
                    "away_score": random.randint(60, 120) if self.sport == "basketball" else random.randint(0, 3),
                })

            current += timedelta(days=1)

        return matches

    def predict_match(self, match: Dict) -> Dict[str, float]:
        """Get prediction for a match."""
        if self.model:
            try:
                # Prepare match data for model
                match_data = {
                    "home_player": {"name": match["home_team"], "ranking": 10},
                    "away_player": {"name": match["away_team"], "ranking": 15},
                    "home_team": {"name": match["home_team"], "rating": 1500},
                    "away_team": {"name": match["away_team"], "rating": 1450},
                }

                result = self.model.predict(match_data)
                return {
                    "home_prob": result.home_probability,
                    "away_prob": result.away_probability,
                }
            except Exception as e:
                logger.debug(f"Model prediction failed: {e}")

        # Fallback: use implied probabilities from odds
        home_implied = 1 / match["home_odds"]
        away_implied = 1 / match["away_odds"]
        total = home_implied + away_implied

        return {
            "home_prob": home_implied / total,
            "away_prob": away_implied / total,
        }

    def find_value_bet(
        self,
        match: Dict,
        prediction: Dict[str, float]
    ) -> Optional[Dict]:
        """Find value bet opportunity."""
        home_edge = prediction["home_prob"] * match["home_odds"] - 1
        away_edge = prediction["away_prob"] * match["away_odds"] - 1

        # Check home bet
        if home_edge >= self.min_edge:
            stake = self.calculate_kelly_stake(
                prediction["home_prob"],
                match["home_odds"]
            )
            if stake > 0:
                return {
                    "bet_on": "home",
                    "odds": match["home_odds"],
                    "stake": stake,
                    "probability": prediction["home_prob"],
                    "edge": home_edge,
                }

        # Check away bet
        if away_edge >= self.min_edge:
            stake = self.calculate_kelly_stake(
                prediction["away_prob"],
                match["away_odds"]
            )
            if stake > 0:
                return {
                    "bet_on": "away",
                    "odds": match["away_odds"],
                    "stake": stake,
                    "probability": prediction["away_prob"],
                    "edge": away_edge,
                }

        return None

    async def run(
        self,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """Run backtest on historical data."""
        logger.info(f"Starting backtest: {self.sport} from {start_date} to {end_date}")

        # Load data
        matches = await self.load_historical_data(start_date, end_date)
        logger.info(f"Loaded {len(matches)} historical matches")

        # Process each match
        for match in matches:
            # Get prediction
            prediction = self.predict_match(match)

            # Find value bet
            value_bet = self.find_value_bet(match, prediction)

            if value_bet:
                # Create bet record
                bet = BacktestBet(
                    match_id=match["id"],
                    date=match["match_date"],
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    bet_on=value_bet["bet_on"],
                    odds=value_bet["odds"],
                    stake=value_bet["stake"],
                    predicted_prob=value_bet["probability"],
                    actual_winner=match["winner"],
                )

                # Update bankroll
                self.bankroll += bet.profit_loss
                self.bets.append(bet)
                self.bankroll_history.append(self.bankroll)

        # Calculate results
        return self._calculate_results(start_date, end_date)

    def _calculate_results(
        self,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """Calculate backtest results and metrics."""
        if not self.bets:
            return BacktestResult(
                model_name=self.model_name,
                sport=self.sport,
                start_date=start_date,
                end_date=end_date,
                initial_bankroll=self.initial_bankroll,
                final_bankroll=self.initial_bankroll,
                total_bets=0,
                wins=0,
                losses=0,
                total_staked=0,
                total_profit=0,
                roi=0,
                win_rate=0,
                avg_odds=0,
                avg_edge=0,
                max_drawdown=0,
                sharpe_ratio=0,
            )

        wins = sum(1 for b in self.bets if b.won)
        losses = len(self.bets) - wins
        total_staked = sum(b.stake for b in self.bets)
        total_profit = sum(b.profit_loss for b in self.bets)

        # Calculate max drawdown
        peak = self.initial_bankroll
        max_drawdown = 0
        for bankroll in self.bankroll_history:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate Sharpe ratio (simplified)
        returns = [b.profit_loss / b.stake if b.stake > 0 else 0 for b in self.bets]
        if len(returns) > 1:
            import statistics
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            model_name=self.model_name,
            sport=self.sport,
            start_date=start_date,
            end_date=end_date,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=self.bankroll,
            total_bets=len(self.bets),
            wins=wins,
            losses=losses,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=(total_profit / total_staked * 100) if total_staked > 0 else 0,
            win_rate=(wins / len(self.bets) * 100) if self.bets else 0,
            avg_odds=sum(b.odds for b in self.bets) / len(self.bets),
            avg_edge=sum((b.predicted_prob * b.odds - 1) for b in self.bets) / len(self.bets),
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe,
            bets=self.bets,
        )


def print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {result.model_name} ({result.sport})")
    print("="*60)

    print(f"\nPeriod: {result.start_date} to {result.end_date}")
    print(f"Initial Bankroll: ${result.initial_bankroll:,.2f}")
    print(f"Final Bankroll: ${result.final_bankroll:,.2f}")

    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
    print("-"*40)

    profit_color = "\033[92m" if result.total_profit >= 0 else "\033[91m"
    reset = "\033[0m"

    print(f"Total Bets: {result.total_bets}")
    print(f"Wins/Losses: {result.wins}/{result.losses}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Total Staked: ${result.total_staked:,.2f}")
    print(f"Total Profit: {profit_color}${result.total_profit:,.2f}{reset}")
    print(f"ROI: {profit_color}{result.roi:.2f}%{reset}")

    print("\n" + "-"*40)
    print("RISK METRICS")
    print("-"*40)

    print(f"Average Odds: {result.avg_odds:.2f}")
    print(f"Average Edge: {result.avg_edge:.1%}")
    print(f"Max Drawdown: {result.max_drawdown:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

    print("\n" + "="*60 + "\n")


async def main_async(args):
    """Main async function."""
    backtester = Backtester(
        model_name=args.model,
        sport=args.model,  # model name = sport for now
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        max_stake_pct=args.max_stake,
    )

    result = await backtester.run(args.start, args.end)
    print_results(result)

    # Save results if requested
    if args.output:
        output_data = result.to_dict()
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")

    return result.roi > 0


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS AI Backtesting"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tennis", "basketball"],
        help="Model/sport to backtest"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll"
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly fraction (0-1)"
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge to place bet"
    )
    parser.add_argument(
        "--max-stake",
        type=float,
        default=0.05,
        help="Maximum stake as fraction of bankroll"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("NEXUS AI - Backtesting Engine")
    print("="*60)

    success = asyncio.run(main_async(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
