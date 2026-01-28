"""
Collect Historical Data for ML Training.

Checkpoint: 5.4
Run: python scripts/collect_historical.py [--sport SPORT] [--leagues LEAGUES] [--days DAYS]

Examples:
    # Collect last year of Premier League data
    python scripts/collect_historical.py --sport football --leagues PL --days 365

    # Collect all default football leagues
    python scripts/collect_historical.py --sport football --days 365

    # Collect basketball data
    python scripts/collect_historical.py --sport basketball --leagues NBA --days 365
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from core.data.enums import Sport
from data.collectors import (
    HistoricalDataCollector,
    CollectionResult,
    CollectionConfig,
    DEFAULT_LEAGUES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect historical match data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_historical.py --sport football --leagues PL,LaLiga --days 365
  python scripts/collect_historical.py --sport basketball --leagues NBA --days 180
  python scripts/collect_historical.py --all --days 365
        """
    )

    parser.add_argument(
        "--sport",
        type=str,
        choices=["football", "basketball", "tennis"],
        default="football",
        help="Sport to collect (default: football)"
    )

    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="Comma-separated list of leagues (e.g., PL,LaLiga,SerieA). Uses defaults if not specified."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to collect (default: 365)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect all default sports and leagues"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/historical",
        help="Output directory (default: data/historical)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without actually collecting"
    )

    return parser.parse_args()


def save_results(results: List[CollectionResult], output_dir: Path, format: str = "json"):
    """Save collection results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_matches = 0

    for result in results:
        if not result.matches:
            continue

        filename = f"{result.sport.value}_{result.league}_{result.start_date.strftime('%Y%m%d')}_{result.end_date.strftime('%Y%m%d')}"

        if format == "json":
            filepath = output_dir / f"{filename}.json"
            data = {
                "metadata": result.get_summary(),
                "matches": [m.to_dict() for m in result.matches],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "jsonl":
            filepath = output_dir / f"{filename}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for match in result.matches:
                    f.write(json.dumps(match.to_dict(), ensure_ascii=False) + "\n")

        total_matches += len(result.matches)
        logger.info(f"Saved {len(result.matches)} matches to {filepath}")

    return total_matches


def print_summary(results: List[CollectionResult]):
    """Print collection summary."""
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    total_matches = 0
    total_with_stats = 0
    total_with_odds = 0
    total_errors = 0

    for result in results:
        status_icon = {
            "success": "[OK]",
            "partial": "[~~]",
            "failed": "[!!]",
            "no_data": "[--]",
        }.get(result.status.value, "[??]")

        print(f"\n{status_icon} {result.sport.value}/{result.league}")
        print(f"    Matches: {result.total_collected}")
        print(f"    With stats: {result.matches_with_stats}")
        print(f"    With odds: {result.matches_with_odds}")
        print(f"    Sources: {', '.join(result.sources_used)}")
        print(f"    Duration: {result.duration_seconds:.1f}s")

        if result.errors:
            print(f"    Errors: {result.total_errors}")
            for err in result.errors[:3]:  # Show first 3 errors
                print(f"      - {err[:60]}...")

        total_matches += result.total_collected
        total_with_stats += result.matches_with_stats
        total_with_odds += result.matches_with_odds
        total_errors += result.total_errors

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_matches} matches collected")
    print(f"  With detailed stats: {total_with_stats} ({100*total_with_stats/max(1,total_matches):.1f}%)")
    print(f"  With odds data: {total_with_odds} ({100*total_with_odds/max(1,total_matches):.1f}%)")
    print(f"  Errors: {total_errors}")
    print("=" * 60)


async def main():
    args = parse_args()

    # Determine sport
    sport_map = {
        "football": Sport.FOOTBALL,
        "basketball": Sport.BASKETBALL,
        "tennis": Sport.TENNIS,
    }

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Parse leagues
    leagues = None
    if args.leagues:
        leagues = [l.strip() for l in args.leagues.split(",")]

    output_dir = Path(args.output)

    # Dry run - just show what would happen
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No data will be collected")
        print("=" * 60)

        if args.all:
            for sport, sport_leagues in DEFAULT_LEAGUES.items():
                print(f"\n{sport.value}:")
                for league in sport_leagues:
                    print(f"  - {league}")
        else:
            sport = sport_map[args.sport]
            target_leagues = leagues or DEFAULT_LEAGUES.get(sport, [])
            print(f"\nSport: {args.sport}")
            print(f"Leagues: {', '.join(target_leagues)}")

        print(f"\nDate range: {start_date.date()} to {end_date.date()}")
        print(f"Output: {output_dir}")
        return

    # Collect data
    print("\n" + "=" * 60)
    print("NEXUS Historical Data Collector")
    print("=" * 60)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output: {output_dir}")

    all_results = []

    async with HistoricalDataCollector() as collector:
        if args.all:
            # Collect all sports
            print("\nCollecting all configured sports and leagues...")
            results_by_sport = await collector.collect_all(
                start_date=start_date,
                end_date=end_date,
            )
            for sport_results in results_by_sport.values():
                all_results.extend(sport_results)
        else:
            # Collect specific sport
            sport = sport_map[args.sport]
            print(f"\nCollecting {args.sport} data...")
            results = await collector.collect_multiple_leagues(
                sport=sport,
                leagues=leagues,
                start_date=start_date,
                end_date=end_date,
            )
            all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results
    if all_results:
        print(f"\nSaving to {output_dir}...")
        total_saved = save_results(all_results, output_dir, args.format)
        print(f"Total: {total_saved} matches saved")

        # Also save combined metadata
        metadata_file = output_dir / "collection_metadata.json"
        metadata = {
            "collection_date": datetime.utcnow().isoformat(),
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "results": [r.get_summary() for r in all_results],
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    asyncio.run(main())
