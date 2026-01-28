"""
Integrated Data Collection and Model Training.

Combines data collection from sport_datasets_AI_report.md sources
with ML model training in a single pipeline.

Usage:
    python scripts/collect_and_train.py --sport football --days 365
    python scripts/collect_and_train.py --all-sports
    python scripts/collect_and_train.py --check-data
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from core.datasets import DatasetManager, DatasetConfig
from core.ml.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect data and train models"
    )
    
    parser.add_argument("--sport", type=str, help="Sport to process")
    parser.add_argument("--days", type=int, default=365, help="Days of data")
    parser.add_argument("--all-sports", action="store_true", help="All sports")
    parser.add_argument("--check-data", action="store_true", help="Check existing data")
    parser.add_argument("--min-matches", type=int, default=1000, help="Minimum matches required")
    
    return parser.parse_args()


async def check_existing_data():
    """Check what data is already collected."""
    logger.info("Checking existing data...")
    
    data_dir = Path("data/historical")
    if not data_dir.exists():
        logger.info("No historical data directory found")
        return
    
    for sport_dir in data_dir.iterdir():
        if sport_dir.is_dir():
            files = list(sport_dir.glob("*.csv"))
            total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
            logger.info(f"  {sport_dir.name}: {len(files)} files, {total_size:.1f} MB")


async def collect_for_sport(sport: str, days: int, min_matches: int) -> bool:
    """Collect and validate data for a sport."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {sport.upper()}")
    logger.info(f"{'='*60}")
    
    manager = DatasetManager()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    config = DatasetConfig(
        sport=sport,
        start_date=start_date,
        end_date=end_date,
        min_matches=min_matches,
    )
    
    try:
        # Collect data
        matches = await manager.collect(config)
        
        if len(matches) < min_matches:
            logger.warning(f"Insufficient matches: {len(matches)} < {min_matches}")
            return False
        
        # Generate report
        report = manager.generate_report(sport, matches)
        logger.info(f"Collection report:")
        logger.info(f"  Total matches: {report.total_matches}")
        logger.info(f"  Sources: {', '.join(report.sources_used)}")
        logger.info(f"  Quality score: {report.quality_score:.2f}")
        logger.info(f"  Date range: {report.date_range[0].date()} to {report.date_range[1].date()}")
        
        # Save to training format
        output_path = f"data/historical/{sport}/training_data.csv"
        success = manager.save_to_training_format(matches, output_path, format="csv")
        
        if success:
            logger.info(f"Data saved to {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to collect {sport}: {e}")
        return False


async def main():
    args = parse_args()
    
    if args.check_data:
        await check_existing_data()
        return 0
    
    sports = ["football", "basketball", "tennis", "hockey", "baseball", "handball"] if args.all_sports else [args.sport]
    
    if not sports or not sports[0]:
        logger.error("Must specify --sport or --all-sports")
        return 1
    
    logger.info("="*60)
    logger.info("NEXUS AI - Data Collection & Training Pipeline")
    logger.info("="*60)
    logger.info(f"Sports: {', '.join(sports)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Min matches: {args.min_matches}")
    logger.info("")
    
    results = {}
    for sport in sports:
        success = await collect_for_sport(sport, args.days, args.min_matches)
        results[sport] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Summary")
    logger.info(f"{'='*60}")
    
    for sport, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {sport}")
    
    all_success = all(results.values())
    
    if all_success:
        logger.info("\n✓ All sports processed successfully")
        logger.info("\nNext steps:")
        logger.info("  1. Train models: python scripts/train_models.py --all")
        logger.info("  2. Run predictions: python main.py --analyze football")
    else:
        logger.warning("\n⚠ Some sports failed - check logs above")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
