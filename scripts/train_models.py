"""
ML Model Training Pipeline.

Based on sport_datasets_AI_report.md training techniques:
- Ensemble methods (XGBoost, Random Forest)
- Deep Learning (LSTM for sequence modeling)
- Poisson regression for goals
- Transfer learning between leagues

Usage:
    python scripts/train_models.py --sport football --days 365
    python scripts/train_models.py --all
    python scripts/train_models.py --sport basketball --model-type goals
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from core.datasets import DatasetManager, DatasetConfig
from core.ml.models import GoalsModel, HandicapModel
from core.ml.registry import ModelRegistry
from core.ml.features import FeaturePipeline, FeatureVector
from core.ml.training import OnlineTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ML models for sports prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_models.py --sport football --days 365
  python scripts/train_models.py --sport basketball --model-type goals
  python scripts/train_models.py --all --parallel
        """
    )
    
    parser.add_argument(
        "--sport",
        type=str,
        choices=["football", "basketball", "tennis", "hockey", "baseball", "handball"],
        help="Sport to train models for"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to use (default: 365)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["goals", "handicap", "both"],
        default="both",
        help="Type of model to train (default: both)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train models for all sports"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train multiple models in parallel"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/registry",
        help="Output directory for trained models"
    )
    
    return parser.parse_args()


async def collect_training_data(
    sport: str,
    days: int,
) -> List[Any]:
    """Collect historical data for training."""
    logger.info(f"Collecting {days} days of {sport} data...")
    
    manager = DatasetManager()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    config = DatasetConfig(
        sport=sport,
        start_date=start_date,
        end_date=end_date,
    )
    
    try:
        matches = await manager.collect(config)
        logger.info(f"Collected {len(matches)} matches for {sport}")
        return matches
    except Exception as e:
        logger.error(f"Failed to collect data for {sport}: {e}")
        return []


def prepare_features(matches: List[Any]) -> List[FeatureVector]:
    """Convert matches to feature vectors."""
    pipeline = FeaturePipeline()
    features = []
    
    for match in matches:
        try:
            # Create simplified match data structure
            from core.data.schemas import MatchData, Team, DataQuality
            
            match_data = MatchData(
                match_id=match.match_id,
                home_team=Team(name=match.home_team),
                away_team=Team(name=match.away_team),
                match_date=match.match_date,
                sport=match.sport,
                league=match.raw_data.get("league", "unknown"),
                home_score=match.home_score,
                away_score=match.away_score,
                quality=DataQuality(
                    completeness=0.8 if match.home_stats else 0.5,
                    reliability=0.8,
                ),
            )
            
            feature_vector = pipeline.extract(match_data)
            features.append(feature_vector)
        except Exception as e:
            logger.debug(f"Failed to extract features: {e}")
            continue
    
    return features


def prepare_targets(matches: List[Any]) -> List[tuple]:
    """Extract target values (goals) from matches."""
    targets = []
    
    for match in matches:
        if match.home_score is not None and match.away_score is not None:
            targets.append((float(match.home_score), float(match.away_score)))
    
    return targets


async def train_goals_model(
    sport: str,
    matches: List[Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Train goals prediction model."""
    logger.info(f"Training goals model for {sport}...")
    
    # Prepare data
    features = prepare_features(matches)
    targets = prepare_targets(matches)
    
    if len(features) < 50:
        logger.warning(f"Insufficient data for goals model: {len(features)} samples")
        return {"success": False, "error": "Insufficient data"}
    
    # Create and train model
    model = GoalsModel()
    
    try:
        metrics = model.train(features, targets, validation_split=0.2)
        
        # Save model
        registry = ModelRegistry(output_dir)
        version = registry.save_model(model, metrics=metrics)
        
        logger.info(f"Goals model trained. MAE: {metrics.get('mae_total', 'N/A'):.3f}")
        logger.info(f"Model saved as version {version}")
        
        return {
            "success": True,
            "model_type": "goals",
            "sport": sport,
            "version": version,
            "metrics": metrics,
            "samples": len(features),
        }
    except Exception as e:
        logger.error(f"Goals model training failed: {e}")
        return {"success": False, "error": str(e)}


async def train_handicap_model(
    sport: str,
    matches: List[Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Train handicap prediction model."""
    logger.info(f"Training handicap model for {sport}...")
    
    # Prepare data
    features = prepare_features(matches)
    
    if len(features) < 50:
        logger.warning(f"Insufficient data for handicap model: {len(features)} samples")
        return {"success": False, "error": "Insufficient data"}
    
    # Create and train model
    model = HandicapModel()
    
    try:
        # For handicap, we need win/loss targets
        targets = []
        for match in matches:
            if match.result:
                targets.append(match.result)
        
        metrics = model.train(features, targets, validation_split=0.2)
        
        # Save model
        registry = ModelRegistry(output_dir)
        version = registry.save_model(model, metrics=metrics)
        
        logger.info(f"Handicap model trained. Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
        logger.info(f"Model saved as version {version}")
        
        return {
            "success": True,
            "model_type": "handicap",
            "sport": sport,
            "version": version,
            "metrics": metrics,
            "samples": len(features),
        }
    except Exception as e:
        logger.error(f"Handicap model training failed: {e}")
        return {"success": False, "error": str(e)}


async def train_sport_models(
    sport: str,
    args,
) -> List[Dict[str, Any]]:
    """Train all models for a sport."""
    results = []
    
    # Collect data
    matches = await collect_training_data(sport, args.days)
    
    if len(matches) < 100:
        logger.error(f"Insufficient data for {sport}: {len(matches)} matches")
        return [{"success": False, "error": "Insufficient data"}]
    
    # Train models
    if args.model_type in ["goals", "both"]:
        result = await train_goals_model(sport, matches, args.output_dir)
        results.append(result)
    
    if args.model_type in ["handicap", "both"]:
        result = await train_handicap_model(sport, matches, args.output_dir)
        results.append(result)
    
    return results


async def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("NEXUS AI - ML Model Training Pipeline")
    logger.info("=" * 60)
    
    # Determine sports to train
    if args.all:
        sports = ["football", "basketball", "tennis", "hockey", "baseball", "handball"]
    else:
        if not args.sport:
            logger.error("Must specify --sport or --all")
            return 1
        sports = [args.sport]
    
    logger.info(f"Training models for: {', '.join(sports)}")
    logger.info(f"Using {args.days} days of historical data")
    logger.info(f"Model types: {args.model_type}")
    logger.info("")
    
    # Train models
    all_results = {}
    
    if args.parallel and len(sports) > 1:
        # Train in parallel
        tasks = [train_sport_models(sport, args) for sport in sports]
        results_list = await asyncio.gather(*tasks)
        all_results = dict(zip(sports, results_list))
    else:
        # Train sequentially
        for sport in sports:
            results = await train_sport_models(sport, args)
            all_results[sport] = results
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    total_success = 0
    total_failed = 0
    
    for sport, results in all_results.items():
        logger.info(f"\n{sport.upper()}:")
        for result in results:
            if result.get("success"):
                logger.info(f"  ✓ {result['model_type']}: v{result['version']} (samples: {result['samples']})")
                total_success += 1
            else:
                logger.info(f"  ✗ {result.get('model_type', 'unknown')}: {result.get('error', 'Failed')}")
                total_failed += 1
    
    logger.info("")
    logger.info(f"Total: {total_success} succeeded, {total_failed} failed")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sports": sports,
            "days": args.days,
            "model_type": args.model_type,
        },
        "results": all_results,
    }
    
    report_path = Path(args.output_dir) / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to {report_path}")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
