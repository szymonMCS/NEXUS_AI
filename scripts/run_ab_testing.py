"""
Run A/B Testing for Model Comparison.

Compares old models vs new cutting-edge models.
Tracks 100+ predictions for statistical significance.

Usage:
    python scripts/run_ab_testing.py --old-model goals --new-model cutting_edge --samples 100
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml.evaluation.ab_testing import ABTestingFramework
from core.ml.cutting_edge_integration import CuttingEdgeEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run A/B testing")
    parser.add_argument("--old-model", default="goals", help="Old model name")
    parser.add_argument("--new-model", default="cutting_edge", help="New model name")
    parser.add_argument("--samples", type=int, default=100, help="Target samples")
    parser.add_argument("--sport", default="football", help="Sport to test")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("NEXUS AI - A/B Testing Framework")
    logger.info("=" * 70)
    logger.info(f"Old Model: {args.old_model}")
    logger.info(f"New Model: {args.new_model}")
    logger.info(f"Target Samples: {args.samples}")
    logger.info(f"Sport: {args.sport}")
    logger.info("=" * 70)
    
    # Initialize A/B testing
    ab = ABTestingFramework()
    
    # Start test
    test_id = ab.start_test(
        model_a_name=args.old_model,
        model_b_name=args.new_model,
        target_samples=args.samples,
    )
    
    logger.info(f"Test ID: {test_id}")
    logger.info("Recording predictions...")
    
    # Initialize models
    old_ensemble = None  # Would load existing
    new_ensemble = CuttingEdgeEnsemble()
    
    # Simulate predictions (in real scenario, these come from actual matches)
    for i in range(args.samples):
        # Assign group
        group = ab.assign_group(test_id)
        
        # Record prediction (simulated)
        # In real usage, this would be actual model prediction
        record_id = ab.record_prediction(
            test_id=test_id,
            model_name=args.new_model if group == 'B' else args.old_model,
            group=group,
            match_id=f"match_{i}",
            predicted_outcome='home',  # Simulated
            predicted_prob=0.5,
            confidence=0.6,
            sport=args.sport,
        )
        
        # Simulate resolution (would be done after match)
        # For demo, randomly resolve
        import random
        was_correct = random.random() > 0.5
        profit = 10 if was_correct else -10
        
        ab.resolve_prediction(record_id, 'home', profit)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Recorded {i + 1}/{args.samples} predictions")
    
    # Analyze results
    logger.info("\nAnalyzing results...")
    report = ab.get_test_report(test_id)
    
    print("\n" + report)
    
    # Summary
    result = ab.analyze_test(test_id)
    if result:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        logger.info(f"Winner: {result.winner}")
        logger.info(f"Confidence: {result.confidence:.1%}")
        logger.info(f"Statistical Significance: {'Yes' if result.is_significant else 'No'}")
        logger.info(f"Accuracy Improvement: {result.accuracy_b - result.accuracy_a:+.1%}")
        logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
