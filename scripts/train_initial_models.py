"""
Train Initial ML Models.

Checkpoint: 5.5
Run: python scripts/train_initial_models.py [--data-dir DIR] [--output-dir DIR]

This script:
1. Loads historical match data
2. Converts to training examples
3. Trains Goals Model (Poisson regression)
4. Trains Handicap Model (Gradient Boosting)
5. Saves models to registry

Examples:
    # Train with default settings
    python scripts/train_initial_models.py

    # Train with custom data directory
    python scripts/train_initial_models.py --data-dir data/historical --output-dir models/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from core.data.enums import Sport
from core.ml.training.examples import TrainingExample, TrainingBatch, TrainingConfig
from core.ml.models.goals_model import GoalsModel
from core.ml.models.handicap_model import HandicapModel
from core.ml.registry import ModelRegistry, ModelVersion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train initial ML models from historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/historical",
        help="Directory containing historical data files (default: data/historical)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required to train (default: 100)"
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )

    parser.add_argument(
        "--sport",
        type=str,
        choices=["football", "basketball", "all"],
        default="football",
        help="Sport to train models for (default: football)"
    )

    parser.add_argument(
        "--skip-goals",
        action="store_true",
        help="Skip training goals model"
    )

    parser.add_argument(
        "--skip-handicap",
        action="store_true",
        help="Skip training handicap model"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show stats without training"
    )

    return parser.parse_args()


def load_historical_data(data_dir: Path, sport_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load historical match data from JSON files."""
    matches = []

    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return matches

    for filepath in data_dir.glob("*.json"):
        if filepath.name == "collection_metadata.json":
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both formats: {matches: []} and direct list
            file_matches = data.get("matches", data) if isinstance(data, dict) else data

            if not isinstance(file_matches, list):
                continue

            # Filter by sport if specified
            for match in file_matches:
                if sport_filter and match.get("sport") != sport_filter:
                    continue
                matches.append(match)

            logger.info(f"Loaded {len(file_matches)} matches from {filepath.name}")

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    return matches


def convert_to_training_examples(matches: List[Dict[str, Any]]) -> TrainingBatch:
    """Convert historical matches to training examples."""
    batch = TrainingBatch(source="historical_data")

    for match in matches:
        try:
            # Create features from match data
            features = extract_features_from_match(match)

            example = TrainingExample(
                example_id=f"hist_{match.get('match_id', '')}",
                match_id=match.get("match_id", ""),
                features=features,
                actual_home_goals=match.get("home_goals"),
                actual_away_goals=match.get("away_goals"),
            )

            if example.is_complete:
                batch.add(example)

        except Exception as e:
            logger.debug(f"Error converting match: {e}")
            continue

    return batch


def extract_features_from_match(match: Dict[str, Any]) -> Dict[str, float]:
    """Extract feature dictionary from a historical match."""
    features = {}

    # Basic features we can compute from the match itself
    home_goals = match.get("home_goals", 0) or 0
    away_goals = match.get("away_goals", 0) or 0

    # Shot-based features (if available)
    home_shots = match.get("home_shots")
    away_shots = match.get("away_shots")
    if home_shots is not None and away_shots is not None:
        features["home_shots"] = float(home_shots)
        features["away_shots"] = float(away_shots)
        features["shot_ratio"] = home_shots / max(1, away_shots)

    home_sot = match.get("home_shots_on_target")
    away_sot = match.get("away_shots_on_target")
    if home_sot is not None and away_sot is not None:
        features["home_shots_on_target"] = float(home_sot)
        features["away_shots_on_target"] = float(away_sot)
        if home_shots:
            features["home_shot_accuracy"] = home_sot / max(1, home_shots)
        if away_shots:
            features["away_shot_accuracy"] = away_sot / max(1, away_shots)

    # Possession
    home_poss = match.get("home_possession")
    away_poss = match.get("away_possession")
    if home_poss is not None:
        features["home_possession"] = float(home_poss)
        features["away_possession"] = float(away_poss or (1 - home_poss))

    # Corners
    home_corners = match.get("home_corners")
    away_corners = match.get("away_corners")
    if home_corners is not None:
        features["home_corners"] = float(home_corners)
        features["away_corners"] = float(away_corners or 0)

    # Cards
    home_yellow = match.get("home_yellow_cards")
    away_yellow = match.get("away_yellow_cards")
    if home_yellow is not None:
        features["home_yellow_cards"] = float(home_yellow)
        features["away_yellow_cards"] = float(away_yellow or 0)

    # Odds-based features
    odds_home = match.get("odds_home")
    odds_draw = match.get("odds_draw")
    odds_away = match.get("odds_away")
    if odds_home and odds_draw and odds_away:
        features["odds_home"] = float(odds_home)
        features["odds_draw"] = float(odds_draw)
        features["odds_away"] = float(odds_away)
        # Implied probabilities
        total_prob = 1/odds_home + 1/odds_draw + 1/odds_away
        features["implied_prob_home"] = (1/odds_home) / total_prob
        features["implied_prob_draw"] = (1/odds_draw) / total_prob
        features["implied_prob_away"] = (1/odds_away) / total_prob

    odds_over = match.get("odds_over_25")
    odds_under = match.get("odds_under_25")
    if odds_over and odds_under:
        features["odds_over_25"] = float(odds_over)
        features["odds_under_25"] = float(odds_under)

    # If we have very few features, add some dummy features for testing
    if len(features) < 3:
        features["home_strength"] = 0.5 + random.uniform(-0.2, 0.2)
        features["away_strength"] = 0.5 + random.uniform(-0.2, 0.2)
        features["league_avg_goals"] = 2.5 + random.uniform(-0.5, 0.5)

    return features


def split_data(
    batch: TrainingBatch,
    validation_split: float = 0.2,
    shuffle: bool = True,
) -> Tuple[TrainingBatch, TrainingBatch]:
    """Split batch into train and validation sets."""
    examples = batch.get_complete_examples()

    if shuffle:
        random.shuffle(examples)

    split_idx = int(len(examples) * (1 - validation_split))

    train_batch = TrainingBatch(source=f"{batch.source}_train")
    train_batch.examples = examples[:split_idx]

    val_batch = TrainingBatch(source=f"{batch.source}_val")
    val_batch.examples = examples[split_idx:]

    return train_batch, val_batch


def train_goals_model(
    train_batch: TrainingBatch,
    val_batch: TrainingBatch,
    output_dir: Path,
) -> Optional[GoalsModel]:
    """Train and evaluate goals model."""
    logger.info("Training Goals Model (Poisson Regression)...")

    model = GoalsModel()

    # Get feature vectors from training batch
    if not train_batch.examples:
        logger.error("No training examples!")
        return None

    # Convert to FeatureVector objects (model expects List[FeatureVector])
    X_train = train_batch.get_feature_vectors()
    y_train = train_batch.get_goals_targets()

    if not X_train or not y_train:
        logger.error("No valid training data!")
        return None

    feature_names = X_train[0].get_feature_names() if X_train else []
    logger.info(f"Using {len(feature_names)} features: {feature_names[:5]}...")

    # Train model
    try:
        model.train(X_train, y_train, validation_split=0.2)
        logger.info("Goals model trained successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

    # Evaluate on validation set
    if val_batch.examples:
        X_val = val_batch.get_feature_vectors()
        y_val = val_batch.get_goals_targets()

        correct = 0
        total = 0

        for features, (actual_home, actual_away) in zip(X_val, y_val):
            pred = model.predict(features)
            if pred:
                # Check if over/under 2.5 prediction is correct
                actual_total = actual_home + actual_away
                predicted_over = pred.over_25_prob > 0.5
                actual_over = actual_total > 2.5

                if predicted_over == actual_over:
                    correct += 1
                total += 1

        if total > 0:
            accuracy = correct / total
            logger.info(f"Validation accuracy (over/under 2.5): {accuracy:.2%}")

    # Save model (pass Path object, not string)
    model_path = output_dir / "goals_model.json"
    model.save(model_path)
    logger.info(f"Goals model saved to {model_path}")

    return model


def train_handicap_model(
    train_batch: TrainingBatch,
    val_batch: TrainingBatch,
    output_dir: Path,
) -> Optional[HandicapModel]:
    """Train and evaluate handicap model."""
    logger.info("Training Handicap Model (Gradient Boosting)...")

    model = HandicapModel()

    # Get feature vectors
    if not train_batch.examples:
        logger.error("No training examples!")
        return None

    # Convert to FeatureVector objects (model expects List[FeatureVector])
    X_train = train_batch.get_feature_vectors()
    y_train = train_batch.get_margin_targets()

    if not X_train or not y_train:
        logger.error("No valid training data!")
        return None

    feature_names = X_train[0].get_feature_names() if X_train else []
    logger.info(f"Using {len(feature_names)} features for handicap model")

    # Train model
    try:
        model.train(X_train, y_train, validation_split=0.2)
        logger.info("Handicap model trained successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

    # Evaluate on validation set
    if val_batch.examples:
        X_val = val_batch.get_feature_vectors()
        y_val = val_batch.get_margin_targets()

        correct = 0
        total = 0

        for features, actual_margin in zip(X_val, y_val):
            pred = model.predict(features)
            if pred:
                # Check if result direction prediction is correct
                predicted_home_win = pred.expected_margin > 0.5
                actual_home_win = actual_margin > 0

                if predicted_home_win == actual_home_win:
                    correct += 1
                total += 1

        if total > 0:
            accuracy = correct / total
            logger.info(f"Validation accuracy (home/away winner): {accuracy:.2%}")

    # Save model (pass Path object, not string)
    model_path = output_dir / "handicap_model.json"
    model.save(model_path)
    logger.info(f"Handicap model saved to {model_path}")

    return model


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("NEXUS ML Model Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    sport_filter = None if args.sport == "all" else args.sport
    matches = load_historical_data(data_dir, sport_filter)

    if not matches:
        print("\n[ERROR] No historical data found!")
        print(f"Run: python scripts/collect_historical.py --sport {args.sport}")
        return

    print(f"\nLoaded {len(matches)} matches")

    # Convert to training examples
    batch = convert_to_training_examples(matches)
    complete_examples = batch.get_complete_examples()

    print(f"Converted to {len(complete_examples)} complete training examples")

    if len(complete_examples) < args.min_samples:
        print(f"\n[ERROR] Not enough samples (need {args.min_samples}, have {len(complete_examples)})")
        print("Collect more data or reduce --min-samples")
        return

    # Show data stats
    print("\n--- Data Statistics ---")
    if complete_examples:
        sample = complete_examples[0]
        print(f"Features per example: {len(sample.features)}")
        print(f"Feature names: {list(sample.features.keys())[:10]}...")

        # Goal distribution
        total_goals = [e.actual_total_goals for e in complete_examples if e.actual_total_goals is not None]
        if total_goals:
            avg_goals = sum(total_goals) / len(total_goals)
            over_25_pct = sum(1 for g in total_goals if g > 2.5) / len(total_goals)
            print(f"Average total goals: {avg_goals:.2f}")
            print(f"Over 2.5 goals: {over_25_pct:.1%}")

    if args.dry_run:
        print("\n[DRY RUN] Training skipped")
        return

    # Split data
    train_batch, val_batch = split_data(batch, args.validation_split)
    print(f"\nTraining set: {len(train_batch)} examples")
    print(f"Validation set: {len(val_batch)} examples")

    # Train models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    models_trained = []

    if not args.skip_goals:
        goals_model = train_goals_model(train_batch, val_batch, output_dir)
        if goals_model:
            models_trained.append("goals_model")

    if not args.skip_handicap:
        handicap_model = train_handicap_model(train_batch, val_batch, output_dir)
        if handicap_model:
            models_trained.append("handicap_model")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models trained: {', '.join(models_trained) or 'none'}")
    print(f"Output directory: {output_dir}")

    # Save training metadata
    metadata = {
        "training_date": datetime.utcnow().isoformat(),
        "data_directory": str(data_dir),
        "total_matches": len(matches),
        "training_examples": len(train_batch),
        "validation_examples": len(val_batch),
        "models_trained": models_trained,
        "sport_filter": sport_filter,
    }

    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
