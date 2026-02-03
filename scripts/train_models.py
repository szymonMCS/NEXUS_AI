#!/usr/bin/env python3
"""
Train ML models for NEXUS AI.
Usage:
    python scripts/train_models.py --sport tennis
    python scripts/train_models.py --sport basketball --samples 500
"""

import argparse
import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from core.ml import (
    StatisticalEnsemble,
    get_model_registry,
    SKLEARN_AVAILABLE
)

# Try to import ML models
try:
    from core.ml.models.sklearn_models import RandomForestPredictor
except ImportError:
    RandomForestPredictor = None


def generate_sample_matches(sport: str, n_samples: int = 100) -> list:
    """
    Generate sample historical matches for training.
    In production, this would fetch from database/API.
    """
    matches = []
    
    # Sample player/team names by sport
    players_by_sport = {
        "tennis": [
            "Nadal", "Djokovic", "Alcaraz", "Medvedev", "Sinner", "Zverev",
            "Fritz", "Rublev", "Rune", "Tsitsipas", "Berrettini", "FAA",
            "Paul", "Shelton", "Draper", "De Minaur", "Hurkacz", "Cilic"
        ],
        "basketball": [
            "Lakers", "Warriors", "Celtics", "Nets", "Bucks", "Heat",
            "Suns", "Mavericks", "76ers", "Knicks", "Clippers", "Nuggets"
        ]
    }
    
    players = players_by_sport.get(sport, ["Team A", "Team B", "Team C", "Team D"])
    
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        # Random players
        home = random.choice(players)
        away = random.choice([p for p in players if p != home])
        
        # Random date within last year
        match_date = base_date + timedelta(days=random.randint(0, 365))
        
        # Generate realistic scores based on sport
        if sport == "tennis":
            # Tennis: best of 3 or 5 sets
            home_score = random.randint(0, 3)
            away_score = random.randint(0, 3)
            # Ensure not both at max
            if home_score == away_score:
                if random.random() > 0.5:
                    home_score += 1
                else:
                    away_score += 1
        elif sport == "basketball":
            # Basketball: higher scores
            home_score = random.randint(80, 130)
            away_score = random.randint(80, 130)
        else:
            home_score = random.randint(0, 5)
            away_score = random.randint(0, 5)
        
        # Additional features
        match = {
            "match_id": f"{sport}_{i:04d}",
            "home_player": home,
            "away_player": away,
            "home_team": home,  # Alias
            "away_team": away,  # Alias
            "home_score": home_score,
            "away_score": away_score,
            "date": match_date,
            "sport": sport,
            "league": f"{sport.upper()} League",
            # Features that would be available before match
            "home_rank": random.randint(1, 100),
            "away_rank": random.randint(1, 100),
            "home_win_rate": random.uniform(0.4, 0.8),
            "away_win_rate": random.uniform(0.4, 0.8),
        }
        
        matches.append(match)
    
    # Sort by date
    matches.sort(key=lambda x: x["date"])
    
    return matches


def train_statistical_models(sport: str, matches: list) -> StatisticalEnsemble:
    """Train statistical ensemble (ELO + Form + H2H)."""
    print(f"\n{'='*60}")
    print(f"Training Statistical Ensemble for {sport.upper()}")
    print(f"{'='*60}")
    print(f"Training samples: {len(matches)}")
    
    # Create and train model
    model = StatisticalEnsemble(sport=sport)
    model.fit(matches)
    
    # Save to registry
    registry = get_model_registry()
    filepath = registry.save_model(model)
    
    print(f"Model saved: {filepath}")
    print(f"Training complete!")
    
    return model


def train_ml_models(sport: str, matches: list) -> RandomForestPredictor:
    """Train ML models (Random Forest) if sklearn available."""
    if not SKLEARN_AVAILABLE or RandomForestPredictor is None:
        print("\n⚠️  scikit-learn not available, skipping ML training")
        return None
    
    print(f"\n{'='*60}")
    print(f"Training Random Forest for {sport.upper()}")
    print(f"{'='*60}")
    
    # Extract features
    from scripts.train_models import extract_features
    
    X = []
    y = []
    
    for match in matches:
        features = extract_features(match)
        X.append([
            features.get("home_rank", 50) / 100,
            features.get("away_rank", 50) / 100,
            features.get("home_win_rate", 0.5),
            features.get("away_win_rate", 0.5),
            features.get("h2h_advantage", 0),
        ])
        
        # Label: 1 if home wins, 0 if away wins
        if match["home_score"] > match["away_score"]:
            y.append(1)
        else:
            y.append(0)
    
    import numpy as np
    X = np.array(X)
    y = np.array(y)
    
    # Train
    model = RandomForestPredictor(
        name="RandomForest",
        sport=sport,
        n_estimators=100,
        max_depth=10
    )
    
    feature_names = ["home_rank", "away_rank", "home_win_rate", "away_win_rate", "h2h_advantage"]
    model.fit(X, y, feature_names=feature_names)
    
    # Save
    registry = get_model_registry()
    filepath = registry.save_model(model)
    
    print(f"Accuracy: {model.metadata.accuracy:.3f}")
    print(f"Model saved: {filepath}")
    
    return model


def extract_features(match: dict) -> dict:
    """Extract features from match data."""
    return {
        "home_rank": match.get("home_rank", 50),
        "away_rank": match.get("away_rank", 50),
        "home_win_rate": match.get("home_win_rate", 0.5),
        "away_win_rate": match.get("away_win_rate", 0.5),
    }


def test_model(model, sport: str):
    """Test trained model on sample matches."""
    print(f"\n{'='*60}")
    print(f"Testing Model for {sport.upper()}")
    print(f"{'='*60}")
    
    # Create test matches
    test_matches = [
        {
            "home_player": "Player A",
            "away_player": "Player B",
            "home_rank": 5,
            "away_rank": 15,
            "home_win_rate": 0.75,
            "away_win_rate": 0.60,
        },
        {
            "home_player": "Underdog",
            "away_player": "Favorite",
            "home_rank": 50,
            "away_rank": 3,
            "home_win_rate": 0.40,
            "away_win_rate": 0.85,
        },
    ]
    
    for match in test_matches:
        result = model.predict(match)
        print(f"\n{match['home_player']} (rank {match['home_rank']}) vs {match['away_player']} (rank {match['away_rank']})")
        print(f"  Prediction: Home {result.home_win_prob:.1%} vs Away {result.away_win_prob:.1%}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Model: {result.model_name}")


async def main():
    parser = argparse.ArgumentParser(description="Train NEXUS AI models")
    parser.add_argument("--sport", type=str, default="tennis", help="Sport to train")
    parser.add_argument("--samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML models (statistical only)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("NEXUS AI - Model Training")
    print("="*60)
    print(f"Sport: {args.sport}")
    print(f"Samples: {args.samples}")
    
    # Generate training data
    print("\nGenerating training data...")
    matches = generate_sample_matches(args.sport, args.samples)
    
    # Save training data for reference
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / f"{args.sport}_training_data.json", "w") as f:
        json.dump(matches, f, indent=2, default=str)
    
    print(f"Training data saved: data/training/{args.sport}_training_data.json")
    
    # Train statistical models
    stat_model = train_statistical_models(args.sport, matches)
    
    # Train ML models (if available)
    ml_model = None
    if not args.no_ml:
        ml_model = train_ml_models(args.sport, matches)
    
    # Test models
    test_model(stat_model, args.sport)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved in: models/{args.sport}/")
    print("\nYou can now use these models for predictions:")
    print(f"  python -c \"from core.ml import get_prediction_service; svc = get_prediction_service(); print(svc.predict('{args.sport}', 'PlayerA', 'PlayerB'))\"")


if __name__ == "__main__":
    asyncio.run(main())
