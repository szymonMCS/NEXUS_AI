#!/usr/bin/env python3
"""
NEXUS AI - Train models on ScoreNetworkData
Properly handles feature alignment between train/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/score_network")
OUTPUT_DIR = Path("models/score_network")


def load_discipline_data(discipline):
    """Load train and test data for a discipline"""
    train_path = DATA_DIR / f"{discipline}_train.csv"
    test_path = DATA_DIR / f"{discipline}_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Prepare aligned features for train and test"""
    # Remove non-numeric and metadata columns
    exclude_cols = ['_source_file', 'Unnamed', 'id', 'index', 'date', 'name', 'player', 'team']
    
    # Get numeric columns only
    train_numeric = train_df.select_dtypes(include=[np.number])
    test_numeric = test_df.select_dtypes(include=[np.number])
    
    # Find common columns
    common_cols = list(set(train_numeric.columns) & set(test_numeric.columns))
    common_cols = [c for c in common_cols if not any(x in c.lower() for x in exclude_cols)]
    
    if len(common_cols) < 3:
        logger.warning(f"Too few common features: {len(common_cols)}")
        return None, None, None
    
    # Limit to top features by variance in training set
    if len(common_cols) > 25:
        variances = train_df[common_cols].var()
        common_cols = variances.nlargest(25).index.tolist()
    
    # Extract features
    X_train = train_df[common_cols].fillna(0).values
    X_test = test_df[common_cols].fillna(0).values
    
    # Create synthetic target (binary classification based on mean of first 3 features)
    # This is for demonstration - in real scenario you'd have actual targets
    threshold_train = np.mean(X_train[:, :3])
    threshold_test = np.mean(X_test[:, :3])
    
    y_train = (X_train[:, 0] > threshold_train).astype(int)
    y_test = (X_test[:, 0] > threshold_test).astype(int)
    
    # Ensure we have both classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # Use random target as fallback
        np.random.seed(42)
        y_train = np.random.randint(0, 2, len(X_train))
        y_test = np.random.randint(0, 2, len(X_test))
    
    return X_train, X_test, y_train, y_test, common_cols


def train_models(X_train, X_test, y_train, y_test, discipline):
    """Train models for a discipline"""
    logger.info(f"\nTraining models for {discipline}...")
    logger.info(f"Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # Random Forest
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        results['random_forest'] = {
            'model': rf,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
        }
        logger.info(f"  Random Forest: Train={train_acc:.4f}, Test={test_acc:.4f}")
    except Exception as e:
        logger.error(f"  RF error: {e}")
    
    # MLP
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        n_components = min(15, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        
        mlp.fit(X_train_pca, y_train)
        
        train_acc = mlp.score(X_train_pca, y_train)
        test_acc = mlp.score(X_test_pca, y_test)
        
        results['mlp'] = {
            'model': mlp,
            'scaler': scaler,
            'pca': pca,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
        }
        logger.info(f"  MLP: Train={train_acc:.4f}, Test={test_acc:.4f}")
    except Exception as e:
        logger.error(f"  MLP error: {e}")
    
    return results


def save_models(discipline, models, feature_cols, output_dir):
    """Save trained models"""
    disc_dir = output_dir / discipline
    disc_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_data in models.items():
        filepath = disc_dir / f"{model_name}_{timestamp}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"  Saved: {filepath.name}")
    
    # Save feature list
    with open(disc_dir / f"features_{timestamp}.json", 'w') as f:
        json.dump({'feature_names': feature_cols}, f)
    
    return timestamp


def main():
    logger.info("="*60)
    logger.info("NEXUS AI - ScoreNetworkData Model Training")
    logger.info("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all disciplines
    disciplines = set()
    for f in DATA_DIR.glob("*_train.csv"):
        disc = f.stem.replace("_train", "")
        disciplines.add(disc)
    
    logger.info(f"Found {len(disciplines)} disciplines")
    
    all_results = []
    
    for discipline in sorted(disciplines):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {discipline.upper()}")
        logger.info(f"{'='*60}")
        
        # Load data
        train_df, test_df = load_discipline_data(discipline)
        if train_df is None:
            logger.warning(f"No data found for {discipline}")
            continue
        
        logger.info(f"Loaded: Train={len(train_df)}, Test={len(test_df)}")
        
        # Prepare features
        prepared = prepare_features(train_df, test_df)
        if prepared[0] is None:
            logger.warning(f"Could not prepare features for {discipline}")
            continue
        
        X_train, X_test, y_train, y_test, feature_cols = prepared
        
        # Train models
        models = train_models(X_train, X_test, y_train, y_test, discipline)
        
        if not models:
            logger.warning(f"No models trained for {discipline}")
            continue
        
        # Save models
        timestamp = save_models(discipline, models, feature_cols, OUTPUT_DIR)
        
        # Record results
        result = {
            'discipline': discipline,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(feature_cols),
            'models': {
                name: {
                    'train_accuracy': data['train_accuracy'],
                    'test_accuracy': data['test_accuracy']
                }
                for name, data in models.items()
            }
        }
        all_results.append(result)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    
    for r in all_results:
        logger.info(f"\n{r['discipline'].upper()}:")
        logger.info(f"  Samples: {r['train_samples']:,} train, {r['test_samples']:,} test")
        logger.info(f"  Features: {r['features']}")
        for model_name, metrics in r['models'].items():
            logger.info(f"  {model_name}: Test={metrics['test_accuracy']:.4f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'disciplines': all_results
    }
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nModels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
