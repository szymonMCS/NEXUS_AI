#!/usr/bin/env python3
"""
NEXUS AI v3.0 - Fast Model Training
Optimized version for quicker training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all Excel files"""
    all_data = []
    
    for file in sorted(data_dir.glob("seasons-*.xlsx")):
        logger.info(f"Loading {file.name}...")
        xls = pd.ExcelFile(file)
        
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(file, sheet_name=sheet)
                if len(df) > 0 and 'FTR' in df.columns:
                    df['league'] = sheet
                    df['season'] = file.stem.replace('seasons-', '')
                    all_data.append(df)
            except:
                pass
    
    return pd.concat(all_data, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick feature engineering"""
    logger.info("Engineering features...")
    
    df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Odds features
    if 'AvgH' in df.columns:
        df['odds_home'] = df['AvgH'].fillna(2.0)
        df['odds_draw'] = df['AvgD'].fillna(3.4)
        df['odds_away'] = df['AvgA'].fillna(3.6)
    else:
        df['odds_home'] = df.get('B365H', 2.0)
        df['odds_draw'] = df.get('B365D', 3.4)
        df['odds_away'] = df.get('B365A', 3.6)
    
    # Implied probabilities
    total = (1/df['odds_home'] + 1/df['odds_draw'] + 1/df['odds_away'])
    df['prob_home'] = (1/df['odds_home']) / total
    df['prob_draw'] = (1/df['odds_draw']) / total
    df['prob_away'] = (1/df['odds_away']) / total
    
    # Market confidence (lower = more uncertain)
    df['market_confidence'] = df[['prob_home', 'prob_draw', 'prob_away']].max(axis=1)
    
    # Over/Under odds if available
    if 'Avg>2.5' in df.columns:
        df['over_25_prob'] = 1 / df['Avg>2.5'].fillna(2.0)
        df['under_25_prob'] = 1 / df['Avg<2.5'].fillna(2.0)
    else:
        df['over_25_prob'] = 0.5
        df['under_25_prob'] = 0.5
    
    # Simple team strength proxy (using league average goals)
    df['home_advantage'] = 1.0  # Home team advantage factor
    
    # Goal-based features if available
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        df['home_goals'] = df['FTHG'].fillna(0)
        df['away_goals'] = df['FTAG'].fillna(0)
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['goal_diff'] = df['home_goals'] - df['away_goals']
    else:
        df['home_goals'] = 0
        df['away_goals'] = 0
        df['total_goals'] = 0
        df['goal_diff'] = 0
    
    # Half-time features
    if 'HTHG' in df.columns and 'HTAG' in df.columns:
        df['ht_home'] = df['HTHG'].fillna(0)
        df['ht_away'] = df['HTAG'].fillna(0)
        df['ht_diff'] = df['ht_home'] - df['ht_away']
    else:
        df['ht_home'] = 0
        df['ht_away'] = 0
        df['ht_diff'] = 0
    
    # Stats features (fill with median if missing)
    stats_map = {
        'HS': 'home_shots', 'AS': 'away_shots',
        'HST': 'home_shots_target', 'AST': 'away_shots_target',
        'HC': 'home_corners', 'AC': 'away_corners',
        'HF': 'home_fouls', 'AF': 'away_fouls',
        'HY': 'home_yellow', 'AY': 'away_yellow',
        'HR': 'home_red', 'AR': 'away_red',
    }
    
    for col, name in stats_map.items():
        if col in df.columns:
            df[name] = df[col].fillna(df[col].median())
        else:
            df[name] = 0
    
    return df


def create_feature_matrix(df: pd.DataFrame) -> tuple:
    """Create X, y matrices"""
    
    feature_cols = [
        'odds_home', 'odds_draw', 'odds_away',
        'prob_home', 'prob_draw', 'prob_away',
        'market_confidence', 'over_25_prob', 'under_25_prob',
        'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
        'home_yellow', 'away_yellow', 'home_red', 'away_red',
        'ht_diff', 'total_goals', 'goal_diff'
    ]
    
    # Fill any remaining NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(df[col].median() if df[col].median() == df[col].median() else 0)
    
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df['FTR'])
    
    logger.info(f"Feature matrix: {X.shape}, Classes: {np.bincount(y)}")
    return X, y, feature_cols


def train_models(X_train, y_train, X_val, y_val):
    """Train all models"""
    
    models = {}
    
    # 1. Random Forest
    logger.info("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_train = rf.score(X_train, y_train)
    rf_val = rf.score(X_val, y_val)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=3)
    
    logger.info(f"  Train: {rf_train:.4f}, Val: {rf_val:.4f}, CV: {cv_scores.mean():.4f}")
    
    models['random_forest'] = {
        'model': rf,
        'train_acc': rf_train,
        'val_acc': rf_val,
        'cv_acc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': rf.feature_importances_.tolist()
    }
    
    # 2. MLP with PCA
    logger.info("\n[2/3] Training MLP Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    n_comp = min(15, X_train.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    logger.info(f"  PCA variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )
    
    mlp.fit(X_train_pca, y_train)
    
    mlp_train = mlp.score(X_train_pca, y_train)
    mlp_val = mlp.score(X_val_pca, y_val)
    
    logger.info(f"  Train: {mlp_train:.4f}, Val: {mlp_val:.4f}, Iter: {mlp.n_iter_}")
    
    models['mlp'] = {
        'model': mlp,
        'scaler': scaler,
        'pca': pca,
        'train_acc': mlp_train,
        'val_acc': mlp_val,
        'n_iter': mlp.n_iter_
    }
    
    # 3. Gradient Boosting
    logger.info("\n[3/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    gb.fit(X_train, y_train)
    
    gb_train = gb.score(X_train, y_train)
    gb_val = gb.score(X_val, y_val)
    
    logger.info(f"  Train: {gb_train:.4f}, Val: {gb_val:.4f}")
    
    models['gradient_boosting'] = {
        'model': gb,
        'train_acc': gb_train,
        'val_acc': gb_val
    }
    
    return models


def save_models(models, feature_cols, output_dir: Path):
    """Save trained models"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, data in models.items():
        filepath = output_dir / f"{name}_{timestamp}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved: {filepath}")
    
    # Metadata
    metadata = {
        'timestamp': timestamp,
        'models': list(models.keys()),
        'performance': {
            name: {'train': data['train_acc'], 'val': data['val_acc']}
            for name, data in models.items()
        },
        'features': feature_cols
    }
    
    meta_path = output_dir / f"metadata_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    logger.info("=" * 60)
    logger.info("NEXUS AI v3.0 - Fast Model Training")
    logger.info("=" * 60)
    
    # Load data
    data_dir = Path("data/raw/football_data")
    logger.info(f"\nLoading data from {data_dir}...")
    df = load_data(data_dir)
    logger.info(f"Total matches: {len(df)}")
    
    # Feature engineering
    logger.info("\nFeature engineering...")
    df = engineer_features(df)
    
    # Create matrices
    X, y, feature_cols = create_feature_matrix(df)
    
    # Split
    logger.info("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)
    models = train_models(X_train, y_train, X_val, y_val)
    
    # Test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)
    
    for name, data in models.items():
        if name == 'mlp':
            X_test_scaled = data['scaler'].transform(X_test)
            X_test_pca = data['pca'].transform(X_test_scaled)
            test_acc = data['model'].score(X_test_pca, y_test)
        else:
            test_acc = data['model'].score(X_test, y_test)
        logger.info(f"{name:20s}: {test_acc:.4f}")
        data['test_acc'] = test_acc
    
    # Save
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODELS")
    logger.info("=" * 60)
    metadata = save_models(models, feature_cols, Path("models/trained"))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Random Forest:      {models['random_forest']['test_acc']:.2%}")
    logger.info(f"MLP Neural Net:     {models['mlp']['test_acc']:.2%}")
    logger.info(f"Gradient Boosting:  {models['gradient_boosting']['test_acc']:.2%}")
    
    return models


if __name__ == "__main__":
    models = main()
