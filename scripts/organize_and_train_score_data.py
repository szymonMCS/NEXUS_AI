#!/usr/bin/env python3
"""
NEXUS AI - ScoreNetworkData Organization, Augmentation & Training
Processes D:\ScoreNetworkData with limits per discipline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import gzip
from datetime import datetime
from sklearn.model_selection import train_test_split
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

# Configuration
SOURCE_DIR = Path("D:/ScoreNetworkData")
OUTPUT_DIR = Path("data/score_network")
MAX_SAMPLES_PER_DISCIPLINE = 500000  # Limit samples
MIN_SAMPLES = 5000  # Minimum to keep discipline

DISCIPLINE_MAPPING = {
    'tennis': ['tennis_atp', 'atp_matches', 'wta', 'tennis-m-shots', 'tennis-w-shots', 'AustralianOpen', 'table_tennis'],
    'basketball': ['nba', 'wnba', 'nba_draft', 'nba_2223', 'nba_per100possessions', 'nba_shooting', 'nba_wingspan', 'nba-player-stats', 'sga_stats', 'nba_players_all_seasons'],
    'american_football': ['nfl', 'NFL', 'nfl_combine', 'nfl_mahomes', 'NFLPoints', 'nfl-team-statistics'],
    'baseball': ['mlb', 'batting', 'pitchers', 'judge_batting', 'betts_batting', 'ohtani_batting', 'mlb_team', 'mlb-standings', 'mlb_umpires', 'verlander'],
    'hockey': ['nhl', 'NHL', 'phf-shots', 'pwhl'],
    'soccer': ['soccer', 'epl', 'laliga', 'handball_bundesliga', 'world_cup', 'international_matches', 'nwsl'],
    'mma': ['mma', 'UFC', 'BullRiders', 'sumo'],
    'volleyball': ['volleyball', 'VNL'],
    'lacrosse': ['lacrosse'],
    'golf': ['golf', 'DGPT', 'PGA'],
    'esports': ['lol', 'LOL'],
    'motorsports': ['nascar'],
    'olympics': ['olympic', 'beijing', 'speed_skating', 'rowing', 'diving', 'racewalking', 'boston_marathon', 'ironman'],
}


def identify_discipline(filename):
    fname_lower = filename.lower()
    for discipline, patterns in DISCIPLINE_MAPPING.items():
        for pattern in patterns:
            if pattern.lower() in fname_lower:
                return discipline
    return None  # Skip "other"


def load_sample(filepath, max_rows=50000):
    """Load file with row limit"""
    try:
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath, nrows=max_rows, low_memory=False)
        elif filepath.suffix == '.gz':
            return pd.read_csv(filepath, compression='gzip', nrows=max_rows, low_memory=False)
        elif filepath.suffix == '.xlsx':
            return pd.read_excel(filepath, nrows=max_rows)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
    except Exception as e:
        logger.warning(f"  Could not load {filepath.name}: {e}")
        return None


def augment_data(df, discipline):
    """Quick augmentation"""
    if df is None or len(df) < 100:
        return df
    
    augmented = [df]
    num_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit columns
    
    if len(num_cols) > 0:
        df_noise = df.copy()
        for col in num_cols:
            if df[col].std() > 0 and df[col].dtype in ['float64', 'int64']:
                noise = np.random.normal(0, df[col].std() * 0.01, len(df))
                df_noise[col] = df_noise[col] + noise
        augmented.append(df_noise)
    
    return pd.concat(augmented, ignore_index=True)


def create_simple_features(df, discipline):
    """Create simple features for ML"""
    df = df.copy()
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to useful columns (exclude IDs, indices)
    exclude_patterns = ['id', 'index', 'unnamed', '_source']
    feature_cols = [c for c in numeric_cols if not any(p in c.lower() for p in exclude_patterns)]
    
    # Limit to top 30 features by variance
    if len(feature_cols) > 30:
        variances = df[feature_cols].var()
        feature_cols = variances.nlargest(30).index.tolist()
    
    # Fill NaN
    df_features = df[feature_cols].fillna(0)
    
    return df_features, feature_cols


def train_discipline_model(train_df, test_df, discipline):
    """Train models for a discipline"""
    logger.info(f"\nTraining models for {discipline}...")
    
    # Create features
    X_train, feature_cols = create_simple_features(train_df, discipline)
    X_test, _ = create_simple_features(test_df, discipline)
    
    # Create synthetic target (clustering-like)
    # Use mean of features as target proxy for demonstration
    y_train = (X_train.iloc[:, :3].mean(axis=1) > X_train.iloc[:, :3].mean(axis=1).median()).astype(int)
    y_test = (X_test.iloc[:, :3].mean(axis=1) > X_test.iloc[:, :3].mean(axis=1).median()).astype(int)
    
    if len(np.unique(y_train)) < 2:
        logger.warning(f"  Only one class, skipping training")
        return None
    
    results = {}
    
    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)
        results['random_forest'] = {'accuracy': rf_acc, 'model': rf}
        logger.info(f"  Random Forest: {rf_acc:.4f}")
    except Exception as e:
        logger.error(f"  RF error: {e}")
    
    # MLP
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=min(10, X_train.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
        mlp.fit(X_train_pca, y_train)
        mlp_acc = mlp.score(X_test_pca, y_test)
        results['mlp'] = {'accuracy': mlp_acc, 'model': mlp, 'scaler': scaler, 'pca': pca}
        logger.info(f"  MLP: {mlp_acc:.4f}")
    except Exception as e:
        logger.error(f"  MLP error: {e}")
    
    return results


def process_discipline(files, discipline_name, output_path):
    """Process files for one discipline"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {discipline_name.upper()}")
    logger.info(f"{'='*60}")
    
    all_data = []
    total_loaded = 0
    
    for file in files:
        if total_loaded >= MAX_SAMPLES_PER_DISCIPLINE:
            break
        
        remaining = MAX_SAMPLES_PER_DISCIPLINE - total_loaded
        df = load_sample(file, max_rows=min(50000, remaining))
        
        if df is not None and len(df) > 0:
            df['_source_file'] = file.name
            all_data.append(df)
            total_loaded += len(df)
            logger.info(f"  {file.name[:50]:50s} | Rows: {len(df):6d} | Total: {total_loaded:7d}")
    
    if not all_data or total_loaded < MIN_SAMPLES:
        logger.warning(f"  SKIPPED: Only {total_loaded} samples (< {MIN_SAMPLES})")
        return None
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} rows")
    
    # Augment
    augmented = augment_data(combined, discipline_name)
    logger.info(f"After augmentation: {len(augmented):,} rows")
    
    # Split
    train_df, test_df = train_test_split(augmented, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Save
    train_path = output_path / f"{discipline_name}_train.csv"
    test_path = output_path / f"{discipline_name}_test.csv"
    
    # Save sample (not full for large datasets)
    save_limit = 100000
    train_df.head(save_limit).to_csv(train_path, index=False)
    test_df.head(save_limit//4).to_csv(test_path, index=False)
    
    # Train models
    model_results = train_discipline_model(train_df, test_df, discipline_name)
    
    return {
        'discipline': discipline_name,
        'original_samples': len(combined),
        'augmented_samples': len(augmented),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': len(combined.columns),
        'model_results': {k: {'accuracy': v['accuracy']} for k, v in (model_results or {}).items()}
    }


def main():
    logger.info("="*60)
    logger.info("NEXUS AI - ScoreNetworkData Pipeline")
    logger.info("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan files
    logger.info(f"\nScanning {SOURCE_DIR}...")
    all_files = []
    for ext in ['*.csv', '*.csv.gz', '*.xlsx', '*.json']:
        all_files.extend(SOURCE_DIR.rglob(ext))
    
    exclude = ['README', 'notes', 'dictionary', 'atp_rankings', 'atp_players', 'players_till']
    all_files = [f for f in all_files if not any(p in f.name for p in exclude)]
    
    logger.info(f"Found {len(all_files)} files")
    
    # Group by discipline
    discipline_files = {k: [] for k in DISCIPLINE_MAPPING.keys()}
    skipped = 0
    
    for file in all_files:
        disc = identify_discipline(file.name)
        if disc:
            discipline_files[disc].append(file)
        else:
            skipped += 1
    
    logger.info(f"Skipped {skipped} files (no discipline match)")
    
    # Print summary
    for disc, files in sorted(discipline_files.items(), key=lambda x: -len(x[1])):
        total_mb = sum(f.stat().st_size for f in files) / (1024*1024)
        logger.info(f"  {disc:20s}: {len(files):3d} files, {total_mb:8.1f} MB")
    
    # Process each discipline
    results = []
    for disc, files in discipline_files.items():
        if len(files) > 0:
            result = process_discipline(files, disc, OUTPUT_DIR)
            if result:
                results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Disciplines kept: {len(results)}")
    
    for r in results:
        logger.info(f"\n{r['discipline'].upper()}:")
        logger.info(f"  Samples: {r['augmented_samples']:,} (augmented)")
        logger.info(f"  Train: {r['train_samples']:,}, Test: {r['test_samples']:,}")
        for model, metrics in r['model_results'].items():
            logger.info(f"  {model}: {metrics['accuracy']:.4f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'disciplines': results
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
