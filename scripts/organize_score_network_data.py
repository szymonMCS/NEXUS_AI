#!/usr/bin/env python3
"""
NEXUS AI - ScoreNetworkData Organization & Augmentation
Segregates sports data, splits train/test, performs augmentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
import gzip
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SOURCE_DIR = Path("D:/ScoreNetworkData")
OUTPUT_DIR = Path("data/score_network")
MIN_SAMPLES = 1000  # Minimum samples to keep a discipline

# Discipline mapping based on file patterns
DISCIPLINE_MAPPING = {
    # Tennis
    'tennis': [
        'tennis_atp', 'atp_matches', 'atp_players', 'atp_rankings',
        'wta', 'tennis-m-shots', 'tennis-w-shots', 'AustralianOpen',
        'table_tennis'
    ],
    # Basketball
    'basketball': [
        'nba', 'wnba', 'caitlin_clark', 'nba_draft', 'nba_2223',
        'nba_per100possessions', 'nba_shooting', 'nba_wingspan',
        'nba-player-stats', 'sga_stats', 'nba_players_all_seasons',
        'd3_wsoc2022'  # Division 3 soccer/basketball
    ],
    # American Football
    'american_football': [
        'nfl', 'NFL', 'nfl_combine', 'nfl_mahomes', 'NFLPoints',
        'nfl-team-statistics'
    ],
    # Baseball
    'baseball': [
        'mlb', 'batting', 'pitchers', 'judge_batting', 'betts_batting',
        'ohtani_batting', 'mlb_team', 'mlb-standings', 'mlb_umpires',
        'verlander'
    ],
    # Hockey
    'hockey': [
        'nhl', 'NHL', 'phf-shots', 'pwhl'
    ],
    # Soccer/Football
    'soccer': [
        'soccer', 'epl', 'laliga', 'handball_bundesliga', 'world_cup',
        'international_matches', 'cricket_asia', 'nwsl'
    ],
    # MMA/Fighting
    'mma': [
        'mma', 'UFC', 'BullRiders', 'sumo'
    ],
    # Volleyball
    'volleyball': [
        'volleyball', 'VNL', 'ncaa_waterpolo'
    ],
    # Lacrosse
    'lacrosse': [
        'lacrosse'
    ],
    # Golf
    'golf': [
        'golf', 'DGPT', 'PGA'
    ],
    # Motorsports
    'motorsports': [
        'nascar'
    ],
    # Olympics/Track
    'olympics': [
        'olympic', 'beijing', 'speed_skating', 'rowing', 'diving',
        'racewalking', 'boston_marathon', 'ironman', 'erg'
    ],
    # eSports
    'esports': [
        'lol', 'LOL'
    ],
    # Winter Sports
    'winter_sports': [
        'skating', 'canoe'
    ],
    # Swimming
    'swimming': [
        'swimming'
    ],
    # Gymnastics
    'gymnastics': [
        'gymnastics', 'fencing'
    ],
    # Extreme Sports
    'extreme': [
        'pickleBall'
    ]
}


def identify_discipline(filename):
    """Identify sport discipline from filename"""
    fname_lower = filename.lower()
    
    for discipline, patterns in DISCIPLINE_MAPPING.items():
        for pattern in patterns:
            if pattern.lower() in fname_lower:
                return discipline
    
    return 'other'


def get_file_size_mb(filepath):
    """Get file size in MB"""
    return filepath.stat().st_size / (1024 * 1024)


def load_dataframe(filepath):
    """Load file as DataFrame"""
    try:
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.gz':
            return pd.read_csv(filepath, compression='gzip')
        elif filepath.suffix == '.xlsx':
            return pd.read_excel(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not load {filepath}: {e}")
        return None


def augment_data(df, discipline):
    """Perform data augmentation based on discipline"""
    if df is None or len(df) == 0:
        return df
    
    augmented = [df]
    
    # Noise injection for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        # Add Gaussian noise
        df_noise = df.copy()
        for col in num_cols:
            if df[col].std() > 0:
                noise = np.random.normal(0, df[col].std() * 0.01, len(df))
                df_noise[col] = df_noise[col] + noise
        augmented.append(df_noise)
    
    # Feature engineering based on discipline
    if discipline == 'tennis':
        # Create synthetic rankings variation
        if 'rank' in str(df.columns).lower():
            df_rank = df.copy()
            rank_cols = [c for c in df.columns if 'rank' in c.lower()]
            for col in rank_cols:
                if df_rank[col].dtype in [np.int64, np.float64]:
                    df_rank[col] = df_rank[col] * np.random.uniform(0.95, 1.05, len(df_rank))
            augmented.append(df_rank)
    
    elif discipline == 'basketball':
        # Add performance variations
        if 'points' in str(df.columns).lower() or 'score' in str(df.columns).lower():
            df_perf = df.copy()
            score_cols = [c for c in df.columns if any(x in c.lower() for x in ['points', 'score', 'pts'])]
            for col in score_cols:
                if df_perf[col].dtype in [np.int64, np.float64]:
                    df_perf[col] = (df_perf[col] * np.random.uniform(0.9, 1.1, len(df_perf))).astype(int)
            augmented.append(df_perf)
    
    elif discipline in ['soccer', 'american_football']:
        # Score variation
        if any(x in str(df.columns).lower() for x in ['goal', 'score', 'point']):
            df_score = df.copy()
            goal_cols = [c for c in df.columns if any(x in c.lower() for x in ['goal', 'score', 'point'])]
            for col in goal_cols:
                if df_score[col].dtype in [np.int64, np.float64]:
                    df_score[col] = (df_score[col] * np.random.uniform(0.85, 1.15, len(df_score))).clip(0).astype(int)
            augmented.append(df_score)
    
    # Combine all augmentations
    result = pd.concat(augmented, ignore_index=True)
    logger.info(f"  Augmentation: {len(df)} → {len(result)} samples ({len(result)/len(df):.1f}x)")
    return result


def process_discipline(files, discipline_name, output_path):
    """Process all files for a discipline"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {discipline_name.upper()}")
    logger.info(f"{'='*60}")
    
    all_data = []
    
    for file in files:
        logger.info(f"Loading: {file.name}")
        df = load_dataframe(file)
        if df is not None:
            df['_source_file'] = file.name
            all_data.append(df)
            logger.info(f"  Rows: {len(df)}, Cols: {len(df.columns)}")
    
    if not all_data:
        logger.warning(f"No data loaded for {discipline_name}")
        return None
    
    # Combine all files
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined: {len(combined)} total rows")
    
    # Skip if too small
    if len(combined) < MIN_SAMPLES:
        logger.warning(f"Discipline {discipline_name} has only {len(combined)} samples (< {MIN_SAMPLES}), SKIPPING")
        return None
    
    # Data augmentation
    logger.info("Applying augmentation...")
    augmented = augment_data(combined, discipline_name)
    
    # Split train/test
    train_df, test_df = train_test_split(
        augmented, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Save
    train_path = output_path / f"{discipline_name}_train.csv"
    test_path = output_path / f"{discipline_name}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved: {train_path.name}, {test_path.name}")
    
    return {
        'discipline': discipline_name,
        'original_samples': len(combined),
        'augmented_samples': len(augmented),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': len(combined.columns),
        'files_processed': len(files)
    }


def main():
    """Main organization function"""
    logger.info("="*60)
    logger.info("NEXUS AI - ScoreNetworkData Organization")
    logger.info("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan source directory
    logger.info(f"\nScanning: {SOURCE_DIR}")
    all_files = []
    
    for ext in ['*.csv', '*.csv.gz', '*.xlsx', '*.json']:
        all_files.extend(SOURCE_DIR.rglob(ext))
    
    # Exclude certain files
    exclude_patterns = ['README', 'notes', 'dictionary']
    all_files = [f for f in all_files if not any(p in f.name for p in exclude_patterns)]
    
    logger.info(f"Found {len(all_files)} data files")
    
    # Group by discipline
    discipline_files = {k: [] for k in DISCIPLINE_MAPPING.keys()}
    discipline_files['other'] = []
    
    for file in all_files:
        disc = identify_discipline(file.name)
        discipline_files[disc].append(file)
    
    # Print summary
    logger.info("\nDiscipline Summary:")
    for disc, files in sorted(discipline_files.items(), key=lambda x: -len(x[1])):
        total_size = sum(get_file_size_mb(f) for f in files)
        logger.info(f"  {disc:20s}: {len(files):4d} files, {total_size:8.1f} MB")
    
    # Process each discipline
    results = []
    for disc, files in discipline_files.items():
        if len(files) > 0:
            result = process_discipline(files, disc, OUTPUT_DIR)
            if result:
                results.append(result)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'source_dir': str(SOURCE_DIR),
        'output_dir': str(OUTPUT_DIR),
        'total_files': len(all_files),
        'disciplines_processed': len(results),
        'disciplines': results
    }
    
    summary_path = OUTPUT_DIR / 'organization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("ORGANIZATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Kept disciplines: {len(results)}")
    total_train = sum(r['train_samples'] for r in results)
    total_test = sum(r['test_samples'] for r in results)
    logger.info(f"Total samples: Train={total_train:,}, Test={total_test:,}")
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
