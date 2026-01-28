#!/usr/bin/env python3
"""
NEXUS AI - Integrate ScoreNetworkData into main project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime

NEXUS_DATA_DIR = Path("data/sports")
SCORE_NETWORK_DIR = Path("data/score_network")

def integrate_to_nexus():
    """Copy and integrate ScoreNetwork data to NEXUS structure"""
    print("="*60)
    print("INTEGRATING SCORE NETWORK DATA TO NEXUS AI")
    print("="*60)
    
    # Create sport directories
    sport_mapping = {
        'tennis': 'tennis',
        'basketball': 'basketball',
        'american_football': 'american_football',
        'baseball': 'baseball',
        'hockey': 'hockey',
        'soccer': 'soccer',
        'mma': 'mma',
        'olympics': 'olympics'
    }
    
    total_train = 0
    total_test = 0
    
    for score_discipline, nexus_sport in sport_mapping.items():
        train_file = SCORE_NETWORK_DIR / f"{score_discipline}_train.csv"
        test_file = SCORE_NETWORK_DIR / f"{score_discipline}_test.csv"
        
        if not train_file.exists():
            continue
        
        # Create directory
        sport_dir = NEXUS_DATA_DIR / nexus_sport / "score_network"
        sport_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        shutil.copy2(train_file, sport_dir / "train.csv")
        shutil.copy2(test_file, sport_dir / "test.csv")
        
        # Count samples
        train_df = pd.read_csv(train_file, nrows=1000)
        test_df = pd.read_csv(test_file, nrows=1000)
        
        train_rows = len(pd.read_csv(train_file, usecols=[0]))
        test_rows = len(pd.read_csv(test_file, usecols=[0]))
        
        total_train += train_rows
        total_test += test_rows
        
        print(f"[OK] {nexus_sport:20s}: Train={train_rows:7,}, Test={test_rows:7,}")
    
    print("="*60)
    print(f"TOTAL: Train={total_train:,}, Test={total_test:,}")
    print(f"Data location: {NEXUS_DATA_DIR}")
    
    # Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source': 'ScoreNetworkData (D:\\ScoreNetworkData)',
        'sports': list(sport_mapping.values()),
        'total_train': total_train,
        'total_test': total_test,
        'note': 'Integrated from external dataset'
    }
    
    with open(NEXUS_DATA_DIR / 'score_network_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {NEXUS_DATA_DIR / 'score_network_metadata.json'}")

if __name__ == "__main__":
    integrate_to_nexus()
