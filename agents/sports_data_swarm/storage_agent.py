"""
Storage Agent - Saves formatted data to datasets for ML training.
Supports multiple output formats: CSV, JSON, Parquet.
"""

import os
import json
import csv
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

# Import TaskResult from base_agent
try:
    from base_agent import TaskResult
except ImportError:
    # For inline imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from base_agent import TaskResult

logger = logging.getLogger(__name__)


class StorageAgent:
    """
    Agent responsible for storing processed sports data.
    
    Features:
    - Multiple output formats (CSV, JSON, Parquet)
    - Automatic dataset versioning
    - Train/test split generation
    - Metadata tracking
    - Compression support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "StorageAgent"
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'datasets/sports_data')
        self.format = self.config.get('format', 'csv')
        self.split_ratio = self.config.get('split_ratio', 0.8)  # Train/test split
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute storage task.
        
        Task params:
        - sport: Sport type
        - data: Formatted data to save
        - output_dir: Output directory
        - format: Output format (csv, json, parquet)
        - split: Whether to create train/test split
        """
        try:
            sport = task['sport']
            data = task.get('data', {})
            output_dir = task.get('output_dir', self.output_dir)
            fmt = task.get('format', self.format)
            create_split = task.get('split', True)
            
            records = data.get('records', [])
            
            if not records:
                return TaskResult(success=False, error="No records to save")
            
            logger.info(f"[{self.name}] Saving {len(records)} {sport} records")
            
            # Create directories
            raw_dir = Path(output_dir) / 'raw'
            processed_dir = Path(output_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save full dataset
            filename_base = f"{sport}_dataset_{timestamp}"
            
            if fmt == 'csv':
                filepath = await self._save_csv(records, processed_dir, filename_base)
            elif fmt == 'json':
                filepath = await self._save_json(records, processed_dir, filename_base)
            elif fmt == 'parquet':
                filepath = await self._save_parquet(records, processed_dir, filename_base)
            else:
                return TaskResult(success=False, error=f"Unsupported format: {fmt}")
            
            # Create train/test split if requested
            split_info = {}
            if create_split and len(records) > 100:
                split_info = await self._create_train_test_split(
                    records, processed_dir, filename_base, fmt
                )
            
            # Save metadata
            metadata = {
                'sport': sport,
                'created_at': datetime.now().isoformat(),
                'record_count': len(records),
                'format': fmt,
                'filepath': str(filepath),
                'schema_compliance': data.get('compliance', {}),
                'train_test_split': split_info,
                'fields': list(records[0].keys()) if records else []
            }
            
            meta_path = processed_dir / f"{filename_base}_metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"[{self.name}] Saved to {filepath}")
            logger.info(f"[{self.name}] Metadata saved to {meta_path}")
            
            return TaskResult(
                success=True,
                data={
                    'filepath': str(filepath),
                    'metadata_path': str(meta_path),
                    'record_count': len(records),
                    'format': fmt
                },
                records_processed=len(records),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return TaskResult(success=False, error=str(e))
    
    async def _save_csv(self, records: List[Dict], directory: Path, filename_base: str) -> Path:
        """Save records as CSV."""
        filepath = directory / f"{filename_base}.csv"
        
        if not records:
            return filepath
        
        # Get all unique fields
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())
        
        # Prioritize fields
        priority_fields = ['game_id', 'match_id', 'date', 'home_team', 'away_team', 
                          'player1_name', 'player2_name', 'home_score', 'away_score',
                          'home_win', 'player1_win', 'league', 'tournament']
        
        fieldnames = [f for f in priority_fields if f in all_fields]
        fieldnames.extend(sorted(all_fields - set(priority_fields)))
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        return filepath
    
    async def _save_json(self, records: List[Dict], directory: Path, filename_base: str) -> Path:
        """Save records as JSON."""
        filepath = directory / f"{filename_base}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, default=str)
        
        return filepath
    
    async def _save_parquet(self, records: List[Dict], directory: Path, filename_base: str) -> Path:
        """Save records as Parquet (requires pandas)."""
        try:
            import pandas as pd
            filepath = directory / f"{filename_base}.parquet"
            
            df = pd.DataFrame(records)
            df.to_parquet(filepath, index=False, compression='snappy')
            
            return filepath
        except ImportError:
            logger.warning("Pandas not available, falling back to CSV")
            return await self._save_csv(records, directory, filename_base)
    
    async def _create_train_test_split(self, records: List[Dict], directory: Path, 
                                        filename_base: str, fmt: str) -> Dict:
        """Create train/test split."""
        import random
        
        # Shuffle and split
        shuffled = records.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * self.split_ratio)
        train_records = shuffled[:split_idx]
        test_records = shuffled[split_idx:]
        
        # Save splits
        if fmt == 'csv':
            train_path = directory / f"{filename_base}_train.csv"
            test_path = directory / f"{filename_base}_test.csv"
            
            all_fields = set()
            for r in records:
                all_fields.update(r.keys())
            fieldnames = sorted(all_fields)
            
            for path, data in [(train_path, train_records), (test_path, test_records)]:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
                    
        elif fmt == 'json':
            train_path = directory / f"{filename_base}_train.json"
            test_path = directory / f"{filename_base}_test.json"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_records, f, default=str)
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(test_records, f, default=str)
        
        logger.info(f"[{self.name}] Train set: {len(train_records)}, Test set: {len(test_records)}")
        
        return {
            'train_count': len(train_records),
            'test_count': len(test_records),
            'train_path': str(directory / f"{filename_base}_train.{fmt}"),
            'test_path': str(directory / f"{filename_base}_test.{fmt}"),
            'split_ratio': self.split_ratio
        }


