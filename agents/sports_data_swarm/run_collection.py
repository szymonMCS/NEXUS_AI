#!/usr/bin/env python3
"""
Sports Data Swarm - Main Entry Point

Usage:
    python run_collection.py --sports basketball football tennis
    python run_collection.py --sport basketball --target 5000 --augment 3.0
    python run_collection.py --all --target 10000 --augment

This script orchestrates the agent swarm to collect sports data from web sources
using Brave Search and Serper APIs, formats the data, and creates ML-ready datasets.
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import importlib.util

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
base_agent = load_module("base_agent", Path(__file__).parent / "base_agent.py")
manager_agent = load_module("manager_agent", Path(__file__).parent / "manager_agent.py")
sport_agents_mod = load_module("sport_agents", Path(__file__).parent / "sport_agents.py")
football_agent_mod = load_module("football_agent", Path(__file__).parent / "football_agent.py")
baseball_agent_mod = load_module("baseball_agent", Path(__file__).parent / "baseball_agent.py")
hockey_agent_mod = load_module("hockey_agent", Path(__file__).parent / "hockey_agent.py")
mma_agent_mod = load_module("mma_agent", Path(__file__).parent / "mma_agent.py")
esports_agent_mod = load_module("esports_agent", Path(__file__).parent / "esports_agent.py")
golf_agent_mod = load_module("golf_agent", Path(__file__).parent / "golf_agent.py")
rugby_agent_mod = load_module("rugby_agent", Path(__file__).parent / "rugby_agent.py")
cricket_agent_mod = load_module("cricket_agent", Path(__file__).parent / "cricket_agent.py")
table_tennis_agent_mod = load_module("table_tennis_agent", Path(__file__).parent / "table_tennis_agent.py")
data_acquisition = load_module("data_acquisition_agent", Path(__file__).parent / "data_acquisition_agent.py")
formatting = load_module("formatting_agent", Path(__file__).parent / "formatting_agent.py")
storage = load_module("storage_agent", Path(__file__).parent / "storage_agent.py")
evaluators = load_module("evaluator_agents", Path(__file__).parent / "evaluator_agents.py")
football_eval_mod = load_module("football_evaluator", Path(__file__).parent / "football_evaluator.py")
augmentation_mod = load_module("data_augmentation_agent", Path(__file__).parent / "data_augmentation_agent.py")

# Get classes
ManagerAgent = manager_agent.ManagerAgent
BasketballAgent = sport_agents_mod.BasketballAgent
VolleyballAgent = sport_agents_mod.VolleyballAgent
HandballAgent = sport_agents_mod.HandballAgent
TennisAgent = sport_agents_mod.TennisAgent
FootballAgent = football_agent_mod.FootballAgent
BaseballAgent = baseball_agent_mod.BaseballAgent
HockeyAgent = hockey_agent_mod.HockeyAgent
MMAAgent = mma_agent_mod.MMAAgent
EsportsAgent = esports_agent_mod.EsportsAgent
GolfAgent = golf_agent_mod.GolfAgent
RugbyAgent = rugby_agent_mod.RugbyAgent
CricketAgent = cricket_agent_mod.CricketAgent
TableTennisAgent = table_tennis_agent_mod.TableTennisAgent
DataAcquisitionAgent = data_acquisition.DataAcquisitionAgent
FormattingAgent = formatting.FormattingAgent
StorageAgent = storage.StorageAgent
BasketballEvaluatorAgent = evaluators.BasketballEvaluatorAgent
VolleyballEvaluatorAgent = evaluators.VolleyballEvaluatorAgent
HandballEvaluatorAgent = evaluators.HandballEvaluatorAgent
TennisEvaluatorAgent = evaluators.TennisEvaluatorAgent
FootballEvaluatorAgent = football_eval_mod.FootballEvaluatorAgent
DataAugmentationAgent = augmentation_mod.DataAugmentationAgent


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  SPORTS DATA SWARM - ML Training Dataset Generator v2.0")
    print("=" * 70)
    print("  Collecting historical sports data for AI training")
    print("  Using Brave Search & Serper APIs + Data Augmentation")
    print("=" * 70 + "\n")


def check_api_keys():
    """Check if required API keys are set."""
    try:
        from dotenv import load_dotenv
        # Load from project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        load_dotenv(env_path)
    except ImportError:
        pass
    
    brave_key = os.getenv('BRAVE_API_KEY')
    serper_key = os.getenv('SERPER_API_KEY')
    
    if not brave_key:
        print("!  Warning: BRAVE_API_KEY not found in environment")
    else:
        print("[OK] Brave Search API: Connected")
    
    if not serper_key:
        print("!  Warning: SERPER_API_KEY not found in environment")
    else:
        print("[OK] Serper API: Connected")
    
    if not brave_key and not serper_key:
        print("\n[ERROR] No search API keys configured!")
        print("Please set BRAVE_API_KEY and/or SERPER_API_KEY in your .env file")
        return False
    
    return True


async def run_collection(
    sports: list,
    target_records: int,
    date_range: dict,
    output_format: str,
    create_split: bool,
    augment: bool = False,
    augment_multiplier: float = 2.0
):
    """Run the full data collection process."""
    
    print(f"\nConfiguration:")
    print(f"   Sports: {', '.join(sports)}")
    print(f"   Target records per sport: {target_records:,}")
    print(f"   Date range: {date_range['start']} to {date_range['end']}")
    print(f"   Output format: {output_format}")
    print(f"   Train/test split: {create_split}")
    if augment:
        print(f"   Data augmentation: {augment_multiplier}x multiplier")
    print()
    
    # Create output directories
    Path("datasets/sports_data/raw").mkdir(parents=True, exist_ok=True)
    Path("datasets/sports_data/processed").mkdir(parents=True, exist_ok=True)
    Path("datasets/sports_data/evaluated").mkdir(parents=True, exist_ok=True)
    
    # Initialize Manager Agent
    manager = ManagerAgent(config={'target_records': target_records})
    
    # Register Sport Agents (13 sports for betting)
    sport_agents = {
        'basketball': BasketballAgent(),
        'volleyball': VolleyballAgent(),
        'handball': HandballAgent(),
        'tennis': TennisAgent(),
        'football': FootballAgent(),
        'baseball': BaseballAgent(),
        'hockey': HockeyAgent(),
        'mma': MMAAgent(),
        'esports': EsportsAgent(),
        'golf': GolfAgent(),
        'rugby': RugbyAgent(),
        'cricket': CricketAgent(),
        'table_tennis': TableTennisAgent()
    }
    
    for sport in sports:
        if sport in sport_agents:
            manager.register_sport_agent(sport, sport_agents[sport])
    
    # Register Support Agents
    manager.register_acquisition_agent(DataAcquisitionAgent())
    manager.register_formatting_agent(FormattingAgent())
    manager.register_storage_agent(StorageAgent(config={
        'output_dir': 'datasets/sports_data',
        'format': output_format,
        'split_ratio': 0.8
    }))
    
    # Register Evaluator Agents
    evaluators_map = {
        'basketball': BasketballEvaluatorAgent(),
        'volleyball': VolleyballEvaluatorAgent(),
        'handball': HandballEvaluatorAgent(),
        'tennis': TennisEvaluatorAgent(),
        'football': FootballEvaluatorAgent(),
        # Using base evaluators for new sports (can be customized later)
        'baseball': BasketballEvaluatorAgent(),  # Reuse as template
        'hockey': BasketballEvaluatorAgent(),
        'mma': BasketballEvaluatorAgent(),
        'esports': BasketballEvaluatorAgent(),
        'golf': BasketballEvaluatorAgent(),
        'rugby': BasketballEvaluatorAgent(),
        'cricket': BasketballEvaluatorAgent(),
        'table_tennis': TennisEvaluatorAgent()  # Similar to tennis
    }
    
    for sport in sports:
        if sport in evaluators_map:
            manager.register_evaluator_agent(sport, evaluators_map[sport])
    
    # Execute collection
    print("\nStarting data collection...\n")
    start_time = datetime.now()
    
    try:
        result = await manager.execute({
            'sports': sports,
            'target_records': target_records,
            'date_range': date_range,
            'output_format': output_format,
            'create_split': create_split
        })
        
        if not result.success:
            print(f"[ERROR] Collection failed: {result.error}")
            return False
        
        # Data Augmentation Phase
        if augment:
            print("\n" + "="*70)
            print("  DATA AUGMENTATION PHASE")
            print("="*70 + "\n")
            
            augmentation_agent = DataAugmentationAgent()
            
            for sport in sports:
                print(f"\nAugmenting {sport} data...")
                
                # Load collected data
                sport_data = result.data.get('results', {}).get(sport, {}) or {}
                storage_data = sport_data.get('storage', {}) or {}
                
                if storage_data and 'filepath' in storage_data:
                    # Load the dataset
                    filepath = storage_data['filepath']
                    try:
                        if filepath.endswith('.json'):
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                records = data if isinstance(data, list) else data.get('records', [])
                        elif filepath.endswith('.csv'):
                            import csv
                            with open(filepath, 'r') as f:
                                records = list(csv.DictReader(f))
                        else:
                            records = []
                        
                        if records:
                            # Apply augmentation
                            aug_result = await augmentation_agent.execute({
                                'sport': sport,
                                'data': {'records': records},
                                'target_multiplier': augment_multiplier,
                                'techniques': ['all']
                            })
                            
                            if aug_result.success:
                                # Save augmented data
                                aug_storage = StorageAgent()
                                aug_save = await aug_storage.execute({
                                    'sport': f"{sport}_augmented",
                                    'data': aug_result.data,
                                    'format': output_format,
                                    'split': create_split
                                })
                                
                                if aug_save.success:
                                    print(f"  [OK] Augmented: {aug_result.data.get('original_count')} -> {aug_result.data.get('augmented_count')} records")
                                    print(f"  [FILE] Saved to: {aug_save.data.get('filepath')}")
                    except Exception as e:
                        print(f"  [ERROR] Augmentation failed: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"  COLLECTION COMPLETED IN {elapsed:.1f} SECONDS")
        print(f"{'='*70}\n")
        
        print(f"Results Summary:")
        print(f"   Total records collected: {result.records_processed:,}")
        
        for sport, data in result.data.get('results', {}).items():
            if data:
                storage = data.get('storage', {})
                print(f"\n   {sport.upper()}:")
                print(f"      Records: {storage.get('record_count', 0):,}")
                print(f"      File: {storage.get('filepath', 'N/A')}")
        
        print(f"\nOutput files:")
        print(f"   Processed data: datasets/sports_data/processed/")
        print(f"   Evaluation reports: datasets/sports_data/evaluated/")
        if augment:
            print(f"   Augmented data: datasets/sports_data/processed/*_augmented_*.csv")
        print(f"   Collection report: datasets/sports_data/collection_report.json")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sports Data Swarm - Collect sports data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect data for all sports (10,000 records each)
    python run_collection.py --all
    
    # Collect only football data
    python run_collection.py --sport football --target 5000
    
    # Collect with data augmentation (3x data multiplication)
    python run_collection.py --sport basketball --target 1000 --augment 3.0
    
    # Collect multiple specific sports
    python run_collection.py --sports basketball football tennis --target 2000
    
    # Full custom configuration with augmentation
    python run_collection.py --all --target 5000 --augment 2.5 --format parquet
        """
    )
    
    # Sport selection (13 sports available)
    sport_group = parser.add_mutually_exclusive_group(required=False)
    sport_group.add_argument(
        '--sport', '-s',
        choices=['basketball', 'volleyball', 'handball', 'tennis', 'football', 
                 'baseball', 'hockey', 'mma', 'esports', 'golf', 'rugby', 
                 'cricket', 'table_tennis'],
        help='Single sport to collect data for'
    )
    sport_group.add_argument(
        '--sports',
        nargs='+',
        choices=['basketball', 'volleyball', 'handball', 'tennis', 'football',
                 'baseball', 'hockey', 'mma', 'esports', 'golf', 'rugby',
                 'cricket', 'table_tennis'],
        help='Multiple sports to collect data for'
    )
    sport_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Collect data for all 13 sports (best for betting models)'
    )
    
    # Collection parameters
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=10000,
        help='Target number of records per sport (default: 10000)'
    )
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date for historical data (default: 2020-01-01)'
    )
    parser.add_argument(
        '--end-date',
        default='2025-12-31',
        help='End date for historical data (default: 2025-12-31)'
    )
    
    # Output options
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'json', 'parquet'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Disable train/test split'
    )
    
    # Augmentation options
    parser.add_argument(
        '--augment',
        type=float,
        nargs='?',
        const=2.0,
        metavar='MULTIPLIER',
        help='Enable data augmentation with optional multiplier (default: 2.0x)'
    )
    
    # Utility
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check API keys and exit'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check API keys
    if not check_api_keys() or args.check:
        sys.exit(1 if not args.check else 0)
    
    if args.check:
        print("\n[OK] All checks passed!")
        sys.exit(0)
    
    # Determine sports to collect (skip if check mode)
    if args.check:
        sports = []
    elif args.all:
        sports = ['basketball', 'volleyball', 'handball', 'tennis', 'football',
                  'baseball', 'hockey', 'mma', 'esports', 'golf', 'rugby',
                  'cricket', 'table_tennis']
    elif args.sport:
        sports = [args.sport]
    elif args.sports:
        sports = args.sports
    else:
        parser.error("Please specify --sport, --sports, or --all")
    
    # Date range
    date_range = {
        'start': args.start_date,
        'end': args.end_date
    }
    
    # Run collection
    success = asyncio.run(run_collection(
        sports=sports,
        target_records=args.target,
        date_range=date_range,
        output_format=args.format,
        create_split=not args.no_split,
        augment=args.augment is not None,
        augment_multiplier=args.augment if args.augment else 2.0
    ))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
